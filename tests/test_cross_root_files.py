"""A recording outside the open folder — named by its full path, and read.

In the Ask tab, "who speaks" reported 0 people for several recordings in a
row, then *"Error calling tool 'file_info': Not found (404)"*. The search
spans every root, and a hit's path is relative to ITS root; the per-file
tools resolved it against the folder open in the app — a different root —
and found nothing, for a recording with plenty of voice windows.

What these tests pin:

* the MCP server names a hit relative to the open folder when it lies
  there, else by its full path (as ``where_person_speaks`` and
  ``semantic_search`` always did);
* the per-file reading routes take a full path as ``?file=`` — within the
  allowed roots, anything else is refused with 403;
* the clip player plays a recording by its full path;
* the search's file filter takes a full path: that recording, or a folder.
"""

from __future__ import annotations

import asyncio
import shutil
import subprocess

import pytest

import whisper_tm.mcp_server as mcp

SAMPLE = {
    "filename": "talk.mp4",
    "complete": True,
    "language": "de",
    "model": "turbo",
    "speakers": ["SPEAKER_00"],
    "segments": [
        {"speaker": "SPEAKER_00", "start": float(i), "end": i + 1.0, "text": f"Sentence {i}"}
        for i in range(6)
    ],
}


@pytest.fixture
def env(tmp_path, monkeypatch):
    """App with its own database; a transcript under root A, B is open."""
    monkeypatch.setenv("HOME", str(tmp_path))
    from whisper_tm.config import config as cfg

    cfg.load(None)
    root_a, root_b = tmp_path / "a", tmp_path / "b"
    (root_a / "show").mkdir(parents=True)
    root_b.mkdir()
    (root_a / "show" / "talk.mp4").write_bytes(b"x" * 64)
    cfg._data.setdefault("directories", {})["media"] = str(root_a)
    cfg._data.setdefault("security", {})["allowed_roots"] = [str(tmp_path)]
    cfg._resolve_directories()
    cfg._ensure_directories()
    from whisper_tm import extensions as ext
    from whisper_tm.api import search as search_api
    from whisper_tm.api import transcription as transcription_api
    from whisper_tm.app import create_app

    client = create_app().test_client()
    # The API modules bind `db` at import time; with one app per test the
    # binding has to follow (as in test_lookup_by_identity)
    monkeypatch.setattr(search_api, "db", ext.db)
    monkeypatch.setattr(transcription_api, "db", ext.db)
    full = str(root_a / "show" / "talk.mp4")
    ext.db.upsert_transcript("show/talk.mp4", SAMPLE, media_dir=str(root_a), abs_path=full)
    from whisper_tm.scanner import register_single_file

    register_single_file(full, db=ext.db, root=str(root_a))
    cfg.media_dir = root_b  # the user opened another folder
    yield client, full, root_a
    cfg.media_dir = root_a


class TestTheRoutesTakeAFullPath:
    def test_the_transcript(self, env):
        client, full, _ = env
        assert client.get("/api/transcript/show/talk.mp4").status_code == 404
        r = client.get("/api/transcript", query_string={"file": full})
        assert r.status_code == 200, r.get_json()
        assert [s["text"] for s in r.get_json()["segments"]][:2] == ["Sentence 0", "Sentence 1"]

    def test_the_file_entry(self, env):
        client, full, _ = env
        r = client.get("/api/scan_entry", query_string={"file": full})
        assert r.status_code == 200, r.get_json()
        assert r.get_json()["file"]["abs_path"] == full
        assert (
            client.get("/api/scan_entry", query_string={"file": "show/talk.mp4"}).status_code == 400
        )

    def test_summaries_and_translations(self, env):
        client, full, _ = env
        assert client.get("/api/summaries", query_string={"file": full}).get_json() == {
            "summaries": [],
            "pending": False,
        }
        assert client.get("/api/translations", query_string={"file": full}).get_json() == {
            "translations": []
        }

    @pytest.mark.parametrize(
        "route",
        [
            "/api/transcript",
            "/api/scan_entry",
            "/api/summaries",
            "/api/translations",
            "/api/translation/en",
        ],
    )
    def test_outside_the_allowed_roots_is_refused(self, env, route):
        client = env[0]
        assert client.get(route, query_string={"file": "/etc/passwd"}).status_code == 403


class TestTheSearchFilter:
    def test_a_recording_or_a_folder_by_full_path(self, env):
        client, full, root_a = env
        for value, hits in (
            (full, 6),
            (str(root_a / "show"), 6),
            (str(root_a / "sh"), 0),
            (str(root_a / "other"), 0),
        ):
            d = client.get("/api/search", query_string={"q": "Sentence", "file": value}).get_json()
            assert d["total"] == hits, value


ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not available")


class TestTheClipPlayer:
    @pytest.fixture
    def app(self, tmp_path, monkeypatch):
        from flask import Flask

        from whisper_tm.config import config

        lib, other, clips = tmp_path / "lib", tmp_path / "open", tmp_path / "clips"
        for d in (lib, other, clips):
            d.mkdir()
        if shutil.which("ffmpeg"):
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-f",
                    "lavfi",
                    "-i",
                    "sine=frequency=440:duration=5",
                    str(lib / "tone.wav"),
                ],
                capture_output=True,
                timeout=60,
            )
        monkeypatch.setattr(type(config), "media_dir", property(lambda self: other))
        monkeypatch.setattr(type(config), "dirs", property(lambda self: {"clips": clips}))
        monkeypatch.setitem(
            config._data.setdefault("security", {}), "allowed_roots", [str(tmp_path)]
        )
        from whisper_tm.api.media import media_bp

        a = Flask(__name__)
        a.register_blueprint(media_bp)
        a.lib = lib
        return a

    @ffmpeg
    def test_a_recording_outside_the_open_folder(self, app):
        r = app.test_client().get(
            "/api/clip/audio",
            query_string={"file": str(app.lib / "tone.wav"), "start": 1, "end": 3},
        )
        assert r.status_code == 200 and r.mimetype == "audio/mpeg"

    def test_outside_the_allowed_roots(self, app):
        r = app.test_client().get(
            "/api/clip/audio", query_string={"file": "/etc/hostname", "start": 0, "end": 2}
        )
        assert r.status_code == 403

    @ffmpeg
    def test_an_older_relative_player_finds_its_one_recording(self, app, tmp_path, monkeypatch):
        """Players written before (and in Open WebUI) name a hit of another
        root relative to that root — it plays when exactly one recording
        has that name, and not at all when two do."""
        from whisper_tm import extensions as ext
        from whisper_tm.database import Database

        db = Database(tmp_path / "wx.db")
        db.initialize()
        monkeypatch.setattr(ext, "db", db)
        tx = {"complete": True, "segments": [{"start": 0.0, "end": 4.0, "text": "a"}]}
        db.upsert_transcript(
            "tone.wav", tx, media_dir=str(app.lib), abs_path=str(app.lib / "tone.wav")
        )
        try:
            ask = {"file": "tone.wav", "start": 1, "end": 3}
            assert app.test_client().get("/api/clip/audio", query_string=ask).status_code == 200
            second = tmp_path / "second"
            second.mkdir()
            db.upsert_transcript(
                "tone.wav", tx, media_dir=str(second), abs_path=str(second / "tone.wav")
            )
            assert app.test_client().get("/api/clip/audio", query_string=ask).status_code == 404
        finally:
            db.close_all()


class TestTheMcpNamesThem:
    def test_named(self):
        assert mcp._named("show/talk.mp4", "/nas", "/nas") == "show/talk.mp4"
        assert mcp._named("videos/a/talk.mp4", "/nas", "/nas/videos") == "a/talk.mp4"
        assert (
            mcp._named("videos/a/talk.mp4", "/nas", "/nas/Archive/old") == "/nas/videos/a/talk.mp4"
        )
        assert mcp._named("/nas/x.mp4", "/nas", "/other") == "/nas/x.mp4"
        assert mcp._named("x.mp4", "", "/other") == "x.mp4"

    def test_a_full_path_travels_as_a_parameter(self, monkeypatch):
        asked = []
        monkeypatch.setattr(mcp, "CONN", mcp.Connection("http://127.0.0.1:5077"))
        monkeypatch.setattr(mcp.CONN, "fetch", lambda path, **p: asked.append((path, p)) or {})
        mcp._fetch_file("/api/transcript", "/nas/videos/a b.mp4")
        mcp._fetch_file("/api/transcript", "videos/#tag #other.mp4")
        assert asked == [
            ("/api/transcript", {"file": "/nas/videos/a b.mp4"}),
            ("/api/transcript/videos/%23tag%20%23other.mp4", {}),
        ]

    def test_search_hits_outside_the_open_folder(self, monkeypatch):
        monkeypatch.setattr(mcp, "CONN", mcp.Connection("http://127.0.0.1:5077"))
        monkeypatch.setattr(
            mcp.CONN,
            "fetch",
            lambda path, **p: {
                "total": 2,
                "files": 2,
                "media_dir": "/nas/Archive/old",
                "results": [
                    {
                        "filename": "videos/Archive/talk.mp4",
                        "file_dir": "/nas",
                        "start": 12.0,
                        "start_fmt": "0:12",
                        "text": "money",
                    },
                    {
                        "filename": "song.mp3",
                        "file_dir": "/nas/Archive/old",
                        "start": 3.0,
                        "start_fmt": "0:03",
                        "text": "money",
                    },
                ],
            },
        )
        server = mcp.build_server()
        tool = asyncio.run(server.get_tool("search"))
        out = asyncio.run(tool.run({"text": "money"})).structured_content
        assert [h["file"] for h in out["hits"]] == ["/nas/videos/Archive/talk.mp4", "song.mp3"]
