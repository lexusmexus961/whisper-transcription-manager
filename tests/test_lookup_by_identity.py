"""
Lookups that must work no matter which folder is open.

``rel_path`` is relative to the root that was open when a file was
transcribed. Three places joined it with the root open NOW, and failed as
soon as another folder was open:

* the transcript export skipped every transcript and handed out an empty
  file;
* the search's reading pane looked transcripts up by ``rel_path`` alone;
* the "embedded" badges in the file list used the stored ``rel_path`` as
  the key, while the list matches on names relative to the open folder.
"""

import io
import json
import zipfile

import pytest

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
    """App with its own database; one transcript under root A, B is open."""
    monkeypatch.setenv("HOME", str(tmp_path))
    from whisper_tm.config import config as cfg

    cfg.load(None)
    root_a, root_b = tmp_path / "a", tmp_path / "b"
    (root_a / "show").mkdir(parents=True)
    root_b.mkdir()
    cfg._data.setdefault("directories", {})["media"] = str(root_a)
    cfg._resolve_directories()
    cfg._ensure_directories()
    from whisper_tm import extensions as ext
    from whisper_tm.api import rag as rag_api
    from whisper_tm.api import search as search_api
    from whisper_tm.app import create_app

    client = create_app().test_client()
    # The API modules bind `db` at import time; with one app per test the
    # binding has to follow (same as the other API tests do).
    monkeypatch.setattr(rag_api, "db", ext.db)
    monkeypatch.setattr(search_api, "db", ext.db)
    ext.db.upsert_transcript(
        "show/talk.mp4", SAMPLE, media_dir=str(root_a), abs_path=str(root_a / "show" / "talk.mp4")
    )
    cfg.media_dir = root_b  # the user opened another folder
    yield client, root_a, root_b
    cfg.media_dir = root_a


@pytest.mark.ffmpeg
def test_jsonl_export_finds_transcripts_of_another_root(env):
    client, root_a, _ = env
    r = client.get("/api/export/transcripts")
    lines = [json.loads(line) for line in r.get_data(as_text=True).splitlines()]
    assert len(lines) == 1
    assert lines[0]["_abs_path"] == str(root_a / "show" / "talk.mp4")
    assert lines[0]["_rel_path"] == "show/talk.mp4"
    assert len(lines[0]["segments"]) == 6


@pytest.mark.ffmpeg
def test_zip_export_names_entries_by_identity(env):
    client, root_a, _ = env
    r = client.get("/api/export/transcripts?format=zip")
    names = zipfile.ZipFile(io.BytesIO(r.get_data())).namelist()
    assert names == [str(root_a / "show" / "talk.mp4").lstrip("/") + ".json"]


def test_reading_pane_context_uses_the_hits_root(env):
    client, root_a, _ = env
    r = client.get(
        "/api/search/context",
        query_string={"file": "show/talk.mp4", "dir": str(root_a), "start": 3, "n": 1},
    )
    assert r.status_code == 200, r.get_json()
    texts = [s["text"] for s in r.get_json()["segments"]]
    assert "Sentence 3" in texts


def test_reading_pane_without_root_does_not_guess(env):
    """Without `dir`, the open root (B) is used — and the file isn't there.
    The old lookup would have returned whatever transcript shared the
    relative path."""
    client, _, _ = env
    r = client.get("/api/search/context", query_string={"file": "show/talk.mp4", "start": 3})
    assert r.status_code == 404


def test_list_refresh_skips_a_file_outside_the_open_folder(env):
    """A job finished under root A while B is open: the list has no row for
    it, and a 404 here made the frontend rescan all of B after every job."""
    client, root_a, _ = env
    abs_path = str(root_a / "show" / "talk.mp4")
    r = client.get("/api/scan_entry/show/talk.mp4", query_string={"abs": abs_path})
    assert r.status_code == 200
    assert r.get_json() == {"file": None, "outside": True}


def test_list_refresh_inside_the_open_folder_still_answers(env, monkeypatch):
    client, root_a, _ = env
    from whisper_tm import extensions as ext
    from whisper_tm.config import config as cfg

    abs_path = str(root_a / "show" / "talk.mp4")
    ext.db.upsert_media_file("show/talk.mp4", abs_path=abs_path)
    cfg.media_dir = root_a
    r = client.get("/api/scan_entry/show/talk.mp4", query_string={"abs": abs_path})
    assert r.status_code == 200
    assert r.get_json()["file"]["name"] == "show/talk.mp4"


def test_list_refresh_in_a_subfolder_opened_since(env):
    """The queue names the file relative to the entry's root ("show/…"),
    the open folder is "show" itself — the upload dialog opens the uploads
    folder that way. The name joined to the open folder was a 404."""
    client, root_a, _ = env
    from whisper_tm import extensions as ext
    from whisper_tm.config import config as cfg

    abs_path = str(root_a / "show" / "talk.mp4")
    ext.db.upsert_media_file("show/talk.mp4", abs_path=abs_path)
    cfg.media_dir = root_a / "show"
    r = client.get("/api/scan_entry/show/talk.mp4", query_string={"abs": abs_path})
    assert r.status_code == 200
    assert r.get_json()["file"]["name"] == "talk.mp4"  # the list's name
