"""The transcript export, narrowed down: the folder open in the app, and
when the recordings came into the library — "everything downloaded into
the uploads folder today" — as JSON, readable text or subtitles.

* ``folder=open``: the open folder, its subfolders too unless
  ``subfolders=0``;
* ``since`` / ``until``: when the recording came into the library (its
  media row), the transcript's own time without one;
* ``as=txt``: one line per segment with its time and the speaker's name,
  ``as=srt``: subtitles; the files named below the folder, two that end
  up with the same name both kept;
* the count the export dialog shows first, and what is refused.
"""

import io
import time
import zipfile

import pytest

SAMPLE = {
    "complete": True,
    "language": "en",
    "model": "turbo",
    "speakers": ["SPEAKER_00", "SPEAKER_01"],
    "speaker_names": {"SPEAKER_00": "Alice"},
    "segments": [
        {
            "speaker": f"SPEAKER_0{i % 2}",
            "start": 61.0 * i,
            "end": 61.0 * i + 2.5,
            "text": f"Sentence {i}",
        }
        for i in range(3)
    ],
}


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    from whisper_tm.config import config as cfg

    cfg.load(None)
    root = tmp_path / "media"
    (root / "uploads" / "deep").mkdir(parents=True)
    (root / "other").mkdir()
    cfg._data.setdefault("directories", {})["media"] = str(root)
    cfg._resolve_directories()
    cfg._ensure_directories()
    from whisper_tm import extensions as ext
    from whisper_tm.api import rag as rag_api
    from whisper_tm.app import create_app

    client = create_app().test_client()
    monkeypatch.setattr(rag_api, "db", ext.db)
    for rel in ("uploads/talk.mp4", "uploads/talk.m4a", "uploads/deep/old.mp4", "other/x.mp4"):
        ext.db.upsert_transcript(
            rel, {**SAMPLE, "filename": rel}, media_dir=str(root), abs_path=str(root / rel)
        )
    # The old one came into the library a week ago (its media row says so)
    week = time.time() - 7 * 86400
    with ext.db.connection() as c:
        n = c.execute(
            "UPDATE media_files SET created_at = ? WHERE abs_path = ?",
            (week, str(root / "uploads/deep/old.mp4")),
        ).rowcount
        if not n:  # no media row: the transcript's own time counts
            c.execute(
                "UPDATE transcripts SET created_at = ? WHERE abs_path = ?",
                (week, str(root / "uploads/deep/old.mp4")),
            )
    cfg.media_dir = root / "uploads"  # the uploads folder is open
    yield client, root
    cfg.media_dir = root


def _names(r):
    assert r.status_code == 200, r.get_data(as_text=True)[:300]
    return sorted(zipfile.ZipFile(io.BytesIO(r.get_data())).namelist())


@pytest.mark.ffmpeg
def test_the_open_folder_and_its_subfolders(env):
    client, _ = env
    assert _names(client.get("/api/export/transcripts?format=zip&folder=open")) == [
        "deep/old.mp4.json",
        "talk.m4a.json",
        "talk.mp4.json",
    ]
    assert _names(client.get("/api/export/transcripts?format=zip&folder=open&subfolders=0")) == [
        "talk.m4a.json",
        "talk.mp4.json",
    ]
    # Without a filter: everything, as before
    assert len(_names(client.get("/api/export/transcripts?format=zip"))) == 4


@pytest.mark.ffmpeg
def test_added_today(env):
    client, _ = env
    midnight = time.time() - 3600
    d = client.get(f"/api/export/transcripts/count?folder=open&since={midnight}").get_json()
    assert d["count"] == 2 and d["examples"] == ["talk.m4a", "talk.mp4"]
    old = client.get(f"/api/export/transcripts/count?folder=open&until={midnight}").get_json()
    assert old["count"] == 1 and old["examples"] == ["old.mp4"]


@pytest.mark.ffmpeg
def test_as_text_with_times_and_names(env):
    client, _ = env
    r = client.get(
        f"/api/export/transcripts?format=zip&folder=open&as=txt&since={time.time() - 60}"
    )
    z = zipfile.ZipFile(io.BytesIO(r.get_data()))
    assert sorted(z.namelist()) == ["talk (2).txt", "talk.txt"]  # .mp4 and .m4a both kept
    assert z.read("talk.txt").decode().splitlines() == [
        "[0:00:00 - 0:00:02] Alice: Sentence 0",
        "[0:01:01 - 0:01:03] Speaker 2: Sentence 1",
        "[0:02:02 - 0:02:04] Alice: Sentence 2",
    ]
    srt = client.get("/api/export/transcripts?format=zip&folder=open&subfolders=0&as=srt")
    first = zipfile.ZipFile(io.BytesIO(srt.get_data())).read("talk.srt").decode()
    assert first.startswith("1\n00:00:00,000 --> 00:00:02,500\nAlice: Sentence 0\n")


@pytest.mark.ffmpeg
def test_refused(env):
    client, _ = env
    assert client.get("/api/export/transcripts?format=zip&as=docx").status_code == 400
    assert client.get("/api/export/transcripts?since=today").status_code == 400
    assert client.get("/api/export/transcripts/count?until=x").status_code == 400
