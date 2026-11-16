"""What the extension asks about a page's videos (core/ingest.py).

Found in an extension review: ``/api/ingest/known`` matched any page-supplied
title against every file name in the upload folder, so a page could probe
which files exist. Now:

* a request without a title matches by the video id alone — the extension
  sends titles only on YouTube pages, and a cross-site page can't post at
  all (the server's Origin check);
* with a title: the noted downloads (their titles and file names) and, as
  before, the upload folder's file names — a video downloaded before the
  ids were noted is still recognized;
* every download notes its id — the web UI's YouTube dialog and the MCP
  server (core/ytdl) as well as the extension's batches — or those videos
  were never recognized on YouTube;
* one call asks about 200 videos at most, reads the index and the upload
  folder once, and asks the filesystem only about a match.
"""

from __future__ import annotations

import os

import pytest

from whisper_tm.core import ingest

YT = "https://www.youtube.com/watch?v="


@pytest.fixture
def uploads(tmp_path, monkeypatch):
    from whisper_tm.config import config

    up = tmp_path / "uploads"
    up.mkdir()
    monkeypatch.setattr(type(config), "upload_dir", property(lambda self: up))
    monkeypatch.setattr(type(config), "data_dir", property(lambda self: tmp_path / "data"))
    return up


@pytest.fixture
def no_db(monkeypatch):
    """known() reads a found file's state — stubbed, the lookup is what counts."""

    class Conn:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    class Db:
        def connection(self):
            return Conn()

    monkeypatch.setattr(ingest, "_db", lambda: Db())
    monkeypatch.setattr(
        ingest, "file_state", lambda *a, **k: {"summary": "", "transcribed": False, "queue": ""}
    )


def test_without_a_title_no_file_name_is_matched(uploads, no_db):
    (uploads / "Private_call.m4a").write_bytes(b"x")
    rows = ingest.known(
        [
            {"url": "https://example.com/Private_call", "title": ""},
            {"url": YT + "aaaaaaaaaaa", "title": ""},
        ]
    )
    assert [r["abs_path"] for r in rows] == ["", ""]
    assert ingest._find("https://example.com/x", "") == ""


def test_with_a_title_a_download_from_before_the_index(uploads, no_db):
    f = uploads / "How_Fast_Is_It_Really.m4a"
    f.write_bytes(b"x")
    rows = ingest.known([{"url": YT + "aaaaaaaaaaa", "title": "How Fast Is It Really?"}])
    assert rows[0]["abs_path"] == str(f)
    assert ingest._find(YT + "aaaaaaaaaaa", "How Fast Is It Really?") == str(f)


def test_a_noted_download_by_id_and_by_its_title(uploads, no_db):
    f = uploads / "A_talk.m4a"
    f.write_bytes(b"x")
    ingest._remember("bbbbbbbbbbb", str(f), "A talk")
    assert ingest._find(YT + "bbbbbbbbbbb", "") == str(f)
    assert ingest._find(YT + "ccccccccccc", "A talk") == str(f)  # other link, same video
    rows = ingest.known(
        [{"url": "https://example.com/page", "title": ""}, {"url": YT + "bbbbbbbbbbb", "title": ""}]
    )
    assert [r["abs_path"] for r in rows] == ["", str(f)]


def test_a_noted_file_that_is_gone(uploads, no_db):
    f = uploads / "A_talk.m4a"
    f.write_bytes(b"x")
    ingest._remember("bbbbbbbbbbb", str(f), "A talk")
    f.unlink()
    assert ingest._find(YT + "bbbbbbbbbbb", "A talk") == ""


def test_at_most_200_per_call(uploads, no_db):
    rows = ingest.known([{"url": YT + f"{i:011d}", "title": ""} for i in range(500)])
    assert len(rows) == 200


def test_one_read_per_call_and_the_disk_only_on_a_match(uploads, no_db, monkeypatch):
    for i in range(30):
        f = uploads / f"Video_{i}.m4a"
        f.write_bytes(b"x")
        ingest._remember(f"v{i:010d}", str(f), f"Video {i}")
    reads, exists = [], []
    real_index, real_exists = ingest._index, os.path.exists
    monkeypatch.setattr(ingest, "_index", lambda: reads.append(1) or real_index())
    monkeypatch.setattr(ingest.os.path, "exists", lambda p: exists.append(p) or real_exists(p))
    videos = [{"url": YT + f"n{i:010d}", "title": f"Nothing {i}"} for i in range(100)] + [
        {"url": YT + "v0000000003", "title": "Video 3"}
    ]
    rows = ingest.known(videos)
    assert rows[-1]["abs_path"] == str(uploads / "Video_3.m4a")
    assert len(reads) == 1
    assert len(exists) <= 2, len(exists)


def test_a_youtube_dialog_download_is_noted(uploads, monkeypatch, tmp_path):
    from whisper_tm.core import ytdl
    from whisper_tm.jobs import JobManager

    jm = JobManager(logs_dir=tmp_path / "logs")
    monkeypatch.setattr(ytdl.ext, "job_manager", jm)
    monkeypatch.setattr(
        ytdl,
        "enumerate_entries",
        lambda u: [{"url": u, "title": "A dialog video", "id": "ddddddddddd"}],
    )

    def download(job, entry, tmp_dir, dest_dir, **kw):
        target = dest_dir / "A_dialog_video.m4a"
        target.write_bytes(b"x" * 2000)
        return target

    monkeypatch.setattr(ytdl, "_download_one", download)
    monkeypatch.setattr("whisper_tm.scanner.register_single_file", lambda *a, **k: None)
    jm.create("ytdl_x", filename="x", job_type="youtube")
    ytdl._worker("ytdl_x", [YT + "ddddddddddd"])
    assert jm.get("ytdl_x").status == "done"
    assert ingest._find(YT + "ddddddddddd", "") == str(uploads / "A_dialog_video.m4a")
