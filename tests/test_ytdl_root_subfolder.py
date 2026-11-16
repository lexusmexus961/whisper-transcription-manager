"""Link download while a SUBDIRECTORY is open as root.

With `…/@SomeChannel` open as root, a link was added. The file
landed in uploads/, but `register_single_file` calculated relative to the
OPEN root → None → `files: []`, `in_library: False`; the UI reported
"outside the media directory" and queued nothing — the download disappeared
from view.

Now: first relative to the current root, otherwise relative to the uploads
directory itself (the UI jumps there anyway).
"""

from __future__ import annotations

import pytest

from tests.test_ytdl_multiple_links import FakeJob, FakeJobManager, _entries  # noqa: E402
from whisper_tm.core import ytdl
from whisper_tm.database import Database
from whisper_tm.scanner import register_single_file


@pytest.fixture
def run(monkeypatch, tmp_path):
    j = FakeJob()
    monkeypatch.setattr(ytdl.ext, "job_manager", FakeJobManager(j))
    media = tmp_path / "media"
    uploads = media / "whisper-tm" / "uploads"
    uploads.mkdir(parents=True)
    (tmp_path / "work").mkdir()
    monkeypatch.setattr(ytdl.config, "_resolved_uploads", uploads, raising=False)
    monkeypatch.setattr(ytdl.config, "_resolved_work", tmp_path / "work", raising=False)
    monkeypatch.setattr(ytdl.config, "_resolved_chunks", tmp_path / "work", raising=False)
    # The OPEN root is a subdirectory that does NOT contain uploads/
    below = media / "videos" / "channel"
    below.mkdir(parents=True)
    monkeypatch.setattr(ytdl.config, "_resolved_media", below, raising=False)
    db = Database(tmp_path / "w.db")
    db.initialize()
    import whisper_tm.extensions as ext

    monkeypatch.setattr(ext, "db", db)

    def _load(job, entry, tmp, target_dir, report_it=None, audio_only=None):
        p = target_dir / f"{entry['title']}.m4a"
        p.write_bytes(b"x" * 1000)
        return p

    monkeypatch.setattr(ytdl, "_download_one", _load)
    monkeypatch.setattr(ytdl, "enumerate_entries", lambda u: _entries("led"))
    return j, db, uploads, below


def test_registration_falls_back_to_the_upload_directory(run):
    j, db, uploads, below = run
    ytdl._worker("j1", ["https://y/led"])
    assert j.status == "done"
    e = j.result
    assert e["ok"] == 1
    assert e["files"] == ["led.m4a"], e  # relative to uploads/, not empty
    assert e["in_library"] is True
    # and the row is in the DB, uniquely identified via abs_path
    with db.connection() as conn:
        rows = conn.execute("SELECT rel_path, abs_path FROM media_files").fetchall()
    assert len(rows) == 1 and rows[0][1].endswith("/whisper-tm/uploads/led.m4a")


def test_inside_the_root_rel_path_stays_relative_to_the_root(run, monkeypatch):
    j, db, uploads, below = run
    monkeypatch.setattr(ytdl.config, "_resolved_media", uploads.parent.parent, raising=False)
    ytdl._worker("j2", ["https://y/led"])
    assert j.result["files"] == ["whisper-tm/uploads/led.m4a"]


def test_register_single_file_with_its_own_root(run):
    j, db, uploads, below = run
    f = uploads / "x.m4a"
    f.write_bytes(b"y" * 10)
    assert register_single_file(f) is None  # open root: subdirectory
    assert register_single_file(f, root=uploads) == "x.m4a"
