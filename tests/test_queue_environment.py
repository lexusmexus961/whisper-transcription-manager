"""Errors of the machine pause the queue; they don't fail every file.

A NAS unmounted, a full disk, a read-only target, no internet for a model
download: every pending file used to be booked as an error, one after
another, and the queue emptied itself into the error bucket. Now such an
error pauses the queue with its cause and the entry stays queued; fixed,
"Resume" goes on where it stopped. An error of the file itself is booked as
before.
"""

from __future__ import annotations

import json

import pytest

from whisper_tm.core import queue as q


@pytest.fixture
def db(tmp_path, monkeypatch):
    from whisper_tm.database import Database

    d = Database(tmp_path / "library.db")
    d.initialize()
    monkeypatch.setattr(q, "_db", lambda: d)
    q._paused = False
    yield d
    q._paused = False


@pytest.fixture
def jm(monkeypatch):
    class FakeJM:
        warnings: list = []

        def system_warning(self, msg):
            self.warnings.append(msg)

    f = FakeJM()
    f.warnings = []
    monkeypatch.setattr(q, "_jm", lambda: f)
    monkeypatch.setattr(q, "qlog", lambda m: None)
    return f


def _entry(db, filename="a.mkv", base="/media"):
    with db.connection() as conn:
        cur = conn.execute(
            "INSERT INTO queue_items (filename, base_dir, settings_json, status, position) "
            "VALUES (?, ?, ?, 'processing', 1)",
            (filename, base, json.dumps({})),
        )
    return {"id": cur.lastrowid, "filename": filename, "dir": base, "settings": {}}


def _status(db, item_id):
    with db.connection() as conn:
        r = conn.execute(
            "SELECT status, error FROM queue_items WHERE id = ?", (item_id,)
        ).fetchone()
    return r["status"], r["error"]


@pytest.mark.parametrize(
    ("error", "cause"),
    [
        ("ffmpeg: [Errno 28] No space left on device", "The disk is full"),
        ("OSError: [Errno 30] Read-only file system: '/work/a.wav'", "read-only"),
        ("OSError: [Errno 107] Transport endpoint is not connected", "network share"),
        (
            "HTTPSConnectionPool(host='huggingface.co', port=443): Max retries exceeded with url: /api",
            "internet",
        ),
        ("We couldn't connect to 'https://huggingface.co' to load this file", "internet"),
        (
            "Cannot find the requested files in the disk cache and outgoing traffic has been disabled",
            "Offline",
        ),
    ],
)
def test_what_the_machine_says(error, cause):
    assert cause in q.environment_cause(error)


@pytest.mark.parametrize(
    "error",
    [
        "Audio extraction failed",
        "Invalid data found when processing input",
        "CUDA out of memory",
        "Cancelled",
        "",
    ],
)
def test_what_the_file_says(error):
    assert q.environment_cause(error) == ""


def test_the_queue_pauses_and_the_entry_stays(db, jm):
    it = _entry(db)
    assert q._environment_pause(it, "[Errno 28] No space left on device")
    assert q.is_paused() and _status(db, it["id"]) == ("pending", "")
    assert jm.warnings and jm.warnings[0].startswith(
        "The disk is full — queue paused, a.mkv stays queued"
    )
    # A file's own error is not the machine's
    other = _entry(db, "b.mkv")
    q._paused = False
    assert not q._environment_pause(other, "Invalid data found when processing input")
    assert not q.is_paused() and _status(db, other["id"])[0] == "processing"


def test_a_library_gone(tmp_path):
    lib = tmp_path / "nas"
    assert q._library_gone(lib)  # not there
    lib.mkdir()
    assert q._library_gone(lib)  # an empty mount point
    (lib / "a.mkv").write_bytes(b"x")
    assert not q._library_gone(lib)


def test_the_errors_go_through_it():
    import inspect

    src = inspect.getsource(q)
    after_oom = src[src.index("if _oom_retry(item, err, job):") :]
    assert after_oom.index("elif _environment_pause(item, err, job):") < after_oom.index(
        '_finish(item["id"], "error", err)'
    )
    not_found = src[src.index('_finish(item["id"], "error", "Not found")') - 600 :]
    assert "_library_gone(vdir)" in not_found


def test_no_job_on_a_full_work_disk(monkeypatch, tmp_path):
    import shutil as sh
    from collections import namedtuple

    from whisper_tm.config import config

    monkeypatch.setattr(type(config), "work_dir", property(lambda self: tmp_path))
    usage = namedtuple("usage", "total used free")
    monkeypatch.setattr(sh, "disk_usage", lambda p: usage(100e9, 99.5e9, 0.5e9))
    assert "0.5 GB free in the work folder" in q._work_space_short()
    monkeypatch.setattr(sh, "disk_usage", lambda p: usage(100e9, 50e9, 50e9))
    assert q._work_space_short() == ""
    import inspect

    src = inspect.getsource(q)
    assert src.index("short = _work_space_short()") < src.index(
        "job = jm.create(jid, filename=filename"
    )


def test_ffmpeg_says_what_the_machine_says():
    from whisper_tm.core.audio_utils import ffmpeg_failure, ffmpeg_reason

    out = (
        "Input #0, matroska ...\n[out#0/wav] Error writing trailer: No space left on device\n"
        "[aost#0:0] Error submitting a packet to the muxer\nConversion failed!\n"
    )
    assert ffmpeg_reason(out) == "[out#0/wav] Error writing trailer: No space left on device"
    assert ffmpeg_failure(out) == "[out#0/wav] Error writing trailer: No space left on device"
    assert (
        ffmpeg_reason("a\nInvalid data found when processing input\n")
        == "Invalid data found when processing input"
    )
    assert (
        ffmpeg_failure("x" * 600 + "\nbad").endswith("bad")
        and len(ffmpeg_failure("x" * 600)) == 500
    )
    assert "disk is full" in q.environment_cause("FFmpeg: " + ffmpeg_failure(out)).lower()
