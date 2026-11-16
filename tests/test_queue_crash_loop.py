"""A file that takes the server down isn't run forever.

Every start put the entries that were running back into the queue without
counting. A file that crashed the process (a library's segfault, the
out-of-memory killer) ran first after every restart — and Docker's restart
policy or systemd's Restart= turned that into a loop. Now an entry the server
went down under runs alone next time (so a second crash names it, not the
files beside it), and the second time it is booked as an error.
"""

from __future__ import annotations

import pytest

import whisper_tm.extensions as ext
from whisper_tm.config import config
from whisper_tm.core import queue as q
from whisper_tm.database import Database
from whisper_tm.jobs import JobManager


@pytest.fixture
def env(tmp_path, monkeypatch):
    q.stop_worker()
    media = tmp_path / "media"
    media.mkdir()
    db = Database(tmp_path / "test.db")
    db.initialize()
    monkeypatch.setattr(ext, "db", db)
    monkeypatch.setattr(ext, "job_manager", JobManager(logs_dir=tmp_path / "logs"))
    monkeypatch.setitem(config._data.setdefault("queue", {}), "lanes", 3)
    monkeypatch.setitem(config._data["queue"], "long_file_sec", 3600)
    q._running.clear()
    q._current = None
    q._abort = q._paused = q._stopping = q._shutting_down = False
    q._requeue_on_abort.clear()
    q._solo.clear()
    monkeypatch.setattr(q, "is_transcript_complete", lambda *a, **k: False)
    monkeypatch.setattr(q, "is_skipped", lambda *a, **k: False)
    monkeypatch.setattr(q, "transcript_path_for", lambda rel, **k: tmp_path / "t" / (rel + ".json"))
    monkeypatch.setattr(type(q.config), "media_dir", property(lambda self: media))
    for name in ("a.mp4", "b.mp4", "c.mp4"):
        (media / name).write_bytes(b"x")
    q.enqueue(["a.mp4", "b.mp4", "c.mp4"], {}, str(media))
    ids = {i["filename"]: i["id"] for i in q.list_items("pending")}
    yield {"db": db, "ids": ids}
    q.stop_worker()
    q._running.clear()


def _crash(env, *names):
    """The server goes down while these run: they stay 'processing'."""
    with env["db"].connection() as conn:
        for name in names:
            conn.execute(
                "UPDATE queue_items SET status = 'processing' WHERE id = ?", (env["ids"][name],)
            )
    q._running.clear()
    q._bump()


def _row(env, name):
    with env["db"].connection() as conn:
        return conn.execute(
            "SELECT status, error, interrupted FROM queue_items WHERE id = ?", (env["ids"][name],)
        ).fetchone()


def test_the_second_time_the_server_goes_down_under_it_is_an_error(env):
    _crash(env, "a.mp4")
    assert q.requeue_stale() == 1
    assert tuple(_row(env, "a.mp4")) == ("pending", "", 1)

    _crash(env, "a.mp4")
    assert q.requeue_stale() == 0
    status, error, interrupted = _row(env, "a.mp4")
    assert status == "error" and "went down twice" in error and interrupted == 2

    # The user's retry starts over
    assert q.retry_item(env["ids"]["a.mp4"])[0]
    assert tuple(_row(env, "a.mp4")) == ("pending", "", 0)


def test_an_interrupted_entry_runs_alone(env):
    _crash(env, "a.mp4", "b.mp4")
    assert q.requeue_stale() == 2
    q._bump()

    first = q._claim_next()
    assert first["filename"] == "a.mp4" and first["interrupted"] == 1
    assert q._claim_next() is None  # nothing starts beside it
    q._release(first)

    second = q._claim_next()
    assert second["filename"] == "b.mp4"  # next in line, alone again
    assert q._claim_next() is None
    q._release(second)

    third = q._claim_next()
    assert third["filename"] == "c.mp4" and third["interrupted"] == 0
    q._release(third)


def test_it_waits_for_the_running_ones_when_next_in_line(env):
    """In line behind others it waits until they are done — the files
    after it don't go first."""
    _crash(env, "b.mp4")
    q.requeue_stale()
    q._bump()
    first = q._claim_next()
    assert first["filename"] == "a.mp4"
    assert q._claim_next() is None  # b is next, and a runs
    q._release(first)
    alone = q._claim_next()
    assert alone["filename"] == "b.mp4"
    q._release(alone)
    assert q._claim_next()["filename"] == "c.mp4"


def test_a_clean_shutdown_counts_nothing(env):
    """A shutdown puts its running entries back itself (not 'processing')."""
    item = q._claim_next()
    q._requeue(item["id"])
    q._release(item)
    assert q.requeue_stale() == 0
    assert tuple(_row(env, item["filename"])) == ("pending", "", 0)


def test_a_new_database_has_the_column(tmp_path):
    db = Database(tmp_path / "library.db")
    db.initialize()
    with db.connection() as conn:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(queue_items)")}
    db.close_all()
    assert "interrupted" in cols
