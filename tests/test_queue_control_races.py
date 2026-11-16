"""Queue control under lanes — the races found in a review (core/queue.py).

* An out-of-memory entry that runs while the queue is paused ("run now")
  went back to waiting at once, was taken again at once — and so on,
  forever: a paused queue counted as "stop waiting for VRAM", and the
  second out-of-memory never reached the rule that pauses.
* "Abort" un-paused and woke the lanes before it emptied the queue: a
  stopped queue's lanes took pending entries in between, and those ran to
  the end after a total abort.
* A stop that came while a lane was claiming its next entry did not reach
  that entry: it ran to the end on a paused queue.
* Two enqueues at the same time read the queue before either wrote: the
  second one's "empty queue → start a new run" deleted the first one's
  fresh entries, and the same file went in twice.
* A file queued without a root ('' = the media root) and the same file
  queued with the root spelled out were two entries.
"""

from __future__ import annotations

import contextlib
import threading
import time

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
    q._running.clear()
    q._current = None
    q._abort = q._paused = q._stopping = q._shutting_down = q._aborting = False
    q._requeue_on_abort.clear()
    q._solo.clear()
    q._log.clear()
    monkeypatch.setattr(q, "is_transcript_complete", lambda *a, **k: False)
    monkeypatch.setattr(q, "is_skipped", lambda *a, **k: False)
    transcripts = tmp_path / "transcripts"
    transcripts.mkdir()
    monkeypatch.setattr(q, "transcript_path_for", lambda rel, **k: transcripts / (rel + ".json"))
    monkeypatch.setattr(type(q.config), "media_dir", property(lambda self: media))
    names = [f"f{i:02d}.mp4" for i in range(8)]
    for name in names:
        (media / name).write_bytes(b"x")
    yield {"db": db, "media": media, "names": names}
    q.stop_worker()
    q._running.clear()
    q._solo.clear()
    q._abort = q._paused = q._stopping = q._shutting_down = False


def _until(pred, timeout=10.0):
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(0.02)
    return False


def _statuses():
    return {i["filename"]: i["status"] for i in q.list_items()}


class Runs:
    """transcription_worker stand-in: records the runs, ends as told."""

    def __init__(self, outcome="done", delay=0.05):
        self.outcome, self.delay = outcome, delay
        self.runs: list[str] = []
        self.lock = threading.Lock()

    def __call__(self, job_id, source, filename, **kwargs):
        job = ext.job_manager.get(job_id)
        with self.lock:
            self.runs.append(filename)
        time.sleep(self.delay)
        if job.cancel:
            ext.job_manager.fail(job, "Cancelled")
        elif self.outcome == "oom":
            ext.job_manager.fail(job, "CUDA out of memory. Tried to allocate 2 GiB")
        else:
            ext.job_manager.complete(job, {"processing_time": "1s"})


def _use(monkeypatch, worker):
    monkeypatch.setattr("whisper_tm.core.transcription.transcription_worker", worker)


class TestOutOfMemoryWhilePaused:
    def test_a_run_now_entry_backs_off_and_then_pauses(self, env, monkeypatch):
        from whisper_tm.core import vram

        runs = Runs("oom", delay=0.01)
        _use(monkeypatch, runs)
        monkeypatch.setattr(q, "_unload_models", lambda: None)
        waits = []

        def wait_for_free_vram(min_mb, max_sec, **kw):
            waits.append(min_mb)
            return True, min_mb

        monkeypatch.setattr(vram, "wait_for_free_vram", wait_for_free_vram)
        q.enqueue(env["names"][:2], {}, str(env["media"]))
        q.pause()
        new = q.enqueue(["f05.mp4"], {}, str(env["media"]), front=True)
        assert q.run_now([e["id"] for e in new]) == 1
        q.start_worker()
        assert _until(lambda: q._solo == set() and not q.busy(), timeout=5)
        time.sleep(0.3)
        # Waited for VRAM before the retry, then: second time → stays paused
        assert runs.runs == ["f05.mp4", "f05.mp4"], runs.runs
        assert waits, "the retry did not wait for free VRAM"
        assert q.is_paused() and _statuses()["f05.mp4"] == "pending"
        warnings = ext.job_manager.system_warnings()
        assert any("VRAM" in w["text"] for w in warnings)

    def test_a_stop_still_ends_the_wait(self, env, monkeypatch):
        """The run-now entry no longer counts a pause as "stop waiting" —
        but a stop (which ends run-now) still does."""
        item = {"id": 99, "settings": {}}
        q._paused = True
        q._solo.add(99)
        assert q._oom_wait_over(item, None) is False
        q._solo.clear()
        assert q._oom_wait_over(item, None) is True


class TestAbortOnAStoppedQueue:
    def test_no_lane_takes_an_entry_while_it_is_cleared(self, env, monkeypatch):
        runs = Runs(delay=0.05)
        _use(monkeypatch, runs)
        q.enqueue(env["names"], {}, str(env["media"]))
        q.pause()
        q.start_worker()
        time.sleep(0.2)
        real_clear = q.clear_pending

        def slow_clear():
            time.sleep(0.5)  # what a big queue's DELETE gives the lanes
            return real_clear()

        monkeypatch.setattr(q, "clear_pending", slow_clear)
        q.abort()
        time.sleep(0.5)
        assert runs.runs == [], f"ran after a total abort: {runs.runs}"
        assert q.list_items("pending") == [] and not q.busy()
        assert not q.is_paused()


class TestAnAbortThatFails:
    """The big DELETE can time out on a locked database: the abort must not
    leave its marks behind — after Resume every entry was booked cancelled."""

    def test_the_marks_go_with_the_error(self, env, monkeypatch):
        import sqlite3

        def locked():
            raise sqlite3.OperationalError("database is locked")

        monkeypatch.setattr(q, "clear_pending", locked)
        with pytest.raises(sqlite3.OperationalError):
            q.abort()
        assert q._aborting is False and q._paused is False

    def test_resume_clears_a_left_mark(self, env):
        q._aborting = True
        q.resume()
        assert q._aborting is False
        q.enqueue(env["names"][:1], {}, str(env["media"]))
        item = q._claim_next()
        try:
            assert not item.get("_cancel")
        finally:
            q._release(item)


class TestStopWhileALaneClaims:
    def test_the_entry_claimed_meanwhile_goes_back(self, env, monkeypatch):
        runs = Runs(delay=0.3)
        _use(monkeypatch, runs)
        q.enqueue(env["names"][:1], {}, str(env["media"]))
        real = q._long_file_sec
        fired = []

        def stop_in_the_middle():
            # _claim_next asks this between reading the pause and its UPDATE
            if not fired:
                fired.append(1)
                q.stop()
            return real()

        monkeypatch.setattr(q, "_long_file_sec", stop_in_the_middle)
        monkeypatch.setitem(config._data["queue"], "lanes", 1)
        q.start_worker()
        assert _until(lambda: fired and not q.busy(), timeout=5)
        time.sleep(0.4)
        assert _statuses()["f00.mp4"] == "pending", _statuses()

    def test_a_claim_after_an_abort_is_cancelled(self, env, monkeypatch):
        q.enqueue(env["names"][:1], {}, str(env["media"]))
        q._aborting = True  # abort() is emptying the queue
        try:
            item = q._claim_next()
            assert item["_cancel"] is True and item["id"] not in q._requeue_on_abort
        finally:
            q._aborting = False
            q._release(item)


class TestEnqueueAtTheSameTime:
    def _hooked(self, env, monkeypatch, other):
        """Run ``other`` in a second thread right after this enqueue read the
        queue — the moment the old code wrote nothing yet."""
        db = env["db"]
        real = db.connection
        me = threading.get_ident()
        fired = []

        class Conn:
            def __init__(self, c):
                self._c = c

            def execute(self, sql, *a):
                cur = self._c.execute(sql, *a)
                if (
                    threading.get_ident() == me
                    and not fired
                    and "SELECT id, filename, base_dir, status" in sql
                ):
                    fired.append(1)
                    t = threading.Thread(target=other)
                    t.start()
                    t.join(0.5)
                    fired.append(t)
                return cur

            def __getattr__(self, name):
                return getattr(self._c, name)

        @contextlib.contextmanager
        def connection():
            with real() as c:
                yield Conn(c) if threading.get_ident() == me else c

        monkeypatch.setattr(db, "connection", connection)
        return fired

    def test_a_new_run_does_not_delete_the_other_ones_entries(self, env, monkeypatch):
        media = str(env["media"])
        fired = self._hooked(env, monkeypatch, lambda: q.enqueue(["f01.mp4"], {}, media))
        q.enqueue(["f00.mp4"], {}, media)
        fired[1].join(10)
        assert sorted(_statuses()) == ["f00.mp4", "f01.mp4"]

    def test_the_same_file_goes_in_once(self, env, monkeypatch):
        media = str(env["media"])
        q.enqueue(["f07.mp4"], {}, media)  # the run isn't new
        fired = self._hooked(env, monkeypatch, lambda: q.enqueue(["f00.mp4"], {}, media))
        q.enqueue(["f00.mp4"], {}, media)
        fired[1].join(10)
        assert [i["filename"] for i in q.list_items()].count("f00.mp4") == 1


class TestEnqueueReadsOnlyWhatItAsksAbout:
    """The write lock is held from the first read on: reading every waiting
    entry for the duplicate check (tens of thousands in a big queue) held it for each
    autoscan enqueue. Only the requested names are looked up."""

    def test_one_file_into_a_big_queue(self, env, monkeypatch):
        with env["db"].connection() as conn:
            conn.executemany(
                "INSERT INTO queue_items (filename, base_dir, settings_json, status, "
                "position) VALUES (?, ?, '{}', 'pending', ?)",
                [(f"old/{i}.mp4", str(env["media"]), i) for i in range(3000)],
            )
        db = env["db"]
        real = db.connection
        read = []

        class Rows(list):
            def fetchone(self):
                return self[0] if self else None

            def fetchall(self):
                return list(self)

        class Conn:
            def __init__(self, c):
                self._c = c

            def execute(self, sql, *a):
                cur = self._c.execute(sql, *a)
                if sql.lstrip().upper().startswith("SELECT"):
                    rows = Rows(cur.fetchall())
                    read.append(len(rows))
                    return rows
                return cur

            def __getattr__(self, name):
                return getattr(self._c, name)

        @contextlib.contextmanager
        def connection():
            with real() as c:
                yield Conn(c)

        monkeypatch.setattr(db, "connection", connection)
        assert q.enqueue(["f00.mp4"], {}, str(env["media"]))
        assert q.enqueue(["f00.mp4"], {}, str(env["media"])) == []  # still a duplicate
        monkeypatch.setattr(db, "connection", real)
        assert max(read) < 10, read


class TestTheMediaRootSpelledTwoWays:
    def test_no_second_entry(self, env):
        assert q.enqueue(["f00.mp4"], {}, None)
        assert q.enqueue(["f00.mp4"], {}, str(env["media"])) == []
        assert len(q.list_items()) == 1

    def test_the_library_selection_sees_it_queued(self, env):
        from whisper_tm.core import library_select as ls
        from whisper_tm.core import paths

        q.enqueue(["f00.mp4"], {}, None)
        with env["db"].connection() as conn:
            queued, _failed = ls._queue_sets(conn)
        assert paths.canon(env["media"] / "f00.mp4") in queued


class TestEnglishOnly:
    def test_moving_says_so_in_english(self, env):
        q.enqueue(env["names"][:2], {}, str(env["media"]))
        first = q.list_items("pending")[0]
        assert q.move_item(first["id"], 1) == (True, "Moved")

    def test_the_overlap_examples_run(self):
        import doctest

        finder = doctest.DocTestFinder()
        runner = doctest.DocTestRunner()
        for t in finder.find(
            q._trim_overlap, "_trim_overlap", globs={"_trim_overlap": q._trim_overlap}
        ):
            runner.run(t)
        assert runner.failures == 0 and runner.tries == 2
