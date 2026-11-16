"""
Tests for core/queue.py — the persistent server-side queue.

Core promise secured here: The queue runs WITHOUT client intervention.
No test here sends an HTTP request to start the next job — the worker
fetches the work itself.
"""

import json
import threading
import time

import pytest

import whisper_tm.extensions as ext
from whisper_tm.core import queue as q
from whisper_tm.database import Database
from whisper_tm.jobs import JobManager


@pytest.fixture
def queue_env(tmp_path, monkeypatch):
    """Fresh DB + JobManager + empty worker state."""
    media = tmp_path / "media"
    media.mkdir()

    # Stop a worker from an earlier test (e.g. started via create_app) first
    # — otherwise it grabs the entries from this test.
    q.stop_worker()
    # One lane: these tests pin the order and control of one worker
    # (tests/test_queue_lanes.py covers more), whatever config an earlier
    # test left loaded.
    monkeypatch.setitem(q.config._data.setdefault("queue", {}), "lanes", 1)
    q._running.clear()

    db = Database(tmp_path / "test.db")
    db.initialize()
    monkeypatch.setattr(ext, "db", db)
    monkeypatch.setattr(ext, "job_manager", JobManager(logs_dir=tmp_path / "logs"))

    # Reset worker globals between tests
    q._current = None
    q._abort = False
    q._paused = False
    q._stopping = False
    q._shutting_down = False
    q._requeue_on_abort.clear()
    q._solo.clear()
    q._log.clear()

    # Neutralize filesystem dependencies: the file always "exists",
    # never has a transcript and is on no skip list.
    transcripts = tmp_path / "transcripts"
    transcripts.mkdir()
    monkeypatch.setattr(q, "is_transcript_complete", lambda *a, **k: False)
    monkeypatch.setattr(q, "is_skipped", lambda *a, **k: False)
    monkeypatch.setattr(q, "transcript_path_for", lambda rel, **k: transcripts / (rel + ".json"))
    # media_dir is a property — replace on the class, not on the object
    monkeypatch.setattr(type(q.config), "media_dir", property(lambda self: media))

    for name in ("a.mp4", "b.mp4", "c.mp4"):
        (media / name).write_bytes(b"x")

    yield {"db": db, "media": media}

    q.stop_worker()


def _fake_worker(processed, delay=0.0, fail_on=None):
    """Replacement for transcription_worker: marks the job as done."""

    def run(job_id, source, filename, **kwargs):
        if delay:
            time.sleep(delay)
        job = ext.job_manager.get(job_id)
        processed.append(filename)
        if fail_on and filename in fail_on:
            ext.job_manager.fail(job, "Test error")
        else:
            ext.job_manager.complete(job, {"processing_time": "1s"})

    return run


def _wait_until(pred, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if pred():
            return True
        time.sleep(0.05)
    return False


def _wait_for_item(filename, timeout=10.0):
    """Get entry as soon as the worker doesn't have it anymore."""
    _wait_until(
        lambda: _item(filename) is not None and _item(filename)["status"] != "processing", timeout
    )
    item = _item(filename)
    assert item is not None, f"{filename} is no longer in the queue"
    return item


def _item(filename):
    for i in q.list_items():
        if i["filename"] == filename:
            return i
    return None


# ── Persistence ───────────────────────────────────────────────────────────


def test_enqueue_persists_to_db(queue_env):
    """Queued files are in the DB — not in the browser."""
    added = q.enqueue(["a.mp4", "b.mp4"], {"model": "turbo"}, str(queue_env["media"]))
    assert len(added) == 2

    items = q.list_items("pending")
    assert [i["filename"] for i in items] == ["a.mp4", "b.mp4"]
    assert items[0]["settings"]["model"] == "turbo"


def test_enqueue_skips_duplicates(queue_env):
    q.enqueue(["a.mp4"], {}, str(queue_env["media"]))
    added = q.enqueue(["a.mp4", "b.mp4"], {}, str(queue_env["media"]))
    assert [i["filename"] for i in added] == ["b.mp4"]
    assert len(q.list_items("pending")) == 2


def test_requeue_stale_resumes_after_restart(queue_env):
    """An entry interrupted during server stop is re-queued."""
    q.enqueue(["a.mp4"], {}, str(queue_env["media"]))
    item_id = q.list_items("pending")[0]["id"]

    # Server dies in the middle of processing. We deliberately write
    # bypassing the queue API (real crash doesn't have clean calls) —
    # so the cache must be discarded manually. During normal operation
    # every mutation does it itself; queue.py is the only writer.
    with queue_env["db"].connection() as conn:
        conn.execute("UPDATE queue_items SET status='processing' WHERE id=?", (item_id,))
    q._bump()
    assert q.list_items("pending") == []

    # Restart
    assert q.requeue_stale() == 1
    assert [i["filename"] for i in q.list_items("pending")] == ["a.mp4"]


# ── The worker runs without a client ─────────────────────────────────────────


def test_worker_processes_whole_queue_without_any_client(queue_env, monkeypatch):
    """THE core test: Three files, one enqueue call, no further push.

    Previously the frontend had to push the next job after each one
    (processNextQueued). Here nobody calls anything — the worker
    empties the queue on its own.
    """
    processed = []
    monkeypatch.setattr(
        "whisper_tm.core.transcription.transcription_worker", _fake_worker(processed)
    )

    q.enqueue(["a.mp4", "b.mp4", "c.mp4"], {}, str(queue_env["media"]))
    q.start_worker()

    assert _wait_until(lambda: len(processed) == 3), f"only {processed} processed"
    assert processed == ["a.mp4", "b.mp4", "c.mp4"]

    snap = q.snapshot()
    assert snap["done_count"] == 3
    assert snap["pending"] == 0
    assert snap["running"] is False


def test_worker_picks_up_items_enqueued_while_idle(queue_env, monkeypatch):
    """Worker sleeps on empty queue and wakes up when items are enqueued."""
    processed = []
    monkeypatch.setattr(
        "whisper_tm.core.transcription.transcription_worker", _fake_worker(processed)
    )

    q.start_worker()
    assert _wait_until(lambda: q.snapshot()["running"] is False)

    q.enqueue(["a.mp4"], {}, str(queue_env["media"]))
    assert _wait_until(lambda: processed == ["a.mp4"])


def test_failed_job_does_not_stop_the_queue(queue_env, monkeypatch):
    """An error on file 1 must not block files 2 and 3."""
    processed = []
    monkeypatch.setattr(
        "whisper_tm.core.transcription.transcription_worker",
        _fake_worker(processed, fail_on={"a.mp4"}),
    )

    q.enqueue(["a.mp4", "b.mp4", "c.mp4"], {}, str(queue_env["media"]))
    q.start_worker()

    assert _wait_until(lambda: len(processed) == 3)
    snap = q.snapshot()
    assert snap["fail_count"] == 1
    assert snap["done_count"] == 2


def test_missing_file_is_marked_error_not_hung(queue_env, monkeypatch):
    processed = []
    monkeypatch.setattr(
        "whisper_tm.core.transcription.transcription_worker", _fake_worker(processed)
    )

    q.enqueue(["does_not_exist.mp4", "a.mp4"], {}, str(queue_env["media"]))
    q.start_worker()

    assert _wait_until(lambda: processed == ["a.mp4"])
    items = {i["filename"]: i for i in q.list_items()}
    assert items["does_not_exist.mp4"]["status"] == "error"
    assert items["a.mp4"]["status"] == "done"


# ── Control ───────────────────────────────────────────────────────────────


def test_pause_holds_the_queue(queue_env, monkeypatch):
    processed = []
    monkeypatch.setattr(
        "whisper_tm.core.transcription.transcription_worker", _fake_worker(processed)
    )

    q.pause()
    q.enqueue(["a.mp4"], {}, str(queue_env["media"]))
    q.start_worker()

    time.sleep(0.5)
    assert processed == []  # paused → nothing runs
    assert q.is_paused() is True

    q.resume()
    assert _wait_until(lambda: processed == ["a.mp4"])


def test_remove_and_move_reorder_pending_items(queue_env):
    q.enqueue(["a.mp4", "b.mp4", "c.mp4"], {}, str(queue_env["media"]))
    items = q.list_items("pending")
    ids = {i["filename"]: i["id"] for i in items}

    ok, _ = q.move_item(ids["c.mp4"], -1)
    assert ok
    assert [i["filename"] for i in q.list_items("pending")] == ["a.mp4", "c.mp4", "b.mp4"]

    ok, _ = q.remove_item(ids["a.mp4"])
    assert ok
    assert [i["filename"] for i in q.list_items("pending")] == ["c.mp4", "b.mp4"]


def test_move_beyond_edge_is_rejected(queue_env):
    q.enqueue(["a.mp4", "b.mp4"], {}, str(queue_env["media"]))
    first = q.list_items("pending")[0]["id"]
    ok, msg = q.move_item(first, -1)
    assert not ok and "edge" in msg


# ── "Process this file now" ──────────────────────────────────────────────────
#
# ▲ moves exactly one position. With a huge queue waiting, that's no progress —
# move_to_top() jumps.


def _api_client():
    """Minimal app with just the Queue blueprint.

    Deliberately NOT create_app(): that accesses the real database and
    starts a second worker alongside the running app.
    """
    from flask import Flask

    from whisper_tm.api.queue import queue_bp

    app = Flask(__name__)
    app.register_blueprint(queue_bp)
    return app.test_client()


def test_move_to_top_jumps_over_everything_pending(queue_env):
    q.enqueue([f"d{i}.mp4" for i in range(50)], {}, str(queue_env["media"]))
    ids = {i["filename"]: i["id"] for i in q.list_items("pending")}

    ok, _ = q.move_to_top(ids["d42.mp4"])

    assert ok
    assert q.list_items("pending")[0]["filename"] == "d42.mp4"


def test_move_to_top_leaves_the_remaining_order_alone(queue_env):
    """Only one entry moves — the remainder must not get scrambled
    (and a huge queue's rows must not be renumbered)."""
    q.enqueue([f"d{i}.mp4" for i in range(10)], {}, str(queue_env["media"]))
    ids = {i["filename"]: i["id"] for i in q.list_items("pending")}

    q.move_to_top(ids["d7.mp4"])

    names = [i["filename"] for i in q.list_items("pending")]
    assert names[0] == "d7.mp4"
    assert names[1:] == [f"d{i}.mp4" for i in range(10) if i != 7]


def test_move_to_top_twice_puts_the_last_choice_first(queue_env):
    """Two clicks in a row: the last one moved to front stands first."""
    q.enqueue(["a.mp4", "b.mp4", "c.mp4"], {}, str(queue_env["media"]))
    ids = {i["filename"]: i["id"] for i in q.list_items("pending")}

    q.move_to_top(ids["b.mp4"])
    q.move_to_top(ids["c.mp4"])

    assert [i["filename"] for i in q.list_items("pending")] == ["c.mp4", "b.mp4", "a.mp4"]


def test_move_to_top_on_the_first_one_is_no_error(queue_env):
    q.enqueue(["a.mp4", "b.mp4"], {}, str(queue_env["media"]))
    first = q.list_items("pending")[0]["id"]

    ok, msg = q.move_to_top(first)

    assert ok and ("front" in msg or "front" in msg)
    assert [i["filename"] for i in q.list_items("pending")] == ["a.mp4", "b.mp4"]


def test_move_to_top_does_not_touch_the_running_entry(queue_env):
    """Moving it to front would mean cancelling a model run to reach a spot already occupied."""
    q.enqueue(["running.mp4", "b.mp4"], {}, str(queue_env["media"]))
    with queue_env["db"].connection() as conn:
        conn.execute("UPDATE queue_items SET status='processing' WHERE filename='running.mp4'")
        iid = conn.execute("SELECT id FROM queue_items WHERE filename='running.mp4'").fetchone()[
            "id"
        ]

    ok, msg = q.move_to_top(iid)

    assert not ok and ("Already running" in msg or "already running" in msg)
    with queue_env["db"].connection() as conn:
        r = conn.execute("SELECT status FROM queue_items WHERE id = ?", (iid,)).fetchone()
    assert r["status"] == "processing"


def test_move_to_top_does_not_wake_a_paused_queue(queue_env):
    """Reordering is not a command. Whoever paused the queue doesn't want it to resume
    from a sort click."""
    q.enqueue(["a.mp4", "b.mp4"], {}, str(queue_env["media"]))
    q.pause()
    second = q.list_items("pending")[1]["id"]

    q.move_to_top(second)

    assert q.is_paused() is True


def test_move_to_top_discards_the_cache(queue_env):
    """The SSE stream reads the queue from a generation cache — without
    _bump() the UI would show the old order."""
    q.enqueue(["a.mp4", "b.mp4", "c.mp4"], {}, str(queue_env["media"]))
    q.list_items("pending")  # fill the cache
    ids = {i["filename"]: i["id"] for i in q.list_items("pending")}

    q.move_to_top(ids["c.mp4"])

    assert [i["filename"] for i in q.list_items("pending")][0] == "c.mp4"


def test_api_move_top_moves_it_up(queue_env):
    q.enqueue(["a.mp4", "b.mp4", "c.mp4"], {}, str(queue_env["media"]))
    ids = {i["filename"]: i["id"] for i in q.list_items("pending")}

    r = _api_client().post("/api/queue/move", json={"id": ids["c.mp4"], "top": True})

    assert r.status_code == 200 and r.get_json()["ok"] is True
    assert q.list_items("pending")[0]["filename"] == "c.mp4"


def test_api_move_top_needs_no_direction(queue_env):
    """`top` without `direction` must not fail the parameter check."""
    q.enqueue(["a.mp4", "b.mp4"], {}, str(queue_env["media"]))
    ids = {i["filename"]: i["id"] for i in q.list_items("pending")}

    r = _api_client().post("/api/queue/move", json={"id": ids["b.mp4"], "top": True})

    assert r.status_code == 200


def test_api_move_without_top_stays_step_by_step(queue_env):
    """Counter-check: ▲/▼ must not become jumps due to the new branching."""
    q.enqueue([f"d{i}.mp4" for i in range(5)], {}, str(queue_env["media"]))
    ids = {i["filename"]: i["id"] for i in q.list_items("pending")}

    r = _api_client().post("/api/queue/move", json={"id": ids["d4.mp4"], "direction": -1})

    assert r.status_code == 200
    assert [i["filename"] for i in q.list_items("pending")] == [
        "d0.mp4",
        "d1.mp4",
        "d2.mp4",
        "d4.mp4",
        "d3.mp4",
    ]


def test_a_stale_reader_does_not_overwrite_the_fresh_pending_list(queue_env, monkeypatch):
    """Two concurrent readers, a change in between.

    The slow reader (A) fetched its list BEFORE the change. If it then stores
    it under the NEW generation in the cache, the UI continues showing
    the old order — until something else changes the queue.
    For move-to-top this is fatal: "at the front" wouldn't be visible.

    This case is real: the SSE stream reads the queue every second per
    open tab, in parallel with every request.
    """
    q.enqueue(["a.mp4"], {}, str(queue_env["media"]))

    inside = threading.Event()
    onward = threading.Event()
    real = q._row_to_item

    def slow(row):
        inside.set()
        onward.wait(5)
        return real(row)

    monkeypatch.setattr(q, "_row_to_item", slow)
    result = {}
    t = threading.Thread(target=lambda: result.update(listing=q.list_items("pending", limit=200)))
    t.start()
    try:
        assert inside.wait(5), "Reader A never read"
        monkeypatch.setattr(q, "_row_to_item", real)  # B reads normally

        q.enqueue(["new_value.mp4"], {}, str(queue_env["media"]), front=True)
        # Reader B fetches fresh state and stores it.
        assert q.list_items("pending", limit=200)[0]["filename"] == "new_value.mp4"
    finally:
        onward.set()
        t.join(10)

    names = [i["filename"] for i in q.list_items("pending", limit=200)]
    assert names[0] == "new_value.mp4", f"stale list from cache: {names}"


def test_api_move_top_on_an_unknown_id(queue_env):
    r = _api_client().post("/api/queue/move", json={"id": 999999, "top": True})
    assert r.status_code == 400
    assert "not found" in r.get_json()["error"]


def _cancellable(started, cancelled):
    """Worker that responds to job.cancel."""

    def run(job_id, source, filename, **kwargs):
        job = ext.job_manager.get(job_id)
        started.set()
        for _ in range(150):
            if job.cancel:
                cancelled.append(filename)
                return  # Job remains 'processing' → cancelled
            time.sleep(0.02)
        ext.job_manager.complete(job, {})

    return run


def test_stop_pauses_but_KEEPS_the_queue(queue_env, monkeypatch):
    """Regression: stopping must NOT delete the queue.

    Previously stop() called clear_pending(). Because __main__._shutdown()
    triggered exactly that via batch_stop(), a simple SIGTERM permanently
    deleted the whole persistent queue from the DB.
    """
    started, cancelled = threading.Event(), []
    monkeypatch.setattr(
        "whisper_tm.core.transcription.transcription_worker", _cancellable(started, cancelled)
    )

    q.enqueue(["a.mp4", "b.mp4", "c.mp4"], {}, str(queue_env["media"]))
    q.start_worker()
    assert started.wait(5)

    q.stop()  # Default: do NOT delete
    assert _wait_until(lambda: cancelled == ["a.mp4"])

    assert q.is_paused() is True
    # b and c must still be there — that's exactly what was lost before.
    # a.mp4 (was running) comes back to its place: stopping means halting,
    # not discarding.
    assert _wait_until(
        lambda: [i["filename"] for i in q.list_items("pending")] == ["a.mp4", "b.mp4", "c.mp4"]
    )


def test_stop_requeues_the_running_item(queue_env, monkeypatch):
    """Regression: the running entry must not burn when stopping.

    Previously it landed as 'error' ("Cancelled") — whoever paused the queue briefly
    lost exactly the file that was running and had no way to get it back
    without a retry route.
    """
    started, cancelled = threading.Event(), []
    monkeypatch.setattr(
        "whisper_tm.core.transcription.transcription_worker", _cancellable(started, cancelled)
    )

    q.enqueue(["a.mp4", "b.mp4"], {}, str(queue_env["media"]))
    q.start_worker()
    assert started.wait(5)

    q.stop()
    assert _wait_until(lambda: cancelled == ["a.mp4"])

    a = _wait_for_item("a.mp4")
    assert a["status"] == "pending", f"a.mp4 is {a['status']}, expected pending"
    assert a["error"] == ""
    snap = q.snapshot()
    assert snap["fail_count"] == 0
    assert snap["cancel_count"] == 0


def test_abort_cancels_running_and_empties_queue(queue_env, monkeypatch):
    """Total abort: waiting entries are deleted, the running job
    remains as 'cancelled' in history.

    Previously the running entry was also silently deleted — then it was no longer
    clear in the file list that processing was aborted.
    Now the same treatment applies as with targeted individual abort: visible + requeueable by retry.
    """
    started, cancelled = threading.Event(), []
    monkeypatch.setattr(
        "whisper_tm.core.transcription.transcription_worker", _cancellable(started, cancelled)
    )

    q.enqueue(["a.mp4", "b.mp4", "c.mp4"], {}, str(queue_env["media"]))
    q.start_worker()
    assert started.wait(5)  # a.mp4 running

    ok, removed = q.abort()
    assert ok and removed == 2  # b + c (pending) gone immediately
    assert _wait_until(lambda: cancelled == ["a.mp4"])

    # a.mp4 remains as 'cancelled' — visible and requeueable
    assert _wait_until(lambda: (_item("a.mp4") or {}).get("status") == "cancelled")
    snap = q.snapshot()
    assert snap["pending"] == 0
    assert snap["cancel_count"] == 1
    assert snap["fail_count"] == 0
    assert snap["retryable_count"] == 1
    assert _wait_until(lambda: q.snapshot()["running"] is False)
    assert q._paused is False  # no pause state — clean start


def test_abort_while_paused_clears_all_pending(queue_env):
    """Abort without running job: all waiting entries disappear."""
    q.enqueue(["a.mp4", "b.mp4", "c.mp4"], {}, str(queue_env["media"]))
    q.pause()

    ok, removed = q.abort()
    assert ok and removed == 3
    assert q.list_items("pending") == []
    assert q.snapshot()["total"] == 0
    assert q._paused is False


def test_targeted_cancel_marks_cancelled_not_error(queue_env, monkeypatch):
    """A targeted job cancellation (/api/cancel) is not a failure.

    It discards the entry ('cancelled'), but does not count as an error — and
    remains requeueable by retry.
    """
    started, cancelled = threading.Event(), []
    monkeypatch.setattr(
        "whisper_tm.core.transcription.transcription_worker", _cancellable(started, cancelled)
    )

    q.enqueue(["a.mp4"], {}, str(queue_env["media"]))
    q.start_worker()
    assert started.wait(5)

    # This is exactly what the route /api/cancel/<job_id> does
    cur = q.snapshot()["current_job_id"]
    ext.job_manager.get(cur).cancel = True
    assert _wait_until(lambda: cancelled == ["a.mp4"])

    a = _wait_for_item("a.mp4")
    assert a["status"] == "cancelled"
    snap = q.snapshot()
    assert snap["fail_count"] == 0
    assert snap["retryable_count"] == 1

    # …and the path back to the queue is open. The worker is still
    # running and can grab the entry again immediately (and even finish
    # processing it) — the only thing that matters is: it is NOT 'cancelled' anymore.
    ok, _ = q.retry_item(a["id"])
    assert ok
    assert _item("a.mp4")["status"] in ("pending", "processing", "done")


def test_retry_failed_requeues_everything(queue_env):
    """retry_failed() retrieves failed AND cancelled entries."""
    items = q.enqueue(["a.mp4", "b.mp4", "c.mp4"], {}, str(queue_env["media"]))
    q._finish(items[0]["id"], "error", "Boom")
    q._finish(items[1]["id"], "cancelled", "Cancelled")
    q._finish(items[2]["id"], "done")

    assert q.retry_failed() == 2

    by_name = {i["filename"]: i for i in q.list_items()}
    assert by_name["a.mp4"]["status"] == "pending"
    assert by_name["a.mp4"]["error"] == ""  # old error message is gone
    assert by_name["b.mp4"]["status"] == "pending"
    assert by_name["c.mp4"]["status"] == "done"  # Completed stays completed


def test_retry_items_requeues_only_named_ids(queue_env):
    """retry_items() (filtered retry from the quick filters) touches only
    the named entries — and among those, only failed/cancelled ones."""
    items = q.enqueue(["a.mp4", "b.mp4", "c.mp4", "d.mp4"], {}, str(queue_env["media"]))
    q._finish(items[0]["id"], "error", "Boom")
    q._finish(items[1]["id"], "cancelled", "Cancelled")
    q._finish(items[2]["id"], "error", "Boom 2")
    q._finish(items[3]["id"], "done")

    # a + b (+ d, but that is finished and gets ignored) — c stays untouched
    assert q.retry_items([items[0]["id"], items[1]["id"], items[3]["id"]]) == 2

    by_name = {i["filename"]: i for i in q.list_items()}
    assert by_name["a.mp4"]["status"] == "pending"
    assert by_name["b.mp4"]["status"] == "pending"
    assert by_name["c.mp4"]["status"] == "error"  # not named → untouched
    assert by_name["d.mp4"]["status"] == "done"
    assert q.retry_items([]) == 0


def test_remove_pending_for_files_that_vanished(queue_env):
    """Walk cleanup: pending entries for files that have vanished get
    removed; everything else (other files, already completed ones) stays."""
    media = str(queue_env["media"])
    items = q.enqueue(["gone.mp4", "stays.mp4", "done.mp4"], {}, media)
    q._finish(items[2]["id"], "done")

    n = q.remove_pending_for([f"{media}/gone.mp4", f"{media}/never-existed.mp4"])
    assert n == 1
    by_name = {i["filename"]: i["status"] for i in q.list_items()}
    assert "gone.mp4" not in by_name
    assert by_name["stays.mp4"] == "pending"
    assert by_name["done.mp4"] == "done"


def test_retry_item_rejects_finished_entries(queue_env):
    """Finished or waiting entries are not a retry case."""
    items = q.enqueue(["a.mp4", "b.mp4"], {}, str(queue_env["media"]))
    q._finish(items[0]["id"], "done")

    ok, msg = q.retry_item(items[0]["id"])
    assert not ok and "failed" in msg
    ok, _ = q.retry_item(items[1]["id"])  # still pending
    assert not ok
    ok, msg = q.retry_item(999999)
    assert not ok and "not found" in msg


def test_shutdown_requeues_running_item_and_keeps_pending(queue_env, monkeypatch):
    """Shutting down must preserve the queue completely.

    The running entry goes back to 'pending' (runs again on the next
    start), the waiting ones stay untouched.
    """
    started, cancelled = threading.Event(), []
    monkeypatch.setattr(
        "whisper_tm.core.transcription.transcription_worker", _cancellable(started, cancelled)
    )

    q.enqueue(["a.mp4", "b.mp4", "c.mp4"], {}, str(queue_env["media"]))
    q.start_worker()
    assert started.wait(5)

    q.shutdown(timeout=10)  # like SIGTERM / Ctrl+C

    items = q.list_items()
    assert [i["filename"] for i in items] == ["a.mp4", "b.mp4", "c.mp4"]
    assert all(i["status"] == "pending" for i in items), (
        f"not everything pending after shutdown: {[(i['filename'], i['status']) for i in items]}"
    )

    q._shutting_down = False


def test_clear_is_the_only_thing_that_deletes(queue_env):
    """Only the explicit clear_pending() call may remove entries."""
    q.enqueue(["a.mp4", "b.mp4"], {}, str(queue_env["media"]))
    assert q.clear_pending() == 2
    assert q.list_items("pending") == []


def test_start_worker_after_stop_worker_works_again(queue_env, monkeypatch):
    """stop_worker() must not permanently disable start_worker().

    Previously _stopping=True remained set if the old thread was still
    alive on restart — start_worker() then returned immediately and no
    worker ran.
    """
    processed = []
    monkeypatch.setattr(
        "whisper_tm.core.transcription.transcription_worker", _fake_worker(processed)
    )

    q.start_worker()
    q.stop_worker()
    q.start_worker()  # must set up a worker again

    q.enqueue(["a.mp4"], {}, str(queue_env["media"]))
    assert _wait_until(lambda: processed == ["a.mp4"])


def test_start_worker_is_idempotent(queue_env, monkeypatch):
    processed = []
    monkeypatch.setattr(
        "whisper_tm.core.transcription.transcription_worker", _fake_worker(processed)
    )

    q.start_worker()
    q.start_worker()
    q.start_worker()

    q.enqueue(["a.mp4"], {}, str(queue_env["media"]))
    assert _wait_until(lambda: processed == ["a.mp4"])
    time.sleep(0.3)
    assert processed == ["a.mp4"]  # not processed more than once


# ── Via the HTTP API (the way the WebUI does it) ──────────────────────────


def test_api_enqueue_then_worker_runs_without_further_requests(tmp_path, monkeypatch):
    """Complete run through the real API.

    The client makes EXACTLY ONE request (/api/queue/add) and is then
    gone — as if the user had closed the browser. All three files must
    still be processed.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    media = tmp_path / "media"
    media.mkdir()
    for name in ("a.mp4", "b.mp4", "c.mp4"):
        (media / name).write_bytes(b"x")

    from whisper_tm.config import config as cfg

    q.stop_worker()
    cfg.load(None)
    cfg._data.setdefault("directories", {})["media"] = str(media)
    cfg._resolve_directories()
    cfg._ensure_directories()
    # This test checks exactly the autostart behavior — regardless of what
    # is in the developer's config.yaml.
    cfg.set("queue", "autostart", True)

    processed = []
    monkeypatch.setattr(
        "whisper_tm.core.transcription.transcription_worker", _fake_worker(processed)
    )
    monkeypatch.setattr(q, "is_transcript_complete", lambda *a, **k: False)
    monkeypatch.setattr(q, "is_skipped", lambda *a, **k: False)

    from whisper_tm.app import create_app

    client = create_app().test_client()  # starts the worker

    r = client.post(
        "/api/queue/add",
        json={
            "filenames": ["a.mp4", "b.mp4", "c.mp4"],
            "model": "turbo",
            "language": "de",
        },
    )
    assert r.status_code == 200 and r.get_json()["added"] == 3

    # From here on NO more client contact — the worker must finish on its own.
    try:
        assert _wait_until(lambda: len(processed) == 3), f"only {processed} processed"
        assert processed == ["a.mp4", "b.mp4", "c.mp4"]

        state = client.get("/api/queue").get_json()
        assert state["done_count"] == 3
        assert state["pending"] == 0
        assert state["running"] is False
    finally:
        q.stop_worker()


def test_clear_pending_keeps_running_job(queue_env, monkeypatch):
    """Clearing the queue does not cancel the RUNNING file."""
    started = threading.Event()
    release = threading.Event()
    processed = []

    def slow(job_id, source, filename, **kwargs):
        started.set()
        release.wait(5)
        processed.append(filename)
        ext.job_manager.complete(ext.job_manager.get(job_id), {})

    monkeypatch.setattr("whisper_tm.core.transcription.transcription_worker", slow)

    q.enqueue(["a.mp4", "b.mp4", "c.mp4"], {}, str(queue_env["media"]))
    q.start_worker()
    assert started.wait(5)

    assert q.clear_pending() == 2  # b + c gone, a keeps running
    release.set()

    assert _wait_until(lambda: processed == ["a.mp4"])
    assert q.snapshot()["done_count"] == 1


# ── Nothing may go off the rails without a watching browser ──────────────


def test_live_buffers_stay_bounded_without_a_reader(tmp_path):
    """The SSE buffers must not grow unbounded without an open WebUI.

    new_log/new_segments are drained ONLY by a reading SSE client. Since
    the queue keeps running even without a browser, a server that nobody
    watches for days would otherwise fill them up to the memory limit.
    """
    from whisper_tm.jobs import MAX_NEW_LOG, MAX_NEW_SEGMENTS, JobManager

    jm = JobManager(logs_dir=tmp_path)
    job = jm.create("j1", filename="long.mp4")

    for i in range(MAX_NEW_LOG * 3):
        jm.log(job, f"Line {i}")
    for i in range(MAX_NEW_SEGMENTS * 3):
        jm.add_live_segments(job, [{"text": f"seg {i}", "start_fmt": "0:00", "end_fmt": "0:01"}])

    assert len(job.new_log) <= MAX_NEW_LOG
    assert len(job.new_segments) <= MAX_NEW_SEGMENTS
    # The MOST RECENT entries must be preserved (ring buffer, not truncation)
    assert job.new_log[-1].endswith(f"Line {MAX_NEW_LOG * 3 - 1}")
    assert job.new_segments[-1]["text"] == f"seg {MAX_NEW_SEGMENTS * 3 - 1}"


def test_long_running_job_is_not_reaped_as_stale(tmp_path):
    """A long but alive job must not be cleaned up.

    The cleanup used to go by start_time — a transcription that takes
    longer than _stale_max_age (large-v3 + vocal isolation on a
    multi-hour recording) was therefore removed from the JobManager
    MID-RUN. Afterward, the queue worker could no longer find a job
    object and marked the entry as "cancelled".
    """
    from whisper_tm.jobs import JobManager

    jm = JobManager(logs_dir=tmp_path, max_age_secs=1)  # _stale_max_age = 6s
    job = jm.create("long", filename="3h_video.mp4")
    job.status = "processing"

    # Job has been running for 10 minutes — but just reported progress
    job.start_time = time.time() - 600
    job.update(progress=42)  # sets last_activity

    assert jm.cleanup() == 0, "a live job was cleaned up"
    assert jm.get("long") is not None

    # No more sign of life → worker is dead → may be removed
    job.last_activity = time.time() - 600
    assert jm.cleanup() == 1
    assert jm.get("long") is None


# ── Autostart can be turned off ────────────────────────────────────────────


def test_autostart_false_starts_paused_and_keeps_the_queue(queue_env, monkeypatch):
    """queue.autostart: false → worker runs, but processes nothing.

    The queue stays fully intact; processing only starts after an
    explicit resume() ("Resume" in the WebUI).
    """
    processed = []
    monkeypatch.setattr(
        "whisper_tm.core.transcription.transcription_worker", _fake_worker(processed)
    )

    q.enqueue(["a.mp4", "b.mp4"], {}, str(queue_env["media"]))

    # This is what create_app() does with autostart: false
    q.pause()
    q.start_worker()

    time.sleep(0.6)
    assert processed == [], "processing happened despite autostart:false"
    assert q.is_paused() is True
    assert len(q.list_items("pending")) == 2, "queue was touched"

    # Only clicking "Resume" gets it going
    q.resume()
    assert _wait_until(lambda: processed == ["a.mp4", "b.mp4"])


# ── Cache: saves DB accesses without serving stale data ───────────────────


def test_cache_avoids_a_db_hit_when_nothing_changed(queue_env, monkeypatch):
    """The SSE stream polls every second. Without a cache that was a DB
    transaction every time — thousands of them within half an hour,
    just for watching."""
    q.enqueue(["a.mp4", "b.mp4"], {}, str(queue_env["media"]))

    reads = []
    real_conn = queue_env["db"].connection

    def counting_conn():
        reads.append(1)
        return real_conn()

    monkeypatch.setattr(queue_env["db"], "connection", counting_conn)

    # Per generation, every kind of query is read ONCE, after that
    # everything comes from the cache. snapshot() and the waiting list are
    # kept separate here: snapshot() deliberately no longer builds the list
    # — it used to be fully materialized (as many dicts as there were
    # entries, hundreds of thousands), even though snapshot() never touches it.
    q.snapshot()
    q.list_items("pending")
    after_first = len(reads)

    for _ in range(20):
        q.snapshot()
        q.list_items("pending")

    assert len(reads) == after_first, (
        f"{len(reads) - after_first} unnecessary DB accesses despite unchanged queue"
    )


def test_limited_pending_query_does_not_build_the_whole_list(queue_env, monkeypatch):
    """The stream shows 200 entries — it must not build a dict per entry.

    Found via a call trace: _row_to_item ran millions of times, because
    the waiting list was fully rematerialized on every queue change and
    only then trimmed down to 200.
    """
    q.enqueue([f"f{i}.mp4" for i in range(50)], {}, str(queue_env["media"]))

    built = []
    real = q._row_to_item
    monkeypatch.setattr(q, "_row_to_item", lambda r: built.append(1) or real(r))

    header = q.list_items("pending", limit=5)
    assert len(header) == 5
    assert len(built) == 5, f"{len(built)} entries built for 5 shown"

    # A second request of the same length no longer costs anything at all
    built.clear()
    assert len(q.list_items("pending", limit=5)) == 5
    assert built == []

    # A SHORTER request is served from the existing list
    built.clear()
    assert len(q.list_items("pending", limit=3)) == 3
    assert built == []


def test_every_mutation_invalidates_the_cache(queue_env):
    """The dangerous part of the cache: a forgotten invalidation silently
    serves stale data. Every mutation is played through here."""
    media = str(queue_env["media"])

    q.enqueue(["a.mp4", "b.mp4", "c.mp4"], {}, media)
    assert q.snapshot()["pending"] == 3
    assert len(q.list_items("pending")) == 3

    ids = {i["filename"]: i["id"] for i in q.list_items("pending")}

    # move
    q.move_item(ids["c.mp4"], -1)
    assert [i["filename"] for i in q.list_items("pending")] == ["a.mp4", "c.mp4", "b.mp4"]

    # remove
    q.remove_item(ids["a.mp4"])
    assert q.snapshot()["pending"] == 2
    assert [i["filename"] for i in q.list_items("pending")] == ["c.mp4", "b.mp4"]

    # claim: the worker takes the next one for itself
    item = q._claim_next()
    assert item["filename"] == "c.mp4"
    assert q.snapshot()["pending"] == 1

    # requeue: on shutdown the running entry goes back into the queue
    q._requeue(item["id"])
    assert q.snapshot()["pending"] == 2
    assert [i["filename"] for i in q.list_items("pending")] == ["c.mp4", "b.mp4"]

    # finish
    item = q._claim_next()
    q._finish(item["id"], "done")
    assert q.snapshot()["done_count"] == 1
    assert q.snapshot()["pending"] == 1

    # clear
    q.clear_pending()
    assert q.snapshot()["pending"] == 0
    assert q.list_items("pending") == []


def test_api_queue_returns_only_pending_items(tmp_path, monkeypatch):
    """/api/queue['items'] may contain ONLY waiting entries.

    The frontend takes the list 1:1 as the queue
    (applyQueueState → queueItems). If list_items() were called here
    without a filter, the WebUI would also show long-finished and failed
    entries as "waiting" — and on a large queue would send along thousands
    of entries including settings.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    media = tmp_path / "media"
    media.mkdir()
    for n in ("a.mp4", "b.mp4", "c.mp4"):
        (media / n).write_bytes(b"x")

    from whisper_tm.config import config as cfg

    q.stop_worker()
    cfg.load(None)
    cfg._data.setdefault("directories", {})["media"] = str(media)
    cfg._resolve_directories()
    cfg._ensure_directories()
    cfg.set("queue", "autostart", False)  # worker must not touch anything here

    monkeypatch.setattr(q, "is_transcript_complete", lambda *a, **k: False)
    monkeypatch.setattr(q, "is_skipped", lambda *a, **k: False)

    from whisper_tm.app import create_app

    client = create_app().test_client()
    try:
        q.clear_pending()
        q.enqueue(["a.mp4", "b.mp4", "c.mp4"], {}, str(media))

        # Complete one entry, set one to error
        items = q.list_items("pending")
        q._finish(items[0]["id"], "done")
        q._finish(items[1]["id"], "error", "broken")

        d = client.get("/api/queue").get_json()

        assert {i["status"] for i in d["items"]} == {"pending"}
        assert [i["filename"] for i in d["items"]] == ["c.mp4"]
        # The tally lives in the counters, not in the list
        assert d["done_count"] == 1 and d["fail_count"] == 1
    finally:
        q.stop_worker()


class TestConfigDefaults:
    """The settings must actually determine the job defaults.

    Previously, `transcription.default_*` and `vocal_isolation.default_*`
    were pure dummies: the UI saved them, but `_normalize`
    and the API plugged in hardcoded values instead. These tests establish
    that a missing request value comes from the config — and that an
    explicitly set request value still wins.
    """

    @pytest.fixture
    def cfg(self):
        """Set config — and then RESTORE it afterward.

        `config` is a singleton; without the reset, every subsequent test
        would continue running with large-v3/French.
        """
        from whisper_tm.config import config as c

        if not c.is_loaded:
            c.load()
        overrides = {
            ("transcription", "default_model"): "large-v3",
            ("transcription", "default_language"): "fr",
            ("transcription", "default_diarize"): False,
            ("transcription", "default_num_speakers"): 3,
            ("vocal_isolation", "default_save_alongside"): True,
        }
        previous = {k: c.get(*k, default=None) for k in overrides}
        for (sec, key), val in overrides.items():
            c.set(sec, key, val)
        try:
            yield c
        finally:
            for (sec, key), val in previous.items():
                c.set(sec, key, val)

    def test_defaults_come_from_config(self, cfg):
        opts = q._normalize({})
        assert opts["model_name"] == "large-v3"
        assert opts["language"] == "fr"
        assert opts["diarize"] is False
        assert opts["min_speakers"] == 3 and opts["max_speakers"] == 3
        assert opts["save_audio_alongside"] is True

    def test_request_still_wins(self, cfg):
        opts = q._normalize(
            {
                "model": "turbo",
                "language": "de",
                "diarize": True,
                "numSpeakers": 5,
            }
        )
        assert opts["model_name"] == "turbo"
        assert opts["language"] == "de"
        assert opts["diarize"] is True
        assert opts["min_speakers"] == 5

    def test_a_lavasr_request_from_an_old_entry_is_ignored(self, cfg):
        """LavaSR was removed; queue entries saved before still carry the
        key. It must not reach the worker, whose signature no longer has it."""
        assert "lavasr" not in q._normalize({"lavasr": True})

    def test_explicit_false_is_not_overridden(self, cfg):
        """A deliberate 'off' must not be overridden by the config default."""
        assert q._normalize({"diarize": False})["diarize"] is False
        cfg.set("transcription", "default_diarize", True)
        assert q._normalize({"diarize": False})["diarize"] is False

    def test_garbage_model_falls_back(self, cfg):
        """A nonsensical config value must not break transcription."""
        cfg.set("transcription", "default_model", "does-not-exist")
        assert q._normalize({})["model_name"] == "turbo"


def test_a_root_change_lets_old_entries_keep_running(queue_env, monkeypatch):
    """Regression: directory changed (+ restart) → all entries
    from the earlier root died with "path not allowed".

    An entry's root was checked at ENQUEUE time — the persistent queue
    must still process it afterward, even if allowed_roots() (allowed
    roots) no longer contains the old root.
    """
    from whisper_tm.core import security

    old_value = queue_env["media"]
    added = q.enqueue(["a.mp4"], {}, str(old_value))  # root here: checked+allowed
    assert len(added) == 1

    # "Restart with a different root": media_dir now points elsewhere, the
    # old root drops out of the allow-list (the anchor is the start/current
    # root).
    new_value = old_value.parent / "new-root"
    new_value.mkdir()
    monkeypatch.setattr(type(q.config), "media_dir", property(lambda self: new_value))
    assert not security.root_allowed(str(old_value / "a.mp4")), (
        "test setup: old root should now be DISALLOWED"
    )

    processed = []
    monkeypatch.setattr(
        "whisper_tm.core.transcription.transcription_worker", _fake_worker(processed)
    )
    q.start_worker()

    assert _wait_until(lambda: (_item("a.mp4") or {}).get("status") in ("done", "error")), (
        "worker never touched the entry"
    )
    item = _item("a.mp4")
    assert item["status"] == "done", f"entry ended as {item['status']}: {item.get('error')}"
    assert processed == ["a.mp4"]


# ── Run without transcription ("no transcription" in the dialog) ──────────


def _done(name="a.mp4"):
    return (_item(name) or {}).get("status") in ("done", "error", "skipped")


def test_a_separation_run_does_not_delete_the_existing_transcript(queue_env, monkeypatch):
    """The "New" button sends `retranscribe` along — even if "no
    transcription" was chosen in the dialog. But a separation-only run
    doesn't produce a new transcript: if it deleted the old one, the data
    would be gone for good.

    Second part: it also must not be skipped as "transcript already
    present" — an already-transcribed file is the normal case for
    separation.
    """
    deleted = []
    monkeypatch.setattr(
        "whisper_tm.api.transcription._delete_transcript_everywhere",
        lambda fn, **k: deleted.append(fn),
    )
    monkeypatch.setattr(q, "is_transcript_complete", lambda *a, **k: True)

    processed = []
    monkeypatch.setattr(
        "whisper_tm.core.transcription.transcription_worker", _fake_worker(processed)
    )
    q.enqueue(["a.mp4"], {"model": "none", "retranscribe": True}, str(queue_env["media"]))
    q.start_worker()

    assert _wait_until(_done), "worker never touched the entry"
    assert deleted == [], "separation run deleted the transcript"
    assert processed == ["a.mp4"], "separation run was skipped"
    assert _item("a.mp4")["status"] == "done"


def test_a_separation_run_reaches_the_worker_with_the_mode_set(queue_env, monkeypatch):
    """`model: none` must arrive as separation mode — not as a Whisper run."""
    seen = {}

    def run(job_id, source, filename, **kwargs):
        seen.update(kwargs)
        ext.job_manager.complete(ext.job_manager.get(job_id), {"processing_time": "1s"})

    monkeypatch.setattr("whisper_tm.core.transcription.transcription_worker", run)
    q.enqueue(["a.mp4"], {"model": "none", "vocal_isolation": False}, str(queue_env["media"]))
    q.start_worker()

    assert _wait_until(_done), "worker never touched the entry"
    assert seen.get("separation_only") is True
    assert seen.get("vocal_isolation") is True, "without separation it would be a no-op"
    assert seen.get("save_audio_alongside") is True, "otherwise no result remains"
    assert seen.get("model_name") != "none"


def test_a_separation_run_merges_nothing(queue_env, monkeypatch):
    """A still-checked "summarize" box must not, without a transcript, let
    the LLM loose on a transcript that doesn't exist."""
    called = []
    monkeypatch.setattr(q, "_queue_summary", lambda *a, **k: called.append(1))
    monkeypatch.setattr("whisper_tm.core.transcription.transcription_worker", _fake_worker([]))
    q.enqueue(["a.mp4"], {"model": "none", "summarize": True}, str(queue_env["media"]))
    q.start_worker()

    assert _wait_until(_done), "worker never touched the entry"
    assert called == []


class TestTheSummaryFollows:
    """The dialog's summary follows the transcript in the LLM queue. It ran
    in the lane itself before the entry counted as done: while the language
    model didn't answer (a server on another machine, for half an hour),
    every lane waited and nothing was transcribed. The entry is done with
    its transcript now; the page that shows it still learns that a summary
    is coming (summary_pending) — otherwise it arrives a minute later while
    the view says "0 summaries"."""

    def _run(self, queue_env, monkeypatch, summarize, endpoint="http://llm:8080"):
        from whisper_tm.core.llm_queue import llm_queue

        monkeypatch.setitem(q.config._data.setdefault("llm", {}), "endpoint", endpoint)
        llm_queue.stop()  # the tests look at what waits
        seen = {}
        abs_a = str(queue_env["media"] / "a.mp4")

        def run(job_id, source, filename, **kwargs):
            seen["retry_if"] = ext.job_manager.get(job_id).retry_if
            ext.job_manager.complete(ext.job_manager.get(job_id), {"processing_time": "1s"})

        monkeypatch.setattr("whisper_tm.core.transcription.transcription_worker", run)
        q.enqueue(
            ["a.mp4"],
            {
                "model": "turbo",
                "summarize": summarize,
                "summarize_preset": "Timeline",
                "summarize_model": "qwen3",
            },
            str(queue_env["media"]),
        )
        q.start_worker()
        assert _wait_until(_done), "worker never touched the entry"
        seen["pending"] = q.summary_pending(abs_a)
        with ext.db.connection() as conn:
            seen["tasks"] = [
                dict(r)
                for r in conn.execute(
                    "SELECT id, kind, filename, params_json, status FROM llm_queue"
                )
            ]
        return seen

    def test_the_entry_is_done_and_the_summary_waits_in_the_llm_queue(self, queue_env, monkeypatch):
        seen = self._run(queue_env, monkeypatch, summarize=True)
        assert _item("a.mp4")["status"] == "done"
        [task] = seen["tasks"]
        params = json.loads(task["params_json"])
        assert (task["kind"], task["filename"], task["status"]) == ("summary", "a.mp4", "pending")
        assert params == {
            "media_dir": str(queue_env["media"]),
            "auto": True,
            "label": "Timeline",
            "preset": "Timeline",
            "model": "qwen3",
        }
        assert seen["pending"] is True
        assert any("Summary queued: a.mp4 (Timeline)" in line for line in q._log)

    def test_a_written_or_failed_summary_ends_the_waiting(self, queue_env, monkeypatch):
        seen = self._run(queue_env, monkeypatch, summarize=True)
        abs_a = str(queue_env["media"] / "a.mp4")
        for status in ("running", "error"):
            with ext.db.connection() as conn:
                conn.execute(
                    "UPDATE llm_queue SET status = ? WHERE id = ?", (status, seen["tasks"][0]["id"])
                )
            assert q.summary_pending(abs_a) is (status == "running")
        assert q.summary_pending(str(queue_env["media"] / "b.mp4")) is False

    def test_without_the_box_nothing_is_coming(self, queue_env, monkeypatch):
        seen = self._run(queue_env, monkeypatch, summarize=False)
        assert seen["tasks"] == [] and seen["pending"] is False

    def test_without_a_language_model_it_says_so(self, queue_env, monkeypatch):
        seen = self._run(queue_env, monkeypatch, summarize=True, endpoint="")
        assert seen["tasks"] == [] and _item("a.mp4")["status"] == "done"
        assert any("No summary of a.mp4" in line for line in q._log)

    def test_an_oom_of_a_queue_job_is_taken_up_again(self, queue_env, monkeypatch):
        # The page must not show a ❌ for it: _oom_retry puts the entry back
        seen = self._run(queue_env, monkeypatch, summarize=False)
        assert seen["retry_if"] is q._is_oom_error


def test_a_normal_run_still_summarizes(queue_env, monkeypatch):
    """Counter-check: the brake above applies ONLY to the separation run."""
    called = []
    monkeypatch.setattr(q, "_queue_summary", lambda *a, **k: called.append(1))
    monkeypatch.setattr("whisper_tm.core.transcription.transcription_worker", _fake_worker([]))
    q.enqueue(["a.mp4"], {"model": "turbo", "summarize": True}, str(queue_env["media"]))
    q.start_worker()

    assert _wait_until(_done), "worker never touched the entry"
    assert called == [1]
