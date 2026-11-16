"""Queue lanes: several files in flight at once (core/queue.py).

Goal: concurrency in the spirit of vLLM — several files at once without
loading the model twice, and without breaking anything. ``queue.lanes``
worker threads take one entry each; the models are shared through
core/gpu_gate.py.

What these tests pin:

* lanes really overlap, and every entry runs exactly once — never claimed
  twice, never lost, whatever the timing;
* two entries with the same name (under two roots) are never in flight
  together: the job id, the work folder and the job log hang on the name;
* at most one long file is in flight; short ones pass it;
* stop puts EVERY running entry back, shutdown too; abort cancels every
  one; a stop that comes before an entry's job exists still reaches it;
* the snapshot lists every running entry, current_* stays the oldest;
* an out-of-memory entry stays 'processing' while its lane waits, so no
  other lane takes it onto the same full card;
* a low-VRAM entry runs alone on the GPU;
* the queue's time left divides by how many ran at once;
* the diarization worker answers one request at a time, each its own;
* the low-VRAM choice of one lane is not another lane's;
* a paused queue runs what was just brought in — only that, the rest stays
  paused and waits; stop, pause and resume end that; the add route and a
  video from the browser extension ask for it.
"""

from __future__ import annotations

import json
import re
import threading
import time

import pytest

import whisper_tm.extensions as ext
from whisper_tm.config import config
from whisper_tm.core import queue as q
from whisper_tm.core.gpu_gate import GATE
from whisper_tm.database import Database
from whisper_tm.jobs import JobManager


@pytest.fixture
def lanes_env(tmp_path, monkeypatch):
    """Fresh DB + JobManager, three lanes, files that always exist."""
    q.stop_worker()
    media = tmp_path / "media"
    media.mkdir()
    db = Database(tmp_path / "test.db")
    db.initialize()
    monkeypatch.setattr(ext, "db", db)
    monkeypatch.setattr(ext, "job_manager", JobManager(logs_dir=tmp_path / "logs"))
    monkeypatch.setitem(config._data.setdefault("queue", {}), "lanes", 3)
    monkeypatch.setitem(config._data["queue"], "long_file_sec", 3600)
    # Spare VRAM runs an entry alone — on by itself on a CPU-only machine or
    # a small card, where these tests would then see one lane at a time
    monkeypatch.setitem(config._data.setdefault("hardware", {}), "low_vram", False)
    q._running.clear()
    q._current = None
    q._abort = q._paused = q._stopping = q._shutting_down = False
    q._requeue_on_abort.clear()
    q._solo.clear()
    q._log.clear()
    monkeypatch.setattr(q, "is_transcript_complete", lambda *a, **k: False)
    monkeypatch.setattr(q, "is_skipped", lambda *a, **k: False)
    transcripts = tmp_path / "transcripts"
    transcripts.mkdir()
    monkeypatch.setattr(q, "transcript_path_for", lambda rel, **k: transcripts / (rel + ".json"))
    monkeypatch.setattr(type(q.config), "media_dir", property(lambda self: media))
    names = [f"f{i:02d}.mp4" for i in range(12)]
    for name in names:
        (media / name).write_bytes(b"x")
    yield {"db": db, "media": media, "names": names, "tmp": tmp_path}
    q.stop_worker()
    q._running.clear()
    q._solo.clear()


class Recorder:
    """A transcription_worker stand-in that records who ran when."""

    def __init__(self, delay=0.15, hold=None):
        self.delay = delay
        self.hold = hold  # Event: block until set
        self.lock = threading.Lock()
        self.inside: set[str] = set()
        self.most = 0
        self.runs: list[str] = []
        self.started = threading.Event()

    def __call__(self, job_id, source, filename, **kwargs):
        job = ext.job_manager.get(job_id)
        with self.lock:
            self.inside.add(filename)
            self.most = max(self.most, len(self.inside))
            self.runs.append(filename)
        self.started.set()
        try:
            end = time.time() + 10
            if self.hold is not None:
                while not self.hold.is_set() and not job.cancel and time.time() < end:
                    time.sleep(0.01)
            else:
                time.sleep(self.delay)
            if job.cancel:
                ext.job_manager.fail(job, "Cancelled")
            else:
                ext.job_manager.complete(job, {"processing_time": "1s"})
        finally:
            with self.lock:
                self.inside.discard(filename)


def _until(pred, timeout=10.0):
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(0.02)
    return False


def _statuses():
    return {i["filename"]: i["status"] for i in q.list_items()}


def _use(monkeypatch, worker):
    monkeypatch.setattr("whisper_tm.core.transcription.transcription_worker", worker)


class TestOverlap:
    def test_three_lanes_overlap_and_each_entry_runs_once(self, lanes_env, monkeypatch):
        rec = Recorder(delay=0.2)
        _use(monkeypatch, rec)
        q.enqueue(lanes_env["names"], {}, str(lanes_env["media"]))
        q.start_worker()
        assert _until(lambda: set(_statuses().values()) == {"done"}, 20)
        assert rec.most == 3
        assert sorted(rec.runs) == sorted(lanes_env["names"])  # once each
        assert len(q._workers) == 3
        assert [t.name for t in q._workers] == [f"queue-worker-{i}" for i in (1, 2, 3)]

    def test_each_start_has_its_own_place(self, lanes_env, monkeypatch):
        """Lanes starting together read the snapshot after each other's
        claims: the log said "(2/11)", "(3/11)", "(3/11)" and no "(1/11)"."""
        rec = Recorder(delay=0.05)
        _use(monkeypatch, rec)
        q.enqueue(lanes_env["names"], {}, str(lanes_env["media"]))
        q.start_worker()
        assert _until(lambda: set(_statuses().values()) == {"done"}, 20)
        places = sorted(
            int(re.search(r"\((\d+)/12\)$", line).group(1))
            for line in q._log
            if "Starting:" in line
        )
        assert places == list(range(1, 13))

    def test_one_lane_is_the_serial_queue(self, lanes_env, monkeypatch):
        monkeypatch.setitem(config._data["queue"], "lanes", 1)
        rec = Recorder(delay=0.02)
        _use(monkeypatch, rec)
        q.enqueue(lanes_env["names"][:5], {}, str(lanes_env["media"]))
        q.start_worker()
        assert _until(lambda: set(_statuses().values()) == {"done"})
        assert rec.most == 1 and rec.runs == lanes_env["names"][:5]
        assert [t.name for t in q._workers] == ["queue-worker"]

    def test_lanes_are_capped(self, monkeypatch):
        monkeypatch.setitem(config._data.setdefault("queue", {}), "lanes", 99)
        assert q.lanes() == q.MAX_LANES
        monkeypatch.setitem(config._data["queue"], "lanes", 0)
        assert q.lanes() == 1
        monkeypatch.setitem(config._data["queue"], "lanes", "two")
        assert q.lanes() == 1

    def test_claims_race_never_double(self, lanes_env):
        """Eight threads claiming at once from 40 entries: 40 distinct claims."""
        names = [f"r{i:02d}.mp4" for i in range(40)]
        for n in names:
            (lanes_env["media"] / n).write_bytes(b"x")
        q.enqueue(names, {}, str(lanes_env["media"]))
        got, lock = [], threading.Lock()
        go = threading.Event()

        def claim():
            go.wait()
            while True:
                item = q._claim_next()
                if item is None:
                    return
                with lock:
                    got.append(item["id"])
                q._release(item)  # free the name for the next claim

        threads = [threading.Thread(target=claim) for _ in range(8)]
        for t in threads:
            t.start()
        go.set()
        for t in threads:
            t.join(10)
        assert len(got) == 40 and len(set(got)) == 40


class TestWhatNeverRunsTogether:
    def test_the_same_name_under_two_roots_waits(self, lanes_env, monkeypatch):
        other = lanes_env["tmp"] / "other"
        other.mkdir()
        (other / "f00.mp4").write_bytes(b"x")
        monkeypatch.setitem(
            config._data.setdefault("security", {}), "allowed_roots", [str(lanes_env["tmp"])]
        )
        q.enqueue(["f00.mp4"], {}, str(lanes_env["media"]))
        q.enqueue(["f00.mp4"], {}, str(other))
        q.enqueue(["f01.mp4"], {}, str(lanes_env["media"]))
        first = q._claim_next()
        second = q._claim_next()
        assert first["filename"] == "f00.mp4"
        assert second["filename"] == "f01.mp4"  # the twin was passed over
        assert q._claim_next() is None
        q._release(first)
        third = q._claim_next()
        assert third["filename"] == "f00.mp4" and third["dir"] == str(other)

    def test_only_one_long_file_at_a_time(self, lanes_env):
        db, media = lanes_env["db"], lanes_env["media"]
        with db.connection() as conn:
            for name, dur in (("f00.mp4", 7200), ("f01.mp4", 5400), ("f02.mp4", 30)):
                conn.execute(
                    "INSERT INTO media_files (rel_path, abs_path, duration_sec) VALUES (?, ?, ?)",
                    (name, f"{media}/{name}", dur),
                )
        q.enqueue(["f00.mp4", "f01.mp4", "f02.mp4"], {}, str(media))
        first = q._claim_next()
        assert first["filename"] == "f00.mp4" and first["_long"]
        second = q._claim_next()
        assert second["filename"] == "f02.mp4"  # the other long one waits
        assert q._claim_next() is None
        q._release(first)
        assert q._claim_next()["filename"] == "f01.mp4"


class TestControl:
    def _three_running(self, lanes_env, monkeypatch):
        hold = threading.Event()
        rec = Recorder(hold=hold)
        _use(monkeypatch, rec)
        q.enqueue(lanes_env["names"][:5], {}, str(lanes_env["media"]))
        q.start_worker()
        assert _until(lambda: q.snapshot()["running_count"] == 3)
        assert _until(lambda: all(r["job_id"] for r in q.snapshot()["running_items"]))
        return rec, hold

    def test_stop_puts_every_running_entry_back(self, lanes_env, monkeypatch):
        rec, hold = self._three_running(lanes_env, monkeypatch)
        ok, msg = q.stop()
        assert ok and "3 running jobs" in msg
        assert _until(lambda: q.snapshot()["running_count"] == 0)
        assert set(_statuses().values()) == {"pending"}, _statuses()
        assert q.snapshot()["paused"] is True
        hold.set()

    def test_abort_cancels_every_running_entry(self, lanes_env, monkeypatch):
        rec, hold = self._three_running(lanes_env, monkeypatch)
        ok, removed = q.abort()
        assert ok and removed == 2
        assert _until(lambda: q.snapshot()["running_count"] == 0)
        assert sorted(_statuses().values()) == ["cancelled"] * 3
        hold.set()

    def test_shutdown_puts_every_running_entry_back(self, lanes_env, monkeypatch):
        rec, hold = self._three_running(lanes_env, monkeypatch)
        q.shutdown(timeout=10)
        try:
            assert set(_statuses().values()) == {"pending"}, _statuses()
        finally:
            q._shutting_down = False
            hold.set()

    def test_a_stop_before_the_job_exists_still_reaches_it(self, lanes_env, monkeypatch):
        rec = Recorder(delay=0.05)
        _use(monkeypatch, rec)
        q.enqueue(["f00.mp4"], {}, str(lanes_env["media"]))
        item = q._claim_next()  # claimed, no job yet
        assert item["job_id"] == ""
        q.stop()
        q._process(item)
        q._release(item)
        assert _statuses() == {"f00.mp4": "pending"}

    def test_one_entry_is_cancelled_before_its_job_exists(self, lanes_env, monkeypatch):
        """The ✕ on a running row whose lane still checks the file."""
        rec = Recorder(delay=0.05)
        _use(monkeypatch, rec)
        q.enqueue(["f00.mp4", "f01.mp4"], {}, str(lanes_env["media"]))
        item = q._claim_next()
        other = q._claim_next()
        assert q.cancel_item(item["id"]) is True
        assert q.cancel_item(99999) is False
        q._process(item)
        q._release(item)
        q._process(other)
        q._release(other)
        assert _statuses() == {"f00.mp4": "cancelled", "f01.mp4": "done"}
        assert q.cancel_item(item["id"]) is False  # not running anymore

    def test_the_cancel_route(self, lanes_env, monkeypatch):
        from flask import Flask

        from whisper_tm.api.queue import queue_bp

        app = Flask(__name__)
        app.register_blueprint(queue_bp)
        client = app.test_client()
        q.enqueue(["f00.mp4"], {}, str(lanes_env["media"]))
        item = q._claim_next()
        try:
            assert client.post("/api/queue/cancel", json={"item_id": item["id"]}).status_code == 200
            assert q._running[item["id"]]["_cancel"] is True
            assert client.post("/api/queue/cancel", json={"item_id": 4242}).status_code == 404
            assert client.post("/api/queue/cancel", json={}).status_code == 400
        finally:
            q._release(item)

    def test_snapshot_lists_every_running_entry(self, lanes_env, monkeypatch):
        rec, hold = self._three_running(lanes_env, monkeypatch)
        snap = q.snapshot()
        items = snap["running_items"]
        assert snap["running_count"] == 3 and snap["lanes"] == 3
        assert [i["filename"] for i in items] == lanes_env["names"][:3]
        assert snap["current_file"] == items[0]["filename"]
        assert snap["current_item_id"] == items[0]["item_id"]
        assert all(i["abs"].endswith(i["filename"]) for i in items)
        assert q.busy() is True
        editor = q.editor_list({})
        assert len(editor["processing_items"]) == 3
        assert editor["processing"]["id"] == items[0]["item_id"]
        hold.set()
        assert _until(lambda: q.snapshot()["running_count"] == 0)
        assert q.busy() is False

    def test_removing_an_entry_a_lane_just_took(self, lanes_env):
        q.enqueue(["f00.mp4"], {}, str(lanes_env["media"]))
        item = q._claim_next()
        ok, msg = q.remove_item(item["id"])
        assert not ok and "running" in msg.lower()
        assert _statuses() == {"f00.mp4": "processing"}
        q._release(item)


class TestOutOfMemory:
    def test_the_entry_stays_running_while_its_lane_waits(self, lanes_env, monkeypatch):
        """As 'pending' another lane would take it onto the same full card."""
        from whisper_tm.core import vram

        waiting, go_on = threading.Event(), threading.Event()
        seen = {}

        def wait_for_free_vram(min_mb, max_sec, **kw):
            seen["status"] = _statuses()["f00.mp4"]
            seen["exclusive"] = GATE.state()["exclusive"]
            waiting.set()
            go_on.wait(5)
            return True, 9000

        monkeypatch.setattr(vram, "wait_for_free_vram", wait_for_free_vram)
        q.enqueue(["f00.mp4", "f01.mp4"], {}, str(lanes_env["media"]))
        item = q._claim_next()
        t = threading.Thread(
            target=q._oom_retry, args=(item, "CUDA out of memory. Tried to allocate")
        )
        t.start()
        assert waiting.wait(5)
        assert seen == {"status": "processing", "exclusive": "waiting after a VRAM overflow"}
        go_on.set()
        t.join(5)
        q._release(item)
        row = next(i for i in q.list_items() if i["filename"] == "f00.mp4")
        assert row["status"] == "pending"
        assert row["settings"]["low_vram"] is True and row["settings"]["oom_attempts"] == 1

    def test_a_low_vram_entry_runs_alone(self, lanes_env, monkeypatch):
        seen = []

        def worker(job_id, source, filename, **kwargs):
            seen.append((filename, GATE.state()["exclusive"]))
            ext.job_manager.complete(ext.job_manager.get(job_id), {})

        _use(monkeypatch, worker)
        q.enqueue(["f00.mp4"], {"low_vram": True}, str(lanes_env["media"]))
        q.enqueue(["f01.mp4"], {"low_vram": False}, str(lanes_env["media"]))
        for _ in range(2):
            item = q._claim_next()
            q._process(item)
            q._release(item)
        assert dict(seen) == {"f00.mp4": "low-VRAM run", "f01.mp4": None}


class TestReviewFindings:
    """Defects an adversarial review found in the first lanes version."""

    def test_the_oom_wait_unloads_inside_its_own_exclusive(self, lanes_env, monkeypatch):
        """Measured after the other lanes had reloaded their models, the
        wait ran ten minutes and paused the queue for nothing."""
        from whisper_tm.core import vram

        order = []
        monkeypatch.setattr(
            q, "_unload_models", lambda: order.append(("unload", GATE.state()["exclusive"]))
        )
        monkeypatch.setattr(
            vram,
            "wait_for_free_vram",
            lambda *a, **k: (order.append(("wait", GATE.state()["exclusive"])), (True, 9000))[1],
        )
        q.enqueue(["f00.mp4"], {}, str(lanes_env["media"]))
        item = q._claim_next()
        assert q._oom_retry(item, "CUDA out of memory")
        q._release(item)
        assert order == [
            ("unload", "waiting after a VRAM overflow"),
            ("wait", "waiting after a VRAM overflow"),
        ]
        assert _statuses() == {"f00.mp4": "pending"} and not q.is_paused()

    def test_a_cancel_while_it_waits_for_the_other_lanes(self, lanes_env, monkeypatch):
        """Second OOM: the lane waits for the gate; a ✕ meanwhile must end
        it as cancelled — not pause the queue over it."""
        monkeypatch.setattr(q, "_unload_models", lambda: None)
        q.enqueue(["f00.mp4"], {"oom_attempts": 1}, str(lanes_env["media"]))
        item = q._claim_next()
        release, holding = threading.Event(), threading.Event()

        def other_lane():
            with GATE.stage("separation"):
                holding.set()
                release.wait(10)

        threading.Thread(target=other_lane, daemon=True).start()
        assert holding.wait(5)
        t = threading.Thread(target=q._oom_retry, args=(item, "CUDA out of memory"))
        t.start()
        assert _until(lambda: GATE.state()["waiting_exclusive"] == 1)
        assert q.cancel_item(item["id"])
        t.join(5)
        release.set()
        q._release(item)
        assert _statuses() == {"f00.mp4": "cancelled"} and not q.is_paused()

    def test_an_abort_on_its_way_counts_as_cancel(self, lanes_env):
        q._abort, q._paused = True, False  # abort() before job.cancel
        try:
            assert q._oom_cancel_outcome({"id": 1}, None) == "cancelled"
            q._paused = True  # a stop
            assert q._oom_cancel_outcome({"id": 1}, None) == "retry"
        finally:
            q._abort = q._paused = False

    def test_stop_then_abort_cancels_instead_of_rerunning(self, lanes_env, monkeypatch):
        hold = threading.Event()
        _use(monkeypatch, Recorder(hold=hold))
        q.enqueue(lanes_env["names"][:3], {}, str(lanes_env["media"]))
        q.start_worker()
        assert _until(
            lambda: (
                all(r["job_id"] for r in q.snapshot()["running_items"])
                and q.snapshot()["running_count"] == 3
            )
        )
        q.stop()
        q.abort()
        assert _until(lambda: q.snapshot()["running_count"] == 0)
        time.sleep(0.3)
        assert sorted(_statuses().values()) == ["cancelled"] * 3, _statuses()
        hold.set()

    def test_one_file_under_two_spellings_waits(self, lanes_env):
        """sub/x.mp4 under /media and x.mp4 under /media/sub: one file."""
        media = lanes_env["media"]
        (media / "sub").mkdir()
        (media / "sub" / "x.mp4").write_bytes(b"x")
        q.enqueue(["sub/x.mp4"], {}, str(media))
        q.enqueue(["x.mp4"], {}, str(media / "sub"))
        q.enqueue(["f01.mp4"], {}, str(media))
        first = q._claim_next()
        second = q._claim_next()
        assert (first["filename"], second["filename"]) == ("sub/x.mp4", "f01.mp4")
        assert q._claim_next() is None
        q._release(first)
        assert q._claim_next()["dir"] == str(media / "sub")

    def test_the_low_vram_choice_is_pinned_for_the_run(self, lanes_env, monkeypatch):
        """Read live, "Spare VRAM" switched on mid-run made a lane without
        the exclusive unload models under the others."""
        seen = []

        def worker(job_id, source, filename, **kwargs):
            seen.append((filename, kwargs["low_vram"], GATE.state()["exclusive"]))
            ext.job_manager.complete(ext.job_manager.get(job_id), {})

        _use(monkeypatch, worker)
        monkeypatch.setitem(config._data.setdefault("hardware", {}), "low_vram", True)
        q.enqueue(["f00.mp4"], {}, str(lanes_env["media"]))  # no choice
        q.enqueue(["f01.mp4"], {"low_vram": False}, str(lanes_env["media"]))
        for _ in range(2):
            item = q._claim_next()
            q._process(item)
            q._release(item)
        assert seen == [("f00.mp4", True, "low-VRAM run"), ("f01.mp4", False, None)]

    def test_a_low_vram_entry_cancelled_while_it_waits_for_the_gpu(self, lanes_env, monkeypatch):
        """It ends as cancelled, and its job ends too — left 'queued' it
        counted as running for the idle unloader forever."""
        _use(monkeypatch, Recorder(delay=0.01))
        q.enqueue(["f00.mp4"], {"low_vram": True}, str(lanes_env["media"]))
        item = q._claim_next()
        release, holding = threading.Event(), threading.Event()

        def other_lane():
            with GATE.stage("diarization"):
                holding.set()
                release.wait(10)

        threading.Thread(target=other_lane, daemon=True).start()
        assert holding.wait(5)
        t = threading.Thread(target=q._process, args=(item,))
        t.start()
        assert _until(lambda: GATE.state()["waiting_exclusive"] == 1)
        assert q.cancel_item(item["id"])
        t.join(5)
        release.set()
        q._release(item)
        assert not t.is_alive()
        assert _statuses() == {"f00.mp4": "cancelled"}
        assert ext.job_manager.list_active() == []

    def test_releasing_the_separator_from_inside_another_stage(self, monkeypatch):
        """Window separation runs inside the voice-vector stage; its
        low-VRAM release used to raise there and fail every window."""
        from whisper_tm.core import vocal_isolation as vi

        monkeypatch.setattr(vi, "_separator", "loaded")
        with GATE.stage("vectors"):
            vi.release_separator()  # no exception
        assert vi._separator == "loaded"

    @pytest.mark.parametrize(
        "error", [RuntimeError("CUDA out of memory. Tried"), InterruptedError("Cancelled")]
    )
    def test_window_separation_passes_a_full_card_and_a_cancel_on(
        self, tmp_path, monkeypatch, error
    ):
        """Skipped, every window failed and the entry ended as error —
        not as the OOM the queue retries, nor as a cancel."""
        import torch

        from whisper_tm.core import speaker
        from whisper_tm.core import vocal_isolation as vi

        def fail(*a, **k):
            raise error

        monkeypatch.setattr(vi, "isolate_vocals", fail)
        separate = speaker.window_separator(tmp_path)
        with pytest.raises(type(error)):
            separate(torch.zeros(1, 16000))


class TestALibraryThatExits:
    """With a full card: audio-separator calls
    sys.exit(1) when its model won't load. SystemExit passed every
    ``except Exception`` and ended all three lanes without a word, their
    entries left 'processing' — the same as one worker before lanes."""

    def test_the_lane_survives_and_books_the_entry(self, lanes_env, monkeypatch):
        rec = Recorder(delay=0.01)

        def worker(job_id, source, filename, **kwargs):
            if filename == "f00.mp4":
                raise SystemExit(1)
            rec(job_id, source, filename, **kwargs)

        _use(monkeypatch, worker)
        monkeypatch.setitem(config._data["queue"], "lanes", 1)
        q.enqueue(["f00.mp4", "f01.mp4"], {}, str(lanes_env["media"]))
        q.start_worker()
        assert _until(lambda: _statuses() == {"f00.mp4": "error", "f01.mp4": "done"})
        assert all(t.is_alive() for t in q._workers)
        row = next(i for i in q.list_items() if i["filename"] == "f00.mp4")
        assert row["error"] == "SystemExit: 1"

    def test_a_model_that_wont_load_is_an_error_with_its_reason(self):
        import logging

        from whisper_tm.core import vocal_isolation as vi

        class Sep:
            def load_model(self, model_filename):
                log = logging.getLogger("audio_separator.separator.separator")
                log.error(
                    "Failed to load Roformer model: CUDA out of memory. Tried to allocate 20.00 MiB"
                )
                log.error("Please try deleting the model file and run again")
                raise SystemExit(1)

        with pytest.raises(RuntimeError, match="CUDA out of memory"):
            vi._load_model(Sep(), "vocals_mel_band_roformer.ckpt")

    def test_another_lanes_error_is_not_this_ones_reason(self):
        import logging

        from whisper_tm.core import vocal_isolation as vi

        with vi._SeparatorErrorLog() as log:
            t = threading.Thread(
                target=lambda: logging.getLogger("audio_separator").error(
                    "CUDA out of memory (other lane)"
                )
            )
            t.start()
            t.join()
            logging.getLogger("audio_separator").error("Failed to process file a.wav: boom")
        assert log.reason() == "boom"


class TestJobLogs:
    def test_two_names_with_one_safe_name_keep_two_logs(self, tmp_path, monkeypatch):
        """'a/b.mp4' and 'a_b.mp4' finishing in the same second."""
        import whisper_tm.jobs as jobs

        monkeypatch.setattr(jobs.time, "strftime", lambda fmt, *a: "20240301_120000")
        jm = JobManager(logs_dir=tmp_path / "logs")
        names = []
        for i, name in enumerate(("a/b.mp4", "a_b.mp4")):
            job = jm.create(f"j{i}", filename=name, job_type="transcription")
            jm.log(job, f"log of {name}")
            names.append(jm._persist_log(job))
        assert names == ["20240301_120000_a_b.mp4.log", "20240301_120000_a_b.mp4-2.log"]
        texts = sorted((tmp_path / "logs" / n).read_text() for n in names)
        assert "log of a/b.mp4" in texts[0] and "log of a_b.mp4" in texts[1]


class TestTheStreamsLog:
    def test_new_lines_still_arrive_once_the_ring_is_full(self, lanes_env, monkeypatch):
        import types

        from flask import Flask

        import whisper_tm.api.queue as api_queue

        monkeypatch.setattr(
            api_queue,
            "time",
            types.SimpleNamespace(time=time.time, sleep=lambda s: q.qlog("tick line")),
        )
        app = Flask(__name__)
        app.register_blueprint(api_queue.queue_bp)
        for i in range(q.MAX_LOG + 20):
            q.qlog(f"line {i}")
        resp = app.test_client().get("/api/queue/stream", buffered=False)
        events = []
        for chunk in resp.response:
            text = chunk.decode() if isinstance(chunk, bytes) else chunk
            if text.startswith("data: "):
                events.append(json.loads(text[6:]))
            if len(events) == 4:
                break
        resp.close()
        assert len(events[0]["new_log"]) == q.MAX_LOG
        assert [len(e.get("new_log", [])) for e in events[1:]] == [1, 1, 1]
        assert all(e["new_log"][0].endswith("tick line") for e in events[1:])


class TestTimeLeft:
    def test_shares_of_the_wall_clock(self):
        share = q._clock_shares([(0, 30), (10, 20), (10, 40)])
        assert share(0, 30) == pytest.approx(10 + 10 / 3 + 10 / 2)
        assert share(10, 20) == pytest.approx(10 / 3)
        assert share(10, 40) == pytest.approx(10 / 3 + 10 / 2 + 10)
        # They add up to the time anything ran
        assert share(0, 30) + share(10, 20) + share(10, 40) == pytest.approx(40)

    def _done(self, db, media, name, dur, a, b):
        with db.connection() as conn:
            conn.execute(
                "INSERT INTO media_files (rel_path, abs_path, duration_sec) VALUES (?, ?, ?)",
                (name, f"{media}/{name}", dur),
            )
            conn.execute(
                "INSERT INTO queue_items (filename, base_dir, settings_json, "
                "status, position, started_at, finished_at) VALUES "
                "(?, ?, '{\"job\": \"voices\"}', 'done', 0, ?, ?)",
                (name, str(media), a, b),
            )

    def test_three_lanes_a_pause_and_a_long_file_beside_short_ones(self, lanes_env):
        """Three cases from a review: the samples are the same costs
        whether the entries ran three at a time, long ago, or next to one
        long file — dividing their sum by a recent parallelism was wrong
        in each."""
        db, media = lanes_env["db"], lanes_env["media"]
        now = time.time()
        t0 = now - 4 * 3600  # long before a pause
        # three lanes of 9-s runs that each cost 3 s
        for i in range(30):
            a = t0 + (i // 3) * 9
            self._done(db, media, f"n{i}.aac", 10, a, a + 9)
        # one long entry beside two lanes of short ones, more recently
        t1 = now - 3600
        self._done(db, media, "long.aac", 1800, t1, t1 + 600)
        for i in range(40):
            a = t1 + i * 15
            self._done(db, media, f"s{i}.aac", 10, a, a + 15)
            self._done(db, media, f"t{i}.aac", 10, a, a + 15)
        with db.connection() as conn:  # something left
            conn.execute(
                "INSERT INTO queue_items (filename, base_dir, settings_json, "
                "status, position) VALUES ('f00.mp4', ?, '{\"job\": \"voices\"}', "
                "'pending', 1)",
                (str(media),),
            )
        inputs = q.eta_inputs(now=now)
        by_dur = {}
        for dur, secs in inputs["samples"]["voices"]:
            by_dur.setdefault(dur, []).append(secs)
        assert by_dur[1800] == [pytest.approx(200.0)]  # 600 s / 3
        assert sorted(by_dur[10])[:30] == [pytest.approx(3.0)] * 30  # 9 s / 3
        assert sorted(by_dur[10])[30:] == [pytest.approx(5.0)] * 80  # 15 s / 3
        assert 2.5 < inputs["parallelism"] < 3.1

    def test_a_running_entry_counts_its_share(self, lanes_env):
        from whisper_tm.core import eta

        inputs = {
            "pending": {},
            "samples": {"voices": [(5.0, 3.0)] * 30 + [(600.0, 33.0)] * 30},
            "running": [{"kind": "voices", "duration": 600.0, "started_at": 1000.0}] * 3,
            "last_hour": {},
        }
        est = eta.queue(lanes_env["db"], inputs, now=1030.0)
        expected = eta.fit(*eta.groups_of(inputs["samples"]["voices"])).seconds(1, 600.0)
        assert est["sec"] == pytest.approx(3 * (expected - 10.0))

    def test_the_cost_of_a_run_beside_others(self, lanes_env, monkeypatch):
        """The transcript's compute time is the run's share (one lane: its
        wall time, unchanged)."""
        q.enqueue(["f00.mp4", "f01.mp4"], {}, str(lanes_env["media"]))
        a = q._claim_next()
        time.sleep(0.2)
        b = q._claim_next()
        time.sleep(0.2)
        ratio_a, ratio_b = q._cost_ratio(a), q._cost_ratio(b)
        q._release(a)
        q._release(b)
        assert ratio_a == pytest.approx(0.75, abs=0.08)  # 0.2 alone + 0.2 shared
        assert ratio_b == pytest.approx(0.5, abs=0.08)


class TestDiarizationWorker:
    def test_two_lanes_each_get_their_own_answer(self, tmp_path):
        """The answers carry no request id: one request at a time."""
        import sys
        import textwrap

        from whisper_tm.core import diar_process

        script = tmp_path / "worker.py"
        script.write_text(
            textwrap.dedent("""
            import json, sys, time
            print(json.dumps({"kind": "ready"}), flush=True)
            for line in sys.stdin:
                req = json.loads(line)
                time.sleep(0.2)
                name = req["wav"].rsplit("/", 1)[-1]
                print(json.dumps({"kind": "result", "ok": True,
                                  "segments": [[0, 1, name]]}), flush=True)
        """)
        )
        worker = diar_process.Worker("cpu", command=[sys.executable, str(script)])
        results = {}

        def ask(name):
            results[name] = worker.diarize_it(tmp_path / name)

        try:
            threads = [threading.Thread(target=ask, args=(f"{n}.wav",)) for n in ("a", "b", "c")]
            for t in threads:
                t.start()
            for t in threads:
                t.join(20)
        finally:
            worker.terminate("test over")
        assert {k: v[0][2] for k, v in results.items()} == {
            "a.wav": "a.wav",
            "b.wav": "b.wav",
            "c.wav": "c.wav",
        }


class TestLowVramPerLane:
    def test_one_lanes_choice_is_not_anothers(self):
        from whisper_tm.core import vram

        seen = {}
        set_it, read_it = threading.Event(), threading.Event()

        def lane_a():
            vram.set_low_vram_override(True)
            set_it.set()
            read_it.wait(5)
            seen["a"] = vram.low_vram()
            vram.set_low_vram_override(None)

        def lane_b():
            set_it.wait(5)
            vram.set_low_vram_override(False)
            seen["b"] = vram.low_vram()
            read_it.set()

        threads = [threading.Thread(target=f) for f in (lane_a, lane_b)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(5)
        assert seen == {"a": True, "b": False}


class TestOnlyTheNewWhilePaused:
    """When the queue is paused, only the new video runs; the rest stays
    paused. Adding a file "to the front" used to resume the whole
    queue — and its whole backlog with it."""

    def _backlog(self, lanes_env, monkeypatch, delay=0.05):
        rec = Recorder(delay=delay)
        _use(monkeypatch, rec)
        q.enqueue(lanes_env["names"][:4], {}, str(lanes_env["media"]))
        q.pause()
        return rec

    def test_only_the_new_entry_runs(self, lanes_env, monkeypatch):
        rec = self._backlog(lanes_env, monkeypatch)
        new = q.enqueue(["f05.mp4"], {}, str(lanes_env["media"]), front=True)
        assert q.run_now([e["id"] for e in new]) == 1
        q.start_worker()
        assert _until(lambda: _statuses()["f05.mp4"] == "done")
        time.sleep(0.4)  # nothing else starts after it
        assert rec.runs == ["f05.mp4"]
        assert [s for n, s in _statuses().items() if n != "f05.mp4"] == ["pending"] * 4
        snap = q.snapshot()
        assert snap["paused"] is True and snap["solo"] == 0 and q._solo == set()

    def test_while_it_runs_the_queue_says_so(self, lanes_env, monkeypatch):
        hold = threading.Event()
        rec = Recorder(hold=hold)
        _use(monkeypatch, rec)
        q.enqueue(lanes_env["names"][:2], {}, str(lanes_env["media"]))
        q.pause()
        new = q.enqueue(["f05.mp4"], {}, str(lanes_env["media"]), front=True)
        q.run_now([e["id"] for e in new])
        q.start_worker()
        assert _until(lambda: q.snapshot()["running_count"] == 1)
        snap = q.snapshot()
        assert snap["paused"] is True and snap["solo"] == 1
        assert [r["filename"] for r in snap["running_items"]] == ["f05.mp4"]
        hold.set()

    def test_without_a_pause_nothing_changes(self, lanes_env):
        new = q.enqueue(["f05.mp4"], {}, str(lanes_env["media"]), front=True)
        assert q.run_now([e["id"] for e in new]) == 0 and q._solo == set()

    def test_stop_ends_it_and_resume_runs_everything(self, lanes_env, monkeypatch):
        hold = threading.Event()
        rec = Recorder(hold=hold)
        _use(monkeypatch, rec)
        q.enqueue(lanes_env["names"][:2], {}, str(lanes_env["media"]))
        q.pause()
        new = q.enqueue(["f05.mp4"], {}, str(lanes_env["media"]), front=True)
        q.run_now([e["id"] for e in new])
        q.start_worker()
        assert _until(lambda: q.snapshot()["running_count"] == 1)
        q.stop()  # back to waiting, not again
        assert _until(lambda: q.snapshot()["running_count"] == 0)
        assert q._solo == set() and set(_statuses().values()) == {"pending"}
        time.sleep(0.3)
        assert q.snapshot()["running_count"] == 0
        hold.set()
        q.resume()  # now all of it
        assert _until(lambda: set(_statuses().values()) == {"done"})

    def test_one_that_is_gone_is_not_waited_for(self, lanes_env):
        q.enqueue(lanes_env["names"][:1], {}, str(lanes_env["media"]))
        q.pause()
        new = q.enqueue(["f05.mp4"], {}, str(lanes_env["media"]), front=True)
        q.run_now([e["id"] for e in new])
        assert q.remove_item(new[0]["id"])[0] is True
        q._prune_solo()
        assert q._solo == set()

    def test_the_add_route_runs_just_these(self, lanes_env, monkeypatch):
        from flask import Flask

        from whisper_tm.api.queue import queue_bp

        self._backlog(lanes_env, monkeypatch)
        app = Flask(__name__)
        app.register_blueprint(queue_bp)
        client = app.test_client()
        media = str(lanes_env["media"])
        d = client.post(
            "/api/queue/add", json={"filenames": ["f05.mp4"], "base_dir": media, "front": True}
        ).get_json()
        assert d["run_now"] == 1 and d["paused"] is True
        assert q.is_paused() is True  # not resumed
        d = client.post(
            "/api/queue/add", json={"filenames": ["f06.mp4"], "base_dir": media}
        ).get_json()
        assert d["run_now"] == 0  # not "front": it waits like the rest
        assert len(q._solo) == 1
