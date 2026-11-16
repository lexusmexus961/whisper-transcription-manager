"""CUDA OOM: an item NEVER becomes an error — it waits, or the queue halts.

Backstory: the old rule was to retry once with "go easy on VRAM" and book
it as an error on the second OOM. In a long run the retry could fail every
time, because it started in the SAME second as the error, with a card that
was just as full (e.g. another program holding part of it). And when
``hardware.low_vram`` was already globally on, the retry changed nothing at
all. Files were then marked broken, although nothing was wrong with them.

The contract:

1. Never an error: back to ``pending``, with ``low_vram``, halved
   batch and the counter ``oom_attempts`` — which lives IN the database.
2. Wait first (``hardware.oom_min_free_mb`` / ``oom_wait_min``),
   then retry.
3. If the card doesn't free up, or even the eased-off run fails:
   pause the queue, system warning, incident — the item stays waiting.
"""

from __future__ import annotations

import json
import types

import pytest

from whisper_tm.core import queue as q
from whisper_tm.core import vram


@pytest.fixture
def db(tmp_path, monkeypatch):
    """A real, empty queue database — not the running server's."""
    from whisper_tm.database import Database

    d = Database(tmp_path / "library.db")
    d.initialize()
    monkeypatch.setattr(q, "_db", lambda: d)
    monkeypatch.setattr(q, "_recommended_batch", lambda: 16)
    q._paused = False
    q._abort = False
    q._shutting_down = False
    yield d
    q._paused = False


class FakeJM:
    def __init__(self):
        self.warnings = []

    def system_warning(self, msg):
        self.warnings.append(msg)


@pytest.fixture
def jm(monkeypatch):
    f = FakeJM()
    monkeypatch.setattr(q, "_jm", lambda: f)
    return f


@pytest.fixture
def voice_map(monkeypatch):
    """The card is immediately free — waiting is a no-op."""
    monkeypatch.setattr(vram, "wait_for_free_vram", lambda *a, **k: (True, 9000))


@pytest.fixture
def qlog(monkeypatch):
    lines = []
    monkeypatch.setattr(q, "qlog", lambda m: lines.append(m))
    return lines


def _entry(db, filename="a.mkv", settings=None) -> dict:
    settings = settings if settings is not None else {"model": "large-v3"}
    with db.connection() as conn:
        cur = conn.execute(
            "INSERT INTO queue_items (filename, base_dir, settings_json, "
            "status, position) VALUES (?, ?, ?, 'processing', 1)",
            (filename, "/media", json.dumps(settings)),
        )
        vid = cur.lastrowid
    return {"id": vid, "filename": filename, "settings": dict(settings)}


def _status(db, item_id):
    with db.connection() as conn:
        r = conn.execute(
            "SELECT status, settings_json, error FROM queue_items WHERE id = ?", (item_id,)
        ).fetchone()
    return r["status"], json.loads(r["settings_json"] or "{}"), r["error"]


def _back_to_processing(db, item_id):
    with db.connection() as conn:
        conn.execute("UPDATE queue_items SET status = 'processing' WHERE id = ?", (item_id,))


class TestOnlyOnOOM:
    def test_an_oom_returns_to_the_queue(self, db, voice_map, jm, qlog):
        it = _entry(db)
        assert q._oom_retry(it, "CUDA failed with error out of memory")
        status, formerly, error = _status(db, it["id"])
        assert status == "pending", "the item is not queued again"
        assert not error and not q.is_paused()

    def test_and_it_does_so_gently_with_half_the_batch_and_a_counter(self, db, voice_map, jm, qlog):
        """The second attempt must behave differently than the first —
        otherwise it's gambling, not retrying."""
        it = _entry(db)
        q._oom_retry(it, "CUDA failed with error out of memory")
        _, formerly, _ = _status(db, it["id"])
        assert formerly["low_vram"] is True
        assert formerly["batch_size"] == 8 and formerly["oom_attempts"] == 1

    def test_the_remaining_settings_stay(self, db, voice_map, jm, qlog):
        it = _entry(
            db,
            settings={
                "model": "large-v3",
                "language": "de",
                "diarize": True,
                "vocal_isolation": True,
            },
        )
        q._oom_retry(it, "CUDA failed with error out of memory")
        _, formerly, _ = _status(db, it["id"])
        assert formerly["model"] == "large-v3" and formerly["language"] == "de"
        assert formerly["diarize"] is True and formerly["vocal_isolation"] is True

    @pytest.mark.parametrize(
        "error",
        [
            "No audio track found",
            "ffmpeg exited with code 1",
            "Aborted",
            "",
        ],
    )
    def test_other_errors_are_not_retried(self, db, voice_map, jm, error):
        it = _entry(db)
        assert not q._oom_retry(it, error)
        assert _status(db, it["id"])[0] == "processing"

    def test_the_spelling_does_not_matter(self, db, voice_map, jm, qlog):
        it = _entry(db)
        assert q._oom_retry(it, "RuntimeError: CUDA Out Of Memory")

    def test_pyannotes_own_wording_is_an_oom_too(self, db, voice_map, jm, qlog):
        # pyannote re-raises a CUDA OOM as MemoryError without the words;
        # that once booked a diarization on a full card as error
        it = _entry(db)
        assert q._oom_retry(
            it,
            "Diarization failed: batch_size ( 32) is probably too "
            "large. Try with a smaller value until memory error "
            "disappears.",
        )
        assert _status(db, it["id"])[0] == "pending"


class TestWaitFirst:
    def test_it_waits_with_the_configured_limits(self, db, jm, qlog, monkeypatch):
        calls = []
        monkeypatch.setattr(vram, "total_mb", lambda *a, **k: 24576)

        def wait(min_mb, max_sec, **kw):
            calls.append((min_mb, max_sec))
            kw["report_it"](1200)
            return True, 8500

        monkeypatch.setattr(vram, "wait_for_free_vram", wait)
        it = _entry(db)
        assert q._oom_retry(it, "out of memory")
        assert calls == [(8000, 600.0)]  # defaults: 8000 MB, 10 min
        assert any("Waiting for free VRAM: 1200 MB" in z for z in qlog)
        assert any("VRAM free (8500 MB)" in z for z in qlog)
        assert not q.is_paused()

    @pytest.mark.parametrize("card,expected", [(8192, 6553), (6144, 4915), (24576, 8000)])
    def test_a_smaller_card_waits_for_what_it_can_free(
        self, db, jm, qlog, monkeypatch, card, expected
    ):
        # 8000 MB free never comes on an 8 GB card: the wait ran its full
        # ten minutes and paused the queue after every out-of-memory
        calls = []
        monkeypatch.setattr(vram, "total_mb", lambda *a, **k: card)

        def wait(min_mb, max_sec, **kw):
            calls.append(min_mb)
            return True, min_mb

        monkeypatch.setattr(vram, "wait_for_free_vram", wait)
        assert q._oom_retry(_entry(db), "out of memory")
        assert calls == [expected]

    def test_card_does_not_free_up_then_pause_instead_of_retry(self, db, jm, qlog, monkeypatch):
        """Rule 3: don't even try — pause, warn, lose nothing."""
        monkeypatch.setattr(vram, "wait_for_free_vram", lambda *a, **k: (False, 1500))
        it = _entry(db)
        assert q._oom_retry(it, "out of memory")
        status, formerly, error = _status(db, it["id"])
        assert status == "pending" and not error  # NEVER error
        assert q.is_paused()
        assert len(jm.warnings) == 1 and "queue paused" in jm.warnings[0]
        assert "1500 MB free" in jm.warnings[0]
        assert any(z.startswith("⏸") for z in qlog)

    def test_the_user_pauses_while_it_waits(self, db, jm, qlog, monkeypatch):
        """The user has already stopped it — no second warning on top."""

        def wait(*a, **k):
            q._paused = True
            return False, 100

        monkeypatch.setattr(vram, "wait_for_free_vram", wait)
        it = _entry(db)
        assert q._oom_retry(it, "out of memory")
        assert _status(db, it["id"])[0] == "pending" and jm.warnings == []

    def test_cancelling_the_file_ends_the_wait(self, db, jm, qlog, monkeypatch):
        """The cancel button used to do nothing while it waited for VRAM —
        the wait only looked at the queue flags, never at the job."""
        job = types.SimpleNamespace(cancel=False)

        def wait(*a, abort, **k):
            job.cancel = True
            assert abort()
            return False, 400

        monkeypatch.setattr(vram, "wait_for_free_vram", wait)
        it = _entry(db)
        assert q._oom_retry(it, "out of memory", job)
        assert _status(db, it["id"])[0] == "cancelled" and jm.warnings == []

    def test_stop_and_quick_resume_while_waiting_keeps_it(self, db, jm, qlog, monkeypatch):
        """Stop sets job.cancel and marks the entry for requeueing; a resume
        a second later clears the queue flags, but the wait still ends and
        the entry stays pending."""
        job = types.SimpleNamespace(cancel=False)

        def wait(*a, abort, **k):
            job.cancel = True
            q._requeue_on_abort.add(it["id"])
            assert abort()
            return False, 400

        monkeypatch.setattr(vram, "wait_for_free_vram", wait)
        it = _entry(db)
        assert q._oom_retry(it, "out of memory", job)
        assert _status(db, it["id"])[0] == "pending"
        assert it["id"] not in q._requeue_on_abort

    def test_waiting_switched_off(self, db, jm, qlog, monkeypatch):
        monkeypatch.setattr(
            q.config,
            "get",
            lambda *a, **k: 0 if a[:2] == ("hardware", "oom_min_free_mb") else k.get("default"),
        )
        monkeypatch.setattr(
            vram,
            "wait_for_free_vram",
            lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not wait")),
        )
        it = _entry(db)
        assert q._oom_retry(it, "out of memory")
        assert _status(db, it["id"])[0] == "pending" and not q.is_paused()


class TestSecondOOM:
    def test_the_second_oom_pauses_and_stays_pending(self, db, voice_map, jm, qlog):
        """No error, no loop: the item stays, the queue stops,
        the user finds out."""
        it = _entry(db)
        assert q._oom_retry(it, "out of memory")
        _, formerly, _ = _status(db, it["id"])
        _back_to_processing(db, it["id"])
        it["settings"] = formerly
        assert q._oom_retry(it, "out of memory")
        status, formerly, error = _status(db, it["id"])
        assert status == "pending" and not error
        assert formerly["oom_attempts"] == 2 and formerly["batch_size"] == 4
        assert q.is_paused()
        assert len(jm.warnings) == 1 and "half the batch" in jm.warnings[0]

    def test_the_batch_never_drops_below_two(self, db, voice_map, jm, qlog):
        it = _entry(db, settings={"batch_size": 3, "oom_attempts": 4})
        q._oom_retry(it, "out of memory")
        _, formerly, _ = _status(db, it["id"])
        assert formerly["batch_size"] == 2 and formerly["oom_attempts"] == 5

    def test_two_different_files_do_not_disturb_each_other(self, db, voice_map, jm, qlog):
        a, b = _entry(db, "a.mkv"), _entry(db, "b.mkv")
        assert q._oom_retry(a, "out of memory")
        assert q._oom_retry(b, "out of memory")
        assert not q.is_paused()

    def test_the_counter_is_in_the_database(self, db, voice_map, jm, qlog):
        """Deliberately persistent: after a restart the run shouldn't
        start over at batch 16, and the badge should show it."""
        it = _entry(db)
        q._oom_retry(it, "out of memory")
        _, formerly, _ = _status(db, it["id"])
        assert formerly["oom_attempts"] == 1
        assert not hasattr(q, "_oom_retried")


class TestWiringUp:
    def test_the_error_branch_asks_first(self):
        from pathlib import Path

        source = (
            Path(__file__).resolve().parent.parent / "whisper_tm" / "core" / "queue.py"
        ).read_text(encoding="utf-8")
        i = source.index('elif job and job.status == "error" and not job.cancel:')
        body = source[i : i + 500]
        assert "_oom_retry(item, err, job)" in body
        assert body.index("_oom_retry") < body.index('_finish(item["id"], "error"')

    def test_a_failed_retry_does_not_bring_the_run_down(self):
        from pathlib import Path

        source = (
            Path(__file__).resolve().parent.parent / "whisper_tm" / "core" / "queue.py"
        ).read_text(encoding="utf-8")
        i = source.index("def _oom_retry(")
        body = source[i : source.index("\ndef _requeue(", i)]
        assert "except Exception" in body and "return False" in body

    def test_batch_size_reaches_the_worker(self):
        """queue._normalize passes batch_size through, transcription_worker takes it."""
        assert q._normalize({"batch_size": 4})["batch_size"] == 4
        assert q._normalize({"batch_size": "8"})["batch_size"] == 8
        assert q._normalize({})["batch_size"] is None
        assert q._normalize({"batch_size": 0})["batch_size"] is None
        import inspect

        from whisper_tm.core.transcription import transcription_worker

        assert "batch_size" in inspect.signature(transcription_worker).parameters


class TestFreeVram:
    def test_free_mb_from_total_and_used(self, monkeypatch):
        from whisper_tm.core import debug

        monkeypatch.setattr(debug, "gpu_mem_used_mb", lambda: {0: 9800})
        monkeypatch.setattr(debug, "gpu_static", lambda: {0: {"name": "x", "total_mb": 12288}})
        assert vram.free_mb(0) == 2488
        monkeypatch.setattr(debug, "gpu_mem_used_mb", lambda: {})
        assert vram.free_mb(0) is None

    def test_wait_until_free_and_cancel(self, monkeypatch):
        value_list = iter([1000, 3000, 9000])
        monkeypatch.setattr(vram, "free_mb", lambda *a, **k: next(value_list))
        reported = []
        ok, free = vram.wait_for_free_vram(8000, 60, report_it=reported.append, tick=0.01)
        assert ok and free == 9000 and reported == [1000]
        monkeypatch.setattr(vram, "free_mb", lambda *a, **k: 500)
        ok, free = vram.wait_for_free_vram(8000, 0.05, tick=0.01)
        assert not ok and free == 500
        ok, free = vram.wait_for_free_vram(8000, 60, abort=lambda: True, tick=0.01)
        assert not ok
        monkeypatch.setattr(vram, "free_mb", lambda *a, **k: None)
        assert vram.wait_for_free_vram(8000, 60) == (True, None)
