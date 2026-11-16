"""The GPU gate: which file may use which model when (core/gpu_gate.py).

The goal: concurrency like vLLM, without loading a model twice. Queue lanes
run several files at once; every model is loaded once and serves one file
at a time.

What these tests pin:

* a stage (one model) takes one thread at a time; two different stages may
  run together; a thread inside a stage may enter it again or another one;
* a stage that would run next to another one waits until enough VRAM is
  free — alone it always goes;
* an exclusive waits for every running stage, holds new ones off and
  goes before them; its owner's stages pass; asking for it from inside a
  stage raises instead of waiting for itself;
* ``try_exclusive`` takes the GPU only when nothing runs;
* a cancelled job leaves the wait — for a stage and for the exclusive —
  instead of first taking its turn (a cancelled file used to wait for the
  separator, then would have separated a 30-minute chunk), and its progress
  says what it waits for;
* the time spent waiting is handed to the pipeline once.
"""

from __future__ import annotations

import threading
import time

import pytest

from whisper_tm.core.gpu_gate import GpuGate


def _gate(free=None):
    return GpuGate(free_mb=(lambda: free[0]) if free is not None else (lambda: None))


def _in_thread(fn):
    t = threading.Thread(target=fn, daemon=True)
    t.start()
    return t


def _hold(gate, name, release, entered=None):
    """A thread that sits in a stage until ``release`` is set."""

    def run():
        with gate.stage(name):
            if entered is not None:
                entered.set()
            release.wait(5)

    return _in_thread(run)


def _until(pred, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(0.01)
    return False


class TestStages:
    def test_one_thread_per_model(self):
        gate, inside, most = _gate(), [], [0]
        lock = threading.Lock()

        def use():
            with gate.stage("separation"):
                with lock:
                    inside.append(1)
                    most[0] = max(most[0], len(inside))
                time.sleep(0.05)
                with lock:
                    inside.pop()

        threads = [_in_thread(use) for _ in range(4)]
        for t in threads:
            t.join(5)
        assert most[0] == 1

    def test_two_models_run_together(self):
        gate = _gate()
        entered = threading.Event()
        release = threading.Event()

        def separation():
            with gate.stage("separation"):
                entered.set()
                release.wait(5)

        t = _in_thread(separation)
        assert entered.wait(5)
        done = threading.Event()
        _hold(gate, "diarization", release, entered=done)
        assert done.wait(2), "diarization waited for the separator"
        release.set()
        t.join(5)

    def test_nested_stages_of_one_thread_pass(self):
        gate = _gate()
        with gate.stage("vectors"):
            with gate.stage("separation"):
                with gate.stage("separation"):
                    assert gate.holds_stage()
        assert not gate.holds_stage()
        assert gate.state()["stages"] == {}

    def test_next_to_another_it_waits_for_free_vram(self):
        free = [500]
        gate = _gate(free)
        release = threading.Event()
        entered = threading.Event()

        def separation():
            with gate.stage("separation"):
                entered.set()
                release.wait(5)

        t = _in_thread(separation)
        assert entered.wait(5)
        got = threading.Event()

        def diarization():
            with gate.stage("diarization", need_mb=1500):
                got.set()

        _in_thread(diarization)
        assert not got.wait(0.3), "started next to the separator with 500 MB free"
        free[0] = 4000
        assert got.wait(3), "did not start once the VRAM was free"
        release.set()
        t.join(5)

    def test_alone_it_always_goes(self):
        gate = _gate([10])
        with gate.stage("diarization", need_mb=1500):
            pass

    def test_unknown_free_vram_does_not_block(self):
        gate = _gate()
        release = threading.Event()
        t = _hold(gate, "separation", release)
        assert _until(lambda: gate.state()["stages"])
        with gate.stage("diarization", need_mb=99999):
            pass
        release.set()
        t.join(5)


class TestExclusive:
    def test_waits_for_running_stages_and_holds_new_ones_off(self):
        gate = _gate()
        release = threading.Event()
        order = []

        def separation():
            with gate.stage("separation"):
                release.wait(5)
                order.append("separation done")

        t = _in_thread(separation)
        assert _until(lambda: gate.state()["stages"])

        def unload():
            with gate.exclusive("unloading"):
                order.append("exclusive")
                time.sleep(0.1)
                order.append("exclusive done")

        u = _in_thread(unload)
        assert _until(lambda: gate.state()["waiting_exclusive"] == 1)

        # A new stage now waits behind the exclusive
        def diarization():
            with gate.stage("diarization"):
                order.append("diarization")

        late = _in_thread(diarization)
        time.sleep(0.1)
        assert "diarization" not in order
        release.set()
        for x in (t, u, late):
            x.join(5)
        assert order == ["separation done", "exclusive", "exclusive done", "diarization"]

    def test_the_owners_stages_and_exclusives_pass(self):
        gate = _gate()
        with gate.exclusive("low-VRAM run"):
            with gate.stage("separation"):
                with gate.exclusive("unloading"):
                    pass
            with gate.stage("asr"):
                pass
        assert gate.state()["exclusive"] is None

    def test_asking_for_it_inside_a_stage_raises(self):
        gate = _gate()
        with gate.stage("separation"):
            with pytest.raises(RuntimeError, match="wait for itself"):
                with gate.exclusive("unloading"):
                    pass

    def test_whisper_is_a_stage_of_its_own(self):
        """ASR never runs next to another stage (5.6–6.8 GB alone)."""
        gate = _gate()
        release = threading.Event()
        t = _hold(gate, "separation", release)
        assert _until(lambda: gate.state()["stages"])
        got = threading.Event()
        _hold(gate, "asr", threading.Event(), entered=got)
        assert not got.wait(0.3)
        release.set()
        assert got.wait(3)
        t.join(5)

    def test_try_exclusive_only_when_idle(self):
        gate = _gate()
        release = threading.Event()
        t = _hold(gate, "separation", release)
        assert _until(lambda: gate.state()["stages"])
        with gate.try_exclusive("idle unload") as got:
            assert got is False
        release.set()
        t.join(5)
        gate2 = _gate()
        with gate2.try_exclusive("idle unload") as got:
            assert got is True
            assert gate2.state()["exclusive"] == "idle unload"
        assert gate2.state()["exclusive"] is None


class FakeJob:
    def __init__(self):
        self.cancel = False
        self.message = "🎤 Vocal Isolation…"
        self.messages = []

    def update(self, message=None, **kw):
        self.message = message
        self.messages.append(message)


class TestCancelWhileWaiting:
    def test_a_cancelled_job_leaves_the_stage_wait(self):
        gate, job = _gate(), FakeJob()
        release = threading.Event()
        t = _hold(gate, "separation", release)
        assert _until(lambda: gate.state()["stages"])
        threading.Timer(0.2, lambda: setattr(job, "cancel", True)).start()
        t0 = time.time()
        with pytest.raises(InterruptedError):
            with gate.stage("separation", job=job):
                pass
        assert time.time() - t0 < 3
        assert job.messages == ["⏳ Waiting for the separator — another file is using it"]
        assert gate.state()["stages"] == {"separation": t.ident}
        release.set()
        t.join(5)
        assert gate.state()["stages"] == {}

    def test_a_cancelled_job_leaves_the_exclusive_wait(self):
        gate, job = _gate(), FakeJob()
        release = threading.Event()
        t = _hold(gate, "diarization", release)
        assert _until(lambda: gate.state()["stages"])
        threading.Timer(0.2, lambda: setattr(job, "cancel", True)).start()
        with pytest.raises(InterruptedError):
            with gate.exclusive("low-VRAM run", job=job):
                pass
        assert job.messages == ["⏳ Waiting for the other files to leave the GPU"]
        state = gate.state()
        assert state["waiting_exclusive"] == 0 and state["exclusive"] is None
        # Nothing stays blocked behind the exclusive that gave up
        done = threading.Event()
        _hold(gate, "vectors", threading.Event(), entered=done)
        assert done.wait(3)
        release.set()
        t.join(5)

    def test_the_note_goes_when_the_wait_is_over(self):
        """Three running files used to all say they were waiting."""
        gate, job = _gate(), FakeJob()
        release = threading.Event()
        t = _hold(gate, "separation", release)
        assert _until(lambda: gate.state()["stages"])
        threading.Timer(0.2, release.set).start()
        with gate.stage("separation", job=job):
            assert job.message == "🎤 Vocal Isolation…"
        assert job.messages == [
            "⏳ Waiting for the separator — another file is using it",
            "🎤 Vocal Isolation…",
        ]
        t.join(5)

    def test_no_note_when_it_does_not_wait(self):
        gate, job = _gate(), FakeJob()
        with gate.stage("separation", job=job):
            pass
        assert job.messages == []


class TestWaitTime:
    def test_the_wait_is_handed_over_once(self):
        gate = _gate()
        release = threading.Event()
        t = _hold(gate, "separation", release)
        assert _until(lambda: gate.state()["stages"])
        gate.take_wait()
        threading.Timer(0.3, release.set).start()
        with gate.stage("separation"):
            pass
        assert gate.take_wait() >= 0.25
        assert gate.take_wait() == 0.0
        t.join(5)


class TestWhoWaits:
    """`waiting()` — a holder that keeps a stage for long (the reranker's
    server, core/rerank_server.py) gives the card back when someone waits."""

    def test_stages_and_exclusives_waiting_are_counted(self):
        gate = _gate()
        release = threading.Event()
        t = _hold(gate, "separation", release)
        assert _until(lambda: gate.state()["stages"])
        assert gate.waiting() == 0
        same = _hold(gate, "separation", release)  # waits for the stage
        assert _until(lambda: gate.waiting() == 1)
        assert gate.state()["waiting_stages"] == 1

        def unload():
            with gate.exclusive("unloading"):
                pass

        u = _in_thread(unload)
        assert _until(lambda: gate.waiting() == 2)
        release.set()
        for x in (t, same, u):
            x.join(5)
        assert gate.waiting() == 0 and gate.state()["waiting_stages"] == 0

    def test_a_cancelled_wait_is_no_longer_counted(self):
        gate = _gate()
        release = threading.Event()
        t = _hold(gate, "rerank", release)
        assert _until(lambda: gate.state()["stages"])
        stop = threading.Event()
        out = []

        def waiter():
            try:
                with gate.stage("rerank", cancel=stop.is_set):
                    out.append("entered")
            except InterruptedError:
                out.append("cancelled")

        w = _in_thread(waiter)
        assert _until(lambda: gate.waiting() == 1)
        stop.set()
        w.join(5)
        assert out == ["cancelled"] and gate.waiting() == 0
        release.set()
        t.join(5)
