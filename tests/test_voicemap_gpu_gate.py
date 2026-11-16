"""The voice map's neighbourhood asks the GPU gate before using the card.

The map's kNN used to load the whole inventory onto CUDA (half precision,
plus 1.25-GB product stripes) outside ``core/gpu_gate.GATE`` — next to three
queue lanes and other GPU programs on a shared card, a map refresh could push
a transcription into out-of-memory.

Now the GPU path takes a gate stage ("voicemap", with what it needs). Is the
gate held (Whisper runs exclusively, or the memory is short next to another
stage), the map does not wait long: after ``GPU_WAIT_SEC`` it computes on the
CPU — the same neighbours, exact by construction.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from whisper_tm.core import gpu_gate
from whisper_tm.core import speaker_neighbours as nb

pytestmark = pytest.mark.skipif(not nb.HAVE_NUMBA, reason="numba missing")


def _x(n=300, d=16, seed=3):
    X = np.random.default_rng(seed).normal(size=(n, d))
    return (X / np.linalg.norm(X, axis=1, keepdims=True)).astype(np.float32)


@pytest.fixture
def gate(monkeypatch):
    g = gpu_gate.GpuGate(free_mb=lambda: 10**6)
    monkeypatch.setattr(gpu_gate, "GATE", g)
    seen = []
    real = nb._Products

    class _Spy(real):
        def __init__(self, X, device):
            seen.append((device, g.state() if hasattr(g, "state") else None, dict(g._stages)))
            super().__init__(X, "cpu")  # the computation itself on the CPU

    monkeypatch.setattr(nb, "_Products", _Spy)
    monkeypatch.setattr(nb, "GPU_WAIT_SEC", 0.2)
    monkeypatch.setattr(nb, "GPU_WAIT_UPDATE_SEC", 0.2)
    monkeypatch.setattr(nb, "_cuda_available", lambda: True)
    return g, seen


def test_the_gpu_path_runs_inside_a_gate_stage(gate):
    g, seen = gate
    X = _x()
    idx, v64, _ = nb.compute(X, 5, "cuda")
    assert seen[0][0] == "cuda"
    assert "voicemap" in seen[0][2]
    assert not g._stages  # left again
    ref_idx, ref_v, _ = nb.compute(X, 5, "cpu")
    assert (idx == ref_idx).all()


def test_a_held_gate_sends_the_map_to_the_cpu(gate):
    g, seen = gate
    taken, release = threading.Event(), threading.Event()

    def whisper():
        with g.exclusive("asr"):
            taken.set()
            release.wait(5)

    t = threading.Thread(target=whisper)
    t.start()
    taken.wait(5)
    try:
        nb.compute(_x(), 5, "cuda")
    finally:
        release.set()
        t.join()
    assert seen[0][0] == "cpu"


def test_the_cpu_path_never_asks_the_gate(gate, monkeypatch):
    g, seen = gate

    def no(*a, **k):
        raise AssertionError("the CPU needs no gate")

    monkeypatch.setattr(g, "stage", no)
    nb.compute(_x(), 5, "cpu")
    assert seen[0][0] == "cpu"


def _hold(g, secs):
    """Whisper holding the card exclusively for ``secs``."""
    taken = threading.Event()

    def whisper():
        with g.exclusive("asr"):
            taken.set()
            time.sleep(secs)

    t = threading.Thread(target=whisper)
    t.start()
    taken.wait(5)
    return t


def test_the_full_computation_waits_for_the_card(gate, monkeypatch):
    """With three lanes Whisper holds the gate nearly all the time — five
    seconds sent a full kNN over a large voice inventory to the CPU
    (minutes instead of seconds). The full computation waits its turn."""
    g, seen = gate
    monkeypatch.setattr(nb, "GPU_WAIT_SEC", 5.0)
    monkeypatch.setattr(nb, "GPU_WAIT_UPDATE_SEC", 0.05)
    t = _hold(g, 0.6)
    try:
        nb.compute(_x(), 5, "cuda")
    finally:
        t.join()
    assert seen[0][0] == "cuda"


def test_the_default_wait_is_minutes_not_seconds():
    import inspect
    import re

    src = inspect.getsource(nb)
    full = float(re.search(r"^GPU_WAIT_SEC = ([\d.]+)", src, re.M).group(1))
    assert full >= 120


def test_a_big_update_does_not_wait_long(gate, monkeypatch):
    g, seen = gate
    monkeypatch.setattr(nb, "GPU_WAIT_SEC", 5.0)
    monkeypatch.setattr(nb, "GPU_WAIT_UPDATE_SEC", 0.05)
    monkeypatch.setattr(nb, "UPDATE_ON_CPU", 0)  # force the GPU route
    X = _x(400)
    ids = np.arange(400, dtype=np.int64)
    idx, v64, _ = nb.compute(X[:300], 5, "cpu")
    seen.clear()
    t = _hold(g, 1.6)  # the gate checks a give-up once a second
    try:
        nb.update(ids[:300], idx, v64, X, ids, "cuda")
    finally:
        t.join()
    assert seen and seen[0][0] == "cpu"


def test_without_cuda_no_gate_and_no_wait(gate, monkeypatch):
    g, seen = gate
    monkeypatch.setattr(nb, "_cuda_available", lambda: False)

    def no(*a, **k):
        raise AssertionError("no card, no gate")

    monkeypatch.setattr(g, "stage", no)
    nb.compute(_x(), 5, "cuda")
    assert seen[0][0] == "cpu"
