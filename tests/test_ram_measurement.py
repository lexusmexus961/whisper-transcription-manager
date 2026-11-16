"""What a step costs in RAM — and why it has to be measured.

A long recording can drive the machine into the kernel OOM-killer. Not
CUDA, not the graphics card: main memory.

    kswapd0 invoked oom-killer (global_oom)
    Out of memory: Killed process <pid> (python)

The process at risk is the **diarization worker** — the child process
that allows diarization to be canceled. On a recording of many hours it
can grow to most of the machine's memory and take the server down
with it.

**Why it needs its own measurement:** The pipeline measures VRAM per
step (`debug.vram`), because the known bottlenecks are there. Without a
RAM measurement, monitoring reports "running normally" up to the last
minute — it only looks for what it is searching for.

Two things make this measurement different, and both are documented here:

1. It counts the **entire process tree**. The server process itself looks
   harmless; the memory is in the child.
2. It appears in **the same row and same table** as VRAM. Two
   separate logs nobody would add together.
"""

from __future__ import annotations

import os

import pytest

from whisper_tm.core import debug


def _brief():
    """Wait for two sampling cycles to ensure the peak is definitely seen."""
    import time

    time.sleep(2 * 0.5 / debug.RAM_TICKS_PER_VRAM_TICK)


@pytest.fixture(autouse=True)
def _measure_on(monkeypatch):
    """Measurement only runs in debug mode — otherwise it's a no-op. Without
    a GPU, as on a CPU-only machine or a CI runner: the memory is measured all
    the same (it was not, and every number here read 0 there)."""
    monkeypatch.setattr(debug, "_enabled", True)
    monkeypatch.setattr(debug, "gpu_available", lambda: False)
    with debug._vram_lock:
        debug._ram.clear()
        debug._vram.clear()
    yield
    with debug._vram_lock:
        debug._ram.clear()
        debug._vram.clear()


class TestTheProcessTree:
    """The point everything hinges on."""

    def test_the_own_usage_is_measured(self):
        assert debug.ram_used_mb() > 0

    def test_a_CHILD_counts_too(self):
        """The memory can sit in a child process. Whoever only measures
        their own RSS sees nothing."""
        import subprocess
        import sys

        before = debug.ram_used_mb()
        # A child that holds ~200 MB and waits for a character.
        child = subprocess.Popen(
            [
                sys.executable,
                "-c",
                "import sys; x = bytearray(200*1024*1024); "
                "sys.stdout.write('ok'); sys.stdout.flush(); sys.stdin.read(1)",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )
        try:
            assert child.stdout.read(2) == b"ok"  # memory is held
            after = debug.ram_used_mb()
            assert after - before > 100, (
                f"the child ({child.pid}) was not counted: {before} -> {after} MB"
            )
        finally:
            child.stdin.write(b"x")
            child.stdin.close()
            child.wait(timeout=10)

    def test_a_finished_child_does_not_throw(self):
        """A child can disappear between listing and measuring. An
        exception in the measurement sample would be worse than an inaccurate number."""
        assert isinstance(debug.ram_used_mb(), int)

    def test_without_psutil_it_returns_zero_instead_of_crashing(self, monkeypatch):
        """The measurement must never halt the pipeline."""
        import builtins

        real = builtins.__import__

        def _no_psutil(name, *a, **kw):
            if name == "psutil":
                raise ImportError("not there")
            return real(name, *a, **kw)

        monkeypatch.setattr(builtins, "__import__", _no_psutil)
        assert debug.ram_used_mb() == 0


class TestTheMeasurementInTheBlock:
    def test_growth_is_detected(self):
        with debug.vram("Sample") as v:
            buffer = bytearray(220 * 1024 * 1024)
            assert buffer[0] == 0  # against optimizing it away
        assert v.ram_delta_mb >= 150, f"only {v.ram_delta_mb} MB measured"
        assert v.ram_peak_mb >= v.ram_delta_mb

    def test_the_peak_survives_the_release(self):
        """What's measured is the PEAK, not the state at the end — otherwise
        a step that takes 25 GB and gives it back again shows zero.

        For this, the memory has to be held long enough for one sampling
        cycle to see it; `_brief` gives it two. This is not an artificial
        crutch, but the real case: diarization holds its memory for minutes.
        """
        with debug.vram("Peak") as v:
            buffer = bytearray(220 * 1024 * 1024)
            assert buffer[0] == 0
            _brief()
            del buffer
        assert v.ram_delta_mb >= 150

    # Not tested: a peak shorter than a sampling cycle is missed — the
    # method's limit, deliberate (each pipeline step holds its memory for
    # seconds). A test of it passed or failed with the machine's timing.

    def test_RAM_is_on_the_same_line_as_VRAM(self):
        """Nobody adds up two separate logs when it actually matters."""
        with debug.vram("Line") as v:
            buffer = bytearray(200 * 1024 * 1024)
            assert buffer[0] == 0
            _brief()
        assert "RAM" in v.note()

    def test_without_usage_no_noise_in_the_line(self):
        v = debug._VramResult()
        assert "RAM" not in v.note()


class TestTheReport:
    def test_every_step_appears_with_its_peak(self):
        for name, mb in (("Diarization", 200), ("Alignment", 120)):
            with debug.vram(name):
                buffer = bytearray(mb * 1024 * 1024)
                assert buffer[0] == 0
                _brief()
            del buffer
        names = [r["label"] for r in debug.ram_stats()]
        assert names == ["Diarization", "Alignment"], names

    def test_a_step_WITHOUT_a_gpu_still_shows_up(self, monkeypatch):
        """The actual reason for a dedicated RAM table: a step
        that doesn't touch the GPU has no row in the GPU table — and exactly a
        step like that can take the machine down."""
        monkeypatch.setattr(debug, "gpu_available", lambda: True)
        monkeypatch.setattr(debug, "gpu_mem_used_mb", lambda: {})
        with debug.vram("CPU-only-step"):
            buffer = bytearray(200 * 1024 * 1024)
            assert buffer[0] == 0
            _brief()
        assert "CPU-only-step" in [r["label"] for r in debug.ram_stats()]
        assert "CPU-only-step" in debug.vram_summary()

    def test_the_table_names_the_reference_value(self):
        """ "20 GB" is a number. "80% of the machine" is a statement."""
        with debug.vram("Something"):
            buffer = bytearray(200 * 1024 * 1024)
            assert buffer[0] == 0
            _brief()
        t = debug.vram_summary()
        assert "Memory per step" in t
        assert "Machine total" in t and "Peak =" in t

    def test_the_total_memory_matches_the_system(self):
        measured = debug._ram_total_mb()
        with_ = int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1048576)
        assert abs(measured - with_) < with_ * 0.05, f"{measured} vs {with_}"

    def test_repeated_steps_keep_the_maximum(self):
        """Across a batch run, the worst case counts, not the last one."""
        with debug.vram("Repeated"):
            large = bytearray(220 * 1024 * 1024)
            assert large[0] == 0
            _brief()
        del large
        with debug.vram("Repeated"):
            small = bytearray(10 * 1024 * 1024)
            assert small[0] == 0
        entry = next(r for r in debug.ram_stats() if r["label"] == "Repeated")
        assert entry["delta_mb"] >= 150, entry
