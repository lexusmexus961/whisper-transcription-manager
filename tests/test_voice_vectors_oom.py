"""Voice vectors at full VRAM: save them, or fail — never stay silent.

**Observed in a running batch** (another program on the GPU at the same
time): a file finished as ✅, but in the last step it said again and again

    Voice vector SPEAKER_01#1 skipped: CUDA out of memory … MiB free

and only a few of its windows remained. The same warning appeared many
times in the log, in large blocks. If the voice vectors fail, the whole job must fail —
otherwise its result is invalid. And was there perhaps no speech in the
windows? No — such windows fall out in ``plan_windows`` before
calculation; the skipped ones were real losses.

The sequence now: clear cache and try again → CPU for the remainder of the
file → only then an error, with "out of memory" in the text, so the queue
retries once with VRAM conservation.
"""

from __future__ import annotations

import numpy as np
import pytest

from whisper_tm.core import speaker as sp

OOM = (
    "CUDA out of memory. Tried to allocate 40.00 MiB. GPU 0 has a total "
    "capacity of 12.00 GiB of which 100.00 MiB is free."
)


def _vec(seed, dim=192):
    r = np.random.default_rng(seed).standard_normal(dim).astype(np.float32)
    return r / np.linalg.norm(r)


class _Model:
    """Replica of Embedder: fails by script, counts calls per device."""

    def __init__(self, script):
        # script: list of Exceptions or None per GPU call
        self.script = list(script)
        self.calls = {"cuda": 0, "cpu": 0}

    def load(self, device="cuda", job=None):
        def compute(w):
            self.calls[device] += 1
            if device == "cuda" and self.script:
                f = self.script.pop(0)
                if f is not None:
                    raise f
            return _vec(1)

        return compute


@pytest.fixture
def plan_and_audio():
    audio = np.zeros(120 * sp.SR, dtype=np.float32)
    plan = sp.plan_windows([{"start": 0.0, "end": 120.0, "speaker": "A"}])
    assert len(plan) == 4
    return audio, plan


@pytest.fixture(autouse=True)
def _no_real_cache_clear(monkeypatch):
    cleared = []
    monkeypatch.setattr(sp, "_clear_gpu_cache", lambda: cleared.append(1))
    return cleared


class TestSaving:
    def test_second_attempt_after_cache_clear(
        self, monkeypatch, plan_and_audio, _no_real_cache_clear
    ):
        """A one-time bottleneck costs no window and no CPU round."""
        m = _Model([RuntimeError(OOM)])  # only first call stalls
        monkeypatch.setattr(sp, "load_model", m.load)
        audio, plan = plan_and_audio
        rows = sp.compute_vectors(audio, plan, device="cuda")
        assert len(rows) == 4
        assert m.calls == {"cuda": 5, "cpu": 0}
        assert _no_real_cache_clear == [1]

    def test_if_full_cpu_computes_rest(self, monkeypatch, plan_and_audio):
        """Two OOMs in a row → remainder of file runs on CPU."""
        m = _Model([RuntimeError(OOM), RuntimeError(OOM)])
        monkeypatch.setattr(sp, "load_model", m.load)
        audio, plan = plan_and_audio
        rows = sp.compute_vectors(audio, plan, device="cuda")
        assert len(rows) == 4  # no window missing
        assert m.calls["cuda"] == 2  # no GPU after that
        assert m.calls["cpu"] == 4

    def test_cpu_round_in_job_log(self, monkeypatch, plan_and_audio):
        m = _Model([RuntimeError(OOM), RuntimeError(OOM)])
        monkeypatch.setattr(sp, "load_model", m.load)
        warnings = []
        from whisper_tm import extensions

        class _JM:
            def warn(self, job, text):
                warnings.append(text)

        # outside the app, job_manager is None
        monkeypatch.setattr(extensions, "job_manager", _JM())
        audio, plan = plan_and_audio
        sp.compute_vectors(audio, plan, device="cuda", job=object())
        assert any("CPU" in w for w in warnings)


class TestFailing:
    def test_no_silent_gap_anymore(self, monkeypatch, plan_and_audio):
        """The actual incident: an error that is not an OOM flies out —
        instead of silently skipping the window as before."""
        m = _Model([None, RuntimeError("CUDA error: an illegal memory access")])
        monkeypatch.setattr(sp, "load_model", m.load)
        audio, plan = plan_and_audio
        with pytest.raises(RuntimeError, match="Voice vector A#2"):
            sp.compute_vectors(audio, plan, device="cuda")

    def test_cpu_also_fails_but_oom_text_is_preserved(self, monkeypatch, plan_and_audio):
        """The queue recognizes an OOM only by the text — it must not be lost,
        otherwise there is no retry with VRAM conservation."""

        class _BrokenCPU(_Model):
            def load(self, device="cuda", job=None):
                if device == "cpu":

                    def broken(w):
                        raise RuntimeError("CPU: " + OOM)

                    return broken
                return super().load(device, job)

        m = _BrokenCPU([RuntimeError(OOM), RuntimeError(OOM)])
        monkeypatch.setattr(sp, "load_model", m.load)
        audio, plan = plan_and_audio
        with pytest.raises(RuntimeError) as e:
            sp.compute_vectors(audio, plan, device="cuda")
        assert "out of memory" in str(e.value).lower()
        assert "also fails on" in str(e.value)


class TestWorkerLetsItThrough:
    def test_voice_vector_error_no_longer_becomes_warning(self):
        """transcription.py used to catch EVERYTHING here and wrote warn()."""
        from pathlib import Path

        source = Path("whisper_tm/core/transcription.py").read_text()
        assert "voice vectors skipped" not in source
        i = source.index('raise RuntimeError(f"Voice vectors failed')
        block = source[i - 1200 : i]
        # InterruptedError (user cancellation) still passes through
        assert "except InterruptedError:" in block

    def test_alignment_block_error_no_longer_becomes_warning(self):
        from pathlib import Path

        source = Path("whisper_tm/core/transcription.py").read_text()
        assert "aligned.extend(chunk_segs)" not in source
        assert 'f"Alignment block {ci + 1}/{nc} failed: ' in source


class TestNoSpeechIsNotError:
    def test_silent_windows_drop_out_before_the_computation(self):
        """The question "or was there no speech?" — windows
        below MIN_SPEECH never enter the plan."""
        plan = sp.plan_windows([{"start": 0.0, "end": 1.0, "speaker": "A"}])
        assert plan == []
