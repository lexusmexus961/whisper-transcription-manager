"""The diarization worker's VRAM limit (hardware.diarization_vram_cap_mb).

pyannote itself needs under 2 GiB, but cuDNN takes a convolution workspace
of most of the card whenever the card is free — the worker near 10 GiB
for a few-second voice note, a 12 GiB card almost full. Limited per
process, cuDNN falls back to a frugal algorithm: the same segments, and
faster. See ``diar_worker._set_vram_cap``.

Like ``test_diar_cancellable``, these run the REAL worker as a child
process — only ``torch`` and ``whisperx`` are small stand-ins put in front
on PYTHONPATH. The stand-in torch records every limit the worker sets, so
the order "limit → model → request → lift → retry → limit again" is
checked on the real protocol, without a GPU.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from whisper_tm.core import diar_process

REPO = Path(__file__).resolve().parent.parent

FAKE_TORCH = """
    import json, os

    def _log(*entry):
        with open(os.environ["FAKE_TORCH_LOG"], "a") as fh:
            fh.write(json.dumps(entry) + "\\n")

    class OutOfMemoryError(RuntimeError):
        pass

    class _Device:
        def __init__(self, spec):
            self.type, _, index = str(spec).partition(":")
            self.index = int(index) if index else None

    def device(spec):
        return _Device(spec)

    class _Props:
        total_memory = 12 * 2**30          # a 12 GiB card

    class _Cuda:
        OutOfMemoryError = OutOfMemoryError

        def current_device(self):
            return 0

        def get_device_properties(self, index):
            return _Props()

        def set_per_process_memory_fraction(self, fraction, index):
            # The real one raises a TypeError for anything but a float
            assert isinstance(fraction, float), type(fraction)
            _log("limit", round(fraction, 4), index)

        def empty_cache(self):
            pass

    cuda = _Cuda()
"""

FAKE_WHISPERX = """
    def load_audio(path):
        return [0.0] * 16000
"""

FAKE_DIARIZE = """
    import json, os

    class _Frame:
        def __init__(self, rows):
            self.rows = rows

        def iterrows(self):
            return enumerate(self.rows)

    class DiarizationPipeline:
        def __init__(self, token=None, device="cpu"):
            self.calls = 0

        def __call__(self, audio, **kwargs):
            self.calls += 1
            with open(os.environ["FAKE_TORCH_LOG"], "a") as fh:
                fh.write(json.dumps(["diarize", self.calls]) + "\\n")
            mode = os.environ.get("FAKE_FAILURE", "")
            if mode == "always" or (mode == "once" and self.calls == 1):
                import torch
                raise torch.cuda.OutOfMemoryError(
                    "CUDA out of memory. Tried to allocate 9.16 GiB")
            if mode == "pyannote_always" or (mode == "pyannote" and self.calls == 1):
                # What pyannote's Inference really does with a CUDA OOM
                try:
                    raise RuntimeError("CUDA out of memory. Tried to allocate 20.00 MiB")
                except RuntimeError:
                    raise MemoryError("batch_size ( 32) is probably too large. Try "
                                      "with a smaller value until memory error disappears.")
            if mode == "corrupt":
                raise ValueError("File is corrupt")
            return _Frame([{"start": 0.0, "end": 1.5, "speaker": "SPEAKER_00"}])
"""


@pytest.fixture
def fakes(tmp_path, monkeypatch):
    """Stand-in torch/whisperx for the child process; returns its log reader."""
    root = tmp_path / "fakes"
    (root / "torch").mkdir(parents=True)
    (root / "whisperx").mkdir()
    (root / "torch" / "__init__.py").write_text(textwrap.dedent(FAKE_TORCH))
    (root / "whisperx" / "__init__.py").write_text(textwrap.dedent(FAKE_WHISPERX))
    (root / "whisperx" / "diarize.py").write_text(textwrap.dedent(FAKE_DIARIZE))
    log = tmp_path / "torch.log"
    monkeypatch.setenv("PYTHONPATH", f"{root}:{REPO}")
    monkeypatch.setenv("FAKE_TORCH_LOG", str(log))
    monkeypatch.delenv("FAKE_FAILURE", raising=False)

    def entries():
        if not log.exists():
            return []
        return [json.loads(line) for line in log.read_text().splitlines()]

    yield entries
    diar_process.terminate_all("Test ended")


def _cap(monkeypatch, value):
    from whisper_tm.config import config

    real = config.get

    def faked(*path, default=None):
        if path == ("hardware", "diarization_vram_cap_mb"):
            return value
        return real(*path, default=default)

    monkeypatch.setattr(config, "get", faked)


def _worker(device="cuda"):
    # The real command line — the wiring from config to child is the point
    return diar_process.Worker(device)


class TestTheLimit:
    def test_the_limit_is_set_before_the_model_loads(self, fakes, monkeypatch, tmp_path):
        _cap(monkeypatch, 4096)
        w = _worker()
        segments = w.diarize_it(tmp_path / "a.wav")
        assert segments == [(0.0, 1.5, "SPEAKER_00")]
        # 4096 MiB of a 12 GiB card
        assert fakes() == [["limit", 0.3333, 0], ["diarize", 1]]

    def test_zero_means_no_limit(self, fakes, monkeypatch, tmp_path):
        _cap(monkeypatch, 0)
        _worker().diarize_it(tmp_path / "a.wav")
        assert fakes() == [["diarize", 1]]

    def test_a_cpu_worker_gets_no_limit(self, fakes, monkeypatch, tmp_path):
        _cap(monkeypatch, 4096)
        _worker("cpu").diarize_it(tmp_path / "a.wav")
        assert fakes() == [["diarize", 1]]

    def test_a_limit_above_the_card_is_the_whole_card(self, fakes, monkeypatch, tmp_path):
        _cap(monkeypatch, 65536)
        _worker().diarize_it(tmp_path / "a.wav")
        assert fakes()[0] == ["limit", 1.0, 0]


class TestSoftLimit:
    def test_a_request_that_needs_more_runs_once_without_it(
        self, fakes, monkeypatch, tmp_path, caplog
    ):
        _cap(monkeypatch, 4096)
        monkeypatch.setenv("FAKE_FAILURE", "once")
        w = _worker()
        with caplog.at_level("INFO", logger="whisper_tm.core.diar_process"):
            segments = w.diarize_it(tmp_path / "a.wav")
        assert segments == [(0.0, 1.5, "SPEAKER_00")]
        assert fakes() == [
            ["limit", 0.3333, 0],
            ["diarize", 1],
            ["limit", 1.0, 0],
            ["diarize", 2],
            ["limit", 0.3333, 0],
        ]  # and limited again
        assert "ran once without it" in caplog.text
        # The next request runs limited, in the same worker
        w.diarize_it(tmp_path / "b.wav")
        assert fakes()[-1] == ["diarize", 3]

    def test_pyannotes_own_memory_error_counts_too(self, fakes, monkeypatch, tmp_path):
        # Measured on the GPU with a 200 MiB limit: pyannote re-raises the
        # CUDA OOM as MemoryError("batch_size (32) is probably too large")
        _cap(monkeypatch, 4096)
        monkeypatch.setenv("FAKE_FAILURE", "pyannote")
        segments = _worker().diarize_it(tmp_path / "a.wav")
        assert segments == [(0.0, 1.5, "SPEAKER_00")]
        assert ["limit", 1.0, 0] in fakes()

    def test_a_wrapped_oom_is_reported_as_one(self, fakes, monkeypatch, tmp_path):
        # Card really full: the retry fails too — the error must say "out of
        # memory", or the queue books it as an error instead of waiting
        _cap(monkeypatch, 4096)
        monkeypatch.setenv("FAKE_FAILURE", "pyannote_always")
        with pytest.raises(RuntimeError, match="CUDA out of memory — batch_size"):
            _worker().diarize_it(tmp_path / "a.wav")

    def test_a_real_shortage_still_fails_and_restores_the_limit(self, fakes, monkeypatch, tmp_path):
        _cap(monkeypatch, 4096)
        monkeypatch.setenv("FAKE_FAILURE", "always")
        w = _worker()
        with pytest.raises(RuntimeError, match="out of memory"):
            w.diarize_it(tmp_path / "a.wav")
        assert fakes()[-1] == ["limit", 0.3333, 0]
        assert w.alive  # a failed request keeps the worker

    def test_other_errors_are_not_retried(self, fakes, monkeypatch, tmp_path):
        _cap(monkeypatch, 4096)
        monkeypatch.setenv("FAKE_FAILURE", "corrupt")
        with pytest.raises(RuntimeError, match="File is corrupt"):
            _worker().diarize_it(tmp_path / "a.wav")
        assert fakes() == [["limit", 0.3333, 0], ["diarize", 1]]
