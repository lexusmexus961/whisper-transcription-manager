"""An OOM in Vocal Isolation must arrive as an OOM.

**The failure mode**: with another program using the GPU at the same
time, several files in a row fail permanently ❌ with

    Separation failed after 2 attempts: no output files

while above it in the log stood

    audio_separator: Failed to process file …: CUDA out of memory.
    Tried to allocate 128.00 MiB … 100.00 MiB is free

``Separator.separate()`` catches every exception itself, logs it and returns
an empty list. Our error text didn't know about the OOM — and
the queue, which retries once with VRAM conservation on "out of memory"
(``core/queue.py``), only saw "no output files" and gave up on the file.

A related problem: turning on ``hardware.low_vram``
in settings had no effect, because each waiting
entry had ``low_vram: false`` from the dialog baked in, and that
overrode the config. Whether memory is tight is decided when
calculating, not when queueing: the global setting is now a floor.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from whisper_tm.core import vocal_isolation as vi
from whisper_tm.core import vram


class _SeparatorMock:
    """Mock of audio_separator: log errors, return nothing."""

    def __init__(self, error, output_dir=None):
        self.error = error
        self.output_dir = output_dir
        self.calls = 0

    def separate(self, path):
        self.calls += 1
        if self.error is None:
            (Path(self.output_dir) / "vocals.wav").write_bytes(b"\0" * 2000)
            return ["vocals.wav"]
        logging.getLogger("audio_separator.separator.separator").error(
            "Failed to process file %s: %s", path, self.error
        )
        return []


@pytest.fixture
def separator_environment(tmp_path, monkeypatch):
    """_run_separator without GPU, without ffmpeg, without model."""
    api_out = tmp_path / "api_out"
    api_out.mkdir()
    input_file = tmp_path / "in.wav"
    input_file.write_bytes(b"RIFF" + b"\0" * 100)
    holder = {}

    def _build(error):
        sep = _SeparatorMock(error, api_out)
        holder["sep"] = sep
        return sep

    monkeypatch.setattr(vi, "_sep_api_dir", lambda: api_out)
    monkeypatch.setattr(vi, "_pad_for_separation", lambda p, job=None: (Path(p), None))
    monkeypatch.setattr(vi, "_get_separator", lambda *a, **k: holder["sep"])
    monkeypatch.setattr(vi.time, "sleep", lambda s: None)
    return input_file, tmp_path / "out", _build, holder


class TestErrorTextKnowsOOM:
    OOM = (
        "CUDA out of memory. Tried to allocate 128.00 MiB. GPU 0 has a "
        "total capacity of 12.00 GiB of which 100.00 MiB is free."
    )

    def test_oom_appears_in_error_text(self, separator_environment):
        """The actual fix: the queue checks ``"out of memory" in error``."""
        input_file, out, build, _ = separator_environment
        build(self.OOM)
        with pytest.raises(RuntimeError) as e:
            vi._run_separator(input_file, out, "mel_roformer_kim.ckpt")
        assert "out of memory" in str(e.value).lower()
        assert "no output files" in str(e.value).lower()

    def test_path_stays_outside(self, separator_environment):
        """\"Failed to process file /…/sep_in_0000.wav: \" is just noise."""
        input_file, out, build, _ = separator_environment
        build(self.OOM)
        with pytest.raises(RuntimeError) as e:
            vi._run_separator(input_file, out, "m.ckpt")
        assert "Failed to process file" not in str(e.value)
        assert str(input_file) not in str(e.value)

    def test_no_second_attempt_on_oom(self, separator_environment):
        """Three seconds later the memory still belongs to the other program."""
        input_file, out, build, holder = separator_environment
        build(self.OOM)
        with pytest.raises(RuntimeError) as e:
            vi._run_separator(input_file, out, "m.ckpt", retry=1)
        assert holder["sep"].calls == 1
        assert "after 1 attempt" in str(e.value).lower()

    def test_other_errors_continue_retrying(self, separator_environment):
        """For comparison: the second attempt remains for everything else."""
        input_file, out, build, holder = separator_environment
        build("sep_single.wav is empty or not valid")
        with pytest.raises(RuntimeError) as e:
            vi._run_separator(input_file, out, "m.ckpt", retry=1)
        assert holder["sep"].calls == 2
        assert "after 2 attempts" in str(e.value).lower()
        assert "is empty or not valid" in str(e.value)

    def test_success_stays_success(self, separator_environment):
        input_file, out, build, _ = separator_environment
        build(None)
        result = vi._run_separator(input_file, out, "m.ckpt")
        assert [p.name for p in result] == ["vocals.wav"]

    def test_handler_no_longer_attached_to_logger_afterwards(self, separator_environment):
        """Otherwise a handler accumulates per chunk."""
        input_file, out, build, _ = separator_environment
        build(None)
        vi._run_separator(input_file, out, "m.ckpt")
        lg = logging.getLogger("audio_separator")
        assert not any(isinstance(h, vi._SeparatorErrorLogger) for h in lg.handlers)


class TestTheEntryDecides:
    @pytest.fixture(autouse=True)
    def _cleanup(self):
        yield
        vram.set_low_vram_override(None)

    def _config(self, monkeypatch, value):
        monkeypatch.setattr(
            vram.config,
            "get",
            lambda *path, default=None: value if path == ("hardware", "low_vram") else default,
        )

    def test_entry_false_wins_over_global_true(self, monkeypatch):
        """Switched off in the dialog, yet every run unloaded."""
        self._config(monkeypatch, True)
        vram.set_low_vram_override(False)
        assert vram.low_vram() is False

    def test_without_a_choice_the_global_default_applies(self, monkeypatch):
        self._config(monkeypatch, True)
        vram.set_low_vram_override(None)
        assert vram.low_vram() is True

    def test_entry_true_switches_on(self, monkeypatch):
        """The OOM retry attempt sets exactly that."""
        self._config(monkeypatch, False)
        vram.set_low_vram_override(True)
        assert vram.low_vram() is True

    def test_without_both_off(self, monkeypatch):
        self._config(monkeypatch, False)
        vram.set_low_vram_override(None)
        assert vram.low_vram() is False
        vram.set_low_vram_override(False)
        assert vram.low_vram() is False
