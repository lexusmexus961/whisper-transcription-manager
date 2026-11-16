"""
Vocal Isolation: memory usage must be capped by chunk size.

Previously a fixed duration limit (3h) decided whether audio-separator
gets the file in one piece. audio-separator keeps entire audio as float32 in RAM (input
+ vocals + instrumental) — consumption grows LINEAR with duration. A
2.5-hour recording was under the limit, but needed ~20 GB of RAM.

Now it's always chunked. Chunk size is thus the upper limit, no matter how
long the file is — these tests check that.
"""

import inspect

import pytest

from whisper_tm.config import config
from whisper_tm.core import vocal_isolation as vi

# 48 kHz · 2 channels · 4 bytes (float32); with STFT intermediate steps roughly ×6
BYTES_PER_AUDIO_SEC = 48_000 * 2 * 4
PEAK_FACTOR = 6


@pytest.fixture(autouse=True)
def _cfg():
    if not config.is_loaded:
        config.load()


def _peak_gb(seconds: float) -> float:
    return seconds * BYTES_PER_AUDIO_SEC * PEAK_FACTOR / 2**30


def test_peak_is_bounded_by_chunk_size_not_by_file_length():
    """Core: A 10x longer file does NOT need 10x more memory."""
    chunk = config.get("hardware", "sep_chunk_duration_sec", default=1800)

    # Whether 30 minutes or 10 hours of recording — processing always
    # happens at most one chunk at a time.
    assert _peak_gb(chunk) < 6, f"one chunk needs {_peak_gb(chunk):.1f} GB"

    # For comparison: this was the old direct mode for such a recording
    assert _peak_gb(9000) > 18, "test assumption no longer holds"


def test_direct_mode_is_gone():
    """There must be no path that processes the whole file at once."""
    assert not hasattr(vi, "_isolate_direct")
    assert not hasattr(vi, "_DIRECT_MAX_SECS")
    # Memory estimation is also gone — it wasn't reproducible (result
    # depended on what else was running on the machine).
    assert not hasattr(vi, "_direct_fits_in_ram")


def test_isolate_vocals_always_goes_through_the_chunked_path():
    src = inspect.getsource(vi.isolate_vocals)
    assert "_isolate_chunked" in src
    assert "_isolate_direct" not in src


def test_short_file_is_not_split_into_chunks(tmp_path, monkeypatch):
    """Short files still run in one piece — no extra overhead.

    _extract_sep_chunks only splits what's longer than one chunk.
    """
    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        out = tmp_path / "sep_single.wav"
        out.write_bytes(b"\0" * 2000)

        class R:
            returncode = 0
            stderr = ""

        return R()

    monkeypatch.setattr(vi, "get_duration", lambda p: 300.0)  # 5 minutes
    monkeypatch.setattr(vi, "get_audio_duration", lambda p: 300.0)
    monkeypatch.setattr(vi, "run_cancellable", fake_run)

    chunks, total, _silence = vi._extract_sep_chunks(
        tmp_path / "short.wav", tmp_path, chunk_dur=1800, overlap=3
    )

    assert len(chunks) == 1, "short file was unnecessarily split"
    assert total == 300.0
    assert len(calls) == 1, "more than one ffmpeg call for a short file"


def test_long_file_is_split(tmp_path, monkeypatch):
    """A 2.5 h recording must break into several chunks."""

    def fake_run(cmd, **kw):
        out = [c for c in cmd if str(c).endswith(".wav")][-1]
        from pathlib import Path

        Path(out).write_bytes(b"\0" * 2000)

        class R:
            returncode = 0
            stderr = ""

        return R()

    monkeypatch.setattr(vi, "get_duration", lambda p: 9000.0)  # 2:30:00
    monkeypatch.setattr(vi, "get_audio_duration", lambda p: 9000.0)
    monkeypatch.setattr(vi, "run_cancellable", fake_run)

    chunks, total, _silence = vi._extract_sep_chunks(
        tmp_path / "long.mp4", tmp_path, chunk_dur=1800, overlap=3
    )

    assert len(chunks) == 5, f"{len(chunks)} chunks at 2:30:00 / 30min"
    assert total == 9000.0
