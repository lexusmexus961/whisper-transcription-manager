"""
Separation: Short files must not crash the separator anymore.

Roformer models process audio in windows of stft_hop × (dim_t − 1)
samples (mel_roformer_kim: 485 100 = 11.0 s @ 44.1 kHz). audio-separator
has a fallback for short files, but it only kicks in under 10.0 s —
files between 10.0 s and 11.0 s crashed ("The size of tensor a (0)
must match …"). The fix pads audio that's too short
before separation with silence up to _SEP_MIN_AUDIO_SEC and cuts the stems
back to exact original length afterwards (the reference
implementation ZFTurbo/Music-Source-Separation-Training does it the same way).

These tests check the padding/cutting with real ffmpeg,
without loading the separation model.
"""

import subprocess

import pytest

from whisper_tm.config import config
from whisper_tm.core import vocal_isolation as vi
from whisper_tm.core.audio_utils import get_duration_safe

# Every test here runs ffmpeg (skipped without it)
pytestmark = pytest.mark.ffmpeg


@pytest.fixture(autouse=True)
def _cfg():
    if not config.is_loaded:
        config.load()


def _make_wav(path, seconds, sr=48000):
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:duration={seconds}",
            "-ar",
            str(sr),
            "-ac",
            "2",
            "-acodec",
            "pcm_s16le",
            str(path),
        ],
        capture_output=True,
        check=True,
    )
    return path


def test_the_dead_zone_is_padded(tmp_path):
    """The lower edge: 10.0 s falls in the dead zone [10.0; 11.0)."""
    wav = _make_wav(tmp_path / "short.wav", 10.0)
    padded, orig_dur = vi._pad_for_separation(wav)
    assert orig_dur == pytest.approx(10.0, abs=0.1)
    assert padded != wav
    # Name must stay identical — the separator derives the stem names from it
    assert padded.name == wav.name
    assert get_duration_safe(padded) >= vi._SEP_MIN_AUDIO_SEC - 0.05


def test_a_very_short_file_is_padded(tmp_path):
    """Second dead zone: below the fallback window (~2.55 s)."""
    wav = _make_wav(tmp_path / "mini.wav", 1.5)
    padded, orig_dur = vi._pad_for_separation(wav)
    assert orig_dur == pytest.approx(1.5, abs=0.1)
    assert get_duration_safe(padded) >= vi._SEP_MIN_AUDIO_SEC - 0.05


def test_a_long_file_stays_untouched(tmp_path):
    wav = _make_wav(tmp_path / "long.wav", 15.0)
    padded, orig_dur = vi._pad_for_separation(wav)
    assert padded == wav  # no copy, no detour
    assert orig_dur is None  # → no trim-back


def test_trim_restores_original_length(tmp_path):
    """Timeline: stem after pad+unpad exactly as long as the original."""
    wav = _make_wav(tmp_path / "short.wav", 10.0)
    padded, orig_dur = vi._pad_for_separation(wav)
    # padded stands in for a stem at window length
    vi._unpad_stem(padded, orig_dur)
    assert get_duration_safe(padded) == pytest.approx(orig_dur, abs=0.05)


def test_the_minimum_length_covers_the_roformer_window():
    """Kim window: 441 × 1100 = 485,100 samples @ 44.1 kHz = 11.0 s —
    the threshold must lie above it (see model YAML inference.dim_t)."""
    assert vi._SEP_MIN_AUDIO_SEC > 441 * 1100 / 44_100
