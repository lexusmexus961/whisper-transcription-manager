"""Waveform and loudness (whisper_tm/audio/levels.py) against what EBU
R128 / ITU-R BS.1770-4 define."""

from __future__ import annotations

import numpy as np
import pytest

from whisper_tm.audio import levels
from whisper_tm.audio.levels import SR, Levels


def _measure(x: np.ndarray, piece: int = 0) -> dict:
    lv = Levels()
    if piece:
        for i in range(0, len(x), piece):
            lv.feed(x[i : i + piece])
    else:
        lv.feed(x)
    return lv.finish()


def _sine(freq: float, amp: float, seconds: float) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_k_weighting_is_the_standards_at_48k():
    """The analog prototypes reproduce BS.1770's 48 kHz coefficients."""
    b1, a1, b2, a2 = levels._biquad_k(48000)
    np.testing.assert_allclose(
        b1, [1.53512485958697, -2.69169618940638, 1.19839281085285], atol=1e-8
    )
    np.testing.assert_allclose(a1, [1.0, -1.69065929318241, 0.73248077421585], atol=1e-8)
    np.testing.assert_allclose(b2, [1.0, -2.0, 1.0])
    np.testing.assert_allclose(a2, [1.0, -1.99004745483398, 0.99007225036621], atol=1e-8)


def test_a_sine_at_minus_20_dbfs_is_minus_23_lufs():
    """BS.1770: a 997 Hz sine at full scale reads -3.01 LUFS; 20 dB down
    -23.01 (here at 16 kHz, within a few hundredths)."""
    r = _measure(_sine(997, 0.1, 20))
    assert r["lufs"] == pytest.approx(-23.0, abs=0.1)
    assert r["peak_db"] == pytest.approx(-20.0, abs=0.05)
    assert r["silence"] == 0.0 and r["clipped"] == 0.0
    assert r["lra"] == pytest.approx(0.0, abs=0.1)
    assert r["duration"] == pytest.approx(20.0)


def test_pieces_measure_as_the_whole():
    rng = np.random.default_rng(1)
    x = (rng.normal(0, 0.1, SR * 30) * np.repeat(rng.uniform(0.05, 1, 30), SR)).astype(np.float32)
    whole = _measure(x)
    parts = _measure(x, piece=12_345)
    assert whole["levels"] == parts["levels"]
    for k in ("lufs", "lra", "peak_db", "silence", "clipped", "duration"):
        assert parts[k] == pytest.approx(whole[k], abs=1e-6), k


def test_loudness_range_of_two_levels():
    """Half the time 20 dB quieter: the short-term loudness has two levels,
    and the range (P10–P95, gated 20 LU below) spans them."""
    loud = _sine(997, 0.3, 30)
    soft = _sine(997, 0.03, 30)
    r = _measure(np.concatenate([loud, soft]))
    assert r["lra"] == pytest.approx(20.0, abs=0.6)


def test_silence_and_too_short():
    r = _measure(np.zeros(SR * 5, np.float32))
    assert r["lufs"] is None and r["peak_db"] is None
    assert r["silence"] == 1.0
    r = _measure(np.zeros(SR // 10, np.float32))  # 100 ms: no 400 ms block
    assert r["lufs"] is None and r["silence"] is None and r["lra"] is None


def test_quiet_parts_count_as_silence():
    x = np.concatenate([_sine(440, 0.3, 6), np.zeros(SR * 4, np.float32)])
    r = _measure(x)
    assert r["silence"] == pytest.approx(0.4, abs=0.03)


def test_clipping():
    x = np.clip(_sine(220, 2.0, 2), -1.0, 1.0)
    r = _measure(x)
    assert r["clipped"] > 0.5
    assert r["peak_db"] == pytest.approx(0.0, abs=0.01)


def test_waveform_is_core_waveforms_format():
    """20 values a second, the peak of each 50 ms on -60…0 dBFS → 0…255 —
    what the clip editor's overview (core/waveform) holds."""
    from whisper_tm.core import waveform

    x = np.concatenate([_sine(440, a, 1) for a in (1.0, 0.1, 0.001, 0.0)])
    r = _measure(x)
    lv = np.frombuffer(r["levels"], np.uint8)
    assert len(lv) == 4 * waveform.RATE
    assert waveform.RATE == levels.RATE and waveform.FLOOR_DB == levels.FLOOR_DB
    ints = (np.abs(x) * 32768).astype(np.int32).reshape(-1, SR // waveform.RATE).max(axis=1)
    assert np.all(np.abs(lv.astype(int) - waveform._levels(ints).astype(int)) <= 1)
    assert lv[0] == 255 and lv[-1] == 0


def test_int16_input():
    x = _sine(997, 0.1, 5)
    a = _measure(x)
    b = _measure((x * 32768).astype(np.int16))
    assert b["lufs"] == pytest.approx(a["lufs"], abs=0.01)
