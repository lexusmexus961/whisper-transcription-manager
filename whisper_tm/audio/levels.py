"""Waveform and loudness of a recording, fed in pieces.

* **The waveform** — 20 values a second, each the peak of its 50 ms in
  decibels, -60…0 dBFS onto 0…255: the clip editor's overview
  (core/waveform.py), which then needs no decoding of its own. 72 kB an
  hour.
* **Loudness** as EBU R128 / ITU-R BS.1770-4 measures it: the sound
  K-weighted (a high shelf of +4 dB above ~1.7 kHz, a high-pass at 38 Hz),
  its mean square every 100 ms; 400 ms blocks give the momentary loudness,
  3 s windows the short-term loudness.

  - ``lufs``: integrated loudness — blocks above -70 LUFS, then those at
    most 10 LU below their mean;
  - ``lra``: loudness range (EBU Tech 3342) — the 10th to 95th percentile
    of the short-term loudness, gated at -70 LUFS and 20 LU below;
  - ``peak_db``: the highest sample, dBFS;
  - ``silence``: the share of 400 ms blocks quieter than ``SILENT_LUFS``;
  - ``clipped``: the share of samples at full scale.

Measured from the 16 kHz mono sound the transcription decodes anyway: one
pass serves both. Loudness above 8 kHz is not in it — for speech nothing,
for bright music a few tenths of a dB; the peak is the 16 kHz signal's,
not the true peak of the original.
"""

from __future__ import annotations

import math

import numpy as np

SR = 16000
#: Waveform values a second — the format of core/waveform.py
RATE = 20
FLOOR_DB = -60.0
#: Momentary loudness below this counts as silence
SILENT_LUFS = -55.0
#: A 100 ms step of the loudness measurement
_STEP = SR // 10
_ABS_GATE = -70.0
#: Full scale counts as clipped from here on (int16 32,734 and above)
_CLIP = 0.999


def _biquad_k(fs: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """BS.1770's two K-weighting stages for sample rate ``fs`` — the analog
    prototypes behind the standard's 48 kHz coefficients (B. De Man,
    "Evaluation of implementations of the EBU R128 loudness measurement",
    AES 2016), which they reproduce exactly at 48 kHz."""
    # Stage 1: high shelf
    f0, g, q = 1681.974450955533, 3.999843853973347, 0.7071752369554196
    k = math.tan(math.pi * f0 / fs)
    vh = 10.0 ** (g / 20.0)
    vb = vh**0.4996667741545416
    a0 = 1.0 + k / q + k * k
    b1 = np.array(
        [(vh + vb * k / q + k * k) / a0, 2.0 * (k * k - vh) / a0, (vh - vb * k / q + k * k) / a0]
    )
    a1 = np.array([1.0, 2.0 * (k * k - 1.0) / a0, (1.0 - k / q + k * k) / a0])
    # Stage 2: high-pass
    f0, q = 38.13547087602444, 0.5003270373238773
    k = math.tan(math.pi * f0 / fs)
    a0 = 1.0 + k / q + k * k
    b2 = np.array([1.0, -2.0, 1.0])
    a2 = np.array([1.0, 2.0 * (k * k - 1.0) / a0, (1.0 - k / q + k * k) / a0])
    return b1, a1, b2, a2


def waveform_levels(peaks: np.ndarray) -> np.ndarray:
    """Linear peaks (0…1) → 0…255 over FLOOR_DB…0 dBFS (as core/waveform)."""
    db = 20.0 * np.log10(np.maximum(peaks, 1.0 / 32768.0))
    return np.clip((db - FLOOR_DB) * (255.0 / -FLOOR_DB), 0, 255).astype(np.uint8)


def _loudness(z: np.ndarray) -> np.ndarray:
    return -0.691 + 10.0 * np.log10(np.maximum(z, 1e-20))


class Levels:
    """Waveform and loudness of a recording fed in pieces."""

    def __init__(self):
        from scipy.signal import lfilter_zi

        self._b1, self._a1, self._b2, self._a2 = _biquad_k(SR)
        self._z1 = lfilter_zi(self._b1, self._a1) * 0.0
        self._z2 = lfilter_zi(self._b2, self._a2) * 0.0
        self._per = SR // RATE
        self._wave_rest = np.zeros(0, np.float32)
        self._k_rest = np.zeros(0, np.float64)
        self._levels: list[np.ndarray] = []
        self._energy: list[np.ndarray] = []
        self.samples = 0
        self.peak = 0.0
        self.clipped = 0

    def feed(self, x: np.ndarray) -> None:
        """More sound: float32 at 16 kHz (int16 is scaled)."""
        from scipy.signal import lfilter

        if x.dtype == np.int16:
            x = x.astype(np.float32) * np.float32(1 / 32768)
        elif x.dtype != np.float32:
            x = x.astype(np.float32)
        if not len(x):
            return
        self.samples += len(x)
        ax = np.abs(x)
        self.peak = max(self.peak, float(ax.max()))
        self.clipped += int(np.count_nonzero(ax >= _CLIP))
        # Waveform: peak of every 50 ms
        w = np.concatenate([self._wave_rest, ax]) if len(self._wave_rest) else ax
        whole = len(w) // self._per * self._per
        if whole:
            self._levels.append(waveform_levels(w[:whole].reshape(-1, self._per).max(axis=1)))
        self._wave_rest = w[whole:].copy()
        # Loudness: K-weighted, mean square of every 100 ms
        y, self._z1 = lfilter(self._b1, self._a1, x.astype(np.float64), zi=self._z1)
        y, self._z2 = lfilter(self._b2, self._a2, y, zi=self._z2)
        y = np.concatenate([self._k_rest, y]) if len(self._k_rest) else y
        whole = len(y) // _STEP * _STEP
        if whole:
            self._energy.append(np.mean(np.square(y[:whole]).reshape(-1, _STEP), axis=1))
        self._k_rest = y[whole:].copy()

    def finish(self) -> dict:
        """``levels`` (bytes, RATE a second) and the loudness figures — None
        where the recording is too short or silent for them."""
        if len(self._wave_rest):
            self._levels.append(waveform_levels(self._wave_rest.max(keepdims=True)))
            self._wave_rest = np.zeros(0, np.float32)
        levels = np.concatenate(self._levels) if self._levels else np.zeros(0, np.uint8)
        e = np.concatenate(self._energy) if self._energy else np.zeros(0)
        out = {
            "duration": self.samples / SR,
            "levels": levels.tobytes(),
            "peak_db": round(20.0 * math.log10(self.peak), 2) if self.peak > 0 else None,
            "clipped": round(self.clipped / self.samples, 6) if self.samples else None,
            "lufs": None,
            "lra": None,
            "silence": None,
        }
        if len(e) >= 4:
            # 400 ms blocks every 100 ms
            c = np.concatenate([[0.0], np.cumsum(e)])
            z = (c[4:] - c[:-4]) / 4.0
            lk = _loudness(z)
            out["silence"] = round(float(np.mean(lk < SILENT_LUFS)), 4)
            gated = z[lk > _ABS_GATE]
            if len(gated):
                rel = _loudness(np.array([gated.mean()]))[0] - 10.0
                gated = z[(lk > _ABS_GATE) & (lk > rel)]
                if len(gated):
                    out["lufs"] = round(float(_loudness(np.array([gated.mean()]))[0]), 2)
        if len(e) >= 30:
            # 3 s windows every 100 ms (EBU Tech 3342)
            c = np.concatenate([[0.0], np.cumsum(e)])
            st = _loudness((c[30:] - c[:-30]) / 30.0)
            st = st[st > _ABS_GATE]
            if len(st):
                rel = _loudness(np.array([np.mean(10.0 ** ((st + 0.691) / 10.0))]))[0] - 20.0
                st = st[st > rel]
                if len(st):
                    out["lra"] = round(float(np.percentile(st, 95) - np.percentile(st, 10)), 2)
        return out
