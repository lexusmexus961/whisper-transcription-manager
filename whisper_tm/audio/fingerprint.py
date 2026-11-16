"""Fingerprints: pairs of spectral peaks ("landmarks", A. Wang, 2003).

The sound at 16 kHz becomes a spectrogram (1,024 samples a frame, a frame
every 256 samples = 16 ms, 62.5 Hz … 4 kHz). Its peaks — the largest value
within ±12 frames and ±10 bins — survive re-encoding, another volume and a
good deal of noise; the strongest few of every second are kept. Each peak is
paired with the next ones in time (up to ~1 s later, up to ±1 kHz apart), and
a pair becomes a 21-bit hash: its first frequency, the frequency step and
the time step. A hash says nothing about where it happened; the index keeps
that (``t``, the anchor's frame). A snippet matches a recording where many of
its hashes occur at the SAME time offset — chance spreads over all offsets.

**Measured on a real library (October 2026)**: 14 recordings that exist
twice in other formats (mkv/mp4/webm/opus/mp3; the transcripts 76–100 % the
same) — snippets of one copy searched in an index holding the other, beside
several hundred hours of other sound.

* Stored: 6 peaks a second, each paired with the next 2 → about 10 hashes a
  second of sound (216 kB an hour, 48 bits each), extracted at ~1,000× real
  time on one core.
* Asked: a snippet is fingerprinted denser (12 peaks a second, the next 6
  each) and four times, shifted by a quarter frame each: its frames lie
  anywhere between the stored ones. Shifting raised the score of a 3-second
  snippet from 12 to 30 and its recall from 88 % to 100 %.
* The score is the number of DIFFERENT snippet hashes found at the best
  offset (±1 frame). Raw counts let a steady sound — hum, a held tone —
  count one hash again and again: time-reversed snippets (which can't be
  in the index) reached 10 raw, 5 different. Real copies: 3 s → 15 (10th
  percentile), 5 s → 32, 10 s → 68; with mp3 at 32 kbit/s, noise at the
  same level as the sound, or -20 dB about the same; with a second
  recording mixed in as loud as the snippet, 10 s were found in 95 % of the
  cases, 3 s in 74 %.
* Not found: copies played faster or slower, or pitched (the hashes are
  frequencies and time steps), and a snippet re-recorded through a
  microphone in a room is a matter of luck.

``VERSION`` names everything above: another value means the stored
fingerprints are of another kind, and the index has to be built again.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

SR = 16000
N_FFT = 1024
HOP = 256
#: Frames a second (62.5)
FPS = SR / HOP
#: The bins kept: 62.5 Hz up to 4 kHz — where speech and most music have
#: their peaks, and below what low bit rates cut
FMIN, FMAX = 4, 256
#: A peak is the largest value within ±NB_T frames and ±NB_F bins
NB_T, NB_F = 12, 10
#: Nothing quieter is a peak (dB of a full-scale sine = 0): digital
#: silence and the noise floor of a quiet recording have no landmarks
FLOOR_DB = -75.0
#: A pair: the second peak 2…63 frames (32 ms … 1 s) after the first, at
#: most 63 bins (~1 kHz) above or below it
DT_MIN, DT_MAX = 2, 63
DF_MAX = 63
#: first bin (8 bits) · bin step (7) · frame step (6)
HASH_BITS = 21
#: Peaks are chosen per second: a second is 62.5 frames, so seconds are
#: counted as ``2 * t // 125`` and whole seconds come in steps of 125 frames
_TWO_SECONDS = 125
#: The stored kind of fingerprint (see above)
VERSION = 1


class Density(NamedTuple):
    peaks_per_sec: int
    fanout: int


#: What the index keeps of every recording
STORE = Density(6, 2)
#: What a snippet is asked with — denser: its pairs include most stored ones
QUERY = Density(12, 6)
#: A snippet is fingerprinted this many times, shifted by HOP / SHIFTS
SHIFTS = 4

_WIN = np.hanning(N_FFT).astype(np.float32)
_NORM = np.float32(2.0 / float(_WIN.sum()))
_LOW = np.float32(-1e9)


def spectrogram(x: np.ndarray) -> np.ndarray:
    """Rows of dB values (``FMAX - FMIN`` bins), one per HOP samples of
    ``x`` (float32, 16 kHz); a frame that would run past the end is left
    out."""
    if len(x) < N_FFT:
        return np.zeros((0, FMAX - FMIN), np.float32)
    frames = np.lib.stride_tricks.sliding_window_view(x, N_FFT)[::HOP]
    out = np.empty((len(frames), FMAX - FMIN), np.float32)
    for i in range(0, len(frames), 4096):
        spec = np.fft.rfft(frames[i : i + 4096] * _WIN, axis=1)[:, FMIN:FMAX]
        mag = np.abs(spec).astype(np.float32) * _NORM
        out[i : i + 4096] = 20.0 * np.log10(mag + np.float32(1e-9))
    return out


def _peaks(S: np.ndarray, row0: int, lo: int, hi: int) -> tuple[np.ndarray, ...]:
    """The peaks of ``S`` (its first row is frame ``row0``) in frames
    ``lo`` … ``hi``: frames, bins, values."""
    from scipy.ndimage import maximum_filter1d

    m = maximum_filter1d(S, 2 * NB_T + 1, axis=0, mode="constant", cval=_LOW)
    m = maximum_filter1d(m, 2 * NB_F + 1, axis=1, mode="constant", cval=_LOW)
    a, b = lo - row0, hi - row0
    part = S[a:b]
    t, f = np.nonzero((part == m[a:b]) & (part > FLOOR_DB))
    return t + lo, f, part[t, f]


def _strongest(t: np.ndarray, f: np.ndarray, mag: np.ndarray, per_sec: int):
    """The ``per_sec`` strongest peaks of every second, in time order."""
    if not len(t):
        return t, f
    second = (2 * t) // _TWO_SECONDS
    order = np.lexsort((-mag, second))
    t, f, second = t[order], f[order], second[order]
    first = np.searchsorted(second, second, side="left")
    keep = (np.arange(len(second)) - first) < per_sec
    t, f = t[keep], f[keep]
    order = np.lexsort((f, t))
    return t[order], f[order]


def _pairs(t: np.ndarray, f: np.ndarray, fanout: int, lo: int, hi: int):
    """Hashes and anchor frames of the pairs whose first peak lies in
    ``lo`` … ``hi``: each paired with the next ``fanout`` peaks in reach."""
    n = len(t)
    if n < 2:
        return np.zeros(0, np.uint32), np.zeros(0, np.uint32)
    anchor = (t >= lo) & (t < hi)
    paired = np.zeros(n, np.int64)
    hashes, times = [], []
    for k in range(1, n):
        a = np.arange(n - k)
        dt = t[a + k] - t[a]
        reach = dt <= DT_MAX
        if not reach.any():
            break
        df = f[a + k] - f[a]
        ok = reach & (dt >= DT_MIN) & (np.abs(df) <= DF_MAX) & anchor[a] & (paired[a] < fanout)
        if not ok.any():
            continue
        a, dt, df = a[ok], dt[ok], df[ok]
        paired[a] += 1
        hashes.append(
            (f[a].astype(np.uint32) << 13)
            | ((df + DF_MAX).astype(np.uint32) << 6)
            | dt.astype(np.uint32)
        )
        times.append(t[a].astype(np.uint32))
    if not hashes:
        return np.zeros(0, np.uint32), np.zeros(0, np.uint32)
    return np.concatenate(hashes), np.concatenate(times)


class Fingerprinter:
    """The landmarks of a recording fed in pieces — exactly those of the
    whole recording at once, without holding its spectrogram (two minutes of
    it at most): a 24-hour recording is 5.4 million frames."""

    #: Frames worked on at a time: a minute (a multiple of whole seconds)
    CHUNK = _TWO_SECONDS * 30
    #: What a chunk needs beyond its end: the second after it (its peaks are
    #: partners of the last anchors), and the peak search's reach
    _AFTER = 63 + NB_T

    def __init__(self, density: Density = STORE):
        self.density = density
        self._samples = np.zeros(0, np.float32)
        self._spec = np.zeros((0, FMAX - FMIN), np.float32)
        self._row0 = 0  # frame of self._spec[0]
        self._frames = 0  # frames computed
        self._done = 0  # anchors before this frame are out
        self._hashes: list[np.ndarray] = []
        self._times: list[np.ndarray] = []

    @property
    def frames(self) -> int:
        return self._frames

    def feed(self, x: np.ndarray) -> None:
        """More sound: float32 at 16 kHz (int16 is scaled)."""
        if x.dtype != np.float32:
            x = (
                x.astype(np.float32) * np.float32(1 / 32768)
                if x.dtype == np.int16
                else x.astype(np.float32)
            )
        self._samples = np.concatenate([self._samples, x]) if len(self._samples) else x
        if len(self._samples) >= N_FFT:
            n = 1 + (len(self._samples) - N_FFT) // HOP
            rows = spectrogram(self._samples[: (n - 1) * HOP + N_FFT])
            self._spec = np.concatenate([self._spec, rows])
            self._samples = self._samples[n * HOP :].copy()
            self._frames += n
        while self._frames >= self._done + self.CHUNK + self._AFTER:
            self._chunk(self._done + self.CHUNK)

    def finish(self) -> tuple[np.ndarray, np.ndarray]:
        """Hashes (uint32) and their anchor frames (uint32), in time order."""
        while self._done < self._frames:
            self._chunk(min(self._frames, self._done + self.CHUNK))
        if not self._hashes:
            return np.zeros(0, np.uint32), np.zeros(0, np.uint32)
        h, t = np.concatenate(self._hashes), np.concatenate(self._times)
        order = np.argsort(t, kind="stable")
        return h[order], t[order]

    def _chunk(self, hi: int) -> None:
        lo = self._done
        top = min(self._frames, hi + self._AFTER)
        first = max(self._row0, lo - NB_T)
        S = self._spec[first - self._row0 : top - self._row0]
        t, f, mag = _peaks(S, first, lo, min(top, hi + 63))
        t, f = _strongest(t, f, mag, self.density.peaks_per_sec)
        h, at = _pairs(t, f, self.density.fanout, lo, hi)
        if len(h):
            self._hashes.append(h)
            self._times.append(at)
        self._done = hi
        cut = max(0, self._done - NB_T - self._row0)
        if cut:
            self._spec = self._spec[cut:].copy()
            self._row0 += cut


def landmarks(x: np.ndarray, density: Density = STORE) -> tuple[np.ndarray, np.ndarray]:
    """Hashes and anchor frames of a whole piece of sound at once."""
    fp = Fingerprinter(density)
    fp.feed(x)
    return fp.finish()


def query_landmarks(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A snippet's landmarks as it is searched with: denser, and from
    ``SHIFTS`` starts a fraction of a frame apart — distinct (hash, frame)
    pairs."""
    x = np.asarray(x)
    if x.dtype == np.int16:
        x = x.astype(np.float32) * np.float32(1 / 32768)
    hs, ts = [], []
    for s in range(SHIFTS):
        h, t = landmarks(x[int(s * HOP / SHIFTS) :], QUERY)
        hs.append(h)
        ts.append(t)
    h, t = np.concatenate(hs), np.concatenate(ts)
    if not len(h):
        return h, t
    both = np.unique((h.astype(np.uint64) << np.uint64(32)) | t.astype(np.uint64))
    return (
        (both >> np.uint64(32)).astype(np.uint32),
        (both & np.uint64(0xFFFFFFFF)).astype(np.uint32),
    )
