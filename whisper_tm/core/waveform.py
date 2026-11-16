"""The loudness of a recording, for the clip editor's timeline.

The waveform makes cutting easier — but it must not block on a 24-hour
recording.

A 24-hour recording can be an HEVC video of several GB on a network
share. Its sound alone, decoded, is on the order of a minute of ffmpeg —
and the whole file read over the network. So nothing waits for that:

* **A short stretch** — what the timeline shows zoomed in, up to ten
  minutes — is decoded on the spot from where it begins (``-ss`` before
  ``-i`` seeks in the container): around a second, even deep inside such
  a file.
* **A recording the audio analysis has seen** (whisper_tm/audio) has its
  overview kept there already: nothing is decoded.
* **Anything longer**, the whole file above all, comes from an overview:
  20 values a second, decoded once in the background — at low priority, one
  file at a time — and kept in the cache folder (1.7 MB for 24 hours).
  Until it is complete a request gets what is there so far and
  ``pending``; the timeline fills from the left.

A value is the peak of its stretch in decibels, -60…0 dBFS onto 0…255:
eight bits of linear peak would leave a quiet recording (-30 dB) seven
steps.
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import shutil
import subprocess
import tempfile
import threading
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np

from whisper_tm.config import config
from whisper_tm.core import media_formats

logger = logging.getLogger(__name__)

#: Overview values per second: 1,500 of them for a 75-second window — a
#: timeline's width. Narrower windows are decoded on the spot.
RATE = 20
#: A window up to this long is decoded on the spot (about a second, see
#: above); a longer one waits for the overview.
DECODE_MAX_SEC = 600
FLOOR_DB = -60.0
#: The overview's sample rate: 200 samples per value
OVERVIEW_HZ = 4000
#: The cache folder's limit — the least recently used overview goes first
CACHE_MAX_BYTES = 512 * 1024 * 1024
#: An overview still waiting for its turn that nobody has asked for in this
#: long is dropped: someone who opened five long recordings one after the
#: other is looking at the last one.
ABANDON_SEC = 120
#: A failed overview is tried again only after this long — not on every
#: request of a timeline that asks once a second
RETRY_SEC = 300

_lock = threading.Lock()
_overview_slot = threading.Lock()  # one overview decode at a time
_decode_slots = threading.BoundedSemaphore(2)  # on-the-spot decodes at once
_jobs: dict[str, _Overview] = {}
_loaded: OrderedDict[str, bytes] = OrderedDict()  # finished overviews in memory
_LOADED_MAX = 6
_info: dict[str, tuple[float, bool]] = {}  # key → (duration, has sound)


class WaveformError(Exception):
    """The recording has no sound that ffmpeg reads."""


class _Overview:
    """One recording's overview while it is decoded."""

    def __init__(self, key: str, source: Path, duration: float):
        self.key = key
        self.source = source
        self.total = math.ceil(duration * RATE) if duration > 0 else 0
        self.levels = bytearray()
        self.done = False
        self.error = ""
        self.failed_at = 0.0
        self.wanted = time.monotonic()


def _levels(peaks: np.ndarray) -> np.ndarray:
    """Linear peaks (0…32768) → 0…255 over FLOOR_DB…0 dBFS."""
    db = 20.0 * np.log10(np.maximum(peaks, 1) / 32768.0)
    return np.clip((db - FLOOR_DB) * (255.0 / -FLOOR_DB), 0, 255).astype(np.uint8)


def _pooled(values: np.ndarray, first: int, last: int, n: int, then=None) -> list[int | None]:
    """``n`` buckets over ``values[first:last]`` — each the loudest value in
    its part, None where there is nothing (beyond what is known or the end).
    ``then`` maps the buckets' values afterwards."""
    out: list[int | None] = [None] * n
    edges = first + np.floor(np.arange(n) * ((last - first) / n)).astype(np.int64)
    have = edges < min(last, len(values))
    if have.any():
        part = values[first : min(last, len(values))]
        got = np.maximum.reduceat(part, edges[have] - first)
        if then is not None:
            got = then(got)
        for i, v in zip(np.flatnonzero(have), got.tolist(), strict=True):
            out[i] = int(v)
    return out


def _identity(source: Path) -> tuple[str, float, bool]:
    """The cache key — the file as it is now — with its duration and
    whether it has sound (one ffprobe per version of the file)."""
    st = source.stat()
    key = hashlib.sha1(
        f"{source}|{st.st_size}|{st.st_mtime_ns}|{RATE}|{FLOOR_DB}".encode()
    ).hexdigest()[:24]
    with _lock:
        known = _info.get(key)
    if known is None:
        from whisper_tm.core.audio_utils import get_media_info

        info = get_media_info(source)
        known = (float(info.get("duration") or 0.0), bool(info.get("has_audio")))
        with _lock:
            _info[key] = known
    return key, known[0], known[1]


def _cache_dir() -> Path:
    return Path(config.cache_dir) / "waveforms"


def _remember(key: str, data: bytes) -> None:
    """Call with _lock held."""
    _loaded[key] = data
    _loaded.move_to_end(key)
    while len(_loaded) > _LOADED_MAX:
        _loaded.popitem(last=False)


def _stored(source: Path) -> bytes | None:
    """The waveform the audio analysis keeps for this version of the file
    (whisper_tm/audio: the same 20 values a second) — None when it has
    none."""
    try:
        from whisper_tm.audio.runner import enabled

        if not enabled():
            return None
        from whisper_tm.audio.store import get_store
        from whisper_tm.core import paths

        st = source.stat()
        return get_store().levels(paths.canon(source), st.st_size, st.st_mtime) or None
    except Exception as e:  # noqa: BLE001 — decoding is the way it always was
        logger.debug("Stored waveform of %s: %s", source, e)
        return None


def _finished(key: str, source: Path | None = None) -> bytes | None:
    """A complete overview — in memory, in the cache folder or (with
    ``source``) kept by the audio analysis."""
    with _lock:
        if key in _loaded:
            _loaded.move_to_end(key)
            return _loaded[key]
    path = _cache_dir() / f"{key}.u8"
    try:
        data = path.read_bytes()
    except OSError:
        data = _stored(source) if source is not None else None
        if data is not None:
            with _lock:
                _remember(key, data)
        return data
    try:
        os.utime(path)  # used: the last to be pruned
    except OSError:
        pass
    with _lock:
        _remember(key, data)
    return data


def _ffmpeg(*args: str, nice: bool = False) -> list[str]:
    cmd = ["ffmpeg", "-nostdin", "-v", "error", *args]
    if nice and shutil.which("nice"):
        # The queue beside it is what the machine is for
        cmd = ["nice", "-n", "10", *cmd]
    return cmd


def _decode(source: Path, start: float, end: float, n: int) -> list[int | None]:
    """The window decoded on the spot, ``n`` buckets."""
    dur = end - start
    # Some samples per bucket, however narrow the window (16 kHz at most)
    hz = int(min(16000, max(2000, math.ceil(8 * n / dur))))
    cmd = _ffmpeg(
        "-ss",
        f"{start:.3f}",
        *media_formats.source(source),
        "-t",
        f"{dur:.3f}",
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(hz),
        "-f",
        "s16le",
        "-",
    )
    try:
        with _decode_slots:
            r = subprocess.run(cmd, capture_output=True, timeout=120)
    except subprocess.TimeoutExpired:
        raise WaveformError("Reading the sound took too long") from None
    if r.returncode != 0:
        raise WaveformError(
            r.stderr.decode("utf-8", "replace").strip()[-300:] or "ffmpeg could not read the sound"
        )
    raw = r.stdout[: len(r.stdout) // 2 * 2]
    peaks = np.abs(np.frombuffer(raw, dtype="<i2").astype(np.int32))
    # The window's samples as they would be — where the file ends sooner,
    # the buckets beyond stay empty
    return _pooled(peaks, 0, max(1, round(dur * hz)), n, then=_levels)


def _overview(key: str, source: Path, duration: float) -> _Overview | None:
    """The overview being decoded — started when it isn't yet. None when it
    has just been finished (it is in memory then)."""
    with _lock:
        if key in _loaded:
            return None
        job = _jobs.get(key)
        if job is None or (job.error and time.monotonic() - job.failed_at > RETRY_SEC):
            job = _jobs[key] = _Overview(key, source, duration)
            threading.Thread(
                target=_run, args=(job,), name="waveform overview", daemon=True
            ).start()
        job.wanted = time.monotonic()
        return job


def _run(job: _Overview) -> None:
    with _overview_slot:
        if time.monotonic() - job.wanted > ABANDON_SEC:
            with _lock:
                _jobs.pop(job.key, None)
            return
        try:
            _decode_overview(job)
        except Exception as e:  # noqa: BLE001 — a waveform is never worth more
            logger.warning("Waveform of %s: %s", job.source, e)
            job.failed_at = time.monotonic()
            job.error = str(e)[:300] or type(e).__name__
            return
    with _lock:
        _remember(job.key, bytes(job.levels))
        _jobs.pop(job.key, None)


def _decode_overview(job: _Overview) -> None:
    per = OVERVIEW_HZ // RATE
    cmd = _ffmpeg(
        *media_formats.source(job.source),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(OVERVIEW_HZ),
        "-f",
        "s16le",
        "-",
        nice=True,
    )
    t0 = time.monotonic()
    # stderr into a file: a damaged recording can write more errors than a
    # pipe holds, and ffmpeg would stop, waiting for someone to read them
    with tempfile.TemporaryFile() as err:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=err)
        rest = b""
        try:
            while True:
                data = proc.stdout.read(per * 2 * 1000)  # 50 s of sound
                if not data:
                    break
                data = rest + data
                whole = len(data) // (per * 2) * (per * 2)
                rest = data[whole:]
                if whole:
                    peaks = np.abs(np.frombuffer(data[:whole], dtype="<i2").astype(np.int32))
                    level = _levels(peaks.reshape(-1, per).max(axis=1)).tobytes()
                    with _lock:
                        job.levels.extend(level)
        finally:
            proc.stdout.close()
            proc.wait()
        if len(rest) >= 2:  # the last, shorter stretch
            peaks = np.abs(np.frombuffer(rest[: len(rest) // 2 * 2], dtype="<i2").astype(np.int32))
            with _lock:
                job.levels.extend(_levels(peaks.max(keepdims=True)).tobytes())
        # A damaged end can make ffmpeg fail after all of it; a stop halfway
        # (a network share gone for a moment) must not be kept as the whole file
        if proc.returncode != 0 and not (job.total and len(job.levels) >= 0.99 * job.total):
            err.seek(0)
            raise WaveformError(
                err.read().decode("utf-8", "replace").strip()[-300:]
                or "ffmpeg could not read the sound"
            )
    folder = _cache_dir()
    folder.mkdir(parents=True, exist_ok=True)
    tmp = folder / f"{job.key}.{os.getpid()}.part"
    tmp.write_bytes(bytes(job.levels))
    os.replace(tmp, folder / f"{job.key}.u8")
    job.done = True
    logger.info(
        "Waveform of %s: %d values in %.0f s",
        job.source.name,
        len(job.levels),
        time.monotonic() - t0,
    )
    _prune(folder)


def _prune(folder: Path) -> None:
    """Keep the cache folder under its limit, the least recently used first —
    and without the halves a server stopped mid-write left behind."""
    for part in folder.glob("*.part"):
        try:
            if time.time() - part.stat().st_mtime > 3600:
                part.unlink(missing_ok=True)
        except OSError:
            pass
    try:
        files = sorted(
            ((p.stat().st_mtime, p.stat().st_size, p) for p in folder.glob("*.u8")),
            key=lambda f: f[0],
        )
    except OSError:
        return
    total = sum(f[1] for f in files)
    for _mtime, size, path in files:
        if total <= CACHE_MAX_BYTES:
            break
        path.unlink(missing_ok=True)
        total -= size


def peaks(source: Path, start: float, end: float, n: int) -> dict:
    """The loudness of ``start``…``end`` (seconds) in ``n`` buckets.

    Returns ``peaks`` (0…255, None where nothing is known yet or the file
    has ended), and ``pending`` with ``progress`` (0…1) while the overview it
    comes from is still being decoded — ask again for the rest.
    """
    n = max(16, min(4096, int(n)))
    start = max(0.0, float(start))
    end = max(float(end), start + 0.01)
    key, duration, has_sound = _identity(source)
    if not has_sound:
        raise WaveformError("This recording has no sound")
    reply = {"start": start, "end": end, "n": n, "pending": False, "progress": 1.0}
    first, last = int(start * RATE), max(int(start * RATE) + 1, math.ceil(end * RATE))
    # Finer than the overview's 20 values a second: only decoding shows it
    if (end - start) * RATE < n:
        return {**reply, "peaks": _decode(source, start, end, n)}
    done = _finished(key, source)
    if done is not None:
        return {**reply, "peaks": _pooled(np.frombuffer(done, np.uint8), first, last, n)}
    if end - start <= DECODE_MAX_SEC:
        return {**reply, "peaks": _decode(source, start, end, n)}
    job = _overview(key, source, duration)
    if job is None:
        done = _finished(key)
        if done is not None:
            return {**reply, "peaks": _pooled(np.frombuffer(done, np.uint8), first, last, n)}
        return {**reply, "peaks": [None] * n, "pending": True, "progress": 0.0}
    with _lock:
        known = np.frombuffer(bytes(job.levels), np.uint8)
        error, total = job.error, job.total
    if error:
        raise WaveformError(error)
    if job.done:
        return {**reply, "peaks": _pooled(known, first, last, n)}
    return {
        **reply,
        "peaks": _pooled(known, first, last, n),
        "pending": True,
        "progress": round(min(1.0, len(known) / total), 3) if total else 0.0,
    }
