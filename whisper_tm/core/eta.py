"""How long the rest takes — for the library's open files and for the queue.

One model for both: a file costs a fixed amount plus a share of its length,

    seconds = a + b × duration

``a`` per file (provisioning, ffmpeg, loading the audio, model calls), ``b``
per second of audio. A plain "audio ÷ speed" misses the fixed part
completely — for a voice note of a few seconds the fixed part is most of
the cost, close to real time — and a queue can hold many thousands of such
short notes.

``a`` and ``b`` come from two groups of recent jobs, short (< 60 s) and
long, and reproduce the measured total time of both exactly. A single
average speed is far off for short files. Recent jobs, because the pace
follows the pipeline: an all-time average lags behind changes to it.

Files without a known duration (not indexed yet — e.g. freshly added
videos) get one from their size, at the bytes per second of the
indexed files with the same extension. Only what has neither stays
uncounted.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, replace

SHORT_SEC = 60.0  # border between the two groups of a fit
MIN_GROUP = 20  # samples per group needed for the two-part fit
WINDOW_DAYS = 30  # transcriptions: recent enough to follow the pipeline
MIN_SAMPLES = 200  # …unless the window is thinner — then all of them
RATE_MIN_FILES = 20  # files per extension for a bytes-per-second figure


@dataclass(frozen=True)
class Model:
    a: float  # seconds per file
    b: float  # seconds per second of audio
    samples: int
    rough: bool = False  # one group missing: a plain ratio, no fixed part
    window_days: int | None = None  # transcriptions: the days fitted on (None = all)

    def seconds(self, files: float, audio_sec: float) -> float:
        return self.a * files + self.b * audio_sec

    def as_dict(self) -> dict:
        return {
            "a": round(self.a, 3),
            "b": round(self.b, 5),
            "samples": self.samples,
            "rough": self.rough,
            "window_days": self.window_days,
        }


def fit(short: tuple, long: tuple) -> Model | None:
    """Fit ``a`` and ``b`` from (files, audio seconds, seconds) per group.

    Solves ``files·a + audio·b = seconds`` for both groups at once. With too
    few samples in one group — or a solution without meaning (a < 0) — it
    falls back to one ratio for everything, as before this module, and
    says so (``rough``).
    """
    ns, ds, ts = short
    nl, dl, tl = long
    if ns + nl == 0 or ds + dl <= 0:
        return None
    if ns >= MIN_GROUP and nl >= MIN_GROUP:
        det = ns * dl - nl * ds
        if det > 0:
            b = (tl * ns - ts * nl) / det
            a = (ts - ds * b) / ns
            if a >= 0 and b > 0:
                return Model(a, b, ns + nl)
    return Model(0.0, (ts + tl) / (ds + dl), ns + nl, rough=True)


def groups_of(samples) -> tuple[tuple, tuple]:
    """(duration, seconds) pairs → the two groups ``fit`` takes."""
    short, long = [0, 0.0, 0.0], [0, 0.0, 0.0]
    for dur, secs in samples:
        g = short if dur < SHORT_SEC else long
        g[0] += 1
        g[1] += dur
        g[2] += secs
    return tuple(short), tuple(long)


def audio_from_size(rows, rates: dict) -> tuple[int, float, int]:
    """Durations for files that have none: (ext, files, bytes) rows →
    (files estimated, their audio seconds, files left unknown)."""
    estimated, audio, unknown = 0, 0.0, 0
    for ext, files, size in rows:
        rate = rates.get((ext or "").lower())
        if rate and size:
            estimated += files
            audio += size / rate
        else:
            unknown += files
    return estimated, audio, unknown


# ── Cached inputs: fits and byte rates change slowly ─────────────────────

_cache: dict = {}
_cache_lock = threading.Lock()


def _cached(key, ttl: float, compute):
    now = time.time()
    with _cache_lock:
        hit = _cache.get(key)
        if hit and now - hit[0] < ttl:
            return hit[1]
    value = compute()
    with _cache_lock:
        _cache[key] = (now, value)
    return value


def clear_cache() -> None:
    with _cache_lock:
        _cache.clear()


def transcription_model(db) -> Model | None:
    """Model of a transcription, from the transcripts of the last 30 days."""

    def compute():
        since = time.time() - WINDOW_DAYS * 86400
        short, long = db.eta_transcription_groups(since)
        window = WINDOW_DAYS
        if short[0] + long[0] < MIN_SAMPLES:
            short, long = db.eta_transcription_groups(0)
            window = None
        model = fit(short, long)
        return replace(model, window_days=window) if model else None

    return _cached(("transcription", id(db)), 300, compute)


def byte_rates(db) -> dict:
    return _cached(("rates", id(db)), 3600, lambda: db.eta_byte_rates(RATE_MIN_FILES))


# ── The two estimates ────────────────────────────────────────────────────


def library(db, abs_dir: str, stats: dict) -> dict:
    """Time left for the open files under ``abs_dir``.

    ``stats`` is the result of ``get_media_stats_under`` (pending files with
    and without a duration). Returns the ``eta`` block of /api/files/stats.
    """
    model = transcription_model(db)
    known = int(stats.get("pending_with_dur") or 0)
    audio = float(stats.get("pending_media_dur_sec") or 0.0)
    est, est_audio, unknown = audio_from_size(
        db.eta_pending_without_duration(abs_dir), byte_rates(db)
    )
    out = {
        "files": known + est + unknown,
        "known": known,
        "audio_sec": audio,
        "by_size": est,
        "size_audio_sec": est_audio,
        "unknown": unknown,
        "sec": 0.0,
        "model": model.as_dict() if model else None,
    }
    if model:
        out["sec"] = model.seconds(known + est, audio + est_audio)
    return out


def cached_queue(db, inputs_fn) -> dict:
    """``queue`` at most every 20 s: the pending join runs over the whole
    queue (about 0.1 s for 100,000 entries), and the sidebar polls."""
    return _cached(("queue", id(db)), 20, lambda: queue(db, inputs_fn()))


def _queue_model(db, inputs: dict, kind: str) -> tuple[Model | None, str]:
    """The model for one kind of queued job, and where it comes from.

    From the finished entries of the current run; a transcription without
    enough of them borrows the library's model. A voices-only run has no
    stand-in — its estimate stays rough until 20 of each length group are
    done, and there is none before the first one finished.
    """
    model = fit(*groups_of(inputs.get("samples", {}).get(kind, [])))
    if (model is None or model.rough) and kind != "voices":
        library_model = transcription_model(db)
        if library_model:
            return library_model, "library"
    return model, "queue"


def queue(db, inputs: dict, now: float | None = None) -> dict:
    """Time left for the queue, from ``core.queue.eta_inputs()``.

    ``sec`` is None while no kind can be estimated yet (a fresh run of a
    kind nothing is known about); a kind without a model is left out of
    the sum, says so (``sec: None``) and makes the sum ``partial``.
    """
    now = now or time.time()
    rates = byte_rates(db)
    kinds, total, any_model = [], 0.0, False
    for kind, p in sorted(inputs.get("pending", {}).items()):
        model, source = _queue_model(db, inputs, kind)
        est, est_audio, unknown = audio_from_size(p.get("no_duration", []), rates)
        files, audio = p["known"] + est, p["audio_sec"] + est_audio
        sec = None
        if model:
            sec = model.seconds(files, audio)
            # Nothing to go by at all: at the average of the rest
            if unknown:
                sec += unknown * (sec / files if files else model.a)
            total += sec
            any_model = True
        kinds.append(
            {
                "kind": kind,
                "files": files + unknown,
                "known": p["known"],
                "audio_sec": p["audio_sec"],
                "by_size": est,
                "size_audio_sec": est_audio,
                "unknown": unknown,
                "sec": sec,
                "source": source,
                "model": model.as_dict() if model else None,
            }
        )
    running = inputs.get("running")
    if running is None:  # inputs from before lanes
        running = [inputs["current"]] if inputs.get("current") else []
    for cur in running:
        model, _ = _queue_model(db, inputs, cur["kind"])
        if model:
            # The samples are shares of the wall clock (core/queue.py
            # eta_inputs); a running entry's share so far is its wall time
            # divided by the entries running beside it.
            expected = model.seconds(1, cur.get("duration") or 0.0)
            ran = (now - (cur.get("started_at") or now)) / max(1, len(running))
            total += max(0.0, expected - ran)
            any_model = True
    return {
        "sec": total if any_model else None,
        "kinds": kinds,
        "partial": any(k["sec"] is None for k in kinds),
        "parallelism": inputs.get("parallelism", 1.0),
        "last_hour": inputs.get("last_hour", {}),
    }
