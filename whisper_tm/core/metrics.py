"""System load history: CPU, RAM, GPU and VRAM over the last days.

One background thread measures every ``SAMPLE_SEC`` seconds and keeps the
latest reading for the live meters in the top bar (/api/system/load). Every
``WRITE_SEC`` seconds it writes the mean — and the peak of CPU and GPU —
of the readings since the last write into its own small SQLite file,
``metrics.db`` next to the main database. Own file on purpose: the main DB
is several GB and gets backed up; a history of meter readings needs
neither. Rows older than ``RETENTION_DAYS`` are dropped once an hour.

:func:`history` serves the performance view: it averages the rows into at
most ``max_points`` buckets, so a 30-day range costs the same as an hour.

Measuring once centrally also fixes the CPU figure: ``psutil.cpu_percent``
measures since its previous call, so two callers (the meters and a sampler)
would each see half-intervals.

**VRAM by process** (core/vram_procs.py): every reading
also asks the driver whose memory it is — on the GPU WhisperX runs on, and
only there. Each interval stores how much of the VRAM was this app's
(``vram_own_mb``) and everything else's (``vram_foreign_mb``) — NULL in
rows from before, so the view shows no split it doesn't have — and, in
``vram_procs``, per source the mean, the lowest and the highest of the
interval's readings: *"how much does the diarization swing"* is the
distance between the two.

**Live**: the view's "Live 90s" range shows the raw
readings of the last minute and a half, scrolling with the clock. Every
reading also goes into an in-memory ring buffer of the last
``LIVE_KEEP_SEC`` seconds (:func:`live`, /api/system/live). While someone
watches — each request of the live view counts for ``LIVE_WATCH_SEC`` —
the sampler adds a reading every ``LIVE_SEC`` between two regular ones,
as long as one reading stays cheaper than ``LIVE_MAX_COST_SEC`` (through
NVML about 1.4 ms, through nvidia-smi about 33 ms). Those extra readings
go into the ring buffer only: the top-bar meters and the 30-s rows keep
getting exactly the readings they got before, every ``SAMPLE_SEC``, with
psutil's CPU figure over the whole ``SAMPLE_SEC`` — an extra call of
``psutil.cpu_percent`` would shorten that window. The buffer's CPU figure
is its own: busy share since the buffer's previous reading.
"""

from __future__ import annotations

import contextlib
import logging
import math
import sqlite3
import threading
import time
from collections import deque
from pathlib import Path

logger = logging.getLogger(__name__)

SAMPLE_SEC = 2.0
WRITE_SEC = 30.0
RETENTION_DAYS = 30

#: The live view: what it shows, how fine while someone watches, how long a
#: request of it counts as "someone watches", how long the buffer reaches
#: back, and the most one reading may cost for the finer resolution.
LIVE_SPAN_SEC = 90.0
LIVE_SEC = 1.0
LIVE_WATCH_SEC = 10.0
LIVE_KEEP_SEC = 300.0
LIVE_MAX_COST_SEC = 0.15

_lock = threading.Lock()
_latest: dict | None = None
_pending: list[dict] = []
_thread: threading.Thread | None = None
_stop = threading.Event()
_db_path: Path | None = None
#: The live view's readings, oldest first — at most one per LIVE_SEC.
_ring: deque[dict] = deque(maxlen=int(LIVE_KEEP_SEC / LIVE_SEC) + 10)
#: Until when (time.monotonic) a live viewer counts as watching.
_watch_until = 0.0
#: What the latest reading cost, in seconds (None: none taken yet).
_cost: float | None = None

_SCHEMA = """
CREATE TABLE IF NOT EXISTS samples (
    ts            REAL PRIMARY KEY,   -- end of the interval, unix seconds
    cpu           REAL,               -- percent, mean over the interval
    cpu_max       REAL,
    gpu           REAL,               -- percent busy (NULL without GPU info)
    gpu_max       REAL,
    ram_mb        REAL,               -- system RAM in use (total - available)
    ram_total_mb  REAL,
    vram_mb       REAL,               -- all processes on the working GPU
    vram_total_mb REAL,
    vram_own_mb   REAL,               -- of that: this app's processes
    vram_foreign_mb REAL              -- of that: every other process
)
"""

_SCHEMA_PROCS = """
CREATE TABLE IF NOT EXISTS vram_procs (
    ts      REAL NOT NULL,            -- the interval's end, as in samples
    key     TEXT NOT NULL,            -- "own:server", "foreign:Discord", ...
    label   TEXT,
    own     INTEGER NOT NULL,         -- 1: this app or a process it started
    mb      REAL,                     -- mean over the interval, absent = 0
    mb_max  REAL,
    mb_min  REAL,                     -- lowest reading, absent = 0
    PRIMARY KEY (ts, key)
)
"""


#: How many sources the history names one by one; the rest of each kind
#: is summed as "Other programs" / "Other (own)".
HISTORY_SOURCES = 6


# ── Measuring ────────────────────────────────────────────────────────────


def _amd_load() -> dict:
    """GPU load and VRAM of the first AMD card from sysfs (amdgpu driver).

    Only used without an NVIDIA card — the index would not match
    ``hardware.gpu_index`` otherwise. Display only; untested on real AMD
    hardware (see the setup guide's AMD page).
    """
    for card in sorted(Path("/sys/class/drm").glob("card[0-9]*")):
        dev = card / "device"
        try:
            if "-" in card.name or (dev / "vendor").read_text().strip() != "0x1002":
                continue
            mb = 1024 * 1024
            return {
                "util": int((dev / "gpu_busy_percent").read_text()),
                "vram_used_mb": int((dev / "mem_info_vram_used").read_text()) // mb,
                "vram_total_mb": int((dev / "mem_info_vram_total").read_text()) // mb,
                "name": "AMD GPU",
            }
        except (OSError, ValueError):
            continue
    return {}


def measure(cpu: bool = True) -> dict:
    """One reading: CPU, RAM, GPU, VRAM. Every part is optional.

    ``cpu=False`` leaves psutil's CPU figure out (``None``): the live
    view's extra readings. psutil measures since its previous call — one
    more call would shorten the window of the next regular reading."""
    from whisper_tm.config import config

    out: dict = {
        "ts": time.time(),
        "cpu": None,
        "ram_used_mb": None,
        "ram_total_mb": None,
        "gpu": None,
    }
    try:
        import psutil

        if cpu:
            out["cpu"] = round(psutil.cpu_percent(interval=None))
        vm = psutil.virtual_memory()
        out["ram_used_mb"] = (vm.total - vm.available) // (1024 * 1024)
        out["ram_total_mb"] = vm.total // (1024 * 1024)
    except Exception as e:  # noqa — the meters are a convenience
        logger.debug("CPU/RAM load: %s", e)
    try:
        from whisper_tm.core import debug as _debug

        if _debug.gpu_available():
            idx = int(config.get("hardware", "gpu_index", default=0) or 0)
            procs = None
            try:
                # The GPU WhisperX runs on: where this app holds memory, and
                # before it holds any, the configured one
                from whisper_tm.core import vram_procs

                idx, procs = vram_procs.snapshot(idx)
            except Exception as e:  # noqa — the split is extra, the total stays
                logger.debug("VRAM by process: %s", e)
            static_info = _debug.gpu_static()
            info = static_info.get(idx) or next(iter(static_info.values()), {})
            readings = _debug.gpu_used_and_util()
            used, util = readings.get(idx) or next(iter(readings.values()), (0, None))
            out["gpu"] = {
                "index": idx,
                "name": info.get("name", ""),
                "util": util,
                "vram_used_mb": used,
                "vram_total_mb": info.get("total_mb", 0),
            }
            if procs is not None:
                out["gpu"]["procs"] = procs
        else:
            out["gpu"] = _amd_load() or None
    except Exception as e:  # noqa
        logger.debug("GPU load: %s", e)
    return out


def latest() -> dict:
    """The most recent reading — measured on the spot if the sampler isn't running."""
    with _lock:
        if _latest is not None and time.time() - _latest["ts"] < SAMPLE_SEC * 3:
            return dict(_latest)
    return measure()


# ── Live ─────────────────────────────────────────────────────────────────


class _CpuSince:
    """Busy share of all CPUs since this counter's previous reading.

    ``psutil.cpu_percent``'s formula on a reference of its own, so the live
    buffer gets the CPU over its own (shorter) windows while psutil's
    reference keeps spanning the regular readings' ``SAMPLE_SEC``."""

    def __init__(self) -> None:
        self._last: tuple[float, float] | None = None

    def read(self) -> int | None:
        try:
            import psutil

            t = psutil.cpu_times()
        except Exception:  # noqa — the meters are a convenience
            return None
        # As psutil counts: guest time is already in user/nice, and iowait
        # is idle time Linux keeps apart
        total = sum(t) - getattr(t, "guest", 0.0) - getattr(t, "guest_nice", 0.0)
        busy = total - t.idle - getattr(t, "iowait", 0.0)
        last, self._last = self._last, (busy, total)
        if last is None or total <= last[1]:
            return None
        return round(max(0.0, min(100.0, (busy - last[0]) / (total - last[1]) * 100)))


_cpu_live = _CpuSince()


def watch() -> None:
    """Someone has the live view open: finer readings for ``LIVE_WATCH_SEC``."""
    global _watch_until
    _watch_until = time.monotonic() + LIVE_WATCH_SEC


def watched() -> bool:
    return time.monotonic() < _watch_until


def live_step() -> float:
    """Seconds between two readings in the buffer right now: ``LIVE_SEC``
    while someone watches and a reading is cheap, else ``SAMPLE_SEC``."""
    cheap = _cost is None or _cost <= LIVE_MAX_COST_SEC
    return LIVE_SEC if watched() and cheap else SAMPLE_SEC


def _live_entry(r: dict, cpu: int | None) -> dict:
    """A reading as the live view draws it: the history's field names, the
    VRAM per source as :func:`aggregate` counts it (processes of one source
    add up, an unknown amount is left out) and the sources' names apart."""
    g = r.get("gpu") or {}
    e = {
        "t": round(r["ts"], 3),
        "cpu": cpu,
        "gpu": g.get("util"),
        "ram_mb": r.get("ram_used_mb"),
        "ram_total_mb": r.get("ram_total_mb"),
        "vram_mb": g.get("vram_used_mb"),
        "vram_total_mb": g.get("vram_total_mb"),
        "vram_own_mb": None,
        "vram_foreign_mb": None,
        "procs": None,
        "labels": None,
    }
    if g.get("procs") is not None:
        by: dict[str, float] = {}
        labels: dict[str, tuple[str, bool]] = {}
        for p in g["procs"]:
            if p.get("mb") is None:
                continue
            by[p["key"]] = by.get(p["key"], 0) + p["mb"]
            labels[p["key"]] = (p.get("label") or p["key"], bool(p.get("own")))
        e["procs"], e["labels"] = by, labels
        e["vram_own_mb"] = sum(mb for k, mb in by.items() if labels[k][1])
        e["vram_foreign_mb"] = sum(mb for k, mb in by.items() if not labels[k][1])
    return e


def _remember(r: dict) -> None:
    """Into the live buffer — with the CPU since the buffer's previous reading.

    Never raises: whatever goes wrong here must not cost the regular
    reading its place in the meters and the 30-s rows."""
    try:
        entry = _live_entry(r, _cpu_live.read())
        with _lock:
            _ring.append(entry)
    except Exception as e:  # noqa — the live view is a convenience
        logger.debug("Live buffer: %s", e)


def live(since: float | None = None, now: float | None = None) -> dict:
    """The buffered readings newer than ``since`` (unix seconds).

    Without ``since`` the last ``LIVE_SPAN_SEC`` and a little more — the
    view trails the clock by a moment and scrolls on between two answers.
    ``sources``: name and kind of every source in ``readings[].procs``;
    ``step``: the buffer's current resolution in seconds."""
    now = time.time() if now is None else now
    if since is None or not math.isfinite(since):
        since = now - LIVE_SPAN_SEC - 10
    since = max(since, now - LIVE_KEEP_SEC)
    with _lock:
        fresh = [e for e in _ring if e["t"] > since]
    readings, sources = [], {}
    for e in fresh:
        readings.append({k: v for k, v in e.items() if k != "labels"})
        for key, (label, own) in (e["labels"] or {}).items():
            sources[key] = {"label": label, "own": own}
    return {
        "now": now,
        "step": live_step(),
        "span": LIVE_SPAN_SEC,
        "readings": readings,
        "sources": sources,
    }


# ── Storage ──────────────────────────────────────────────────────────────


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path), timeout=5)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(_SCHEMA)
    conn.execute(_SCHEMA_PROCS)
    return conn


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def _peak(vals):
    vals = [v for v in vals if v is not None]
    return max(vals) if vals else None


def aggregate(readings: list[dict]) -> dict | None:
    """Collapse the readings of one write interval into one row.

    ``procs``: per source the mean over the readings that asked the
    driver (a source absent from one counts 0 there) and its peak;
    several processes of one source (three browser renderers) add up."""
    if not readings:
        return None
    gpus = [r["gpu"] for r in readings if r.get("gpu")]
    row = {
        "ts": readings[-1]["ts"],
        "cpu": _mean(r.get("cpu") for r in readings),
        "cpu_max": _peak(r.get("cpu") for r in readings),
        "gpu": _mean(g.get("util") for g in gpus),
        "gpu_max": _peak(g.get("util") for g in gpus),
        "ram_mb": _mean(r.get("ram_used_mb") for r in readings),
        "ram_total_mb": _peak(r.get("ram_total_mb") for r in readings),
        "vram_mb": _mean(g.get("vram_used_mb") for g in gpus),
        "vram_total_mb": _peak(g.get("vram_total_mb") for g in gpus),
        "vram_own_mb": None,
        "vram_foreign_mb": None,
        "procs": [],
    }
    lists = [g["procs"] for g in gpus if g.get("procs") is not None]
    if lists:
        per_reading: list[dict[str, float]] = []
        meta: dict[str, tuple[str, bool]] = {}
        for procs in lists:
            here: dict[str, float] = {}
            for p in procs:
                if p.get("mb") is None:
                    continue
                here[p["key"]] = here.get(p["key"], 0.0) + p["mb"]
                meta[p["key"]] = (p.get("label") or p["key"], bool(p.get("own")))
            per_reading.append(here)
        n = len(lists)
        row["vram_own_mb"] = (
            sum(mb for here in per_reading for k, mb in here.items() if meta[k][1]) / n
        )
        row["vram_foreign_mb"] = (
            sum(mb for here in per_reading for k, mb in here.items() if not meta[k][1]) / n
        )
        # A source absent from a reading held 0 there: its lowest is 0
        row["procs"] = [
            {
                "key": k,
                "label": meta[k][0],
                "own": int(meta[k][1]),
                "mb": sum(here.get(k, 0.0) for here in per_reading) / n,
                "mb_max": max(here.get(k, 0.0) for here in per_reading),
                "mb_min": min(here.get(k, 0.0) for here in per_reading),
            }
            for k in meta
        ]
    return row


def write(path: Path, row: dict) -> None:
    values = {"vram_own_mb": None, "vram_foreign_mb": None, **row}
    with contextlib.closing(_connect(path)) as conn, conn:
        conn.execute(
            "INSERT OR REPLACE INTO samples (ts, cpu, cpu_max, gpu, gpu_max, "
            "ram_mb, ram_total_mb, vram_mb, vram_total_mb, vram_own_mb, "
            "vram_foreign_mb) VALUES (:ts, :cpu, :cpu_max, :gpu, :gpu_max, "
            ":ram_mb, :ram_total_mb, :vram_mb, :vram_total_mb, :vram_own_mb, "
            ":vram_foreign_mb)",
            values,
        )
        if row.get("procs"):
            conn.executemany(
                "INSERT OR REPLACE INTO vram_procs (ts, key, label, own, mb, "
                "mb_max, mb_min) VALUES (:ts, :key, :label, :own, :mb, "
                ":mb_max, :mb_min)",
                [{"mb_min": None, **p, "ts": row["ts"]} for p in row["procs"]],
            )


def prune(path: Path, now: float | None = None) -> int:
    cutoff = (now or time.time()) - RETENTION_DAYS * 86400
    with contextlib.closing(_connect(path)) as conn, conn:
        conn.execute("DELETE FROM vram_procs WHERE ts < ?", (cutoff,))
        return conn.execute("DELETE FROM samples WHERE ts < ?", (cutoff,)).rowcount


#: The widest history asked for at once — more than any retention keeps
HISTORY_SPAN_MAX_SEC = 10 * 365 * 86400


def history(
    path: Path | None = None,
    start: float | None = None,
    end: float | None = None,
    max_points: int = 600,
) -> dict:
    """Rows between ``start`` and ``end``, averaged into ≤ ``max_points`` buckets.

    Each point carries the bucket's mean and the peak of CPU and GPU; RAM
    and VRAM totals are the bucket's maximum (a total doesn't average).
    ``bucket`` tells the client how wide a point is — a gap wider than a
    few buckets means the server was off, and the line should break there.
    """
    path = path or _db_path
    # Finite and at most :data:`HISTORY_SPAN_MAX_SEC` wide: ±1e308 put "inf"
    # into the SQL, nan and inf came back as bare NaN/Infinity
    now = time.time()
    end = end if end and math.isfinite(end) else now
    end = min(max(end, 0.0), now + 86400)
    if start is None or not math.isfinite(start) or start >= end:
        start = end - 86400
    start = max(start, end - HISTORY_SPAN_MAX_SEC)
    bucket = max(WRITE_SEC, (end - start) / max(1, max_points))
    out = {
        "start": start,
        "end": end,
        "bucket": bucket,
        "points": [],
        "first": None,
        "retention_days": RETENTION_DAYS,
    }
    if not path or not Path(path).exists():
        return out
    with contextlib.closing(_connect(Path(path))) as conn:
        # The bucket bound, not formatted into the SQL
        args = {"b": bucket, "start": start, "end": end}
        rows = conn.execute(
            """
            SELECT CAST(ts / :b AS INTEGER) * :b + :b / 2 AS t,
                   AVG(cpu), MAX(cpu_max), AVG(gpu), MAX(gpu_max),
                   AVG(ram_mb), MAX(ram_total_mb), AVG(vram_mb), MAX(vram_total_mb),
                   AVG(vram_own_mb), AVG(vram_foreign_mb),
                   CAST(ts / :b AS INTEGER), COUNT(vram_own_mb)
            FROM samples WHERE ts >= :start AND ts <= :end
            GROUP BY CAST(ts / :b AS INTEGER) ORDER BY t
        """,
            args,
        ).fetchall()
        first = conn.execute("SELECT MIN(ts) FROM samples").fetchone()[0]
        by_source = conn.execute(
            """
            SELECT CAST(ts / :b AS INTEGER), key, SUM(mb), MAX(mb_max),
                   MAX(label), MAX(own), MIN(mb_min), COUNT(*)
            FROM vram_procs WHERE ts >= :start AND ts <= :end
            GROUP BY CAST(ts / :b AS INTEGER), key
        """,
            args,
        ).fetchall()
    keys = (
        "t",
        "cpu",
        "cpu_max",
        "gpu",
        "gpu_max",
        "ram_mb",
        "ram_total_mb",
        "vram_mb",
        "vram_total_mb",
        "vram_own_mb",
        "vram_foreign_mb",
    )
    out["points"] = [
        {
            k: (round(v, 1) if isinstance(v, float) else v)
            for k, v in zip(keys, r[: len(keys)], strict=True)
        }
        for r in rows
    ]
    out["first"] = first
    out["vram_by"] = _sources(rows, by_source)
    return out


def _sources(rows: list, by_source: list) -> dict:
    """Per source a value per point: the bucket's mean over the intervals
    that knew their processes. The largest sources of each kind by name
    (by their peak in the range), the rest of each kind summed.

    ``low``/``high`` per named source: the lowest and the highest reading
    in the bucket — how far it swung. An interval without the source held
    0 of it, so a bucket where it came or went starts at 0."""
    where = {r[-2]: i for i, r in enumerate(rows)}
    known = [r[-1] for r in rows]
    peak: dict[str, float] = {}
    meta: dict[str, tuple[str, bool]] = {}
    for _bk, key, _sum, mx, label, own, _mn, _count in by_source:
        peak[key] = max(peak.get(key, 0.0), mx or 0.0)
        meta[key] = (label or key, bool(own))
    named: dict[str, str] = {}
    for own in (True, False):
        ranked = sorted((k for k in peak if meta[k][1] == own), key=lambda k: (-peak[k], k))
        for i, k in enumerate(ranked):
            named[k] = k if i < HISTORY_SOURCES else ("own:other" if own else "foreign:other")
    series: dict[str, list] = {}
    low: dict[str, list] = {}
    high: dict[str, list] = {}
    for bk, key, total, mx, _label, _own, mn, count in by_source:
        i = where.get(bk)
        if i is None or not known[i]:
            continue
        target = named[key]
        values = series.setdefault(target, [None] * len(rows))
        values[i] = (values[i] or 0.0) + total / known[i]
        if target == key:
            # Absent from an interval: it held 0 there. Recorded before the
            # lowest reading was (mb_min NULL): unknown — no band, not 0.
            lowest = 0.0 if count < known[i] else (math.nan if mn is None else mn)
            low.setdefault(key, [None] * len(rows))[i] = lowest
            high.setdefault(key, [None] * len(rows))[i] = mx
    for key, values in series.items():
        for i, v in enumerate(values):
            if not known[i]:
                continue
            if v is None:
                values[i] = 0.0
            else:
                values[i] = round(v, 1)
            if key in low:
                if low[key][i] is None:
                    low[key][i] = high[key][i] = 0.0
                elif math.isnan(low[key][i]):
                    low[key][i] = high[key][i] = None
                else:
                    low[key][i] = round(low[key][i], 1)
                    high[key][i] = round(high[key][i], 1)
    labels = {"own:other": ("Other (own)", True), "foreign:other": ("Other programs", False)}
    order = sorted(
        series,
        key=lambda k: (not (meta.get(k) or labels[k])[1], -max(v or 0.0 for v in series[k]), k),
    )
    return {
        "keys": [
            {"key": k, "label": (meta.get(k) or labels[k])[0], "own": (meta.get(k) or labels[k])[1]}
            for k in order
        ],
        "series": series,
        "low": low,
        "high": high,
    }


# ── Sampler thread ───────────────────────────────────────────────────────


def _extra_readings(due: float) -> bool:
    """While someone watches the live view: a reading every ``LIVE_SEC``
    until the regular one at ``due`` (time.monotonic) — into the live
    buffer only. True when the sampler is to stop."""
    global _cost
    at = due - SAMPLE_SEC + LIVE_SEC
    while at <= due - LIVE_SEC / 2 and live_step() == LIVE_SEC:
        if _stop.wait(max(0.0, at - time.monotonic())):
            return True
        try:
            t0 = time.perf_counter()
            r = measure(cpu=False)
            _cost = time.perf_counter() - t0
            _remember(r)
        except Exception as e:  # the sampler must never die
            logger.debug("Metrics sampler (live): %s", e)
        at += LIVE_SEC
    return False


def _run(path: Path) -> None:
    global _latest, _cost
    last_write = time.time()
    last_prune = 0.0
    measure()  # prime psutil's CPU reference
    _cpu_live.read()  # and the live buffer's
    while True:
        # The regular reading SAMPLE_SEC after the last one, as ever; the
        # live view's extra readings fit in between without moving it
        due = time.monotonic() + SAMPLE_SEC
        if _extra_readings(due) or _stop.wait(max(0.0, due - time.monotonic())):
            break
        try:
            t0 = time.perf_counter()
            r = measure()
            _cost = time.perf_counter() - t0
            with _lock:
                _latest = r
                _pending.append(r)
            _remember(r)
            now = time.time()
            if now - last_write >= WRITE_SEC:
                with _lock:
                    batch, _pending[:] = list(_pending), []
                row = aggregate(batch)
                if row:
                    write(path, row)
                last_write = now
            if now - last_prune >= 3600:
                prune(path)
                last_prune = now
        except Exception as e:  # the sampler must never die
            logger.debug("Metrics sampler: %s", e)


def start(db_dir: Path) -> None:
    """Start the sampler (once). ``db_dir`` is where ``metrics.db`` lives."""
    global _thread, _db_path
    if _thread and _thread.is_alive():
        return
    _db_path = Path(db_dir) / "metrics.db"
    _stop.clear()
    _thread = threading.Thread(target=_run, args=(_db_path,), name="metrics", daemon=True)
    _thread.start()


def stop() -> None:
    _stop.set()
