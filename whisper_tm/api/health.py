"""
Health check: A glance that answers "Is everything healthy?".

GET /api/health/detail collects everything the health GUI shows — deliberately
in ONE endpoint, so the overlay gets by with one request and the indicator
lights stay consistent with each other (same point in time):

- GPU: VRAM used/total, resident models, idle-unloader threshold
- Queue: state + rate (files/h, x-realtime rolling) + error rate
- DB: reachable, size, schema version, auto-backup status
- NAS/media directory: reachable + latency sample
- System warnings (banner mechanism) + prefilter tally since start

The lean /api/health (app.py) stays untouched — it is the cheap
30-s online ping from the frontend; this one is the expensive detail view
(FS access, several DB aggregates) and is only fetched when the overlay is open.
"""

import logging
import os
import threading
import time

from flask import Blueprint, jsonify, request

from whisper_tm.config import config
from whisper_tm.extensions import db, job_manager

logger = logging.getLogger(__name__)

health_bp = Blueprint("health", __name__)

_SERVER_START = time.time()

#: How long the health view waits for the media folder to answer.
NAS_TIMEOUT_S = 5.0
_nas_lock = threading.Lock()
_nas_probe: dict | None = None

# Rolling rate window: large enough for stable numbers, small enough
# to show slowdowns (sluggish NAS, model reloaded) promptly.
_RATE_WINDOW_S = 3600
_ERROR_WINDOW_S = 86400


def _gpu_block() -> dict:
    out = {
        "available": False,
        "name": "",
        "vram_total_mb": 0,
        "vram_used_mb": 0,
        "resident_models": [],
        "idle_min": config.get("hardware", "model_idle_min", default=10),
    }
    try:
        from whisper_tm.core import debug as _debug

        if _debug.gpu_available():
            out["available"] = True
            static_text = _debug.gpu_static()
            used = _debug.gpu_mem_used_mb()
            idx = config.get("hardware", "gpu_index", default=0)
            info = static_text.get(idx) or next(iter(static_text.values()), {})
            out["name"] = info.get("name", "")
            out["vram_total_mb"] = info.get("total_mb", 0)
            out["vram_used_mb"] = used.get(idx, next(iter(used.values()), 0))
    except Exception as e:  # noqa — GPU info is a convenience, not a must
        logger.debug("GPU block: %s", e)

    # Resident models: ASR registry + separator flag. Deliberately names
    # only — the health GUI answers "what's holding VRAM right now?", nothing more.
    from whisper_tm.core import vram_procs

    out["resident_models"] = vram_procs.resident_models()
    return out


def _llama_block() -> dict:
    """Local llama.cpp server: installed? running? does the config fit?

    Deliberately tolerant: anyone using a remote provider (OpenAI, Ollama
    on another machine) simply gets "not installed" here — that's not a
    deficiency, just a statement of fact.
    """
    try:
        from whisper_tm.core import llama_server

        return llama_server.check()
    except Exception as e:  # noqa — info is a convenience
        logger.debug("llama block: %s", e)
        return {"installed": False, "warnings": [], "hints": []}


def _queue_block() -> dict:
    out = {
        "running": False,
        "paused": False,
        "pending": 0,
        "done": 0,
        "error": 0,
        "skipped": 0,
        "cancelled": 0,
        "processed": 0,
        "total": 0,
        "current_file": None,
        "running_count": 0,
        "speed": {
            "files_per_h": 0.0,
            "x_realtime": 0.0,
            "audio_h_per_h": 0.0,
            "window_min": _RATE_WINDOW_S // 60,
            "done_in_window": 0,
        },
        "errors_24h": {"done": 0, "error": 0, "percent": 0.0},
    }
    try:
        from whisper_tm.core import queue as q

        snap = q.snapshot()
        for k in ("running", "paused", "pending", "processed", "total"):
            out[k] = snap.get(k, out[k])
        out["done"] = snap.get("done_count", 0)
        out["error"] = snap.get("fail_count", 0)
        out["skipped"] = snap.get("skip_count", 0)
        out["cancelled"] = snap.get("cancel_count", 0)
        out["current_file"] = snap.get("current_file")
        # Queue lanes: current_file is the oldest running file; the card
        # names it plus how many more run beside it.
        out["running_count"] = snap.get("running_count") or (1 if out["current_file"] else 0)
    except Exception as e:  # noqa
        logger.debug("Queue snapshot: %s", e)

    if not db:
        return out
    try:
        with db.connection() as conn:
            # Rate from the most recently FINISHED transcripts: audio
            # duration vs. compute time — the usual x-realtime definition.
            r = conn.execute(
                "SELECT COUNT(*) AS n, SUM(duration_sec) AS audio,"
                "       SUM(processing_secs) AS compute"
                "  FROM transcripts"
                " WHERE complete = 1 AND updated_at > unixepoch() - ?",
                (_RATE_WINDOW_S,),
            ).fetchone()
            n = r["n"] or 0
            audio = r["audio"] or 0.0
            compute_val = r["compute"] or 0.0
            out["speed"]["done_in_window"] = n
            out["speed"]["files_per_h"] = round(n * 3600 / _RATE_WINDOW_S, 1)
            out["speed"]["audio_h_per_h"] = round(audio / _RATE_WINDOW_S, 2)
            if compute_val > 0:
                out["speed"]["x_realtime"] = round(audio / compute_val, 1)

            f = conn.execute(
                "SELECT SUM(status = 'error') AS error,"
                "       SUM(status IN ('done','error','skipped')) AS finished"
                "  FROM queue_items"
                " WHERE finished_at > unixepoch() - ?",
                (_ERROR_WINDOW_S,),
            ).fetchone()
            error = f["error"] or 0
            terminated = f["finished"] or 0
            out["errors_24h"] = {
                "done": terminated - error,
                "error": error,
                "percent": round(error / terminated * 100, 1) if terminated else 0.0,
            }
    except Exception as e:  # noqa
        logger.debug("Queue rate: %s", e)
    return out


def _db_block() -> dict:
    out = {"ok": False, "size": 0, "schema_version": 0, "backup_counter": 0, "backup_every": 0}
    if not db:
        return out
    try:
        with db.connection() as conn:
            row = conn.execute("SELECT value FROM schema_info WHERE key = 'version'").fetchone()
            out["schema_version"] = int(row["value"]) if row else 0
        out["size"] = db.get_db_size()
        out["backup_counter"] = db.get_backup_counter()
        out["backup_every"] = config.get("database", "backup_every_changes", default=25)
        out["ok"] = True
    except Exception as e:  # noqa
        logger.debug("DB block: %s", e)
    return out


def _nas_block() -> dict:
    """Reachability + latency of the media directory.

    listdir instead of stat: the CIFS mount may answer stat from the
    attribute cache (actimeo); a directory listing has to reach the server.
    """
    path = str(config.media_dir)
    out = {"ok": False, "latency_ms": None, "path": path}
    global _nas_probe
    with _nas_lock:
        probe = _nas_probe
        if probe is None or not probe["thread"].is_alive():
            # Single flight: a new probe only when none is still hanging —
            # a hung mount would otherwise collect one stuck thread per call
            probe = {"path": path, "result": None}

            def run(p=probe):
                t0 = time.perf_counter()
                try:
                    os.listdir(p["path"])
                    p["result"] = ("ok", round((time.perf_counter() - t0) * 1000, 1))
                except OSError as e:
                    p["result"] = ("error", str(e)[:200])

            probe["thread"] = threading.Thread(target=run, name="nas-probe", daemon=True)
            probe["thread"].start()
            _nas_probe = probe
    # A hung NAS used to tie up the request thread for as long as the mount
    # hung; now the answer comes after at most NAS_TIMEOUT_S
    probe["thread"].join(NAS_TIMEOUT_S)
    result = probe["result"]
    if result is None:
        out["error"] = f"NAS not responding (no answer within {NAS_TIMEOUT_S:.0f} s)"
    elif result[0] == "ok":
        out["latency_ms"] = result[1]
        out["ok"] = True
    else:
        out["error"] = result[1]
    return out


@health_bp.route("/api/system/load")
def system_load():
    """Live load for the top-bar meters: CPU, RAM, GPU, VRAM.

    The latest reading of the metrics sampler (core/metrics.py, every 2 s),
    so polling it costs nothing; measured on the spot if the sampler isn't
    running.
    """
    from whisper_tm.core import metrics

    r = metrics.latest()
    r.pop("ts", None)
    if r.get("gpu"):
        # The meters show totals; the split is /api/system/vram
        r["gpu"] = {k: v for k, v in r["gpu"].items() if k != "procs"}
    return jsonify(r)


@health_bp.route("/api/system/vram")
def system_vram():
    """VRAM by process right now — for the performance view.

    This app's processes by role (the server with the models it holds),
    every other program by name, and ``unattributed_mb``: what the driver
    counts as used but assigns to no process. From the metrics sampler's
    latest reading (every 2 s), so asking costs nothing.
    """
    from whisper_tm.core import metrics, vram_procs

    g = metrics.latest().get("gpu") or {}
    procs = g.get("procs")
    # `index`/`name`: the GPU WhisperX runs on — the only one listed
    out = {
        "available": procs is not None,
        "index": g.get("index"),
        "name": g.get("name", ""),
        "total_mb": g.get("vram_total_mb"),
        "used_mb": g.get("vram_used_mb"),
        "procs": [],
        "own_mb": None,
        "foreign_mb": None,
        "unattributed_mb": None,
    }
    if procs is None:
        return jsonify(out)
    models = vram_procs.resident_models()
    rows = []
    for p in procs:
        row = {
            k: p.get(k)
            for k in ("pid", "mb", "type", "program", "user", "own", "role", "label", "key")
        }
        if p.get("role") == "server":
            row["models"] = models
        rows.append(row)
    counted = [p for p in procs if p.get("mb") is not None]
    out["procs"] = rows
    out["own_mb"] = sum(p["mb"] for p in counted if p.get("own"))
    out["foreign_mb"] = sum(p["mb"] for p in counted if not p.get("own"))
    out["unattributed_mb"] = max(
        0, (g.get("vram_used_mb") or 0) - out["own_mb"] - out["foreign_mb"]
    )
    return jsonify(out)


@health_bp.route("/api/system/history")
def system_history():
    """Load history for the performance view: CPU, GPU, RAM and VRAM over time.

    Query: ``start`` and ``end`` as unix seconds (default: the last 24 h),
    ``points`` — at most this many buckets (default 600, max 2000). Each
    point has the bucket's mean plus the CPU/GPU peak; ``bucket`` is the
    bucket width in seconds, ``first`` the oldest stored sample.
    """
    import math

    from whisper_tm.core import metrics

    # Only finite numbers; metrics.history clamps the range
    end = request.args.get("end", type=float)
    end = end if end and math.isfinite(end) else time.time()
    start = request.args.get("start", type=float)
    if start is None or not math.isfinite(start) or start >= end:
        start = end - 86400
    points = max(10, min(request.args.get("points", default=600, type=int) or 600, 2000))
    return jsonify(metrics.history(start=start, end=end, max_points=points))


@health_bp.route("/api/system/live")
def system_live():
    """The raw readings of the last seconds — the performance view's live mode.

    Query: ``since`` (unix seconds) — only the readings newer than that, so
    the view fetches what it hasn't got yet; without it the last 90 s (and a
    little more). From the sampler's in-memory buffer, so asking costs
    nothing. Each request also counts as "someone watches": for the next
    ``metrics.LIVE_WATCH_SEC`` the sampler adds a reading every
    ``metrics.LIVE_SEC`` (``step`` in the answer says what it does).
    """
    from whisper_tm.core import metrics

    metrics.watch()
    return jsonify(metrics.live(since=request.args.get("since", type=float)))


@health_bp.route("/api/vram/unload", methods=["POST"])
def vram_unload():
    """Unload all resident models now ("Unload VRAM" in the top bar).

    Answers 409 while a transcription runs — unloading then would kill its
    diarization worker. VRAM before/after comes from the driver (all
    processes), so the number also covers what PyTorch's cache held.
    """
    from whisper_tm.core import model_idle

    before = _gpu_block()
    r = model_idle.unload_now()
    if not r["ok"]:
        return jsonify(
            {
                "ok": False,
                "error": "A transcription is running — models stay loaded until it is done.",
            }
        ), 409
    after = _gpu_block()
    freed = max(0, (before.get("vram_used_mb") or 0) - (after.get("vram_used_mb") or 0))
    return jsonify(
        {
            "ok": True,
            "unloaded": r["unloaded"],
            "models": before.get("resident_models", []),
            "freed_mb": freed,
            "vram_used_mb": after.get("vram_used_mb", 0),
            "vram_total_mb": after.get("vram_total_mb", 0),
        }
    )


@health_bp.route("/api/health/detail")
def health_detail():
    """Everything the health overlay shows, in one response: version, uptime, GPU
    (VRAM, resident models), llama.cpp, queue rate and error rate, database,
    NAS latency, system warnings and the vocal-isolation prefilter tally.
    """
    from whisper_tm import __version__

    try:
        from whisper_tm.core.vocal_isolation import prefilter_balance

        prefilter = prefilter_balance()
    except Exception:  # noqa
        prefilter = {}
    warnings = []
    try:
        if job_manager:
            warnings = job_manager.system_warnings()
    except Exception as e:  # noqa
        logger.debug("Warnings: %s", e)
    return jsonify(
        {
            "version": __version__,
            "uptime_s": int(time.time() - _SERVER_START),
            "gpu": _gpu_block(),
            "llama": _llama_block(),
            "queue": _queue_block(),
            "db": _db_block(),
            "nas": _nas_block(),
            "warnings": warnings,
            "prefilter": prefilter,
        }
    )
