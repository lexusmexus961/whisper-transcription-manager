"""Idle unloader for resident models.

Models remain resident normally (3.0a) — correct for queue continuous
operation, but blocks the GPU when server only runs in background
(e.g. while another program uses the same card). So: after configurable idle
(`hardware.model_idle_min`, default 10, 0 = never) ALL resident
models are unloaded and VRAM freed. Next job simply reloads them
(one-time ~3–5 s).

A SHARED timestamp for Whisper/alignment/diarization AND separator:
while a job runs, it touches at least every few minutes one of the
models (chunks, blocks) — timestamp stays fresh, nothing unloaded
mid-job.
"""

from __future__ import annotations

import logging
import threading
import time

from whisper_tm.config import config

logger = logging.getLogger("whisper_tm.core.transcription")

_last_use = time.time()
_lock = threading.Lock()
_watchdog: threading.Thread | None = None


def report_usage() -> None:
    """Called from every model access — keeps models alive."""
    global _last_use
    _last_use = time.time()
    _start_watchdog()


def _idle_min() -> float:
    try:
        return float(config.get("hardware", "model_idle_min", default=10))
    except (TypeError, ValueError):
        return 10.0


def _something_loaded() -> bool:
    # Diarization worker counts too: it keeps its model in its OWN process
    # (so hanging can be killed), but occupies same VRAM. Without it here
    # watchdog would think it's done while ~1 GB still sits on card.
    from whisper_tm.core import diar_process, transcription, vocal_isolation

    return (
        bool(transcription._models)
        or vocal_isolation._separator is not None
        or diar_process.something_running()
    )


def _job_running() -> bool:
    """Is a transcription running right now? Then unload NOTHING.

    The comment above assumes a running job touches "at least every few
    minutes one of the models". True for normal-length files — fails for
    very long ones: a recording many hours long sticks in ONE step for
    minutes (separation well over ten minutes, diarization similar),
    without touching a model again.

    Without this check a job can die that way: the watchdog thinks the
    models are idle, `release_models` takes the diarization worker with
    it (it keeps its VRAM share in its own process), and the job fails with
    "diarization worker unexpectedly terminated".

    The usage report in worker idle loop fixes the specific case; this query
    is the general lock. A running job is never idle — no matter how long
    its current step takes.
    """
    try:
        from whisper_tm.extensions import job_manager

        return bool(job_manager and job_manager.list_active())
    except Exception:
        # When in doubt, do NOT unload: a model held too long costs
        # VRAM, one unloaded too early costs an entire job.
        return True


def unload_all() -> None:
    """Unload every resident model: ASR, alignment, diarization worker, separator.

    Shared by the idle watchdog and the "Unload VRAM" button in the top
    bar. Callers check :func:`_job_running` first — this function does not,
    so tests and shutdown can use it unconditionally.
    """
    from whisper_tm.core import transcription, vocal_isolation

    transcription.release_models()
    vocal_isolation.release_separator()


def _stop_media_service() -> bool:
    """The Images & sounds service, stopped unless an index run or a search
    uses it. Releasing its GPU model leaves the process with its CUDA
    context and cache (~1.8 GB measured on the card); only stopping it
    gives that back. It starts again with the next search (~4 s)."""
    try:
        from whisper_tm.core.media_server import SERVER
        from whisper_tm.media import indexer

        if indexer.INDEXER.running():
            return False
        return SERVER.stop("unload on request")
    except Exception as e:  # noqa: BLE001 — the other models are unloaded either way
        logger.info("Media service not stopped: %s", e)
        return False


def unload_now() -> dict:
    """Manual unload (top-bar button). Refuses while a job is running.

    A running job would lose its diarization worker (see
    :func:`_job_running`), so the button never overrides that lock.
    Between two queue files it is allowed: the next file simply reloads.
    """
    if _job_running():
        return {"ok": False, "reason": "job_running"}
    media = _stop_media_service()
    if not _something_loaded():
        return {"ok": True, "unloaded": media}
    from whisper_tm.core.gpu_gate import GATE

    # Only if no model is in use this very moment — a queue lane may have
    # taken its next file between the check above and here.
    with GATE.try_exclusive("unload on request") as got:
        if not got:
            return {"ok": False, "reason": "job_running"}
        unload_all()
    logger.info("💤 Models unloaded on request — VRAM freed")
    return {"ok": True, "unloaded": True}


def _watchdog_run() -> None:
    while True:
        time.sleep(60)
        try:
            limit = _idle_min()
            if limit <= 0:
                continue  # 0 = never unload
            idle = time.time() - _last_use
            if idle < limit * 60 or not _something_loaded():
                continue
            if _job_running():
                # Not idle, just a long step (see _job_running).
                logger.debug("Idle limit reached but job running — nothing unloaded")
                continue
            from whisper_tm.core.gpu_gate import GATE

            # The check above has a gap — a queue lane may claim its next
            # file right now. Unload only while no model is in use.
            with GATE.try_exclusive("idle unload") as got:
                if not got:
                    continue
                unload_all()
            logger.info(
                "💤 Models unloaded after %.0f min idle — "
                "VRAM freed (limit: %.0f min, "
                "hardware.model_idle_min; next job reloads "
                "them automatically)",
                idle / 60,
                limit,
            )
        except Exception as e:  # Watchdog must never die
            logger.debug("Idle watchdog: %s", e)


def _start_watchdog() -> None:
    global _watchdog
    with _lock:
        if _watchdog is None or not _watchdog.is_alive():
            _watchdog = threading.Thread(target=_watchdog_run, name="model-idle", daemon=True)
            _watchdog.start()
