"""
Low-VRAM mode: config default + per-run override.

Why: on a 12-GB card, large-v3 + separator resident from a previous run,
plus an alignment model can end in "CUDA failed with error out of memory".
The global setting ``hardware.low_vram`` didn't help because you
can't set it per file — but that's exactly what's needed: one large run
should unload models, many small ones should remain resident.

``transcription_worker`` sets the override at the start and cleans it up in
``finally``. Per thread: with queue lanes several files run at once, each in
its own thread from start to end — one entry's choice must not flip
another's caching, and the first to finish must not reset the other's.
(A low-VRAM entry runs alone anyway: core/queue.py holds the GPU gate
exclusively for it.)

``hardware.low_vram: auto`` (the default) switches it on for a card of up to
11 GB — a smaller card stayed resident and ran into the out-of-memory retry
with every long file.
"""

from __future__ import annotations

import logging
import sys
import threading

from whisper_tm.config import config

logger = logging.getLogger(__name__)

_local = threading.local()

#: A card smaller than this spares VRAM unless set otherwise: Whisper, the
#: separator, an alignment model and the diarization together peak at
#: about 8.4 GB, and while large-v3 loads beside a resident separator at
#: 11.5 GB. The bound lies between the 11 GB and the 12 GB cards: "under
#: 10" left a 10 GB card (10.0 GiB exactly) to the out-of-memory retries.
SMALL_CARD_GB = 11.5

_card: float | None = None


def card_gb() -> float | None:
    """Memory of the card the jobs run on (``hardware.gpu_index``) in GiB —
    None without one. NVIDIA's is read without a CUDA context (NVML); an
    AMD card's once torch is loaded."""
    global _card
    if _card is not None:
        return _card
    index = 0
    try:
        # For testing: the size the card is taken to have
        assumed = float(config.get("hardware", "assume_vram_mb", default=0) or 0)
        if assumed > 0:
            return assumed / 1024
    except (TypeError, ValueError):
        pass
    try:
        index = int(config.get("hardware", "gpu_index", default=0) or 0)
        from whisper_tm.core import debug

        mb = (debug.gpu_static().get(index) or {}).get("total_mb")
        if mb:
            _card = mb / 1024
            return _card
    except Exception as e:  # noqa: BLE001 — then torch, or no card
        logger.debug("Card size via NVML: %s", e)
    torch = sys.modules.get("torch")  # never imported just for this
    try:
        if torch is not None and torch.cuda.is_available():
            _card = torch.cuda.get_device_properties(index).total_memory / 1024**3
    except Exception as e:  # noqa: BLE001
        logger.debug("Card size via torch: %s", e)
    return _card


def default_low_vram() -> bool:
    """``hardware.low_vram``: the choice for runs queued without one."""
    return resolve(config.get("hardware", "low_vram", default="auto"))


def resolve(value) -> bool:
    """A ``hardware.low_vram`` value as on or off: true and false decide,
    anything else ("auto", the default) means :func:`small_card`."""
    if isinstance(value, bool):
        return value
    word = str(value or "").strip().lower()
    if word in ("true", "yes", "on", "1"):
        return True
    if word in ("false", "no", "off", "0"):
        return False
    return small_card()


def small_card() -> bool:
    """What "auto" means here: a card under :data:`SMALL_CARD_GB`."""
    size = card_gb()
    return size is not None and size < SMALL_CARD_GB


def set_low_vram_override(value: bool | None) -> None:
    """Set override (True/False) or remove it (None) — for this thread."""
    _local.override = value


def low_vram() -> bool:
    """Does LOW-VRAM apply to the CURRENT run?

    The entry decides; ``hardware.low_vram`` is only the default for runs
    queued without a choice (and the preset of the dialog switch). What
    someone switched off when clicking "Transcribe" must stay off.

    Making the global setting a floor (so that switching it on also
    affected entries already queued as ``false``) fixed one direction and
    broke the other: with sparing switched off in the dialog, every short
    file still unloaded and reloaded Whisper and the separator (a large
    share of a short file's run time).
    Changing entries that already wait is the queue editor's job
    ("Spare VRAM" bulk action), not a global override.

    The OOM retry writes ``True`` into the entry, so it still conserves.
    """
    override = getattr(_local, "override", None)
    if override is not None:
        return override
    return default_low_vram()


# ── Free memory and waiting for it ─────────────────────────────────────
#
# Why: the retry after a CUDA OOM used to start in THE SAME second as the
# error — with a card as full as before (another program may hold part of it;
# the WhisperX step alone needs several GB). Every such retry failed. A retry
# that waits nothing is not one.
#
# Measured is the TOTAL used memory of the card (all processes, via
# NVML/nvidia-smi — core/debug.py), not torch.cuda.*: what a neighbor
# process holds, torch doesn't see.


def total_mb(gpu_index: int | None = None) -> int | None:
    """Total VRAM of the working GPU in MB — None if there's no info. The
    size the card is taken to have when ``hardware.assume_vram_mb`` is set."""
    from whisper_tm.core import debug

    try:
        assumed = int(float(config.get("hardware", "assume_vram_mb", default=0) or 0))
        if assumed > 0 and gpu_index is None:
            return assumed
    except (TypeError, ValueError):
        pass
    try:
        idx = (
            int(gpu_index)
            if gpu_index is not None
            else int(config.get("hardware", "gpu_index", default=0) or 0)
        )
        return int(debug.gpu_static()[idx].get("total_mb") or 0) or None
    except Exception:
        return None


def free_mb(gpu_index: int | None = None) -> int | None:
    """Free VRAM of working GPU in MB — None if there's no info
    (no NVIDIA card, no nvidia-smi/pynvml)."""
    from whisper_tm.core import debug

    try:
        idx = (
            int(gpu_index)
            if gpu_index is not None
            else int(config.get("hardware", "gpu_index", default=0) or 0)
        )
        used = debug.gpu_mem_used_mb()
        static_text = debug.gpu_static()
        if idx not in used or idx not in static_text:
            return None
        total = int(static_text[idx].get("total_mb") or 0)
        if total <= 0:
            return None
        return max(0, total - int(used[idx]))
    except Exception:
        return None


def wait_for_free_vram(
    min_mb: int, max_sec: float, *, abort=None, report_it=None, tick: float = 5.0
) -> tuple[bool, int | None]:
    """Block until ``min_mb`` is free or ``max_sec`` has elapsed.

    Returns ``(reached, last_free_mb)``. ``abort()`` is polled every
    tick (shutdown/pause) and ends the wait with ``False``. Without GPU
    information (``free_mb() is None``), no waiting happens — "unknown"
    then counts as "free", because there is nothing to measure.
    """
    import time

    free = free_mb()
    if free is None:
        return True, None
    t0 = time.monotonic()
    reported = False
    while free < min_mb:
        if abort is not None and abort():
            return False, free
        if time.monotonic() - t0 >= max_sec:
            return False, free
        if report_it is not None and not reported:
            report_it(free)
            reported = True
        time.sleep(min(tick, max(0.1, max_sec - (time.monotonic() - t0))))
        free = free_mb()
        if free is None:
            return True, None
    return True, free
