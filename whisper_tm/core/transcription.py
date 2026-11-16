"""
Transcription worker: WhisperX pipeline with alignment and diarization.
Adapted for the new job management and hardware profile.

NOTE: This file is a bridge to the existing code.
The worker logic stays identical but uses the new JobManager.
"""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import math
import os
import re
import shutil
import sqlite3
import threading
import time
import warnings
from pathlib import Path

from whisper_tm.config import config
from whisper_tm.core import debug, hf_access, media_formats
from whisper_tm.core.audio_utils import (
    CorruptMediaError,
    _is_corrupt_stderr,
    extract_audio,
    get_duration,
    get_duration_safe,
    has_audio_stream,
    run_cancellable,
)
from whisper_tm.core.gpu_gate import GATE
from whisper_tm.core.helpers import (
    add_to_skiplist,
    atomic_write_json,
    find_transcript,
    fmt_seg,
    format_duration,
    format_size,
    format_ts,
    import_clean,
    invalidate_complete_cache,
    is_silent,
    transcript_path_for,
)
from whisper_tm.extensions import db, hw, job_manager
from whisper_tm.hardware import get_whisperx_device

logger = logging.getLogger(__name__)

# pyannote/torch emit std() DoF warning per diarization.
# Set filter ONCE at module level — not per job, or warnings.filters grows unbounded.
warnings.filterwarnings("ignore", message="std\\(\\): degrees of freedom")

# Default repo from whisperx.diarize.DiarizationPipeline — must match installed
# whisperx. Its token, its cache and its gate live in core/hf_access.
DIARIZATION_REPO = hf_access.REPO


# ── Resident models ───────────────────────────────────────────────────
# Default: Whisper, alignment and diarization stay loaded after the first job
# — measured, all models fit together with the separator in
# VRAM (measured peak 8.4 of 12 GB); loading per file
# cost 7.4% of total compute time. hardware.low_vram=true restores the
# old behavior (reload fresh each time, `del` frees again).
_models: dict = {}
_models_lock = threading.Lock()
# Which lanes (threads) use which resident model right now: from their
# _resident_model call until they let go (release_held). Eviction never
# touches a model in use — evicted, it stayed in VRAM through the lane's
# reference anyway, and that lane's next file loaded a second copy.
_holders: dict = {}


def _family(key) -> tuple | None:
    """Which models replace each other: alignment models per device, Whisper
    weights per card. None for everything else (pipelines are light and
    go with their weights, diarization is one per device anyway)."""
    if not isinstance(key, tuple) or not key:
        return None
    if key[0] == "align" and len(key) >= 3:
        return ("align", key[2])
    if key[0] == "asr-weights" and len(key) >= 4:
        return ("asr-weights", key[2], key[3])
    return None


def _family_limit(family: tuple) -> int:
    """How many members of a family stay resident.

    Alignment: two languages (``hardware.resident_align_models``) — a queue
    alternating between two languages does not reload, a third one evicts
    the least recently used. Whisper weights: one set per card — a second
    model would hold another 1.5–3 GB next to it.
    """
    if family[0] == "align":
        try:
            return max(1, int(config.get("hardware", "resident_align_models", default=2)))
        except (TypeError, ValueError):
            return 2
    return 1


def _in_use(key) -> bool:
    """Is a living lane holding ``key``? Call with _models_lock held.
    A thread that ended without letting go (a crash) holds nothing."""
    users = _holders.get(key)
    if not users:
        return False
    alive = {t.ident for t in threading.enumerate()}
    users &= alive
    if not users:
        _holders.pop(key, None)
    return bool(users)


def release_held() -> None:
    """This lane is done with the resident models it fetched — they may be
    evicted again. The worker calls it after each GPU stage and at the end."""
    me = threading.get_ident()
    with _models_lock:
        for k in list(_holders):
            _holders[k].discard(me)
            if not _holders[k]:
                del _holders[k]


def _evict_for(key) -> None:
    """Make room in ``key``'s family before ``key`` loads. Call with
    _models_lock held.

    Without eviction, every language's alignment model and every Whisper
    weight set stayed resident until the idle unloader — which never runs
    while the queue is busy — and a mixed-language queue filled the shared
    card. Only runs when a NEW key loads, inside the loading lane's GPU
    stage; a lane still using an evicted model keeps its own reference, so
    nothing is pulled from under a running step — the memory goes once it
    is done with it.
    """
    family = _family(key)
    if family is None:
        return
    members = [k for k in _models if _family(k) == family]  # oldest use first
    surplus = len(members) - (_family_limit(family) - 1)

    def busy(k):
        # Weights are in use while any pipeline built on them is
        return _in_use(k) or (
            k[0] == "asr-weights"
            and any(p[0] == "asr" and p[1:5] == k[1:5] and _in_use(p) for p in _models)
        )

    # Only what nobody uses — the family may stay above its limit until
    # the lanes let go; a later load evicts then
    gone = [k for k in members if not busy(k)][: max(0, surplus)]
    for k in gone:
        _models.pop(k, None)
        if k[0] == "asr-weights":
            # The pipelines built around these weights hold them too
            for p in [p for p in _models if p[0] == "asr" and p[1:5] == k[1:5]]:
                _models.pop(p, None)
        logger.info("Resident model %s replaced by %s", k, key)
    if gone:
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass


def _resident_model(key, load_fn, job=None, label=""):
    """Fetch or load a model from the residency cache.

    In low_vram mode, NOTHING is cached — the caller keeps the sole
    reference, and its existing `del model` frees the memory as before.
    In residency mode the cache holds a second reference, turning the
    same `del` spots into harmless no-ops. A new language's alignment model
    or new Whisper weights replace the old ones (_evict_for).
    """
    from whisper_tm.core.model_idle import report_usage
    from whisper_tm.core.vram import low_vram

    report_usage()  # keeps the idle-unloader at bay
    if low_vram():  # config OR per-run override from the dialog
        return load_fn()
    # In residency mode, NO reload for an already-cached model — but
    # after a low-VRAM run the cache may hold stale entries; that's
    # fine (the run itself neither used nor added to them).
    with _models_lock:
        if _family(key) is not None or (isinstance(key, tuple) and key[:1] == ("asr",)):
            # Only what eviction looks at (pipelines: their weights' use)
            _holders.setdefault(key, set()).add(threading.get_ident())
        if key not in _models:
            _evict_for(key)
            _models[key] = load_fn()
        else:
            # Most recently used last: eviction takes the front
            _models[key] = _models.pop(key)
            if job is not None and label:
                job_manager.log(job, f"{label}: resident (no reload)")
        return _models[key]


def _whisperx(*parts):
    """whisperx with the parts the caller uses — imported where no caller's
    frames can be kept alive by them (helpers.import_clean). Its
    ``load_model``, ``align`` … would import their module on first use,
    inside the caller."""
    for part in parts:
        import_clean(f"whisperx.{part}")
    return import_clean("whisperx")


def _whisper_weights(model_name, device, device_index, compute_type):
    """The Whisper weights alone (CTranslate2) — built as whisperx.load_model
    builds them inside, with its defaults."""
    WhisperModel = import_clean("whisperx.asr").WhisperModel
    if compute_type == "default":
        compute_type = "float16" if device == "cuda" else "float32"
    try:
        return WhisperModel(
            model_name,
            device=device,
            device_index=device_index,
            compute_type=compute_type,
            cpu_threads=4,
        )
    except Exception as e:
        # Offline before its first download: the library said "Max retries
        # exceeded with url" or that a snapshot folder is missing
        from whisper_tm.core import prefetch

        if prefetch.is_download_failure(e):
            raise prefetch.download_error(
                f"The Whisper model {model_name}", prefetch.whisper_source(model_name), e
            ) from e
        raise


def _asr_pipeline(
    model_name, device, device_index, compute_type, language, vocab_prompt, asr_extra, job=None
):
    """The WhisperX pipeline for one language and vocabulary — around Whisper
    weights loaded ONCE per model and card.

    A pipeline is light: tokenizer, decoding options, a small VAD. The
    weights are not — large-v3 holds ~3 GB on the card. Every language,
    vocabulary and decoding option used to load them anew, next to the ones
    already resident: an English upload ("auto") loaded a second large-v3
    beside the German one of the queue's other lanes, and a 12 GB card ran
    out of memory in its first attempt.
    """
    whisperx = _whisperx("asr")
    lang = language if language != "auto" else None
    # Without the cache (low-VRAM runs) both come fresh, and the pipeline
    # holds the only reference to the weights: its `del` frees them as before.
    weights = _resident_model(
        ("asr-weights", model_name, device, device_index, compute_type),
        lambda: _whisper_weights(model_name, device, device_index, compute_type),
        job=job,
        label=f"WhisperX {model_name} weights",
    )
    return _resident_model(
        (
            "asr",
            model_name,
            device,
            device_index,
            compute_type,
            lang,
            vocab_prompt or None,
            tuple(sorted(asr_extra.items())),
        ),
        lambda: whisperx.load_model(
            model_name,
            device,
            device_index=device_index,
            compute_type=compute_type,
            language=lang,
            asr_options=({"initial_prompt": vocab_prompt, **asr_extra} if vocab_prompt else None),
            model=weights,
        ),
        job=job,
        label=f"WhisperX {model_name}",
    )


def release_models():
    """Unload all resident ASR models (tests / model switch / emergency).

    The diarization worker belongs here too, even though it sits in its
    own process: its VRAM usage counts just like that of `_models`, and
    the idle-unloader calls exactly this function. Without this line, ~1 GB
    would stay stuck on the card while the server believes it is unloaded.
    """
    from whisper_tm.core import diar_process

    if not GATE.may_take_exclusive():
        # Inside a GPU stage of a run that doesn't own the GPU: the
        # exclusive would wait for itself. The models stay.
        logger.debug("Models stay loaded: released from inside a GPU stage")
        return
    # Not while another queue lane uses one of them: it would hold its own
    # reference (a second copy on the next load) and lose its diarization.
    with GATE.exclusive("unloading models"):
        with _models_lock:
            _models.clear()
        diar_process.terminate_all("Freeing models")
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass


def _asr_blocks(wav, work, total_dur, block_sec, overlap_sec, job=None):
    """Prepare audio for transcription: normally AS ONE PIECE.

    An earlier 600-s split cut sentences off
    mid-word at every boundary. Whisper needs the split neither for quality
    nor for memory (both measured) — only for EXTREMELY long
    files does the audio array (~0.23 GB per hour as float32) push against
    memory limits. Only then is it split into huge blocks, and WITH overlap
    on both sides: speech spanning a block boundary is thus produced in
    full in both blocks; the midpoint dedup in the caller keeps exactly one
    copy. Nothing gets cut anymore.

    Returns: list of (path, log_start, ext_start, log_end) —
    log_* is the half-open responsibility window, ext_start the start
    of the actually extracted (overlapped) audio.
    """
    if total_dur <= block_sec:
        return [(wav, 0.0, 0.0, float("inf"))]

    n = math.ceil(total_dur / block_sec)
    blocks = []
    for i in range(n):
        log_start = i * block_sec
        log_end = (i + 1) * block_sec if i < n - 1 else float("inf")
        ext_start = max(0.0, log_start - overlap_sec)
        ext_end = min(total_dur, (i + 1) * block_sec + overlap_sec)
        bp = work / f"asr_block_{i:03d}.wav"
        r = run_cancellable(
            [
                "ffmpeg",
                "-y",
                "-ss",
                f"{ext_start:.3f}",
                "-i",
                str(wav),
                "-t",
                f"{ext_end - ext_start:.3f}",
                "-acodec",
                "pcm_s16le",
                "-rf64",
                "auto",
                str(bp),
            ],
            job=job,
            timeout=7200,
            label=f"ASR block {i + 1}/{n}",
        )
        if r.returncode != 0 or not bp.exists() or bp.stat().st_size <= 1000:
            raise RuntimeError(
                f"ASR block {i + 1}/{n} extraction failed: {(r.stderr or '')[-300:]}"
            )
        blocks.append((bp, float(log_start), ext_start, log_end))
    if job and job_manager:
        job_manager.log(
            job,
            f"File over {block_sec / 3600:.0f}h boundary — "
            f"{n} ASR blocks with {overlap_sec:.0f}s "
            f"overlap (midpoint dedup, lossless)",
        )
    return blocks


def diarization_possible(hf_token: str | None) -> bool:
    """Can diarization run — WITH or WITHOUT a Hugging Face token?

    Originally the check everywhere was simply ``diarize and hf_token``.
    That's too strict: the token is **only needed for downloading**. If the
    repo is fully in the local cache, ``DiarizationPipeline(token=None)``
    loads without complaint — exactly how the tools under ``tools/`` have
    always worked.

    The bug was nasty because it **fails silently**: without a token the
    banner still reported "Diarization: yes", the phase count quietly
    dropped by one, and the transcript came back without speakers — no
    warning, no log, nothing visibly failed — most likely with a fresh
    configuration.
    """
    if hf_token:
        return True
    return _repo_fully_cached(DIARIZATION_REPO)


#: Is an HF repo fully in the local cache — core/hf_access.repo_cached
_repo_fully_cached = hf_access.repo_cached


class _HfOfflineIfCached:
    """Context manager: switch the HF hub offline when the repo is
    fully in the local cache.

    huggingface_hub makes a HEAD request against huggingface.co (ETag
    check) per file when loading, even with a full cache. If the connection
    is down or slow, every check blocks for a 10s timeout x 5 retries with
    backoff — diarization then "hangs" for minutes even though everything
    is local. With a full cache we therefore force local_files_only;
    without a cache everything stays at the normal online behavior
    (initial download).
    """

    def __init__(self, repo_id: str):
        self.repo_id = repo_id
        self._prev = None  # None = untouched

    def __enter__(self):
        if not _repo_fully_cached(self.repo_id):
            return self  # no (full) cache → load online
        from huggingface_hub import constants

        # constants.HF_HUB_OFFLINE is read at runtime — the env var
        # alone no longer suffices after import.
        self._prev = constants.HF_HUB_OFFLINE
        constants.HF_HUB_OFFLINE = True
        logger.info("HF cache complete for %s — loading offline (no ETag check)", self.repo_id)
        return self

    def __exit__(self, *exc):
        if self._prev is not None:
            from huggingface_hub import constants

            constants.HF_HUB_OFFLINE = self._prev
        return False


def _diar_deadline(total_dur: float) -> float:
    """How long may a diarization run before it counts as hung?

    Measured on a 12 GB card with about two hours of speech:
    **about 75x real time**. The
    default factor of 0.5 allows HALF the audio duration — that's roughly
    37 times the measured time.

    The generosity is intentional: this limit is meant to catch a **hang**
    (pyannote that never returns and holds the GPU), not to punish a slow
    card or a CPU fallback. Anyone who wants it tighter sets
    ``hardware.diarization_deadline_factor`` lower; 0 disables the limit
    (leaving only manual cancellation).

    The floor of 900 s protects short files: for a 30-s note a 15-s
    deadline would be nonsense, since loading the model alone already
    costs 5 s.
    """
    try:
        factor = float(config.get("hardware", "diarization_deadline_factor", default=0.5))
    except (TypeError, ValueError):
        factor = 0.5
    if factor <= 0:
        return float("inf")
    return max(900.0, float(total_dur or 0) * factor)


def _diarize(wav, min_speakers, max_speakers, device, hf_token, job, total_dur, load_done, t_load):
    """Diarize — cancellable, with a time limit.

    The normal path runs in a dedicated, long-lived child process
    (:mod:`whisper_tm.core.diar_process`). Reason: pyannote has no
    time limit, and a hanging call in the same process cannot be stopped
    — Python can't kill a thread, and the CUDA context would be left
    orphaned. In an overnight batch run, that's the difference between
    "one file is missing" and "nothing ran anymore".

    ``hardware.diarization_subprocess: false`` restores the old path in
    the same process — as an emergency exit for environments where no
    child process can start. Diarization is then not cancellable again;
    that is logged as such.
    """
    child_process = bool(config.get("hardware", "diarization_subprocess", default=True))

    if not child_process:
        # ── Old path: in the same process, resident, NOT cancellable ──
        job_manager.log(
            job,
            "⚠ Diarization runs in the same process "
            "(hardware.diarization_subprocess=false) — "
            "a hang is then not cancellable",
        )
        _wx = _whisperx("audio", "diarize")
        DiarizationPipeline = import_clean("whisperx.diarize").DiarizationPipeline
        with _HfOfflineIfCached(DIARIZATION_REPO):
            diar = _resident_model(
                ("diar", device),
                # token=None is valid and the normal case when the model
                # is already in the cache.
                lambda: DiarizationPipeline(token=hf_token or None, device=device),
                job=job,
                label="Diarization",
            )
        load_done("Diarization model loaded", time.time() - t_load)
        dk = {}
        if min_speakers:
            dk["min_speakers"] = min_speakers
        if max_speakers:
            dk["max_speakers"] = max_speakers
        # On this path, the audio loads here — on the normal path the
        # worker does it itself (0.6-1.7 s per hour, and without voice
        # vectors the parent process skips it entirely).
        return diar(_wx.load_audio(str(wav)), **dk)

    # ── Normal path: own process ────────────────────────────────────────
    from whisper_tm.core import diar_process

    worker = diar_process.take(
        device, hf_token=hf_token, offline=_repo_fully_cached(DIARIZATION_REPO)
    )
    already_there = worker.alive
    worker.start_it(job=job)
    if already_there:
        job_manager.log(job, "Diarization: resident (no reload)")
    load_done("Diarization model loaded", time.time() - t_load)

    # ── Cutting out silence ────────────────────────────────────────────
    #
    # Diarization has become the most expensive step (on a day-long
    # recording it took longer than separation) and until now ran over the
    # whole file — including hours of guaranteed silence. A long recording
    # can be more than half silence, and detecting that on the final
    # 16-kHz track costs only seconds.
    #
    # Cutting happens in ONE piece: pyannote reassigns its labels per
    # call, so chunkwise "SPEAKER_00" in the third part would be a
    # different person than in the seventh (see core/diar_silence.py).
    target, parts = wav, []
    try:
        if config.get("hardware", "diarization_skip_silence", default=True):
            target, parts = _diar_cut_silence(wav, total_dur, job)
            if target is None:
                # Everything is silent — there is literally nothing to diarize.
                job_manager.log(job, "Diarization skipped: the audio track is silent throughout")
                return diar_process.as_dataframe([])
    except Exception as e:
        # A cost-saving measure must never cost a job.
        job_manager.warn(job, f"Silence cutting skipped: {str(e)[:120]}")
        target, parts = wav, []

    deadline = _diar_deadline(total_dur)
    segments = worker.diarize_it(
        target, min_speakers=min_speakers, max_speakers=max_speakers, job=job, deadline_s=deadline
    )
    if parts:
        from whisper_tm.core import diar_silence

        before = len(segments)
        segments = diar_silence.times_back(segments, parts)
        logger.debug(
            "Diarization: %d segments from the cut version → %d in original time",
            before,
            len(segments),
        )
    return diar_process.as_dataframe(segments)


def _folder_writable(folder: Path) -> bool:
    """Whether a transcript can be written into ``folder`` — tried, not
    guessed from mode bits (a read-only share, a mount's options and an
    ACL are invisible to os.access)."""
    probe = folder / f".wtm-write-test-{os.getpid()}-{threading.get_ident()}"
    try:
        folder.mkdir(parents=True, exist_ok=True)
        probe.write_bytes(b"")
        probe.unlink()
        return True
    except OSError:
        try:
            probe.unlink(missing_ok=True)
        except OSError:
            pass
        return False


def _diarize_or_none(
    wav,
    min_speakers,
    max_speakers,
    device,
    hf_token,
    *,
    job,
    total_dur,
    load_done,
    t_load,
    voices_only=False,
):
    """The speakers of the recording — or ``None`` when finding them failed
    in a way another try won't mend (the model can't be downloaded, the
    token was revoked, the worker was killed, the time limit on a slow
    CPU). The transcript is done by then and the speakers are an extra: it
    is saved without them and the job says so, as without a token. A full
    card stays the queue's (it waits and tries again and never books an
    error, queue._oom_retry), a cancel stays a cancel, and a voices-only
    run has nothing else to give."""
    try:
        return _diarize(
            wav,
            min_speakers,
            max_speakers,
            device,
            hf_token,
            job=job,
            total_dur=total_dur,
            load_done=load_done,
            t_load=t_load,
        )
    except InterruptedError:
        raise
    except Exception as e:
        if getattr(job, "cancel", False):
            raise InterruptedError("Cancelled") from e
        from whisper_tm.core.queue import _is_oom_error

        if voices_only or _is_oom_error(str(e)):
            raise
        logger.warning("Speaker recognition failed — saved without speakers: %s", e)
        job_manager.warn(
            job, f"Speaker recognition failed — saved without speakers: {str(e)[:300]}"
        )
        return None


def _diar_cut_silence(wav, total_dur, job):
    """Remove the silent parts from the audio track, for diarization.

    Returns ``(file, parts)``:

    * ``(wav, [])`` — not cut, everything as before;
    * ``(short.wav, parts)`` — cut, ``parts`` are the speech parts in
      ORIGINAL TIME and are needed afterwards to map back;
    * ``(None, [])`` — the file is silent throughout.
    """
    from whisper_tm.core import diar_silence

    duration = float(total_dur or 0) or get_duration_safe(wav)
    if not duration:
        return wav, []

    t0 = time.time()
    silence = diar_silence.find_silence(wav)
    parts = diar_silence.speech_parts(silence, duration)
    search_s = time.time() - t0

    if not parts:
        return None, []
    if not diar_silence.cut_worth_it(parts, duration):
        logger.debug(
            "Silence cutting not worth it (%.0f%% silence, %d parts, search %.1fs)",
            100 * (1 - sum(b - a for a, b in parts) / duration),
            len(parts),
            search_s,
        )
        return wav, []

    short = Path(wav).with_name("diar_without_silence.wav")
    r = run_cancellable(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-i",
            str(wav),
            "-af",
            diar_silence.select_filter(parts),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-acodec",
            "pcm_s16le",
            str(short),
        ],
        job=job,
        timeout=3600,
        label="Cutting silence for diarization",
    )
    if r.returncode != 0 or not short.exists() or short.stat().st_size < 1000:
        logger.warning(
            "Silence cutting failed (rc=%s) — diarizing the whole file instead", r.returncode
        )
        short.unlink(missing_ok=True)
        return wav, []

    keep = sum(b - a for a, b in parts)
    # The cut must be as long as the parts it keeps: the diarization maps its
    # times back through them. An ffmpeg whose filter kept everything (Ubuntu
    # 26.04's aselect) gave the whole file — with the parts' map, every
    # speaker would have landed at the wrong time.
    got = _wav_seconds(short)
    if got is not None and abs(got - keep) > max(1.0, 0.01 * keep):
        logger.warning(
            "Silence cut is %.1f s long, the speech parts %.1f s — "
            "diarizing the whole file instead",
            got,
            keep,
        )
        short.unlink(missing_ok=True)
        return wav, []
    message = (
        f"Silence skipped: {(duration - keep) / 3600:.1f} h of "
        f"{duration / 3600:.1f} h ({100 * (1 - keep / duration):.0f} %) — "
        f"diarizing {len(parts)} speech parts "
        f"(detection {search_s:.1f}s)"
    )
    job_manager.log(job, message)
    # ALSO into the pipeline log. Previously this
    # line was only in the job log — the most important diagnostic of the
    # function was missing exactly where one looks while monitoring
    # (main.log/pipeline.log). Whether the cut worked could only be
    # inferred indirectly from the diarization duration.
    logger.info("Diarization: %s", message)
    return short, parts


def _wav_seconds(path) -> float | None:
    """Length of a PCM WAV from its header, or None when it can't be read."""
    import wave

    try:
        with wave.open(str(path), "rb") as w:
            return w.getnframes() / float(w.getframerate())
    except (OSError, EOFError, wave.Error):
        return None


def _single_pass_cmd(source, audio_copy, wav) -> list[str]:
    """The single-pass ffmpeg: the source read once, its audio track copied
    (``audio_copy``) and decoded to a 16-kHz WAV (``wav``) at the same time.

    **The video is read along and thrown away** (the third output, ``-f
    null``). Example: a large screen recording (.mov with high-bitrate
    video) read from a NAS at a few MB/s — the step would have taken over
    an hour. Its sound lies between the video frames, a small packet
    every few milliseconds; with the video unused,
    ffmpeg's mov demuxer jumps from one sound packet to the next, and every
    jump is a round trip to the NAS (tens of thousands of small reads for
    a few minutes of film). Reading the video along makes it one sequential
    read — several times faster, even on a busy link, though it reads all
    the bytes.
    Matroska reads every cluster anyway, an audio file has no video (the
    ``?``); the sound written is the same to the byte (test files in
    every common container: mp4, mov, mkv, webm, avi, wmv, flv, 3gp,
    mpg, mp3 with a cover, m4a, ogg, opus, wma, wav).
    """
    return [
        "ffmpeg",
        "-y",
        *media_formats.source(source),
        "-map",
        "0:a:0",
        "-c",
        "copy",
        str(audio_copy),
        "-map",
        "0:a:0",
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        "-rf64",
        "auto",
        str(wav),
        "-map",
        "0:v:0?",
        "-c",
        "copy",
        "-f",
        "null",
        "-",
    ]


def _voices_into_transcript(database, source_path, filename, turns, check=None) -> dict:
    """What a voices-only run found, written into the transcript.

    Example: a recording transcribed without telling speakers apart got
    several voices from
    its voices run — on the map, in "Who was there?" — while its transcript
    still said "—" in every line.

    * **Speakers** only into a transcript that has none: each sentence gets
      the label of the diarization turn it overlaps most
      (speaker.label_segments) — the labels its voices carry, so "only this
      person's sentences" and the map point at the same voice. A transcript
      diarized before keeps its speakers; its voices were aligned to them.
    * **Names** where one is clear (speaker.names_for_labels: the map's,
      else the recognition's at the gallery threshold) — only for speakers
      that have no name yet. A name someone gave is never replaced.
    * **Sentences** the diarization gave to the wrong voice move to the
      right one when ``check`` is given (speaker.recheck_sentences, called
      as ``check(segments, labels)`` with the labels the transcript uses).

    Text, times and everything else stay as they are. Written like a
    transcription writes: the JSON (if there is one) through a temporary
    file, then the database. Returns what was done, for the run's record.
    """
    from whisper_tm.core import speaker as _speaker
    from whisper_tm.core.paths import canon

    tp = (
        find_transcript(filename, abs_path=source_path)
        if config.transcript_mode != "database"
        else None
    )
    if tp:
        with open(tp, encoding="utf-8") as f:
            tdata = json.load(f)
    else:
        tdata = database.get_transcript(str(source_path)) if database else None
    segs = (tdata or {}).get("segments") or []
    if not segs:
        return {"segments": 0, "names": {}, "reason": "no transcript"}

    def _named(sp):
        return bool(sp) and str(sp) not in ("—", "-")

    written = 0
    if turns and not any(_named(x.get("speaker")) for x in segs):
        labels = _speaker.label_segments(
            [(float(x.get("start") or 0), float(x.get("end") or 0)) for x in segs], turns
        )
        for x, label in zip(segs, labels, strict=True):
            x["speaker"] = label
            written += _named(label)
        if written:
            tdata["speakers"] = sorted({lb for lb in labels if _named(lb)})
    present = {str(x.get("speaker")) for x in segs if _named(x.get("speaker"))}
    checked = check(segs, present) if check is not None and present else None
    if checked and checked["moved"]:
        present = {str(x.get("speaker")) for x in segs if _named(x.get("speaker"))}
        tdata["speakers"] = sorted(present)
    names = dict(tdata.get("speaker_names") or {})
    added = {
        label: n
        for label, n in _speaker.names_for_labels(canon(str(source_path))).items()
        if label in present and not names.get(label)
    }
    names.update({label: n["name"] for label, n in added.items()})
    if not written and not added and not (checked and checked["moved"]):
        return {"segments": 0, "names": {}, **({"sentence_check": checked} if checked else {})}
    if names:
        tdata["speaker_names"] = names
    if tp:
        atomic_write_json(tp, tdata)
    if database:
        database.upsert_transcript(
            filename, tdata, media_dir=str(config.media_dir), abs_path=str(source_path)
        )
    return {
        "segments": written,
        "names": {
            label: f"{n['name']} ({n['source']}, {n['windows']} of {n['of']} windows)"
            for label, n in added.items()
        },
        **({"sentence_check": checked} if checked else {}),
    }


def _recheck(check_with, segments, device, job, labels=None) -> dict:
    """The sentence check (speaker.recheck_sentences) on ``segments``.

    ``check_with`` is ``(audio, voices per label)`` from the voice-vector
    step. ``labels``: the labels the transcript uses — a voice it doesn't
    use (a fresh label of a voices-only run) takes no sentences.
    """
    from whisper_tm.core import speaker as _speaker

    audio, centroids = check_with
    if labels is not None:
        centroids = {k: v for k, v in centroids.items() if k in labels}
    if len(centroids) < 2:
        return {"checked": 0, "moved": 0, "moves": [], "reason": "fewer than two voices to compare"}
    return _speaker.recheck_sentences(
        audio, segments, centroids, device=device, job=job, abort=lambda: bool(job.cancel)
    )


def _log_recheck(job, checked: dict) -> None:
    """The sentence check's outcome in the job log."""
    if not checked:
        return
    job_manager.log(
        job,
        f"Sentence check: {checked['moved']} of {checked['checked']} "
        f"sentences moved to another voice"
        + (f" ({checked['secs']:.1f} s)" if "secs" in checked else "")
        + (f" — {checked['reason']}" if checked.get("reason") else ""),
    )
    for m in checked.get("moves", [])[:20]:
        job_manager.log(
            job,
            f"  {format_ts(m['start'])} {m['old']} → {m['new']} ({m['sim']:.2f} vs {m['was']:.2f})",
        )


def _one_pass_error(returncode: int, stderr: str, d_copy: float, d_wav: float) -> Exception:
    """Names WHY the single-pass produced no usable output.

    Originally, the single-pass threw the same message for every
    cause ("check container/codec"). Analyzing the failed entries
    revealed four clearly distinct cases behind it — and because none of
    them was named, none reached the skiplist either; every one kept
    coming back into the queue.

    Returns the exception instead of raising it: this keeps the abort
    visible at the call site, and the cause → message mapping is
    checkable without the 1000-line monster function.
    """
    from whisper_tm.core.audio_utils import ffmpeg_reason, plain_reason

    stderr = (stderr or "").strip()
    # (a system error anywhere in the output first: "No space left on
    # device" used to hide behind the last line, "Conversion failed!")
    last_one = ffmpeg_reason(stderr)
    if _is_corrupt_stderr(stderr):
        # Broken means final — this exception leads to the skiplist.
        return CorruptMediaError(f"File is corrupt: {plain_reason(stderr)}")
    if returncode != 0:
        reason = f"ffmpeg aborted (code {returncode})"
        if last_one:
            reason += f": {last_one}"
    else:
        # ffmpeg was satisfied, the two outputs aren't: typical for
        # containers whose header names a guessed duration (raw ADTS-AAC,
        # and files with absurd values — a short sound effect can report
        # days instead of under a second).
        reason = (
            f"Audio-track copy and WAV have different lengths "
            f"({d_copy:.1f}s vs {d_wav:.1f}s) — the duration in "
            f"the container is not believable"
        )
    return RuntimeError(f"Single-pass provisioning failed: {reason}")


def transcription_worker(
    job_id,
    source_path,
    filename,
    language=None,
    min_speakers=None,
    max_speakers=None,
    diarize=True,
    model_name="turbo",
    vocal_isolation=False,
    vocal_model="bs_roformer",
    save_audio_alongside=False,
    word_timestamps=None,
    export_alongside=None,
    low_vram=None,
    vocab_preset=None,
    separation_only=False,
    speaker_vectors=None,
    batch_size=None,
    voices_only=False,
    refresh_voices=False,
):
    """Main transcription worker. Runs in its own thread.

    ``batch_size``: WhisperX batch for this run; None = hardware
    recommendation. The queue's OOM path (core/queue._oom_retry)
    halves it per attempt — the peak of the WhisperX step (measured
    5.6-6.8 GB at batch 16, large-v3) depends mostly on this.

    Uses the new JobManager for state and logging,
    and the HardwareProfile for device selection.

    ``voices_only``: diarization and voice vectors for a file that already
    has its transcript — the same provisioning, separation, diarization and
    vector code as a transcription, without Whisper. The voices land in
    ``speaker.db``, the transcript's recorded settings get a ``voices``
    entry, and a transcript without speakers gets the voices' labels and
    the names that are clear (_voices_into_transcript) — its text never
    changes.

    ``refresh_voices`` (with ``voices_only``): the file's voices are made
    anew, and the names, trash and history of the old ones go to the new
    voices that are the same people (speaker.carry_along, keep_decisions).
    Where that isn't clear for every decided voice, the file keeps its old
    voices — the run says so in its log and under ``voices.renewal``.

    ``separation_only``: run without transcription — provision the file, get
    audio, separate the vocal track, place MP3s next to the source file,
    done. Comes from the transcription dialog ("no transcription"). There
    is deliberately no second worker for this: the first three phases are
    identical, and a separate path would mean maintaining the same
    separation logic in two places.

    ``low_vram``: the entry's choice (None = setting hardware.low_vram, see
    core/vram.py). True unloads the resident models of the prior run
    BEFORE this run and caches nothing during it — for runs that would
    otherwise fail on VRAM (e.g. large-v3 resident + separator +
    Japanese alignment → CUDA OOM on a 12 GB card).
    """
    import torch

    whisperx = _whisperx("asr", "alignment", "audio", "diarize")

    from whisper_tm.core.vram import low_vram as _lv
    from whisper_tm.core.vram import set_low_vram_override

    job = job_manager.get(job_id)
    if not job:
        return

    try:
        job.abs_path = str(source_path)  # for the incidents table
    except Exception:
        pass
    set_low_vram_override(low_vram if isinstance(low_vram, bool) else None)
    if _lv():
        # Free existing residency — a low-VRAM run loading ALONGSIDE the
        # cached models would be worse than none.
        try:
            from whisper_tm.core.vocal_isolation import release_separator

            release_models()
            release_separator()
            job_manager.log(
                job, "🧠 Saving VRAM: unloading resident models, no caching in this run"
            )
        except Exception as e:
            logger.debug("Low-VRAM preparation: %s", e)

    job.start_time = time.time()
    hf_token = config.hf_token

    # WhisperX/ctranslate2 only accepts "cuda" or "cpu" (no "cuda:0").
    # For multi-GPU, the GPU index is passed cleanly to the model via
    # device_index (ctranslate2/whisperx.load_model). NO runtime mutation
    # of CUDA_VISIBLE_DEVICES anymore: it has no effect after the torch
    # import and, as a process-global env var, would be a race between
    # job threads.
    device = get_whisperx_device(hw) if hw else "cuda"
    device_index = hw.best_gpu_index if (hw and hw.best_device == "cuda") else 0

    compute_type = config.get("hardware", "compute_type", default="auto")
    if compute_type == "auto" and hw:
        compute_type = hw.recommended_compute_type

    # Whisper transcribes AS ONE PIECE.
    # An earlier 600-s split cut sentences off mid-word at every block
    # boundary (affecting many long transcripts) and gained neither
    # quality nor meaningful memory savings (+413 MB from 5 to
    # 60 min). Only EXTREMELY long files (audio array ~0.23 GB/h RAM) run
    # in huge blocks WITH overlap and midpoint dedup — lossless at the
    # seams.
    asr_block_sec = config.get("hardware", "asr_block_max_sec", default=54000)
    asr_overlap_sec = 30.0
    # Per-job override or config default
    if word_timestamps is None:
        word_timestamps = config.get("transcription", "word_timestamps", default=False)
    if export_alongside is None:
        export_alongside = config.get("transcription", "export_alongside", default=False)
    # Voice vectors: only useful if diarizing — without speaker segments
    # there is nothing to cut. Hence no separate condition further below;
    # it's already tied to `diarize` here.
    _vectors_on = voices_only or (
        bool(speaker_vectors)
        if speaker_vectors is not None
        else bool(config.get("transcription", "speaker_vectors", default=True))
    )
    # Only becomes true once separation has actually taken effect — not
    # merely because it was requested.
    _audio_separated = False
    # Without vocal isolation the vectors would be the RAW version and thus
    # not comparable with the others (for the vector, separation is
    # proven: 100% vs 89% hit rate). Instead of just warning, the
    # 30-s WINDOWS are then separated individually — per window and per
    # track practically the same vector results (0.984/0.994), it's only
    # a matter of time, and at 30 seconds it comes out differently than
    # at seven hours.
    _window_separation = bool(
        config.get("transcription", "speaker_window_separation", default=True)
    )
    # Each sentence measured against the recording's voices after the
    # diarization (speaker.recheck_sentences) — only where the whole audio
    # is separated: the voices are, and a raw sentence doesn't compare.
    _sentence_check = bool(config.get("speaker", "check_sentences", default=True))
    # (audio, voices) for the check, held until the sentences are final
    _check_with = None
    # Whether the recording's folder takes the transcript file ("alongside")
    _beside_ok = True
    # What the check did — part of the recorded settings
    _checked: dict = {}
    # What the voice-vector step reported — part of the recorded settings.
    _voice: dict = {}
    # Voices-only: how the fresh labels were matched to the transcript's
    _voice_labels: dict = {}
    # Voices-only: the turns under the labels the voices were stored with —
    # what the transcript's sentences get (_voices_into_transcript)
    _label_turns: list = []
    dirs = config.dirs
    uploads_dir = dirs["uploads"]
    # The folder takes the end of the job ID (its hash included): the ID
    # holds the whole relative path, and on Windows a deep one under the
    # default work folder plus the separator's subfolders went past the
    # 260 characters a path may have there
    work = dirs["work"] / job_id[-48:]
    work.mkdir(parents=True, exist_ok=True)

    job_manager.log(job, f"Job started: {filename}")
    job_manager.log(job, f"Device: {device}, Compute: {compute_type}")
    job_manager.log(job, f"Workdir: {work}")

    if vocal_isolation:
        job_manager.log(job, f"Vocal Isolation: {vocal_model}")

    try:
        t_start = time.time()

        def elapsed():
            return format_duration(time.time() - t_start)

        _step_t = [time.time()]
        # A wait from before this run (a low-VRAM run waiting for the other
        # lanes to let go, or what the previous file on this lane left
        # behind) is no step of it.
        _before = GATE.take_wait()
        if _before >= 1.0:
            job_manager.log(
                job,
                f"⏳ Waited {_before:.1f} s for the GPU before starting (other files in flight)",
            )

        # Structure phases in the log: header line at the start ("now doing
        # X with file Y"), closing line with duration at the end. Without
        # this the log is a flat wasteland of lines where you can't find
        # where a phase begins.
        #
        # The total count is computed from the ACTUAL options, not
        # hardwired — otherwise the counter would show jumps like
        # [3/9] → [5/9] when vocal isolation is off, making it worthless.
        # Check ONCE, then use the same answer everywhere: the cache test
        # costs a filesystem access, and three different answers within one
        # run would be a bug waiting to happen.
        _diar_possible = bool(diarize) and diarization_possible(hf_token)
        if diarize and not _diar_possible:
            job_manager.warn(
                job,
                "Diarization requested but not possible: no "
                "Hugging Face token set AND the model "
                f"{DIARIZATION_REPO} is not fully in the cache. "
                "The transcript will be produced WITHOUT speakers. "
                f"Once: accept the conditions at {hf_access.REPO_URL}, "
                "create a Read token and enter it under Settings → "
                "Speakers (or run whisper-tm --hf-token).",
            )
        if voices_only and not _diar_possible:
            # Voices come from diarization; without it this job has nothing
            # to do — failing says so, a silent "done" would not.
            raise RuntimeError(
                "Voices need diarization, which is not "
                "possible here (Hugging Face token / model cache)"
            )
        _phase_n = [0]
        if separation_only:
            # check, audio, vocal isolation — the run ends after that.
            _PHASES = 3
        elif voices_only:
            # check, audio, [vocal isolation], diarization — no Whisper.
            _PHASES = 3 + (1 if vocal_isolation else 0)
        else:
            _PHASES = (
                6  # always: check, audio,
                + (1 if vocal_isolation else 0)  # splitting, Whisper,
                + (1 if _diar_possible else 0)
            )  # alignment, saving

        def phase(title, **fields):
            _phase_n[0] += 1
            # For the graphical progress map in the frontend: which phase
            # is currently running, since when (live timer), how many total.
            try:
                job.phase = {
                    "index": _phase_n[0],
                    "total": _PHASES,
                    "title": title,
                    "start": time.time(),
                }
            except Exception:
                pass
            debug.section(title, subject=filename, step=_phase_n[0], total=_PHASES, **fields)

        def step_done(label, group=None, vram=None):
            now = time.time()
            # Without the time spent waiting for a model another queue lane
            # was using: the step stats compare work, and the wait says
            # nothing about this file.
            waited = GATE.take_wait()
            dt = max(0.0, now - _step_t[0] - waited)
            _step_t[0] = now
            if waited >= 1.0:
                job_manager.log(
                    job, f"⏳ {label}: waited {waited:.1f} s for the GPU (other files in flight)"
                )
            job_manager.record_step(job, label, dt, group=group)
            # Attach this section's VRAM peak (only filled in debug mode) to
            # the job log and the closing line — so it shows right next to
            # the duration how much GPU memory the step cost.
            note = vram.note() if vram is not None else ""
            if note:
                job_manager.log(job, f"📊 VRAM {note}")
            debug.section_done(label, dt, note=note)

        def load_done(label, secs):
            """Record pure model loading time separately.

            Deliberately WITHOUT group: the frontend builds its phase
            timeline from the ungrouped top-level steps (matched by label)
            — the loading time should appear as a substep of its phase
            (transcription/alignment/diarization), not vanish into its own
            group.

            Resets the phase clock (_step_t) so that the subsequent
            step_done() only measures INFERENCE and doesn't lump
            loading+inference together — this way you can see exactly how
            long loading took.
            """
            job_manager.record_step(job, label, secs)
            debug.section_done(label, secs)
            _step_t[0] = time.time()
            # A wait before the load lies before the new clock start.
            waited = GATE.take_wait()
            if waited >= 1.0:
                job_manager.log(
                    job, f"⏳ {label}: waited {waited:.1f} s for the GPU (other files in flight)"
                )

        # The header names what this run does: a voices or separation run
        # that announces "TRANSCRIPTION · Model: large-v3" sends whoever
        # reads the log looking for a Whisper step that never happens.
        _whisper = not (voices_only or separation_only)
        debug.section(
            ("VOICES" if voices_only else "SEPARATION" if separation_only else "TRANSCRIPTION"),
            subject=filename,
            **({"Model": model_name, "Language": language} if _whisper else {}),
            Device=device,
            Diarization=("yes" if _diar_possible else "NOT POSSIBLE" if diarize else "no"),
            **{"Vocal Isolation": vocal_model if vocal_isolation else "no"},
        )

        # ── Check whether the file is being written to ──────────────────
        phase("Check & provision file")
        job.update(status="processing", progress=1, message="Checking file…")
        job_manager.log(job, f"Checking: {source_path} ({format_size(source_path.stat().st_size)})")

        # Only uploads can still be actively written to during processing
        # — secure that with a 2s stability check there. Stable library
        # files don't need it (saves 2s per job).
        try:
            is_upload = source_path.resolve().is_relative_to(uploads_dir.resolve())
        except OSError:
            is_upload = True  # play it safe when unsure
        try:
            size1 = source_path.stat().st_size
            if is_upload:
                time.sleep(2)
                size2 = source_path.stat().st_size
                if size1 != size2:
                    raise RuntimeError(
                        f"File is being actively written to ({format_size(size2 - size1)}/2s)"
                    )
        except OSError as e:
            raise RuntimeError(f"File not readable: {e}") from e

        # ── Copy / symlink ────────────────────────────────
        file_size = source_path.stat().st_size
        work_file = work / Path(filename).name

        # Check whether source and work dir are on the same filesystem
        same_fs = False
        try:
            same_fs = source_path.stat().st_dev == work.stat().st_dev
        except OSError:
            pass

        if same_fs:
            # Same filesystem: symlink instead of copy (saves I/O + time)
            #
            # There used to be an additional condition here:
            # "...and the source is not in uploads/". The reason for that
            # is LONG GONE: the `finally` block used to delete the upload
            # source after the run back then, and a symlink would have
            # pointed into thin air. Since a later rework, uploads/
            # is part of the library and no longer gets deleted (see the
            # comment below in `finally`).
            #
            # The exception stayed in place and thereby became a bug: it
            # prevented the symlink but left `same_fs` at True — and the
            # single-pass provisioning below only kicks in for
            # `not same_fs`. Result: for an uploaded file NO work file
            # whatsoever was created, and the next step failed with
            # "file not readable (ffprobe code 1) … No such file or
            # directory". Any installation where the library and work
            # directory are on the same drive is affected — i.e. the
            # normal case without a NAS. Found while testing the MCP
            # server, which follows exactly this path: upload, transcribe.
            try:
                work_file.symlink_to(source_path)
                job.update(progress=5, message=f"Symlink created · {elapsed()}")
                job_manager.log(job, f"Symlink instead of copy (same FS, {format_size(file_size)})")
                step_done("File linked")
            except OSError:
                # Fallback: copy after all (e.g. on permission issues)
                same_fs = False

        wav = work / "audio.wav"
        combo_ok = False
        if not same_fs and not work_file.exists():
            # Single-pass provisioning: ffmpeg reads the NAS source ONCE and
            # simultaneously writes the work copy and the 16-kHz WAV. The
            # work copy is ONLY THE AUDIO TRACK (stream copy into .mka):
            # the pipeline never needs the video, and for
            # a 25-GB recording over 500 Mbit that saves minutes per file.
            # Extracting directly from the NAS AND later pulling the split
            # chunks from the NAS as well would instead double the read
            # volume.
            audio_copy = work / "source_audio.mka"
            # First check whether there is an audio track at all. The slow
            # path (extract_audio) has always done this, the single-pass
            # path did not — and `-map 0:a:0` fails without an audio track
            # with EINVAL, which landed in the error filter as a
            # meaningless "check container/codec" instead of on the
            # skiplist. An analysis of failed entries: most errors were
            # exactly that (videos without audio, plus photos named .MOV),
            # many others were 0 bytes in size, and a few were broken
            # containers. An ffprobe only reads the headers; until then the
            # single-pass read the entire file from the NAS only to then
            # reliably fail.
            #
            # has_audio_stream separates the three cases by itself: broken
            # files and 0 bytes ("end of file") throw CorruptMediaError, a
            # transient ffprobe error throws RuntimeError instead — only
            # the first two may go permanently onto the skiplist.
            #
            # The check deliberately CONTINUES WHEN IN DOUBT. It sits on
            # the NAS path, where it didn't exist before, and its ffprobe
            # gets 30 s while the single-pass itself gets 14400 s. A CIFS
            # hiccup must therefore never fail a job that would have
            # succeeded before: it may prevent a hopeless run, but must
            # never be an obstacle that wasn't there before. Only a clear
            # "no audio stream" and CorruptMediaError stop it.
            try:
                has_audio_track = has_audio_stream(source_path)
            except CorruptMediaError:
                raise
            except Exception as e:
                has_audio_track = True
                job_manager.log(
                    job,
                    f"Audio track pre-check inconclusive ({e}) — "
                    f"proceeding with single-pass anyway",
                )
            if not has_audio_track:
                raise RuntimeError("No audio track present – file skipped")
            job.update(
                progress=2, message=f"Copying + extracting audio track (1 pass)… · {elapsed()}"
            )
            job_manager.log(
                job,
                "Single-pass: audio-track copy (.mka, stream copy) + 16-kHz WAV in one NAS pass",
            )
            r = run_cancellable(
                _single_pass_cmd(source_path, audio_copy, wav),
                job=job,
                timeout=14400,
                label="Single-pass: audio-track copy + 16k",
            )
            # Plausibility via DURATION (the audio track is naturally much
            # smaller than the file): both outputs must have the same length.
            d_copy = get_duration_safe(audio_copy) if audio_copy.exists() else 0
            d_wav = get_duration_safe(wav) if wav.exists() else 0
            combo_ok = (
                r.returncode == 0
                and wav.exists()
                and wav.stat().st_size > 1000
                and d_copy > 0
                and d_wav > 0
                and abs(d_copy - d_wav) <= max(2.0, 0.01 * d_wav)
            )
            if combo_ok:
                work_file = audio_copy
                job_manager.log(
                    job,
                    f"Audio track: {format_size(audio_copy.stat().st_size)} "
                    f"instead of {format_size(file_size)} copied",
                )
                step_done("Audio track copied + audio extracted (1 pass)")
            else:
                # No more silent fallback to copy + extraction: the
                # single-pass ran without a single failure in log
                # analysis, and duplicate code paths are to be avoided. If it does fail — exotic
                # container/codec — the job should fail visibly and the
                # file should land in the ✕ filter, not silently trigger a
                # second NAS pass.
                audio_copy.unlink(missing_ok=True)
                wav.unlink(missing_ok=True)
                raise _one_pass_error(r.returncode, getattr(r, "stderr", ""), d_copy, d_wav)

        # ── Extract audio ────────────────────────────────
        phase("Audio extraction", target="16 kHz mono WAV")
        if job.cancel:
            raise InterruptedError("Cancelled")
        if not combo_ok:
            job.update(progress=5, message=f"Extracting audio… · {elapsed()}")
            job_manager.log(job, "FFmpeg: audio extraction (16kHz mono)")
            extract_audio(work_file, wav, job=job)

        if not wav.exists():
            raise RuntimeError("Audio extraction failed")

        total_dur = get_duration(wav)
        wav_size = wav.stat().st_size
        job.update(
            progress=8,
            message=f"Audio: {format_ts(total_dur)} · {format_size(wav_size)} · {elapsed()}",
        )
        step_done("Audio extraction")
        job_manager.log(job, f"Audio: {format_ts(total_dur)}, {format_size(wav_size)}")

        # A library folder that can't be written ("alongside" mode, a
        # read-only share): the transcript is kept in the database — said
        # now, not after the GPU work, which used to be lost at the very end
        if config.transcript_mode == "alongside" and not (voices_only or separation_only):
            _beside_ok = _folder_writable(
                transcript_path_for(filename, abs_path=source_path).parent
            )
            if not _beside_ok:
                job_manager.warn(
                    job,
                    "The recording's folder can't be written (read-only?) — "
                    "the transcript is kept in the database only",
                )

        # Fingerprints, waveform and loudness (whisper_tm/audio) from this
        # WAV — the original mix, before a separation replaces it — on a
        # thread of their own while the GPU works: the recording is not
        # read again for them
        try:
            from whisper_tm.audio.runner import after_decoding

            after_decoding(wav, source_path, log=lambda m: job_manager.log(job, m))
        except Exception as e:  # noqa: BLE001 — never worth a transcript
            logger.debug("Audio analysis not started: %s", e)

        # Only delete work_file if the separator no longer needs it
        if not vocal_isolation:
            work_file.unlink(missing_ok=True)

        # ── Vocal Isolation (optional) ───────────────────────
        if vocal_isolation:
            phase("Vocal Isolation", Model=vocal_model)
            from whisper_tm.core.vocal_isolation import isolate_vocals

            if not (hw and hw.has_audio_separator):
                raise RuntimeError("audio-separator not installed!")
            if job.cancel:
                raise InterruptedError("Cancelled")

            job.update(progress=8, message=f"🎤 Vocal Isolation… · {elapsed()}")

            # work_file DIRECTLY to separator (no second audio extract!)
            # audio-separator handles FFmpeg internally
            # IMPORTANT: resolve symlinks! audio-separator can have issues with symlinks
            iso_source = work_file if work_file.exists() else source_path
            if iso_source.is_symlink():
                iso_source = iso_source.resolve()
                job_manager.log(job, f"Symlink resolved: {iso_source}")

            from whisper_tm.core.vocal_isolation import SeparationUnavailable

            skipped = False
            try:
                vocals_path, inst_path = isolate_vocals(
                    iso_source,
                    work,
                    job=job,
                    elapsed_fn=elapsed,
                    model_key=vocal_model,
                    # Kept next to the file or the whole result: in full quality
                    full_vocals=bool(save_audio_alongside or separation_only),
                )
            except SeparationUnavailable as e:
                # Offline before the model's first download, say: every
                # file failed for it. Transcribed without the separation —
                # the recorded settings say "not separated" — and said so.
                if separation_only:
                    raise
                _separation_unavailable(job_manager, job, e)
                vocals_path = inst_path = None
                skipped = True
            work_file.unlink(missing_ok=True)

            if vocals_path and vocals_path.exists():
                # Placed BEFORE the 16-kHz conversion: dropping the file
                # next to the source is the only result of a pure separation
                # run, while the conversion is only food for WhisperX.
                if save_audio_alongside:
                    _save_audio_alongside(source_path, vocals_path, inst_path, job)

                if separation_only:
                    step_done("Vocal Isolation")
                    for d in [work / "vocal_iso_work", work / "pass2_work"]:
                        if d.exists():
                            shutil.rmtree(d, ignore_errors=True)
                    _secs = round(time.time() - t_start, 1)
                    # Before `complete()` — after that the log is written
                    # and any further line is lost (see below).
                    job_manager.log(
                        job, f"✅ Separated, no transcript requested ({format_duration(_secs)})"
                    )
                    job_manager.complete(
                        job,
                        {
                            "filename": filename,
                            "vocal_isolation": True,
                            "vocal_model": vocal_model,
                            "separation_only": True,
                            "processing_time": format_duration(_secs),
                            "processing_secs": _secs,
                            "step_timings": [
                                {
                                    "label": s["label"],
                                    "secs": s["secs"],
                                    **({"group": s["group"]} if "group" in s else {}),
                                }
                                for s in job.step_timings
                            ],
                            "warning_count": job.warning_count,
                            "warnings": job.warnings,
                        },
                    )
                    return

                # Vocals → 16kHz mono WAV for WhisperX
                wav_clean = work / "audio_clean.wav"
                # -rf64 auto: see audio_utils.extract_audio — 4-GiB limit
                # of the WAV format, here after 37.3 h at 16 kHz mono.
                _r = run_cancellable(
                    [
                        "ffmpeg",
                        "-y",
                        "-i",
                        str(vocals_path),
                        "-ar",
                        "16000",
                        "-ac",
                        "1",
                        "-acodec",
                        "pcm_s16le",
                        "-rf64",
                        "auto",
                        str(wav_clean),
                    ],
                    job=job,
                    timeout=3600,
                )
                if _r.returncode != 0:
                    from whisper_tm.core.audio_utils import ffmpeg_failure

                    job_manager.warn(
                        job,
                        f"FFmpeg (Vocals→16kHz) failed "
                        f"(code {_r.returncode}): {ffmpeg_failure(_r.stderr)}",
                    )

                if wav_clean.exists() and wav_clean.stat().st_size > 1000:
                    wav.unlink()
                    wav_clean.rename(wav)
                    # Only HERE has the audio actually been separated. The
                    # voice vectors record this version too (v_raw vs
                    # v_separated); if this were derived from the REQUEST
                    # instead of the RESULT, a failed separation run would
                    # silently carry its raw vectors into the database as
                    # separated ones — and nobody could tell afterward.
                    _audio_separated = True
                    step_done("Vocal Isolation")
                    job_manager.log(job, f"Vocals → 16kHz: {format_size(wav.stat().st_size)}")
                else:
                    job_manager.warn(job, "Vocal Isolation failed, using original")
            elif separation_only and getattr(job, "separation_silent", False):
                # Nothing but silence in the recording: there was nothing to
                # separate — no tracks, and no error either (a failed run
                # would come back on every retry).
                _secs = round(time.time() - t_start, 1)
                job_manager.warn(
                    job, "The recording is silent throughout — nothing to separate, no tracks saved"
                )
                job_manager.complete(
                    job,
                    {
                        "filename": filename,
                        "vocal_isolation": True,
                        "vocal_model": vocal_model,
                        "separation_only": True,
                        "silent": True,
                        "processing_time": format_duration(_secs),
                        "processing_secs": _secs,
                        "step_timings": [
                            {
                                "label": s["label"],
                                "secs": s["secs"],
                                **({"group": s["group"]} if "group" in s else {}),
                            }
                            for s in job.step_timings
                        ],
                        "warning_count": job.warning_count,
                        "warnings": job.warnings,
                    },
                )
                return
            elif separation_only:
                # Without transcription, separation is the whole job —
                # "just use the original" would be a silent no-op here.
                raise RuntimeError("Separation produced no vocals track")
            elif not skipped:
                job_manager.warn(job, "No vocals found, using original")

            for d in [work / "vocal_iso_work", work / "pass2_work"]:
                if d.exists():
                    shutil.rmtree(d, ignore_errors=True)

            total_dur = get_duration(wav)

        # ── Gain boost for very quiet recordings ──────────────────────
        # Runs AFTER vocal isolation on the final ASR audio (the separation
        # models are invariant to pure gain — measured). Whisper is
        # level-invariant down to ~-40 dB — only below that (typically:
        # quiet voice notes) does a PURE gain help measurably. Deliberately
        # NO compression, no loudnorm: both raise the noise floor during
        # speech pauses and feed exactly the passages where Whisper
        # hallucinates.
        gain_below_db = float(config.get("transcription", "gain_below_db", default=-40.0))
        try:
            _m = None
            if gain_below_db < 0:  # >= 0 means: gain off
                _vd = run_cancellable(
                    [
                        "ffmpeg",
                        "-i",
                        str(wav),
                        "-af",
                        "volumedetect",
                        "-f",
                        "null",
                        "-",
                    ],
                    job=job,
                    timeout=1800,
                    label="Level measurement",
                )
                _m = re.search(r"mean_volume:\s*(-?[\d.]+) dB", _vd.stderr or "")
            if _m and float(_m.group(1)) < gain_below_db:
                _mean_db = float(_m.group(1))
                _gain = min(-26.0 - _mean_db, 30.0)  # target ~-26 dB, capped
                _target = work / "audio_gain.wav"
                _g = run_cancellable(
                    [
                        "ffmpeg",
                        "-y",
                        "-i",
                        str(wav),
                        "-af",
                        f"volume={_gain:.1f}dB",
                        "-acodec",
                        "pcm_s16le",
                        "-rf64",
                        "auto",
                        str(_target),
                    ],
                    job=job,
                    timeout=3600,
                    label="Gain (quiet recording)",
                )
                if _g.returncode == 0 and _target.exists() and _target.stat().st_size > 1000:
                    wav.unlink()
                    _target.rename(wav)
                    job_manager.log(
                        job,
                        f"Very quiet recording "
                        f"({_mean_db:.1f} dB) — pure gain "
                        f"+{_gain:.1f} dB applied",
                    )
        except Exception as _e:
            job_manager.warn(job, f"Level check skipped: {str(_e)[:100]}")

        if voices_only:
            # Voices only (queue job "voices"): everything up to here ran
            # exactly as for a transcription — provisioning, separation,
            # gain — and diarization and vectors below run the same way
            # again. Only Whisper and alignment are skipped; of the
            # transcript only speakers and names can change, never its text
            # (_voices_into_transcript). One path instead of a second
            # worker, so both cannot drift apart (the old backfill tool
            # separated 16-kHz audio and diarized without the silence cut).
            nc, _batch, vocab_prompt = 0, None, ""
            det_lang = language or ""
            aligned = []
        else:
            # ── Split audio into huge blocks if needed (extreme cases only) ──────
            phase("Audio splitting")
            if job.cancel:
                raise InterruptedError("Cancelled")
            job.update(progress=12, message=f"Preparing audio… · {elapsed()}")

            # (block_path, log_start, ext_start, log_end):
            #   log_*  = logical window (half-open, decides via segment
            #            MIDPOINT which block keeps a boundary segment)
            #   ext_*  = actually extracted audio (± overlap) — speech spanning
            #            the block boundary thus ALWAYS lies fully within one
            #            of the blocks; nothing gets cut mid-word anymore.
            chunks = _asr_blocks(wav, work, total_dur, asr_block_sec, asr_overlap_sec, job=job)
            nc = len(chunks)
            step_done("Audio splitting")
            job_manager.log(
                job,
                "Transcribing as one piece (no chunking)"
                if nc == 1
                else f"{nc} ASR blocks of {asr_block_sec}s (overlapped)",
            )

            # ── WhisperX transcription ───────────────────────────
            _batch = int(batch_size) if batch_size else (hw.recommended_batch_size if hw else 16)
            phase(
                "Speech recognition (WhisperX)",
                Model=model_name,
                Language=language,
                Device=device,
                Batch=_batch,
            )
            if job.cancel:
                raise InterruptedError("Cancelled")
            job.update(progress=15, message=f"Loading model: {model_name}… · {elapsed()}")
            job_manager.log(
                job,
                f"Loading WhisperX model: {model_name} on {device} "
                f"(GPU index {device_index}, compute={compute_type})",
            )

            # Vocabulary preset → initial_prompt: whisperx applies it to EVERY
            # batch window. It's part of asr_options and thus BAKED INTO the
            # model — hence it belongs in the residency key (a different
            # preset = a different model object, reload ~3.5s).
            from whisper_tm.core.vocabulary import (
                WHISPER_PROMPT_BUDGET,
                estimate_tokens,
                preset_prompt,
            )

            vocab_prompt = preset_prompt(vocab_preset)
            # Repetition brakes — ONLY when vocabulary is active. WhisperX's
            # batch path passes on only these two of all Whisper safety gates
            # to generate(); temperature fallback, compression_ratio_threshold,
            # log_prob_threshold and no_speech_threshold are in its defaults
            # but are read nowhere (checked whisperx/asr.py:61-72 against
            # :376-399). Without them there is nothing to catch a runaway
            # repetition — in Whisper itself, the gzip rate of a five-times
            # repeated word list would long since have forced a new attempt.
            # Only with vocabulary, because otherwise they could also clip
            # genuine repetitions in spoken language.
            _asr_extra = {}
            if vocab_prompt:
                _ngram = int(config.get("transcription", "vocab_no_repeat_ngram", default=5) or 0)
                _rep = float(
                    config.get("transcription", "vocab_repetition_penalty", default=1.1) or 1.0
                )
                if _ngram > 0:
                    _asr_extra["no_repeat_ngram_size"] = _ngram
                if _rep > 1.0:
                    _asr_extra["repetition_penalty"] = _rep
            if vocab_prompt:
                _vt = estimate_tokens(vocab_prompt)
                job_manager.log(
                    job,
                    f"📖 Vocabulary: {vocab_preset} "
                    f"(~{_vt}/{WHISPER_PROMPT_BUDGET} tokens)"
                    + (f", repetition brake {_asr_extra}" if _asr_extra else ""),
                )
                if _vt > WHISPER_PROMPT_BUDGET:
                    job_manager.warn(
                        job,
                        f"Vocabulary list over the Whisper "
                        f"budget (~{_vt} tokens) — only its "
                        f"tail takes effect",
                    )

            # VRAM context spans loading AND inference — the peak usage mostly
            # occurs only during inference (batch_size). WhisperX runs over
            # ctranslate2, so only the TOTAL GPU usage (debug.vram) measures
            # the real demand, not torch.cuda.*_allocated.
            with GATE.stage("asr", job=job), debug.vram(f"WhisperX {model_name}") as _vr_whisper:
                _t_load = time.time()
                try:
                    model = _asr_pipeline(
                        model_name,
                        device,
                        device_index,
                        compute_type,
                        language,
                        vocab_prompt,
                        _asr_extra,
                        job=job,
                    )
                except Exception as e:
                    if "model.bin" in str(e) or "Unable to open" in str(e):
                        raise RuntimeError(
                            f"Model '{model_name}' is not in CTranslate2 format. "
                            f"WhisperX requires CT2-converted models (with model.bin). "
                            f"Please choose a compatible model."
                        ) from e
                    raise
                load_done("Model loaded", time.time() - _t_load)

                all_segments = []
                det_lang = language
                for ci, (chunk_path, log_start, ext_start, log_end) in enumerate(chunks):
                    if job.cancel:
                        del model
                        gc.collect()
                        torch.cuda.empty_cache()
                        raise InterruptedError("Cancelled")

                    pct = 18 + int(ci / nc * 30)
                    job.update(
                        progress=pct,
                        message=(
                            f"Transcribing… · {elapsed()}"
                            if nc == 1
                            else f"Transcribing block {ci + 1}/{nc}… · {elapsed()}"
                        ),
                    )

                    audio = whisperx.load_audio(str(chunk_path))
                    result = model.transcribe(audio, batch_size=_batch)

                    if ci == 0 and result.get("language"):
                        det_lang = result["language"]
                        job_manager.log(job, f"Language detected: {det_lang}")

                    keep = []
                    for seg in result.get("segments", []):
                        seg["start"] += ext_start
                        seg["end"] += ext_start
                        # Midpoint dedup at block boundaries: the overlap lets
                        # boundary segments form in FULL in BOTH blocks;
                        # a segment is only kept from the block within whose
                        # logical window its midpoint falls.
                        middle = (seg["start"] + seg["end"]) / 2
                        if log_start <= middle < log_end:
                            keep.append(seg)
                    all_segments.extend(keep)
                    if nc > 1:
                        job_manager.log(job, f"Block {ci + 1}/{nc}: {len(keep)} segments")

                    # #010: public accessor instead of direct access to the
                    # private job_manager._lock (appends thread-safely under the lock).
                    job_manager.add_live_segments(job, [fmt_seg(s) for s in keep])

            del model
            release_held()  # the Whisper weights may be replaced again
            gc.collect()
            torch.cuda.empty_cache()
            step_done("Transcription", vram=_vr_whisper)

            # ── Alignment ────────────────────────────────────────
            phase("Alignment (word timestamps)")
            # IMPORTANT: alignment is OPTIONAL. For languages without a
            # wav2vec2 align model (many rare languages) or when no language
            # was detected, the job must NOT die — the transcription in
            # all_segments is already valid (with offsets). If alignment
            # fails, we use the unaligned segments (without word timestamps).
            if job.cancel:
                raise InterruptedError("Cancelled")
            job.update(progress=52, message=f"Alignment… · {elapsed()}")

            # Speed lever: alignment provides WORD timestamps.
            # These are only needed by word-level timestamps and by
            # diarization (word→speaker assignment). If both are off — the
            # new default —, alignment costs ~18% of job time for nothing but
            # minimally finer segment boundaries; the Whisper timestamps
            # suffice for search. So skip it.
            align_needed = bool(word_timestamps or _diar_possible)

            model_a = meta = None
            align_ran = False  # only True once whisperx.align actually ran
            align_lang = det_lang if det_lang and det_lang != "auto" else None
            if not align_needed:
                align_lang = None
                job_manager.log(
                    job,
                    "Alignment skipped — neither "
                    "word timestamps nor diarization active "
                    "(saves ~18% job time)",
                )
            with (
                GATE.stage("alignment", job=job),
                debug.vram(f"Alignment {align_lang or '—'}") as _vr_align,
            ):
                if align_lang is None:
                    if align_needed:
                        # Only THEN is this worth a warning — the user wanted
                        # word timestamps/diarization but gets no words.
                        job_manager.warn(
                            job, "No language detected for alignment — skipping word alignment"
                        )
                else:
                    job_manager.log(job, f"Alignment for: {align_lang}")
                    _t_align_load = time.time()
                    try:
                        model_a, meta = _resident_model(
                            ("align", align_lang, device),
                            lambda: whisperx.load_align_model(
                                language_code=align_lang, device=device
                            ),
                            job=job,
                            label=f"Alignment {align_lang}",
                        )
                        load_done("Alignment model loaded", time.time() - _t_align_load)
                    except Exception as e:
                        # A full card is not a missing model: with queue
                        # lanes another file's step can fill it while this
                        # one loads. Saved without word times the run would
                        # count as done — the queue's OOM retry is the way.
                        if "out of memory" in str(e).lower():
                            raise
                        model_a = meta = None
                        from whisper_tm.core import prefetch

                        if prefetch.is_download_failure(e):
                            job_manager.warn(
                                job,
                                f"The alignment model for '{align_lang}' "
                                f"could not be downloaded — {prefetch.hint(e)} "
                                f"— using segment timestamps",
                            )
                        else:
                            job_manager.warn(
                                job,
                                f"No alignment model for '{align_lang}' "
                                f"({str(e)[:80]}) — using segment timestamps",
                            )
                    if model_a is not None:
                        # whisperx.align loads NLTK's sentence splitter and
                        # downloads it when missing — offline every block
                        # then failed with "Resource punkt_tab not found",
                        # and the whole file with it
                        from whisper_tm.core import prefetch

                        try:
                            prefetch.punkt()
                        except Exception as e:  # noqa: BLE001
                            del model_a, meta
                            model_a = meta = None
                            release_held()
                            gc.collect()
                            job_manager.warn(
                                job,
                                "NLTK's sentence splitter (punkt_tab) could not be "
                                f"downloaded from {prefetch.PUNKT_SOURCE} — "
                                f"{prefetch.hint(e)} — using segment timestamps",
                            )

                if model_a is None:
                    # Fallback: unaligned segments (text + segment timestamps
                    # correct, just no word timestamps). Job continues normally.
                    aligned = [dict(s) for s in all_segments]
                else:
                    aligned = []
                    for ci, (chunk_path, log_start, ext_start, log_end) in enumerate(chunks):
                        if job.cancel:
                            del model_a, meta
                            gc.collect()
                            torch.cuda.empty_cache()
                            raise InterruptedError("Cancelled")
                        pct = 55 + int(ci / nc * 15)
                        job.update(
                            progress=pct,
                            message=(
                                f"Alignment… · {elapsed()}"
                                if nc == 1
                                else f"Alignment block {ci + 1}/{nc}… · {elapsed()}"
                            ),
                        )
                        # Segments were already assigned unambiguously to a
                        # block during transcription via their midpoint — here
                        # the same half-open criterion, so each segment ends up
                        # in EXACTLY one alignment run. Thanks to overlap its
                        # audio lies fully within the extended block.
                        chunk_segs = [
                            s
                            for s in all_segments
                            if log_start <= (s["start"] + s["end"]) / 2 < log_end
                        ]
                        local = []
                        for s in chunk_segs:
                            ls = dict(s)
                            ls["start"] = max(0.0, ls["start"] - ext_start)
                            ls["end"] -= ext_start
                            local.append(ls)

                        try:
                            audio = whisperx.load_audio(str(chunk_path))
                            ar = whisperx.align(
                                local, model_a, meta, audio, device, return_char_alignments=False
                            )
                            for s in ar["segments"]:
                                s["start"] += ext_start
                                s["end"] += ext_start
                                if "words" in s:
                                    for w in s["words"]:
                                        if "start" in w:
                                            w["start"] += ext_start
                                        if "end" in w:
                                            w["end"] += ext_start
                            aligned.extend(ar["segments"])
                        except InterruptedError:
                            raise
                        except Exception as e:
                            # Previously: kept the block unaligned, job
                            # continued as ✅ — a transcript missing word
                            # timestamps in patches, with no way to tell just by
                            # looking at it. Design decision: a sub-step that
                            # fails invalidates the whole run. (If there is NO
                            # alignment model at all for the language, that is
                            # not a failure but a known limitation — the path via
                            # `model_a is None` above remains for that.)
                            raise RuntimeError(
                                f"Alignment block {ci + 1}/{nc} failed: {str(e)[:300]}"
                            ) from e

                    align_ran = True
                    del model_a, meta
                    release_held()  # the alignment model may be replaced again
                    gc.collect()
            torch.cuda.empty_cache()
            step_done("Alignment", vram=_vr_align)

            # Word timestamps requested, alignment RAN — but not a single word
            # got a time (with CJK languages wav2vec2 occasionally produces
            # nothing usable). Without a warning the job would look successful
            # even though the SRT fine-splitting silently fails to happen
            # The failures before this (no language/no
            # model) already warn on their own.
            if word_timestamps and align_ran and not any(s.get("words") for s in aligned):
                job_manager.warn(
                    job,
                    "Word timestamps: alignment produced no "
                    "word times — subtitles remain on "
                    "segment boundaries",
                )

            for cp, _ls, _es, _le in chunks:
                if cp != wav and cp.exists():
                    cp.unlink()

        # ── Diarization ──────────────────────────────────────
        if job.cancel:
            raise InterruptedError("Cancelled")
        diar_segs = None  # stays None without diarization (marker step)
        if _diar_possible:
            phase(
                "Speaker recognition (diarization)",
                speakers=(f"{min_speakers}" if min_speakers else "automatic"),
            )
            job.update(progress=73, message=f"Diarization… · {elapsed()}")
            job_manager.log(job, "Loading diarization model (pyannote)")

            # VRAM context spans loading + inference of the pyannote model —
            # this is the candidate most likely to be kept together with the
            # Whisper model on one GPU, while separation moves to the second.
            # It measures via nvidia-smi and thus also sees the worker
            # process (like with audio-separator).
            with (
                GATE.stage("diarization", job=job),
                debug.vram("Diarization (pyannote)") as _vr_diar,
            ):
                _t_diar_load = time.time()
                est_secs = max(10, total_dur / 10)
                diar_done = threading.Event()

                def _diar_timer():
                    t0 = time.time()
                    while not diar_done.is_set():
                        dt = time.time() - t0
                        ratio = min(dt / est_secs, 0.95)
                        prog = int(76 + ratio * 11)
                        job.update(
                            progress=prog,
                            message=(
                                f"Diarization… {format_duration(dt)} / "
                                f"~{format_duration(est_secs)} · {elapsed()}"
                            ),
                        )
                        diar_done.wait(2.0)

                timer_t = threading.Thread(target=_diar_timer, daemon=True)
                timer_t.start()
                try:
                    diar_segs = _diarize_or_none(
                        wav,
                        min_speakers,
                        max_speakers,
                        device,
                        hf_token,
                        job=job,
                        total_dur=total_dur,
                        load_done=load_done,
                        t_load=_t_diar_load,
                        voices_only=voices_only,
                    )
                finally:
                    # ALWAYS stop the timer — otherwise the thread keeps
                    # overwriting job.progress/message forever after a
                    # diarization failure.
                    diar_done.set()
                    timer_t.join(timeout=1)

            if diar_segs is None:
                # The speaker step failed (said above): the transcript
                # without speakers, as without a token
                final = aligned
            else:
                # The whole audio in memory — only needed for the voice
                # vectors now. On the subprocess path the worker loads its own
                # copy; without vectors nothing more is loaded here at all
                # (measured 0.6-1.7 s per hour, which then simply doesn't happen).
                full_audio = whisperx.load_audio(str(wav)) if _vectors_on else None

                if job.cancel:
                    del full_audio
                    gc.collect()
                    torch.cuda.empty_cache()
                    raise InterruptedError("Cancelled")

                job.update(progress=88, message=f"Assigning speakers… · {elapsed()}")
                # assign_word_speakers iterates per segment over seg["words"] —
                # if alignment was skipped/failed, the key is missing and the
                # call would die with KeyError. Guard with an empty word list.
                for _s in aligned:
                    _s.setdefault("words", [])
                res_sp = whisperx.assign_word_speakers(diar_segs, {"segments": aligned})
                final = res_sp["segments"]
                step_done("Diarization", vram=_vr_diar)

                # ── Capturing voice vectors ─────────────────────────────
                # Right here everything needed for this is available, and only
                # here: `full_audio` is the whole 16-kHz audio in memory
                # (separation ran right at the start and replaced audio.wav),
                # `diar_segs` are the fresh speaker segments. Two lines further
                # down both are freed and `wav` is deleted — until now the
                # speaker information was completely discarded at this point.
                #
                # No ffmpeg, no temp file, no second model pass over the
                # audio: cutting happens in memory. Cost is about one second
                # per 30-s window.
                if _vectors_on:
                    try:
                        from whisper_tm.core import speaker as _speaker
                        from whisper_tm.core.paths import canon as _canon

                        job.update(progress=90, message=f"Voice vectors… · {elapsed()}")
                        # Voices-only: the transcript was diarized by an earlier
                        # run whose labels this fresh diarization does not
                        # reproduce — name the new turns after the transcript's
                        # speakers so vectors and sentences keep one label
                        # (speaker.align_labels). Audio and windows unchanged.
                        _turns_for_vectors = diar_segs
                        if voices_only and db:
                            _aligned_turns, _label_info = _speaker.align_labels(
                                _speaker.normalize_turns(diar_segs),
                                db.transcript_speakers(_canon(str(source_path))),
                            )
                            if _label_info:
                                _turns_for_vectors = _aligned_turns
                                _voice_labels.update(_label_info)
                                _renamed = {
                                    k: v for k, v in _label_info["aligned"].items() if k != v
                                }
                                job_manager.log(
                                    job,
                                    "Voice labels aligned to the transcript: "
                                    + (
                                        ", ".join(f"{k}→{v}" for k, v in _renamed.items())
                                        or "already identical"
                                    )
                                    + (
                                        " · without counterpart: "
                                        + ", ".join(
                                            f"{k}→{v}" for k, v in _label_info["unmatched"].items()
                                        )
                                        if _label_info["unmatched"]
                                        else ""
                                    ),
                                )
                        _sv = _speaker.carry_along(
                            _canon(str(source_path)),
                            full_audio,
                            _turns_for_vectors,
                            device=device,
                            job=job,
                            abort=lambda: bool(job.cancel),
                            separated=_audio_separated,
                            work=work / "speaker_separation",
                            window_separation=_window_separation,
                            keep_decisions=bool(voices_only and refresh_voices),
                        )
                        _voice = _sv
                        _label_turns = _speaker.normalize_turns(_turns_for_vectors)
                        if (
                            _sentence_check
                            and _audio_separated
                            and len(_sv.get("centroids") or {}) >= 2
                        ):
                            _check_with = (full_audio, _sv["centroids"])
                        _renewal = _sv.get("renewal")
                        if _renewal:
                            for _c in _renewal["carried"]:
                                job_manager.log(
                                    job,
                                    f"Voice renewed: {_c['new']} · {_c['windows']} "
                                    f"window{'s' if _c['windows'] != 1 else ''} "
                                    f"{_c['name']} ← {', '.join(_c['from'])} "
                                    f"(at least {_c['sim']:.2f})",
                                )
                            for _u in _renewal["unplaced"]:
                                job_manager.log(
                                    job,
                                    f"No clear counterpart: {_u['old']} ({_u['name']}"
                                    + (
                                        f"; nearest new window in {_u['best']} at {_u['sim']:.2f}"
                                        if _u["best"]
                                        else ""
                                    )
                                    + ")",
                                )
                            job_manager.log(
                                job,
                                (
                                    f"Voices renewed: {_renewal['labels_old']} old → "
                                    f"{_renewal['labels_new']} new speakers, "
                                    f"{_renewal['names']} windows named, "
                                    f"{_renewal['trash']} in the trash, "
                                    f"{_renewal['history_entries']} history entries followed "
                                    f"({_renewal['secs']:.1f} s)"
                                )
                                if _renewal["replaced"]
                                else "Voices kept as they were: not every decided voice has a "
                                "clear counterpart in the new run",
                            )
                        if _sv.get("windows"):
                            job_manager.log(
                                job,
                                f"Voice vectors: {_sv['windows']} windows "
                                f"from {_sv['labels']} identities "
                                f"({_sv['variant']}, {_sv['model']})",
                            )
                        else:
                            job_manager.log(job, f"Voice vectors: none ({_sv.get('reason', '?')})")
                        # Own time entry in the ⏱ summary: a new step that
                        # appears nowhere would be blamed on diarization during
                        # the next performance comparison.
                        step_done("Voice vectors")
                    except InterruptedError:
                        raise
                    except Exception as e:
                        # A failing sub-step invalidates the whole run: a warn()
                        # and a ✅ left files with a few of their vectors and
                        # no way to tell the GPU was full at that moment. The
                        # error text stays untruncated so the queue can tell an
                        # OOM and retry once with VRAM saving (speaker.
                        # compute_vectors salvages what it can before). The
                        # one exception: offline without the model's weights —
                        # the transcript is kept, recorded without a voice
                        # step, so "voices missing" finds it for a voices-only
                        # run once the weights are there.
                        from whisper_tm.core.speaker import VoiceWeightsUnavailable

                        if not isinstance(e, VoiceWeightsUnavailable) or voices_only:
                            raise RuntimeError(f"Voice vectors failed: {str(e)[:400]}") from e
                        job_manager.warn(job, f"Voice vectors skipped: {e}")
                        step_done("Voice vectors")

                del full_audio
                gc.collect()
                torch.cuda.empty_cache()
                # In low_vram mode the worker process does NOT keep the model
                # beyond the file — the same rule as for the resident models
                # in the own process. (`_lv`, not `low_vram`: the name here is
                # the function's per-run parameter, not the query.)
                if _lv():
                    from whisper_tm.core import diar_process

                    diar_process.terminate_all("low_vram")
        else:
            final = aligned

        if wav.exists():
            wav.unlink()

        if voices_only:
            _secs = round(time.time() - t_start, 1)
            voices = {
                "source": "voices-only run",
                "app_version": _voices_version()[0],
                "app_commit": _voices_version()[1],
                "at": round(time.time()),
                "diarized": diar_segs is not None,
                "separated": bool(_audio_separated),
                "vocal_model": vocal_model if vocal_isolation else "",
                "speakers": (
                    [min_speakers, max_speakers] if (min_speakers or max_speakers) else None
                ),
                "windows": int(_voice.get("windows") or 0),
                "model": _voice.get("model") or "",
                "reason": _voice.get("reason") or "",
                # {} = transcript without speakers, nothing to align to
                "labels": _voice_labels,
            }
            if _voice.get("renewal"):
                _rn = _voice["renewal"]
                voices["renewal"] = {
                    "replaced": _rn["replaced"],
                    "labels_old": _rn["labels_old"],
                    "labels_new": _rn["labels_new"],
                    "carried": len(_rn["carried"]),
                    "names": _rn["names"],
                    "trash": _rn["trash"],
                    "history_entries": _rn["history_entries"],
                    "unplaced": [u["old"] for u in _rn["unplaced"]],
                }
            # Speakers and names into the transcript. A sub-step like any
            # other: if it fails, the run fails (and is tried again).
            try:
                _cw = _check_with
                _tr = _voices_into_transcript(
                    db,
                    source_path,
                    filename,
                    _label_turns,
                    check=(
                        (lambda segs, labels: _recheck(_cw, segs, device, job, labels))
                        if _cw is not None
                        else None
                    ),
                )
            except InterruptedError:
                raise
            except Exception as e:
                raise RuntimeError(
                    f"Writing the speakers into the transcript failed: {str(e)[:400]}"
                ) from e
            finally:
                _check_with = _cw = None  # the whole audio goes now
            _log_recheck(job, _tr.get("sentence_check"))
            voices["transcript"] = _tr
            if _tr["segments"]:
                job_manager.log(job, f"Transcript: speakers for {_tr['segments']} segments")
            for _lb, _nm in _tr["names"].items():
                job_manager.log(job, f"Transcript: {_lb} → {_nm}")
            if db:
                from whisper_tm.core.paths import canon as _canon

                db.merge_transcript_settings(_canon(str(source_path)), {"voices": voices})
            _kept = voices.get("renewal", {}).get("replaced") is False
            _told = f"{_tr['segments']} segments got speakers" if _tr["segments"] else ""
            if _tr["names"]:
                _told += (", " if _told else "") + f"{len(_tr['names'])} named"
            job_manager.log(
                job,
                (
                    "✅ Voices kept as they were"
                    if _kept
                    else f"✅ Voices: {voices['windows']} windows"
                )
                + f", transcript: {_told or 'unchanged'} "
                f"({format_duration(_secs)})",
            )
            job_manager.complete(
                job,
                {
                    "filename": filename,
                    "voices_only": True,
                    "voices": voices,
                    "processing_time": format_duration(_secs),
                    "processing_secs": _secs,
                    "step_timings": [
                        {
                            "label": s["label"],
                            "secs": s["secs"],
                            **({"group": s["group"]} if "group" in s else {}),
                        }
                        for s in job.step_timings
                    ],
                    "warning_count": job.warning_count,
                    "warnings": job.warnings,
                },
            )
            return

        # ── Filter & save ──────────────────────────────
        phase("Saving transcript")
        job.update(progress=92, message=f"Removing silence… · {elapsed()}")
        final.sort(key=lambda s: s["start"])

        # CJK: Whisper BPE scatters spaces between characters
        # ("こ ん に ち は") — wrong in the written form, so it's removed. Runs
        # for ALL CJK runs, not just with word timestamps.
        from whisper_tm.core.segmentation import NO_SPACE_LANGS

        if det_lang in NO_SPACE_LANGS:
            from whisper_tm.core.segmentation import strip_cjk_spaces

            for s in final:
                s["text"] = strip_cjk_spaces(s.get("text", ""))

        # ── Sorting out prompt echoes ────────────────────────────────────
        # Only when a vocabulary was in play: then a segment consisting
        # exclusively of its words cannot be spoken content, only the
        # continuation of our own input.
        if vocab_prompt:
            try:
                from whisper_tm.core.postprocessing import remove_prompt_echoes

                _before = len(final)
                final = remove_prompt_echoes(final, vocab_prompt)
                _gone = _before - len(final)
                if _gone:
                    job_manager.warn(
                        job,
                        f"{_gone} segment(s) consisted only of the "
                        f'vocabulary "{vocab_preset}" — removed '
                        f"(prompt echo). Comma lists are prone to "
                        f"this; a sentence is more stable.",
                    )
            except Exception as e:
                job_manager.warn(job, f"Prompt echo check skipped: {str(e)[:120]}")

        # With word timestamps: split segments at word boundaries —
        # subtitle quality (only show what is CURRENTLY being said). The
        # four phases (repair, repetitions, split, merge) have lived
        # testably in core/segmentation.py.
        if word_timestamps:
            from whisper_tm.core.segmentation import word_boundary_segmentation

            final = word_boundary_segmentation(final, det_lang, job=job)

        # ── Sentence check ─────────────────────────────────────────────
        # On the final sentences: a subtitle split above may have parted
        # two speakers that one Whisper segment held.
        if _check_with is not None:
            job.update(progress=93, message=f"Checking each sentence's voice… · {elapsed()}")
            try:
                _checked = _recheck(_check_with, final, device, job)
            except InterruptedError:
                raise
            except Exception as e:
                # A sub-step like any other: if it fails, the run fails
                raise RuntimeError(f"Sentence check failed: {str(e)[:400]}") from e
            finally:
                _check_with = None  # the whole audio goes now
            _log_recheck(job, _checked)
            step_done("Sentence check")

        # ── Post-processing: scores & markers ─────────────────────────
        # Reversible — only markers, no segment is changed/deleted:
        # mark_segments sets exclusively `phantom_score` and
        # `overlapped` and does not touch text, times, speakers or words.
        #
        # RUNS AFTER THE SPLIT, and this matters. It
        # used to run before it — and the split builds new dicts in its
        # third phase, which lost both markers. In the database this
        # meant every diarized segment had a zero;
        # `overlapped` had never had a single hit in the entire dataset.
        #
        # Just passing the fields through would NOT have been enough: the
        # markers apply to the boundaries they were computed on. A
        # 20-second segment with 7 seconds of overlap would otherwise pass
        # its marker on to all five subtitles, including the four without
        # overlap. And "Thank you very much." is suspicious as its own
        # subtitle, but not in the middle of twenty seconds of real
        # speech. Hence: cut first, judge afterward.
        try:
            from whisper_tm.core.postprocessing import mark_segments

            _marker_stats = mark_segments(final, diar_segs)
            if _marker_stats["phantom"] or _marker_stats["overlapped"]:
                job_manager.log(
                    job,
                    f"Post-processing: "
                    f"{_marker_stats['phantom']} suspected phantom cases, "
                    f"{_marker_stats['overlapped']} overlapping segments "
                    f"marked",
                )
        except Exception as e:
            # Markers are a bonus feature — they must never cost the job.
            job_manager.warn(job, f"Post-processing skipped: {str(e)[:120]}")

        min_chars = config.get("transcription", "min_segment_chars", default=0)
        formatted = [
            fmt_seg(s, include_words=word_timestamps)
            for s in final
            if not is_silent(s, min_chars=min_chars)
        ]
        speakers = sorted(set(s["speaker"] for s in formatted if s["speaker"] != "—"))

        processing_secs = round(time.time() - t_start, 1)
        proc_fmt = format_duration(processing_secs)

        # ⏱ Timing summary with real-time factors — the one place where
        # pipeline performance per file can be read off.
        try:
            _rt = (total_dur / processing_secs) if processing_secs else 0
            logger.info(
                "⏱ Summary %s: audio %s · total %s (%.1fx real time)",
                filename,
                format_ts(total_dur),
                proc_fmt,
                _rt,
            )
            for _s in job.step_timings:
                _secs = _s.get("secs") or 0
                _factor = f" · {total_dur / _secs:.1f}x" if _secs > 2 and total_dur else ""
                logger.info("⏱   %-38s %8.1fs%s", _s.get("label", "?")[:38], _secs, _factor)
            _ff_secs = getattr(job, "ffmpeg_secs", 0.0)
            _ff_calls = getattr(job, "ffmpeg_calls", 0)
            if _ff_calls:
                logger.info(
                    "⏱   %-38s %8.1fs · %d calls (details: ▶/⏱ lines in the debug log)",
                    "external tools (ffmpeg & co.)",
                    _ff_secs,
                    _ff_calls,
                )
        except Exception:
            pass

        from whisper_tm import __version__ as _app_version
        from whisper_tm.core.version import version_info as _version_info

        # How this transcript was made — everything that decided the result,
        # as it actually took effect, not only as it was requested. Stored
        # with the transcript (DB column settings_json, and in the JSON) so a
        # later selection can say "everything made without diarization
        # before commit X" exactly.
        settings = {
            "source": "recorded",
            "app_version": _app_version,
            "app_commit": _version_info().get("commit") or "",
            "model": model_name,
            "language_requested": language or "",
            "language": det_lang,
            "device": device,
            "compute_type": compute_type,
            "batch_size": _batch,
            "low_vram": bool(_lv()),
            "vocal_isolation": bool(vocal_isolation),
            "vocal_model": vocal_model if vocal_isolation else "",
            "separated": bool(_audio_separated),
            "separation_chunk_sec": config.get("hardware", "sep_chunk_duration_sec", default=1800),
            "asr_blocks": nc,
            "asr_block_sec": asr_block_sec,
            "vocab_preset": vocab_preset or "",
            "vocabulary_chars": len(vocab_prompt or ""),
            "word_timestamps": bool(word_timestamps),
            "export_alongside": bool(export_alongside),
            "diarize": bool(diarize),
            "speakers": ([min_speakers, max_speakers] if (min_speakers or max_speakers) else None),
            "diarized": diar_segs is not None,
            "speaker_vectors": bool(_vectors_on),
            "voice_vectors_step": bool(_voice),
            "voice_windows": int(_voice.get("windows") or 0),
            "voice_model": _voice.get("model") or "",
            "window_separation": bool(_window_separation),
            **({"sentence_check": _checked} if _checked else {}),
        }
        result = {
            "filename": filename,
            "language": det_lang,
            "model": model_name,
            # Which app version was used for transcription? Lands in the DB
            # and UI — e.g. to specifically re-transcribe older data from a
            # previous version after a pipeline improvement.
            "app_version": _app_version,
            "vocal_isolation": vocal_isolation,
            "vocal_model": vocal_model if vocal_isolation else None,
            "segments": formatted,
            "speakers": speakers,
            "total_duration": format_ts(total_dur),
            "total_duration_sec": round(total_dur, 2),
            "total_chunks": nc,
            "processing_time": proc_fmt,
            "processing_secs": processing_secs,
            "step_timings": [
                {
                    "label": s["label"],
                    "secs": s["secs"],
                    **({"group": s["group"]} if "group" in s else {}),
                }
                for s in job.step_timings
            ],
            "complete": True,
            "warning_count": job.warning_count,
            "warnings": job.warnings,
            "settings": settings,
        }

        transcript_mode = config.transcript_mode
        # The database first: the app reads from it, and a folder that can't
        # be written — a read-only library — mustn't cost the transcript.
        # Saved to the DB — always. The former switch database.store_transcripts
        # could disable this and was thus fatal in mode
        # transcript_mode="database": no JSON gets written there, nothing
        # in the DB — the transcript ended up nowhere, yet the job still
        # reported "complete". The DB carries search, scan status and the
        # queue; it's mandatory.
        db_err = None
        if db:
            # sqlite3.OperationalError ("database is locked") is transient →
            # retry a few times with a short backoff.
            for _attempt in range(3):
                try:
                    # abs_path from the ACTUAL location, not from a guessed
                    # root: `filename` is relative to the directory the job
                    # was queued with, and the queue is persistent — by the
                    # time the job runs, config.media_dir may long since
                    # point elsewhere (docs/PATHS.md).
                    db.upsert_transcript(
                        filename, result, media_dir=str(config.media_dir), abs_path=str(source_path)
                    )
                    db_err = None
                    break
                except sqlite3.OperationalError as e:
                    db_err = e
                    job_manager.warn(job, f"DB locked, attempt {_attempt + 1}/3: {e}")
                    if _attempt < 2:
                        time.sleep(0.5 * (_attempt + 1))
                except Exception as e:
                    db_err = e
                    break
            if db_err is not None:
                # In DB-only mode there is NO JSON fallback — a silently
                # logged error would lose the transcript while the job
                # still reports "complete". So fail hard.
                if transcript_mode == "database":
                    raise RuntimeError(
                        f"DB storage failed (DB-only mode, no JSON fallback): {db_err}"
                    )
                job_manager.warn(job, f"DB storage failed (JSON fallback available): {db_err}")

        # Then the JSON (except in DB-only mode). The actual location of the
        # media file determines where it goes — not the directory currently
        # selected in the frontend (docs/PATHS.md). With transcript_mode
        # "alongside" it lands in .transcripts/ NEXT TO the file.
        if transcript_mode != "database":
            tp = transcript_path_for(filename, abs_path=source_path)
            # Written atomically (helpers.atomic_write_json: a short, hashed
            # temporary name in the same folder, then os.replace) — a crash
            # mid-write leaves no truncated transcript, and the name stays
            # under the filesystem's 255 bytes (OSError 36 once cost a whole
            # transcription here).
            try:
                tp.parent.mkdir(parents=True, exist_ok=True)
                atomic_write_json(tp, result)
            except OSError as e:
                # A folder that can't be written — the database has it
                if not db or db_err is not None:
                    raise
                if _beside_ok:
                    job_manager.warn(
                        job,
                        f"Transcript file not written ({e.strerror or e}) — "
                        f"it is kept in the database",
                    )
                else:
                    job_manager.log(job, f"Transcript file not written ({e.strerror or e})")

        # In DB-only mode: log note
        if transcript_mode == "database":
            job_manager.log(job, "💾 DB-only mode: no JSON written")

        # Export SRT/TXT next to the video (Plex/Jellyfin)
        # export_alongside is already resolved here (None → config default,
        # True/False → explicit value). Only export when True.
        if export_alongside:
            job_manager.log(job, f"📺 Exporting SRT next to: {source_path.parent}/")
            try:
                from whisper_tm.core.export import export_alongside as _do_export

                exported = _do_export(source_path, result, force=True)
                if exported:
                    names = ", ".join(Path(p).name for p in exported)
                    job_manager.log(job, f"📄 Exported: {names}")
                else:
                    job_manager.log(
                        job, "⚠ Export empty — check formats (config: export_alongside_formats)"
                    )
            except Exception as e:
                import traceback

                job_manager.log(job, f"⚠ Export failed: {e}\n{traceback.format_exc()}")

        # Invalidate cache so the scan sees the new status
        invalidate_complete_cache(filename)

        # Log EVERYTHING BEFORE `complete()`.
        #
        # `complete()` and `fail()` write the job log to disk in one go
        # (`jobs/__init__.py::_persist_log`, an `open(…, "w")` over the
        # collected lines). Anything after that only lands in the in-memory
        # list and is gone with the job.
        #
        # That's why the closing line was missing from EVERY job log: it
        # was one line too late. Only noticed when the new storage line
        # behind it also failed to show up — until then this was a log
        # ending in the middle of post-processing, and nobody missed a
        # "Done!" that never existed.
        job_manager.log(job, f"✅ Done! {len(formatted)} segments, {proc_fmt}")
        # The server's own memory state belongs here too, not just the
        # VRAM figure: over a run spanning days, the change per job is the
        # only way to spot a leak before the machine starts swapping
        # (debug.py).
        try:
            _mem = debug.memory_report()
            if _mem:
                job_manager.log(job, f"🧠 Server memory: {_mem}")
            # Speak up clearly once above the threshold — in the job log,
            # the server log, and as a banner in the UI. A value that only
            # sits in one log line among a thousand others warns no one.
            _warn = debug.memory_limit_reached()
            if _warn:
                job_manager.log(job, f"⚠ {_warn}")
                job_manager.system_warning(_warn, key="memory")
                logger.warning("%s", _warn)
        except Exception as e:
            logger.debug("Memory report not possible: %s", e)
        job_manager.complete(job, result)

    except InterruptedError:
        job_manager.log(job, "❌ Cancelled")
        job_manager.fail(job, "Cancelled")
        try:
            torch.cuda.empty_cache()
        except Exception as e:
            logger.debug("Unhandled exception: %s", e)

    except CorruptMediaError as e:
        # The file is broken (e.g. an aborted recording without a moov
        # atom). Retrying would fail again right here — hence onto the
        # skiplist, otherwise it would land back in the queue on every
        # retry.
        err_msg = str(e)
        job_manager.log(job, f"❌ {err_msg}")
        try:
            add_to_skiplist(filename, "corrupt", abs_path=source_path)
        except Exception as se:
            logger.warning("add_to_skiplist failed: %s", se)
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        job_manager.fail(job, err_msg)

    except Exception as e:
        err_msg = str(e)
        job_manager.log(job, f"❌ Error: {err_msg}")
        _log_traceback(job_manager, job, filename)
        if "no audio" in err_msg.lower():
            try:
                add_to_skiplist(filename, "no_audio", abs_path=source_path)
            except Exception as e:
                logger.warning("add_to_skiplist failed: %s", e)
        # Also clear GPU cache on the error path (models may have been
        # loaded mid alignment/diarization)
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        # CUDA OOM: the resident models are now part of the problem —
        # free them so the next run (or a retry with "save VRAM") starts
        # with a clean card instead of dying again right away.
        if "out of memory" in err_msg.lower():
            try:
                from whisper_tm.core.vocal_isolation import release_separator

                with GATE.exclusive("cleanup after a VRAM overflow"):
                    release_models()
                    release_separator()
                    torch.cuda.empty_cache()
                job_manager.log(
                    job, "🧠 OOM: unloaded resident models — retry starts with free VRAM"
                )
            except Exception as se:
                logger.debug("OOM cleanup: %s", se)
        job_manager.fail(job, err_msg)

    finally:
        set_low_vram_override(None)
        # Whatever this run still held (an error mid-stage): let go, or the
        # models would count as in use for as long as this lane lives
        release_held()
        if work.exists():
            shutil.rmtree(work, ignore_errors=True)
        # NO MORE deleting the source. uploads/ used to be a staging area
        # in the data directory — after a successful transcription the
        # source file was discarded. Since a later rework,
        # uploads/ IS part of the library (media_dir/uploads, DB
        # registered, transcript alongside): the old cleanup DESTROYED
        # freshly downloaded videos after transcription — the player found
        # only a 404, and the transcript was left orphaned. Anyone who
        # wants to get rid of uploads deletes them deliberately via the
        # list.


def _log_traceback(job_manager, job, filename: str) -> None:
    """A failed job's traceback: into its log (kept as a file) and the debug
    log — not onto the console, where the queue says it in one line. One
    offline failure printed some 130 lines there. Call it from the except
    block."""
    import traceback

    job_manager.log(job, traceback.format_exc().rstrip())
    logger.debug("%s failed", filename, exc_info=True)


#: The banner for a separation that can't load: once per server run, not
#: once per file of a queue
_separation_said = False


def _separation_unavailable(job_manager, job, error: Exception) -> None:
    """Say that a file is transcribed without its vocal isolation."""
    global _separation_said
    job_manager.warn(job, f"Vocal isolation unavailable — transcribed without it: {error}")
    if not _separation_said:
        _separation_said = True
        job_manager.system_warning(
            f"Vocal isolation is unavailable: {str(error)[:200]}. Files are "
            'transcribed without it (their recorded settings say "not separated") '
            "until its model loads — with an internet connection for the first "
            "download — or vocal isolation is switched off (Settings → Vocal isolation)."
        )


def _voices_version() -> tuple[str, str]:
    from whisper_tm import __version__
    from whisper_tm.core.version import version_info

    return __version__, version_info().get("commit") or ""


#: The MP3s next to the source: VBR V0, the encoder's best — a track kept
#: next to the file is wanted at full bitrate (was V5)
ALONGSIDE_MP3 = ["-codec:a", "libmp3lame", "-q:a", "0"]


def _mp3_alongside(stem_wav, target, job) -> int:
    """One track as MP3 at ``target`` — through a temporary name in the same
    folder, renamed only once ffmpeg succeeded. Returns its size; raises on
    failure (and leaves nothing behind)."""
    tmp = target.with_name(
        f".wtm-{hashlib.sha256(target.name.encode('utf-8')).hexdigest()[:16]}.mp3.tmp"
    )
    try:
        # "-f mp3": the extension is .tmp, ffmpeg would not know the format
        r = run_cancellable(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(stem_wav),
                *ALONGSIDE_MP3,
                "-f",
                "mp3",
                str(tmp),
            ],
            job=job,
            timeout=3600,
            label=f"Track next to the file: {target.name}",
        )
        if r.returncode != 0 or not tmp.exists() or tmp.stat().st_size < 1000:
            raise RuntimeError(f"ffmpeg (code {r.returncode}): {(r.stderr or '').strip()[-200:]}")
        os.replace(tmp, target)
    finally:
        tmp.unlink(missing_ok=True)
    return target.stat().st_size


def _save_audio_alongside(source_path, vocals_path, inst_path, job):
    """Save HQ vocals + instrumental as MP3 next to the source file.

    A failed track is a warning (the transcript does not depend on it) — a
    cancellation is not: it ends the run. Previously it was caught here
    with everything else (``InterruptedError`` is an ``OSError``), and a
    separation-only run cancelled in this step completed as "separated",
    a truncated MP3 next to the recording. Each MP3 is written under a
    temporary name and renamed once ffmpeg succeeded.
    """
    src_stem = source_path.stem
    src_dir = source_path.parent
    v_mp3_size = 0
    i_mp3_size = 0
    try:
        # The tracks are the app's own output — continuous scanning must
        # not take them for new recordings and transcribe them. Noted
        # before they exist: a scan may run while they are written.
        from whisper_tm.core.autoscan import note_made

        note_made(src_dir / f"{src_stem}_vocals.mp3", src_dir / f"{src_stem}_instrumental.mp3")

        if vocals_path and vocals_path.exists():
            alongside_v = src_dir / f"{src_stem}_vocals.mp3"
            try:
                v_mp3_size = _mp3_alongside(vocals_path, alongside_v, job)
                job_manager.log(job, f"💾 Vocals: {alongside_v.name} ({format_size(v_mp3_size)})")
            except InterruptedError:
                raise
            except Exception as e:
                job_manager.warn(job, f"Saving the vocals next to the file failed: {e}")

        if inst_path and inst_path.exists():
            alongside_i = src_dir / f"{src_stem}_instrumental.mp3"
            try:
                i_mp3_size = _mp3_alongside(inst_path, alongside_i, job)
                job_manager.log(
                    job, f"💾 Instrumental: {alongside_i.name} ({format_size(i_mp3_size)})"
                )
            except InterruptedError:
                raise
            except Exception as e:
                job_manager.warn(job, f"Saving the instrumental next to the file failed: {e}")

        # Compare MP3 sizes — this is where real content differences show up
        if v_mp3_size > 0 and i_mp3_size > 0:
            ratio = v_mp3_size / i_mp3_size if i_mp3_size > 0 else 0
            diff_pct = abs(v_mp3_size - i_mp3_size) / max(v_mp3_size, i_mp3_size) * 100
            job_manager.log(
                job, f"📊 Vocals/instrumental MP3 ratio: {ratio:.2f} (difference: {diff_pct:.1f}%)"
            )
            if diff_pct < 5:
                job_manager.warn(
                    job,
                    "⚠ Vocals and instrumental MP3s are nearly the same size — "
                    "separation may have had little effect. "
                    "This is normal for mono audio with mixed speech+music.",
                )
    except InterruptedError:
        raise
    except Exception as e:
        job_manager.warn(job, f"Saving audio failed: {e}")
