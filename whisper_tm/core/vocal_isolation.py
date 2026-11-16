"""
Vocal Isolation: Audio-Separator, Chunked Separation.

Completely ported to new architecture:
- JobManager instead of global job_log/job_warn
- Config instead of constants
- Job dataclass instead of dict (job.cancel instead of job.get("cancel"))
"""

from __future__ import annotations

import logging
import math
import shutil
import threading
import time
from pathlib import Path

from whisper_tm.config import config
from whisper_tm.core import debug, media_formats
from whisper_tm.core.audio_utils import (
    find_output_wav,
    find_output_wavs,
    get_audio_duration,
    get_duration,
    get_duration_safe,
    run_cancellable,
)
from whisper_tm.core.gpu_gate import GATE
from whisper_tm.core.helpers import format_size, format_ts
from whisper_tm.extensions import job_manager

logger = logging.getLogger("whisper_tm.core.vocal_isolation")

# ══════════════════════════════════════════════════════════════════════════
#  Model catalog (metadata for annotated GUI selection)
# ══════════════════════════════════════════════════════════════════════════
# Key MUST match config.vocal_isolation.models. Delivered to frontend via
# /api/vocal_isolation_status, which builds dropdowns from it
# (group → <optgroup>, label → visible text, desc → tooltip).
#
#   group    Group heading in dropdown (order = GROUP_ORDER)
#   label    short visible name incl. short tag
#   quality  1..5 (vocal quality, rough reference)
#   speed    1..5 (5 = fastest)
#   desc     what the model is AND why better/worse (tooltip)
#
# SDR values = audio-separator multisong scale (vocals/music), NOT MUSDB and
# NOT measured on speech — to be understood as relative reference.
GROUP_ORDER = ["Recommended", "Maximum Quality", "Faster"]

#: Proven checkpoint used when the configured one is missing or will not
#: load. Deliberately not offered in the GUI selection.
FALLBACK_MODEL_FILE = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"


class SeparationUnavailable(RuntimeError):
    """No separation model loads — neither the configured one nor the
    fallback (offline before the first download, say). A transcription goes
    on without the separation; a separation-only job has nothing to do."""


MODEL_INFO = {
    # The model palette is reduced to ONE
    # visible model — mel_roformer_kim won on speed (1.49x), loop metric
    # AND word recall (+23 points); all other candidates measured
    # worse or unproven-slower. bs_roformer stays as a
    # silent fallback in the code (see _get_separator), but no longer
    # appears in the selection. Goal: a simple, reliable program — one
    # truth instead of eight options.
    "mel_roformer_kim": {
        "group": "Recommended",
        "label": "MelBand Roformer (Kim) \u2b50 Standard",
        "quality": 5,
        "speed": 3,
        "desc": "Mel-band model tuned for the human voice. Best "
        "measured model in our tests: highest vocal SDR "
        "(~12.6), 1.49\u00d7 faster than BS-Roformer and +23 points "
        "word recall on noisy material.",
    },
}


def model_catalog() -> list[dict]:
    """Annotated model list for the GUI.

    Visible is what is in MODEL_INFO (5.3: deliberately only the
    standard now). Config may carry additional entries — e.g. the
    silent bs_roformer fallback or manual experiments —, which
    then don't appear in the selection.
    """
    models = config.get("vocal_isolation", "models", default={}) or {}
    out = [{"name": name, **MODEL_INFO[name]} for name in models if name in MODEL_INFO]
    if not out and models:
        # Safety net: config knows NO known model → offer the first one anyway,
        # otherwise the dialog would be empty and unusable.
        name = next(iter(models))
        out = [
            {
                "name": name,
                "group": "Recommended",
                "label": name,
                "quality": 0,
                "speed": 0,
                "desc": "",
            }
        ]
    order = {g: i for i, g in enumerate(GROUP_ORDER)}
    out.sort(key=lambda e: (order.get(e["group"], 99), -e["quality"]))
    return out


# ══════════════════════════════════════════════════════════════════════════
#  Audio-Separator
# ══════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════
#  Resident Separator (Python API instead of CLI subprocess)
#
#  The CLI subprocess cost ~4.8s fixed
#  cost per call (process start + CUDA context + model load) — hours
#  across a large collection. The model now stays loaded (VRAM peak of all
#  models combined: about 8.4 GB, measured on a 12 GB card).
#  hardware.low_vram=true restores the old behavior (unload after each
#  file) for cards with less memory.
# ══════════════════════════════════════════════════════════════════════════

_separator = None  # audio_separator.Separator with model loaded
_separator_key = None  # (model_file, single_stem, settings) of the loaded model
_separator_lock = threading.Lock()  # serialize loading + inference (one GPU)


class _SeparatorErrorLog(logging.Handler):
    """Records what ``audio_separator`` actually failed on.

    ``Separator.separate()`` catches EVERY exception from ``_separate_file``
    itself, only writes it to the log ("Failed to process file …: CUDA
    out of memory …") and returns an empty list. For us, only "no output
    files" made it through — and the queue, which triggers a retry with
    VRAM-sparing on "out of memory", never saw an OOM.
    Seen in practice: several files in a row failed permanently while
    another program was using the GPU at the same time.

    The handler is attached to the package's logger only during
    ``separate()`` and remembers the last error message; that flows into
    the error text.
    """

    def __init__(self):
        super().__init__(level=logging.ERROR)
        self.last_one = None
        self.out_of_memory = None
        # Only this thread's records: with queue lanes another file's
        # separation error must not become this one's reason.
        self._thread = threading.get_ident()

    def emit(self, record):
        if record.thread != self._thread:
            return
        try:
            self.last_one = record.getMessage()
        except Exception:
            self.last_one = str(record.msg)
        if "out of memory" in self.last_one.lower():
            self.out_of_memory = self.last_one

    def reason(self, maxlen=300):
        """The part after "Failed to process file <path>: " — the path
        itself is only noise in the error text. A full card wins over the
        lines the library logs after it ("Please try deleting the model
        file …"): the queue's OOM retry goes by these words."""
        text = self.out_of_memory or self.last_one
        if not text:
            return None
        marker = "Failed to process file "
        if text.startswith(marker) and ": " in text[len(marker) :]:
            text = text[len(marker) :].split(": ", 1)[1]
        return text[:maxlen]

    def __enter__(self):
        logging.getLogger("audio_separator").addHandler(self)
        return self

    def __exit__(self, *exc):
        logging.getLogger("audio_separator").removeHandler(self)
        return False


_sep_api_out = None  # process-wide API output folder (see _sep_api_dir)

# Prefilter balance since server start (health check). The
# job logs show the balance per file — here it additionally accumulates
# so the GUI can show "since start: X separated / Y clean / Z silence".
_prefilter_total = {"separate": 0, "clean": 0, "silence": 0}
_prefilter_lock = threading.Lock()


def _prefilter_count(class_name: str) -> None:
    with _prefilter_lock:
        _prefilter_total[class_name] = _prefilter_total.get(class_name, 0) + 1


def prefilter_balance() -> dict:
    """Copy of the total prefilter balance since server start (health check)."""
    with _prefilter_lock:
        return dict(_prefilter_total)


def _sep_api_dir() -> Path:
    """ONE fixed output folder for all API calls of the process.

    audio-separator copies output_dir into the model instance during
    load_model — a folder that changes per call would therefore go nowhere
    from the second job on (the acceptance test found exactly that).
    That's why the resident separator ALWAYS writes here; _run_separator
    then moves the stems to their respective target (shutil.move: rename,
    or copy across filesystem boundaries if needed).
    """
    global _sep_api_out
    if _sep_api_out is None:
        try:
            basis = Path(config.dirs["work"])
        except Exception:
            # Standalone use (tests/scripts without full app config)
            basis = Path.home() / ".cache" / "whisper-tm"
        _sep_api_out = basis / "_sep_api_out"
    _sep_api_out.mkdir(parents=True, exist_ok=True)
    return _sep_api_out


def _low_vram() -> bool:
    # Config OR per-run override from the dialog (core/vram.py)
    from whisper_tm.core.vram import low_vram

    return low_vram()


def _drop_separator():
    """Let go of the resident separator and hand its VRAM back.

    Dropping the reference alone is not enough: what the model holds can
    sit in reference cycles until the next garbage collection, and torch's
    caching allocator keeps freed blocks for this process — the
    diarization worker, a process of its own on the same card, would get
    none of it. Must be called under _separator_lock.
    """
    global _separator, _separator_key
    if _separator is None:
        return
    _separator = None
    _separator_key = None
    import gc

    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass


def release_separator():
    """Unload resident separation model (low_vram / tests / shutdown).

    Not from inside a GPU stage of another owner: the exclusive would wait
    for itself. Then the model stays — loaded is not wrong, a run that
    fails over it would be (window separation in the voice-vector stage).
    """
    if not GATE.may_take_exclusive():
        logger.debug("Separator stays loaded: released from inside a GPU stage")
        return
    with GATE.exclusive("unloading the separator"), _separator_lock:
        _drop_separator()


def _load_model(sep, model_file) -> None:
    """``Separator.load_model`` — without ending the thread.

    audio-separator calls ``sys.exit(1)`` when a model fails to load (its
    MDXC loader, for example on CUDA out of memory). SystemExit passes
    every ``except Exception``: it ended the queue's worker thread without
    a word, its entry left 'processing' — all queue lanes at once when
    another program filled the card. Here it becomes the error
    it should have been, with the logged reason, so a full card reaches
    the queue's OOM retry.
    """
    with _SeparatorErrorLog() as log:
        try:
            sep.load_model(model_filename=model_file)
        except SystemExit:
            raise RuntimeError(
                f"Separation model {model_file} did not load: "
                f"{log.reason() or 'audio-separator gave up'}"
            ) from None


def _get_separator(model_file, api_out_dir, single_stem=None, job=None):
    """Get resident separator instance; only loads the model on change.

    output_dir/single_stem are copied by audio-separator into the model
    instance during load_model — that's why api_out_dir is a FIXED folder
    per process, from which the stems are moved to the target after each
    call (rename on the same partition, free). Must be called under
    _separator_lock.
    """
    global _separator, _separator_key
    from whisper_tm.core.helpers import import_clean
    from whisper_tm.core.model_idle import report_usage

    # From a clean stack, like whisperx (helpers.import_clean)
    Separator = import_clean("audio_separator.separator").Separator

    report_usage()  # keeps the idle-unloader at bay
    use_autocast = bool(config.get("vocal_isolation", "use_autocast", default=True))
    # Effective parameters, measured: for Roformer (.ckpt)
    # only overlap is relevant — batch_size is bit-identically ineffective,
    # segment_size has no effect without an override flag. For MDX (.onnx)
    # all three have an effect.
    mdx_params = {
        "segment_size": config.get("vocal_isolation", "mdx_segment_size", default=256),
        "overlap": config.get("vocal_isolation", "mdx_overlap", default=0.25),
        "batch_size": config.get("vocal_isolation", "mdx_batch_size", default=1),
        "hop_length": 1024,
        "enable_denoise": False,
    }
    mdxc_params = {
        "segment_size": 256,
        "override_model_segment_size": False,
        "overlap": config.get("vocal_isolation", "mdxc_overlap", default=8),
        "batch_size": 1,
        "pitch_shift": 0,
    }
    # The settings belong in the key, not just the model: the Separator
    # takes them once, when it is built. Keyed by model alone, a changed
    # overlap or autocast only took effect once the model happened to be
    # unloaded (idle unload, "Unload VRAM", restart). Frozen to plain
    # values: the dicts are rebuilt on every call and handed to the
    # library, and unchanged settings must compare equal and keep the
    # loaded model.
    settings = (use_autocast, tuple(sorted(mdx_params.items())), tuple(sorted(mdxc_params.items())))
    # The model file stays first — vram_procs.resident_models() names the
    # separator by key[0].
    key = (model_file, single_stem or "", settings)
    if _separator is not None and _separator_key == key:
        return _separator

    # Release the old model before the new one loads — the whole way, as
    # release_separator() does, not just the reference (see _drop_separator).
    _drop_separator()
    t0 = time.time()
    sep = Separator(
        output_dir=str(api_out_dir),
        output_format="WAV",
        use_autocast=use_autocast,
        output_single_stem=single_stem,
        mdx_params=mdx_params,
        mdxc_params=mdxc_params,
        log_level=logging.WARNING,
        model_file_dir=str(separation_model_dir()),
    )
    try:
        _load_model(sep, model_file)
    except Exception as e:
        # Fallback: if the default checkpoint doesn't load (broken
        # download, network down on first fetch), falls back to
        # bs_roformer — proven, but deliberately NOT in the GUI selection.
        # Not for a full card: that passes, and another model's vectors
        # would not compare with the rest.
        fallback = FALLBACK_MODEL_FILE
        if "out of memory" in str(e).lower():
            raise
        if model_file == fallback:
            raise SeparationUnavailable(f"{model_file} does not load: {str(e)[:160]}") from e
        logger.warning(
            "Separation model %s does not load (%s) — falling back to %s",
            model_file,
            str(e)[:120],
            fallback,
        )
        try:
            _load_model(sep, fallback)
        except Exception as e2:
            if "out of memory" in str(e2).lower():
                raise
            # Neither loads (offline before the first download): the
            # banner of a working fallback would send the user the wrong way
            raise SeparationUnavailable(
                f"{model_file} does not load ({str(e)[:100]}), nor the fallback "
                f"{fallback} ({str(e2)[:100]})"
            ) from e2
        if job_manager:
            # UNMISSABLE (by design): not just the job log, but a
            # system-wide red banner in the WebUI until acknowledged.
            job_manager.system_warning(
                f"🚨 SEPARATION MODEL FALLBACK ACTIVE: {model_file} does not "
                f"load ({str(e)[:100]}) — running bs_roformer (slower, "
                f"slightly weaker). It is tried again with every file; check "
                f"its download (Settings → Vocal isolation)."
            )
            if job:
                job_manager.warn(
                    job,
                    f"Separation model {model_file} does not load — falling back to bs_roformer",
                )
        model_file = fallback
        key = (model_file, single_stem or "", settings)
    _separator = sep
    _separator_key = key
    if job and job_manager:
        job_manager.log(
            job,
            f"Separation model loaded ({model_file}, "
            f"autocast={use_autocast}): "
            f"{time.time() - t0:.1f}s — stays resident"
            + (" (low_vram: will be unloaded after the file)" if _low_vram() else ""),
        )
    return sep


# ── Minimum length for separation (Roformer chunk window) ──
# audio-separator processes Roformer models in windows of
# stft_hop_length × (dim_t − 1) samples (model YAML):
#   mel_roformer_kim  441 × 1100 = 485,100 samples = 11.0 s @ 44.1 kHz
#   bs_roformer       441 ×  800 = 352,800 samples =  8.0 s
# If the audio is shorter than ONE window, it ends up in the "last chunk"
# branch of the demix loop (mdxc_separator.py, overlap_add): the start
# index is computed as length − window, then negative, the target slice
# in the result tensor becomes empty → "The size of tensor a (0) must
# match the size of tensor b (<audio_length>)". The library does have a
# short-file fallback (smaller windows of 2.55 s each, issue #232), but
# it only kicks in BELOW 10.0 s — the threshold is smaller than the Kim
# window. Result: files between 10.0 s and 11.0 s crash (seen with a
# clip whose audio track was ~10.2 s; empirically reproduced:
# 10.3 s crashes, 11.2 s runs), and below 2.55 s the same risk exists in
# the fallback path.
#
# The reference implementation (ZFTurbo/Music-Source-Separation-Training,
# demix in utils/model_utils.py) pads chunks that are too short before
# the model to exactly chunk_size — with silence for very short material —
# and trims back afterward. audio-separator lacks exactly this pad step.
# Instead of patching the library, the same thing happens here one level
# up: audio below the minimum length is padded with silence before
# separation and the stems are trimmed back to the original length. ONE
# code path for all short files, covering all dead zones, independent of
# the library fallback.
#
# CAUTION with new separation models: the threshold must be ABOVE the
# largest window (stft_hop_length × (dim_t − 1) / sample_rate) of all
# configured models — it's in the model YAML under inference.
_SEP_MIN_AUDIO_SEC = 11.2  # 11.0 s Kim window + safety margin


def _pad_for_separation(input_file, job=None):
    """Pad audio that's too short to the separation minimum length.

    Returns (file_for_separator, original_duration|None); None means:
    long enough, nothing padded, no trim-back needed.
    """
    dur = get_duration_safe(input_file)
    if not dur or dur >= _SEP_MIN_AUDIO_SEC:
        return input_file, None
    # Same file name in a subfolder: the separator derives the stem
    # names from the input name — a suffix would leak through to the
    # download file names.
    pad_dir = input_file.parent / "sep_pad"
    pad_dir.mkdir(parents=True, exist_ok=True)
    padded = pad_dir / input_file.name
    r = run_cancellable(
        [
            "ffmpeg",
            "-y",
            *media_formats.source(input_file),
            "-af",
            f"apad=whole_dur={_SEP_MIN_AUDIO_SEC}",
            "-acodec",
            "pcm_s16le",
            str(padded),
        ],
        job=job,
        timeout=300,
        label="Pad short file",
    )
    if r.returncode != 0 or not padded.exists() or padded.stat().st_size <= 1000:
        raise RuntimeError(
            f"Padding short file failed (rc={r.returncode}): {(r.stderr or '')[-200:]}"
        )
    if job and job_manager:
        job_manager.log(
            job,
            f"  Audio only {dur:.1f}s — padded with silence to "
            f"{_SEP_MIN_AUDIO_SEC}s for separation (Roformer minimum "
            f"window), stems will be trimmed back",
        )
    return padded, dur


def _unpad_stem(stem_wav, orig_dur, job=None):
    """Trim the padding silence back off — stems must have exactly the
    original length, otherwise the timeline shifts (desync)."""
    tmp = stem_wav.with_name(stem_wav.stem + "_unpad.wav")
    r = run_cancellable(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(stem_wav),
            "-t",
            f"{orig_dur:.3f}",
            "-acodec",
            "pcm_s16le",
            str(tmp),
        ],
        job=job,
        timeout=300,
        label="Trim padding silence",
    )
    if r.returncode != 0 or not tmp.exists() or tmp.stat().st_size <= 1000:
        raise RuntimeError(
            f"Trim-back after padding failed (rc={r.returncode}): {(r.stderr or '')[-200:]}"
        )
    tmp.replace(stem_wav)


def separation_model_dir() -> Path:
    """Where the separation model lives permanently.

    ``audio-separator`` stores its checkpoints by default under
    ``/tmp/audio-separator-models/``. On many systems ``/tmp`` is a
    **tmpfs in RAM** — the directory is empty on every restart, and the
    871 MB of ``vocals_mel_band_roformer.ckpt`` get downloaded again
    (after a restart the file carries a fresh download timestamp).

    This costs more than just bandwidth. The download blocks the first
    job after every restart, and anyone restarting offline instead gets
    the silent fallback to a different separation model — with vectors
    that no longer match the previously computed ones.

    The model directory is placed next to the remainder of the application's
    cache, i.e. where the chunks also live. A dedicated config value
    would be a control for a question nobody wants to answer
    differently; anyone who does need it can set
    ``AUDIO_SEPARATOR_MODEL_DIR`` — audio-separator reads that variable
    itself and thereby overrides this value too.
    """
    target = Path(config.cache_dir) / "separation_models"
    try:
        target.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        # Better the old location than a crash: without a directory nothing loads at all.
        logger.warning(
            "Model directory %s not usable (%s) — staying with audio-separator's default", target, e
        )
        return Path("/tmp/audio-separator-models")
    return target


def _run_separator(input_file, output_dir, model_file, job=None, single_stem=None, retry=1):
    """Separate one file via the resident Python API.

    Stems land in output_dir (as before with the CLI). Raises RuntimeError
    if no output was produced after all attempts.
    """
    input_file = Path(input_file)
    if input_file.is_symlink():
        input_file = input_file.resolve()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Short-file protection: below the Roformer window the separator
    # crashes (see _SEP_MIN_AUDIO_SEC) → pad beforehand.
    input_file, orig_dur = _pad_for_separation(input_file, job=job)

    sep_label = f"Separation {Path(model_file).stem[:40]}"
    last_error = None
    for attempt in range(1 + retry):
        if job and job.cancel:
            raise InterruptedError("Cancelled")
        try:
            with GATE.stage("separation", job=job), _separator_lock:
                api_out = _sep_api_dir()
                for old_value in api_out.iterdir():
                    if old_value.is_file():
                        old_value.unlink(missing_ok=True)
                sep = _get_separator(model_file, api_out, single_stem=single_stem, job=job)
                with (
                    debug.vram(sep_label, poll=1.0) as _vr_sep,
                    _SeparatorErrorLog() as log_capture,
                ):
                    sep.separate(str(input_file))
                created = [f for f in api_out.iterdir() if f.is_file() and f.stat().st_size > 1000]
                for f in created:
                    shutil.move(str(f), str(output_dir / f.name))
            if job and job_manager and _vr_sep.note():
                job_manager.log(job, f"📊 VRAM {sep_label}: {_vr_sep.note()}")
            if created:
                if orig_dur is not None:
                    for f in created:
                        _unpad_stem(output_dir / f.name, orig_dur, job=job)
                    input_file.unlink(missing_ok=True)
                if job and job_manager:
                    job_manager.log(
                        job,
                        "  Separator Output: "
                        + ", ".join(
                            f"{f.name} ({format_size((output_dir / f.name).stat().st_size)})"
                            for f in created
                        ),
                    )
                return [output_dir / f.name for f in created]
            last_error = "no output files"
            if log_capture.reason():
                last_error += f" ({log_capture.reason()})"
        except InterruptedError:
            raise
        except Exception as e:
            last_error = str(e)[:300]

        # With VRAM full, a second attempt three seconds later gains
        # nothing — the memory belongs to someone else. Bail out
        # immediately so the queue can trigger its VRAM-sparing retry.
        if "out of memory" in (last_error or "").lower():
            break

        if attempt < retry:
            if job and job_manager:
                job_manager.warn(
                    job, f"Separation failed ({last_error}), attempt {attempt + 1}, retrying in 3s…"
                )
            time.sleep(3)

    if orig_dur is not None:
        input_file.unlink(missing_ok=True)
    raise RuntimeError(
        f"Separation failed after {attempt + 1} attempt{'s' if attempt else ''}: {last_error}"
    )


# ══════════════════════════════════════════════════════════════════════════
#  Chunk extraction
# ══════════════════════════════════════════════════════════════════════════


def _extract_sep_chunks(
    source, work_dir, chunk_dur, overlap, job=None, sample_rate=48000, channels=2
):
    """
    Extract overlapping audio chunks directly from the source file.
    Returns: (chunks_list, total_duration)
    chunks_list: [(chunk_path, trim_start, trim_end, chunk_duration), ...]
    """
    # The CONTAINER duration determines the split — deliberately not the
    # audio-track duration. The latter would be shorter and wouldn't
    # even address the empty chunks at the end, but relies on a piece of
    # metadata: if it reports too little (broken or imprecise file), the
    # remainder would be silently cut off and the job would still count as
    # successful. A too-short transcript that nobody notices is worse
    # than an abort. Empty chunks are handled below with silence; the
    # separation skips them.
    total_dur = get_duration(source)
    audio_dur = get_audio_duration(source)

    if total_dur <= chunk_dur + overlap:
        single = work_dir / "sep_single.wav"
        cmd = [
            "ffmpeg",
            "-y",
            *media_formats.source(source),
            "-vn",
            "-map",
            "0:a:0",
            "-acodec",
            "pcm_s16le",
            "-ar",
            str(sample_rate),
            "-ac",
            str(channels),
            "-threads",
            "0",
            str(single),
        ]
        r = run_cancellable(cmd, job=job, timeout=14400)
        if r.returncode != 0:
            from whisper_tm.core.audio_utils import ffmpeg_failure

            raise RuntimeError(f"Audio extraction failed: {ffmpeg_failure(r.stderr, 300)}")
        if not single.exists() or single.stat().st_size < 1000:
            raise RuntimeError("Audio extraction: no output file")
        return [(single, 0.0, 0.0, total_dur)], total_dur, set()

    chunks = []
    silence_idx: set[int] = set()  # chunks without audio, padded with silence
    step = chunk_dur
    n = math.ceil(total_dur / step)
    chunk_dir = work_dir / "sep_chunks_in"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    for i in range(n):
        if job and job.cancel:
            raise InterruptedError("Cancelled")

        raw_start = i * step
        raw_end = min((i + 1) * step, total_dur)
        ext_start = max(0.0, raw_start - overlap) if i > 0 else 0.0
        ext_end = min(total_dur, raw_end + overlap) if i < n - 1 else total_dur
        trim_start = raw_start - ext_start
        trim_end = ext_end - raw_end

        cp = chunk_dir / f"sep_in_{i:04d}.wav"
        dur = ext_end - ext_start
        cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            f"{ext_start:.3f}",
            *media_formats.source(source),
            "-t",
            f"{dur:.3f}",
            "-vn",
            "-map",
            "0:a:0",
            "-acodec",
            "pcm_s16le",
            "-ar",
            str(sample_rate),
            "-ac",
            str(channels),
            "-threads",
            "0",
            str(cp),
        ]
        r = run_cancellable(cmd, job=job, timeout=1800, label=f"Sep chunk extraction {i + 1}/{n}")

        # IMPORTANT: a missing chunk would shift ALL subsequent timestamps
        # during concat (the vocals track is joined seamlessly) → video/
        # transcript desync. A chunk must therefore never simply be dropped.
        #
        # But it CAN legitimately be EMPTY: there are recordings whose
        # audio track doesn't span the whole file — it starts later, ends
        # earlier, or only exists in the middle (typical for DVD/TV
        # recordings where the audio only starts with the broadcast).
        # ffmpeg then returns "Output file is empty, nothing was encoded".
        #
        # Instead of aborting, silence of exactly the expected length is
        # generated. This leaves the timeline untouched and the
        # transcript timestamps continue to match the video file.
        #
        # An empty stretch exits 0 (ffmpeg n9: "Output file is empty,
        # nothing was encoded" is a warning). A non-zero exit is a real
        # failure — a read or decode error — and must not quietly turn up
        # to 30 minutes of speech into silence.
        if r.returncode != 0 and "output file is empty" not in (r.stderr or "").lower():
            cp.unlink(missing_ok=True)
            raise RuntimeError(
                f"Chunk {i + 1}/{n} extraction failed (rc={r.returncode}): "
                f"{(r.stderr or '').strip()[-400:]}"
            )
        empty = not cp.exists() or cp.stat().st_size <= 1000
        if empty:
            silence = [
                "ffmpeg",
                "-y",
                "-f",
                "lavfi",
                "-i",
                f"anullsrc=r={sample_rate}:cl={'stereo' if channels == 2 else 'mono'}",
                "-t",
                f"{dur:.3f}",
                "-acodec",
                "pcm_s16le",
                "-threads",
                "0",
                str(cp),
            ]
            rs = run_cancellable(silence, job=job, timeout=300, label="Generate silence chunk")
            if rs.returncode != 0 or not cp.exists() or cp.stat().st_size <= 1000:
                raise RuntimeError(
                    f"Chunk {i + 1}/{n} extraction failed and silence "
                    f"could not be generated: {(r.stderr or '')[-400:]}"
                )
            silence_idx.add(i)
            if job is not None:
                job_manager.log(
                    job,
                    f"Chunk {i + 1}/{n} without audio ({ext_start:.0f}–"
                    f"{ext_end:.0f}s) — padded with silence, "
                    f"timeline preserved",
                )
        chunks.append((cp, trim_start, trim_end, dur))

    if not chunks:
        raise RuntimeError("No audio chunks extracted")
    if len(silence_idx) == len(chunks):
        # Not a single chunk carried audio. Continuing would mean
        # separating a pure silence track and reporting it as a success.
        raise RuntimeError(
            f"No audio track extractable from {source.name} ({len(chunks)} chunks, all empty)"
        )
    if silence_idx and job is not None:
        job_manager.log(
            job,
            f"{len(silence_idx)} of {len(chunks)} chunks without audio "
            f"(audio track {audio_dur:.0f}s of {total_dur:.0f}s) — "
            f"separation is skipped there",
        )
    return chunks, total_dur, silence_idx


# ══════════════════════════════════════════════════════════════════════════
#  Chunk prefilter
#
#  Core idea: how loud is it when nobody is speaking? Music/game audio
#  continues during speech pauses; a clean recording is silent there.
#  Calibrated against 160 windows of real separation ground truth
#  (0 false skips) and a searchability benchmark. Measured
#  to save a large share of separation time — very long recordings
#  often consist mostly of quiet stretches.
# ══════════════════════════════════════════════════════════════════════════


def _chunk_prefilter(chunk_path, chunk_dur, job=None):
    """Classifies an extracted chunk before separation.

    Return value:
      "silence"   practically no signal (< prefilter_silence_db) — the
                  separator would reject it anyway ("empty or not valid")
                  → silence path
      "clean"     speech with silent pauses — nothing to separate, the
                  original directly becomes the vocals track
      "separate"  background present → separation needed

    Hybrid rule (recalibrated, normalization-proof):
      speech fraction ≥ 5%  → skip if pause level < speech level − 25 dB
      otherwise             → skip if pause level < −50 dBFS (absolute)
    Conservative: a chunk only counts as "clean" if ALL sample windows
    skip; any unclear window forces separation.
    """
    if not config.get("vocal_isolation", "prefilter", default=True):
        return "separate"

    rel_db = float(config.get("vocal_isolation", "prefilter_rel_db", default=-25.0))
    abs_db = float(config.get("vocal_isolation", "prefilter_abs_db", default=-50.0))
    silence_db = float(config.get("vocal_isolation", "prefilter_silence_db", default=-60.0))
    min_speech = float(config.get("vocal_isolation", "prefilter_min_speech_share", default=0.05))

    import librosa
    import numpy as np
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    sr = 16000
    window_duration = 60.0
    # Sample windows distributed across the chunk: short = 1 window,
    # otherwise 3 (start/middle/end, 5% margin excluded).
    if chunk_dur <= window_duration * 1.5:
        starts = [0.0]
    else:
        edge = chunk_dur * 0.05
        usable = chunk_dur - 2 * edge - window_duration
        starts = [edge + usable * i / 2 for i in range(3)]

    try:
        for start in starts:
            y, _ = librosa.load(
                str(chunk_path), sr=sr, mono=True, offset=max(0.0, start), duration=window_duration
            )
            if len(y) < sr:
                # Too short for the window logic — but do NOT blindly separate.
                #
                # This is exactly where digitally silent short videos (under
                # one second, all samples zero) used to end up. The separator
                # rejects them ("is empty or not valid" — it checks
                # `np.any(mix)`), the caller tries a second time, finds the
                # same silence, and the WHOLE job fails:
                #
                #   ❌ Separation failed after 2 attempts:
                #      no output files
                #
                # Observed in a large batch on short phone videos. In a big
                # collection this is not a one-off, but a whole class of error.
                #
                # That the separator rejects something like this is described
                # above in the docstring as the case this prefilter is SUPPOSED
                # to catch — the short-file shortcut just never reached it.
                if not len(y) or not np.any(y):
                    return "silence"
                short_rms = float(np.sqrt(np.mean(y**2)))
                if 20 * np.log10(max(short_rms, 1e-9)) < silence_db:
                    return "silence"
                return "separate"  # short, but there's something on it
            rms = float(np.sqrt(np.mean(y**2)))
            total_db = 20 * np.log10(max(rms, 1e-9))
            if total_db < silence_db:
                continue  # window is practically silent → skip

            sections = get_speech_timestamps(
                y, VadOptions(min_silence_duration_ms=300, speech_pad_ms=100)
            )
            mask = np.zeros(len(y), dtype=bool)
            for a in sections:
                mask[a["start"] : a["end"]] = True
            speech_share = float(mask.mean())
            pauses = y[~mask]
            if len(pauses) < sr // 2:
                return "separate"  # no assessable pause → separate
            pause_rms = float(np.sqrt(np.mean(pauses**2)))
            pause_db = 20 * np.log10(max(pause_rms, 1e-9))

            if speech_share >= min_speech:
                speech = y[mask]
                speech_rms = float(np.sqrt(np.mean(speech**2))) if len(speech) else 0.0
                speech_db = 20 * np.log10(max(speech_rms, 1e-9))
                if pause_db >= speech_db + rel_db:
                    return "separate"  # pauses too loud relative to speech
            else:
                if pause_db >= abs_db:
                    return "separate"  # barely any speech, but level present
    except Exception as e:
        # The prefilter must NEVER cost a job — separate when in doubt.
        if job and job_manager:
            job_manager.warn(job, f"Prefilter error ({str(e)[:120]}) — separating chunk to be safe")
        return "separate"

    # All windows skipped. Silence or clean speech?
    try:
        y, _ = librosa.load(str(chunk_path), sr=8000, mono=True, duration=min(chunk_dur, 300.0))
        rms = float(np.sqrt(np.mean(y**2)))
        if 20 * np.log10(max(rms, 1e-9)) < silence_db:
            return "silence"
    except Exception:
        pass
    return "clean"


# ══════════════════════════════════════════════════════════════════════════
#  Concat & Trim
# ══════════════════════════════════════════════════════════════════════════


def _concat_chunks(chunk_paths, output_path, job=None):
    """Merge WAV files with ffmpeg concat."""
    if len(chunk_paths) == 1:
        shutil.copy2(str(chunk_paths[0]), str(output_path))
        return
    concat_list = output_path.parent / f"{output_path.stem}_concat.txt"
    with open(concat_list, "w", encoding="utf-8") as f:
        for cp in chunk_paths:
            # #050: escape single quotes in the path for the ffmpeg concat
            # format ( '  ->  '\'' ), otherwise a path with ' breaks the concat.
            safe = str(cp).replace("'", "'\\''")
            f.write(f"file '{safe}'\n")
    # -rf64 auto: WAV has a 32-bit size field and is thus limited to 4 GiB.
    # At 48 kHz stereo s16 (192 kB/s) that's reached after 6.21h —
    # ffmpeg then writes a file whose header overflows ("Filesize
    # invalid for wav, output file will be broken"). ffprobe reads it to the
    # end, but soundfile only up to 22,369.62 s (= exactly 4 GiB). The result
    # was shifted timestamps and a truncated end.
    # "auto" only writes an RF64 header above the limit; below that it
    # remains an ordinary RIFF/WAV — so nothing changes for shorter files.
    r = run_cancellable(
        [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_list),
            "-acodec",
            "pcm_s16le",
            "-rf64",
            "auto",
            str(output_path),
        ],
        job=job,
        timeout=7200,
        label="Vocals concat",
    )
    concat_list.unlink(missing_ok=True)
    # #038: check returncode + output size — otherwise a silently failed
    # (partial) concat counts as valid output → corrupt/too-short file.
    if r.returncode != 0 or not output_path.exists() or output_path.stat().st_size <= 1000:
        raise RuntimeError(
            f"Concat failed (rc={r.returncode}, {len(chunk_paths)} chunks): "
            f"{(r.stderr or '')[-200:]}"
        )


def _trim_chunk(
    input_wav,
    output_wav,
    trim_start,
    trim_end,
    total_dur,
    job=None,
    sample_rate=None,
    channels=None,
):
    """Trim the overlap off a processed chunk.

    sample_rate/channels set → output is also converted
    (rebuild 3.0c: vocals chunks go into the concat as 16 kHz mono — saves
    ~18 GB of write/read load for a 13.6h file, and Whisper only needs
    16 kHz mono anyway). Then it is ALWAYS converted, even without a trim —
    otherwise a 48 kHz chunk would sit among all-16 kHz chunks in the concat.
    """
    convert = sample_rate is not None or channels is not None
    if trim_start <= 0.01 and trim_end <= 0.01 and not convert:
        # Copied, never handed back as is: the input lies in the separator's
        # output folder, which the caller deletes right after this call —
        # the concat then found nothing.
        shutil.copyfile(str(input_wav), str(output_wav))
        return output_wav
    # #036: measure the ACTUAL output duration of the separated file instead
    # of trusting the requested chunk_dur — the separator can slightly
    # change the length; otherwise the cut error accumulates across all
    # chunks (desync).
    actual_dur = get_duration_safe(input_wav)
    if actual_dur and actual_dur > 0:
        total_dur = actual_dur
    keep_start = trim_start
    keep_dur = total_dur - trim_start - trim_end
    if keep_dur <= 0:
        # Nothing of the chunk's own stretch is left — passing the
        # untrimmed file on would shift the timeline (#035), and it is
        # deleted with its folder anyway.
        raise RuntimeError(
            f"Chunk trim impossible: {total_dur:.2f}s minus overlap "
            f"{trim_start:.2f}s + {trim_end:.2f}s leaves nothing"
        )
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_wav),
        "-ss",
        f"{keep_start:.3f}",
        "-t",
        f"{keep_dur:.3f}",
    ]
    if sample_rate is not None:
        cmd += ["-ar", str(sample_rate)]
    if channels is not None:
        cmd += ["-ac", str(channels)]
    cmd += ["-acodec", "pcm_s16le", str(output_wav)]
    r = run_cancellable(cmd, job=job, timeout=300, label="Chunk trim")
    # #035: check returncode/existence — on a silent failure do NOT pass
    # through untrimmed (with overlap), that would shift the timeline (desync).
    if r.returncode != 0 or not output_wav.exists() or output_wav.stat().st_size <= 1000:
        raise RuntimeError(f"Chunk trim failed (rc={r.returncode}): {(r.stderr or '')[-200:]}")
    return output_wav


def _write_silence(output_wav, duration_sec, sample_rate=48000, channels=2, job=None):
    """Generate a silence WAV of exact length.

    #032: used when a chunk delivers NO vocals/stem file. Without this
    placeholder, the concat would pull the following chunks together
    seamlessly and shift the whole timeline (video/transcript desync).
    Silence of exactly the same length preserves the timeline
    (silence→silence is harmless).
    """
    layout = "mono" if channels == 1 else "stereo"
    dur = max(0.0, float(duration_sec))
    r = run_cancellable(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"anullsrc=r={sample_rate}:cl={layout}",
            "-t",
            f"{dur:.3f}",
            "-ac",
            str(channels),
            "-ar",
            str(sample_rate),
            "-acodec",
            "pcm_s16le",
            str(output_wav),
        ],
        job=job,
        timeout=300,
        label="Silence fill",
    )
    if r.returncode != 0 or not output_wav.exists():
        raise RuntimeError(
            f"Silence generation failed (rc={r.returncode}): {(r.stderr or '')[-200:]}"
        )
    return output_wav


# ══════════════════════════════════════════════════════════════════════════
#  Chunked Separation Pipeline
# ══════════════════════════════════════════════════════════════════════════


def no_vocals_text(n_chunks: int, separated: int) -> str:
    """Warning text when no section delivered voice — distinguished by CASE.

    Previously two very different cases got the same sentence ("No vocals
    after chunk processing, all chunks without vocals output"), and it
    sounded like an error even though usually there wasn't one:

      a) The prefilter classified EVERY section as silence — there was
         nothing to separate. The recording simply has no audio. Such
         files end with an empty transcript, and that's correct.
      b) Separation happened, and not a single separation delivered voice.
         **This** is the case worth looking into.

    The difference is in the prefilter balance and costs nothing. Anyone
    who later looks at a file's warnings shouldn't be sent in the wrong
    direction.
    """
    if separated <= 0:
        return (
            f"Entire recording without audio: all {n_chunks} sections are "
            f"silence — nothing to separate, transcript stays empty"
        )
    return (
        f"No vocals after chunk processing: {separated} sections "
        f"were separated, none delivered voice"
    )


def _note_silent(job) -> None:
    """Mark on the job that the recording holds nothing to separate — the
    prefilter found only silence. Not a failed separation: a separation-only
    run then ends without tracks, but without an error."""
    if job is not None:
        try:
            job.separation_silent = True
        except Exception:  # noqa: BLE001 — a note, nothing depends on it failing
            pass


def chunked_separate(
    source_file,
    work_dir,
    model_key="bs_roformer",
    job=None,
    elapsed_fn=None,
    single_stem=None,
    progress_base=10,
    progress_span=50,
    pass_label="Pass 1",
    sample_rate=48000,
    channels=2,
    full_vocals=False,
):
    """
    Robust chunked separation pipeline.

    ``full_vocals``: the vocals track in the extraction format (like the
    instrumental) instead of 16 kHz mono — for a track someone listens to.

    Returns: (vocals_path, instrumental_path)
    """
    models = config.get("vocal_isolation", "models", default={})
    default_model = config.get("vocal_isolation", "default_model", default="bs_roformer")
    model_file = models.get(model_key, models.get(default_model, FALLBACK_MODEL_FILE))
    # The chunk size IS the memory upper bound: audio-separator holds one
    # chunk entirely as float32 in RAM (input + vocals + instrumental).
    # 30 minutes ≈ 4 GB peak — regardless of how long the file is overall.
    # Anyone with less RAM lowers hardware.sep_chunk_duration_sec.
    sep_chunk_dur = config.get("hardware", "sep_chunk_duration_sec", default=1800)
    sep_overlap = config.get("hardware", "sep_overlap_sec", default=3)

    el = elapsed_fn() if elapsed_fn else ""
    if job and job_manager:
        job_manager.log(job, f"🎤 {pass_label}: Analyzing audio length…")

    total_dur = get_duration(source_file)
    source_size = source_file.stat().st_size
    need_chunks = total_dur > sep_chunk_dur + sep_overlap
    n_est = max(1, math.ceil(total_dur / sep_chunk_dur)) if need_chunks else 1

    if job and job_manager:
        job_manager.log(job, f"Source: {format_ts(total_dur)}, {format_size(source_size)}")
        if need_chunks:
            job_manager.log(
                job, f"Chunking: yes ({n_est} chunks of ~{sep_chunk_dur // 60}min each)"
            )
        else:
            job_manager.log(job, "Chunking: no (direct processing)")

    # ── Extract chunks ────────────────────────────────────────────
    _t0 = time.time()
    if job:
        job.update(
            progress=progress_base, message=f"🎤 {pass_label}: Extracting audio chunks… · {el}"
        )

    chunks, _, silence_idx = _extract_sep_chunks(
        source_file,
        work_dir,
        sep_chunk_dur,
        sep_overlap,
        job=job,
        sample_rate=sample_rate,
        channels=channels,
    )

    if job and job_manager:
        job_manager.record_step(
            job, f"{pass_label}: chunk extraction", time.time() - _t0, group=pass_label
        )

    n_chunks = len(chunks)

    if n_chunks == 0:
        if job and job_manager:
            job_manager.warn(job, f"{pass_label}: no audio chunks (silence?)")
        return None, None

    # ── Short file: direct ───────────────────────────────────────────
    if not need_chunks and n_chunks == 1:
        _t1 = time.time()
        _w1 = GATE.wait_total()  # waits for the separator are no work
        if job:
            job.update(progress=progress_base + 2, message=f"🎤 {pass_label}: {model_key}… · {el}")
        if job and job_manager:
            job_manager.log(job, f"Direct processing: {format_size(chunks[0][0].stat().st_size)}")

        # Prefilter also in the direct path (3.1a) — collections often
        # consist mostly of short files (voice memos etc.), and that's
        # exactly where the share of clean recordings is highest (a sizable
        # part of short-file separation time would be unnecessary). "clean" →
        # original becomes the vocals track, "silence" → nothing to get.
        _tv = time.time()
        class_direct = _chunk_prefilter(chunks[0][0], chunks[0][3], job=job)
        _prefilter_count(class_direct)
        logger.debug(
            "⏱ Prefilter (direct): %s (%.1fs, file %.0fs)",
            class_direct,
            time.time() - _tv,
            chunks[0][3],
        )
        if class_direct != "separate":
            if job and job_manager:
                job_manager.log(job, f"Prefilter: {class_direct} — separation skipped")
                job_manager.record_step(
                    job,
                    f"{pass_label}: prefilter (direct)",
                    time.time() - _t1 - GATE.wait_total() + _w1,
                    group=pass_label,
                )
            if class_direct == "silence":
                _note_silent(job)
                return None, None
            # Whisper needs 16 kHz mono; a track someone listens to keeps
            # the extraction format, as on the chunked path (otherwise
            # short clean recordings came out at 16 kHz mono next to the
            # file despite full_vocals)
            d_sr, d_ch = (sample_rate, channels) if full_vocals else (16000, 1)
            direct_vocals = work_dir / "vocals_direct.wav"
            r = run_cancellable(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(chunks[0][0]),
                    "-ar",
                    str(d_sr),
                    "-ac",
                    str(d_ch),
                    "-acodec",
                    "pcm_s16le",
                    "-rf64",
                    "auto",
                    str(direct_vocals),
                ],
                job=job,
                timeout=1800,
                label=f"Prefilter direct {d_sr // 1000}k",
            )
            if r.returncode == 0 and direct_vocals.exists() and direct_vocals.stat().st_size > 1000:
                if chunks[0][0].name != source_file.name:
                    chunks[0][0].unlink(missing_ok=True)
                return direct_vocals, None
            # Conversion failed → separate normally to be safe
            if job and job_manager:
                job_manager.warn(job, "Prefilter direct path failed — separating normally")

        out_dir = work_dir / "sep_direct"
        out_dir.mkdir(parents=True, exist_ok=True)
        _run_separator(chunks[0][0], out_dir, model_file, job=job, single_stem=single_stem)

        if job and job_manager:
            job_manager.record_step(
                job,
                f"{pass_label}: separation (direct)",
                time.time() - _t1 - GATE.wait_total() + _w1,
                group=pass_label,
            )

        if chunks[0][0].name != source_file.name:
            chunks[0][0].unlink(missing_ok=True)

        if single_stem:
            v = find_output_wav(out_dir, stem_hint=single_stem.lower())
            if not v and job and job_manager:
                out_files = list(out_dir.iterdir()) if out_dir.exists() else []
                job_manager.warn(
                    job, f"No '{single_stem}' file! Files: {[f.name for f in out_files]}"
                )
            return v, None
        else:
            v, inst = find_output_wavs(out_dir)
            if not v and job and job_manager:
                out_files = list(out_dir.iterdir()) if out_dir.exists() else []
                job_manager.warn(job, f"No vocals file! Files: {[f.name for f in out_files]}")
            return v, inst

    # ── Long file: chunks individually ───────────────────────────────────
    if job and job_manager:
        job_manager.log(job, f"{n_chunks} chunks extracted, starting separation…")

    processed_vocals = []
    processed_inst = []
    real_vocal_count = 0  # #032: real (not silence-padded) vocals chunks
    class_balance = {"separate": 0, "clean": 0, "silence": 0}
    chunk_out_base = work_dir / "sep_chunks_out"
    chunk_out_base.mkdir(parents=True, exist_ok=True)
    trimmed_dir = work_dir / "sep_trimmed"
    trimmed_dir.mkdir(parents=True, exist_ok=True)

    for idx, (chunk_path, trim_start, trim_end, chunk_dur) in enumerate(chunks):
        if job and job.cancel:
            raise InterruptedError("Cancelled")

        _tc = time.time()
        _wc = GATE.wait_total()  # waits for the separator are no work
        el = elapsed_fn() if elapsed_fn else ""
        pct = progress_base + 3 + int((idx / n_chunks) * (progress_span - 6))
        chunk_size = format_size(chunk_path.stat().st_size) if chunk_path.exists() else "?"

        if job:
            job.update(progress=pct, message=f"🎤 {pass_label}: Chunk {idx + 1}/{n_chunks}… · {el}")
        if job and job_manager:
            job_manager.log(
                job,
                f"Chunk {idx + 1}/{n_chunks}: "
                f"{format_ts(chunk_dur)}, {chunk_size} "
                f"(trim: {trim_start:.1f}s / {trim_end:.1f}s)",
            )

        chunk_out = chunk_out_base / f"chunk_{idx:04d}"
        # parents=True: without it, mkdir raises FileNotFoundError as soon
        # as the base folder is missing — and it can disappear during the
        # run (cache folder cleared, disk filled up). That's how a job
        # that already had an hour of GPU time behind it once died on a
        # directory that can be recreated in one line.
        chunk_out.mkdir(parents=True, exist_ok=True)

        # Rebuild 3.0c: vocals chunks go into the concat as 16 kHz mono —
        # Whisper doesn't need anything else anyway, and the intermediate
        # file shrinks from 9.4 to 1.57 GB for a 13.6h file. The
        # instrumental track (listening artifact) stays at extraction format.
        # Unless the vocals are a listening artifact too: saved next to the
        # file, they came out at 16 kHz mono beside a 44.1-kHz stereo
        # instrumental; a saved track should keep full quality. Whisper
        # gets its 16 kHz from the worker then.
        voc_sr, voc_ch = (sample_rate, channels) if full_vocals else (16000, 1)
        useful_duration = chunk_dur - trim_start - trim_end

        # Three-class prefilter (3.1a): silence / clean / separate.
        _tv = time.time()
        class_name = (
            "silence" if idx in silence_idx else _chunk_prefilter(chunk_path, chunk_dur, job=job)
        )
        class_balance[class_name] = class_balance.get(class_name, 0) + 1
        _prefilter_count(class_name)
        logger.debug(
            "⏱ Prefilter chunk %d/%d: %s (%.1fs, chunk %.0fs)",
            idx + 1,
            n_chunks,
            class_name,
            time.time() - _tv,
            chunk_dur,
        )
        if class_name == "separate" and job and job_manager:
            job_manager.log(
                job, f"Chunk {idx + 1}/{n_chunks}: prefilter → separate ({time.time() - _tv:.1f}s)"
            )

        if class_name != "separate":
            if class_name == "silence":
                # No audio at this point — separation would be GPU time for
                # guaranteed silence (and the separator rejects silent WAVs
                # anyway with "empty or not valid", task 1.2).
                if job and job_manager:
                    job_manager.log(
                        job, f"Chunk {idx + 1}/{n_chunks}: silence — separation skipped"
                    )
                _write_silence(
                    trimmed_dir / f"v_{idx:04d}.wav", useful_duration, voc_sr, voc_ch, job=job
                )
            else:
                # "clean": speech with silent pauses — there is nothing to
                # separate. The ORIGINAL becomes the vocals track; this
                # avoids both the separation AND its artifacts on clean
                # material (misheard words).
                if job and job_manager:
                    job_manager.log(
                        job,
                        f"Chunk {idx + 1}/{n_chunks}: clean "
                        f"speech — separation skipped, "
                        f"original becomes vocals track",
                    )
                real_vocal_count += 1
                _trim_chunk(
                    chunk_path,
                    trimmed_dir / f"v_{idx:04d}.wav",
                    trim_start,
                    trim_end,
                    chunk_dur,
                    job=job,
                    sample_rate=voc_sr,
                    channels=voc_ch,
                )
            processed_vocals.append(trimmed_dir / f"v_{idx:04d}.wav")
            if not single_stem:
                # 1.2 fix: also pad the instrumental track with silence —
                # otherwise its timeline would be shifted against the video.
                _write_silence(
                    trimmed_dir / f"i_{idx:04d}.wav",
                    useful_duration,
                    sample_rate,
                    channels,
                    job=job,
                )
                processed_inst.append(trimmed_dir / f"i_{idx:04d}.wav")
            chunk_path.unlink(missing_ok=True)
            if job and job_manager:
                job_manager.record_step(
                    job,
                    f"{pass_label}: chunk {idx + 1}/{n_chunks} (prefilter: {class_name})",
                    time.time() - _tc - GATE.wait_total() + _wc,
                    group=pass_label,
                )
            continue

        try:
            _ts = time.time()
            _run_separator(
                chunk_path, chunk_out, model_file, job=job, single_stem=single_stem, retry=1
            )
            _sep_s = time.time() - _ts
            logger.debug(
                "⏱ Separation chunk %d/%d: %.1fs (%.1f× real-time)",
                idx + 1,
                n_chunks,
                _sep_s,
                chunk_dur / _sep_s if _sep_s > 0 else 0,
            )
        except RuntimeError as e:
            if job and job_manager:
                job_manager.warn(job, f"Chunk {idx + 1} failed: {e}")
            raise

        if single_stem:
            voc = find_output_wav(chunk_out, stem_hint=single_stem.lower())
            if voc:
                real_vocal_count += 1
                trimmed = trimmed_dir / f"v_{idx:04d}.wav"
                processed_vocals.append(
                    _trim_chunk(
                        voc,
                        trimmed,
                        trim_start,
                        trim_end,
                        chunk_dur,
                        job=job,
                        sample_rate=voc_sr,
                        channels=voc_ch,
                    )
                )
            else:
                # #032: fill missing stem file with silence of exact length,
                # otherwise all following chunks drift over time (desync).
                # raw fraction = chunk_dur - trim_start - trim_end.
                if job and job_manager:
                    out_files = list(chunk_out.iterdir()) if chunk_out.exists() else []
                    job_manager.warn(
                        job,
                        f"Chunk {idx + 1}: no '{single_stem}' file! "
                        f"Filling with silence (timeline preserved). "
                        f"Files: {[f.name for f in out_files]}",
                    )
                silent = trimmed_dir / f"v_{idx:04d}.wav"
                _write_silence(silent, useful_duration, voc_sr, voc_ch, job=job)
                processed_vocals.append(silent)
        else:
            voc, inst = find_output_wavs(chunk_out)
            if job and job_manager:
                job_manager.log(
                    job,
                    f"  Chunk {idx + 1} assignment: "
                    f"vocals={voc.name if voc else 'NONE'}, "
                    f"inst={inst.name if inst else 'NONE'}",
                )
            if voc:
                real_vocal_count += 1
                trimmed_v = trimmed_dir / f"v_{idx:04d}.wav"
                processed_vocals.append(
                    _trim_chunk(
                        voc,
                        trimmed_v,
                        trim_start,
                        trim_end,
                        chunk_dur,
                        job=job,
                        sample_rate=voc_sr,
                        channels=voc_ch,
                    )
                )
            else:
                # #032: see _write_silence — fill missing vocals with
                # equal-length silence, otherwise the concat shifts the
                # timeline (desync).
                if job and job_manager:
                    out_files = list(chunk_out.iterdir()) if chunk_out.exists() else []
                    job_manager.warn(
                        job,
                        f"Chunk {idx + 1}: no vocals file! "
                        f"Filling with silence (timeline preserved). "
                        f"Files: {[f.name for f in out_files]}",
                    )
                silent_v = trimmed_dir / f"v_{idx:04d}.wav"
                _write_silence(silent_v, useful_duration, voc_sr, voc_ch, job=job)
                processed_vocals.append(silent_v)
            if inst:
                # Converted to the extraction format, like the silence that
                # stands in for skipped chunks: audio-separator writes
                # 44.1 kHz, the placeholders are 48 kHz, and the concat
                # demuxer does not resample — every 48-kHz piece played
                # 8.8 % too long and the track drifted off the video.
                trimmed_i = trimmed_dir / f"i_{idx:04d}.wav"
                processed_inst.append(
                    _trim_chunk(
                        inst,
                        trimmed_i,
                        trim_start,
                        trim_end,
                        chunk_dur,
                        job=job,
                        sample_rate=sample_rate,
                        channels=channels,
                    )
                )
            else:
                # 1.2 fix: also pad when the instrumental stem is missing,
                # otherwise the instrumental timeline shifts.
                silent_i = trimmed_dir / f"i_{idx:04d}.wav"
                _write_silence(silent_i, useful_duration, sample_rate, channels, job=job)
                processed_inst.append(silent_i)

        chunk_path.unlink(missing_ok=True)
        shutil.rmtree(chunk_out, ignore_errors=True)

        if job and job_manager:
            job_manager.record_step(
                job,
                f"{pass_label}: chunk {idx + 1}/{n_chunks}",
                time.time() - _tc - GATE.wait_total() + _wc,
                group=pass_label,
            )
            job_manager.log(job, f"Chunk {idx + 1}/{n_chunks} ✓")

    # Prefilter balance of the file — the one line where you read off the
    # gain (a large share of separation time hangs on the clean/silence skips).
    _balance = (
        f"Prefilter balance: {class_balance['separate']} separated, "
        f"{class_balance['clean']} clean-skipped, "
        f"{class_balance['silence']} silence"
    )
    logger.info("⏱ %s (%d chunks)", _balance, n_chunks)
    if job and job_manager:
        job_manager.log(job, _balance)

    # #032: processed_vocals is never empty due to the silence padding;
    # that's why we check for REAL vocals. If NO chunk delivered real
    # vocals, separation was completely unsuccessful → treat as "no
    # vocals" as before (instead of a complete silence track).
    if real_vocal_count == 0:
        # Two very different cases that used to get the same sentence —
        # and it sounded like an error even though usually there wasn't one:
        #
        #   a) The prefilter classified EVERY section as silence; there
        #      was simply nothing to separate. The recording has no
        #      audio. Such files end with an empty transcript —
        #      correctly so.
        #   b) Separation happened, and not a single separation delivered
        #      voice. THIS is the case worth looking into.
        #
        # Writing "No vocals after chunk processing" for both sends
        # anyone who later looks at the warnings in the wrong direction —
        # the difference is in the balance and costs nothing.
        if class_balance.get("separate", 0) <= 0:
            _note_silent(job)
        text = no_vocals_text(n_chunks, class_balance.get("separate", 0))
        if job and job_manager:
            job_manager.warn(job, text)
        logger.warning("%s", text)
        return None, None

    # ── Merge all chunks ─────────────────────────────────────
    _t_concat = time.time()
    el = elapsed_fn() if elapsed_fn else ""
    if job:
        job.update(
            progress=progress_base + progress_span - 3, message=f"🎤 {pass_label}: Merging… · {el}"
        )
    if job and job_manager:
        job_manager.log(
            job,
            f"Concat: {len(processed_vocals)} vocals chunks"
            + (f", {len(processed_inst)} instrumental chunks" if processed_inst else ""),
        )

    final_vocals = work_dir / "vocals_combined.wav"
    _concat_chunks(processed_vocals, final_vocals, job=job)

    final_inst = None
    if processed_inst:
        final_inst = work_dir / "instrumental_combined.wav"
        _concat_chunks(processed_inst, final_inst, job=job)

    shutil.rmtree(trimmed_dir, ignore_errors=True)

    if job and job_manager:
        job_manager.record_step(
            job, f"{pass_label}: merge", time.time() - _t_concat, group=pass_label
        )
        v_size = format_size(final_vocals.stat().st_size) if final_vocals.exists() else "?"
        job_manager.log(
            job,
            f"🎤 {pass_label} done: vocals={v_size}"
            + (
                f", instrumental={format_size(final_inst.stat().st_size)}"
                if final_inst and final_inst.exists()
                else ""
            ),
        )

    return final_vocals, final_inst


# ══════════════════════════════════════════════════════════════════════════
#  High-Level Vocal Isolation
# ══════════════════════════════════════════════════════════════════════════


def isolate_vocals(
    source_file, work_dir, job=None, elapsed_fn=None, model_key="bs_roformer", full_vocals=False
):
    """
    Vocal isolation: separates speech from music/effects.

    ALWAYS chunked. The chunk size is fixed in `hardware.sep_chunk_duration_sec`.

    Why always chunk instead of "only for long files":
    audio-separator decodes the entire audio as float32 and holds input,
    vocals AND instrumental in memory simultaneously (~384 KB per second of
    audio, roughly six times that with the STFT intermediate steps).
    Consumption therefore grows LINEARLY with duration — a 3-hour recording
    needs over 20 GB in one go and fills up the host.

    Previously a fixed duration threshold (3h) decided whether it's computed
    in one go; it was simply set too high. Estimating free RAM would be the
    obvious fix, but it's worse than it sounds: the result would depend on
    whatever else happens to be running on the machine (the same file would
    behave differently each time), and the whole calculation would stand or
    fall on a guessed factor.

    A fixed chunk size instead gives a CONSTANT upper bound, no matter how
    long the file is: 30 minutes ≈ 4 GB peak, always.

    Files shorter than one chunk still run through in one piece
    (_extract_sep_chunks doesn't even split them) — nothing changes for those.

    Returns: (vocals_path, instrumental_path)
    """
    iso_work = work_dir / "vocal_iso_work"
    iso_work.mkdir(parents=True, exist_ok=True)

    models = config.get("vocal_isolation", "models", default={})
    default_model = config.get("vocal_isolation", "default_model", default="bs_roformer")
    model_file = models.get(model_key, models.get(default_model, FALLBACK_MODEL_FILE))
    chunk_dur = config.get("hardware", "sep_chunk_duration_sec", default=1800)

    # #053: safe duration query (does not raise).
    total_dur = get_duration_safe(source_file)

    if job and job_manager:
        job_manager.log(job, f"🎤 Vocal Isolation: {source_file.name}")
        job_manager.log(
            job,
            f"Duration: {format_ts(total_dur)}, Size: {format_size(source_file.stat().st_size)}",
        )
        job_manager.log(job, f"Model: {model_file}, Chunk size: {chunk_dur // 60}min")

    try:
        return _isolate_chunked(
            source_file,
            iso_work,
            model_key,
            job=job,
            elapsed_fn=elapsed_fn,
            full_vocals=full_vocals,
        )
    finally:
        # 3.0a: default is RESIDENT (model stays loaded for the next file,
        # saves 4.8s per call). Only in low-VRAM mode is it unloaded after
        # every file — the old behavior for small cards.
        if _low_vram():
            release_separator()


def _isolate_chunked(
    source_file, work_dir, model_key, job=None, elapsed_fn=None, full_vocals=False
):
    """Separation in fixed-size chunks — the only path (see the caller).

    A file shorter than one chunk is not split, which the log reports as
    "Chunking: no". A warning here used to claim "direct run doesn't fit in
    RAM" for every file, from the time a RAM estimate still chose between
    two paths.
    """
    vocals_path, inst_path = chunked_separate(
        source_file,
        work_dir,
        model_key=model_key,
        job=job,
        elapsed_fn=elapsed_fn,
        single_stem=None,
        progress_base=8,
        progress_span=45,
        pass_label="Vocal Isolation",
        full_vocals=full_vocals,
    )
    if not vocals_path or not vocals_path.exists():
        return None, None
    return vocals_path, inst_path
