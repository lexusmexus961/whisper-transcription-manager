"""The child process that performs diarization — killable without affecting the app.

Invocation (never by hand, always via :mod:`whisper_tm.core.diar_process`)::

    python -m whisper_tm.core.diar_worker cuda

**Why a separate process at all?**

pyannote runs without a time limit. If it hangs, the entire job hangs: Python cannot
kill a thread, and simply abandoning the call would leave the CUDA context
orphaned — the GPU would stay occupied until the server restarts. With a
batch run over tens of thousands of files this is not an edge case, but a
complete failure overnight.

A separate process solves exactly this: ``kill()`` terminates it hard, the
operating system cleans up its CUDA context completely, the parent process continues
running. The price would be reloading the model — therefore this process is
**long-lived**: It loads once and then serves request after request.

Measured (a 12 GB card): process startup to fully loaded model
**4.96 s** (1.01 s ``import torch``, 3.12 s ``import whisperx.diarize``,
0.83 s Pipeline). A disposable process per file would cost, for 100,000 files,
approximately **138 hours** additional — therefore long-lived, not per request new.

**The protocol** is line-wise JSON: requests come on stdin, responses
go to stdout, everything else (log messages from pyannote, torch warnings,
tracebacks) to stderr.

Requests::

    {"kind": "diarize", "wav": "/path/audio.wav",
     "min_speakers": 2, "max_speakers": 4}
    {"kind": "end"}

Responses::

    {"kind": "ready", "vram_cap_mb": 4096}  — once only, model is ready
    {"kind": "result", "ok": true,  "segments": [[start, end, "SPEAKER_00"], …]}
    {"kind": "result", "ok": false, "error": "…", "trace": "…"}

A result carries ``"uncapped": true`` when the request needed more than
the VRAM limit and ran once without it (see ``_set_vram_cap``).

The segments are **lists, not objects**: with a long recording these are
a few thousand entries, and ``["start","end","speaker"]`` three times per
line to repeat would be three times the data volume without any benefit.
"""

from __future__ import annotations

import gc
import json
import os
import sys
import traceback

# ── stdout belongs to the protocol, nobody else ─────────────────────────
#
# A single `print` line from a library would destroy the response stream
# — the parent process reads line by line JSON. pyannote and torch
# do log to stderr, but this is nothing one should rely on (tqdm, progress bars,
# debug prints in dependencies).
#
# Therefore: take a private copy of the real stdout, and then redirect file descriptor 1
# to stderr. Anyone from here on who writes to stdout ends up in the
# log — not in the protocol.
_log = os.fdopen(os.dup(1), "w", buffering=1)
os.dup2(2, 1)


def _send(obj: dict) -> None:
    # ASCII: the pipe is in Windows' ANSI code page, and a path in an
    # error text with a character outside it ended the worker
    _log.write(json.dumps(obj) + "\n")
    _log.flush()


# Set by diar_process from hardware.diarization_vram_cap_mb
_CAP_ENV = "WTM_DIAR_VRAM_CAP_MB"


def _cap_fraction(cap_mb: int, total_bytes: int) -> float:
    """The share of the card that ``cap_mb`` MiB is — 1.0 means no limit."""
    if cap_mb <= 0 or total_bytes <= 0:
        return 1.0
    return min(1.0, cap_mb * 2**20 / total_bytes)


def _set_vram_cap(device: str, cap_mb: int) -> int:
    """Limit this process's VRAM to ``cap_mb`` MiB (0 lifts the limit).

    pyannote itself needs about 1.6 GiB, whatever the recording's length
    (measured: a batch of short voice notes, and a two-hour recording). But for the
    convolutions of the speaker-embedding model cuDNN takes, per input
    shape, the first algorithm whose workspace it can get — on a free card
    one with a 9.2 GiB workspace, fetched in a single allocation inside the
    WeSpeaker ResNet. The worker stood at 10 GiB for a few-second voice
    note, the card at 11.9 of 12 GiB.

    Limited, that allocation fails inside PyTorch and cuDNN falls back to
    the next, frugal algorithm: the same segments for every file, and
    faster — 66 s instead of 88 s, since 9 GiB are no longer fetched and
    returned per file. (``CUDNN_CONV_WSCAP_DBG`` was tried first: the cuDNN
    v8 path of PyTorch ignores it.)

    Returns the limit in effect, 0 when there is none.
    """
    if not str(device).startswith("cuda"):
        return 0
    try:
        import torch

        dev = torch.device(device)
        index = dev.index if dev.index is not None else torch.cuda.current_device()
        total = torch.cuda.get_device_properties(index).total_memory
        torch.cuda.set_per_process_memory_fraction(_cap_fraction(cap_mb, total), index)
        return max(cap_mb, 0)
    except Exception as e:
        print(f"VRAM limit not set: {e}", file=sys.stderr)
        return 0


def _is_oom(e: BaseException | None) -> bool:
    """CUDA out of memory — also when pyannote re-raised it as its own.

    pyannote's Inference turns it into ``MemoryError("batch_size (32) is
    probably too large …")``, without the words; the original stays
    attached as the context, so the whole chain is looked at.
    """
    for _ in range(10):
        if e is None:
            return False
        if "out of memory" in str(e).lower():
            return True
        e = e.__cause__ or e.__context__
    return False


def _release_cuda_cache(device: str) -> None:
    """Hand the allocator's unused blocks back to the driver.

    The model stays loaded; only the cache goes. Without this the worker
    kept what pyannote peaked at for the whole next file — measured 9.4 GiB
    after a 4.6-minute recording, which left the next separation 83 MiB and
    an out-of-memory retry on every second file of a voices-only queue.
    (That peak was cuDNN's convolution workspace, which the VRAM limit of
    ``_set_vram_cap`` now keeps from arising at all.)
    """
    if not str(device).startswith("cuda"):
        return
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass


def main() -> int:
    device = sys.argv[1] if len(sys.argv) > 1 else "cpu"

    # The imports cost approximately 4.1 s total and happen exactly once.
    import warnings

    # Same filter as in the parent process: pyannote/torch give per
    # diarization a std()-DoF hint.
    warnings.filterwarnings("ignore", message="std\\(\\): degrees of freedom")

    try:
        cap_mb = int(float(os.environ.get(_CAP_ENV) or 0))
    except ValueError:
        cap_mb = 0

    try:
        import whisperx
        from whisperx.diarize import DiarizationPipeline

        # Before the model: from its first allocation on, the limit holds
        cap_mb = _set_vram_cap(device, cap_mb) if cap_mb > 0 else 0

        # HF_HUB_OFFLINE is set by the parent process via the environment — here this is
        # the right way (unlike in the running server), because
        # huggingface_hub is only imported now and the constant is read from
        # the environment at import time.
        pipe = DiarizationPipeline(token=os.environ.get("HF_TOKEN") or None, device=device)
    except Exception as e:
        _send(
            {
                "kind": "result",
                "ok": False,
                "error": f"Diarization model cannot be loaded: {e}",
                "trace": traceback.format_exc()[-2000:],
            }
        )
        return 1

    _send({"kind": "ready", "vram_cap_mb": cap_mb})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            job = json.loads(line)
        except Exception as e:
            _send({"kind": "result", "ok": False, "error": f"Unreadable request: {e}"})
            continue

        if job.get("kind") == "end":
            break

        try:
            kwargs = {}
            if job.get("min_speakers"):
                kwargs["min_speakers"] = int(job["min_speakers"])
            if job.get("max_speakers"):
                kwargs["max_speakers"] = int(job["max_speakers"])

            # Audio is loaded HERE, not sent through the pipe: one
            # hour of 16-kHz mono is 220 MiB in memory, decoding
            # costs measured 0.6–1.7 s. Through a pipe it would be slower and
            # significantly more error-prone.
            audio = whisperx.load_audio(str(job["wav"]))
            uncapped = False
            try:
                df = pipe(audio, **kwargs)
            except Exception as e:
                # The limit is aimed at cuDNN's appetite, not at a real
                # need: a request that genuinely needs more runs once
                # without it instead of failing.
                if not cap_mb or not _is_oom(e):
                    raise
                _release_cuda_cache(device)
                _set_vram_cap(device, 0)
                try:
                    df = pipe(audio, **kwargs)
                finally:
                    _release_cuda_cache(device)
                    _set_vram_cap(device, cap_mb)
                uncapped = True

            segments = [
                [float(r["start"]), float(r["end"]), str(r["speaker"])] for _, r in df.iterrows()
            ]
            # Release audio and result IMMEDIATELY.
            #
            # The worker is long-lived — that is its purpose —, and precisely
            # because of this it holds the audio of the last request without this `del`
            # until the next. With a 20-hour recording that's
            # 4.6 GB (20 h × 16 kHz × 4 bytes as float32).
            #
            # Measured after two files:
            # **5.93 GiB RSS idle**, while the parent process
            # simultaneously held its model and the separator. On a
            # 32-GB machine this is the difference between "fits" and
            # "swaps".
            del audio, df
            gc.collect()
            _release_cuda_cache(device)
            _send(
                {
                    "kind": "result",
                    "ok": True,
                    "segments": segments,
                    **({"uncapped": True} if uncapped else {}),
                }
            )
        except Exception as e:
            _release_cuda_cache(device)
            # A failed request does NOT end the worker — the model
            # is loaded, and the next request can easily succeed
            # (corrupt file, impossible speaker count). Only a crash or
            # a kill() ends it.
            error = str(e)[:500]
            # pyannote's re-raised OOM lacks the words; say them, so the
            # queue waits for free VRAM and retries instead of booking an
            # error
            if _is_oom(e) and "out of memory" not in error.lower():
                error = f"CUDA out of memory — {error}"
            _send(
                {
                    "kind": "result",
                    "ok": False,
                    "error": error,
                    "trace": traceback.format_exc()[-2000:],
                }
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
