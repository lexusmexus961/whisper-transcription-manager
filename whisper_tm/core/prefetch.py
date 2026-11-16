"""
The models the settings need, downloaded before the first job.

Whisper, the alignment model, NLTK's sentence splitter and the separation
model each download on first use — offline, behind a blocked host or a
proxy, the first job ended in a library's raw error ("Max retries exceeded
with url", NLTK's "Resource punkt_tab not found"). ``whisper-tm
--prefetch`` (``./start.sh --prefetch``; setup.sh offers it) fetches them
now and says for each what it fetched from where, or why not; at job time
:func:`download_error` turns a failed download into the same kind of
message.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass

from whisper_tm.config import config

logger = logging.getLogger(__name__)

#: Download sizes (MB) of the Whisper models, for the setup's question
WHISPER_MB = {
    "tiny": 75,
    "base": 145,
    "small": 485,
    "medium": 1530,
    "large-v1": 3090,
    "large-v2": 3090,
    "large-v3": 3090,
    "large": 3090,
    "turbo": 1620,
    "large-v3-turbo": 1620,
    "distil-large-v3": 1510,
}

#: What a failed download says, in the libraries' words (the exception
#: chain is read: huggingface_hub wraps requests, which wraps urllib3)
_NETWORK = (
    "max retries exceeded",
    "temporary failure in name resolution",
    "name or service not known",
    "network is unreachable",
    "connection refused",
    "connection reset",
    "connection aborted",
    "timed out",
    "unable to connect to proxy",
    "proxyerror",
    "urlopen error",
    "localentrynotfounderror",
    "cannot find the requested files",
    "outgoing traffic has been disabled",
    "check your internet connection",
    "proxied fetch",
    "certificate verify failed",
)


@dataclass
class Model:
    what: str  # "Whisper large-v3"
    source: str  # "huggingface.co/Systran/faster-whisper-large-v3"
    fetch: Callable[[], object] | None
    mb: int = 0  # download size, roughly
    why_not: str = ""  # not needed / can't be fetched now — and why


def _chain(error: BaseException):
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        yield error
        error = error.__cause__ or error.__context__


def is_download_failure(error: BaseException) -> bool:
    """Did a model fail to load because it couldn't be downloaded?"""
    for e in _chain(error):
        text = f"{type(e).__name__}: {e}".lower()
        if any(needle in text for needle in _NETWORK):
            return True
    return False


def hint(error: BaseException) -> str:
    """What to do about a failed download."""
    from whisper_tm.core import commands

    text = " ".join(str(e) for e in _chain(error)).lower()
    if "proxied fetch" in text:
        return "NLTK downloads nothing through a proxy unless NLTK_ALLOW_PROXIED_URLOPEN=1 is set"
    if "certificate verify failed" in text:
        return "the TLS certificate was refused (a proxy or a filter in between?)"
    return (
        f"no internet connection? With one, {commands.start('--prefetch')} "
        "downloads every model the settings need"
    )


def download_error(what: str, source: str, error: BaseException) -> RuntimeError:
    """A model's first download failed — said as that. The library's own
    words stay at the end; "could not be downloaded from" is what the queue
    pauses on (queue.ENVIRONMENT_ERRORS)."""
    first = str(error).strip().splitlines()[0][:240] if str(error).strip() else type(error).__name__
    return RuntimeError(f"{what} could not be downloaded from {source} — {hint(error)} ({first})")


# ── The models ────────────────────────────────────────────────────────────


def whisper_source(name: str) -> str:
    try:
        from faster_whisper.utils import _MODELS

        repo = _MODELS.get(name, name)
    except Exception:  # noqa: BLE001 — only for the message
        repo = name
    return f"huggingface.co/{repo}"


def _whisper(name: str) -> None:
    from faster_whisper.utils import download_model

    download_model(name)


def _alignment_name(lang: str) -> str:
    from whisperx.alignment import DEFAULT_ALIGN_MODELS_HF, DEFAULT_ALIGN_MODELS_TORCH

    return DEFAULT_ALIGN_MODELS_TORCH.get(lang) or DEFAULT_ALIGN_MODELS_HF.get(lang) or ""


def _alignment(name: str) -> None:
    """As whisperx.load_align_model fetches it — on the CPU, then let go."""
    import torchaudio

    if name in torchaudio.pipelines.__all__:
        torchaudio.pipelines.__dict__[name].get_model()
        return
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

    Wav2Vec2Processor.from_pretrained(name)
    Wav2Vec2ForCTC.from_pretrained(name)


def punkt() -> None:
    """NLTK's sentence splitter, which whisperx's alignment loads (and
    downloads, failing with a LookupError when it can't)."""
    import nltk

    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True, raise_on_error=True)
        nltk.data.find("tokenizers/punkt_tab")


PUNKT_SOURCE = "raw.githubusercontent.com (NLTK data)"


def _separation(model_file: str) -> None:
    import tempfile

    from whisper_tm.core.vocal_isolation import separation_model_dir

    folder = separation_model_dir()
    # Downloaded already: audio-separator would still ask GitHub for its
    # list of models first — and fail offline
    if (folder / model_file).exists():
        return
    from audio_separator.separator import Separator

    with tempfile.TemporaryDirectory() as out:
        Separator(
            output_dir=out, model_file_dir=str(folder), log_level=logging.WARNING
        ).download_model_files(model_file)


def models() -> list[Model]:
    """What the configured defaults need, in the order a job loads it."""
    from whisper_tm.core import commands

    out: list[Model] = []
    name = str(config.get("transcription", "default_model", default="large-v3") or "large-v3")
    out.append(
        Model(
            f"Whisper {name}", whisper_source(name), lambda: _whisper(name), WHISPER_MB.get(name, 0)
        )
    )

    lang = str(config.get("transcription", "default_language", default="en") or "auto")
    if lang == "auto":
        out.append(
            Model(
                "Alignment",
                "",
                None,
                why_not=(
                    "the language is detected per file — its model downloads with the "
                    "first file of that language"
                ),
            )
        )
    else:
        try:
            align = _alignment_name(lang)
        except Exception as e:  # noqa: BLE001 — whisperx not importable
            align = ""
            logger.debug("alignment models: %s", e)
        if align:
            import torchaudio

            torch_model = align in torchaudio.pipelines.__all__
            out.append(
                Model(
                    f"Alignment for {lang} ({align})",
                    "download.pytorch.org" if torch_model else f"huggingface.co/{align}",
                    lambda: _alignment(align),
                    360 if torch_model else 1200,
                )
            )
        else:
            out.append(
                Model(
                    f"Alignment for {lang}",
                    "",
                    None,
                    why_not="no alignment model for this language — segment times only",
                )
            )
    out.append(Model("Sentence splitter (NLTK punkt_tab)", PUNKT_SOURCE, punkt, 40))

    if config.get("vocal_isolation", "default_enabled", default=True):
        key = str(config.get("vocal_isolation", "default_model", default="mel_roformer_kim"))
        files = config.get("vocal_isolation", "models", default={}) or {}
        file = files.get(key) or key
        out.append(
            Model(
                f"Vocal isolation ({key})",
                "github.com (UVR model repository)",
                lambda: _separation(file),
                870,
            )
        )
    else:
        out.append(Model("Vocal isolation", "", None, why_not="off by default — not needed"))

    if config.get("transcription", "default_diarize", default=False):
        from whisper_tm.core import hf_access
        from whisper_tm.core.speaker import REDIMNET_URL, _redimnet_weights

        out.append(
            Model("Voice vectors (ReDimNet)", REDIMNET_URL.split("/")[2], _redimnet_weights, 20)
        )
        token = str(config.get("transcription", "hf_token", default="") or "").strip()
        if token or hf_access.model_cached():
            out.append(
                Model(
                    "Speaker diarization (pyannote)",
                    f"huggingface.co/{hf_access.REPO}",
                    lambda: hf_access.download(token),
                    hf_access.MODEL_MB,
                )
            )
        else:
            out.append(
                Model(
                    "Speaker diarization (pyannote)",
                    "",
                    None,
                    why_not=(f"needs the Hugging Face token first: {commands.start('--hf-token')}"),
                )
            )
    else:
        out.append(Model("Speaker diarization", "", None, why_not="off by default — not needed"))
    return out


def total_mb() -> int:
    return sum(m.mb for m in models() if m.fetch)


def _say(mark: str, text: str) -> None:
    print(f"  {mark} {text}", flush=True)


def cli() -> int:
    """``whisper-tm --prefetch``: download what the settings need. Exit
    code 1 when something could not be downloaded."""
    if not config._loaded:
        config.load(None)
    # The model the app will start with on THIS machine, not the shipped one
    try:
        from whisper_tm.hardware import apply_machine_defaults

        _, chosen = apply_machine_defaults()
        print(f"\n{chosen}")
    except Exception as e:  # noqa: BLE001 — then the configured defaults
        logger.debug("machine defaults: %s", e)
    print()
    print("Models the settings need")
    print("=" * 50)
    failed = 0
    for m in models():
        if m.fetch is None:
            _say("·", f"{m.what} — {m.why_not}")
            continue
        t0 = time.time()
        _say("…", f"{m.what} — from {m.source}")
        try:
            m.fetch()
        except Exception as e:  # noqa: BLE001 — every model gets its line
            failed += 1
            why = download_error(m.what, m.source, e) if is_download_failure(e) else e
            _say("✗", str(why)[:400])
            continue
        _say("✓", f"{m.what} — ready ({time.time() - t0:.0f} s)")
    if failed:
        _say("·", "The first job tries again; so does this command.")
    return 1 if failed else 0
