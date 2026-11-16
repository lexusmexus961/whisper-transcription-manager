"""Models downloaded before the first job, and failed downloads said as such.

Whisper, the alignment model, NLTK's sentence splitter and the separation
model each downloaded on first use. Offline the first job ended in a
library's raw error — "Max retries exceeded with url", or NLTK's "Resource
punkt_tab not found", which failed every alignment block and the file with
it. ``whisper-tm --prefetch`` downloads what the settings need; at job time
a failed download names the model and where it should have come from, and
the queue pauses on it.
"""

from __future__ import annotations

import subprocess
import sys
import types
from pathlib import Path

import pytest

from whisper_tm.config import config
from whisper_tm.core import prefetch, queue

ROOT = Path(__file__).resolve().parents[1]
OFFLINE = (
    "HTTPSConnectionPool(host='huggingface.co', port=443): Max retries exceeded "
    "with url: /api/models/Systran/faster-whisper-large-v3/revision/main"
)


@pytest.fixture
def settings(monkeypatch):
    def set_(section, key, value):
        monkeypatch.setitem(config._data.setdefault(section, {}), key, value)

    set_("transcription", "default_model", "large-v3")
    set_("transcription", "default_language", "en")
    set_("transcription", "default_diarize", False)
    set_("vocal_isolation", "default_enabled", True)
    return set_


def _names(models):
    return [(m.what, bool(m.fetch)) for m in models]


def test_what_the_defaults_need(settings):
    pytest.importorskip("whisperx")
    got = prefetch.models()
    names = _names(got)
    assert names[0] == ("Whisper large-v3", True)
    assert got[0].source == "huggingface.co/Systran/faster-whisper-large-v3"
    assert names[1] == ("Alignment for en (WAV2VEC2_ASR_BASE_960H)", True)
    assert ("Sentence splitter (NLTK punkt_tab)", True) in names
    assert ("Vocal isolation (mel_roformer_kim)", True) in names
    assert ("Speaker diarization", False) in names  # off by default
    assert prefetch.total_mb() == 3090 + 360 + 40 + 870  # ~4.4 GB


def test_what_other_settings_need(settings):
    settings("transcription", "default_language", "auto")
    settings("vocal_isolation", "default_enabled", False)
    settings("transcription", "default_diarize", True)
    settings("transcription", "hf_token", "")
    from whisper_tm.core import hf_access

    cached = hf_access.model_cached()
    got = {m.what: m for m in prefetch.models()}
    assert got["Alignment"].fetch is None and "detected per file" in got["Alignment"].why_not
    assert got["Vocal isolation"].fetch is None
    assert got["Voice vectors (ReDimNet)"].fetch is not None
    pyannote = got["Speaker diarization (pyannote)"]
    assert (pyannote.fetch is not None) == cached
    if not cached:
        assert "--hf-token" in pyannote.why_not


def test_a_failed_download_is_said_as_one():
    error = RuntimeError("could not fetch")
    error.__cause__ = ConnectionError(OFFLINE)
    assert prefetch.is_download_failure(error)
    said = str(
        prefetch.download_error(
            "The Whisper model large-v3", "huggingface.co/Systran/faster-whisper-large-v3", error
        )
    )
    assert said.startswith(
        "The Whisper model large-v3 could not be downloaded from "
        "huggingface.co/Systran/faster-whisper-large-v3 — no internet"
    )
    assert "--prefetch" in said
    # …and the queue pauses on it instead of failing every file
    assert queue.environment_cause(said) == "A model could not be downloaded (no internet?)"


def test_other_errors_are_not_download_failures():
    assert not prefetch.is_download_failure(RuntimeError("CUDA out of memory"))
    assert not prefetch.is_download_failure(ValueError("Invalid model size 'large-v9'"))


def test_nltk_behind_a_proxy(monkeypatch):
    """Newer NLTK refuses to download through a proxy — said, with the switch."""
    nltk = pytest.importorskip("nltk")

    def missing(_):
        raise LookupError("Resource punkt_tab not found.")

    def refuse(*a, **k):
        raise ValueError(
            "Error loading punkt_tab: Security Violation [pathsec.urlopen]: "
            "refusing a proxied fetch of 'https://raw.githubusercontent.com/…'"
        )

    monkeypatch.setattr(nltk.data, "find", missing)
    monkeypatch.setattr(nltk, "download", refuse)
    with pytest.raises(ValueError) as caught:
        prefetch.punkt()
    assert prefetch.is_download_failure(caught.value)
    assert "NLTK_ALLOW_PROXIED_URLOPEN=1" in prefetch.hint(caught.value)


def test_the_command_says_each_model(monkeypatch, capsys):
    def offline():
        raise ConnectionError(OFFLINE)

    monkeypatch.setattr(
        prefetch,
        "models",
        lambda: [
            prefetch.Model("Whisper tiny", "huggingface.co/x", lambda: None, 75),
            prefetch.Model("Alignment for en", "download.pytorch.org", offline, 360),
            prefetch.Model("Vocal isolation", "", None, why_not="off by default — not needed"),
        ],
    )
    monkeypatch.setattr(config, "_loaded", True, raising=False)
    assert prefetch.cli() == 1
    out = capsys.readouterr().out
    assert "✓ Whisper tiny — ready" in out
    assert "✗ Alignment for en could not be downloaded from download.pytorch.org" in out
    assert "· Vocal isolation — off by default — not needed" in out


def test_a_whisper_model_that_cannot_be_downloaded(monkeypatch):
    from whisper_tm.core import transcription as tx

    class Offline:
        def __init__(self, *a, **k):
            raise RuntimeError(OFFLINE)

    monkeypatch.setattr(
        tx, "import_clean", lambda name: types.SimpleNamespace(WhisperModel=Offline)
    )
    with pytest.raises(
        RuntimeError,
        match="The Whisper model large-v3 could not be downloaded "
        "from huggingface.co/Systran/faster-whisper-large-v3",
    ):
        tx._whisper_weights("large-v3", "cpu", 0, "default")


def test_the_switch_exists():
    out = subprocess.run(
        [sys.executable, "-m", "whisper_tm", "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    ).stdout
    assert "--prefetch" in out
