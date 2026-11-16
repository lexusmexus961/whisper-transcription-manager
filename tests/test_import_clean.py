"""A first import must not keep its caller's frames — and models — alive.

Symptom: VRAM stayed in use after an unload. "Unload VRAM" freed the
separator, but large-v3's weights — about 4 GB — stayed on the card until
the server was restarted, every time.

The weights sat in a closure cell of ``_asr_pipeline``'s frame, and that
frame was alive: the first ``import whisperx.asr`` ran inside it (whisperx
imports its parts on first use), and down that import optuna (through
pyannote) tried ``import redis`` and *kept* the failure to raise it later
(``optuna._imports.try_import``). An exception keeps its traceback; a
traceback keeps every frame above it — the whole stack of whoever imported
first, locals and all, for as long as the process runs.

``helpers.import_clean`` imports from a thread of its own: the frames kept
are that thread's, which hold nothing. These tests rebuild the mechanism
without a GPU, and pin that the transcription imports its libraries that
way.
"""

from __future__ import annotations

import gc
import importlib
import sys
import weakref
from pathlib import Path

import pytest

from whisper_tm.core.helpers import import_clean

REPO = Path(__file__).resolve().parent.parent

# A module that keeps the failure of an optional import — as optuna does
STASH = """
try:
    import a_module_that_is_not_there_at_all  # noqa: F401
except ImportError as e:
    KEPT = e
"""


@pytest.fixture
def stash(tmp_path, monkeypatch):
    for name in ("stash_plain", "stash_clean"):
        (tmp_path / f"{name}.py").write_text(STASH, encoding="utf-8")
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.syspath_prepend(str(tmp_path))
    yield
    for name in ("stash_plain", "stash_clean"):
        sys.modules.pop(name, None)


class Weights:
    """Stands for a model: what must be freed."""


def _held_after(importer, name: str) -> bool:
    """Import ``name`` from a frame that holds a model, let go of the model —
    is it still alive?"""
    weights = Weights()
    ref = weakref.ref(weights)

    def load():
        held = weights  # a local, as _asr_pipeline's cell was
        importer(name)
        return held is not None

    load()
    weights = None  # let go — the closure's cell too
    gc.collect()
    return ref() is not None


def test_a_plain_first_import_keeps_the_callers_model(stash):
    """The mechanism itself — why the 4 GB stayed."""
    assert _held_after(importlib.import_module, "stash_plain") is True


def test_a_clean_first_import_does_not(stash):
    assert _held_after(import_clean, "stash_clean") is False
    assert sys.modules["stash_clean"].KEPT.__class__.__name__ == "ModuleNotFoundError"


def test_a_failing_import_still_fails_in_the_caller(tmp_path, monkeypatch):
    (tmp_path / "broken_mod.py").write_text("raise ImportError('nope')\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "broken_mod", raising=False)
    with pytest.raises(ImportError, match="nope"):
        import_clean("broken_mod")


def _code(path: str) -> str:
    """Just the code — comments name what they replaced."""
    text = (REPO / path).read_text(encoding="utf-8")
    return "\n".join(ln for ln in text.splitlines() if not ln.lstrip().startswith("#"))


def test_the_transcription_imports_its_libraries_clean():
    code = _code("whisper_tm/core/transcription.py")
    assert "from whisperx.asr import" not in code and "from whisperx.diarize import" not in code
    assert "\n    import whisperx\n" not in code and "import whisperx as" not in code
    assert 'import_clean("whisperx.asr")' in code
    sep = _code("whisper_tm/core/vocal_isolation.py")
    assert "from audio_separator.separator import" not in sep
    assert 'import_clean("audio_separator.separator")' in sep
