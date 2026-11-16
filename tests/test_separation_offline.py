"""A separation model that can't load no longer stops every transcription.

Vocal isolation is on by default. Offline before its model's first download,
the model and its fallback both failed to load: every transcription failed,
and a banner said the fallback was active and to restart the server. Now
the separation reports that it is unavailable, the file is transcribed
without it (with a warning in its job and one system warning per run), and
the fallback banner only shows when the fallback did load.
"""

from __future__ import annotations

import sys
import threading
import types
from types import SimpleNamespace

import pytest

from whisper_tm.core import transcription as tx
from whisper_tm.core import vocal_isolation as vi

MODEL = "vocals_mel_band_roformer.ckpt"
OFFLINE = (
    "HTTPSConnectionPool(host='github.com', port=443): Max retries exceeded "
    "with url: /x (Caused by NameResolutionError: Temporary failure in name resolution)"
)


class Manager:
    def __init__(self):
        self.system, self.warned = [], []

    def system_warning(self, text):
        self.system.append(text)

    def warn(self, job, text):
        self.warned.append(text)

    def log(self, job, text):
        pass


@pytest.fixture
def separator(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    failing: dict[str, str] = {}

    class FakeSeparator:
        def __init__(self, **kw):
            self.model = None

        def load_model(self, model_filename):
            if model_filename in failing:
                raise RuntimeError(failing[model_filename])
            self.model = model_filename

    module = types.ModuleType("audio_separator.separator")
    module.Separator = FakeSeparator
    monkeypatch.setitem(sys.modules, "audio_separator.separator", module)
    monkeypatch.setattr("whisper_tm.core.model_idle.report_usage", lambda: None)
    monkeypatch.setattr(vi, "separation_model_dir", lambda: tmp_path)
    monkeypatch.setattr(vi, "_separator", None)
    monkeypatch.setattr(vi, "_separator_key", None)
    monkeypatch.setattr(vi, "_separator_lock", threading.Lock())
    manager = Manager()
    monkeypatch.setattr(vi, "job_manager", manager)

    def get(model=MODEL):
        with vi._separator_lock:
            return vi._get_separator(model, tmp_path, job=object())

    return SimpleNamespace(get=get, failing=failing, manager=manager)


def test_neither_model_loads_offline(separator):
    separator.failing[MODEL] = OFFLINE
    separator.failing[vi.FALLBACK_MODEL_FILE] = OFFLINE
    with pytest.raises(vi.SeparationUnavailable, match="nor the fallback"):
        separator.get()
    # The banner of a working fallback would send the user the wrong way
    assert not any("FALLBACK ACTIVE" in t for t in separator.manager.system)


def test_the_fallback_loads_and_says_so(separator):
    separator.failing[MODEL] = OFFLINE
    sep = separator.get()
    assert sep.model == vi.FALLBACK_MODEL_FILE
    banner = " ".join(separator.manager.system)
    assert "FALLBACK ACTIVE" in banner and "restart" not in banner


def test_a_full_card_is_no_missing_model(separator):
    """Out of memory goes to the queue's retry, as before."""
    separator.failing[MODEL] = "CUDA out of memory. Tried to allocate 2.00 GiB"
    with pytest.raises(RuntimeError) as caught:
        separator.get()
    assert not isinstance(caught.value, vi.SeparationUnavailable)


def test_the_fallback_itself_configured(separator):
    separator.failing[vi.FALLBACK_MODEL_FILE] = OFFLINE
    with pytest.raises(vi.SeparationUnavailable):
        separator.get(vi.FALLBACK_MODEL_FILE)


def test_the_file_is_transcribed_without_it_and_said_once(monkeypatch):
    manager = Manager()
    monkeypatch.setattr(tx, "_separation_said", False)
    for _ in range(3):
        tx._separation_unavailable(manager, object(), vi.SeparationUnavailable("x does not load"))
    assert len(manager.warned) == 3 and "transcribed without it" in manager.warned[0]
    assert len(manager.system) == 1 and "Settings → Vocal isolation" in manager.system[0]


def test_the_worker_catches_it_and_a_separation_only_job_does_not():
    import inspect

    source = inspect.getsource(tx.transcription_worker)
    block = source[source.index("except SeparationUnavailable as e:") :]
    block = block[: block.index("work_file.unlink")]
    assert "if separation_only:\n                    raise" in block
    assert "_separation_unavailable(job_manager, job, e)" in block
