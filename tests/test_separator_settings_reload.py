"""A changed separation setting must reach the next job.

The resident separator was cached by ``(model_file, single_stem)`` alone.
``vocal_isolation.mdxc_overlap`` and ``vocal_isolation.use_autocast`` are
handed to audio-separator's ``Separator`` once, when it is built — so a value
changed in the settings only took effect once the model happened to be
unloaded: by the idle unload, the "Unload VRAM" button or a restart. Until
then every job kept running with the old value while the settings page
showed the new one.

Now the settings are part of the key — as values, so that unchanged
settings keep the loaded model (a reload costs seconds, and it would cost
them on every job) — and the model that gets replaced is released the whole
way before the new one loads, the same way ``release_separator()`` does it.
"""

from __future__ import annotations

import sys
import threading
import types
import weakref
from types import SimpleNamespace

import pytest

from whisper_tm.config import config
from whisper_tm.core import vocal_isolation as vi

MODEL = "vocals_mel_band_roformer.ckpt"


@pytest.fixture
def separators(tmp_path, monkeypatch):
    """``_get_separator`` without audio-separator, GPU or checkpoint."""
    torch = pytest.importorskip("torch")
    built: list[weakref.ref] = []

    class FakeSeparator:
        def __init__(self, **kw):
            # Checked at the moment the new one is built: the old model
            # must already be gone, not just unreferenced.
            self.predecessor_alive = any(r() is not None for r in built)
            self.kw = kw
            # A reference cycle, as real models tend to have one: dropping
            # the reference alone does not free it, a garbage collection does.
            self._me = self
            built.append(weakref.ref(self))

        def load_model(self, model_filename):
            self.model = model_filename

    module = types.ModuleType("audio_separator.separator")
    module.Separator = FakeSeparator
    monkeypatch.setitem(sys.modules, "audio_separator.separator", module)
    # The real one starts the idle watchdog thread
    monkeypatch.setattr("whisper_tm.core.model_idle.report_usage", lambda: None)
    monkeypatch.setattr(vi, "separation_model_dir", lambda: tmp_path)
    emptied: list[int] = []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: emptied.append(1))
    monkeypatch.setattr(vi, "_separator", None)
    monkeypatch.setattr(vi, "_separator_key", None)
    # A lock of the test's own: should a replacement ever wait on the lock
    # its caller holds, only this test's thread stays stuck on it.
    monkeypatch.setattr(vi, "_separator_lock", threading.Lock())
    section = config._data.setdefault("vocal_isolation", {})

    def get(model=MODEL):
        """As ``_run_separator`` calls it: under ``_separator_lock``. In a
        thread with a time limit, so that a replacement taking the
        (non-reentrant) lock itself fails here instead of hanging."""
        result = {}

        def work():
            with vi._separator_lock:
                result["sep"] = vi._get_separator(model, tmp_path)

        t = threading.Thread(target=work, daemon=True)
        t.start()
        t.join(10)
        assert not t.is_alive(), "_get_separator waits on _separator_lock"
        return result.pop("sep")

    return SimpleNamespace(
        get=get,
        built=built,
        emptied=emptied,
        set=lambda key, value: monkeypatch.setitem(section, key, value),
    )


class TestReload:
    def test_unchanged_settings_keep_the_loaded_model(self, separators):
        """The dicts are rebuilt on every call — equal values must still
        count as unchanged, or every job would reload the model."""
        first = separators.get()
        assert separators.get() is first
        assert len(separators.built) == 1

    def test_a_changed_overlap_reaches_the_next_job(self, separators):
        separators.set("mdxc_overlap", 8)
        separators.get()
        separators.set("mdxc_overlap", 4)
        sep = separators.get()
        assert len(separators.built) == 2
        assert sep.kw["mdxc_params"]["overlap"] == 4

    def test_a_changed_autocast_reaches_the_next_job(self, separators):
        separators.set("use_autocast", True)
        separators.get()
        separators.set("use_autocast", False)
        sep = separators.get()
        assert len(separators.built) == 2
        assert sep.kw["use_autocast"] is False

    def test_values_count_not_their_spelling(self, separators):
        """8 and 8.0 are the same overlap — the settings page may save
        either, and that is no reason to reload."""
        separators.set("mdxc_overlap", 8)
        first = separators.get()
        separators.set("mdxc_overlap", 8.0)
        assert separators.get() is first

    def test_the_old_model_is_released_before_the_new_one_loads(self, separators):
        """Not just the reference: collected, and torch's cache handed back
        — the diarization worker is a process of its own on the same card."""
        separators.set("mdxc_overlap", 8)
        separators.get()  # not kept: only vi holds it
        separators.set("mdxc_overlap", 4)
        sep = separators.get()
        assert sep.predecessor_alive is False
        assert separators.built[0]() is None
        assert separators.emptied == [1]

    def test_the_resident_model_is_still_named_by_its_file(self, separators, monkeypatch):
        """The key grew; the GPU view still reads the model from key[0]."""
        from whisper_tm.core import transcription, vram_procs

        monkeypatch.setattr(transcription, "_models", {})
        separators.get()
        assert vram_procs.resident_models() == [f"Separator ({MODEL})"]


def test_release_separator_still_releases(separators):
    """The freeing moved into a helper; the public way out still uses it."""
    separators.get()
    vi.release_separator()
    assert vi._separator is None and vi._separator_key is None
    assert separators.built[0]() is None
    assert separators.emptied == [1]
