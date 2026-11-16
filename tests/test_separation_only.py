"""Run without transcription ("no transcription" in the dialog).

The dedicated separation dialog with /api/separate has been dropped: separation
now runs through the same dialog and the same queue as a
transcription. These tests pin down the spots where that used to
break — above all: 'none' is NOT a Whisper model and must not
silently fall back to the config default, or the run transcribes after all.
"""

import inspect

import pytest

from whisper_tm.core import queue as q


@pytest.fixture
def cfg():
    """Set config and restore it — `config` is a singleton."""
    from whisper_tm.config import config as c

    if not c.is_loaded:
        c.load()
    overrides = {
        ("transcription", "default_model"): "large-v3",
        ("vocal_isolation", "default_enabled"): False,
        ("vocal_isolation", "default_save_alongside"): False,
    }
    previous = {k: c.get(*k, default=None) for k in overrides}
    for (sec, key), val in overrides.items():
        c.set(sec, key, val)
    try:
        yield c
    finally:
        for (sec, key), val in previous.items():
            c.set(sec, key, val)


class TestNormalize:
    def test_none_is_recognized_as_a_separation_run(self, cfg):
        assert q._normalize({"model": "none"})["separation_only"] is True

    def test_a_normal_run_is_not_a_separation_run(self, cfg):
        assert q._normalize({"model": "turbo"})["separation_only"] is False
        assert q._normalize({})["separation_only"] is False

    def test_none_never_reaches_whisper(self, cfg):
        """'none' must not slip through as a model name."""
        opts = q._normalize({"model": "none"})
        assert opts["model_name"] != "none"
        from whisper_tm.config import VALID_WHISPER_MODELS

        assert opts["model_name"] in VALID_WHISPER_MODELS

    def test_separation_and_storing_are_set_for_a_separation_run(self, cfg):
        """Without both the run would be a no-op — despite config defaults 'off'."""
        opts = q._normalize(
            {"model": "none", "vocal_isolation": False, "save_audio_alongside": False}
        )
        assert opts["vocal_isolation"] is True
        assert opts["save_audio_alongside"] is True

    def test_a_normal_run_keeps_the_deliberate_off(self, cfg):
        """The enforcement above applies ONLY to the separation-only run."""
        opts = q._normalize(
            {"model": "turbo", "vocal_isolation": False, "save_audio_alongside": False}
        )
        assert opts["vocal_isolation"] is False
        assert opts["save_audio_alongside"] is False

    def test_a_nonsense_model_still_falls_back(self, cfg):
        """The new 'none' branch must not override the unknown-model fallback."""
        opts = q._normalize({"model": "does-not-exist"})
        assert opts["model_name"] == "large-v3"
        assert opts["separation_only"] is False


def test_normalize_matches_the_worker_signature():
    """_normalize() feeds transcription_worker via **opts — every key
    needs a matching parameter there, or the run dies with TypeError."""
    from whisper_tm.core.transcription import transcription_worker

    params = set(inspect.signature(transcription_worker).parameters)
    assert set(q._normalize({"model": "none"})) <= params


def test_the_worker_knows_the_separation_mode():
    from whisper_tm.core.transcription import transcription_worker

    params = inspect.signature(transcription_worker).parameters
    assert params["separation_only"].default is False
    source = inspect.getsource(transcription_worker)
    # The run ends after separation: no splitting, no Whisper.
    assert "if separation_only:" in source
    assert source.index("if separation_only:") < source.index("Audio splitting")


def test_no_separate_separation_path_any_more():
    """The old second path is gone and must not come back."""
    import importlib

    for module in ("whisper_tm.api.separation", "whisper_tm.core.separation"):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(module)
