"""The separation model must not reside in a tmpfs.

``audio-separator`` puts its checkpoints (several hundred MB each, e.g.
``vocals_mel_band_roformer.ckpt``) by default in
``/tmp/audio-separator-models/``, and ``/tmp`` on many systems is a
**tmpfs in RAM**: empty after every restart, so the model is
**downloaded again** after each one.

This costs not just bandwidth. The download blocks the first job
after every restart, and anyone restarting offline gets instead the
silent fallback to a different separation model — with voice vectors that
no longer match the previously calculated ones.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from whisper_tm.config import config
from whisper_tm.core import vocal_isolation as vi


@pytest.fixture(autouse=True)
def _cfg():
    if not config.is_loaded:
        config.load()


class TestLocation:
    def test_not_library_default(self):
        """Core: away from ``/tmp/audio-separator-models``.

        Deliberately NOT checking "nowhere under /tmp" — pytest creates its own
        disposable directories there, and a test config may rightfully point there.
        Meant is the library's default, which ends up in tmpfs during live operation.
        """
        target = vi.separation_model_dir()
        assert target != Path("/tmp/audio-separator-models")
        assert target.name == "separation_models"

    def test_real_operation_outside_tmpfs(self, monkeypatch):
        """Against ACTUAL config, not what a previous test left behind.

        The session runs in a throwaway HOME under /tmp (conftest.py) — for
        this one path computation, which writes nothing, the account's own.
        """
        import os
        import pwd

        from platformdirs import user_cache_dir

        monkeypatch.setenv("HOME", pwd.getpwuid(os.getuid()).pw_dir)
        real = Path(user_cache_dir("whisper-tm"))
        if str(real).startswith("/tmp/"):
            pytest.skip("XDG_CACHE_HOME points to tmpfs itself")
        assert not str(real / "separation_models").startswith("/tmp/")

    def test_next_to_other_cache(self):
        """Where chunks also live — a separate config value
        would be a signal for a question nobody wants to answer differently."""
        assert vi.separation_model_dir().parent == config.cache_dir

    def test_directory_is_created(self):
        assert vi.separation_model_dir().is_dir()

    def test_on_error_prefer_old_location(self, monkeypatch):
        """Without directory, nothing loads — then rather the
        library default than a crash."""

        def broken(*a, **k):
            raise OSError("no space")

        monkeypatch.setattr(Path, "mkdir", broken)
        assert vi.separation_model_dir() == Path("/tmp/audio-separator-models")


class TestWiring:
    def test_separator_gets_directory(self):
        """Otherwise the prettiest path is useless."""
        source = inspect.getsource(vi)
        index = source.index("sep = Separator(")
        # Up to the closing paren of the CALL, not to the first one
        # overall — that's in `separation_model_directory()` itself.
        call = source[index : source.index("\n    )", index)]
        assert "model_file_dir=" in call, call
        assert "separation_model_dir()" in call, call

    def test_config_provides_cache_directory(self):
        """Without `config.cache_dir` the helper would fail with
        AttributeError, specifically on the first separation run."""
        assert isinstance(config.cache_dir, Path)
        assert config.cache_dir.is_absolute()
