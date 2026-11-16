"""The data folder and the database's folder are the owner's only; files
with passwords and tokens that others can read are named at start."""

import logging
import os
import stat

import pytest

from whisper_tm.config import ConfigManager

pytestmark = pytest.mark.skipif(os.name != "posix", reason="POSIX modes")


def test_folders_become_private(tmp_path):
    a, b = tmp_path / "data", tmp_path / "db"
    a.mkdir(mode=0o755)
    b.mkdir(mode=0o755)
    os.chmod(a, 0o755)
    os.chmod(b, 0o775)
    ConfigManager._keep_private(a, b, tmp_path / "missing")
    assert stat.S_IMODE(a.stat().st_mode) == 0o700
    assert stat.S_IMODE(b.stat().st_mode) == 0o700


def test_readable_secret_files_are_named(tmp_path, caplog):
    from whisper_tm import app as app_mod
    from whisper_tm.config import config

    cfg = tmp_path / "config.yaml"
    cfg.write_text(f"directories:\n  media: {tmp_path}\n", encoding="utf-8")
    os.chmod(cfg, 0o644)
    config.load(str(cfg))
    with caplog.at_level(logging.WARNING):
        app_mod._warn_readable_secrets()
    assert "config.yaml is readable by other accounts" in caplog.text
    config.load(None)
