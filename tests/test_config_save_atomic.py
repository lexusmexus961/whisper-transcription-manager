"""
Saving the configuration.

``save()`` truncated config.yaml and wrote it anew in place, with mode 0644:
a crash in between left half a file that the next start loaded without a
word, and every local user could read the API keys, tokens and password.
"""

from __future__ import annotations

import os
import stat

import pytest
import yaml

from whisper_tm.config import ConfigManager


@pytest.fixture
def cfg(tmp_path):
    file = tmp_path / "config.yaml"
    file.write_text("server:\n  port: 5123\n", encoding="utf-8")
    os.chmod(file, 0o644)
    c = ConfigManager()
    c.load(file)
    return c, file


def test_saved_file_is_readable_only_by_its_owner(cfg):
    c, file = cfg
    c.set("llm", "api_key", "sk-x")
    c.save()
    assert stat.S_IMODE(file.stat().st_mode) == 0o600
    assert yaml.safe_load(file.read_text())["llm"]["api_key"] == "sk-x"


def test_a_failed_write_leaves_the_old_file_whole(cfg, monkeypatch):
    c, file = cfg
    before = file.read_text()

    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(os, "fsync", boom)
    c.set("server", "port", 6000)
    with pytest.raises(OSError):
        c.save()
    assert file.read_text() == before
    assert [p.name for p in file.parent.iterdir()] == ["config.yaml"]


def test_a_value_yaml_cannot_read_back_is_refused_not_written(cfg):
    """yaml.dump wrote Python tags that safe_load refuses at the next start."""
    c, file = cfg
    before = file.read_text()
    c.set("misc", "thing", object())
    with pytest.raises(yaml.YAMLError):
        c.save()
    assert file.read_text() == before


def test_a_symlinked_config_is_written_where_it_points(tmp_path):
    real = tmp_path / "real.yaml"
    real.write_text("{}\n", encoding="utf-8")
    link = tmp_path / "config.yaml"
    link.symlink_to(real)
    c = ConfigManager()
    c.load(link)
    c.set("server", "port", 5333)
    c.save(link)
    assert link.is_symlink()
    assert yaml.safe_load(real.read_text())["server"]["port"] == 5333
