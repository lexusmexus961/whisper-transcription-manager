"""
``database.backup_path`` is resolved like every other folder (config.backup_dir).

It was used as written: ``~/backups`` became a folder named ``~`` in the
working directory, and ``${NAS}/db`` a folder named ``${NAS}``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from whisper_tm.config import ConfigManager


def _cfg(tmp_path, backup_path: str) -> ConfigManager:
    file = tmp_path / "config.yaml"
    file.write_text(f'database:\n  backup_path: "{backup_path}"\n', encoding="utf-8")
    c = ConfigManager()
    c.load(file)
    return c


def test_unset_is_none(tmp_path):
    assert _cfg(tmp_path, "").backup_dir is None


def test_home_is_expanded(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    assert _cfg(tmp_path, "~/backups").backup_dir == tmp_path / "home" / "backups"


def test_a_variable_is_expanded(tmp_path, monkeypatch):
    monkeypatch.setenv("WTM_TEST_NAS", str(tmp_path / "nas"))
    assert _cfg(tmp_path, "${WTM_TEST_NAS}/db").backup_dir == tmp_path / "nas" / "db"


def test_an_undefined_variable_leaves_it_unset(tmp_path, monkeypatch):
    monkeypatch.delenv("WTM_TEST_MISSING", raising=False)
    assert _cfg(tmp_path, "${WTM_TEST_MISSING}/db").backup_dir is None


def test_a_relative_path_is_relative_to_the_config_file(tmp_path):
    assert _cfg(tmp_path, "backups").backup_dir == tmp_path / "backups"


@pytest.mark.parametrize("raw", ["~/backups", "backups"])
def test_the_database_backs_up_into_the_resolved_folder(tmp_path, monkeypatch, raw):
    from whisper_tm import config as config_module
    from whisper_tm.database import Database

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    c = _cfg(tmp_path, raw)
    monkeypatch.setattr(config_module, "config", c)
    (tmp_path / "data").mkdir()
    db = Database(tmp_path / "data" / "library.db")
    db.initialize()
    target = db.backup_target()
    assert target == c.backup_dir and "~" not in target.parts
    assert db._backup_settings()[0] == str(c.backup_dir)
    assert Path(target).is_absolute()


# The other file and folder settings outside ``directories`` take the same
# rule (config.path_setting): rag.path, the Images & sounds index, its cache
# and its environment were used as written as well.


def _settings(tmp_path, text: str) -> ConfigManager:
    file = tmp_path / "config.yaml"
    file.write_text(text, encoding="utf-8")
    c = ConfigManager()
    c.load(file)
    return c


def test_the_vector_store_path_is_resolved(tmp_path, monkeypatch):
    from whisper_tm import config as config_module
    from whisper_tm.rag import service

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    c = _settings(tmp_path, 'rag:\n  path: "~/vectors/rag.db"\n')
    monkeypatch.setattr(config_module, "config", c)
    monkeypatch.setattr(service, "config", c)
    assert service.vector_db_path() == str(tmp_path / "home" / "vectors" / "rag.db")


def test_the_media_paths_are_resolved(tmp_path, monkeypatch):
    from whisper_tm import config as config_module
    from whisper_tm.media import environment, indexer, store

    monkeypatch.setenv("WTM_TEST_MEDIA", str(tmp_path / "m"))
    c = _settings(
        tmp_path,
        "media_search:\n"
        '  path: "index/media.db"\n'
        '  cache_dir: "${WTM_TEST_MEDIA}/cache"\n'
        '  environment: "envs/media"\n',
    )
    monkeypatch.setattr(config_module, "config", c)
    monkeypatch.setattr(
        environment, "_cfg", lambda name, default=None: c.get("media_search", name, default=default)
    )
    monkeypatch.setattr(
        indexer, "_cfg", lambda name, default=None: c.get("media_search", name, default=default)
    )
    assert store.store_path() == tmp_path / "index" / "media.db"
    assert indexer.cache_dir() == tmp_path / "m" / "cache"
    assert environment.env_dir() == tmp_path / "envs" / "media"


def test_an_undefined_variable_in_a_setting_is_named(tmp_path, monkeypatch, caplog):
    monkeypatch.delenv("WTM_TEST_MISSING", raising=False)
    c = _settings(tmp_path, 'rag:\n  path: "${WTM_TEST_MISSING}/rag.db"\n')
    with caplog.at_level("WARNING"):
        assert c.path_setting("rag", "path") is None
    assert "rag.path ignored" in caplog.text
