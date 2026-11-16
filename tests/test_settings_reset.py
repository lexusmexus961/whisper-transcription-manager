"""
Settings reset: config values go back to defaults, but WHERE things live
(directories, database file, server address) is kept — otherwise the next
start opens a different database and "transcripts are kept" is not true.
"""

import pytest

from whisper_tm.config import config


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    own = tmp_path / "config.yaml"
    own.write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("WHISPER_TM_CONFIG", str(own))
    config.load(None)
    config._data.setdefault("directories", {})["media"] = str(tmp_path / "media")
    config._data["directories"]["work"] = str(tmp_path / "work")
    config._data.setdefault("database", {})["path"] = str(tmp_path / "own.db")
    config._data.setdefault("server", {})["port"] = 5123
    config._resolve_directories()
    config._ensure_directories()
    from whisper_tm.app import create_app

    yield create_app().test_client(), tmp_path
    config.load(None)


def test_reset_keeps_locations(client):
    c, tmp = client
    config._data.setdefault("scanner", {})["watch_interval_sec"] = 999
    r = c.post("/api/settings/reset")
    assert r.status_code == 200 and r.get_json()["ok"]
    assert config._data["scanner"].get("watch_interval_sec") != 999
    assert config._data["database"]["path"] == str(tmp / "own.db")
    assert config._data["directories"]["media"] == str(tmp / "media")
    assert config._data["directories"]["work"] == str(tmp / "work")
    assert config._data["server"]["port"] == 5123


def test_reset_keeps_what_the_page_cannot_set_back(client):
    """Images & sounds' and the audio analysis' databases, the reranker's
    start and the MCP access: file-only or local-only, so a reset would
    have pointed at empty databases or dropped values for good."""
    c, tmp = client
    own = {
        ("media_search", "path"): str(tmp / "m.db"),
        ("media_search", "environment"): str(tmp / "env"),
        ("audio_analysis", "path"): str(tmp / "a.db"),
        ("rerank", "host"): "127.0.0.2",
        ("youtube", "allow_private_hosts"): True,
        ("mcp", "require_token"): False,
    }
    for (s, k), v in own.items():
        config._data.setdefault(s, {})[k] = v
    assert c.post("/api/settings/reset").get_json()["ok"]
    for (s, k), v in own.items():
        assert config._data[s][k] == v, (s, k)
