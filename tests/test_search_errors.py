"""
/api/search and /api/search/facets: only a real FTS5 syntax error is the
user's fault (400). A locked or broken database is a server error — it must
be logged and answered as 500, not disguised as "check quotation marks".
"""

import logging
import sqlite3

import pytest


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    from whisper_tm.config import config

    config.load(None)
    config._data.setdefault("directories", {})["media"] = str(tmp_path)
    config._resolve_directories()
    config._ensure_directories()
    from whisper_tm.app import create_app

    return create_app().test_client()


def _raise(exc):
    def f(*a, **kw):
        raise exc

    return f


class _FailingDB:
    """Stands in for the module's database: search and facets raise."""

    def __init__(self, exc):
        self.search = self.search_facets = _raise(exc)


@pytest.mark.parametrize(
    "path,method",
    [("/api/search?q=hello", "search"), ("/api/search/facets?q=hello", "search_facets")],
)
def test_locked_database_is_500_and_logged(client, monkeypatch, caplog, path, method):
    del method  # both endpoints fail through the same stub
    from whisper_tm.api import search as api

    monkeypatch.setattr(api, "db", _FailingDB(sqlite3.OperationalError("database is locked")))
    with caplog.at_level(logging.WARNING):
        r = client.get(path)
    assert r.status_code == 500
    assert "locked" in r.get_json()["error"]
    assert any(
        "database is locked" in rec.getMessage() or "database is locked" in str(rec.exc_info)
        for rec in caplog.records
    )


@pytest.mark.parametrize(
    "message", ['fts5: syntax error near "AND"', "unterminated string", "no such column: foo"]
)
def test_fts_syntax_error_stays_400(client, monkeypatch, message):
    from whisper_tm.api import search as api

    monkeypatch.setattr(api, "db", _FailingDB(sqlite3.OperationalError(message)))
    r = client.get("/api/search?q=hello")
    assert r.status_code == 400
    assert "syntax" in r.get_json()["error"]
