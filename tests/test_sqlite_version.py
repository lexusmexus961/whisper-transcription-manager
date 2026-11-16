"""The server refuses to start on an SQLite older than 3.38 — with a reason.

The database uses unixepoch() (SQLite 3.38): on Ubuntu 22.04's Python (SQLite
3.37) the first scan, queue entry and statistics failed with "no such
function: unixepoch", long after the start looked fine.
"""

from __future__ import annotations

import sqlite3

from whisper_tm import __main__ as main_mod


def test_a_new_sqlite_passes():
    if sqlite3.sqlite_version_info >= (3, 38):
        assert main_mod._sqlite_too_old() == ""


def test_an_old_sqlite_is_named_with_the_way_out(monkeypatch):
    monkeypatch.setattr(sqlite3, "sqlite_version_info", (3, 37, 2))
    monkeypatch.setattr(sqlite3, "sqlite_version", "3.37.2")
    why = main_mod._sqlite_too_old()
    assert "3.37.2" in why and "3.38" in why and "setup.sh" in why
