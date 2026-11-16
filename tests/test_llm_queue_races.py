"""LLM queue: a cancel is never overwritten, a result is never lost quietly
(core/llm_queue.py).

* The worker read the next entry and then set it running in a second,
  unguarded statement: a cancel that landed in between was overwritten,
  and the cancelled task ran.
* A cancel of an entry the worker had just taken — 'running' in the
  database, not yet its current one — found it neither pending nor
  current, and was lost.
* Every progress and status write swallowed its database error at debug
  level: a locked database at the end of a long summary left the entry
  'running' with its result gone, and nothing in the log said why.
"""

from __future__ import annotations

import contextlib
import logging
import sqlite3
import threading

import pytest

import whisper_tm.extensions as ext
from whisper_tm.core import llm_queue as lq
from whisper_tm.database import Database


@pytest.fixture
def queue(tmp_path, monkeypatch):
    db = Database(tmp_path / "test.db")
    db.initialize()
    monkeypatch.setattr(ext, "db", db)
    return lq.LlmQueue(), db


def _status(db, entry_id):
    with db.connection() as c:
        return c.execute("SELECT status FROM llm_queue WHERE id = ?", (entry_id,)).fetchone()[0]


class Rows:
    def __init__(self, rows):
        self._rows = list(rows)

    def fetchone(self):
        return self._rows.pop(0) if self._rows else None

    def fetchall(self):
        rows, self._rows = self._rows, []
        return rows


def test_a_cancel_while_the_worker_takes_the_entry_holds(queue, monkeypatch):
    llm, db = queue
    eid = llm.enqueue("summary", "a.mp4", {})
    real = db.connection
    me = threading.get_ident()
    fired = []

    class Conn:
        def __init__(self, c):
            self._c = c

        def execute(self, sql, *a):
            cur = self._c.execute(sql, *a)
            if "status='pending'" in sql and not fired:
                fired.append(1)
                rows = cur.fetchall()
                # The cancel arrives right after the worker's first look
                self._c.commit()
                t = threading.Thread(target=llm.cancel, args=(eid,))
                t.start()
                t.join(5)
                return Rows(rows)
            return cur

        def __getattr__(self, name):
            return getattr(self._c, name)

    @contextlib.contextmanager
    def connection():
        with real() as c:
            yield Conn(c) if threading.get_ident() == me else c

    monkeypatch.setattr(db, "connection", connection)
    taken = llm._next_one()
    monkeypatch.setattr(db, "connection", real)
    assert fired
    ran = []
    monkeypatch.setattr("whisper_tm.core.llm_tasks.execute", lambda e, q: ran.append(e) or {})
    if taken is not None:
        llm._run_one(taken)
    assert ran == [], "the cancelled task ran"
    assert _status(db, eid) == "cancelled"


def test_a_cancel_between_taking_and_running_is_not_lost(queue, monkeypatch):
    llm, db = queue
    eid = llm.enqueue("summary", "a.mp4", {})
    entry = llm._next_one()  # running, not yet current
    assert llm.cancel(eid) is True
    ran = []
    monkeypatch.setattr("whisper_tm.core.llm_tasks.execute", lambda e, q: ran.append(e) or {})
    llm._run_one(entry)
    assert ran == []
    assert _status(db, eid) == "cancelled"


def test_the_final_write_retries_a_locked_database(queue, monkeypatch, caplog):
    llm, db = queue
    eid = llm.enqueue("summary", "a.mp4", {})
    real = db.connection
    fails = [2]

    @contextlib.contextmanager
    def connection():
        if fails[0]:
            fails[0] -= 1
            raise sqlite3.OperationalError("database is locked")
        with real() as c:
            yield c

    monkeypatch.setattr(db, "connection", connection)
    monkeypatch.setattr(lq.time, "sleep", lambda s: None)
    llm._update(eid, status="done", result_json="{}")
    monkeypatch.setattr(db, "connection", real)
    assert _status(db, eid) == "done"


def test_a_lost_final_write_is_said_loudly(queue, monkeypatch, caplog):
    llm, db = queue
    eid = llm.enqueue("summary", "a.mp4", {})

    @contextlib.contextmanager
    def connection():
        raise sqlite3.OperationalError("database is locked")
        yield

    monkeypatch.setattr(db, "connection", connection)
    monkeypatch.setattr(lq.time, "sleep", lambda s: None)
    with caplog.at_level(logging.WARNING, logger="whisper_tm.core.llm_queue"):
        llm._update(eid, status="done", result_json="{}")
    assert any("database is locked" in r.getMessage() for r in caplog.records)
