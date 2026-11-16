"""Saving a summary never loses another one.

Two failures this guards against:

* **Lost update.** Every writer read the list, appended and wrote it back
  — the queue's summary and a "Save" in the dialog finishing together kept
  only one of them.
* **A failed read wrote over the database.** A DB read error was logged at
  debug level and the JSON backup taken instead; the next save then
  replaced the database's list with that (possibly older) one.

Now changes go through ``summarize.change_summaries``: one writer at a time,
read strictly (a DB error stops the change), and the JSON backup is
written through a temporary file.
"""

from __future__ import annotations

import threading
import time

import pytest

from whisper_tm.core import summarize as s


class _Db:
    def __init__(self):
        self.rows: dict[str, list] = {}
        self.fail_read = False
        self.writes = 0

    def get_summaries(self, filename, media_dir=None):
        if self.fail_read:
            raise RuntimeError("database is locked")
        rows = self.rows.get(filename)
        return None if rows is None else [dict(r) for r in rows]

    def replace_summaries(self, filename, summaries, media_dir=None):
        self.writes += 1
        self.rows[filename] = [dict(r) for r in summaries]


@pytest.fixture
def db(tmp_path, monkeypatch):
    from whisper_tm.config import config

    d = _Db()
    monkeypatch.setattr("whisper_tm.extensions.db", d)
    monkeypatch.setattr(type(config), "transcript_dir", property(lambda self: tmp_path))
    monkeypatch.setattr(type(config), "media_dir", property(lambda self: tmp_path))
    return d


def test_two_writers_at_once_both_land(db, monkeypatch):
    read = s.load_summaries

    def slow(*a, **k):
        out = read(*a, **k)
        time.sleep(0.05)  # the window the other writer used
        return out

    monkeypatch.setattr(s, "load_summaries", slow)

    def add(text):
        s.change_summaries("a.mp4", lambda lst: lst.append({"text": text}))

    ts = [threading.Thread(target=add, args=(t,)) for t in ("one", "two")]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert sorted(x["text"] for x in db.rows["a.mp4"]) == ["one", "two"]


def test_a_failed_read_writes_nothing(db):
    db.rows["a.mp4"] = [{"text": "kept"}]
    db.fail_read = True
    with pytest.raises(RuntimeError, match="locked"):
        s.change_summaries("a.mp4", lambda lst: lst.append({"text": "new"}))
    assert db.writes == 0 and db.rows["a.mp4"] == [{"text": "kept"}]


def test_the_change_returns_its_result_and_saves(db):
    saved, n = s.change_summaries("a.mp4", lambda lst: (lst.append({"text": "x"}), len(lst))[1])
    assert saved is True and n == 1
    assert s.load_summaries_json("a.mp4") == [{"text": "x"}]
    assert not list(s.summaries_path("a.mp4").parent.glob("*.tmp"))
