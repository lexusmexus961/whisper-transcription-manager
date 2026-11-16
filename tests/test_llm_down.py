"""A language model that is down: the tasks wait, Ask says it plainly, the
setup guide sees the server setup.sh configured.

* The LLM queue failed every entry while its server was down — a folder of
  200 summaries was 200 errors after one stopped llama-server. Now the entry
  goes back to wait, the queue tries again after 30 s, then longer (up to
  10 minutes), and "Try now" brings the next try forward.
* Ask showed the library's text: "HTTPConnectionPool(host=…): Max retries
  exceeded with url …".
* Right after setup.sh --llama the setup guide said "none configured": it
  counted the saved provider list, while the setup writes llm.endpoint.
"""

from __future__ import annotations

import time

import pytest
import requests

import whisper_tm.extensions as ext
from whisper_tm.config import config
from whisper_tm.core import chat
from whisper_tm.core import llm_queue as lq
from whisper_tm.core.llm_client import LlmConnectionError
from whisper_tm.database import Database

DOWN = (
    "Connection failed: HTTPConnectionPool(host='127.0.0.1', port=8080): Max retries "
    "exceeded with url: /v1/chat/completions (Caused by NewConnectionError(… "
    "Connection refused))"
)


@pytest.mark.parametrize(
    "text,down",
    [
        (DOWN, True),
        ("Timeout on connection establishment", True),
        ('LLM server 503: {"error":{"code":503,"message":"Loading model"}}', True),
        ("Timeout after 1800s", False),  # a slow answer is the entry's
        ("LLM server 500: context size exceeded", False),
    ],
)
def test_what_counts_as_down(text, down):
    assert lq.server_down(LlmConnectionError(text)) is down
    wrapped = RuntimeError("Summary failed")
    wrapped.__cause__ = LlmConnectionError(text)
    assert lq.server_down(wrapped) is down
    assert not lq.server_down(RuntimeError(DOWN))  # only the client's own error


@pytest.fixture
def queue(tmp_path, monkeypatch):
    db = Database(tmp_path / "test.db")
    db.initialize()
    monkeypatch.setattr(ext, "db", db)
    return lq.LlmQueue(), db


def _row(db, entry_id):
    with db.connection() as c:
        return c.execute(
            "SELECT status, message, error, parts_json FROM llm_queue WHERE id = ?", (entry_id,)
        ).fetchone()


def _take(llm):
    return llm._next_one()


def test_the_tasks_wait_for_the_server(queue, monkeypatch):
    llm, db = queue
    first = llm.enqueue("summary", "a.mp4", {})
    llm.enqueue("summary", "b.mp4", {})
    outcome = {"raise": LlmConnectionError(DOWN)}

    def execute(entry, q):
        q.partial_result(entry["id"], {"part": 1})
        if outcome["raise"]:
            raise outcome["raise"]
        return {"text": "summary"}

    from whisper_tm.core import llm_tasks

    monkeypatch.setattr(llm_tasks, "execute", execute)

    entry = _take(llm)
    assert entry["id"] == first
    llm._run_one(entry)
    status, message, error, parts = _row(db, first)
    assert (status, message, error) == ("pending", "Waiting for the language model", "")
    assert parts == "[]"  # it starts over
    snap = llm.snapshot()
    assert 25 <= snap["waiting"]["next_in"] <= 30 and snap["counts"]["error"] == 0
    assert "can't be reached" in " ".join(snap["log"])

    # Down again: the next try waits twice as long
    llm._run_one(_take(llm))
    assert 55 <= llm.snapshot()["waiting"]["next_in"] <= 60

    # "Try now" (the Resume of a queue that isn't paused)
    llm.pause(False)
    assert llm.snapshot()["waiting"]["next_in"] == 0
    assert "Resumed" not in " ".join(llm.snapshot()["log"])

    outcome["raise"] = None
    llm._run_one(_take(llm))
    snap = llm.snapshot()
    assert snap["waiting"] is None and snap["counts"]["done"] == 1
    assert "answers again" in " ".join(snap["log"])


def test_an_answer_error_still_fails_the_entry(queue, monkeypatch):
    llm, db = queue
    eid = llm.enqueue("summary", "a.mp4", {})
    from whisper_tm.core import llm_tasks

    def execute(entry, q):
        raise LlmConnectionError("LLM server 500: context size exceeded")

    monkeypatch.setattr(llm_tasks, "execute", execute)
    llm._run_one(_take(llm))
    assert _row(db, eid)[0] == "error" and llm.snapshot()["waiting"] is None


def test_the_loop_takes_nothing_before_the_next_try(queue, monkeypatch):
    llm, _db = queue
    llm.enqueue("summary", "a.mp4", {})
    llm._waiting = {"since": time.time(), "next_try": time.time() + 30, "tries": 1, "reason": "x"}
    taken = []
    monkeypatch.setattr(llm, "_next_one", lambda: taken.append(1))

    def once(timeout=None):
        llm._running = False  # one round of the loop
        return True

    monkeypatch.setattr(llm._wakeup, "wait", once)
    llm._running = True
    llm._loop()
    assert taken == []


@pytest.mark.parametrize(
    "url,error,expect",
    [
        (
            "http://127.0.0.1:8080/v1/chat/completions",
            requests.exceptions.ConnectionError("x"),
            "The language model at 127.0.0.1:8080 can't be reached — nothing answers there; "
            "is its server running? The local llama.cpp server starts with ./start-llm.sh.",
        ),
        (
            "http://10.0.0.5:11434/v1/chat/completions",
            requests.exceptions.ConnectTimeout("x"),
            "The language model at 10.0.0.5:11434 can't be reached — it didn't answer in time; "
            "is its server running? Settings → AI models shows the address.",
        ),
    ],
)
def test_ask_says_it_plainly(url, error, expect):
    assert chat._unreachable(url, error) == expect


def test_the_guide_sees_the_endpoint_the_setup_wrote(monkeypatch):
    from whisper_tm.core import diagnostics, llm_provider

    section = config._data.setdefault("llm", {})
    monkeypatch.setitem(section, "providers", [])
    monkeypatch.setitem(section, "endpoint", "http://127.0.0.1:8080/v1")
    monkeypatch.setattr(llm_provider, "_local_provider", lambda: None)
    check = next(
        c for c in diagnostics._feature_checks() if c["title"].startswith("Language model")
    )
    assert check["status"] == diagnostics.OK and "none" not in check["detail"]
    assert "Standard" in check["detail"]
