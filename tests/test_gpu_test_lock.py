"""
The setup guide's GPU test holds the GPU.

It checked "is a job running?" once, then loaded Whisper on the card for up
to ten minutes — a queue file starting meanwhile met it there (CUDA out of
memory on a card shared with another process), and two clicks ran two tests.
Now the test holds the queue's GPU gate exclusively while it runs, and a
second one is refused.
"""

from __future__ import annotations

import threading

import pytest
from flask import Flask

from whisper_tm.core import diagnostics as dg
from whisper_tm.core.gpu_gate import GATE


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr("whisper_tm.core.model_idle._job_running", lambda: False)
    from whisper_tm.api.setup import setup_bp

    app = Flask(__name__)
    app.register_blueprint(setup_bp)
    return app.test_client()


def test_the_test_holds_the_gpu_gate(client, monkeypatch):
    seen = {}

    def fake(idx=0, timeout=600):
        # A queue stage asking now must not get in
        with GATE.try_exclusive("probe") as got:
            seen["other_could_take_it"] = got
        seen["state"] = GATE.state()
        return {"ok": True, "steps": []}

    monkeypatch.setattr(dg, "run_gpu_test_subprocess", fake)
    assert client.post("/api/setup/gpu_test").get_json()["ok"] is True
    assert seen["other_could_take_it"] is False
    # …and gives it back afterwards
    with GATE.try_exclusive("after") as got:
        assert got is True


def test_refused_while_a_queue_stage_runs(client, monkeypatch):
    called = []
    monkeypatch.setattr(dg, "run_gpu_test_subprocess", lambda *a, **k: called.append(1))
    entered, leave = threading.Event(), threading.Event()

    def stage():
        with GATE.stage("separation"):
            entered.set()
            leave.wait(5)

    t = threading.Thread(target=stage)
    t.start()
    try:
        entered.wait(5)
        r = client.post("/api/setup/gpu_test")
        assert r.status_code == 409
        assert called == []
    finally:
        leave.set()
        t.join(5)


def test_a_second_test_is_refused_while_one_runs(client, monkeypatch):
    started, finish = threading.Event(), threading.Event()

    def slow(idx=0, timeout=600):
        started.set()
        finish.wait(5)
        return {"ok": True, "steps": []}

    monkeypatch.setattr(dg, "run_gpu_test_subprocess", slow)
    results = {}
    t = threading.Thread(
        target=lambda: results.setdefault("first", client.post("/api/setup/gpu_test").status_code)
    )
    t.start()
    try:
        started.wait(5)
        assert client.post("/api/setup/gpu_test").status_code == 409
    finally:
        finish.set()
        t.join(5)
    assert results["first"] == 200
