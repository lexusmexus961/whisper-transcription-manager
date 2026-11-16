"""
The NAS probe of /api/health/detail can't hang a request (UI review #24).

``os.listdir`` on the media folder had no timeout: a hung NAS mount tied up
a server thread for every health request. Now the probe runs in a worker
thread, the request waits at most NAS_TIMEOUT_S and answers "NAS not
responding", and no second probe starts while one is still hanging.
"""

import threading
import time

from whisper_tm.api import health


def test_a_hung_nas_answers_in_time_and_only_one_probe_hangs(monkeypatch):
    release = threading.Event()
    calls = []

    def hung_listdir(path):
        calls.append(path)
        release.wait(10)
        return []

    monkeypatch.setattr(health, "NAS_TIMEOUT_S", 0.2)
    monkeypatch.setattr(health, "_nas_probe", None)
    monkeypatch.setattr(health.os, "listdir", hung_listdir)
    try:
        t0 = time.monotonic()
        a = health._nas_block()
        b = health._nas_block()
        assert time.monotonic() - t0 < 2
        assert a["ok"] is False and "not responding" in a["error"]
        assert b["ok"] is False and "not responding" in b["error"]
        assert len(calls) == 1
    finally:
        release.set()
        health._nas_probe["thread"].join(5)


def test_a_reachable_nas_reports_its_latency(monkeypatch):
    monkeypatch.setattr(health, "_nas_probe", None)
    monkeypatch.setattr(health.os, "listdir", lambda p: [])
    out = health._nas_block()
    assert out["ok"] is True and out["latency_ms"] is not None
