"""Switching continuous scanning off and on again mid-round (core/autoscan.py).

``stop()`` used to only set the shared stop signal, and ``start()``
cleared that same signal — a loop still inside a long round over a network
share never saw it. Off-and-on within a round left two loops scanning side by
side: the same files acted on twice, and one walk's cache cleanup iterating
a dict the other one's threads were filling.
"""

from __future__ import annotations

import threading
import time

from whisper_tm.core import autoscan as au


def test_the_old_loop_ends_and_one_scans(monkeypatch):
    scan = au.AutoScan()
    entered = threading.Event()
    release = threading.Event()
    rounds: list[str] = []
    inside = []
    most = []

    def round_():
        rounds.append(threading.current_thread().name)
        inside.append(1)
        most.append(len(inside))
        try:
            entered.set()
            release.wait(5)
        finally:
            inside.pop()
        return {}

    monkeypatch.setattr(scan, "round", round_)
    monkeypatch.setattr(au, "setting", lambda name: {"interval_min": 15}.get(name, False))

    scan.start()
    assert entered.wait(5)
    old = scan._thread
    scan.stop()  # off …
    entered.clear()
    scan.start()  # … and on, while the old round runs
    new = scan._thread
    assert new is not old
    release.set()
    old.join(3)
    try:
        assert not old.is_alive(), "the old loop kept scanning after it was switched off"
        assert new.is_alive()
        assert max(most) == 1, "two rounds ran at the same time"
    finally:
        scan.stop()
        new.join(3)
    assert not new.is_alive()


def test_stop_wakes_a_waiting_loop(monkeypatch):
    scan = au.AutoScan()
    monkeypatch.setattr(scan, "round", lambda: {})
    monkeypatch.setattr(au, "setting", lambda name: {"interval_min": 15}.get(name, False))
    scan.start()
    t = scan._thread
    time.sleep(0.2)  # now in its 15-minute wait
    scan.stop()
    t.join(3)
    assert not t.is_alive()
