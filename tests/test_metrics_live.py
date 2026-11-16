"""The performance view's live mode on the server (``core/metrics.py``).

The system performance tab shows the last 1 min 30 s live and keeps
scrolling. Checked: every reading
lands in an in-memory buffer the view asks for deltas of
(/api/system/live); while someone watches, the sampler adds a reading
between two regular ones — and those extra readings never reach the
top-bar meters or the 30-s rows: with a viewer or without, the rows are
the same, to the last digit.

The sampler loop runs on a fake clock: its waits advance the clock
instead of sleeping, so a minute of sampling takes no time and every
reading's place in it is exact.
"""

from __future__ import annotations

from collections import deque, namedtuple
from types import SimpleNamespace

import pytest
from flask import Flask

from whisper_tm.core import metrics

T0 = 1_790_000_000.0


@pytest.fixture(autouse=True)
def fresh(monkeypatch):
    """Every test starts with an empty buffer, nobody watching, no cost."""
    monkeypatch.setattr(metrics, "_ring", deque(maxlen=metrics._ring.maxlen))
    monkeypatch.setattr(metrics, "_pending", [])
    monkeypatch.setattr(metrics, "_latest", None)
    monkeypatch.setattr(metrics, "_watch_until", 0.0)
    monkeypatch.setattr(metrics, "_cost", None)


def _proc(key, mb, own, label=None):
    return {"key": key, "label": label or key.split(":", 1)[1], "own": own, "mb": mb, "pid": 1}


def _reading(ts, procs=None, cpu=5, util=40, vram=6000, ram=8000):
    gpu = {"util": util, "vram_used_mb": vram, "vram_total_mb": 12288, "name": "x"}
    if procs is not None:
        gpu["procs"] = procs
    return {"ts": ts, "cpu": cpu, "ram_used_mb": ram, "ram_total_mb": 64000, "gpu": gpu}


def _fill(times, **kw):
    for t in times:
        metrics._ring.append(metrics._live_entry(_reading(t, **kw), 7))


# ── The buffer ───────────────────────────────────────────────────────────


class TestTheBuffer:
    def test_a_reading_as_the_view_draws_it(self):
        e = metrics._live_entry(
            _reading(
                T0 + 0.12345,
                [
                    _proc("own:server", 2000, True, "whisper-tm server"),
                    _proc("foreign:chrome", 100, False),
                    _proc("foreign:chrome", 50, False),  # a second renderer
                    _proc("foreign:game", None, False),  # amount unknown
                ],
            ),
            33,
        )
        assert e["t"] == round(T0 + 0.12345, 3)
        assert (e["cpu"], e["gpu"], e["ram_mb"], e["vram_mb"]) == (33, 40, 8000, 6000)
        assert e["procs"] == {"own:server": 2000, "foreign:chrome": 150}
        assert (e["vram_own_mb"], e["vram_foreign_mb"]) == (2000, 150)
        assert e["labels"]["own:server"] == ("whisper-tm server", True)

    def test_without_the_driver_no_split(self):
        e = metrics._live_entry(_reading(T0), 1)
        assert e["procs"] is None and e["vram_own_mb"] is None and e["vram_mb"] == 6000

    def test_without_a_gpu(self):
        e = metrics._live_entry(
            {"ts": T0, "cpu": 1, "ram_used_mb": 2, "ram_total_mb": 3, "gpu": None}, 4
        )
        assert e["gpu"] is None and e["vram_mb"] is None and e["ram_mb"] == 2

    def test_the_first_call_gets_the_last_90_seconds(self):
        _fill([T0 - 200 + i for i in range(201)])
        d = metrics.live(now=T0)
        ts = [r["t"] for r in d["readings"]]
        assert ts == sorted(ts)
        assert ts[0] > T0 - metrics.LIVE_SPAN_SEC - 11 and ts[0] >= T0 - 100
        assert ts[-1] == T0 and d["now"] == T0 and d["span"] == metrics.LIVE_SPAN_SEC

    def test_since_answers_only_what_is_new(self):
        _fill([T0 + i for i in range(10)])
        d = metrics.live(since=T0 + 6, now=T0 + 10)
        assert [r["t"] for r in d["readings"]] == [T0 + 7, T0 + 8, T0 + 9]
        assert metrics.live(since=T0 + 9, now=T0 + 10)["readings"] == []

    def test_since_reaches_back_no_further_than_the_buffer(self):
        _fill([T0 - 400, T0 - 299, T0])
        d = metrics.live(since=0, now=T0)
        assert [r["t"] for r in d["readings"]] == [T0 - 299, T0]

    def test_a_since_that_is_no_number_counts_as_none(self):
        _fill([T0 - 1])
        assert len(metrics.live(since=float("nan"), now=T0)["readings"]) == 1

    def test_sources_are_named_once_and_the_readings_stay_lean(self):
        _fill([T0, T0 + 1], procs=[_proc("own:server", 10, True, "whisper-tm server")])
        d = metrics.live(since=T0 - 1, now=T0 + 2)
        assert d["sources"] == {"own:server": {"label": "whisper-tm server", "own": True}}
        assert all("labels" not in r for r in d["readings"])

    def test_the_buffer_is_bounded(self):
        _fill([T0 + i for i in range(metrics._ring.maxlen + 50)])
        assert len(metrics._ring) == metrics._ring.maxlen
        assert metrics._ring.maxlen * metrics.LIVE_SEC >= metrics.LIVE_KEEP_SEC


class TestTheCpu:
    def test_the_buffer_counts_busy_time_since_its_previous_reading(self, monkeypatch):
        import psutil

        Times = namedtuple(
            "scputimes", "user nice system idle iowait irq softirq steal guest guest_nice"
        )
        seq = iter(
            [
                Times(10, 0, 10, 80, 0, 0, 0, 0, 0, 0),
                # 100 s of CPU time later: 30 busy, 60 idle, 10 waiting
                # for I/O — and 5 of the busy ones as a guest, which
                # Linux counts in "user" already
                Times(30, 0, 20, 140, 10, 0, 0, 0, 5, 0),
            ]
        )
        monkeypatch.setattr(psutil, "cpu_times", lambda: next(seq))
        c = metrics._CpuSince()
        assert c.read() is None  # nothing to compare with yet
        assert c.read() == 30

    def test_an_extra_reading_leaves_psutils_window_alone(self, monkeypatch):
        """psutil.cpu_percent measures since its previous call — the regular
        reading's window is SAMPLE_SEC only if nothing else calls it."""
        import psutil

        calls = []
        monkeypatch.setattr(psutil, "cpu_percent", lambda interval=None: calls.append(1) or 12.0)
        from whisper_tm.core import debug

        monkeypatch.setattr(debug, "gpu_available", lambda: False)
        monkeypatch.setattr(metrics, "_amd_load", lambda: {})
        r = metrics.measure(cpu=False)
        assert calls == [] and r["cpu"] is None and r["ram_used_mb"] is not None
        assert metrics.measure()["cpu"] == 12 and calls == [1]


# ── Someone watches ──────────────────────────────────────────────────────


class Clock:
    """time.time/monotonic/perf_counter in one — moved only by the test."""

    def __init__(self, now):
        self.now = now

    def time(self):
        return self.now

    monotonic = perf_counter = time


class Stop:
    """The sampler's stop event: a wait moves the clock instead of sleeping."""

    def __init__(self, clock, until):
        self.clock, self.until = clock, until

    def wait(self, sec):
        self.clock.now += sec
        return self.clock.now >= self.until


class TestTheViewer:
    def test_a_request_counts_for_ten_seconds(self, monkeypatch):
        clock = Clock(T0)
        monkeypatch.setattr(metrics, "time", clock)
        assert not metrics.watched() and metrics.live_step() == metrics.SAMPLE_SEC
        metrics.watch()
        clock.now += metrics.LIVE_WATCH_SEC - 0.5
        assert metrics.watched() and metrics.live_step() == metrics.LIVE_SEC
        clock.now += 1
        assert not metrics.watched() and metrics.live_step() == metrics.SAMPLE_SEC

    def test_an_expensive_reading_keeps_the_regular_step(self, monkeypatch):
        metrics.watch()
        monkeypatch.setattr(metrics, "_cost", metrics.LIVE_MAX_COST_SEC + 0.01)
        assert metrics.live_step() == metrics.SAMPLE_SEC
        monkeypatch.setattr(metrics, "_cost", 0.0014)  # NVML: about 1.4 ms
        assert metrics.live_step() == metrics.LIVE_SEC


def _sample(monkeypatch, seconds, watch_until=0.0, cost=0.0, start_cost=None):
    """Run the real sampler loop for ``seconds`` of fake time.

    Regular readings (``cpu=True``) and the live view's extras
    (``cpu=False``) get clearly different values, so an extra reading that
    leaked into the rows or the meters would show at once."""
    clock = Clock(T0)
    monkeypatch.setattr(metrics, "time", clock)
    monkeypatch.setattr(metrics, "_stop", Stop(clock, T0 + seconds))
    monkeypatch.setattr(metrics, "_watch_until", watch_until)
    monkeypatch.setattr(metrics, "_cost", start_cost)
    monkeypatch.setattr(metrics, "_ring", deque(maxlen=metrics._ring.maxlen))
    monkeypatch.setattr(metrics, "_pending", [])
    monkeypatch.setattr(metrics, "_latest", None)
    monkeypatch.setattr(metrics, "_cpu_live", SimpleNamespace(read=lambda: 77))
    taken, meters, rows = [], [], []

    def measure(cpu=True):
        r = {
            "ts": clock.now,
            "cpu": 10 + int(clock.now) % 5 if cpu else None,
            "ram_used_mb": 1000 if cpu else 9000,
            "ram_total_mb": 64000,
            "gpu": {
                "util": 20 if cpu else 95,
                "vram_used_mb": 4000 if cpu else 11000,
                "vram_total_mb": 12288,
                "name": "x",
            },
        }
        clock.now += cost
        taken.append(("regular" if cpu else "extra", r["ts"] - T0))
        return r

    monkeypatch.setattr(metrics, "measure", measure)
    monkeypatch.setattr(metrics, "write", lambda path, row: rows.append(row))
    monkeypatch.setattr(metrics, "prune", lambda path: 0)

    # What the top bar would see: _latest after every regular reading
    real_remember = metrics._remember

    def remember(r):
        meters.append(metrics._latest["ts"] - T0 if metrics._latest else None)
        real_remember(r)

    monkeypatch.setattr(metrics, "_remember", remember)
    metrics._run(None)
    return SimpleNamespace(
        taken=taken[1:],
        rows=rows,
        meters=meters,  # [0]: the priming
        ring=list(metrics._ring),
        latest=metrics._latest,
    )


class TestTheSampler:
    def test_nobody_watching_no_extra_readings(self, monkeypatch):
        run = _sample(monkeypatch, 61)
        assert {kind for kind, _ in run.taken} == {"regular"}
        assert [t for _, t in run.taken] == [2.0 * k for k in range(1, 31)]
        assert len(run.ring) == 30  # the buffer fills anyway

    def test_a_viewer_gets_a_reading_every_second(self, monkeypatch):
        run = _sample(monkeypatch, 61, watch_until=T0 + 1000)
        assert [t for _, t in run.taken] == [float(k) for k in range(1, 61)]
        assert [kind for kind, _ in run.taken[:4]] == ["extra", "regular", "extra", "regular"]
        ts = [e["t"] - T0 for e in run.ring]
        assert ts == [float(k) for k in range(1, 61)]
        # The buffer's own CPU figure for both kinds
        assert {e["cpu"] for e in run.ring} == {77}

    def test_the_30s_rows_are_the_same_with_and_without_a_viewer(self, monkeypatch):
        alone = _sample(monkeypatch, 95)
        watched = _sample(monkeypatch, 95, watch_until=T0 + 1000)
        assert len(alone.rows) == 3
        assert watched.rows == alone.rows
        # Nothing of an extra reading in them: 15 regular readings each
        for row in watched.rows:
            assert row["gpu"] == row["gpu_max"] == 20
            assert row["ram_mb"] == 1000 and row["vram_mb"] == 4000

    def test_the_meters_only_ever_see_regular_readings(self, monkeypatch):
        run = _sample(monkeypatch, 31, watch_until=T0 + 1000)
        seen = [t for t in run.meters if t is not None]
        assert seen and all(t % 2 == 0 for t in seen)
        assert run.latest["ts"] - T0 == 30.0 and run.latest["cpu"] is not None

    def test_the_extra_readings_end_with_the_viewer(self, monkeypatch):
        run = _sample(monkeypatch, 41, watch_until=T0 + 10)
        extras = [t for kind, t in run.taken if kind == "extra"]
        assert extras == [1.0, 3.0, 5.0, 7.0, 9.0]
        assert [t for kind, t in run.taken if kind == "regular"] == [2.0 * k for k in range(1, 21)]

    def test_expensive_readings_stay_at_the_regular_step(self, monkeypatch):
        """Through a slow driver (> 150 ms a reading) the live view gets the
        regular 2-s readings, not one more reading per second."""
        run = _sample(monkeypatch, 30, watch_until=T0 + 1000, cost=0.2, start_cost=0.2)
        assert {kind for kind, _ in run.taken} == {"regular"}


# ── The endpoint ─────────────────────────────────────────────────────────


class TestTheEndpoint:
    @pytest.fixture
    def client(self):
        from whisper_tm.api.health import health_bp

        app = Flask(__name__)
        app.register_blueprint(health_bp)
        return app.test_client()

    def test_the_view_asks_and_watches(self, client):
        import time

        now = time.time()
        _fill(
            [now - 3, now - 2, now - 1],
            procs=[
                _proc("own:server", 900, True, "whisper-tm server"),
                _proc("foreign:game", 300, False),
            ],
        )
        assert not metrics.watched()
        d = client.get("/api/system/live").get_json()
        assert metrics.watched()  # the sampler goes finer now
        assert d["step"] == metrics.LIVE_SEC
        assert len(d["readings"]) == 3
        r = d["readings"][-1]
        assert r["procs"] == {"own:server": 900, "foreign:game": 300}
        assert (r["vram_own_mb"], r["vram_foreign_mb"]) == (900, 300)
        assert d["sources"]["foreign:game"] == {"label": "game", "own": False}

    def test_since_gets_the_delta(self, client):
        import time

        now = time.time()
        _fill([now - 3, now - 2, now - 1])
        first = client.get("/api/system/live").get_json()["readings"]
        d = client.get(f"/api/system/live?since={first[1]['t']}").get_json()
        assert [r["t"] for r in d["readings"]] == [first[2]["t"]]

    def test_a_since_that_is_no_number_is_ignored(self, client):
        import time

        _fill([time.time() - 1])
        assert len(client.get("/api/system/live?since=abc").get_json()["readings"]) == 1

    def test_the_meters_stay_as_they_were(self, client, monkeypatch):
        """/api/system/load answers the regular reading, never the buffer's."""
        import time

        monkeypatch.setattr(metrics, "_latest", _reading(time.time(), cpu=42))
        _fill([time.time()])  # the buffer says 7
        assert client.get("/api/system/load").get_json()["cpu"] == 42
