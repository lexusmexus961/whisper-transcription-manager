"""Load history (``core/metrics.py``) behind the performance view.

Rows are written directly with fixed timestamps, so every test shows which
numbers must come out — no sampler thread, no real CPU.
"""

from __future__ import annotations

import pytest
from flask import Flask

from whisper_tm.core import metrics

T0 = 1_790_000_000.0


def _row(ts, cpu=10.0, gpu=50.0, ram=8000.0, vram=4000.0, **kw):
    r = {
        "ts": ts,
        "cpu": cpu,
        "cpu_max": cpu,
        "gpu": gpu,
        "gpu_max": gpu,
        "ram_mb": ram,
        "ram_total_mb": 64000.0,
        "vram_mb": vram,
        "vram_total_mb": 12288.0,
    }
    r.update(kw)
    return r


def _reading(ts, cpu, util, vram=4000, ram=8000):
    return {
        "ts": ts,
        "cpu": cpu,
        "ram_used_mb": ram,
        "ram_total_mb": 64000,
        "gpu": {"util": util, "vram_used_mb": vram, "vram_total_mb": 12288, "name": "x"},
    }


@pytest.fixture
def db(tmp_path):
    return tmp_path / "metrics.db"


class TestAggregate:
    def test_mean_and_peak(self):
        row = metrics.aggregate([_reading(T0, 10, 20), _reading(T0 + 2, 30, 100)])
        assert row["ts"] == T0 + 2
        assert row["cpu"] == 20 and row["cpu_max"] == 30
        assert row["gpu"] == 60 and row["gpu_max"] == 100
        assert row["vram_total_mb"] == 12288

    def test_without_gpu(self):
        r = {"ts": T0, "cpu": 5, "ram_used_mb": 1, "ram_total_mb": 2, "gpu": None}
        row = metrics.aggregate([r])
        assert row["gpu"] is None and row["vram_mb"] is None and row["cpu"] == 5

    def test_nothing_to_aggregate(self):
        assert metrics.aggregate([]) is None


class TestHistory:
    def test_empty_without_file(self, db):
        h = metrics.history(db, start=T0, end=T0 + 3600)
        assert h["points"] == [] and h["first"] is None

    def test_buckets_average_and_keep_the_peak(self, db):
        # 20 rows, 30 s apart = 10 minutes; asked for 2 points → 5-minute buckets
        for i in range(20):
            metrics.write(db, _row(T0 + i * 30, cpu=float(i), gpu=100.0 if i == 3 else 0.0))
        h = metrics.history(db, start=T0, end=T0 + 600, max_points=2)
        assert h["bucket"] == 300
        assert len(h["points"]) <= 3
        first = h["points"][0]
        assert first["gpu_max"] == 100
        assert 0 < first["gpu"] < 100  # the mean, not the peak

    def test_never_finer_than_the_write_interval(self, db):
        h = metrics.history(db, start=T0, end=T0 + 60, max_points=600)
        assert h["bucket"] == metrics.WRITE_SEC

    def test_the_range_cuts(self, db):
        for i in range(10):
            metrics.write(db, _row(T0 + i * 3600))
        h = metrics.history(db, start=T0 + 2 * 3600, end=T0 + 5 * 3600, max_points=600)
        ts = [p["t"] for p in h["points"]]
        assert min(ts) >= T0 + 2 * 3600 - h["bucket"] and max(ts) <= T0 + 5 * 3600 + h["bucket"]
        assert len(h["points"]) == 4
        assert h["first"] == T0

    def test_prune_drops_old_rows(self, db):
        metrics.write(db, _row(T0))
        metrics.write(db, _row(T0 + metrics.RETENTION_DAYS * 86400 + 10))
        assert metrics.prune(db, now=T0 + metrics.RETENTION_DAYS * 86400 + 20) == 1


class TestEndpoints:
    @pytest.fixture
    def client(self, db, monkeypatch):
        monkeypatch.setattr(metrics, "_db_path", db)
        from whisper_tm.api.health import health_bp

        app = Flask(__name__)
        app.register_blueprint(health_bp)
        return app.test_client()

    def test_history(self, client, db):
        for i in range(5):
            metrics.write(db, _row(T0 + i * 30))
        d = client.get(f"/api/system/history?start={T0 - 10}&end={T0 + 200}").get_json()
        assert d["points"] and d["bucket"] >= metrics.WRITE_SEC

    def test_history_defaults_to_a_day(self, client):
        d = client.get("/api/system/history").get_json()
        assert round(d["end"] - d["start"]) == 86400

    @pytest.mark.parametrize(
        "query",
        ["start=-1e308&end=1e308", "start=nan&end=inf", "start=-inf&end=nan", "start=1e308"],
    )
    def test_history_takes_only_finite_times(self, client, db, query):
        # ±1e308 used to give a 500 ("no such column: inf"), nan/inf bare NaN in the JSON
        import json
        import math

        metrics.write(db, _row(T0))
        r = client.get(f"/api/system/history?{query}")
        assert r.status_code == 200
        d = json.loads(r.get_data(as_text=True), parse_constant=lambda c: pytest.fail(c))
        assert all(math.isfinite(d[k]) for k in ("start", "end", "bucket"))
        assert d["end"] - d["start"] <= metrics.HISTORY_SPAN_MAX_SEC

    def test_load_uses_the_sampler_reading(self, client, monkeypatch):
        import time

        monkeypatch.setattr(metrics, "_latest", _reading(time.time(), 42, 7))
        d = client.get("/api/system/load").get_json()
        assert d["cpu"] == 42 and d["gpu"]["util"] == 7
        assert "ts" not in d
