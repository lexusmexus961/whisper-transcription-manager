"""VRAM by process (core/vram_procs.py) and its history (core/metrics.py).

Shows which process uses VRAM, so foreign consumers are visible too.
Checked: the driver's report is read per GPU,
this app's processes are told from the rest (by role where it knows them),
the history keeps who held how much per interval — and rows from before
the split stay without one instead of inventing it.
"""

from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
import time

import pytest
from flask import Flask

from whisper_tm.core import metrics, vram_procs

T0 = 1_790_000_000.0

REPORT = """
==============NVSMI LOG==============

Timestamp                                              : Mon Jan  1 12:00:00 2024
Driver Version                                         : 615.71.09

Attached GPUs                                          : 2
GPU 00000000:09:00.0
    Processes
        GPU instance ID                   : N/A
        Compute instance ID               : N/A
        Process ID                        : 91301
            Type                          : C+G
            Name                          : /usr/bin/kwin_wayland
            Used GPU Memory               : 14 MiB
        GPU instance ID                   : N/A
        Compute instance ID               : N/A
        Process ID                        : 92715
            Type                          : C
            Name                          : /app/discord/Discord --type=renderer --a=b,c --x=y:z
            Used GPU Memory               : 251 MiB
        GPU instance ID                   : N/A
        Compute instance ID               : N/A
        Process ID                        : 777
            Type                          : G
            Name                          : game.exe
            Used GPU Memory               : Not available

GPU 00000000:0A:00.0
    Processes                             : None
"""


class TestTheDriverReport:
    def test_every_gpu_and_process(self):
        gpus = vram_procs.parse_smi(REPORT)
        assert list(gpus) == ["00000000:09:00.0", "00000000:0a:00.0"]
        first = gpus["00000000:09:00.0"]
        assert [p["pid"] for p in first] == [91301, 92715, 777]
        assert first[0] == {"pid": 91301, "type": "C+G", "name": "/usr/bin/kwin_wayland", "mb": 14}
        # A name with colons and commas stays whole
        assert first[1]["name"].endswith("--a=b,c --x=y:z") and first[1]["mb"] == 251
        # Graphics-only, memory unknown: listed, not counted
        assert first[2]["type"] == "G" and first[2]["mb"] is None
        assert gpus["00000000:0a:00.0"] == []

    def test_the_right_card_by_bus_id(self, monkeypatch):
        monkeypatch.setattr(vram_procs, "_smi_report", lambda: REPORT)
        monkeypatch.setattr(
            vram_procs, "_bus_to_index", {"00000000:09:00.0": 1, "00000000:0a:00.0": 0}
        )
        from whisper_tm.core import debug

        monkeypatch.setattr(debug, "_init_gpu_backend", lambda: "smi")
        assert vram_procs.raw_processes(0) == []
        assert [p["pid"] for p in vram_procs.raw_processes(1)] == [91301, 92715, 777]

    def test_no_driver_no_list(self, monkeypatch):
        from whisper_tm.core import debug

        monkeypatch.setattr(debug, "_init_gpu_backend", lambda: "none")
        assert vram_procs.raw_processes(0) is None


class TestWhoIsWho:
    @pytest.mark.parametrize(
        "name,cmd,want",
        [
            (
                "python3.12",
                ["python", "-m", "whisper_tm", "--host", "0.0.0.0"],
                "python -m whisper_tm",
            ),
            ("python", ["/x/python", "/srv/tools/train.py", "--lr", "3"], "python train.py"),
            ("Discord", ["/app/discord/Discord", "--type=renderer"], "Discord"),
            ("", ["/usr/bin/kwin_wayland"], "kwin_wayland"),
        ],
    )
    def test_program_names(self, name, cmd, want):
        assert vram_procs._program(name, cmd) == want

    def test_this_server_is_own_and_named(self):
        d = vram_procs.describe(os.getpid(), roles={os.getpid(): "server"})
        assert d["own"] is True and d["role"] == "server"

    def test_a_child_is_own_a_stranger_is_not(self):
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        try:
            d = vram_procs.describe(child.pid, roles={})
            assert d["own"] is True and d["role"] is None
            assert d["program"].startswith("python")
        finally:
            child.kill()
            child.wait()
        # pid 1 is nobody's child of ours
        assert vram_procs.describe(1, roles={})["own"] is False

    def test_a_vanished_process_keeps_the_drivers_name(self):
        d = vram_procs.describe(2**22 + 12345, "/opt/game/bin/game --fullscreen", roles={})
        assert d == {"program": "game", "user": None, "own": False, "role": None}

    def test_no_command_line_is_kept(self):
        d = vram_procs.describe(os.getpid(), roles={})
        assert set(d) == {"program", "user", "own", "role"}

    def test_the_list_groups_by_role_and_program(self, monkeypatch):
        me = os.getpid()
        raw = [
            {"pid": me, "type": "C", "name": "python", "mb": 3000},
            {"pid": 91301, "type": "C+G", "name": "/usr/bin/kwin_wayland", "mb": 14},
            {"pid": 555, "type": "C", "name": "worker", "mb": 1200},
        ]
        monkeypatch.setattr(vram_procs, "raw_processes", lambda idx: raw)
        monkeypatch.setattr(vram_procs, "_role_pids", lambda: {me: "server", 555: "diarization"})
        real = vram_procs.describe

        def fake(pid, smi_name="", roles=None):
            if pid == 555:
                return {"program": "python -m x", "user": "u", "own": True, "role": "diarization"}
            if pid == 91301:
                return {"program": "kwin_wayland", "user": "u", "own": False, "role": None}
            return real(pid, smi_name, roles)

        monkeypatch.setattr(vram_procs, "describe", fake)
        procs = vram_procs.processes(0)
        assert [p["pid"] for p in procs] == [me, 555, 91301]  # largest first
        assert [p["key"] for p in procs] == [
            "own:server",
            "own:diarization",
            "foreign:kwin_wayland",
        ]
        assert procs[0]["label"] == "whisper-tm server"
        assert procs[1]["label"] == "Diarization worker"


class TestWhichGpu:
    """Only the GPU WhisperX runs on — the card this app holds memory on."""

    @staticmethod
    def _lists(monkeypatch, lists, own_pids):
        monkeypatch.setattr(vram_procs, "gpu_lists", lambda: lists)
        monkeypatch.setattr(vram_procs, "_role_pids", lambda: {})
        monkeypatch.setattr(
            vram_procs,
            "describe",
            lambda pid, name="", roles=None: {
                "program": name or str(pid),
                "user": "u",
                "own": pid in own_pids,
                "role": None,
            },
        )

    def test_where_the_app_holds_memory(self, monkeypatch):
        self._lists(
            monkeypatch,
            {
                0: [{"pid": 10, "type": "C", "name": "game", "mb": 6000}],
                1: [
                    {"pid": 20, "type": "C", "name": "whisperx", "mb": 3000},
                    {"pid": 30, "type": "C+G", "name": "kwin", "mb": 14},
                ],
            },
            own_pids={20},
        )
        idx, procs = vram_procs.snapshot(configured=0)
        assert idx == 1
        assert [p["pid"] for p in procs] == [20, 30]  # the game is not here

    def test_before_it_holds_any_the_configured_one(self, monkeypatch):
        self._lists(
            monkeypatch,
            {0: [{"pid": 10, "type": "C", "name": "game", "mb": 6000}], 1: []},
            own_pids=set(),
        )
        assert vram_procs.snapshot(configured=1) == (1, [])

    def test_without_the_driver(self, monkeypatch):
        monkeypatch.setattr(vram_procs, "gpu_lists", lambda: None)
        assert vram_procs.snapshot(configured=0) == (0, None)

    def test_the_sampler_measures_that_card(self, monkeypatch):
        from whisper_tm.core import debug

        monkeypatch.setattr(debug, "gpu_available", lambda: True)
        monkeypatch.setattr(
            debug,
            "gpu_static",
            lambda: {
                0: {"name": "Other card", "total_mb": 8000},
                1: {"name": "Test GPU", "total_mb": 12288},
            },
        )
        monkeypatch.setattr(debug, "gpu_used_and_util", lambda: {0: (100, 5), 1: (5000, 90)})
        monkeypatch.setattr(vram_procs, "snapshot", lambda configured: (1, []))
        g = metrics.measure()["gpu"]
        assert (g["index"], g["name"], g["vram_used_mb"], g["util"]) == (1, "Test GPU", 5000, 90)


# ── The history ──────────────────────────────────────────────────────────


def _reading(ts, procs, vram=4000):
    return {
        "ts": ts,
        "cpu": 5,
        "ram_used_mb": 1,
        "ram_total_mb": 2,
        "gpu": {
            "util": 1,
            "vram_used_mb": vram,
            "vram_total_mb": 12288,
            "name": "x",
            "procs": procs,
        },
    }


def _p(key, mb, own):
    return {"key": key, "label": key.split(":", 1)[1], "own": own, "mb": mb, "pid": 1}


class TestAggregate:
    def test_mean_with_absence_as_zero_and_the_peak(self):
        row = metrics.aggregate(
            [
                _reading(T0, [_p("own:server", 3000, True), _p("foreign:game", 600, False)]),
                _reading(T0 + 2, [_p("own:server", 1000, True)]),
            ]
        )
        got = {p["key"]: p for p in row["procs"]}
        assert got["own:server"]["mb"] == 2000 and got["own:server"]["mb_max"] == 3000
        assert got["foreign:game"]["mb"] == 300 and got["foreign:game"]["mb_max"] == 600
        assert row["vram_own_mb"] == 2000 and row["vram_foreign_mb"] == 300

    def test_processes_of_one_program_add_up(self):
        row = metrics.aggregate(
            [_reading(T0, [_p("foreign:chrome", 100, False), _p("foreign:chrome", 250, False)])]
        )
        assert row["procs"] == [
            {
                "key": "foreign:chrome",
                "label": "chrome",
                "own": 0,
                "mb": 350,
                "mb_max": 350,
                "mb_min": 350,
            }
        ]

    def test_the_lowest_reading_counts_absence_as_zero(self):
        row = metrics.aggregate(
            [
                _reading(T0, [_p("own:diarization", 900, True)]),
                _reading(T0 + 2, [_p("own:diarization", 2500, True)]),
                _reading(T0 + 4, []),  # the worker was gone
            ]
        )
        (d,) = row["procs"]
        assert (d["mb_min"], d["mb"], d["mb_max"]) == (0.0, 3400 / 3, 2500)

    def test_unknown_memory_is_not_counted(self):
        row = metrics.aggregate([_reading(T0, [{**_p("foreign:game", None, False)}])])
        assert row["procs"] == [] and row["vram_foreign_mb"] == 0

    def test_without_the_split_nothing_is_invented(self):
        r = _reading(T0, None)
        del r["gpu"]["procs"]
        row = metrics.aggregate([r])
        assert row["vram_own_mb"] is None and row["procs"] == []


@pytest.fixture
def db(tmp_path):
    return tmp_path / "metrics.db"


def _write(db, ts, procs, vram=4000):
    metrics.write(db, metrics.aggregate([_reading(ts, procs, vram)]))


class TestHistory:
    def test_sources_per_point(self, db):
        for i in range(4):
            _write(
                db, T0 + i * 30, [_p("own:server", 3000, True), _p("foreign:game", 500 * i, False)]
            )
        h = metrics.history(db, start=T0 - 1, end=T0 + 120, max_points=600)
        keys = [k["key"] for k in h["vram_by"]["keys"]]
        assert keys == ["own:server", "foreign:game"]  # own first
        assert h["vram_by"]["keys"][1]["own"] is False
        assert h["vram_by"]["series"]["own:server"] == [3000.0] * 4
        assert h["vram_by"]["series"]["foreign:game"] == [0.0, 500.0, 1000.0, 1500.0]
        assert [p["vram_own_mb"] for p in h["points"]] == [3000.0] * 4

    def test_rows_from_before_have_no_split(self, db):
        metrics.write(
            db,
            {
                "ts": T0,
                "cpu": 1,
                "cpu_max": 1,
                "gpu": 1,
                "gpu_max": 1,
                "ram_mb": 1,
                "ram_total_mb": 2,
                "vram_mb": 4000,
                "vram_total_mb": 12288,
            },
        )
        _write(db, T0 + 30, [_p("own:server", 3000, True)])
        h = metrics.history(db, start=T0 - 1, end=T0 + 60, max_points=600)
        assert [p["vram_own_mb"] for p in h["points"]] == [None, 3000.0]
        assert h["vram_by"]["series"]["own:server"] == [None, 3000.0]

    def test_a_bucket_averages_its_intervals(self, db):
        # Two intervals in one 60-s bucket: the game held 800 in one of them
        _write(db, T0 + 1, [_p("own:server", 1000, True), _p("foreign:game", 800, False)])
        _write(db, T0 + 31, [_p("own:server", 1000, True)])
        h = metrics.history(db, start=T0, end=T0 + 60, max_points=1)
        assert h["vram_by"]["series"]["foreign:game"] == [400.0]

    def test_the_rest_is_summed(self, db, monkeypatch):
        monkeypatch.setattr(metrics, "HISTORY_SOURCES", 2)
        _write(db, T0, [_p(f"foreign:p{i}", 100 * (i + 1), False) for i in range(5)])
        h = metrics.history(db, start=T0 - 1, end=T0 + 1, max_points=600)
        s = h["vram_by"]["series"]
        assert set(s) == {"foreign:p4", "foreign:p3", "foreign:other"}
        assert s["foreign:other"] == [600.0]  # 100 + 200 + 300
        labels = {k["key"]: k["label"] for k in h["vram_by"]["keys"]}
        assert labels["foreign:other"] == "Other programs"

    def test_each_own_process_shows_how_far_it_swung(self, db):
        """How much the diarization swings: the lowest and the highest
        reading per bucket, next to the mean."""
        metrics.write(
            db,
            metrics.aggregate(
                [
                    _reading(
                        T0 + 1, [_p("own:diarization", 400, True), _p("own:server", 3000, True)]
                    ),
                    _reading(
                        T0 + 3, [_p("own:diarization", 3800, True), _p("own:server", 3000, True)]
                    ),
                ]
            ),
        )
        metrics.write(
            db,
            metrics.aggregate(
                [
                    _reading(
                        T0 + 31, [_p("own:diarization", 1200, True), _p("own:server", 3000, True)]
                    ),
                ]
            ),
        )
        h = metrics.history(db, start=T0, end=T0 + 60, max_points=1)  # one 60-s bucket
        by = h["vram_by"]
        assert by["low"]["own:diarization"] == [400.0]
        assert by["high"]["own:diarization"] == [3800.0]
        assert by["series"]["own:diarization"] == [(2100 + 1200) / 2]
        assert by["low"]["own:server"] == by["high"]["own:server"] == [3000.0]

    def test_a_process_that_came_in_the_bucket_starts_at_zero(self, db):
        _write(db, T0 + 1, [_p("own:server", 1000, True)])
        _write(db, T0 + 31, [_p("own:server", 1000, True), _p("own:diarization", 800, True)])
        h = metrics.history(db, start=T0, end=T0 + 60, max_points=1)
        assert h["vram_by"]["low"]["own:diarization"] == [0.0]
        assert h["vram_by"]["high"]["own:diarization"] == [800.0]

    def test_before_the_lowest_was_recorded_the_swing_is_unknown(self, db):
        """Rows from before mb_min existed: no band — not a band down to 0."""
        _write(db, T0 + 1, [_p("own:server", 1000, True)])
        with sqlite3.connect(db) as c:
            c.execute("UPDATE vram_procs SET mb_min = NULL")
        _write(db, T0 + 61, [_p("own:server", 1200, True)])
        h = metrics.history(db, start=T0, end=T0 + 120, max_points=2)  # 60-s buckets
        assert h["vram_by"]["low"]["own:server"] == [None, 1200.0]
        assert h["vram_by"]["high"]["own:server"] == [None, 1200.0]
        assert h["vram_by"]["series"]["own:server"] == [1000.0, 1200.0]

    def test_summed_sources_have_no_swing(self, db, monkeypatch):
        monkeypatch.setattr(metrics, "HISTORY_SOURCES", 1)
        _write(db, T0, [_p("foreign:a", 100, False), _p("foreign:b", 200, False)])
        h = metrics.history(db, start=T0 - 1, end=T0 + 1, max_points=600)
        assert "foreign:other" in h["vram_by"]["series"]
        assert "foreign:other" not in h["vram_by"]["low"]

    def test_prune_takes_the_sources_along(self, db):
        _write(db, T0, [_p("own:server", 1, True)])
        metrics.prune(db, now=T0 + metrics.RETENTION_DAYS * 86400 + 10)
        with sqlite3.connect(db) as c:
            assert c.execute("SELECT COUNT(*) FROM vram_procs").fetchone()[0] == 0


# ── The endpoints ────────────────────────────────────────────────────────


class TestEndpoints:
    @pytest.fixture
    def client(self, db, monkeypatch):
        monkeypatch.setattr(metrics, "_db_path", db)
        from whisper_tm.api.health import health_bp

        app = Flask(__name__)
        app.register_blueprint(health_bp)
        return app.test_client()

    def _latest(self, monkeypatch, procs):
        r = _reading(time.time(), procs, vram=5000)
        monkeypatch.setattr(metrics, "_latest", r)

    def test_the_live_split(self, client, monkeypatch):
        self._latest(
            monkeypatch,
            [
                {
                    **_p("own:server", 3000, True),
                    "role": "server",
                    "program": "python -m whisper_tm",
                    "user": "me",
                    "type": "C",
                    "label": "whisper-tm server",
                },
                {
                    **_p("foreign:game", 1500, False),
                    "role": None,
                    "program": "game",
                    "user": "me",
                    "type": "G",
                },
                {
                    **_p("foreign:x", None, False),
                    "role": None,
                    "program": "x",
                    "user": "root",
                    "type": "C",
                },
            ],
        )
        monkeypatch.setattr(vram_procs, "resident_models", lambda: ["large-v3"])
        d = client.get("/api/system/vram").get_json()
        assert d["available"] is True
        assert d["name"] == "x"  # the card it is about
        assert d["own_mb"] == 3000 and d["foreign_mb"] == 1500
        assert d["unattributed_mb"] == 500
        assert d["procs"][0]["models"] == ["large-v3"]
        assert "models" not in d["procs"][1]
        assert not any("cmd" in p for p in d["procs"])

    def test_without_the_driver(self, client, monkeypatch):
        r = _reading(time.time(), None)
        del r["gpu"]["procs"]
        monkeypatch.setattr(metrics, "_latest", r)
        d = client.get("/api/system/vram").get_json()
        assert d["available"] is False and d["procs"] == []

    def test_the_meters_stay_lean(self, client, monkeypatch):
        self._latest(monkeypatch, [_p("own:server", 1, True)])
        d = client.get("/api/system/load").get_json()
        assert "procs" not in d["gpu"] and d["gpu"]["vram_used_mb"] == 5000


class TestTheView:
    def test_the_view_asks_for_the_split(self):
        from pathlib import Path

        js = (
            Path(__file__).resolve().parent.parent / "whisper_tm" / "static" / "js" / "perfview.js"
        ).read_text(encoding="utf-8")
        assert "/api/system/vram" in js
        for fn in (
            "function pvStack(",
            "function pvStackMarks(",
            "function pvStackLegend(",
            "function pvProcsInner(",
            "function pvTipSources(",
            "function pvOwnChart(",
        ):
            assert fn in js, fn
        # The poll stops with the view
        close = js[js.index("function closePerfView") : js.index("async function pvLoadProcs")]
        assert "clearInterval(pv.procsTimer)" in close


def test_model_names_read_like_names(monkeypatch):
    """The ASR registry's keys are tuples — shown as "large-v3 (cuda,
    float16)", not as Python's "('large-v3', 'cuda', 'float16')"."""
    from whisper_tm.core import transcription, vocal_isolation

    monkeypatch.setattr(
        transcription,
        "_models",
        {("large-v3", "cuda", "float16"): 1, ("speaker_embed", "cuda"): 2, "plain": 3},
    )
    monkeypatch.setattr(vocal_isolation, "_separator", object())
    monkeypatch.setattr(vocal_isolation, "_separator_key", ("mel_roformer", "cuda"))
    assert vram_procs.resident_models() == [
        "large-v3 (cuda, float16)",
        "plain",
        "speaker_embed (cuda)",
        "Separator (mel_roformer)",
    ]
