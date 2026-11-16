"""Incidents: the memory for everything that went wrong.

**Reason:** A file could be missing most of its speaker vectors (GPU
full) and still show as ✅ in the list. Which other files suffered the same fate could
only be reconstructed from randomly surviving logs — the job itself lives
for one hour, `job_history` caps at 10,000 lines, and nobody searches a
directory of many thousands of log files. Errors must therefore be saved
permanently, so what went wrong can be looked up at any time.

The entire chain is tested: Table → `report()` → Job events
(warn/fail/system) → OOM retry of queue → HTTP API → MCP tool.
"""

from __future__ import annotations

import time

import pytest

from whisper_tm.database import Database


@pytest.fixture
def db(tmp_path, monkeypatch):
    d = Database(tmp_path / "test.db")
    d.initialize()
    import whisper_tm.extensions as ext

    monkeypatch.setattr(ext, "db", d)
    return d


class TestTheTable:
    def test_writing_and_reading(self, db):
        db.report_incident(
            "warning",
            "Voice vectors: GPU full, falling back to the CPU",
            abs_path="/m/a.mkv",
            filename="a.mkv",
            step="Voice vectors",
        )
        db.report_incident(
            "error", "Separation failed", abs_path="/m/b.mkv", filename="b.mkv", log_file="x.log"
        )
        r = db.incidents()
        assert r["total"] == 2
        assert r["kinds"] == {"warning": 1, "error": 1}
        assert r["items"][0]["abs_path"] == "/m/b.mkv"  # newest first
        assert r["items"][0]["log_file"] == "x.log"

    def test_filter_by_file_kind_and_text(self, db):
        for i in range(3):
            db.report_incident("warning", f"w{i}", abs_path="/m/a.mkv", filename="a.mkv")
        db.report_incident("oom", "VRAM exhausted", abs_path="/m/b.mkv", filename="sub/b.mkv")
        assert db.incidents(abs_path="/m/a.mkv")["total"] == 3
        assert db.incidents(filename="sub/b.mkv")["total"] == 1
        assert db.incidents(filename="b.mkv")["total"] == 1  # suffix is enough
        assert db.incidents(kind="oom")["total"] == 1
        assert db.incidents(search="exhausted")["total"] == 1
        assert db.incidents(since=time.time() + 60)["total"] == 0

    def test_the_counter_counts_events(self, db):
        db.report_incident("warning", "500 windows skipped", abs_path="/m/a.mkv", count=500)
        r = db.incidents()
        assert r["total"] == 1 and r["events"] == 500

    def test_an_unknown_kind_becomes_a_warning(self, db):
        db.report_incident("nonsense", "x")
        assert db.incidents()["items"][0]["kind"] == "warning"

    def test_per_file(self, db):
        db.report_incident("warning", "a", abs_path="/m/a.mkv")
        db.report_incident("warning", "b", abs_path="/m/a.mkv")
        db.report_incident("error", "c", abs_path="/m/c.mkv")
        assert db.incidents_per_file(["/m/a.mkv", "/m/c.mkv", "/m/none.mkv"]) == {
            "/m/a.mkv": 2,
            "/m/c.mkv": 1,
        }

    def test_the_table_is_never_truncated(self):
        """job_history has a cap — incidents intentionally not."""
        from pathlib import Path

        source = Path("whisper_tm/database/__init__.py").read_text()
        i = source.index("def report_incident")
        j = source.index("def incidents(")
        assert "DELETE" not in source[i:j]


class TestTheJobEvents:
    """warn()/fail()/system_warning() automatically land in the table."""

    @pytest.fixture
    def jm(self, tmp_path):
        from whisper_tm.core import incidents
        from whisper_tm.jobs import JobManager

        m = JobManager(logs_dir=tmp_path / "logs")
        incidents.attach(m)
        return m

    def test_a_warning_with_file_and_step(self, db, jm):
        job = jm.create("j1", filename="rel/a.mkv")
        job.abs_path = "/m/rel/a.mkv"
        job.phase = {"title": "Voice vectors", "index": 7, "total": 8}
        jm.warn(job, "GPU full, falling back to the CPU")
        r = db.incidents()
        assert r["total"] == 1
        v = r["items"][0]
        assert v["kind"] == "warning" and v["abs_path"] == "/m/rel/a.mkv"
        assert v["filename"] == "rel/a.mkv" and v["step"] == "Voice vectors"
        assert v["job_id"] == "j1"

    def test_an_error_with_a_log_file(self, db, jm):
        job = jm.create("j2", filename="b.mkv")
        job.abs_path = "/m/b.mkv"
        jm.log(job, "anything")  # to create a log file
        jm.fail(job, "Separation failed after 1 attempt: CUDA out of memory")
        v = db.incidents(kind="error")["items"][0]
        assert "out of memory" in v["text"]
        assert v["log_file"].endswith(".log")

    def test_a_user_cancel_is_not_an_incident(self, db, jm):
        job = jm.create("j3", filename="c.mkv")
        jm.fail(job, "Cancelled")
        assert db.incidents()["total"] == 0

    def test_a_system_warning_without_a_file(self, db, jm):
        jm.system_warning("Separation model switched to Fallback")
        v = db.incidents(kind="system")["items"][0]
        assert v["abs_path"] == "" and "Fallback" in v["text"]

    def test_without_a_database_nothing_breaks(self, jm, monkeypatch):
        import whisper_tm.extensions as ext

        monkeypatch.setattr(ext, "db", None)
        job = jm.create("j4", filename="d.mkv")
        jm.warn(job, "x")  # must not throw
        jm.fail(job, "y")


class TestTheQueueReportsOOM:
    def test_an_oom_retry_becomes_an_incident(self, db, monkeypatch):
        from whisper_tm.core import queue as q

        monkeypatch.setattr(q, "_requeue", lambda item_id: None)
        monkeypatch.setattr(q, "qlog", lambda msg: None)
        # no real wait for the map: immediately "free"
        from whisper_tm.core import vram

        monkeypatch.setattr(vram, "wait_for_free_vram", lambda *a, **k: (True, 9000))
        item = {"id": 7, "filename": "rel/x.mkv", "base_dir": "/m", "settings": {"low_vram": False}}
        # DB update of settings runs against our test DB (queue_items
        # empty → UPDATE hits nothing, but also doesn't throw)
        assert q._oom_retry(item, "RuntimeError: CUDA failed with error out of memory") is True
        v = db.incidents(kind="oom")["items"]
        assert len(v) == 1
        assert v[0]["abs_path"] == "/m/rel/x.mkv" and v[0]["filename"] == "rel/x.mkv"
        assert "VRAM conservation" in v[0]["text"] and "attempt 1" in v[0]["text"]


class TestTheApi:
    @pytest.fixture
    def client(self, db, monkeypatch):
        from flask import Flask

        import whisper_tm.api.incidents as api

        monkeypatch.setattr(api, "db", db)
        app = Flask(__name__)
        app.register_blueprint(api.incidents_bp)
        return app.test_client()

    def test_listing_and_filter(self, client, db):
        db.report_incident("warning", "w", abs_path="/m/a.mkv", filename="a.mkv")
        db.report_incident("error", "f", abs_path="/m/b.mkv", filename="b.mkv")
        r = client.get("/api/incidents").get_json()
        assert r["total"] == 2 and len(r["items"]) == 2
        assert client.get("/api/incidents?file=a.mkv").get_json()["total"] == 1
        assert client.get("/api/incidents?file=/m/b.mkv").get_json()["total"] == 1
        assert client.get("/api/incidents?kind=error").get_json()["total"] == 1
        assert client.get("/api/incidents?q=w").get_json()["total"] == 1
        assert client.get("/api/incidents?kind=whatever").status_code == 400

    def test_limit_and_offset(self, client, db):
        for i in range(5):
            db.report_incident("warning", f"w{i}")
        r = client.get("/api/incidents?limit=2&offset=2").get_json()
        assert r["total"] == 5 and len(r["items"]) == 2
        assert r["items"][0]["text"] == "w2"


class TestTheMcpTool:
    def test_incidents_translates_the_kinds(self, monkeypatch):
        from whisper_tm import mcp_server as m

        calls = []

        class _V:
            def fetch(self, path, **params):
                calls.append((path, params))
                return {
                    "total": 1,
                    "events": 23,
                    "kinds": {"warning": 1},
                    "offset": 0,
                    "limit": 50,
                    "items": [
                        {
                            "timestamp": 1_800_000_000,
                            "filename": "a.mkv",
                            "abs_path": "/m/a.mkv",
                            "kind": "warning",
                            "step": "Voice vectors",
                            "text": "GPU full",
                            "count": 23,
                            "log_file": "a.log",
                        }
                    ],
                }

        monkeypatch.setattr(m, "conn", lambda: _V())
        import asyncio

        server = m.build_server()
        tool = asyncio.run(server.get_tool("incidents"))
        r = tool.fn(file="a.mkv", kind="warning", days=7)
        assert calls[0][1]["kind"] == "warning" and calls[0][1]["days"] == 7
        assert r["by_kind"] == {"warning": 1}
        v = r["incidents"][0]
        assert v["kind"] == "warning" and v["count"] == 23
        assert v["step"] == "Voice vectors" and v["log_file"] == "a.log"
        assert v["time"].startswith("20")


# ── The UI: Chip in details + collapsible list ─────────────

import json
import os
import shutil
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
HARNESS = REPO / "tests" / "js" / "fileview_harness.js"
FILEVIEW = REPO / "whisper_tm" / "static" / "js" / "fileview.js"


def _js(body: str) -> dict:
    # esc() and fmtDate() come from state.js/utils — supplied in harness
    source = (
        f"const {{buildIt}} = require({str(HARNESS)!r});\n"
        "const h = buildIt('/mnt/media/videos');\n"
        "h.run(`function esc(s){return String(s??'').replace(/&/g,'&amp;').replace(/</g,'&lt;')"
        ".replace(/>/g,'&gt;').replace(/\"/g,'&quot;');}"
        " function escJs(s){return String(s??'');}"
        " function fmtDate(x){return x?String(x):'';}"
        " function chip(){return '';}`);\n"
        "const output = (o) => console.log('§§' + JSON.stringify(o));\n"
        f"{body}\n"
    )
    done = subprocess.run(
        ["node", "-e", source],
        capture_output=True,
        text=True,
        timeout=60,
        env={"FILEVIEW_JS": str(FILEVIEW), "PATH": os.environ.get("PATH", "")},
    )
    if done.returncode != 0:
        raise AssertionError(f"node failed:\n{done.stderr}")
    lines = [z for z in done.stdout.splitlines() if z.startswith("§§")]
    assert lines, f"no result:\n{done.stdout}\n{done.stderr}"
    return json.loads(lines[-1][2:])


@pytest.mark.skipif(shutil.which("node") is None, reason="node not available")
class TestTheInterface:
    def test_the_chip_is_in_the_details_even_without_a_transcript(self):
        # The Details list (recording.js) — also under a recording that has
        # no transcript yet (Rec.stateHtml puts it there)
        r = _js("""
            h.sandbox.queueMicrotask = (fn) => {};
            const html = h.value("Rec.detailsHtml({name: 'a/b.mkv', status: 'pending', size_fmt: '1 MB'}, null)");
            const state = h.value("Rec.stateHtml({name: 'a/b.mkv', status: 'pending', size_fmt: '1 MB'})");
            output({html, state});
        """)
        html = r["html"]
        assert 'class="fi-chip fi-incidents"' in html
        assert 'data-file="/mnt/media/videos/a/b.mkv"' in html
        assert html.index("File and media") < html.index("fi-incidents")
        assert 'class="fi-chip fi-incidents"' in r["state"]

    def test_the_list_shows_time_step_text_and_counter(self):
        r = _js("""
            const d = {total: 2, events: 24, kinds: {warning: 1, error: 1}, items: [
                {timestamp: 1_800_000_000, kind: 'warning', step: 'Voice vectors', text: 'GPU full', count: 23},
                {timestamp: 1_800_000_100, kind: 'error', step: '', text: 'Separation <broken>', count: 1},
            ]};
            h.sandbox.__d = d;
            output({html: h.value("fiIncidentsHtml(__d)")});
        """)
        html = r["html"]
        assert "2 incidents (24 events)" in html
        assert "Voice vectors" in html and "×23" in html
        assert "&lt;broken&gt;" in html  # escaped
        assert "1 warning" in html and "1 error" in html

    def test_no_incidents_says_so_too(self):
        r = _js("""output({html: h.value("fiIncidentsHtml({total: 0, items: []})")});""")
        assert "No incidents" in r["html"]


def test_a_state_warning_replaces_its_predecessor(tmp_path):
    """The memory warning describes a state and fires after every job while
    it lasts; with a key, the newest replaces the older instead of stacking
    (otherwise five near-identical lines can cover the header)."""
    from whisper_tm.jobs import JobManager

    m = JobManager(logs_dir=tmp_path / "logs")
    m.system_warning("event one")
    m.system_warning("memory 30 GiB", key="memory")
    m.system_warning("memory 31 GiB", key="memory")
    assert [w["text"] for w in m.system_warnings()] == ["event one", "memory 31 GiB"]
