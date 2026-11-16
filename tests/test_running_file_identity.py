"""The running file is recognized by its identity, not its name.

Bug: Click on '⟳ Processing...' in the Queue tab → the browser
jumps to the file's folder (navigate_to_file), the file is named there
only by its basename — and `f.name === queueCurrentFile` (the
queue name is relative to the entry's root) was wrong: no overlay,
no progress, instead buttons 'Transcribe'/'Add to queue'.

The snapshot now carries `current_abs`/`current_dir`, each list entry
its `abs_path`, and `isRunningFile(f)` (state.js) compares via these.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import whisper_tm.extensions as ext
from whisper_tm.core import queue as q
from whisper_tm.database import Database

REPO = Path(__file__).resolve().parent.parent
JS = REPO / "whisper_tm" / "static" / "js"


@pytest.fixture
def db(tmp_path, monkeypatch):
    q.stop_worker()
    d = Database(tmp_path / "q.db")
    d.initialize()
    monkeypatch.setattr(ext, "db", d)
    q._current = None
    return d


def test_the_snapshot_carries_the_running_files_identity(db):
    with db.connection() as conn:
        conn.execute(
            "INSERT INTO queue_items (filename, base_dir, settings_json, status, position, job_id)"
            " VALUES ('videos/a/b.mkv', '/mnt/media', '{}', 'processing', 1, 'j1')"
        )
        row = conn.execute("SELECT * FROM queue_items").fetchone()
    q._current = q._row_to_item(row)
    s = q.snapshot()
    assert s["current_file"] == "videos/a/b.mkv"
    assert s["current_abs"] == "/mnt/media/videos/a/b.mkv"
    assert s["current_dir"] == "/mnt/media"
    q._current = None
    s = q.snapshot()
    assert s["current_abs"] is None and s["current_dir"] is None


def test_the_stream_payload_passes_the_identity_through():
    source = (REPO / "whisper_tm" / "api" / "queue.py").read_text(encoding="utf-8")
    assert '"current_abs": snap.get("current_abs")' in source
    assert '"current_dir": snap.get("current_dir")' in source


def test_list_entries_carry_abs_path():
    from whisper_tm.scanner import db_row_to_entry

    r = {
        "abs_path": "/mnt/media/videos/a/b.mkv",
        "rel_path": "videos/a/b.mkv",
        "display_name": "b.mkv",
        "file_size": 10,
        "modified_at": 1.0,
        "extension": ".mkv",
        "is_video": 1,
        "status": "pending",
    }
    e = db_row_to_entry(r, "/mnt/media/videos")
    assert e["name"] == "a/b.mkv" and e["abs_path"] == "/mnt/media/videos/a/b.mkv"


def test_is_running_file_compares_via_abs_path():
    """state.js in node sandbox: name different, abs_path same → running."""
    import shutil
    import subprocess

    if shutil.which("node") is None:
        pytest.skip("node missing")
    code = f"""
        const vm = require('vm'); const fs = require('fs');
        const sb = {{console, document: {{getElementById: () => null, addEventListener() {{}}}}, window: {{addEventListener() {{}}}}, localStorage: {{getItem: () => null, setItem() {{}}}}}};
        sb.globalThis = sb; vm.createContext(sb);
        vm.runInContext(fs.readFileSync({str(JS / "state.js")!r}, 'utf8'), sb);
        const r = vm.runInContext(`
            queueCurrentFile = 'videos/a/b.mkv'; queueCurrentAbs = '/mnt/media/videos/a/b.mkv';
            [isRunningFile({{name: 'b.mkv', abs_path: '/mnt/media/videos/a/b.mkv'}}),
             isRunningFile({{name: 'videos/a/b.mkv', abs_path: '/mnt/media/videos/a/b.mkv'}}),
             isRunningFile({{name: 'videos/a/b.mkv', abs_path: '/other/videos/a/b.mkv'}}),
             isRunningFile({{name: 'videos/a/b.mkv'}}),
             isRunningFile({{name: 'b.mkv'}}),
             isRunningFile(null)]
        `, sb);
        console.log(JSON.stringify(r));
    """
    done = subprocess.run(["node", "-e", code], capture_output=True, text=True, timeout=30)
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout.strip().splitlines()[-1]) == [
        True,
        True,
        False,
        True,
        False,
        False,
    ]


def test_the_comparison_sites_use_the_identity():
    """No more `=== queueCurrentFile` name comparison at the places that
    control overlay, live preview, and highlighting."""
    fv = (JS / "fileview.js").read_text(encoding="utf-8")
    assert "f.name === queueCurrentFile" not in fv and "f.name !== queueCurrentFile" not in fv
    rec = (JS / "recording.js").read_text(encoding="utf-8")
    # overlay on/off in the file view, the running state's note (recording.js)
    assert fv.count("isRunningFile(f)") + rec.count("isRunningFile(f)") >= 3
    fl = (JS / "filelist.js").read_text(encoding="utf-8")
    assert "f.name === queueCurrentFile" not in fl
    assert "openQueueFile(queueCurrentFile, queueCurrentDir || '')" in fl
    b = (JS / "batch.js").read_text(encoding="utf-8")
    assert "activeFile === d.current_file" not in b
    # Queue lanes: the live stream follows the watched running entry (the
    # open file's, else the oldest) — with its identity, as before.
    assert "connectJobSSEForLive(jobId, w.filename, w.abs)" in b
    assert "runningItemMatches(r, activeFileObj)" in b
    assert "watches()" in b and "queueCurrentAbs = d.current_abs || null" in b


def test_waiting_entries_carry_their_identity(db):
    """The waiting and failed entries the browser marks rows with: an upload
    waits as "uploads/x.m4a" under the media root, while the upload dialog
    opens the uploads folder, where the row is "x.m4a" — it showed "+Q"."""
    with db.connection() as conn:
        conn.execute(
            "INSERT INTO queue_items (filename, base_dir, settings_json, status, position)"
            " VALUES ('uploads/x.m4a', '/mnt/media', '{}', 'pending', 1)"
        )
    assert q.list_items("pending")[0]["abs"] == "/mnt/media/uploads/x.m4a"


def test_queued_rows_compare_via_abs_path():
    """state.js in node: queued and failed marks follow the identity."""
    import shutil
    import subprocess

    if shutil.which("node") is None:
        pytest.skip("node missing")
    code = f"""
        const vm = require('vm'); const fs = require('fs');
        const sb = {{console, document: {{getElementById: () => null, addEventListener() {{}}}}, window: {{addEventListener() {{}}}}, localStorage: {{getItem: () => null, setItem() {{}}}}}};
        sb.globalThis = sb; vm.createContext(sb);
        vm.runInContext(fs.readFileSync({str(JS / "state.js")!r}, 'utf8'), sb);
        const r = vm.runInContext(`
            queueItems = [{{filename: 'uploads/x.m4a', abs: '/m/uploads/x.m4a'}}, {{filename: 'old.mp4'}}];
            setQueueOutcomes([{{id: 7, status: 'error', filename: 'uploads/y.m4a', abs: '/m/uploads/y.m4a'}}]);
            const keys = queuedKeys();
            [isQueuedFile({{name: 'x.m4a', abs_path: '/m/uploads/x.m4a'}}),
             queuedIn(keys, {{name: 'x.m4a', abs_path: '/m/uploads/x.m4a'}}),
             queuedIn(keys, {{name: 'uploads/x.m4a', abs_path: '/other/uploads/x.m4a'}}),
             isQueuedFile({{name: 'old.mp4', abs_path: '/m/old.mp4'}}),
             queuedIn(keys, {{name: 'old.mp4', abs_path: '/m/old.mp4'}}),
             (queueOutcomeOf({{name: 'y.m4a', abs_path: '/m/uploads/y.m4a'}}) || {{}}).id,
             queueOutcomeOf({{name: 'uploads/y.m4a', abs_path: '/n/uploads/y.m4a'}})]
        `, sb);
        console.log(JSON.stringify(r));
    """
    done = subprocess.run(["node", "-e", code], capture_output=True, text=True, timeout=30)
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout.strip().splitlines()[-1]) == [
        True,
        True,
        False,
        True,
        True,
        7,
        None,
    ]
