"""
Tests for tools/migrate_paths.py.

The tool rewrites an existing database and cannot be undone. It was outside
version control and untested for a long time; both are fixed now.
"""

import importlib.util
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from whisper_tm.database import Database

_REPO = Path(__file__).resolve().parent.parent
_TOOL = _REPO / "tools" / "migrate_paths.py"


def _load():
    spec = importlib.util.spec_from_file_location("migrate_paths", _TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tx(filename):
    return {
        "filename": filename,
        "language": "de",
        "model": "turbo",
        "vocal_isolation": False,
        "vocal_model": None,
        "segments": [
            {
                "speaker": "S0",
                "start": 0.0,
                "end": 1.0,
                "start_fmt": "0:00",
                "end_fmt": "0:01",
                "text": "hello",
            }
        ],
        "speakers": ["S0"],
        "total_duration": "0:01",
        "total_duration_sec": 1.0,
        "processing_time": "1s",
        "processing_secs": 1.0,
        "total_chunks": 1,
        "complete": True,
        "warning_count": 0,
        "warnings": [],
    }


@pytest.fixture
def db_file(tmp_path):
    d = Database(tmp_path / "m.db")
    d.initialize()
    for i in range(5):
        d.upsert_transcript(f"series/f{i}.mp4", _tx(f"series/f{i}.mp4"), media_dir="/old/root")
    d.close_all()
    return tmp_path / "m.db"


class TestNorm:
    """norm() has gone through core/paths since the rework (docs/PATHS.md)."""

    def test_relative_to_the_root(self):
        m = _load()
        assert m.norm("/nas/video", "/nas/video", "a.mp4") == "a.mp4"

    def test_a_different_root_is_recomputed(self):
        m = _load()
        assert m.norm("/nas/video", "/nas/video/series", "f1.mp4") == "series/f1.mp4"

    def test_outside_returns_none(self):
        m = _load()
        assert m.norm("/nas/video", "/somewhere/else", "a.mp4") is None

    def test_a_neighbouring_folder_is_not_below_it(self):
        """The old check via startswith classified /nas/videos as being
        underneath /nas/video."""
        m = _load()
        assert m.norm("/nas/video", "/nas/videos", "a.mp4") is None

    def test_the_result_uses_forward_slashes(self):
        m = _load()
        assert m.norm("C:/video", "C:/video/series", "f1.mp4") == "series/f1.mp4"


class TestDryRun:
    def test_without_apply_nothing_is_written(self, db_file):
        before = db_file.read_bytes()
        r = subprocess.run(
            [sys.executable, str(_TOOL), "--db", str(db_file), "--root", "/old/root"],
            capture_output=True,
            text=True,
            cwd=_REPO,
        )
        assert r.returncode == 0, r.stdout
        assert "Dry run" in r.stdout
        assert db_file.read_bytes() == before

    def test_a_foreign_root_is_refused(self, db_file):
        """Safety brake: if the given root matches almost no row, it's
        probably the wrong one — better to abort than to rewrite half
        the database."""
        r = subprocess.run(
            [sys.executable, str(_TOOL), "--db", str(db_file), "--root", "/entirely/other/root"],
            capture_output=True,
            text=True,
            cwd=_REPO,
        )
        assert r.returncode != 0
        assert "does not match" in r.stdout

    def test_a_missing_db_reports_cleanly(self, tmp_path):
        r = subprocess.run(
            [sys.executable, str(_TOOL), "--db", str(tmp_path / "gone.db"), "--root", "/nas"],
            capture_output=True,
            text=True,
            cwd=_REPO,
        )
        assert r.returncode != 0
        assert "does not exist" in r.stdout


class TestApply:
    def test_the_marker_is_written(self, db_file):
        """Without a marker there was no way to tell whether the tool had ever run."""
        r = subprocess.run(
            [sys.executable, str(_TOOL), "--db", str(db_file), "--root", "/old/root", "--apply"],
            capture_output=True,
            text=True,
            cwd=_REPO,
        )
        assert r.returncode == 0, r.stderr
        conn = sqlite3.connect(db_file)
        value = conn.execute("SELECT value FROM schema_info WHERE key='migrate_paths'").fetchone()
        conn.close()
        assert value is not None
        import json

        data = json.loads(value[0])
        assert data["root"] == "/old/root"
        assert "timestamp" in data and "report" in data

    def test_the_database_stays_consistent(self, db_file):
        subprocess.run(
            [sys.executable, str(_TOOL), "--db", str(db_file), "--root", "/old/root", "--apply"],
            capture_output=True,
            text=True,
            cwd=_REPO,
            check=True,
        )
        conn = sqlite3.connect(db_file)
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
        assert conn.execute("SELECT COUNT(*) FROM transcripts").fetchone()[0] == 5
        conn.close()

    def test_a_second_run_changes_nothing(self, db_file):
        """Idempotency: a repeat run must not rewrite again."""
        for _ in range(2):
            r = subprocess.run(
                [
                    sys.executable,
                    str(_TOOL),
                    "--db",
                    str(db_file),
                    "--root",
                    "/old/root",
                    "--apply",
                ],
                capture_output=True,
                text=True,
                cwd=_REPO,
            )
            assert r.returncode == 0, r.stderr
        conn = sqlite3.connect(db_file)
        assert conn.execute("SELECT COUNT(*) FROM transcripts").fetchone()[0] == 5
        conn.close()

    def test_a_backup_is_created(self, db_file):
        subprocess.run(
            [sys.executable, str(_TOOL), "--db", str(db_file), "--root", "/old/root", "--apply"],
            capture_output=True,
            text=True,
            cwd=_REPO,
            check=True,
        )
        copies = list(db_file.parent.glob("*.db*"))
        assert len(copies) > 1, [p.name for p in copies]
