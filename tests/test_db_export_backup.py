"""
Downloading the database and backing it up by hand.

Export: the database runs in WAL mode, and ``GET /api/db/export`` sent the
live file as it lay on disk. The latest commits can still sit in the -wal
file — they were not in the download at all — and a write during the
download mixed pages from before and after it. Now a snapshot is written
with the SQLite backup API, sent, and deleted once the response is closed.

Backup: ``POST /api/db/backup`` without a body wrote a full copy next to the
database every time (several GB each on a large library), although the configured backup folder is
where backups belong (``Database.backup_target``). From the settings page it
never even got that far: the button posts no body, and ``request.json``
answers that with 415.
"""

from __future__ import annotations

import os
import shutil
import sqlite3
import tempfile
import time
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace

import pytest

from whisper_tm.config import config


@pytest.fixture
def app(tmp_path, monkeypatch):
    """An app on a throwaway config and database; the queue stays paused."""
    monkeypatch.setenv("HOME", str(tmp_path))
    own = tmp_path / "config.yaml"
    own.write_text("queue:\n  autostart: false\n  lanes: 1\n", encoding="utf-8")
    monkeypatch.setenv("WHISPER_TM_CONFIG", str(own))
    (tmp_path / "data").mkdir()
    config.load(None)
    config._data.setdefault("directories", {})["media"] = str(tmp_path / "media")
    config._data["directories"]["work"] = str(tmp_path / "work")
    config._data.setdefault("database", {})["path"] = str(tmp_path / "data" / "library.db")
    config._data["database"]["backup_path"] = ""
    # No startup backup: every backup in these tests is one a test asked for
    config._data["database"]["auto_backup"] = False
    config._resolve_directories()
    config._ensure_directories()
    # The export's second choice is the system temp folder, and it looks for
    # leftovers there: a folder of the test's own, never the real /tmp.
    system_tmp = tmp_path / "system-tmp"
    system_tmp.mkdir()
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(system_tmp))

    from whisper_tm import extensions as ext
    from whisper_tm.api import settings as settings_api
    from whisper_tm.app import create_app

    client = create_app().test_client()
    # settings.py binds `db` at import time — without this, the database of
    # an earlier test's app would answer.
    monkeypatch.setattr(settings_api, "db", ext.db)
    yield SimpleNamespace(
        client=client, db=ext.db, tmp=tmp_path, system_tmp=system_tmp, db_file=Path(ext.db._db_path)
    )
    monkeypatch.undo()
    config.load(None)


def _setting(path: Path, key: str):
    with closing(sqlite3.connect(path)) as conn:
        row = conn.execute("SELECT value FROM user_settings WHERE key = ?", (key,)).fetchone()
    return row[0] if row else None


def _snapshots(folder: Path, db_file: Path) -> list[Path]:
    return [p for p in folder.iterdir() if p.name.startswith(f".{db_file.name}.export-")]


def _backups(folder: Path, db_file: Path) -> list[Path]:
    return sorted(folder.glob(f"{db_file.stem}_*.db")) if folder.exists() else []


def _snapshot_folders(monkeypatch) -> list[Path]:
    """Where the export creates its snapshots. Looking into the folders
    afterwards can't tell: the name is gone as soon as the file is open."""
    used: list[Path] = []
    real = tempfile.mkstemp

    def mkstemp(*args, **kwargs):
        fd, name = real(*args, **kwargs)
        used.append(Path(name).parent)
        return fd, name

    monkeypatch.setattr(tempfile, "mkstemp", mkstemp)
    return used


# ── Export ────────────────────────────────────────────────────────────────


def test_the_download_holds_what_was_committed_just_before(app):
    with app.db.connection() as conn:
        conn.execute("PRAGMA wal_autocheckpoint=0")
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    app.db.set_setting("export_probe", "committed just now")
    # The row is only in the -wal file: the live file as it lies — what the
    # export used to send — does not have it.
    as_it_lies = app.tmp / "as-it-lies.db"
    shutil.copyfile(app.db_file, as_it_lies)
    assert _setting(as_it_lies, "export_probe") is None

    resp = app.client.get("/api/db/export")
    try:
        assert resp.status_code == 200, resp.get_data(as_text=True)
        assert "whisper-tm-backup.db" in resp.headers["Content-Disposition"]
        body = resp.get_data()
    finally:
        resp.close()
    download = app.tmp / "download.db"
    download.write_bytes(body)
    assert _setting(download, "export_probe") == "committed just now"
    with closing(sqlite3.connect(download)) as conn:
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"


def test_nothing_of_the_snapshot_stays_behind(app, monkeypatch):
    used = _snapshot_folders(monkeypatch)
    resp = app.client.get("/api/db/export")
    assert resp.status_code == 200
    assert used == [app.db_file.parent], "not written next to the database"
    # The name is gone while the download runs — a server dying now leaves
    # nothing behind — and the open file still delivers every byte.
    assert not _snapshots(app.db_file.parent, app.db_file)
    body = resp.get_data()
    resp.close()
    assert body.startswith(b"SQLite format 3\x00")
    assert int(resp.headers["Content-Length"]) == len(body)
    assert not _snapshots(app.db_file.parent, app.db_file)


def test_where_an_open_file_cannot_be_deleted_the_close_removes_it(app, monkeypatch):
    """Windows refuses to delete an open file. Then the response's close has
    to do it — also for a download the browser aborted. send_file's
    direct_passthrough used to keep that close from ever being called."""
    real = Path.unlink
    refused: list[Path] = []

    def unlink(self, missing_ok=False):
        if self.name.startswith(f".{app.db_file.name}.export-") and not refused:
            refused.append(self)
            raise PermissionError(13, "The file is being used by another process", str(self))
        return real(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", unlink)
    resp = app.client.get("/api/db/export")
    assert resp.status_code == 200
    assert refused, "the export no longer removes the name early — adjust this test"
    assert len(_snapshots(app.db_file.parent, app.db_file)) == 1
    resp.close()  # aborted: nothing was read
    assert not _snapshots(app.db_file.parent, app.db_file)


def test_leftovers_of_a_dead_export_go_on_the_next_call(app):
    """A server killed during a download never runs the close callback."""
    prefix = f".{app.db_file.name}.export-"
    dead = [
        app.db_file.parent / f"{prefix}dead.tmp",
        app.db_file.parent / f"{prefix}dead.tmp-journal",
        app.system_tmp / f"{prefix}dead-too.tmp",
    ]
    an_hour_ago = time.time() - 3600
    for p in dead:
        p.write_bytes(b"half a snapshot")
        os.utime(p, (an_hour_ago, an_hour_ago))
    in_progress = app.db_file.parent / f"{prefix}in-progress.tmp"
    in_progress.write_bytes(b"another request is writing this one")

    app.client.get("/api/db/export").close()
    assert not [p for p in dead if p.exists()]
    assert in_progress.exists(), "a snapshot another request still writes was deleted"


def test_a_full_database_disk_uses_the_temp_folder(app, monkeypatch):
    used = _snapshot_folders(monkeypatch)
    real = shutil.disk_usage

    def usage(path):
        u = real(path)
        return u._replace(free=0) if Path(path) == app.db_file.parent else u

    monkeypatch.setattr(shutil, "disk_usage", usage)
    resp = app.client.get("/api/db/export")
    assert resp.status_code == 200, resp.get_data(as_text=True)
    assert used == [app.system_tmp]
    assert resp.get_data().startswith(b"SQLite format 3\x00")
    resp.close()
    assert not _snapshots(app.system_tmp, app.db_file)
    assert not _snapshots(app.db_file.parent, app.db_file)


def test_without_space_anywhere_nothing_is_written(app, monkeypatch):
    used = _snapshot_folders(monkeypatch)
    real = shutil.disk_usage
    monkeypatch.setattr(shutil, "disk_usage", lambda path: real(path)._replace(free=0))
    resp = app.client.get("/api/db/export")
    assert resp.status_code == 507
    assert "free space" in resp.get_json()["error"]
    assert used == []
    assert not _snapshots(app.db_file.parent, app.db_file)
    assert not _snapshots(app.system_tmp, app.db_file)


# ── Manual backup ─────────────────────────────────────────────────────────


def test_without_a_body_the_backup_goes_to_the_configured_folder(app):
    nas = app.tmp / "nas"
    config._data["database"]["backup_path"] = str(nas)
    app.db.set_setting(app.db._BACKUP_COUNTER_KEY, "7")

    r = app.client.post("/api/db/backup")  # what the button sends: no body
    assert r.status_code == 200, r.get_data(as_text=True)
    d = r.get_json()
    assert Path(d["path"]).parent == nas
    assert d["backups"] == 1
    assert not _backups(app.db_file.parent / "backups", app.db_file), (
        "a full copy landed next to the database again"
    )
    # It went where the automatic backup goes, so it counts as that checkpoint
    assert app.db.get_backup_counter() == 0


def test_without_a_configured_folder_it_stays_next_to_the_database(app):
    """Better a weak backup than none — Database.backup_target's fallback."""
    local = app.db_file.parent / "backups"
    assert app.client.post("/api/db/backup").status_code == 200
    r = app.client.post("/api/db/backup")
    assert r.status_code == 200
    assert Path(r.get_json()["path"]).parent == local
    assert r.get_json()["backups"] == len(_backups(local, app.db_file)) == 2


def test_external_without_a_folder_is_refused(app):
    r = app.client.post("/api/db/backup", json={"target": "external"})
    assert r.status_code == 400


def test_external_goes_to_the_folder_as_before(app):
    nas = app.tmp / "nas"
    config._data["database"]["backup_path"] = str(nas)
    app.db.set_setting(app.db._BACKUP_COUNTER_KEY, "3")
    r = app.client.post("/api/db/backup", json={"target": "external"})
    assert r.status_code == 200
    assert Path(r.get_json()["path"]).parent == nas
    assert r.get_json()["backups"] == 1
    assert app.db.get_backup_counter() == 0


def test_an_unreachable_folder_is_reported_not_swapped_for_a_local_copy(app):
    blocker = app.tmp / "not-a-folder"
    blocker.write_text("a file where the NAS folder should be", encoding="utf-8")
    config._data["database"]["backup_path"] = str(blocker / "backups")
    r = app.client.post("/api/db/backup")
    assert r.status_code == 500
    assert "unreachable" in r.get_json()["error"]
    assert not _backups(app.db_file.parent / "backups", app.db_file)


class TestDeleteBacksUpFirst:
    """'Delete the database' made no copy — in "Database only" mode that was
    the only copy of every transcript."""

    def _mark(self, app):
        with app.db.connection() as conn:
            conn.execute("INSERT OR REPLACE INTO user_settings (key, value) VALUES ('m', 'kept')")

    def test_a_backup_comes_first(self, app):
        self._mark(app)
        d = app.client.post("/api/db/delete").get_json()
        assert d["ok"] and d["backup"]
        assert _setting(Path(d["backup"]), "m") == "kept"  # the copy has it
        with app.db.connection() as conn:
            assert (
                conn.execute("SELECT COUNT(*) FROM user_settings WHERE key='m'").fetchone()[0] == 0
            )

    def test_refused_while_a_job_runs(self, app, monkeypatch):
        from whisper_tm.core import queue as tx
        from whisper_tm.core.llm_queue import llm_queue

        self._mark(app)
        monkeypatch.setattr(tx, "busy", lambda: True)
        assert app.client.post("/api/db/delete").status_code == 409
        monkeypatch.setattr(tx, "busy", lambda: False)
        monkeypatch.setattr(llm_queue, "_current", {"id": 1})
        assert app.client.post("/api/db/delete").status_code == 409
        with app.db.connection() as conn:
            assert (
                conn.execute("SELECT COUNT(*) FROM user_settings WHERE key='m'").fetchone()[0] == 1
            )

    def test_the_llm_worker_is_held_and_its_pause_restored(self):
        from whisper_tm.core.llm_queue import llm_queue

        was = llm_queue._paused
        with llm_queue.held():
            assert llm_queue._paused and llm_queue._take_lock.locked()
        assert llm_queue._paused == was and not llm_queue._take_lock.locked()

    def test_without_a_backup_nothing_is_deleted(self, app, monkeypatch):
        self._mark(app)
        monkeypatch.setattr(app.db, "backup", lambda dest_dir=None: None)
        r = app.client.post("/api/db/delete")
        assert r.status_code == 500 and "nothing was deleted" in r.get_json()["error"]
        with app.db.connection() as conn:
            assert (
                conn.execute("SELECT value FROM user_settings WHERE key='m'").fetchone()[0]
                == "kept"
            )


class TestImportReadsEveryTranscriptFolder:
    """'Import transcripts' read only <library>/.transcripts: after losing
    the database, transcripts beside recordings in subfolders stayed out."""

    def test_a_subfolders_transcripts_come_back(self, app):
        import json

        media = app.tmp / "media"
        (media / "sub" / ".transcripts").mkdir(parents=True)
        (media / "sub" / "b.mp3").write_bytes(b"x")
        (media / ".hidden" / ".transcripts").mkdir(parents=True)
        doc = {
            "filename": "sub/b.mp3",
            "language": "en",
            "model": "turbo",
            "complete": True,
            "segments": [{"speaker": "S0", "start": 0.0, "end": 1.0, "text": "hello there"}],
        }
        (media / "sub" / ".transcripts" / "b.mp3.json").write_text(
            json.dumps(doc), encoding="utf-8"
        )
        (media / ".hidden" / ".transcripts" / "c.mp3.json").write_text(
            json.dumps(doc), encoding="utf-8"
        )
        d = app.client.post("/api/db/sync").get_json()
        assert d["ok"] and d["imported"] == 1, d  # not the hidden one
        with app.db.connection() as conn:
            rows = [r[0] for r in conn.execute("SELECT abs_path FROM transcripts")]
        assert rows == [str(media / "sub" / "b.mp3")]
