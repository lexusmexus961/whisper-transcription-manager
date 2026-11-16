"""No full copy of the database on every server start.

``create_app()`` used to back up on EVERY start, so a few restarts a day
filled the backup folder with full copies of a large database — tens of
gigabytes on the system disk. A backup on every server start is not needed,
just the rotation configured in the settings. And a second copy next to the
database on THE SAME disk is the weakest form of backup that exists.

So: one target (``database.backup_path``, e.g. a network share) and two triggers —
every N transcript changes as before, and on start at most every N
hours (``database.backup_min_hours``, default 24).
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import pytest

from whisper_tm.database import Database


@pytest.fixture()
def db(tmp_path, monkeypatch):
    d = Database(str(tmp_path / "library.db"))
    d.initialize()
    # Something to lose: an empty library is not backed up
    with d.connection() as conn:
        conn.execute("INSERT INTO media_files (rel_path, abs_path) VALUES ('a.mp4', '/m/a.mp4')")
    return d


def test_an_empty_library_is_not_backed_up(tmp_path):
    d = Database(str(tmp_path / "empty.db"))
    d.initialize()
    assert d.backup_due() is None


def _provide_config(monkeypatch, **value_list):
    """Replace backup config access with fixed values."""
    from whisper_tm import config as cfgmod

    real = cfgmod.config.get

    def faked(*parts, default=None):
        if len(parts) == 2 and parts[0] == "database" and parts[1] in value_list:
            return value_list[parts[1]]
        return real(*parts, default=default)

    monkeypatch.setattr(cfgmod.config, "get", faked)


class TestTarget:
    def test_configured_path_wins(self, db, tmp_path, monkeypatch):
        nas = tmp_path / "nas"
        _provide_config(monkeypatch, backup_path=str(nas))
        assert db.backup_target() == nas

    def test_without_path_stays_next_to_database(self, db, tmp_path, monkeypatch):
        """Better a weak backup than none at all."""
        _provide_config(monkeypatch, backup_path="")
        assert db.backup_target() == Path(db._db_path).parent / "backups"

    def test_whitespace_counts_as_not_set(self, db, monkeypatch):
        _provide_config(monkeypatch, backup_path="   ")
        assert db.backup_target() == Path(db._db_path).parent / "backups"


class TestDue:
    def test_without_existing_backup_creates_one(self, db, tmp_path, monkeypatch):
        nas = tmp_path / "nas"
        _provide_config(monkeypatch, backup_path=str(nas), backup_min_hours=24, max_backups=5)
        p = db.backup_due()
        assert p is not None and p.exists()

    def test_fresh_backup_prevents_second(self, db, tmp_path, monkeypatch):
        """The core: two server starts soon after each other cost space once,
        not twice. That's exactly what filled the disk before."""
        nas = tmp_path / "nas"
        _provide_config(monkeypatch, backup_path=str(nas), backup_min_hours=24, max_backups=5)
        assert db.backup_due() is not None
        assert db.backup_due() is None, "second start backed up again"
        assert len(list(nas.glob("library_*.db"))) == 1

    def test_old_backup_triggers_again(self, db, tmp_path, monkeypatch):
        nas = tmp_path / "nas"
        _provide_config(monkeypatch, backup_path=str(nas), backup_min_hours=24, max_backups=5)
        first_one = db.backup_due()
        # Back-date by 25 hours.
        old_value = time.time() - 25 * 3600
        os.utime(first_one, (old_value, old_value))
        assert db.backup_due() is not None
        assert len(list(nas.glob("library_*.db"))) == 2

    def test_zero_hours_is_old_behavior(self, db, tmp_path, monkeypatch):
        """Those who really want it on every start can have it — it's just
        no longer the default."""
        nas = tmp_path / "nas"
        _provide_config(monkeypatch, backup_path=str(nas), backup_min_hours=0, max_backups=5)
        assert db.backup_due() is not None
        assert db.backup_due() is not None

    def test_nonsense_value_falls_back_to_default(self, db, tmp_path, monkeypatch):
        """A "24h" in the field must not cause backing up on every start again
        — the default is the safe side."""
        nas = tmp_path / "nas"
        _provide_config(
            monkeypatch, backup_path=str(nas), backup_min_hours="twenty-four", max_backups=5
        )
        assert db.backup_due() is not None
        assert db.backup_due() is None

    def test_backup_lands_at_configured_target(self, db, tmp_path, monkeypatch):
        """No longer beside: local storage is exactly what filled the disk."""
        nas = tmp_path / "nas"
        _provide_config(monkeypatch, backup_path=str(nas), backup_min_hours=24, max_backups=5)
        db.backup_due()
        local = Path(db._db_path).parent / "backups"
        assert list(nas.glob("library_*.db"))
        assert not local.exists() or not list(local.glob("library_*.db"))


class TestLeftovers:
    def test_the_staging_copy_of_an_interrupted_backup_goes(self, db):
        """A start killed during its backup left a copy as big as the
        database next to it — for good."""
        folder = Path(db._db_path).parent
        stale = folder / ".library_20261012_202946_277290.tmp"
        stale.write_bytes(b"x" * 1000)
        journal = folder / ".library_20261012_202946_277290.tmp-journal"
        journal.write_bytes(b"j")
        keep = [
            folder / ".library_notes.tmp",  # not a backup's name
            folder / "library_20261012_202946_277290.db",  # a finished backup
            folder / ".other_20261012_202946_277290.tmp",
        ]  # another database's
        for k in keep:
            k.write_bytes(b"k")
        assert db.remove_backup_leftovers() == 2
        assert not stale.exists() and not journal.exists()
        assert all(k.exists() for k in keep)

    def test_a_backup_leaves_none(self, db, tmp_path):
        assert db.backup(tmp_path / "nas") is not None
        assert db.remove_backup_leftovers() == 0


class TestInTheBackground:
    def test_the_start_does_not_wait_for_it(self):
        """A large database takes minutes to copy — a start that seemed to
        hang got killed mid-copy."""
        source = (Path(__file__).resolve().parent.parent / "whisper_tm" / "app.py").read_text(
            encoding="utf-8"
        )
        assert "threading.Thread(target=_backup_on_start" in source
        assert "ext.db.remove_backup_leftovers()" in source

    def test_one_backup_at_a_time(self, db, monkeypatch):
        """Each stages a copy as big as the database: the start's backup and
        "Back up now" side by side needed twice the space."""
        running, most = [0], [0]

        def slow(dest_dir):
            running[0] += 1
            most[0] = max(most[0], running[0])
            time.sleep(0.05)
            running[0] -= 1

        monkeypatch.setattr(db, "_backup", slow)
        threads = [threading.Thread(target=db.backup) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert most[0] == 1

    def test_a_failure_is_said_not_raised(self, monkeypatch, caplog):
        import whisper_tm.extensions as ext
        from whisper_tm import app

        class Broken:
            def backup_due(self):
                raise RuntimeError("disk on fire")

        monkeypatch.setattr(ext, "db", Broken())
        app._backup_on_start()
        assert "disk on fire" in caplog.text


class TestWiring:
    def test_start_calls_due_check(self):
        """No more `backup()` — otherwise the whole check would be bypassed."""
        source = (Path(__file__).resolve().parent.parent / "whisper_tm" / "app.py").read_text(
            encoding="utf-8"
        )
        assert "ext.db.backup_due()" in source
        assert "ext.db.backup()" not in source

    def test_setting_in_ui(self):
        """Project rule: no setting without UI (read, write, act)."""
        repo = Path(__file__).resolve().parent.parent
        js = "".join(
            (repo / "whisper_tm" / "static" / "js" / name).read_text(encoding="utf-8")
            for name in ("settings.js", "settings_pages.js", "settings_tools.js")
        )
        assert "database.backup_min_hours" in js
