"""The voice database is backed up: names, trash and history can't be
recomputed. A copy a week at start, next to the library's backups, the
newest three kept."""

import os
import sqlite3
import time

from whisper_tm.database.speaker import SpeakerDB


def _db(tmp_path):
    db = SpeakerDB(tmp_path / "speaker.db")
    db.initialize()
    with db.connection() as c:
        c.execute("CREATE TABLE marker (x)")
        c.execute("INSERT INTO marker VALUES (42)")
    return db


def test_a_backup_is_a_whole_copy(tmp_path):
    db = _db(tmp_path)
    out = db.backup(tmp_path / "backups")
    assert out and out.name.startswith("speaker_") and out.suffix == ".db"
    with sqlite3.connect(out) as c:
        assert c.execute("SELECT x FROM marker").fetchone()[0] == 42
    assert not list(tmp_path.glob(".speaker_*.tmp"))


def test_the_newest_three_are_kept(tmp_path):
    db = _db(tmp_path)
    for _ in range(5):
        db.backup(tmp_path / "backups")
    assert len(list((tmp_path / "backups").glob("speaker_*.db"))) == SpeakerDB.BACKUPS_KEPT == 3


def test_a_copy_a_week(tmp_path):
    db = _db(tmp_path)
    first = db.backup_due(tmp_path / "backups")
    assert first is not None
    assert db.backup_due(tmp_path / "backups") is None  # too soon
    old = time.time() - 8 * 86400
    os.utime(first, (old, old))
    assert db.backup_due(tmp_path / "backups") is not None


def test_an_unusable_destination_is_no_error(tmp_path):
    db = _db(tmp_path)
    blocker = tmp_path / "file"
    blocker.write_text("x")
    assert db.backup(blocker / "backups") is None
