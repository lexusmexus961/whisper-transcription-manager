"""
An orphan (transcript present, file gone) whose file comes back must stop
being an orphan on the next walk — and must not be listed twice.
"""

import pytest

from whisper_tm.database import Database


@pytest.fixture
def db(tmp_path):
    d = Database(tmp_path / "test.db")
    d.initialize()
    return d


def _status(db, abs_path):
    with db.connection() as conn:
        return conn.execute(
            "SELECT status FROM media_files WHERE abs_path = ?", (abs_path,)
        ).fetchone()[0]


def _orphan(db, tmp_path):
    root = tmp_path / "media"
    root.mkdir()
    (root / "anchor.mp4").write_bytes(b"x")
    anchor = str(root / "anchor.mp4")
    back = str(root / "back.mp4")
    db.bulk_upsert_media(
        [
            {"rel_path": "anchor.mp4", "abs_path": anchor},
            {"rel_path": "back.mp4", "abs_path": back, "status": "done"},
        ]
    )
    db.upsert_transcript(
        "back.mp4",
        {"complete": True, "segments": [{"speaker": "S0", "start": 0, "end": 1, "text": "hi"}]},
        media_dir=str(root),
        abs_path=back,
    )
    db.clean_orphaned_media(root, {anchor})
    assert _status(db, back) == "orphan"
    return root, back


def test_returning_file_loses_orphan_status(db, tmp_path):
    root, back = _orphan(db, tmp_path)
    (root / "back.mp4").write_bytes(b"x")
    db.bulk_upsert_media([{"rel_path": "back.mp4", "abs_path": back, "status": "done"}])
    assert _status(db, back) == "done"


def test_bulk_upsert_keeps_other_statuses(db, tmp_path):
    """Only 'orphan' is a statement about existence; 'skipped' etc. are the
    user's and stay as they are."""
    db.bulk_upsert_media([{"rel_path": "s.mp4", "abs_path": "/m/s.mp4"}])
    with db.connection() as conn:
        conn.execute("UPDATE media_files SET status='skipped' WHERE abs_path='/m/s.mp4'")
    db.bulk_upsert_media([{"rel_path": "s.mp4", "abs_path": "/m/s.mp4", "status": "pending"}])
    assert _status(db, "/m/s.mp4") == "skipped"


def test_walk_lists_returned_orphan_once(db, tmp_path):
    from whisper_tm.scanner import scan_media_directory

    root, back = _orphan(db, tmp_path)
    (root / "back.mp4").write_bytes(b"x")
    res = scan_media_directory(root, recursive=True, db=db)
    names = [f["name"] for f in res.files]
    assert names.count("back.mp4") == 1, names
    assert res.orphans == 0
