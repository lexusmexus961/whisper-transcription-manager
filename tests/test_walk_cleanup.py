"""
Walk cleanup: rows of disappeared files disappear from DB + queue.

Previously, media_files rows of deleted/renamed files would remain
marked as "open" forever — each "queue all open" batch would pull them back
into the queue, where they failed as "not found" (e.g. thousands of dead
paths after a network share was renamed).

Cleanup only runs after a COMPLETE walk and has three safeguards:
existence check (ignore patterns/permissions), quantity protection (half-dead
mount) and transcript protection (CASCADE would tear transcripts).
"""

import pytest

from whisper_tm.database import Database


@pytest.fixture
def db(tmp_path):
    d = Database(tmp_path / "test.db")
    d.initialize()
    return d


def _create(db, root, rel, status="pending"):
    db.bulk_upsert_media(
        [
            {
                "rel_path": rel,
                "abs_path": str(root / rel),
                "status": status,
            }
        ]
    )
    return str(root / rel)


def test_a_vanished_file_is_removed(db, tmp_path):
    root = tmp_path / "media"
    root.mkdir()
    (root / "stays.mp4").write_bytes(b"x")
    present = _create(db, root, "stays.mp4")
    gone = _create(db, root, "deleted.mp4")  # file does not exist

    deleted, orphaned = db.clean_orphaned_media(root, {present})
    assert deleted == [gone] and orphaned == 0
    rows = {r["rel_path"] for r in db.get_media_files_under(str(root))}
    assert rows == {"stays.mp4"}


def test_the_existence_check_protects_skipped_files(db, tmp_path):
    """Walk didn't see the file (ignore/permissions) — but it exists."""
    root = tmp_path / "media"
    (root / "sub").mkdir(parents=True)
    (root / "sub" / "skipped.mp4").write_bytes(b"x")
    _create(db, root, "sub/skipped.mp4")

    # Walk saw NOTHING of it — existence check must save it
    deleted, orphaned = db.clean_orphaned_media(root, {"/something/else"})
    assert deleted == [] and orphaned == 0


def test_volume_guard_on_half_dead_mount(db, tmp_path):
    """If 'everything' disappears, nothing is deleted (mount failure pattern)."""
    root = tmp_path / "media"
    root.mkdir()
    path_list = [_create(db, root, f"gone_{i:03d}.mp4") for i in range(60)]
    (root / "only.mp4").write_bytes(b"x")
    present = _create(db, root, "only.mp4")

    deleted, orphaned = db.clean_orphaned_media(root, {present})
    assert deleted == [] and orphaned == 0  # 60/61 gone → refused
    assert len(db.get_media_files_under(str(root))) == 61
    assert path_list  # (for readability only: the 60 are still there)


def test_rows_with_a_transcript_stay(db, tmp_path):
    """CASCADE protection: file gone, but transcript present → row stays."""
    root = tmp_path / "media"
    root.mkdir()
    (root / "anchor.mp4").write_bytes(b"x")
    anchor = _create(db, root, "anchor.mp4")
    gone_with_tx = _create(db, root, "gone_with_transcript.mp4", status="done")
    db.upsert_transcript(
        "gone_with_transcript.mp4",
        {
            "complete": True,
            "segments": [
                {
                    "speaker": "S",
                    "start": 0,
                    "end": 1,
                    "start_fmt": "0:00",
                    "end_fmt": "0:01",
                    "text": "hello",
                }
            ],
        },
        media_dir=str(root),
    )

    deleted, orphaned = db.clean_orphaned_media(root, {anchor})
    assert deleted == []  # only the protected one was gone
    assert orphaned == 1  # … and is now 'orphan'
    rows = {r["rel_path"]: r["status"] for r in db.get_media_files_under(str(root))}
    assert rows["gone_with_transcript.mp4"] == "orphan"
    # second run doesn't mark again (idempotent)
    _, again = db.clean_orphaned_media(root, {anchor})
    assert again == 0
    assert db.get_transcript("gone_with_transcript.mp4", media_dir=str(root)) is not None
    assert gone_with_tx  # row persists (transcript anchor)


def test_an_empty_review_deletes_nothing(db, tmp_path):
    """Without seen files (broken walk), nothing is ever deleted."""
    root = tmp_path / "media"
    root.mkdir()
    _create(db, root, "x.mp4")
    assert db.clean_orphaned_media(root, set()) == ([], 0)
