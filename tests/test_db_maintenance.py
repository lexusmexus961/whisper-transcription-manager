"""
Database maintenance that must not hold the write lock for long:
the foreign-key index behind ON DELETE CASCADE, the JSON sync without a
full-text rebuild, and the statistics refresh.
"""

import json

import pytest

from whisper_tm.database import Database


@pytest.fixture
def db(tmp_path):
    d = Database(tmp_path / "test.db")
    d.initialize()
    return d


def _tx(name: str) -> dict:
    return {
        "filename": name,
        "complete": True,
        "language": "en",
        "segments": [{"speaker": "S0", "start": 0.0, "end": 1.0, "text": f"hello {name}"}],
    }


class TestMediaIdIndex:
    """Deleting a media_files row cascades into transcripts. Without an
    index on transcripts.media_id SQLite scans the whole transcripts table
    per deleted row (a noticeable delay per row on a large library)."""

    def test_fresh_db_has_media_id_index(self, db):
        with db.connection() as conn:
            plan = conn.execute(
                "EXPLAIN QUERY PLAN SELECT 1 FROM transcripts WHERE media_id = ?", (1,)
            ).fetchall()
        assert any("idx_transcript_media" in str(tuple(r)) for r in plan), plan

    def test_a_missing_index_is_made_at_start(self, tmp_path):
        path = tmp_path / "library.db"
        d = Database(path)
        d.initialize()
        with d.connection() as conn:
            conn.execute("DROP INDEX IF EXISTS idx_transcript_media")
        d.close_all()
        d2 = Database(path)
        d2.initialize()
        with d2.connection() as conn:
            names = {
                r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='index'")
            }
        assert "idx_transcript_media" in names


class TestSyncWithoutFtsRebuild:
    """The FTS triggers keep the index consistent. A full 'rebuild' after
    every sync of more than ten JSONs held the write lock for tens of
    seconds on a large library (millions of segments)."""

    def test_bulk_sync_does_not_rebuild_fts(self, db, tmp_path, monkeypatch):
        tdir = tmp_path / "media" / ".transcripts"
        tdir.mkdir(parents=True)
        for i in range(15):
            (tmp_path / "media" / f"f{i}.mp4").write_bytes(b"x")
            (tdir / f"f{i}.mp4.json").write_text(json.dumps(_tx(f"f{i}.mp4")))
        calls = []
        monkeypatch.setattr(db, "rebuild_fts_index", lambda: calls.append(1))
        res = db.sync_from_json_dir(tdir, media_dir=str(tmp_path / "media"))
        assert res["imported"] == 15
        assert calls == []
        # ...and the search still finds the imported text via the triggers
        out = db.search("f7")
        assert out["total"] == 1


class TestStatisticsRefresh:
    def test_refresh_is_limited_and_scoped(self, db, monkeypatch):
        """A full ANALYZE covers every segment and the FTS tables under
        the write lock; the planner only needs media_files/transcripts."""
        db.bulk_upsert_media(
            [{"rel_path": f"a{i}.mp4", "abs_path": f"/m/a{i}.mp4"} for i in range(3)]
        )
        statements = []
        conn = db._get_connection()
        conn.set_trace_callback(statements.append)
        try:
            db.refresh_statistics(min_growth=1)
        finally:
            conn.set_trace_callback(None)
        analyze = [s for s in statements if s.strip().upper().startswith("ANALYZE")]
        assert analyze, statements
        assert all(s.strip().upper() != "ANALYZE" for s in analyze), analyze
        assert any("analysis_limit" in s.lower() for s in statements)
        with db.connection() as conn:
            assert (
                conn.execute(
                    "SELECT COUNT(*) FROM sqlite_stat1 WHERE tbl = 'media_files'"
                ).fetchone()[0]
                > 0
            )


def test_media_delete_uses_index(db):
    """The cascade lookup itself: deleting a media row must search, not scan."""
    db.upsert_transcript("a.mp4", _tx("a.mp4"), media_dir="/m", abs_path="/m/a.mp4")
    with db.connection() as conn:
        mid = conn.execute("SELECT id FROM media_files WHERE abs_path = '/m/a.mp4'").fetchone()[0]
        plan = conn.execute(
            "EXPLAIN QUERY PLAN SELECT 1 FROM transcripts WHERE media_id = ?", (mid,)
        ).fetchall()
    assert not any("SCAN transcripts" in str(tuple(r)) for r in plan)
