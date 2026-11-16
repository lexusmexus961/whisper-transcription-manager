"""
Tests for the media indexer and index DB methods.
"""

import time

import pytest

from whisper_tm.database import Database
from whisper_tm.scanner.indexer import _RateEstimator


class TestRateEstimator:
    def test_first_update_only_seeds(self):
        est = _RateEstimator()
        est.update(0, 100.0)  # baseline only
        assert est.rate is None
        assert est.eta_sec(50) is None

    def test_seeds_rate_on_first_real_interval(self):
        est = _RateEstimator()
        est.update(0, 0.0)
        est.update(10, 1.0)  # 10 files in 1s → 10/s
        assert est.rate == pytest.approx(10.0)
        # 90 remaining at 10/s → 9s
        assert est.eta_sec(90) == pytest.approx(9.0)

    def test_no_progress_interval_ignored(self):
        est = _RateEstimator()
        est.update(0, 0.0)
        est.update(10, 1.0)
        est.update(10, 2.0)  # no progress (dn=0) → rate stays
        est.update(10, 1.0)  # time backwards (dt<=0) → ignored
        assert est.rate == pytest.approx(10.0)

    def test_ewma_adapts_toward_new_rate(self):
        # From 5/s to 50/s: EWMA must move toward the new rate,
        # without jumping there immediately (smoothing).
        est = _RateEstimator(tau_sec=8.0)
        t = 0.0
        est.update(0, t)
        n = 0
        for _ in range(20):  # warm up at 5/s
            n += 15
            t += 15 / 5.0
            est.update(n, t)
        assert est.rate == pytest.approx(5.0, abs=0.5)
        for _ in range(30):  # switch to 50/s
            n += 15
            t += 15 / 50.0
            est.update(n, t)
        # clearly above old rate and toward 50, but smoothed
        assert 5.0 < est.rate <= 50.0
        assert est.rate > 30.0

    def test_sec_per_file_is_inverse_rate(self):
        est = _RateEstimator()
        est.update(0, 0.0)
        est.update(4, 1.0)  # 4/s → 0.25 s/file
        assert est.sec_per_file() == pytest.approx(0.25)

    def test_eta_guards(self):
        est = _RateEstimator()
        est.update(0, 0.0)
        est.update(10, 1.0)
        assert est.eta_sec(0) is None  # nothing left open
        assert est.eta_sec(-5) is None


@pytest.fixture
def db(tmp_path):
    d = Database(tmp_path / "test.db")
    d.initialize()
    yield d
    d.close_all()


class TestSchemaV2:
    def test_new_columns_exist(self, db):
        with db.connection() as conn:
            cols = {r["name"] for r in conn.execute("PRAGMA table_info(media_files)")}
        for c in (
            "width",
            "height",
            "has_video",
            "audio_codec",
            "video_codec",
            "bitrate",
            "indexed_at",
            "index_mtime",
            "abs_path",
        ):
            assert c in cols, f"column {c} missing"

    def test_version_bumped(self, db):
        from whisper_tm.database import SCHEMA_VERSION

        with db.connection() as conn:
            v = conn.execute("SELECT value FROM schema_info WHERE key='version'").fetchone()
        assert v["value"] == str(SCHEMA_VERSION)


class TestMediaMetadata:
    def test_bulk_set_and_get(self, db):
        db.bulk_set_media_metadata(
            [
                {
                    "rel_path": "a.mp4",
                    "duration_sec": 12.5,
                    "has_audio": True,
                    "has_video": True,
                    "width": 1920,
                    "height": 1080,
                    "audio_codec": "aac",
                    "video_codec": "h264",
                    "bitrate": 4_000_000,
                    "media_title": "Title",
                    "indexed_at": time.time(),
                    "index_mtime": 100.0,
                },
            ]
        )
        meta = db.get_media_meta_batch()
        assert "a.mp4" in meta
        m = meta["a.mp4"]
        assert m["media_duration_sec"] == 12.5
        assert m["media_has_video"] is True
        assert m["media_width"] == 1920
        assert m["media_audio_codec"] == "aac"
        assert m["indexed"] is True

    def test_unindexed_not_in_meta_batch(self, db):
        # A file without indexed_at should NOT show up in the meta batch
        db.upsert_media_file("none.mp4", status="pending")
        assert "none.mp4" not in db.get_media_meta_batch()

    def test_index_state_batch(self, db):
        now = time.time()
        db.bulk_set_media_metadata(
            [
                {"rel_path": "a.mp4", "duration_sec": 1, "indexed_at": now, "index_mtime": 42.0},
            ]
        )
        state = db.get_index_state_batch()
        assert "a.mp4" in state
        indexed_at, index_mtime = state["a.mp4"]
        assert indexed_at > 0 and index_mtime == 42.0

    def test_metadata_survives_scan_upsert(self, db):
        # Index sets metadata; a subsequent scan upsert (bulk_upsert_media)
        # must NOT overwrite duration/indexed_at.
        db.bulk_set_media_metadata(
            [
                {
                    "rel_path": "a.mp4",
                    "duration_sec": 99.0,
                    "indexed_at": 123.0,
                    "index_mtime": 5.0,
                    "width": 640,
                    "height": 480,
                },
            ]
        )
        db.bulk_upsert_media(
            [
                {"rel_path": "a.mp4", "file_size": 1000, "modified_at": 5.0, "status": "done"},
            ]
        )
        meta = db.get_media_meta_batch()["a.mp4"]
        assert meta["media_duration_sec"] == 99.0  # preserved
        assert meta["media_width"] == 640


class TestIndexStats:
    def test_stats_counts(self, db):
        now = time.time()
        db.bulk_set_media_metadata(
            [
                {
                    "rel_path": "a.mp4",
                    "duration_sec": 10,
                    "has_video": True,
                    "indexed_at": now,
                    "index_mtime": 1,
                },
                {
                    "rel_path": "b.mp3",
                    "duration_sec": 20,
                    "has_video": False,
                    "indexed_at": now,
                    "index_mtime": 1,
                },
            ]
        )
        db.upsert_media_file("c.mp4", status="pending")  # not indexed
        stats = db.get_index_stats()
        assert stats["total"] == 3
        assert stats["indexed"] == 2
        assert stats["unindexed"] == 1
        assert stats["video_count"] == 1
        assert stats["total_duration_sec"] == 30

    def test_stats_subfolder_filter(self, db):
        now = time.time()
        db.bulk_set_media_metadata(
            [
                {"rel_path": "sub/a.mp4", "duration_sec": 10, "indexed_at": now, "index_mtime": 1},
                {
                    "rel_path": "other/b.mp4",
                    "duration_sec": 20,
                    "indexed_at": now,
                    "index_mtime": 1,
                },
            ]
        )
        stats = db.get_index_stats(subfolder_prefix="sub")
        assert stats["total"] == 1
        assert stats["total_duration_sec"] == 10


class TestIndexDirectory:
    """The index_directory() generator with a fake file iteration."""

    def test_probe_metadata_shape(self, monkeypatch, db, tmp_path):
        # A real media file via ffmpeg would be expensive — mock get_media_info
        from whisper_tm.scanner import indexer

        # Big enough for the duration plausibility filter:
        # below 256 B per second of claimed duration, the duration counts
        # as broken metadata and is set to 0 — so the 5 s from the fake
        # sample needs at least ~1.3 kB of file.
        (tmp_path / "a.wav").write_bytes(b"x" * 8192)
        (tmp_path / "b.wav").write_bytes(b"x" * 8192)

        def fake_iter(root, **kw):
            return [tmp_path / "a.wav", tmp_path / "b.wav"]

        def fake_probe(fp):
            return {
                "duration": 5.0,
                "has_audio": True,
                "has_video": False,
                "width": 0,
                "height": 0,
                "audio_codec": "pcm",
                "video_codec": "",
                "bitrate": 128000,
                "title": "",
                "ok": True,
            }

        monkeypatch.setattr(indexer, "collect_media_files", fake_iter)
        monkeypatch.setattr(indexer, "get_media_info", fake_probe)

        progs = list(indexer.index_directory(tmp_path, db=db, discover_transcripts=False))
        final = progs[-1]
        assert final["complete"] is True
        assert final["indexed"] == 2
        assert final["failed"] == 0
        # Metadata landed in the DB
        meta = db.get_media_meta_batch()
        assert meta["a.wav"]["media_duration_sec"] == 5.0

    def test_resume_skips_indexed(self, monkeypatch, db, tmp_path):
        from whisper_tm.scanner import indexer

        f = tmp_path / "a.wav"
        f.write_bytes(b"x")

        monkeypatch.setattr(indexer, "collect_media_files", lambda root, **kw: [f])
        monkeypatch.setattr(
            indexer,
            "get_media_info",
            lambda fp: {
                "duration": 1.0,
                "has_audio": True,
                "has_video": False,
                "width": 0,
                "height": 0,
                "audio_codec": "",
                "video_codec": "",
                "bitrate": 0,
                "title": "",
                "ok": True,
            },
        )

        # First run indexes
        list(indexer.index_directory(tmp_path, db=db, discover_transcripts=False))
        # Second run: same mtime → skip
        final = list(indexer.index_directory(tmp_path, db=db, discover_transcripts=False))[-1]
        assert final["indexed"] == 0
        assert final["skipped"] == 1


class TestParentChildCoverage:
    """Regression: indexing the parent recursively → a scan of the
    subfolder must recognize the files as indexed (stable abs_path key).
    Otherwise the "directory not indexed" popup wrongly reappears."""

    def _fake_probe(self):
        return {
            "duration": 1.0,
            "has_audio": True,
            "has_video": False,
            "width": 0,
            "height": 0,
            "audio_codec": "",
            "video_codec": "",
            "bitrate": 0,
            "title": "",
            "ok": True,
        }

    def test_child_scan_sees_parent_index(self, monkeypatch, db, tmp_path):
        from whisper_tm.config import config
        from whisper_tm.scanner import indexer, scan_media_directory

        config.load(None)  # scan_media_directory needs resolved config

        (tmp_path / "a.wav").write_bytes(b"x")
        sub = tmp_path / "interviews"
        sub.mkdir()
        (sub / "b.wav").write_bytes(b"x")
        (sub / "c.wav").write_bytes(b"x")

        monkeypatch.setattr(indexer, "get_media_info", lambda fp: self._fake_probe())

        # Index the parent recursively → 3 files
        final = list(
            indexer.index_directory(tmp_path, recursive=True, db=db, discover_transcripts=False)
        )[-1]
        assert final["indexed"] == 3

        # Scan the child → both files count as indexed (via abs_path)
        res = scan_media_directory(sub, recursive=True, db=db)
        assert res.index["total"] == 2
        assert res.index["unindexed"] == 0, res.index

        # Re-indexing the child skips everything (no renewed FFprobe)
        final2 = list(
            indexer.index_directory(sub, recursive=True, db=db, discover_transcripts=False)
        )[-1]
        assert final2["skipped"] == 2
        assert final2["indexed"] == 0

        # No duplicates: media_files has exactly 3 rows (not 5)
        with db.connection() as conn:
            n = conn.execute("SELECT COUNT(*) AS c FROM media_files").fetchone()["c"]
        assert n == 3, f"duplicate created: {n} rows instead of 3"

    def test_abs_path_backfilled_by_scan(self, monkeypatch, db, tmp_path):
        """Old v2 data (abs_path empty) gets backfilled by a parent scan
        — no force-reindex needed."""
        from whisper_tm.config import config
        from whisper_tm.scanner import indexer, scan_media_directory

        config.load(None)

        (tmp_path / "a.wav").write_bytes(b"x")
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "b.wav").write_bytes(b"x")

        monkeypatch.setattr(indexer, "get_media_info", lambda fp: self._fake_probe())
        list(indexer.index_directory(tmp_path, recursive=True, db=db, discover_transcripts=False))
        # simulate old DB: clear abs_path
        with db.connection() as conn:
            conn.execute("UPDATE media_files SET abs_path=''")

        # Parent scan (automatic on open) backfills abs_path
        scan_media_directory(tmp_path, recursive=True, db=db)
        with db.connection() as conn:
            filled = conn.execute(
                "SELECT COUNT(*) AS c FROM media_files WHERE abs_path != ''"
            ).fetchone()["c"]
        assert filled == 2

        # Afterward the child scan recognizes the file as indexed
        res = scan_media_directory(sub, recursive=True, db=db)
        assert res.index["unindexed"] == 0, res.index
