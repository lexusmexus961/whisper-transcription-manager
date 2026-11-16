"""
Tests for the paginated file list (UI rework 5.1):
get_media_files_page / get_media_stats_under / get_media_names_under
and the /api/files endpoints.

Key points:
- Scope via abs_path range, NEVER via rel_path (DB holds multiple roots)
- Row form identical to get_media_file_at (db_row_to_entry contract)
- Counter definitions == list definitions (tab filter)
- error/cancelled via the MOST RECENT queue outcome per file
"""

import pytest

from whisper_tm.database import Database

ROOT = "/data/videos"
FOREIGN = "/data/other"


@pytest.fixture
def db(tmp_path):
    d = Database(tmp_path / "test.db")
    d.initialize()
    with d.connection() as conn:
        # Two roots: 6 files in own, 2 in foreign.
        own_ones = [
            ("a.mp4", "done", 100, 10),
            ("b.mp4", "pending", 200, 20),
            ("sub/c.mp4", "pending", 300, 30),
            ("d.mp4", "skipped", 400, 40),
            ("e.mp4", "orphan", 500, 50),
            ("Latest.mp4", "done", 600, 60),
        ]
        for rel, status, size, mod in own_ones:
            conn.execute(
                "INSERT INTO media_files (rel_path, abs_path, status, file_size,"
                " modified_at) VALUES (?, ?, ?, ?, ?)",
                (rel, f"{ROOT}/{rel}", status, size, mod),
            )
        for rel in ("x.mp4", "a.mp4"):  # a.mp4 also under the FOREIGN root
            conn.execute(
                "INSERT INTO media_files (rel_path, abs_path, status, file_size,"
                " modified_at) VALUES (?, ?, 'done', 1, 1)",
                (rel, f"{FOREIGN}/{rel}"),
            )
        # Transcript for a.mp4 (own root)
        conn.execute(
            "INSERT INTO transcripts (rel_path, abs_path, complete, word_count,"
            " duration_sec, processing_secs, segment_count, updated_at,"
            " processing_time) VALUES ('a.mp4', ?, 1, 42, 60.0, 6.0, 3, 111,"
            " '6s')",
            (f"{ROOT}/a.mp4",),
        )
        # Queue history for b.mp4: first error, then cancelled → counts as
        # cancelled (most recent outcome wins); sub/c.mp4 is running now.
        conn.execute("INSERT INTO queue_items (filename, status) VALUES ('b.mp4', 'error')")
        conn.execute("INSERT INTO queue_items (filename, status) VALUES ('b.mp4', 'cancelled')")
        conn.execute(
            "INSERT INTO queue_items (filename, status) VALUES ('sub/c.mp4', 'processing')"
        )
    return d


class TestPage:
    def test_scope_covers_only_its_own_root(self, db):
        r = db.get_media_files_page(ROOT)
        assert r["total"] == 6
        names = [z["rel_path"] for z in r["rows"]]
        assert "x.mp4" not in names

    def test_a_foreign_root_stays_separate(self, db):
        r = db.get_media_files_page(FOREIGN)
        assert r["total"] == 2

    def test_offset_limit_continuous(self, db):
        all_items = db.get_media_files_page(ROOT, limit=10)["rows"]
        page = db.get_media_files_page(ROOT, limit=2, offset=2)["rows"]
        assert [z["rel_path"] for z in page] == [z["rel_path"] for z in all_items[2:4]]

    def test_sort_size_desc(self, db):
        r = db.get_media_files_page(ROOT, sort="size", sort_dir="desc")
        sizes = [z["file_size"] for z in r["rows"]]
        assert sizes == sorted(sizes, reverse=True)

    def test_sort_tx_date(self, db):
        r = db.get_media_files_page(ROOT, sort="tx_date", sort_dir="desc")
        assert r["rows"][0]["rel_path"] == "a.mp4"  # the only transcript

    def test_row_shape_matches_the_single_fetch(self, db):
        """Contract: page row == get_media_file_at row (scan_entry)."""
        page = db.get_media_files_page(ROOT, name_filter="a.mp4")
        line = next(z for z in page["rows"] if z["rel_path"] == "a.mp4")
        single = db.get_media_file_at(f"{ROOT}/a.mp4")
        assert set(line.keys()) == set(single.keys())
        assert line == single

    def test_tab_done(self, db):
        r = db.get_media_files_page(ROOT, tab="done")
        assert r["total"] == 2
        assert all(z["status"] == "done" for z in r["rows"])

    def test_tab_pending_without_the_running_one(self, db):
        # b.mp4 + sub/c.mp4 are pending, but sub/c.mp4 is running now
        r = db.get_media_files_page(ROOT, tab="pending")
        assert [z["rel_path"] for z in r["rows"]] == ["b.mp4"]

    def test_tab_processing(self, db):
        r = db.get_media_files_page(ROOT, tab="processing")
        assert [z["rel_path"] for z in r["rows"]] == ["sub/c.mp4"]

    def test_the_cancelled_tab_uses_the_latest_outcome(self, db):
        """b.mp4 has error THEN cancelled — counts only as cancelled."""
        assert db.get_media_files_page(ROOT, tab="error")["total"] == 0
        r = db.get_media_files_page(ROOT, tab="cancelled")
        assert [z["rel_path"] for z in r["rows"]] == ["b.mp4"]

    def test_tab_orphan(self, db):
        r = db.get_media_files_page(ROOT, tab="orphan")
        assert [z["rel_path"] for z in r["rows"]] == ["e.mp4"]

    def test_name_filter(self, db):
        r = db.get_media_files_page(ROOT, name_filter="Latest")
        assert r["total"] == 1
        assert r["rows"][0]["rel_path"] == "Latest.mp4"

    def test_name_filter_folds_case_beyond_ascii(self, db):
        """LIKE folds ASCII only: "über" missed "Über …" in the list."""
        with db.connection() as conn:
            conn.execute(
                "INSERT INTO media_files (rel_path, abs_path, status, file_size,"
                " modified_at) VALUES (?, ?, 'pending', 1, 1)",
                ("Über Ärger.mp4", f"{ROOT}/Über Ärger.mp4"),
            )
        for q in ("über", "ÜBER", "ärger", "*ärger*", "über ärger.mp4"):
            r = db.get_media_files_page(ROOT, name_filter=q)
            assert [z["rel_path"] for z in r["rows"]] == ["Über Ärger.mp4"], q
        assert db.get_media_files_page(ROOT, name_filter="latest")["total"] == 1

    def test_name_filter_wildcards_harmless(self, db):
        assert db.get_media_files_page(ROOT, name_filter="%")["total"] == 0

    def test_star_pattern_matches_file_names(self, db):
        """The filter advertises *replay* and *.mp4 — so * must work,
        also for files in subfolders (sub/c.mp4)."""
        assert db.get_media_files_page(ROOT, name_filter="*.mp4")["total"] == 6
        rows = db.get_media_files_page(ROOT, name_filter="c*")["rows"]
        assert [z["rel_path"] for z in rows] == ["sub/c.mp4"]
        assert db.get_media_files_page(ROOT, name_filter="*test*")["total"] == 1

    def test_star_pattern_is_anchored(self, db):
        """A pattern is the whole name: 'a*' is a.mp4, not Latest.mp4."""
        rows = db.get_media_files_page(ROOT, name_filter="a*")["rows"]
        assert [z["rel_path"] for z in rows] == ["a.mp4"]

    def test_question_mark_is_one_character(self, db):
        assert db.get_media_files_page(ROOT, name_filter="?.mp4")["total"] == 5

    def test_unembedded_with_keys(self, db):
        r = db.get_media_files_page(ROOT, tab="unembedded", embedded_keys={"a.mp4"})
        assert [z["rel_path"] for z in r["rows"]] == ["Latest.mp4"]

    def test_unembedded_knows_the_identity_keys(self, db):
        """The vector store keys transcripts by their identity; matched
        against the rel_path alone, no file ever counted as embedded."""
        keys = {f"{ROOT}/a.mp4"}
        r = db.get_media_files_page(ROOT, tab="unembedded", embedded_keys=keys)
        assert [z["rel_path"] for z in r["rows"]] == ["Latest.mp4"]
        assert db.get_media_names_under(ROOT, tab="unembedded", embedded_keys=keys) == [
            "Latest.mp4"
        ]

    def test_the_name_sort_goes_by_the_identity(self, db):
        """rel_path is relative to the folder open when a file was indexed:
        "b.mp4" (indexed with sub2 open) sorted before "sub2/a.mp4"."""
        with db.connection() as conn:
            for rel, name in (("sub2/a.mp4", "a.mp4"), ("b.mp4", "b.mp4")):
                conn.execute(
                    "INSERT INTO media_files (rel_path, abs_path, status) VALUES (?, ?, 'pending')",
                    (rel, f"{ROOT}/sub2/{name}"),
                )
        rows = db.get_media_files_page(f"{ROOT}/sub2")["rows"]
        assert [z["abs_path"] for z in rows] == [f"{ROOT}/sub2/a.mp4", f"{ROOT}/sub2/b.mp4"]
        down = db.get_media_files_page(f"{ROOT}/sub2", sort_dir="desc")["rows"]
        assert [z["abs_path"] for z in down] == [f"{ROOT}/sub2/b.mp4", f"{ROOT}/sub2/a.mp4"]


class TestStats:
    def test_counters_match_the_list(self, db):
        """Every tab counter == total of the corresponding list query."""
        s = db.get_media_stats_under(ROOT)
        for tab in ("done", "pending", "processing", "error", "cancelled", "orphan"):
            expected = db.get_media_files_page(ROOT, tab=tab)["total"]
            assert s[tab] == expected, tab
        assert s["total"] == 6
        assert s["skipped"] == 1

    def test_sums(self, db):
        s = db.get_media_stats_under(ROOT)
        assert s["total_size"] == 100 + 200 + 300 + 400 + 500 + 600
        assert s["total_words"] == 42
        assert s["transcribed_audio_sec"] == 60.0

    def test_remainder_and_speed_only_from_what_counts(self, db):
        """Remaining = only waiting durations (not skipped ones); rate = only
        transcripts WITH words (silent 24h recordings would distort it otherwise)."""
        with db.connection() as conn:
            # Durations: b (pending) 100 s, sub/c (pending, no duration), d (skipped) 999 s
            conn.execute(
                "UPDATE media_files SET duration_sec = 100 WHERE rel_path = 'b.mp4' AND abs_path LIKE ?",
                (ROOT + "/%",),
            )
            conn.execute("UPDATE media_files SET duration_sec = 999 WHERE rel_path = 'd.mp4'")
            # Latest.mp4: done, 0 words, 1 h of silence "processed" in 60 s
            conn.execute(
                "INSERT INTO transcripts (rel_path, abs_path, complete, word_count,"
                " duration_sec, processing_secs, segment_count, updated_at, processing_time)"
                " VALUES ('Latest.mp4', ?, 1, 0, 3600.0, 60.0, 0, 111, '60s')",
                (f"{ROOT}/Latest.mp4",),
            )
        s = db.get_media_stats_under(ROOT)
        assert s["pending_media_dur_sec"] == 100.0  # d.mp4 (skipped) doesn't count
        assert s["pending_without_dur"] == 1  # sub/c.mp4
        assert s["speech_audio_sec"] == 60.0 and s["speech_proc_sec"] == 6.0  # only a.mp4
        assert s["transcribed_audio_sec"] == 3660.0  # total chip stays incl. silence

    def test_a_foreign_root_stays_separate(self, db):
        s = db.get_media_stats_under(FOREIGN)
        assert s["total"] == 2 and s["done"] == 2


class TestNames:
    def test_order_matches_the_page(self, db):
        names = db.get_media_names_under(ROOT)
        page = db.get_media_files_page(ROOT, limit=10)["rows"]
        assert names == [z["rel_path"] for z in page]

    def test_with_status(self, db):
        lines = db.get_media_names_under(ROOT, tab="orphan", with_status=True)
        assert lines == [("e.mp4", "orphan")]

    def test_names_are_relative_to_the_folder_listed(self, db):
        """Handed out as names, the stored rel_paths made a select-all under
        a subfolder hold "sub2/a.mp4" where the list says "a.mp4" — and act
        on files that aren't there."""
        with db.connection() as conn:
            for rel, name in (("sub2/a.mp4", "a.mp4"), ("b.mp4", "b.mp4")):
                conn.execute(
                    "INSERT INTO media_files (rel_path, abs_path, status) VALUES (?, ?, 'pending')",
                    (rel, f"{ROOT}/sub2/{name}"),
                )
        assert db.get_media_names_under(f"{ROOT}/sub2") == ["a.mp4", "b.mp4"]
        assert db.get_media_names_under(f"{ROOT}/sub2", with_status=True) == [
            ("a.mp4", "pending"),
            ("b.mp4", "pending"),
        ]


class TestEndpoints:
    @pytest.fixture
    def client(self, db, tmp_path, monkeypatch):
        from whisper_tm.config import config

        config.load(None)
        # patch media_dir directly onto the resolved value (the root
        # doesn't exist on disk; the setter would reject that) —
        # and reset it AFTER the test, otherwise later tests inherit
        # /data/videos as the media directory.
        old_raw = config._data.get("directories", {}).get("media", "")
        old_resolved = config._resolved_media
        config._data.setdefault("directories", {})["media"] = ROOT
        from pathlib import Path

        config._resolved_media = Path(ROOT)
        import whisper_tm.extensions as ext

        monkeypatch.setattr(ext, "db", db)
        # api.files binds ext.db at import time — patch onto the test db
        import whisper_tm.api.files as files_mod

        monkeypatch.setattr(files_mod, "db", db)
        from flask import Flask

        app = Flask(__name__)
        app.register_blueprint(files_mod.files_bp)
        yield app.test_client()
        config._data["directories"]["media"] = old_raw
        config._resolved_media = old_resolved

    def test_files_endpoint(self, client):
        d = client.get("/api/files?limit=3").get_json()
        assert d["total"] == 6
        assert len(d["files"]) == 3
        assert d["files"][0]["name"]  # entry form (db_row_to_entry)

    def test_files_stats_endpoint(self, client):
        d = client.get("/api/files/stats").get_json()
        assert d["total"] == 6 and d["done"] == 2

    def test_files_names_with_status(self, client):
        d = client.get("/api/files/names?tab=done&with_status=1").get_json()
        assert d["total"] == 2
        assert len(d["names"]) == len(d["status"]) == 2
