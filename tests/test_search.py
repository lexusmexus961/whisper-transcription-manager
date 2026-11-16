"""
Tests for search: Facet aggregation (DB) and search API endpoints.

The facets are the data basis for the heatmap in the large search view — they
must refer to ALL results, not just the currently loaded page.
That's exactly what the tests here check.
"""

import pytest

from whisper_tm.database import Database


@pytest.fixture
def db(tmp_path):
    d = Database(tmp_path / "test.db")
    d.initialize()
    return d


def _tx(filename, segments, duration):
    return {
        "filename": filename,
        "language": "de",
        "model": "turbo",
        "vocal_isolation": False,
        "vocal_model": None,
        "segments": segments,
        "speakers": sorted({s["speaker"] for s in segments}),
        "total_duration": "0:10:00",
        "total_duration_sec": duration,
        "processing_time": "5s",
        "processing_secs": 5.0,
        "total_chunks": 1,
        "complete": True,
        "warning_count": 0,
        "warnings": [],
    }


def _seg(speaker, start, text):
    return {
        "speaker": speaker,
        "start": start,
        "end": start + 2.0,
        "start_fmt": f"0:00:{int(start) % 60:02d}",
        "end_fmt": "0:00:00",
        "text": text,
    }


@pytest.fixture
def facet_db(db):
    # video1: 100s long, hits clustered at the start (0s, 5s, 10s)
    db.upsert_transcript(
        "video1.mp4",
        _tx(
            "video1.mp4",
            [
                _seg("SPEAKER_00", 0.0, "Server migration pending."),
                _seg("SPEAKER_00", 5.0, "The migration is running."),
                _seg("SPEAKER_01", 10.0, "Migration complete."),
                _seg("SPEAKER_01", 90.0, "Something completely different."),
            ],
            100.0,
        ),
    )
    # video2: 100s long, one hit right at the end
    db.upsert_transcript(
        "video2.mp4",
        _tx(
            "video2.mp4",
            [
                _seg("SPEAKER_00", 0.0, "Irrelevant preamble."),
                _seg("SPEAKER_00", 99.0, "Finally, the migration."),
            ],
            100.0,
        ),
    )
    return db


class TestSearchFacets:
    def test_totals_and_file_count(self, facet_db):
        f = facet_db.search_facets("Migration")
        assert f["total"] == 4  # 3x video1 + 1x video2
        assert f["file_count"] == 2
        assert len(f["files"]) == 2

    def test_sorted_by_hits(self, facet_db):
        f = facet_db.search_facets("Migration")
        assert f["files"][0]["filename"] == "video1.mp4"
        assert f["files"][0]["hits"] == 3
        assert f["files"][1]["hits"] == 1

    def test_histogram_length_matches_buckets(self, facet_db):
        f = facet_db.search_facets("Migration", buckets=10)
        assert f["buckets"] == 10
        for entry in f["files"]:
            assert len(entry["histogram"]) == 10
            assert sum(entry["histogram"]) == entry["hits"]

    def test_histogram_localizes_hits_in_time(self, facet_db):
        """The core of the heatmap: the buckets must show WHERE the hits are."""
        f = facet_db.search_facets("Migration", buckets=10)
        by_name = {e["filename"]: e for e in f["files"]}
        # video1: all 3 hits in the first 10% of the runtime → bucket 0/1
        v1 = by_name["video1.mp4"]["histogram"]
        assert sum(v1[:2]) == 3
        assert sum(v1[2:]) == 0
        # video2: single hit at 99s of 100s → last bucket
        v2 = by_name["video2.mp4"]["histogram"]
        assert v2[-1] == 1
        assert sum(v2[:-1]) == 0

    def test_two_recordings_of_one_name_stay_two(self, db):
        """rel_path is relative to the folder open at transcription: two
        "intro.mp4" in two folders opened that way were counted as one."""
        for folder in ("/lib/a", "/lib/b"):
            db.upsert_transcript(
                "intro.mp4",
                _tx(
                    "intro.mp4",
                    [
                        _seg("SPEAKER_00", 1.0, "The migration starts."),
                    ],
                    10.0,
                ),
                media_dir=folder,
                abs_path=f"{folder}/intro.mp4",
            )
        f = db.search_facets("migration")
        assert f["file_count"] == 2
        assert sorted(e["media_dir"] for e in f["files"]) == ["/lib/a", "/lib/b"]
        assert [e["hits"] for e in f["files"]] == [1, 1]

    def test_buckets_clamped(self, facet_db):
        assert facet_db.search_facets("Migration", buckets=99999)["buckets"] == 200
        assert facet_db.search_facets("Migration", buckets=0)["buckets"] == 40

    def test_file_filter(self, facet_db):
        f = facet_db.search_facets("Migration", file_filter="video2")
        assert f["file_count"] == 1
        assert f["total"] == 1

    def test_speaker_filter(self, facet_db):
        f = facet_db.search_facets("Migration", speaker_filter="SPEAKER_01")
        assert f["total"] == 1

    def test_no_match(self, facet_db):
        f = facet_db.search_facets("xyznonexistent")
        assert f["total"] == 0
        assert f["files"] == []

    def test_empty_query(self, facet_db):
        f = facet_db.search_facets("   ")
        assert f["total"] == 0
        assert f["files"] == []

    def test_facet_total_matches_search_total(self, facet_db):
        """Facets and search must not diverge — otherwise the sidebar
        shows different numbers than the results list."""
        assert facet_db.search_facets("Migration")["total"] == facet_db.search("Migration")["total"]

    def test_facets_independent_of_pagination(self, facet_db):
        """The facets know all hits, even when the page shows only one."""
        page = facet_db.search("Migration", limit=1)
        assert len(page["results"]) == 1
        assert facet_db.search_facets("Migration")["total"] == 4


class TestFileExact:
    """file_exact for the drill-down of the hit card: there the path is
    known and must match exactly, otherwise the card counts foreign files too."""

    @pytest.fixture
    def nested(self, db):
        # "a/clip.mp4" is a substring of "b/a/clip.mp4" — exactly the case
        # where LIKE %…% sweeps up too much.
        db.upsert_transcript(
            "a/clip.mp4",
            _tx(
                "clip.mp4",
                [
                    _seg("SPEAKER_00", 0.0, "Migration here."),
                ],
                10.0,
            ),
        )
        db.upsert_transcript(
            "b/a/clip.mp4",
            _tx(
                "clip.mp4",
                [
                    _seg("SPEAKER_00", 0.0, "Migration there."),
                    _seg("SPEAKER_00", 5.0, "Migration again."),
                ],
                10.0,
            ),
        )
        return db

    def test_substring_matches_both(self, nested):
        assert nested.search("Migration", file_filter="a/clip.mp4")["total"] == 3

    def test_exact_matches_one(self, nested):
        r = nested.search("Migration", file_filter="a/clip.mp4", file_exact=True)
        assert r["total"] == 1
        assert r["results"][0]["filename"] == "a/clip.mp4"

    def test_exact_matches_nested(self, nested):
        r = nested.search("Migration", file_filter="b/a/clip.mp4", file_exact=True)
        assert r["total"] == 2

    def test_exact_unknown_path_is_empty(self, nested):
        assert nested.search("Migration", file_filter="clip.mp4", file_exact=True)["total"] == 0

    def test_api_passes_flag(self, client):
        r = client.get("/api/search?q=migration&file=x&file_exact=1")
        assert r.status_code == 200


class TestSearchStillWorks:
    """Regression: search() now uses the same clause builder as the
    facets — the existing semantics must stay unchanged."""

    def test_search_with_filters(self, facet_db):
        r = facet_db.search("Migration", file_filter="video1", speaker_filter="SPEAKER_00")
        assert r["total"] == 2
        assert all(x["speaker"] == "SPEAKER_00" for x in r["results"])

    def test_truncated_flag(self, facet_db):
        assert facet_db.search("Migration", limit=1)["truncated"] is True
        assert facet_db.search("Migration", limit=99)["truncated"] is False

    def test_empty_query_returns_empty(self, facet_db):
        assert facet_db.search("   ")["total"] == 0


# ── API endpoints ──────────────────────────────────────────────────────────


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    from whisper_tm.config import config

    config.load(None)
    config._data.setdefault("directories", {})["media"] = str(tmp_path)
    config._resolve_directories()
    config._ensure_directories()
    from whisper_tm.app import create_app

    return create_app().test_client()


class TestFacetsEndpoint:
    def test_short_query_is_empty_not_error(self, client):
        r = client.get("/api/search/facets?q=a")
        assert r.status_code == 200
        assert r.get_json()["total"] == 0

    def test_long_query_rejected(self, client):
        r = client.get("/api/search/facets?q=" + "x" * 501)
        assert r.status_code == 400

    def test_response_shape(self, client):
        d = client.get("/api/search/facets?q=migration").get_json()
        for key in ("files", "total", "file_count", "buckets"):
            assert key in d

    def test_search_endpoint_pagination_params(self, client):
        d = client.get("/api/search?q=migration&limit=10&offset=0").get_json()
        assert "results" in d and "total" in d


class TestTheRootOfAHit:
    """A hit's root comes from its identity. The stored ``media_dir`` is not
    always the folder ``rel_path`` is relative to (some transcripts carried
    their file's own folder), and such a hit's context
    said "Transcript not found" — seen clicking through the search."""

    @pytest.fixture
    def skewed(self, db, tmp_path):
        root = tmp_path / "nas"
        f = root / "videos" / "Archive" / "clip.mp4"
        db.upsert_transcript(
            "videos/Archive/clip.mp4",
            _tx(
                "clip.mp4",
                [
                    _seg("SPEAKER_00", 0.0, "Budget review today."),
                    _seg("SPEAKER_00", 4.0, "And one more sentence."),
                ],
                10.0,
            ),
            media_dir=str(root),
            abs_path=str(f),
        )
        with db.connection() as conn:
            conn.execute("UPDATE transcripts SET media_dir = ?", (str(f.parent),))
        return db, root

    def test_the_helper(self):
        from whisper_tm.database import _root_of

        assert _root_of("a/b.mp4", "/x/a/b.mp4", "/x/a") == "/x"
        assert _root_of("a/b.mp4", "/y/other.mp4", "/x") == "/x", "doesn't fit: as stored"
        assert _root_of("a/b.mp4", "", "/x") == "/x"
        assert _root_of("a/b.mp4", None, None) == ""
        assert _root_of("/abs/b.mp4", "/abs/b.mp4", "/m") == "/m"
        assert _root_of("b.mp4", "/b.mp4", "") == "/"

    def test_search_and_facets_hand_on_the_right_root(self, skewed):
        db, root = skewed
        hit = db.search("Budget")["results"][0]
        assert hit["filename"] == "videos/Archive/clip.mp4" and hit["media_dir"] == str(root)
        assert db.search_facets("Budget")["files"][0]["media_dir"] == str(root)

    def test_the_context_of_such_a_hit(self, skewed, monkeypatch):
        from flask import Flask

        from whisper_tm.api import search as search_api

        db, root = skewed
        monkeypatch.setattr(search_api, "db", db)
        a = Flask(__name__)
        a.register_blueprint(search_api.search_bp)
        c = a.test_client()
        hit = c.get("/api/search?q=Budget").get_json()["results"][0]
        assert hit["file_dir"] == str(root)
        d = c.get(
            "/api/search/context",
            query_string={
                "file": hit["filename"],
                "dir": hit["file_dir"],
                "start": hit["start"],
                "n": 3,
            },
        ).get_json()
        assert d["segments"][0]["text"] == "Budget review today."

    def test_a_file_hit_hands_on_its_own_root(self, skewed, monkeypatch):
        # File hits carried media_dir while rel_path was relative to
        # another root — the joined path didn't exist
        from flask import Flask

        from whisper_tm.api import search as search_api

        db, root = skewed
        monkeypatch.setattr(search_api, "db", db)
        monkeypatch.setattr(search_api.config, "get", lambda *a, **k: k.get("default"))
        a = Flask(__name__)
        a.register_blueprint(search_api.search_bp)
        files = a.test_client().get("/api/search?q=clip&mode=fts").get_json()["files"]
        assert files and files[0]["filename"] == "videos/Archive/clip.mp4"
        assert files[0]["file_dir"] == str(root)


class TestNumbersFromTheRequest:
    """limit=-5 was no limit at all (a response of many megabytes); huge
    offsets and word counts were OverflowErrors (500); the speakers' list
    took seconds on every call."""

    @pytest.fixture
    def api(self, facet_db, monkeypatch):
        from flask import Flask

        from whisper_tm.api import search as search_api

        monkeypatch.setattr(search_api, "db", facet_db)
        search_api._speakers_cache.update(at=-1e9, db=None, speakers=[])
        a = Flask(__name__)
        a.register_blueprint(search_api.search_bp)
        return a.test_client(), search_api, facet_db

    def test_limit_offset_and_counts_are_clamped(self, api):
        c, _m, _db = api
        assert c.get("/api/search?q=migration&mode=fts").get_json()["total"] >= 3
        d = c.get("/api/search?q=migration&mode=fts&limit=-5").get_json()
        assert len(d["results"]) == 1
        assert c.get(f"/api/search?q=migration&mode=fts&offset={10**30}").status_code == 200
        r = c.get(f"/api/search?mode=files&word_count_min={10**30}&offset={10**30}")
        assert r.status_code == 200

    def test_the_speakers_are_kept_a_minute(self, api, monkeypatch):
        c, search_api, db = api
        first = c.get("/api/search/speakers").get_json()["speakers"]
        db.upsert_transcript(
            "video9.mp4", _tx("video9.mp4", [_seg("SPEAKER_42", 0.0, "Someone new.")], 10.0)
        )
        assert c.get("/api/search/speakers").get_json()["speakers"] == first
        monkeypatch.setattr(search_api, "SPEAKERS_TTL_SEC", 0.0)
        assert "SPEAKER_42" in c.get("/api/search/speakers").get_json()["speakers"]
