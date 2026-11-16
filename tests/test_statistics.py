"""
Tests for stats aggregation (``api/stats.py``).

The core of the change is that global and directory stats go through
*the same* function — just with or without subdirectories. This suite therefore
checks above all that the range cuts cleanly: not too much (the folder
``Series`` must not collect ``Series2``), not too little (subtrees count
completely), and above all that old data from **foreign roots** is not
counted in.

The last point is the real reason why filtering is done via ``abs_path``
and not via ``rel_path``: ``docs/PATHS.md`` declares ``rel_path`` to be
a mere display value, which becomes stale after a root change and is not
unique across multiple roots. The database intentionally keeps rows from
earlier roots — a stat over ``rel_path`` would have silently counted them.

Rows are inserted directly via SQL. This is intentional: the aggregation
is tested, not the write path — and so each test visibly shows
what numbers must come out.
"""

from __future__ import annotations

import time

import pytest
from flask import Flask

from whisper_tm.api.stats import (
    clear_cache,
    compute,
    normalize_prefix,
    stats_bp,
)
from whisper_tm.database import Database

ROOT = "/media/active"
FOREIGN = "/media/elsewhere"


# ── Setup ────────────────────────────────────────────────────────────────


def _transcript_record(
    conn,
    rel,
    *,
    root=ROOT,
    duration=600.0,
    compute_time=60.0,
    words=1000,
    segments=50,
    speaker=2,
    language="de",
    model="turbo",
    complete=1,
    warnings=0,
    vocal=0,
    age_days=1,
    abs_path=None,
):
    ap = abs_path if abs_path is not None else f"{root}/{rel}"
    conn.execute(
        """
        INSERT INTO transcripts
            (rel_path, abs_path, language, model, duration_sec, processing_secs,
             word_count, char_count, segment_count, speaker_count, warning_count,
             vocal_isolation, complete, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """,
        (
            rel,
            ap,
            language,
            model,
            duration,
            compute_time,
            words,
            words * 6,
            segments,
            speaker,
            warnings,
            vocal,
            complete,
            time.time() - age_days * 86400,
        ),
    )
    return conn.execute("SELECT last_insert_rowid() AS id").fetchone()["id"]


def _media_file(
    conn,
    rel,
    *,
    root=ROOT,
    status="done",
    size=1000,
    video=1,
    duration=600.0,
    extension=".mp4",
    indexed_at=1.0,
    skip_reason="",
):
    conn.execute(
        """
        INSERT INTO media_files
            (rel_path, abs_path, extension, is_video, file_size, duration_sec,
             status, skip_reason, indexed_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """,
        (rel, f"{root}/{rel}", extension, video, size, duration, status, skip_reason, indexed_at),
    )


def _segments(conn, transcript_id, texts, speaker="SPEAKER_00"):
    for i, t in enumerate(texts):
        conn.execute(
            """
            INSERT INTO segments
                (transcript_id, seg_index, speaker, start_sec, end_sec, text)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            (transcript_id, i, speaker, i * 5.0, i * 5.0 + 4.0, t),
        )


@pytest.fixture(autouse=True)
def _cache_free():
    """Without this, the second test in the module sees the numbers of the first."""
    clear_cache()
    yield
    clear_cache()


@pytest.fixture
def root(monkeypatch):
    """Pin ``config.media_dir`` to a fixed root."""
    from whisper_tm.config import config

    monkeypatch.setattr(type(config), "media_dir", property(lambda self: ROOT))
    return ROOT


@pytest.fixture
def db(tmp_path, monkeypatch, root):
    """Database with a small, hand-verifiable library.

    Setup (deliberately stocked with traps):

        Series/S1/a.mp4      600 s   done
        Series/S1/b.mp4     1200 s   done
        Series/S2/c.mp4      120 s   done
        Series2/d.mp4       3000 s   done   ← must NOT count toward "Series"
        loose.mp4            300 s   done   ← directly in the root
        Series/S1/open_count.mp4    –     pending  ← Medium without transcript

    And outside the active root, below /media/elsewhere:

        Series/S1/a.mp4     9999 s   done   ← same rel_path as at the top!
    """
    d = Database(tmp_path / "stats.db")
    d.initialize()
    with d.connection() as conn:
        t1 = _transcript_record(
            conn,
            "Series/S1/a.mp4",
            duration=600,
            compute_time=60,
            words=1200,
            segments=40,
            speaker=2,
        )
        _transcript_record(
            conn,
            "Series/S1/b.mp4",
            duration=1200,
            compute_time=200,
            words=3000,
            segments=90,
            speaker=3,
            warnings=2,
            vocal=1,
        )
        _transcript_record(
            conn,
            "Series/S2/c.mp4",
            duration=120,
            compute_time=30,
            words=200,
            segments=10,
            speaker=1,
            language="en",
            model="large-v3",
        )
        _transcript_record(
            conn,
            "Series2/d.mp4",
            duration=3000,
            compute_time=500,
            words=9000,
            segments=300,
            speaker=5,
        )
        _transcript_record(
            conn, "loose.mp4", duration=300, compute_time=50, words=500, segments=20, speaker=1
        )

        # Old data from another root — same rel_path, different file.
        _transcript_record(
            conn,
            "Series/S1/a.mp4",
            root=FOREIGN,
            duration=9999,
            compute_time=9999,
            words=99999,
            segments=999,
            speaker=9,
        )

        _media_file(conn, "Series/S1/a.mp4", size=1000)
        _media_file(conn, "Series/S1/b.mp4", size=2000)
        _media_file(conn, "Series/S1/open_count.mp4", status="pending", size=4000)
        _media_file(conn, "Series/S2/c.mp4", size=500, extension=".mkv")
        _media_file(conn, "Series2/d.mp4", size=8000)
        _media_file(conn, "loose.mp4", size=100, video=0, extension=".mp3")
        _media_file(conn, "Series/S1/a.mp4", root=FOREIGN, size=777000)

        _segments(
            conn,
            t1,
            [
                "The motor runs very quietly and the motor is quiet",
                "Ein Motor braucht Wartung",
            ],
        )

    monkeypatch.setattr("whisper_tm.api.stats.db", d)
    return d


@pytest.fixture
def client(db):
    app = Flask(__name__)
    app.register_blueprint(stats_bp)
    app.config["TESTING"] = False  # real response instead of passed-through error
    return app.test_client()


# ── Prefix Normalization ─────────────────────────────────────────────────


class TestPrefix:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("", ""),
            ("/", ""),
            ("Series", "Series"),
            ("/Series/S1/", "Series/S1"),
            ("Series\\S1", "Series/S1"),  # Windows separator
            ("../../etc", "etc"),  # Traversal filtered out
            ("..", ""),
            ("./Series/./S1", "Series/S1"),
            ("Series//S1", "Series/S1"),  # double separator
        ],
    )
    def test_normalized(self, raw, expected):
        assert normalize_prefix(raw) == expected


# ── Foreign Roots ────────────────────────────────────────────────────────


class TestRootSeparation:
    """The reason for filtering via abs_path instead of rel_path."""

    def test_global_counts_only_the_active_root(self, db):
        """The legacy rows below /media/elsewhere must not show up.

        filtered by rel_path, it would be 6 transcripts and 15,219 s duration —
        including the 9,999 s from the foreign file.
        """
        r = compute("")
        assert r["transcripts"]["total"] == 5
        assert r["transcripts"]["duration"] == pytest.approx(600 + 1200 + 120 + 3000 + 300)
        assert r["transcripts"]["words"] == 1200 + 3000 + 200 + 9000 + 500

    def test_same_rel_path_does_not_collide(self, db):
        """'Series/S1' exists under both roots — only one is counted."""
        r = compute("Series/S1")
        assert r["transcripts"]["total"] == 2
        assert r["transcripts"]["duration"] == pytest.approx(1800)

    def test_media_from_a_foreign_root_stays_out(self, db):
        """The 777 MB of the foreign file must not inflate the data volume."""
        r = compute("")
        assert r["media"]["size"] == 1000 + 2000 + 4000 + 500 + 8000 + 100
        assert r["progress"]["total"] == 6

    def test_a_root_change_changes_the_result(self, db, monkeypatch):
        from whisper_tm.config import config

        before = compute("")["transcripts"]["total"]
        clear_cache()
        monkeypatch.setattr(type(config), "media_dir", property(lambda self: FOREIGN))
        after = compute("")["transcripts"]["total"]
        assert before == 5
        assert after == 1


# ── Scope: global ────────────────────────────────────────────────


class TestGlobal:
    def test_counts_the_root(self, db):
        r = compute("")
        assert "error" not in r
        assert r["scope"]["global"] is True
        assert r["progress"]["total"] == 6  # including the pending medium
        assert r["progress"]["done"] == 5

    def test_the_label_names_the_open_folder(self, db):
        """The numbers cover the open folder, not every root in the
        database — so the label names that folder instead of claiming
        "Entire library"."""
        from pathlib import Path

        from whisper_tm.config import config

        r = compute("")
        assert r["scope"]["label"] == Path(str(config.media_dir)).name
        assert r["scope"]["root"] == str(config.media_dir)
        assert r["progress"]["pending"] == 1

    def test_realtime_factor(self, db):
        """speed = audio seconds per second of compute time."""
        r = compute("")["transcripts"]
        assert r["speed"] == pytest.approx(5220 / 840, abs=0.01)

    def test_progress_percent(self, db):
        assert compute("")["progress"]["pct"] == pytest.approx(83.3, abs=0.1)

    def test_the_coverage_says_when_it_is_only_hidden(self, db, monkeypatch):
        """`rag.enabled` only shows the coverage. With it off the page said
        "Not enabled" while the search used a built index — it tells the
        two apart now."""
        from pathlib import Path

        from whisper_tm.rag import service

        monkeypatch.setattr(service, "get_rag_config", lambda: {"enabled": False})
        clear_cache()
        v = compute("")["vector"]
        assert v["active"] is False and v["hidden"] is True
        js = (
            Path(__file__).resolve().parent.parent / "whisper_tm" / "static" / "js" / "statsview.js"
        ).read_text(encoding="utf-8")
        assert "Show coverage</b>. The search uses what is embedded either way." in js
        assert "Not enabled." not in js


# ── Scope: directory ──────────────────────────────────────────────


class TestDirectoryScope:
    def test_the_subtree_is_complete(self, db):
        """ "Series" covers S1 and S2, so three transcripts."""
        r = compute("Series")
        assert r["scope"]["global"] is False
        assert r["scope"]["path"] == "Series"
        assert r["transcripts"]["total"] == 3
        assert r["transcripts"]["duration"] == pytest.approx(600 + 1200 + 120)

    def test_the_prefix_does_not_reach_too_far(self, db):
        """The classic:: "Series" must not include "Series2".

        Without the separator in the LIKE pattern it would be 4 instead of 3 transcripts, and the
        duration would be about 3000 s from Series2 too high.
        """
        series = compute("Series")["transcripts"]
        series2 = compute("Series2")["transcripts"]
        assert series["total"] == 3
        assert series2["total"] == 1
        assert series["duration"] == pytest.approx(1920)
        assert series2["duration"] == pytest.approx(3000)

    def test_deeper_level(self, db):
        r = compute("Series/S1")
        assert r["transcripts"]["total"] == 2
        assert r["transcripts"]["words"] == 1200 + 3000

    def test_media_follows_the_scope(self, db):
        r = compute("Series/S1")["progress"]
        assert r["total"] == 3  # a, b, open_count
        assert r["done"] == 2
        assert r["pending"] == 1

    def test_size_per_directory(self, db):
        assert compute("Series/S1")["media"]["size"] == 1000 + 2000 + 4000

    def test_an_unknown_directory_is_empty_not_broken(self, db):
        r = compute("Does/Not/Exist")
        assert "error" not in r
        assert r["transcripts"]["total"] == 0
        assert r["progress"]["total"] == 0
        assert r["progress"]["pct"] == 0.0  # no division by zero
        assert r["transcripts"]["speed"] == 0.0

    def test_the_sum_of_the_parts_equals_the_whole(self, db):
        """Consistency check between the scopes."""
        whole = compute("")["transcripts"]["words"]
        parts = sum(compute(p)["transcripts"]["words"] for p in ("Series", "Series2"))
        assert whole == parts + 500  # + the loose file in the root

    def test_without_a_media_directory(self, db, monkeypatch):
        from whisper_tm.config import config

        monkeypatch.setattr(type(config), "media_dir", property(lambda self: ""))
        r = compute("")
        assert r["transcripts"]["total"] == 0
        assert "media directory" in r["scope"]["hint"]


# ── LIKE special characters ────────────────────────────────────────────────────


class TestSpecialCharacters:
    @pytest.fixture
    def db_special(self, tmp_path, monkeypatch, root):
        d = Database(tmp_path / "special.db")
        d.initialize()
        with d.connection() as conn:
            _transcript_record(conn, "100%_Done/a.mp4", words=10)
            _transcript_record(conn, "100X0Done/b.mp4", words=20)
            _transcript_record(conn, "1000Done/c.mp4", words=40)
        monkeypatch.setattr("whisper_tm.api.stats.db", d)
        return d

    def test_percent_is_not_a_wildcard(self, db_special):
        """``%`` in a folder name must not match arbitrary characters.

        Without escaping, ``100%_Done/%`` would also match ``100X0Done`` and
        ``1000Done`` — the stats of a folder would then show
        foreign files.
        """
        r = compute("100%_Done")
        assert r["transcripts"]["total"] == 1
        assert r["transcripts"]["words"] == 10

    def test_underscore_is_not_a_wildcard(self, db_special):
        assert compute("100X0Done")["transcripts"]["total"] == 1


# ── Distributions ──────────────────────────────────────────────────────────


class TestDistributions:
    def test_duration_buckets_are_always_complete(self, db):
        """All four levels come back, even the empty ones."""
        b = compute("")["duration_buckets"]
        assert [x["key"] for x in b] == ["under_5min", "5_30min", "30min_2h", "over_2h"]
        by_key = {x["key"]: x["cnt"] for x in b}
        # Boundaries are inclusive at the bottom: 300 s counts as "5–30 min", not
        # to "< 5 min". Otherwise no one notices, but shifts the bars.
        assert by_key["under_5min"] == 1  # 120 s
        assert by_key["5_30min"] == 3  # 300 s, 600 s, 1200 s
        assert by_key["30min_2h"] == 1  # 3000 s
        assert by_key["over_2h"] == 0
        assert sum(x["cnt"] for x in b) == 5  # no file falls through the grid

    def test_buckets_for_an_empty_range(self, db):
        b = compute("Does/Not/Exist")["duration_buckets"]
        assert len(b) == 4
        assert all(x["cnt"] == 0 for x in b)

    def test_languages_and_models(self, db):
        r = compute("")
        languages = {x["name"]: x["cnt"] for x in r["languages"]}
        assert languages == {"de": 4, "en": 1}
        models = {x["name"]: x["cnt"] for x in r["models"]}
        assert models == {"turbo": 4, "large-v3": 1}

    def test_languages_follow_the_scope(self, db):
        languages = {x["name"]: x["cnt"] for x in compute("Series/S1")["languages"]}
        assert languages == {"de": 2}

    def test_extensions(self, db):
        e = {x["name"]: x["cnt"] for x in compute("")["extensions"]}
        assert e == {".mp4": 4, ".mkv": 1, ".mp3": 1}

    def test_features(self, db):
        f = compute("")["features"]
        assert f["vocal_iso"] == 1
        assert f["with_warning"] == 1
        assert f["diarized"] == 3  # speaker_count > 1: 2, 3, 5

    def test_diarized_is_what_ran_not_how_many_spoke(self, db):
        """Counting two or more speakers told a library diarized at 88 %
        that it was 12 %: a diarized monologue, or a voices-only run since,
        is diarized too. Rows without recorded settings still go by the
        speakers (the three above)."""
        import json

        with db.connection() as conn:
            for rel, settings, speakers in (
                ("Talks/mono.mp4", {"diarized": True}, 1),
                ("Talks/later.mp4", {"diarized": False, "voices": {"made": 1}}, 1),
                ("Talks/plain.mp4", {"diarized": False}, 1),
                ("Talks/odd.mp4", {"diarized": False}, 2),
            ):
                tid = _transcript_record(conn, rel, speaker=speakers)
                conn.execute(
                    "UPDATE transcripts SET settings_json=? WHERE id=?", (json.dumps(settings), tid)
                )
        clear_cache()
        assert compute("")["features"]["diarized"] == 3 + 2

    def test_activity_always_has_30_days(self, db):
        """Gaps are filled, otherwise the timeline collapses."""
        a = compute("")["activity"]
        assert len(a) == 30
        assert sum(x["cnt"] for x in a) == 5
        assert a == sorted(a, key=lambda x: x["tag"])

    def test_history_lists_only_active_days_in_order(self, db):
        """The whole timeline, but only days with activity — the client
        fills the calendar. Counts, audio and words add up per day."""
        h = compute("")["history"]
        assert h == sorted(h, key=lambda x: x["tag"])
        assert all(x["cnt"] > 0 for x in h)
        assert sum(x["cnt"] for x in h) == 5
        assert sum(x["words"] for x in h) == sum(x["words"] for x in [compute("")["transcripts"]])

    def test_history_follows_the_scope(self, db):
        assert sum(x["cnt"] for x in compute("Series/S1")["history"]) == 2


# ── Subdirectories ───────────────────────────────────────────────────────────


class TestSubdirectories:
    def test_global_shows_the_top_level(self, db):
        """Media files are counted, not transcripts.

        Series has 4 media files (a, b, open_count, S2/c), but only 3 transcripts —
        the open file belongs in the column, otherwise it's missing from the overview.
        """
        u = {x["name"]: x["total"] for x in compute("")["subfolders"]}
        assert u == {"Series": 4, "Series2": 1, ".": 1}

    def test_a_directory_shows_its_direct_children(self, db):
        u = {x["name"]: x["total"] for x in compute("Series")["subfolders"]}
        assert u == {"S1": 3, "S2": 1}

    def test_progress_comes_from_the_media_files(self, db):
        """Otherwise there would be folder column always 100%.

        A transcript only exists once the file is finished — if one counted
        Progress over transcripts, the column measures itself and
        contradicts the percentage in the header.
        """
        s1 = next(x for x in compute("Series")["subfolders"] if x["name"] == "S1")
        assert s1["total"] == 3  # a, b, open_count
        assert s1["done"] == 2  # one is pending
        assert s1["done"] < s1["total"]

    def test_the_text_volume_comes_from_the_transcripts(self, db):
        s1 = next(x for x in compute("Series")["subfolders"] if x["name"] == "S1")
        assert s1["words"] == 1200 + 3000
        assert s1["duration"] == pytest.approx(1800)

    def test_the_path_is_navigable(self, db):
        """The returned path must directly be the next scope."""
        children = compute("Series")["subfolders"]
        s1 = next(x for x in children if x["name"] == "S1")
        assert s1["path"] == "Series/S1"
        assert compute(s1["path"])["transcripts"]["total"] == 2

    def test_no_folders_from_foreign_roots(self, db):
        names = {x["name"] for x in compute("")["subfolders"]}
        assert "elsewhere" not in names


# ── Records ───────────────────────────────────────────────────────────────


class TestRecords:
    def test_longest_and_shortest(self, db):
        r = {x["key"]: x for x in compute("")["records"]}
        assert r["longest"]["path"] == "Series2/d.mp4"
        assert r["shortest"]["path"] == "Series/S2/c.mp4"

    def test_the_records_follow_the_scope(self, db):
        r = {x["key"]: x for x in compute("Series")["records"]}
        assert r["longest"]["path"] == "Series/S1/b.mp4"  # not Series2
        assert r["most_words"]["path"] == "Series/S1/b.mp4"

    def test_record_ignores_foreign_root(self, db):
        """The foreign file has 9999 s — it must not hold the record."""
        r = {x["key"]: x for x in compute("")["records"]}
        assert r["longest"]["value"] == 3000

    def test_the_display_path_is_relative(self, db):
        """The displayed path is relative to the root, not absolute."""
        r = {x["key"]: x for x in compute("")["records"]}
        assert not r["longest"]["path"].startswith("/")

    def test_an_empty_scope_without_records(self, db):
        assert compute("Does/Not/Exist")["records"] == []


# ── Text analysis ───────────────────────────────────────────────────────────


class TestText:
    def test_only_on_request(self, db):
        assert "text" not in compute("")
        assert "text" in compute("", with_text=True)

    def test_top_words_without_stopwords(self, db, monkeypatch):
        # A library in two languages names both (the default is English
        # only); the sample has a line of each
        from whisper_tm.config import config

        orig = config.get
        monkeypatch.setattr(
            config,
            "get",
            lambda *k, default=None: (
                ["de", "en"] if k == ("search", "stopword_languages") else orig(*k, default=default)
            ),
        )
        t = compute("", with_text=True)["text"]
        words = {x["word"]: x["cnt"] for x in t["top_words"]}
        assert words["motor"] == 3
        # the stop words of both lists are missing
        assert not ({"the", "and", "is", "very", "ein"} & set(words))

    def test_english_stopwords_by_default(self, db):
        t = compute("", with_text=True)["text"]
        words = {x["word"] for x in t["top_words"]}
        assert not ({"the", "and", "is", "very"} & words) and "ein" in words

    def test_bigrams(self, db):
        t = compute("", with_text=True)["text"]
        phrases = {x["phrase"] for x in t["top_bigrams"]}
        # Stop words fall away without replacement: "Motor runs very quietly" → "runs quietly"
        assert "motor runs" in phrases

    def test_the_sample_is_declared(self, db):
        t = compute("", with_text=True)["text"]
        assert t["sample"] == 2  # only t1 has segments

    def test_the_text_follows_the_scope(self, db):
        """Series2 has no segments — the analysis must be empty, not global."""
        t = compute("Series2", with_text=True)["text"]
        assert t["sample"] == 0
        assert t["top_words"] == []


# ── Scope: whole library ─────────────────────────────────────────────────


class TestLibrary:
    """The explicit exception: every root in the database, not just the open one."""

    def test_counts_every_root(self, db):
        r = compute("", library=True)
        assert "error" not in r
        assert r["scope"]["library"] is True
        assert r["scope"]["label"] == "Whole library"
        assert r["transcripts"]["total"] == 6  # 5 active + 1 foreign
        assert r["progress"]["total"] == 7
        assert r["media"]["size"] == 1000 + 2000 + 4000 + 500 + 8000 + 100 + 777000

    def test_the_path_is_ignored(self, db):
        r = compute("Series/S1", library=True)
        assert r["scope"]["path"] == ""
        assert r["transcripts"]["total"] == 6

    def test_no_subfolder_table(self, db):
        """Drilling down is relative to the open root — meaningless across roots."""
        assert compute("", library=True)["subfolders"] == []

    def test_records_show_the_absolute_path(self, db):
        rec = {r["key"]: r for r in compute("", library=True)["records"]}
        assert rec["longest"]["path"] == f"{FOREIGN}/Series/S1/a.mp4"

    def test_rows_without_identity_stay_out(self, db):
        with db.connection() as conn:
            _transcript_record(conn, "old.mp4", abs_path="")
        assert compute("", library=True)["transcripts"]["total"] == 6

    def test_the_path_scope_is_unchanged(self, db):
        r = compute("")
        assert r["scope"]["library"] is False
        assert r["transcripts"]["total"] == 5


# ── HTTP endpoint ─────────────────────────────────────────────────────────


class TestEndpoint:
    def test_global(self, client):
        r = client.get("/api/stats")
        assert r.status_code == 200
        assert r.get_json()["scope"]["global"] is True

    def test_directory_scope(self, client):
        d = client.get("/api/stats?path=Series/S1").get_json()
        assert d["scope"]["path"] == "Series/S1"
        assert d["transcripts"]["total"] == 2

    def test_text_flag(self, client):
        assert "text" not in client.get("/api/stats").get_json()
        assert "text" in client.get("/api/stats?text=1").get_json()

    @pytest.mark.parametrize(
        "path",
        [
            "../../etc/passwd",
            "..",
            "%",
            "_",
            "'; DROP TABLE transcripts; --",
            "Series/../../..",
            "a" * 5000,
        ],
    )
    def test_no_five_hundreds(self, client, path):
        """Useless paths must return empty, not with a stack trace."""
        r = client.get("/api/stats", query_string={"path": path})
        assert r.status_code == 200, f"{path!r} → {r.status_code}"
        assert "error" not in r.get_json()

    def test_traversal_delivers_no_foreign_numbers(self, client):
        """An escape attempt must not the root above it show."""
        d = client.get("/api/stats", query_string={"path": "../elsewhere"}).get_json()
        assert d["transcripts"]["total"] == 0

    def test_injection_leaves_the_table_intact(self, client, db):
        client.get("/api/stats", query_string={"path": "'; DROP TABLE transcripts; --"})
        with db.connection() as conn:
            n = conn.execute("SELECT COUNT(*) AS n FROM transcripts").fetchone()["n"]
        assert n == 6

    def test_the_cache_serves_again_and_can_be_cleared(self, client, db):
        assert client.get("/api/stats").get_json()["transcripts"]["total"] == 5
        with db.connection() as conn:
            _transcript_record(conn, "Series/S1/new_value.mp4", words=1)

        # Within the TTL the old state stays — that's intentional.
        assert client.get("/api/stats").get_json()["transcripts"]["total"] == 5

        clear_cache()
        assert client.get("/api/stats").get_json()["transcripts"]["total"] == 6

    def test_the_cache_separates_scopes(self, client):
        """The directory call must not recycle the global response."""
        g = client.get("/api/stats").get_json()
        v = client.get("/api/stats?path=Series/S1").get_json()
        assert g["transcripts"]["total"] == 5
        assert v["transcripts"]["total"] == 2

    def test_library_scope(self, client):
        d = client.get("/api/stats?scope=library").get_json()
        assert d["scope"]["library"] is True
        assert d["transcripts"]["total"] == 6

    def test_the_cache_separates_library_and_root(self, client):
        assert client.get("/api/stats").get_json()["transcripts"]["total"] == 5
        assert client.get("/api/stats?scope=library").get_json()["transcripts"]["total"] == 6
        assert client.get("/api/stats").get_json()["transcripts"]["total"] == 5

    def test_the_cache_separates_roots(self, client, monkeypatch):
        """After a root change, the old response must not come back."""
        from whisper_tm.config import config

        assert client.get("/api/stats").get_json()["transcripts"]["total"] == 5
        monkeypatch.setattr(type(config), "media_dir", property(lambda self: FOREIGN))
        assert client.get("/api/stats").get_json()["transcripts"]["total"] == 1

    def test_without_a_database(self, monkeypatch):
        monkeypatch.setattr("whisper_tm.api.stats.db", None)
        app = Flask(__name__)
        app.register_blueprint(stats_bp)
        r = app.test_client().get("/api/stats")
        assert r.status_code == 501


# ── Empty library ─────────────────────────────────────────────────────────


class TestEmpty:
    def test_all_zero_instead_of_a_crash(self, tmp_path, monkeypatch, root):
        d = Database(tmp_path / "empty.db")
        d.initialize()
        monkeypatch.setattr("whisper_tm.api.stats.db", d)
        r = compute("", with_text=True)
        assert "error" not in r
        assert r["transcripts"]["total"] == 0
        assert r["transcripts"]["duration"] == 0
        assert r["transcripts"]["speed"] == 0.0
        assert r["progress"]["pct"] == 0.0
        assert r["subfolders"] == []
        assert r["records"] == []
        assert r["text"]["top_words"] == []
        assert len(r["duration_buckets"]) == 4


def test_words_of_every_script():
    """[a-zA-ZÀ-ž] gave a Russian, Greek or Hindi library no word at all,
    and letters alone cut Hindi words at their vowel signs. The file view's
    statistics count the same way (stats.js)."""
    from pathlib import Path

    from whisper_tm.core.stopwords import words

    assert words("Привет мир, नमस्ते दुनिया, Καλημέρα, Grüße, 日本語", set()) == [
        "привет",
        "мир",
        "नमस्ते",
        "दुनिया",
        "καλημέρα",
        "grüße",
        "日本語",
    ]
    js = (
        Path(__file__).resolve().parent.parent / "whisper_tm" / "static" / "js" / "stats.js"
    ).read_text(encoding="utf-8")
    assert "/[\\p{L}\\p{M}]+/gu" in js and "match(/[a-zß-ž]+/g)" not in js
    # One word count on the page: the card above and the words below
    assert "${fmtNumber(s.words)}</b> word${s.words === 1 ? '' : 's'}" in js


# ── Any folder of the library (the Library page's tree) ──────────────────


class TestFolderScope:
    def test_a_folder_outside_the_open_one(self, client, monkeypatch):
        import whisper_tm.core.security as sec

        monkeypatch.setattr(sec, "root_allowed", lambda p: str(p).startswith("/media/"))
        # The open folder is ROOT; the foreign tree is still a library folder
        d = client.get(f"/api/stats?folder={FOREIGN}/Series").get_json()
        assert d["progress"]["total"] == 1 and d["transcripts"]["duration"] == 9999
        # and a folder inside the open root, given absolutely, is its own root
        d = client.get(f"/api/stats?folder={ROOT}/Series").get_json()
        assert d["progress"]["total"] == 4
        assert {f["name"] for f in d.get("folders", [])} >= {"S1", "S2"} or "subfolders" in d

    def test_a_folder_outside_the_allowed_trees_is_refused(self, client, monkeypatch):
        import whisper_tm.core.security as sec

        monkeypatch.setattr(sec, "root_allowed", lambda p: False)
        assert client.get("/api/stats?folder=/etc").status_code == 403
