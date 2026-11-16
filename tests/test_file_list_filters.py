"""The file list's filter panel (P3.02): type, length, added, language,
has (summary, translation, warnings, transcript), subfolder and people —
in get_media_files_page, get_media_names_under and /api/files."""

from __future__ import annotations

import time

import pytest
from flask import Flask

from whisper_tm.database import Database

ROOT = "/lib/media"


@pytest.fixture
def db(tmp_path):
    d = Database(tmp_path / "test.db")
    d.initialize()
    now = time.time()
    files = [
        # rel, video, seconds, added days ago, status
        ("talks/a.mp4", 1, 3600, 1, "done"),
        ("talks/b.mp3", 0, 600, 40, "done"),
        ("talks/deep/c.m4a", 0, 120, 2, "pending"),
        ("memos/d.wav", 0, 30, 0.1, "done"),
        ("memos/e.mkv", 1, 7200, 100, "pending"),
    ]
    with d.connection() as c:
        for rel, video, sec, days, status in files:
            c.execute(
                "INSERT INTO media_files (rel_path, abs_path, is_video, duration_sec, created_at,"
                " status, file_size, modified_at) VALUES (?, ?, ?, ?, ?, ?, 1, 1)",
                (rel, f"{ROOT}/{rel}", video, sec, now - days * 86400, status),
            )
        for rel, lang, warn in (
            ("talks/a.mp4", "en", 0),
            ("talks/b.mp3", "de", 2),
            ("memos/d.wav", "en", 0),
        ):
            c.execute(
                "INSERT INTO transcripts (rel_path, abs_path, complete, language, warning_count,"
                " word_count, updated_at) VALUES (?, ?, 1, ?, ?, 10, 1)",
                (rel, f"{ROOT}/{rel}", lang, warn),
            )
        c.execute(
            "INSERT INTO summaries (rel_path, abs_path, text) VALUES ('talks/a.mp4', ?, 'x')",
            (f"{ROOT}/talks/a.mp4",),
        )
        c.execute(
            "INSERT INTO translations (rel_path, abs_path, language, segments_json)"
            " VALUES ('memos/d.wav', ?, 'de', '[]')",
            (f"{ROOT}/memos/d.wav",),
        )
    return d


def names(db, **filters):
    page = db.get_media_files_page(ROOT, filters=filters or None)
    got = sorted(r["rel_path"] for r in page["rows"])
    assert page["total"] == len(got)
    # the "select all" list agrees with the page
    assert sorted(db.get_media_names_under(ROOT, filters=filters or None)) == got
    return got


def test_no_filter_lists_everything(db):
    assert len(names(db)) == 5


@pytest.mark.parametrize(
    "filters, expected",
    [
        ({"type": "video"}, ["memos/e.mkv", "talks/a.mp4"]),
        ({"type": "audio"}, ["memos/d.wav", "talks/b.mp3", "talks/deep/c.m4a"]),
        ({"len_min": 3600}, ["memos/e.mkv", "talks/a.mp4"]),
        ({"len_max": 120}, ["memos/d.wav", "talks/deep/c.m4a"]),
        ({"len_min": 100, "len_max": 1000}, ["talks/b.mp3", "talks/deep/c.m4a"]),
        ({"added_days": 7}, ["memos/d.wav", "talks/a.mp4", "talks/deep/c.m4a"]),
        ({"language": "en"}, ["memos/d.wav", "talks/a.mp4"]),
        ({"has": ["summary"]}, ["talks/a.mp4"]),
        ({"has": ["translation"]}, ["memos/d.wav"]),
        ({"has": ["warnings"]}, ["talks/b.mp3"]),
        ({"has": ["transcript"]}, ["memos/d.wav", "talks/a.mp4", "talks/b.mp3"]),
        ({"subfolder": "talks"}, ["talks/a.mp4", "talks/b.mp3", "talks/deep/c.m4a"]),
        ({"subfolder": "talks/deep/"}, ["talks/deep/c.m4a"]),
        (
            {"people": ["Alice"], "people_paths": [f"{ROOT}/talks/b.mp3", "/elsewhere/x.mp3"]},
            ["talks/b.mp3"],
        ),
        ({"people": ["Nobody"], "people_paths": []}, []),
    ],
)
def test_each_filter(db, filters, expected):
    assert names(db, **filters) == expected


def test_filters_combine_with_and(db):
    assert names(db, type="audio", added_days=7, has=["transcript"]) == ["memos/d.wav"]


def test_filters_combine_with_the_status_tab_and_the_name_filter(db):
    page = db.get_media_files_page(ROOT, tab="done", name_filter="talks", filters={"type": "audio"})
    assert [r["rel_path"] for r in page["rows"]] == ["talks/b.mp3"]


def test_a_subfolder_cannot_climb_out(db):
    assert len(names(db, subfolder="../media")) == 5, "a '..' path is ignored, not followed"


def test_paging_counts_the_filtered_total(db):
    page = db.get_media_files_page(ROOT, limit=1, offset=1, filters={"type": "audio"})
    assert page["total"] == 3 and len(page["rows"]) == 1


# ── /api/files parses them ──────────────────────────────────────────────


@pytest.fixture
def client(db, monkeypatch):
    from whisper_tm import extensions as ext
    from whisper_tm.api import files as api
    from whisper_tm.config import config

    monkeypatch.setattr(ext, "db", db)
    monkeypatch.setattr(api, "db", db)
    monkeypatch.setattr(type(config), "media_dir", property(lambda self: ROOT))

    class FakeSpeakers:
        def person_spots(self, name):
            return [(f"{ROOT}/talks/a.mp4", "SPEAKER_00")] if name == "Alice" else []

    import whisper_tm.database.speaker as sp

    monkeypatch.setattr(sp, "speaker_db", lambda: FakeSpeakers())
    app = Flask(__name__)
    app.register_blueprint(api.files_bp)
    return app.test_client()


def test_the_api_takes_the_panel_parameters(client):
    def got(q):
        d = client.get("/api/files?" + q).get_json()
        return sorted(f["name"] for f in d["files"])

    assert got("f_type=video&f_len_min=3600") == ["memos/e.mkv", "talks/a.mp4"]
    assert got("f_has=summary,bogus") == ["talks/a.mp4"]
    assert got("f_sub=memos&f_lang=en") == ["memos/d.wav"]
    assert got("f_people=Alice") == ["talks/a.mp4"]
    assert got("f_added=7&tab=pending") == ["talks/deep/c.m4a"]


def test_a_value_that_does_not_parse_switches_only_that_filter_off(client):
    d = client.get("/api/files?f_len_min=abc&f_type=video").get_json()
    assert sorted(f["name"] for f in d["files"]) == ["memos/e.mkv", "talks/a.mp4"]
    names_ = client.get("/api/files/names?f_type=video").get_json()["names"]
    assert sorted(names_) == ["memos/e.mkv", "talks/a.mp4"]


def test_files_not_indexed_yet_go_by_extension_and_the_transcript_length(tmp_path):
    d = Database(tmp_path / "raw.db")
    d.initialize()
    with d.connection() as c:
        for rel in ("clip.MP4", "talk.m4a", "known.mp4"):
            c.execute(
                "INSERT INTO media_files (rel_path, abs_path, status, file_size, modified_at,"
                " is_video, duration_sec, indexed_at) VALUES (?, ?, 'done', 1, 1, 0, 0, ?)",
                (rel, f"{ROOT}/{rel}", 5 if rel == "known.mp4" else 0),
            )
        c.execute(
            "INSERT INTO transcripts (rel_path, abs_path, complete, duration_sec, updated_at)"
            " VALUES ('talk.m4a', ?, 1, 900, 1)",
            (f"{ROOT}/talk.m4a",),
        )

    def got(**f):
        return sorted(r["rel_path"] for r in d.get_media_files_page(ROOT, filters=f)["rows"])

    # known.mp4 was indexed and holds no video (an audio-only container)
    assert got(type="video") == ["clip.MP4"]
    assert got(type="audio") == ["known.mp4", "talk.m4a"]
    assert got(len_min=600) == ["talk.m4a"]


def test_what_needs_attention_can_be_listed(tmp_path):
    d = Database(tmp_path / "att.db")
    d.initialize()
    with d.connection() as c:
        for rel, st, why in (
            ("mute.mp4", "skipped", "no_audio"),
            ("later.mp3", "skipped", "manual"),
            ("quiet.wav", "done", ""),
            ("talk.mp3", "done", ""),
        ):
            c.execute(
                "INSERT INTO media_files (rel_path, abs_path, status, skip_reason, file_size, modified_at)"
                " VALUES (?, ?, ?, ?, 1, 1)",
                (rel, f"{ROOT}/{rel}", st, why),
            )
        for rel, words in (("quiet.wav", 0), ("talk.mp3", 12)):
            c.execute(
                "INSERT INTO transcripts (rel_path, abs_path, complete, word_count, updated_at) VALUES (?, ?, 1, ?, 1)",
                (rel, f"{ROOT}/{rel}", words),
            )

    def got(h):
        rows = d.get_media_files_page(ROOT, filters={"has": [h]})["rows"]
        return sorted(r["rel_path"] for r in rows)

    assert got("no_audio") == ["mute.mp4"]
    assert got("skipped") == ["later.mp3"]
    assert got("empty") == ["quiet.wav"]
    assert sorted(d.get_media_names_under(ROOT, filters={"has": ["empty"]})) == ["quiet.wav"]


def test_a_length_of_zero_is_a_filter_like_any_other(client):
    # 0 counted as "not set" when the query was built: no join, and the
    # filter's own column was missing — a 500
    for query in ("f_len_min=0", "f_len_max=0", "f_len_min=0&f_len_max=100000"):
        r = client.get("/api/files?" + query)
        assert r.status_code == 200, (query, r.get_data(as_text=True)[:200])
