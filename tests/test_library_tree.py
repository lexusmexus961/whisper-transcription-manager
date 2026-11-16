"""The Library's folder tree (/api/library/tree): counts per folder with
its subfolders, one level per request, cached, and cheap on a big library."""

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
    with d.connection() as c:
        for rel, status in (
            ("a/1.mp3", "done"),
            ("a/2.mp3", "pending"),
            ("a/b/3.mp3", "done"),
            ("a/b/4.mp3", "pending"),
            ("c/5.mp3", "skipped"),
            ("6.mp3", "done"),
        ):
            c.execute(
                "INSERT INTO media_files (rel_path, abs_path, status, file_size, modified_at)"
                " VALUES (?, ?, ?, 1, 1)",
                (rel, f"{ROOT}/{rel}", status),
            )
        c.execute(
            "INSERT INTO media_files (rel_path, abs_path, status, file_size, modified_at)"
            " VALUES ('x.mp3', '/elsewhere/x.mp3', 'done', 1, 1)"
        )
        # waiting: a/2; failed then waiting again: a/b/4 (counts as waiting); failed: c/5
        for fn, st in (
            ("a/2.mp3", "pending"),
            ("a/b/4.mp3", "error"),
            ("a/b/4.mp3", "pending"),
            ("c/5.mp3", "error"),
        ):
            c.execute(
                "INSERT INTO queue_items (filename, base_dir, status) VALUES (?, ?, ?)",
                (fn, ROOT, st),
            )
    return d


def test_counts_include_the_subfolders(db):
    from whisper_tm.api import library

    t = library.build(db, ROOT)
    c = t["counts"]
    assert c[ROOT] == {"files": 6, "done": 3, "open": 2, "failed": 1, "queued": 2}
    assert c[ROOT + "/a"] == {"files": 4, "done": 2, "open": 2, "failed": 0, "queued": 2}
    assert c[ROOT + "/a/b"] == {"files": 2, "done": 1, "open": 1, "failed": 0, "queued": 1}
    assert c[ROOT + "/c"] == {"files": 1, "done": 0, "open": 0, "failed": 1, "queued": 0}
    assert t["children"][ROOT] == ["a", "c"] and t["children"][ROOT + "/a"] == ["b"]
    assert "/elsewhere" not in c


@pytest.fixture
def client(db, monkeypatch):
    import whisper_tm.core.security as sec
    from whisper_tm import extensions as ext
    from whisper_tm.api import library

    monkeypatch.setattr(ext, "db", db)
    monkeypatch.setattr(sec, "allowed_roots", lambda: [ROOT])
    library.clear_cache()
    app = Flask(__name__)
    app.register_blueprint(library.library_bp)
    return app.test_client()


def test_one_level_per_request(client):
    d = client.get("/api/library/tree").get_json()
    assert d["path"] == "" and d["counts"]["files"] == 6
    assert [(x["name"], x["path"], x["has_children"]) for x in d["children"]] == [
        ("a", "a", True),
        ("c", "c", False),
    ]
    d = client.get("/api/library/tree?path=a").get_json()
    assert [x["path"] for x in d["children"]] == ["a/b"]
    assert d["crumbs"] == [{"name": "media", "path": ""}, {"name": "a", "path": "a"}]


def test_outside_or_unknown_folders_are_refused(client):
    assert client.get("/api/library/tree?path=../x").status_code == 400
    assert client.get("/api/library/tree?path=nope").status_code == 404


def test_watched_folders_and_their_subfolders(client, monkeypatch):
    from pathlib import Path

    from whisper_tm.core import autoscan

    monkeypatch.setattr(autoscan, "setting", lambda name: True if name == "enabled" else [])
    monkeypatch.setattr(autoscan, "watched_folders", lambda: [Path(ROOT + "/a")])
    d = client.get("/api/library/tree").get_json()
    assert {x["path"]: x["watched"] for x in d["children"]} == {"a": "self", "c": ""}
    d = client.get("/api/library/tree?path=a").get_json()
    assert d["watched"] == "self" and d["children"][0]["watched"] == "parent"
    # the whole library watched: the root itself, everything below through it
    monkeypatch.setattr(autoscan, "watched_folders", lambda: [Path(ROOT)])
    d = client.get("/api/library/tree").get_json()
    assert d["watched"] == "self" and {x["watched"] for x in d["children"]} == {"parent"}
    # off: nothing is watched
    monkeypatch.setattr(autoscan, "setting", lambda name: False)
    d = client.get("/api/library/tree").get_json()
    assert d["watched"] == "" and {x["watched"] for x in d["children"]} == {""}


def test_a_big_library_is_cheap_and_cached(tmp_path):
    from whisper_tm.api import library

    d = Database(tmp_path / "big.db")
    d.initialize()
    with d.connection() as c:
        c.executemany(
            "INSERT INTO media_files (rel_path, abs_path, status, file_size, modified_at) VALUES (?, ?, ?, 1, 1)",
            (
                (
                    f"s{i % 40}/e{i % 400}/f{i}.mp4",
                    f"{ROOT}/s{i % 40}/e{i % 400}/f{i}.mp4",
                    "done" if i % 3 else "pending",
                )
                for i in range(200_000)
            ),
        )
    library.clear_cache()
    t0 = time.perf_counter()
    t = library.tree(d, ROOT)
    first = time.perf_counter() - t0
    t0 = time.perf_counter()
    library.tree(d, ROOT)
    again = time.perf_counter() - t0
    assert t["counts"][ROOT]["files"] == 200_000
    assert first < 5.0, f"first build took {first:.2f} s"
    assert again < 0.01, "the second call is served from the cache"


# ── Who speaks the most ──────────────────────────────────────────────────


def test_who_speaks_the_most_counts_like_a_recordings_speakers_tab(tmp_path):
    from whisper_tm.api import library
    from whisper_tm.database.speaker import SpeakerDB

    d = Database(tmp_path / "w.db")
    d.initialize()
    s = SpeakerDB(tmp_path / "s.db")
    s.initialize()
    with d.connection() as c:
        for rel, segs in (
            ("a/1.mp3", [("SPEAKER_00", 0, 30), ("SPEAKER_01", 30, 40), ("SPEAKER_00", 40, 50)]),
            ("a/2.mp3", [("SPEAKER_00", 0, 20)]),
            ("b/3.mp3", [("SPEAKER_00", 0, 100)]),
        ):
            tid = c.execute(
                "INSERT INTO transcripts (rel_path, abs_path, complete, updated_at) VALUES (?, ?, 1, 1)",
                (rel, f"{ROOT}/{rel}"),
            ).lastrowid
            for i, (sp, a, b) in enumerate(segs):
                c.execute(
                    "INSERT INTO segments (transcript_id, seg_index, speaker, start_sec, end_sec, text)"
                    " VALUES (?, ?, ?, ?, ?, 'x')",
                    (tid, i, sp, a, b),
                )
    with s.connection() as c:
        people = {
            n: c.execute("INSERT INTO speaker_people (name) VALUES (?)", (n,)).lastrowid
            for n in ("Alice", "Bob")
        }
        counter = iter(range(1000))

        def voice(rel, label, person, n=1):
            for _ in range(n):
                vid = c.execute(
                    "INSERT INTO speaker_vectors (abs_path, label, windows, start_sec, speech_sec, model)"
                    " VALUES (?, ?, ?, 0, 30, 'm')",
                    (f"{ROOT}/{rel}", label, next(counter)),
                ).lastrowid
                c.execute(
                    "INSERT INTO speaker_assignment (vector_id, person_id) VALUES (?, ?)",
                    (vid, people[person]),
                )

        voice("a/1.mp3", "SPEAKER_00", "Alice", 2)
        voice("a/1.mp3", "SPEAKER_00", "Bob", 1)  # outvoted: the voice is Alice's
        voice("a/1.mp3", "SPEAKER_01", "Bob")
        voice("a/2.mp3", "SPEAKER_00", "Bob")
        voice("b/3.mp3", "SPEAKER_00", "Bob")
    whole = library.speakers_in(d, s, ROOT)
    assert whole == [
        {"name": "Bob", "speech_sec": 130.0, "recordings": 3},
        {"name": "Alice", "speech_sec": 40.0, "recordings": 1},
    ]
    assert library.speakers_in(d, s, ROOT + "/a") == [
        {"name": "Alice", "speech_sec": 40.0, "recordings": 1},
        {"name": "Bob", "speech_sec": 30.0, "recordings": 2},
    ]
