"""Search for text, but only where a specific person speaks.

The large search can be narrowed to the recordings of a selected speaker,
e.g. search for "x" only where Alice speaks.

**This is not the Speaker filter that already existed.** That filters on
`segments.speaker` — the diarization identifier (`SPEAKER_00`), which only
applies WITHIN a recording; in the next file, `SPEAKER_00` is someone
else. A person, on the other hand, is recognized across the entire collection,
and their positions are stored in a different database
(`speaker.db`). That's why they are looked up there and passed as a list,
rather than joined in the query.

**Two levels of precision**, because both questions occur:

* *"the recordings where they appear"* — the main use. There every word
  counts, even others': a common word only hits in the recordings the
  person appears in, instead of across the whole library.
* *"only their sentences"* — far fewer hits. For that, identifier and
  `segments.speaker` match, both are `SPEAKER_00`.
"""

import pytest

from tests.test_search import _seg, _tx  # noqa: F401
from whisper_tm.database import Database
from whisper_tm.database.speaker import SpeakerDB


@pytest.fixture
def db(tmp_path):
    d = Database(tmp_path / "test.db")
    d.initialize()
    # three recordings, same word in all — speakers differ
    d.upsert_transcript(
        "with_person.mp4",
        _tx(
            "with_person.mp4",
            [
                _seg("SPEAKER_00", 0.0, "The cake was good."),
                _seg("SPEAKER_01", 5.0, "The cake was bad."),
            ],
            100.0,
        ),
    )
    d.upsert_transcript(
        "also_with.mp4",
        _tx(
            "also_with.mp4",
            [
                _seg("SPEAKER_03", 0.0, "Another cake."),
            ],
            100.0,
        ),
    )
    d.upsert_transcript(
        "without_person.mp4",
        _tx(
            "without_person.mp4",
            [
                _seg("SPEAKER_00", 0.0, "Cake without them."),
            ],
            100.0,
        ),
    )
    return d


def _path(db, name):
    """Address the recording as the speaker DB knows it."""
    with db.connection() as c:
        r = c.execute("SELECT abs_path FROM transcripts WHERE rel_path = ?", (name,)).fetchone()
    return r["abs_path"]


class TestTheNarrowing:
    def test_without_filter_all_found(self, db):
        assert db.search("cake")["total"] == 4

    def test_only_person_recordings(self, db):
        """The main use: everything from recordings they appear in
        — even sentences from others."""
        d = db.search(
            "cake",
            filter_opts={
                "person_spots": [
                    (_path(db, "with_person.mp4"), "SPEAKER_00"),
                    (_path(db, "also_with.mp4"), "SPEAKER_03"),
                ]
            },
        )
        assert d["total"] == 3, [r["filename"] for r in d["results"]]
        assert {r["filename"] for r in d["results"]} == {"with_person.mp4", "also_with.mp4"}

    def test_only_their_sentences(self, db):
        d = db.search(
            "cake",
            filter_opts={
                "person_spots": [
                    (_path(db, "with_person.mp4"), "SPEAKER_00"),
                    (_path(db, "also_with.mp4"), "SPEAKER_03"),
                ],
                "person_segments_only": True,
            },
        )
        assert d["total"] == 2
        assert {r["speaker"] for r in d["results"]} == {"SPEAKER_00", "SPEAKER_03"}

    def test_empty_list_finds_nothing(self, db):
        """If you search for a person with no voice assigned yet,
        you should see zero hits — not the whole library."""
        assert db.search("cake", filter_opts={"person_spots": []})["total"] == 0

    def test_facets_count_same(self, db):
        """Otherwise sidebar shows different numbers than hit list."""
        fo = {"person_spots": [(_path(db, "with_person.mp4"), "SPEAKER_00")]}
        assert (
            db.search_facets("cake", filter_opts=fo)["total"]
            == db.search("cake", filter_opts=fo)["total"]
            == 2
        )

    def test_filter_does_not_persist(self, db):
        """TEMP table lives on connection — a later search without
        filter must not get anything from it."""
        db.search(
            "cake", filter_opts={"person_spots": [(_path(db, "with_person.mp4"), "SPEAKER_00")]}
        )
        assert db.search("cake")["total"] == 4


class TestThePositionsFromSpeakerDb:
    @pytest.fixture
    def sdb(self, tmp_path):
        import numpy as np

        d = SpeakerDB(tmp_path / "speaker.db")
        d.initialize()
        v = np.ones(192, dtype=np.float32)
        d.write(
            "/media/a.mp4",
            "redimnet_m",
            [
                {
                    "label": "SPEAKER_00",
                    "windows": 0,
                    "start_sec": 0.0,
                    "speech_sec": 30.0,
                    "segments": 3,
                    "v_separated": v,
                },
                {
                    "label": "SPEAKER_01",
                    "windows": 1,
                    "start_sec": 30.0,
                    "speech_sec": 30.0,
                    "segments": 3,
                    "v_separated": v,
                },
            ],
        )
        d.write(
            "/media/b.mp4",
            "redimnet_m",
            [
                {
                    "label": "SPEAKER_02",
                    "windows": 0,
                    "start_sec": 0.0,
                    "speech_sec": 30.0,
                    "segments": 3,
                    "v_separated": v,
                }
            ],
        )
        return d

    def test_they_come_as_pairs(self, sdb):
        sdb.assign_many([1, 3], "Alice")
        spots = set(sdb.person_spots("Alice"))
        assert ("/media/a.mp4", "SPEAKER_00") in spots
        assert ("/media/b.mp4", "SPEAKER_02") in spots
        assert ("/media/a.mp4", "SPEAKER_01") not in spots, "a foreign label came along"

    def test_upper_and_lower_case_do_not_matter(self, sdb):
        sdb.assign_many([1], "Alice")
        assert sdb.person_spots("alice") == sdb.person_spots("Alice")

    def test_an_unknown_person_returns_empty(self, sdb):
        assert sdb.person_spots("DoesNotExist") == []


class TestThePathThroughTheApi:
    """From `?person=Alice` to the query — the spot where the second
    database comes into play."""

    @pytest.fixture
    def client(self, tmp_path, monkeypatch, db):
        import numpy as np
        from flask import Flask

        from whisper_tm.config import config

        sdb = SpeakerDB(tmp_path / "speaker.db")
        sdb.initialize()
        sdb.write(
            _path(db, "with_person.mp4"),
            "redimnet_m",
            [
                {
                    "label": "SPEAKER_00",
                    "windows": 0,
                    "start_sec": 0.0,
                    "speech_sec": 30.0,
                    "segments": 3,
                    "v_separated": np.ones(192, dtype=np.float32),
                }
            ],
        )
        sdb.assign_many([1], "Alice")
        monkeypatch.setattr("whisper_tm.database.speaker.speaker_db", lambda: sdb)
        monkeypatch.setattr("whisper_tm.api.search.db", db)
        monkeypatch.setattr(type(config), "media_dir", property(lambda self: tmp_path / "media"))

        from whisper_tm.api.search import search_bp

        a = Flask(__name__)
        a.register_blueprint(search_bp)
        return a.test_client()

    def test_without_a_person_the_search_finds_everything(self, client):
        d = client.get("/api/search?q=cake").get_json()
        assert d["total"] == 4

    def test_with_a_person_only_their_recordings(self, client):
        d = client.get("/api/search?q=cake&person=Alice").get_json()
        assert d["total"] == 2
        assert {r["filename"] for r in d["results"]} == {"with_person.mp4"}

    def test_with_a_person_and_segments(self, client):
        d = client.get("/api/search?q=cake&person=Alice&person_segments=1").get_json()
        assert d["total"] == 1
        assert d["results"][0]["speaker"] == "SPEAKER_00"

    def test_an_unknown_person_finds_nothing(self, client):
        """Not EVERYTHING — that would be the silent variant of "filter ignored"."""
        d = client.get("/api/search?q=cake&person=Nobody").get_json()
        assert d["total"] == 0
