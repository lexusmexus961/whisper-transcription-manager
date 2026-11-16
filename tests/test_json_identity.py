"""
Which identity a discovered transcript JSON gets.

The scanner finds ``.transcripts/`` directories throughout the tree and takes
their JSONs into the database. The question each time is: which media file does
this JSON belong to?

Previously, the ``filename`` field in the JSON was considered authoritative.
It is not: it was written during transcription relative to the THEN ``media_dir``,
and that is switchable at runtime. Whoever had a subdirectory open back then has
just a file name there —

    JSON  /nas/videos/recordings/.transcripts/clip.mp4.json
    field  "filename": "clip.mp4"
    → abs_from("clip.mp4", "/nas") = "/nas/clip.mp4"        does not exist

An index run over a large library thus produced many rows with an identity
under which nothing lies. Now what matters is the storage location: a
``.transcripts/x.mp4.json`` belongs to ``../x.mp4``.
"""

from __future__ import annotations

import json

import pytest

from whisper_tm.database import Database


@pytest.fixture
def environment(tmp_path, monkeypatch):
    from whisper_tm.config import config

    media = tmp_path / "media"
    low = media / "videos" / "recordings"
    low.mkdir(parents=True)
    (low / "clip.mp4").write_bytes(b"x")
    # same-name file directly in the root — the confusion partner
    (media / "clip.mp4").write_bytes(b"y")

    monkeypatch.setattr(type(config), "media_dir", property(lambda self: media))
    monkeypatch.setattr(type(config), "transcript_mode", property(lambda self: "alongside"))
    db = Database(tmp_path / "t.db")
    db.initialize()
    return {"media": media, "low": low, "db": db}


def _store_json(folder, name, filename_field, text="content"):
    td = folder / ".transcripts"
    td.mkdir(exist_ok=True)
    (td / f"{name}.json").write_text(
        json.dumps(
            {
                "complete": True,
                "filename": filename_field,
                "segments": [{"start": 0, "end": 1, "text": text}],
            }
        ),
        encoding="utf-8",
    )
    return td


class TestStorageLocationDecides:
    def test_a_stale_filename_field_is_not_believed(self, environment):
        """The real failure case: field is a basename from an old root."""
        td = _store_json(environment["low"], "clip.mp4", "clip.mp4")
        environment["db"].sync_from_json_dir(td, media_dir=str(environment["media"]))

        with environment["db"].connection() as c:
            rows = c.execute("SELECT abs_path FROM transcripts").fetchall()
        assert len(rows) == 1
        assert rows[0]["abs_path"] == str(environment["low"] / "clip.mp4"), (
            "identity from the storage location, not from the filename field"
        )

    def test_the_identity_points_at_an_existing_file(self, environment):
        import os

        td = _store_json(environment["low"], "clip.mp4", "clip.mp4")
        environment["db"].sync_from_json_dir(td, media_dir=str(environment["media"]))

        with environment["db"].connection() as c:
            ap = c.execute("SELECT abs_path FROM transcripts").fetchone()["abs_path"]
        assert os.path.exists(ap), f"orphaned identity: {ap}"

    def test_a_misleading_field_from_a_foreign_folder(self, environment):
        """Even a field that points at ANOTHER existing file must not
        override the storage location."""
        td = _store_json(
            environment["low"], "clip.mp4", filename_field="clip.mp4", text="belongs to low"
        )
        environment["db"].sync_from_json_dir(td, media_dir=str(environment["media"]))

        with environment["db"].connection() as c:
            ap = c.execute("SELECT abs_path FROM transcripts").fetchone()["abs_path"]
        assert ap != str(environment["media"] / "clip.mp4"), (
            "attached to the same-named file in the root"
        )


class TestCentralFolder:
    def test_the_field_still_counts_when_it_resolves(self, environment):
        """In the central collection folder, the storage location reveals
        nothing — there the filename field stays the source, as long as
        it points at a real file."""
        central = environment["media"] / ".transcripts"
        central.mkdir(exist_ok=True)
        (central / "videos__recordings__clip.mp4.json").write_text(
            json.dumps(
                {
                    "complete": True,
                    "filename": "videos/recordings/clip.mp4",
                    "segments": [{"start": 0, "end": 1, "text": "x"}],
                }
            ),
            encoding="utf-8",
        )

        environment["db"].sync_from_json_dir(central, media_dir=str(environment["media"]))
        with environment["db"].connection() as c:
            ap = c.execute("SELECT abs_path FROM transcripts").fetchone()["abs_path"]
        assert ap == str(environment["low"] / "clip.mp4")

    def test_the_field_is_taken_without_an_existence_check(self, environment):
        """In the central folder, the field is NOT checked against disk.

        Deliberate trade-off: an existence check would, with a briefly
        unreachable NAS, cause transcripts to never even make it into the
        database — bigger damage than a row that temporarily points into
        the void. `tools/repair_identities.py` cleans up such rows.

        The orphaned identities above did not come from here, but from this
        branch running BEFORE location resolution (see the class above).
        """
        central = environment["media"] / ".transcripts"
        central.mkdir(exist_ok=True)
        (central / "does_not_exist.mp4.json").write_text(
            json.dumps(
                {
                    "complete": True,
                    "filename": "does/not/exist.mp4",
                    "segments": [{"start": 0, "end": 1, "text": "x"}],
                }
            ),
            encoding="utf-8",
        )

        environment["db"].sync_from_json_dir(central, media_dir=str(environment["media"]))
        with environment["db"].connection() as c:
            row = c.execute("SELECT abs_path FROM transcripts").fetchone()
        assert row["abs_path"] == str(environment["media"] / "does/not/exist.mp4")
