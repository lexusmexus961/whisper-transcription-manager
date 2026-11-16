"""Central transcript names are one per recording.

The flat name of ``transcript_mode: central`` turned "/" into
"__" and ":" into "_" — not reversible. ``/m/a/b__c.mp4`` and
``/m/a__b/c.mp4`` both became ``m__a__b__c.mp4.json``, and the second
transcript overwrote the first. A NEW transcript of a path whose flat name
is ambiguous gets a hash of the path; every other name stays as it was.

A transcript already written under the old flat name must
stay THE transcript of its recording — not a second file beside a new one.
So the old name is used while it exists (and holds this recording's
transcript): edits go into it, the scanner sees it, a re-sync finds one
file, and deleting removes every file of the recording.
"""

from __future__ import annotations

import json

import pytest

from whisper_tm.config import config
from whisper_tm.core import helpers


@pytest.fixture
def central(tmp_path, monkeypatch):
    monkeypatch.setitem(config._data.setdefault("directories", {}), "transcript_mode", "central")
    monkeypatch.setattr(type(config), "transcript_dir", property(lambda self: tmp_path))
    return tmp_path


@pytest.mark.parametrize(
    "a, b",
    [
        ("/m/a/b__c.mp4", "/m/a__b/c.mp4"),
        ("/m/x:y.mp4", "/m/x_y.mp4"),
        ("/m/a_/b.mp4", "/m/a/_b.mp4"),
    ],
)
def test_paths_that_used_to_collide_get_two_names(central, a, b):
    assert helpers.transcript_path_for(a) != helpers.transcript_path_for(b)


def test_an_unambiguous_name_is_unchanged(central):
    assert helpers.transcript_path_for("/m/show/ep 1.mp4").name == "m__show__ep 1.mp4.json"


def _write(path, filename):
    path.write_text(
        json.dumps({"filename": filename, "complete": True, "segments": []}), encoding="utf-8"
    )


def test_an_existing_old_name_stays_the_transcript(central):
    old = central / "m__a__b__c.mp4.json"
    _write(old, "a/b__c.mp4")
    # The one location — for reading, for writing an edit, for the scanner
    assert helpers.transcript_path_for("/m/a/b__c.mp4") == old
    assert helpers.find_transcript("/m/a/b__c.mp4") == old


def test_without_an_old_file_the_new_name_is_used(central):
    p = helpers.transcript_path_for("/m/a/b__c.mp4")
    assert p.name != "m__a__b__c.mp4.json" and p.name.endswith(".json")


def test_the_twins_old_file_is_not_taken(central):
    """/m/a/b/c.mp4 (unambiguous) owns the flat name m__a__b__c.mp4.json —
    /m/a__b/c.mp4 must not read or overwrite it."""
    twin = central / "m__a__b__c.mp4.json"
    _write(twin, "a/b/c.mp4")
    assert helpers.transcript_path_for("/m/a/b/c.mp4") == twin
    mine = helpers.transcript_path_for("/m/a__b/c.mp4")
    assert mine != twin
    assert helpers.find_transcript("/m/a__b/c.mp4") is None


def test_every_file_of_the_recording_is_listed_for_deleting(central):
    old = central / "m__a__b__c.mp4.json"
    _write(old, "a/b__c.mp4")
    canonical = "/m/a/b__c.mp4"
    new = helpers._transcript_path_central_hashed(canonical)
    _write(new, "a/b__c.mp4")
    assert set(helpers.transcript_files_for(canonical)) == {old, new}


def test_delete_removes_both_files(central, tmp_path, monkeypatch):
    from whisper_tm import extensions as ext
    from whisper_tm.api import transcription as api

    monkeypatch.setattr(ext, "db", None)
    monkeypatch.setattr(api, "db", None)
    old = central / "m__a__b__c.mp4.json"
    _write(old, "a/b__c.mp4")
    new = helpers._transcript_path_central_hashed("/m/a/b__c.mp4")
    _write(new, "a/b__c.mp4")
    api._delete_transcript_everywhere("/m/a/b__c.mp4")
    assert not old.exists() and not new.exists()
    assert helpers.find_transcript("/m/a/b__c.mp4") is None


def test_an_edit_lands_in_the_old_file(central, tmp_path, monkeypatch):
    from whisper_tm import extensions as ext
    from whisper_tm.api import transcription as api

    monkeypatch.setattr(ext, "db", None)
    monkeypatch.setattr(api, "db", None)
    media = tmp_path / "m" / "a"
    media.mkdir(parents=True)
    rec = media / "b__c.mp4"
    rec.write_bytes(b"x")
    flat = str(rec).strip("/").replace("/", "__") + ".json"
    old = central / flat
    _write(old, str(rec))
    api._save_transcript_data(
        str(rec), {"filename": str(rec), "complete": True, "segments": [{"text": "Edited."}]}
    )
    assert json.loads(old.read_text(encoding="utf-8"))["segments"][0]["text"] == "Edited."
    assert [p.name for p in central.glob("*.json")] == [flat]


def test_the_scanner_sees_the_old_file(central):
    from pathlib import Path

    from whisper_tm.scanner import indexer

    old = central / "m__a__b__c.mp4.json"
    _write(old, "a/b__c.mp4")
    assert indexer._transcript_beside("a/b__c.mp4", Path("/m/a/b__c.mp4"))
