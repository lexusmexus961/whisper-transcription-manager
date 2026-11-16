"""Names for a recording's speakers: one name per speaker, one speaker per
name, and never vouched for by the recording itself.

``names_for_labels`` writes names into transcripts, and two
things let it write wrong ones:

* **The recording vouched for itself.** Its unnamed speaker was held against
  the whole gallery, including the named windows of the SAME recording. The
  shared channel lifts every similarity inside a recording, so a second
  speaker cleared the threshold against the first one's windows and got
  that person's name.
* **Two speakers, one name.** Nothing stopped the recognition from giving
  the same person to two labels — or a name the map already gave another.
"""

from __future__ import annotations

import numpy as np
import pytest

from whisper_tm.core import speaker as sp
from whisper_tm.core import speaker_inventory as inventory
from whisper_tm.database.speaker import SpeakerDB

DIM = 192


def _basis(seed):
    v = np.random.default_rng(seed).normal(size=DIM)
    return v / np.linalg.norm(v)


def _near(basis, seed, spread=0.02):
    v = basis + spread * np.random.default_rng(seed).normal(size=DIM)
    return (v / np.linalg.norm(v)).astype(np.float32)


HERE, ALICE_ELSEWHERE, CAROL = _basis(1), _basis(2), _basis(3)


@pytest.fixture
def db(tmp_path, monkeypatch):
    d = SpeakerDB(tmp_path / "speaker.db")
    d.initialize()
    monkeypatch.setattr("whisper_tm.database.speaker.speaker_db", lambda: d)
    inventory.discard()
    yield d
    inventory.discard()


def _write(db, path, speakers):
    """``speakers``: {label: [vector, …]}"""
    rows = []
    for label, vs in speakers.items():
        for i, v in enumerate(vs):
            rows.append(
                {
                    "label": label,
                    "windows": i + 1,
                    "start_sec": i * 40.0,
                    "speech_sec": 30.0,
                    "segments": 2,
                    "parts": [(i * 40.0, i * 40.0 + 30)],
                    "v_separated": v,
                }
            )
    db.write(path, sp.MODEL_BRAND, rows)
    out: dict[str, list[int]] = {}
    for w in db.windows(path):
        out.setdefault(w["label"], []).append(w["id"])
    return out


def test_the_recording_does_not_vouch_for_itself(db):
    # SPEAKER_00 is Alice, named on the map. SPEAKER_01 is someone else, but
    # the same channel makes it alike to SPEAKER_00 — and only to it.
    rec = _write(
        db,
        "/m/talk.mkv",
        {
            "SPEAKER_00": [_near(HERE, s) for s in (10, 11, 12)],
            "SPEAKER_01": [_near(HERE, s, 0.03) for s in (20, 21)],
        },
    )
    other = _write(db, "/m/alice.mkv", {"SPEAKER_00": [_near(ALICE_ELSEWHERE, 30)]})
    db.assign_many(rec["SPEAKER_00"] + other["SPEAKER_00"], "Alice")

    got = sp.names_for_labels("/m/talk.mkv", threshold=0.75)
    assert got["SPEAKER_00"]["source"] == "map"
    assert "SPEAKER_01" not in got


def test_one_name_goes_to_one_speaker(db):
    carol = _write(db, "/m/carol.mkv", {"SPEAKER_00": [_near(CAROL, 40)]})
    db.assign_many(carol["SPEAKER_00"], "Carol")
    _write(
        db,
        "/m/two.mkv",
        {
            "SPEAKER_00": [_near(CAROL, s) for s in (50, 51, 52)],  # more votes
            "SPEAKER_01": [_near(CAROL, s) for s in (60, 61)],
        },
    )

    got = sp.names_for_labels("/m/two.mkv", threshold=0.75)
    assert got["SPEAKER_00"]["name"] == "Carol"
    assert "SPEAKER_01" not in got


def test_a_name_the_map_gave_is_not_given_again(db):
    alice = _write(db, "/m/alice.mkv", {"SPEAKER_00": [_near(ALICE_ELSEWHERE, 70)]})
    db.assign_many(alice["SPEAKER_00"], "Alice")
    rec = _write(
        db,
        "/m/three.mkv",
        {
            "SPEAKER_00": [_near(HERE, 80)],
            "SPEAKER_01": [_near(ALICE_ELSEWHERE, s) for s in (90, 91)],
        },
    )
    db.assign_many(rec["SPEAKER_00"], "Alice")  # the map says so

    got = sp.names_for_labels("/m/three.mkv", threshold=0.75)
    assert got["SPEAKER_00"] == {"name": "Alice", "source": "map", "windows": 1, "of": 1}
    assert "SPEAKER_01" not in got


def test_the_held_gallery_is_used_not_the_database(db, monkeypatch):
    """Reading every named voice's vector from the
    database on each transcription is what the held inventory is for."""
    carol = _write(db, "/m/carol.mkv", {"SPEAKER_00": [_near(CAROL, 40)]})
    db.assign_many(carol["SPEAKER_00"], "Carol")
    _write(db, "/m/four.mkv", {"SPEAKER_00": [_near(CAROL, s) for s in (50, 51)]})
    inventory.inventory()  # held, as in the app

    def no(*a, **k):
        raise AssertionError("names_for_labels read the gallery from the database")

    monkeypatch.setattr(db, "person_vectors", no)
    got = sp.names_for_labels("/m/four.mkv", threshold=0.75)
    assert got["SPEAKER_00"]["name"] == "Carol"


def test_no_second_implementation_is_left_behind():
    import inspect

    src = inspect.getsource(sp.names_for_labels)
    assert src.count("b = inventory()") == 1
    assert src.count("votes: dict[str, Counter]") == 1
