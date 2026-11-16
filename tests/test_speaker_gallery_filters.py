"""The recognition gallery holds only voices that may vote.

Two gaps this guards against:

* **Trashed voices voted on.** Discarding keeps a voice's name, and
  ``person_vectors`` read every named voice — so a wrongly named voice that
  was thrown out in the review still made "Who was there?", "Where does X
  speak" and the names written into transcripts say X.
* **Other models voted on.** The matrix is read for one model, the gallery
  for all: after a checkpoint change (``write`` keeps the old model's rows)
  the comparison went across models — or, with another dimension, failed.
"""

from __future__ import annotations

import numpy as np
import pytest

from whisper_tm.core import speaker as speaker_core
from whisper_tm.core import speaker_inventory as inventory
from whisper_tm.database.speaker import SpeakerDB

DIM = 192


def _vec(seed, basis=None, spread=0.01, dim=DIM):
    r = np.random.default_rng(seed)
    v = basis if basis is not None else r.normal(size=dim)
    v = v + spread * r.normal(size=dim)
    return (v / np.linalg.norm(v)).astype(np.float32)


ALICE, BOB = _vec(1, spread=0), _vec(2, spread=0)


@pytest.fixture
def db(tmp_path, monkeypatch):
    d = SpeakerDB(tmp_path / "speaker.db")
    d.initialize()
    monkeypatch.setattr("whisper_tm.database.speaker.speaker_db", lambda: d)
    inventory.discard()
    yield d
    inventory.discard()


def _rec(db, path, basis, n, seed, model=speaker_core.MODEL_BRAND, dim=DIM, label="SPEAKER_00"):
    db.write(
        path,
        model,
        [
            {
                "label": label,
                "windows": i + 1,
                "start_sec": i * 40.0,
                "speech_sec": 30.0,
                "segments": 2,
                "parts": [(i * 40.0, i * 40.0 + 30)],
                "v_separated": _vec(seed + i, basis if dim == DIM else None, dim=dim),
            }
            for i in range(n)
        ],
    )
    return [w["id"] for w in db.windows(path) if w["model"] == model]


def test_trashed_voices_leave_the_gallery(db):
    alice = _rec(db, "/m/alice.mkv", ALICE, 3, 10)
    wrong = _rec(db, "/m/wrong.mkv", BOB, 2, 20)  # sounds like Bob
    bob_only = _rec(db, "/m/bob.mkv", BOB, 3, 30)
    db.assign_many(alice + wrong, "Alice")
    assert any(
        p["person"] == "Alice" for p in speaker_core.who_was_there("/m/bob.mkv", threshold=0.9)
    )

    db.move_to_trash(wrong, reason="people review")
    assert len(db.person_vectors()["Alice"]) == 3
    assert len(inventory.inventory()["gallery"]["Alice"]) == 3
    assert not any(
        p["person"] == "Alice" for p in speaker_core.who_was_there("/m/bob.mkv", threshold=0.9)
    )
    assert inventory.where_it_occurs("Alice", threshold=0.9)["found"] == 1
    assert bob_only  # the recording itself is untouched


def test_gallery_is_read_for_the_inventory_model(db):
    alice = _rec(db, "/m/alice.mkv", ALICE, 3, 10)
    old = _rec(db, "/m/alice.mkv", None, 2, 40, model="old/model-1", dim=256)
    db.assign_many(alice + old, "Alice")
    assert len(db.person_vectors(model=speaker_core.MODEL_BRAND)["Alice"]) == 3
    g = inventory.inventory()["gallery"]
    assert g["Alice"].shape == (3, DIM)
