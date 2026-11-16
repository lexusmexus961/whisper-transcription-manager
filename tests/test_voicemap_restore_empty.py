"""An empty selection restores nothing — not the whole trash.

A review found: ``ids = d.get("ids") or d.get("vector_ids")`` turned an empty
list into "no list", and no list means "restore everything" — every
discarded voice back in the library from a click on an empty selection.
Only a request WITHOUT ``ids`` (the "everything" button) restores all.
"""

from __future__ import annotations

import numpy as np
import pytest
from flask import Flask

from whisper_tm.core import speaker as speaker_core
from whisper_tm.database.speaker import SpeakerDB


@pytest.fixture
def client(tmp_path, monkeypatch):
    db = SpeakerDB(tmp_path / "speaker.db")
    db.initialize()
    monkeypatch.setattr("whisper_tm.database.speaker.speaker_db", lambda: db)
    monkeypatch.setattr("whisper_tm.api.speaker.speaker_db", lambda: db)
    r = np.random.default_rng(1)
    db.write(
        "/m/a.mkv",
        speaker_core.MODEL_BRAND,
        [
            {
                "label": "SPEAKER_00",
                "windows": i + 1,
                "start_sec": i * 30.0,
                "speech_sec": 30.0,
                "segments": 1,
                "v_separated": (r.normal(size=192)).astype(np.float32),
            }
            for i in range(3)
        ],
    )
    ids = [w["id"] for w in db.windows("/m/a.mkv")]
    db.move_to_trash(ids)
    from whisper_tm.api.speaker import speaker_bp

    a = Flask(__name__)
    a.register_blueprint(speaker_bp)
    return a.test_client(), db


@pytest.mark.parametrize("key", ["ids", "vector_ids"])
def test_an_empty_list_restores_nothing(client, key):
    c, db = client
    r = c.post("/api/voicemap/trash/restore", json={key: []})
    assert r.status_code == 400
    assert len(db.trash_ids()) == 3


def test_without_ids_everything_comes_back(client):
    c, db = client
    r = c.post("/api/voicemap/trash/restore", json={})
    assert r.status_code == 200 and r.get_json()["back"] == 3
    assert not db.trash_ids()
