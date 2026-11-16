"""The speaker endpoints.

Tested is the **effect**, not the existence: assign and then query
the server status, not just "answers with 200".
"""

import numpy as np
import pytest
from flask import Flask

from whisper_tm.core import speaker as speaker_core
from whisper_tm.database.speaker import SpeakerDB


def _vec(seed):
    return np.random.default_rng(seed).normal(size=192).astype(np.float32)


@pytest.fixture
def app(tmp_path, monkeypatch):
    from whisper_tm.config import config

    media = tmp_path / "media"
    media.mkdir()
    (media / "recording.mkv").write_bytes(b"x")
    monkeypatch.setattr(type(config), "media_dir", property(lambda self: media))

    db = SpeakerDB(tmp_path / "speaker.db")
    db.initialize()
    monkeypatch.setattr("whisper_tm.database.speaker.speaker_db", lambda: db)
    monkeypatch.setattr("whisper_tm.api.speaker.speaker_db", lambda: db)

    from whisper_tm.api.speaker import speaker_bp

    a = Flask(__name__)
    a.register_blueprint(speaker_bp)
    a.config["TESTING"] = False
    a.speaker_db = db
    a.media = media
    return a


def _fill(db, media):
    path = str(media / "recording.mkv")
    alice = _vec(1)
    db.write(
        path,
        speaker_core.MODEL_BRAND,
        [
            {
                "label": "SPEAKER_00",
                "windows": 1,
                "start_sec": 0.0,
                "speech_sec": 30.0,
                "segments": 5,
                "v_separated": alice + 0.05 * _vec(2),
            },
            {
                "label": "SPEAKER_00",
                "windows": 2,
                "start_sec": 300.0,
                "speech_sec": 30.0,
                "segments": 5,
                "v_separated": alice + 0.05 * _vec(3),
            },
            {
                "label": "SPEAKER_01",
                "windows": 1,
                "start_sec": 600.0,
                "speech_sec": 12.0,
                "segments": 2,
                "v_separated": _vec(9),
            },
        ],
    )
    return path, alice


def test_recording_without_vectors_explains_itself(app):
    r = app.test_client().get("/api/speaker/recording/recording.mkv")
    assert r.status_code == 200
    d = r.get_json()
    assert d["windows"] == 0 and d["people"] == []
    assert "Voices only" in d["hint"]  # says what to do


def test_recording_shows_identifiers_and_times(app):
    _fill(app.speaker_db, app.media)
    d = app.test_client().get("/api/speaker/recording/recording.mkv").get_json()
    assert d["windows"] == 3
    assert d["model"] == speaker_core.MODEL_BRAND
    identifiers = {k["label"]: k for k in d["labels"]}
    assert identifiers["SPEAKER_00"]["windows"] == 2
    assert identifiers["SPEAKER_00"]["speech_sec"] == 60.0
    # sorted by speech time — largest identifier first
    assert d["labels"][0]["label"] == "SPEAKER_00"


def test_person_is_found_in_recording(app):
    """Core: name it, and the recording knows the person afterwards."""
    path, alice = _fill(app.speaker_db, app.media)
    gallery = str(app.media / "voice_message.opus")
    app.speaker_db.write(
        gallery,
        speaker_core.MODEL_BRAND,
        [
            {
                "label": "SPEAKER_00",
                "windows": 1,
                "start_sec": 0.0,
                "speech_sec": 30.0,
                "segments": 1,
                "v_separated": alice + 0.05 * _vec(4),
            }
        ],
    )
    vid = app.speaker_db.windows(gallery)[0]["id"]

    c = app.test_client()
    assert (
        c.post("/api/speaker/assign", json={"vector_id": vid, "person": "Alice"}).status_code == 200
    )

    d = c.get("/api/speaker/recording/recording.mkv").get_json()
    people = {p["person"]: p for p in d["people"]}
    assert "Alice" in people
    assert people["Alice"]["windows"] == 2
    assert people["Alice"]["timestamps"] == [0, 5]  # Minute 0 and 5
    assert people["Alice"]["labels"] == ["SPEAKER_00"]

    # and dissolve again
    assert c.post("/api/speaker/assign", json={"vector_id": vid, "person": None}).status_code == 200
    assert c.get("/api/speaker/recording/recording.mkv").get_json()["people"] == []


def test_threshold_has_effect(app):
    path, alice = _fill(app.speaker_db, app.media)
    gallery = str(app.media / "s.opus")
    app.speaker_db.write(
        gallery,
        speaker_core.MODEL_BRAND,
        [
            {
                "label": "SPEAKER_00",
                "windows": 1,
                "start_sec": 0.0,
                "speech_sec": 30.0,
                "segments": 1,
                "v_separated": alice + 0.05 * _vec(4),
            }
        ],
    )
    app.speaker_db.assign(app.speaker_db.windows(gallery)[0]["id"], "Alice")
    c = app.test_client()
    assert c.get("/api/speaker/recording/recording.mkv?threshold=0.999").get_json()["people"] == []
    assert c.get("/api/speaker/recording/recording.mkv?threshold=0.5").get_json()["people"]


def test_threshold_stays_in_range(app):
    _fill(app.speaker_db, app.media)
    for value in ("abc", "-5", "17"):
        d = (
            app.test_client()
            .get(f"/api/speaker/recording/recording.mkv?threshold={value}")
            .get_json()
        )
        assert 0.0 <= d["threshold"] <= 1.0


def test_without_a_threshold_the_setting_applies_and_is_reported(app, monkeypatch):
    """The file view asks without ``?threshold=`` and starts its slider at
    the ``threshold`` that comes back — so the setting must be what
    decides, and what is reported."""
    from whisper_tm.config import config

    path, alice = _fill(app.speaker_db, app.media)
    gallery = str(app.media / "s.opus")
    app.speaker_db.write(
        gallery,
        speaker_core.MODEL_BRAND,
        [
            {
                "label": "SPEAKER_00",
                "windows": 1,
                "start_sec": 0.0,
                "speech_sec": 30.0,
                "segments": 1,
                "v_separated": alice + 0.05 * _vec(4),
            }
        ],
    )
    app.speaker_db.assign(app.speaker_db.windows(gallery)[0]["id"], "Alice")
    speaker = config._data.setdefault("speaker", {})
    c = app.test_client()

    monkeypatch.setitem(speaker, "threshold_gallery", 0.999)
    d = c.get("/api/speaker/recording/recording.mkv").get_json()
    assert d["threshold"] == 0.999
    assert d["people"] == []

    monkeypatch.setitem(speaker, "threshold_gallery", 0.5)
    d = c.get("/api/speaker/recording/recording.mkv").get_json()
    assert d["threshold"] == 0.5
    assert d["people"]

    # A value of the request's own still wins — that is the slider
    d = c.get("/api/speaker/recording/recording.mkv?threshold=0.999").get_json()
    assert d["threshold"] == 0.999
    assert d["people"] == []


def test_window_list(app):
    _fill(app.speaker_db, app.media)
    d = app.test_client().get("/api/speaker/windows/recording.mkv").get_json()
    assert len(d["windows"]) == 3
    assert all("v_separated" not in f for f in d["windows"])  # no raw data
    assert d["windows"][0]["start_sec"] == 0.0


def test_assignment_without_id(app):
    r = app.test_client().post("/api/speaker/assign", json={"person": "Alice"})
    assert r.status_code == 400


def test_statistics_names_the_operating_point(app):
    _fill(app.speaker_db, app.media)
    d = app.test_client().get("/api/speaker/statistics").get_json()
    assert d["windows"] == 3 and d["files"] == 1
    assert d["operating_point"]["model"] == "redimnet_m/vb2-ptn"
    assert d["operating_point"]["window_length_sec"] == 30.0


def test_unusable_paths_do_not_give_five_hundred(app):
    c = app.test_client()
    for name in ("../../etc/passwd", "..", "../outside.mp4"):
        assert c.get(f"/api/speaker/recording/{name}").status_code < 500
        assert c.get(f"/api/speaker/windows/{name}").status_code < 500
