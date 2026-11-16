"""
How a transcript was made (schema v25, ``transcripts.settings_json``).

New transcriptions record their effective settings; older ones get them
reconstructed from what the database still knows. The reconstruction must
say which values it only inferred — a selection like "everything made
without diarization" is only as good as that distinction.
"""

import json

import pytest

from whisper_tm.database import Database

ROOT = "/media"
SAMPLE = {
    "complete": True,
    "language": "de",
    "model": "large-v3",
    "vocal_isolation": True,
    "vocal_model": "mel_roformer_kim",
    "app_version": "2.1.0",
    "total_chunks": 1,
    "segments": [{"speaker": "SPEAKER_00", "start": 0.0, "end": 5.0, "text": "Hello"}],
}


@pytest.fixture
def db(tmp_path):
    d = Database(tmp_path / "wx.db")
    d.initialize()
    yield d
    d.close_all()


def _settings(db, name):
    with db.connection() as c:
        raw = c.execute(
            "SELECT settings_json FROM transcripts WHERE abs_path = ?", (f"{ROOT}/{name}",)
        ).fetchone()[0]
    return json.loads(raw) if raw else None


def _add(db, name, data):
    return db.upsert_transcript(name, data, media_dir=ROOT, abs_path=f"{ROOT}/{name}")


class TestRecorded:
    def test_stored_with_the_transcript(self, db):
        _add(db, "a.mp4", {**SAMPLE, "settings": {"source": "recorded", "diarize": True}})
        assert _settings(db, "a.mp4") == {"source": "recorded", "diarize": True}

    def test_rereading_an_old_json_keeps_them(self, db):
        """The start-up sync reads JSON files from before v25 again — they
        have no settings, and that must not erase the recorded ones."""
        _add(db, "a.mp4", {**SAMPLE, "settings": {"source": "recorded", "diarize": True}})
        _add(db, "a.mp4", SAMPLE)
        assert _settings(db, "a.mp4")["source"] == "recorded"
