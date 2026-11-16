"""
Queue job kind "voices": diarization + voice vectors for a transcribed file.

It must run the transcription's own path (same worker, same separation,
same diarization) — the old backfill tool separated 16-kHz audio and
diarized without the silence cut, so its voices weren't comparable. And it
must never change the transcript's text, nor replace voices that are
already there: their names hang on the vector ids. Speakers and names it
may add (tests/test_voices_into_transcript.py).
"""

import inspect
import json

import pytest

from whisper_tm.core import queue as q
from whisper_tm.database import Database


@pytest.fixture
def cfg():
    from whisper_tm.config import config as c

    if not c.is_loaded:
        c.load()
    return c


class TestNormalize:
    def test_the_job_kind_sets_voices_only(self, cfg):
        assert q._normalize({"job": "voices"})["voices_only"] is True
        assert q._normalize({})["voices_only"] is False

    def test_voices_are_always_made_the_comparable_way(self, cfg):
        """Diarized, separated, vectors on — whatever the dialog said."""
        o = q._normalize(
            {
                "job": "voices",
                "diarize": False,
                "vocal_isolation": False,
                "speaker_vectors": False,
                "save_audio_alongside": True,
            }
        )
        assert (o["diarize"], o["vocal_isolation"], o["speaker_vectors"]) == (True, True, True)
        assert o["save_audio_alongside"] is False


def test_the_worker_runs_voices_on_its_own_path():
    from whisper_tm.core.transcription import transcription_worker

    assert inspect.signature(transcription_worker).parameters["voices_only"].default is False
    source = inspect.getsource(transcription_worker)
    # Whisper and alignment are skipped, diarization is the shared one.
    assert source.index("if voices_only:") < source.index("Audio splitting")
    assert "_diarize_or_none(" in source and "carry_along(" in source


def test_a_file_with_voices_is_skipped(monkeypatch, cfg):
    from whisper_tm.database import speaker as speaker_db_module

    finished = []
    monkeypatch.setattr(q, "_db", lambda: None)
    monkeypatch.setattr(q, "_jm", lambda: None)
    monkeypatch.setattr(q, "_finish", lambda item_id, status, *a: finished.append(status))
    monkeypatch.setattr(q, "qlog", lambda *a, **k: None)

    class _HasVoices:
        def has_vectors(self, *a, **k):
            return True

    monkeypatch.setattr(speaker_db_module, "speaker_db", lambda: _HasVoices())

    q._process({"id": 1, "filename": "a.mp4", "dir": "/media", "settings": {"job": "voices"}})
    assert finished == ["skipped"]


def test_a_file_without_transcript_is_skipped(monkeypatch, cfg):
    """Voices only for transcribed files: a later transcription would
    otherwise replace them, names included, without any warning."""
    from whisper_tm.database import speaker as speaker_db_module

    finished, logged = [], []
    monkeypatch.setattr(q, "_db", lambda: None)
    monkeypatch.setattr(q, "_jm", lambda: None)
    monkeypatch.setattr(q, "_finish", lambda item_id, status, *a: finished.append(status))
    monkeypatch.setattr(q, "qlog", lambda msg, *a, **k: logged.append(msg))
    monkeypatch.setattr(q, "is_transcript_complete", lambda *a, **k: False)

    class _NoVoices:
        def has_vectors(self, *a, **k):
            return False

    monkeypatch.setattr(speaker_db_module, "speaker_db", lambda: _NoVoices())

    q._process({"id": 1, "filename": "a.mp4", "dir": "/media", "settings": {"job": "voices"}})
    assert finished == ["skipped"]
    assert "no transcript" in logged[-1]


def test_merging_settings_keeps_how_the_transcript_was_made(tmp_path):
    db = Database(tmp_path / "wx.db")
    db.initialize()
    try:
        db.upsert_transcript(
            "a.mp4",
            {
                "complete": True,
                "segments": [],
                "settings": {"source": "recorded", "diarize": False},
            },
            media_dir="/media",
            abs_path="/media/a.mp4",
        )
        assert db.merge_transcript_settings("/media/a.mp4", {"voices": {"windows": 3}})
        assert not db.merge_transcript_settings("/media/missing.mp4", {"voices": {}})
        with db.connection() as c:
            s = json.loads(c.execute("SELECT settings_json FROM transcripts").fetchone()[0])
        assert s == {"source": "recorded", "diarize": False, "voices": {"windows": 3}}
    finally:
        db.close_all()


def test_the_queue_does_not_rewrite_the_transcript_after_voices():
    """After a job the queue re-reads the JSON and upserts it with the job's
    log link. For a voices (or separation) run that rewrote the untouched
    transcript and pointed its log link at the wrong run."""
    source = inspect.getsource(q._process_entry)  # _process settles around it
    rewrite = source.index("tp2 = transcript_path_for(")
    guard = source.rfind(
        'elif job.status == "done" and not opts.get("separation_only")', 0, rewrite
    )
    voices = source.rfind('if job.status == "done" and opts.get("voices_only")', 0, guard)
    assert 0 < voices < guard < rewrite
