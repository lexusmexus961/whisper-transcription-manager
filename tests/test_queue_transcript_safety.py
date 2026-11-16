"""A queue run never costs a transcript it did not replace (core/queue.py).

Two review findings:

* **Re-transcription deleted first.** ``retranscribe`` removed the old
  transcript (JSON, database row) and the file's voice vectors — names
  included — before the file was even looked for. A NAS that was briefly
  gone, a skip list, a failed or cancelled run: every one of them ended with
  the old transcript and its named voices gone, and nothing new in their
  place. Now the old ones stay until the new run is ``done``; on any other
  outcome they are what they were.
* **The overlap repair kept the wrong name.** When root and filename
  overlapped (see test_queue_root_overlap.py), the path was repaired but the
  name was not: "transcript present" was asked of a path that does not
  exist, and the "incomplete transcript" cleanup deleted the real file's
  COMPLETE transcript.

These tests use the real transcript location, the real database and the
real voice database — only the transcription itself is a stand-in.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import whisper_tm.extensions as ext
from whisper_tm.core import queue as q
from whisper_tm.core.helpers import invalidate_complete_cache, transcript_path_for
from whisper_tm.database import Database
from whisper_tm.jobs import JobManager

OLD_TEXT = "the old, good transcript"
NEW_TEXT = "the new transcript"


@pytest.fixture
def env(tmp_path, monkeypatch):
    from whisper_tm.config import config
    from whisper_tm.database import speaker as sp

    q.stop_worker()
    monkeypatch.setitem(config._data.setdefault("queue", {}), "lanes", 1)
    monkeypatch.setitem(config._data.setdefault("directories", {}), "transcript_mode", "alongside")
    media = tmp_path / "nas"
    (media / "videos" / "Archive").mkdir(parents=True)
    monkeypatch.setattr(type(config), "media_dir", property(lambda self: media))
    monkeypatch.setattr(type(config), "db_path", property(lambda self: tmp_path / "test.db"))
    db = Database(tmp_path / "test.db")
    db.initialize()
    monkeypatch.setattr(ext, "db", db)
    monkeypatch.setattr(ext, "job_manager", JobManager(logs_dir=tmp_path / "logs"))
    monkeypatch.setattr(sp, "_db", None)
    q._running.clear()
    q._current = None
    q._abort = q._paused = q._stopping = q._shutting_down = False
    q._requeue_on_abort.clear()
    q._solo.clear()
    q._log.clear()
    yield {"db": db, "media": media}
    q.stop_worker()
    sp._db = None


def _transcribed(env, rel: str, base=None) -> tuple[str, str]:
    """A file with a complete transcript and two named voices."""
    from whisper_tm.core import paths
    from whisper_tm.core.speaker import MODEL_BRAND
    from whisper_tm.database.speaker import speaker_db

    base = base or env["media"]
    f = base / rel
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_bytes(b"x" * 2048)
    abs_path = paths.canon(f)
    data = {
        "complete": True,
        "segments": [{"start": 0, "end": 5, "text": OLD_TEXT, "speaker": "SPEAKER_00"}],
    }
    tp = transcript_path_for(rel, abs_path=abs_path)
    tp.parent.mkdir(parents=True, exist_ok=True)
    tp.write_text(json.dumps(data), encoding="utf-8")
    env["db"].upsert_transcript(rel, data, media_dir=str(base), abs_path=abs_path)
    sdb = speaker_db()
    sdb.write(
        abs_path,
        MODEL_BRAND,
        [
            {
                "label": lb,
                "windows": 1,
                "start_sec": i * 30.0,
                "speech_sec": 30.0,
                "v_separated": np.ones(8, dtype=np.float32) * (i + 1),
            }
            for i, lb in enumerate(("SPEAKER_00", "SPEAKER_01"))
        ],
    )
    ids = [w["id"] for w in sdb.windows(abs_path)]
    sdb.write_names([(ids[0], "Alice", "manual"), (ids[1], "Bob", "manual")])
    invalidate_complete_cache(rel, media_dir=str(base))
    return abs_path, str(tp)


def _voices(abs_path):
    from whisper_tm.database.speaker import speaker_db

    return sorted((w["label"], w["person"]) for w in speaker_db().windows(abs_path))


def _text(tp):
    return json.loads(open(tp, encoding="utf-8").read())["segments"][0]["text"]


def _db_text(env, abs_path):
    t = env["db"].get_transcript(abs_path)
    return t["segments"][0]["text"] if t else None


def _worker(outcome, *, new_vectors=False, write=False):
    """The transcription stand-in: may write new vectors / a new transcript
    first, then ends as ``outcome`` (done, error, cancel, oom)."""

    def run(job_id, source, filename, **kwargs):
        from whisper_tm.core import paths
        from whisper_tm.core.speaker import MODEL_BRAND
        from whisper_tm.database.speaker import speaker_db

        job = ext.job_manager.get(job_id)
        abs_path = paths.canon(source)
        if new_vectors:
            speaker_db().write(
                abs_path,
                MODEL_BRAND,
                [
                    {
                        "label": "SPEAKER_07",
                        "windows": 1,
                        "start_sec": 0.0,
                        "speech_sec": 30.0,
                        "v_separated": np.zeros(8, dtype=np.float32),
                    }
                ],
            )
        if write:
            data = {"complete": True, "segments": [{"start": 0, "end": 5, "text": NEW_TEXT}]}
            tp = transcript_path_for(filename, abs_path=source)
            tp.write_text(json.dumps(data), encoding="utf-8")
            ext.db.upsert_transcript(
                filename, data, media_dir=str(source.parent), abs_path=abs_path
            )
        if outcome == "done":
            ext.job_manager.complete(job, {"processing_time": "1s"})
        elif outcome == "cancel":
            job.cancel = True
            ext.job_manager.fail(job, "Cancelled")
        elif outcome == "oom":
            ext.job_manager.fail(job, "CUDA out of memory")
        else:
            ext.job_manager.fail(job, "Test error")

    return run


def _run_one(monkeypatch, worker):
    monkeypatch.setattr("whisper_tm.core.transcription.transcription_worker", worker)
    item = q._claim_next()
    assert item is not None
    try:
        q._process(item)
    finally:
        q._release(item)
    return {i["filename"]: i for i in q.list_items()}


class TestRetranscribeKeepsTheOldOnesUntilDone:
    @pytest.mark.parametrize("outcome", ["error", "cancel"])
    @pytest.mark.parametrize("new_vectors", [False, True])
    def test_a_run_that_does_not_finish(self, env, monkeypatch, outcome, new_vectors):
        abs_path, tp = _transcribed(env, "a.mp4")
        before = _voices(abs_path)
        q.enqueue(["a.mp4"], {"retranscribe": True}, str(env["media"]))
        items = _run_one(monkeypatch, _worker(outcome, new_vectors=new_vectors))
        assert items["a.mp4"]["status"] in ("error", "cancelled")
        assert _text(tp) == OLD_TEXT
        assert _db_text(env, abs_path) == OLD_TEXT
        assert _voices(abs_path) == before == [("SPEAKER_00", "Alice"), ("SPEAKER_01", "Bob")]

    def test_new_voices_that_reuse_the_old_ids_are_still_put_back(self, env, monkeypatch):
        """Deleted rows free their ids: two new voices carry the old two ids."""
        from whisper_tm.core.speaker import MODEL_BRAND
        from whisper_tm.database.speaker import speaker_db

        abs_path, tp = _transcribed(env, "a.mp4")

        def worker(job_id, source, filename, **kwargs):
            speaker_db().write(
                abs_path,
                MODEL_BRAND,
                [
                    {
                        "label": lb,
                        "windows": 1,
                        "start_sec": 0.0,
                        "speech_sec": 30.0,
                        "v_separated": np.zeros(8, dtype=np.float32),
                    }
                    for lb in ("SPEAKER_08", "SPEAKER_09")
                ],
            )
            ext.job_manager.fail(ext.job_manager.get(job_id), "Test error")

        q.enqueue(["a.mp4"], {"retranscribe": True}, str(env["media"]))
        _run_one(monkeypatch, worker)
        assert _voices(abs_path) == [("SPEAKER_00", "Alice"), ("SPEAKER_01", "Bob")]

    def test_a_restore_never_touches_another_files_voices(self, env, monkeypatch):
        """Freed ids are reused: while the run failed, another lane (the
        voice renewal, say) stored a second file's voices under the ids the
        old ones had. Putting the old ones back must not replace those."""
        from whisper_tm.core.speaker import MODEL_BRAND
        from whisper_tm.database.speaker import speaker_db

        abs_path, tp = _transcribed(env, "a.mp4")
        other = env["media"] / "b.mp4"
        other.write_bytes(b"x")
        other_abs = str(other)

        def worker(job_id, source, filename, **kwargs):
            sdb = speaker_db()
            sdb.write(
                abs_path,
                MODEL_BRAND,
                [  # frees both old ids
                    {
                        "label": "SPEAKER_07",
                        "windows": 1,
                        "start_sec": 0.0,
                        "speech_sec": 30.0,
                        "v_separated": np.zeros(8, dtype=np.float32),
                    }
                ],
            )
            sdb.write(
                other_abs,
                MODEL_BRAND,
                [  # takes the freed id
                    {
                        "label": lb,
                        "windows": 1,
                        "start_sec": i * 30.0,
                        "speech_sec": 30.0,
                        "v_separated": np.ones(8, dtype=np.float32),
                    }
                    for i, lb in enumerate(("SPEAKER_00", "SPEAKER_01"))
                ],
            )
            ids = [w["id"] for w in sdb.windows(other_abs)]
            sdb.write_names([(ids[0], "Carol", "manual"), (ids[1], "Dave", "manual")])
            ext.job_manager.fail(ext.job_manager.get(job_id), "Test error")

        q.enqueue(["a.mp4"], {"retranscribe": True}, str(env["media"]))
        _run_one(monkeypatch, worker)
        assert _voices(other_abs) == [("SPEAKER_00", "Carol"), ("SPEAKER_01", "Dave")]
        assert _voices(abs_path) == [("SPEAKER_00", "Alice"), ("SPEAKER_01", "Bob")]

    def test_a_run_that_wrote_its_transcript_and_then_failed(self, env, monkeypatch):
        abs_path, tp = _transcribed(env, "a.mp4")
        q.enqueue(["a.mp4"], {"retranscribe": True}, str(env["media"]))
        _run_one(monkeypatch, _worker("error", new_vectors=True, write=True))
        assert _text(tp) == OLD_TEXT
        assert _db_text(env, abs_path) == OLD_TEXT
        assert _voices(abs_path) == [("SPEAKER_00", "Alice"), ("SPEAKER_01", "Bob")]

    def test_an_out_of_memory_run_goes_back_and_keeps_them(self, env, monkeypatch):
        from whisper_tm.core import vram

        abs_path, tp = _transcribed(env, "a.mp4")
        monkeypatch.setattr(q, "_unload_models", lambda: None)
        monkeypatch.setattr(vram, "wait_for_free_vram", lambda *a, **k: (True, 9999))
        q.enqueue(["a.mp4"], {"retranscribe": True}, str(env["media"]))
        items = _run_one(monkeypatch, _worker("oom", new_vectors=True))
        assert items["a.mp4"]["status"] == "pending"
        assert _text(tp) == OLD_TEXT
        assert _voices(abs_path) == [("SPEAKER_00", "Alice"), ("SPEAKER_01", "Bob")]

    def test_a_missing_file_costs_nothing(self, env, monkeypatch):
        abs_path, tp = _transcribed(env, "a.mp4")
        (env["media"] / "a.mp4").unlink()  # the share is gone
        q.enqueue(["a.mp4"], {"retranscribe": True}, str(env["media"]))
        ran = []
        items = _run_one(monkeypatch, lambda *a, **k: ran.append(1))
        assert items["a.mp4"]["status"] == "error" and ran == []
        assert _text(tp) == OLD_TEXT
        assert _db_text(env, abs_path) == OLD_TEXT
        assert len(_voices(abs_path)) == 2

    def test_a_stop_before_the_job_exists(self, env, monkeypatch):
        abs_path, tp = _transcribed(env, "a.mp4")
        q.enqueue(["a.mp4"], {"retranscribe": True}, str(env["media"]))
        item = q._claim_next()
        item["_cancel"] = True
        q._requeue_on_abort.add(item["id"])
        monkeypatch.setattr("whisper_tm.core.transcription.transcription_worker", _worker("cancel"))
        try:
            q._process(item)
        finally:
            q._release(item)
        assert q.list_items()[0]["status"] == "pending"
        assert _text(tp) == OLD_TEXT and len(_voices(abs_path)) == 2

    def test_done_replaces_them(self, env, monkeypatch):
        """The point of a re-transcription still holds: when it is done,
        the new transcript stands and the old voices are gone."""
        abs_path, tp = _transcribed(env, "a.mp4")
        q.enqueue(["a.mp4"], {"retranscribe": True}, str(env["media"]))
        items = _run_one(monkeypatch, _worker("done", write=True))
        assert items["a.mp4"]["status"] == "done"
        assert _text(tp) == NEW_TEXT
        assert _db_text(env, abs_path) == NEW_TEXT
        assert _voices(abs_path) == []

    def test_done_with_new_voices_keeps_only_the_new(self, env, monkeypatch):
        abs_path, tp = _transcribed(env, "a.mp4")
        q.enqueue(["a.mp4"], {"retranscribe": True}, str(env["media"]))
        _run_one(monkeypatch, _worker("done", new_vectors=True, write=True))
        assert _voices(abs_path) == [("SPEAKER_07", None)]


class TestTheOverlapRepairUsesTheRepairedName:
    def test_a_complete_transcript_is_kept_and_the_file_skipped(self, env, monkeypatch):
        sub = env["media"] / "videos" / "Archive"
        abs_path, tp = _transcribed(env, "film.mp4", base=sub)
        # The batch.js case: name relative to the media root, root a subfolder
        q.enqueue(["videos/Archive/film.mp4"], {}, str(sub))
        ran = []
        items = _run_one(monkeypatch, lambda *a, **k: ran.append(1))
        assert ran == [], "an already transcribed file was transcribed again"
        assert items["videos/Archive/film.mp4"]["status"] == "skipped"
        assert Path(tp).is_file(), "the complete transcript was deleted"
        assert _text(tp) == OLD_TEXT
        assert len(_voices(abs_path)) == 2

    def test_an_untranscribed_file_runs_under_its_real_name(self, env, monkeypatch):
        sub = env["media"] / "videos" / "Archive"
        (sub / "new.mp4").write_bytes(b"x" * 2048)
        q.enqueue(["videos/Archive/new.mp4"], {}, str(sub))
        seen = []

        def worker(job_id, source, filename, **kwargs):
            seen.append((str(source), filename))
            ext.job_manager.complete(ext.job_manager.get(job_id), {})

        _run_one(monkeypatch, worker)
        assert seen == [(str(sub / "new.mp4"), "new.mp4")]

    def test_a_retranscription_of_a_repaired_entry_keeps_them_on_failure(self, env, monkeypatch):
        sub = env["media"] / "videos" / "Archive"
        abs_path, tp = _transcribed(env, "film.mp4", base=sub)
        q.enqueue(["videos/Archive/film.mp4"], {"retranscribe": True}, str(sub))
        _run_one(monkeypatch, _worker("error", new_vectors=True, write=True))
        assert _text(tp) == OLD_TEXT
        assert _voices(abs_path) == [("SPEAKER_00", "Alice"), ("SPEAKER_01", "Bob")]
