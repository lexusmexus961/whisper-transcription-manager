"""A translation run with gaps never throws away a complete track.

Before, a run failed only when NOTHING came back. Three of ten chunks
lost to a server hiccup still replaced the language's existing, complete
track with the 70 % one — and a summary of that language silently read the
original text in the gaps.

Now the gaps of a new run are filled from the track that was there: what the
new run translated wins, what it missed keeps its earlier translation — but
only where the earlier entry still belongs to the same segment (after a
re-transcription segment 4 can be another sentence). Every entry
saved carries its segment's ``start``/``end`` for that check; an earlier
entry without them, or with other times, is not taken.
"""

from __future__ import annotations

import pytest

from whisper_tm.core import llm_tasks


class _Queue:
    def progress(self, *a, **k):
        pass

    def partial_result(self, *a, **k):
        pass

    def abort_requested(self, _id):
        return False


class _Db:
    def __init__(self, track):
        self.track = track
        self.saved = None

    def get_translation(self, filename, language, media_dir=None):
        return self.track

    def replace_translation(self, filename, language, track, model="", media_dir=None):
        self.saved = track
        return True


SEGS = [{"text": f"Sentence {i}", "start": float(i), "end": i + 1.0} for i in range(6)]


def _t(i, text):
    """A saved entry: id, text and its segment's times."""
    return {"id": i, "text": text, "start": float(i), "end": i + 1.0}


@pytest.fixture
def run(monkeypatch, tmp_path):
    from whisper_tm.config import config

    monkeypatch.setattr(llm_tasks, "_load_segments", lambda f, language="": SEGS)
    monkeypatch.setattr(llm_tasks, "_back_up_translations_json", lambda *a: None)
    monkeypatch.setattr(type(config), "media_dir", property(lambda self: tmp_path))

    def go(existing, new, missing):
        db = _Db(existing)
        monkeypatch.setattr(llm_tasks, "_db", lambda: db)
        monkeypatch.setattr(
            llm_tasks.trans,
            "translate",
            lambda *a, **k: {
                "translations": new,
                "missing": missing,
                "chunks": 3,
                "chunks_failed": 1 if missing else 0,
            },
        )
        r = llm_tasks._translation(
            {"id": 1, "filename": "a.mp4", "params": {"language": "en", "model": "m"}}, _Queue()
        )
        return r, db.saved

    return go


def test_gaps_keep_the_earlier_translation(run):
    old = [_t(i, f"old {i}") for i in range(6)]
    new = [{"id": i, "text": f"new {i}"} for i in (0, 1, 2, 3)]
    r, saved = run(old, new, [4, 5])
    assert saved == [_t(i, f"new {i}") for i in range(4)] + [_t(4, "old 4"), _t(5, "old 5")]
    assert r["missing"] == 0 and r["kept_from_before"] == 2


def test_an_earlier_entry_of_another_segment_is_not_taken(run):
    """After a re-transcription id 4 is a different sentence."""
    old = [_t(i, f"old {i}") for i in range(4)] + [
        {"id": 4, "text": "old 4", "start": 17.0, "end": 19.5},  # other times
        {"id": 5, "text": "old 5"},
    ]  # no times at all
    new = [{"id": i, "text": f"new {i}"} for i in (0, 1, 2, 3)]
    r, saved = run(old, new, [4, 5])
    assert saved == [_t(i, f"new {i}") for i in range(4)]
    assert r["missing"] == 2 and r["kept_from_before"] == 0


def test_without_an_earlier_track_the_gaps_stay_gaps(run):
    new = [{"id": i, "text": f"new {i}"} for i in (0, 1)]
    r, saved = run(None, new, [2, 3, 4, 5])
    assert saved == [_t(0, "new 0"), _t(1, "new 1")]
    assert r["missing"] == 4 and r["kept_from_before"] == 0


def test_a_complete_run_replaces_everything(run):
    old = [_t(i, f"old {i}") for i in range(6)]
    new = [{"id": i, "text": f"new {i}"} for i in range(6)]
    _r, saved = run(old, new, [])
    assert saved == [_t(i, f"new {i}") for i in range(6)]
