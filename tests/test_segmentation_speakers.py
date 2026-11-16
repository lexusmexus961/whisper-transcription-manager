"""Subtitle cues keep their speakers apart.

Background: the word-boundary split never cut where the speaker of the
words changed (whisperx.assign_word_speakers gives every word its own), and
the merge of short cues joined neighbours of different speakers — a 0.3-s
"Yes." of SPEAKER_01 went into SPEAKER_00's cue and was attributed to the
wrong person.
"""

from __future__ import annotations

import pytest

from whisper_tm.core.segmentation import word_boundary_segmentation


@pytest.fixture(autouse=True)
def limits(monkeypatch):
    from whisper_tm.config import config

    t = config._data.setdefault("transcription", {})
    monkeypatch.setitem(t, "max_segment_duration", 3.5)
    monkeypatch.setitem(t, "max_segment_words", 8)
    monkeypatch.setitem(t, "word_pause_threshold", 0.3)
    monkeypatch.setitem(t, "split_natural_only", False)


def _w(word, start, end, speaker):
    return {"word": word, "start": start, "end": end, "speaker": speaker}


def test_a_short_answer_of_another_speaker_stays_its_own():
    segs = [
        {
            "start": 0.0,
            "end": 2.0,
            "text": "Are you coming tonight?",
            "speaker": "SPEAKER_00",
            "words": [
                _w("Are", 0.0, 0.4, "SPEAKER_00"),
                _w("you", 0.4, 0.8, "SPEAKER_00"),
                _w("coming", 0.8, 1.4, "SPEAKER_00"),
                _w("tonight?", 1.4, 2.0, "SPEAKER_00"),
            ],
        },
        {
            "start": 2.02,
            "end": 2.3,
            "text": "Yes.",
            "speaker": "SPEAKER_01",
            "words": [_w("Yes.", 2.02, 2.3, "SPEAKER_01")],
        },
    ]
    out = word_boundary_segmentation(segs, "en")
    assert [(s["speaker"], s["text"]) for s in out] == [
        ("SPEAKER_00", "Are you coming tonight?"),
        ("SPEAKER_01", "Yes."),
    ]


def test_a_long_segment_is_split_where_the_speaker_changes():
    words = [_w(f"a{i}", i * 0.3, i * 0.3 + 0.28, "SPEAKER_00") for i in range(6)] + [
        _w(f"b{i}", 1.8 + i * 0.3, 1.8 + i * 0.3 + 0.28, "SPEAKER_01") for i in range(6)
    ]
    segs = [
        {
            "start": 0.0,
            "end": 3.6,
            "text": " ".join(w["word"] for w in words),
            "speaker": "SPEAKER_00",
            "words": words,
        }
    ]
    out = word_boundary_segmentation(segs, "en")
    for s in out:
        assert {w["speaker"] for w in s["words"]} == {s["speaker"]}
    assert [s["speaker"] for s in out] == ["SPEAKER_00", "SPEAKER_01"]


def test_without_speakers_short_cues_still_merge():
    segs = [
        {
            "start": 0.0,
            "end": 2.0,
            "text": "Hello there friend.",
            "speaker": "—",
            "words": [
                _w("Hello", 0.0, 0.6, None),
                _w("there", 0.6, 1.2, None),
                _w("friend.", 1.2, 2.0, None),
            ],
        },
        {
            "start": 2.02,
            "end": 2.3,
            "text": "Hm.",
            "speaker": "—",
            "words": [_w("Hm.", 2.02, 2.3, None)],
        },
    ]
    out = word_boundary_segmentation(segs, "en")
    assert len(out) == 1 and out[0]["text"] == "Hello there friend. Hm."
