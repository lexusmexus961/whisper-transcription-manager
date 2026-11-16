"""The quotation marks Whisper puts around a line (core/helpers.edge_quotes).

Whisper wraps a line that sounds like dialogue in guillemets (»…«), a habit
from books and subtitles. At a segment's edge they are noise, and the voice
extractor would write them into a TTS dataset. They go at the edge — in new
transcripts (fmt_seg) and in the extractor's text; a quote inside a sentence
stays.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from whisper_tm.core.helpers import edge_quotes, fmt_seg

REPO = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize(
    "text, expected",
    [
        ("»Call Bob today.", ("Call Bob today.", "»", "")),
        ("»The report is ready for review.«", ("The report is ready for review.", "»", "«")),
        ("that the test passed again.«", ("that the test passed again.", "", "«")),
        (" » Good. ", ("Good.", "»", "")),
        ("‹Yes.›", ("Yes.", "‹", "›")),
        # A quote inside the sentence is the sentence's own
        ("He called it »the draft«", ("He called it »the draft«", "", "")),
        ("»Draft« is fine.", ("»Draft« is fine.", "", "")),
        ("He said: »Come.«", ("He said: »Come.«", "", "")),
        ("»Come here«, he said.", ("»Come here«, he said.", "", "")),
        # Nothing would be left — it stays as it is
        ("»", ("»", "", "")),
        ("«", ("«", "", "")),
        ("Normal.", ("Normal.", "", "")),
        ("", ("", "", "")),
    ],
)
def test_the_rule(text, expected):
    assert edge_quotes(text) == expected


class TestTheWords:
    def test_the_first_and_the_last_word_lose_the_mark(self):
        seg = {
            "start": 0.0,
            "end": 1.2,
            "text": "»Call Bob today.«",
            "words": [
                {"word": "»Call", "start": 0.1},
                {"word": "Bob"},
                {"word": "today.«", "end": 1.2},
            ],
        }
        out = fmt_seg(seg, include_words=True)
        assert out["text"] == "Call Bob today."
        assert [w["word"] for w in out["words"]] == ["Call", "Bob", "today."]
        assert out["words"][0]["start"] == 0.1 and out["words"][-1]["end"] == 1.2

    def test_a_mark_that_was_a_word_of_its_own_goes_with_it(self):
        seg = {
            "start": 0.0,
            "end": 1.0,
            "text": "» Good.",
            "words": [{"word": "»"}, {"word": "Good."}],
        }
        assert [w["word"] for w in fmt_seg(seg, include_words=True)["words"]] == ["Good."]

    def test_a_line_without_marks_is_untouched(self):
        seg = {
            "start": 0.0,
            "end": 1.0,
            "text": "He said: »Come.«",
            "words": [{"word": "He"}, {"word": "»Come.«"}],
        }
        out = fmt_seg(seg, include_words=True)
        assert out["text"] == "He said: »Come.«" and out["words"][-1]["word"] == "»Come.«"


def test_new_transcripts_come_without_them():
    seg = {
        "start": 0.0,
        "end": 2.0,
        "speaker": "SPEAKER_00",
        "text": "»But the budget still needs a second look,",
        "words": [
            {"word": "»But", "start": 0.0, "end": 0.3, "score": 0.9},
            {"word": "look,", "start": 1.5, "end": 2.0, "score": 0.8},
        ],
    }
    out = fmt_seg(seg, include_words=True)
    assert out["text"] == "But the budget still needs a second look,"
    assert [w["word"] for w in out["words"]] == ["But", "look,"]
    # The pipeline's own segment is not changed underneath it
    assert seg["text"].startswith("»") and seg["words"][0]["word"] == "»But"
    assert fmt_seg(seg)["text"] == "But the budget still needs a second look,"


def test_the_extractor_takes_the_sentence_without_them(tmp_path, monkeypatch):
    import whisper_tm.extensions as ext
    from whisper_tm.core import voice_extract as vx
    from whisper_tm.database import Database

    db = Database(tmp_path / "lib.db")
    db.initialize()
    monkeypatch.setattr(ext, "db", db)
    abs_path = str(tmp_path / "line_0001.wav")
    db.upsert_transcript(
        "line_0001.wav",
        {
            "complete": True,
            "segments": [
                {"speaker": "SPEAKER_00", "start": 0.0, "end": 2.0, "text": "»Call Bob today."},
                {"speaker": "SPEAKER_00", "start": 2.0, "end": 4.0, "text": "He said: »Come.«"},
            ],
        },
        media_dir=str(tmp_path),
        abs_path=abs_path,
    )
    got = vx._sentences({(abs_path, "SPEAKER_00")})[(abs_path, "SPEAKER_00")]
    assert [s["text"] for s in got] == ["Call Bob today.", "He said: »Come.«"]
