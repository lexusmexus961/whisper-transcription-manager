"""The audio sample must play what the vector was created from.

Listening to the suggestions showed groups of clearly different speakers —
several people from different recordings in one group. The model, meanwhile,
held the group's centroids as nearly identical (far closer than random
voices).

Both were right. A window is 30 seconds of NET speech, glued from pieces
distributed across the entire recording; only the START was stored. The audio
sample cut 30 seconds at a time from there — and thus almost never the vector's
material:

    listened: the first 30 seconds — often without the voice in question
    span:     the window's pieces — spread over many minutes, sometimes hours

So groups were decided based on audio the procedure had never seen.
Now the spans are in the `parts` column, and ffmpeg glues exactly
them — with the same plan that `build_windows` creates.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from whisper_tm.api.speaker import (
    MAX_PARTS,
    MONTAGE_PIECES,
    _audio_sample_parts,
    _montage_from_spots,
    _montage_parts,
    _sample_command,
)
from whisper_tm.database.speaker import SpeakerDB


class TestReadingSpans:
    def test_stored_spans_come_back(self):
        assert _audio_sample_parts("[[10,15],[100,130.5]]") == [(10.0, 15.0), (100.0, 130.5)]

    def test_old_rows_have_none_and_that_is_no_error(self):
        """Windows from before the column existed have none. For them it stays
        the old way — an exception would be worst here."""
        for raw in (None, "", "null", "[]"):
            assert _audio_sample_parts(raw) == []

    def test_broken_values_cost_no_audio_sample(self):
        for raw in ("broken", "{}", "[[1]]", "[[5,5]]", "[[9,2]]", "[[-3,-1]]"):
            assert _audio_sample_parts(raw) == [], raw

    def test_too_many_pieces_are_capped(self):
        """Each piece is a separate ffmpeg input; the command line grows
        linearly with it."""
        raw = json.dumps([[i * 10, i * 10 + 3] for i in range(60)])
        assert len(_audio_sample_parts(raw)) == MAX_PARTS

    def test_capped_by_length_emitted_by_time(self):
        """The longest pieces carry the vector most strongly — but an edit in
        jump order would be hard to follow."""
        raw = json.dumps(
            [[0, 1], [100, 120], [200, 201], [300, 330]]
            + [[400 + i, 400.5 + i] for i in range(MAX_PARTS)]
        )
        t = _audio_sample_parts(raw)
        assert (100.0, 120.0) in t and (300.0, 330.0) in t, "long pieces missing"
        assert t == sorted(t), "not in chronological order"


class TestTheFfmpegCall:
    def test_one_input_per_piece(self):
        """`-ss` BEFORE `-i` jumps in the container, instead of decoding from
        the start. With a recording over six hours, that is the whole difference."""
        b = _sample_command(Path("/x/a.mp4"), Path("/t/o.mp3"), [(10, 15), (100, 130)], 0, 30)
        assert b.count("-i") == 2
        for timestamp, duration in (("10.000", "5.000"), ("100.000", "30.000")):
            k = b.index(timestamp)
            assert b[k - 1] == "-ss", "jump marker missing"
            assert b[k + 1] == "-t" and b[k + 2] == duration, b[k - 1 : k + 4]
            # the input's options, then -i: -format_whitelist belongs to it too
            assert b[k + 3 : k + 6] == ["-format_whitelist", b[k + 4], "-i"], "-ss not BEFORE -i"
        assert "concat=n=2:v=0:a=1[out]" in " ".join(b)

    def test_without_pieces_the_old_cut_stays(self):
        b = _sample_command(Path("/x/a.mp4"), Path("/t/o.mp3"), [], 42.0, 30.0)
        assert b.count("-i") == 1
        assert "-filter_complex" not in b
        assert "42.000" in b and "30.000" in b

    def test_the_output_format_is_always_stated(self):
        """`-f mp3` is not optional: written to `.mp3.tmp`, and ffmpeg cannot
        guess the format from that extension (rc=234)."""
        for parts in ([], [(1, 2)]):
            b = _sample_command(Path("/x/a.mp4"), Path("/t/o.mp3.tmp"), parts, 0, 30)
            assert b[-3:-1] == ["-f", "mp3"], b[-4:]


class TestStorage:
    @pytest.fixture
    def db(self, tmp_path):
        d = SpeakerDB(tmp_path / "s.db")
        d.initialize()
        return d

    def _line(self, parts=None):
        return {
            "label": "SPEAKER_00",
            "windows": 1,
            "start_sec": 10.0,
            "speech_sec": 30.0,
            "segments": 3,
            "parts": parts,
            "v_separated": np.ones(8, dtype=np.float32),
        }

    def test_the_spans_are_stored(self, db):
        db.write("/a.mp4", "M", [self._line([(1.0, 5.0), (9.0, 35.5)])])
        with db.connection() as c:
            raw = c.execute("SELECT parts FROM speaker_vectors").fetchone()[0]
        assert _audio_sample_parts(raw) == [(1.0, 5.0), (9.0, 35.5)]

    def test_without_spans_the_column_stays_empty(self, db):
        db.write("/a.mp4", "M", [self._line(None)])
        with db.connection() as c:
            assert c.execute("SELECT parts FROM speaker_vectors").fetchone()[0] is None

    def test_two_decimal_places_are_enough(self, db):
        """ffmpeg cannot cut finer anyway, and with a large library at 18 pieces
        per window, it more than halves the size of the column."""
        db.write("/a.mp4", "M", [self._line([(1.23456, 5.98765)])])
        with db.connection() as c:
            raw = c.execute("SELECT parts FROM speaker_vectors").fetchone()[0]
        assert raw == "[[1.23,5.99]]", raw


def test_the_window_plan_delivers_the_spans_at_all():
    """The source of truth: `build_windows` knows the pieces long ago,
    `plan_windows` passes them through. They were only thrown away when
    writing — and exactly this gap was the original error."""
    from whisper_tm.core import speaker as speaker_core

    turns = [{"start": t, "end": t + 6.0, "speaker": "SPEAKER_00"} for t in range(0, 600, 60)]
    plan = speaker_core.plan_windows(turns)
    assert plan, "no window planned"
    p = plan[0]
    assert p["parts"], "the plan does not carry the pieces"
    assert abs(sum(b - a for a, b in p["parts"]) - p["speech_sec"]) < 0.05, (
        "the pieces do not add up to net speech time"
    )
    assert p["parts"][0][0] == p["start_sec"], "start does not match the plan"


class TestMontageForOldWindows:
    """The workaround for windows without stored spans.

    Recalculating them would mean repeating separation and diarization: days
    of GPU time for a large inventory, for better listening, not better
    data. From transcript segments, only a small share of windows could be
    reconstructed.

    The montage covers the same STRETCH the vector came from, just not the
    same spots: six samples across the span instead of the first 30 seconds.
    """

    def test_the_montage_covers_the_whole_span(self):
        t = _montage_parts(0, 2400, 30)  # 40 minutes
        assert t[0][0] == 0.0
        assert abs(t[-1][1] - 2400) < 0.1, "end of span missing"
        assert len(t) == 6

    def test_it_lasts_as_long_as_the_window(self):
        """30 seconds of audio stays 30 seconds of audio — just distributed."""
        for span, duration in ((2400, 30), (24000, 30), (600, 25)):
            t = _montage_parts(0, span, duration)
            assert abs(sum(b - a for a, b in t) - duration) < 0.5, (span, duration)

    def test_the_pieces_do_not_overlap(self):
        t = _montage_parts(100, 5000, 30)
        for (a1, b1), (a2, b2) in zip(t, t[1:]):
            assert b1 <= a2 + 0.01, t

    def test_no_piece_is_too_short_to_recognize(self):
        """Below three seconds you can't hear tone and timbre reliably — then
        fewer cuts are better."""
        for span in (60, 200, 2400, 24000):
            for a, b in _montage_parts(0, span, 30):
                assert b - a >= 3.0 - 1e-6, (span, a, b)

    def test_for_a_short_span_it_is_not_worth_it(self):
        """If the span is barely longer than the window, the head cut IS
        already almost everything — then stay with the old way."""
        assert _montage_parts(0, 40, 30) == []
        assert _montage_parts(0, 30, 30) == []
        assert _montage_parts(0, 0, 30) == []

    def test_it_fits_into_the_same_ffmpeg_call(self):
        """No second editing path: montage and real spans go through the same
        function — otherwise the two drift apart."""
        t = _montage_parts(0, 2400, 30)
        b = _sample_command(Path("/x/a.mp4"), Path("/t/o.mp3"), t, 0, 30)
        assert b.count("-i") == len(t)
        assert f"concat=n={len(t)}" in " ".join(b)


class TestMontageAimsAtSpeech:
    """The blind montage was not enough.

    30 seconds of speech can span 40 minutes, so 1.25% of the distance. Six evenly
    spaced cuts therefore almost always land in silence or with other people,
    rarely on the voice in question.

    The transcript segments tell us WHEN this speaker spoke. Targeted, the
    samples land on the voice in question where the blind cuts missed it.
    """

    def test_the_samples_sit_on_the_speech_spots(self):
        spots = [
            (10.0, 14.0),
            (500.0, 508.0),
            (900.0, 903.0),
            (1400.0, 1412.0),
            (2000.0, 2004.0),
            (2200.0, 2260.0),
        ]
        t = _montage_from_spots(spots, 30)
        for a, b in t:
            assert any(s <= a < e for s, e in spots), (a, b)

    def test_they_spread_over_the_whole_speech(self):
        """Evenly across the LIST, not across time: that gives you beginning,
        middle and end, even if the positions are uneven."""
        spots = [(float(i), float(i) + 4.0) for i in range(0, 600, 10)]
        t = _montage_from_spots(spots, 30)
        assert len(t) == MONTAGE_PIECES
        assert t[0][0] < 60 and t[-1][0] > 480, t

    def test_every_sample_starts_at_a_word_boundary(self):
        """At the start of the position, not in the middle — there is a word
        there, not half a vowel."""
        spots = [(100.0, 130.0), (300.0, 330.0)]
        for a, _ in _montage_from_spots(spots, 20):
            assert a in (100.0, 300.0), a

    def test_two_samples_never_overlap(self):
        """With few, long positions, two samples can fall into the same —
        then one stays."""
        t = _montage_from_spots([(0.0, 300.0)], 30)
        for (a1, b1), (a2, b2) in zip(t, t[1:]):
            assert b1 <= a2, t

    def test_without_speech_spots_the_blind_montage_remains(self):
        """For recordings without transcript segments — the fallback must be
        there, otherwise there would be no sample at all."""
        assert _montage_from_spots([], 30) == []
        assert _montage_parts(0, 2400, 30), "blind path missing"
