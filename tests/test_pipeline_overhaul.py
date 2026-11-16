"""Tests for the pipeline overhaul.

Covers the new components that do NOT need a GPU:
  - ASR block formation (one piece / giant blocks with overlap)
  - Midpoint dedup criterion
  - Post-processing: Phantom score, overlapping intervals
  - Duration plausibility filter of scanner
  - Segment markers survive fmt_seg and DB round
"""

import wave
from pathlib import Path

import pytest

from whisper_tm.core.postprocessing import (
    mark_segments,
    overlapping_intervals,
    phantom_score,
)
from whisper_tm.scanner.indexer import _plausible_duration

# ══════════════════════════════════════════════════════════════════════════
#  ASR block formation (no hard cuts anymore)
# ══════════════════════════════════════════════════════════════════════════


def _silent_wav(path: Path, seconds: float, rate: int = 16000):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * int(seconds * rate))


class TestASRBlocks:
    def test_short_file_runs_in_one_piece(self, tmp_path):
        from whisper_tm.core.transcription import _asr_blocks

        wav = tmp_path / "short.wav"
        _silent_wav(wav, 2.0)
        blocks = _asr_blocks(wav, tmp_path, total_dur=2.0, block_sec=54000, overlap_sec=30.0)
        assert blocks == [(wav, 0.0, 0.0, float("inf"))], (
            "Files below the limit must NOT be split — exactly the 600-s split cut sentences"
        )

    @pytest.mark.ffmpeg
    def test_long_file_gets_overlapped_blocks(self, tmp_path):
        from whisper_tm.core.transcription import _asr_blocks

        wav = tmp_path / "long.wav"
        _silent_wav(wav, 25.0)
        # Mini scale: 10-s blocks with 2 s overlap on 25 s audio
        blocks = _asr_blocks(wav, tmp_path, total_dur=25.0, block_sec=10, overlap_sec=2.0)
        assert len(blocks) == 3
        # Block 2 starts logically at 10 s, but its audio already at 8 s —
        # speech above the boundary lies completely in one of the blocks.
        _, log_start, ext_start, log_end = blocks[1]
        assert log_start == 10.0
        assert ext_start == 8.0
        assert log_end == 20.0
        # Last block is open at the top (no segment falls out the back)
        assert blocks[-1][3] == float("inf")
        # All block files exist and contain audio
        for bp, *_rest in blocks:
            assert bp.exists() and bp.stat().st_size > 1000

    def test_midpoint_dedup_assigns_uniquely(self):
        # The criterion from the transcription loop, recomputed in isolation:
        # a boundary segment (9.5–11.5 s, midpoint 10.5) belongs to block 2
        # [10, 20), NOT to block 1 [0, 10) — and only to exactly one.
        seg = {"start": 9.5, "end": 11.5}
        midpoint = (seg["start"] + seg["end"]) / 2
        assignment = [i for i, (lo, hi) in enumerate([(0, 10), (10, 20)]) if lo <= midpoint < hi]
        assert assignment == [1]


# ══════════════════════════════════════════════════════════════════════════
#  Post-processing
# ══════════════════════════════════════════════════════════════════════════


class TestPhantomScore:
    def test_calibrated_phantoms_are_recognized(self):
        # Known large-v3 hallucination artifacts (German subtitles)
        for text in (
            "Vielen Dank.",
            "Bis zum nächsten Mal.",
            "Vielen, vielen Dank.",
            "Untertitelung im Auftrag des ZDF, 2017",
        ):
            assert phantom_score(text, None, None) > 0, text

    def test_real_speech_remains_unmarked(self):
        for text in (
            "The meeting starts at nine.",
            "Thanks, that was great!",
            "Thank you for the flowers, Bob.",
            "We need more chairs.",
        ):
            assert phantom_score(text, "SPEAKER_00", -0.2) == 0.0, text

    def test_gates_raise_score(self):
        # Observation: hallucinations mostly have NO speaker
        with_speaker = phantom_score("Vielen Dank.", "SPEAKER_01", -0.1)
        without_speaker = phantom_score("Vielen Dank.", None, -0.1)
        uncertain_too = phantom_score("Vielen Dank.", None, -1.5)
        assert with_speaker < without_speaker < uncertain_too
        # Hit WITH speaker stays below 0.5 — real thank yous exist
        assert with_speaker < 0.5

    def test_score_is_reversible_only_marker(self):
        seg = {"text": "Vielen Dank.", "speaker": "—", "start": 1.0, "end": 2.0}
        mark_segments([seg])
        assert seg["text"] == "Vielen Dank.", "Text must NEVER change"
        assert seg["phantom_score"] > 0


class TestOverlapped:
    def test_intervals_from_diarization(self):
        import pandas as pd

        # Two speakers overlap from 5–8 s
        df = pd.DataFrame(
            [
                {"start": 0.0, "end": 8.0, "speaker": "A"},
                {"start": 5.0, "end": 12.0, "speaker": "B"},
            ]
        )
        iv = overlapping_intervals(df)
        assert len(iv) == 1
        assert iv[0] == pytest.approx((5.0, 8.0))

    def test_a_segment_in_the_overlap_is_marked(self):
        import pandas as pd

        df = pd.DataFrame(
            [
                {"start": 0.0, "end": 8.0, "speaker": "A"},
                {"start": 5.0, "end": 12.0, "speaker": "B"},
            ]
        )
        inside = {"text": "jumbled", "start": 5.5, "end": 7.5, "speaker": "A"}
        outside = {"text": "alone", "start": 9.0, "end": 11.0, "speaker": "B"}
        stats = mark_segments([inside, outside], df)
        assert inside.get("overlapped") == 1
        assert "overlapped" not in outside
        assert stats["overlapped"] == 1

    def test_no_crash_without_diarization(self):
        seg = {"text": "hello", "start": 0.0, "end": 1.0}
        stats = mark_segments([seg], None)
        assert stats["overlapped"] == 0


# ══════════════════════════════════════════════════════════════════════════
#  Scanner plausibility
# ══════════════════════════════════════════════════════════════════════════


class TestPlausibleDuration:
    def test_an_int32_overflow_is_discarded(self):
        # A tiny 500-byte FLV claiming 2,147,483.647 s
        assert _plausible_duration(2147483.647, 500) == 0.0

    def test_normal_files_stay_untouched(self):
        assert _plausible_duration(3600.0, 50_000_000) == 3600.0

    def test_an_extremely_compressed_voice_note_survives(self):
        # 60 s mono-opus with ~6 kbps ≈ 45 kB — must survive the threshold
        assert _plausible_duration(60.0, 45_000) == 60.0

    def test_without_a_duration_no_effect(self):
        assert _plausible_duration(0, 12345) == 0


# ══════════════════════════════════════════════════════════════════════════
#  Markers survive fmt_seg and the DB round
# ══════════════════════════════════════════════════════════════════════════


class TestMarkerPersistence:
    def test_fmt_seg_passes_markers_through(self):
        from whisper_tm.core.helpers import fmt_seg

        seg = {
            "start": 1.0,
            "end": 2.0,
            "text": "Vielen Dank.",
            "speaker": "—",
            "avg_logprob": -1.234,
            "phantom_score": 0.85,
            "overlapped": 1,
        }
        f = fmt_seg(seg)
        assert f["avg_logprob"] == -1.234
        assert f["phantom_score"] == 0.85
        assert f["overlapped"] == 1

    def test_fmt_seg_without_markers_is_unchanged(self):
        # Old transcripts/tests: without markers the output does NOT change
        from whisper_tm.core.helpers import fmt_seg

        f = fmt_seg({"start": 1.0, "end": 2.0, "text": "hello"})
        assert "avg_logprob" not in f
        assert "phantom_score" not in f
        assert "overlapped" not in f

    def test_the_db_stores_and_reads_markers(self, tmp_path):
        from whisper_tm.database import Database

        db = Database(tmp_path / "t.db")
        db.initialize()
        db.upsert_transcript(
            "a/b.mp4",
            {
                "segments": [
                    {
                        "speaker": "—",
                        "start": 0.0,
                        "end": 1.0,
                        "start_fmt": "0:00",
                        "end_fmt": "0:01",
                        "text": "Vielen Dank.",
                        "avg_logprob": -0.9,
                        "phantom_score": 0.7,
                        "overlapped": 1,
                    },
                ],
                "complete": True,
            },
        )
        with db.connection() as conn:
            row = conn.execute(
                "SELECT avg_logprob, phantom_score, overlapped FROM segments"
            ).fetchone()
        assert row["avg_logprob"] == pytest.approx(-0.9)
        assert row["phantom_score"] == pytest.approx(0.7)
        assert row["overlapped"] == 1
