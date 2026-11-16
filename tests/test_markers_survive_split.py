"""Markers must survive the word boundary split.

The error that these tests pin down went unnoticed for a long time: `word_boundary_segmentation`
builds NEW dicts with five keys in its third phase — the fourth phase and the
two before it modify the existing ones in place. `avg_logprob`, `phantom_score`
and `overlapped` fell out exactly at this one point.

The database showed it all too clearly, without anyone noticing:

===================  ============  =============  ============
Run                  avg_logprob   phantom_score  overlapped
===================  ============  =============  ============
with diarization     never set     never set      never set
without diarization  mostly set    sometimes set  never set
===================  ============  =============  ============

The difference is exactly this split: diarization forces alignment, alignment
triggers the split. `overlapped` never had a single match in the entire dataset,
because it can't exist without diarization and was always discarded with
diarization.

Two guarantees follow from this, and they are NOT the same:

1. **`avg_logprob` must be inherited** — it comes from the decoder and cannot
   be recalculated after cutting.
2. **`phantom_score` and `overlapped` must NOT be inherited** — they apply to
   the old boundaries. After splitting they must be determined anew.
"""

import pytest

from whisper_tm.core.postprocessing import mark_segments
from whisper_tm.core.segmentation import word_boundary_segmentation


def _word(w, a, b):
    return {"word": w, "start": a, "end": b, "score": 0.9}


def _segment(text, a, b, words, **remainder):
    return {
        "start": a,
        "end": b,
        "text": text,
        "speaker": "SPEAKER_00",
        "words": words,
        **remainder,
    }


# ── What MUST be inherited ───────────────────────────────────────────────


def test_avg_logprob_survives_the_split():
    """From the decoder, not recalculable after cutting."""
    words = [_word(f"w{i}", i * 1.0, i * 1.0 + 0.8) for i in range(14)]
    seg = _segment(" ".join(w["word"] for w in words), 0.0, 14.0, words, avg_logprob=-0.42)
    out = word_boundary_segmentation([seg], "de")
    assert len(out) > 1, "The split didn't split at all — test is worthless"
    assert all(s.get("avg_logprob") == -0.42 for s in out), (
        f"avg_logprob lost: {[s.get('avg_logprob') for s in out]}"
    )


def test_without_avg_logprob_no_key_appears():
    """Old transcripts and tests should remain unchanged."""
    words = [_word(f"w{i}", i * 1.0, i * 1.0 + 0.8) for i in range(14)]
    out = word_boundary_segmentation([_segment("x " * 14, 0.0, 14.0, words)], "de")
    assert all("avg_logprob" not in s for s in out)


# ── What MUST NOT be inherited ────────────────────────────────────────────


def test_overlap_is_not_inherited_blindly():
    """Otherwise every subtitle carries the parent segment's marker.

    A 20-second segment with 7 seconds of overlap counts as overlapped. If it
    becomes five subtitles, not all five may inherit the marker — four of them
    don't touch the overlap at all.
    """
    words = [_word(f"w{i}", i * 1.0, i * 1.0 + 0.8) for i in range(14)]
    seg = _segment("x " * 14, 0.0, 14.0, words, overlapped=1, phantom_score=0.85)
    out = word_boundary_segmentation([seg], "de")
    assert all("overlapped" not in s for s in out)
    assert all("phantom_score" not in s for s in out)


def test_markers_after_the_split_hit_the_right_parts():
    """Order is the actual substance of the repair.

    Only the FIRST third of the recording overlaps here. After the split, only
    the subtitle at that location should be marked.
    """
    pd = pytest.importorskip("pandas")
    words = [_word(f"w{i}", i * 1.0, i * 1.0 + 0.8) for i in range(14)]
    seg = _segment("x " * 14, 0.0, 14.0, words, avg_logprob=-0.5)

    # two speakers overlapping from 0 to 4 s
    diar = pd.DataFrame(
        [{"start": 0.0, "end": 4.0, "speaker": "A"}, {"start": 0.0, "end": 4.0, "speaker": "B"}]
    )

    out = word_boundary_segmentation([seg], "de")
    mark_segments(out, diar)

    early = [s for s in out if s["end"] <= 4.0]
    late = [s for s in out if s["start"] >= 5.0]
    assert early and late, "split didn't divide as expected"
    assert all(s.get("overlapped") == 1 for s in early)
    assert all("overlapped" not in s for s in late), "segments outside overlap were also marked"


def test_the_phantom_score_judges_the_cut_text():
    """ "Thank you." as its own subtitle is suspicious — the same two words in
    real speech are not."""
    long_seg = _segment(
        "also ich habe das gestern gemacht und dann vielen dank gesagt und "
        "danach sind wir gegangen",
        0.0,
        16.0,
        [
            _word(w, i * 1.0, i * 1.0 + 0.8)
            for i, w in enumerate(
                "also ich habe das gestern gemacht und dann "
                "vielen dank gesagt und danach sind wir "
                "gegangen".split()
            )
        ],
        avg_logprob=-0.5,
    )
    stats = mark_segments([dict(long_seg)], None)
    assert stats["phantom"] == 0, "in running text it's not a phantom"

    short = _segment(
        "Vielen Dank.", 0.0, 1.2, [_word("Vielen", 0.0, 0.6), _word("Dank.", 0.6, 1.2)]
    )
    short["speaker"] = "—"  # no speaker: diarization found nothing
    stats2 = mark_segments([short], None)
    assert stats2["phantom"] == 1
    assert short["phantom_score"] >= 0.7


# ── And the path to output ───────────────────────────────────────────────────


def test_fmt_seg_passes_the_markers_on():
    from whisper_tm.core.helpers import fmt_seg

    s = fmt_seg(
        {
            "start": 0.0,
            "end": 1.0,
            "text": "x",
            "speaker": "A",
            "avg_logprob": -0.4321,
            "phantom_score": 0.7,
            "overlapped": 1,
        }
    )
    assert s["avg_logprob"] == -0.432
    assert s["phantom_score"] == 0.7
    assert s["overlapped"] == 1


def test_marking_segments_changes_only_the_markers():
    """The guarantee that makes refactoring safe: text, times, speaker, and
    words are left untouched."""
    words = [_word("Vielen", 0.0, 0.6), _word("Dank.", 0.6, 1.2)]
    seg = _segment("Vielen Dank.", 0.0, 1.2, words)
    seg["speaker"] = "—"
    before = {k: seg[k] for k in ("start", "end", "text", "speaker", "words")}
    mark_segments([seg], None)
    for k, v in before.items():
        assert seg[k] == v, f"{k} was changed"
    assert "phantom_score" in seg


# ── Diarization without HuggingFace Token ────────────────────────────────


def test_diarization_works_without_a_token_when_the_model_is_there(monkeypatch):
    """The token is ONLY needed for downloading.

    Previously, everywhere it said `diarize and hf_token`. Without token,
    diarization didn't happen — the banner still reported "Diarization: yes",
    phase count dropped by one, and the transcript came back without speakers.
    No error, no warning, nothing.
    """
    from whisper_tm.core.transcription import diarization_possible

    monkeypatch.setattr("huggingface_hub.snapshot_download", lambda *a, **k: "/cache/pyannote")
    assert diarization_possible(None) is True
    assert diarization_possible("") is True


def test_without_a_token_and_without_a_cache_it_is_impossible(monkeypatch):
    from whisper_tm.core.transcription import diarization_possible

    def raises(*a, **k):
        raise OSError("not in cache")

    monkeypatch.setattr("huggingface_hub.snapshot_download", raises)
    assert diarization_possible(None) is False
    # a token saves it: then download is allowed
    assert diarization_possible("hf_xyz") is True


# ── Tool detection: check what is actually used ───────────────────────────


def test_audio_separator_is_detected_through_the_package(monkeypatch):
    """Detection must ask the same as usage.

    `vocal_isolation.py` does `from audio_separator.separator import Separator`.
    Detection used to call only the COMMAND `audio-separator` — that's
    in `.venv/bin/` and only in PATH when the environment is activated. If the
    server starts via the interpreter path, it reported "not installed" even
    though the import works: the run aborted while the same library in `tools/`
    separated fine.
    """
    from whisper_tm.hardware import HardwareProfile, _detect_tools

    def no_command(*a, **k):
        raise FileNotFoundError("audio-separator not in PATH")

    monkeypatch.setattr("subprocess.run", no_command)

    p = HardwareProfile()
    _detect_tools(p)
    assert p.has_audio_separator is True, "package importable but detection says no"
