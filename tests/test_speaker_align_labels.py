"""
Voices-only runs diarize again; their labels are renamed after the existing
transcript's speakers so speaker_vectors.label and segments.speaker agree —
the audio, the turns and the windows stay as the diarization delivered them.
"""

from whisper_tm.core.speaker import align_labels


def T(sp, a, b):
    return {"speaker": sp, "start": float(a), "end": float(b)}


def test_swapped_labels_are_renamed():
    turns = [T("SPEAKER_00", 0, 10), T("SPEAKER_01", 10, 20), T("SPEAKER_00", 20, 25)]
    tx = [("SPEAKER_01", 0, 10), ("SPEAKER_00", 10, 20), ("SPEAKER_01", 20, 25)]
    out, info = align_labels(turns, tx)
    assert [t["speaker"] for t in out] == ["SPEAKER_01", "SPEAKER_00", "SPEAKER_01"]
    assert info["aligned"] == {"SPEAKER_00": "SPEAKER_01", "SPEAKER_01": "SPEAKER_00"}
    assert info["unmatched"] == {}
    # times untouched
    assert [(t["start"], t["end"]) for t in out] == [(0, 10), (10, 20), (20, 25)]


def test_identical_labels_stay():
    turns = [T("SPEAKER_00", 0, 5), T("SPEAKER_01", 5, 9)]
    tx = [("SPEAKER_00", 0, 5), ("SPEAKER_01", 5, 9)]
    out, info = align_labels(turns, tx)
    assert [t["speaker"] for t in out] == ["SPEAKER_00", "SPEAKER_01"]
    assert info["aligned"] == {"SPEAKER_00": "SPEAKER_00", "SPEAKER_01": "SPEAKER_01"}


def test_extra_speaker_gets_a_free_label_not_someone_elses():
    # new SPEAKER_02 talks where the transcript has SPEAKER_00 — but
    # SPEAKER_00 is already the counterpart of the new SPEAKER_01
    turns = [T("SPEAKER_01", 0, 30), T("SPEAKER_02", 30, 34), T("SPEAKER_00", 40, 60)]
    tx = [("SPEAKER_00", 0, 34), ("SPEAKER_01", 40, 60)]
    out, info = align_labels(turns, tx)
    assert info["aligned"] == {"SPEAKER_01": "SPEAKER_00", "SPEAKER_00": "SPEAKER_01"}
    assert info["unmatched"] == {"SPEAKER_02": "SPEAKER_02"}
    labels = [t["speaker"] for t in out]
    assert labels == ["SPEAKER_00", "SPEAKER_02", "SPEAKER_01"]


def test_free_label_never_collides_with_transcript_labels():
    turns = [T("SPEAKER_00", 0, 10), T("SPEAKER_01", 50, 60)]
    tx = [("SPEAKER_00", 0, 10), ("SPEAKER_01", 20, 30), ("SPEAKER_02", 30, 40)]
    out, info = align_labels(turns, tx)
    assert info["unmatched"] == {"SPEAKER_01": "SPEAKER_03"}
    assert out[1]["speaker"] == "SPEAKER_03"


def test_no_majority_means_no_link():
    # the new speaker is split 50/50 between two transcript speakers
    turns = [T("SPEAKER_00", 0, 20)]
    tx = [("SPEAKER_00", 0, 9), ("SPEAKER_01", 9, 20)]
    out, info = align_labels(turns, tx)
    assert info["aligned"] == {"SPEAKER_00": "SPEAKER_01"}  # 11 s of 20 = majority
    turns = [T("SPEAKER_00", 0, 30)]
    tx = [("SPEAKER_00", 0, 10), ("SPEAKER_01", 10, 20), ("SPEAKER_02", 20, 30)]
    out, info = align_labels(turns, tx)
    assert info["aligned"] == {}
    assert info["unmatched"] == {"SPEAKER_00": "SPEAKER_03"}


def test_transcript_without_speakers_changes_nothing():
    turns = [T("SPEAKER_00", 0, 10), T("SPEAKER_01", 10, 20)]
    out, info = align_labels(turns, [("—", 0, 10), ("", 10, 20)])
    assert info == {}
    assert [t["speaker"] for t in out] == ["SPEAKER_00", "SPEAKER_01"]
    out, info = align_labels(turns, [])
    assert info == {}


def test_input_turns_are_not_mutated():
    turns = [T("SPEAKER_00", 0, 10)]
    align_labels(turns, [("SPEAKER_05", 0, 10)])
    assert turns[0]["speaker"] == "SPEAKER_00"
