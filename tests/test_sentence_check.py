"""
The sentence check (core/speaker.check_sentences): after diarization each
sentence is measured on its own against the voices of the same recording
and moves to another voice only when that one is clearly it.

The failure it targets: two people trading short sentences that diarization
keeps under one label — every sentence in that stretch gets the name of
whoever the label is.
"""

import inspect
import re

import numpy as np
import pytest

from whisper_tm.core import speaker as sp


def _flat(text: str) -> str:
    """Source text with the formatter's line breaks taken out: one space
    for every run of whitespace, none inside brackets, no trailing commas."""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r",\s*([)\]}])", r"\1", text)
    text = re.sub(r"([(\[{])\s+", r"\1", text)
    return re.sub(r"\s+([)\]}])", r"\1", text)


def _unit(*xs):
    v = np.asarray(xs, np.float32)
    return v / np.linalg.norm(v)


#: Three voices, as the recording's centroids
VOICES = {"SPEAKER_00": _unit(1, 0, 0), "SPEAKER_01": _unit(0, 1, 0), "SPEAKER_02": _unit(0, 0, 1)}


def _audio(spans):
    """Audio whose sample value says whose voice it is: ``(a, b, code)``."""
    end = max(b for _a, b, _c in spans)
    x = np.zeros(int(end * sp.SR) + sp.SR, np.float32)
    for a, b, code in spans:
        x[int(a * sp.SR) : int(b * sp.SR)] = code
    return x


#: The fake model: the code in the audio picks the vector
CODES = {
    1.0: _unit(1, 0, 0),
    2.0: _unit(0, 1, 0),
    3.0: _unit(0, 0, 1),
    # between 00 and 01, a little closer to 01: a close call
    4.0: _unit(0.95, 1, 0),
    # nobody of the recording
    5.0: _unit(-1, -1, 0.2),
}


def _embed(w):
    code = float(np.round(np.median(w.numpy()), 1))
    return CODES[code]


def _seg(a, b, speaker, **kw):
    return {"start": a, "end": b, "speaker": speaker, **kw}


def test_a_sentence_clearly_someone_else_moves():
    audio = _audio([(0, 3, 1.0), (3, 6, 2.0)])
    segs = [_seg(0, 3, "SPEAKER_00"), _seg(3, 6, "SPEAKER_00")]
    moves = sp.check_sentences(audio, segs, VOICES, _embed)
    assert [(m["index"], m["old"], m["new"]) for m in moves] == [(1, "SPEAKER_00", "SPEAKER_01")]
    assert moves[0]["sim"] > 0.99 and moves[0]["was"] < 0.01


def test_close_calls_stay_with_the_diarization():
    """It heard the sentence in context; the check hears it alone."""
    audio = _audio([(0, 3, 4.0)])
    assert sp.check_sentences(audio, [_seg(0, 3, "SPEAKER_00")], VOICES, _embed) == []


def test_nobody_similar_enough_stays():
    audio = _audio([(0, 3, 5.0)])
    assert sp.check_sentences(audio, [_seg(0, 3, "SPEAKER_00")], VOICES, _embed) == []


def test_short_unlabelled_and_voiceless_sentences_are_left_alone():
    audio = _audio([(0, 0.8, 2.0), (1, 4, 2.0), (4, 7, 2.0)])
    segs = [
        _seg(0, 0.8, "SPEAKER_00"),  # under a second
        _seg(1, 4, "—"),  # no label
        _seg(4, 7, "SPEAKER_09"),
    ]  # a label without a voice
    assert sp.check_sentences(audio, segs, VOICES, _embed) == []


def test_a_sentence_past_the_audio_is_skipped():
    audio = _audio([(0, 3, 2.0)])
    assert sp.check_sentences(audio, [_seg(50, 53, "SPEAKER_00")], VOICES, _embed) == []


def test_one_voice_has_nothing_to_compare():
    audio = _audio([(0, 3, 2.0)])
    one = {"SPEAKER_00": VOICES["SPEAKER_00"]}
    assert sp.check_sentences(audio, [_seg(0, 3, "SPEAKER_00")], one, _embed) == []


def test_the_numbers_decide():
    audio = _audio([(0, 3, 2.0)])
    segs = [_seg(0, 3, "SPEAKER_00")]
    assert sp.check_sentences(audio, segs, VOICES, _embed, min_sim=1.01) == []
    assert sp.check_sentences(audio, segs, VOICES, _embed, min_sec=5) == []
    assert sp.check_sentences(audio, segs, VOICES, _embed, margin=1.5) == []


def test_cancel_is_heard():
    audio = _audio([(0, 3, 2.0)])
    with pytest.raises(InterruptedError):
        sp.check_sentences(audio, [_seg(0, 3, "SPEAKER_00")], VOICES, _embed, abort=lambda: True)


def test_recheck_applies_the_moves_to_sentence_and_words(monkeypatch):
    """The subtitle split reads the words' speakers — they move too."""
    monkeypatch.setattr(sp, "load_model", lambda device="cuda", job=None: _embed)
    audio = _audio([(0, 3, 1.0), (3, 6, 2.0)])
    segs = [
        _seg(0, 3, "SPEAKER_00"),
        _seg(
            3,
            6,
            "SPEAKER_00",
            words=[
                {"word": "a", "speaker": "SPEAKER_00"},
                {"word": "b", "speaker": "SPEAKER_02"},
                {"word": "c"},
            ],
        ),
    ]
    out = sp.recheck_sentences(audio, segs, VOICES, device="cpu")
    assert segs[0]["speaker"] == "SPEAKER_00"
    assert segs[1]["speaker"] == "SPEAKER_01"
    assert [w.get("speaker") for w in segs[1]["words"]] == ["SPEAKER_01", "SPEAKER_02", None]
    assert out["checked"] == 2 and out["moved"] == 1
    assert out["moves"] == [
        {"start": 3.0, "old": "SPEAKER_00", "new": "SPEAKER_01", "sim": 1.0, "was": 0.0}
    ]


def test_recheck_settings_come_from_the_config(monkeypatch):
    from whisper_tm.config import config

    monkeypatch.setattr(sp, "load_model", lambda device="cuda", job=None: _embed)
    real = config.get

    def get(*keys, default=None):
        if keys == ("speaker", "check_min_sim"):
            return 1.01  # out of range: the default holds
        if keys == ("speaker", "check_min_sec"):
            return 5.0  # every sentence here is shorter
        return real(*keys, default=default)

    monkeypatch.setattr(config, "get", get)
    st = sp.check_settings()
    assert st["min_sim"] == sp.CHECK_MIN_SIM and st["min_sec"] == 5.0
    segs = [_seg(0, 3, "SPEAKER_00")]
    assert sp.recheck_sentences(_audio([(0, 3, 2.0)]), segs, VOICES, device="cpu")["moved"] == 0


def test_label_centroids_average_unit_windows():
    lines = [
        {"label": "A", "v_separated": np.array([2, 0], np.float32)},
        {"label": "A", "v_separated": np.array([0, 5], np.float32)},
        {"label": "B", "v_separated": np.array([0, 1], np.float32)},
        {"label": "C", "v_separated": None},
        {"label": "D", "v_separated": np.array([0, 0], np.float32)},
    ]
    c = sp.label_centroids(lines)
    assert sorted(c) == ["A", "B"]
    assert np.allclose(c["A"], _unit(1, 1), atol=1e-6)
    assert sp.label_centroids(lines, "v_raw") == {}


def test_recheck_in_a_voices_run_only_moves_to_the_transcripts_labels(monkeypatch):
    """A voices-only run's fresh label (no counterpart in the transcript)
    takes no sentences."""
    from whisper_tm.core import transcription as tr

    class Job:
        cancel = False

    monkeypatch.setattr(sp, "load_model", lambda device="cuda", job=None: _embed)
    audio = _audio([(0, 3, 2.0)])
    segs = [_seg(0, 3, "SPEAKER_00")]
    out = tr._recheck((audio, VOICES), segs, "cpu", Job(), labels={"SPEAKER_00", "SPEAKER_02"})
    assert out["moved"] == 0 and segs[0]["speaker"] == "SPEAKER_00"
    out = tr._recheck((audio, VOICES), segs, "cpu", Job(), labels={"SPEAKER_00"})
    assert out["moved"] == 0 and "fewer than two" in out["reason"]
    out = tr._recheck((audio, VOICES), segs, "cpu", Job(), labels={"SPEAKER_00", "SPEAKER_01"})
    assert out["moved"] == 1 and segs[0]["speaker"] == "SPEAKER_01"


def test_the_pipeline_checks_the_final_sentences():
    """After the subtitle split (it may part two speakers one Whisper segment
    held), before the markers — and only on separated audio."""
    from whisper_tm.core import transcription as tr

    src = (
        inspect.getsource(tr.transcribe_file)
        if hasattr(tr, "transcribe_file")
        else inspect.getsource(tr)
    )
    src = _flat(src)
    split = src.index("final = word_boundary_segmentation(final")
    check = src.index("_checked = _recheck(_check_with, final")
    marks = src.index("mark_segments(final, diar_segs)")
    assert split < check < marks
    assert "_sentence_check and _audio_separated" in src
