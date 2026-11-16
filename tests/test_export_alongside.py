"""
"SRT next to the video file" — the complete export path, written to disk.

Background (audit): the feature was suspected to be broken. A code trace
found the chain intact; these tests verify it by actually running export_alongside()
to disk — including configuration quirks (formats as string OR list, force override)
and SRT correctness rules (#031: no negative/backwards timestamps).
"""

from __future__ import annotations

import pytest

from whisper_tm.core.export import _srt_time, _to_srt, export_alongside


@pytest.fixture
def source(tmp_path):
    f = tmp_path / "video.mp4"
    f.write_bytes(b"x")
    return f


SEGVS = [
    {"start": 0.0, "end": 2.5, "text": " Hello world ", "speaker": "SPEAKER_00"},
    {"start": 2.5, "end": 5.0, "text": "Second sentence", "speaker": "SPEAKER_01"},
]


def test_srt_appears_next_to_the_source(source):
    created = export_alongside(source, {"segments": SEGVS}, force=True)
    srt = source.parent / "video.srt"
    assert [str(srt)] == created
    content = srt.read_text(encoding="utf-8")
    assert "1\n00:00:00,000 --> 00:00:02,500\n[SPEAKER_00] Hello world" in content
    assert "[SPEAKER_01] Second sentence" in content


def test_formats_as_a_comma_string_and_a_list(source, monkeypatch):
    from whisper_tm.config import config

    # String form (how settings panel saves it)
    config.set("transcription", "export_alongside_formats", "srt, txt")
    created = export_alongside(source, {"segments": SEGVS}, force=True)
    assert {p.rsplit(".", 1)[1] for p in created} == {"srt", "txt"}
    # List form (YAML default)
    config.set("transcription", "export_alongside_formats", ["json"])
    created = export_alongside(source, {"segments": SEGVS}, force=True)
    assert created and created[0].endswith("video.json")
    config.set("transcription", "export_alongside_formats", ["srt"])


def test_config_gate_without_force(source):
    from whisper_tm.config import config

    config.set("transcription", "export_alongside", False)
    assert export_alongside(source, {"segments": SEGVS}) == []
    # force=True (the dialog checkbox) overrides the config switch
    assert export_alongside(source, {"segments": SEGVS}, force=True)


def test_srt_times_never_run_backwards():
    # end < start (#031) → clamped to start instead of invalid timecode
    srt = _to_srt([{"start": 5.0, "end": 3.0, "text": "x"}])
    assert "00:00:05,000 --> 00:00:05,000" in srt
    assert _srt_time(-2) == "00:00:00,000"


def test_word_timestamps_affect_srt_only_through_segment_boundaries():
    """SRT uses segment boundaries; `words` themselves do NOT appear in the cue.

    This is the documented design: "timestamp per word" refines segments
    (worker splits at word boundaries), the SRT formatter stays simple.
    This test ensures word remnants never leak into the subtitle text.
    """
    srt = _to_srt(
        [{"start": 0, "end": 1, "text": "Hi", "words": [{"word": "Hi", "start": 0, "end": 1}]}]
    )
    assert "Hi" in srt and "word" not in srt
