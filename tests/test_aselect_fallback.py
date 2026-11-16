"""Cuts that keep only some parts of the audio work with every ffmpeg.

Ubuntu 26.04's ffmpeg 8.0.1 passes every frame through ``aselect`` — even
``aselect='0'``. The silence cut before diarization then came out as long as
the original while its times were mapped back through the parts it should have
kept, and an export "without pauses" kept them. Without a working aselect the
filter is ``asplit`` → ``atrim`` per part → ``concat``; both kinds are run here
on real ffmpeg, so a machine whose aselect works tests the fallback as well.
"""

import shutil
import subprocess
import wave

import pytest

from whisper_tm.core import audio_utils, diar_silence

pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg")

PARTS = [(1.0, 4.0), (7.0, 9.0), (10.5, 11.0)]  # 5.5 s of 12
#: The probe itself, before any test replaces it
REAL_PROBE = audio_utils.aselect_works


@pytest.fixture()
def tone(tmp_path):
    p = tmp_path / "tone.wav"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "sine=f=440:d=12:sample_rate=16000",
            "-ac",
            "1",
            str(p),
        ],
        check=True,
    )
    return p


def _cut(src, out, af):
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-i",
            str(src),
            "-af",
            af,
            "-ac",
            "1",
            "-ar",
            "16000",
            "-acodec",
            "pcm_s16le",
            str(out),
        ],
        check=True,
        capture_output=True,
    )
    with wave.open(str(out), "rb") as w:
        return w.getnframes() / w.getframerate()


@pytest.fixture(params=[True, False], ids=["aselect", "atrim-fallback"])
def mode(request, monkeypatch):
    monkeypatch.setattr(audio_utils, "aselect_works", lambda: request.param)
    return request.param


def test_the_kept_parts_are_exactly_what_remains(tone, tmp_path, mode):
    if mode and not REAL_PROBE():
        pytest.skip("this ffmpeg's aselect keeps every frame")
    got = _cut(tone, tmp_path / "out.wav", audio_utils.keep_audio_filter(PARTS))
    # aselect keeps whole frames (1,024 samples: 64 ms here), atrim is exact
    assert abs(got - 5.5) < 0.15, got


def test_the_diarization_cut_too(tone, tmp_path, mode):
    if mode and not REAL_PROBE():
        pytest.skip("this ffmpeg's aselect keeps every frame")
    got = _cut(tone, tmp_path / "out.wav", diar_silence.select_filter(PARTS))
    # aselect keeps whole frames (1,024 samples: 64 ms here), atrim is exact
    assert abs(got - 5.5) < 0.15, got


def test_a_single_part_needs_no_split(mode):
    f = audio_utils.keep_audio_filter([(2.0, 3.0)])
    assert ("asplit" not in f) and (("aselect" in f) == mode)


def test_the_probe_reads_this_ffmpeg():
    REAL_PROBE.cache_clear()
    assert isinstance(REAL_PROBE(), bool)


def test_a_cut_of_the_wrong_length_is_not_used(tmp_path, monkeypatch):
    """The pipeline checks the cut against the parts it keeps: a cut as long
    as the original would have put every speaker at the wrong time."""
    from whisper_tm.core import transcription as tr

    wav = tmp_path / "audio.wav"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "sine=d=60:sample_rate=16000",
            "-ac",
            "1",
            str(wav),
        ],
        check=True,
    )
    monkeypatch.setattr(diar_silence, "find_silence", lambda w: [(10.0, 50.0)])
    monkeypatch.setattr(diar_silence, "speech_parts", lambda s, d: [(0.0, 10.0), (50.0, 60.0)])
    # A filter that keeps everything, as the broken aselect did
    monkeypatch.setattr(diar_silence, "select_filter", lambda parts: "anull")
    out, parts = tr._diar_cut_silence(str(wav), 60.0, None)
    assert str(out) == str(wav) and parts == []
