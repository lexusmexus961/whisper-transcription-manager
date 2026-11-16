"""A digitally silent short file must not cost a job.

**Observed in a long batch run**, on two phone videos of one second each:

    ERROR audio_separator … sep_pad/sep_single.wav is empty or not valid
    ERROR audio_separator … Failed to process file …
    ❌ Error: …/phone_clip.mp4:
       Separation failed after 2 attempts: no output files

The file wasn't broken — it was **silent**. `audio_separator` checks
`np.any(mix)` and rejects anything whose sample values are all zero.

The prefilter is supposed to catch exactly that; its docstring names the case
literally. It never caught it though, because a shortcut for short files
came first:

    if len(y) < sr:
        return "separate"      # unreadable/too short → safely separate

In a large library, that's not a curiosity but an error class: EVERY
silent recording under one second fails — tried twice, even though the
second attempt finds the same silence.
"""

from __future__ import annotations

import subprocess

import pytest

from whisper_tm.core import vocal_isolation as vi

# Every test here runs ffmpeg (skipped without it)
pytestmark = pytest.mark.ffmpeg


def _wav(path, duration, *, still, rate=16000):
    source = (
        f"anullsrc=r={rate}:cl=mono"
        if still
        else f"sine=frequency=440:duration={duration}:sample_rate={rate}"
    )
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            source,
            "-t",
            str(duration),
            "-ac",
            "1",
            "-ar",
            str(rate),
            "-acodec",
            "pcm_s16le",
            str(path),
        ],
        check=True,
        capture_output=True,
    )
    return path


class TestShortAndSilent:
    def test_silent_short_file_does_not_go_to_separation(self, tmp_path):
        """The actual fix."""
        p = _wav(tmp_path / "still.wav", 0.8, still=True)
        assert vi._chunk_prefilter(p, 0.8) == "silence"

    def test_short_with_signal_continues_to_be_separated(self, tmp_path):
        """The shortcut wasn't pointless: for window logic one
        second is too little. If there's something, it stays at "separate"."""
        p = _wav(tmp_path / "tone.wav", 0.8, still=False)
        assert vi._chunk_prefilter(p, 0.8) == "separate"

    def test_silence_over_one_second_was_always_recognized(self, tmp_path):
        """For demarcation: the long path already worked."""
        p = _wav(tmp_path / "long_silent.wav", 3.0, still=True)
        assert vi._chunk_prefilter(p, 3.0) == "silence"

    def test_disabled_prefilter_remains_unaffected(self, tmp_path, monkeypatch):
        from whisper_tm.config import config

        real = config.get

        def mocked(*path, default=None):
            if path == ("vocal_isolation", "prefilter"):
                return False
            return real(*path, default=default)

        monkeypatch.setattr(config, "get", mocked)
        p = _wav(tmp_path / "still2.wav", 0.8, still=True)
        assert vi._chunk_prefilter(p, 0.8) == "separate"


class TestThreshold:
    @pytest.mark.parametrize("duration", [0.2, 0.5, 0.99])
    def test_each_length_under_one_second(self, tmp_path, duration):
        """The shortcut applies to everything under `sr` samples."""
        p = _wav(tmp_path / f"s{duration}.wav", duration, still=True)
        assert vi._chunk_prefilter(p, duration) == "silence"
