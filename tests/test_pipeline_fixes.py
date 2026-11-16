"""
Tests for the Pipeline Review fixes (as far as testable without torch/whisperx):
- parse_bool (robust bool parsing)
- _mean_volume_db (fallback heuristic)
"""

import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from whisper_tm.core.helpers import parse_bool

FFMPEG = shutil.which("ffmpeg")


class TestParseBool:
    def test_real_bools(self):
        assert parse_bool(True) is True
        assert parse_bool(False) is False

    def test_string_falsey(self):
        # Exactly the bug: bool("false") would be True
        for s in ("false", "False", "0", "no", "off", ""):
            assert parse_bool(s) is False, s

    def test_string_truthy(self):
        for s in ("true", "1", "yes", "on", "anything"):
            assert parse_bool(s) is True, s

    def test_none_is_false(self):
        assert parse_bool(None) is False


@pytest.mark.skipif(not FFMPEG, reason="ffmpeg not available")
class TestMeanVolume:
    def test_returns_float(self):
        from whisper_tm.core.audio_utils import _mean_volume_db

        work = Path(tempfile.mkdtemp())
        src = work / "a.wav"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=440:duration=2",
                "-ar",
                "16000",
                "-ac",
                "1",
                str(src),
            ],
            capture_output=True,
        )
        vol = _mean_volume_db(src)
        assert vol is not None
        assert -60 < vol < 0  # reasonable dBFS range

    def test_missing_file_none(self):
        from whisper_tm.core.audio_utils import _mean_volume_db

        assert _mean_volume_db(Path("/nonexistent/x.wav")) is None
