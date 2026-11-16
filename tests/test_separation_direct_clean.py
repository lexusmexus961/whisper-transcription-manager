"""A short clean recording kept next to the file is in full quality too.

Background: the direct path (a recording shorter than one separation
chunk — the common case) turned a chunk the prefilter called "clean"
into 16 kHz mono vocals whatever ``full_vocals`` said; the chunked path
already honoured it. Real ffmpeg output checked.
"""

from __future__ import annotations

import json
import shutil
import subprocess

import pytest

from whisper_tm.core import vocal_isolation as vi

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None, reason="ffmpeg not available"
)


def _probe(path):
    d = json.loads(
        subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "a:0",
                "-show_entries",
                "stream=sample_rate,channels:format=duration",
                "-of",
                "json",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout
    )
    s = d["streams"][0]
    return int(s["sample_rate"]), int(s["channels"]), float(d["format"]["duration"])


@pytest.fixture
def clean(tmp_path, monkeypatch):
    src = tmp_path / "memo.wav"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=4",
            "-ar",
            "44100",
            "-ac",
            "2",
            str(src),
        ],
        check=True,
        timeout=60,
    )
    monkeypatch.setattr(vi, "_chunk_prefilter", lambda *a, **k: "clean")
    monkeypatch.setattr(vi, "_run_separator", lambda *a, **k: pytest.fail("no separation"))
    for d in ("a", "b"):
        (tmp_path / d).mkdir()
    return src


def test_full_vocals_keep_the_extraction_format(clean, tmp_path):
    vocals, inst = vi.chunked_separate(clean, tmp_path / "a", full_vocals=True)
    rate, ch, dur = _probe(vocals)
    assert (rate, ch) == (48000, 2) and abs(dur - 4.0) < 0.05
    assert inst is None


def test_for_whisper_alone_they_stay_16k_mono(clean, tmp_path):
    vocals, _ = vi.chunked_separate(clean, tmp_path / "b")
    rate, ch, dur = _probe(vocals)
    assert (rate, ch) == (16000, 1) and abs(dur - 4.0) < 0.05
