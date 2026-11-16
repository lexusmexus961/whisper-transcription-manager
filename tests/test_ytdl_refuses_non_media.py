"""A link download that isn't a recording never enters the library: a
manifest or playlist file named like a video would make every later read
open what it lists."""

import shutil
import subprocess
from pathlib import Path

import pytest

from whisper_tm.core import media_formats


@pytest.mark.skipif(shutil.which("ffprobe") is None, reason="no ffprobe")
def test_the_check_tells_a_manifest_from_a_recording(tmp_path):
    real = tmp_path / "real.mp3"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "sine=d=1", str(real)], check=True
    )
    fake = tmp_path / "video.mp4"
    fake.write_text(f"ffconcat version 1.0\nfile '{real}'\n", encoding="utf-8")
    hls = tmp_path / "clip.webm"
    hls.write_text("#EXTM3U\n#EXT-X-VERSION:3\n#EXTINF:1,\nreal.mp3\n", encoding="utf-8")
    assert media_formats.is_recording(real)
    assert not media_formats.is_recording(fake)
    assert not media_formats.is_recording(hls)


def test_the_download_removes_what_fails_it():
    src = (Path(__file__).resolve().parent.parent / "whisper_tm/core/ytdl.py").read_text(
        encoding="utf-8"
    )
    block = src[src.index("if not media_formats.is_recording(target):") :][:300]
    assert "target.unlink()" in block and "not an audio or video file" in block
