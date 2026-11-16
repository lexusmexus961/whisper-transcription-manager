"""The single pass reads the video along — and writes the same sound.

On a network share, the first step of a large screen recording (.mov) can
read at only a few MB/s. Its sound sits between the video
frames, a small packet every 21 ms, and with the video unused ffmpeg's mov
demuxer jumped from one sound packet to the next — every jump a round trip
to the share. Read along and thrown away, the video makes it one sequential
read: several times as much film a second (core/transcription._single_pass_cmd).

What these tests pin, with real ffmpeg:

* the video goes into a null output — and only if there is one (``?``);
* the sound written is the same to the byte as without it, for a video
  with interleaved sound and for a file that is sound only;
* the worker builds its command there, after the check for a sound track.
"""

from __future__ import annotations

import hashlib
import inspect
import shutil
import subprocess

import pytest

from whisper_tm.core import transcription

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not available")


def _ffmpeg(*args: str) -> None:
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", *args], check=True, capture_output=True, timeout=120
    )


def _packets(path) -> str:
    """The audio track's packets, without the container's own bytes (its
    muxing date and ids differ from run to run)."""
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:a:0", "-c", "copy", "-f", "md5", "-"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    return out.stdout.strip()


def _run(cmd) -> int:
    return subprocess.run(cmd, capture_output=True, timeout=120).returncode


def test_the_video_is_read_into_nothing_when_there_is_one():
    cmd = transcription._single_pass_cmd("in.mov", "a.mka", "a.wav")
    assert cmd[-7:] == ["-map", "0:v:0?", "-c", "copy", "-f", "null", "-"]
    # the two outputs it is for, unchanged
    assert cmd.index("a.mka") < cmd.index("a.wav") < cmd.index("0:v:0?")


def test_the_worker_builds_it_there_after_the_sound_check():
    source = inspect.getsource(transcription.transcription_worker)
    assert source.index("has_audio_stream(source_path)") < source.index(
        "_single_pass_cmd(source_path"
    )
    assert '"ffmpeg", "-y", "-i", str(source_path)' not in source  # no second copy of it


@needs_ffmpeg
@pytest.mark.parametrize("kind", ["video with sound", "sound only"])
def test_the_sound_is_the_same_to_the_byte(tmp_path, kind):
    src = tmp_path / ("in.mov" if kind == "video with sound" else "in.m4a")
    if kind == "video with sound":
        # Interleaved as a camera or OBS writes it: frames and sound packets
        # in turn
        _ffmpeg(
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=320x240:rate=25:duration=6",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=6",
            "-c:v",
            "mpeg4",
            "-q:v",
            "2",
            "-c:a",
            "aac",
            "-shortest",
            str(src),
        )
    else:
        _ffmpeg("-f", "lavfi", "-i", "sine=frequency=330:duration=6", "-c:a", "aac", str(src))
    new = transcription._single_pass_cmd(src, tmp_path / "new.mka", tmp_path / "new.wav")
    old = new[: new.index("0:v:0?") - 1]  # the same, without the read-along
    old = [
        str(tmp_path / "old.mka")
        if a.endswith("new.mka")
        else str(tmp_path / "old.wav")
        if a.endswith("new.wav")
        else a
        for a in old
    ]
    assert _run(new) == 0 and _run(old) == 0
    wav = {v: hashlib.md5((tmp_path / f"{v}.wav").read_bytes()).hexdigest() for v in ("new", "old")}
    assert wav["new"] == wav["old"]
    assert _packets(tmp_path / "new.mka") == _packets(tmp_path / "old.mka") != ""
