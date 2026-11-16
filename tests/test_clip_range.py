"""A clip is the range chosen — to the frame.

A saved clip didn't match the range selected: the clip was cut as a stream copy, and a copy can only begin at a keyframe:
ffmpeg went back to the one before the start, and in a YouTube video they
lie seconds apart. 0:03–0:30 of a video with keyframes at 0 s and 5 s
came out 30 s long instead of 27 — starting 3 s early.

What these tests pin, with real material (keyframes every five seconds, as
in a downloaded video):

* a video clip begins where it was chosen and is as long as chosen;
* a container that can't hold the re-encoded clip (.webm) gives an .mp4;
* sound alone stays a stream copy — cut at the packet, close enough;
* the clip's name says the range to the tenth of a second the editor shows.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest
from flask import Flask

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None, reason="ffmpeg not available"
)


def _ffmpeg(*args: str) -> None:
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", *args], check=True, capture_output=True, timeout=120
    )


def _has_encoder(name: str) -> bool:
    out = subprocess.run(
        ["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True, timeout=30
    ).stdout
    return f" {name} " in out


@pytest.fixture
def app(tmp_path, monkeypatch):
    from whisper_tm.config import config

    media = tmp_path / "media"
    media.mkdir()
    clips = tmp_path / "clips"
    clips.mkdir()
    if shutil.which("ffmpeg"):
        # Twelve seconds of a picture that brightens with time — 20 steps a
        # second, so a frame says when it is — with sound, a keyframe every
        # five seconds: at 0, 5 and 10
        _ffmpeg(
            "-f",
            "lavfi",
            "-i",
            "color=c=black:size=160x90:rate=25:duration=12,format=yuv420p,"
            "geq=lum='mod(20*T,256)':cb=128:cr=128",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=12",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-g",
            "125",
            "-keyint_min",
            "125",
            "-sc_threshold",
            "0",
            "-c:a",
            "aac",
            "-shortest",
            str(media / "video.mp4"),
        )
        _ffmpeg(
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=330:duration=6",
            "-c:a",
            "aac",
            str(media / "tone.m4a"),
        )
        if _has_encoder("libvpx-vp9"):
            _ffmpeg(
                "-f",
                "lavfi",
                "-i",
                "testsrc=size=160x90:rate=25:duration=6",
                "-c:v",
                "libvpx-vp9",
                "-deadline",
                "realtime",
                "-g",
                "125",
                str(media / "video.webm"),
            )
    monkeypatch.setattr(type(config), "media_dir", property(lambda self: media))
    monkeypatch.setattr(type(config), "dirs", property(lambda self: {"clips": clips}))

    from whisper_tm.api.media import media_bp

    a = Flask(__name__)
    a.register_blueprint(media_bp)
    a.media = media
    a.clips = clips
    return a


def _duration(path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    return float(out.stdout.strip())


def _when(path) -> float:
    """The moment a clip's first frame shows: its brightness / 20 — read
    from the Y plane as written (a grey conversion stretches 16–235)."""
    raw = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(path),
            "-frames:v",
            "1",
            "-s",
            "32x18",
            "-pix_fmt",
            "yuv420p",
            "-f",
            "rawvideo",
            "-",
        ],
        capture_output=True,
        timeout=30,
        check=True,
    ).stdout
    y = raw[: 32 * 18]
    return sum(y) / len(y) / 20


@needs_ffmpeg
def test_a_video_clip_is_the_range_chosen(app):
    r = app.test_client().post(
        "/api/clip", json={"filename": "video.mp4", "start": 3.2, "end": 8.4}
    )
    d = r.get_json()
    assert r.status_code == 200 and d["ok"], d
    assert d["filename"] == "video_clip_0-00-03.2_to_0-00-08.4.mp4"
    clip = app.clips / d["filename"]
    # A copy came out 8.4 s long — from the keyframe at 0
    assert abs(_duration(clip) - 5.2) < 0.1, _duration(clip)
    # … and it begins at 3.2, not at the keyframe before it
    assert abs(_when(clip) - 3.2) < 0.2, _when(clip)


@needs_ffmpeg
def test_a_webm_clip_becomes_an_mp4(app):
    if not (app.media / "video.webm").exists():
        pytest.skip("no VP9 encoder")
    d = (
        app.test_client()
        .post("/api/clip", json={"filename": "video.webm", "start": 1.0, "end": 3.5})
        .get_json()
    )
    assert d["ok"] and d["filename"].endswith(".mp4"), d
    assert abs(_duration(app.clips / d["filename"]) - 2.5) < 0.1


@needs_ffmpeg
def test_sound_alone_stays_a_copy(app, monkeypatch):
    from whisper_tm.api import media

    seen = []
    real = media.run_cancellable

    def run(cmd, **kw):
        seen.append(cmd)
        return real(cmd, **kw)

    monkeypatch.setattr(media, "run_cancellable", run)
    d = (
        app.test_client()
        .post("/api/clip", json={"filename": "tone.m4a", "start": 1.0, "end": 3.0})
        .get_json()
    )
    assert d["ok"] and d["filename"] == "tone_clip_0-00-01.0_to_0-00-03.0.m4a", d
    assert abs(_duration(app.clips / d["filename"]) - 2.0) < 0.06
    assert seen and "copy" in seen[0] and "libx264" not in seen[0]


def test_the_name_says_the_range_to_the_tenth():
    from whisper_tm.api.media import _clip_ts

    assert _clip_ts(3.29) == "0-00-03.3"
    assert _clip_ts(59.96) == "0-01-00.0"
    assert _clip_ts(3725.04) == "1-02-05.0"
