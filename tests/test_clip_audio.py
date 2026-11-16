"""An audio clip via GET — so a player can play it.

The existing `/api/clip` cuts on POST and returns a filename. But an
`<audio>` element in the browser can only GET, and that's exactly what's needed
for the MCP server path: the language model writes a line for each finding,
the chat window turns that into a player.

Cutting happens only during playback. A hit list quickly has thirty findings;
cutting all of them in advance would mean thirty times ffmpeg for snippets,
of which two are listened to.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest
from flask import Flask

ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not available")


@pytest.fixture
def app(tmp_path, monkeypatch):
    from whisper_tm.config import config

    media = tmp_path / "media"
    media.mkdir()
    clips = tmp_path / "clips"
    clips.mkdir()
    if shutil.which("ffmpeg"):
        # Five seconds of sine — real material so the cut actually cuts
        # and doesn't pass by a dummy file.
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=440:duration=5",
                str(media / "tone.wav"),
            ],
            capture_output=True,
            timeout=60,
        )
    else:
        (media / "tone.wav").write_bytes(b"x")

    monkeypatch.setattr(type(config), "media_dir", property(lambda self: media))
    monkeypatch.setattr(type(config), "dirs", property(lambda self: {"clips": clips}))

    from whisper_tm.api.media import media_bp

    a = Flask(__name__)
    a.register_blueprint(media_bp)
    a.config["TESTING"] = False
    a.clips = clips
    return a


class TestTheRejections:
    """What doesn't work must say why — a 500 helps nobody."""

    @pytest.mark.parametrize(
        "question",
        [
            "",  # no parameters at all
            "?file=tone.wav",  # without times
            "?file=tone.wav&start=3&end=1",  # backwards
            "?file=tone.wav&start=x&end=y",  # not numbers
        ],
    )
    def test_unusable_parameters(self, app, question):
        r = app.test_client().get("/api/clip/audio" + question)
        assert r.status_code == 400, r.data[:200]

    def test_clip_too_long(self, app):
        """A typo in `end` must not trigger hours of re-encoding."""
        r = app.test_client().get("/api/clip/audio?file=tone.wav&start=0&end=99999")
        assert r.status_code == 400
        assert b"too long" in r.data

    def test_file_does_not_exist(self, app):
        r = app.test_client().get("/api/clip/audio?file=missing_it.wav&start=0&end=5")
        assert r.status_code == 404

    def test_no_escape_from_media_root(self, app):
        r = app.test_client().get("/api/clip/audio?file=../../etc/passwd&start=0&end=5")
        assert r.status_code in (400, 403, 404), r.status_code


@ffmpeg
class TestTheCut:
    def test_delivers_playable_audio(self, app):
        r = app.test_client().get("/api/clip/audio?file=tone.wav&start=1&end=3")
        assert r.status_code == 200, r.data[:300]
        assert r.mimetype == "audio/mpeg"
        assert len(r.data) > 1000, "too small for two seconds of mp3"
        assert r.data[:3] in (b"ID3", b"\xff\xfb", b"\xff\xf3"), r.data[:8]

    def test_not_attachment_but_playable(self, app):
        """`as_attachment` would put the file in the download folder
        instead of playing it."""
        r = app.test_client().get("/api/clip/audio?file=tone.wav&start=0&end=2")
        assert "attachment" not in (r.headers.get("Content-Disposition") or "")

    def test_without_video_track(self, app):
        """From 30 s video becomes 360 kB instead of 40 MB."""
        app.test_client().get("/api/clip/audio?file=tone.wav&start=0&end=2")
        file = next(app.clips.glob("sample_*.mp3"))
        off = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "stream=codec_type",
                "-of",
                "csv=p=0",
                str(file),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout
        assert "video" not in off, off
        assert "audio" in off, off

    def test_same_spot_twice_cuts_once(self, app):
        """The name carries the cut within it — second time no ffmpeg."""
        c = app.test_client()
        c.get("/api/clip/audio?file=tone.wav&start=1&end=3")
        file = next(app.clips.glob("sample_*.mp3"))
        before = file.stat().st_mtime_ns

        import whisper_tm.api.media as media_mod

        called = []
        real = media_mod.run_cancellable
        media_mod.run_cancellable = lambda *a, **kw: called.append(1) or real(*a, **kw)
        try:
            r = c.get("/api/clip/audio?file=tone.wav&start=1&end=3")
        finally:
            media_mod.run_cancellable = real
        assert r.status_code == 200
        assert called == [], "cut again"
        assert file.stat().st_mtime_ns == before

    def test_different_spots_are_different_files(self, app):
        c = app.test_client()
        c.get("/api/clip/audio?file=tone.wav&start=0&end=2")
        c.get("/api/clip/audio?file=tone.wav&start=2&end=4")
        assert len(list(app.clips.glob("sample_*.mp3"))) == 2


def _loudest(path) -> float:
    out = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostats",
            "-i",
            str(path),
            "-af",
            "volumedetect",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    ).stderr
    return float(out.split("max_volume:")[1].split("dB")[0])


@ffmpeg
def test_a_quiet_recording_is_brought_to_speech_loudness(app, tmp_path):
    """A stream 45 dB below full scale played back as silence. The sample
    is normalized."""
    from whisper_tm.config import config

    quiet = config.media_dir / "quiet.wav"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=300:duration=6,volume=-45dB",
            str(quiet),
        ],
        capture_output=True,
        timeout=60,
    )
    assert _loudest(quiet) < -40
    r = app.test_client().get("/api/clip/audio?file=quiet.wav&start=0&end=5")
    assert r.status_code == 200
    sample = next(app.clips.glob("sample_quiet_*.mp3"))
    # A steady tone at -18 LUFS peaks near -15 dB (speech, spikier, higher)
    assert -24 < _loudest(sample) < -1
