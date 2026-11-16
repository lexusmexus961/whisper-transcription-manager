"""Playback transcode: what goes into the cache, and what does not.

Issues fixed in ``/api/video``:

* ffmpeg's exit code was never checked: a transcode that broke off (a full
  cache disk) still had its partial MP3 renamed into the cache — served,
  truncated, until the recording changed.
* A fixed 120-s limit: a multi-hour recording never finished, and every
  click in the transcript started the same doomed transcode again.
* Plain PCM WAV (pcm_s16le — what nearly every .wav is) was transcoded,
  although every browser plays it.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest
from flask import Flask

from whisper_tm.api import media

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not available")


@pytest.fixture
def app(tmp_path, monkeypatch):
    from whisper_tm.config import config

    mdir = tmp_path / "media"
    mdir.mkdir()
    work = tmp_path / "cache" / "work"
    work.mkdir(parents=True)
    monkeypatch.setattr(type(config), "media_dir", property(lambda self: mdir))
    monkeypatch.setattr(type(config), "work_dir", property(lambda self: work))
    media.clear_path_cache()
    media._transcode_failed.clear()
    for name, codec in (("plain.wav", "pcm_s16le"), ("deep.wav", "pcm_s24le")):
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=440:duration=3",
                "-c:a",
                codec,
                str(mdir / name),
            ],
            check=True,
            timeout=60,
        )
    a = Flask(__name__)
    a.register_blueprint(media.media_bp)
    a.cache = tmp_path / "cache" / "playback_cache"
    return a


def _cached(app):
    return sorted(p.name for p in app.cache.glob("*.mp3")) if app.cache.exists() else []


def test_plain_pcm_wav_is_served_as_it_is(app, monkeypatch):
    real = media.run_cancellable

    def run(cmd, **k):
        if "libmp3lame" in cmd:
            pytest.fail("a plain WAV was transcoded")
        return real(cmd, **k)

    monkeypatch.setattr(media, "run_cancellable", run)
    r = app.test_client().get("/api/video/plain.wav")
    assert r.status_code == 200 and _cached(app) == []


def test_a_failed_transcode_is_not_cached(app, monkeypatch):
    real = media.run_cancellable

    def broken(cmd, **k):
        r = real(cmd, **k)
        if "libmp3lame" in cmd:
            return subprocess.CompletedProcess(cmd, 1, r.stdout, "No space left on device")
        return r

    monkeypatch.setattr(media, "run_cancellable", broken)
    r = app.test_client().get("/api/video/deep.wav")
    assert r.status_code == 200
    assert _cached(app) == []
    assert not list(app.cache.glob("*.tmp"))


def test_a_failed_transcode_is_not_retried_on_every_click(app, monkeypatch):
    calls = []

    def failing(cmd, **k):
        if "libmp3lame" in cmd:
            calls.append(1)
            raise RuntimeError("FFmpeg timeout")
        return real(cmd, **k)

    real = media.run_cancellable
    monkeypatch.setattr(media, "run_cancellable", failing)
    client = app.test_client()
    for _ in range(3):
        assert client.get("/api/video/deep.wav").status_code == 200
    assert len(calls) == 1


def test_a_good_transcode_is_cached(app):
    assert app.test_client().get("/api/video/deep.wav").status_code == 200
    assert len(_cached(app)) == 1


def test_the_limit_grows_with_the_recording():
    assert media._transcode_timeout(60) == 120
    assert media._transcode_timeout(10 * 3600) > 120
