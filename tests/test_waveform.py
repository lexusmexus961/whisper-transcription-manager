"""The clip editor's waveform — and that a long recording never blocks it.

The waveform makes cutting easier — but it must never block, even for a
24-hour recording. Such a recording can be a video of several GB on a
network share; decoding its sound takes about a minute. So
(core/waveform.py):

* a short window is decoded on the spot — exact, to the bucket;
* a long one comes from an overview decoded in the background, one file at
  a time; until it is complete the answer is what is there and
  ``pending``, never a wait;
* the overview is kept in the cache folder, under a limit, and read from
  there the next time — no ffmpeg;
* an overview nobody waits for any more is dropped, a failed one is not
  tried again on every request.

The material is real sound decoded by the real ffmpeg: three seconds of
silence, three of a tone, three of silence.
"""

from __future__ import annotations

import os
import shutil
import threading
import time
import wave

import numpy as np
import pytest
from flask import Flask

from whisper_tm.core import waveform as w

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None, reason="ffmpeg not available"
)

HZ = 16000
#: The tone at half of full scale: -6 dB, level 229 of 255
TONE = range(225, 233)


@pytest.fixture
def media(tmp_path, monkeypatch):
    folder = tmp_path / "media"
    folder.mkdir()
    tone = (0.5 * 32767 * np.sin(2 * np.pi * 440 * np.arange(3 * HZ) / HZ)).astype("<i2")
    silence = np.zeros(3 * HZ, "<i2")
    with wave.open(str(folder / "talk.wav"), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(HZ)
        f.writeframes(np.concatenate([silence, tone, silence]).tobytes())
    (folder / "notes.wav").write_text("not sound at all", encoding="utf-8")
    monkeypatch.setattr(w, "_cache_dir", lambda: tmp_path / "cache")
    for name in ("_jobs", "_info"):
        monkeypatch.setattr(w, name, {})
    monkeypatch.setattr(w, "_loaded", w.OrderedDict())
    return folder


def _until(cond, timeout=20.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.05)
    return False


def test_a_value_is_decibels():
    got = w._levels(np.array([32767, 3277, 328, 33, 0])).tolist()
    assert got == [254, 170, 85, 0, 0]  # 0, -20, -40, -60 dB and below


def test_pooling_takes_the_loudest_and_leaves_the_unknown_empty():
    values = np.arange(10, dtype=np.uint8)
    assert w._pooled(values, 0, 20, 4) == [4, 9, None, None]
    assert w._pooled(values, 2, 6, 2) == [3, 5]
    assert w._pooled(values, 12, 20, 2) == [None, None]


class TestOnTheSpot:
    def test_a_short_window_is_exact(self, media):
        d = w.peaks(media / "talk.wav", 0, 9, 90)
        assert d["pending"] is False and d["n"] == 90 and len(d["peaks"]) == 90
        quiet, loud, after = d["peaks"][:29], d["peaks"][31:59], d["peaks"][61:]
        assert max(quiet) == 0 and max(after) == 0
        assert set(loud) <= set(TONE)

    def test_beyond_the_end_is_empty_not_silent(self, media):
        d = w.peaks(media / "talk.wav", 6, 12, 60)
        assert d["peaks"][:25] == [0] * 25 and d["peaks"][-20:] == [None] * 20

    def test_no_sound_is_said(self, media):
        with pytest.raises(w.WaveformError, match="no sound"):
            w.peaks(media / "notes.wav", 0, 5, 50)


class TestTheOverview:
    @pytest.fixture(autouse=True)
    def long_windows(self, monkeypatch):
        # Nine seconds count as long here: past the on-the-spot limit
        monkeypatch.setattr(w, "DECODE_MAX_SEC", 2)

    def test_a_long_window_never_waits(self, media):
        with w._overview_slot:  # another overview runs
            t = time.time()
            d = w.peaks(media / "talk.wav", 0, 9, 90)
            assert time.time() - t < 1.0
            assert d["pending"] is True and d["progress"] == 0.0 and d["peaks"] == [None] * 90
        assert _until(lambda: not w.peaks(media / "talk.wav", 0, 9, 90)["pending"])
        d = w.peaks(media / "talk.wav", 0, 9, 90)
        assert max(d["peaks"][:29]) == 0 and set(d["peaks"][31:59]) <= set(TONE)

    def test_it_is_kept_and_read_again_without_ffmpeg(self, media, monkeypatch):
        w.peaks(media / "talk.wav", 0, 9, 90)
        assert _until(lambda: not w.peaks(media / "talk.wav", 0, 9, 90)["pending"])
        kept = list((media.parent / "cache").glob("*.u8"))
        assert len(kept) == 1 and kept[0].stat().st_size == 9 * w.RATE

        def no_ffmpeg(*a, **k):
            raise AssertionError("ffmpeg ran again")

        monkeypatch.setattr(w.subprocess, "Popen", no_ffmpeg)
        monkeypatch.setattr(w.subprocess, "run", no_ffmpeg)
        w._loaded.clear()  # as after a restart: from the folder
        d = w.peaks(media / "talk.wav", 0, 9, 90)
        assert d["pending"] is False and set(d["peaks"][31:59]) <= set(TONE)

    def test_a_changed_file_is_read_anew(self, media):
        a = w._identity(media / "talk.wav")[0]
        os.utime(media / "talk.wav", (1, 1))
        assert w._identity(media / "talk.wav")[0] != a

    def test_one_nobody_waits_for_is_dropped(self, media, monkeypatch):
        decoded = []
        monkeypatch.setattr(w, "_decode_overview", lambda job: decoded.append(job))
        with w._overview_slot:
            w.peaks(media / "talk.wav", 0, 9, 90)
            job = next(iter(w._jobs.values()))
            job.wanted -= w.ABANDON_SEC + 1  # asked for long ago
        assert _until(lambda: not w._jobs) and decoded == []

    def test_a_failed_one_is_not_tried_on_every_request(self, media, monkeypatch):
        runs = []

        def fails(job):
            runs.append(1)
            raise w.WaveformError("the NAS went away")

        monkeypatch.setattr(w, "_decode_overview", fails)
        try:  # pending, or failed already
            w.peaks(media / "talk.wav", 0, 9, 90)
        except w.WaveformError:
            pass
        assert _until(lambda: any(j.error for j in w._jobs.values()))
        for _ in range(3):
            with pytest.raises(w.WaveformError, match="NAS"):
                w.peaks(media / "talk.wav", 0, 9, 90)
        assert len(runs) == 1
        monkeypatch.setattr(w, "RETRY_SEC", 0)  # later: once more
        try:
            w.peaks(media / "talk.wav", 0, 9, 90)
        except w.WaveformError:
            pass
        assert _until(lambda: len(runs) == 2)

    def test_one_at_a_time(self, media, monkeypatch):
        """Five long recordings opened one after the other decode one by
        one, not five ffmpegs at once on the NAS."""
        at_once, most = [0], [0]
        gate = threading.Event()

        def slow(job):
            at_once[0] += 1
            most[0] = max(most[0], at_once[0])
            gate.wait(2)
            at_once[0] -= 1
            job.done = True

        monkeypatch.setattr(w, "_decode_overview", slow)
        for i in range(3):
            shutil.copy(media / "talk.wav", media / f"copy{i}.wav")
            w.peaks(media / f"copy{i}.wav", 0, 9, 90)
        time.sleep(0.3)
        gate.set()
        assert _until(lambda: not w._jobs) and most[0] == 1


class TestKeptByTheAudioAnalysis:
    """A recording the audio analysis has seen (whisper_tm/audio) has its
    overview kept there: a long window is answered at once, with no ffmpeg
    — and only for the version of the file it was made of."""

    @pytest.fixture
    def analyzed(self, media, tmp_path, monkeypatch):
        from whisper_tm.audio import analyze, runner
        from whisper_tm.audio import store as store_mod

        monkeypatch.setattr(w, "DECODE_MAX_SEC", 2)
        monkeypatch.setattr(
            runner, "_cfg", lambda name, default=None: {"enabled": True}.get(name, default)
        )
        monkeypatch.setattr(store_mod, "store_path", lambda: tmp_path / "audio.db")
        store_mod._stores.clear()
        analyze.analyze_file(store_mod.get_store(), str(media / "talk.wav"))
        yield media
        store_mod.close_all()

    def test_a_long_window_comes_from_it(self, analyzed, monkeypatch):
        def no_decoding(*a, **k):
            raise AssertionError("the sound was decoded")

        monkeypatch.setattr(w, "_decode", no_decoding)
        monkeypatch.setattr(w, "_overview", no_decoding)
        d = w.peaks(analyzed / "talk.wav", 0, 9, 90)
        assert d["pending"] is False
        assert max(d["peaks"][:29]) == 0 and set(d["peaks"][31:59]) <= set(TONE)

    def test_not_for_a_changed_file(self, analyzed):
        os.utime(analyzed / "talk.wav", (1, 1))
        with w._overview_slot:  # the decoding waits its turn
            d = w.peaks(analyzed / "talk.wav", 0, 9, 90)
        assert d["pending"] is True

    def test_switched_off_it_is_not_asked(self, analyzed, monkeypatch):
        from whisper_tm.audio import runner

        monkeypatch.setattr(
            runner, "_cfg", lambda name, default=None: {"enabled": False}.get(name, default)
        )
        with w._overview_slot:
            assert w.peaks(analyzed / "talk.wav", 0, 9, 90)["pending"] is True


def test_the_cache_folder_keeps_its_limit(tmp_path, monkeypatch):
    for i, name in enumerate(("old", "middle", "new")):
        p = tmp_path / f"{name}.u8"
        p.write_bytes(b"x" * 100)
        os.utime(p, (1000 + i, 1000 + i))
    (tmp_path / "left.123.part").write_bytes(b"x")  # a write a stop cut short
    os.utime(tmp_path / "left.123.part", (1000, 1000))
    (tmp_path / "now.456.part").write_bytes(b"x")  # one being written
    monkeypatch.setattr(w, "CACHE_MAX_BYTES", 250)
    w._prune(tmp_path)
    assert sorted(p.stem for p in tmp_path.glob("*.u8")) == ["middle", "new"]
    assert [p.name for p in tmp_path.glob("*.part")] == ["now.456.part"]


class TestTheRoute:
    @pytest.fixture
    def client(self, media, monkeypatch):
        from whisper_tm.config import config

        monkeypatch.setattr(type(config), "media_dir", property(lambda self: media))
        from whisper_tm.api.media import media_bp

        app = Flask(__name__)
        app.register_blueprint(media_bp)
        return app.test_client()

    def test_the_window_in_buckets(self, client):
        d = client.get("/api/clip/waveform?file=talk.wav&start=0&end=9&n=90").get_json()
        assert d["n"] == 90 and d["pending"] is False and set(d["peaks"][31:59]) <= set(TONE)

    def test_what_it_refuses(self, client):
        assert client.get("/api/clip/waveform?file=talk.wav&start=5&end=2").status_code == 400
        assert client.get("/api/clip/waveform?file=talk.wav&start=x&end=2").status_code == 400
        assert client.get("/api/clip/waveform?file=gone.wav&start=0&end=2").status_code == 404
        assert client.get("/api/clip/waveform?file=../../etc/passwd&start=0&end=2").status_code in (
            400,
            403,
            404,
        )
        r = client.get("/api/clip/waveform?file=notes.wav&start=0&end=2")
        assert r.status_code == 422 and "no sound" in r.get_json()["error"]
