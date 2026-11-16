"""Analyzing the library (whisper_tm/audio/runner.py): during a
transcription from its WAV, and the background run over what the library
scan found. Real ffmpeg, synthetic sound."""

from __future__ import annotations

import os
import shutil
import sqlite3
import time
import wave
from pathlib import Path

import numpy as np
import pytest

from tests.test_audio_fingerprint import as_int16, song, write
from whisper_tm.audio import runner
from whisper_tm.audio import store as store_mod
from whisper_tm.audio.fingerprint import SR

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not available")


@pytest.fixture
def env(tmp_path, monkeypatch):
    cfg = {"enabled": True, "during_transcription": True, "parallel_files": 2, "min_free_gb": 0}
    monkeypatch.setattr(runner, "_cfg", lambda name, default=None: cfg.get(name, default))
    monkeypatch.setattr(store_mod, "store_path", lambda: tmp_path / "audio.db")
    store_mod._stores.clear()
    main = tmp_path / "main.db"
    con = sqlite3.connect(main)
    con.execute(
        "CREATE TABLE media_files (id INTEGER PRIMARY KEY, abs_path TEXT, file_size INTEGER, "
        "modified_at REAL, duration_sec REAL, has_audio INTEGER)"
    )
    con.commit()
    con.close()
    monkeypatch.setattr(runner, "_main_db", lambda: main)
    runner.forget_scope()
    lib = tmp_path / "lib"
    lib.mkdir()

    def scanned(*paths: Path, duration: float = 0.0):
        """As the library scan writes them into the main database."""
        with sqlite3.connect(main) as c:
            for p in paths:
                st = p.stat()
                c.execute(
                    "INSERT INTO media_files(abs_path, file_size, modified_at, duration_sec, has_audio) "
                    "VALUES(?,?,?,?,1)",
                    (str(p), st.st_size, st.st_mtime, duration),
                )
        runner.forget_scope()

    class Env:
        pass

    e = Env()
    e.cfg, e.lib, e.main, e.scanned = cfg, lib, main, scanned
    yield e
    if runner.RUNNER.running():
        runner.RUNNER.stop()
        runner.RUNNER.join(20)
    store_mod.close_all()
    runner.forget_scope()


def _wait(pred, timeout=60.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.05)
    return False


def test_the_run_analyzes_what_the_scan_found(env):
    a, b = song(1, 30), song(2, 20)
    pa = write(a, env.lib / "a.mp3", "-b:a", "96k")
    pb = write(b, env.lib / "b.m4a", "-c:a", "aac")
    pc = write(a, env.lib / "a again.opus", "-c:a", "libopus")
    broken = env.lib / "broken.mp4"
    broken.write_bytes(b"not a video")
    env.scanned(pa, pb, pc, broken, duration=25.0)
    assert runner.estimate(fresh=True)["files"] == 4
    assert runner.RUNNER.start(reason="test")
    assert _wait(lambda: not runner.RUNNER.running())
    st = runner.RUNNER.status()
    assert st["files_done"] == 4 and st["errors"] == 1, st
    store = store_mod.get_store()
    assert store.file(str(pa))["status"] == "done"
    assert store.file(str(broken))["status"] == "error"
    assert store.counts()["same"] == 1  # a.mp3 and "a again.opus"
    assert store.meta("run_wanted") is False
    # Nothing is left — the broken one counts as analyzed until it changes
    assert runner.estimate(fresh=True)["files"] == 0
    runner.RUNNER.start(reason="again")
    assert _wait(lambda: not runner.RUNNER.running())
    assert runner.RUNNER.status()["files_done"] == 0


def test_a_changed_file_is_analyzed_again(env):
    p = write(song(3, 15), env.lib / "x.mp3")
    env.scanned(p)
    runner.RUNNER.start()
    assert _wait(lambda: not runner.RUNNER.running())
    first = store_mod.get_store().file(str(p))["id"]
    write(song(4, 15), p)
    os.utime(p, (time.time() + 5, time.time() + 5))
    with sqlite3.connect(env.main) as c:
        st = p.stat()
        c.execute("UPDATE media_files SET file_size=?, modified_at=?", (st.st_size, st.st_mtime))
    runner.forget_scope()
    assert runner.estimate(fresh=True)["files"] == 1
    runner.RUNNER.start()
    assert _wait(lambda: not runner.RUNNER.running())
    assert store_mod.get_store().file(str(p))["id"] != first


def test_folders_limit_the_run(env):
    (env.lib / "keep").mkdir()
    (env.lib / "other").mkdir()
    p1 = write(song(5, 10), env.lib / "keep" / "1.mp3")
    p2 = write(song(6, 10), env.lib / "other" / "2.mp3")
    env.scanned(p1, p2)
    env.cfg["folders"] = [str(env.lib / "keep")]
    rows = runner.scope(fresh=True)
    assert [r[0] for r in rows] == [str(p1)]


def test_a_stopped_run_stores_nothing_half(env, monkeypatch):
    p = write(song(7, 60), env.lib / "long.wav")
    env.scanned(p, duration=60.0)
    from whisper_tm.audio import analyze

    real = analyze.decode

    def slow(*a, **kw):
        for x in real(*a, **kw):
            time.sleep(0.2)
            yield x

    monkeypatch.setattr(analyze, "decode", slow)
    runner.RUNNER.start()
    assert _wait(lambda: runner.RUNNER.status().get("current"))
    runner.RUNNER.stop(wanted=True)
    assert _wait(lambda: not runner.RUNNER.running())
    store = store_mod.get_store()
    assert store.file(str(p)) is None
    assert store.meta("run_wanted") is True  # goes on with the next start


def test_reading_is_paced(env, monkeypatch):
    """max_read_mb_s: the run sleeps between pieces so that the file's share
    read so far stays under the limit."""
    p = write(song(8, 40), env.lib / "paced.wav")  # ~1.3 MB
    env.scanned(p, duration=40.0)
    env.cfg["max_read_mb_s"] = 1.0
    env.cfg["parallel_files"] = 1
    t0 = time.monotonic()
    runner.RUNNER.start()
    assert _wait(lambda: not runner.RUNNER.running(), 30)
    assert time.monotonic() - t0 >= 0.9
    assert store_mod.get_store().file(str(p))["status"] == "done"


def _wav(path: Path, x: np.ndarray) -> Path:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(as_int16(x).tobytes())
    return path


def _drain_hook():
    pool = runner._hook_pool
    if pool is not None:
        pool.shutdown(wait=True)
        runner._hook_pool = None


def test_during_transcription_from_the_wav(env, tmp_path):
    """The transcription hands its 16 kHz WAV over and goes on: replacing
    it with the separated vocals right after takes nothing from the
    analysis — the file was opened before."""
    x = song(9, 25)
    source = write(x, env.lib / "talk.mp4", "-c:a", "aac")
    wav = _wav(tmp_path / "audio.wav", x)
    logged = []
    assert runner.after_decoding(wav, source, log=logged.append)
    _wav(tmp_path / "vocals.wav", song(10, 25)).replace(wav)
    _drain_hook()
    row = store_mod.get_store().file(str(source))
    assert row["status"] == "done" and row["source"] == "transcription"
    assert row["duration"] == pytest.approx(25.0, abs=0.01)
    assert logged and "Audio analysis" in logged[0]
    from whisper_tm.audio.analyze import search_snippet

    assert search_snippet(store_mod.get_store(), as_int16(x[5 * SR : 11 * SR]))["hits"][0]["file"][
        "path"
    ] == str(source)
    # Analyzed once: the next transcription of the same file doesn't again
    assert not runner.after_decoding(wav, source)


def test_during_transcription_off(env, tmp_path):
    env.cfg["during_transcription"] = False
    x = song(11, 5)
    assert not runner.after_decoding(_wav(tmp_path / "a.wav", x), write(x, env.lib / "a.mp3"))
    env.cfg["during_transcription"] = True
    env.cfg["enabled"] = False
    assert not runner.after_decoding(_wav(tmp_path / "a.wav", x), env.lib / "a.mp3")


def test_a_bad_wav_never_raises(env, tmp_path):
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"RIFF....WAVEjunk")
    src = write(song(12, 5), env.lib / "b.mp3")
    assert runner.after_decoding(bad, src) is True
    _drain_hook()
    assert store_mod.get_store().file(str(src)) is None
    assert runner.after_decoding(tmp_path / "missing.wav", src) is False


def test_a_run_going_at_shutdown_goes_on(env, monkeypatch):
    started = []
    monkeypatch.setattr(runner.RUNNER, "start", lambda reason="": started.append(reason) or True)
    monkeypatch.setattr(runner, "_auto_thread", object())
    monkeypatch.setattr(runner, "_auto_loop", lambda: None)
    store_mod.get_store().set_meta("run_wanted", True)
    runner._auto_thread = None
    runner.on_server_start()
    assert started == ["resumed"]


def test_gone_recordings_are_forgotten_where_their_folder_is(env):
    """A recording deleted from its folder goes; one whose whole folder is
    gone (a share not mounted, a moved folder) stays."""
    from whisper_tm.audio.analyze import analyze_file

    st = store_mod.get_store()
    (env.lib / "sub").mkdir()
    keep = [write(song(20 + i, 5), env.lib / f"k{i}.mp3") for i in range(8)]
    gone = write(song(30, 5), env.lib / "gone.mp3")
    far = write(song(31, 5), env.lib / "sub" / "far.mp3")
    for p in [*keep, gone, far]:
        analyze_file(st, str(p))
    gone.unlink()
    shutil.rmtree(env.lib / "sub")
    assert runner.prune_missing(st) == 1
    assert st.file(str(gone)) is None
    assert st.file(str(far)) is not None and st.file(str(keep[0])) is not None


def test_nothing_is_forgotten_when_too_many_seem_gone(env):
    from whisper_tm.audio.analyze import analyze_file

    st = store_mod.get_store()
    files = [write(song(40 + i, 5), env.lib / f"f{i}.mp3") for i in range(3)]
    for p in files:
        analyze_file(st, str(p))
    files[0].unlink()
    files[1].unlink()
    assert runner.prune_missing(st) == 0
    assert st.file(str(files[0])) is not None


def test_a_video_without_sound_says_so(env):
    """ffmpeg's last line for a video without a sound track is "Error
    opening output files: Invalid argument" — the analysis says what it is."""
    import subprocess

    from whisper_tm.audio.analyze import analyze_file

    p = env.lib / "silent clip.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=64x48:rate=10",
            "-t",
            "2",
            "-c:v",
            "libx264",
            str(p),
        ],
        check=True,
    )
    res = analyze_file(store_mod.get_store(), str(p))
    assert res["error"] == "no sound track"
    assert store_mod.get_store().file(str(p))["error"] == "no sound track"
