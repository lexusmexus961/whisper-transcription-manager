"""The routes of the audio analysis (api/audio.py): searching a stretch or a
file of one's own, the duplicates, one recording's analysis, the run, and
what is refused. Real ffmpeg, synthetic sound (tests/test_audio_fingerprint)."""

from __future__ import annotations

import io
import shutil
import sqlite3

import numpy as np
import pytest
from flask import Flask

from tests.test_audio_fingerprint import song, write
from whisper_tm.audio import analyze, runner
from whisper_tm.audio import store as store_mod

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not available")


@pytest.fixture
def env(tmp_path, monkeypatch):
    from whisper_tm import extensions as ext
    from whisper_tm.api import audio as api
    from whisper_tm.config import config

    lib = tmp_path / "lib"
    lib.mkdir()
    monkeypatch.setattr(type(config), "media_dir", property(lambda self: lib))
    monkeypatch.setattr(ext, "db", None)
    cfg = {"enabled": True}
    monkeypatch.setattr(runner, "_cfg", lambda name, default=None: cfg.get(name, default))
    monkeypatch.setattr(store_mod, "store_path", lambda: tmp_path / "audio.db")
    store_mod._stores.clear()
    main = tmp_path / "main.db"
    with sqlite3.connect(main) as c:
        c.execute(
            "CREATE TABLE media_files (id INTEGER PRIMARY KEY, abs_path TEXT, file_size INTEGER, "
            "modified_at REAL, duration_sec REAL, has_audio INTEGER)"
        )
    monkeypatch.setattr(runner, "_main_db", lambda: main)
    runner.forget_scope()

    a, b, c = song(1, 40), song(2, 30), song(3, 35)
    files = {
        "a.wav": write(a, lib / "a.wav"),
        "a_copy.mp3": write(a, lib / "a_copy.mp3", "-b:a", "64k"),
        "b.m4a": write(b, lib / "b.m4a", "-c:a", "aac"),
        "compilation.mp3": write(np.concatenate([c, a * 0.5, b]), lib / "compilation.mp3"),
        "c.opus": write(c, lib / "c.opus", "-c:a", "libopus"),
    }
    store = store_mod.get_store()
    for p in files.values():
        analyze.analyze_file(store, str(p))
    (lib / "silent.wav").write_bytes(b"RIFF0000WAVEfmt ")
    analyze.analyze_file(store, str(lib / "silent.wav"))

    app = Flask(__name__)
    app.register_blueprint(api.audio_bp)

    class Env:
        pass

    e = Env()
    e.cfg, e.lib, e.files, e.store, e.main = cfg, lib, files, store, main
    e.songs = {"a": a, "b": b, "c": c}
    e.client = app.test_client()
    yield e
    if runner.RUNNER.running():
        runner.RUNNER.stop()
        runner.RUNNER.join(20)
    store_mod.close_all()


def test_info(env):
    d = env.client.get("/api/audio/info").get_json()
    assert d["enabled"] and d["counts"]["files"] == 5 and d["counts"]["errors"] == 1
    assert d["counts"]["same"] == 1 and d["counts"]["contains"] >= 3
    assert d["run"]["running"] is False


def test_a_stretch_of_a_recording(env):
    r = env.client.post("/api/audio/search", json={"file": "a.wav", "start": 12, "end": 18})
    assert r.status_code == 200, r.data
    d = r.get_json()
    got = {h["name"]: h for h in d["hits"]}
    # The stretch itself (a.wav at 12 s) is not a hit; its copies are
    assert set(got) == {"a_copy.mp3", "compilation.mp3"}
    assert got["a_copy.mp3"]["at"] == pytest.approx(12.0, abs=0.1)
    assert got["compilation.mp3"]["at"] == pytest.approx(35 + 12, abs=0.1)
    assert got["compilation.mp3"]["file_dir"] == str(env.lib)
    assert got["compilation.mp3"]["filename"] == "compilation.mp3"
    assert got["a_copy.mp3"]["lufs"] is not None
    assert d["source"]["analyzed"] is True
    # A full path in the allowed roots works the same
    r = env.client.post(
        "/api/audio/search", json={"file": str(env.files["a.wav"]), "start": 12, "end": 18}
    )
    assert {h["name"] for h in r.get_json()["hits"]} == set(got)


def test_a_file_of_ones_own(env, tmp_path):
    clip = write(env.songs["b"][8 * 16000 : 16 * 16000], tmp_path / "mine.mp3", "-b:a", "96k")
    with open(clip, "rb") as f:
        r = env.client.post(
            "/api/audio/search-file",
            data={"file": (f, "mine.mp3")},
            content_type="multipart/form-data",
        )
    assert r.status_code == 200, r.data
    got = {h["name"]: h["at"] for h in r.get_json()["hits"]}
    assert got["b.m4a"] == pytest.approx(8.0, abs=0.1)
    assert got["compilation.mp3"] == pytest.approx(75 + 8, abs=0.1)


@pytest.mark.parametrize(
    "body,code",
    [
        ({"file": "a.wav", "start": 3, "end": 3.5}, 400),  # under a second
        ({"file": "a.wav", "start": 0, "end": 400}, 400),  # over five minutes
        ({"file": "a.wav", "start": "x", "end": 4}, 400),
        ({"file": "", "start": 0, "end": 4}, 400),
        ({"file": "nope.wav", "start": 0, "end": 4}, 404),
        ({"file": "../outside.wav", "start": 0, "end": 4}, 403),
        ({"file": "/etc/passwd", "start": 0, "end": 4}, 403),
    ],
)
def test_what_a_search_refuses(env, body, code):
    assert env.client.post("/api/audio/search", json=body).status_code == code


def test_uploads_that_are_refused(env):
    r = env.client.post(
        "/api/audio/search-file",
        data={"file": (io.BytesIO(b"x"), "notes.txt")},
        content_type="multipart/form-data",
    )
    assert r.status_code == 400
    r = env.client.post(
        "/api/audio/search-file",
        data={"file": (io.BytesIO(b"garbage"), "x.mp3")},
        content_type="multipart/form-data",
    )
    assert r.status_code == 422
    r = env.client.post("/api/audio/search-file", data={}, content_type="multipart/form-data")
    assert r.status_code == 400


def test_switched_off(env):
    env.cfg["enabled"] = False
    assert env.client.get("/api/audio/info").get_json()["enabled"] is False
    assert (
        env.client.post(
            "/api/audio/search", json={"file": "a.wav", "start": 0, "end": 5}
        ).status_code
        == 503
    )
    assert env.client.get("/api/audio/duplicates").status_code == 503
    assert env.client.get("/api/audio/overview").status_code == 503
    assert env.client.get("/api/audio/levels?ids=1").status_code == 503
    assert env.client.post("/api/audio/run/start").status_code == 503


def test_duplicates(env):
    d = env.client.get("/api/audio/duplicates?kind=same").get_json()
    assert d["total"] == 1 and d["sort"] == "space"
    g = d["groups"][0]
    assert sorted(f["name"] for f in g["files"]) == ["a.wav", "a_copy.mp3"]
    assert g["count"] == 2 and g["length"] == pytest.approx(40.0, abs=0.1)
    # Both are sound only: the larger is kept, the other is the extra copy
    assert {f["name"]: f["largest"] for f in g["files"]} == {"a.wav": True, "a_copy.mp3": False}
    assert g["extra"] == 1 and g["extra_bytes"] == env.files["a_copy.mp3"].stat().st_size
    assert all(abs(f["pos"]) < 0.1 and f["filename"] and f["file_dir"] for f in g["files"])
    d = env.client.get("/api/audio/duplicates?kind=contains").get_json()
    assert d["total"] == 1 and d["sort"] == "parts"
    w = d["wholes"][0]
    assert w["whole"]["name"] == "compilation.mp3"
    held = {p["name"]: p["at"] for p in w["parts"]}
    assert held["b.m4a"] == pytest.approx(75.0, abs=0.1)
    assert held["c.opus"] == pytest.approx(0.0, abs=0.1)
    assert "a.wav" in held or "a_copy.mp3" in held
    assert [p["at"] for p in w["parts"]] == sorted(p["at"] for p in w["parts"])
    assert w["covered"] == pytest.approx(105.0, abs=1.0)
    # One recording's
    d = env.client.get("/api/audio/duplicates?kind=contains&file=b.m4a").get_json()
    assert d["total"] == 1 and d["wholes"][0]["whole"]["name"] == "compilation.mp3"
    assert env.client.get("/api/audio/duplicates?kind=same&file=b.m4a").get_json()["total"] == 0
    assert env.client.get("/api/audio/duplicates?file=unknown.wav").get_json()["analyzed"] is False
    # Letters of a path, a shortest length
    assert env.client.get("/api/audio/duplicates?kind=same&q=A_COPY").get_json()["total"] == 1
    assert env.client.get("/api/audio/duplicates?kind=same&min_sec=41").get_json()["total"] == 0
    assert env.client.get("/api/audio/duplicates?kind=contains&q=c.opus").get_json()["total"] == 1
    # Refused
    assert env.client.get("/api/audio/duplicates?kind=other").status_code == 400
    assert env.client.get("/api/audio/duplicates?limit=x").status_code == 400


def test_the_overview(env):
    d = env.client.get("/api/audio/overview").get_json()
    assert d["same"] == {
        "groups": 1,
        "files": 2,
        "extra": 1,
        "extra_bytes": env.files["a_copy.mp3"].stat().st_size,
    }
    assert d["contains"]["wholes"] == 1 and d["contains"]["parts"] >= 3
    assert d["analyzed"] == 5 and d["folders_total"] == 1
    f = d["folders"][0]
    # Both copies in the one folder: a pair of that folder with itself
    assert f["within"] and f["a"].endswith("/lib") and f["a_files"] == 2 and f["a_total"] == 5
    assert env.client.get("/api/audio/overview?q=nothing-like-it").get_json()["same"]["groups"] == 0
    assert env.client.get("/api/audio/overview?min_sec=x&folders=y").status_code == 200


def test_the_waveforms(env):
    import base64

    a = env.store.file(str(env.files["a.wav"]))
    d = env.client.get(f"/api/audio/levels?ids={a['id']},999&n=50").get_json()
    v = base64.b64decode(d["levels"][str(a["id"])])
    assert len(v) == 50 and max(v) > 100 and "999" not in d["levels"]
    assert env.client.get("/api/audio/levels?ids=x").status_code == 400
    assert env.client.get("/api/audio/levels?ids=" + ",".join(["1"] * 301)).status_code == 400


def test_one_recording(env):
    d = env.client.get("/api/audio/file?file=a.wav").get_json()
    assert d["analyzed"] and d["copies"] == 1 and d["contained_in"] == 1 and d["current"]
    f = d["file"]
    assert f["duration"] == pytest.approx(40.0, abs=0.05)
    assert -40 < f["lufs"] < -5 and f["peak_db"] <= 0 and 0 <= f["silence"] <= 1
    d = env.client.get("/api/audio/file?file=compilation.mp3").get_json()
    assert d["contains"] >= 3
    assert env.client.get("/api/audio/file?file=unknown.wav").get_json()["analyzed"] is False


def test_retry_and_clear(env):
    d = env.client.post("/api/audio/retry-failed").get_json()
    assert d["forgotten"] == 1
    assert env.store.file(str(env.lib / "silent.wav")) is None
    assert env.client.post("/api/audio/clear").get_json()["ok"]
    assert env.client.get("/api/audio/info").get_json()["counts"]["files"] == 0


def test_the_run_from_the_page(env):
    p = write(song(9, 10), env.lib / "new.mp3")
    with sqlite3.connect(env.main) as c:
        st = p.stat()
        c.execute(
            "INSERT INTO media_files(abs_path, file_size, modified_at, duration_sec, has_audio) "
            "VALUES(?,?,?,10,1)",
            (str(p), st.st_size, st.st_mtime),
        )
    runner.forget_scope()
    assert env.client.get("/api/audio/estimate?fresh=1").get_json()["files"] == 1
    d = env.client.post("/api/audio/run/start").get_json()
    assert d["started"]
    runner.RUNNER.join(30)
    assert env.store.file(str(p))["status"] == "done"
    assert env.client.post("/api/audio/run/stop").get_json()["stopping"] is False
    # The index can't be emptied under a running run
    import threading

    hold = threading.Event()
    runner.RUNNER._thread = threading.Thread(target=hold.wait, daemon=True)
    runner.RUNNER._thread.start()
    try:
        assert env.client.post("/api/audio/clear").status_code == 409
    finally:
        hold.set()
        runner.RUNNER._thread.join()
