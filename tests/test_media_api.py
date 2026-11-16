"""The routes of Images & sounds (api/media_search.py) — searching, file
queries, pictures and clips, the language model, the index, the environment.
The service is a stand-in; ffmpeg is real (clips are cut from a test video)."""

from __future__ import annotations

import io
import os
import shutil
import subprocess
import time

import numpy as np
import pytest
from flask import Flask
from PIL import Image

from whisper_tm.api import media_search as api
from whisper_tm.core import media_server
from whisper_tm.media import cuts, describe, environment, indexer, llm, search
from whisper_tm.media import store as store_mod
from whisper_tm.media.service import b64

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not available")


def _vec(seed: int) -> np.ndarray:
    v = np.random.default_rng(seed).standard_normal(16).astype(np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture
def env(tmp_path, monkeypatch):
    cfg = {
        "enabled": True,
        "cache_dir": str(tmp_path / "cache"),
        "results_per_kind": 5,
        "clip_sec": 4,
        "llm_translate_sounds": True,
        "llm_check_count": 3,
        "thumb_width": 64,
        "kinds": ["image", "audio"],
    }
    get = lambda name, default=None: cfg.get(name, default)  # noqa: E731
    for mod in (api, indexer, llm, media_server, environment):
        monkeypatch.setattr(mod, "_cfg", get)
    monkeypatch.setattr(store_mod, "store_path", lambda: tmp_path / "media.db")
    store_mod._stores.clear()
    from whisper_tm import extensions as ext

    monkeypatch.setattr(ext, "db", None)
    indexer._scope.update(key=None)  # another test's library, kept for minutes
    monkeypatch.setattr(
        environment, "check", lambda force=False: {"ok": True, "problems": [], "model_cached": True}
    )
    monkeypatch.setattr(llm, "configured", lambda: True)
    monkeypatch.setattr(llm, "vision_available", lambda: True)
    monkeypatch.setattr(llm, "sound_label", lambda q: "dog barking")
    search._tags["vectors"] = None

    # Media: a 6 s video with sound, a 6 s sound
    video = tmp_path / "lib" / "street.mp4"
    sound = tmp_path / "lib" / "note.m4a"
    video.parent.mkdir()
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
            "testsrc=size=160x120:rate=10",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440",
            "-t",
            "6",
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            "-shortest",
            str(video),
        ],
        check=True,
    )
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
            "sine=frequency=220",
            "-t",
            "6",
            "-c:a",
            "aac",
            str(sound),
        ],
        check=True,
    )
    store = store_mod.get_store()
    store.ensure_dims(16)
    thumbs = indexer.thumbs_dir()
    fv = store.start_file(str(video), video=True, duration=6.0, settings="fp")
    t1 = thumbs / str(fv["id"]) / "20.jpg"
    t1.parent.mkdir(parents=True)
    Image.new("RGB", (64, 48), (200, 30, 30)).save(t1)
    store.add_moments(
        fv["id"],
        [
            {
                "t": 2.0,
                "start": 0.0,
                "dur": 4.0,
                "thumb": str(t1),
                "text": "[SPEAKER_00] The police are over there.",
                "vectors": {"image": _vec(1), "audio": _vec(2)},
            },
            {
                "t": 4.0,
                "start": 2.0,
                "dur": 4.0,
                "thumb": str(thumbs / str(fv["id"]) / "40.jpg"),
                "text": "",
                "vectors": {"image": _vec(3)},
            },
        ],
        next_t=6.0,
        prev=None,
        done=True,
        seconds=1.0,
    )
    fa = store.start_file(str(sound), video=False, duration=6.0, settings="fp")
    store.add_moments(
        fa["id"],
        [
            {
                "t": 3.0,
                "start": 0.0,
                "dur": 6.0,
                "thumb": None,
                "text": "",
                "vectors": {"audio": _vec(4)},
            }
        ],
        next_t=6.0,
        prev=None,
        done=True,
        seconds=1.0,
    )
    ms = {
        r[0]: r for r in store.connect().__enter__().execute("SELECT id, file_id, t FROM moments")
    }

    calls = []

    def fake_call(path, payload=None, *, timeout=120):
        calls.append((path, payload))
        if path == "/embed":
            return {"vectors": [b64(_vec(1 if "police" in t else 99)) for t in payload["texts"]]}
        if path == "/embed-file":
            return {"vector": b64(_vec(4)), "ms": 5}
        return {}

    monkeypatch.setattr(media_server.SERVER, "call", fake_call)
    app = Flask(__name__)
    app.register_blueprint(api.media_search_bp)
    yield {
        "client": app.test_client(),
        "cfg": cfg,
        "store": store,
        "calls": calls,
        "video": video,
        "sound": sound,
        "moments": sorted(ms),
        "thumb": t1,
        "tmp": tmp_path,
    }
    store_mod._stores.clear()


class TestInfoAndSearch:
    def test_info(self, env):
        d = env["client"].get("/api/media-search/info").get_json()
        assert d["enabled"] and d["ready"] and d["environment_ok"]
        assert d["counts"]["kinds"]["image"] == 2 and d["counts"]["kinds"]["audio"] == 2
        assert d["llm"]["vision"] and d["results_per_kind"] == 5 and d["dims"] == 16

    def test_a_text_asks_pictures_as_typed_and_sounds_by_their_label(self, env):
        d = env["client"].get("/api/media-search?q=police").get_json()
        assert (
            d["source"] == {"type": "text", "text": "police"} and d["sound_label"] == "dog barking"
        )
        embedded = [
            p["texts"][0] for path, p in env["calls"] if path == "/embed" and len(p["texts"]) == 1
        ]
        assert embedded[:2] == ["police", "dog barking"]
        img = d["results"]["image"]
        assert img[0]["score"] == pytest.approx(1.0, abs=1e-3)  # the still made of the same vector
        h = img[0]
        assert h["thumb"] == f"/api/media-search/thumb/{h['moment']}" and h["video"]
        assert h["name"] == "street.mp4" and h["file_dir"] == str(env["video"].parent)
        assert h["text"].startswith("[SPEAKER_00]") and isinstance(h["tags"], list)
        assert set(d["results"]) == set(store_mod.KINDS)

    def test_similar_leaves_the_item_itself_out(self, env):
        first = (
            env["client"]
            .get("/api/media-search?q=police&kinds=image")
            .get_json()["results"]["image"][0]
        )
        d = env["client"].get(f"/api/media-search?like={first['id']}&kinds=image").get_json()
        assert d["source"]["type"] == "like" and first["id"] not in [
            h["id"] for h in d["results"]["image"]
        ]

    def test_fewer_numbers_and_folders(self, env):
        d = env["client"].get("/api/media-search?q=police&dims=128&kinds=image").get_json()
        assert d["dims"] == 16  # never more than stored
        d = env["client"].get(f"/api/media-search?q=police&folders={env['tmp']}/other").get_json()
        assert all(v == [] for v in d["results"].values())

    def test_bad_requests(self, env):
        c = env["client"]
        assert c.get("/api/media-search?like=abc").get_json()["error"] == "like is an item id"
        # Ids past SQLite's integers were a 500 (OverflowError); 5,000
        # characters took half a minute to embed
        assert c.get(f"/api/media-search?like={2**70}").status_code == 404
        assert c.get(f"/api/media-search/thumb/{2**70}").status_code == 404
        assert c.get(f"/api/media-search/clip/{2**70}").status_code == 404
        assert c.get("/api/media-search?q=" + "x" * 501).status_code == 400
        assert c.post("/api/media-search/check/nope/cancel").status_code == 404
        assert c.get("/api/media-search?q=a").status_code == 400
        assert c.get("/api/media-search?q=abc&dims=300").status_code == 400
        assert c.get("/api/media-search?like=999999").status_code == 404
        assert c.get("/api/media-search?qid=nope").status_code == 404
        env["cfg"]["enabled"] = False
        assert c.get("/api/media-search?q=abc").status_code == 503

    def test_a_file_as_the_query(self, env):
        buf = io.BytesIO()
        Image.new("RGB", (8, 8)).save(buf, "PNG")
        r = env["client"].post(
            "/api/media-search/query-file",
            data={
                "file": (io.BytesIO(buf.getvalue()), "me.png", "image/png"),
                "q": "at night",
                "with_sound": "1",
            },
            content_type="multipart/form-data",
        )
        d = r.get_json()
        assert r.status_code == 200 and d["kind"] == "image" and d["text"] == "at night"
        sent = [p for path, p in env["calls"] if path == "/embed-file"][-1]
        assert sent["kind"] == "image" and sent["text"] == "at night" and sent["with_sound"]
        assert not list((indexer.cache_dir() / "uploads").iterdir())  # the upload is gone again
        res = env["client"].get(f"/api/media-search?qid={d['qid']}&kinds=audio").get_json()
        assert res["source"]["type"] == "file" and res["results"]["audio"][0][
            "score"
        ] == pytest.approx(1.0, abs=1e-3)

    def test_a_file_that_is_neither_picture_nor_sound(self, env):
        r = env["client"].post(
            "/api/media-search/query-file",
            data={"file": (io.BytesIO(b"%PDF"), "a.pdf", "application/pdf")},
            content_type="multipart/form-data",
        )
        assert r.status_code == 400


class TestPicturesAndClips:
    def test_a_still(self, env):
        m = env["moments"][0]
        r = env["client"].get(f"/api/media-search/thumb/{m}")
        assert r.status_code == 200 and r.mimetype == "image/jpeg"

    def test_a_lost_still_is_cut_again_from_the_recording(self, env):
        m = env["moments"][1]
        r = env["client"].get(f"/api/media-search/thumb/{m}")
        assert r.status_code == 200 and Image.open(io.BytesIO(r.data)).size[0] == 64

    def test_no_file_outside_the_thumbnail_folder(self, env):
        with env["store"].connect() as c:
            c.execute("UPDATE moments SET thumb=? WHERE id=?", ("/etc/passwd", env["moments"][0]))
        assert env["client"].get(f"/api/media-search/thumb/{env['moments'][0]}").status_code == 404

    def test_a_clip_of_a_video_moment_and_of_a_sound(self, env):
        c = env["client"]
        v = c.get(f"/api/media-search/clip/{env['moments'][0]}")
        assert v.status_code == 200 and v.mimetype == "video/mp4" and len(v.data) > 1000
        t0 = time.monotonic()
        assert c.get(f"/api/media-search/clip/{env['moments'][0]}").status_code == 200
        assert time.monotonic() - t0 < 1.0  # kept, not cut again
        a = c.get(f"/api/media-search/clip/{env['moments'][2]}")
        assert a.status_code == 200 and a.mimetype == "audio/mp4"
        part = c.get(f"/api/media-search/clip/{env['moments'][0]}", headers={"Range": "bytes=0-99"})
        assert part.status_code == 206 and len(part.data) == 100
        assert c.get("/api/media-search/clip/999999").status_code == 404

    def test_the_clip_cache_keeps_its_size(self, env):
        folder = indexer.cache_dir() / "clips"
        folder.mkdir(parents=True, exist_ok=True)
        old = time.time() - 3600
        for k in range(5):
            (folder / f"old{k}.mp4").write_bytes(b"x" * 400_000)
            os.utime(folder / f"old{k}.mp4", (old + k, old + k))
        env["cfg"]["cache_max_mb"] = 1
        api._trim_clips(folder)
        assert sum(p.stat().st_size for p in folder.iterdir()) <= 1024 * 1024

    def test_a_new_moment_with_a_reused_id_gets_its_own_clip(self, env):
        # Clips were named clips/<moment id>.mp4 — SQLite reuses the ids of deleted rows,
        # so a new moment got an old recording's clip
        c, store = env["client"], env["store"]
        mid = env["moments"][2]
        first = c.get(f"/api/media-search/clip/{mid}").data
        store.remove_files([str(env["sound"])])
        other = env["tmp"] / "lib" / "other.m4a"
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
                "sine=frequency=880",
                "-t",
                "3",
                "-c:a",
                "aac",
                str(other),
            ],
            check=True,
        )
        f = store.start_file(str(other), video=False, duration=3.0, settings="fp")
        store.add_moments(
            f["id"],
            [
                {
                    "t": 1.5,
                    "start": 0.0,
                    "dur": 3.0,
                    "thumb": None,
                    "text": "",
                    "vectors": {"audio": _vec(5)},
                }
            ],
            next_t=3.0,
            prev=None,
            done=True,
            seconds=1.0,
        )
        assert store.file_moments(f["id"])[0]["id"] == mid, "the id was not reused"
        second = c.get(f"/api/media-search/clip/{mid}")
        assert second.status_code == 200 and second.data != first

    def test_an_unreadable_recording_has_no_clip(self, env):
        # Previously a 500 for a recording ffmpeg can't read — there is nothing to serve
        env["sound"].write_bytes(b"not a recording")
        assert env["client"].get(f"/api/media-search/clip/{env['moments'][2]}").status_code == 404


class TestTheCutCache:
    """media/cuts.py — the cache Images & sounds and Find cut into."""

    def test_trim_after_real_cuts_only_and_a_hit_renews_the_file(self, tmp_path):
        out, made, trims = tmp_path / "a.mp4", [], []

        def make(o):
            made.append(o)
            o.write_bytes(b"x")
            return True

        assert cuts.cached(out, make, after_cut=lambda: trims.append(1)) == out
        os.utime(out, (1, 1))
        assert cuts.cached(out, make, after_cut=lambda: trims.append(1)) == out
        assert len(made) == 1 and trims == [1], "trimmed on a cache hit"
        assert out.stat().st_mtime > 1000, "a hit did not renew the file's time"

    def test_young_files_are_never_trimmed(self, tmp_path):
        # A concurrent trim once deleted a clip right before send_file
        for k in range(3):
            (tmp_path / f"{k}.mp4").write_bytes(b"x" * 600_000)
        os.utime(tmp_path / "0.mp4", (1, 1))
        cuts.trim(tmp_path, 1, suffixes=(".mp4",))
        assert sorted(p.name for p in tmp_path.iterdir()) == ["1.mp4", "2.mp4"]

    def test_one_cut_per_file_while_others_wait(self, tmp_path):
        # Previously the first request dropped the file's lock while a second
        # waited on it; a third made a new one — two ffmpeg on one .part
        import threading

        out, running, overlap = tmp_path / "c.mp4", [0], []
        go = [threading.Event() for _ in range(3)]
        entered = threading.Semaphore(0)

        def make(o):
            n = len(overlap)
            running[0] += 1
            overlap.append(running[0])
            entered.release()
            go[n].wait(5)
            running[0] -= 1
            return False  # failed: the next waiter tries again

        def start():
            t = threading.Thread(target=cuts.cached, args=(out, make))
            t.start()
            return t

        threads = [start(), start()]
        assert entered.acquire(timeout=5)  # the first cuts, the second waits
        go[0].set()
        assert entered.acquire(timeout=5)  # the second cuts now
        threads.append(start())  # a third comes while it does
        time.sleep(0.2)
        go[1].set()
        assert entered.acquire(timeout=5)
        go[2].set()
        for t in threads:
            t.join(10)
        assert overlap == [1, 1, 1], overlap
        assert not cuts._locks


class TestTheLanguageModel:
    def test_a_check_judges_each_picture_and_reports_as_it_goes(self, env, monkeypatch):
        monkeypatch.setattr(llm, "judge", lambda q, thumb: 8.5 if q == "police" else 1.0)
        m = env["moments"][0]
        d = (
            env["client"]
            .post("/api/media-search/check", json={"query": "police", "moments": [m]})
            .get_json()
        )
        for _ in range(100):
            st = env["client"].get(f"/api/media-search/check/{d['id']}").get_json()
            if not st["running"]:
                break
            time.sleep(0.02)
        assert st["done"] == 1 and st["scores"] == {str(m): 8.5}
        assert env["client"].post(f"/api/media-search/check/{d['id']}/cancel").get_json() == {
            "cancelled": False
        }

    def test_a_check_needs_pictures_and_a_model_that_sees(self, env, monkeypatch):
        c = env["client"]
        assert (
            c.post(
                "/api/media-search/check", json={"query": "x", "moments": [env["moments"][2]]}
            ).status_code
            == 400
        )
        monkeypatch.setattr(llm, "vision_available", lambda: False)
        assert (
            c.post(
                "/api/media-search/check", json={"query": "x", "moments": [env["moments"][0]]}
            ).status_code
            == 503
        )
        assert c.get("/api/media-search/check/nope").status_code == 404

    def test_describe_writes_once_and_makes_it_searchable(self, env, monkeypatch):
        said = []
        monkeypatch.setattr(
            llm, "describe", lambda thumb: said.append(thumb) or "Police by a street."
        )
        m = env["moments"][0]
        d = env["client"].post(f"/api/media-search/describe/{m}", json={}).get_json()
        assert d == {"text": "Police by a street.", "model": "", "cached": False}
        assert env["store"].counts()["kinds"]["description"] == 1
        again = env["client"].post(f"/api/media-search/describe/{m}", json={}).get_json()
        assert again["cached"] and len(said) == 1
        hits = (
            env["client"]
            .get("/api/media-search?q=police&kinds=description")
            .get_json()["results"]["description"]
        )
        assert hits and hits[0]["moment"] == m

    def test_a_describing_run_goes_into_the_llm_queue(self, env, monkeypatch):
        monkeypatch.setattr(api, "_describe_entry", lambda: None)
        got = []
        monkeypatch.setattr(
            describe, "enqueue", lambda limit, folders: got.append((limit, folders)) or 7
        )
        env["cfg"]["llm_describe_limit"] = 50
        r = env["client"].post("/api/media-search/describe-run", json={})
        assert r.status_code == 200 and r.get_json()["entry"] == 7 and got == [(50, [])]
        monkeypatch.setattr(api, "_describe_entry", lambda: {"id": 7, "status": "running"})
        assert env["client"].post("/api/media-search/describe-run", json={}).status_code == 409
        monkeypatch.setattr(api, "_describe_entry", lambda: None)
        r = env["client"].post("/api/media-search/describe-run", json={"limit": "many"})
        assert r.status_code == 400 and got == [(50, [])]

    def test_a_search_failure_is_not_blamed_on_like(self, env, monkeypatch):
        # Previously any ValueError of the search was answered "like is an item id"
        def broken(**kw):
            raise ValueError("something inside")

        monkeypatch.setattr(search, "search", broken)
        r = env["client"].get("/api/media-search?q=police")
        assert r.status_code == 500


class TestTheIndexAndTheEnvironment:
    def test_index_state_answers_at_once_and_the_estimate_follows(self, env, monkeypatch):
        import threading

        go = threading.Event()
        real = indexer.estimate
        monkeypatch.setattr(indexer, "estimate", lambda s=None, fresh=False: go.wait(5) and real(s))
        indexer.forget_estimate()
        t0 = time.monotonic()
        d = env["client"].get("/api/media-search/index").get_json()
        assert time.monotonic() - t0 < 2  # not waiting for the estimate
        assert d["counts"]["moments"] == 3 and d["descriptions"]["stills"] == 2
        assert d["estimate"] is None and d["estimate_computing"]
        assert d["ram_bytes"] >= 0 and "running" in d["run"]
        go.set()
        for _ in range(200):
            d = env["client"].get("/api/media-search/index").get_json()
            if d["estimate"]:
                break
            time.sleep(0.01)
        assert (
            d["estimate"]["files"] == 0 and "share_done" in d["estimate"]
        )  # no transcripts in this library
        assert "estimate" not in env["client"].get("/api/media-search/index?estimate=0").get_json()

    def test_start_needs_the_environment(self, env, monkeypatch):
        monkeypatch.setattr(
            environment, "check", lambda force=False: {"ok": False, "problems": ["not installed"]}
        )
        assert env["client"].post("/api/media-search/index/start", json={}).status_code == 409
        monkeypatch.setattr(environment, "check", lambda force=False: {"ok": True, "problems": []})
        started = []
        monkeypatch.setattr(
            indexer.INDEXER,
            "start",
            lambda rebuild=False, reason="": started.append(rebuild) or True,
        )
        d = env["client"].post("/api/media-search/index/start", json={"rebuild": True}).get_json()
        assert d["started"] and started == [True]

    def test_clear_empties_index_and_cache(self, env):
        (indexer.cache_dir() / "clips").mkdir(parents=True, exist_ok=True)
        d = env["client"].post("/api/media-search/index/clear").get_json()
        assert d["cleared"] and d["counts"]["moments"] == 0
        assert (
            not (indexer.cache_dir() / "thumbs").exists()
            and not (indexer.cache_dir() / "clips").exists()
        )
        assert env["store"].meta("thumb_bytes") == 0

    def test_clear_leaves_a_folder_it_did_not_write(self, env):
        """cache_dir is a setting: pointed at someone else's folder, clearing
        must not delete their "thumbs"."""
        theirs = indexer.cache_dir() / "thumbs"
        theirs.mkdir(parents=True, exist_ok=True)
        (theirs / "1.jpg").write_bytes(b"x")
        (theirs / "notes.txt").write_text("mine")
        d = env["client"].post("/api/media-search/index/clear").get_json()
        assert d["cleared"] and str(theirs) in d["kept"]
        assert (theirs / "notes.txt").exists() and (theirs / "1.jpg").exists()

    def test_nothing_is_cleared_while_a_run_goes(self, env, monkeypatch):
        monkeypatch.setattr(indexer.INDEXER, "running", lambda: True)
        assert env["client"].post("/api/media-search/index/clear").status_code == 409
        assert env["client"].post("/api/media-search/environment/install").status_code == 409
        assert env["client"].post("/api/media-search/service/stop").status_code == 409

    def test_environment_and_install(self, env, monkeypatch):
        monkeypatch.setattr(environment, "install_state", lambda: {"running": False, "lines": []})
        d = env["client"].get("/api/media-search/environment").get_json()
        assert d["check"]["ok"] and d["install"]["running"] is False and "running" in d["service"]
        monkeypatch.setattr(environment, "start_install", lambda: True)
        monkeypatch.setattr(media_server.SERVER, "stop", lambda reason="", force=False: True)
        assert env["client"].post("/api/media-search/environment/install").get_json()["started"]
