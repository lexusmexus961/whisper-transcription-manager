"""Index runs of Images & sounds (media/indexer.py): which recordings, which
moments, resuming, kinds added later, errors, the disk, and the GPU — a
guest behind the gate."""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from whisper_tm.core import media_server
from whisper_tm.core.gpu_gate import GpuGate
from whisper_tm.media import indexer
from whisper_tm.media import store as store_mod
from whisper_tm.media.service import b64


@pytest.fixture
def lib(tmp_path, monkeypatch):
    """A library of three recordings and a media index of its own."""
    from whisper_tm import extensions as ext
    from whisper_tm.database import Database

    db = Database(tmp_path / "lib.db")
    db.initialize()
    monkeypatch.setattr(ext, "db", db)
    root = tmp_path / "media"
    (root / "streams").mkdir(parents=True)
    (root / "voice").mkdir(parents=True)
    cfg = {
        "cache_dir": str(tmp_path / "cache"),
        "kinds": ["image", "video", "audio"],
        "dims": 512,
        "video_interval_sec": 20,
        "sound_window_sec": 10,
        "clip_frames": 0,
        "dedupe_similarity": 0.97,
        "index_device": "cpu",
        "chunk_moments": 4,
        "yield_after_sec": 0,
        "folders": [],
        "parallel_files": 1,
    }
    monkeypatch.setattr(indexer, "_cfg", lambda name, default=None: cfg.get(name, default))
    monkeypatch.setattr(store_mod, "store_path", lambda: tmp_path / "media.db")
    store_mod._stores.clear()
    monkeypatch.setattr(indexer, "QUEUE_WAIT_SEC", 0.02)
    monkeypatch.setattr(indexer, "OOM_PAUSE_SEC", 0.01)
    monkeypatch.setattr(indexer, "YIELD_PAUSE_SEC", 0.01)
    monkeypatch.setattr(indexer, "DISK_WAIT_SEC", 0.01)
    monkeypatch.setattr(indexer, "disk_free_gb", lambda: 500.0)
    indexer._scope.update(key=None)
    indexer.forget_estimate()
    t = [1000.0]

    def add(rel, duration, *, exists=True):
        p = root / rel
        if exists:
            p.write_bytes(b"x")
        segs = [
            {"start": s, "end": s + 4, "text": f"said at {s}", "speaker": "SPEAKER_00"}
            for s in range(0, int(duration), 5)
        ]
        db.upsert_transcript(
            rel,
            {"complete": True, "segments": segs, "total_duration_sec": duration},
            media_dir=str(root),
            abs_path=str(p),
        )
        t[0] += 10
        with db.connection() as c:  # newest first is the order of a run
            c.execute(
                "UPDATE transcripts SET duration_sec=?, updated_at=? WHERE abs_path=?",
                (duration, t[0], str(p)),
            )
        return str(p)

    files = {
        "stream": add("streams/live.mp4", 70.0),
        "note": add("voice/note.opus", 25.0),
        "gone": add("streams/gone.mkv", 40.0, exists=False),
    }
    yield {"cfg": cfg, "files": files, "add": add, "root": root, "tmp": tmp_path, "db": db}
    store_mod._stores.clear()
    db.close_all()


class FakeService:
    """Answers the indexer like media/service.py would — a distinct vector
    per moment and kind; repeats where ``repeat_at`` says (or the request
    knows them); ``thumb_bytes`` per kept still."""

    VISUAL = ("image", "video", "videoaudio")

    def __init__(self):
        self.calls = []
        self.fail: list = []
        self.repeat_at: set = set()
        self.thumb_size = 0

    def __call__(self, path, payload=None, *, timeout=120):
        self.calls.append((path, dict(payload or {})))
        if self.fail:
            raise self.fail.pop(0)
        if path == "/release":
            return {"released": True}
        out = []
        known = payload.get("repeats")
        looks = payload["video"] and (
            ("image" in payload["kinds"])
            if known is not None
            else bool(set(self.VISUAL) & set(payload["kinds"]))
        )
        thumbs = 0
        for k, m in enumerate(payload["moments"]):
            rep = bool(known[k]) if known is not None else (looks and m["t"] in self.repeat_at)
            still = payload["video"] and not rep and (looks or known is not None)
            if still and looks and payload.get("thumbs", True):
                thumbs += self.thumb_size
            vecs = {}
            for kind in payload["kinds"]:
                if kind in self.VISUAL and (not payload["video"] or rep):
                    continue
                v = (
                    np.random.default_rng(int(m["t"] * 10) + len(kind))
                    .standard_normal(16)
                    .astype(np.float32)
                )
                vecs[kind] = b64(v / np.linalg.norm(v))
            out.append({"repeat": rep, "still": still, "vectors": vecs})
        return {
            "moments": out,
            "prev": b64(np.ones(16, dtype=np.float32) / 4),
            "timing": {},
            "thumb_bytes": thumbs,
        }


@pytest.fixture
def service(monkeypatch):
    fake = FakeService()
    monkeypatch.setattr(media_server.SERVER, "call", fake)
    return fake


def _run(ix=None, rebuild=False):
    ix = ix or indexer.Indexer()
    assert ix.start(rebuild=rebuild)
    for _ in range(500):
        if not ix.running():
            break
        time.sleep(0.01)
    return ix.status()


class TestPlanning:
    def test_video_moments_and_their_sound_windows(self):
        s = {"video_interval_sec": 20, "sound_window_sec": 10}
        ms = indexer.plan_moments(70.0, True, s)
        assert [m["t"] for m in ms] == [10, 30, 50, 69.5]
        assert ms[0]["start"] == 5 and ms[0]["dur"] == 10
        assert ms[-1]["start"] == 60 and ms[-1]["dur"] == 10  # kept inside the file

    def test_audio_windows_back_to_back(self):
        ms = indexer.plan_moments(25.0, False, {"video_interval_sec": 20, "sound_window_sec": 10})
        assert [(m["start"], m["dur"]) for m in ms] == [(0, 10), (10, 10), (20, 5)]

    def test_resuming_starts_at_the_next_moment(self):
        s = {"video_interval_sec": 20, "sound_window_sec": 10}
        assert [m["t"] for m in indexer.plan_moments(70.0, True, s, frm=30)] == [30, 50, 69.5]

    @pytest.mark.parametrize("video", [True, False])
    def test_resuming_from_the_stored_next_t_skips_nothing(self, video):
        # next_t is a moment's centre; sounds used to be filtered by their
        # start — a resumed sound file lost one window for good
        s = {"video_interval_sec": 20, "sound_window_sec": 10}
        full = indexer.plan_moments(95.0, video, s)
        for n in range(1, len(full)):
            again = indexer.plan_moments(95.0, video, s, frm=full[n]["t"])
            assert again == full[n:], n
            assert indexer.count_moments(95.0, video, s, full[n]["t"]) == len(full) - n

    def test_a_short_video_has_one_moment(self):
        assert (
            len(indexer.plan_moments(5.0, True, {"video_interval_sec": 20, "sound_window_sec": 10}))
            == 1
        )

    def test_frames_per_clip_automatic(self):
        assert indexer.frames_for({"clip_frames": 0}, 100) == 8
        assert indexer.frames_for({"clip_frames": 0}, 22_000) == 4
        assert indexer.frames_for({"clip_frames": 6}, 22_000) == 6
        assert indexer.frames_for({"clip_frames": 0}, 22_000, kept=8) == 8  # decided once

    @pytest.mark.parametrize("video", [True, False])
    def test_moments_are_counted_without_making_them(self, video):
        s = {"video_interval_sec": 20, "sound_window_sec": 10}
        for dur in (0.4, 1.0, 5.0, 10.0, 10.5, 11.0, 25.0, 69.9, 70.0, 3600.7):
            for frm in (0.0, 10.0, 15.0, 30.0, 45.0, 69.5, 5000.0):
                assert indexer.count_moments(dur, video, s, frm) == len(
                    indexer.plan_moments(dur, video, s, frm)
                ), (dur, frm)

    def test_settings_are_clamped_and_kinds_known(self, lib):
        lib["cfg"].update(
            kinds=["image", "smell", "audio"], dims=300, video_interval_sec=0.1, folders="/a/\n/b\n"
        )
        s = indexer.settings()
        assert s["kinds"] == ["image", "audio"]
        assert s["dims"] == 256 and s["video_interval_sec"] == 2
        assert s["folders"] == ["/a", "/b"]

    def test_the_fingerprint_follows_what_the_moments_depend_on(self, lib):
        s = indexer.settings()
        a = indexer.fingerprint(s, (512, "int8"))
        assert indexer.fingerprint(s, (768, "int8")) != a
        assert indexer.fingerprint(s, (512, "float16")) != a
        assert indexer.fingerprint({**s, "video_interval_sec": 10}, (512, "int8")) != a
        # kinds and clips are kept per file: no reason to start over
        assert indexer.fingerprint({**s, "kinds": ["audio"], "clip_sec": 4}, (512, "int8")) == a

    def test_what_a_file_should_have(self, lib):
        s = {**indexer.settings(), "kinds": list(indexer.ALL_KINDS)}
        assert indexer.kind_specs(s, 4, True) == [
            "image",
            "video@4x10",
            "videoaudio@4x10",
            "audio",
            "speech",
        ]
        assert indexer.kind_specs(s, 4, False) == ["audio", "speech"]

    def test_what_a_file_needs(self):
        want = ["image", "video@4x10", "audio"]
        done = {"settings": "b", "status": "done", "kinds": ["audio", "image", "video@4x10"]}
        assert indexer.plan_file(None, "b", want) == ("full", want)
        assert indexer.plan_file({**done, "settings": "a"}, "b", want) == ("full", want)
        assert indexer.plan_file(done, "b", want) == ("done", [])
        assert indexer.plan_file(done, "b", want + ["speech"]) == ("fill", ["speech"])
        assert indexer.plan_file(done, "b", ["image", "video@8x10", "audio"]) == (
            "fill",
            ["video@8x10"],
        )
        assert indexer.plan_file(done, "b", ["audio"]) == ("done", [])  # switched off: kept
        part = {**done, "status": "partial"}
        assert indexer.plan_file(part, "b", want) == ("continue", sorted(want))
        assert indexer.plan_file(part, "b", ["audio"]) == ("full", ["audio"])


class TestScope:
    def test_newest_first_and_folders(self, lib):
        s = indexer.settings()
        paths = [f["path"] for f in indexer.scope_files(s)]
        assert paths == [lib["files"]["gone"], lib["files"]["note"], lib["files"]["stream"]]
        lib["cfg"]["folders"] = [str(lib["root"] / "voice")]
        assert [f["path"] for f in indexer.scope_files(indexer.settings())] == [
            lib["files"]["note"]
        ]

    def test_estimate(self, lib):
        e = indexer.estimate()
        assert e["files"] == 3 and e["todo"] == 3 and e["clip_frames"] == 8
        # live.mp4 4 moments + gone.mkv 2, note.opus 3 windows
        assert e["moments"] == 9
        assert e["vectors"] == 6 * 2 + 9  # stills+clips of video moments, a sound each
        assert e["vectors_by_kind"] == {"image": 6, "video": 6, "audio": 9}
        assert e["bytes"] == 21 * 512  # 8-bit: a byte a number
        assert e["thumb_bytes"] == 6 * indexer.THUMB_BYTES
        assert e["gpu_hours"] >= 0 and e["hours_done"] == 0 and e["share_done"] == 0
        assert e["free_gb"] == 500.0 and e["min_free_gb"] == 20

    def test_a_thumbnail_limit_of_0_means_none_in_the_estimate_too(self, lib):
        # the run writes every still with 0 (_thumbs_on); the estimate
        # counted 0 as no room at all
        lib["cfg"]["thumbs_max_mb"] = 0
        assert indexer.estimate()["thumb_bytes"] == 6 * indexer.THUMB_BYTES
        lib["cfg"]["thumbs_max_mb"] = 1
        indexer.forget_estimate()
        assert indexer.estimate()["thumb_bytes"] == min(6 * indexer.THUMB_BYTES, 1024**2)

    def test_estimate_after_a_run_and_what_a_new_kind_would_take(self, lib, service):
        _run()
        e = indexer.estimate()
        assert e["todo"] == 1 and e["done"] == 2  # the missing one stays to do
        assert e["share_done"] == pytest.approx(95 / 135, abs=1e-3)  # 70 s + 25 s of 135 s
        lib["cfg"]["kinds"] = ["image", "video", "audio", "speech"]
        e = indexer.estimate()
        assert e["fills"] == 2 and e["share_done"] == 0
        assert e["vectors_by_kind"] == {"speech": 4 + 3 + 2, "image": 2, "video": 2, "audio": 2}

    def test_the_estimate_is_never_waited_for(self, lib, monkeypatch):
        go = threading.Event()
        real = indexer.estimate
        monkeypatch.setattr(indexer, "estimate", lambda s=None, fresh=False: go.wait(5) and real(s))
        first = indexer.estimate_cached()
        assert first["value"] is None and first["computing"]
        go.set()
        for _ in range(200):
            got = indexer.estimate_cached()
            if got["value"]:
                break
            time.sleep(0.01)
        assert got["value"]["files"] == 3 and not got["computing"]
        lib["cfg"]["folders"] = [str(lib["root"] / "voice")]  # other settings: a new one
        assert indexer.estimate_cached()["value"] is None


class TestRuns:
    def test_a_run_indexes_what_is_there_and_counts_what_is_not(self, lib, service):
        st = _run()
        assert st["message"] == "Done" and st["files_done"] == 3 and st["missing"] == 1
        store = store_mod.get_store()
        c = store.counts()
        assert (
            c["kinds"]["image"] == 4 and c["kinds"]["video"] == 4 and c["kinds"]["audio"] == 4 + 3
        )
        assert store.file(lib["files"]["stream"])["status"] == "done"
        assert store.file(lib["files"]["gone"]) is None
        # the chunks: 4 moments each, the clip frames automatic (8 below 500 hours)
        idx = [p for path, p in service.calls if path == "/index"]
        assert all(len(p["moments"]) <= 4 and p["clip_frames"] == 8 for p in idx)
        # what is said comes along without speaker labels, and shown with them
        first = idx[-1]["moments"][0] if idx[-1]["video"] else idx[0]["moments"][0]
        assert "[SPEAKER" not in first["text"] and "SPEAKER_00" in first["shown"]

    def test_a_second_run_skips_what_is_done_and_redoes_on_new_settings(self, lib, service):
        _run()
        n = len(service.calls)
        st = _run()
        # only the missing one is still to do (it may turn up) — nothing is sent
        assert st["files_total"] == 1 and st["missing"] == 1
        assert [c for c in service.calls[n:] if c[0] == "/index"] == []
        lib["cfg"]["video_interval_sec"] = 30
        st = _run()
        assert st["files_total"] == 3
        assert store_mod.get_store().counts()["kinds"]["image"] == 3  # 70 s every 30 s

    def test_a_stopped_run_continues_where_it_stopped(self, lib, service, monkeypatch):
        ix = indexer.Indexer()
        orig = service.__call__

        def stop_after_first(path, payload=None, *, timeout=120):
            out = orig(path, payload, timeout=timeout)
            if path == "/index":
                ix._stop.set()
            return out

        monkeypatch.setattr(media_server.SERVER, "call", stop_after_first)
        st = _run(ix)
        assert st["message"] == "Stopped"
        store = store_mod.get_store()
        first = store.counts()["moments"]
        assert 0 < first < 7
        monkeypatch.setattr(media_server.SERVER, "call", service)
        _run()
        c = store.counts()
        assert c["moments"] == 7 and c["kinds"]["image"] == 4  # nothing twice

    def test_a_service_gone_after_a_stop_is_a_stop_not_a_failure(self, lib, service, monkeypatch):
        """Shutting down stops the run, then the service — the call in flight
        breaks with a closed connection."""
        import http.client

        ix = indexer.Indexer()

        def gone(path, payload=None, *, timeout=120):
            if path == "/index":
                ix._stop.set()
                raise http.client.RemoteDisconnected(
                    "Remote end closed connection without response"
                )
            return service(path, payload, timeout=timeout)

        monkeypatch.setattr(media_server.SERVER, "call", gone)
        st = _run(ix)
        assert st["message"] == "Stopped" and not st.get("error")

    def test_out_of_gpu_memory_retries_smaller(self, lib, service):
        err = media_server.MediaUnavailable("CUDA out of memory")
        err.oom = True
        service.fail = [err]
        st = _run()
        assert st["message"] == "Done"
        idx = [p for path, p in service.calls if path == "/index"]
        assert idx[0]["batch"] == 4 and idx[1]["batch"] == 2

    def test_a_file_the_service_cannot_read_is_marked_and_skipped(self, lib, service):
        err = media_server.MediaUnavailable("file not found: x")
        err.status = 404
        service.fail = [err]
        st = _run()
        assert st["message"] == "Done" and len(st["errors"]) == 1
        states = store_mod.get_store().file_states()
        assert sorted(s["status"] for s in states.values()) == ["done", "error"]

    def test_a_failing_file_doesnt_end_the_run_five_in_a_row_do(self, lib, service):
        err = media_server.MediaUnavailable("RuntimeError: broken stream")
        err.status = 500
        service.fail = [err]
        st = _run()
        assert st["message"] == "Done" and st["errors"][0]["error"].startswith("RuntimeError")
        for k in range(6):
            lib["add"](f"streams/broken{k}.mp4", 30.0)
        service.fail = [err] * 6
        st = _run()
        assert st["message"].startswith("Failed: 5 recordings in a row failed")

    def test_nothing_to_index_is_said(self, lib, service):
        lib["cfg"]["kinds"] = []
        st = _run()
        assert "Nothing to index" in st["message"]

    def test_rebuild_empties_first(self, lib, service):
        _run()
        st = _run(rebuild=True)
        assert st["files_total"] == 3
        assert store_mod.get_store().counts()["kinds"]["image"] == 4

    def test_describing_is_queued_after_a_run_when_asked(self, lib, service, monkeypatch):
        from whisper_tm.media import describe, llm

        queued = []
        lib["cfg"]["llm_describe_after_index"] = True
        monkeypatch.setattr(llm, "configured", lambda: True)
        monkeypatch.setattr(llm, "vision_available", lambda: True)
        monkeypatch.setattr(describe, "enqueue", lambda limit, folders: queued.append(limit) or 1)
        # thumbnails the fake service doesn't write: give the moments some
        orig = service.__call__

        def with_thumbs(path, payload=None, *, timeout=120):
            for m in (payload or {}).get("moments") or []:
                if m.get("thumb"):
                    import os

                    os.makedirs(os.path.dirname(m["thumb"]), exist_ok=True)
                    with open(m["thumb"], "wb") as fh:
                        fh.write(b"jpg")
            return orig(path, payload, timeout=timeout)

        monkeypatch.setattr(media_server.SERVER, "call", with_thumbs)
        _run()
        assert queued == [1000]


class TestKindsAddedLater:
    def test_a_new_kind_goes_onto_the_moments_that_are_there(self, lib, service):
        lib["cfg"]["kinds"] = ["image", "audio"]
        _run()
        store = store_mod.get_store()
        before = store.counts(fresh=True)
        n = len(service.calls)
        lib["cfg"]["kinds"] = ["image", "videoaudio", "audio", "speech"]
        st = _run()
        assert st["message"] == "Done" and st["fills"] == 2
        sent = [p for path, p in service.calls[n:] if path == "/index"]
        stream = [p for p in sent if p["video"]]
        assert stream and all(
            p["kinds"] == ["speech", "videoaudio"] and p["repeats"] == [False] * len(p["moments"])
            for p in stream
        )  # the repeats it knows, no new look
        assert all(p["kinds"] == ["speech"] and "repeats" not in p for p in sent if not p["video"])
        after = store.counts(fresh=True)
        assert after["moments"] == before["moments"]  # no moment made again
        assert (
            after["kinds"]["image"] == before["kinds"]["image"]
            and after["kinds"]["videoaudio"] == 4
        )
        assert after["kinds"]["speech"] == 4 + 3
        assert store.file(lib["files"]["stream"])["kinds"] == [
            "audio",
            "image",
            "speech",
            "videoaudio@8x10",
        ]
        # and then nothing is left to do
        n = len(service.calls)
        assert _run()["files_total"] == 1 and not [c for c in service.calls[n:] if c[0] == "/index"]

    def test_pictures_added_to_a_file_indexed_without_them(self, lib, service):
        lib["cfg"]["kinds"] = ["audio"]
        service.repeat_at = {30}
        _run()
        store = store_mod.get_store()
        row = store.file(lib["files"]["stream"])
        assert all(m["thumb"] is None and not m["repeat"] for m in store.file_moments(row["id"]))
        n = len(service.calls)
        lib["cfg"]["kinds"] = ["image", "audio"]
        _run()
        sent = [p for path, p in service.calls[n:] if path == "/index" and p["video"]]
        assert sent and all("repeats" not in p for p in sent)  # looked at: which repeat
        ms = store.file_moments(row["id"])
        assert [m["repeat"] for m in ms] == [False, True, False, False]
        assert ms[0]["thumb"] == indexer.thumb_path(lib["files"]["stream"], 10)
        assert ms[1]["thumb"] == ms[0]["thumb"]  # a repeat shows the last kept still
        c = store.counts(fresh=True)
        assert c["kinds"]["image"] == 3 and c["repeats"] == 1 and c["stills"] == 3

    def test_other_clip_frames_make_only_the_clips_again(self, lib, service):
        lib["cfg"]["clip_frames"] = 4
        _run()
        store = store_mod.get_store()
        n = len(service.calls)
        lib["cfg"]["clip_frames"] = 8
        _run()
        sent = [p for path, p in service.calls[n:] if path == "/index"]
        assert sent and all(p["kinds"] == ["video"] and p["clip_frames"] == 8 for p in sent)
        c = store.counts(fresh=True)
        assert c["kinds"]["video"] == 4 and c["kinds"]["image"] == 4

    def test_automatic_frames_are_kept_when_the_library_grows(self, lib, service, monkeypatch):
        _run()
        assert store_mod.get_store().meta("clip_frames") == 8
        monkeypatch.setattr(indexer, "AUTO_FRAMES_HOURS", 0.001)  # now far more than "500 hours"
        n = len(service.calls)
        _run()
        assert not [c for c in service.calls[n:] if c[0] == "/index"]

    def test_a_pass_under_way_with_other_kinds_starts_over(self, lib, service, monkeypatch):
        ix = indexer.Indexer()
        orig = service.__call__

        def stop_after_first(path, payload=None, *, timeout=120):
            out = orig(path, payload, timeout=timeout)
            if path == "/index":
                ix._stop.set()
            return out

        monkeypatch.setattr(media_server.SERVER, "call", stop_after_first)
        _run(ix)
        monkeypatch.setattr(media_server.SERVER, "call", service)
        lib["cfg"]["kinds"] = ["image", "video", "audio", "speech"]
        _run()
        store = store_mod.get_store()
        c = store.counts(fresh=True)
        assert c["moments"] == 7 and c["kinds"]["speech"] == 7  # nothing twice, nothing missing


class TestTheDisk:
    def test_the_run_waits_for_room_and_goes_on(self, lib, service, monkeypatch):
        free = [5.0, 5.0, 50.0]
        monkeypatch.setattr(
            indexer, "disk_free_gb", lambda: free.pop(0) if len(free) > 1 else free[0]
        )
        ix = indexer.Indexer()
        seen = []
        orig = ix._set

        def spy(**kw):
            if kw.get("waiting") == "disk":
                seen.append(kw["message"])
            return orig(**kw)

        ix._set = spy
        st = _run(ix)
        assert st["message"] == "Done" and seen and "5.0 GB free" in seen[0] and "20 GB" in seen[0]

    def test_a_stop_ends_the_wait(self, lib, service, monkeypatch):
        monkeypatch.setattr(indexer, "disk_free_gb", lambda: 1.0)
        ix = indexer.Indexer()
        assert ix.start()
        for _ in range(200):
            if ix.status().get("waiting") == "disk":
                break
            time.sleep(0.01)
        assert ix.status()["waiting"] == "disk"
        ix.stop()
        for _ in range(200):
            if not ix.running():
                break
            time.sleep(0.01)
        assert ix.status()["message"] == "Stopped" and not [
            c for c in service.calls if c[0] == "/index"
        ]

    def test_thumbnails_stop_at_their_limit(self, lib, service):
        lib["cfg"].update(thumbs_max_mb=1, chunk_moments=2)
        service.thumb_size = 600 * 1024
        _run()
        sent = [p for path, p in service.calls if path == "/index" and p["video"]]
        assert [p["thumbs"] for p in sent] == [True, False]  # 2 stills of 0.6 MB, then full
        assert store_mod.get_store().meta("thumb_bytes") == 2 * 600 * 1024


class TestTheGpu:
    def test_a_turn_holds_the_gate_and_gives_it_back_when_someone_waits(
        self, lib, service, monkeypatch
    ):
        from whisper_tm.core import gpu_gate

        gate = GpuGate(free_mb=lambda: 100_000)
        monkeypatch.setattr(gpu_gate, "GATE", gate)
        lib["cfg"]["index_device"] = "cuda"
        seen = []
        orig = service.__call__

        def watch(path, payload=None, *, timeout=120):
            if path == "/index":
                seen.append(dict(gate.state()["stages"]))
            return orig(path, payload, timeout=timeout)

        monkeypatch.setattr(media_server.SERVER, "call", watch)
        monkeypatch.setattr(gate, "waiting", lambda: 1)  # the queue always waits
        st = _run()
        assert st["message"] == "Done"
        assert all("media" in s for s in seen)  # every chunk inside the stage
        releases = [p for path, p in service.calls if path == "/release"]
        assert len(releases) >= len(seen) - 1  # given back after each turn
        assert gate.state()["stages"] == {}  # nothing left held

    def test_whisper_waiting_gets_the_card(self, lib, service, monkeypatch):
        """A stage held by the index run blocks an exclusive (Whisper); the
        run's yield lets it in after the chunk in hand."""
        from whisper_tm.core import gpu_gate

        gate = GpuGate(free_mb=lambda: 100_000)
        monkeypatch.setattr(gpu_gate, "GATE", gate)
        lib["cfg"]["index_device"] = "cuda"
        got = threading.Event()
        orig = service.__call__

        def first_chunk_then_whisper(path, payload=None, *, timeout=120):
            out = orig(path, payload, timeout=timeout)
            if path == "/index" and not hasattr(first_chunk_then_whisper, "started"):
                first_chunk_then_whisper.started = True

                def whisper():
                    with gate.exclusive("asr"):
                        got.set()

                threading.Thread(target=whisper, daemon=True).start()
                time.sleep(0.05)
            return out

        monkeypatch.setattr(media_server.SERVER, "call", first_chunk_then_whisper)
        st = _run()
        assert st["message"] == "Done" and got.wait(2)

    def test_queue_first_waits_while_a_transcription_runs(self, lib, service, monkeypatch):
        from whisper_tm.core import gpu_gate
        from whisper_tm.core import queue as q

        monkeypatch.setattr(gpu_gate, "GATE", GpuGate(free_mb=lambda: 100_000))
        lib["cfg"].update(index_device="cuda", index_while_queue_runs=False)
        busy = [3]

        def is_busy():
            busy[0] -= 1
            return busy[0] > 0

        monkeypatch.setattr(q, "busy", is_busy)
        monkeypatch.setattr(q, "is_paused", lambda: False)
        st = _run()
        assert st["message"] == "Done" and busy[0] <= 0


class TestSeveralAtOnce:
    """``parallel_files``: recordings side by side, each its chunks in order."""

    def test_two_recordings_overlap_and_the_result_is_the_same(self, lib, service, monkeypatch):
        lib["cfg"]["parallel_files"] = 2
        inside, most = [0], [0]
        lock = threading.Lock()
        orig = service.__call__

        def slow(path, payload=None, *, timeout=120):
            if path == "/index":
                with lock:
                    inside[0] += 1
                    most[0] = max(most[0], inside[0])
                time.sleep(0.03)
                try:
                    return orig(path, payload, timeout=timeout)
                finally:
                    with lock:
                        inside[0] -= 1
            return orig(path, payload, timeout=timeout)

        monkeypatch.setattr(media_server.SERVER, "call", slow)
        st = _run()
        assert st["message"] == "Done" and st["files_done"] == 3 and st["missing"] == 1
        assert most[0] == 2  # two requests at the service
        c = store_mod.get_store().counts()
        assert c["moments"] == 7 and c["kinds"]["image"] == 4 and c["kinds"]["audio"] == 7
        assert st["moments_done"] == 7
        # a file's chunks stay in order (its repeats follow its last still)
        stream = [
            p["moments"][0]["t"]
            for path, p in service.calls
            if path == "/index" and p["path"] == lib["files"]["stream"]
        ]
        assert stream == sorted(stream)

    def test_the_turn_ends_only_when_no_request_is_at_the_service(self, lib, service, monkeypatch):
        """The model is released between requests, never under one."""
        from whisper_tm.core import gpu_gate

        gate = GpuGate(free_mb=lambda: 100_000)
        monkeypatch.setattr(gpu_gate, "GATE", gate)
        monkeypatch.setattr(gate, "waiting", lambda: 1)  # the queue always waits
        lib["cfg"].update(index_device="cuda", parallel_files=2)
        for k in range(3):
            lib["add"](f"streams/more{k}.mp4", 50.0)
        inside, bad = [0], []
        lock = threading.Lock()
        orig = service.__call__

        def watch(path, payload=None, *, timeout=120):
            if path == "/release":
                with lock:
                    if inside[0]:
                        bad.append(inside[0])
                return orig(path, payload, timeout=timeout)
            with lock:
                inside[0] += 1
            time.sleep(0.02)
            try:
                return orig(path, payload, timeout=timeout)
            finally:
                with lock:
                    inside[0] -= 1

        monkeypatch.setattr(media_server.SERVER, "call", watch)
        st = _run()
        assert st["message"] == "Done" and st["files_done"] == 6
        assert bad == []
        assert any(path == "/release" for path, _ in service.calls)
        assert gate.state()["stages"] == {}

    def test_a_failure_in_one_stops_the_other(self, lib, service, monkeypatch):
        lib["cfg"]["parallel_files"] = 2
        for k in range(6):
            lib["add"](f"streams/broken{k}.mp4", 30.0)
        err = media_server.MediaUnavailable("RuntimeError: broken stream")
        err.status = 500
        service.fail = [err] * 9
        st = _run()
        assert st["message"].startswith("Failed: 5 recordings in a row failed")
        assert not st["running"]

    def test_a_stop_stops_both(self, lib, service, monkeypatch):
        lib["cfg"]["parallel_files"] = 2
        ix = indexer.Indexer()
        orig = service.__call__

        def stop_after_first(path, payload=None, *, timeout=120):
            out = orig(path, payload, timeout=timeout)
            if path == "/index":
                ix._stop.set()
            return out

        monkeypatch.setattr(media_server.SERVER, "call", stop_after_first)
        st = _run(ix)
        assert st["message"] == "Stopped" and not st.get("error")
        assert len([c for c in service.calls if c[0] == "/index"]) <= 2


def test_the_config_check_names_values_the_indexer_cannot_use(monkeypatch):
    from whisper_tm.config import config

    orig = config.get

    def get(*keys, default=None):
        if keys == ("media_search",):
            return {"kinds": ["image", "smell"], "dims": 300, "index_device": "tpu"}
        return orig(*keys, default=default)

    monkeypatch.setattr(config, "get", get)
    warnings = config.validate()
    assert any("unknown kind(s) smell" in w for w in warnings)
    assert any("media_search.dims" in w for w in warnings)
    assert any("media_search.index_device" in w for w in warnings)
