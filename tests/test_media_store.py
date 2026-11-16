"""The media index (media/store.py): files, moments, vectors, ranking.

The vectors are stored as 8-bit numbers (all 768 of the
model — less disk than 512 as float16, and better for sounds), RAM keeps
them as they are in blocks (packed for fbgemm), and a search scores its
best again with the stored vectors. Counts move with every write instead
of counting millions of rows on every look of the settings page.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest

from whisper_tm.media import store as store_mod
from whisper_tm.media.store import KINDS, MediaStore, decode, encode, unit


def _v(*xs) -> np.ndarray:
    return np.asarray(xs, dtype=np.float32)


@pytest.fixture
def store(tmp_path):
    s = MediaStore(tmp_path / "media.db")
    s.ensure_dims(4)
    yield s
    s.close()


def _file(
    store, path="/lib/a.mp4", settings="fp1", video=True, duration=100.0, kinds=None, fresh=False
):
    return store.start_file(
        path, video=video, duration=duration, settings=settings, kinds=kinds, fresh=fresh
    )


def _moments(store, fid, vectors: list[dict], done=True, t0=0.0, thumbs=None):
    rows = [
        {
            "t": t0 + 10 * k,
            "start": t0 + 10 * k,
            "dur": 10,
            "repeat": False,
            "thumb": (thumbs[k] if thumbs else None),
            "text": f"said {k}",
            "vectors": vec,
        }
        for k, vec in enumerate(vectors)
    ]
    return store.add_moments(
        fid, rows, next_t=t0 + 10 * len(rows), prev=None, done=done, seconds=1.0
    )


class TestTheFormat:
    def test_8_bit_numbers_keep_the_direction(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            v = rng.normal(size=768).astype(np.float32)
            w = decode(encode(v, 768))
            assert len(encode(v, 768)) == 768  # one byte a number
            assert float(unit(v, 768) @ w) > 0.999

    def test_vectors_are_cut_to_the_index_size(self, store):
        f = _file(store)
        assert _moments(store, f["id"], [{"image": _v(3, 4, 0, 0, 9, 9)}]) == 1  # longer: cut
        it = store.item(store.rank(_v(1, 0, 0, 0), "image", 1)[0][0])
        assert it["vector"].shape == (4,) and np.allclose(it["vector"], [0.6, 0.8, 0, 0], atol=1e-2)
        assert store.format() == "int8"

    def test_the_index_size_is_fixed_by_the_first_write(self, tmp_path):
        s = MediaStore(tmp_path / "m.db")
        assert s.dims() is None
        assert s.ensure_dims(512) == 512 and s.format() == "int8"
        assert s.ensure_dims(256) == 512  # later calls keep it
        s.clear()
        assert s.dims() is None  # an empty index takes a new size


class TestFiles:
    def test_a_file_indexed_with_other_settings_starts_over(self, store):
        f = _file(store, settings="fp1", kinds=["image"])
        _moments(store, f["id"], [{"image": _v(1, 0, 0, 0)}, {"image": _v(0, 1, 0, 0)}])
        assert store.counts()["moments"] == 2
        g = _file(store, settings="fp1")  # same settings: kept
        assert g["id"] == f["id"] and g["status"] == "done" and g["kinds"] == ["image"]
        h = _file(store, settings="fp2")  # other settings: its moments go
        assert h["status"] == "partial" and h["moments"] == 0
        assert store.counts()["moments"] == 0 and store.counts()["kinds"]["image"] == 0
        assert store.rank(_v(1, 0, 0, 0), "image", 5) == []

    def test_fresh_starts_over_with_the_same_settings(self, store):
        f = _file(store, kinds=["audio"])
        _moments(store, f["id"], [{"audio": _v(1, 0, 0, 0)}], done=False)
        g = _file(store, kinds=["audio", "speech"], fresh=True)
        assert g["kinds"] == ["audio", "speech"] and g["moments"] == 0 and g["status"] == "partial"
        assert store.counts()["kinds"]["audio"] == 0

    def test_a_partial_file_keeps_where_it_stopped(self, store):
        f = _file(store, kinds=["audio"])
        store.add_moments(
            f["id"],
            [{"t": 5, "start": 0, "dur": 10, "vectors": {"audio": _v(1, 0, 0, 0)}}],
            next_t=15.0,
            prev="abc",
            done=False,
            seconds=2.0,
        )
        row = store.file("/lib/a.mp4")
        assert row["status"] == "partial" and row["next_t"] == 15.0 and row["prev"] == "abc"
        assert store.file_states() == {
            "/lib/a.mp4": {
                "status": "partial",
                "settings": "fp1",
                "kinds": ["audio"],
                "next_t": 15.0,
                "duration": 100.0,
            }
        }

    def test_kinds_added_later_go_onto_the_moments(self, store):
        f = _file(store, kinds=["image"])
        _moments(store, f["id"], [{"image": _v(1, 0, 0, 0)}, {"image": _v(0, 1, 0, 0)}])
        ms = store.file_moments(f["id"])
        assert [m["t"] for m in ms] == [0, 10]
        assert (
            store.add_items(
                [(ms[0]["id"], "speech", _v(0, 0, 1, 0)), (ms[1]["id"], "speech", _v(0, 0, 0, 1))]
            )
            == 2
        )
        store.set_file_kinds(f["id"], ["image", "speech"])
        store.finish_file(f["id"], 3.0)
        assert store.file("/lib/a.mp4")["kinds"] == ["image", "speech"]
        assert store.counts()["kinds"]["speech"] == 2
        assert store.rank(_v(0, 0, 1, 0), "speech", 1)[0][1] == ms[0]["id"]

    def test_a_kind_made_again_goes_first(self, store):
        f = _file(store)
        _moments(store, f["id"], [{"image": _v(1, 0, 0, 0), "video": _v(0, 1, 0, 0)}])
        assert store.delete_kinds(f["id"], ["video"]) == 1
        c = store.counts()
        assert c["kinds"]["video"] == 0 and c["kinds"]["image"] == 1
        assert store.rank(_v(0, 1, 0, 0), "video", 5) == []  # gone from the search at once

    def test_a_later_look_at_the_pictures(self, store, tmp_path):
        f = _file(store, kinds=["audio"])
        _moments(store, f["id"], [{"audio": _v(1, 0, 0, 0)}, {"audio": _v(0, 1, 0, 0)}])
        assert store.counts()["stills"] == 0
        ms = store.file_moments(f["id"])
        store.update_moments(
            f["id"], [(ms[0]["id"], False, "/t/0.jpg"), (ms[1]["id"], True, "/t/0.jpg")]
        )
        c = store.counts()
        assert c["stills"] == 1 and c["repeats"] == 1
        assert c == store.counts(fresh=True) | {"files": c["files"]}  # moved = counted
        assert store.file("/lib/a.mp4")["repeats"] == 1

    def test_remove_and_clear(self, store):
        f = _file(store)
        _moments(store, f["id"], [{"image": _v(1, 0, 0, 0)}])
        g = _file(store, path="/lib/b.mp4")
        _moments(store, g["id"], [{"image": _v(0, 1, 0, 0)}])
        assert store.remove_files(["/lib/a.mp4"]) == 1
        assert [r[1] for r in store.rank(_v(1, 0, 0, 0), "image", 5)] != []
        assert store.counts()["kinds"]["image"] == 1
        store.set_meta("thumb_bytes", 5)
        store.clear()
        assert store.counts()["moments"] == 0 and store.file_states() == {}
        assert store.meta("thumb_bytes") is None


class TestCounting:
    def test_counts(self, store, tmp_path):
        f = _file(store)
        _moments(
            store,
            f["id"],
            [{"image": _v(1, 0, 0, 0), "audio": _v(0, 1, 0, 0)}, {"audio": _v(0, 0, 1, 0)}],
            thumbs=["/t/1.jpg", None],
        )
        c = store.counts()
        assert c["kinds"]["image"] == 1 and c["kinds"]["audio"] == 2
        assert set(c["kinds"]) == set(KINDS)
        assert (
            c["files"] == {"done": 1} and c["moments"] == 2 and c["dims"] == 4 and c["stills"] == 1
        )

    def test_counts_move_with_every_write_without_counting_again(self, store, monkeypatch):
        f = _file(store)
        _moments(store, f["id"], [{"image": _v(1, 0, 0, 0)}])
        store.counts()
        calls = []
        real = store._count_all
        monkeypatch.setattr(store, "_count_all", lambda: calls.append(1) or real())
        g = _file(store, path="/lib/b.mp4")
        _moments(store, g["id"], [{"image": _v(0, 1, 0, 0), "speech": _v(0, 0, 1, 0)}] * 3)
        store.delete_kinds(g["id"], ["speech"])
        c = store.counts()
        assert calls == []  # moved, not counted
        assert c["kinds"]["image"] == 4 and c["kinds"]["speech"] == 0 and c["moments"] == 4
        fresh = store.counts(fresh=True)
        assert fresh["kinds"] == c["kinds"] and fresh["moments"] == c["moments"]


class TestRanking:
    def test_best_first_with_exact_scores(self, store):
        f = _file(store)
        _moments(
            store,
            f["id"],
            [{"image": _v(1, 0, 0, 0)}, {"image": _v(0.6, 0.8, 0, 0)}, {"image": _v(0, 1, 0, 0)}],
        )
        hits = store.rank(_v(1, 0, 0, 0), "image", 2)
        assert [round(h[2], 2) for h in hits] == [1.0, 0.6]
        assert all(len(h) == 4 and h[3].shape == (4,) for h in hits)  # item, moment, score, vector

    def test_full_blocks_are_packed_8_bit_and_rank_like_all_numbers(self, tmp_path, monkeypatch):
        monkeypatch.setattr(store_mod, "BLOCK", 4)
        s = MediaStore(tmp_path / "p.db")
        s.ensure_dims(16)
        f = _file(s)
        rng = np.random.default_rng(3)
        V = rng.normal(size=(9, 16)).astype(np.float32)
        s.matrices()
        _moments(s, f["id"], [{"image": v} for v in V])
        m = s.matrices()["image"]
        assert [b["n"] for b in m.blocks] == [4, 4] and m.tail["n"] == 1
        assert all("packed" in b for b in m.blocks) == bool(store_mod._FBGEMM["ok"])
        q = rng.normal(size=16).astype(np.float32)
        exact = np.argsort(-(unit(V, 16) @ unit(q, 16)))[:5]
        ids = [
            r[0] for r in sorted(s.connect().__enter__().execute("SELECT id, moment_id FROM items"))
        ]
        assert [h[0] for h in s.rank(q, "image", 5)] == [ids[i] for i in exact]
        # read again from the database: full blocks straight from the stored bytes
        s2 = MediaStore(tmp_path / "p.db")
        m2 = s2.matrices()["image"]
        assert [b["n"] for b in m2.blocks] == [4, 4] and m2.tail["n"] == 1
        assert [h[0] for h in s2.rank(q, "image", 5)] == [ids[i] for i in exact]
        assert s2.ram_bytes() >= 9 * (16 + 24)

    def test_without_fbgemm_the_blocks_stay_float16(self, tmp_path, monkeypatch):
        monkeypatch.setattr(store_mod, "BLOCK", 2)
        monkeypatch.setitem(store_mod._FBGEMM, "ok", False)
        s = MediaStore(tmp_path / "f.db")
        s.ensure_dims(4)
        f = _file(s)
        s.matrices()
        _moments(
            s,
            f["id"],
            [{"audio": _v(1, 0, 0, 0)}, {"audio": _v(0, 1, 0, 0)}, {"audio": _v(0, 0, 1, 0)}],
        )
        m = s.matrices()["audio"]
        assert "X16" in m.blocks[0] and "packed" not in m.blocks[0]
        assert s.rank(_v(0, 1, 0, 0), "audio", 1)[0][2] == pytest.approx(1.0, abs=1e-2)

    def test_exclude_and_file_filter(self, store):
        a = _file(store, path="/lib/x/a.mp4")
        b = _file(store, path="/lib/y/b.mp4")
        _moments(store, a["id"], [{"image": _v(1, 0, 0, 0)}])
        _moments(store, b["id"], [{"image": _v(0.9, 0.1, 0, 0)}])
        first = store.rank(_v(1, 0, 0, 0), "image", 5)
        assert len(first) == 2
        without = store.rank(_v(1, 0, 0, 0), "image", 5, exclude=first[0][0])
        assert [h[0] for h in without] == [first[1][0]]
        only_y = store.rank(_v(1, 0, 0, 0), "image", 5, files=store.file_ids_under(["/lib/y/"]))
        assert len(only_y) == 1 and only_y[0][0] == first[1][0]
        assert (
            store.rank(_v(1, 0, 0, 0), "image", 5, files=store.file_ids_under(["/elsewhere/"]))
            == []
        )

    def test_fewer_numbers_matryoshka(self, store):
        f = _file(store)
        # Equal in the first two numbers, apart in the last two
        _moments(store, f["id"], [{"image": _v(1, 1, 1, 0)}, {"image": _v(1, 1, 0, 1)}])
        full = store.rank(_v(1, 1, 1, 0), "image", 2)
        assert full[0][2] > full[1][2]
        cut = store.rank(_v(1, 1, 1, 0), "image", 2, dims=2)
        assert cut[0][2] == pytest.approx(cut[1][2], abs=1e-2) == pytest.approx(1.0, abs=1e-2)

    def test_new_rows_join_a_loaded_matrix_in_place(self, store):
        f = _file(store)
        _moments(store, f["id"], [{"image": _v(0, 1, 0, 0)}])
        m = store.matrices()["image"]
        assert len(m) == 1
        gen = store.generation
        g = _file(store, path="/lib/b.mp4")
        _moments(store, g["id"], [{"image": _v(1, 0, 0, 0)}])
        assert store.generation > gen and store.matrices()["image"] is m and len(m) == 2
        assert store.rank(_v(1, 0, 0, 0), "image", 1)[0][2] == pytest.approx(1.0, abs=1e-2)

    def test_blocks_fill_and_grow_without_copying_the_rest(self, store, monkeypatch):
        monkeypatch.setattr(store_mod, "BLOCK", 4)
        f = _file(store)
        store.matrices()
        for k in range(3):
            _moments(store, f["id"], [{"audio": _v(1, k, 0, 0)}] * 3, t0=100 * k)
        m = store.matrices()["audio"]
        assert [b["n"] for b in m.blocks] == [4, 4] and m.tail["n"] == 1 and len(m) == 9
        best = store.rank(_v(1, 2, 0, 0), "audio", 3)
        assert len(best) == 3 and all(h[2] == pytest.approx(1.0, abs=1e-2) for h in best)

    def test_removed_rows_are_not_found_and_many_read_again(self, store):
        f = _file(store)
        _moments(store, f["id"], [{"video": _v(1, 0, 0, 0)}, {"video": _v(0, 1, 0, 0)}])
        m = store.matrices()["video"]
        store.delete_kinds(f["id"], ["video"])
        assert store.rank(_v(1, 0, 0, 0), "video", 5) == []  # the second pass doesn't find them
        assert store.matrices()["video"] is not m  # more than a tenth gone: read again
        assert len(store.matrices()["video"]) == 0

    def test_rows_written_while_reading_are_not_lost(self, store, monkeypatch):
        f = _file(store)
        _moments(store, f["id"], [{"image": _v(1, 0, 0, 0)}])
        real = store._load
        started, go = threading.Event(), threading.Event()

        def slow():
            out = real()
            started.set()
            go.wait(5)
            return out

        monkeypatch.setattr(store, "_load", slow)
        t = threading.Thread(target=store.matrices)
        t.start()
        started.wait(5)
        g = _file(store, path="/lib/b.mp4")
        _moments(store, g["id"], [{"image": _v(0, 1, 0, 0)}])  # while it is read
        go.set()
        t.join(5)
        assert len(store.matrices()["image"]) == 2
        assert store.rank(_v(0, 1, 0, 0), "image", 1)[0][2] == pytest.approx(1.0, abs=1e-2)

    def test_an_empty_kind_finds_nothing(self, store):
        assert store.rank(_v(1, 0, 0, 0), "videoaudio", 5) == []

    def test_ram_holds_every_number_as_8_bit(self, tmp_path):
        s = MediaStore(tmp_path / "r.db")
        s.ensure_dims(768)
        f = s.start_file("/lib/a.mp4", video=True, duration=10, settings="x")
        rng = np.random.default_rng(1)
        s.add_moments(
            f["id"],
            [
                {"t": k, "start": k, "dur": 1, "vectors": {"image": rng.normal(size=768)}}
                for k in range(10)
            ],
            next_t=10,
            prev=None,
            done=True,
            seconds=0,
        )
        assert s.ram_bytes() == 0
        m = s.matrices()["image"]
        assert m.dims == 768 and len(m) == 10 and s.ram_bytes() > 0

    def test_fewer_numbers_go_deeper_before_the_exact_order(self, store, monkeypatch):
        monkeypatch.setattr(store_mod, "SHORTLIST", 1)
        monkeypatch.setattr(store_mod, "SHORTLIST_FEWER", 3)
        f = _file(store)
        _moments(
            store,
            f["id"],
            [{"image": _v(1, 1, 1, 0)}, {"image": _v(1, 1, 0, 1)}, {"image": _v(1, 1, -1, 0)}],
        )
        cut = store.rank(_v(1, 1, 1, 0), "image", 3, dims=2)
        assert len(cut) == 3 and all(h[2] == pytest.approx(1.0, abs=1e-2) for h in cut)


class TestDescriptions:
    def test_described_moments_and_their_vectors(self, store, tmp_path):
        thumb = tmp_path / "t.jpg"
        thumb.write_bytes(b"jpg")
        f = _file(store)
        rows = [
            {
                "t": 1,
                "start": 0,
                "dur": 10,
                "repeat": False,
                "thumb": str(thumb),
                "vectors": {"image": _v(1, 0, 0, 0)},
            },
            {"t": 11, "start": 10, "dur": 10, "repeat": True, "thumb": str(thumb), "vectors": {}},
            {"t": 21, "start": 20, "dur": 10, "repeat": False, "thumb": None, "vectors": {}},
        ]
        store.add_moments(f["id"], rows, next_t=30, prev=None, done=True, seconds=0)
        todo = store.undescribed(10)
        assert (
            len(todo) == 1 and todo[0]["text"] is None
        )  # repeats and moments without a still don't count
        assert todo[0]["t"] == 1 and todo[0]["path"] == f["path"]  # to cut a still never written
        mid = todo[0]["id"]
        store.set_description(mid, "A police car at night.", "m")
        assert store.describe(mid)["text"] == "A police car at night."
        # Written but not embedded: still to do, with its text
        again = store.undescribed(10)
        assert again[0]["text"] == "A police car at night."
        assert store.add_items([(mid, "description", _v(0, 0, 1, 0))]) == 1
        assert store.undescribed(10) == []
        assert store.description_counts() == {
            "described": 1,
            "stills": 1,
        }  # repeats and no-picture moments aside
        assert store.rank(_v(0, 0, 1, 0), "description", 1)[0][1] == mid
        # Again for the same moment: replaced, not doubled
        store.add_items([(mid, "description", _v(0, 0, 0, 1))])
        assert store.counts()["kinds"]["description"] == 1
        assert store.counts(fresh=True)["kinds"]["description"] == 1

    def test_only_these_folders_and_the_newest_first(self, store, tmp_path):
        thumb = tmp_path / "t.jpg"
        thumb.write_bytes(b"jpg")
        for path in ("/lib/x_1/a.mp4", "/lib/x%1/b.mp4", "/lib/y/c.mp4"):
            f = _file(store, path=path)
            store.add_moments(
                f["id"],
                [{"t": 1, "start": 0, "dur": 10, "thumb": str(thumb), "vectors": {}}],
                next_t=10,
                prev=None,
                done=True,
                seconds=0,
            )
        assert [m["path"] for m in store.undescribed(10, ["/lib/x_1/"])] == [
            "/lib/x_1/a.mp4"
        ]  # _ is no wildcard
        assert [m["path"] for m in store.undescribed(10)] == [
            "/lib/y/c.mp4",
            "/lib/x%1/b.mp4",
            "/lib/x_1/a.mp4",
        ]
        assert len(store.undescribed(1)) == 1

    def test_unit_helper(self):
        u = unit(_v(3, 4, 12), 2)
        assert np.allclose(u, [0.6, 0.8])
