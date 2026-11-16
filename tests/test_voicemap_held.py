"""The map is held by what it depends on (core/speaker_graph.py).

Holding the map costs RAM, but saves the long waits. The guided assignment
reloads the map after every "Apply". An Apply changes names, and the map
used to recompute everything for it — neighbourhood, grouping, edges, tens
of MB of JSON: close to a minute per click on a library of several hundred
thousand voices. Now a name change costs a fraction of a second.

Checked here: that the held parts are only reused when nothing they depend
on moved, and that the answer is byte for byte what the endpoint sent
before — against a copy of the old code.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from flask import Flask

from whisper_tm.core import speaker as speaker_core
from whisper_tm.core import speaker_graph, speaker_knn
from whisper_tm.core import speaker_inventory as inventory
from whisper_tm.database.speaker import SpeakerDB

DIM = 192


def _vec(seed):
    return np.random.default_rng(seed).normal(size=DIM).astype(np.float32)


@pytest.fixture
def app(tmp_path, monkeypatch):
    from whisper_tm.config import config

    media = tmp_path / "media"
    media.mkdir()
    monkeypatch.setattr(type(config), "media_dir", property(lambda self: media))
    db = SpeakerDB(tmp_path / "speaker.db")
    db.initialize()
    monkeypatch.setattr("whisper_tm.database.speaker.speaker_db", lambda: db)
    monkeypatch.setattr("whisper_tm.api.speaker.speaker_db", lambda: db)
    inventory.discard()
    speaker_knn.clear_cache()
    speaker_graph.clear_cache()

    from whisper_tm.api.speaker import speaker_bp

    a = Flask(__name__)
    a.register_blueprint(speaker_bp)
    a.db = db
    a.media = media
    yield a
    inventory.discard()
    speaker_knn.clear_cache()
    speaker_graph.clear_cache()


def _stock(app, files=6, each=7, seed=0):
    """Speakers that recur across files, so groups span recordings."""
    rng = np.random.default_rng(seed)
    people = [_vec(100 + p) for p in range(4)]
    for d in range(files):
        path = str(app.media / f"folder{d % 2}" / f"rec{d}.mkv")
        rows = []
        for i in range(each):
            base = people[(d + i) % len(people)]
            rows.append(
                {
                    "label": f"SPEAKER_0{i % 3}",
                    "windows": i,
                    "start_sec": i * 31.5,
                    "speech_sec": 12.0 + i,
                    "segments": 2,
                    "v_separated": base + rng.uniform(0.2, 0.9) * _vec(1000 + d * 50 + i),
                }
            )
        app.db.write(path, speaker_core.MODEL_BRAND, rows)


class _Count:
    """Counts calls into the expensive steps: a neighbourhood computed from
    scratch (``knn``), one updated to changed vectors (``updates``), a
    grouping (``group``)."""

    def __init__(self, monkeypatch):
        from whisper_tm.core import speaker_neighbours as sn

        self.knn = 0
        self.updates = 0
        self.group = 0
        real_compute, real_update = sn.compute, sn.update
        real_group = speaker_knn._group_voices

        def compute(*a, **k):
            self.knn += 1
            return real_compute(*a, **k)

        def update(*a, **k):
            self.updates += 1
            return real_update(*a, **k)

        def group(*a, **k):
            self.group += 1
            return real_group(*a, **k)

        monkeypatch.setattr(sn, "compute", compute)
        monkeypatch.setattr(sn, "update", update)
        monkeypatch.setattr(speaker_knn, "_group_voices", group)


# ── What is held, and when it is not ─────────────────────────────────────


class TestHeld:
    def test_a_name_change_recomputes_nothing_but_the_names(self, app, monkeypatch):
        _stock(app)
        count = _Count(monkeypatch)
        c = app.test_client()
        before = c.get("/api/voicemap/graph?threshold=0.5").get_json()
        assert (count.knn, count.group) == (1, 1)
        first = before["nodes"]["id"][0]
        app.db.assign(first, "Alice")
        after = c.get("/api/voicemap/graph?threshold=0.5").get_json()
        assert (count.knn, count.group) == (1, 1), "recomputed for a name"
        assert after["nodes"]["name"][0] == "Alice"
        nr = after["nodes"]["group_nr"][0]
        legend = {g["nr"]: g for g in after["groups"]}
        assert legend[nr]["name"] == "Alice"
        # Everything else is unchanged
        for key in ("edges", "file_names", "threshold", "voices", "group_count"):
            assert after[key] == before[key], key
        for key in ("id", "group_nr", "label", "start", "windows", "speech_sec", "file_idx"):
            assert after["nodes"][key] == before["nodes"][key], key

    def test_new_vectors_are_recomputed(self, app, monkeypatch):
        _stock(app)
        count = _Count(monkeypatch)
        c = app.test_client()
        n = c.get("/api/voicemap/graph").get_json()["voices"]
        app.db.write(
            str(app.media / "new.mkv"),
            speaker_core.MODEL_BRAND,
            [
                {
                    "label": "SPEAKER_00",
                    "windows": 0,
                    "start_sec": 0.0,
                    "speech_sec": 30.0,
                    "segments": 1,
                    "v_separated": _vec(9),
                }
            ],
        )
        d = c.get("/api/voicemap/graph").get_json()
        assert d["voices"] == n + 1
        # The neighbourhood followed the new voice instead of starting over
        assert (count.knn, count.updates, count.group) == (1, 1, 2)
        assert "new.mkv" in d["file_names"]

    def test_the_trash_is_recomputed(self, app, monkeypatch):
        _stock(app)
        count = _Count(monkeypatch)
        c = app.test_client()
        d = c.get("/api/voicemap/graph").get_json()
        app.db.move_to_trash([d["nodes"]["id"][0]])
        after = c.get("/api/voicemap/graph").get_json()
        assert after["voices"] == d["voices"] - 1
        # Updated, or — the small set is dense — computed afresh
        assert count.knn + count.updates >= 2
        assert c.get("/api/voicemap/graph?trash=1").get_json()["voices"] == 1

    def test_the_slider_reuses_the_neighbourhood(self, app, monkeypatch):
        _stock(app)
        count = _Count(monkeypatch)
        c = app.test_client()
        for t in (0.3, 0.5, 0.7, 0.5):
            c.get(f"/api/voicemap/graph?threshold={t}")
        assert count.knn == 1
        assert count.group == 3  # 0.5 twice: the second is held

    def test_a_moved_media_root_is_a_new_view(self, app, monkeypatch):
        _stock(app)
        c = app.test_client()
        d = c.get("/api/voicemap/graph").get_json()
        assert all(not f.startswith("/") for f in d["file_names"])
        from whisper_tm.config import config

        elsewhere = app.media / "folder0"
        monkeypatch.setattr(type(config), "media_dir", property(lambda self: elsewhere))
        d2 = c.get("/api/voicemap/graph").get_json()
        assert d2["root"] == str(elsewhere)
        assert any(f.startswith("/") for f in d2["file_names"])

    def test_held_arrays_cannot_be_changed_by_a_caller(self, app):
        _stock(app)
        X = inventory.inventory()["X"]
        idx, value = speaker_knn.held_neighbours(X, ("m", 1), k=5, device="cpu")
        with pytest.raises(ValueError):
            idx[0, 0] = 1
        with pytest.raises(ValueError):
            value[0, 0] = 1.0
        again = speaker_knn.held_neighbours(X, ("m", 1), k=5, device="cpu")
        assert again[0] is idx


class TestInventory:
    def test_a_name_keeps_the_matrix(self, app):
        _stock(app, files=2)
        a = inventory.inventory()
        app.db.assign(a["meta"][0]["id"], "Alice")
        b = inventory.inventory()
        assert b is not a and b["X"] is a["X"] and b["meta"] is a["meta"]
        assert "Alice" in b["gallery"] and "Alice" not in a["gallery"]
        assert b["generation"] > a["generation"]
        assert b["inventory_generation"] == a["inventory_generation"]

    @staticmethod
    def _same_as_a_full_read(state):
        fresh = inventory.inventory(force=True)
        assert np.array_equal(state["X"], fresh["X"])
        assert state["X"].dtype == fresh["X"].dtype
        assert state["meta"] == fresh["meta"]
        assert state["id_to_row"] == fresh["id_to_row"]
        assert np.array_equal(state["ids"], fresh["ids"])
        assert state["gallery"].keys() == fresh["gallery"].keys()

    def test_a_new_recording_is_followed_not_reread(self, app):
        _stock(app, files=4)
        inventory.inventory()
        app.db.write(
            str(app.media / "x.mkv"),
            speaker_core.MODEL_BRAND,
            [
                {
                    "label": "SPEAKER_01",
                    "windows": i,
                    "start_sec": 3.0 * i,
                    "speech_sec": 30.0,
                    "segments": 1,
                    "v_separated": _vec(50 + i),
                }
                for i in range(3)
            ],
        )
        # An instance attribute over the method — `monkeypatch.undo()` would
        # also undo the fixture's patches
        app.db.all_vectors = lambda *a, **k: pytest.fail("read everything again")
        try:
            state = inventory.inventory()
        finally:
            del app.db.all_vectors
        assert len(state["X"]) == 4 * 7 + 3
        self._same_as_a_full_read(state)

    def test_a_deleted_recording_is_followed(self, app):
        _stock(app, files=4)
        inventory.inventory()
        app.db.remove(str(app.media / "folder1" / "rec1.mkv"))
        state = inventory.inventory()
        assert len(state["X"]) == 3 * 7
        self._same_as_a_full_read(state)

    def test_the_trash_reads_nothing(self, app):
        _stock(app, files=2)
        a = inventory.inventory()
        app.db.move_to_trash([a["meta"][0]["id"]])
        app.db.vectors_by_id = lambda *a, **k: pytest.fail("read vectors for the trash")
        try:
            b = inventory.inventory()
        finally:
            del app.db.vectors_by_id
        assert b["X"] is a["X"] and b["meta"] is a["meta"]
        assert b["inventory_generation"] > a["inventory_generation"]

    def test_new_ids_below_held_ones_land_in_id_order(self, app):
        """SQLite hands out rising ids — but should a lower one ever come,
        the held matrix must still be in id order, as a full read is."""
        _stock(app, files=3)
        inventory.inventory()
        victim = str(app.media / "folder0" / "rec0.mkv")
        rows = app.db.windows(victim)
        low = [r["id"] for r in rows]
        # Delete the file behind the inventory's back, then bring it back
        # under its OLD (lower) ids — as if SQLite had reused them.
        X_all, meta_all = app.db.all_vectors()
        keep = [m for m in meta_all if m["id"] in low]
        vecs = {m["id"]: X_all[i] for i, m in enumerate(meta_all) if m["id"] in low}
        state = inventory.inventory()
        state["ids"] = state["ids"][~np.isin(state["ids"], low)]
        state["meta"] = [m for m in state["meta"] if m["id"] not in low]
        state["X"] = state["X"][[i for i, m in enumerate(meta_all) if m["id"] not in low]]
        app.db._changed(inventory=True)
        followed = inventory.inventory()
        assert followed["ids"].tolist() == sorted(followed["ids"].tolist())
        assert len(keep) == len(vecs) == len(low)
        self._same_as_a_full_read(followed)

    def test_rows_by_id_without_a_dict(self):
        rows = inventory._Rows(np.array([3, 7, 10, 42], dtype=np.int64))
        assert rows[7] == 1 and rows.get(42) == 3
        assert rows.get(5) is None and rows.get(99) is None and rows.get(1) is None
        assert 3 in rows and 8 not in rows
        assert dict(rows) == {3: 0, 7: 1, 10: 2, 42: 3}
        assert inventory._Rows(np.zeros(0, np.int64)).get(1) is None

    def test_new_vectors_reload_the_matrix(self, app):
        _stock(app, files=2)
        a = inventory.inventory()
        app.db.write(
            str(app.media / "x.mkv"),
            speaker_core.MODEL_BRAND,
            [
                {
                    "label": "SPEAKER_00",
                    "windows": 0,
                    "start_sec": 0.0,
                    "speech_sec": 30.0,
                    "segments": 1,
                    "v_separated": _vec(5),
                }
            ],
        )
        b = inventory.inventory()
        assert b["X"] is not a["X"] and len(b["X"]) == len(a["X"]) + 1


# ── Byte for byte what the endpoint sent before ──────────────────────────


def _old_answer(X, meta, names, removed, trash_only, limit, link_method, k, root):
    """`map_graph` as it was before the held parts (without the empty case)."""
    if removed:
        keep = [i for i, m in enumerate(meta) if (m["id"] in removed) == trash_only]
        X, meta = X[keep], [meta[i] for i in keep]
    device = "cpu"
    full_idx, full_value, _ = speaker_knn.neighbours(X, k=speaker_knn.K, device=device)
    run = speaker_knn._group_voices(
        X, limit, speaker_knn.K, link_method, device, None, (full_idx, full_value)
    )
    markers = np.asarray(run["markers"])
    idx, value = full_idx[:, :k], full_value[:, :k]
    edges = {}
    for i in range(len(X)):
        for s in range(idx.shape[1]):
            j = int(idx[i, s])
            w = float(value[i, s])
            if j == i or w < limit:
                continue
            a, b = (i, j) if i < j else (j, i)
            edges[(a, b)] = round(w, 4)

    def display(p):
        p = p or ""
        return p[len(root) :].lstrip("/") if p.startswith(root) else p

    total_sum = {}
    for (a, b), w in edges.items():
        if markers[a] == markers[b] and markers[a] > 0:
            total_sum.setdefault(int(markers[a]), []).append(w)
    by_group = {}
    for i, m in enumerate(markers):
        m = int(m)
        if m > 0:
            by_group.setdefault(m, []).append(i)
    groups = []
    for nr in sorted(by_group):
        members = by_group[nr]
        value_list = total_sum.get(nr, [])
        counted = {}
        for i in members:
            n = names.get(meta[i]["id"])
            if n:
                counted[n] = counted.get(n, 0) + 1
        groups.append(
            {
                "nr": nr,
                "n": len(members),
                "mean": round(sum(value_list) / len(value_list), 3) if value_list else None,
                "name": max(counted, key=counted.get) if counted else None,
                "people": sorted(counted.items(), key=lambda t: -t[1]),
            }
        )
    groups.sort(key=lambda g: -g["n"])
    file_names, file_index, file_idx = [], {}, []
    for m in meta:
        shown = display(m["abs_path"])
        if shown not in file_index:
            file_index[shown] = len(file_names)
            file_names.append(shown)
        file_idx.append(file_index[shown])
    out = {
        "threshold": round(limit, 4),
        "link_method": link_method,
        "k": k,
        "voices": len(X),
        "files": len({m["abs_path"] for m in meta}),
        "group_count": len(groups),
        "loners": int(sum(1 for g in groups if g["n"] == 1)),
        "root": root,
        # The trash count comes with every answer — the map's
        # switch to the trash shows it (it came with an empty one only).
        "trash": len(removed),
        "nodes": {
            "id": [m["id"] for m in meta],
            "group_nr": [int(g) for g in markers],
            "file_idx": file_idx,
            "label": [m["label"] for m in meta],
            "start": [m["start_sec"] for m in meta],
            "windows": [m["windows"] for m in meta],
            "speech_sec": [m["speech_sec"] for m in meta],
            "name": [names.get(m["id"]) for m in meta],
        },
        "file_names": file_names,
        "edges": [[a, b, w] for (a, b), w in edges.items()],
        "groups": groups,
    }
    return json.dumps(out, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n"


class TestSameBytesAsBefore:
    @pytest.mark.parametrize(
        "threshold,link_method,k,trash_only",
        [
            (0.5, "average", 5, False),
            (0.3, "complete", 3, False),
            (0.7, "single", 8, False),
            (0.5, "average", 5, True),
        ],
    )
    def test_the_answer_is_unchanged(self, app, threshold, link_method, k, trash_only):
        _stock(app, files=8, each=9)
        inv = inventory.inventory()
        meta = inv["meta"]
        # A few names — two people in one group, so the legend has to
        # decide — and a trash
        app.db.assign(meta[0]["id"], "Alice")
        app.db.assign(meta[4]["id"], "Alice")
        app.db.assign(meta[8]["id"], "Bob")
        app.db.assign(meta[12]["id"], "Frank Ö")  # escaped as ASCII
        app.db.move_to_trash([meta[i]["id"] for i in (3, 20, 21, 40)])
        inv = inventory.inventory()
        want = _old_answer(
            inv["X"],
            inv["meta"],
            app.db.names_per_vector(),
            app.db.trash_ids(),
            trash_only,
            threshold,
            link_method,
            k,
            str(app.media),
        )
        q = f"threshold={threshold}&link_method={link_method}&k={k}"
        if trash_only:
            q += "&trash=1"
        got = app.test_client().get("/api/voicemap/graph?" + q).data.decode()
        assert got == want


class TestTheEdges:
    """`speaker_graph.edges` against the double loop it replaced."""

    @staticmethod
    def _loop(idx, value, limit):
        edges = {}
        for i in range(idx.shape[0]):
            for s in range(idx.shape[1]):
                j, w = int(idx[i, s]), float(value[i, s])
                if j == i or w < limit:
                    continue
                edges[(min(i, j), max(i, j))] = round(w, 4)
        return edges

    @pytest.mark.parametrize("seed", range(20))
    def test_order_and_values(self, seed):
        r = np.random.default_rng(seed)
        n, k = int(r.integers(2, 60)), int(r.integers(1, 8))
        idx = r.integers(0, n, size=(n, k)).astype(np.int32)
        # Both directions of a pair with values differing in the last bit,
        # values exactly at the limit, and self loops
        value = r.uniform(0.2, 1.0, size=(n, k)).astype(np.float32)
        limit = float(value[0, 0]) if seed % 2 else 0.6
        value[r.random(size=(n, k)) < 0.1] = np.float32(0.6)
        idx[r.random(size=(n, k)) < 0.05] = 0
        a, b, w = speaker_graph.edges(idx, value, limit)
        want = self._loop(idx, value, limit)
        assert list(zip(a.tolist(), b.tolist())) == list(want)
        assert w.tolist() == list(want.values())

    def test_rounding_agrees_with_python_for_every_float32_boundary(self):
        # Values whose 4th decimal is a 5 — where np.round and round could
        # part ways if the scaled value were not exact.
        v = np.arange(0, 10001, dtype=np.float64) / 10000 + 0.00005
        w32 = v.astype(np.float32).astype(np.float64)
        assert np.round(w32, 4).tolist() == [round(x, 4) for x in w32.tolist()]


class TestTheFastEncoders:
    """Legend and edge list are joined by hand — they must still be what
    `json.dumps` writes, down to the byte."""

    @pytest.mark.parametrize(
        "mean", [None, 0.9, 0.123, 1.0, 0.0, 1e-05, 0.30000000000000004, float("nan"), float("inf")]
    )
    def test_a_legend_entry(self, mean):
        want = speaker_graph._dump({"nr": 17, "n": 3, "mean": mean, "name": None, "people": []})
        assert speaker_graph._legend_entry(17, 3, mean) == want

    def test_the_edge_list(self):
        r = np.random.default_rng(3)
        a = r.integers(0, 10**6, 5000)
        b = r.integers(0, 10**6, 5000)
        w = np.round(r.uniform(0, 1, 5000).astype(np.float32).astype(np.float64), 4)
        w[:3] = [1.0, 0.0, 1e-05]
        want = speaker_graph._dump(
            [[x, y, z] for x, y, z in zip(a.tolist(), b.tolist(), w.tolist())]
        )
        assert speaker_graph._edge_list(a, b, w) == want
        w[5] = np.nan  # the loop kept NaN; json writes it as NaN
        want = speaker_graph._dump(
            [[x, y, z] for x, y, z in zip(a.tolist(), b.tolist(), w.tolist())]
        )
        assert speaker_graph._edge_list(a, b, w) == want
