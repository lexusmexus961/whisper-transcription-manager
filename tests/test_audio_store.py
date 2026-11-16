"""The fingerprint index on disk (audio/index.py) and audio.db
(audio/store.py): segments written and merged, entries in memory searched
at once, analyses replaced, a crash before writing, another version."""

from __future__ import annotations

import json

import numpy as np
import pytest

from tests.test_audio_fingerprint import song
from whisper_tm.audio import analyze, fingerprint, index
from whisper_tm.audio import store as store_mod
from whisper_tm.audio.fingerprint import SR


def _entries(n, seed=0, ids=10):
    rng = np.random.default_rng(seed)
    h = rng.integers(0, index.BUCKETS, n).astype(np.uint32)
    e = index.pack(rng.integers(1, ids + 1, n), rng.integers(0, 1 << 20, n))
    return h, e


# ── Segments ─────────────────────────────────────────────────────────────


def test_pack_and_unpack():
    ids = np.array([1, 2, index.MAX_ID])
    fr = np.array([0, 12345, index.MAX_T])
    got_ids, got_fr = index.unpack(index.pack(ids, fr))
    assert got_ids.tolist() == ids.tolist() and got_fr.tolist() == fr.tolist()


def test_a_segment_answers_what_was_written(tmp_path):
    h, e = _entries(50_000)
    assert index.write(tmp_path / "seg-000001.afp", h, e) == 50_000
    seg = index.Segment(tmp_path / "seg-000001.afp")
    q = np.unique(h[:300])
    got, which = seg.lookup(q)
    want = sorted((int(hh), int(ee)) for hh, ee in zip(h, e, strict=True) if hh in set(q.tolist()))
    assert sorted(zip(q[which].tolist(), got.tolist(), strict=True)) == want
    assert seg.bucket_sizes(q).sum() == len(want)
    # A bucket over the limit is left out
    big = q[np.argmax(seg.bucket_sizes(q))]
    got, _ = seg.lookup(np.array([big], np.uint32), max_bucket=0)
    assert len(got) == seg.bucket_sizes(np.array([big]))[0]
    seg.close()


def test_a_cut_off_segment_is_refused(tmp_path):
    h, e = _entries(1000)
    p = tmp_path / "seg-000001.afp"
    index.write(p, h, e)
    p.write_bytes(p.read_bytes()[:-7])
    with pytest.raises(ValueError):
        index.Segment(p)


def test_merge_keeps_everything_but_the_dead(tmp_path):
    parts = [_entries(20_000, seed=s) for s in range(3)]
    segs = []
    for i, (h, e) in enumerate(parts):
        index.write(tmp_path / f"seg-{i:06d}.afp", h, e)
        segs.append(index.Segment(tmp_path / f"seg-{i:06d}.afp"))
    n = index.merge(segs, tmp_path / "seg-000009.afp", dead=np.array([3, 7]))
    merged = index.Segment(tmp_path / "seg-000009.afp")
    h = np.concatenate([p[0] for p in parts])
    e = np.concatenate([p[1] for p in parts])
    ids, _ = index.unpack(e)
    keep = ~np.isin(ids, [3, 7])
    assert n == merged.count == int(keep.sum())
    q = np.unique(h)
    got, which = merged.lookup(q)
    assert sorted(zip(q[which].tolist(), got.tolist(), strict=True)) == sorted(
        zip(h[keep].tolist(), e[keep].tolist(), strict=True)
    )


def test_a_stopped_merge_leaves_nothing(tmp_path):
    h, e = _entries(5000)
    index.write(tmp_path / "seg-000001.afp", h, e)
    seg = index.Segment(tmp_path / "seg-000001.afp")
    with pytest.raises(InterruptedError):
        index.merge([seg], tmp_path / "seg-000002.afp", stop=lambda: True)
    assert not list(tmp_path.glob("seg-000002*"))


# ── The store ────────────────────────────────────────────────────────────


@pytest.fixture
def make_store(tmp_path):
    made = []

    def make():
        s = store_mod.AudioStore(tmp_path / "audio.db")
        made.append(s)
        return s

    yield make
    for s in made:
        s.close()


def _add(st, path, x, **kw):
    res, h, t = analyze.measure([x])
    return analyze._store_result(
        st, path, 1000 + len(x), 5.0, res, h, t, source="test", seconds=0.1, **kw
    )


def _find(st, x):
    return analyze.search_snippet(st, x)["hits"]


def test_found_in_memory_then_in_a_segment_then_after_a_restart(make_store):
    st = make_store()
    a, b = song(1, 40), song(2, 40)
    _add(st, "/lib/a.mp3", a)
    _add(st, "/lib/b.mp3", b)
    snippet = b[10 * SR : 16 * SR]
    assert _find(st, snippet)[0]["file"]["path"] == "/lib/b.mp3"
    assert st.counts()["pending"] > 0 and st.counts()["segments"] == 0
    st.flush()
    assert st.counts()["pending"] == 0 and st.counts()["segments"] == 1
    assert _find(st, snippet)[0]["file"]["path"] == "/lib/b.mp3"
    st.close()
    again = make_store()
    hits = _find(again, snippet)
    assert hits[0]["file"]["path"] == "/lib/b.mp3" and abs(hits[0]["at"] - 10) <= 0.1


def test_analyzed_again_replaces_the_old_fingerprints(make_store):
    st = make_store()
    old, new = song(3, 30), song(4, 30)
    first = _add(st, "/lib/x.mp3", old)
    st.flush()
    second = _add(st, "/lib/x.mp3", new)  # the file changed
    assert second != first
    assert _find(st, old[5 * SR : 11 * SR]) == []
    assert _find(st, new[5 * SR : 11 * SR])[0]["id"] == second
    with st.connect() as c:
        assert c.execute("SELECT count(*) FROM dead WHERE id=?", (first,)).fetchone()[0] == 1
    # Replaced while still in memory: dropped from memory, not marked dead
    third = _add(st, "/lib/x.mp3", old)
    assert _find(st, new[5 * SR : 11 * SR]) == []
    with st.connect() as c:
        assert c.execute("SELECT count(*) FROM dead WHERE id=?", (second,)).fetchone()[0] == 0
    assert _find(st, old[5 * SR : 11 * SR])[0]["id"] == third


def test_ids_are_never_reused(make_store):
    st = make_store()
    first = _add(st, "/lib/a.mp3", song(5, 12))
    st.remove("/lib/a.mp3")
    assert _add(st, "/lib/b.mp3", song(6, 12)) > first


def test_a_crash_before_writing_analyzes_them_again(make_store, tmp_path):
    st = make_store()
    _add(st, "/lib/a.mp3", song(7, 20))
    st.flush()
    _add(st, "/lib/b.mp3", song(8, 20))  # only in memory
    st._stop.set()  # "killed": nothing written
    st._closed = True
    again = make_store()
    assert again.file("/lib/a.mp3") is not None
    assert again.file("/lib/b.mp3") is None  # the next run takes it again


def test_merges_follow_the_tiers_and_drop_the_dead(make_store, monkeypatch):
    monkeypatch.setattr(store_mod, "_TIER_BASE", 1)
    st = make_store()
    recs = {f"/lib/{i}.mp3": song(20 + i, 15) for i in range(5)}
    for path, x in recs.items():
        _add(st, path, x)
        st.flush()
    st.remove("/lib/0.mp3")
    assert st.counts()["segments"] == 5
    while st.merge():
        pass
    assert st.counts()["segments"] < 5
    st.merge(list(st._segments))  # everything: the dead are gone for good
    assert st.counts()["segments"] == 1
    with st.connect() as c:
        assert c.execute("SELECT count(*) FROM dead").fetchone()[0] == 0
    assert _find(st, recs["/lib/0.mp3"][2 * SR : 8 * SR]) == []
    for path in list(recs)[1:]:
        assert _find(st, recs[path][2 * SR : 8 * SR])[0]["file"]["path"] == path
    assert not list(st.folder.glob("merge-*.json"))


def test_an_interrupted_merge_is_finished_at_the_next_start(make_store):
    st = make_store()
    _add(st, "/lib/a.mp3", song(30, 15))
    st.flush()
    _add(st, "/lib/b.mp3", song(31, 15))
    st.flush()
    names = [s.path.name for s in st._segments]
    st.merge(list(st._segments))
    merged = [s.path.name for s in st._segments][0]
    st.close()
    # As if the server died after the new segment and before the old ones went
    folder = st.folder
    for n in names:
        (folder / n).write_bytes((folder / merged).read_bytes())
    (folder / f"merge-{merged[:-4]}.json").write_text(
        json.dumps({"output": merged, "sources": names})
    )
    again = make_store()
    assert [s.path.name for s in again._segments] == [merged]
    assert not list(folder.glob("merge-*.json"))


def test_another_version_starts_over(make_store, monkeypatch):
    st = make_store()
    _add(st, "/lib/a.mp3", song(40, 15))
    st.flush()
    st.close()
    monkeypatch.setattr(store_mod, "VERSION", fingerprint.VERSION + 1)
    again = make_store()
    assert again.counts()["files"] == 0 and not list(again.folder.glob("seg-*.afp"))


def test_failed_recordings_and_currency(make_store):
    st = make_store()
    st.add("/lib/broken.mp4", 10, 1.0, error="moov atom not found", source="run")
    assert st.is_current("/lib/broken.mp4", 10, 1.0)
    assert not st.is_current("/lib/broken.mp4", 11, 1.0)
    assert not st.is_current("/lib/broken.mp4", 10, 2.0)
    assert st.counts()["errors"] == 1 and st.counts()["files"] == 0


def test_levels_belong_to_one_version_of_the_file(make_store):
    st = make_store()
    _add(st, "/lib/a.mp3", song(50, 6))
    size = 1000 + 6 * SR
    lv = st.levels("/lib/a.mp3", size, 5.0)
    assert lv is not None and len(lv) == 6 * 20
    assert st.levels("/lib/a.mp3", size + 1, 5.0) is None
    assert st.levels("/lib/a.mp3", size, 6.0) is None
    assert st.levels("/lib/other.mp3") is None


def test_relations_are_stored_and_go_with_their_recordings(make_store):
    st = make_store()
    a, b = song(60, 40), song(61, 30)
    _add(st, "/lib/a.mp3", a)
    _add(st, "/lib/b.mp3", b)
    _add(st, "/lib/a copy.mp3", a * 0.7)
    _add(st, "/lib/both.mp3", np.concatenate([a, b]))
    with st.connect() as c:
        rows = c.execute(
            "SELECT a.path, b.path, r.kind, r.offset FROM relations r "
            "JOIN files a ON a.id = r.a JOIN files b ON b.id = r.b"
        ).fetchall()
    got = {(r[0], r[1], r[2]): r[3] for r in rows}
    assert ("/lib/a.mp3", "/lib/a copy.mp3", "same") in got
    assert got[("/lib/both.mp3", "/lib/b.mp3", "contains")] == pytest.approx(40.0, abs=0.1)
    assert got[("/lib/both.mp3", "/lib/a.mp3", "contains")] == pytest.approx(0.0, abs=0.1)
    st.remove("/lib/both.mp3")
    with st.connect() as c:
        assert c.execute("SELECT count(*) FROM relations WHERE kind='contains'").fetchone()[0] == 0


def test_clear(make_store):
    st = make_store()
    _add(st, "/lib/a.mp3", song(70, 10))
    st.flush()
    st.clear()
    c = st.counts()
    assert c["files"] == 0 and c["segments"] == 0 and c["pending"] == 0
    assert not list(st.folder.glob("seg-*.afp"))


def test_copies_ask_the_sound(make_store, monkeypatch):
    """core/copies asks audio.db for a pair: "same", or which one contains
    the other and from where — nothing for recordings it doesn't know."""
    from whisper_tm.audio import runner
    from whisper_tm.core import copies

    st = make_store()
    monkeypatch.setattr(store_mod, "store_path", lambda: st.path)
    store_mod._stores[str(st.path)] = st
    monkeypatch.setattr(
        runner, "_cfg", lambda name, default=None: {"enabled": True}.get(name, default)
    )
    a, b = song(80, 30), song(81, 20)
    _add(st, "/lib/a.mp3", a)
    _add(st, "/lib/b.mp3", b)
    _add(st, "/lib/a2.mp3", a)
    _add(st, "/lib/ab.mp3", np.concatenate([a, b]))
    assert copies.sound_relation("/lib/a2.mp3", "/lib/a.mp3")[0] == "same"
    kind, whole, at = copies.sound_relation("/lib/b.mp3", "/lib/ab.mp3")
    assert kind == "contains" and whole == "/lib/ab.mp3" and at == pytest.approx(30.0, abs=0.1)
    assert copies.sound_relation("/lib/a.mp3", "/lib/b.mp3") is None
    assert copies.sound_relation("/lib/a.mp3", "/elsewhere/x.mp3") is None
    store_mod._stores.pop(str(st.path), None)


def test_segment_names_are_never_given_twice(make_store):
    """A merge writes its output for minutes; a flush meanwhile must not
    pick the same name (one of the two would be lost)."""
    st = make_store()
    names = {st._next_name() for _ in range(5)}
    assert len(names) == 5
    _add(st, "/lib/a.mp3", song(90, 10))
    st.flush()
    assert st._segments[-1].path.name not in names


def test_a_search_snapshot_survives_a_merge(make_store):
    st = make_store()
    a = song(91, 20)
    for i in range(3):
        _add(st, f"/lib/{i}.mp3", a if i == 0 else song(92 + i, 20))
        st.flush()
    snap = st.snapshot()  # a search in progress
    st.merge(list(st._segments))
    h, t = fingerprint.query_landmarks(a[5 * SR : 11 * SR])
    hits = analyze.search(snap, h, t)
    assert hits and hits[0]["id"] == st.file("/lib/0.mp3")["id"]


def test_a_long_snippet_is_fingerprinted_as_the_index_is(make_store):
    st = make_store()
    a = song(95, 120)
    _add(st, "/lib/long.mp3", a)
    r = analyze.search_snippet(st, a[10 * SR : 70 * SR])  # a minute
    assert r["hits"][0]["file"]["path"] == "/lib/long.mp3"
    assert abs(r["hits"][0]["at"] - 10) <= 0.1
    assert r["hashes"] < 15 * 60  # stored density, not 4 × the query's
