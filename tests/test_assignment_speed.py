"""Speed up speaker assignment without changing its result.

Goal: the speaker calculation well under a minute on a large library.

Measured on a large library — hundreds of thousands of windows, **tens of
thousands of centroids**:

    ==========================  =======
    load from database              1.3 s
    build centroids                 0.7 s
    matrix product                  2.0 s
    **linkage (scipy)**          **29.7 s**
    ==========================  =======

So linkage is practically everything. It's O(n²) in time AND memory:
the distance matrix alone would be several GiB, plus the float64 copy in scipy.

Two levers, both without any change to the result:

**1. Cache was on the wrong counter.** It listened to `generation`, which changes
with every edit — including renaming. Every click on "that's Alice" threw away
the calculation, even though a name doesn't move a single centroid. Exactly in
the work step you do hundreds of times in a row. Now it listens to
`inventory_generation`: new recording, deleted recording, trash — nothing else.

**2. Linkage component-wise.** Every finished group is a clique below the cut and
lies entirely in ONE connected component; across component boundaries, merging
only happens above the cut, and that throws `fcluster` away. On the same data:

    ==========================  ======  =========================
    scipy, in one piece (before)  33 s
    decomposed + scipy            19 s  same groups, identical
    decomposed + fastcluster      13 s  same groups, identical
    ==========================  ======  =========================

The centroids break into thousands of components, most of them singletons.
But 33 s doesn't become 3: one giant component holds most of the centroids,
linked via chains.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from tests.test_speaker_assignment import _voice, _write  # noqa: F401
from whisper_tm.core import speaker_assignment as assignment
from whisper_tm.core import speaker_inventory as inventory
from whisper_tm.database.speaker import SpeakerDB

DIM = 192


def _in_one_go(C: np.ndarray, cut: float) -> np.ndarray:
    """The previous version — used as the benchmark in the test."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    if len(C) < 2:
        return np.zeros(len(C), dtype=int)
    D = 1.0 - (C @ C.T)
    np.fill_diagonal(D, 0.0)
    np.clip(D, 0.0, 2.0, out=D)
    Z = linkage(squareform(D, checks=False), method="complete")
    return fcluster(Z, t=1.0 - cut, criterion="distance")


@pytest.fixture(autouse=True)
def _recount_at_once(monkeypatch):
    """The tests here watch the background recount itself — without the
    minute between two recounts (`TestTheRecountIsThrottled` has that)."""
    monkeypatch.setattr(assignment, "RECOUNT_EVERY_SEC", 0)


def _wait_for(limit: float = 5.0) -> None:
    """Wait for the background run — only in the test.

    In production nobody waits for it; here, though, the check needs to
    see what it did.
    """
    end = time.perf_counter() + limit
    while assignment.INVENTORY._computing and time.perf_counter() < end:
        time.sleep(0.02)


def _partition(markers) -> list[tuple]:
    """Labels are arbitrary — the partition is not."""
    sort_by = {}
    for i, m in enumerate(np.asarray(markers).tolist()):
        sort_by.setdefault(m, []).append(i)
    return sorted(tuple(v) for v in sort_by.values())


class TestTheResultStaysTheSame:
    """The only point at which speed would be off-limits: if it changes
    the groups."""

    @pytest.mark.parametrize("seed", range(6))
    def test_random_positions_give_the_same_split(self, seed, monkeypatch):
        monkeypatch.setattr(assignment, "control_value", lambda kind: 0.6)
        r = np.random.default_rng(seed)
        # A few real clusters plus scatter — otherwise everything falls
        # apart into loners and the test checks nothing.
        cores = r.normal(size=(7, DIM))
        raw = np.vstack([k + 0.35 * r.normal(size=(9, DIM)) for k in cores])
        C = (raw / np.linalg.norm(raw, axis=1, keepdims=True)).astype(np.float32)
        assert _partition(assignment.Inventory._concat(C)) == _partition(_in_one_go(C, 0.6))

    def test_even_when_everything_is_connected(self, monkeypatch):
        """A single giant component is the worst case — and the real
        one: in a large library most centroids are connected."""
        monkeypatch.setattr(assignment, "control_value", lambda kind: 0.3)
        r = np.random.default_rng(3)
        raw = np.ones((40, DIM)) + 0.05 * r.normal(size=(40, DIM))
        C = (raw / np.linalg.norm(raw, axis=1, keepdims=True)).astype(np.float32)
        assert _partition(assignment.Inventory._concat(C)) == _partition(_in_one_go(C, 0.3))

    def test_when_nothing_is_connected(self, monkeypatch):
        """Nothing but loners: each is its own group, and none may be
        lost."""
        monkeypatch.setattr(assignment, "control_value", lambda kind: 0.99)
        r = np.random.default_rng(4)
        raw = r.normal(size=(25, DIM))
        C = (raw / np.linalg.norm(raw, axis=1, keepdims=True)).astype(np.float32)
        g = assignment.Inventory._concat(C)
        assert len(set(g.tolist())) == 25
        assert _partition(g) == _partition(_in_one_go(C, 0.99))

    def test_a_single_centroid(self, monkeypatch):
        monkeypatch.setattr(assignment, "control_value", lambda kind: 0.7)
        C = np.zeros((1, DIM), np.float32)
        C[0, 0] = 1.0
        assert len(assignment.Inventory._concat(C)) == 1

    def test_the_chain_still_breaks_apart(self, monkeypatch):
        """The whole reason full linkage exists: A~B~C, but A and C are
        unrelated to each other. Connectivity alone would turn this into
        ONE group — the decomposition must not bring that back through
        the back door."""
        monkeypatch.setattr(assignment, "control_value", lambda kind: 0.8)
        a = np.zeros(DIM)
        a[0] = 1.0
        c = np.zeros(DIM)
        c[1] = 1.0
        b = (a + c) / np.linalg.norm(a + c)  # ~0.707 to both
        C = np.stack([a, b, c]).astype(np.float32)
        C /= np.linalg.norm(C, axis=1, keepdims=True)
        g = assignment.Inventory._concat(C)
        assert len(set(g.tolist())) == 3, "the chain was merged back into one group"


class TestTheCacheListensToTheInventory:
    """The most expensive bug wasn't a computation error, but a cache
    that discarded at the wrong time."""

    @pytest.fixture
    def db(self, tmp_path, monkeypatch):
        from whisper_tm.config import config

        media = tmp_path / "media"
        media.mkdir()
        monkeypatch.setattr(type(config), "media_dir", property(lambda self: media))
        d = SpeakerDB(tmp_path / "speaker.db")
        d.initialize()
        monkeypatch.setattr("whisper_tm.database.speaker.speaker_db", lambda: d)
        inventory.discard()
        assignment.INVENTORY.discard()

        class Help:
            pass

        h = Help()
        h.db = d
        h.media = media
        for file in ("one.mkv", "two.mkv"):
            _write(h, file, "SPEAKER_00", _voice(11), n=4, sec=90.0)
        return h

    def _computations(self, db, monkeypatch):
        """Counts how often the computation actually runs."""
        counter = {"n": 0}
        real = assignment.Inventory._compute

        def _with(X, meta, mp=None):
            counter["n"] += 1
            return real(X, meta, mp)

        monkeypatch.setattr(assignment.Inventory, "_compute", staticmethod(_with))
        return counter

    def test_a_name_does_not_discard_the_computation(self, db, monkeypatch):
        """*The* everyday case: a hundred assignments in a row."""
        z = self._computations(db, monkeypatch)
        assignment.INVENTORY.fetch()
        assert z["n"] == 1
        db.db.assign_many([1, 2], "Alice")
        assignment.INVENTORY.fetch()
        assert z["n"] == 1, "a name discarded the whole computation"

    def test_new_voices_do_discard_it(self, db, monkeypatch):
        """The computation runs in the BACKGROUND — the
        request gets the old state immediately. The new recording still
        has to arrive though."""
        z = self._computations(db, monkeypatch)
        assignment.INVENTORY.fetch()
        _write(db, "three.mkv", "SPEAKER_00", _voice(22), n=4, sec=90.0)
        assignment.INVENTORY.fetch()
        _wait_for()
        assert z["n"] == 2, "the new recording didn't arrive in the grouping"

    def test_the_trash_does_too(self, db, monkeypatch):
        z = self._computations(db, monkeypatch)
        state_info = assignment.INVENTORY.fetch()
        db.db.move_to_trash([int(state_info["vectors"][0][0])], "Test")
        assignment.INVENTORY.fetch()
        _wait_for()
        assert z["n"] == 2

    def test_assigning_computes_nothing_at_all(self, db, monkeypatch):
        """The most expensive step in the whole flow — and it wasn't in
        the cache, but right next to it.

        `apply()` manually called `INVENTORY.discard()` twice, with
        the reasoning "generation has changed": once after setting, once
        after propagation. Every assignment thus made the grouping
        recompute TWICE, at 13 seconds each — dozens of computations in
        one hour of assigning.
        """
        z = self._computations(db, monkeypatch)
        v = assignment.suggestion()
        assert z["n"] == 1
        assignment.apply(v["vector_ids"], "Alice")
        assert z["n"] == 1, "the assignment recomputed the grouping"
        assignment.suggestion()
        assert z["n"] == 1, "the next suggestion recomputed it"

    def test_the_name_still_arrives_at_once(self, db):
        """Without recomputation, nothing may get lost: the assigned
        voices must immediately count as taken."""
        v = assignment.suggestion()
        before = v["progress"]["open_hours"]
        out = assignment.apply(v["vector_ids"], "Alice")
        assert out["set_flag"] == len(v["vector_ids"])
        assert out["progress"]["open_hours"] < before, (
            "the assignment didn't show up in the progress"
        )
        onward = assignment.suggestion()
        if onward is not None:
            assert onward["group_nr"] != v["group_nr"], "the same group is presented again"

    def test_the_markers_run_separately(self, db):
        before = (db.db.generation, db.db.inventory_generation)
        db.db.assign_many([1], "Alice")
        assert db.db.generation > before[0], "the general marker stands still"
        assert db.db.inventory_generation == before[1], "a name moved the inventory marker"


class TestTheTimingsAreInTheAnswer:
    """Exact timings, so production shows what is costing time where.

    The numbers have to be right there in the response, not in the log:
    whoever has to go looking for them never sees them. And they must
    answer the question that matters day to day — *was it recomputed, or
    was the cache warm?*
    """

    @pytest.fixture
    def ready(self, tmp_path, monkeypatch):
        from whisper_tm.config import config

        media = tmp_path / "media"
        media.mkdir()
        monkeypatch.setattr(type(config), "media_dir", property(lambda self: media))
        d = SpeakerDB(tmp_path / "speaker.db")
        d.initialize()
        monkeypatch.setattr("whisper_tm.database.speaker.speaker_db", lambda: d)
        inventory.discard()
        assignment.INVENTORY.discard()

        class Help:
            pass

        h = Help()
        h.db = d
        h.media = media
        for file in ("one.mkv", "two.mkv"):
            _write(h, file, "SPEAKER_00", _voice(11), n=4, sec=90.0)
        return h

    def test_every_suggestion_brings_its_breakdown(self, ready):
        v = assignment.suggestion()
        assert v is not None
        z = v["times"]
        steps = [s["step"] for s in z["steps"]]
        for expected in ("Collection", "open mask", "sort", "Samples", "Progress"):
            assert expected in steps, steps
        assert z["total_sec"] >= 0

    def test_the_first_time_the_computation_is_included(self, ready):
        """The first call computes — then it must also show WHERE the
        time went: loading, centroids, edges, linking."""
        assignment.INVENTORY.discard()
        v = assignment.suggestion()
        steps = [s["step"] for s in v["times"]["inventory"]]
        assert any(s.startswith("Load vectors") for s in steps), steps
        assert any(s.startswith("Centroids") for s in steps), steps
        assert any(s.startswith("Edges") for s in steps), steps
        assert any(s.startswith("Linking") for s in steps), steps

    def test_the_second_time_the_inventory_is_warm(self, ready):
        """The number that lets you see immediately in production
        whether the cache is hitting."""
        assignment.suggestion()
        v = assignment.suggestion()
        inventory_sec = [s["sec"] for s in v["times"]["steps"] if s["step"] == "Collection"][0]
        assert inventory_sec < 0.05, f"the inventory was recomputed: {inventory_sec}s"

    def test_the_edge_count_is_in_the_name(self, ready):
        """Without it, "edges 3.1 s" says nothing — with it, you see
        immediately whether it's the volume or the computation path
        that's the problem."""
        assignment.INVENTORY.discard()
        v = assignment.suggestion()
        names = [s["step"] for s in v["times"]["inventory"]]
        assert any(s.startswith("Edges (") for s in names), names

    def test_the_measurements_also_reach_the_overview(self, ready):
        """`/api/debug/timings` shows them without debug mode — there you
        see the trend across many calls instead of just the last one."""
        from whisper_tm.core import debug

        assignment.INVENTORY.discard()
        assignment.suggestion()
        markers = {r["label"] for r in debug.timings()}
        assert "Assignment" in markers
        assert "Assignment · linking" in markers, markers
        assert any(m.startswith("Suggestion") for m in markers), markers


class TestServingStaleResults:
    """The assignment itself must not be slow either.

    The assignment itself takes 0.2 s afterward — what remained was the
    NEXT suggestion: the queue drops off new voices every few minutes,
    and then the next click costs 15 seconds even though the user did
    nothing.

    For the question "which group do I take next," though, a grouping
    from five minutes ago is just as usable. So: deliver the old one
    immediately and compute the new one in the background.
    """

    @pytest.fixture
    def db(self, tmp_path, monkeypatch):
        from whisper_tm.config import config

        media = tmp_path / "media"
        media.mkdir()
        monkeypatch.setattr(type(config), "media_dir", property(lambda self: media))
        d = SpeakerDB(tmp_path / "speaker.db")
        d.initialize()
        monkeypatch.setattr("whisper_tm.database.speaker.speaker_db", lambda: d)
        inventory.discard()
        assignment.INVENTORY.discard()

        class Help:
            pass

        h = Help()
        h.db = d
        h.media = media
        for file in ("one.mkv", "two.mkv"):
            _write(h, file, "SPEAKER_00", _voice(11), n=4, sec=90.0)
        return h

    def test_the_first_time_it_waits(self, db):
        """Without an old state there's nothing to deliver."""
        assignment.INVENTORY.discard()
        state_info = assignment.INVENTORY.fetch()
        assert len(state_info["C"]) > 0

    def test_new_voices_do_not_hold_the_user_up(self, db, monkeypatch):
        """The core: after new vectors, the old state comes back
        IMMEDIATELY — the computation runs alongside."""
        old_value = assignment.INVENTORY.fetch()
        runs = []
        real = assignment.Inventory._compute

        def _slow(X, meta, mp=None):
            runs.append(1)
            time.sleep(0.4)  # an "expensive" computation
            return real(X, meta, mp)

        monkeypatch.setattr(assignment.Inventory, "_compute", staticmethod(_slow))
        _write(db, "three.mkv", "SPEAKER_00", _voice(22), n=4, sec=90.0)

        t0 = time.perf_counter()
        same = assignment.INVENTORY.fetch()
        duration = time.perf_counter() - t0
        assert duration < 0.2, f"it waited: {duration:.2f}s"
        assert same is old_value, "the old state was not returned"

        for _ in range(60):  # give the thread time
            time.sleep(0.05)
            if not assignment.INVENTORY._computing:
                break
        new_value = assignment.INVENTORY.fetch()
        assert new_value is not old_value, "the background run swapped nothing in"
        assert len(new_value["key"]) > len(old_value["key"])

    def test_only_one_computes_at_a_time(self, db, monkeypatch):
        """Two simultaneous computations cost double the memory and
        deliver the same result."""
        assignment.INVENTORY.fetch()
        runs = []
        real = assignment.Inventory._compute

        def _slow(X, meta, mp=None):
            runs.append(1)
            time.sleep(0.3)
            return real(X, meta, mp)

        monkeypatch.setattr(assignment.Inventory, "_compute", staticmethod(_slow))
        _write(db, "three.mkv", "SPEAKER_00", _voice(22), n=4, sec=90.0)
        for _ in range(10):  # ten requests in quick succession
            assignment.INVENTORY.fetch()
        for _ in range(60):
            time.sleep(0.05)
            if not assignment.INVENTORY._computing:
                break
        assert len(runs) == 1, f"{len(runs)} computations at once"

    def test_the_suggestion_says_how_old_its_state_is(self, db):
        v = assignment.suggestion()
        assert v["times"]["inventory_age_sec"] is not None
        assert v["times"]["inventory_age_sec"] >= 0
        assert "recomputing" in v["times"]


class TestTheSliderStaysInstant:
    """The slider is the one exception to "deliver stale".

    Whoever turns the threshold is waiting precisely for its effect. If
    the old state came back there, the slider would look like a dummy —
    the same bug that `test_changed_cut_discards_result`
    was once written against.
    """

    @pytest.fixture
    def db(self, tmp_path, monkeypatch):
        from whisper_tm.config import config

        media = tmp_path / "media"
        media.mkdir()
        monkeypatch.setattr(type(config), "media_dir", property(lambda self: media))
        d = SpeakerDB(tmp_path / "speaker.db")
        d.initialize()
        monkeypatch.setattr("whisper_tm.database.speaker.speaker_db", lambda: d)
        inventory.discard()
        assignment.INVENTORY.discard()

        class Help:
            pass

        h = Help()
        h.db = d
        h.media = media
        basis = _voice(31)
        for i in range(3):
            _write(
                h,
                f"s{i}.mkv",
                "SPEAKER_00",
                basis + 0.25 * _voice(40 + i),
                n=4,
                sec=100.0,
                seed=40 + i,
            )
        return h

    def test_a_new_cut_takes_effect_in_the_same_call(self, db, monkeypatch):
        from whisper_tm.config import config

        wide = assignment.INVENTORY.fetch()["n_groups"]
        monkeypatch.setitem(config._data.setdefault("speaker", {}), "assignment_cut", 0.99)
        narrow = assignment.INVENTORY.fetch()["n_groups"]
        assert narrow > wide, f"the slider only takes effect later ({wide} → {narrow})"


def _speaker_app(tmp_path, monkeypatch):
    from flask import Flask

    from whisper_tm.config import config

    media = tmp_path / "media"
    media.mkdir()
    monkeypatch.setattr(type(config), "media_dir", property(lambda self: media))
    d = SpeakerDB(tmp_path / "speaker.db")
    d.initialize()
    monkeypatch.setattr("whisper_tm.database.speaker.speaker_db", lambda: d)
    monkeypatch.setattr("whisper_tm.api.speaker.speaker_db", lambda: d)
    inventory.discard()
    assignment.INVENTORY.discard()
    from whisper_tm.api.speaker import speaker_bp

    a = Flask(__name__)
    a.register_blueprint(speaker_bp)

    class Help:
        pass

    h = Help()
    h.db = d
    h.media = media
    h.client = a.test_client()
    # Two voices that recur: 11 in three recordings, 22 in two
    for file in ("one.mkv", "two.mkv", "three.mkv"):
        _write(h, file, "SPEAKER_00", _voice(11), n=4, sec=90.0)
    for file in ("four.mkv", "five.mkv"):
        _write(h, file, "SPEAKER_01", _voice(22), n=4, sec=60.0)
    return h


class TestTheRecountIsThrottled:
    """A running queue changes the inventory every few seconds. Each
    suggestion after such a change started a recount — 8–14 s on all cores,
    back to back while the user clicked. Now: at most one a minute."""

    def test_a_fresh_grouping_is_not_recounted_at_once(self, tmp_path, monkeypatch):
        monkeypatch.setattr(assignment, "RECOUNT_EVERY_SEC", 60)
        h = _speaker_app(tmp_path, monkeypatch)
        assignment.INVENTORY.fetch()
        calls = []
        monkeypatch.setattr(assignment.Inventory, "_trigger_recount", lambda self: calls.append(1))
        _write(h, "six.mkv", "SPEAKER_00", _voice(33), n=4, sec=90.0)
        assignment.INVENTORY.fetch()
        assert calls == []

    def test_an_older_one_is(self, tmp_path, monkeypatch):
        monkeypatch.setattr(assignment, "RECOUNT_EVERY_SEC", 60)
        h = _speaker_app(tmp_path, monkeypatch)
        state = assignment.INVENTORY.fetch()
        state["computed_at"] -= 61
        calls = []
        monkeypatch.setattr(assignment.Inventory, "_trigger_recount", lambda self: calls.append(1))
        _write(h, "six.mkv", "SPEAKER_00", _voice(33), n=4, sec=90.0)
        assignment.INVENTORY.fetch()
        assert calls == [1]


class TestDiscardingDoesNotWait:
    """ "Discard" in the guided assignment recomputed the whole grouping in
    the click — 8–14 s per click on a large library. Now the held grouping
    closes what was discarded, as a name closes it."""

    @pytest.fixture
    def h(self, tmp_path, monkeypatch):
        monkeypatch.setattr(assignment, "RECOUNT_EVERY_SEC", 60)
        return _speaker_app(tmp_path, monkeypatch)

    def _count(self, monkeypatch):
        n = {"compute": 0}
        real = assignment.Inventory._compute

        def counted(X, meta, mp=None):
            n["compute"] += 1
            return real(X, meta, mp)

        monkeypatch.setattr(assignment.Inventory, "_compute", staticmethod(counted))
        return n

    def test_the_click_computes_nothing(self, h, monkeypatch):
        v = assignment.suggestion()
        n = self._count(monkeypatch)
        r = h.client.post(
            "/api/speaker/assignment/apply", json={"vector_ids": v["vector_ids"], "trash": True}
        )
        assert r.status_code == 200 and r.get_json()["discarded"] == len(v["vector_ids"])
        assert n["compute"] == 0

    def test_the_discarded_group_is_not_suggested_again(self, h):
        v = assignment.suggestion()
        h.client.post(
            "/api/speaker/assignment/apply", json={"vector_ids": v["vector_ids"], "trash": True}
        )
        after = assignment.suggestion()
        trash = h.db.trash_ids()
        assert after is None or not (set(after["vector_ids"]) & trash)
        assert after is None or after["group_nr"] != v["group_nr"]

    def test_progress_counts_it_as_decided(self, h):
        before = assignment.progress(assignment.INVENTORY.fetch())
        v = assignment.suggestion()
        r = h.client.post(
            "/api/speaker/assignment/apply", json={"vector_ids": v["vector_ids"], "trash": True}
        )
        assert r.get_json()["progress"]["open_hours"] < before["open_hours"]

    def test_a_name_is_not_spread_onto_discarded_voices(self, h):
        state = assignment.INVENTORY.fetch()
        own = [
            i
            for i, (path, _label) in enumerate(state["key"])
            if path.endswith(("one.mkv", "two.mkv", "three.mkv"))
        ]
        third = [
            v for i in own if state["key"][i][0].endswith("three.mkv") for v in state["vectors"][i]
        ]
        first = [
            v for i in own if state["key"][i][0].endswith("one.mkv") for v in state["vectors"][i]
        ]
        h.db.move_to_trash(third, "Test")
        out = assignment.apply(first, "Alice")
        names = h.db.names_per_vector()
        assert not any(v in names for v in third), "a name landed in the trash"
        assert out["spread"] > 0  # two.mkv still got it


class TestTheMasksAreTheLoopsTheyReplaced:
    """`_named` and `discarded` in numpy — against the Python loops of
    before, on centroids with no, one, partial and conflicting names."""

    @staticmethod
    def _state(seed: int):
        r = np.random.default_rng(seed)
        vectors, start = [], 1
        for _ in range(int(r.integers(50, 400))):
            k = int(r.integers(1, 9))
            vectors.append(list(range(start, start + k)))
            start += k
        ids = np.arange(1, start)
        names = {
            int(v): str(r.choice(["Alice", "Bob", "Carol"]))
            for v in r.choice(ids, size=len(ids) // 3, replace=False)
        }
        return {"vectors": vectors, "C": np.zeros((len(vectors), 4), np.float32)}, names, ids, r

    @pytest.mark.parametrize("seed", range(20))
    def test_named(self, seed):
        state, names, _ids, _r = self._state(seed)
        has, single = assignment._named(state, names)
        want_has = [any(v in names for v in vids) for vids in state["vectors"]]
        want_single = {}
        for i, vids in enumerate(state["vectors"]):
            here = {names[v] for v in vids if v in names}
            if len(here) == 1:
                want_single[i] = here.pop()
        assert has.tolist() == want_has
        assert single == want_single and list(single) == list(want_single)

    @pytest.mark.parametrize("seed", range(10))
    def test_discarded(self, seed, monkeypatch):
        state, _names, ids, r = self._state(seed)
        trash = {int(v) for v in r.choice(ids, size=len(ids) // 2, replace=False)}
        # whole centroids too, so "all voices discarded" happens
        for vids in state["vectors"][::7]:
            trash.update(vids)

        class DB:
            trash_generation = 5

            def trash_ids(self):
                return trash

        monkeypatch.setattr("whisper_tm.database.speaker.speaker_db", lambda: DB())
        state["trash_generation"] = 4
        got = assignment.discarded(state)
        want = [all(v in trash for v in vids) for vids in state["vectors"]]
        assert got.tolist() == want
        # held for the same trash state, and None once the grouping knows it
        assert assignment.discarded(state) is got
        state["trash_generation"] = 5
        assert assignment.discarded(state) is None
