"""Auto-search: At which threshold is THIS person's group largest?

The search scans a person's group across all thresholds and reports the
point where it is largest — some people only form a sizeable group within
a narrow threshold window.

That's the core: **The threshold is not a global quality knob.** The best
threshold differs from person to person, and a step of 0.01 can lie
between "doesn't exist" and a dozen voices. By hand you only find that by
trying.

**Two results, not one.** "Largest" is literally what's asked, but
alone it's misleading: a large group can keep growing at low thresholds
while its purity drops to 0.40, because the group then consists 60% of
others. Therefore, the most balanced threshold (F₁ from purity and
coverage) stands equally alongside.

**What "their group" means.** The same rule as everywhere on the map: A
group's name is the most frequently assigned name in it. Without this rule,
the answer would always be the lowest threshold where everything is a lump."""

import numpy as np
import pytest
from flask import Flask

from whisper_tm.core import speaker as speaker_core
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

    from whisper_tm.api.speaker import speaker_bp

    a = Flask(__name__)
    a.register_blueprint(speaker_bp)
    a.db = db
    a.media = media
    yield a
    inventory.discard()


def _person(app, path, middle_seed, n, spread, person=None, ab=0):
    """A cloud around its own centre, optionally named."""
    basis = _vec(middle_seed)
    app.db.write(
        str(app.media / path),
        speaker_core.MODEL_BRAND,
        [
            {
                "label": "SPEAKER_00",
                "windows": ab + i,
                "start_sec": (ab + i) * 300.0,
                "speech_sec": 30.0,
                "segments": 2,
                "v_separated": basis + spread * _vec(9000 + middle_seed * 50 + i),
            }
            for i in range(n)
        ],
    )
    ids = [z["id"] for z in app.db.windows(str(app.media / path))]
    if person:
        for vid in ids:
            app.db.assign(vid, person, source="test")
    inventory.discard()
    return ids


def _search(app, person, **q):
    p = "&".join(f"{k}={v}" for k, v in q.items())
    return (
        app.test_client()
        .get(f"/api/voicemap/threshold-search?person={person}" + ("&" + p if p else ""))
        .get_json()
    )


class TestBaseCase:
    def test_a_clearly_separated_person_is_found(self, app):
        """Two tight clouds, far apart. One of them is called Carol."""
        _person(app, "a.mkv", 1, 8, 0.02, "Carol")
        _person(app, "b.mkv", 2, 8, 0.02, "Bob")
        d = _search(app, "Carol", step=0.05)

        assert d["total"] == 8
        assert d["largest"], "no threshold found"
        g = d["largest"]
        assert g["n"] == 8, f"Carol's group is only {g['n']} voices large"
        assert g["purity"] == 1.0, "foreign voices in the group"
        assert g["coverage"] == 1.0

    def test_the_curve_comes_back_complete(self, app):
        """It's the actual find — you SEE how narrow the window is in
        which a person exists."""
        _person(app, "a.mkv", 1, 6, 0.02, "Carol")
        d = _search(app, "Carol", from_value=0.5, to_value=0.8, step=0.05)
        thresholds = [p["threshold"] for p in d["points"]]
        assert thresholds == sorted(thresholds), "the points are unsorted"
        assert min(thresholds) >= 0.5 and max(thresholds) <= 0.8
        for p in d["points"]:
            assert {"threshold", "n", "hits", "purity", "coverage", "f1", "component"} <= set(p)

    def test_upper_and_lower_case_do_not_matter(self, app):
        """Same rule as in the person table (COLLATE NOCASE)."""
        _person(app, "a.mkv", 1, 6, 0.02, "Carol")
        a = _search(app, "carol", step=0.1)
        b = _search(app, "CAROL", step=0.1)
        assert a["total"] == b["total"] == 6

    def test_without_a_person_a_clear_error(self, app):
        r = app.test_client().get("/api/voicemap/threshold-search")
        assert r.status_code == 400
        assert "person" in r.get_json()["error"]

    def test_an_unknown_person_is_explained(self, app):
        """Not "no results", but why: the search needs a named
        anchor."""
        _person(app, "a.mkv", 1, 6, 0.02, "Bob")
        d = _search(app, "Nobody", step=0.1)
        assert d["total"] == 0
        assert d["points"] == []
        assert "anchor" in d["hint"]

    def test_an_empty_inventory_does_not_crash(self, app):
        d = _search(app, "Carol")
        assert d["points"] == []
        assert d["hint"]


class TestTheRuleWhichGroupCounts:
    """A group's name is the most frequently assigned name in it."""

    def test_a_group_dominated_by_others_does_not_count(self, app):
        """Without this rule, the answer would always be the lowest
        threshold where everything is a lump — and the "found group" would
        in reality be called something else."""
        # A large cloud "Dave" and two voices "Carol" right in the middle of it.
        _person(app, "large.mkv", 5, 20, 0.02, "Dave")
        _person(app, "small.mkv", 5, 2, 0.02, "Carol")
        d = _search(app, "Carol", from_value=0.1, to_value=0.4, step=0.1)
        # At such low thresholds everything is ONE group — and it's called
        # "Dave", not "Carol".
        for p in d["points"]:
            assert p["n"] <= 2, (
                f"at threshold {p['threshold']} a group with {p['n']} "
                f"voices counts as the Carol group, even though Dave dominates it"
            )

    def test_purity_and_coverage_are_reported(self, app):
        """A group of 400 voices, of which 30 are the person being
        searched for, is not a find but a warning."""
        _person(app, "a.mkv", 1, 10, 0.02, "Carol")
        _person(app, "b.mkv", 1, 10, 0.02, "Bob", ab=100)  # same cloud!
        d = _search(app, "Carol", step=0.05)
        with_ = [p for p in d["points"] if p["n"] > 0]
        assert with_, "found nothing at all"
        for p in with_:
            assert 0.0 <= p["purity"] <= 1.0
            assert 0.0 <= p["coverage"] <= 1.0
            assert p["hits"] <= p["n"]
            assert p["hits"] <= d["total"]

    def test_both_recommendations_stand_side_by_side(self, app):
        _person(app, "a.mkv", 1, 8, 0.02, "Carol")
        _person(app, "b.mkv", 2, 8, 0.02, "Bob")
        d = _search(app, "Carol", step=0.05)
        assert d["largest"] and d["best"]
        # "largest" (largest) is maximal by n, "best" (best) by f1.
        assert d["largest"]["n"] == max(p["n"] for p in d["points"])
        assert d["best"]["f1"] == max(p["f1"] for p in d["points"])


class TestTwoStage:
    """Without a two-stage approach, the search took tens of seconds
    instead of a few — at low thresholds even a small person hangs
    in the giant component (a dozen voices can sit at 0.40 in a lump of
    several thousand)."""

    def test_without_a_step_the_search_runs_in_two_stages(self, app):
        _person(app, "a.mkv", 1, 8, 0.02, "Carol")
        d = _search(app, "Carol")
        assert d["two_stage"] is True
        coarse = [p for p in d["points"] if p["coarse"]]
        assert coarse, "no coarse grid in the response"
        # The coarse grid has 0.05 spacing …
        distance = round(coarse[1]["threshold"] - coarse[0]["threshold"], 4)
        assert distance == 0.05
        # … and finer points lie in between.
        assert len(d["points"]) > len(coarse)

    def test_with_a_step_exactly_that_grid_is_computed(self, app):
        _person(app, "a.mkv", 1, 8, 0.02, "Carol")
        d = _search(app, "Carol", from_value=0.5, to_value=0.7, step=0.05)
        assert d["two_stage"] is False
        assert [p["threshold"] for p in d["points"]] == [0.5, 0.55, 0.6, 0.65, 0.7]

    def test_the_fine_grid_lies_around_the_maximum(self, app):
        _person(app, "a.mkv", 1, 8, 0.02, "Carol")
        _person(app, "b.mkv", 2, 8, 0.02, "Bob")
        d = _search(app, "Carol")
        peak = d["largest"]["threshold"]
        near = [p["threshold"] for p in d["points"] if abs(p["threshold"] - peak) <= 0.05]
        # Around the peak the resolution must be fine (0.01 steps).
        assert len(near) >= 6, f"only {len(near)} points around {peak}"

    def test_the_two_stage_search_finds_the_same_maximum(self, app):
        """The price of the two-stage approach must not be the result."""
        _person(app, "a.mkv", 1, 8, 0.02, "Carol")
        _person(app, "b.mkv", 2, 8, 0.02, "Bob")
        auto = _search(app, "Carol")
        full = _search(app, "Carol", from_value=0.4, to_value=0.95, step=0.01)
        assert auto["largest"]["n"] == full["largest"]["n"]


class TestBounds:
    def test_the_range_is_capped(self, app):
        """A single call must not tie up the server for minutes."""
        _person(app, "a.mkv", 1, 6, 0.02, "Carol")
        d = _search(app, "Carol", from_value=0.05, to_value=0.99, step=0.005)
        assert len(d["points"]) <= 200

    def test_swapped_bounds_are_swapped_back(self, app):
        _person(app, "a.mkv", 1, 6, 0.02, "Carol")
        d = _search(app, "Carol", from_value=0.8, to_value=0.6, step=0.05)
        s = [p["threshold"] for p in d["points"]]
        assert s == sorted(s) and min(s) >= 0.6 and max(s) <= 0.8

    def test_the_neighbourhood_is_computed_only_once(self):
        """It does not depend on the threshold. Computing it 51 times
        would mean 51 seconds for a result that was already available
        after the first one."""
        from pathlib import Path

        q = (
            Path(__file__).resolve().parent.parent / "whisper_tm" / "api" / "speaker.py"
        ).read_text(encoding="utf-8")
        i = q.index("def map_threshold_search(")
        j = q.index("@speaker_bp.route", i + 10)
        body = q[i:j]
        assert body.count("speaker_knn.held_neighbours(") == 1
        assert "speaker_knn.neighbours(" not in body
        # This used to read `"neighbourhood" not in body or "_finer" in body`
        # — and because `_finer` IS in the body, the assertion was ALWAYS true.
        # A test that can't check anything is worse than none: it looks
        # like protection. Now the thing itself is actually checked.
        assert "speaker_knn.components(" in body, (
            "the search does not decompose itself, but lets group_voices compute it"
        )
        assert body.index("speaker_knn.held_neighbours(") < body.index("def measure("), (
            "the neighbourhood is computed INSIDE the loop"
        )

    def test_only_the_affected_component_is_refined(self):
        """Fine-slicing the whole inventory costs most of a second per threshold."""
        from pathlib import Path

        q = (
            Path(__file__).resolve().parent.parent / "whisper_tm" / "api" / "speaker.py"
        ).read_text(encoding="utf-8")
        i = q.index("def map_threshold_search(")
        j = q.index("@speaker_bp.route", i + 10)
        body = q[i:j]
        assert "affected = np.unique(comp[is_wanted])" in body
        assert "speaker_knn.group_voices(" not in body, (
            "the search calls the full run — then the decomposition is useless"
        )


class TestTheCoreTakesAPrecomputedNeighbourhood:
    def test_grouping_accepts_it(self):
        import numpy as np

        from whisper_tm.core import speaker_knn as sk

        rng = np.random.default_rng(3)
        X = np.vstack([rng.normal(size=(8, 64)) + 5 * i for i in range(3)])
        X = X.astype(np.float32)
        Xn = sk._nrm(X)
        idx, value, _ = sk.neighbours(Xn, k=5, device="cpu")
        without = sk.group_voices(X, 0.6, k=5, device="cpu")["markers"]
        with_ = sk.group_voices(X, 0.6, k=5, device="cpu", neighbourhood=(idx, value))["markers"]
        assert list(without) == list(with_), "the precomputed neighbourhood changes the result"


class TestOneNeighbourhoodForBoth:
    """`map_graph` used to compute kNN TWICE.

    Once in `group_voices` with K=30, then again with k=5 for the
    edges — a full second per map load at several thousand voices,
    for the same computation. The `neighbourhood` parameter had existed
    since the auto-search, but was not used here.
    """

    def test_the_first_k_columns_are_the_k_strongest(self):
        """The basis for the merging. `torch.topk` returns results sorted
        in descending order; if that weren't the case, the map would show
        different edges than before — weaker ones, and nobody would have
        noticed."""
        import numpy as np

        from whisper_tm.core import speaker_knn as sk

        rng = np.random.default_rng(7)
        X = sk._nrm(rng.normal(size=(200, 64)).astype(np.float32))
        i30, w30, _ = sk.neighbours(X, k=30, device="cpu")
        i5, w5, _ = sk.neighbours(X, k=5, device="cpu")
        assert np.allclose(w30[:, :5], w5), "the weights deviate"
        assert (i30[:, :5] == i5).all(), "the neighbours deviate"
        assert (np.diff(w30, axis=1) <= 1e-6).all(), "not sorted descending"

    def test_the_map_computes_the_neighbourhood_only_once(self):
        # The map is built in core/speaker_graph.py; the
        # endpoint itself no longer touches the neighbourhood.
        from pathlib import Path

        root = Path(__file__).resolve().parent.parent / "whisper_tm"
        q = (root / "api" / "speaker.py").read_text(encoding="utf-8")
        i = q.index("def map_graph(")
        endpoint = q[i : q.index("@speaker_bp.route", i + 10)]
        assert "neighbours(" not in endpoint
        g = (root / "core" / "speaker_graph.py").read_text(encoding="utf-8")
        body = g[g.index("def cut(") : g.index("def edges(")]
        assert body.count("speaker_knn.held_neighbours(") == 1, "kNN runs twice per map load again"
        assert "speaker_knn.neighbours(" not in g
        assert "neighbourhood=(full_idx, full_value)" in body
        assert "full_idx[:, :k]" in body


class TestWiringUp:
    def test_the_button_calls_the_function(self):
        from pathlib import Path

        static = Path(__file__).resolve().parent.parent / "whisper_tm" / "static"
        html = (static / "index.html").read_text(encoding="utf-8")
        js = (static / "js" / "voicemap_graph.js").read_text(encoding="utf-8")
        assert "graphThresholdSearch()" in html
        assert "function graphThresholdSearch(" in js
        assert 'id="mapSearchBar"' in html

    def test_a_click_on_the_curve_sets_the_threshold(self):
        """Namely via the same path as the slider — otherwise the display,
        slider, and inventory drift apart."""
        from pathlib import Path

        js = (
            Path(__file__).resolve().parent.parent
            / "whisper_tm"
            / "static"
            / "js"
            / "voicemap_graph.js"
        ).read_text(encoding="utf-8")
        i = js.index("function graphSetThreshold(")
        r = js[i : i + 400]
        assert "graphThresholdSet(" in r
        assert "mapThreshold" in r
        assert "graphSetThreshold(" in js[:i], "the curve doesn't call it"

    def test_the_name_comes_from_the_filter_field_or_the_selection(self):
        """Having to type the name a third time would be exactly the
        friction that keeps a tool from being used."""
        from pathlib import Path

        js = (
            Path(__file__).resolve().parent.parent
            / "whisper_tm"
            / "static"
            / "js"
            / "voicemap_graph.js"
        ).read_text(encoding="utf-8")
        r = js[js.index("function graphSearchName(") :][:600]
        assert "mapFText" in r
        assert "MAP.chosen" in r

    def test_a_superseded_search_no_longer_writes(self):
        """Two clicks in a row: the older result must not overwrite the
        newer one — the same rule as for graph loading (`seq`)."""
        from pathlib import Path

        js = (
            Path(__file__).resolve().parent.parent
            / "whisper_tm"
            / "static"
            / "js"
            / "voicemap_graph.js"
        ).read_text(encoding="utf-8")
        i = js.index("function graphThresholdSearch(")
        r = js[i : i + 1600]
        assert "const mine = ++_graphSearchRun;" in r
        assert r.count("mine !== _graphSearchRun") == 2
