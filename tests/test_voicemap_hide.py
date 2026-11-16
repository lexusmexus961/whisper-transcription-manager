"""Really hide entire groups — and thereby pack closer together.

Three requirements for the same view:

    - groups below a minimum size (say, 15 nodes) leave the view entirely
    - the nodes pack together more closely
    - selecting some groups hides everything else completely

**Hiding is different from filtering.** Filters are a VIEW: non-matches are
dampened, not removed, because otherwise "who is that?" would get a different
answer depending on perspective. That stays. Here it is about the SECTION you
want to look at — and a section may really be empty.

The difference has consequences, and they are the heart of these tests:
**Hidden nodes no longer participate in the force play.** With that, the first
requirement answers the second right along — hundreds of tiny groups that fill
the edge otherwise press the large ones together.

An invisible node that still pushes or catches clicks would be the worst state
of all. That is exactly what these tests aim for.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
HARNESS = Path(__file__).resolve().parent / "js" / "voicemap_harness.js"
MAP_JS = REPO / "whisper_tm" / "static" / "js" / "voicemap_graph.js"
PANEL_JS = MAP_JS.parent / "voicemap_panel.js"

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node not available")


def js(body: str) -> dict:
    source = (
        f"const {{API, paper}} = require({str(HARNESS)!r});\n"
        "const output = (o) => console.log('§§' + JSON.stringify(o));\n"
        f"{body}\n"
    )
    done = subprocess.run(
        ["node", "-e", source],
        capture_output=True,
        text=True,
        timeout=60,
        env={"MAP_JS": str(MAP_JS), "PANEL_JS": str(PANEL_JS), "PATH": os.environ.get("PATH", "")},
    )
    if done.returncode != 0:
        raise AssertionError(f"node failed:\n{done.stderr}")
    lines = [z for z in done.stdout.splitlines() if z.startswith("§§")]
    assert lines, f"no result:\n{done.stdout}\n{done.stderr}"
    return json.loads(lines[-1][2:])


# Three groups: one large (20), one medium (8), one tiny (2),
# plus four unbound ones. A typical distribution in miniature.
STRUCTURE = """
const x = [], y = [], gv = [], members = [];
const setIt = (nr, n, mx) => {
    for (let i = 0; i < n; i++) {
        x.push(mx + (i % 5) * 6); y.push(Math.floor(i / 5) * 6); gv.push(nr);
        members.push({id: members.length + 1, display: `g${nr}_${i}.mp4`, person: '',
                  label: 'S' + i});
    }
};
setIt(1, 20, 0);
setIt(2, 8, 400);
setIt(3, 2, 800);
setIt(0, 4, 1200);
API.mapSet({n: x.length, x: Float32Array.from(x), y: Float32Array.from(y),
                 sx: 1, ox: 0, oy: 0, layer: 'graph', members: members,
                 name: members.map(() => '')});
API.GRAPH.groupOf = Int32Array.from(gv);
API.GRAPH.groups = [{nr: 1, n: 20, name: 'Bob'}, {nr: 2, n: 8, name: ''},
                     {nr: 3, n: 2, name: ''}];
API.GRAPH.showOutlines = true;
API.GRAPH.showNames = true;
const countActive = () => {
    let n = 0;
    for (let i = 0; i < API.GRAPH.active.length; i++) n += API.GRAPH.active[i];
    return n;
};
"""


class TestMinimumSize:
    def test_without_a_limit_everything_is_included(self):
        r = js(
            STRUCTURE
            + """
        API.GRAPH.minGroup = 1; API.graphBuildActive();
        output({active: countActive(), total: 34});
        """
        )
        assert r["active"] == 34

    def test_small_groups_drop_out_entirely(self):
        """With a minimum of 15, group 2 (8) and 3 (2) must go, the unbound
        ones as well."""
        r = js(
            STRUCTURE
            + """
        API.GRAPH.minGroup = 15; API.graphBuildActive();
        output({active: countActive()});
        """
        )
        assert r["active"] == 20, "only the group of 20 may remain"

    def test_the_unbound_ones_disappear_from_two_on(self):
        """Group 0 is the voices without a binding — in a large library the
        bulk of the edge noise. By definition they have size 1."""
        r = js(
            STRUCTURE
            + """
        API.GRAPH.minGroup = 2; API.graphBuildActive();
        output({active: countActive()});
        """
        )
        assert r["active"] == 30  # 20 + 8 + 2, without the four unbound ones


class TestSelection:
    def test_only_the_selected_ones_remain(self):
        r = js(
            STRUCTURE
            + """
        API.GRAPH.selection = new Set([2]); API.graphBuildActive();
        output({active: countActive()});
        """
        )
        assert r["active"] == 8

    def test_several_groups_work(self):
        r = js(
            STRUCTURE
            + """
        API.GRAPH.selection = new Set([1, 3]); API.graphBuildActive();
        output({active: countActive()});
        """
        )
        assert r["active"] == 22

    def test_an_empty_selection_means_all(self):
        """A picture without a single point is not a state you want to end
        up in without having meant to."""
        r = js(
            STRUCTURE
            + """
        API.GRAPH.selection = null; API.graphBuildActive();
        output({active: countActive()});
        """
        )
        assert r["active"] == 34

    def test_selection_and_minimum_size_work_together(self):
        r = js(
            STRUCTURE
            + """
        API.GRAPH.selection = new Set([1, 3]);
        API.GRAPH.minGroup = 15;
        API.graphBuildActive();
        output({active: countActive()});
        """
        )
        assert r["active"] == 20, "group 3 is selected, but too small"


class TestReallyOutside:
    """The core: not just invisible, but not participating at all."""

    def test_hidden_nodes_do_not_move(self):
        r = js(
            STRUCTURE
            + """
        API.GRAPH.minGroup = 15; API.graphBuildActive();
        API.GRAPH.edges = Int32Array.from([]);
        API.GRAPH.kWeight = Float32Array.from([]);
        API.GRAPH.vx = new Float32Array(34);
        API.GRAPH.vy = new Float32Array(34);
        API.GRAPH.ticks = 0; API.GRAPH.steps = 100;
        const before = Array.from(API.MAP_X());
        for (let s = 0; s < 20; s++) API.graphStep();
        const after = Array.from(API.MAP_X());
        let moved_active = 0, moved_inactive = 0;
        for (let i = 0; i < 34; i++) {
            const d = Math.abs(after[i] - before[i]);
            if (API.GRAPH.active[i]) { if (d > 0.01) moved_active++; }
            else if (d > 1e-9) moved_inactive++;
        }
        output({moved_active, moved_inactive});
        """
        )
        assert r["moved_inactive"] == 0, "hidden nodes were moved"
        assert r["moved_active"] > 0, "the visible ones do not move at all"

    def test_hidden_edges_are_not_drawn(self):
        r = js(
            STRUCTURE
            + """
        API.GRAPH.edges = Int32Array.from([0, 1, 20, 21, 0, 20]);
        API.GRAPH.kWeight = Float32Array.from([0.9, 0.9, 0.7]);
        API.GRAPH.minGroup = 15; API.graphBuildActive();
        const {c, p} = paper();
        API.graphDrawEdges(c, 1, 0, 0, 2000, 2000);
        // Only edge 0–1 lies entirely within the visible group.
        output({points: p.curves.flat().length});
        """
        )
        assert r["points"] == 2, "an edge into the hidden area was drawn"

    def test_hidden_groups_get_no_outline(self):
        r = js(
            STRUCTURE
            + """
        API.GRAPH.minGroup = 15; API.graphBuildActive();
        const {c, p} = paper();
        API.graphDrawHulls(c, 1, 0, 0, 2000, 2000);
        output({strokes: p.strokes});
        """
        )
        assert r["strokes"] == 1

    def test_hidden_groups_get_no_label(self):
        r = js(
            STRUCTURE
            + """
        API.GRAPH.selection = new Set([2]); API.graphBuildActive();
        const {c, p} = paper();
        API.graphDrawNames(c, 1, 0, 400, 4000, 800);
        output({texts: p.texts.map(t => t.t)});
        """
        )
        assert "Bob" not in r["texts"], "the name of a hidden group"


class TestCompaction:
    def test_a_lower_density_pulls_closer_together(self):
        """Compaction, measured: spread of the visible points after
        the same number of steps, once with full and once with lowered
        repulsion."""
        r = js(
            STRUCTURE
            + """
        const run = (density) => {
            API.mapSet({x: Float32Array.from(x), y: Float32Array.from(y)});
            API.GRAPH.density = density;
            API.GRAPH.drag = 1.0;
            API.GRAPH.minGroup = 1;
            API.graphBuildActive();
            API.GRAPH.vx = new Float32Array(34);
            API.GRAPH.vy = new Float32Array(34);
            API.GRAPH.ticks = 0; API.GRAPH.steps = 200;
            API.GRAPH.edges = Int32Array.from([]);
            API.GRAPH.kWeight = Float32Array.from([]);
            for (let s = 0; s < 120; s++) API.graphStep();
            const X = API.MAP_X(), Y = API.MAP_Y();
            let mx = 0, my = 0;
            for (let i = 0; i < 20; i++) { mx += X[i]; my += Y[i]; }
            mx /= 20; my /= 20;
            let s2 = 0;
            for (let i = 0; i < 20; i++) s2 += (X[i]-mx)**2 + (Y[i]-my)**2;
            return Math.sqrt(s2 / 20);
        };
        output({full: run(1.0), dense: run(0.2)});
        """
        )
        assert r["dense"] < r["full"], (
            f"densification has no effect: {r['dense']:.1f} vs {r['full']:.1f}"
        )

    def test_the_slider_goes_up_to_25(self):
        """Previously capped at 10, which was too weak a pull."""
        source = MAP_JS.read_text("utf-8")
        assert "const DRAG_MAX = 25.0;" in source


class TestWiringUp:
    def test_every_new_button_has_a_function(self):
        html = (REPO / "whisper_tm" / "static" / "index.html").read_text("utf-8")
        source = MAP_JS.read_text("utf-8")
        for f in ("graphMinGroupSet", "graphDensitySet"):
            assert f in html, f"{f} missing from the markup"
            assert f"function {f}" in source
        for f in (
            "graphToggleSelection",
            "graphOnlyThisGroup",
            "graphClearSelection",
            "graphUnhide",
        ):
            assert f"function {f}" in source
            assert f"{f}(" in source, f"{f} is never called"

    def test_the_hide_state_is_in_the_markup(self):
        html = (REPO / "whisper_tm" / "static" / "index.html").read_text("utf-8")
        assert 'id="mapHideState"' in html

    def test_hit_testing_and_fitting_know_the_criterion(self):
        """An invisible point that still catches clicks would be the worst
        of the three errors — and fitting the view around invisible points
        would leave the visible ones as crumbs in the middle."""
        k = (REPO / "whisper_tm" / "static" / "js" / "voicemap.js").read_text("utf-8")
        for fn, length in (
            ("function mapBuildGrid", 1500),
            ("function mapFit", 1500),
            ("function mapDraw", 4500),
        ):
            place = k.index(fn)
            body = k[place : place + length]
            assert "mapVisible(" in body, fn

    def test_the_selection_does_not_survive_a_threshold_change(self):
        """Group numbers shift with the threshold — "only group 29" then
        showed a different group."""
        source = MAP_JS.read_text("utf-8")
        place = source.index("async function mapLoadGraph")
        body = source[place : source.index("graphInitialLayout()", place)]
        assert "GRAPH.selection = null" in body


# ── Same names into one ball ──────────────────────────────────────────
#
#  Separate groups that carry the same name should come together into
#  one ball.
#
#  In practice one name is often spread across many groups (one large
#  group plus several small splinters). At threshold 0.65 these parts are
#  not mathematically connected.
#
#  The group is a claim made by the machine, the assigned name is a
#  statement made by a human. Whoever has called several groups "me" has
#  said they are one person — keeping them apart in the view puts the
#  calculation above the statement.

BALL = """
const x = [], y = [], gv = [], members = [];
const setIt = (nr, n, mx, my) => {
    for (let i = 0; i < n; i++) {
        x.push(mx + (i % 4) * 5); y.push(my + Math.floor(i / 4) * 5); gv.push(nr);
        members.push({id: members.length + 1, display: `g${nr}.mp4`, person: '', label: 'S'});
    }
};
setIt(1, 20, 0, 0);          // "me", large
setIt(2, 6, 900, 0);         // "me", splinter far away
setIt(3, 5, 0, 900);         // "me", second splinter
setIt(4, 18, 900, 900);      // "Alice", separate person
API.mapSet({n: x.length, x: Float32Array.from(x), y: Float32Array.from(y),
                 sx: 1, ox: 0, oy: 0, layer: 'graph', members: members,
                 name: members.map(() => '')});
API.GRAPH.groupOf = Int32Array.from(gv);
API.GRAPH.groups = [{nr: 1, n: 20, name: 'me'}, {nr: 4, n: 18, name: 'Alice'},
                     {nr: 2, n: 6, name: 'me'}, {nr: 3, n: 5, name: 'me'}];
API.GRAPH.showNames = true;
API.GRAPH.showOutlines = true;
const distance = () => {
    // Distance between the two "me" centroids (group 1 vs. group 2).
    const X = API.MAP_X(), Y = API.MAP_Y();
    let ax=0, ay=0, bx=0, by=0;
    for (let i = 0; i < 20; i++) { ax += X[i]; ay += Y[i]; }
    for (let i = 20; i < 26; i++) { bx += X[i]; by += Y[i]; }
    ax/=20; ay/=20; bx/=6; by/=6;
    return Math.round(Math.hypot(ax-bx, ay-by));
};
const runLayout = (cluster) => {
    API.mapSet({x: Float32Array.from(x), y: Float32Array.from(y)});
    API.GRAPH.clusterNames = cluster;
    API.GRAPH.density = 1.0; API.GRAPH.drag = 1.0; API.GRAPH.minGroup = 1;
    API.graphBuildActive();
    API.GRAPH.edges = Int32Array.from([]);
    API.GRAPH.kWeight = Float32Array.from([]);
    API.GRAPH.vx = new Float32Array(49); API.GRAPH.vy = new Float32Array(49);
    API.GRAPH.ticks = 0; API.GRAPH.steps = 400;
    for (let s = 0; s < 250; s++) API.graphStep();
    return distance();
};
"""


class TestNameClusters:
    def test_identical_names_move_together(self):
        """The behaviour, measured: distance between the two
        "me" centroids after the same number of steps.

        They do NOT become a single point, and that is not a weakness of
        the implementation: 31 nodes cannot occupy the same spot. The
        repulsion between the points sets a natural packing density, and
        a "ball" of 31 voices simply has a radius. Measured at pull 1 and
        density 1: 549 → 318, so roughly halved; with pull 3 and density
        0.25 it goes down to 140. What belongs together then lies
        adjacent and carries ONE outline — physically nothing more is
        possible.
        """
        r = js(BALL + "output({separate: runLayout(false), clustered: runLayout(true)});")
        assert r["clustered"] < r["separate"] * 0.6, (
            f"barely moved together: {r['clustered']} vs {r['separate']}"
        )

    def test_one_outline_per_person_instead_of_per_group(self):
        r = js(
            BALL
            + """
        API.GRAPH.clusterNames = true; API.graphBuildActive();
        const a = paper(); API.graphDrawHulls(a.c, 1, 0, 0, 4000, 4000);
        API.GRAPH.clusterNames = false; API.graphBuildActive();
        const b = paper(); API.graphDrawHulls(b.c, 1, 0, 0, 4000, 4000);
        output({clustered: a.p.strokes, separate: b.p.strokes});
        """
        )
        assert r["clustered"] == 2, "one line per person: me and Alice"
        assert r["separate"] == 4, "one line per group"

    def test_the_name_appears_once_in_the_centre_of_the_cluster(self):
        """All parts share one centroid — without bundling three "me"
        would sit exactly on top of each other."""
        r = js(
            BALL
            + """
        API.GRAPH.clusterNames = true; API.graphBuildActive();
        const {c, p} = paper();
        API.graphDrawNames(c, 1.5, 0, 0, 4000, 4000);
        output({texts: p.texts.map(t => t.t)});
        """
        )
        assert r["texts"].count("me") == 1, r["texts"]

    def test_the_minimum_size_judges_the_person(self):
        """Otherwise, at "15 and up" the small "me" parts would disappear,
        even though "me" with 31 voices is the largest person — the ball
        would lose its satellites and nobody would know why."""
        r = js(
            BALL
            + """
        API.GRAPH.clusterNames = true;
        API.GRAPH.minGroup = 15; API.graphBuildActive();
        let n = 0;
        for (let i = 0; i < API.GRAPH.active.length; i++) n += API.GRAPH.active[i];
        output({active: n});
        """
        )
        assert r["active"] == 49, "all 31 me voices plus 18 Alice"

    def test_without_clustering_the_group_counts_again(self):
        r = js(
            BALL
            + """
        API.GRAPH.clusterNames = false;
        API.GRAPH.minGroup = 15; API.graphBuildActive();
        let n = 0;
        for (let i = 0; i < API.GRAPH.active.length; i++) n += API.GRAPH.active[i];
        output({active: n});
        """
        )
        assert r["active"] == 38, "only the group of 20 and the group of 18"

    def test_different_names_stay_separate(self):
        """Bundling must not throw everything together."""
        r = js(
            BALL
            + """
        API.GRAPH.clusterNames = true;
        output({me1: API.graphKey(1), me2: API.graphKey(2),
                 alice: API.graphKey(4)});
        """
        )
        assert r["me1"] == r["me2"]
        assert r["alice"] != r["me1"]

    def test_upper_and_lower_case_form_the_same_cluster(self):
        r = js(
            BALL
            + """
        API.GRAPH.groups = [{nr: 1, n: 20, name: 'Me'}, {nr: 2, n: 6, name: 'me'}];
        API.GRAPH.clusterNames = true;
        output({a: API.graphKey(1), b: API.graphKey(2)});
        """
        )
        assert r["a"] == r["b"]

    def test_the_switch_is_wired(self):
        html = (REPO / "whisper_tm" / "static" / "index.html").read_text("utf-8")
        assert 'id="mapCluster"' in html
        assert "graphToggleNameClusters(this.checked)" in html
        source = MAP_JS.read_text("utf-8")
        assert "function graphToggleNameClusters" in source
        assert "cl.checked = GRAPH.clusterNames" in source


# ── The sequence player ────────────────────────────────────────────────────
#
#  The playback menu in the map plays all voices of a group one after
#  another.
#
#  Before, there was a start button and nothing else: no pause, no
#  going back, no visible status — and the map jumped somewhere else
#  with every clip, which is the opposite of helpful while listening.

PLAYER = """
const x = [], y = [], gv = [], members = [];
const files = ['a.mp4','a.mp4','a.mp4','b.mp4','b.mp4','c.mp4'];
for (let i = 0; i < 6; i++) {
    x.push(i * 10); y.push(0); gv.push(1);
    members.push({id: i + 1, display: files[i], person: '', label: 'S' + i});
}
x.push(500); y.push(500); gv.push(2); members.push({id: 99, display: 'z.mp4'});
API.mapSet({n: 7, x: Float32Array.from(x), y: Float32Array.from(y),
                 sx: 1, ox: 0, oy: 0, layer: 'graph', members: members,
                 name: members.map(() => '')});
API.GRAPH.groupOf = Int32Array.from(gv);
API.GRAPH.groups = [{nr: 1, n: 6, name: 'Bob'}, {nr: 2, n: 1, name: ''}];
API.GRAPH.minGroup = 1; API.graphBuildActive();
"""


class TestQueuePlayer:
    def test_one_per_recording_bundles_the_windows(self):
        """The right thing for a quick overview across 40 recordings."""
        r = js(PLAYER + "output({row: API.graphBuildRow(1, true)});")
        assert r["row"] == [0, 3, 5], "first window per file"

    def test_all_windows_really_plays_all_of_them(self):
        """It is precisely the MULTIPLE windows of a recording that one
        wants to compare — previously that wasn't possible at all."""
        r = js(PLAYER + "output({row: API.graphBuildRow(1, false)});")
        assert r["row"] == [0, 1, 2, 3, 4, 5]

    def test_hidden_voices_do_not_enter_the_queue(self):
        r = js(
            PLAYER
            + """
        API.GRAPH.selection = new Set([2]); API.graphBuildActive();
        output({row: API.graphBuildRow(1, false)});
        """
        )
        assert r["row"] == []

    def test_a_foreign_group_stays_out(self):
        r = js(PLAYER + "output({row: API.graphBuildRow(2, false)});")
        assert r["row"] == [6]

    def test_the_position_points_at_the_playing_piece(self):
        """Previously `pos` already pointed at the NEXT one (pos++ at
        start). That made "back" impossible to build: you would have had
        to subtract two to go back by one."""
        source = PANEL_JS.read_text("utf-8")
        place = source.index("function graphRowPlay")
        body = source[place : place + 1200]
        assert "r.listing[r.pos]" in body
        assert "r.pos++" not in body, "the position is incremented while playing"

    def test_the_map_does_not_jump_on_every_piece(self):
        """`graphJumpTo` always centers. While listening through a
        group, the view would then move with every clip."""
        source = PANEL_JS.read_text("utf-8")
        place = source.index("function graphRowPlay")
        body = source[place : place + 1200]
        assert "graphJumpTo(" not in body
        assert "graphRowIntoView(" in body

    def test_every_button_is_wired(self):
        source = PANEL_JS.read_text("utf-8")
        for f in (
            "graphRowNext",
            "graphRowPause",
            "graphRowStop",
            "panelTransportState",
            "graphRowIntoView",
            "graphBuildRow",
        ):
            assert f"function {f}" in source, f
            assert f"{f}(" in source
        assert 'id="mapRowState"' in source

    def test_a_new_graph_stops_the_playback(self):
        """The indices afterward point to different voices."""
        source = MAP_JS.read_text("utf-8")
        place = source.index("async function mapLoadGraph")
        assert "graphCancelRowIfNeeded()" in source[place : place + 400]


class TestCountUnderTheName:
    def test_the_number_sits_under_the_name(self):
        """The number of matches is written under each group's name — until
        now it only appeared in the legend, and that shows only ten groups."""
        r = js(
            STRUCTURE
            + """
        API.GRAPH.clusterNames = false;
        const {c, p} = paper();
        API.graphDrawNames(c, 1.5, 400, 300, 4000, 800);
        output({texts: p.texts.map(t => ({t: t.t, y: Math.round(t.y)}))});
        """
        )
        names = [z for z in r["texts"] if not z["t"].isdigit()]
        numbers = [z for z in r["texts"] if z["t"].isdigit()]
        assert numbers, "no count drawn"
        # Every number sits BELOW its name.
        for z in numbers:
            assert any(n["y"] < z["y"] for n in names), z

    def test_what_is_visible_is_what_is_counted(self):
        """Not the number from the group list: if parts are hidden or
        bundled, it would differ from what you can actually count."""
        r = js(
            BALL
            + """
        API.GRAPH.clusterNames = true; API.graphBuildActive();
        const {c, p} = paper();
        API.graphDrawNames(c, 1.5, 100, 100, 4000, 4000);
        output({numbers: p.texts.filter(t => /^\\d+$/.test(t.t)).map(t => t.t)});
        """
        )
        # me = 20 + 6 + 5 = 31 (bundled), Alice = 18.
        assert "31" in r["numbers"], r["numbers"]
        assert "18" in r["numbers"], r["numbers"]


class TestPreviewDuration:
    """A preview limit plays only the first 1, 2, 3, 5 or 10 seconds of
    each clip — listening through 36 windows of 30 s each would take 18
    minutes; with three seconds it's two."""

    def _js(self) -> str:
        return PANEL_JS.read_text("utf-8")

    def _tick(self) -> str:
        """The body of the `timeupdate` listener — that's where the limit lives."""
        q = self._js()
        place = q.index("audioEl.addEventListener('timeupdate'")
        return q[place : q.index("\n    });", place)]

    def test_the_limit_hangs_on_the_playback_progress(self):
        """Via `timeupdate`, NOT via a timer: a `setTimeout` would keep
        running while paused and fire too early while buffering."""
        body = self._tick()
        assert "audioEl.currentTime >= GRAPH.playbackSec" in body
        assert "setTimeout(" not in body, "timer instead of playback progress"

    def test_zero_means_all_the_way_through(self):
        """Otherwise playback would stall immediately at "full"."""
        assert "!GRAPH.playbackSec) return" in self._tick()

    def test_pause_also_halts_the_preview_duration(self):
        """`timeupdate` does not fire while paused — and the extra check
        on `running` catches the case where an event is still queued."""
        assert "!r.running" in self._tick()

    def test_the_limit_applies_only_to_the_queue(self):
        """Previously `ontimeupdate` was attached to the
        element and stayed there even after the sequence had long since
        ended: someone who wanted to listen to a neighbor got three
        seconds — and then the next clip of a sequence they had never
        started."""
        assert "PANEL.owner !== 'row'" in self._tick()

    def test_stopping_releases_the_playback(self):
        """The successor to "stopping releases the time watchdog".

        The listeners are now permanently attached to the audio element
        and are no longer reattached — which is exactly why stopping must
        clear the OWNER. If it stayed set to 'row', the playback limit
        would keep cutting off every single listening sample.
        """
        q = self._js()
        place = q.index("function graphRowStop")
        body = q[place : place + 400]
        assert "PANEL.row = null" in body
        assert "PANEL.owner = null" in body

    def test_the_selection_is_in_the_control_bar(self):
        js_q = self._js()
        assert "graphPlaybackSet" in js_q
        assert "function graphPlaybackSet" in js_q
        # All offered values must be selectable.
        place = js_q.index("PANEL_PREVIEW = [")
        body = js_q[place : js_q.index("]", place)]
        for v in ("0", "1", "2", "3", "5", "10"):
            assert v in body, v

    def test_the_value_is_capped(self):
        """No negative value and nothing beyond the listening sample (30 s)."""
        place = self._js().index("function graphPlaybackSet")
        body = self._js()[place : place + 300]
        assert "Math.max(0" in body and "Math.min(30" in body

    def test_the_value_survives_a_reload(self):
        js_q = MAP_JS.read_text("utf-8")
        assert "playback: GRAPH.playbackSec" in js_q
        assert "m.playback !== undefined" in js_q


class TestPlayOnlyOpenOnes:
    """The unassigned voices of a group can be played in sequence — that
    makes assigning them easier.

    This is the workflow for naming: listen, recognize, type, next.
    Anyone listening through a group to assign it doesn't want to hear
    a voice twenty times that already has a name.
    """

    STRUCTURE = """
    const x = [], y = [], gv = [], members = [];
    const names = ['Bob', '', '', 'Bob', '', ''];
    for (let i = 0; i < 6; i++) {
        x.push(i * 10); y.push(0); gv.push(1);
        members.push({id: i + 1, display: `d${i}.mp4`, person: names[i], label: 'S'});
    }
    API.mapSet({n: 6, x: Float32Array.from(x), y: Float32Array.from(y),
                     sx: 1, ox: 0, oy: 0, layer: 'graph', members: members,
                     name: names});
    API.GRAPH.groupOf = Int32Array.from(gv);
    API.GRAPH.groups = [{nr: 1, n: 6, name: 'Bob'}];
    API.GRAPH.minGroup = 1; API.graphBuildActive();
    """

    def test_only_those_without_an_assignment(self):
        r = js(self.STRUCTURE + "output({r: API.graphBuildRow(1, false, true)});")
        assert r["r"] == [1, 2, 4, 5]

    def test_without_the_restriction_all_of_them(self):
        r = js(self.STRUCTURE + "output({r: API.graphBuildRow(1, false, false)});")
        assert r["r"] == [0, 1, 2, 3, 4, 5]

    def test_together_with_one_per_recording(self):
        """Both restrictions must be combinable."""
        r = js(self.STRUCTURE + "output({r: API.graphBuildRow(1, true, true)});")
        # First OPEN window per file — each file appears once here.
        assert r["r"] == [1, 2, 4, 5]

    def test_a_fully_named_group_yields_no_queue(self):
        r = js("""
        API.mapSet({n: 2, x: Float32Array.from([0, 10]),
                         y: Float32Array.from([0, 0]), sx: 1, ox: 0, oy: 0,
                         layer: 'graph', name: ['Bob', 'Bob'],
                         members: [{id: 1, display: 'a.mp4', person: 'Bob'},
                                      {id: 2, display: 'b.mp4', person: 'Bob'}]});
        API.GRAPH.groupOf = Int32Array.from([1, 1]);
        API.GRAPH.groups = [{nr: 1, n: 2, name: 'Bob'}];
        API.GRAPH.minGroup = 1; API.graphBuildActive();
        output({r: API.graphBuildRow(1, false, true)});
        """)
        assert r["r"] == []

    def test_the_button_appears_only_when_open_ones_exist(self):
        source = PANEL_JS.read_text("utf-8")
        assert "openWindows ?" in source
        assert "open</button>" in source

    def test_after_naming_it_continues(self):
        """Without the jump, with 350 open voices that would be 350 extra
        clicks that decide nothing."""
        k = PANEL_JS.read_text("utf-8")
        place = k.index("async function mapRename")
        body = k[place : k.index("\n}\n", place)]
        assert "graphRowNext(1)" in body

    def test_releasing_does_not_jump(self):
        """Anyone removing an assignment usually wants to listen to the
        voice again right then."""
        k = PANEL_JS.read_text("utf-8")
        place = k.index("if (name && PANEL.row")
        assert "graphRowNext(1)" in k[place : place + 200]


class TestTheDetailsAreDocked:
    """The field used to float over the map — resizable by a handle, but not
    movable, and over voices it could never be moved away from — floating
    panels that cannot be moved get in the way. Now the details sit BESIDE the map: a column with a splitter
    to drag, and a button that hides it."""

    CSS = (REPO / "whisper_tm" / "static" / "css" / "voices.css").read_text("utf-8")
    HTML = (REPO / "whisper_tm" / "static" / "index.html").read_text("utf-8")
    VOICES = (REPO / "whisper_tm" / "static" / "js" / "voices.js").read_text("utf-8")

    def test_beside_the_map_not_over_it(self):
        h = self.HTML
        body = h[h.index('id="mapBody"') :]
        assert (
            body.index('class="map-stage"')
            < body.index('id="mapSplitter"')
            < body.index('id="mapSide"')
        )
        side = self.CSS[self.CSS.index("\n.vx-side {") :]
        side = side[: side.index("}")]
        assert "position:absolute" not in side
        assert "width:var(--vx-side, 460px)" in side

    def test_the_width_can_be_dragged_and_is_remembered(self):
        body = self.VOICES[self.VOICES.index("function vxSplitterWire(") :]
        body = body[: body.index("\n}\n")]
        assert "'mousedown'" in body and "'mousemove'" in body
        assert "--vx-side" in body and "vxSideSave({width" in body

    def test_it_can_be_hidden(self):
        assert "function vxSideToggle" in self.VOICES
        assert ".vx-mapbody.side-closed .vx-side" in self.CSS
        assert "vxSideToggle()" in self.HTML
