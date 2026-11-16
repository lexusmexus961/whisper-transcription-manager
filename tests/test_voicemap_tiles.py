"""Tiles — same voices without physics and without moving objects.

The client side was laggy with rendering, so the map offers a simple static
view without moving objects, made of boxes.

An earlier measurement asked the wrong question: it optimized the one-time
layout run (and broke it in the process, see ``test_voicemap_draw_load.py``),
while the load comes from continuous operation — hundreds of frames of force
play when opening, and then each drag and zoom again thousands of points, tens
of thousands of lines and hundreds of rounded hulls.

One tile per group (area = count), one square per voice. Measured on a real
inventory, median over 20 frames:

    ==========  ========
    Tiles        4.4 ms
    Graph       10.7 ms
    ==========  ========

For drawing alone, that is factor 2.4. The larger part of the gain is not in
this number: the tiles have no continuous loop and no zoom triggering it.

The honest price: **no edges.** The graph answers "why does this voice lie
there?", the tiles "what is there, how much of it, and how does it sound?".
During assignment, the second is almost always the question — that is why tiles
are preset, and that is why the graph remains reachable.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
HARNESS = Path(__file__).resolve().parent / "js" / "tiles_harness.js"
STATIC = REPO / "whisper_tm" / "static"
TILES_JS = STATIC / "js" / "voicemap_tiles.js"
MAP_JS = STATIC / "js" / "voicemap.js"
GRAPH_JS = STATIC / "js" / "voicemap_graph.js"
INDEX = STATIC / "index.html"
# The map's styles live in the voices workspace's sheet.
CSS = STATIC / "css" / "voices.css"

node = pytest.mark.skipif(shutil.which("node") is None, reason="node not available")


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
        env={"TILES_JS": str(TILES_JS), "PATH": os.environ.get("PATH", "")},
    )
    if done.returncode != 0:
        raise AssertionError(f"node failed:\n{done.stderr}")
    lines = [z for z in done.stdout.splitlines() if z.startswith("§§")]
    assert lines, f"no result:\n{done.stdout}\n{done.stderr}"
    return json.loads(lines[-1][2:])


# Four groups of very different size plus two loners — exactly the
# distribution on which a naive tile layout fails: thousands against 1.
STRUCTURE = """
const gv = [], members = [];
const setIt = (nr, n, name) => {
    for (let i = 0; i < n; i++) {
        gv.push(nr);
        members.push({id: members.length + 1, display: `g${nr}/d${i}.mp4`, person: name,
                  label: 'S', windows: 1, start_sec: 0, speech_sec: 30});
    }
};
setIt(1, 200, 'me');
setIt(2, 60, '');
setIt(3, 20, 'Alice');
setIt(4, 20, 'me');      // second group with the same name
setIt(5, 1, '');
setIt(6, 1, '');
API.mapSet({n: gv.length, members: members, name: members.map(m => m.person)});
API.GRAPH.groupOf = Int32Array.from(gv);
API.GRAPH.groups = [
    {nr: 1, n: 200, name: 'me'}, {nr: 2, n: 60, name: ''},
    {nr: 3, n: 20, name: 'Alice'}, {nr: 4, n: 20, name: 'me'},
    {nr: 5, n: 1, name: ''}, {nr: 6, n: 1, name: ''},
];
API.GRAPH.minGroup = 1;
API.GRAPH.clusterNames = false;
API.graphBuildActive();
API.canvas_el(1200, 800);
"""


@node
class TestLayout:
    def test_every_voice_lies_in_exactly_one_tile(self):
        """The basic contract. A lost voice would be invisible — and that
        is exactly the bug that can devalue an entire inventory."""
        d = js(
            STRUCTURE
            + """
        API.tileLayout(true);
        const seen = new Set();
        let duplicates = 0;
        for (const f of API.TILE.fields)
            for (const i of f.idx) {
                if (seen.has(i)) duplicates++;
                seen.add(i);
            }
        output({tiles: API.TILE.fields.length, voices: seen.size,
                 duplicates, n: 302});
        """
        )
        assert d["voices"] == 302, "not all voices are assigned to a tile"
        assert d["duplicates"] == 0
        assert d["tiles"] == 6

    def test_the_area_follows_the_count(self):
        """This is the whole statement of the image. If it isn't true, the
        tile view is a pretty lie."""
        d = js(
            STRUCTURE
            + """
        API.tileLayout(true);
        const f = {};
        for (const k of API.TILE.fields) f[k.nr] = {a: k.w * k.h, n: k.idx.length};
        output(f);
        """
        )
        large, small = d["1"], d["3"]  # 200 against 20 voices
        ratio = (large["a"] / small["a"]) / (large["n"] / small["n"])
        assert 0.85 < ratio < 1.15, f"area and count diverge (factor {ratio:.2f})"

    def test_the_tiles_stay_reasonably_square(self):
        """What the squarified algorithm is for. Long thin strips are
        neither readable nor clickable — with 200 against 1 the real
        danger."""
        d = js(
            STRUCTURE
            + """
        API.tileLayout(true);
        output({s: API.TILE.fields.filter(f => f.idx.length >= 20)
            .map(f => Math.max(f.w / f.h, f.h / f.w))});
        """
        )
        assert max(d["s"]) < 6, f"strips instead of tiles: {d['s']}"

    def test_nothing_sticks_out_of_the_area(self):
        d = js(
            STRUCTURE
            + """
        API.tileLayout(true);
        let out = 0;
        for (const f of API.TILE.fields) {
            if (f.x < -0.01 || f.y < -0.01) out++;
            if (f.x + f.w > 1200.01 || f.y + f.h > 800.01) out++;
        }
        output({out});
        """
        )
        assert d["out"] == 0

    def test_identical_names_can_become_one_tile(self):
        """The same rule as in the graph: eleven tiles "me" side by side
        would be the same misleading as eleven clusters."""
        d = js(
            STRUCTURE
            + """
        API.GRAPH.clusterNames = true;
        API.tileLayout(true);
        const me = API.TILE.fields.filter(f => f.name === 'me');
        output({tiles: API.TILE.fields.length,
                 meTiles: me.length, meVoices: me[0] ? me[0].idx.length : 0});
        """
        )
        assert d["meTiles"] == 1, "two tiles for the same name"
        assert d["meVoices"] == 220, "the second me group is missing from the cluster"
        assert d["tiles"] == 5

    def test_hidden_groups_do_not_appear_at_all(self):
        """`graphActive` applies here just the same — otherwise the tile view
        would show something different from the graph under the same
        settings."""
        d = js(
            STRUCTURE
            + """
        API.GRAPH.minGroup = 20; API.graphBuildActive();
        API.tileLayout(true);
        output({tiles: API.TILE.fields.length,
                 voices: API.TILE.fields.reduce((a, f) => a + f.idx.length, 0)});
        """
        )
        assert d["tiles"] == 4  # 200, 60, 20, 20 — the two singles out
        assert d["voices"] == 300

    def test_the_layout_is_not_computed_for_every_frame(self):
        """The core of the saving: a frame is a redraw, not a recompute."""
        d = js(
            STRUCTURE
            + """
        API.tileLayout(true);
        const before = API.TILE.fields[0];
        API.tileLayout();          // same size → must do nothing
        const same = API.TILE.fields[0] === before;
        API.canvas_el(900, 600);
        API.tileLayout();          // different size → must recompute
        output({same, new_value: API.TILE.fields[0] !== before,
                 width: API.TILE.width});
        """
        )
        assert d["same"], "the layout was recomputed without cause"
        assert d["new_value"] and d["width"] == 900


@node
class TestTheLayoutFollowsTheData:
    """Finding 2 of the code review. Window size alone as a fingerprint was
    not enough: whoever moves the threshold gets new nodes and new indices
    at the SAME canvas size. The layout stayed put — the image was
    byte-for-byte the same (the slider looked broken), and `f.idx` pointed
    to voices of the previous run.
    """

    def test_a_new_load_forces_a_new_layout(self):
        d = js(
            STRUCTURE
            + """
        API.tileLayout(true);
        const before = API.TILE.fields[0];
        API.GRAPH.seq = (API.GRAPH.seq || 0) + 1;    // like mapLoadGraph
        API.tileLayout();
        output({new_value: API.TILE.fields[0] !== before});
        """
        )
        assert d["new_value"], "the layout survives a load"

    def test_a_change_to_the_inventory_forces_it_too(self):
        d = js(
            STRUCTURE
            + """
        API.tileLayout(true);
        const before = API.TILE.fields.length;
        API.GRAPH.minGroup = 20; API.graphBuildActive();
        API.tileLayout();
        output({before, after: API.TILE.fields.length});
        """
        )
        assert d["after"] < d["before"], "the minimum size has no effect on the layout"

    def test_without_a_change_it_stays(self):
        """The saving must not get lost in the process."""
        d = js(
            STRUCTURE
            + """
        API.tileLayout(true);
        const before = API.TILE.fields[0];
        API.tileLayout();
        output({same: API.TILE.fields[0] === before});
        """
        )
        assert d["same"]


@node
class TestHitTesting:
    def test_a_click_finds_the_voice_under_the_cursor(self):
        d = js(
            STRUCTURE
            + """
        API.tileLayout(true);
        const f = API.TILE.fields[0];
        // Center of the third cell of the second row
        const col = 2, row = 1;
        const t = API.tileHits(f.x + (col + 0.5) * f.cell,
                                    f.y + (row + 0.5) * f.cell);
        output({i: t.i, expected: f.idx[row * f.columns + col], field_el: t.field_el});
        """
        )
        assert d["i"] == d["expected"]
        assert d["field_el"] == 0

    def test_next_to_the_tiles_nothing_is_hit(self):
        d = js(
            STRUCTURE
            + """
        API.tileLayout(true);
        output({a: API.tileHits(-50, -50), b: API.tileHits(5000, 5000)});
        """
        )
        assert d["a"]["i"] == -1 and d["b"]["i"] == -1

    def test_in_a_tiles_gap_no_voice_is_hit(self):
        """The last row of a tile is usually not full. Reporting a voice
        there would be worse than none at all: you would hear something
        different from what you clicked."""
        d = js(
            STRUCTURE
            + """
        API.tileLayout(true);
        const f = API.TILE.fields[0];
        const lines = Math.ceil(f.idx.length / f.columns);
        // One column behind the last entry of the last row
        const remainder = f.idx.length - (lines - 1) * f.columns;
        const t = remainder < f.columns
            ? API.tileHits(f.x + (remainder + 0.5) * f.cell,
                                f.y + (lines - 0.5) * f.cell)
            : {i: -1};
        output({i: t.i, remainder, columns: f.columns});
        """
        )
        assert d["i"] == -1, "a gap reports a voice"


@node
class TestDrawing:
    def test_a_frame_is_made_of_rectangles_not_paths(self):
        """This IS the answer to the rendering lag: no curves, no lines, no
        hulls — just `fillRect`."""
        d = js(
            STRUCTURE
            + """
        API.tileLayout(true);
        const {c, p} = paper();
        API.tilesDraw(c, 1200, 800);
        output({rects: p.rects, path_list: p.path_list, curves: p.curvePoints,
                 texts: p.texts.length});
        """
        )
        assert d["rects"] >= 302, "not every voice was drawn"
        assert d["curves"] == 0, "paths are being drawn — that was the problem"
        assert d["texts"] > 0, "no label"

    def test_small_tiles_get_no_name(self):
        """Previously the name was painted at 10 px if needed and ran over
        into the neighbours — with the many narrow tiles that produced a
        strip of merged-together "Group 261 Group 219 Group 253"."""
        d = js(
            STRUCTURE
            + """
        API.tileLayout(true);
        const {c, p} = paper();
        API.tilesDraw(c, 1200, 800);
        // Does every drawn text fit inside its tile?
        let across = 0;
        for (const t of p.texts) {
            const gr = parseFloat((t.font.match(/([\\d.]+)px/) || [0, 0])[1]);
            const width = t.t.length * gr * 0.62;
            const f = API.TILE.fields.find(
                f => t.x >= f.x && t.x <= f.x + f.w && t.y >= f.y && t.y <= f.y + f.h);
            if (!f || width > f.w + 1) across++;
        }
        output({across, texts: p.texts.length});
        """
        )
        assert d["across"] == 0, f"{d['across']} labels overhang their tile"

    def test_filtered_voices_are_dimmed_not_removed(self):
        """The same rule as in the graph: otherwise you could no longer see
        that a group extends beyond the filter."""
        d = js(
            STRUCTURE
            + """
        API.tileLayout(true);
        API.GRAPH.filtered = true;
        API.GRAPH.visible = new Uint8Array(302);
        for (let i = 0; i < 50; i++) API.GRAPH.visible[i] = 1;
        const {c, p} = paper();
        API.tilesDraw(c, 1200, 800);
        output({rects: p.rects, dimmed: p.weakRects});
        """
        )
        assert d["rects"] >= 302, "filtered voices were omitted"
        assert d["dimmed"] > 0, "nothing was dimmed"


class TestSource:
    """What can only be checked from the source text."""

    def test_no_force_simulation_and_no_endless_loop(self):
        # Code only: the header TALKS ABOUT the render loop that is missing here.
        q = "\n".join(
            z
            for z in TILES_JS.read_text(encoding="utf-8").splitlines()
            if not z.lstrip().startswith(("*", "//", "/*"))
        )
        assert "requestAnimationFrame" not in q
        assert "graphStep" not in q
        # `graphRun` occurs exactly ONCE: in the switcher, where the graph
        # gets re-arranged once you actually look at it. Anywhere else it
        # would be out of place here.
        assert q.count("graphRun(") == 1
        i = q.index("function mapView(")
        assert q.index("graphRun(") > i

    def test_the_tiles_are_the_default(self):
        """Unconditional: the remembered view is no longer
        read, because otherwise the map would drift back into the force
        simulation."""
        q = TILES_JS.read_text(encoding="utf-8")
        i = q.index("function tileRememberedView(")
        body = q[i : i + 400]
        assert "return 'tiles';" in body
        assert "localStorage" not in body, "the default is hanging off the remembered value again"

    def test_the_toggle_is_wired(self):
        h = INDEX.read_text(encoding="utf-8")
        assert 'data-view="tiles"' in h and 'data-view="graph"' in h
        assert "mapView('tiles')" in h and "mapView('graph')" in h
        assert ".map-view-btn.active" in CSS.read_text(encoding="utf-8")

    def test_the_drawing_branches_early(self):
        """Another `if` in each of thousands of loop iterations would be the
        wrong place — the tiles draw completely differently."""
        q = MAP_JS.read_text(encoding="utf-8")
        i = q.index("if (TILE.on) { tilesDraw(g, w, h); return; }")
        j = q.index("for (let i = 0; i < MAP.n; i++) {", i)
        assert j > i, "the tile branch comes after the dot loop"

    def test_no_zoom_and_no_dragging_in_the_tiles(self):
        """The image fills the area; there is nothing to move."""
        q = MAP_JS.read_text(encoding="utf-8")
        i = q.index("cv.addEventListener('wheel'")
        assert "if (TILE.on) return;" in q[i : i + 400]

    def test_the_graph_sliders_disappear(self):
        """A slider for a force simulation that isn't currently running is
        a dummy — the same house rule as with the settings."""
        q = TILES_JS.read_text(encoding="utf-8")
        assert "map-graph-only" in q
        h = INDEX.read_text(encoding="utf-8")
        assert h.count("map-graph-only") >= 6

    def test_the_graphs_grid_does_not_apply_there(self):
        """A hit test over the graph coordinates would simply be wrong in
        the tiles — it would report voices at places where they aren't
        actually located."""
        q = MAP_JS.read_text(encoding="utf-8")
        i = q.index("function mapBuildGrid(")
        assert "TILE.on) { MAP.raster = null; return; }" in q[i : i + 500]


class TestThresholdArrows:
    """Up and down arrows next to the threshold, so it can be controlled
    without dragging.

    The slider spans 0.30 to 0.95 over roughly 200 pixels — about three
    pixels per hundredth. Yet exactly one hundredth makes a visible
    difference: 0.62 → 0.61 changes the inventory.
    """

    def test_both_arrows_are_in_the_markup(self):
        h = INDEX.read_text(encoding="utf-8")
        assert 'id="mapThresholdUp"' in h and 'id="mapThresholdDown"' in h
        assert "graphThresholdStep(1)" in h and "graphThresholdStep(-1)" in h

    def test_up_means_up(self):
        """The arrows at the threshold were once the wrong way round — they
        were shown as ▼ above ▲. That is not a matter of taste: the buttons
        are stacked on top of each other, so the upper one MUST increase
        the value."""
        h = INDEX.read_text(encoding="utf-8")
        i = h.index('id="mapThresholdUp"')
        j = h.index('id="mapThresholdDown"')
        assert i < j, "the down arrow sits above the up arrow in the markup"
        assert "▲</button>" in h[i:j] and "▼</button>" in h[j : j + 300]

    def test_the_step_comes_from_the_slider(self):
        """A second number in the JavaScript would eventually drift apart
        from the `step` of the slider."""
        q = GRAPH_JS.read_text(encoding="utf-8")
        i = q.index("function graphThresholdStep(")
        r = q[i : i + 900]
        assert "Number(s.step)" in r
        assert "0.01" in r, "no fallback value if step is missing"

    def test_the_value_stays_within_bounds(self):
        q = GRAPH_JS.read_text(encoding="utf-8")
        i = q.index("function graphThresholdStep(")
        r = q[i : i + 900]
        assert "Math.min(max" in r and "Math.max(min" in r

    def test_it_snaps_to_the_step(self):
        """A value dragged by hand, like 0.6449, would otherwise jump to
        0.6549 — and the display would keep showing 0.65."""
        q = GRAPH_JS.read_text(encoding="utf-8")
        i = q.index("function graphThresholdStep(")
        assert "Math.round(Number(s.value) / level) * level" in q[i : i + 900]

    def test_it_takes_the_same_path_as_dragging(self):
        """Otherwise the display would stay put or nothing would be
        recomputed."""
        q = GRAPH_JS.read_text(encoding="utf-8")
        i = q.index("function graphThresholdStep(")
        assert "graphThresholdSet(s.value)" in q[i : i + 900]


class TestTheGraphComputesOnlyWhenVisible:
    """The graph used to be computed while the tiles were shown.

    That was not right. On opening, hundreds of layout steps ran — many
    seconds of computing for an image nobody is looking at. And that was
    exactly what blocked the interface where you were just trying to click
    on tiles: the tile view was fast and still felt sluggish.
    """

    def test_loading_does_not_lay_out_blindly(self):
        q = GRAPH_JS.read_text(encoding="utf-8")
        i = q.index("async function mapLoadGraph")
        r = q[i : q.index("function graphInitialLayout", i)]
        assert "if (typeof TILE === 'undefined' || !TILE.on) {" in r
        # ... and never past the size cap: too big switches to the tiles
        assert "mapGraphTooBig()" in r and "mapView('tiles')" in r
        assert "else graphRun();" in r
        assert "GRAPH.arranged = false;" in r

    def test_the_run_refuses_in_tile_mode(self):
        """Second safety net: a direct call must not start anything either."""
        q = GRAPH_JS.read_text(encoding="utf-8")
        i = q.index("function graphRun(")
        assert "if (typeof TILE !== 'undefined' && TILE.on) return;" in q[i : i + 900]

    def test_switching_to_tiles_halts_the_run(self):
        q = GRAPH_JS.read_text(encoding="utf-8")
        i = q.index("const tick = () => {")
        assert "TILE.on" in q[i : i + 700]

    def test_and_it_does_not_start_over_afterwards(self):
        """`ticks` stays put — otherwise the already-computed steps would
        be lost every time you switch back and forth."""
        q = GRAPH_JS.read_text(encoding="utf-8")
        i = q.index("const tick = () => {")
        r = q[i : i + 900]
        assert "GRAPH.ticks = 0" not in r

    def test_switching_to_the_graph_lays_out_again(self):
        q = TILES_JS.read_text(encoding="utf-8")
        i = q.index("function mapView(")
        assert "if (!GRAPH.arranged && !GRAPH.running) graphRun();" in q[i : i + 1200]

    def test_done_means_laid_out(self):
        q = GRAPH_JS.read_text(encoding="utf-8")
        i = q.index("GRAPH.running = false;\n            GRAPH.arranged = true;")
        assert i > 0


class TestHidingWorksInTheTilesToo:
    """Finding 3 of the code review. `graphRearrange()` is the path
    through which EVERY hide and selection change reaches the image again
    — and it read `graphInitialLayout(); graphRun();`. `graphRun`
    returns immediately in the tiles: the supposedly hidden groups stayed
    unchanged in the image, while the line underneath claimed the opposite.
    """

    def test_the_detour_through_the_force_simulation_is_closed(self):
        q = GRAPH_JS.read_text(encoding="utf-8")
        i = q.index("function graphRearrange(")
        r = q[i : i + 500]
        assert "TILE.on" in r
        assert "tileLayout(true)" in r and "mapDraw()" in r
        assert r.index("tileLayout(true)") < r.index("graphInitialLayout()")

    def test_every_hide_path_goes_through_it(self):
        """If any one of them calls `graphRun()` directly again, the bug
        is back — that's why they're recorded here by name."""
        q = GRAPH_JS.read_text(encoding="utf-8")
        for f in (
            "graphOnlyThisGroup",
            "graphToggleSelection",
            "graphClearSelection",
            "graphMinGroupSet",
            "graphToggleNameClusters",
        ):
            i = q.index(f"function {f}(")
            body = q[i : q.index("\n}", i)]
            assert "graphRearrange()" in body, f
            assert "graphRun()" not in body, f


class TestTheLegendNoLongerBlocks:
    """The legend ("the ten biggest groups at threshold …") blocked the
    view.

    The legend sat as a fixed box bottom-left over the map; with one large
    group it covered several hundred voices. It was collapsed then —
    and now it doesn't float at all: the largest groups are what
    the docked details show while no voice is chosen. Covering nothing, it
    can show forty instead of ten, and a click opens a group.
    """

    def test_it_lives_in_the_docked_details(self):
        h = INDEX.read_text(encoding="utf-8")
        side = h[h.index('id="mapSide"') :]
        side = side[: side.index("</aside>")]
        assert 'id="mapLegend"' in side and 'id="mapSelection"' in side
        c = CSS.read_text(encoding="utf-8")
        block = c[c.index("\n.map-legend {") :]
        block = block[: block.index("}")]
        assert "position:absolute" not in block

    def test_shown_while_no_voice_is_chosen(self):
        panel = (STATIC / "js" / "voicemap_panel.js").read_text(encoding="utf-8")
        i = panel.index("function panelSideContent(")
        body = panel[i : panel.index("\n}\n", i)]
        assert "legend.style.display = chosen ? 'none' : 'block'" in body
        close = panel[panel.index("function panelClose(") :]
        assert "panelSideContent()" in close[: close.index("\n}\n")]

    def test_a_group_opens_from_the_list(self):
        q = GRAPH_JS.read_text(encoding="utf-8")
        assert 'onclick="graphOpenGroup(${gr.nr})"' in q
        i = q.index("function graphOpenGroup(")
        assert "mapPickVoice(best)" in q[i : q.index("\n}\n", i)]

    def test_the_selection_can_still_be_cleared(self):
        """The checkboxes keep only some groups in the picture — and the
        way back must stay in sight."""
        q = GRAPH_JS.read_text(encoding="utf-8")
        i = q.index("function graphLegend(")
        body = q[i : q.index("\n}\n", i)]
        assert "graphToggleSelection(" in body and "graphClearSelection()" in body


class TestMeasuredOnTheRealInventory:
    """What the browser said against a real inventory of several thousand voices.

    Tiles                    all voices included
    Draw tiles               4.4 ms per frame
    Draw graph               10.7 ms per frame
    Layout (graph)           elongation 1.11 — round again instead of a strand
    Arrow down/up             0.65 → 0.64 → 0.65
    """

    def test_the_measurement_is_in_the_file_header(self):
        header = TILES_JS.read_text(encoding="utf-8")
        header = header[: header.index("const TILE = {")]
        assert "4.4 ms" in header and "10.7 ms" in header
        assert "no\n *  EDGES" in header, "the price of the tiles is missing"


# ── The tiles are the entry point ─────────────────────────────────────────


def test_the_map_always_opens_with_tiles():
    """Opening the map lands on the tile view, not the simulation.

    Previously the last-used view was remembered — whoever once looked at
    the graph landed back in the force simulation on the next open and
    waited for 420 layout steps. The remembered value is no longer allowed
    to decide that, even if it still sits in localStorage.
    """
    d = js("""
        const before = API.tileRememberedView();
        API.setRemembered('graph');                 // old state, stubborn
        output({without: before, withOldState: API.tileRememberedView()});
    """)
    assert d["without"] == "tiles"
    assert d["withOldState"] == "tiles", "an old localStorage entry pulls the simulation back"


def test_highlighted_voices_are_marked_in_the_tiles():
    """The jump from the transcript ("🗣 Map" for a speaker) has landed in
    the tiles — so you also have to be able to see there which
    voice was meant.

    Two levels, and both are needed: a cell is only a few pixels large when
    there are many voices, and you can't find that without the tile
    outline; and the outline alone only tells the group, not the voice.
    """
    d = js(
        STRUCTURE
        + """
        const {c, p} = paper();
        API.mapView('tiles');
        API.tilesDraw(c, 1200, 800);
        const without = p.frame.filter(r => String(r.color).toLowerCase() === '#ffd24a').length;

        // 0,1,2 lie in group 1, 260 in group 3 — so two tiles.
        const {c: c2, p: p2} = paper();
        API.mapSet({highlight: new Set([0, 1, 2, 260])});
        API.tilesDraw(c2, 1200, 800);
        const four = p2.frame.filter(r => String(r.color).toLowerCase() === '#ffd24a');

        // A single voice: one marker, one tile.
        const {c: c3, p: p3} = paper();
        API.mapSet({highlight: new Set([0])});
        API.tilesDraw(c3, 1200, 800);
        const one_of = p3.frame.filter(r => String(r.color).toLowerCase() === '#ffd24a');
        output({without, four: four.length, one_of: one_of.length});
    """
    )
    assert d["without"] == 0, "nothing yellow may be drawn without a highlight"
    # 4 voices + 2 affected tiles
    assert d["four"] == 6, d
    # 1 voice + 1 tile — both levels scale with the selection
    assert d["one_of"] == 2, d


def test_without_a_highlight_the_image_stays_as_it_was():
    """The tiles' drawing cost is measured and documented (4.4 ms). A
    highlight that does work even without a hit would silently make that
    worse."""
    d = js(
        STRUCTURE
        + """
        const a = paper(); API.mapView('tiles');
        API.tilesDraw(a.c, 1200, 800);
        API.mapSet({highlight: null});
        const b = paper(); API.tilesDraw(b.c, 1200, 800);
        output({strokes_a: a.p.strokes, strokes_b: b.p.strokes,
                 rects_a: a.p.rects, rects_b: b.p.rects});
    """
    )
    assert d["strokes_a"] == d["strokes_b"]
    assert d["rects_a"] == d["rects_b"]


# ── The retained frame ────────────────────────────────────────────────────


def test_a_second_frame_does_not_redraw_the_cells():
    """**The core of the speedup.**

    The browser warned that the page was slowing it down. Measured against
    a large inventory — tens of thousands of voices in thousands of tiles —
    one frame cost 156 ms, and moving the mouse over the map
    produced one per mouse move: 155–220 ms, thirty times in a row. Yet
    nothing changes under the pointer except a white rectangle.

    After the fix: 1.3 ms per move.
    """
    d = js(
        STRUCTURE
        + """
        const a = paper(); API.mapView('tiles');
        API.tilesDraw(a.c, 1200, 800);
        // NOT fresh: that is exactly the question — what does the next frame cost?
        const b = paper(false); API.tilesDraw(b.c, 1200, 800);
        output({first: a.p.rects, second: b.p.rects, images: b.p.images});
    """
    )
    assert d["first"] > 0, "the first frame draws nothing — a test without a claim"
    assert d["second"] == 0, (
        f"the second frame redraws {d['second']} rectangles instead of stamping"
    )
    assert d["images"] == 1, "the retained canvas is not used"


def test_the_cursor_is_drawn_over_the_cached_image():
    """The upper layer still has to be created — otherwise the saving would
    come at the cost of no longer seeing what you're pointing at."""
    d = js(
        STRUCTURE
        + """
        const a = paper(); API.mapView('tiles');
        API.tilesDraw(a.c, 1200, 800);
        API.TILE.below = API.TILE.fields[0].idx[0];
        API.TILE.subField = 0;
        const b = paper(false); API.tilesDraw(b.c, 1200, 800);
        output({strokes: b.p.strokes,
                 white: b.p.frame.filter(r => r.color === '#fff').length});
    """
    )
    assert d["white"] == 1, "the marker under the pointer is missing"


def test_a_change_to_the_inventory_discards_the_image():
    """The more dangerous bug of the two: a frame that shows an outdated
    truth. You can't tell just by looking at it."""
    d = js(
        STRUCTURE
        + """
        const a = paper(); API.mapView('tiles');
        API.tilesDraw(a.c, 1200, 800);
        const before = API.TILE.imageStamp;
        API.GRAPH.seq++;                       // a new load run
        const b = paper(false); API.tilesDraw(b.c, 1200, 800);
        output({before, after: API.TILE.imageStamp,
                 redrawn: b.p.rects > 0});
    """
    )
    assert d["before"] != d["after"], "the stamp does not notice the new run"
    assert d["redrawn"], "the frame was not redrawn"


def test_a_new_filter_discards_the_image():
    """Two different filters can have the same number of hits — a mere hit
    count in the stamp would not tell them apart. That's why
    `GRAPH.viewState` counts every change."""
    d = js(
        STRUCTURE
        + """
        const a = paper(); API.mapView('tiles');
        API.tilesDraw(a.c, 1200, 800);
        const before = API.TILE.imageStamp;
        API.GRAPH.viewState = (API.GRAPH.viewState | 0) + 1;
        const b = paper(false); API.tilesDraw(b.c, 1200, 800);
        output({same: before === API.TILE.imageStamp,
                 redrawn: b.p.rects > 0});
    """
    )
    assert not d["same"] and d["redrawn"]


def test_the_handbrake_works():
    """`tileDiscardImage()` is the path for everything the stamp can't
    capture — a name being assigned, for example, recolors the tiles."""
    d = js(
        STRUCTURE
        + """
        const a = paper(); API.mapView('tiles');
        API.tilesDraw(a.c, 1200, 800);
        API.tileDiscardImage();
        const b = paper(false); API.tilesDraw(b.c, 1200, 800);
        output({redrawn: b.p.rects > 0});
    """
    )
    assert d["redrawn"]


def test_the_handbrake_brings_a_new_name_onto_its_tile():
    """A group named in the panel kept its old label on the tile until the
    map was loaded again: the layout, which holds the names, stayed."""
    d = js(
        STRUCTURE
        + """
        const a = paper(); API.mapView('tiles');
        API.tilesDraw(a.c, 1200, 800);
        const nr = API.TILE.fields.find(f => f.nr).nr;
        API.GRAPH.groups.find(g => g.nr === nr).name = 'Alice Example';
        API.tileDiscardImage();
        const b = paper(false); API.tilesDraw(b.c, 1200, 800);
        output({names: API.TILE.fields.map(f => f.name)});
    """
    )
    assert "Alice Example" in d["names"]


def test_a_new_window_size_discards_the_image():
    """Otherwise the old frame would sit distorted in the new canvas."""
    d = js(
        STRUCTURE
        + """
        const a = paper(); API.mapView('tiles');
        API.tilesDraw(a.c, 1200, 800);
        API.canvas_el(900, 600);
        const b = paper(false); API.tilesDraw(b.c, 900, 600);
        output({redrawn: b.p.rects > 0});
    """
    )
    assert d["redrawn"]


def test_naming_discards_the_image():
    """Checked statically: both places that set a name must pull the
    handbrake. Without it the tile would stay in the old color, and the
    freshly named speaker would look unnamed."""
    panel = (
        Path(__file__).resolve().parent.parent
        / "whisper_tm"
        / "static"
        / "js"
        / "voicemap_panel.js"
    ).read_text(encoding="utf-8")
    assert panel.count("tileDiscardImage()") >= 2, (
        "one of the naming spots does not pull the handbrake"
    )


# ── The treemap ────────────────────────────────────────────────────────────


def test_the_treemap_stays_affordable_with_many_items():
    """Two quadratic spots, found at a high threshold.

    ``tileSquarify`` checked a row's aspect ratio by copying the whole row
    per candidate and iterating over it (O(k²) per row), and took the items
    off with ``shift()`` — which shifts the whole array up by one for every
    item. With tens of thousands of tiles that added up to seconds per
    layout, afterwards a few ms.

    The limit here is generous (half a second for 40,000 items): it is
    meant to catch the RETURN of quadratic behaviour, not to grade the
    machine's mood of the day.
    """
    d = js("""
        const entries = [];
        for (let i = 0; i < 40000; i++) entries.push({weight: 1 + (i % 97)});
        const a = Date.now();
        const fields = API.tileSquarify(entries, 0, 0, 1600, 1000);
        output({ms: Date.now() - a, fields: fields.length});
    """)
    assert d["fields"] == 40000, "items were swallowed"
    assert d["ms"] < 500, f"the treemap needs {d['ms']} ms for 40,000 items"


def test_the_treemap_still_divides_the_area_correctly():
    """The rewrite must not change the RESULT.

    A row's worst aspect ratio depends only on the largest and smallest
    area in it — everything in between can never be worse. The short
    calculation is therefore the same statement as the long one, and that
    has to be visible: same count, same area shares, nothing overlaps.
    """
    d = js("""
        const entries = [{weight: 50}, {weight: 30}, {weight: 12},
                        {weight: 5}, {weight: 3}];
        const f = API.tileSquarify(entries, 0, 0, 400, 300);
        const total = f.reduce((s, r) => s + r.w * r.h, 0);
        const shares = f.map(r => Math.round((r.w * r.h) / total * 1000) / 10);
        let overlaps = 0;
        for (let i = 0; i < f.length; i++) for (let j = i + 1; j < f.length; j++) {
            const a = f[i], b = f[j];
            if (a.x < b.x + b.w - 0.01 && b.x < a.x + a.w - 0.01
                && a.y < b.y + b.h - 0.01 && b.y < a.y + a.h - 0.01) overlaps++;
        }
        output({n: f.length, total: Math.round(total), shares, overlaps,
                 weights: f.map(r => r.p.weight)});
    """)
    assert d["n"] == 5
    assert abs(d["total"] - 400 * 300) < 500, "the area does not add up"
    # The shares must follow the weights: 50/30/12/5/3 out of 100.
    assert d["shares"] == [50.0, 30.0, 12.0, 5.0, 3.0], d["shares"]
    assert d["weights"] == [50, 30, 12, 5, 3], "the order is wrong"
    assert d["overlaps"] == 0, "tiles overlap"


def test_a_single_voice_gets_a_crosshair():
    """Going to the speaker map from a chunk in a transcript showed a
    bunch of little boxes for many people.

    Re-measured: what was marked was **exactly one** cell in **one** tile.
    The many light little boxes are the FILTER — the jump restricts the map
    to the recording, and its hundreds of voices are spread across the
    tiles of many people, light against the dimmed remainder.

    The ring around the searched cell was there all along, just
    undiscoverable among tens of thousands of cells, of which hundreds
    light up. Two lines
    across the canvas now point at it.
    """
    d = js(
        STRUCTURE
        + """
        API.mapView('tiles');
        const {c, p} = paper();
        API.mapSet({highlight: new Set([0])});
        API.tilesDraw(c, 1200, 800);
        output({strokes: p.strokes, curves: p.curvePoints,
                 yellow: p.frame.filter(r => String(r.color).toLowerCase() === '#ffd24a').length});
    """
    )
    # Four lines (two interrupted axes) + a ring.
    assert d["curves"] >= 8, f"no crosshair drawn: {d}"


def test_with_many_marked_voices_there_is_no_grid():
    """With a whole speaker ID there are thirty windows — thirty crosses
    would be a grid and would cover the map. The cross answers "where is
    the ONE?", and that question doesn't arise then."""
    one = js(
        STRUCTURE
        + """
        API.mapView('tiles');
        const a = paper();
        API.mapSet({highlight: new Set([0])});
        API.tilesDraw(a.c, 1200, 800);
        output({curves: a.p.curvePoints});
    """
    )
    many = js(
        STRUCTURE
        + """
        API.mapView('tiles');
        const b = paper();
        API.mapSet({highlight: new Set([0, 1, 2, 3])});
        API.tilesDraw(b.c, 1200, 800);
        output({curves: b.p.curvePoints});
    """
    )
    assert one["curves"] > many["curves"], (one, many)
    assert many["curves"] == 0, "it still crosses with four voices"


@node
class TestNamedVersusTotal:
    """A person may have a few hundred assigned voices, while their tile
    in the map shows ten times as many.

    Both are correct. At a high threshold the assigned voices are spread
    over hundreds of groups, and all voices in those groups combined are
    far more than the assigned ones.

    The tile showed only the last number. It's meant to say "this many
    voices are connected to him", but it looked like "this many are his" —
    and the difference is exactly the work that's still left to do.
    """

    STRUCTURE = """
    const gv = [], members = [];
    const setIt = (nr, n, name) => {
        for (let i = 0; i < n; i++) {
            gv.push(nr);
            members.push({id: members.length + 1, display: `g${nr}/d${i}.mp4`,
                      person: name, label: 'S', windows: 1,
                      start_sec: 0, speech_sec: 30});
        }
    };
    setIt(1, 10, 'Frank');      // ten named …
    setIt(1, 90, '');           // … in a group with 90 unnamed
    setIt(2, 40, '');
    API.mapSet({n: gv.length, members: members,
                     name: members.map(m => m.person)});
    API.GRAPH.groupOf = Int32Array.from(gv);
    API.GRAPH.groups = [{nr: 1, n: 100, name: 'Frank'}, {nr: 2, n: 40, name: ''}];
    API.GRAPH.minGroup = 1;
    API.GRAPH.clusterNames = false;
    API.graphBuildActive();
    API.canvas_el(900, 600);
    API.tileLayout(true);
    """

    def test_the_tile_knows_both_numbers(self):
        d = js(
            self.STRUCTURE
            + """
        const f = API.TILE.fields.find(f => f.name === 'Frank');
        output({named: f.named, total: f.idx.length});
        """
        )
        assert d["named"] == 10, "the assigned voices are missing"
        assert d["total"] == 100

    def test_without_names_one_number_is_enough(self):
        """An unnamed group has nothing to distinguish — there "0 / 40"
        would just be noise."""
        d = js(
            self.STRUCTURE
            + """
        const f = API.TILE.fields.find(f => !f.name);
        output({named: f.named, total: f.idx.length});
        """
        )
        assert d["named"] == 0
        assert d["total"] == 40

    def test_a_fully_named_group_shows_only_one_number(self):
        """If everything is assigned, two equal numbers would be silly."""
        d = js("""
        const gv = [], members = [];
        for (let i = 0; i < 12; i++) {
            gv.push(1);
            members.push({id: i + 1, display: `a${i}.mp4`, person: 'Alice',
                      label: 'S', windows: 1, start_sec: 0, speech_sec: 30});
        }
        API.mapSet({n: gv.length, members: members,
                         name: members.map(m => m.person)});
        API.GRAPH.groupOf = Int32Array.from(gv);
        API.GRAPH.groups = [{nr: 1, n: 12, name: 'Alice'}];
        API.GRAPH.minGroup = 1; API.GRAPH.clusterNames = false;
        API.graphBuildActive();
        API.canvas_el(900, 600);
        API.tileLayout(true);
        const f = API.TILE.fields[0];
        output({named: f.named, total: f.idx.length,
                 twoNumbers: f.named > 0 && f.named < f.idx.length});
        """)
        assert d["named"] == d["total"] == 12
        assert d["twoNumbers"] is False
