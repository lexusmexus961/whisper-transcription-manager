"""Outlines around groups — and names that stay when zooming out.

Two problems on the same view:

    - the names of the groups disappeared when zooming out, so a small named
      group could no longer be found
    - many unnamed groups were just gray and blurred into each other

Both are drawing logic, and it could not be tested until now: the map tests
(``test_voicemap_wiring.py``) read only source code and catch renamed
handlers — no calculation error and no wrong thinning.

So here REAL JavaScript in node, drawn on a paper canvas that just records
(``tests/js/voicemap_harness.js``). What is checked is WHAT was drawn, not how it
looks — this does not replace visual inspection, but it catches exactly the
kind of error that made a named group disappear: a thinning that discards the wrong thing.

Without node the tests are skipped; the suite must be able to run without JS
tools.
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
    """Run a snippet against the real file.

    The snippet gets ``API`` and ``paper`` and returns its result via
    ``output(obj)`` — as JSON on the last line, so an accidental ``console.log``
    in the file doesn't break the test.
    """
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


# A small map with three groups: one large anonymous one, one medium
# anonymous one, and "Alice" with two voices — the case that used to fail.
STRUCTURE = """
const x = [], y = [], gv = [];
const setIt = (nr, n, mx, my) => {
    for (let i = 0; i < n; i++) {
        x.push(mx + (i % 5) * 4); y.push(my + Math.floor(i / 5) * 4); gv.push(nr);
    }
};
setIt(1, 40, 0, 0);        // large anonymous group
setIt(2, 8, 300, 0);       // medium anonymous group
setIt(3, 2, 600, 0);       // "Alice" — small and named
setIt(0, 6, 900, 300);     // loner without a group
API.mapSet({n: x.length, x, y, sx: 1, ox: 0, oy: 0, layer: 'graph'});
API.GRAPH.groupOf = gv;
API.GRAPH.groups = [
    {nr: 1, n: 40, name: '', mean: 0.8},
    {nr: 2, n: 8, name: '', mean: 0.7},
    {nr: 3, n: 2, name: 'Alice', mean: 0.9},
];
API.GRAPH.showNames = true;
API.GRAPH.showOutlines = true;
"""


# The same name on several groups — the normal case in practice.
# "me" four times in descending size, "Alice" twice, plus one anonymous.
SHARED_NAMES = """
const x = [], y = [], gv = [], groups = [];
const setIt = (nr, n, mx, name) => {
    for (let i = 0; i < n; i++) { x.push(mx + (i % 5) * 3); y.push(0); gv.push(nr); }
    groups.push({nr, n, name, mean: 0.9});
};
setIt(1, 60, 0, 'me');
setIt(2, 20, 400, 'Alice');
setIt(3, 9, 800, 'me');
setIt(4, 9, 1200, 'me');
setIt(5, 9, 1600, 'me');
setIt(6, 9, 2000, 'Alice');
setIt(7, 9, 2400, '');
API.mapSet({n: x.length, x, y, sx: 1, ox: 0, oy: 0, layer: 'graph'});
API.GRAPH.groupOf = gv;
API.GRAPH.groups = groups;      // descending by size, as from the server
API.GRAPH.showNames = true;
"""


# ── The names ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("zoom", [0.15, 0.26, 0.5, 1.0, 3.0])
def test_a_small_named_group_is_shown_at_every_zoom_level(zoom):
    """The actual finding: an ASSIGNED name must never be thinned out.

    0.26 is the level the graph settles on after fitting — that's exactly
    where Alice was missing, because she has fewer than five voices.
    """
    # Window big enough that the crop check doesn't interfere: when
    # zooming in the groups drift apart, and a name outside the
    # picture is correctly omitted — that's not the question here.
    r = js(
        STRUCTURE
        + f"""
    const {{c, p}} = paper();
    API.graphDrawNames(c, {zoom}, 400, 300, 4000, 800);
    output({{texts: p.texts.map(t => t.t)}});
    """
    )
    assert "Alice" in r["texts"], f"Alice missing at zoom {zoom}: {r['texts']}"


def test_a_named_group_does_not_use_up_the_quota():
    """Otherwise the names would take the anonymous groups' spots.

    With 30 named groups and a quota of 22, otherwise NO anonymous
    group would still be labeled — the thinning would again hit the
    wrong thing, just the other way around.
    """
    r = js("""
    const x = [], y = [], gv = [], groups = [];
    for (let nr = 1; nr <= 30; nr++) {
        for (let i = 0; i < 4; i++) { x.push(nr * 30 + i); y.push(0); gv.push(nr); }
        groups.push({nr, n: 4, name: 'Person' + nr, mean: 0.8});
    }
    for (let nr = 31; nr <= 40; nr++) {
        for (let i = 0; i < 9; i++) { x.push(nr * 30 + i); y.push(60); gv.push(nr); }
        groups.push({nr, n: 9, name: '', mean: 0.8});
    }
    API.mapSet({n: x.length, x, y, sx: 1, ox: 0, oy: 0, layer: 'graph'});
    API.GRAPH.groupOf = gv;
    API.GRAPH.groups = groups;
    API.GRAPH.showNames = true;
    const {c, p} = paper();
    API.graphDrawNames(c, 0.3, 0, 400, 4000, 800);
    const t = p.texts.map(z => z.t);
    output({named: t.filter(s => s.startsWith('Person')).length,
             anonymous: t.filter(s => s.startsWith('Group')).length});
    """)
    assert r["named"] == 30, "all names belong on the map"
    assert r["anonymous"] > 0, "the anonymous groups must not be crowded out"


def test_anonymous_groups_are_thinned_out_when_zooming_out():
    """The thinning stays — it just no longer hits the names.

    Without it the overview would be a carpet of labels; that was the
    reason it exists.
    """
    r = js("""
    const x = [], y = [], gv = [], groups = [];
    for (let nr = 1; nr <= 40; nr++) {
        for (let i = 0; i < 4; i++) { x.push(nr * 30 + i); y.push(0); gv.push(nr); }
        groups.push({nr, n: 4, name: '', mean: 0.8});
    }
    API.mapSet({n: x.length, x, y, sx: 1, ox: 0, oy: 0, layer: 'graph'});
    API.GRAPH.groupOf = gv;
    API.GRAPH.groups = groups;
    API.GRAPH.showNames = true;
    const countTexts = (s) => {
        const {c, p} = paper();
        API.graphDrawNames(c, s, 0, 400, 4000, 800);
        return p.texts.length;
    };
    output({far: countTexts(0.3), near: countTexts(1.5)});
    """)
    # Four voices per group: far out (threshold 5) none at all, close
    # (threshold 3) all. The point is the DIRECTION, not the exact count.
    assert r["far"] < r["near"], f"thinning has no effect: {r['far']} far, {r['near']} close"


def test_the_same_name_appears_only_once_in_the_overview():
    """In practice there are many more named groups than people — one name
    often sits on several groups. In the overview you're looking for the person, not the fragments.

    Checked WITHOUT "bundle names": if bundling is on, all parts of a
    name already share one centroid, and the question no longer arises
    (voicemap_graph.js). This thinning is the case for the unbundled graph,
    which still exists.
    """
    r = js(
        SHARED_NAMES
        + """
    API.GRAPH.clusterNames = false;
    const names = (s) => {
        const {c, p} = paper();
        API.graphDrawNames(c, s, 0, 400, 4000, 800);
        return p.texts.map(t => t.t);
    };
    output({far: names(0.3), near: names(1.5)});
    """
    )
    assert r["far"].count("me") == 1, f"overview: {r['far']}"
    assert r["far"].count("Alice") == 1
    # Zoomed in, all occurrences belong — that's where you look for fragments.
    assert r["near"].count("me") == 4, f"close: {r['near']}"


def test_when_clustered_the_largest_group_gets_the_label():
    """Not just any one: the label must sit where the person mainly is,
    otherwise the name points at a fragment."""
    r = js(
        SHARED_NAMES
        + """
    API.GRAPH.clusterNames = false;
    const {c, p} = paper();
    API.graphDrawNames(c, 0.3, 0, 400, 4000, 800);
    const me = p.texts.find(t => t.t === 'me');
    // The large "me" group (nr 1) sits at x≈0, the fragments further right.
    output({x: me.x, size: parseFloat(me.font.match(/([\\d.]+)px/)[1])});
    """
    )
    assert r["x"] < 200, "the name is not on the largest group"
    # Second piece of evidence via the font size: it grows with the group size.
    # 60 voices give 18.7 px at this zoom level, the fragments of 9 give 13.8 —
    # 16 cleanly separates the two without rebuilding the formula in the test.
    assert r["size"] > 16


def test_a_name_outside_the_viewport_does_not_use_its_slot():
    """The trap with bundling: if the largest "me" group were right next
    to the image, "me" must not be entirely missing — a visible smaller
    group must then step in."""
    r = js(
        SHARED_NAMES
        + """
    API.GRAPH.clusterNames = false;
    const {c, p} = paper();
    // Window placed so the large group (x≈0) falls out on the left.
    API.graphDrawNames(c, 1.0, -900, 400, 800, 800);
    output({texts: p.texts.map(t => t.t)});
    """
    )
    assert "me" in r["texts"], f"the name has vanished entirely: {r['texts']}"


def test_names_off_means_no_names():
    r = js(
        STRUCTURE
        + """
    API.GRAPH.showNames = false;
    const {c, p} = paper();
    API.graphDrawNames(c, 1.0, 400, 300, 1200, 800);
    output({count: p.texts.length});
    """
    )
    assert r["count"] == 0


# ── The outline ──────────────────────────────────────────────────────────


def test_the_hull_discards_interior_points():
    """The convex hull of a square with a point in the middle is the
    square. If the center point were included, the outline would have a dent."""
    r = js("""
    const px = [0, 10, 10, 0, 5], py = [0, 0, 10, 10, 5];
    output({h: API.graphHull(px, py)});
    """)
    assert sorted(r["h"]) == [0, 1, 2, 3]


def test_the_hull_encloses_every_point():
    """The core promise: NO point of the group lies outside.

    Checked on a random cloud — a point outside the hull would be a
    point outside its own outline.
    """
    r = js("""
    // Fixed seed: a test that passes sometimes and not other times is
    // worse than none at all.
    let z = 12345;
    const rand = () => (z = (z * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff;
    const px = [], py = [];
    for (let i = 0; i < 200; i++) { px.push(rand() * 500); py.push(rand() * 300); }
    const h = API.graphHull(px, py);
    // For each hull edge: do all points lie on the same side?
    let outside = 0;
    for (let i = 0; i < h.length; i++) {
        const a = h[i], b = h[(i + 1) % h.length];
        for (let j = 0; j < px.length; j++) {
            const k = (px[b] - px[a]) * (py[j] - py[a])
                    - (py[b] - py[a]) * (px[j] - px[a]);
            if (k < -1e-6) outside++;
        }
    }
    output({corners: h.length, outside});
    """)
    assert r["corners"] >= 3
    assert r["outside"] == 0, "a point lies outside its own hull"


def test_every_group_gets_exactly_one_outline():
    """Three groups, three lines — and NONE for the loners.

    Group 0 are the unbound voices, scattered over the whole map.
    Their hull would be the whole picture and say nothing.
    """
    r = js(
        STRUCTURE
        + """
    const {c, p} = paper();
    API.graphDrawHulls(c, 1.0, 100, 300, 1200, 800);
    output({strokes: p.strokes, fills: p.fills});
    """
    )
    assert r["strokes"] == 3, "one line per group, loners excluded"
    assert r["fills"] == 3


def test_the_outline_lies_outside_the_points():
    """Padded, not on the edge: the line must not touch the outermost
    points, otherwise it looks crossed out."""
    r = js("""
    const x = [0, 40, 40, 0], y = [0, 0, 40, 40], gv = [1, 1, 1, 1];
    API.mapSet({n: 4, x, y, sx: 1, ox: 0, oy: 0, layer: 'graph'});
    API.GRAPH.groupOf = gv;
    API.GRAPH.groups = [{nr: 1, n: 4, name: 'Alice', mean: 0.9}];
    API.GRAPH.showOutlines = true;
    const {c, p} = paper();
    API.graphDrawHulls(c, 1.0, 100, 100, 800, 600);
    const path = p.curves[p.curves.length - 1];
    const xs = path.map(q => q[0]), ys = path.map(q => q[1]);
    output({minX: Math.min(...xs), maxX: Math.max(...xs),
             minY: Math.min(...ys), maxY: Math.max(...ys)});
    """)
    # Points lie (with offset 100/100) between 100 and 140.
    assert r["minX"] < 100 and r["maxX"] > 140
    assert r["minY"] < 100 and r["maxY"] > 140


def test_two_voices_get_an_outline_too():
    """Two points don't span an area — without a special case, no line
    would come out here at all, and of all things the small groups
    (Alice's case) would be left without an outline."""
    r = js("""
    API.mapSet({n: 2, x: [0, 30], y: [0, 0], sx: 1, ox: 0, oy: 0,
                     layer: 'graph'});
    API.GRAPH.groupOf = [1, 1];
    API.GRAPH.groups = [{nr: 1, n: 2, name: 'Alice', mean: 0.9}];
    API.GRAPH.showOutlines = true;
    const {c, p} = paper();
    API.graphDrawHulls(c, 1.0, 100, 100, 800, 600);
    output({strokes: p.strokes, ellipses: p.ellipses});
    """)
    assert r["strokes"] == 1
    assert r["ellipses"] == 1


def test_outlines_off_means_no_outlines():
    r = js(
        STRUCTURE
        + """
    API.GRAPH.showOutlines = false;
    const {c, p} = paper();
    API.graphDrawHulls(c, 1.0, 100, 300, 1200, 800);
    output({strokes: p.strokes});
    """
    )
    assert r["strokes"] == 0


def test_a_filtered_group_is_dimmed_not_removed():
    """The filter principle also applies to the outlines: non-matches
    get dimmed. If the hull disappeared, you'd no longer see that a
    group even exists there."""
    r = js(
        STRUCTURE
        + """
    // Only group 1 is a match.
    const sight = new Uint8Array(API.GRAPH.groupOf.length);
    API.GRAPH.groupOf.forEach((nr, i) => { if (nr === 1) sight[i] = 1; });
    API.GRAPH.visible = sight;
    API.GRAPH.filtered = true;
    const {c, p} = paper();
    API.graphDrawHulls(c, 1.0, 100, 300, 1200, 800);
    output({strokes: p.strokes, alpha: p.strokeAlpha});
    """
    )
    assert r["strokes"] == 3, "non-matches also keep their outline"
    assert 1.0 in r["alpha"], "the matching group stays fully visible"
    assert min(r["alpha"]) < 0.5, "non-matches must be dimmed"


def test_the_outline_carries_the_group_colour():
    """The outline must carry the same color as the points — a separate
    color scale would claim a second grouping."""
    r = js(
        STRUCTURE
        + """
    const {c, p} = paper();
    API.graphDrawHulls(c, 1.0, 100, 300, 1200, 800);
    output({colours: p.strokeColours,
             expected: [1, 2, 3].map(nr => API.graphColour(nr))});
    """
    )
    for color, expected in zip(sorted(r["colours"]), sorted(r["expected"])):
        assert color.startswith(expected), f"{color} doesn't match {expected}"


def test_with_very_many_nodes_no_hulls():
    """Computing a hull over hundreds of thousands of points per frame would be wasteful
    — the same limit as the point-saving mode in voicemap.js."""
    r = js("""
    const n = 20001, x = [], y = [], gv = [];
    for (let i = 0; i < n; i++) { x.push(i % 500); y.push(i % 300); gv.push(1); }
    API.mapSet({n, x, y, sx: 1, ox: 0, oy: 0, layer: 'graph'});
    API.GRAPH.groupOf = gv;
    API.GRAPH.groups = [{nr: 1, n, name: 'many', mean: 0.8}];
    API.GRAPH.showOutlines = true;
    const {c, p} = paper();
    API.graphDrawHulls(c, 1.0, 0, 0, 1200, 800);
    output({strokes: p.strokes});
    """)
    assert r["strokes"] == 0


def test_a_group_outside_the_viewport_costs_nothing():
    r = js(
        STRUCTURE
        + """
    const {c, p} = paper();
    // Shifted far to the left: everything lies outside.
    API.graphDrawHulls(c, 1.0, -5000, -5000, 1200, 800);
    output({strokes: p.strokes});
    """
    )
    assert r["strokes"] == 0


# ── Search scope: name or path ────────────────────────────────────────────
#
#  Searching for Alice can highlight Erin, too. Not a bug, but a FOLDER:
#  `…/alice/alice phone/messages/…`. In a real library a large share of
#  the matches can come only via the path — other people and unnamed
#  voices that happen to sit there.

SEARCH_CASE = """
const m = [
    {display: 'videos/Archive/2024-06-02.mp4', person: 'Alice',  label: 'SPEAKER_00'},
    {display: 'Archive/alice/alice phone/messages/PTT-1.aac', person: 'Erin', label: 'SPEAKER_01'},
    {display: 'Archive/alice/alice phone/messages/PTT-2.aac', person: '',     label: 'SPEAKER_02'},
    {display: 'Archive/alice/PTT-3.aac',                  person: 'Alice',  label: 'SPEAKER_03'},
    {display: 'videos/Archive/2024-07-01.mp4',           person: 'Dave',  label: 'SPEAKER_04'},
];
API.mapSet({n: m.length, x: [0,1,2,3,4], y: [0,0,0,0,0],
                 sx: 1, ox: 0, oy: 0, layer: 'graph', members: m});
API.GRAPH.groupOf = [1, 2, 2, 1, 3];
API.GRAPH.groups = [{nr: 1, n: 2, name: 'Alice'}, {nr: 2, n: 2, name: 'Erin'},
                     {nr: 3, n: 1, name: 'Dave'}];
const search_it = (text, scope) => {
    API.GRAPH.filter = {text, scope, folder: '', from_value: '', to_value: '', naming: 'all_items'};
    API.graphApplyFilter();
    const out = [];
    for (let i = 0; i < 5; i++) if (API.GRAPH.visible[i]) out.push(i);
    return out;
};
"""


def test_everywhere_also_finds_the_folder_with_the_name():
    """The described behavior — and it stays the default. Whoever searches
    for a folder wants exactly that."""
    r = js(SEARCH_CASE + "output({hits: search_it('alice', 'all_items')});")
    # 0 and 3 via the name, 1 and 2 via the path.
    assert r["hits"] == [0, 1, 2, 3]


def test_name_only_leaves_the_folder_match_out():
    """The answer to the problem."""
    r = js(SEARCH_CASE + "output({hits: search_it('alice', 'person')});")
    assert r["hits"] == [0, 3], "Erin still hangs on the folder name"


def test_path_only_finds_the_folder():
    r = js(SEARCH_CASE + "output({hits: search_it('alice', 'file')});")
    # Alice's own file 3 also sits in the folder — 0 does not.
    assert r["hits"] == [1, 2, 3]


def test_the_reason_per_hit_is_recorded():
    """For the status line: 1 = name, 2 = path, 3 = both."""
    r = js(
        SEARCH_CASE
        + """
    search_it('alice', 'all_items');
    output({reason: Array.from(API.GRAPH.textSource)});
    """
    )
    assert r["reason"] == [1, 2, 2, 3, 0]


def test_without_search_text_there_is_no_reason():
    """Otherwise an old justification would stick to a folder filter."""
    r = js(
        SEARCH_CASE
        + """
    API.GRAPH.filter = {text: '', scope: 'all_items', folder: 'videos', from_value: '',
                        to_value: '', naming: 'all_items'};
    API.graphApplyFilter();
    output({reason: API.GRAPH.textSource});
    """
    )
    assert r["reason"] is None


def test_the_bridge_from_the_file_finds_the_recording_by_id():
    """`mapForFile` asks about a RECORDING. By its NAME it lit every other
    recording of that name, too — in a large library many file names belong
    to more than one recording. The server resolves the recording
    and hands its windows; the map lights exactly those ids."""
    source = MAP_JS.read_text("utf-8")
    body = source[source.index("async function mapLocate(") :]
    body = body[: body.index("\n}\n")]
    assert "/api/speaker/windows?file=" in body
    apply = source[source.index("function mapLocateApply(") :]
    apply = apply[: apply.index("\n}\n")]
    assert "GRAPH.rowOf" in apply and "GRAPH.filter.only" in apply
    assert "graphFilterSet('text'" not in apply
    for f in ("mapForFile", "mapForSpeaker", "mapForSpot"):
        b = source[source.index(f"function {f}(") :]
        assert "mapLocate(" in b[: b.index("\n}\n")], f"{f} goes its own way"


# ── Named: the voice or its group? ────────────────────────────────────────
#
#  Why does "only named" gray out half of the 'me' group, even though
#  those voices are in the 'me' group?
#
#  Because the group is a CLAIM by the machine ("sounds the same") and the
#  assignment is a DETERMINATION ("is me"). The group name is moreover just
#  the most common assigned name in it — a group is already called "me" if
#  one part is assigned. Typically only a fraction of the voices carry one.

NAMING = """
const m = [
    {display: 'a.mp4', person: 'me', label: 'S0'},   // 0: assigned
    {display: 'b.mp4', person: '',    label: 'S1'},   // 1: open, group 'me'
    {display: 'c.mp4', person: '',    label: 'S2'},   // 2: open, group 'me'
    {display: 'd.mp4', person: '',    label: 'S3'},   // 3: open, anonymous group
    {display: 'e.mp4', person: 'Alice', label: 'S4'},   // 4: assigned
];
API.mapSet({n: 5, x: [0,1,2,3,4], y: [0,0,0,0,0],
                 sx: 1, ox: 0, oy: 0, layer: 'graph', members: m});
API.GRAPH.groupOf = [1, 1, 1, 2, 3];
API.GRAPH.groups = [{nr: 1, n: 3, name: 'me'}, {nr: 2, n: 1, name: ''},
                     {nr: 3, n: 1, name: 'Alice'}];
const choose = (b) => {
    API.GRAPH.filter = {text: '', scope: 'all_items', folder: '', from_value: '', to_value: '',
                        naming: b};
    API.graphApplyFilter();
    const out = [];
    for (let i = 0; i < 5; i++) if (API.GRAPH.visible[i]) out.push(i);
    return out;
};
"""


def test_voice_named_means_the_assignment():
    """The existing behavior — it stays, because it's the right answer
    to "which voices are confirmed?"."""
    r = js(NAMING + "output({t: choose('named')});")
    assert r["t"] == [0, 4]


def test_in_a_named_group_includes_the_greyed_out_ones():
    """The reading one expects from a group view."""
    r = js(NAMING + "output({t: choose('group_nr')});")
    assert r["t"] == [0, 1, 2, 4], "the 'me' group must be complete"


def test_open_in_a_named_group_is_the_work_list():
    """Exactly the grayed-out half: in a named group, but itself still
    unassigned. "This is probably me — confirm it."."""
    r = js(NAMING + "output({t: choose('open')});")
    assert r["t"] == [1, 2]


def test_voice_unnamed_also_covers_the_anonymous_group():
    """The difference from "open": point 3 belongs to no named group and
    therefore doesn't belong on the work list, but does belong here."""
    r = js(NAMING + "output({t: choose('unnamed')});")
    assert r["t"] == [1, 2, 3]


def test_all_voices_does_not_filter():
    r = js(
        NAMING
        + """
    API.GRAPH.filter = {text: '', scope: 'all_items', folder: '', from_value: '', to_value: '',
                        naming: 'all_items'};
    API.graphApplyFilter();
    output({filtered: API.GRAPH.filtered, visible: API.GRAPH.visible});
    """
    )
    assert r["filtered"] is False
    assert r["visible"] is None


def test_every_option_of_the_selector_is_wired():
    """A value in the markup that the filter doesn't know would be a
    button with no function — exactly the kind of bug that already
    happened here."""
    import re

    html = (REPO / "whisper_tm" / "static" / "index.html").read_text("utf-8")
    block = html[html.index('id="mapFNaming"') :]
    block = block[: block.index("</select>")]
    value_list = set(re.findall(r'<option value="(\w+)"', block))
    source = MAP_JS.read_text("utf-8")
    for w in value_list - {"all_items"}:
        assert f"f.naming === '{w}'" in source, f"'{w}' is checked nowhere"


# ── Wiring ─────────────────────────────────────────────────────────────


def test_the_switch_is_in_the_markup_and_remembers_the_off_state():
    """A remembered ``false`` must not fall back to ``true`` on the next
    load — exactly this bug already occurred once in the pull-strength
    slider (it showed 1 while 10 was in effect)."""
    html = (REPO / "whisper_tm" / "static" / "index.html").read_text("utf-8")
    assert 'id="mapOutlines"' in html
    assert "graphToggleOutlines(this.checked)" in html

    js_source = MAP_JS.read_text("utf-8")
    # The switch must be set to the remembered state on setup.
    assert "ol.checked = GRAPH.showOutlines" in js_source
    assert "m.outlines !== undefined" in js_source, (
        "`if (m.outlines)` would swallow a remembered false"
    )


def test_hulls_are_drawn_beneath_the_edges():
    """Order carries meaning here: an outline drawn OVER the lines would
    cut the graph into areas and claim boundaries where there are
    connections."""
    voice_map = (REPO / "whisper_tm" / "static" / "js" / "voicemap.js").read_text("utf-8")
    hulls = voice_map.index("graphDrawHulls(g, s, ox, oy, w, h)")
    edges = voice_map.index("graphDrawEdges(g, s, ox, oy, w, h)")
    assert hulls < edges
