"""Drawing load in the graph — what helped, what hurt, and what's gone.

Goal: a lighter rendering variant, since the graph lagged noticeably — and
the bubble view was removed.

Measured in the browser on a large map (several thousand voices, tens of
thousands of edges, about a thousand groups), median over 30 frames:

    ============================  ========
    one simulation step             ~50 ms
    entire drawing                  ~12 ms
      thereof points                 ~3 ms
      thereof names                  ~2 ms
      thereof edges                  ~1 ms
      remainder: outlines            ~6 ms
    ============================  ========

**What remained** (unconditionally, while layout runs): only draw every third
frame, and no outlines or names. Both are ornament on an image that's moving
anyway. The fragile part: if ``running`` is deleted only AFTER the last
``mapDraw()``, exactly these outlines and names are missing at the end.

**What was rolled back.** The search window of repulsion had additionally
been shrunk from ±2 to ±1 cell and the range pulled from 270 to 135. On real
data this tore the layout apart: without far-reaching repulsion the large
clusters no longer repel each other, and weak springs between groups pull them
into a diagonal strand. The light run did not help.

The 25 cells are not waste, they ARE the repulsion. If you really want to shed
the drawing load, you must shed the computation entirely — that's what the tile
view is for (``test_voicemap_tiles.py``).

The second part: the bubble view is gone. We verify that nothing
from it remains pointing into nothing — a button with ``onclick="mapView(…)"``
on a deleted function would be the kind of error only a user finds.
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
STATIC = REPO / "whisper_tm" / "static"
MAP_JS = STATIC / "js" / "voicemap_graph.js"
PANEL_JS = MAP_JS.parent / "voicemap_panel.js"
MAP_BASE = STATIC / "js" / "voicemap.js"
INDEX = STATIC / "index.html"

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
        env={"MAP_JS": str(MAP_JS), "PANEL_JS": str(PANEL_JS), "PATH": os.environ.get("PATH", "")},
    )
    if done.returncode != 0:
        raise AssertionError(f"node failed:\n{done.stderr}")
    lines = [z for z in done.stdout.splitlines() if z.startswith("§§")]
    assert lines, f"no result:\n{done.stdout}\n{done.stderr}"
    return json.loads(lines[-1][2:])


# Four groups, far apart — so repulsion has something to do and neighborhood
# search is not trivially empty.
STRUCTURE = """
const x = [], y = [], gv = [];
const setIt = (nr, n, mx, my) => {
    for (let i = 0; i < n; i++) {
        x.push(mx + (i % 6) * 7); y.push(my + Math.floor(i / 6) * 7); gv.push(nr);
    }
};
setIt(1, 60, 0, 0);
setIt(2, 30, 200, 0);
setIt(3, 12, 0, 200);
setIt(4, 4, 200, 200);
API.mapSet({n: x.length, x, y, sx: 1, ox: 0, oy: 0});
API.GRAPH.groupOf = gv;
API.GRAPH.groups = [
    {nr: 1, n: 60, name: 'Ivan', mean: 0.8},
    {nr: 2, n: 30, name: '', mean: 0.7},
    {nr: 3, n: 12, name: '', mean: 0.7},
    {nr: 4, n: 4, name: '', mean: 0.6},
];
API.GRAPH.edges = new Int32Array([0, 1, 1, 2, 60, 61]);
API.GRAPH.kWeight = new Float32Array([0.9, 0.9, 0.9]);
API.GRAPH.on = true;
API.GRAPH.vx = new Float32Array(x.length);
API.GRAPH.vy = new Float32Array(x.length);
API.GRAPH.steps = 420;
API.graphBuildActive();
"""


@node
class TestOrnamentOnlyAtTheEnd:
    """Outlines and names disappear DURING layout — and only then."""

    def _draw(self, running: bool) -> dict:
        return js(
            STRUCTURE
            + f"""
        API.GRAPH.running = {str(running).lower()};
        API.GRAPH.showOutlines = true;
        API.GRAPH.showNames = true;
        const h = paper(), n = paper();
        API.graphDrawHulls(h.c, 1, 0, 0, 900, 900);
        API.graphDrawNames(n.c, 1, 0, 0, 900, 900);
        output({{outlines: h.p.strokes, texts: n.p.texts.length}});
        """
        )

    def test_no_ornament_while_the_run_is_going(self):
        d = self._draw(running=True)
        assert d["outlines"] == 0, "outlines drawn in the middle of the run"
        assert d["texts"] == 0, "names drawn in the middle of the run"

    def test_complete_again_after_the_run(self):
        """The more important case: the final image must not miss anything."""
        d = self._draw(running=False)
        assert d["outlines"] > 0, "outlines are missing after the run"
        assert d["texts"] > 0, "names are missing after the run"


@node
class TestRepulsion:
    """Neighborhood shrinks — forces stay forces."""

    def _steps(self, how_much: int = 12) -> dict:
        return js(
            STRUCTURE
            + f"""
        API.GRAPH.ticks = 0;
        let last_one = 0;
        for (let i = 0; i < {how_much}; i++) last_one = API.graphStep();
        const X = API.MAP_X(), Y = API.MAP_Y();
        let finite = true;
        for (let i = 0; i < X.length; i++) {{
            if (!Number.isFinite(X[i]) || !Number.isFinite(Y[i])) finite = false;
        }}
        // Scatter within the largest group: does it tighten?
        let sx = 0, sy = 0;
        for (let i = 0; i < 60; i++) {{ sx += X[i]; sy += Y[i]; }}
        sx /= 60; sy /= 60;
        let r = 0;
        for (let i = 0; i < 60; i++) r += Math.hypot(X[i] - sx, Y[i] - sy);
        output({{finite, radius: r / 60, idle: last_one, ticks: API.GRAPH.ticks}});
        """
        )

    def test_computes_without_outliers(self):
        d = self._steps()
        assert d["finite"], "NaN/Infinity in positions"
        assert d["ticks"] == 12

    def test_the_group_clusters(self):
        d = self._steps()
        assert d["radius"] > 0, "group did not cluster"


class TestSource:
    """What can only be checked in the text — the order in the tick."""

    def test_the_final_frame_is_drawn_after_the_reset(self):
        """First ``running = false``, THEN draw.

        The other way around, the error would be invisible until the last
        frame: layout would look nice, and at the end outlines and names
        would be missing.
        """
        q = MAP_JS.read_text(encoding="utf-8")
        # The point where the run ENDS — not where it's aborted (`return`),
        # and not in the switch to tiles.
        i = q.index("GRAPH.arranged = true;")
        before = q[i - 200 : i]
        assert "GRAPH.running = false;" in before, "the flag is not deleted before the final image"
        sort_by = q[i : i + 400]
        assert "mapDraw();" in sort_by, "final image not drawn after reset"
        assert sort_by.index("mapDraw();") < sort_by.index("mapBuildGrid();")

    def test_not_every_frame_in_the_light_run(self):
        q = MAP_JS.read_text(encoding="utf-8")
        assert "GRAPH.ticks % 3 === 0" in q

    def test_the_final_frame_is_not_drawn_twice(self):
        """Drawing twice in the last tick would be ~12 ms for nothing."""
        q = MAP_JS.read_text(encoding="utf-8")
        assert "if (!done && GRAPH.ticks % 3 === 0)" in q

    def test_the_repulsion_still_reaches_far(self):
        """The rollback — learned twice, in two different forms.

        **First**: the grid's search window was shrunk from ±2 to ±1 cell
        and the range pulled from 225 to 135. On real data this tore the
        layout into a diagonal strand: without far-reaching repulsion the
        large clusters no longer repel, and weak springs between groups pull
        them together. The light run did not help.

        **Later**: The grid is gone — repulsion now runs via a quadtree
        (see `test_voicemap_tree.py`). So `weite` and `cell` are gone, but the
        RANGE remains, and it's still what makes the map readable: an attempt
        without it let thousands of groups merge into a single disk.

        So the test moves along — it holds what it always held.
        """
        q = MAP_JS.read_text(encoding="utf-8")
        assert "const rich = 225;" in q, "repulsion range is adjusted or missing"
        assert "const weite = 2;" not in q, (
            "grid was replaced by quadtree — remnants suggest a partial rollback"
        )
        assert "GRAPH.leicht" not in q, (
            "the switch was rolled back — it helped nothing and damaged the layout"
        )

    def test_the_run_ends_when_a_new_one_overtakes_it(self):
        """The run ended based on ``MAP.ebene`` — the level no longer exists.

        Without a replacement, an old tick would keep calculating on fields
        that now belong to a new load.
        """
        q = MAP_JS.read_text(encoding="utf-8")
        assert "const mine = GRAPH.seq;" in q
        i = q.index("if (GRAPH.seq !== mine) {")
        stop = q[i : i + 400]
        assert "GRAPH.running = false;" in stop and "return;" in stop
        # …and starts the layout the new load asked for while it ran: its
        # own call bounced off `running`, and the map stayed black
        assert "if (GRAPH.runPending && GRAPH.doneSeq === GRAPH.seq) graphRun();" in stop
        assert "if (GRAPH.running) { GRAPH.runPending = true; return; }" in q


class TestBubbleViewIsGone:
    """Nothing must remain pointing into nothing."""

    # `mapView` is NOT in this list: the name is back, but for
    # something else — the switch between tiles and graph. What must remain
    # gone is the THING: a second load path with its own endpoint and a
    # second level to click into.
    @pytest.mark.parametrize(
        "name",
        [
            "mapLoad",
            "mapOpenGroup",
            "mapApplyData",
        ],
    )
    def test_the_function_is_gone(self, name):
        for file in (MAP_BASE, MAP_JS):
            q = file.read_text(encoding="utf-8")
            assert f"function {name}(" not in q, f"{name} still lives in {file.name}"

    @pytest.mark.parametrize(
        "name",
        [
            "mapLoad(",
            "mapOpenGroup",
        ],
    )
    def test_nobody_calls_it_any_more(self, name):
        """`mapLoad(` with the parenthesis: `mapLoadGraph` is a different,
        living function and must not trip this check."""
        for file in (MAP_BASE, MAP_JS, INDEX):
            q = file.read_text(encoding="utf-8")
            assert name not in q, f"{name} still called in {file.name}"

    def test_no_second_load_path_in_the_frontend(self):
        """The actual point. Tiles use the same data as the graph; bubbles
        fetched their own."""
        for file in (MAP_BASE, MAP_JS, MAP_JS.parent / "voicemap_tiles.js"):
            # Code only, no comments: the header tells the history
            code = "\n".join(
                z
                for z in file.read_text(encoding="utf-8").splitlines()
                if not z.lstrip().startswith(("*", "//", "/*"))
            )
            assert "/api/voicemap/overview" not in code, file.name
            assert "/api/voicemap/group/" not in code, file.name

    def test_no_layer_any_more(self):
        """``MAP.ebene`` was the switch between views.

        A remaining comparison ``ebene === 'groups'`` would now always be
        false — so dead code that looks like it does something.
        """
        for file in (MAP_BASE, MAP_JS):
            # Comment lines may mention it — the reason why the abort now
            # hangs on `seq` belongs exactly there.
            code = [
                z
                for z in file.read_text(encoding="utf-8").splitlines()
                if not z.lstrip().startswith(("*", "//", "/*"))
            ]
            assert "MAP.ebene" not in "\n".join(code), f"MAP.ebene still lives in {file.name}"

    def test_the_overview_button_is_gone_from_the_markup(self):
        """`mapBack` led from member to bubble level.

        The switch itself (`map-view-btn`) is back — it now chooses
        between tiles and graph, that is, between two views of the SAME data.
        There is no level to go back to.
        """
        q = INDEX.read_text(encoding="utf-8")
        assert "mapBack" not in q


# ── Group Index and Treemap ─────────────────────────────────────────────


def test_the_group_name_is_not_looked_up_linearly():
    """**The cause of a "this page is slowing down Firefox" message.**

    Going from the speaker map to Assign made Firefox report the page as
    slowing down the browser.

    In `graphGroupName` there was a `find` over `GRAPH.groups`, in
    `graphColour` additionally a `findIndex`. Both are called ONCE PER VOICE —
    from tile layout and from drawing. With tens of thousands of voices and
    groups, that's on average billions of comparisons per frame.

    Measured on a large library, threshold 0.94:

        Load map        ~15,000 ms  →   ~260 ms
        Base image       ~6,000 ms  →   ~120 ms

    It only became apparent with the new threshold: at 0.71 it was the same
    error with a factor six smaller.
    """
    source = (
        Path(__file__).resolve().parent.parent
        / "whisper_tm"
        / "static"
        / "js"
        / "voicemap_graph.js"
    ).read_text(encoding="utf-8")
    for place in ("function graphGroupName", "function graphColour("):
        body = source[source.index(place) :]
        body = body[: body.index("\n}\n")]
        assert ".find(" not in body and ".findIndex(" not in body, (
            f"{place} searches linearly again — costs per voice"
        )
        assert "graphGroupIndex()" in body


def test_the_index_follows_a_new_group_list():
    """It hangs on the IDENTITY of the list, not its length.

    A same-length new list is the dangerous case: anyone checking length will
    then deliver the names from the PREVIOUS run — silently.
    """
    d = js("""
        API.mapSet({n: 2, name: ['', ''], members: [{}, {}]});
        API.GRAPH.groupOf = new Int32Array([1, 2]);
        API.GRAPH.groups = [{nr: 1, n: 1, name: 'Alice'}, {nr: 2, n: 1, name: ''}];
        const first = API.graphGroupName(1);
        // Same length, different content — the trap.
        API.GRAPH.groups = [{nr: 1, n: 1, name: 'Carol'}, {nr: 2, n: 1, name: ''}];
        const after = API.graphGroupName(1);
        // And renaming IN the entry must work without rebuild.
        API.GRAPH.groups[0].name = 'Ivan';
        output({first: first, after, later: API.graphGroupName(1),
                 empty: API.graphGroupName(2), unknown: API.graphGroupName(99)});
    """)
    assert d["first"] == "Alice"
    assert d["after"] == "Carol", "index sticks to old list"
    assert d["later"] == "Ivan", "renaming doesn't take effect"
    assert d["empty"] == "" and d["unknown"] == ""


# ── Colors ──────────────────────────────────────────────────────────────


def test_the_palette_is_large_enough_and_without_duplicates():
    """More colors, so groups can be told apart better — with too few,
    different names regularly ended up in the same color."""
    d = js("output({n: API.GRAPH_COLORS.length, distinct: new Set(API.GRAPH_COLORS).size});")
    assert d["n"] >= 24, f"only {d['n']} colors"
    assert d["n"] == d["distinct"], "palette contains a color twice"


def _world(names):
    """A map from named groups — one voice per group is enough."""
    listing = ", ".join(
        f"{{nr: {i + 1}, n: {len(names) - i}, name: {n!r}}}" for i, n in enumerate(names)
    ).replace("'", '"')
    return f"""
        API.mapSet({{n: {len(names)}, name: [{",".join(['""'] * len(names))}],
                          members: [{",".join(["{}"] * len(names))}]}});
        API.GRAPH.groupOf = new Int32Array([{",".join(str(i + 1) for i in range(len(names)))}]);
        API.GRAPH.groups = [{listing}];
    """


def test_two_names_never_get_the_same_colour():
    """The core guarantee.

    The hash alone may throw two names to the same color — with ten names
    and 14 colors, a collision by the birthday paradox is more likely than
    not. Now the hash is only the DESIRED slot.
    """
    names = [
        "Alice",
        "Bob",
        "Carol",
        "Dave",
        "Grace",
        "Erin",
        "Heidi",
        "Ivan",
        "Frank",
        "Judy",
        "Mallory",
        "Oscar",
    ]
    d = js(
        _world(names)
        + """
        const colours = {};
        for (let nr = 1; nr <= API.GRAPH.groups.length; nr++)
            colours[API.GRAPH.groups[nr - 1].name] = API.graphColour(nr);
        output({colours, distinct: new Set(Object.values(colours)).size});
    """
    )
    assert d["distinct"] == len(names), (
        f"only {d['distinct']} different colors for {len(names)} names: {d['colours']}"
    )


def test_the_colour_hangs_on_the_name_not_the_rank():
    """Otherwise "Alice" changes color as soon as another group passes her — and
    you search for her on the map every time. That was the whole reason to tie
    color to name."""
    d = js(
        _world(["Alice", "Bob", "Carol"])
        + """
        const before = API.graphColour(1);
        // Same names, different order and different sizes.
        API.GRAPH.groups = [{nr: 3, n: 9, name: 'Carol'},
                             {nr: 1, n: 5, name: 'Alice'},
                             {nr: 2, n: 1, name: 'Bob'}];
        output({before, after: API.graphColour(1)});
    """
    )
    assert d["before"] == d["after"], "Alice changed color"


def test_unnamed_groups_inherit_no_colour_from_a_name():
    """Otherwise the largest unnamed group would have "Alice"'s blue — and the
    map would claim a relationship that doesn't exist."""
    d = js("""
        API.mapSet({n: 4, name: ['', '', '', ''],
                         members: [{}, {}, {}, {}]});
        API.GRAPH.groupOf = new Int32Array([1, 2, 3, 4]);
        API.GRAPH.groups = [{nr: 1, n: 9, name: 'Alice'}, {nr: 2, n: 8, name: ''},
                             {nr: 3, n: 7, name: 'Bob'}, {nr: 4, n: 6, name: ''}];
        output({alice: API.graphColour(1), a: API.graphColour(2),
                 bob: API.graphColour(3), b: API.graphColour(4)});
    """)
    assert len({d["alice"], d["a"], d["bob"], d["b"]}) == 4, d


def test_a_new_name_gets_its_colour_after_the_reset():
    """When naming, the index is discarded — without it the freshly named
    group would stay on the color it had as unnamed."""
    d = js(
        _world(["Alice", "Bob"])
        + """
        const before = API.graphColour(2);
        API.GRAPH.groups[1].name = 'Carol';
        API.graphDropGroupIndex();
        output({before, after: API.graphColour(2),
                 alice: API.graphColour(1)});
    """
    )
    assert d["after"] != d["alice"], "new Carol has Alice's color"


# ── From Transcript to a Location ───────────────────────────────────────


def _provide_world(windows):
    """A recording with given windows: [(label, start_sec), …]."""
    with_ = ", ".join(
        f"{{label: '{k}', start_sec: {t}, display: 'a.mp4', abs_path: '/x/a.mp4', person: ''}}"
        for k, t in windows
    )
    n = len(windows)
    return f"""
        API.mapSet({{n: {n}, name: [{",".join(["''"] * n)}],
                          members: [{with_}]}});
        API.GRAPH.groupOf = new Int32Array([{",".join(["1"] * n)}]);
        API.GRAPH.groups = [{{nr: 1, n: {n}, name: ''}}];
        API.GRAPH.on = true;
        API.GRAPH.visible = null;
    """


def test_the_position_finds_the_window_of_its_own_label():
    """The core: a jump from the transcript to the chunk in the map.

    A window is 30 s NET speech, glued from multiple locations — not a block
    of 30 seconds of recording. "Which window covers second t?" means: within
    THIS speaker ID, the last one that doesn't start after t. Windows of two
    speakers almost always overlap.
    """
    d = js(
        _provide_world(
            [
                ("SPEAKER_00", 0),
                ("SPEAKER_01", 10),
                ("SPEAKER_00", 120),
                ("SPEAKER_01", 200),
                ("SPEAKER_00", 400),
            ]
        )
        + """
        const w = {};
        for (const t of [5, 130, 500, 399, 205])
            w[t] = API.mapSpotWindow('SPEAKER_00', t).hits;
        w.foreign = API.mapSpotWindow('SPEAKER_01', 130).hits;
        w.counts = API.mapSpotWindow('SPEAKER_00', 130);
        output(w);
    """
    )
    assert d["5"] == 0, "second 5 belongs in first window"
    assert d["130"] == 2, "second 130 belongs in second window of ID"
    assert d["399"] == 2, "shortly before next window, old still applies"
    assert d["500"] == 4, "last window covers everything after"
    # 205 s: SPEAKER_01 has a window there, SPEAKER_00 doesn't — still
    # stays with window from SPEAKER_00 at second 120.
    assert d["205"] == 2, "pause belongs to previous window of THIS ID"
    assert d["foreign"] == 1, "other ID has its own sequence"
    assert d["counts"]["ofLabel"] == 3 and d["counts"]["inFile"] == 5


def test_a_position_before_the_first_window_takes_the_first():
    """A short interjection at the start doesn't always make it into a window —
    then the first is the honest answer, not "nothing found"."""
    d = js(
        _provide_world([("SPEAKER_00", 60), ("SPEAKER_00", 300)])
        + """
        output(API.mapSpotWindow('SPEAKER_00', 12));
    """
    )
    assert d["hits"] == 0


def test_without_a_window_for_the_label_nothing_is_highlighted():
    """The dangerous convenience: the nearest FOREIGN window would be easy to
    take — and would be marked as the sought voice. That's a different person."""
    d = js(
        _provide_world([("SPEAKER_01", 0), ("SPEAKER_02", 60)])
        + """
        output(API.mapSpotWindow('SPEAKER_09', 30));
    """
    )
    assert d["hits"] == -1, "fell back to foreign voice"
    assert d["ofLabel"] == 0 and d["inFile"] == 2


def test_the_label_shows_the_span_of_the_window():
    """Locating a line seemed to land a bit before it.

    Example: the line is at 10:54:02, window 9 hit from 10:53:22 to
    10:54:10 — so correct. The sample just starts at the BEGINNING of this
    span, 40 seconds earlier. This is not inaccuracy, it's the design: 30
    seconds net speech from 48 seconds of recording.

    If you only read "window at 10:53", you think it's a miss. With the span
    next to it, you see at once that your location is in it.
    """
    d = js("""
        output({
            span: API.mapSpotText('SPEAKER_17', 39242, true, 20, 467,
                                        39202, 39250),
            open_count:  API.mapSpotText('SPEAKER_17', 39242, true, 20, 467,
                                        39202, Infinity),
        });
    """)
    assert "10:53:22" in d["span"] and "10:54:10" in d["span"], d["span"]
    assert "10:54:02" in d["span"], "own location is missing"
    # The last window of an ID has no successor.
    assert "from 10:53:22" in d["open_count"], d["open_count"]


def test_the_clock_time_has_hours():
    """It used to say "648:06" — 648 minutes. Correct math and unreadable:
    no one calculates back to 10:48 mentally, and in the transcript next to it
    the time already has hours."""
    d = js("""
        output({long: API.mapClockTime(39242), short: API.mapClockTime(125),
                 null: API.mapClockTime(0), junk: API.mapClockTime(-5)});
    """)
    assert d["long"] == "10:54:02", d
    assert d["short"] == "2:05" and d["null"] == "0:00" and d["junk"] == "0:00"


def test_the_span_ends_at_the_next_window_of_the_same_label():
    """Window k ends where k+1 starts — windows of ONE ID follow their speech
    in sequence. The window of another ID in between changes nothing."""
    d = js(
        _provide_world(
            [("SPEAKER_00", 0), ("SPEAKER_01", 60), ("SPEAKER_00", 120), ("SPEAKER_00", 400)]
        )
        + """
        output({middle: API.mapSpotWindow('SPEAKER_00', 130),
                 lastItem: API.mapSpotWindow('SPEAKER_00', 500)});
    """
    )
    assert d["middle"]["from_value"] == 120 and d["middle"]["to_value"] == 400, d["middle"]
    assert d["lastItem"]["to_value"] is None or d["lastItem"]["to_value"] > 1e6, d["lastItem"]


def test_the_sentence_on_the_label_distinguishes_four_situations():
    """Same information for different situations sends the user the wrong way —
    same lesson as with `mapHighlightText`.

    * Window found  → state the time
    * ID has windows, but none until now
    * ID has none, but recording has
    * recording has no voices at all
    """
    d = js("""
        output({
            hits:  API.mapSpotText('SPEAKER_00', 3725, true, 4, 9),
            tooEarly:  API.mapSpotText('SPEAKER_00', 12, false, 4, 9),
            without:     API.mapSpotText('SPEAKER_07', 12, false, 0, 9),
            nothing: API.mapSpotText('SPEAKER_07', 12, false, 0, 0),
        });
    """)
    # With hours: "1:02:05" instead of "62:05".
    assert "1:02:05" in d["hits"], d["hits"]
    assert "SPEAKER_00" in d["hits"]
    assert "no window before 0:12" in d["tooEarly"], d["tooEarly"]
    assert "9 others" in d["without"], d["without"]
    assert "at all" in d["nothing"], d["nothing"]
    assert len({d["hits"], d["tooEarly"], d["without"], d["nothing"]}) == 4


def test_the_button_is_in_the_transcript_and_passes_the_speaker():
    """Without the ID the jump would be ambiguous: at the same second there are
    windows for multiple speakers, and `SPEAKER_06` is just a cut marker WITHIN
    this recording."""
    fv = (
        Path(__file__).resolve().parent.parent / "whisper_tm" / "static" / "js" / "fileview.js"
    ).read_text(encoding="utf-8")
    assert "mapForSpot(" in fv, "button is missing in transcript"
    call_it = fv[fv.index("mapForSpot(") :]
    call_it = call_it[: call_it.index("\n")]
    assert "${start}" in call_it and "s.speaker" in call_it, call_it
    graph = (
        Path(__file__).resolve().parent.parent
        / "whisper_tm"
        / "static"
        / "js"
        / "voicemap_graph.js"
    ).read_text(encoding="utf-8")
    assert "function mapForSpot" in graph


def test_the_jump_does_not_divert_to_a_foreign_voice():
    """The dangerous convenience: if the ID has no window, the nearest foreign
    window would be easy to take — and would be highlighted as the sought
    voice. That's a different person."""
    graph = (
        Path(__file__).resolve().parent.parent
        / "whisper_tm"
        / "static"
        / "js"
        / "voicemap_graph.js"
    ).read_text(encoding="utf-8")
    body = graph[graph.index("function mapForSpot") :]
    body = body[: body.index("\nfunction mapSpotText")]
    assert "if (label && m.label !== label) continue;" in body, "the ID is not enforced"
