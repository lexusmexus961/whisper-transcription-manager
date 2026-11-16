"""The quadtree of repulsion — why it is there and what it must hold.

**The reason.** With tens of thousands of voices the map outgrew its old
repulsion. Repulsion used to run over a fixed grid of 90-px cells; each node
repelled what was in its ±2 cells. On a map of tens of thousands of voices
only a few hundred cells are in use, the densest holding thousands of nodes and
an average one about a hundred; one step took seconds (drawing meanwhile:
well under 100 ms).

So the grid did nothing anymore — at ~100 nodes per cell, each sees thousands of
neighbors, making tens of millions of pairs per frame: about one frame every few
seconds.

**The second, more important finding.** The same grid DISTORTED the map.
Search was via a square of ±2 cells, clipped circularly at 225 px — the two
didn't fit together, and anyone at an angle at the edge either fell in the
window or didn't, depending on where in their cell they were. The largest ball
thus became a rectangle with straight edges instead of a disk.

**What did NOT help.** Pure Barnes-Hut without range. It was fast and round,
but useless: without limits the system behaves like a uniformly charged disk,
all groups melted into ONE circle. Measured by the axis ratio of the
largest group:

    grid (before)             visibly angular, structure there
    tree without range        1.54  — a single disk, structure gone
    tree WITH range           1.01  — round, groups distinguishable

So range was never the error, just the shape of the boundary. The tree discards
a subtree when its nearest corner is too far away — that is perfectly round and
the same in all directions.

These tests safeguard both: that the tree is correct (otherwise the forces would
be wrong without anyone seeing it) and that the approximation stays close enough
to the exact calculation.
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
        f"const {{API}} = require({str(HARNESS)!r});\n"
        "const output = (o) => console.log('§§' + JSON.stringify(o));\n"
        f"{body}\n"
    )
    done = subprocess.run(
        ["node", "-e", source],
        capture_output=True,
        text=True,
        timeout=120,
        env={"MAP_JS": str(MAP_JS), "PANEL_JS": str(PANEL_JS), "PATH": os.environ.get("PATH", "")},
    )
    if done.returncode != 0:
        raise AssertionError(f"node failed:\n{done.stderr}")
    lines = [z for z in done.stdout.splitlines() if z.startswith("§§")]
    assert lines, f"no result:\n{done.stdout}\n{done.stderr}"
    return json.loads(lines[-1][2:])


# A scatter field that is the same for all tests. Own randomness instead of
# Math.random: a test that sees different numbers on every run eventually
# reports an error nobody can reproduce.
SCATTER_FIELD = """
function scatter(n) {
    let z = 12345;
    const w = () => (z = (z * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff;
    const X = new Float32Array(n), Y = new Float32Array(n);
    for (let i = 0; i < n; i++) { X[i] = w() * 2000; Y[i] = w() * 2000; }
    return {X, Y};
}
function build(n, X, Y) {
    API.mapSet({n, x: X, y: Y});
    API.GRAPH.active = null;                 // everyone participates
    API.GRAPH.groupOf = new Int32Array(n);
    return API.treeBuild(X, Y, n);
}
"""


class TestTheTreeIsCorrect:
    """A wrong tree isn't noticeable — the map would just look "somehow
    different". Hence the hard invariants here."""

    def test_every_point_is_in_it_exactly_once(self):
        """The leaves hold linked lists. A point linked in twice
        repels twice; one that's missing, not at all."""
        d = js(
            SCATTER_FIELD
            + """
        const n = 5000, {X, Y} = scatter(n);
        build(n, X, Y);
        const B = API.TREE;
        const seen = new Int32Array(n);
        let leaves = 0;
        for (let k = 0; k < B.cnt; k++) {
            if (B.kind[4*k] !== -1) continue;
            leaves++;
            let q = B.first[k];
            while (q !== -1) { seen[q]++; q = B.next_one[q]; }
        }
        let missing = 0, twice = 0;
        for (let i = 0; i < n; i++) {
            if (seen[i] === 0) missing++;
            if (seen[i] > 1) twice++;
        }
        output({missing, twice, leaves, nodes: B.cnt});
        """
        )
        assert d["missing"] == 0, f"{d['missing']} points are in no leaf"
        assert d["twice"] == 0, f"{d['twice']} points are in one twice"

    def test_the_mass_of_the_root_is_the_number_of_points(self):
        d = js(
            SCATTER_FIELD
            + """
        const n = 3000, {X, Y} = scatter(n);
        build(n, X, Y);
        output({root: API.TREE.mass[0], n});
        """
        )
        assert d["root"] == d["n"]

    def test_every_parent_carries_the_sum_of_its_children(self):
        """The centroids are computed bottom-up by iterating backward
        over the node numbers — that only holds as long as a child
        ALWAYS has a higher number than its parent."""
        d = js(
            SCATTER_FIELD
            + """
        const n = 4000, {X, Y} = scatter(n);
        build(n, X, Y);
        const B = API.TREE;
        let skewed = 0, childBeforeParent = 0;
        for (let k = 0; k < B.cnt; k++) {
            if (B.kind[4*k] === -1) continue;
            let m = 0, sx = 0, sy = 0;
            for (let c = 0; c < 4; c++) {
                const kk = B.kind[4*k+c];
                if (kk <= k) childBeforeParent++;
                m += B.mass[kk]; sx += B.sx[kk]; sy += B.sy[kk];
            }
            if (m !== B.mass[k]) skewed++;
            if (Math.abs(sx - B.sx[k]) > 1e-6) skewed++;
            if (Math.abs(sy - B.sy[k]) > 1e-6) skewed++;
        }
        output({skewed, childBeforeParent});
        """
        )
        assert d["childBeforeParent"] == 0, "a child has a smaller number than its parent"
        assert d["skewed"] == 0

    def test_a_point_lies_in_the_cell_that_holds_it(self):
        """If the quartering were wrong, a point would sit in the
        neighboring quadrant — and the cell distance used for cutoff
        would be a lie."""
        d = js(
            SCATTER_FIELD
            + """
        const n = 2000, {X, Y} = scatter(n);
        build(n, X, Y);
        const B = API.TREE;
        let outside_cell = 0;
        for (let k = 0; k < B.cnt; k++) {
            if (B.kind[4*k] !== -1) continue;
            let q = B.first[k];
            while (q !== -1) {
                const s = B.page[k];
                if (X[q] < B.x0[k] - 1e-6 || X[q] > B.x0[k] + s + 1e-6
                 || Y[q] < B.y0[k] - 1e-6 || Y[q] > B.y0[k] + s + 1e-6) outside_cell++;
                q = B.next_one[q];
            }
        }
        output({outside_cell});
        """
        )
        assert d["outside_cell"] == 0

    def test_coincident_points_do_not_blow_it_up(self):
        """Two voices can lie exactly on top of each other — same
        recording, same start point. Without a depth limit, the tree
        would keep splitting such pairs down to floating-point precision
        and run out of memory."""
        d = js(
            SCATTER_FIELD
            + """
        const n = 600;
        const X = new Float32Array(n).fill(500), Y = new Float32Array(n).fill(500);
        X[0] = 0; Y[0] = 0;                     // one off to the side, otherwise cell 0 is huge
        const ok = build(n, X, Y);
        output({ok, nodes: API.TREE.cnt, root_mass: API.TREE.mass[0]});
        """
        )
        assert d["ok"] is True
        assert d["root_mass"] == 600
        assert d["nodes"] < 400, f"{d['nodes']} tree nodes for 600 identical points"

    def test_below_two_points_there_is_nothing_to_do(self):
        d = js(
            SCATTER_FIELD
            + """
        const X = new Float32Array([1]), Y = new Float32Array([1]);
        output({ok: build(1, X, Y)});
        """
        )
        assert d["ok"] is False

    def test_hidden_ones_do_not_get_in(self):
        """Hidden nodes don't take part in the force interplay — they
        neither repel nor are repelled (see the header of
        voicemap_graph.js). If they were in the tree, they'd do both."""
        d = js(
            SCATTER_FIELD
            + """
        const n = 1000, {X, Y} = scatter(n);
        API.mapSet({n, x: X, y: Y});
        API.GRAPH.groupOf = new Int32Array(n);
        const a = new Uint8Array(n);
        for (let i = 0; i < n; i++) a[i] = i % 2;       // every other one
        API.GRAPH.active = a;
        API.treeBuild(X, Y, n);
        output({root_mass: API.TREE.mass[0]});
        """
        )
        assert d["root_mass"] == 500


class TestTheApproximation:
    """Barnes–Hut doesn't compute exactly. The only question is how far off."""

    def test_it_matches_the_exact_force(self):
        """Against the "everyone against everyone" computation, on a
        field small enough for that. The direction and magnitude of the
        total force per node are compared — not individual
        contributions, since summarizing exactly those is the point.

        5% deviation is generous and still meaningful: a sign error, a
        forgotten mass, or a wrong cell throws the force off by a
        multiple, not by percentages.
        """
        d = js(
            SCATTER_FIELD
            + """
        const n = 900, {X, Y} = scatter(n);
        const strength = 460, reach = 225, limit = reach * reach;

        // Exact: everyone against everyone, with the same range and the
        // same falloff as in the force interplay.
        const eFx = new Float64Array(n), eFy = new Float64Array(n);
        for (let i = 0; i < n; i++) {
            for (let j = 0; j < n; j++) {
                if (i === j) continue;
                const ex = X[i]-X[j], ey = Y[i]-Y[j];
                const d2 = ex*ex + ey*ey;
                if (d2 < 1 || d2 > limit) continue;
                const falloff = 1 - d2/limit;
                const f = strength / d2 * falloff * falloff;
                eFx[i] += ex*f; eFy[i] += ey*f;
            }
        }
        /* Attach the pull toward image center here as well.
         *
         * `graphStep` always computes it in, and without a canvas
         * that's 1200x800. Anyone who leaves it out in the comparison
         * isn't measuring the approximation but this pull instead — on
         * the first run, that's exactly why the test was off. Cooling
         * and clamping are pure factors on the whole vector and don't
         * change the DIRECTION; only that matters below. */
        for (let i = 0; i < n; i++) {
            eFx[i] += (1200/2 - X[i]) * 0.0006;
            eFy[i] += (800/2 - Y[i]) * 0.0006;
        }

        // Approximated: via graphStep, but all other forces silenced.
        API.mapSet({n, x: Float32Array.from(X), y: Float32Array.from(Y)});
        API.GRAPH.active = null;
        API.GRAPH.groupOf = new Int32Array(n);     // no group forces
        API.GRAPH.edges = new Int32Array(0);        // no springs
        API.GRAPH.kWeight = new Float32Array(0);
        API.GRAPH.density = 1.0; API.GRAPH.drag = 1.0;
        API.GRAPH.vx = new Float32Array(n); API.GRAPH.vy = new Float32Array(n);
        API.GRAPH.ticks = 0;
        API.graphStep();
        const gFx = API.GRAPH.vx, gFy = API.GRAPH.vy;

        // graphStep attaches center pull, cooling and clamping. The
        // cooling is a pure factor; the center pull is tiny. For a fair
        // comparison, the common factor is divided out:
        // DIRECTIONS and the ratio of magnitudes are compared.
        let n_ok = 0, angleError = 0, worst = 0;
        let seR = 0, sgR = 0;
        for (let i = 0; i < n; i++) {
            const eMag = Math.hypot(eFx[i], eFy[i]), gMag = Math.hypot(gFx[i], gFy[i]);
            /* Nodes with a nearly vanishing total force stay excluded.
             *
             * For them, the contributions nearly cancel out; a tiny
             * absolute error then rotates the direction by up to 180°,
             * without anything actually being wrong. On the first run,
             * that's exactly why the test reported "169.92° off" — with
             * an average error of 4.45°. The threshold is small compared
             * to the typical force (several hundred) and only cuts off
             * the noise. */
            if (eMag < 0.05 || gMag < 1e-9) continue;
            n_ok++; seR += eMag; sgR += gMag;
            const cos = (eFx[i]*gFx[i] + eFy[i]*gFy[i]) / (eMag*gMag);
            const w = Math.acos(Math.max(-1, Math.min(1, cos))) * 180/Math.PI;
            angleError += w;
            if (w > worst) worst = w;
        }
        output({
            compared: n_ok,
            angle_mean: +(angleError/n_ok).toFixed(2),
            angle_worst: +worst.toFixed(2),
            magnitude_ratio: +(sgR/seR).toFixed(3),
        });
        """
        )
        # Measured at θ=0.8: 3.35°. The threshold leaves room for other
        # machines, but triggers as soon as someone raises θ noticeably
        # (1.1 → 10.11°). For comparison: 0.5 → 1.30°, 0.9 → 4.07°.
        assert d["compared"] > 500, d
        assert d["angle_mean"] < 6.0, (
            f"direction off by {d['angle_mean']}° on average — is TREE_THETA set higher than 0.8?"
        )

    def test_ONE_neighbour_acts_equally_from_every_direction(self):
        """THE test for the incident — and the sharpest one there is here.

        A single neighbor at a fixed distance must push exactly equally
        hard from every direction. Nothing more can be said about
        "direction-independent", and nothing less suffices.

        The old grid failed here. It searched over a SQUARE of ±2 cells
        of 90 px each and cut off circularly at 225 px. A neighbor 200 px
        away is 2.2 cells distant — depending on where the test point
        sits WITHIN its cell, that's a difference of 2 or 3 cells. In the
        second case it lay outside the window and had no effect at all.
        So from some directions came the full force, from others zero —
        and exactly that is what pulled the largest ball into a rectangle
        with straight edges.

        The test doesn't depend on any approximation quality: it holds
        for any `TREE_THETA` and only measures the shape of the range.
        """
        d = js(
            SCATTER_FIELD
            + """
        const MX = 1200/2, MY = 800/2, R = 200, W = 72;
        const value_list = [];
        for (let s = 0; s < W; s++) {
            const w = 2*Math.PI*s/W;
            const X = new Float32Array([MX, MX + Math.cos(w)*R]);
            const Y = new Float32Array([MY, MY + Math.sin(w)*R]);
            API.mapSet({n: 2, x: X, y: Y});
            API.GRAPH.active = null;
            API.GRAPH.groupOf = new Int32Array(2);
            API.GRAPH.edges = new Int32Array(0);
            API.GRAPH.kWeight = new Float32Array(0);
            API.GRAPH.density = 1.0; API.GRAPH.drag = 0;
            API.GRAPH.vx = new Float32Array(2); API.GRAPH.vy = new Float32Array(2);
            API.GRAPH.ticks = 0;
            API.graphStep();
            value_list.push(Math.hypot(API.GRAPH.vx[0], API.GRAPH.vy[0]));
        }
        const large = Math.max(...value_list), small = Math.min(...value_list);
        output({directions: W, largest: +large.toFixed(6),
                 smallest: +small.toFixed(6),
                 span_percent: +(100*(large-small)/large).toFixed(3),
                 had_zeros: small < large*0.01});
        """
        )
        assert d["had_zeros"] is False, (
            "no repulsion at all acted from at least one direction — "
            "exactly the bug of the old search window"
        )
        assert d["span_percent"] < 1.0, (
            f"the force varies by {d['span_percent']} % depending on direction"
        )

    def test_a_ring_largely_cancels_itself_out(self):
        """The same question, but with 720 neighbors — and thereby also a
        measure of how much noise the approximation leaves behind.

        What's checked is the force on a node at the center of a ring.
        Mathematically it must be zero; the approximation leaves a
        residual. Measured against what HALF a ring half pushes:

            θ=0.4/0.5   under 5%
            θ≥0.6       around 37%   (the tree descends one level less deep)

        The ring sits at IMAGE CENTER (without a canvas: 1200×800).
        There, the pull toward the center that `graphStep` always
        attaches is exactly zero — otherwise this test would measure that
        instead of the repulsion. On the first run, that's exactly why it
        was off by a factor of 84.
        """
        d = js(
            SCATTER_FIELD
            + """
        // A ring of 720 points around the center, plus the test point itself.
        const MX = 1200/2, MY = 800/2;
        const R = 150, m = 720, n = m + 1;
        const X = new Float32Array(n), Y = new Float32Array(n);
        X[0] = MX; Y[0] = MY;
        for (let i = 0; i < m; i++) {
            const w = 2*Math.PI*i/m;
            X[i+1] = MX + Math.cos(w)*R;
            Y[i+1] = MY + Math.sin(w)*R;
        }
        API.mapSet({n, x: X, y: Y});
        API.GRAPH.active = null;
        API.GRAPH.groupOf = new Int32Array(n);
        API.GRAPH.edges = new Int32Array(0);
        API.GRAPH.kWeight = new Float32Array(0);
        API.GRAPH.density = 1.0; API.GRAPH.drag = 0;      // no center pull
        API.GRAPH.vx = new Float32Array(n); API.GRAPH.vy = new Float32Array(n);
        API.GRAPH.ticks = 0;
        API.graphStep();

        // How hard does the ring push away A SINGLE neighbor? That's what
        // determines whether the residual force at the center is small.
        const remainder = Math.hypot(API.GRAPH.vx[0], API.GRAPH.vy[0]);
        const one_by_one = 460 / (R*R) * Math.pow(1 - (R*R)/(225*225), 2);
        // What does HALF a ring half push? The residual is measured against
        // that — "small compared to a single neighbor" wouldn't hold for
        // 720 neighbors and an approximation, nor is that the question.
        const half_ring = one_by_one * m * 2/Math.PI;
        output({remainder: +remainder.toFixed(4), one_by_one: +one_by_one.toFixed(5),
                 one_sided: +half_ring.toFixed(3),
                 share_percent: +(100*remainder/half_ring).toFixed(1),
                 ratio: +(remainder/one_by_one).toFixed(1)});
        """
        )
        # The threshold matches the configured θ=0.8 (measured ≈37%) and
        # leaves room upward. It triggers if someone raises θ noticeably —
        # at 1.1 the directional error grows to double.
        #
        # The actual claim about direction-independence is made by the
        # test above; this one additionally measures the noise.
        assert d["share_percent"] < 50.0, (
            f"residual force at the ring's center is {d['share_percent']} % "
            f"of what a ring half pushes — is TREE_THETA too high?"
        )

    def test_beyond_the_range_nothing_acts(self):
        """The range isn't decoration, it's the reason the map shows
        groups at all: without it, on a large map all groups
        merge into a single disk (measured axis ratio 1.54 instead of
        1.01).

        The test point sits at image center, so the pull there doesn't
        move it and the measured force is really the repulsion."""
        d = js(
            SCATTER_FIELD
            + """
        const n = 2;
        const X = new Float32Array([600, 1200]), Y = new Float32Array([400, 400]);
        API.mapSet({n, x: X, y: Y});
        API.GRAPH.active = null;
        API.GRAPH.groupOf = new Int32Array(n);
        API.GRAPH.edges = new Int32Array(0);
        API.GRAPH.kWeight = new Float32Array(0);
        API.GRAPH.density = 1.0; API.GRAPH.drag = 0;
        API.GRAPH.vx = new Float32Array(n); API.GRAPH.vy = new Float32Array(n);
        API.GRAPH.ticks = 0;
        API.graphStep();
        output({force: Math.abs(API.GRAPH.vx[0])});
        """
        )
        assert d["force"] < 1e-6, "600 px apart and yet repulsion"


class TestTheInitialLayout:
    """Why the starting shape is the final shape for large groups.

    The bug that survived two reworks of the force interplay — because
    it never lived there. Group members were scattered into a SQUARE:

        const s = 8 + Math.sqrt(n) * 7;
        x = p.x + (Math.random() - 0.5) * s * 2;      // and the same for y

    For small groups that doesn't matter. The repulsion reaches 225 px,
    i.e. across the whole group, and rounds it out in a few steps. Not
    for a very large group: 10,000 voices give s=708, a square with
    a 1,416 px edge — six times as wide as the range.

    And the interior of such a large, uniformly filled area is
    **force-free**: every node has the same number of neighbors all
    around, everything cancels out. Forces only act in the edge strip.
    So there's nothing that would ever round out the square — the shape
    of the first frame is still there in the last. In the image: straight
    edges, rounded corners, and next to it nothing but round groups.
    """

    def _form(self, n: int) -> dict:
        return js(f"""
        const n = {n};
        API.mapSet({{n, x: new Float32Array(n), y: new Float32Array(n)}});
        const gv = new Int32Array(n).fill(1);
        API.GRAPH.groupOf = gv;
        API.GRAPH.groups = [{{nr: 1, n}}];
        API.GRAPH.active = null;
        API.graphInitialLayout();
        const X = API.MAP_X(), Y = API.MAP_Y();
        let mx = 0, my = 0;
        for (let i = 0; i < n; i++) {{ mx += X[i]; my += Y[i]; }}
        mx /= n; my /= n;
        // Radius enclosing 99% — protects against individual outliers
        const r = Array.from({{length: n}}, (_, i) =>
            Math.hypot(X[i]-mx, Y[i]-my)).sort((a,b)=>a-b);
        const rOuter = r[Math.floor(n*0.99)];
        // For a disk, NOTHING lies outside the circle inscribed in the
        // enclosing square. For a square scatter, that's the four
        // corners: 1 − π/4 = 21.5%.
        let x0=1e9,x1=-1e9,y0=1e9,y1=-1e9;
        for (let i = 0; i < n; i++) {{
            if (X[i]<x0) x0=X[i]; if (X[i]>x1) x1=X[i];
            if (Y[i]<y0) y0=Y[i]; if (Y[i]>y1) y1=Y[i];
        }}
        const rInner = Math.min(x1-x0, y1-y0)/2;
        let outside = 0;
        for (let i = 0; i < n; i++) {{
            if (Math.hypot(X[i]-mx, Y[i]-my) > rInner) outside++;
        }}
        output({{
            n, outside_percent: +(100*outside/n).toFixed(1),
            aspect_ratio: +((x1-x0)/(y1-y0)).toFixed(2),
            radius99: Math.round(rOuter),
        }});
        """)

    def test_a_large_group_starts_as_a_disc(self):
        """A very large group: 10,000 voices in one group.

        For a square scatter, 21.5% of the points lie outside the
        inscribed circle — those are the four corners, and exactly
        those are what you see on the map. For a disk, it's nearly zero.
        """
        d = self._form(10000)
        assert d["outside_percent"] < 3.0, (
            f"{d['outside_percent']} % of the points lie in the corners — "
            f"the group starts as a square (expected: disk)"
        )
        assert 0.9 < d["aspect_ratio"] < 1.1, d

    def test_a_small_one_too(self):
        """For small groups, the layout process irons it out — but
        there's no reason to make it do the work.

        Compared against a REAL random disk of the same size, not
        against a fixed percentage. With just a few hundred points, the
        enclosing rectangle is determined by individual outliers, and
        even a perfect disk then has points outside its inscribed
        circle — at n=40 it's 17%. A fixed threshold would thus measure
        the spread, not the shape.
        """
        d = js("""
        function measure(X, Y, n) {
            let mx = 0, my = 0;
            for (let i = 0; i < n; i++) { mx += X[i]; my += Y[i]; }
            mx /= n; my /= n;
            let x0=1e9,x1=-1e9,y0=1e9,y1=-1e9;
            for (let i = 0; i < n; i++) {
                if (X[i]<x0) x0=X[i]; if (X[i]>x1) x1=X[i];
                if (Y[i]<y0) y0=Y[i]; if (Y[i]>y1) y1=Y[i];
            }
            const rInner = Math.min(x1-x0, y1-y0)/2;
            let out = 0;
            for (let i = 0; i < n; i++) {
                if (Math.hypot(X[i]-mx, Y[i]-my) > rInner) out++;
            }
            return 100*out/n;
        }
        const n = 400;
        API.mapSet({n, x: new Float32Array(n), y: new Float32Array(n)});
        API.GRAPH.groupOf = new Int32Array(n).fill(1);
        API.GRAPH.groups = [{nr: 1, n}];
        API.GRAPH.active = null;
        API.graphInitialLayout();
        const real = measure(API.MAP_X(), API.MAP_Y(), n);

        // For comparison: a disk and a square, 400 points each,
        // averaged over 20 draws (otherwise the comparison is noisy).
        let sDisc = 0, sSquare = 0;
        for (let r = 0; r < 20; r++) {
            const A = new Float32Array(n), B = new Float32Array(n);
            const C = new Float32Array(n), D = new Float32Array(n);
            for (let i = 0; i < n; i++) {
                const w = Math.random()*2*Math.PI, rr = 100*Math.sqrt(Math.random());
                A[i] = rr*Math.cos(w); B[i] = rr*Math.sin(w);
                C[i] = (Math.random()-0.5)*200; D[i] = (Math.random()-0.5)*200;
            }
            sDisc += measure(A, B, n); sSquare += measure(C, D, n);
        }
        output({measured: +real.toFixed(1),
                 disc: +(sDisc/20).toFixed(1),
                 square: +(sSquare/20).toFixed(1)});
        """)
        # Closer to the disk than to the square — and clearly so.
        assert abs(d["measured"] - d["disc"]) < abs(d["measured"] - d["square"]), d
        assert d["square"] > d["disc"] + 5, (
            f"the comparison isn't useful: disk {d['disc']} %, square {d['square']} %"
        )

    def test_the_spread_grows_with_the_group(self):
        """Otherwise a group of 10,000 would sit as a clump on top of
        itself — the reason the spread depends on the size at all."""
        small = self._form(40)["radius99"]
        large = self._form(10000)["radius99"]
        assert large > small * 8, (small, large)


class TestForeignEdges:
    """An edge BETWEEN groups is a hint, not a bond.

    The spring force grows linearly with distance, and groups end up far
    apart. Computed for a large map: a cross-group edge over 2,000 px
    pulls with

        (2000 − 70) · 0.018 · 0.2 · w  ≈  6.95 · w

    while a group holds its member at the edge with only

        0.012 · 550  ≈  6.6

    A SINGLE foreign edge thus outweighs the entire group bond, two tear
    the member out. On a large map, a group with a large share of edges
    leading out of it (around a third) became a cigar instead of a ball
    (axis ratio above 3), while groups with few outside edges stayed
    nearly round.

    After the cap, that group's axis ratio dropped to about 1.2, and all
    other groups became rounder too.
    """

    def _force(self, same_group: bool, distance: float) -> dict:
        """Force on node 0 from ONE edge to node 1.

        The setup is chosen so that no other force is involved: node 0
        sits at image center (no pull there), the distance is above the
        repulsion range, and each group has only one member (the
        centroid is the node itself).
        """
        gv = "[1, 1]" if same_group else "[1, 2]"
        return js(f"""
        const n = 2;
        const X = new Float32Array([600, 600 + {distance}]);
        const Y = new Float32Array([400, 400]);
        API.mapSet({{n, x: X, y: Y}});
        API.GRAPH.active = null;
        API.GRAPH.groupOf = Int32Array.from({gv});
        API.GRAPH.groups = [{{nr: 1, n: 1}}, {{nr: 2, n: 1}}];
        API.GRAPH.edges = Int32Array.from([0, 1]);
        API.GRAPH.kWeight = Float32Array.from([1.0]);   // target = 70
        API.GRAPH.density = 1.0; API.GRAPH.drag = 1.0;
        API.GRAPH.vx = new Float32Array(n); API.GRAPH.vy = new Float32Array(n);
        API.GRAPH.ticks = 0;
        API.graphStep();
        output({{force: +Math.abs(API.GRAPH.vx[0]).toFixed(4)}});
        """)["force"]

    def test_a_foreign_edge_pulls_only_so_far(self):
        """It must not pull any harder at 2,000 px than at 400.

        That's exactly the difference between "hint" and "bond": two
        far-apart clusters are NOT the same thing, no matter how many
        individual similarities there are between them.
        """
        near = self._force(False, 400)
        far = self._force(False, 2000)
        assert far == pytest.approx(near, rel=0.01), (
            f"{near} at 400 px vs {far} at 2,000 px — the cap isn't taking effect"
        )

    def test_and_markedly_weaker_than_without(self):
        """The number in question: uncapped it would be about 5.9."""
        far = self._force(False, 2000)
        unthrottled = (2000 - 70) * 0.018 * 1.0 * 0.2 * 0.85  # incl. cooling
        assert far < unthrottled / 4, (far, unthrottled)

    def test_inside_the_group_it_does_NOT_apply(self):
        """Internally, the edge is a real bond — it holds the cluster
        together and must pull fully regardless of distance. If it were
        capped too, a group that was once pulled apart would never
        collapse back into a ball."""
        near = self._force(True, 400)
        far = self._force(True, 2000)
        assert far > near * 3, (near, far)

    def test_short_foreign_edges_act_unchanged(self):
        """The hint is supposed to do something as long as the groups
        are close together — just not across half the map.

        300 px, not 200: at 200, the two nodes would still be WITHIN the
        repulsion range (225 px), and the measured force would be the sum
        of spring and repulsion. On the first run this was set to 200,
        and the calculation was off by exactly that amount — the test
        would then have checked a force it didn't actually mean to.
        """
        d = 300.0
        measured = self._force(False, d)
        # target = 70 (w=1), stretch 230 < limit 300, cooling 0.85
        expected = (d - 70) * 0.018 * 1.0 * 0.2 * 0.85
        assert measured == pytest.approx(expected, rel=0.05), (measured, expected)
