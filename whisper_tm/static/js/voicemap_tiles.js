/* ══════════════════════════════════════════════════════════════════════════
 *  TILES — the same voices, without physics and without moving objects
 *
 *  The graph view lagged while rendering on the client side; a static view
 *  without moving objects is the simpler answer.
 *
 *  Optimizing the ONE-TIME layout run (420 steps at 53 ms) asked the wrong
 *  question — down to a smaller search window in the repulsion, which then
 *  tore the layout into a diagonal strand.
 *  What costs continuously is the other thing: every drag and every zoom
 *  wheel redraws ~8k points, ~24k lines and ~1k rounded hulls, the
 *  hulls alone 20.4 ms.
 *
 *  THE ANSWER IS NOT LESS PHYSICS, BUT NONE. One tile per group,
 *  one square per voice, area by count. What falls away as a result:
 *
 *      Force simulation   420 × 53 ms   → gone entirely (22 s on opening)
 *      Edges                ~24k lines  → gone
 *      Hulls                 ~1k paths  → gone (alone 20.4 ms per frame)
 *      Fit/zoom            every frame  → gone (the image fills the area)
 *      the render loop                  → gone: drawing happens only
 *                                          when something actually changes
 *
 *  Measured on a map of ~8k voices in ~1k tiles (median over 20
 *  frames): **tiles 4.4 ms, graph 10.7 ms.** For drawing alone that's a
 *  factor of 2.4 — the bigger part of the gain isn't in this number:
 *  the tiles have no continuous loop that recomputes every frame for
 *  22 seconds, and no zooming and dragging that triggers it again. Whoever
 *  doesn't touch anything, consumes nothing.
 *
 *  WHAT THE TILES CANNOT DO, and that's the honest price: they show no
 *  EDGES. The graph answers "why is this voice there?" —
 *  the tiles answer "what's there, how much of it, and what does it sound
 *  like?". For assigning, the second is almost always the right question,
 *  which is why the tiles are the default. The graph stays one click away.
 *
 *  NOT TO BE CONFUSED with the former bubble view. That one
 *  had its own server endpoint, its own layout and a second level to click
 *  into — two data paths for the same thing. The tiles use exactly the same
 *  data as the graph (`MAP.members`,
 *  `GRAPH.groupOf`), the same filters, the same legend and the same
 *  work area. They are a different REPRESENTATION, not a second map.
 * ══════════════════════════════════════════════════════════════════════════ */

const TILE = {
    on: false,               // is the tile view selected?
    fields: [],              // [{key_, nr, name, idx:[…], x, y, w, h, columns, cell}]
    width: 0, height: 0,     // what the layout was computed for
    state_info: '',               // WHAT it was computed FROM, see tileState()
    below: -1,               // voice under the mouse
    subField: -1,
    /* The lower layer, painted once and cached (see tilesDraw).
     * `probe` is the handbrake: whoever changes something the stamp can't
     * capture calls `tileDiscardImage()`. */
    image: null, imageStamp: '', probe: 0, drawn: 0,
};

const TILE_MEMO = 'whisperx.map.view';

/* ── Switching ────────────────────────────────────────────────────────────
 *
 * The force simulation is NOT stopped or restarted on switching: the
 * positions in `MAP.x/y` stay where they are. Whoever switches back to
 * the graph finds the same layout — re-arranging on every switch would be
 * 22 seconds for nothing.
 */
// Above this many voices the force layout cannot run in a browser: one
// step costs 53 ms at ~8k voices (see voicemap_graph.js) and grows with
// the count — at hundreds of thousands of voices the tab froze for good. The tiles show the
// same groups at any size.
const GRAPH_MAX_VOICES = 30000;

function mapGraphTooBig() {
    return (MAP.n || 0) > GRAPH_MAX_VOICES;
}

// The Graph button says why it is off instead of freezing the tab.
function mapGraphButtonState() {
    const big = mapGraphTooBig();
    document.querySelectorAll('.map-view-btn[data-view="graph"]').forEach(b => {
        b.disabled = big;
        b.title = big
            ? `The graph arranges up to ${GRAPH_MAX_VOICES.toLocaleString()} voices — this map has `
              + `${MAP.n.toLocaleString()}. The tiles show the same groups.`
            : 'Graph: every voice a point, similar voices pulled together';
    });
}

function mapView(which) {
    if (which === 'graph' && mapGraphTooBig()) which = 'tiles';
    TILE.on = which === 'tiles';
    try { localStorage.setItem(TILE_MEMO, which); } catch (e) { /* private */ }
    document.querySelectorAll('.map-view-btn').forEach(b =>
        b.classList.toggle('active', b.dataset.view === which));
    // The graph sliders control the force simulation; the tiles have none.
    // The threshold stays visible — it decides WHICH groups exist,
    // and that's the same question in both views.
    document.querySelectorAll('.map-graph-only').forEach(
        e => { e.style.display = TILE.on ? 'none' : ''; });
    if (TILE.on) {
        tileLayout(true);
    } else {
        /* Only arrange NOW, if nobody has done it yet.
         *
         * The graph used to be computed even while the tiles were shown:
         * 420 steps at 53 ms always ran on opening, even when only tiles
         * were visible.
         * Now whoever actually looks at the graph pays the price. */
        if (!GRAPH.arranged && !GRAPH.running) graphRun();
        mapFit();
    }
    mapBuildGrid();
    mapDraw();
}

/* What the map OPENS with — always the tiles.
 *
 * Before, the last used view was remembered. That led to the map often
 * landing in the force simulation on opening, where the tile view is the
 * expected landing place. The simulation is for exploring and needs its 420
 * steps; the tiles stand ready immediately and show the distribution.
 *
 * Switching still works via the buttons — it's just no longer remembered.
 * The key in localStorage stays written but unread: whoever wants to
 * revert the decision still has the value.
 */
function tileRememberedView() {
    return 'tiles';
}

// ── Layout ───────────────────────────────────────────────────────────────

/* Squarified treemap (Bruls/Huizing/van Wijk), the same method as in the
 * search view (`searchmap.js`): fill rows so the tiles stay as square as
 * possible. Long thin strips are neither readable nor clickable —
 * and with ~1k groups, one of which has a few thousand voices and hundreds
 * with just one each, that's exactly the risk.
 */
function tileSquarify(entries, x, y, w, h) {
    const out = [];
    const sum_val = entries.reduce((s, p) => s + p.weight, 0);
    if (!(sum_val > 0) || !(w > 0) || !(h > 0)) return out;
    const scale = (w * h) / sum_val;
    const remainder = entries.map(p => ({p, area: p.weight * scale}));
    let rx = x, ry = y, rw = w, rh = h;

    /* A pointer instead of `shift()`.
     *
     * `shift()` shifts the whole array by one on EVERY item. With ~1k
     * groups that's not noticeable; with ~40k (a high threshold) it's
     * around 800 million moves. The pointer costs nothing. */
    let i = 0;
    while (i < remainder.length && rw > 0.5 && rh > 0.5) {
        const vertical = rw >= rh;
        const page = vertical ? rh : rw;
        const start = i;
        let rowArea = 0, bestOne = Infinity;
        let smallest = Infinity, largest = 0;

        while (i < remainder.length) {
            const cand = remainder[i];
            const new_value = rowArea + cand.area;
            const thick = new_value / page;
            const min = Math.min(smallest, cand.area);
            const max = Math.max(largest, cand.area);
            /* The worst aspect ratio of the row from TWO numbers instead of
             * the whole row.
             *
             * This used to say `for (const it of [...line, cand])` — the
             * row was copied and iterated for each candidate, i.e. O(k²)
             * per row. That's exactly unnecessary: a tile's length is
             * proportional to its area, so the LARGEST area gives the
             * worst `long/thick` and the SMALLEST gives the worst
             * `thick/long`. Everything in between can never be worse. The
             * result is the same, just in O(1) — it's also the form in
             * which Bruls, Huizing and van Wijk state the method. */
            const bad = (thick > 0 && min > 0)
                ? Math.max(thick * thick / min, max / (thick * thick))
                : Infinity;
            if (i > start && bad > bestOne) break;
            rowArea = new_value;
            smallest = min; largest = max;
            bestOne = bad;
            i++;
        }

        const thick = rowArea / page;
        let pos = vertical ? ry : rx;
        for (let k = start; k < i; k++) {
            const it = remainder[k];
            const len = it.area / thick;
            if (vertical) out.push({p: it.p, x: rx, y: pos, w: thick, h: len});
            else out.push({p: it.p, x: pos, y: ry, w: len, h: thick});
            pos += len;
        }
        if (vertical) { rx += thick; rw -= thick; }
        else { ry += thick; rh -= thick; }
    }
    return out;
}

/* What the current layout was computed from.
 *
 * The window size alone is NOT enough as an identifier. Whoever adjusts
 * the threshold gets new nodes, new groups and new indices — at the
 * same canvas size. The layout then stayed put: the image was byte-for-
 * byte identical (the slider looked broken), and worse, `f.idx`
 * pointed to voices from the PREVIOUS run. A click then opened a
 * different voice than the one under the pointer, and with a smaller
 * dataset the drawing read past the end.
 *
 * `GRAPH.seq` counts each load, `activeN` counts each change to filter,
 * minimum size or selection, `clusterNames` the name bundling.
 */
function tileState() {
    return [GRAPH.seq, MAP.n, GRAPH.activeN, GRAPH.clusterNames ? 1 : 0,
            GRAPH.groups.length].join('|');
}

/* The layout is ONLY computed when it could have changed — not per
 * frame. That's exactly the difference to the graph: there every frame
 * was a compute step, here every frame is just a repaint. */
function tileLayout(force = false) {
    const cv = document.getElementById('mapCanvas');
    if (!cv || !MAP.n) { TILE.fields = []; return; }
    const w = cv.clientWidth, h = cv.clientHeight;
    const state_info = tileState();
    if (!force && w === TILE.width && h === TILE.height
        && state_info === TILE.state_info && TILE.fields.length) return;
    TILE.width = w; TILE.height = h; TILE.state_info = state_info;

    /* Bundled like in the graph: `graphKey` merges same-named
     * groups when "bundle names" is on. Eleven tiles "Alice" side by side
     * would be the same misleading result as eleven clusters in the graph. */
    const sort_by = new Map();
    for (let i = 0; i < MAP.n; i++) {
        if (!graphActive(i)) continue;
        const nr = GRAPH.groupOf ? GRAPH.groupOf[i] : 0;
        const key_ = nr ? graphKey(nr) : 'one_by_one';
        let f = sort_by.get(key_);
        if (!f) {
            f = {key_, nr, name: nr ? graphGroupName(nr) : '', idx: []};
            sort_by.set(key_, f);
        }
        f.idx.push(i);
    }
    /* How many of the voices really carry the name?
     *
     * A person may have hundreds of voices assigned while their tile shows
     * ten times as many. Both are true: the assigned voices sit in hundreds
     * of groups that together contain that many voices. The tile so
     * far only showed the second number — it means "this many voices are
     * connected to them", but it looked like "this many are
     * theirs". The difference is exactly the work still to be done. */
    for (const f of sort_by.values()) {
        f.named = f.name
            ? f.idx.reduce((n, i) => n + (MAP.name?.[i] === f.name ? 1 : 0), 0)
            : 0;
    }
    const entries = [...sort_by.values()];
    for (const f of entries) f.weight = f.idx.length;
    /* Descending: the treemap needs sorted items, otherwise the aspect
     * ratios get bad. And the largest group at the top left is also the
     * one you look for first. */
    entries.sort((a, b) => b.weight - a.weight
        || String(a.key_).localeCompare(String(b.key_)));

    const edge = 6;
    TILE.fields = tileSquarify(entries, edge, edge,
                                   Math.max(1, w - 2 * edge),
                                   Math.max(1, h - 2 * edge))
        .map(r => {
            const f = r.p;
            f.x = r.x; f.y = r.y; f.w = r.w; f.h = r.h;
            /* How many columns? Chosen so the cells become as square as
             * possible: n cells in a w×h rectangle give columns
             * ≈ √(n·w/h). Without this, all cells in a wide tile would be
             * tall strips. */
            const n = f.idx.length;
            f.columns = Math.max(1, Math.round(Math.sqrt(n * r.w / Math.max(1, r.h))));
            const lines = Math.ceil(n / f.columns);
            f.cell = Math.min(r.w / f.columns, r.h / Math.max(1, lines));
            return f;
        });
}

// ── Drawing ──────────────────────────────────────────────────────────────

/* The painted image is CACHED.
 *
 * On a large map Firefox warned "this page is slowing down Firefox".
 * Measured on a map of ~80k voices in ~8k tiles:
 *
 *     one frame                        156 ms
 *     and one frame per mouse move, because the highlight is drawn in the
 *     same pass as everything underneath
 *
 * Firefox reports exactly that: the main thread can't get out of drawing
 * anymore. Yet moving over the map changes NOTHING about the ~80k
 * cells — only a white rectangle moves across them.
 *
 * So two layers: the unchanging part is painted once into its own canvas
 * and from then on only stamped; on top come highlight and pointer. The
 * stamp contains everything that can change the lower image —
 * load run, filter, names, size, theme colors. If one were forgotten,
 * you'd get an image showing an older truth, and that's worse than a slow
 * one: you can't tell just by looking.
 */
function tileDiscardImage() {
    TILE.probe = (TILE.probe | 0) + 1;
    // The layout holds the tiles' names: a group named in the panel kept
    // its "Group 1" until the map was loaded again
    TILE.state_info = null;
}

function tilesDraw(g, w, h) {
    tileLayout();
    const style_str = getComputedStyle(document.documentElement);
    const frame = style_str.getPropertyValue('--border').trim() || '#333';
    const tx1 = style_str.getPropertyValue('--tx-1').trim() || '#eee';

    const cv = document.getElementById('mapCanvas');
    // Fallback to 1: a canvas without `width` (or a width of 0) would
    // otherwise give NaN — and a canvas with a NaN size silently paints nothing.
    const dpr = (cv && cv.width && w) ? (cv.width / w) : 1;
    const stamp = [tileState(), w, h, Math.round(dpr * 100),
                     GRAPH.filtered ? 1 : 0, GRAPH.viewState | 0,
                     TILE.probe | 0, frame, tx1].join('|');
    if (TILE.imageStamp !== stamp || !TILE.image) {
        if (!TILE.image) TILE.image = document.createElement('canvas');
        TILE.image.width = Math.max(1, Math.round(w * dpr));
        TILE.image.height = Math.max(1, Math.round(h * dpr));
        const bg = TILE.image.getContext('2d');
        bg.setTransform(dpr, 0, 0, dpr, 0, 0);
        bg.clearRect(0, 0, w, h);
        tileDrawBase(bg, frame, tx1);
        TILE.imageStamp = stamp;
    }
    g.drawImage(TILE.image, 0, 0, w, h);
    tileDrawMarkers(g);
    tileFoot();
}

/* The lower layer: background, cells, borders, names. Everything that
 * does NOT depend on the pointer — and therefore is created at most once
 * per change. */
function tileDrawBase(g, frame, tx1) {
    const sight = GRAPH.filtered ? GRAPH.visible : null;
    let drawn = 0;

    for (const f of TILE.fields) {
        if (f.w < 1 || f.h < 1) continue;
        // Background: the group color, heavily dimmed. It carries the
        // affiliation even where the tile has become too small for
        // individual cells.
        g.globalAlpha = 0.16;
        g.fillStyle = f.idx.length ? graphColourOf(f.idx[0]) : frame;
        g.fillRect(f.x, f.y, f.w, f.h);
        g.globalAlpha = 1;

        /* Cells: one per voice.
         *
         * `fillStyle` and `globalAlpha` are only set on CHANGE. That
         * sounds like a minor detail, but with ~8k cells it isn't:
         * every assignment to `fillStyle` parses a color string. Within
         * a tile the color is almost always the same, so almost all of
         * it is skipped. Measured 4.9 → 4.4 ms per frame. */
        const z = f.cell, slack = z > 5 ? 1 : 0;
        const edge = Math.max(1, z - 2 * slack);
        let color = null, alpha = null;
        for (let k = 0; k < f.idx.length; k++) {
            const i = f.idx[k];
            // Filtered non-matches are DIMMED, not hidden —
            // otherwise you couldn't tell anymore that a group extends
            // beyond the filter (the same rule as in the graph).
            const a = (sight && !sight[i]) ? 0.13 : 1;
            if (a !== alpha) { g.globalAlpha = a; alpha = a; }
            const c = graphColourOf(i);
            if (c !== color) { g.fillStyle = c; color = c; }
            g.fillRect(f.x + (k % f.columns) * z + slack,
                       f.y + ((k / f.columns) | 0) * z + slack, edge, edge);
            drawn++;
        }
        g.globalAlpha = 1;

        // Border last, so it sits on top of the cells.
        g.strokeStyle = frame;
        g.lineWidth = 1;
        g.strokeRect(f.x + 0.5, f.y + 0.5, f.w - 1, f.h - 1);
    }

    // Labels in a second pass: this way no name ends up under the
    // neighboring tile when it extends past its edge.
    g.textAlign = 'center';
    g.textBaseline = 'middle';
    for (const f of TILE.fields) {
        if (f.w < 46 || f.h < 24) continue;
        const title = f.name || (f.nr ? `Group ${f.nr}` : 'no group');
        /* Only label if the name really FITS.
         *
         * This used to say `Math.max(10, …)`: if the name didn't fit, it
         * was still painted at 10 px and ran over the neighboring tiles.
         * With the many narrow tiles on the right, that produced a strip
         * of overlapping "Group 12 Group 34 Group 56". Whoever needs
         * the name of a narrow tile hovers it with the mouse. */
        const size = Math.min(f.w / (title.length * 0.62), f.h * 0.3, 30);
        if (size < 9) continue;
        const mx = f.x + f.w / 2, my = f.y + f.h / 2;
        g.font = `700 ${size.toFixed(1)}px system-ui, sans-serif`;
        // Outline, so the name stays legible on any background.
        g.lineWidth = Math.max(2, size * 0.22);
        g.strokeStyle = 'rgba(0,0,0,0.75)';
        g.strokeText(title, mx, my - size * 0.28);
        g.fillStyle = tx1;
        g.fillText(title, mx, my - size * 0.28);
        if (f.h >= 34) {
            const small = Math.max(9, size * 0.52);
            g.font = `500 ${small.toFixed(1)}px system-ui, sans-serif`;
            // For a named tile, BOTH numbers: the ones assigned
            // and the total that hang together in it.
            const number = (f.named && f.named < f.idx.length)
                ? `${f.named.toLocaleString()} / ${f.idx.length.toLocaleString()}`
                : f.idx.length.toLocaleString();
            g.lineWidth = Math.max(2, small * 0.22);
            g.strokeStyle = 'rgba(0,0,0,0.75)';
            g.strokeText(number, mx, my + size * 0.62);
            g.fillStyle = tx1;
            g.fillText(number, mx, my + size * 0.62);
        }
    }
    g.textBaseline = 'alphabetic';
    TILE.drawn = drawn;
}

/* The upper layer: everything that can change just by moving over the map.
 * It's drawn per frame — and costs nothing, because it paints at most a
 * few dozen rectangles instead of ~80k. */
function tileDrawMarkers(g) {
    /* Highlighted voices (`MAP.highlight`) — the way in from the transcript.
     *
     * In the graph these are yellow rings around individual points. Here
     * that would often be invisible: with ~8k voices, a cell is often
     * only two pixels big. Hence TWO stages — the tile gets a bold
     * border (that can be found), the cell inside it gets a mark (that
     * shows exactly where). Without the first stage you'd have to scan
     * the whole map; without the second you'd only know the group, not
     * the voice.
     */
    if (MAP.highlight && MAP.highlight.size) {
        const cells = [];
        g.strokeStyle = '#ffd24a';
        for (const f of TILE.fields) {
            let inside = 0;
            for (let k = 0; k < f.idx.length; k++) {
                if (!MAP.highlight.has(f.idx[k])) continue;
                inside++;
                cells.push([f.x + (k % f.columns) * f.cell,
                             f.y + ((k / f.columns) | 0) * f.cell, f.cell]);
            }
            if (!inside || f.w < 1 || f.h < 1) continue;
            g.lineWidth = 2.5;
            g.strokeRect(f.x + 1.25, f.y + 1.25, f.w - 2.5, f.h - 2.5);
        }
        for (const [zx, zy, z] of cells) {
            const s = Math.max(5, z);
            g.lineWidth = Math.max(1.5, Math.min(2.5, z * 0.4));
            g.strokeRect(zx - (s - z) / 2, zy - (s - z) / 2, s, s);
        }

        /* ONE crosshair, when it concerns a single voice.
         *
         * Jumping from a transcript chunk to the speaker map seemed to
         * show a bunch of little boxes for many people — yet exactly ONE
         * cell in ONE tile was marked. The many bright little boxes are
         * the FILTER: the jump restricts the map to the recording, and its
         * hundreds of voices are scattered across the tiles of many
         * people, bright against the dimmed remainder.
         *
         * The ring around the sought-after cell was there — just
         * unfindable among tens of thousands of cells, hundreds of which
         * are lit up. Two
         * lines across the canvas point to it; the eye follows them
         * immediately, and nothing is obscured because they're thin and
         * semi-transparent.
         *
         * Only with ONE marked voice: with a whole identity (thirty
         * windows), thirty crosshairs would be a grid. */
        if (cells.length === 1) {
            const [zx, zy, z] = cells[0];
            const mx = zx + z / 2, my = zy + z / 2;
            g.save();
            g.globalAlpha = 0.55;
            g.strokeStyle = '#ffd24a';
            g.lineWidth = 1;
            g.setLineDash([6, 5]);
            g.beginPath();
            g.moveTo(0, my); g.lineTo(Math.max(0, mx - z * 1.5), my);
            g.moveTo(mx + z * 1.5, my); g.lineTo(TILE.width, my);
            g.moveTo(mx, 0); g.lineTo(mx, Math.max(0, my - z * 1.5));
            g.moveTo(mx, my + z * 1.5); g.lineTo(mx, TILE.height);
            g.stroke();
            g.restore();
            // A ring that still works even at two-pixel cell size.
            g.strokeStyle = '#ffd24a';
            g.lineWidth = 2;
            g.beginPath();
            g.arc(mx, my, Math.max(9, z * 0.9), 0, Math.PI * 2);
            g.stroke();
        }
    }

    // Highlight the voice under the mouse.
    if (TILE.below >= 0 && TILE.subField >= 0) {
        const f = TILE.fields[TILE.subField];
        const k = f ? f.idx.indexOf(TILE.below) : -1;
        if (k >= 0) {
            const z = f.cell;
            g.strokeStyle = '#fff';
            g.lineWidth = 2;
            g.strokeRect(f.x + (k % f.columns) * z, f.y + Math.floor(k / f.columns) * z,
                         Math.max(3, z), Math.max(3, z));
        }
    }

}

function tileFoot() {
    const foot = document.getElementById('mapFoot');
    if (!foot) return;
    foot.textContent =
        `${plural(TILE.drawn, 'voice')} in `
        + `${plural(TILE.fields.length, 'tile')} · `
        + 'tile view (no force simulation — area shows the count)';
}

// ── Hit test ─────────────────────────────────────────────────────────────

/* No grid needed: checking ~1k rectangles linearly costs nothing, and
 * within a tile the cell results from two divisions. The graph needs
 * its grid because there thousands of points float freely in space. */
function tileHits(px, py) {
    for (let fi = 0; fi < TILE.fields.length; fi++) {
        const f = TILE.fields[fi];
        if (px < f.x || px > f.x + f.w || py < f.y || py > f.y + f.h) continue;
        const col = Math.floor((px - f.x) / f.cell);
        const row = Math.floor((py - f.y) / f.cell);
        if (col < 0 || col >= f.columns) return {field_el: fi, i: -1};
        const k = row * f.columns + col;
        if (k < 0 || k >= f.idx.length) return {field_el: fi, i: -1};
        return {field_el: fi, i: f.idx[k]};
    }
    return {field_el: -1, i: -1};
}
