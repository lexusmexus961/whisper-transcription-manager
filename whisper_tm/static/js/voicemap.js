/* Voice map — a graph on canvas, and never everything at once.
 *
 * THERE USED TO BE TWO VIEWS HERE. The older one drew a bubble per group and
 * opened it on click into a spiral of its members. It was removed in favour
 * of a lighter rendering path. Its reasoning was a VERY large dataset
 * (hundreds of thousands of voices); the graph, however, is now what is
 * actually used, and two rendering paths in one function made every change
 * cost double.
 *
 * The predecessor (a standalone proof-of-concept page) loads everything and
 * draws everything. That works up to about 5,000 nodes; a large dataset can
 * have hundreds of thousands. Three things about it were, measured, the coffin nails, and
 * all three are avoided here:
 *
 *   1. ONE OBJECT PER NODE. 200,000 JS objects with eight fields each are
 *      hundreds of MB and a GC nightmare. Here: Float32Array for the
 *      positions, Int32Array for numbers — the same pattern as the file
 *      list's virtual scroller (filelist.js: Float64Array for the row tops).
 *   2. LINEAR HIT TESTING. The POC scanned over all nodes on EVERY mouse
 *      move. Here: a fixed grid, built once, then O(1) per query.
 *   3. DRAWING EVERYTHING. Canvas can handle 100,000 circles, but not
 *      100,000 circles plus text plus edges. Here: only the visible
 *      viewport, and below a zoom level fillRect instead of arc — a
 *      beginPath+arc+fill per node and frame is the most expensive way to
 *      paint a point.
 *
 * What the POC did right and what stays here: DPR capped at 2, percentile
 * fitting instead of min/max (a single outlier would otherwise squash
 * everything together), labeling only at the centroid, and NO animation —
 * the coordinates arrive ready-made from the server.
 */

const MAP = {
    n: 0, x: null, y: null, size: null, number: null, name: null,
    // Viewport
    sx: 1, ox: 0, oy: 0,
    // Grid for hit testing
    raster: null, rCell: 48, rMinX: 0, rMinY: 0, rWidth: 0, rHeight: 0,
    underMouse: -1, chosen: null,
    // Permanently highlighted nodes (set of indices). `underMouse`
    // is always EXACTLY ONE and changes on every mouse move; a
    // voice, however, consists of several windows that should be
    // shown together.
    highlight: null,
    seq: 0,
};

const MAP_CELL = 48;          // grid cell in pixels
const MAP_POINT_FROM = 0.35;     // below this: fillRect instead of circle

/* Open the voices workspace ON THE MAP — the way in from the transcript
 * and the search (mapForFile & co.). The top bar opens the tab that was
 * left (voices.js, `vxOpen`). Wiring up, the first view and loading happen
 * when the map tab comes on screen (`vxShowMap`): only then does the
 * canvas have a size — clientWidth is 0 on a hidden element, and the
 * fitting would divide by zero. */
function mapOpen() {
    vxOpen('map');
}

function mapClose() {
    // Stops every player too — a sample playing on after closing sends
    // you looking for the source in another tab.
    vxClose();
}

// ── Grid: hit testing in O(1) instead of O(N) ─────────────────────────────

/* The POC scanned linearly over all nodes on every mouse move. At 200,000
 * that is the first coffin nail. A fixed grid is built ONCE (O(N)) and
 * afterwards answers every query in constant time — the same principle as
 * the prefix sum plus binary search in the virtual scroller, just
 * two-dimensional.
 */
/* Does this point participate in the picture?
 *
 * Entire groups can be hidden — in that case they must
 * neither be drawn nor be hit, and they must not factor into the fitting.
 * An invisible point that catches clicks would be the worst of the three
 * mistakes (voicemap_graph.js).
 */
function mapVisible(i) {
    return typeof graphActive !== 'function' || graphActive(i);
}

function mapBuildGrid() {
    const n = MAP.n;
    if (!n) { MAP.raster = null; return; }
    // In the tiles, `tileHits` answers the question directly from the
    // geometry — a grid over the graph positions would simply be wrong there.
    if (typeof TILE !== 'undefined' && TILE.on) { MAP.raster = null; return; }
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (let i = 0; i < n; i++) {
        if (!mapVisible(i)) continue;
        if (MAP.x[i] < minX) minX = MAP.x[i];
        if (MAP.x[i] > maxX) maxX = MAP.x[i];
        if (MAP.y[i] < minY) minY = MAP.y[i];
        if (MAP.y[i] > maxY) maxY = MAP.y[i];
    }
    const cell_size = Math.max((maxX - minX) / 180, (maxY - minY) / 180, 1e-6);
    const width = Math.max(1, Math.ceil((maxX - minX) / cell_size) + 1);
    const height = Math.max(1, Math.ceil((maxY - minY) / cell_size) + 1);
    const bucket = new Array(width * height);
    for (let i = 0; i < n; i++) {
        if (!mapVisible(i)) continue;
        const cx = Math.floor((MAP.x[i] - minX) / cell_size);
        const cy = Math.floor((MAP.y[i] - minY) / cell_size);
        const k = cy * width + cx;
        (bucket[k] = bucket[k] || []).push(i);
    }
    Object.assign(MAP, {raster: bucket, rCell: cell_size, rMinX: minX,
                          rMinY: minY, rWidth: width, rHeight: height});
}

function mapHits(wx, wy, radius) {
    if (!MAP.raster) return -1;
    const c0x = Math.max(0, Math.floor((wx - radius - MAP.rMinX) / MAP.rCell));
    const c1x = Math.min(MAP.rWidth - 1,
                         Math.floor((wx + radius - MAP.rMinX) / MAP.rCell));
    const c0y = Math.max(0, Math.floor((wy - radius - MAP.rMinY) / MAP.rCell));
    const c1y = Math.min(MAP.rHeight - 1,
                         Math.floor((wy + radius - MAP.rMinY) / MAP.rCell));
    let best = -1, bestD = radius * radius;
    for (let cy = c0y; cy <= c1y; cy++) {
        for (let cx = c0x; cx <= c1x; cx++) {
            const listing = MAP.raster[cy * MAP.rWidth + cx];
            if (!listing) continue;
            for (const i of listing) {
                const dx = MAP.x[i] - wx, dy = MAP.y[i] - wy;
                const d = dx * dx + dy * dy;
                if (d < bestD) { bestD = d; best = i; }
            }
        }
    }
    return best;
}

// ── Drawing ──────────────────────────────────────────────────────────────

function mapFit() {
    const cv = document.getElementById('mapCanvas');
    if (!cv || !MAP.n) return;
    // Percentile instead of min/max: a single outlier would squash the
    // whole map together (experience from the POC).
    //
    // ONLY over the visible points: if you hide hundreds of small groups that fill
    // the entire edge and keep fitting over all of them, the remaining
    // clusters stay a crumb in the middle of an empty picture.
    const sight = [];
    for (let i = 0; i < MAP.n; i++) if (mapVisible(i)) sight.push(i);
    if (!sight.length) return;
    const xs = Float32Array.from(sight, i => MAP.x[i]).sort();
    const ys = Float32Array.from(sight, i => MAP.y[i]).sort();
    const q = (a, p) => a[Math.min(a.length - 1, Math.floor(a.length * p))];
    const bx = Math.max(1e-6, q(xs, 0.98) - q(xs, 0.02));
    const by = Math.max(1e-6, q(ys, 0.98) - q(ys, 0.02));
    const w = cv.clientWidth || 800, h = cv.clientHeight || 600;
    MAP.sx = 0.86 * Math.min(w / bx, h / by);
    MAP.ox = w / 2 - MAP.sx * (q(xs, 0.5));
    MAP.oy = h / 2 - MAP.sx * (q(ys, 0.5));
}

function mapDraw() {
    const cv = document.getElementById('mapCanvas');
    if (!cv) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);   // capped
    const w = cv.clientWidth, h = cv.clientHeight;
    if (cv.width !== w * dpr || cv.height !== h * dpr) {
        cv.width = w * dpr; cv.height = h * dpr;
    }
    const g = cv.getContext('2d');
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, w, h);
    if (!MAP.n) return;

    /* Tiles: a completely different way of drawing — no edges, no hulls,
     * no zoom, no physics. Hence its own path and no additional
     * `if` in every loop below (voicemap_tiles.js). */
    if (TILE.on) { tilesDraw(g, w, h); return; }

    const s = MAP.sx, ox = MAP.ox, oy = MAP.oy;
    const style_str = getComputedStyle(document.documentElement);
    const color = style_str.getPropertyValue('--accent').trim() || '#6db3f2';
    const grey = style_str.getPropertyValue('--tx-3').trim() || '#888';

    // Edges first: lines belong UNDER the points. They are the reason for
    // the graph view — without them the map is a field of points in which
    // you cannot see the structure (voicemap_graph.js).
    // Outlines at the very bottom — even before the edges. An outline OVER
    // the lines would cut the graph into areas and assert boundaries where
    // there are connections (voicemap_graph.js).
    graphDrawHulls(g, s, ox, oy, w, h);
    graphDrawEdges(g, s, ox, oy, w, h);

    // Only the visible viewport. With 200,000 nodes and a zoomed-in
    // window this saves the vast majority.
    const wx0 = (-ox) / s, wx1 = (w - ox) / s;
    const wy0 = (-oy) / s, wy1 = (h - oy) / s;
    /* fillRect instead of circle is a cost-saving measure for VERY MANY
     * nodes, not for small zoom. Without the second condition it kicked in
     * immediately: the force layout spreads over ~2,000 points of width,
     * the fitting then lands below the threshold, and about a thousand voices
     * were drawn as 1.5-pixel crumbs — the colors (i.e. the group membership)
     * were no longer recognizable, and the edges overwhelmed the picture.
     * A thousand circles cost nothing. */
    const points = s < MAP_POINT_FROM && MAP.n > 20000;
    let drawn = 0;

    g.fillStyle = color;
    for (let i = 0; i < MAP.n; i++) {
        if (!mapVisible(i)) continue;
        const x = MAP.x[i], y = MAP.y[i];
        if (x < wx0 || x > wx1 || y < wy0 || y > wy1) continue;
        const px = ox + s * x, py = oy + s * y;
        // Fixed size in PIXELS, independent of zoom: every point is
        // a voice, none carries more weight than another.
        const r = 4;
        // Color per group. A single color for everything wouldn't show what belongs together.
        //
        // Filtered non-matches are DIMMED, not hidden: otherwise you
        // wouldn't see anymore that a group extends beyond the filter
        // — and that is exactly the insight (voicemap_graph.js).
        const out = GRAPH.filtered && GRAPH.visible && !GRAPH.visible[i];
        g.globalAlpha = out ? 0.13 : 1;
        g.fillStyle = graphColourOf(i);
        if (points) {
            // fillRect instead of arc: beginPath+arc+fill 200,000 times per
            // frame is the most expensive way to paint a point.
            g.fillRect(px, py, 1.5, 1.5);
        } else {
            g.beginPath();
            g.arc(px, py, r, 0, 6.283);
            g.fill();
        }
        drawn++;
    }
    g.globalAlpha = 1;

    /* Labeling: ONE label per GROUP, in its color, font size
     * grows with group size and zoom (graphDrawNames).
     *
     * The predecessor labeled the largest NODES with a fixed
     * 11px font. That was doubly unsuitable here: all nodes are the same
     * size, so the same name was stacked dozens of times on top of each
     * other in the center — and at the fitted zoom (0.26) the text wouldn't have
     * appeared anyway.
     */
    graphDrawNames(g, s, ox, oy, w, h);

    // Highlighted voice(s) first — the mouse's white ring should
    // sit on top, otherwise it would disappear underneath.
    if (MAP.highlight && MAP.highlight.size) {
        g.strokeStyle = '#ffd24a';
        g.lineWidth = 2.5;
        for (const i of MAP.highlight) {
            if (i < 0 || i >= MAP.n) continue;
            g.beginPath();
            g.arc(ox + s * MAP.x[i], oy + s * MAP.y[i], 9, 0, 6.283);
            g.stroke();
        }
    }
    if (MAP.underMouse >= 0) {
        const i = MAP.underMouse;
        g.strokeStyle = '#fff';
        g.lineWidth = 2;
        g.beginPath();
        g.arc(ox + s * MAP.x[i], oy + s * MAP.y[i], 7, 0, 6.283);
        g.stroke();
    }
    const foot = document.getElementById('mapFoot');
    if (foot) foot.textContent =
        `${drawn.toLocaleString()} of ${MAP.n.toLocaleString()} visible`
        + (points ? ' · Point view (zoom in for details)' : '');
}

// ── Controls ─────────────────────────────────────────────────────────────

function mapWireUp() {
    const cv = document.getElementById('mapCanvas');
    if (!cv || cv.dataset.wired) return;
    cv.dataset.wired = '1';
    let drags = false, lx = 0, ly = 0, moved = false;

    cv.addEventListener('mousedown', e => {
        drags = true; moved = false; lx = e.clientX; ly = e.clientY;
    });
    window.addEventListener('mouseup', () => { drags = false; });
    cv.addEventListener('mouseleave', () => {
        const h = document.getElementById('mapHint');
        if (h) h.style.display = 'none';
    });
    cv.addEventListener('mousemove', e => {
        if (TILE.on) {
            const r = cv.getBoundingClientRect();
            const t = tileHits(e.clientX - r.left, e.clientY - r.top);
            MAP.mouse = {x: e.clientX - r.left, y: e.clientY - r.top};
            mapHintMove();
            if (t.i !== TILE.below || t.field_el !== TILE.subField) {
                TILE.below = t.i; TILE.subField = t.field_el;
                MAP.underMouse = t.i;
                cv.style.cursor = t.i >= 0 ? 'pointer' : 'default';
                mapHint(t.i);
                mapRequest();
            }
            return;
        }
        if (drags) {
            MAP.ox += e.clientX - lx; MAP.oy += e.clientY - ly;
            lx = e.clientX; ly = e.clientY; moved = true;
            mapRequest();
            return;
        }
        const r = cv.getBoundingClientRect();
        const wx = (e.clientX - r.left - MAP.ox) / MAP.sx;
        const wy = (e.clientY - r.top - MAP.oy) / MAP.sx;
        const hits = mapHits(wx, wy, 12 / MAP.sx);
        MAP.mouse = {x: e.clientX - r.left, y: e.clientY - r.top};
        if (hits !== MAP.underMouse) {
            MAP.underMouse = hits;
            cv.style.cursor = hits >= 0 ? 'pointer' : 'grab';
            mapHint(hits);
            mapRequest();
        }
    });
    // `panelVoiceClicked` instead of `mapPickVoice`: only a real
    // click triggers playback on autoplay — the row also selects, but
    // plays on its own (voicemap_panel.js).
    cv.addEventListener('click', () => {
        if (TILE.on) {
            if (TILE.below >= 0) panelVoiceClicked(TILE.below);
            return;
        }
        if (moved || MAP.underMouse < 0) return;
        panelVoiceClicked(MAP.underMouse);
    });
    cv.addEventListener('wheel', e => {
        // In the tiles there is nothing to zoom: the picture fills the
        // area, and every voice is already as large as it can be. The
        // wheel then belongs to the page.
        if (TILE.on) return;
        e.preventDefault();
        const r = cv.getBoundingClientRect();
        const mx = e.clientX - r.left, my = e.clientY - r.top;
        const f = e.deltaY < 0 ? 1.15 : 1 / 1.15;
        MAP.ox = mx - (mx - MAP.ox) * f;
        MAP.oy = my - (my - MAP.oy) * f;
        MAP.sx *= f;
        mapRequest();
    }, {passive: false});

    // rAF-throttled like the file list's scroll handler — without this you'd
    // compute a full frame on every mouse event.
    window.addEventListener('resize', () => {
        if (TILE.on) tileLayout(true);
        mapRequest();
    });
}

/* The hover label sits next to the pointer, never on top of what it
 * describes — and flips to the other side near the right or lower edge. */
function mapHintMove() {
    const el = document.getElementById('mapHint');
    const cv = document.getElementById('mapCanvas');
    if (!el || !cv || el.style.display === 'none' || !MAP.mouse) return;
    const w = el.offsetWidth, h = el.offsetHeight;
    let x = MAP.mouse.x + 16, y = MAP.mouse.y + 18;
    if (x + w > cv.clientWidth - 6) x = Math.max(6, MAP.mouse.x - w - 16);
    if (y + h > cv.clientHeight - 6) y = Math.max(6, MAP.mouse.y - h - 14);
    el.style.transform = `translate(${Math.round(x)}px, ${Math.round(y)}px)`;
}

let _mapRequested = false;
function mapRequest() {
    if (_mapRequested) return;
    _mapRequested = true;
    requestAnimationFrame(() => { _mapRequested = false; mapDraw(); });
}

function mapHint(i) {
    const el = document.getElementById('mapHint');
    if (!el) return;
    if (i < 0) { el.style.display = 'none'; return; }
    el.style.display = 'block';
    {
        const m = MAP.members[i];
        // The group is the actual information: "who else does this voice
        // sound like?" Without it you'd have to compare the color to the
        // legend.
        const gr = m.group_nr
            ? `<span class="map-small">Group ${m.group_nr}`
              + (graphGroupName(m.group_nr)
                 ? ` · ${esc(graphGroupName(m.group_nr))}` : '')
              + '</span><br>'
            : '<span class="map-small">no group</span><br>';
        /* "SPEAKER_00" is NOT a name, but the label the diarization assigned
         * to this recording — it starts counting fresh for every file.
         * Showing it as a name reads as if the person were called that, and
         * in a group full of SPEAKER_00 it looks like a bug. It isn't one:
         * these are simply the voices nobody has named yet. */
        const title = m.person
            ? esc(m.person)
            : `<span class="map-unnamed">not named yet</span>`;
        el.innerHTML = `<b>${title}</b><br>`
            + gr
            + `${esc(m.display || '')}<br>`
            + `<span class="map-small">at ${Math.round(m.start_sec / 60)} min`
            + (m.speech_sec ? ` · ${Math.round(m.speech_sec)} s speech` : '')
            + '</span>';
    }
    mapHintMove();
}

/* The work panel — selection, audio sample, row player, naming — lives
 * in `voicemap_panel.js`. Previously `mapPickVoice` was here,
 * which rewrote the whole panel via `innerHTML` on EVERY selection; that
 * produced six bugs at once, including two audio samples playing at the
 * same time. The reasoning is documented at the top of that file. */

/* Read a response that says something even in the error case.
 *
 * Clicking "name whole group" once produced a message nobody could make
 * sense of. The cause was mundane and still instructive: the running
 * server didn't know the endpoint yet (**404**), because the JavaScript is
 * reloaded fresh from disk on every call, while the Python part only
 * updates after a restart. So the response was an HTML error page —
 * and `r.json()` failed at the first `<`.
 *
 * A clear "this function doesn't exist here" thus turned into
 * "Unexpected token '<'". Whoever reads that looks for the bug in their
 * data.
 *
 * Hence: check the status first, only then read. And name 404
 * explicitly — along with the fix, which you couldn't otherwise guess.
 */
async function mapResponse(r) {
    if (r.ok) {
        try {
            return await r.json();
        } catch (e) {
            throw new Error(`Response was not JSON (HTTP ${r.status})`);
        }
    }
    if (r.status === 404) {
        throw new Error('The server does not know this function (404). '
            + 'An older version is probably still running — for that the '
            + 'server needs to be restarted, reloading the page is not '
            + 'enough.');
    }
    let d = null;
    try { d = await r.json(); } catch (e) { /* HTML error page, not JSON */ }
    throw new Error(d && d.error ? d.error : `Server responded ${r.status}`);
}

/* The map's line in the head — its own, the other tabs have theirs. */
function mapHead(text) {
    vxState('map', text);
    const h = document.getElementById('mapHint');
    if (h) h.style.display = 'none';
}

function mapReport(text) {
    vxState('map', text);
}

// Esc and a click beside the panel close the workspace (voices.js).
