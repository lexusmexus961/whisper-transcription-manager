/* Loads `voicemap_tiles.js` in node and draws to a paper canvas.
 *
 * Like `voicemap_harness.js`, but for the tiles — with one important
 * difference: this file needs a canvas with SIZE. The layout depends on
 * `clientWidth`/`clientHeight`, and a stub that returns 0 would have let
 * all the checks below run silently against an empty map.
 *
 * What's rebuilt here is deliberately as little as possible. Of
 * `voicemap.js` and `voicemap_graph.js`, only the four functions the tiles
 * actually call are provided: `graphActive`, `graphKey`,
 * `graphGroupName` and `graphColourOf`. Loading them in full would mean
 * testing their bugs along with it — that's what the other test files
 * are for.
 */
const fs = require('fs');
const source = fs.readFileSync(process.env.TILES_JS, 'utf8');

const HEAD = `
var _canvas = {clientWidth: 1200, clientHeight: 800};
var document = {
    getElementById: (id) => id === 'mapCanvas' ? _canvas : null,
    querySelectorAll: () => [],
    documentElement: {},
    // The tiles paint their bottom layer into its own
    // canvas and keep it around. Without this stub the test would run
    // into "document.createElement is not a function" — which it did.
    createElement: (kind) => globalThis.__buildCanvas(kind),
};
var getComputedStyle = () => ({getPropertyValue: () => ''});
var localStorage = {getItem: () => null, setItem: () => {}};
var MAP = {n: 0, members: [], name: []};
var GRAPH = {
    groupOf: null, groups: [], active: null, activeN: 0, seq: 0,
    minGroup: 1, selection: null,
    clusterNames: false, filtered: false, visible: null,
};
var mapDraw = () => {};
var mapFit = () => {};
var mapBuildGrid = () => {};
var graphOnlyThisGroup = () => {};

// The four real helpers, rebuilt — they're tested elsewhere.
function graphActive(i) { return !GRAPH.active || !!GRAPH.active[i]; }
function graphGroupName(nr) {
    const g = GRAPH.groups.find(g => g.nr === nr);
    return g ? (g.name || '') : '';
}
function graphKey(nr) {
    if (GRAPH.clusterNames) {
        const name = graphGroupName(nr);
        if (name) return 'n:' + name.toLowerCase();
    }
    return 'g:' + nr;
}
function graphColourOf(i) {
    const nr = GRAPH.groupOf ? GRAPH.groupOf[i] : 0;
    return nr ? 'hsl(' + (nr * 47 % 360) + ' 70% 60%)' : '#888';
}
/* graphBuildActive in miniature: only the minimum size and the selection.
 *
 * activeN belongs with it — the tile layout uses it to detect that
 * something changed in the collection. The stub didn't set it at first,
 * and the test for it caught fire: it caught the stub, not the code.
 * (NO backticks in this block: the header is itself a template.) */
function graphBuildActive() {
    if (!MAP.n || !GRAPH.groupOf) { GRAPH.active = null; GRAPH.activeN = 0; return; }
    const size = new Map();
    for (const g of GRAPH.groups) {
        const s = graphKey(g.nr);
        size.set(s, (size.get(s) || 0) + g.n);
    }
    const a = new Uint8Array(MAP.n);
    let n = 0;
    for (let i = 0; i < MAP.n; i++) {
        const nr = GRAPH.groupOf[i];
        const gr = nr ? (size.get(graphKey(nr)) || 0) : 1;
        if (gr < GRAPH.minGroup) continue;
        if (GRAPH.selection && !GRAPH.selection.has(nr)) continue;
        a[i] = 1; n++;
    }
    GRAPH.active = a; GRAPH.activeN = n;
}
`;

const FOOT = `
;module.exports = {
    TILE, tileLayout, tileSquarify, tilesDraw, tileHits,
    tileDiscardImage,
    tileRememberedView, mapView,
    GRAPH, graphBuildActive,
    mapSet: (k) => { Object.assign(MAP, k); },
    // What WOULD be in localStorage. This may no longer
    // decide the view on opening — only testable if the test can put
    // something in there.
    setRemembered: (v) => { localStorage.getItem = () => v; },
    // The canvas size here is a test value, not a random pick: whether
    // the layout gets recomputed depends on it changing.
    canvas_el: (w, h) => { _canvas.clientWidth = w; _canvas.clientHeight = h; },
};
`;

const M = {exports: {}};
new Function('module', 'require', HEAD + source + FOOT)(M, require);
const API = M.exports;

/* A context that logs every call instead of drawing.
 *
 * `curvePoints` counts what the tiles specifically should NOT do: paths,
 * arcs, curves. As long as the counter stays at zero, an image really is
 * just a collection of rectangles. */
/* A fresh sheet means: the retained image no longer applies.
 *
 * Otherwise the tests below would check past the retained image — the
 * second sheet would only get a stamp and be empty. Whoever wants to
 * test the RETAINING itself uses `paper(false)`. */
function paper(fresh = true) {
    /* `frame` records the stroked rectangles WITH color and size.
     * Without that, the highlight (yellow border) couldn't be
     * told apart from the normal tile borders — both are strokeRect, and
     * a plain counter would have fired on every image. */
    const p = {rects: 0, weakRects: 0, strokes: 0, path_list: 0,
               curvePoints: 0, texts: [], frame: [], images: 0};
    if (fresh && API.tileDiscardImage) API.tileDiscardImage();
    const c = context(p);
    lastItem = p;
    return {c, p};
}

/* The same recorder for both layers — the retained canvas gets it just
 * like the visible sheet. Both together form the image the human sees,
 * and that's exactly what the tests check. */
function context(p) {
    const c = {
        font: '', textAlign: '', textBaseline: '', lineWidth: 0,
        strokeStyle: '', fillStyle: '', globalAlpha: 1,
        fillRect() { p.rects++; if (c.globalAlpha < 0.5) p.weakRects++; },
        strokeRect(x, y, w, h) {
            p.strokes++;
            p.frame.push({x, y, w, h, color: c.strokeStyle, width: c.lineWidth});
        },
        beginPath() { p.path_list++; },
        moveTo() { p.curvePoints++; },
        lineTo() { p.curvePoints++; },
        quadraticCurveTo() { p.curvePoints++; },
        arc() { p.curvePoints++; },
        ellipse() { p.curvePoints++; },
        closePath() {}, stroke() { p.strokes++; }, fill() {},
        strokeText() {},
        fillText(t, x, y) { p.texts.push({t, x, y, font: c.font}); },
        setTransform() {}, clearRect() {}, drawImage() { p.images++; },
        // For the crosshair of the single selection.
        save() {}, restore() {}, setLineDash() {},
    };
    return c;
}

let lastItem = null;
globalThis.__buildCanvas = () => ({
    width: 0, height: 0,
    getContext: () => context(lastItem || {rects: 0, weakRects: 0,
        strokes: 0, path_list: 0, curvePoints: 0, texts: [], frame: [], images: 0}),
});

module.exports = {API, paper};
