/* Loads `voicemap_graph.js` in node and draws to a paper canvas.
 *
 * Why real JavaScript in the test at all: the earlier map tests
 * (`test_voicemap_wiring.py`) only read text — they catch renamed
 * handlers, but no computation error. The outlines around the groups are
 * geometry (convex hull, outward padding, rounded path); there "runs
 * through" and "is correct" are not the same thing. Map bugs that are
 * only visible in the picture slip past them, and this closes exactly that
 * gap in part: WHAT was drawn can be checked without looking at it.
 *
 * The paper canvas just records. It rasterizes nothing and needs no
 * library — the questions are "does Alice's name appear?" and "does every
 * group have a closed line?", not "which pixels are blue?".
 */
const fs = require('fs');
/* Both files together: the force simulation (voicemap_graph.js) and the
 * workspace (voicemap_panel.js) call each other, and loaded separately each
 * would be missing half. In the browser too they end up one after the
 * other in the same scope. */
const source = [process.env.MAP_JS, process.env.PANEL_JS]
    .filter(Boolean)
    .map(p => fs.readFileSync(p, 'utf8'))
    .join('\n;\n');

/* What the file expects from its environment. Deliberately as little as
 * possible: every stub here is an assumption about the browser that
 * could be wrong. */
const HEAD = `
/* No tree, just the holes the map reaches into. A null from
 * getElementById is the normal case in the test — any spot that can't
 * handle that would also be broken in the browser for a not-yet-built
 * field. (No backticks in this block: the header is itself a template.) */
var document = {
    getElementById: () => null,
    createElement: () => ({style: {}, dataset: {}, classList:
        {add(){}, remove(){}, toggle(){}}, addEventListener: () => {},
        appendChild: () => {}}),
    addEventListener: () => {},
    activeElement: null,
};
var Toast = {error: () => {}};
var localStorage = {getItem: () => null, setItem: () => {}};
var MAP = {n: 0, x: null, y: null, sx: 1, ox: 0, oy: 0};
var mapDraw = () => {};
var esc = (s) => String(s);
// core.js's counted noun: "1 window", "3 windows"
var plural = (n, one, many) => {
    var k = Number(n) || 0;
    return k.toLocaleString() + ' ' + (k === 1 ? one : (many || one + 's'));
};
var fetch = () => Promise.reject(new Error('no network in test'));
// voicemap.js is not loaded: its status check, as the naming calls it.
var mapResponse = async (r) => r.json();
`;
const FOOT = `
;module.exports = {
    GRAPH, graphHull, graphDrawHulls, graphDrawNames, graphColour,
    // The group index. Previously graphGroupName searched
    // linearly — once per voice, with tens of thousands of voices and groups.
    // (NO backticks here: the footer is itself a template.)
    graphGroupName, graphGroupIndex, graphDropGroupIndex,
    GRAPH_COLORS,
    // The text on the highlight sign. Pure arithmetic from three numbers
    // and thus testable without a tree — what it says is the real info.
    mapHighlightText,
    // The jump from the transcript to ONE spot: the selection
    // and the sentence on the sign, both testable without a tree or timer.
    mapSpotWindow, mapSpotText, mapClockTime,
    graphToggleOutlines, graphApplyFilter, graphDrawEdges,
    graphBuildActive, graphActive, graphStep, graphKey,
    graphBuildRow,
    // The quadtree of the repulsion. It's pure computation with no
    // image — and thus the part of the map best tested without a
    // browser. TREE gets filled IN PLACE, hence passed by reference.
    TREE, treeBuild, TREE_THETA, TREE_MAX_DEPTH,
    // The initial layout decides the SHAPE of large groups: their
    // interior is force-free, the arrangement no longer rounds anything
    // out there.
    graphInitialLayout,
    // The workspace (voicemap_panel.js)
    PANEL, graphPlayGroup, graphRowNext, graphRowPause,
    graphRowStop, graphPlaybackSet, graphRowIntoView,
    // The position arrays get changed IN PLACE by the force
    // simulation; a reference handed out once then points at the old
    // array once mapSet replaces it. So fetch fresh on every call.
    MAP_X: () => MAP.x, MAP_Y: () => MAP.y,
    // The recording date read from a file name — the same rule as the
    // extractor's in Python (core/voice_extract.date_from_name).
    graphDateFromName,
    // MAP lives in the module head; only fillable from outside via this path.
    mapSet: (k) => { Object.assign(MAP, k); },
    // The same for localStorage: what the map remembers is part of
    // its behavior and belongs under test.
    catchMemo: (f) => { localStorage.setItem = (k, v) => f(v); },
    // Naming answers arrive late — after a click elsewhere or a reload
    // that renumbered the groups (uifix: mapRename, graphNameGroup).
    mapRename, graphNameGroup,
    setFetch: (f) => { fetch = f; },
    setDocument: (d) => { document = d; },
};
`;

const M = {exports: {}};
new Function('module', 'require', HEAD + source + FOOT)(M, require);
const API = M.exports;

/* A context that logs every call instead of drawing. */
function paper() {
    const p = {texts: [], path_list: 0, strokes: 0, fills: 0, ellipses: 0,
               curves: [], strokeAlpha: [], strokeColours: []};
    const c = {
        font: '', textAlign: '', textBaseline: '', lineWidth: 0, lineJoin: '',
        strokeStyle: '', fillStyle: '', globalAlpha: 1,
        beginPath() { p.path_list++; p.curves.push([]); },
        moveTo(x, y) { p.curves[p.curves.length - 1].push([x, y]); },
        lineTo(x, y) { p.curves[p.curves.length - 1].push([x, y]); },
        quadraticCurveTo(ax, ay, x, y) {
            p.curves[p.curves.length - 1].push([x, y]);
        },
        ellipse(x, y, rx, ry) { p.ellipses++; p.curves[p.curves.length - 1]
            .push([x - rx, y - ry], [x + rx, y + ry]); },
        arc() {}, closePath() {},
        stroke() {
            p.strokes++;
            p.strokeAlpha.push(c.globalAlpha);
            p.strokeColours.push(c.strokeStyle);
        },
        fill() { p.fills++; },
        fillRect() {},
        strokeText() {},
        fillText(t, x, y) { p.texts.push({t, x, y, font: c.font}); },
    };
    return {c, p};
}

module.exports = {API, paper};
