/* Loads `searchview.js` in node, with just enough DOM that it runs.
 *
 * Why at all: without a JS test for the large search view, a bug
 * slipped through that no endpoint test could find — the API
 * answered correctly, the view just asked the wrong thing (see
 * `tests/test_search_view.py`).
 *
 * The elements are deliberately plain objects with `value`, `innerHTML`
 * and `classList`. None of the tested functions need more, and every
 * additional stub would be an assumption about the browser that could
 * be wrong.
 */
const fs = require('fs');

/* BOTH files, in the same order as in the browser.
 *
 * `searchview.js` (the large view) calls `_hlHit` from `search.js` (the
 * small search) — they share the highlighting of hits. Loaded
 * separately, the large one was missing exactly this function, and the
 * view caught the failure instead of crashing: it wrote "Error: ..."
 * into the list and set `done = true`. The test then saw a wrong state
 * instead of an exception — and measured something other than it
 * should have. */
// SEARCH_MAP (searchmap.js, optional): the treemap helpers the hit map
// of the large view builds on (smBuildTree, smSquarify, smChildren).
const source = [process.env.SEARCH_SMALL, process.env.SEARCH_MAP, process.env.SEARCH_JS]
    .filter(Boolean)
    .map((p) => fs.readFileSync(p, 'utf8'))
    .join('\n;\n');

/* SEARCH_STATE (state.js, optional): the app's real `esc` and `escJs`
 * instead of the pass-through stand-ins below. The stand-ins keep the older
 * assertions readable; a test about escaping needs the real thing — taken
 * from state.js itself, so the two can't drift apart. */
let ESCAPING = '';
if (process.env.SEARCH_STATE) {
    const state = fs.readFileSync(process.env.SEARCH_STATE, 'utf8');
    const grab = (name) => {
        const at = state.indexOf('function ' + name + '(');
        if (at < 0) throw new Error('state.js has no ' + name);
        return state.slice(at, state.indexOf('\n}\n', at) + 3)
            .replace('function ' + name + '(', name + ' = function (');
    };
    ESCAPING = grab('esc') + ';\n' + grab('escJs') + ';\n';
}

const HEAD = `
var __elements = {};
var __fetched = [];        // every fetch URL the view requests
var __answer = {total: 0, files: 0, results: [], files: []};

var __anonymous = 0;

function __el(id) {
    if (!__elements[id]) {
        __elements[id] = {
            value: '', textContent: '', style: {},
            dataset: {},
            /* The id is writable AND registers itself. The view builds its
             * blocks with createElement and then sets box.id = 'svFileHits';
             * without this registration getElementById('svFileHits') would
             * never find them, would create an empty placeholder every
             * time, and the old remove() would hit nothing — the block
             * would grow on every rebuild. */
            _id: id,
            get id() { return this._id; },
            set id(v) { this._id = v; __elements[v] = this; },
            /* innerHTML here is not a field but a computation: its own
             * markup PLUS that of the inserted child nodes. Without that
             * the harness would lie in two places:
             *   - remove() deleted the node from __elements, but its
             *     markup stayed in the parent's list — a freshly rebuilt
             *     file block counted 3 + 40 = 43 lines;
             *   - an innerHTML = '...' left the children standing, even
             *     though that's exactly what clears everything inside
             *     in the browser.
             * Either way a real bug would have gone undetected, or clean
             * code would have looked like a bug. */
            _own: '', _children: [], _parents: null,
            get innerHTML() {
                return this._children.map(k => k.innerHTML).join('') + this._own;
            },
            set innerHTML(v) { this._own = v; this._children = []; },
            classList: {
                _s: new Set(),
                add(c) { this._s.add(c); },
                remove(c) { this._s.delete(c); },
                contains(c) { return this._s.has(c); },
                toggle(c) { this._s.has(c) ? this._s.delete(c) : this._s.add(c); },
            },
            focus() {}, select() {}, scrollTo() {}, scrollIntoView() {},
            /* Media element: #videoPlayer sits OUTSIDE the overlay and
             * keeps running while the search stays open. Without these
             * stubs every check of the playback bar falls into a
             * TypeError instead of a statement. */
            paused: true, currentTime: 0, duration: 0,
            play() { this.paused = false; return Promise.resolve(); },
            pause() { this.paused = true; },
            /* Listeners are kept (not fired by anything): a test can call
             * the one the view registered, e.g. the list's scroll. */
            addEventListener(type, fn) {
                this._ls = this._ls || {};
                (this._ls[type] = this._ls[type] || []).push(fn);
            },
            appendChild() {},
            /* remove() was missing on the first pass, and the failure
             * was instructive: the view catches errors while rendering,
             * writes "Error: ..." into the list and sets done = true. The
             * test then saw a plausible but wrong state size instead of
             * an exception. A too-thin harness thus doesn't test less,
             * it tests something different.
             * (NO backticks in this block: the header is a template.) */
            remove() {
                delete __elements[this.id];
                if (this._parents) {
                    this._parents._children =
                        this._parents._children.filter(k => k !== this);
                    this._parents = null;
                }
            },
            insertAdjacentHTML(where, html) { this._own += html; },
            /* The file block and meaning block get placed
             * INTO the hit list (previously beside it, where they took up
             * its width). For that, the harness must at least know that
             * there are children — otherwise the call hits nothing and
             * the test again checks something other than what the view
             * does.
             * The REFERENCE NODE is not simulated: both blocks sit before
             * the text spots anyway, in insertion order (files first,
             * then meaning) — exactly as in the browser. */
            firstChild: null,
            nextSibling: null,
            insertBefore(nodes, _ref) {
                if (!nodes) return nodes;
                nodes._parents = this;
                this._children.push(nodes);
                return nodes;
            },
            querySelectorAll() { return []; },
            querySelector() { return null; },
            getBoundingClientRect() { return {top: 0, height: 0}; },
        };
    }
    return __elements[id];
}

var document = {
    getElementById: __el,
    /* Every call its OWN element. Previously it always returned the
     * same one — the file block and meaning block were the same node and
     * overwrote each other, which in the test would have looked like a
     * vanished block. */
    createElement: () => __el('__anonymous' + (++__anonymous)),
    querySelectorAll: () => [],
    querySelector: () => null,
    addEventListener: () => {},
};
var window = {addEventListener: () => {}, setTimeout: setTimeout};
/* The redraw happens after setting innerHTML — the harness has no
 * refresh rate, so it runs immediately. */
var requestAnimationFrame = (fn) => setTimeout(fn, 0);
/* The helpers from core.js. Kept complete rather than added to on
 * demand: if one is missing, the view catches the exception while
 * rendering and writes "Error: ..." into the list — the test then sees
 * no crash but a wrong state and measures something other than
 * intended. */
var esc = (s) => String(s == null ? '' : s);
var escJs = (s) => String(s == null ? '' : s).replace(/'/g, "\\'");
var fmtDuration = (s) => String(s);
// core.js's counted noun: "1 window", "3 windows"
var plural = (n, one, many) => {
    var k = Number(n) || 0;
    return k.toLocaleString() + ' ' + (k === 1 ? one : (many || one + 's'));
};
var Toast = {error: () => {}, info: () => {}};
var __jumped = [];
var goToSearchResult = (f, t, d) => { __jumped.push({f, t, d});
    return Promise.resolve(); };
/* The view queries TWO endpoints: /api/search for the
 * literal hits and /api/rag/search for the meaning-related ones. A
 * single answer for everything would answer the meaning search with a
 * hit list — the test would then see something that never actually
 * happens. */
var __answers = [];     // [{pattern, answer}] — first match wins
var fetch = (url) => {
    const u = String(url);
    __fetched.push(u);
    const p = __answers.find(x => u.includes(x.pattern));
    return Promise.resolve({json: () => Promise.resolve(p ? p.answer : __answer)});
};
`;

const FOOT = `
// These tests are about the Text tab: Find is the start tab
// (find.js has tests of its own)
;svState.tab = 'text';
;module.exports = {
    svState, svParams, svHasFilter, svFileFallback, openSearchView,
    svClearFilters, svPinFile, svRun,
    svAdvParams, svLoadMore, svSetStatus, svToggleFileMode,
    svRenderFiles, svToggleFiles, svMeaningSearch, svRenderMeaning,
    svToggleMeaning, svSetSideWidth,
    svTab, svListBox, svRenderHits, svSelect, svMoveSelection,
    svSaveState, svLoadState, svSetMode, svMeaningOnly,
    svPlay, svWgToggle, svWgStop, svShowPin,
    closeSearchView,
    svToggleView, svRenderFacets, svSetStatus, svShortName,
    svPersonParams, svShowPersonBox, svSyncPeople,
    svRenderFileList, svMapHeat,
    setupSearchView, svMapLevel, svMapInto, runGlobalSearch, svRedrawRecordings,
    // Where a top-bar query goes
    svRouteQuery,
    // The meaning list ranked by the reranker (situation search)
    svQuickPrepare, svRenderQuick, svQuickWeak, svQuickCopies, svSelectCopy,
    svCopyList, svPct, svQuickHit,
    quick: () => _svQuick,
    smBuildTree: typeof smBuildTree === 'function' ? smBuildTree : null,
    map: () => ({path: _svMapPath, tiles: _svMapTiles}),
    setMapPath: (p) => { _svMapPath = p; },
    ragDown: () => _svRagDown,
    // Test helpers
    el: __el,
    fetched: () => __fetched.slice(),
    clearFetched: () => { __fetched.length = 0; },
    setAnswer: (a) => { __answer = a; __answers = []; },
    answerFor: (pattern, a) => { __answers.push({pattern, answer: a}); },
    jumped: () => __jumped.slice(),
    clearJumped: () => { __jumped.length = 0; },
};
`;

const M = {exports: {}};
new Function('module', 'require', 'setTimeout', 'Promise',
             HEAD + ESCAPING + source + FOOT)(M, require, setTimeout, Promise);

module.exports = M.exports;
