/* The Find tab (find.js) without a browser: plain objects for the elements
 * it writes into, `api()` answering from a table, and the search view's
 * player hooks (svPlay, svGoto, svPlayer) recording their calls.
 *
 * The tiles' DOM surgery (fdRenderTiles) is replaced by a counter — what
 * is tested is what the rows and the preview write, and which hit the
 * preview holds while the list changes under it.
 */
const fs = require('fs');
const vm = require('vm');

function media() {
    const m = {tag: 'media', currentTime: 0, duration: NaN, paused: true, listeners: {}, removed: false,
               addEventListener(e, f) { (this.listeners[e] ||= []).push(f); },
               fire(e) { (this.listeners[e] || []).forEach((f) => f()); },
               pause() { this.paused = true; }, remove() { this.removed = true; }};
    return m;
}

function element(id) {
    const el = {
        id, value: '', textContent: '', innerText: '', checked: false, disabled: false,
        dataset: {}, style: {}, hidden: false, writes: 0, _html: '', media: null,
        classList: {
            _s: new Set(),
            add(c) { this._s.add(c); }, remove(c) { this._s.delete(c); },
            contains(c) { return this._s.has(c); },
            toggle(c, on) { const want = on === undefined ? !this._s.has(c) : on; want ? this._s.add(c) : this._s.delete(c); return want; },
        },
        addEventListener() {}, focus() {}, scrollTop: 0, children: [],
        insertAdjacentHTML(_where, html) { this._html += html; },
        querySelector(sel) {
            if (sel === 'video, audio') return this.media;
            return null;
        },
        querySelectorAll() { return []; },
    };
    Object.defineProperty(el, 'innerHTML', {
        get() { return this._html; },
        set(v) {
            this._html = String(v);
            this.writes += 1;
            // a player written into the box: one media element stands for it
            if (/<(video|audio)\b/.test(this._html)) this.media = media();
        },
    });
    return el;
}

function grab(source, name) {
    const at = source.indexOf('function ' + name + '(');
    if (at < 0) throw new Error('no ' + name);
    return source.slice(at, source.indexOf('\n}\n', at) + 3);
}

function buildIt() {
    const ids = ['fdQuery', 'fdGrid', 'fdAgent', 'fdPreview', 'fdMain', 'fdGo', 'fdStop', 'svQuery',
                 'fdPvName', 'fdPvInfo', 'fdPvMedia', 'fdPvClock', 'svPaneFind'];
    const elements = Object.fromEntries(ids.map((id) => [id, element(id)]));
    const calls = {api: [], play: [], goto: [], tiles: 0, paused: 0};
    const answers = {};
    const state = fs.readFileSync(process.env.STATE_JS, 'utf8');
    const sandbox = {
        document: {
            getElementById: (id) => elements[id] || null,
            querySelector: (sel) => (sel.startsWith('#fdPvMedia') ? elements.fdPvMedia.media : null),
            querySelectorAll: (sel) => (sel.startsWith('#fdPreview') && elements.fdPvMedia.media ? [elements.fdPvMedia.media] : []),
            addEventListener() {},
            createElement: () => element('tmp'),
        },
        api: async (url, opts) => {
            calls.api.push({url, opts: opts ? {method: opts.method, body: opts.body} : null});
            const key = Object.keys(answers).find((k) => url.startsWith(k));
            if (!key) throw new Error('no answer for ' + url);
            const a = answers[key];
            return typeof a === 'function' ? a(url, opts) : a;
        },
        svPlay: (f, t, d) => calls.play.push([f, t, d]),
        svGoto: (f, t, d) => calls.goto.push([f, t, d]),
        svPlayer: () => ({pause() { calls.paused += 1; }}),
        svTab() {}, svRun() {},
        fmtBytes: (b) => `${(b / 1048576).toFixed(1)} MB`,
        setTimeout: () => 0, clearTimeout: () => {},
        console, Math, JSON, Number, String, Array, Object, Set, Map, Promise, encodeURIComponent, RegExp,
    };
    sandbox.globalThis = sandbox;
    vm.createContext(sandbox);
    vm.runInContext(grab(state, 'esc') + '\n' + grab(state, 'escJs'), sandbox);
    vm.runInContext(fs.readFileSync(process.env.FIND_JS, 'utf8'), sandbox, {filename: 'find.js'});
    vm.runInContext('fdRenderTiles = function () { __calls.tiles += 1; };', Object.assign(sandbox, {__calls: calls}));
    return {
        sandbox, elements, calls, answers,
        run: (code) => vm.runInContext(code, sandbox),
        value: (expression) => vm.runInContext(`(${expression})`, sandbox),
    };
}

module.exports = {buildIt};
