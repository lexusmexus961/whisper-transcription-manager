/* The Images & sounds tab (mediasearch.js) without a browser: plain objects
 * for the elements it writes into, `api()` answering from a table, and the
 * search view's player hooks (svPlay, svGoto) recording their calls.
 *
 * The app's own `esc`/`escJs` come from state.js — the tab writes file
 * names and spoken text into HTML, and that is exactly what must not slip.
 */
const fs = require('fs');
const vm = require('vm');

function element(id) {
    return {
        id, value: '', textContent: '', innerHTML: '', innerText: '', checked: false, disabled: false,
        dataset: {}, style: {}, hidden: false, files: [],
        classList: {
            _s: new Set(),
            add(c) { this._s.add(c); }, remove(c) { this._s.delete(c); },
            contains(c) { return this._s.has(c); },
            toggle(c, on) { const want = on === undefined ? !this._s.has(c) : on; want ? this._s.add(c) : this._s.delete(c); return want; },
        },
        addEventListener() {}, querySelector() { return null; }, querySelectorAll() { return []; },
        click() {}, focus() {}, scrollTop: 0,
    };
}

function grab(source, name) {
    const at = source.indexOf('function ' + name + '(');
    if (at < 0) throw new Error('no ' + name);
    return source.slice(at, source.indexOf('\n}\n', at) + 3);
}

function buildIt() {
    const ids = ['svQuery', 'msInfo', 'msControls', 'msExamples', 'msSource', 'msResults', 'msPreview', 'msPvDesc',
                 'msDims', 'msFile', 'msWithText', 'msWithSound'];
    const elements = Object.fromEntries(ids.map((id) => [id, element(id)]));
    const calls = {api: [], play: [], goto: [], fetch: [], tab: [], run: 0};
    const answers = {};
    const state = fs.readFileSync(process.env.STATE_JS, 'utf8');
    const sandbox = {
        document: {
            getElementById: (id) => elements[id] || null,
            querySelector: () => null, querySelectorAll: () => [],
            addEventListener() {},
        },
        api: async (url, opts) => {
            calls.api.push({url, opts: opts ? {method: opts.method, body: opts.body} : null});
            const key = Object.keys(answers).find((k) => url.startsWith(k));
            if (!key) throw new Error('no answer for ' + url);
            const a = answers[key];
            if (a instanceof Error) throw a;
            return typeof a === 'function' ? a(url, opts) : a;
        },
        fetch: async (url) => { calls.fetch.push(url); return {ok: true, json: async () => ({})}; },
        svPlay: (f, t, d) => calls.play.push([f, t, d]),
        svGoto: (f, t, d) => calls.goto.push([f, t, d]),
        svTab: (name) => calls.tab.push(name),
        svRun: () => { calls.run += 1; },
        openSettings: () => {},
        Toast: {error() {}, success() {}},
        URL: {createObjectURL: () => 'blob:x', revokeObjectURL() {}},
        FormData: class { constructor() { this.parts = []; } append(k, v) { this.parts.push([k, v]); } },
        setTimeout: () => 0, clearTimeout: () => {},
        // core.js's counted noun: "1 window", "3 windows"
        plural: (n, one, many) => {
            const k = Number(n) || 0;
            return `${k.toLocaleString()} ${k === 1 ? one : (many || one + 's')}`;
        },
        console, Math, JSON, Number, String, Array, Object, Set, Map, Promise, encodeURIComponent, RegExp,
    };
    sandbox.globalThis = sandbox;
    vm.createContext(sandbox);
    vm.runInContext(grab(state, 'esc') + '\n' + grab(state, 'escJs'), sandbox);
    vm.runInContext(fs.readFileSync(process.env.MEDIA_JS, 'utf8'), sandbox, {filename: 'mediasearch.js'});
    return {
        sandbox, elements, calls, answers,
        run: (code) => vm.runInContext(code, sandbox),
        value: (expression) => vm.runInContext(`(${expression})`, sandbox),
    };
}

module.exports = {buildIt};
