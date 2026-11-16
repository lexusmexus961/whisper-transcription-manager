/* The search view's tab "Same audio" (audiomatch.js, its duplicates
 * audiodups.js) and its settings page (settings_audio.js) without a browser: plain objects for the elements they
 * write into, `api()` answering from a table, and the search view's hooks
 * (svPlay, svGoto, openSearchView, svShowTab) recording their calls.
 *
 * The app's own `esc` comes from state.js — the tab writes file names and
 * folders into HTML, and that is exactly what must not slip.
 */
const fs = require('fs');
const vm = require('vm');

function element(id) {
    return {
        id, value: '', textContent: '', innerHTML: '', className: '', checked: false, disabled: false,
        dataset: {}, style: {}, hidden: false, files: [],
        classList: {
            _s: new Set(),
            add(c) { this._s.add(c); }, remove(c) { this._s.delete(c); },
            contains(c) { return this._s.has(c); },
            toggle(c, on) { const want = on === undefined ? !this._s.has(c) : on; want ? this._s.add(c) : this._s.delete(c); return want; },
        },
        addEventListener() {}, querySelector() { return null; }, querySelectorAll() { return []; },
    };
}

function grab(source, name) {
    const at = source.indexOf('function ' + name + '(');
    if (at < 0) throw new Error('no ' + name);
    return source.slice(at, source.indexOf('\n}\n', at) + 3);
}

function buildIt() {
    const ids = ['svQuery', 'svOverlay', 'svPaneAudio', 'amInfo', 'amSource', 'amViews', 'amResults', 'amPreview',
                 'amSetState', 'amSetRun', 'amSetEstimate', 'amSetStart', 'amSetStop'];
    const elements = Object.fromEntries(ids.map((id) => [id, element(id)]));
    const calls = {api: [], play: [], goto: [], open: [], tab: [], toast: [], fetch: []};
    const answers = {};
    const read = (name) => fs.readFileSync(process.env.JS_DIR + '/' + name, 'utf8');
    const sandbox = {
        document: {
            getElementById: (id) => elements[id] || null,
            querySelector: () => null, querySelectorAll: () => [],
        },
        api: async (url, opts) => {
            calls.api.push({url, opts: opts ? {method: opts.method, body: opts.body} : null});
            const key = Object.keys(answers).find((k) => url.startsWith(k));
            if (!key) throw new Error('no answer for ' + url);
            const a = answers[key];
            if (a instanceof Error) throw a;
            return typeof a === 'function' ? a(url, opts) : a;
        },
        fetch: async (url) => {
            calls.fetch.push(url);
            const key = Object.keys(answers).find((k) => url.startsWith(k));
            return {ok: true, json: async () => (key ? answers[key] : {})};
        },
        svState: {tab: 'text', routed: ''},
        svShowTab: (name) => { calls.tab.push(name); sandbox.svState.tab = name; },
        svTab: (name) => { calls.tab.push(name); sandbox.svState.tab = name; sandbox.amOpen(); },
        openSearchView: (q) => calls.open.push(q),
        svPlay: (f, t, d) => calls.play.push([f, t, d]),
        svGoto: (f, t, d) => calls.goto.push([f, t, d]),
        svPlayer: () => null,
        openSettings: () => {},
        Toast: {error: (m) => calls.toast.push(m), success() {}, info() {}},
        _showToast: (m) => calls.toast.push(m),
        _settingsSaveQuiet: async () => {},
        FormData: class { constructor() { this.parts = []; } append(k, v) { this.parts.push([k, v]); } },
        setInterval: () => 1, clearInterval: () => {}, setTimeout: () => 0, clearTimeout: () => {},
        confirm: () => true,
        // core.js's counted noun: "1 window", "3 windows"
        plural: (n, one, many) => {
            const k = Number(n) || 0;
            return `${k.toLocaleString()} ${k === 1 ? one : (many || one + 's')}`;
        },
        console, Math, JSON, Number, String, Array, Object, Set, Map, Promise, encodeURIComponent, RegExp,
        URLSearchParams, atob, Uint8Array,
        window: {devicePixelRatio: 1},
    };
    sandbox.globalThis = sandbox;
    vm.createContext(sandbox);
    vm.runInContext(grab(read('state.js'), 'esc') + '\n' + grab(read('state.js'), 'escJs'), sandbox);
    // Where the top bar sends a query (searchview.js) and Ask's button (chat.js)
    vm.runInContext(grab(read('searchview.js'), 'svRouteQuery'), sandbox);
    vm.runInContext(grab(read('chat.js'), 'chatDupsHtml'), sandbox);
    // The file view's chips of the analysis (fileview.js)
    vm.runInContext(grab(read('fileview.js'), 'fiAudio').replace(/^function/, 'async function'), sandbox);
    vm.runInContext(read('audiomatch.js'), sandbox, {filename: 'audiomatch.js'});
    vm.runInContext(read('audiodups.js'), sandbox, {filename: 'audiodups.js'});
    vm.runInContext(read('settings_audio.js'), sandbox, {filename: 'settings_audio.js'});
    return {
        sandbox, elements, calls, answers,
        run: (code) => vm.runInContext(code, sandbox),
        value: (expression) => vm.runInContext(`(${expression})`, sandbox),
        settle: async () => { for (let i = 0; i < 20; i++) await new Promise((r) => setImmediate(r)); },
    };
}

module.exports = {buildIt};
