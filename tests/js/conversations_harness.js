/* The search view's tab "Conversations" (conversations.js) and the places
 * that hand over to it — Ask's card (chat.js), Find (find.js), Voices →
 * People (voices_people.js) and the Conversation card of a transcript's 📊
 * (stats.js) — without a browser: plain objects for the elements they
 * write into, `api()` answering from a table (a function can hold an answer
 * back), timers that run when the test says so, and the search view's
 * hooks (svPlay, svGoto, openSearchView, svShowTab, svTab) recording their
 * calls.
 *
 * The app's own `esc`/`escJs` come from state.js, the formatters from
 * core.js and the statistics' building blocks from statsview.js: names and
 * paths are written into HTML, and that is exactly what must not slip.
 */
const fs = require('fs');
const vm = require('vm');

function element(id) {
    return {
        id, value: '', textContent: '', innerHTML: '', className: '', checked: false, disabled: false,
        dataset: {}, style: {}, hidden: false,
        classList: {
            _s: new Set(),
            add(c) { this._s.add(c); }, remove(c) { this._s.delete(c); },
            contains(c) { return this._s.has(c); },
            toggle(c, on) { const want = on === undefined ? !this._s.has(c) : on; want ? this._s.add(c) : this._s.delete(c); return want; },
        },
        addEventListener() {}, focus() {}, querySelector() { return null; }, querySelectorAll() { return []; },
    };
}

function grab(source, name) {
    const at = source.indexOf('function ' + name + '(');
    if (at < 0) throw new Error('no ' + name);
    const start = source.lastIndexOf('\n', at) + 1;          // "async function" too
    return source.slice(start, source.indexOf('\n}\n', at) + 3);
}

function grabConst(source, name) {
    const at = source.indexOf('const ' + name + ' = {');
    if (at < 0) throw new Error('no ' + name);
    return source.slice(at, source.indexOf('\n};\n', at) + 4);
}

function buildIt() {
    const ids = ['svQuery', 'svOverlay', 'svPaneConversations', 'svPaneFind', 'cvInfo', 'cvFilters', 'cvResults',
                 'cvPerson', 'cvList', 'txsConv'];
    const elements = Object.fromEntries(ids.map((id) => [id, element(id)]));
    const calls = {api: [], play: [], goto: [], open: [], tab: [], toast: [], settings: [], shown: [], order: [],
                   confirm: [], seek: []};
    const answers = {};
    const timers = new Map();
    let tid = 0;
    const read = (name) => fs.readFileSync(process.env.JS_DIR + '/' + name, 'utf8');
    const sandbox = {
        document: {
            getElementById: (id) => elements[id] || null,
            // A settings page above the search, when a test says so
            querySelector: (sel) => (sel === '.settings-overlay.visible' && sandbox.__above ? {} : null),
            querySelectorAll: () => [],
            addEventListener() {},
        },
        api: async (url, opts) => {
            calls.api.push({url, opts: opts ? {method: opts.method, body: opts.body} : null});
            // The longest key that fits: '/api/conversations/status' before '/api/conversations'
            const key = Object.keys(answers).filter((k) => url.startsWith(k)).sort((a, b) => b.length - a.length)[0];
            if (!key) throw new Error('no answer for ' + url);
            const a = answers[key];
            const got = typeof a === 'function' ? await a(url, opts) : a;
            if (got instanceof Error) throw got;
            return got;
        },
        svState: {tab: 'text', routed: ''},
        svShowTab: (name) => { calls.tab.push(name); sandbox.svState.tab = name; },
        svTab: (name) => { calls.tab.push('svTab:' + name); sandbox.svState.tab = name; },
        openSearchView: (q) => calls.open.push(q),
        svPlay: (f, t, d) => calls.play.push([f, t, d]),
        svGoto: (f, t, d) => calls.goto.push([f, t, d]),
        openSettings: (p) => calls.settings.push(p),
        vxClose: () => calls.order.push('vxClose'),
        vxOpen: (tab) => calls.order.push('vxOpen:' + tab),
        closeTxStats: () => calls.order.push('closeTxStats'),
        seekTo: (t) => calls.seek.push(t),
        Toast: {error: (m) => calls.toast.push(m), success() {}, info() {}},
        setTimeout: (fn) => { timers.set(++tid, fn); return tid; },
        clearTimeout: (id) => { timers.delete(id); },
        setInterval: () => 1, clearInterval: () => {},
        confirm: (text) => { calls.confirm.push(text); return sandbox.__confirm !== false; },
        console, Math, JSON, Number, String, Array, Object, Set, Map, Promise, Date, RegExp, Error,
        encodeURIComponent, URLSearchParams,
        window: {},
    };
    sandbox.globalThis = sandbox;
    vm.createContext(sandbox);
    const state = read('state.js'), core = read('core.js'), stv = read('statsview.js'), stats = read('stats.js');
    const find = read('find.js');
    vm.runInContext([grab(state, 'esc'), grab(state, 'escJs'),
                     grab(core, 'fmtDuration'), grab(core, 'fmtNumber'), grab(core, 'plural'),
                     grab(stv, 'stvCard'), grab(stv, 'stvBars'), grab(stv, 'stvKpi'), grab(stv, 'stvCompact')].join('\n'), sandbox);
    // The Conversation card of a transcript's 📊 (stats.js)
    vm.runInContext(['var _txsConvSeq = 0;', grab(stats, 'txsFmtClock'), grab(stats, 'txsConvCard'),
                     grab(stats, 'txsConvLoad'), grab(stats, 'txsConvHtml'), grab(stats, 'txsConvMore')].join('\n'), sandbox);
    // Ask's button (chat.js), Find's hand-off (find.js) with the rest of its
    // run stubbed, Voices → People (voices_people.js)
    vm.runInContext(grab(read('chat.js'), 'chatConvsHtml'), sandbox);
    vm.runInContext(['var _fd = {run: null, rev: -1, erev: -1, pick: null}; var __findVisible = true;',
                     'function fdRenderAgent() {} function fdRenderTiles() {} function fdRenderPreview() {}',
                     'function fdVisible() { return __findVisible; }',
                     grabConst(find, 'FD_STATUS'), grab(find, 'fdApply'), grab(find, 'fdShowConversations')].join('\n'), sandbox);
    vm.runInContext(grab(read('voices_people.js'), 'peopleConversations'), sandbox);
    vm.runInContext(read('conversations.js'), sandbox, {filename: 'conversations.js'});
    // Every hand-over recorded on its way into the real cvShow
    const real = sandbox.cvShow;
    sandbox.cvShow = (f) => {
        calls.order.push('cvShow');
        calls.shown.push(JSON.parse(JSON.stringify(f === undefined ? null : f)));
        return real(f);
    };
    const realInVoices = sandbox.cvShowInVoices;
    sandbox.cvShowInVoices = (f) => {
        calls.order.push('cvShowInVoices');
        calls.shown.push(JSON.parse(JSON.stringify(f === undefined ? null : f)));
        return realInVoices(f);
    };
    return {
        sandbox, elements, calls, answers,
        run: (code) => vm.runInContext(code, sandbox),
        value: (expression) => vm.runInContext(`(${expression})`, sandbox),
        // The timers due: the typed filters' half second
        runTimers: () => { for (const [id, fn] of [...timers]) { timers.delete(id); fn(); } },
        timers: () => timers.size,
        settle: async () => { for (let i = 0; i < 20; i++) await new Promise((r) => setImmediate(r)); },
    };
}

module.exports = {buildIt};
