/* `state.js`, `filelist.js`, `batch.js` and `fileactions.js` in node — the
 * four files that turn the queue stream into the running-file marks, the
 * Queue tab, the progress overlay and the cancel buttons.
 *
 * One context like the page's: their top-level `let`s are shared. Every
 * element exists (getElementById makes it on first ask), so the code under
 * test runs its DOM writes and the test reads them back. The stream is fed
 * by hand: tick(payload) is one server tick of /api/queue/stream.
 */
const fs = require('fs');
const path = require('path');
const vm = require('vm');

function element(id) {
    const e = {
        id, value: '', textContent: '', className: '', title: '', hidden: false,
        disabled: false, dataset: {}, style: {}, attrs: {}, children: [], inserted: [],
        scrollTop: 0, clientHeight: 600, clientWidth: 300,
        classList: {
            _s: new Set(),
            add(...cs) { cs.forEach(c => this._s.add(c)); }, remove(c) { this._s.delete(c); },
            contains(c) { return this._s.has(c); },
            toggle(c, on) {
                const want = on === undefined ? !this._s.has(c) : !!on;
                want ? this._s.add(c) : this._s.delete(c);
                return want;
            },
        },
        addEventListener() {}, removeEventListener() {},
        setAttribute(k, v) {
            this.attrs[k] = String(v);
            if (k.startsWith('data-')) this.dataset[k.slice(5)] = String(v);
        },
        getAttribute(k) { return k in this.attrs ? this.attrs[k] : null; },
        appendChild(c) { this.children.push(c); return c; },
        removeChild(c) {
            const i = this.children.indexOf(c);
            if (i >= 0) this.children.splice(i, 1);
            return c;
        },
        prepend(c) { this.children.unshift(c); }, remove() {},
        insertAdjacentHTML(pos, html) { this.inserted.push(html); },
        // A running row's stats span: the one node updateRunningRows writes
        querySelector(sel) { return sel === '.s-run' ? (this._sRun ||= element('s-run')) : null; },
        querySelectorAll(sel) {
            if (sel === '.file-item[data-run]') return this.children.filter(c => c.getAttribute('data-run') != null);
            return [];
        },
        getBoundingClientRect() { return {left: 0, top: 0, width: 300, height: 28}; },
        get firstElementChild() { return element('first'); },
        get innerHTML() { return this._html || ''; },
        set innerHTML(v) { this._html = v; this.children = []; },
    };
    return e;
}

function buildIt() {
    const elements = {};
    const fetched = [];          // {url, method, body}
    const streams = [];          // every EventSource made, in order
    const confirms = [];         // texts confirm() was asked
    const toasts = [];
    const timers = [];           // {fn, ms}: run by runTimers(), never on their own
    const calls = {refreshOne: [], scans: 0, loadServerStats: 0};

    class EventSource {
        constructor(url) { this.url = url; this.closed = false; streams.push(this); }
        close() { this.closed = true; }
    }

    const fakeDocument = {
        getElementById: (id) => (elements[id] ||= element(id)),
        createElement: () => element('new'),
        createDocumentFragment: () => element('fragment'),
        addEventListener() {}, removeEventListener() {},
        querySelector: () => null, querySelectorAll: () => [],
        body: element('body'),
    };

    const sandbox = {
        document: fakeDocument, console,
        window: {addEventListener() {}, location: {href: ''}},
        localStorage: {getItem: () => null, setItem() {}},
        setTimeout: (fn, ms) => { timers.push({fn, ms}); return timers.length; },
        clearTimeout: (id) => { if (timers[id - 1]) timers[id - 1].fn = null; },
        setInterval: () => 0, clearInterval() {},
        requestAnimationFrame: (fn) => { fn(); return 1; },
        EventSource,
        fetch: async (url, opts = {}) => {
            let body = opts.body ?? null;
            try { body = body == null ? null : JSON.parse(body); } catch (e) { /* not JSON */ }
            fetched.push({url, method: opts.method || 'GET', body});
            return {ok: true, status: 200, json: async () => ({})};
        },
        confirm: (text) => { confirms.push(text); return false; },
        Toast: {
            info: (t) => toasts.push(['info', t]), success: (t) => toasts.push(['success', t]),
            error: (t) => toasts.push(['error', t]), warn: (t) => toasts.push(['warn', t]),
        },
        // core.js's counted noun: "1 window", "3 windows"
        plural: (n, one, many) => {
            const k = Number(n) || 0;
            return `${k.toLocaleString()} ${k === 1 ? one : (many || one + 's')}`;
        },
        // From the files that aren't under test here
        updateCounts() {}, loadServerStats() { calls.loadServerStats++; },
        refreshRunningPlaceholder() {}, selectFile() {}, goToSearchResult() {},
        appendLogLines() {}, clearLogViewer() {}, scanFiles: async () => {},
        wtmDebug() {}, openQueueEditor() {},
        Math, JSON, Number, String, Array, Object, Date, Set, Map, Promise, RegExp, Error,
        Float64Array, parseInt, parseFloat, isNaN, encodeURIComponent, decodeURIComponent,
    };
    sandbox.globalThis = sandbox;
    vm.createContext(sandbox);
    const dir = process.env.STATIC_JS;
    for (const name of ['state.js', 'filelist.js', 'batch.js', 'fileactions.js']) {
        vm.runInContext(fs.readFileSync(path.join(dir, name), 'utf8'), sandbox, {filename: name});
    }
    // Spies on the two ways a finished file reaches the list
    sandbox._refreshOneFile = (name, abs) => { calls.refreshOne.push([name, abs || null]); };
    sandbox._scheduleScan = () => { calls.scans++; };

    const run = (code) => vm.runInContext(code, sandbox);
    run('connectQueueSSE()');
    const queueStream = streams.find(s => s.url === '/api/queue/stream');

    const idle = async () => { for (let i = 0; i < 10; i++) await new Promise(r => setImmediate(r)); };
    return {
        sandbox, elements, fetched, streams, confirms, toasts, calls, run, idle,
        value: (expr) => vm.runInContext(`(${expr})`, sandbox),
        el: (id) => fakeDocument.getElementById(id),
        tick: (payload) => queueStream.onmessage({data: JSON.stringify(payload)}),
        // Pending timers (cleared ones excluded), and "time passes": each runs once
        pendingTimers: () => timers.filter(t => t.fn).map(t => t.ms),
        runTimers: async () => {
            for (const t of timers.filter(t => t.fn)) { const fn = t.fn; t.fn = null; fn(); }
            await idle();
        },
    };
}

module.exports = {buildIt};
