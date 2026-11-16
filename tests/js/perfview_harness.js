/* perfview.js in node, for the performance view's Live range: a fake DOM,
 * a clock the test moves, timers and animation frames that run only when
 * the test says so, and a fetch whose answers the test hands out — or
 * fails, or sees aborted (an aborted request rejects with an AbortError,
 * as in the browser).
 *
 * An element's querySelectorAll answers a class (".pv-scroll") or a tag
 * ("svg") with one stand-in per match in its markup — the same stand-ins
 * until the markup changes —, enough to follow what the code does to them.
 */
const fs = require('fs');
const vm = require('vm');

function element(id) {
    const classes = new Set();
    return {
        id, textContent: '', value: '', title: '', hidden: false, disabled: false, open: false,
        style: {}, dataset: {}, attrs: {}, clientWidth: 940, offsetWidth: 200, offsetHeight: 100,
        classList: {
            add: c => classes.add(c), remove: c => classes.delete(c),
            toggle: (c, on) => ((on ?? !classes.has(c)) ? classes.add(c) : classes.delete(c)),
            contains: c => classes.has(c),
        },
        setAttribute(k, v) { this.attrs[k] = String(v); },
        getAttribute(k) { return k in this.attrs ? this.attrs[k] : null; },
        addEventListener() {}, removeEventListener() {}, setPointerCapture() {},
        querySelector: () => null,
        querySelectorAll(sel) {
            const html = this._html || '';
            if (this._qsaFor !== html) { this._qsa = {}; this._qsaFor = html; }
            if (!this._qsa[sel]) {
                const re = sel.startsWith('.') ? new RegExp(`class="${sel.slice(1)}[" ]`, 'g')
                                               : new RegExp(`<${sel}[ >]`, 'g');
                const n = (html.match(re) || []).length;
                this._qsa[sel] = Array.from({length: n}, (_, i) => element(`${sel}#${i}`));
            }
            return this._qsa[sel];
        },
        getBoundingClientRect: () => ({left: 0, top: 0, width: 900, height: 140}),
        get innerHTML() { return this._html || ''; },
        set innerHTML(v) { this._html = String(v); },
    };
}

function buildIt(files) {
    const elements = {};
    const get = id => (elements[id] = elements[id] || element(id));
    const clock = {ms: 1_790_000_000_000};
    class FakeDate extends Date {
        constructor(...a) { if (a.length) super(...a); else super(clock.ms); }
        static now() { return clock.ms; }
    }
    let nextId = 1;
    const timers = [];             // {id, fn, ms, kind, cleared, fired}
    const frames = [];             // {id, fn, cancelled, ran}
    const requests = [];           // {url, resolve, reject, aborted}
    const listeners = {};
    const addTimer = kind => (fn, ms) => {
        const t = {id: nextId++, fn, ms, kind, cleared: false, fired: false};
        timers.push(t);
        return t.id;
    };
    const clear = id => { const t = timers.find(x => x.id === id); if (t) t.cleared = true; };
    const sandbox = {
        console, JSON, Math, Number, String, Array, Object, Promise, Error, Set, Map,
        Infinity, NaN, URLSearchParams, encodeURIComponent, parseFloat, parseInt,
        Date: FakeDate,
        document: {
            visibilityState: 'visible',
            getElementById: get,
            querySelector: () => null, querySelectorAll: () => [],
            addEventListener: (type, fn) => (listeners[type] = listeners[type] || []).push(fn),
            removeEventListener() {},
            elementFromPoint: () => null,
            createElement: () => element(''),
            body: element('body'),
        },
        window: {addEventListener() {}, innerWidth: 1400, innerHeight: 900},
        setTimeout: addTimer('timeout'), clearTimeout: clear,
        setInterval: addTimer('interval'), clearInterval: clear,
        requestAnimationFrame: fn => {
            const f = {id: nextId++, fn, cancelled: false, ran: false};
            frames.push(f);
            return f.id;
        },
        cancelAnimationFrame: id => { const f = frames.find(x => x.id === id); if (f) f.cancelled = true; },
        AbortController, AbortSignal,
        esc: s => String(s ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;')
            .replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&#39;'),
        // core.js's counted noun: "1 window", "3 windows"
        plural: (n, one, many) => {
            const k = Number(n) || 0;
            return `${k.toLocaleString()} ${k === 1 ? one : (many || one + 's')}`;
        },
        fetch: (url, opts = {}) => new Promise((resolve, reject) => {
            const req = {url, resolve, reject, aborted: false};
            requests.push(req);
            const signal = opts.signal;
            if (signal) {
                const abort = () => {
                    req.aborted = true;
                    const e = new Error('The operation was aborted.');
                    e.name = 'AbortError';
                    reject(e);
                };
                if (signal.aborted) abort();
                else signal.addEventListener('abort', abort);
            }
        }),
    };
    sandbox.$ = get;
    sandbox.globalThis = sandbox;
    vm.createContext(sandbox);
    for (const f of files) {
        vm.runInContext(fs.readFileSync(f, 'utf8'), sandbox, {filename: f});
    }
    const settle = () => new Promise(r => setImmediate(r));
    const settled = async () => { await settle(); await settle(); await settle(); };
    return {
        sandbox, elements, requests, timers, frames, clock, get, settle: settled,
        run: code => vm.runInContext(code, sandbox),
        tick: sec => { clock.ms += sec * 1000; },
        // The timeouts still to come and the intervals still running
        pending: () => timers.filter(t => !t.cleared && !t.fired),
        fire: async t => {
            if (t.kind === 'timeout') t.fired = true;
            t.fn();
            await settled();
        },
        // One animation frame: every callback asked for so far, once
        frame: () => {
            const due = frames.filter(f => !f.cancelled && !f.ran);
            due.forEach(f => { f.ran = true; f.fn(clock.ms); });
            return due.length;
        },
        activeFrames: () => frames.filter(f => !f.cancelled && !f.ran).length,
        answer: async (i, body, {status = 200} = {}) => {
            requests[i].resolve({ok: status < 400, status, json: async () => body});
            await settled();
        },
        fail: async (i, error) => {
            requests[i].reject(error || new Error('Failed to fetch'));
            await settled();
        },
        setVisibility: state => {
            sandbox.document.visibilityState = state;
            (listeners.visibilitychange || []).forEach(fn => fn());
        },
    };
}

module.exports = {buildIt};
