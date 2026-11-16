/* Loads one or more static/js files in node with a forgiving fake DOM.
 *
 * Every getElementById answers with an element (created on first use and
 * remembered), so a script runs through its UI code without a browser; the
 * test then inspects the elements it touched. Fetch, EventSource, timers
 * and Toast are recorded fakes the test can drive.
 *
 * Used by tests/test_uifix_fileview_*.py.
 */
const fs = require('fs');
const path = require('path');
const vm = require('vm');

function classList() {
    const s = new Set();
    return {
        _s: s,
        add(...c) { c.forEach(x => s.add(x)); },
        remove(...c) { c.forEach(x => s.delete(x)); },
        contains(c) { return s.has(c); },
        toggle(c, on) {
            const want = on === undefined ? !s.has(c) : !!on;
            want ? s.add(c) : s.delete(c);
            return want;
        },
    };
}

function element(id) {
    return {
        id, value: '', textContent: '', innerHTML: '', className: '', title: '',
        disabled: false, hidden: false, checked: false, href: '',
        dataset: {}, style: {}, children: [], files: [],
        classList: classList(), isConnected: true,
        addEventListener() {}, removeEventListener() {},
        appendChild(c) { this.children.push(c); return c; }, removeChild() {},
        remove() { this.isConnected = false; },
        querySelector() { return null; }, querySelectorAll() { return []; },
        closest() { return null; }, scrollIntoView() {}, focus() {}, select() {},
        getBoundingClientRect() { return {left: 0, width: 100, top: 0, height: 20}; },
    };
}

function buildIt({files = [], fetch: answer = null} = {}) {
    const elements = {};
    const requests = [];
    const toasts = [];
    const timers = [];
    const sources = [];
    const byId = id => (elements[id] = elements[id] || element(id));
    const document = {
        getElementById: byId,
        createElement: kind => element(kind),
        querySelector: () => null,
        querySelectorAll: () => [],
        addEventListener() {},
        body: element('body'),
    };
    const toast = type => (msg, dur) => {
        const el = element('toast');
        el.children = [element('icon'), element('text')];
        el.children[1].textContent = msg;
        toasts.push({type, msg, el});
        return el;
    };
    const sandbox = {
        console, JSON, Math, Number, String, Array, Object, Date, RegExp, Set, Map,
        Promise, Error, parseFloat, parseInt, isNaN, encodeURIComponent,
        decodeURIComponent, URL, URLSearchParams,
        document,
        window: {addEventListener() {}, location: {href: '', hostname: 'app'}},
        localStorage: {getItem: () => null, setItem() {}},
        requestAnimationFrame: fn => fn(),
        setTimeout: (fn, ms) => { timers.push({fn, ms}); return timers.length; },
        clearTimeout() {}, setInterval: () => 0, clearInterval() {},
        Toast: {info: toast('info'), success: toast('success'), warn: toast('warn'),
                error: toast('error')},
        EventSource: function (url) { this.url = url; this.close = () => { this.closed = true; };
                                      sources.push(this); },
        esc: s => String(s ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;')
                                 .replace(/>/g, '&gt;').replace(/"/g, '&quot;')
                                 .replace(/'/g, '&#39;'),
        // core.js's counted noun and speed factor
        plural: (n, one, many) => {
            const k = Number(n) || 0;
            return `${k.toLocaleString()} ${k === 1 ? one : (many || one + 's')}`;
        },
        fmtSpeed: x => {
            x = Number(x) || 0;
            return (x >= 10 ? x.toFixed(0) : x >= 0.1 ? x.toFixed(1) : x.toFixed(2)) + '×';
        },
        fetch: async (url, opts) => {
            requests.push({url, body: opts && opts.body ? JSON.parse(opts.body) : null});
            const r = answer ? await answer(url, opts) : {};
            if (r && r.__status) {
                return {ok: r.__status < 400, status: r.__status,
                        json: async () => r.body || {}, text: async () => ''};
            }
            return {ok: true, status: 200, json: async () => r, text: async () => String(r.text || '')};
        },
    };
    sandbox.window.document = document;
    sandbox.globalThis = sandbox;
    vm.createContext(sandbox);
    const dir = path.resolve(__dirname, '..', '..', 'whisper_tm', 'static', 'js');
    for (const f of files) {
        vm.runInContext(fs.readFileSync(path.join(dir, f), 'utf8'), sandbox, {filename: f});
    }
    return {
        sandbox, elements, requests, toasts, timers, sources, element, byId,
        run: code => vm.runInContext(code, sandbox),
        settle: () => new Promise(resolve => setImmediate(resolve)),
    };
}

module.exports = {buildIt, element};
