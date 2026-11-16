/* Loads one or more static/js files in node with a permissive fake DOM and
 * a fetch whose answers the test hands out in any order — for the "an older
 * answer must not overwrite a newer one" guards of the smaller views
 * (performance, health, statistics, voice history, extract).
 *
 * `fetch` returns a promise that stays pending until the test calls
 * `answer(i, body, {status})` or `fail(i, error)` for request number i.
 * An aborted request (AbortSignal) rejects with an AbortError, like the
 * browser does. Timers are recorded and never fire on their own.
 */
const fs = require('fs');
const vm = require('vm');

function element(id) {
    const classes = new Set();
    return {
        id, innerHTML: '', textContent: '', value: '', title: '', hidden: false,
        disabled: false, style: {}, dataset: {}, clientWidth: 900, scrollTop: 0,
        classList: {
            add: c => classes.add(c), remove: c => classes.delete(c),
            toggle: (c, on) => ((on ?? !classes.has(c)) ? classes.add(c) : classes.delete(c)),
            contains: c => classes.has(c),
        },
        querySelector: () => null, querySelectorAll: () => [],
        addEventListener() {}, removeEventListener() {}, appendChild() {}, prepend() {},
        remove() {}, closest: () => null, focus() {}, blur() {},
        getBoundingClientRect: () => ({left: 0, top: 0, width: 900, height: 400}),
    };
}

function buildIt(files) {
    const elements = {};
    const get = id => (elements[id] = elements[id] || element(id));
    const requests = [];
    const timers = [];
    const sandbox = {
        console, JSON, Math, Number, String, Array, Object, Date, Promise, Error,
        URLSearchParams, Set, Map, encodeURIComponent, parseFloat, parseInt,
        document: {
            getElementById: get,
            querySelector: () => null, querySelectorAll: () => [],
            addEventListener() {}, removeEventListener() {},
            createElement: () => element(''),
            body: element('body'),
        },
        window: {addEventListener() {}},
        localStorage: {getItem: () => null, setItem() {}, removeItem() {}},
        setTimeout: (fn, ms) => { timers.push({fn, ms}); return timers.length; },
        clearTimeout() {}, setInterval: () => 0, clearInterval() {},
        requestAnimationFrame: () => 0,
        AbortController, AbortSignal,
        Toast: {success() {}, error() {}, info() {}, warn() {}},
        esc: s => String(s ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;')
            .replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&#39;'),
        escJs: s => String(s),
        fetch: (url, opts = {}) => new Promise((resolve, reject) => {
            const req = {url, resolve, reject};
            requests.push(req);
            const signal = opts.signal;
            if (signal) {
                const abort = () => {
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
    return {
        sandbox, elements, requests, timers, get, settle,
        run: code => vm.runInContext(code, sandbox),
        answer: async (i, body, {status = 200} = {}) => {
            requests[i].resolve({ok: status < 400, status, json: async () => body});
            await settle(); await settle();
        },
        fail: async (i, error) => {
            requests[i].reject(error || new Error('Failed to fetch'));
            await settle(); await settle();
        },
    };
}

module.exports = {buildIt};
