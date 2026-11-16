/* Loads `deepsearch.js` in node — with a small DOM, fake timers and a
 * scripted fetch.
 *
 * The deep search is a poller, and what has to be right is WHEN it asks:
 * one request at a time, from the cursor on, never for a search no longer
 * shown, slower after failures, and not at all while nobody looks. Real
 * timers would make every such test slow and flaky: here time only moves
 * when a test says so (`tick(ms)`), and an answer only arrives when the
 * script has one (`answer(…)`) — or when a test lets a held one go
 * (`hold` → `release()`), to put a second request "while the first is on
 * its way" or to let an old answer arrive late.
 *
 * The DOM: every id in index.html is an element of its own (as on the
 * page); an element the script draws (`id="dsHits"` inside the view's
 * markup) is found in that markup and reads and writes its slice of it.
 * So what a test reads is what the page would show — rows appended later
 * land inside the view, not in a detached stand-in. An id that is in
 * neither is `null`, as in the browser.
 *
 * The helpers deepsearch.js borrows (svTime, svShortName, svPct,
 * svCopyList, the escaping of state.js, chatWhen) are the real ones, cut
 * out of their files; the search view itself (svTab, svPlay, svGoto) is a
 * stand-in that records.
 */
const fs = require('fs');
const vm = require('vm');

const DIR = process.env.JS_DIR;
const src = (n) => fs.readFileSync(DIR + '/' + n, 'utf8');
function fn(file, name) {
    const s = src(file);
    for (const head of ['async function ' + name + '(', 'function ' + name + '(']) {
        const at = s.indexOf(head);
        if (at >= 0) return s.slice(at, s.indexOf('\n}\n', at) + 3);
    }
    throw new Error(file + ' has no ' + name);
}

// ── Time ─────────────────────────────────────────────────────────────────
const clock = {now: 1700000000000};
let lastTimer = 0;
const timers = new Map();          // id → {at, ms, fn}
function setTimeoutF(f, ms) {
    const id = ++lastTimer;
    timers.set(id, {at: clock.now + (Number(ms) || 0), ms: Number(ms) || 0, fn: f});
    return id;
}
function clearTimeoutF(id) { timers.delete(id); }
const flush = async () => { for (let i = 0; i < 6; i++) await new Promise(r => setImmediate(r)); };
/* Moves the clock by `ms`, running every timer that falls due on the way. */
async function tick(ms) {
    const end = clock.now + (Number(ms) || 0);
    for (;;) {
        let due = null;
        for (const [id, t] of timers) {
            if (t.at <= end && (!due || t.at < due.t.at || (t.at === due.t.at && id < due.id))) due = {id, t};
        }
        if (!due) break;
        timers.delete(due.id);
        clock.now = Math.max(clock.now, due.t.at);
        due.t.fn();
        await flush();
    }
    clock.now = end;
    await flush();
}
class FakeDate extends Date {
    constructor(...a) { super(...(a.length ? a : [clock.now])); }
    static now() { return clock.now; }
}

// ── Fetch ────────────────────────────────────────────────────────────────
const requests = [];               // {url, method, body, aborted}
const script = [];                 // answer rules, first match wins
const held = [];                   // answers a test lets go with release()
const abortError = () => Object.assign(new Error('The operation was aborted'), {name: 'AbortError'});

/* `pattern`: a string the URL contains, or a RegExp. `body`: the JSON, or
 * a function (request) → JSON. opts: method, status (200), once (the rule
 * is used up), hold (the answer waits for release()), fail (a network
 * error instead of an answer). */
function answer(pattern, body, opts = {}) {
    script.unshift(Object.assign({pattern, body}, opts));      // newer rules win
}
function fetchF(url, opts = {}) {
    const method = String(opts.method || 'GET').toUpperCase();
    const req = {url: String(url), method, aborted: false,
                 body: opts.body ? JSON.parse(opts.body) : null};
    requests.push(req);
    const i = script.findIndex(s => (!s.method || s.method === method)
        && (s.pattern instanceof RegExp ? s.pattern.test(req.url) : req.url.includes(s.pattern)));
    const rule = i >= 0 ? script[i] : null;
    if (rule && rule.once) script.splice(i, 1);
    return new Promise((resolve, reject) => {
        let done = false;
        const settle = () => {
            if (done) return;
            done = true;
            if (!rule) return reject(new Error('no scripted answer for ' + method + ' ' + req.url));
            if (rule.fail) return reject(new Error(rule.fail));
            const body = typeof rule.body === 'function' ? rule.body(req) : rule.body;
            const status = rule.status || 200;
            resolve({ok: status >= 200 && status < 300, status,
                     json: () => Promise.resolve(JSON.parse(JSON.stringify(body ?? {})))});
        };
        if (opts.signal) {
            const abort = () => { if (done) return; done = true; req.aborted = true; reject(abortError()); };
            if (opts.signal.aborted) return abort();
            opts.signal.addEventListener('abort', abort);
        }
        if (rule && rule.hold) held.push({settle, req});
        else Promise.resolve().then(settle);
    });
}
function release() {
    const h = held.shift();
    if (h) h.settle();
    return h ? h.req : null;
}

// ── DOM ──────────────────────────────────────────────────────────────────
const STATIC = new Set([...src('../index.html').matchAll(/\sid="([^"]+)"/g)].map(m => m[1]));
const roots = {};

function classes(of) {
    return {
        add: (c) => of.set([...new Set([...of.get(), c])]),
        remove: (c) => of.set(of.get().filter(x => x !== c)),
        contains: (c) => of.get().includes(c),
        toggle: (c, force) => {
            const on = force === undefined ? !of.get().includes(c) : !!force;
            on ? of.set([...new Set([...of.get(), c])]) : of.set(of.get().filter(x => x !== c));
            return on;
        },
    };
}

function root(id) {
    if (roots[id]) return roots[id];
    let cls = [];
    const e = roots[id] = {
        id, value: '', textContent: '', hidden: false, disabled: false, style: {}, dataset: {},
        _html: '', isConnected: true, focused: false,
        get innerHTML() { return this._html; },
        set innerHTML(v) { this._html = String(v); },
        insertAdjacentHTML(where, html) {
            this._html = where === 'afterbegin' ? html + this._html : this._html + html;
        },
        focus() { this.focused = true; }, scrollIntoView() {},
    };
    e.classList = classes({get: () => cls, set: (v) => { cls = v; }});
    return e;
}

/* Where element `id` sits in some root's markup: the root, and the offsets
 * of its opening tag, its content and its closing tag. */
function locate(id) {
    const needle = ` id="${id}"`;
    for (const r of Object.values(roots)) {
        const at = r._html.indexOf(needle);
        if (at < 0) continue;
        const open = r._html.lastIndexOf('<', at);
        const tag = /^<([a-zA-Z][a-zA-Z0-9]*)/.exec(r._html.slice(open))[1];
        const start = r._html.indexOf('>', at) + 1;
        const re = new RegExp(`<${tag}(?=[\\s>])|</${tag}>`, 'g');
        re.lastIndex = start;
        let depth = 1, m;
        while ((m = re.exec(r._html))) {
            depth += m[0][1] === '/' ? -1 : 1;
            if (!depth) return {r, open, start, end: m.index, tag};
        }
        throw new Error('unclosed <' + tag + ' id="' + id + '">');
    }
    return null;
}

const props = {};                  // value, hidden, disabled … of drawn elements
function drawn(id) {
    const where = () => locate(id);
    const p = props[id] = props[id] || {dataset: {}, style: {}};
    const tagAttr = (w) => w.r._html.slice(w.open, w.start);
    const e = {
        id,
        get isConnected() { return !!where(); },
        get innerHTML() { const w = where(); return w ? w.r._html.slice(w.start, w.end) : ''; },
        set innerHTML(v) {
            const w = where();
            if (w) w.r._html = w.r._html.slice(0, w.start) + String(v) + w.r._html.slice(w.end);
        },
        get textContent() { return this.innerHTML.replace(/<[^>]*>/g, ''); },
        set textContent(v) {
            this.innerHTML = String(v).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
        },
        insertAdjacentHTML(where_, html) {
            const w = where();
            if (!w) return;
            const at = where_ === 'afterbegin' ? w.start : w.end;
            w.r._html = w.r._html.slice(0, at) + html + w.r._html.slice(at);
        },
        get dataset() {
            const w = where();
            const out = {};
            if (w) for (const m of tagAttr(w).matchAll(/data-([a-z-]+)="([^"]*)"/g)) out[m[1]] = m[2];
            return Object.assign(out, p.dataset);
        },
        get value() { return p.value ?? ''; }, set value(v) { p.value = v; },
        get hidden() { return !!p.hidden; }, set hidden(v) { p.hidden = !!v; },
        get disabled() { return !!p.disabled; }, set disabled(v) { p.disabled = !!v; },
        style: p.style,
        focus() {}, scrollIntoView() {},
    };
    e.classList = classes({
        get: () => {
            const w = where();
            const m = w && /\sclass="([^"]*)"/.exec(tagAttr(w));
            return m ? m[1].split(/\s+/).filter(Boolean) : [];
        },
        set: (list) => {
            const w = where();
            if (!w) return;
            const tag = tagAttr(w);
            const next = /\sclass="[^"]*"/.test(tag)
                ? tag.replace(/\sclass="[^"]*"/, ` class="${list.join(' ')}"`)
                : tag.replace(/^<([a-zA-Z0-9]+)/, `<$1 class="${list.join(' ')}"`);
            w.r._html = w.r._html.slice(0, w.open) + next + w.r._html.slice(w.start);
        },
    });
    return e;
}

function getElementById(id) {
    if (STATIC.has(id)) return root(id);
    return locate(id) ? drawn(id) : null;
}

const listeners = {};
const document = {
    visibilityState: 'visible',
    getElementById,
    querySelectorAll: () => [],
    querySelector: () => null,
    addEventListener: (type, f) => { (listeners[type] = listeners[type] || []).push(f); },
};

// ── The search view, as a stand-in that records ──────────────────────────
const calls = [];
const toasts = [];
const svState = {tab: 'text', query: ''};
const ctx = {
    console, document, AbortController,
    setTimeout: setTimeoutF, clearTimeout: clearTimeoutF, fetch: fetchF, Date: FakeDate,
    svState,
    svTab: (name) => {
        calls.push(['svTab', name]);
        const leaving = svState.tab;
        svState.tab = name;
        if (leaving === 'deep' && name !== 'deep') ctx.dsLeave();
        if (name === 'deep') ctx.dsOpen();
    },
    svPlay: (f, t, d) => { calls.push(['svPlay', f, t, d]); },
    svGoto: (f, t, d) => { calls.push(['svGoto', f, t, d]); },
    Toast: Object.fromEntries(['success', 'error', 'info', 'warn'].map(k =>
        [k, (msg) => { toasts.push([k, String(msg)]); }])),
};
vm.createContext(ctx);
vm.runInContext([
    fn('state.js', 'esc'), fn('state.js', 'escJs'),
    fn('searchview.js', 'svTime'), fn('searchview.js', 'svShortName'),
    fn('searchview.js', 'svPct'), fn('searchview.js', 'svCopyList'),
    fn('chat.js', 'chatWhen'),
    src('deepsearch.js'),
    'this.DS = DS;',
].join('\n'), ctx);

module.exports = {
    ctx, svState, calls, toasts, requests, held, timers, clock,
    tick, flush, answer, release,
    el: getElementById,
    html: (id) => (getElementById(id) || {innerHTML: ''}).innerHTML,
    /* The search open on the deep-search tab, as svTab('deep') leaves it. */
    show() {
        root('svOverlay').classList.add('visible');
        if (svState.tab !== 'deep') ctx.svTab('deep');
    },
    hideTab() { document.visibilityState = 'hidden'; (listeners.visibilitychange || []).forEach(f => f()); },
    showTab() { document.visibilityState = 'visible'; (listeners.visibilitychange || []).forEach(f => f()); },
    /* The timers that will ask the server again — the poll and the list,
     * not the 20-s guard of a request on its way. */
    pending: () => [...timers.values()].map(t => t.ms),
    polls: (id) => requests.filter(r => r.method === 'GET' && r.url.startsWith('/api/rag/deep/' + id)),
    lists: () => requests.filter(r => r.method === 'GET' && r.url === '/api/rag/deep'),
};
