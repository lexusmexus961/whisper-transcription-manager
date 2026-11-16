/* Loads `summarize.js` in node, with a small fake DOM.
 *
 * Covers the template logic of the summarize dialog: which template is
 * chosen on open, when the system prompt counts as edited, that switching
 * away from an edited prompt can be undone, the hints for speaker labels /
 * timestamps, and what a run sends. A <select> understands the options its
 * innerHTML is given; other elements only carry the properties the dialog
 * reads and writes.
 */
const fs = require('fs');
const vm = require('vm');

function unescape(s) {
    return s.replace(/&quot;/g, '"').replace(/&#39;/g, "'").replace(/&lt;/g, '<')
            .replace(/&gt;/g, '>').replace(/&amp;/g, '&');
}

function element(id) {
    const attrs = {};
    const classes = new Set();
    const el = {
        id, value: '', checked: false, hidden: false, disabled: false,
        textContent: '', title: '', style: {}, dataset: {},
        classList: {
            add: c => classes.add(c), remove: c => classes.delete(c),
            contains: c => classes.has(c),
            toggle: (c, on) => {
                const want = on === undefined ? !classes.has(c) : !!on;
                if (want) classes.add(c); else classes.delete(c);
                return want;
            },
        },
        setAttribute: (k, v) => { attrs[k] = String(v); },
        getAttribute: k => (k in attrs ? attrs[k] : null),
        querySelector: () => element(id + '-child'),
        querySelectorAll: () => [],
        addEventListener() {}, focus() {}, select() {},
    };
    let html = '';
    Object.defineProperty(el, 'innerHTML', {
        get: () => html,
        set: v => {
            html = String(v);
            if (id === 'sumPreset' || id === 'sumSource' || id === 'sumModel') {
                el.options = [...html.matchAll(/<option value="([^"]*)"[^>]*>([^<]*)<\/option>/g)]
                    .map(m => ({value: unescape(m[1]), textContent: unescape(m[2])}));
                el._value = el.options.length ? el.options[0].value : '';
            }
        },
    });
    return el;
}

// A <select>: value only takes what its options offer, and the first option
// appended to an empty one is chosen — like the DOM
function select(id) {
    const el = element(id);
    el.options = [];
    el._value = '';
    Object.defineProperty(el, 'value', {
        get: () => el._value,
        set: v => { el._value = el.options.some(o => o.value === v) ? v : ''; },
    });
    el.appendChild = o => {
        el.options.push({value: o.value, textContent: o.textContent});
        if (el.options.length === 1) el._value = o.value;
    };
    return el;
}

function buildIt({status = {}, prefs = null, answers = {}} = {}) {
    const elements = {};
    const get = id => {
        if (!elements[id]) {
            elements[id] = ['sumPreset', 'sumSource', 'sumModel'].includes(id)
                ? select(id) : element(id);
        }
        return elements[id];
    };
    get('sumTimestamps').checked = true;
    const toggles = ['filename', 'duration', 'language', 'model', 'date', 'speakers'].map(k => {
        const t = element('meta-' + k);
        t.dataset.meta = k;
        t.setAttribute('aria-pressed', String(k === 'filename' || k === 'duration'));
        return t;
    });
    const store = {};
    if (prefs) store.wtm_sumPrefs = JSON.stringify(prefs);
    const requests = [];
    const queued = [];
    const toasts = [];
    const sandbox = {
        console, setTimeout, clearTimeout, JSON, Math, Object, Array, Set, Map, String,
        document: {
            getElementById: get,
            querySelectorAll: sel => {
                if (sel === '#sumMeta .sum-toggle') return toggles;
                return [];
            },
            querySelector: sel => {
                const m = sel.match(/^#sumMeta \.sum-toggle\[data-meta="(\w+)"\]$/);
                if (m) return toggles.find(t => t.dataset.meta === m[1]) || null;
                return null;
            },
            addEventListener() {},
            createElement: () => ({}),
        },
        localStorage: {
            getItem: k => (k in store ? store[k] : null),
            setItem: (k, v) => { store[k] = String(v); },
        },
        fetch: async (url, opts) => {
            requests.push({url, body: opts && opts.body ? JSON.parse(opts.body) : null});
            const [code, body] = answers[url] || [200, {}];
            return {ok: code < 400, status: code, json: async () => body};
        },
        Toast: {success: m => toasts.push(['success', m]), error: m => toasts.push(['error', m]),
                info: m => toasts.push(['info', m])},
        esc: s => String(s ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;')
                                 .replace(/>/g, '&gt;').replace(/"/g, '&quot;'),
        // core.js's counted noun: "1 window", "3 windows"
        plural: (n, one, many) => {
            const k = Number(n) || 0;
            return `${k.toLocaleString()} ${k === 1 ? one : (many || one + 's')}`;
        },
        llmQueueOpen: async () => null,
        llmQueueAdd: async (kind, filename, params) => { queued.push({kind, filename, params}); return 7; },
        llmQueueTrack: () => ({stop() {}}),
        llmResetParts() {}, llmPartsHtml: () => '', llmWaitText: () => '',
        activeTranscript: {filename: 'a.mp4', segments: [{start: 0, text: 'hi'}], speakers: []},
        activeFile: 'a.mp4', files: [], fileByName: () => null,
    };
    sandbox.globalThis = sandbox;
    vm.createContext(sandbox);
    vm.runInContext(fs.readFileSync(process.env.SUMMARIZE_JS, 'utf8'), sandbox,
                    {filename: 'summarize.js'});
    sandbox._sumLlmStatus = status;
    vm.runInContext('_sumLlmStatus = globalThis._sumLlmStatus;', sandbox);
    return {
        elements, get, requests, queued, toasts, store,
        run: code => vm.runInContext(code, sandbox),
    };
}

module.exports = {buildIt};
