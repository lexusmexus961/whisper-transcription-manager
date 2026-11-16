/* Loads `txoptions.js` and/or `llmtools.js` in node, with a small fake DOM.
 *
 * Covers what the LLM dialogs send when the user changes nothing: the model
 * entry (which must leave the model to the task setting) and the correction
 * dialog's score threshold (which must come from the settings). Elements
 * are created on demand and only carry the properties the dialogs read and
 * write. A <select> keeps its options and, like the DOM, takes the first
 * option appended while it has none; its value only takes what its options
 * offer.
 *
 * `fetch` records every request and answers from a table ({url: [status,
 * body]}); an unknown URL gets 200 with an empty object.
 */
const fs = require('fs');
const path = require('path');
const vm = require('vm');

function unescape(s) {
    return s.replace(/&quot;/g, '"').replace(/&#39;/g, "'").replace(/&lt;/g, '<')
            .replace(/&gt;/g, '>').replace(/&amp;/g, '&');
}

function element(id) {
    const classes = new Set();
    return {
        id, value: '', checked: false, disabled: false, hidden: false,
        textContent: '', innerHTML: '', title: '', style: {}, dataset: {},
        classList: {
            add: c => classes.add(c), remove: c => classes.delete(c),
            contains: c => classes.has(c),
            toggle: (c, on) => {
                const want = on === undefined ? !classes.has(c) : !!on;
                if (want) classes.add(c); else classes.delete(c);
                return want;
            },
        },
        querySelector: () => element(id + '-child'),
        querySelectorAll: () => [],
        addEventListener() {}, focus() {},
    };
}

function select(id) {
    const el = element(id);
    el.options = [];
    let chosen = '';
    Object.defineProperty(el, 'value', {
        get: () => chosen,
        set: v => { chosen = el.options.some(o => o.value === v) ? v : ''; },
    });
    let html = '';
    Object.defineProperty(el, 'innerHTML', {
        get: () => html,
        set: v => {
            html = String(v);
            el.options = [...html.matchAll(/<option value="([^"]*)"[^>]*>([^<]*)<\/option>/g)]
                .map(m => ({value: unescape(m[1]), textContent: unescape(m[2])}));
            chosen = el.options.length ? el.options[0].value : '';
        },
    });
    el.appendChild = o => {
        el.options.push({value: String(o.value), textContent: o.textContent});
        if (el.options.length === 1) chosen = String(o.value);
        return o;
    };
    return el;
}

function buildIt({files = [], selects = [], answers = {}} = {}) {
    const elements = {};
    const get = id => {
        if (!elements[id]) elements[id] = selects.includes(id) ? select(id) : element(id);
        return elements[id];
    };
    const requests = [];
    const sandbox = {
        console,
        // Nothing may keep polling after the test: the dialogs' progress
        // loops and the queue badge are not what is tested here
        setTimeout: () => 0, clearTimeout() {}, setInterval: () => 0, clearInterval() {},
        document: {
            getElementById: get,
            addEventListener() {},
            createElement: () => ({}),
            querySelector: () => null,
            querySelectorAll: () => [],
        },
        window: {},
        fetch: async (url, opts) => {
            requests.push({url, body: opts && opts.body ? JSON.parse(opts.body) : null});
            const [code, body] = answers[url] || [200, {}];
            return {ok: code < 400, status: code, json: async () => body};
        },
        Toast: {success() {}, error() {}, info() {}},
        esc: s => String(s ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;')
                                 .replace(/>/g, '&gt;').replace(/"/g, '&quot;'),
    };
    sandbox.globalThis = sandbox;
    vm.createContext(sandbox);
    const dir = process.env.JS_DIR;
    for (const f of files) {
        vm.runInContext(fs.readFileSync(path.join(dir, f), 'utf8'), sandbox, {filename: f});
    }
    return {
        elements, get, requests,
        run: code => vm.runInContext(code, sandbox),
        // Lets the fake fetches and everything chained on them finish
        settle: () => new Promise(r => global.setTimeout(r, 20)),
    };
}

module.exports = {buildIt};
