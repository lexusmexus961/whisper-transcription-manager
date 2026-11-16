/* Loads `settings.js` in node, with a small fake DOM — and, with
 * SETTINGS_EXTRA (paths separated by ":"), the page definitions too.
 *
 * Covers what no endpoint test can see: which form fields the Save button
 * collects (a document-wide query once picked up the queue editor's folder
 * rows and made every save throw), how a control's value keeps its type,
 * the settings search's match logic, and what the pages draw — whose pure
 * functions run here without any DOM.
 *
 * Selectors: the fake understands exactly the forms settings.js uses for
 * collecting — `[attr]`, `tag[attr]`, `tag[attr="value"]`, optionally
 * behind `#settingsOverlay `. Anything else throws, so a changed selector
 * fails the test loudly instead of quietly matching nothing.
 */
const fs = require('fs');
const vm = require('vm');

function matches(el, selector) {
    const m = selector.match(/^([a-z]*)\[([a-z-]+)(?:="([^"]*)")?\]$/);
    if (!m) throw new Error('selector not supported by the harness: ' + selector);
    const [, tag, attr, value] = m;
    if (tag && el.tagName !== tag.toUpperCase()) return false;
    if (!(attr in el.attrs)) return false;
    return value === undefined || el.attrs[attr] === value;
}

// A plain element: tag, attributes (data-* also as dataset), form props
function element(tag, attrs = {}, props = {}) {
    const dataset = {};
    for (const [k, v] of Object.entries(attrs)) {
        if (k.startsWith('data-')) {
            dataset[k.slice(5).replace(/-([a-z])/g, (_, c) => c.toUpperCase())] = v;
        }
    }
    const el = {tagName: tag.toUpperCase(), attrs, dataset, style: {}, min: '', max: '',
                type: '', ...props};
    // Like the DOM: only form controls have a value
    if (!['INPUT', 'SELECT', 'TEXTAREA'].includes(el.tagName)) delete el.value;
    return el;
}

function container(id, children) {
    return {
        id, children,
        classList: {add() {}, remove() {}, contains: () => false, toggle() {}},
        querySelectorAll(selector) {
            return children.filter(c => matches(c, selector));
        },
        querySelector(selector) {
            return this.querySelectorAll(selector)[0] || null;
        },
    };
}

/* `inside`: the elements of the settings dialog; `outside`: elements
 * elsewhere on the page (e.g. the queue editor's folder rows). */
function buildIt({inside = [], outside = []} = {}) {
    const overlay = container('settingsOverlay', inside);
    const all = [...inside, ...outside];
    const listeners = {};
    const document = {
        getElementById: (id) => (id === 'settingsOverlay' ? overlay : null),
        querySelectorAll(selector) {
            if (selector.startsWith('#settingsOverlay ')) {
                return overlay.querySelectorAll(selector.slice('#settingsOverlay '.length));
            }
            return all.filter(c => matches(c, selector));
        },
        querySelector(selector) { return this.querySelectorAll(selector)[0] || null; },
        addEventListener(type, fn) { (listeners[type] = listeners[type] || []).push(fn); },
        createElement: (tag) => element(tag),
        body: {appendChild() {}},
    };
    const store = {};
    const sandbox = {
        document, console,
        window: {},
        localStorage: {
            getItem: (k) => (k in store ? store[k] : null),
            setItem: (k, v) => { store[k] = String(v); },
            removeItem: (k) => { delete store[k]; },
        },
        _wxpDebugEnabled: false,
        setTimeout, clearTimeout,
        location: {hostname: 'test.local'},
        Event: class { constructor(type, opts) { this.type = type; Object.assign(this, opts); } },
        esc: (s) => String(s ?? '')
            .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;').replace(/'/g, '&#39;'),
        // Same as state.js: for values inside a JS string in an onclick
        escJs: (s) => String(s)
            .replace(/\\/g, '\\\\').replace(/'/g, '\\x27').replace(/"/g, '\\x22')
            .replace(/\n/g, '\\n').replace(/\r/g, '\\r')
            .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;'),
    };
    sandbox.globalThis = sandbox;
    vm.createContext(sandbox);
    vm.runInContext(fs.readFileSync(process.env.SETTINGS_JS, 'utf8'), sandbox,
                    {filename: 'settings.js'});
    for (const extra of (process.env.SETTINGS_EXTRA || '').split(':').filter(Boolean)) {
        vm.runInContext(fs.readFileSync(extra, 'utf8'), sandbox, {filename: extra});
    }
    return {
        sandbox, listeners,
        run: (code) => vm.runInContext(code, sandbox),
        value: (expression) => vm.runInContext(`(${expression})`, sandbox),
    };
}

module.exports = {buildIt, element};
