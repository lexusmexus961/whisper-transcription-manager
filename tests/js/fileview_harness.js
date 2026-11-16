/* The file view without a browser — only as much DOM as the header
 * area touches. `fileview.js` brings a lot more (transcript, toolbar,
 * search); only what hangs off the file header is tested here.
 */
const fs = require('fs');
const vm = require('vm');

function element(id) {
    return {
        id, value: '', textContent: '', innerHTML: '', className: '',
        dataset: {}, style: {},
        classList: {
            _s: new Set(),
            add(c) { this._s.add(c); }, remove(c) { this._s.delete(c); },
            contains(c) { return this._s.has(c); },
            toggle(c, force) {
                const expected = force === undefined ? !this._s.has(c) : force;
                expected ? this._s.add(c) : this._s.delete(c);
                return expected;
            },
        },
        addEventListener() {}, appendChild() {}, removeChild() {},
        querySelector() { return null; }, closest() { return null; },
        select() {}, focus() {},
        getBoundingClientRect() { return {left: 0, width: 100, top: 0, height: 20}; },
    };
}

function buildIt(directory) {
    const elements = {dirInput: element('dirInput')};
    elements.dirInput.value = directory || '';
    const created = [];
    const copied = [];

    const fakeDocument = {
        getElementById: (id) => elements[id] || null,
        createElement: (kind) => { const e = element(kind); created.push(e); return e; },
        body: {appendChild() {}, removeChild() {}},
        addEventListener() {},
        execCommand(command) {
            if (command === 'copy' && created.length) {
                copied.push(created[created.length - 1].value);
            }
            return true;
        },
        querySelector() { return null; },
        querySelectorAll() { return []; },
    };

    const sandbox = {
        document: fakeDocument,
        window: {isSecureContext: false, addEventListener() {},
                 location: {href: '', hostname: 'testhost'}},
        navigator: {},                       // no clipboard API, like over http
        console, localStorage: {getItem: () => null, setItem() {}},
        setTimeout: () => 0, clearTimeout: () => {}, setInterval: () => 0,
        clearInterval: () => {},
        fetch: async () => ({ok: true, json: async () => ({})}),
        // core.js's counted noun: "1 window", "3 windows"
        plural: (n, one, many) => {
            const k = Number(n) || 0;
            return `${k.toLocaleString()} ${k === 1 ? one : (many || one + 's')}`;
        },
        Math, JSON, Number, String, Array, Object, Date, parseFloat, parseInt,
        isNaN, encodeURIComponent, decodeURIComponent, RegExp, Set, Map,
    };
    sandbox.globalThis = sandbox;
    vm.createContext(sandbox);
    vm.runInContext(fs.readFileSync(process.env.FILEVIEW_JS, 'utf8'),
                    sandbox, {filename: 'fileview.js'});
    // The recording's header, states and Details tab (recording.js)
    const rec = require('path').join(require('path').dirname(process.env.FILEVIEW_JS), 'recording.js');
    if (fs.existsSync(rec)) vm.runInContext(fs.readFileSync(rec, 'utf8'), sandbox, {filename: 'recording.js'});

    return {
        sandbox, elements, copied,
        run: (code) => vm.runInContext(code, sandbox),
        value: (expression) => vm.runInContext(`(${expression})`, sandbox),
    };
}

module.exports = {buildIt};
