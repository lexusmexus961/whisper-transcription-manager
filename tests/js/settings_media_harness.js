/* Settings → Images & sounds (settings_media.js) without a browser: plain
 * objects for the boxes it writes into, and the helpers it shares with the
 * other settings pages — the coverage bar of the semantic search's index
 * (settings_tools.js), the page links and the byte format — taken from their
 * own files, so the two cards stay one design.
 */
const fs = require('fs');
const vm = require('vm');

function element(id) {
    return {id, innerHTML: '', textContent: '', className: '', hidden: false, disabled: false};
}

function grab(source, name) {
    const at = source.indexOf('function ' + name + '(');
    if (at < 0) throw new Error('no ' + name);
    return source.slice(at, source.indexOf('\n}\n', at) + 3);
}

function buildIt() {
    const ids = ['msSetEnv', 'msSetEnvDetail', 'msSetIndex', 'msSetCoverage', 'msSetEstimate', 'msSetRun',
                 'msSetStart', 'msSetStop', 'msSetDescribe', 'msSetDescribeBtn'];
    const elements = Object.fromEntries(ids.map((id) => [id, element(id)]));
    const read = (name) => fs.readFileSync(process.env.JS_DIR + '/' + name, 'utf8');
    const sandbox = {
        document: {getElementById: (id) => elements[id] || null},
        console, Math, JSON, Number, String, Array, Object, Set, Map, Promise,
        setInterval: () => 1, clearInterval: () => {},
    };
    sandbox.globalThis = sandbox;
    vm.createContext(sandbox);
    const state = read('state.js');
    vm.runInContext([grab(state, 'esc'), grab(state, 'escJs'),
        grab(read('settings.js'), '_fmtBytesS'),
        grab(read('settings_pages.js'), '_sGo'),
        grab(read('settings_tools.js'), '_sCoverageHtml')].join('\n'), sandbox);
    vm.runInContext(read('settings_media.js'), sandbox, {filename: 'settings_media.js'});
    return {
        sandbox, elements,
        run: (code) => vm.runInContext(code, sandbox),
    };
}

module.exports = {buildIt};
