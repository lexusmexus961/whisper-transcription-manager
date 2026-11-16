/* Loads `pathpicker.js` in node, with just enough DOM for the path field.
 *
 * `fetch` answers from `answers` and records every URL; `changeDir` (from
 * directory.js in the browser) is a stub that records what the path field
 * held when it was called — that is the folder the app opens. The queue
 * editor's "Add folder" field is the second place: `qaFolderChosen`
 * records what its field held. Favorites live in `favs`.
 */
const fs = require('fs');
const vm = require('vm');

function element(id) {
    const classes = new Set();
    return {
        id, value: '', textContent: '', className: '', innerHTML: '', style: {}, title: '',
        classList: {add: c => classes.add(c), remove: c => classes.delete(c),
                    toggle: (c, on) => (on ? classes.add(c) : classes.delete(c)),
                    contains: c => classes.has(c)},
        blur() {}, focus() {},
    };
}

function buildIt({answers = {}} = {}) {
    const ids = ['dirInput', 'pathPicker', 'pathField', 'dirStatus', 'ppList',
                 'ppBreadcrumb', 'ppFavs', 'ppHint', 'btnFavStar',
                 'qaFolder', 'qaPathField', 'qaPathPicker', 'qaPpBreadcrumb', 'qaPpFavs',
                 'qaPpList', 'qaPpHint', 'qaFavStar'];
    const elements = Object.fromEntries(ids.map(id => [id, element(id)]));
    const requests = [];
    const opened = [];
    const chosen = [];
    let favs = [];
    const sandbox = {
        console, setTimeout,
        document: {
            getElementById: (id) => elements[id] || null,
            querySelector: () => null,
            addEventListener() {}, removeEventListener() {},
        },
        fetch: async (url) => {
            requests.push(url);
            const [status, body] = answers[url] || [404, {error: 'not found'}];
            return {ok: status < 400, status, json: async () => body};
        },
        changeDir: () => { opened.push(elements.dirInput.value); },
        qaFolderChosen: () => { chosen.push(elements.qaFolder.value); },
        updateFavBtn() {},
        loadFavDirs: () => favs.slice(),
        saveFavDirs: (a) => { favs = a.slice(); },
        esc: (s) => String(s ?? ''),
        escJs: (s) => String(s ?? ''),
    };
    sandbox.globalThis = sandbox;
    vm.createContext(sandbox);
    vm.runInContext(fs.readFileSync(process.env.PATHPICKER_JS, 'utf8'), sandbox,
                    {filename: 'pathpicker.js'});
    return {
        elements, requests, opened, chosen,
        favs: () => favs,
        run: (code) => vm.runInContext(code, sandbox),
    };
}

module.exports = {buildIt};
