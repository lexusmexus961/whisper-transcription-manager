/* `filelist.js` in node — with just enough DOM that the scroll logic
 * runs.
 *
 * A single question is tested: does the spot the user is currently
 * reading stay put when a background reload happens? Everything else
 * (row layout, selection, queue tabs) is incidental here and may stay
 * a stub.
 */
const fs = require('fs');
const vm = require('vm');

function element(id) {
    const e = {
        id, value: '', textContent: '', className: '', dataset: {}, style: {},
        scrollTop: 0, clientHeight: 600, scrollHeight: 0, offsetHeight: 28,
        children: [],
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
        addEventListener() {}, removeEventListener() {},
        setAttribute(k, v) { this.dataset[k.replace(/^data-/, '')] = v; },
        getAttribute(k) { return this.dataset[k.replace(/^data-/, '')] ?? null; },
        appendChild(k) { this.children.push(k); return k; },
        removeChild(k) {
            const i = this.children.indexOf(k);
            if (i >= 0) this.children.splice(i, 1);
            return k;
        },
        insertAdjacentHTML() {},
        querySelector() { return null; },
        querySelectorAll() { return []; },
        closest() { return null; },
        getBoundingClientRect() {
            return {left: 0, top: 0, width: 400, height: 28};
        },
        get innerHTML() { return this._html || ''; },
        set innerHTML(v) {
            this._html = v;
            this.children = [];
            /* When the browser actually clamps scrollTop to 0 — and when
             * it doesn't. That's the difference the bug hinged on:
             *
             * It gets clamped on the next layout, not on assignment.
             * Whoever clears the list and refills it in the SAME tick
             * (that's how `_renderWindow` works while scrolling) sees no
             * jump because of it. Whoever clears it and then waits for a
             * server response (`reloadFileList`) certainly does.
             *
             * A stub that always sets it to 0 immediately could never
             * have shown the bug — it would have shown it everywhere. */
            queueMicrotask(() => {
                if (!this.children.length) this.scrollTop = 0;
            });
        },
    };
    return e;
}

function buildIt({total = 5000} = {}) {
    const elements = {};
    const fetch = (id) => (elements[id] ||= element(id));
    ['fileList', 'fileFilter', 'dirStatus', 'scanSpinner', 'fileCount',
     'batchButtons', 'selCount'].forEach(fetch);

    const fakeDocument = {
        getElementById: (id) => elements[id] || null,
        createElement: () => element('new_value'),
        createDocumentFragment: () => element('ask'),
        addEventListener() {}, removeEventListener() {},
        querySelector: () => null, querySelectorAll: () => [],
        body: {appendChild() {}, removeChild() {}},
    };

    // The server: one page with `total` (total) entries.
    const fetched = [];
    const sandbox = {
        document: fakeDocument,
        window: {addEventListener() {}, location: {href: ''}},
        console,
        setTimeout: (fn) => { return 0; }, clearTimeout: () => {},
        setInterval: () => 0, clearInterval: () => {},
        requestAnimationFrame: (fn) => { fn(); return 1; },
        fetch: async (url) => {
            fetched.push(url);
            // Like on the network: not in the same tick. That's why
            // the browser does a layout pass in between — and exactly
            // there the scroll position got lost.
            await new Promise((r) => setImmediate(r));
            const page = Number((/offset=(\d+)/.exec(url) || [])[1] || 0);
            return {
                ok: true,
                json: async () => ({
                    total: total,
                    offset: page,
                    files: Array.from({length: Math.min(500, total - page)},
                        (_, i) => ({name: `file_${page + i}.mp4`,
                                    display_name: `file_${page + i}.mp4`,
                                    status: 'done', size_fmt: '1 MB',
                                    modified: 0})),
                }),
            };
        },
        Math, JSON, Number, String, Array, Object, Date, Set, Map, Promise,
        Float64Array, Int32Array, parseFloat, parseInt, isNaN,
        encodeURIComponent, decodeURIComponent, RegExp, Error,
        // What comes from other files and has nothing to test here.
        esc: (x) => String(x ?? ''), escJs: (x) => String(x ?? ''),
        updateCounts() {}, renderBatchButtons() {}, loadServerStats() {},
        markActiveFile() {}, filesChanged() {},
        renderQueueList() {}, openQueueFile() {}, moveQueueItem() {},
        clearQueue() {}, deleteTx() {}, goToSearchResult() {},
        selectFile() {}, updateSelectionUI() {}, wtmDebug() {},
        activeTab: 'all', sortKey: 'name', sortAsc: true,
        files: [], queueItems: [], queueCurrentFile: null, queuePending: 0,
        selectedFiles: new Set(), activeFile: null,
        queueOutcomes: new Map(),
        // state.js: queue entries by identity, as there
        queuedKeys: () => new Set(sandbox.queueItems.map(q => q.abs || q.filename)),
        queuedIn: (keys, f) => !!f && ((!!f.abs_path && keys.has(f.abs_path)) || keys.has(f.name)),
        queueOutcomeOf: (f) => (f && ((f.abs_path && sandbox.queueOutcomes.get(f.abs_path))
                                      || sandbox.queueOutcomes.get(f.name))) || null,
    };
    sandbox.globalThis = sandbox;
    vm.createContext(sandbox);
    vm.runInContext(fs.readFileSync(process.env.FILELIST_JS, 'utf8'),
                    sandbox, {filename: 'filelist.js'});

    return {
        sandbox, elements, fetched,
        listing: elements.fileList,
        run: (code) => vm.runInContext(code, sandbox),
        value: (a) => vm.runInContext(`(${a})`, sandbox),
    };
}

module.exports = {buildIt};
