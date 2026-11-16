/* `queue_editor.js` in node — with just enough DOM that filter, selection,
 * dry run → confirmation → real run and the mode bars work.
 *
 * `fetch` records every request and answers it from a table
 * (setAnswer/setStatus). `api()` is core.js's version in miniature:
 * a JSON body, and on !ok an error with body.error or `HTTP <status>`
 * and — like core.js — `err.status`.
 */
const fs = require('fs');
const vm = require('vm');

function element(id) {
    const e = {
        id, value: '', textContent: '', className: '', dataset: {}, style: {},
        checked: false, disabled: false, scrollTop: 0, title: '',
        children: [], _parents: null,
        classList: {
            _s: new Set(),
            add(...cs) { cs.forEach(c => this._s.add(c)); }, remove(c) { this._s.delete(c); },
            contains(c) { return this._s.has(c); },
            toggle(c, force) {
                const expected = force === undefined ? !this._s.has(c) : force;
                expected ? this._s.add(c) : this._s.delete(c);
                return expected;
            },
        },
        addEventListener() {}, removeEventListener() {},
        setAttribute(k, v) { this.dataset[k.replace(/^data-/, '')] = v; },
        appendChild(child) { this.children.push(child); child._parents = this; return child; },
        insertAdjacentHTML(position, html) { this.inserted = (this.inserted || []).concat([[position, html]]); },
        querySelector() { return null; },
        querySelectorAll() { return this.children; },
        closest(sel) {
            let node = this;
            while (node) {
                if (node.selectors && node.selectors.includes(sel)) return node;
                node = node._parents;
            }
            return null;
        },
        focus() { this.focused = true; },
        get innerHTML() { return this._html || ''; },
        set innerHTML(v) { this._html = v; },
    };
    return e;
}

const IDS = ['queueEditorModal', 'qeSettingsModal', 'qeChips', 'qeChipPending', 'qeChipRunning',
    'qeChipError', 'qeHint', 'qeStatus', 'qeRoot', 'qeCrumbs', 'qeSearch', 'qeName', 'qeSettingFilter',
    'qeRecursive', 'qeHits', 'qeSelected', 'qeBtnSelectAll', 'qeBtnClearSelection',
    'qeScopeAll', 'qeScopeSelected', 'qeScopeAllLabel', 'qeScopeSelectedLabel', 'qeFolder',
    'qeList', 'qeFoot', 'qePageInfo', 'qeBtnPrev', 'qeBtnNext', 'qePageSize',
    'qeConfirmation', 'qeConfirmText', 'qeConfirmSample', 'qeCountInput', 'qeBtnCancel',
    'qeBtnConfirm', 'qeBtnSettings', 'qeBtnLowVram', 'qeBtnTop', 'qeBtnBottom',
    'qeBtnRemove', 'qeBtnRetry', 'qsCount', 'qsCountNoun', 'qsSummary', 'qeTxOptions', 'qsBtnApply',
    'qeSummarize', 'qeModel', 'qeLang', 'qeDiarize', 'qeLowVram', 'qeVocalIso', 'qeVocabPreset',
    'qeSpeakerVectors', 'qeWordTimestamps', 'qeExportAlongside', 'qeSaveAudioAlongside'];

function buildIt() {
    const elements = {};
    const fetch = (id) => (elements[id] ||= element(id));
    IDS.forEach(fetch);
    // Option fields: every anchor element sits inside its own .field
    // shell (model in .txo-model), the way renderTxOptions builds them.
    for (const id of ['qeLang', 'qeVocabPreset', 'qeDiarize', 'qeSpeakerVectors', 'qeWordTimestamps',
                      'qeExportAlongside', 'qeVocalIso', 'qeSaveAudioAlongside', 'qeLowVram',
                      'qeSummarize']) {
        const field_el = element('field:' + id); field_el.selectors = ['.field', '.qe-group'];
        elements['field:' + id] = field_el;
        field_el.appendChild(fetch(id));
        elements['qeTxOptions'].appendChild(field_el);
    }
    const modelField = element('field:qeModel'); modelField.selectors = ['.txo-model', '.qe-group'];
    elements['field:qeModel'] = modelField;
    modelField.appendChild(fetch('qeModel'));
    elements['qeTxOptions'].appendChild(modelField);

    const requests = [];
    const answers = [];          // [{pattern, answer, status}]
    const timer = [];
    const toasts = {info: 0, success: 0, warn: 0, error: 0, texts: []};
    const counter = {refreshQueue: 0, renderQueueList: 0, renderFileList: 0, clearSelection: 0, openQueueFile: 0};

    const fakeDocument = {
        getElementById: (id) => elements[id] || null,
        createElement: () => element('new_value'),
        addEventListener(kind, fn) { (fakeDocument.listener[kind] ||= []).push(fn); },
        listener: {},
        querySelector: () => null, querySelectorAll: () => [],
        activeElement: {tagName: 'DIV'},
        body: {appendChild() {}, removeChild() {}},
    };

    const sandbox = {
        document: fakeDocument,
        window: {addEventListener() {}},
        console,
        setTimeout: (fn, ms) => { timer.push({fn, ms}); return timer.length; },
        clearTimeout: (id) => { if (timer[id - 1]) timer[id - 1].fn = null; },
        URLSearchParams,
        fetch: async (url, opts = {}) => {
            const body = opts.body ? JSON.parse(opts.body) : null;
            requests.push({url, method: opts.method || 'GET', body});
            for (const a of answers) {
                if (url.includes(a.pattern)) {
                    const answer = typeof a.answer === 'function' ? a.answer(url, body) : a.answer;
                    const status = a.status || 200;
                    return {ok: status < 400, status, json: async () => answer};
                }
            }
            return {ok: true, status: 200, json: async () => ({ok: true})};
        },
        Toast: {
            info: (t) => { toasts.info++; toasts.texts.push(t); },
            success: (t) => { toasts.success++; toasts.texts.push(t); },
            warn: (t) => { toasts.warn++; toasts.texts.push(t); },
            error: (t) => { toasts.error++; toasts.texts.push(t); },
        },
        esc: (s) => String(s ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;'),
        escJs: (s) => String(s ?? '').replace(/\\/g, '\\\\').replace(/'/g, "\\'"),
        // core.js's counted noun: "1 window", "3 windows"
        plural: (n, one, many) => {
            const k = Number(n) || 0;
            return `${k.toLocaleString()} ${k === 1 ? one : (many || one + 's')}`;
        },
        refreshQueue: async () => { counter.refreshQueue++; },
        renderQueueList: () => { counter.renderQueueList++; },
        renderFileList: () => { counter.renderFileList++; },
        openQueueFile: () => { counter.openQueueFile++; },
        clearSelection: () => { counter.clearSelection++; },
        _queueRows: (x) => x,
        queueBadges: () => [],
        txCollect: () => sandbox.collectValues,
        collectValues: {},
        queuePending: 0, queueCurrentFile: null, queueRetryable: 0, queuePaused: false,
        selectedFiles: new Set(),
        Math, JSON, Number, String, Array, Object, Set, Map, RegExp, Date, parseInt, parseFloat,
        isNaN, encodeURIComponent, decodeURIComponent, Promise,
    };
    // api() like core.js's, in miniature
    sandbox.api = async (url, options = {}) => {
        const opts = {...options};
        if (opts.body && typeof opts.body === 'object') opts.body = JSON.stringify(opts.body);
        const resp = await sandbox.fetch(url, opts);
        if (!resp.ok) {
            let msg = `HTTP ${resp.status}`;
            try { const d = await resp.json(); msg = (d && d.error) || msg; } catch (e) {}
            const err = new Error(msg);
            err.status = resp.status;
            throw err;
        }
        return resp.json();
    };
    sandbox.globalThis = sandbox;
    vm.createContext(sandbox);
    vm.runInContext(fs.readFileSync(process.env.QUEUE_EDITOR_JS, 'utf8'),
                    sandbox, {filename: 'queue_editor.js'});

    // Default answers
    answers.push({pattern: '/api/queue/editor/folders',
                    answer: {ok: true, folders: [], files_here: 0, roots: [], total: 0, generation: 1}});
    answers.push({pattern: '/api/queue/editor/items',
                    answer: {ok: true, total: 0, offset: 0, limit: 200, items: [], roots: [],
                              processing: null, generation: 1, filter: {}}});

    return {
        sandbox, elements, requests, toasts, counter, timer,
        setAnswer: (pattern, answer, status) => answers.unshift({pattern, answer, status}),
        setStatus: (pattern, status, answer) => answers.unshift({pattern, answer, status}),
        // First collect the pending fetch promises, then resolve them in
        // timer order (debounce), then flush the promises once more.
        idle: async () => { for (let i = 0; i < 20; i++) await new Promise(r => setImmediate(r)); },
        runTimers: async () => {
            for (let i = 0; i < 20; i++) await new Promise(r => setImmediate(r));
            const open_count = timer.splice(0);
            for (const t of open_count) if (t.fn) await t.fn();
            for (let i = 0; i < 20; i++) await new Promise(r => setImmediate(r));
        },
        domLoaded: () => (fakeDocument.listener.DOMContentLoaded || []).forEach(fn => fn()),
        press: (ev) => (fakeDocument.listener.keydown || []).forEach(fn => fn(ev)),
        run: (code) => vm.runInContext(code, sandbox),
        value: (expression) => vm.runInContext(`(${expression})`, sandbox),
        wait: (p) => p,
    };
}

module.exports = {buildIt};
