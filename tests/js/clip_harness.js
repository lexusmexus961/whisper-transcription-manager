/* The clip editor without a browser.
 *
 * Only a fraction of the DOM is needed: a few elements with `style`,
 * `value` and `textContent`, plus a timeline with a rectangle. This
 * mock-up doesn't touch anything else in `clip.js` (trimming, preview,
 * downloading) — what's tested is the mouse-button behavior on the
 * timeline, and that's pure arithmetic over pixels.
 */
const fs = require('fs');
const vm = require('vm');

function element(id) {
    return {
        id, value: '', textContent: '', innerHTML: '',
        style: {}, className: '',
        classList: {
            _s: new Set(),
            add(c) { this._s.add(c); }, remove(c) { this._s.delete(c); },
            contains(c) { return this._s.has(c); },
        },
        listener: {},
        addEventListener(kind, fn) { (this.listener[kind] ||= []).push(fn); },
        appendChild() {},
        getBoundingClientRect() { return this.rect || {left: 0, width: 1000, top: 0, height: 40}; },
        focus() {}, blur() {},
    };
}

function buildIt() {
    const elements = {};
    const fetch = (id) => (elements[id] ||= element(id));

    // Timeline: 1000 px wide, starts at x=0 — that way the pixel count
    // is also the per-mille of the total duration, which makes every
    // expectation in the test hand-calculable.
    fetch('clipTimeline').rect = {left: 0, width: 1000, top: 0, height: 40};
    fetch('clipTimelineInner').rect = {left: 0, width: 1000, top: 0, height: 40};
    // Everything the editor touches when setting the markers. If one is
    // missing, `updateClipUI` aborts mid-chain and the test measures
    // silence instead of behavior.
    ['clipPlayhead', 'clipRegion', 'clipIn', 'clipOut', 'clipDuration',
     'clipZoom', 'clipZoomVal', 'clipZoomHint', 'clipEditor',
     'clipSegBars', 'clipTitle'].forEach(fetch);
    fetch('videoPlayer').currentTime = 0;
    fetch('videoPlayer').play = () => {};
    fetch('videoPlayer').pause = () => {};

    const fakeDocument = {
        getElementById: (id) => elements[id] || null,
        createElement: () => element('new_value'),
        listener: {},
        addEventListener(kind, fn) { (this.listener[kind] ||= []).push(fn); },
        removeEventListener() {},
    };

    const sandbox = {
        document: fakeDocument,
        window: {addEventListener() {}, location: {href: ''}},
        console,
        setInterval: () => 0, clearInterval: () => {},
        setTimeout: () => 0, clearTimeout: () => {},
        fetch: async () => ({ok: true, json: async () => ({})}),
        activeTranscript: null,
        activeFilename: 'AbCdEfGhIjK.mp4',
        showToast: () => {},
        Math, JSON, Number, String, Array, Object, Date, parseFloat, parseInt,
        isNaN, encodeURIComponent,
    };
    sandbox.globalThis = sandbox;
    vm.createContext(sandbox);
    vm.runInContext(fs.readFileSync(process.env.CLIP_JS, 'utf8'),
                    sandbox, {filename: 'clip.js'});

    // What the browser triggers after loading — only after this do the
    // mouse listeners hang on the timeline.
    for (const fn of (fakeDocument.listener['DOMContentLoaded'] || [])) fn({});

    const mouse = (button, x) => {
        let prevented = false;
        const e = {
            button, clientX: x, clientY: 10,
            preventDefault() { prevented = true; },
            ctrlKey: false, metaKey: false,
        };
        for (const fn of (elements['clipTimeline'].listener['mousedown'] || [])) fn(e);
        return prevented;
    };

    /* `let clipTotalDur` at the top of clip.js is a lexical binding —
     * it does NOT hang off the global object. Writing
     * `sandbox.clipTotalDur = 1000` from outside just creates a
     * same-named, ineffective property, and the test happily checks
     * nothing at all. Reading and writing must happen inside the context. */
    const run = (code) => vm.runInContext(code, sandbox);
    const value = (expression) => vm.runInContext(`(${expression})`, sandbox);

    return {sandbox, elements, mouse, fetch, run, value};
}

module.exports = {buildIt};
