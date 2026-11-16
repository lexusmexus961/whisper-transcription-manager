// ── Conversations (api/conversations.py, whisper_tm/conversations) ───────
//
// Whole conversations — or the stretch of a long recording where given
// people talk together — found by who took part and how they talked:
//
//   people      all of them together, or any of them (the voice library's
//               names and the names typed in transcripts)
//   length      of the conversation, not of the file: an hour of a
//               six-hour stream where Alice and Bob talk is one result
//   kind        detailed, back-and-forth, an interview, a monologue — and
//               "group" on top of any of them: three people or more
//   sentences   longer or shorter than usual — with people chosen, usual for
//               them (their own conversations); else for the library's
//               conversations in the same language
//
// The server counts and decides (conversations/query.py). The tab shows
// its numbers and never recounts them, so a row says what Ask and the MCP
// tools say about the same conversation.
//
// Opened by its tab, and pre-filtered — always through cvShow(filters),
// with the API's own parameter names — by Ask's find_conversations
// (chat.js), by Find when a description asks for conversations (find.js),
// by Voices → People (voices_people.js) and by the Conversation card of a
// transcript's 📊 (stats.js). ✂ hands the filters on to the voice extractor
// (voices_extract.js): one person's sentences from these conversations, as
// a dataset.
//
// Every transcript write counts its own numbers; what was transcribed
// before is counted by a run in the background (conversations/runner.py).
// While it goes, the info line says that the lists are not complete yet.

const CV_PAGE = 50;
const CV_PILLS = 4;             // people named in a row, the others as "+n"

/* The filters, by the parameter names of GET /api/conversations — what Ask
 * and Find hand over (cvShow) has the same shape. 0 and '' mean "any". Two
 * speakers at least is what makes a conversation; asking for a monologue
 * lowers it (cvMonoSpeakers). */
const CV_NO_FILTERS = () => ({
    people: [], match: 'all',
    min_sec: 0, max_sec: 0,
    kind: '', sentences: '',
    wps_min: 0, wps_max: 0, rate_min: 0, rate_max: 0,
    speakers_min: 2, speakers_max: 0,
    questions_min: 0,
    language: '', q: '', folder: '', date_from: '', date_to: '',
    sort: 'longest', copies: 'fold',
});

// The server's kinds (conversations/query.py), named as the tab names them
const CV_KIND_NAMES = {
    detailed: 'Detailed', lively: 'Back-and-forth', interview: 'Interview', monologue: 'Monologue',
    conversation: 'Conversation', unknown: 'Speakers unknown', silent: 'No speech', group: 'Group',
};
const CV_KIND_TIPS = {
    detailed: 'Long sentences and long turns — people explain things at length',
    lively: 'Many speaker changes and short sentences — a quick back-and-forth',
    interview: 'Two voices, and one of them asks most of the questions',
    monologue: 'One voice speaks most of the time',
    conversation: 'Two voices or more, none of the kinds above',
    unknown: 'Not diarized — who spoke is unknown',
    silent: 'Nothing was said',
    group: 'Three people or more take part — on top of any kind',
};
const CV_KIND_CHIPS = ['', 'detailed', 'lively', 'interview', 'monologue'];

// [min_sec, max_sec, label, title] — "about an hour" is a window around
// it: nobody's hour of talk lasts exactly sixty minutes
const CV_LENGTHS = [
    [0, 0, 'Any', 'Any length'],
    [0, 300, 'Under 5 min', 'Shorter than five minutes'],
    [300, 1800, '5–30 min', 'Five to thirty minutes'],
    [1800, 5400, '30–90 min', 'Half an hour to an hour and a half'],
    [2700, 4500, 'About 1 h', '45 to 75 minutes'],
    [7200, 0, 'Over 2 h', 'Longer than two hours'],
];
// [value, label, title without people, title with people]
const CV_SENTENCES = [
    ['', 'Any', 'Any sentence length', 'Any sentence length'],
    ['longer', 'Longer than usual',
     'More words per sentence than most conversations of the library in the same language',
     'More words per sentence than usual for them — measured against their own conversations'],
    ['shorter', 'Shorter than usual',
     'Fewer words per sentence than most conversations of the library in the same language',
     'Fewer words per sentence than usual for them — measured against their own conversations'],
];
const CV_SORTS = [
    ['longest', 'Longest'], ['shortest', 'Shortest'], ['sentences', 'Longest sentences'],
    ['rate', 'Most speaker changes'], ['questions', 'Most questions'], ['monologue', 'Longest monologue'],
    ['newest', 'Newest'], ['oldest', 'Oldest'], ['together', 'Most together'],
];

const _cv = {
    f: CV_NO_FILTERS(),
    panel: false,               // the ⚙ panel is open
    list: null,                 // {key, total, results, d}: the answer shown (its pages)
    error: '',
    off: false,                 // conversations.enabled is false (503)
    seq: 0,                     // an answer to filters changed since is dropped
    status: null,               // GET /api/conversations/status
    sseq: 0,
    finished: false,            // the background count ended while the tab looked on
    person: null,               // {name, d, error}: the card of the one person in the filter
    pseq: 0,
    people: null,               // [{name, recordings, conversations}] — to check a typed name
    peopleAt: 0,
    unknown: [],                // the names the server does not know (people_unknown)
    emptyWhy: '',               // why the empty list shown is empty (cvEmptyWhy)
    hint: '', hintFor: '',      // a typed name nobody has, and that name folded
    spkAuto: false,             // the speakers minimum was lowered for a monologue
    timer: null,                // follows the background count while it goes
    later: null,                // a typed filter waits for the next key
    keys: false, tips: false,
};

// ── Small things ────────────────────────────────────────────────────────────

function cvClock(sec) {
    sec = Math.max(0, Math.floor(Number(sec) || 0));
    const h = Math.floor(sec / 3600), m = Math.floor((sec % 3600) / 60), s = sec % 60;
    return h ? `${h}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}` : `${m}:${String(s).padStart(2, '0')}`;
}

// A total of hours: "45 min", "3.5 h", "1,240 h"
function cvHours(sec) {
    const h = (Number(sec) || 0) / 3600;
    return h >= 100 ? `${Math.round(h).toLocaleString()} h` : h >= 1 ? `${h.toFixed(1)} h` : `${Math.round(h * 60)} min`;
}

function cvPct(share) {
    return `${Math.round(Math.max(0, Number(share) || 0) * 100)} %`;
}

// A share the server could not count (no sentences): a dash, not "0 %"
function cvPctOr(share) {
    return share === null || share === undefined || share === '' ? '—' : cvPct(share);
}

/* Switched off (conversations.enabled) — the server answers 503 then. A busy
 * database answers 503 too, and is no reason to say "off": the status
 * (enabled: false) has the last word either way. */
function cvIsOff(e) {
    return !!e && e.status === 503 && /switched off/i.test(String(e.message || ''));
}

// A server figure with one decimal — '' when there is none (null: not
// counted, e.g. no sentence punctuation)
function cvFig(x, digits = 1) {
    if (x === null || x === undefined || x === '') return '';
    const n = Number(x);
    return Number.isFinite(n) ? n.toFixed(digits) : '';
}

/* "longer than 78 %" — of the conversations the baseline counts. Under half
 * it reads the other way round: "shorter than 70 %" says the same as
 * "longer than 30 %", and is what one wants to know. */
function cvThan(pct, up, down) {
    if (pct === null || pct === undefined || pct === '') return '';
    const p = Math.round(Number(pct));
    if (!Number.isFinite(p)) return '';
    return p >= 50 ? `${up} than ${p} %` : `${down} than ${100 - p} %`;
}

/* With "longer/shorter sentences" measured against the people chosen
 * (cvSentBaseHtml), a row says where it lies for them — by the server's own
 * thresholds — and the library's comparison moves into the tip: "shorter
 * than 77 %" under every row of "longer than usual" read as a contradiction. */
function cvTheirs() {
    const sb = _cv.list && _cv.list.d ? _cv.list.d.sentences_baseline : null;
    return sb && typeof sb === 'object' && sb.source === 'people' ? sb : null;
}

function cvTheirsWord(wps, sb) {
    const w = Number(wps);
    if (!Number.isFinite(w)) return '';
    // (short: the column is narrow — the info line says it in full)
    if (Number.isFinite(Number(sb.longer_from)) && w >= Number(sb.longer_from)) return 'longer for them';
    if (Number.isFinite(Number(sb.shorter_to)) && w <= Number(sb.shorter_to)) return 'shorter for them';
    return 'usual for them';
}

function cvBase(path) {
    const s = String(path || '');
    const cut = Math.max(s.lastIndexOf('/'), s.lastIndexOf('\\'));
    return cut >= 0 ? s.slice(cut + 1) : s;
}

// The folder a recording lies in — not its file_dir, the root it opens under
function cvFolder(path) {
    const s = String(path || '');
    const cut = Math.max(s.lastIndexOf('/'), s.lastIndexOf('\\'));
    return cut > 0 ? s.slice(0, cut) : '';
}

function cvShortFolder(path, names = 2) {
    const parts = String(path || '').split(/[\\/]/).filter(Boolean);
    if (!parts.length) return String(path || '');
    return (parts.length > names ? '…/' : '') + parts.slice(-names).join('/');
}

// Names compared as the server folds them: case and accents aside
function cvFold(name) {
    return String(name ?? '').toLowerCase().normalize('NFKD').replace(/[̀-ͯ]/g, '').trim();
}

/* A person's colour — the map's, where the map knows the name (voices_people.js).
 * Only a colour value goes into a style attribute, never a name. */
function cvColour(name) {
    let c = '';
    try { c = typeof peopleColour === 'function' ? String(peopleColour(String(name ?? '')) || '') : ''; }
    catch (e) { c = ''; }
    return /^#[0-9a-f]{3,8}$/i.test(c) || /^(?:rgb|hsl)a?\([\d\s.,%]+\)$/i.test(c) ? c : 'var(--accent)';
}

function cvKinds(text) {
    return [...new Set(String(text || '').toLowerCase().split(',').map(k => k.trim())
        .filter(k => Object.prototype.hasOwnProperty.call(CV_KIND_NAMES, k)))];
}

function cvKindName(kind) {
    return CV_KIND_NAMES[kind] || String(kind || '');
}

function cvNames(v) {
    const list = Array.isArray(v) ? v : String(v ?? '').split(',');
    const out = [], seen = new Set();
    for (const x of list) {
        const name = String(x ?? '').trim();
        if (name && !seen.has(cvFold(name))) { seen.add(cvFold(name)); out.push(name); }
    }
    return out;
}

function cvPositive(v) {
    const n = typeof v === 'number' ? v : parseFloat(String(v ?? '').replace(',', '.'));
    return Number.isFinite(n) && n > 0 ? n : 0;
}

// ── Opening the tab ─────────────────────────────────────────────────────────

/* The filters as the API takes them, from what is handed over: unknown keys
 * left out, every value of the type its filter has. */
function cvMerge(o) {
    const f = CV_NO_FILTERS();
    const src = o && typeof o === 'object' ? o : {};
    for (const [k, v] of Object.entries(src)) {
        if (!Object.prototype.hasOwnProperty.call(f, k) || v === null || v === undefined) continue;
        if (k === 'people') f.people = cvNames(v);
        else if (k === 'kind') f.kind = cvKinds(Array.isArray(v) ? v.join(',') : v).join(',');
        else if (typeof f[k] === 'number') f[k] = cvPositive(v);
        else f[k] = String(v).trim();
    }
    // Language codes are lower case wherever they are stored ("DE" from a
    // model would stand beside "de" in the panel's list)
    f.language = f.language.toLowerCase();
    f.speakers_min = Math.round(f.speakers_min);
    f.speakers_max = Math.round(f.speakers_max);
    f.questions_min = Math.min(1, f.questions_min);
    if (f.match !== 'any') f.match = 'all';
    if (f.sentences !== 'longer' && f.sentences !== 'shorter') f.sentences = '';
    if (!CV_SORTS.some(([s]) => s === f.sort)) f.sort = 'longest';
    if (f.copies !== 'all') f.copies = 'fold';
    return f;
}

/* The tab, filtered: {people: ['Alice', 'Bob'], kind: 'detailed',
 * min_sec: 2700, …} — the API's parameters, as Ask's find_conversations
 * and Find hand them over. What is not named starts as new. */
function cvShow(filters) {
    cvSetFilters(filters);
    const keep = document.getElementById('svQuery')?.value || '';
    if (typeof svState !== 'undefined') svState.routed = keep;
    const visible = document.getElementById('svOverlay')?.classList.contains('visible');
    if (visible && typeof svTab === 'function' && svState.tab !== 'conversations') { svTab('conversations'); return; }
    svShowTab('conversations');
    if (visible) { cvOpen(); return; }
    openSearchView(keep);              // its svRun() opens this tab: cvOpen()
}

/* The same filters in the Voices window's Conversations tab — from a
 * person's row: who they talk with. */
function cvShowInVoices(filters) {
    cvSetFilters(filters);
    if (typeof vxOpen === 'function') vxOpen('conversations');
}

/* New filters: what is not named starts as new. */
function cvSetFilters(filters) {
    const given = filters && typeof filters === 'object' ? filters : {};
    _cv.seq++;                         // a load still on its way is of the old filters
    _cv.pseq++;                        // … and so is a person's card
    clearTimeout(_cv.later);           // … and a typed filter still waiting
    _cv.later = null;
    _cv.f = cvMerge(given);
    _cv.spkAuto = false;
    if (!('speakers_min' in given)) cvMonoSpeakers();
    _cv.list = null;
    _cv.error = '';
    _cv.person = null;
    _cv.unknown = [];
    _cv.hint = '';
    _cv.finished = false;
}

// ── Handing over to the voice extractor ─────────────────────────────────────

/* The filters by the API's names, without the order and the copies — what
 * the voice extractor narrows its sentences to (voices_extract.js). It keeps
 * copies apart itself, as it does everywhere. */
function cvApiFilters(src) {
    const f = src ? cvMerge(src) : _cv.f, out = {};
    if (f.people.length) { out.people = [...f.people]; out.match = f.match; }
    for (const k of ['min_sec', 'max_sec', 'kind', 'sentences', 'wps_min', 'wps_max', 'rate_min', 'rate_max',
                     'speakers_min', 'speakers_max', 'questions_min', 'language', 'q', 'folder',
                     'date_from', 'date_to']) {
        const v = f[k];
        if (v !== '' && v !== 0 && v !== null && v !== undefined && !Number.isNaN(v)) out[k] = v;
    }
    return out;
}

/* Conversation filters in a few words — the extractor's summary of them. */
function cvDescribe(src) {
    const f = cvMerge(src), out = [];
    const n = f.people.length;
    if (n) {
        const names = n > 1 ? `${f.people.slice(0, -1).join(', ')} and ${f.people[n - 1]}` : f.people[0];
        out.push(n > 1 ? `${names}${f.match === 'any' ? ' (any of them)' : ' together'}` : names);
    }
    const len = cvLengthIndex(f);
    if (len > 0) out.push(CV_LENGTHS[len][3]);
    const kinds = cvKinds(f.kind);
    if (kinds.length) out.push(kinds.map(cvKindName).join(' + '));
    if (f.sentences) {
        out.push(`${f.sentences === 'longer' ? 'Longer' : 'Shorter'} sentences than usual${n ? ' for them' : ''}`);
    }
    for (const [, label] of cvPanelChips(f)) out.push(label);
    return out;
}

/* ✂ — one person's voice from the conversations shown: the extractor opens
 * with these filters, for the first of the people chosen if any. */
function cvExtract() {
    if (typeof extractFromConversations !== 'function') return;
    extractFromConversations(cvApiFilters(), _cv.f.people[0] || '');
}

async function cvOpen() {
    cvWireKeys();
    cvWireTips();
    _cv.off = false;
    cvRenderFilters();
    cvRenderInfo();
    // Back to the tab with the same filters: the list as it was left. The
    // names to check a typed one against are asked for when the people
    // field gets the focus (cvLoadPeople): on a large library that list is
    // the heaviest thing the tab can ask for, and most visits type no name.
    const fresh = !(_cv.list && _cv.list.key === cvQuery());
    if (!fresh) { cvRenderList(); cvRenderPerson(); }
    await Promise.all([fresh ? cvLoad(false) : null, cvLoadPerson(), cvLoadStatus()]);
}

// Leaving the tab or the search: nothing follows the count any more
function cvLeave() {
    cvStopWatch();
    clearTimeout(_cv.later);
    _cv.later = null;
    if (typeof stvHideTip === 'function') stvHideTip();
}

function cvVisible() {
    const pane = document.getElementById('svPaneConversations');
    if (!pane || !pane.classList.contains('active')) return false;
    // Shown in the Voices window (voices.js moves it there) or in Find
    if (typeof vxConversationsHere === 'function' && vxConversationsHere()) {
        return !!document.getElementById('mapOverlay')?.classList.contains('visible');
    }
    return !!document.getElementById('svOverlay')?.classList.contains('visible');
}

// While the background run counts, the info line follows it — every two
// seconds, and only while the tab is in view
function cvWatch() {
    if (_cv.timer) return;
    _cv.timer = setInterval(() => {
        if (!cvVisible()) { cvStopWatch(); return; }
        cvLoadStatus();
    }, 2000);
}

function cvStopWatch() {
    if (_cv.timer) { clearInterval(_cv.timer); _cv.timer = null; }
}

// ── Asking the server ───────────────────────────────────────────────────────

/* The filters as URL parameters — the people one by one (people=Alice&
 * people=Bob), what is empty or 0 ("any") left out. `extra`: the page. */
function cvQuery(extra) {
    const f = _cv.f, p = new URLSearchParams();
    for (const name of f.people) p.append('people', name);
    const put = (k, v) => {
        if (v === '' || v === null || v === undefined || v === 0 || Number.isNaN(v)) return;
        p.set(k, String(v));
    };
    if (f.people.length) put('match', f.match);
    for (const k of ['min_sec', 'max_sec', 'kind', 'sentences', 'wps_min', 'wps_max', 'rate_min', 'rate_max',
                     'speakers_min', 'speakers_max', 'questions_min', 'language', 'q', 'folder',
                     'date_from', 'date_to']) put(k, f[k]);
    // "Most together" orders by the people's shares: without people, longest
    put('sort', f.sort === 'together' && !f.people.length ? 'longest' : f.sort);
    put('copies', f.copies);
    for (const [k, v] of Object.entries(extra || {})) {
        if (v !== '' && v !== null && v !== undefined) p.set(k, String(v));
    }
    return p.toString();
}

function cvRowKey(r) {
    return `${(r && r.abs_path) || ''}|${Math.round(Number(r && r.start_sec) || 0)}`;
}

async function cvLoad(more) {
    if (!document.getElementById('cvList')) return;
    const seq = ++_cv.seq;
    const key = cvQuery();
    const have = more && _cv.list && _cv.list.key === key ? _cv.list : null;
    if (!have) {
        _cv.list = null;
        _cv.error = '';
        cvRenderList();
        cvRenderInfo();                // the count shown was of the filters before
    } else cvMoreBusy();
    // The next page starts where the last one ended on the server, not at
    // the number of rows shown: rows already shown are left out of a page,
    // and asking from that number again would bring the same page back
    const at = have ? have.next : 0;
    let d;
    try {
        d = await api('/api/conversations?' + cvQuery({ offset: at, limit: CV_PAGE }));
    } catch (e) {
        if (seq !== _cv.seq) return;
        if (cvIsOff(e)) {
            _cv.off = true;
            _cv.list = null;
        } else if (have) {
            if (typeof Toast !== 'undefined') Toast.error(e.message);
        } else _cv.error = e.message || 'Not reachable';
        cvRenderList();
        cvRenderPerson();
        cvRenderInfo();
        return;
    }
    if (seq !== _cv.seq) return;
    d = d && typeof d === 'object' ? d : {};
    let got = Array.isArray(d.results) ? d.results.filter(r => r && typeof r === 'object') : [];
    if (have) {
        // While the background count goes on the order moves between pages:
        // a conversation already shown is not shown twice
        const seen = new Set(have.results.map(cvRowKey));
        got = have.results.concat(got.filter(r => !seen.has(cvRowKey(r))));
    }
    _cv.off = false;
    _cv.error = '';
    const size = Math.round(Number(d.limit)) > 0 ? Math.round(Number(d.limit)) : CV_PAGE;
    _cv.list = { key, total: Math.max(0, Math.round(Number(d.total) || 0)), results: got, d, next: at + size };
    _cv.unknown = Array.isArray(d.people_unknown) ? d.people_unknown.map(String) : [];
    cvRenderList();
    cvRenderInfo();
    cvMarkUnknown();
}

async function cvLoadStatus() {
    const seq = ++_cv.sseq;
    let st;
    try { st = await api('/api/conversations/status'); }
    catch (e) {
        // One answer missed while following the count: what was known stays
        // (the progress line, whether it ran) — the next one brings it on
        if (seq === _cv.sseq) { _cv.status = { ...(_cv.status || {}), error: e.message }; cvRenderInfo(); }
        return;
    }
    if (seq !== _cv.sseq) return;
    const was = !!(_cv.status && _cv.status.run && _cv.status.run.running);
    _cv.status = st && typeof st === 'object' ? st : {};
    const running = !!(_cv.status.run && _cv.status.run.running);
    // Counted to the end while the tab looked on: the list shown is of what
    // was counted then — the complete one is a click away, not forced on
    // someone reading it
    if (was && !running) _cv.finished = true;
    if (_cv.status.enabled === false && !_cv.off) {
        _cv.off = true;
        cvRenderList();
        cvRenderPerson();
    } else if (_cv.list && !_cv.list.results.length && !_cv.off && !_cv.error
               && cvEmptyWhy() !== _cv.emptyWhy) {
        // An empty list says why from the counts ("nothing is counted
        // yet") — and they may come after the list did. Drawn anew only
        // when the reason changed: the status comes every two seconds
        // while the count goes, and a click on "Count now" must not
        // land on a button just replaced
        cvRenderList();
    }
    cvRenderInfo();
    if (running) cvWatch(); else cvStopWatch();
}

// The names that take part somewhere, to check a typed one — at most a
// minute old: names are given on the map all the time
async function cvLoadPeople() {
    if (Array.isArray(_cv.people) && Date.now() - _cv.peopleAt < 60000) return;
    try {
        const d = await api('/api/conversations/people');
        _cv.people = (Array.isArray(d && d.people) ? d.people : []).filter(p => p && p.name);
        _cv.peopleAt = Date.now();
    } catch (e) { /* unchecked then: the server says which names it doesn't know */ }
}

// The card of the one person in the filter (with two or more: no card)
async function cvLoadPerson() {
    const f = _cv.f;
    if (f.people.length !== 1) {
        if (_cv.person) { _cv.person = null; _cv.pseq++; cvRenderPerson(); }
        return;
    }
    const name = f.people[0];
    if (_cv.person && _cv.person.name === name && (_cv.person.d || _cv.person.error)) { cvRenderPerson(); return; }
    const seq = ++_cv.pseq;
    _cv.person = { name, d: null };
    cvRenderPerson();
    let d;
    try { d = await api('/api/conversations/person?person=' + encodeURIComponent(name)); }
    catch (e) {
        if (seq !== _cv.pseq) return;
        _cv.person = cvIsOff(e) ? null : { name, d: null, error: e.message };
        cvRenderPerson();
        return;
    }
    if (seq !== _cv.pseq) return;
    _cv.person = { name, d: d && typeof d === 'object' ? d : {} };
    cvRenderPerson();
}

// ── The info line ───────────────────────────────────────────────────────────

function cvRenderInfo() {
    const el = document.getElementById('cvInfo');
    if (!el) return;
    const st = _cv.status || {}, run = st.run || {}, c = st.counts || {};
    const L = _cv.list, d = L ? L.d : {};
    const off = _cv.off || st.enabled === false;
    const bits = [];
    if (off) bits.push('<span class="ms-err">Conversation statistics are switched off.</span>');
    else if (L) {
        bits.push(`<b>${L.total.toLocaleString()}</b> conversation${L.total === 1 ? '' : 's'}${cvTotalHours(L)}`);
        if (d.baseline && typeof d.baseline === 'object') bits.push(cvBaselineHtml(d.baseline));
        const sb = cvSentBaseHtml(d.sentences_baseline);
        if (sb) bits.push(sb);
    } else if (_cv.error) bits.push(`<span class="ms-err">Not reachable: ${esc(_cv.error)}</span>`);
    else if (st.error) bits.push(`<span class="ms-err">Not reachable: ${esc(st.error)}</span>`);
    else bits.push('Looking for conversations …');
    if (!off) {
        if (run.running && run.stopping) {
            bits.push('<span class="ms-building">⟳ Stopping the count after its current batch …</span>');
        } else if (run.running) {
            const total = Number(run.total) || 0;
            const pct = total > 0 ? Math.min(100, Math.floor((Number(run.done) || 0) / total * 100)) : null;
            const at = pct === null ? ' …' : `: ${pct} %`;
            // Counting again keeps the old numbers readable until each
            // transcript's turn comes: the lists are complete, some of
            // their numbers old — a first count's lists miss what is not
            // counted yet
            bits.push(`<span class="ms-building">⟳ ${run.recount
                ? `Counting everything again${at} — some numbers are still the old ones`
                : `Counting statistics${at} — results are not complete yet`}</span>
                <button class="ms-link" onclick="cvRunStop()" title="Stop counting — it goes on where it stopped the next time">Stop</button>`);
        } else if (Number(c.stale) > 0) {
            bits.push(`${plural(Math.round(Number(c.stale)), 'transcript')} not counted yet
                <button class="ms-link" onclick="cvRunStart(false)" title="Count them in the background — minutes of one CPU core for a large library">Count now</button>`);
        } else if (_cv.finished) {
            bits.push('Counting finished <button class="ms-link" onclick="cvRefresh()">↻ Show the complete list</button>');
        }
        if (L && _cv.unknown.length) {
            bits.push(`<span class="ms-err">${esc(_cv.unknown.join(', '))}: no name of the voice library, nor typed in a transcript</span>`);
        }
        const ignored = (L && Array.isArray(d.ignored) ? d.ignored : []).map(cvIgnoredText).filter(Boolean);
        if (ignored.length) bits.push(`<span class="ms-err">Not understood and left out: ${esc(ignored.join(', '))}</span>`);
    }
    const tail = [];
    if (!off && !run.running && st.counts) {
        tail.push('<button class="ms-link" onclick="cvRunStart(true)" title="Count every transcript again — after an update of the rules, or when numbers look wrong">Count everything again</button>');
    }
    tail.push(`<button class="ms-link" onclick="openSettings('search#conversations')">Settings →</button>`);
    // While the count goes it is drawn every two seconds — anew only when
    // it reads differently, or a click on Stop lands on a button just
    // replaced and is lost
    const html = `${bits.join(' · ')} ${tail.join(' ')}`;
    if (el._cvHtml === html) return;
    el._cvHtml = html;
    el.innerHTML = html;
}

// Their length together — from the server when it says, else added up once
// every page is loaded: a page's sum is no total
function cvTotalHours(L) {
    const d = L.d || {};
    let sec = typeof d.total_sec === 'number' && Number.isFinite(d.total_sec) ? d.total_sec : null;
    if (sec === null) {
        if (L.next < L.total) return '';
        sec = L.results.reduce((a, r) => a + (Number(r.duration_sec) || 0), 0);
    }
    return sec > 0 ? ` · ${cvHours(sec)}` : '';
}

// What "longer than usual" is measured against
function cvBaselineHtml(b) {
    const tip = 'Sentence length and speaker changes are compared with the library’s real conversations '
        + '(two voices or more, punctuated, two minutes at least) in the same language';
    if (b.source === 'defaults') {
        return `<span class="cv-base" title="${esc(tip)}">Compared with general figures — too few conversations are counted yet</span>`;
    }
    const n = Math.round(Number(b.conversations) || 0);
    const lang = b.language ? String(b.language).toUpperCase() : '';
    const where = b.source === 'library' ? ' of the library' : lang ? ` in ${esc(lang)}` : '';
    return `<span class="cv-base" title="${esc(tip)}">Compared with ${n.toLocaleString()} conversation${n === 1 ? '' : 's'}${where}</span>`;
}

/* What "longer/shorter than usual" was measured against
 * (query.py _sentences_baseline): the people's own conversations, or the
 * library's when no people were chosen or they have too few. */
function cvSentBaseHtml(sb) {
    if (!sb || typeof sb !== 'object' || (sb.sentences !== 'longer' && sb.sentences !== 'shorter')) return '';
    const longer = sb.sentences === 'longer';
    const at = cvFig(longer ? sb.longer_from : sb.shorter_to);
    if (!at) return '';
    const words = `${longer ? 'from' : 'up to'} ${at} words per sentence`;
    const n = Math.round(Number(sb.conversations) || 0);
    if (sb.source === 'people') {
        const who = Array.isArray(sb.people) ? sb.people.map(String).join(', ') : '';
        const tip = `Measured against ${who ? `the ${n.toLocaleString()} conversations of ${who}` : 'their own conversations'}`
            + ' — what is usual for them. The kinds stay measured against the library.';
        return `<span class="cv-base" title="${esc(tip)}">${longer ? 'Longer' : 'Shorter'} than usual for them: ${esc(words)}</span>`;
    }
    const few = Number.isFinite(Number(sb.people_conversations)) && sb.people_conversations !== null
        ? ` — they have only ${Math.round(Number(sb.people_conversations)).toLocaleString()} conversation${
            Math.round(Number(sb.people_conversations)) === 1 ? '' : 's'}, too few to measure against their own` : '';
    return `<span class="cv-base" title="Measured against the library’s conversations in the same language">${
        longer ? 'Longer' : 'Shorter'} than usual in the library: ${esc(words + few)}</span>`;
}

function cvIgnoredText(x) {
    if (x === null || x === undefined) return '';
    if (typeof x !== 'object') return String(x);
    return String(x.param || x.name || x.key || '');
}

// ── The filters ─────────────────────────────────────────────────────────────

function cvRenderFilters() {
    const el = document.getElementById('cvFilters');
    if (!el) return;
    el.innerHTML = `<div class="cv-filters">${cvPeopleHtml()}${cvLengthsHtml()}${cvKindsHtml()}${cvSentencesHtml()}
        <span class="cv-tools">${cvToolsHtml()}</span></div>${_cv.panel ? cvPanelHtml() : cvChipsHtml()}`;
}

function cvPeopleHtml() {
    const f = _cv.f;
    const unknown = new Set(_cv.unknown.map(cvFold));
    const chips = f.people.map((name, i) => {
        const bad = unknown.has(cvFold(name));
        return `<span class="cv-person${bad ? ' unknown' : ''}" data-name="${esc(name)}"
            title="${esc(bad ? `${name}: no name of the voice library, nor typed in a transcript` : name)}"><i style="background:${cvColour(name)}"></i>${esc(name)}<button
            class="vx-icon" onclick="cvPersonDrop(${i})" title="Without ${esc(name)}" aria-label="Without ${esc(name)}">✕</button></span>`;
    }).join('');
    // How several people take part — shown while it can matter
    const seg = f.people.length >= 2 || f.match === 'any' ? `<span class="vx-seg cv-match" role="group" aria-label="How they take part">
        <button class="vx-seg-btn${f.match === 'all' ? ' active' : ''}" onclick="cvSetMatch('all')"
                title="Every one of them talks — in a long recording, the stretch where they talk together">All together</button>
        <button class="vx-seg-btn${f.match === 'any' ? ' active' : ''}" onclick="cvSetMatch('any')"
                title="Whole recordings where at least one of them takes part">Any of them</button></span>` : '';
    return `<span class="cv-people">${chips}<input type="text" class="cv-person-in" id="cvPersonIn" data-names
            autocomplete="off" spellcheck="false" aria-label="Add a person" onfocus="cvLoadPeople()" onkeydown="cvPersonKey(event)"
            placeholder="${f.people.length ? 'Add a person …' : 'Who took part? A name, Enter'}">${seg}${
        _cv.hint ? `<span class="cv-hint">${esc(_cv.hint)}</span>` : ''}</span>`;
}

function cvLengthIndex(f = _cv.f) {
    return CV_LENGTHS.findIndex(([a, b]) => a === f.min_sec && b === f.max_sec);
}

function cvLengthsHtml() {
    const on = cvLengthIndex();
    return `<span class="cv-group cv-lens"><span class="cv-flabel">Length</span>${CV_LENGTHS.map(([, , label, title], i) =>
        `<button class="ms-chip${on === i ? ' on' : ''}" data-len="${i}" onclick="cvSetLength(${i})" title="${esc(title)}">${label}</button>`).join('')}</span>`;
}

function cvKindsHtml() {
    const kinds = cvKinds(_cv.f.kind);
    const main = kinds.filter(k => k !== 'group');
    const chip = (k, label, on, title) =>
        `<button class="ms-chip${on ? ' on' : ''}" onclick="cvSetKind('${k}')" title="${esc(title)}">${esc(label)}</button>`;
    const chips = CV_KIND_CHIPS.map(k => chip(k, k ? CV_KIND_NAMES[k] : 'All', k ? main.includes(k) : !main.length,
                                              k ? CV_KIND_TIPS[k] : 'Every kind'));
    // A kind without a chip of its own (handed over by Ask): shown, and off with a click
    for (const k of main) if (!CV_KIND_CHIPS.includes(k)) chips.push(chip(k, cvKindName(k), true, CV_KIND_TIPS[k] || ''));
    return `<span class="cv-group"><span class="cv-flabel">Kind</span>${chips.join('')}<span class="ms-sep"></span>${
        chip('group', '+ Group', kinds.includes('group'), CV_KIND_TIPS.group)}</span>`;
}

function cvSentencesHtml() {
    const people = _cv.f.people.length > 0;
    return `<span class="cv-group"><span class="cv-flabel">Sentences</span><span class="vx-seg" role="group" aria-label="Sentence length">${
        CV_SENTENCES.map(([v, label, title, theirs]) => `<button class="vx-seg-btn${_cv.f.sentences === v ? ' active' : ''}"
            onclick="cvSetSentences('${v}')" title="${esc(people ? theirs : title)}">${label}</button>`).join('')}</span></span>`;
}

function cvToolsHtml() {
    const f = _cv.f, n = cvPanelChips().length;
    const sort = f.sort === 'together' && !f.people.length ? 'longest' : f.sort;
    const sorts = CV_SORTS.filter(([k]) => k !== 'together' || f.people.length)
        .map(([k, label]) => `<option value="${k}"${sort === k ? ' selected' : ''}>${label}</option>`).join('');
    return `<button class="vx-btn ad-panel-btn${_cv.panel ? ' on' : ''}" onclick="cvPanel()" aria-expanded="${_cv.panel}"
            title="Narrow it down: minutes, words per sentence, speaker changes, speakers, questions, language, path, folder, date, copies">⚙ Filters${n ? ` <b>${n}</b>` : ''}</button>
        <select class="ad-sort" onchange="cvSetSort(this.value)" title="Order" aria-label="Order">${sorts}</select>
        <button class="vx-btn ad-again" onclick="cvRefresh()" title="Look again — the background count adds more while it goes">↻</button>
        <button class="vx-btn" onclick="cvExtract()" title="One person's voice from these conversations, sentence by sentence — as a dataset for TTS training or one audio file (Voices → Extract)">✂ Extract</button>`;
}

// ── The ⚙ panel ─────────────────────────────────────────────────────────────

function cvPanelHtml() {
    const f = _cv.f;
    const val = (x) => (x ? esc(String(x)) : '');
    const mins = (s) => (s ? String(Math.round(s / 6) / 10) : '');
    const num = (key, value, attrs, label) => `<input type="number" class="cv-num-in" value="${value}" ${attrs}
        aria-label="${esc(label)}" oninput="cvField('${key}', this.value)" onkeydown="cvFieldKey(event, '${key}')">`;
    const text = (key, value, placeholder, label) => `<input type="search" class="ad-q ad-q-small" value="${val(value)}"
        placeholder="${esc(placeholder)}" aria-label="${esc(label)}" autocomplete="off" spellcheck="false"
        oninput="cvField('${key}', this.value)" onkeydown="cvFieldKey(event, '${key}')">`;
    const row = (label, body, hint) => `<div class="ad-prow"><span class="ad-plabel"${hint ? ` title="${esc(hint)}"` : ''}>${label}</span>
        <span class="ad-pbody">${body}</span></div>`;
    const to = '<span class="ad-to">to</span>';
    const langs = cvLanguages().map(([code, name]) =>
        `<option value="${esc(code)}"${f.language === code ? ' selected' : ''}>${esc(name)}</option>`).join('');
    return `<div class="ad-panel cv-panel">
        ${row('Length', `${num('min_min', mins(f.min_sec), 'min="0" step="1"', 'At least, in minutes')}${to}${
            num('max_min', mins(f.max_sec), 'min="0" step="1"', 'At most, in minutes')}<span class="ad-to">min</span>`,
            'How long the conversation lasts — the stretch of a long recording, or the whole one')}
        ${row('Words per sentence', `${num('wps_min', val(f.wps_min), 'min="0" step="0.5"', 'Words per sentence, at least')}${to}${
            num('wps_max', val(f.wps_max), 'min="0" step="0.5"', 'Words per sentence, at most')}`,
            'Sentences are cut at . ! ? — a transcript without punctuation has no sentence length')}
        ${row('Speaker changes', `${num('rate_min', val(f.rate_min), 'min="0" step="0.5"', 'Speaker changes per minute, at least')}${to}${
            num('rate_max', val(f.rate_max), 'min="0" step="0.5"', 'Speaker changes per minute, at most')}<span class="ad-to">per min</span>`,
            'Per minute of speech — a quick back-and-forth has many')}
        ${row('Speakers', `${num('speakers_min', val(f.speakers_min), 'min="0" step="1"', 'Speakers, at least')}${to}${
            num('speakers_max', val(f.speakers_max), 'min="0" step="1"', 'Speakers, at most')}`,
            'Voices that take part — each with a share of the talk; empty = any number')}
        ${row('Questions', `<span class="ad-to">at least</span>${num('questions_pct', f.questions_min ? String(Math.round(f.questions_min * 100)) : '',
            'min="0" max="100" step="5"', 'Questions, at least, in percent')}<span class="ad-to">% of the sentences</span>`)}
        ${row('Language', `<select class="ad-sel" onchange="cvField('language', this.value, true)" aria-label="Language">
            <option value="">Any</option>${langs}</select>`, 'The language of the transcript')}
        ${row('In the path', text('q', f.q, 'letters of a file or folder name — e.g. interview, 2024-05', 'Text in the path'))}
        ${row('Folder', text('folder', f.folder, 'a folder — e.g. /media/recordings/talks', 'Folder'),
            'Only recordings in this folder and the folders in it')}
        ${row('Recorded', `<input type="date" class="ad-date" value="${val(f.date_from)}" onchange="cvField('date_from', this.value, true)"
                aria-label="Recorded from">${to}<input type="date" class="ad-date" value="${val(f.date_to)}"
                onchange="cvField('date_to', this.value, true)" aria-label="Recorded until">`,
            'The date in the file name, else the file’s own date')}
        ${row('Copies', `<label class="vx-check cv-check"><input type="checkbox"${f.copies === 'fold' ? ' checked' : ''}
                onchange="cvSetCopies(this.checked)"> One row for a recording kept in several files</label>`,
            'The same recording in several containers or folders is shown once, with “+n copies”')}
        <div class="ad-prow ad-pfoot"><button class="ms-link" onclick="cvResetPanel()">Reset the filters</button>
            <button class="ms-link" onclick="cvPanel(false)">Close</button></div>
    </div>`;
}

// The app's transcription languages, the ones of the conversations shown,
// and the one asked for — a transcript can be in any language the model
// detects
function cvLanguages() {
    const out = new Map();
    if (typeof _S_LANGUAGES !== 'undefined' && Array.isArray(_S_LANGUAGES)) {
        for (const [code, name] of _S_LANGUAGES) if (code && code !== 'auto') out.set(String(code), String(name));
    }
    for (const r of (_cv.list && _cv.list.results) || []) {
        const code = String((r && r.language) || '');
        if (code && !out.has(code)) out.set(code, code.toUpperCase());
    }
    if (_cv.f.language && !out.has(_cv.f.language)) out.set(_cv.f.language, _cv.f.language.toUpperCase());
    return [...out];
}

// The panel's filters that are set — each one a chip while it is closed
function cvPanelChips(f = _cv.f) {
    const out = [];
    const range = (a, b, unit) => (a && b ? `${a}–${b} ${unit}` : a ? `≥ ${a} ${unit}` : `≤ ${b} ${unit}`);
    const mins = (s) => Math.round(s / 6) / 10;
    if ((f.min_sec || f.max_sec) && cvLengthIndex(f) < 0) out.push(['length', range(mins(f.min_sec), mins(f.max_sec), 'min')]);
    if (f.wps_min || f.wps_max) out.push(['wps', range(f.wps_min, f.wps_max, 'words per sentence')]);
    if (f.rate_min || f.rate_max) out.push(['rate', range(f.rate_min, f.rate_max, 'speaker changes per min')]);
    if (f.speakers_min !== CV_NO_FILTERS().speakers_min || f.speakers_max) {
        // One speaker at least is no "any": it leaves out what was not diarized
        out.push(['speakers', f.speakers_min > 0
            ? range(f.speakers_min, f.speakers_max, f.speakers_min === 1 && !f.speakers_max ? 'speaker' : 'speakers')
            : f.speakers_max ? `≤ ${f.speakers_max} speakers` : 'Any number of speakers']);
    }
    if (f.questions_min) out.push(['questions_min', `≥ ${Math.round(f.questions_min * 100)} % questions`]);
    if (f.language) out.push(['language', `Language ${f.language.toUpperCase()}`]);
    if (f.q) out.push(['q', `“${f.q}” in the path`]);
    if (f.folder) out.push(['folder', `In ${cvShortFolder(f.folder)}`]);
    if (f.date_from || f.date_to) out.push(['dates', `${f.date_from || '…'} – ${f.date_to || '…'}`]);
    if (f.copies === 'all') out.push(['copies', 'Every copy']);
    return out;
}

function cvChipsHtml() {
    const chips = cvPanelChips();
    if (!chips.length) return '';
    return `<div class="ad-chips cv-chips">${chips.map(([key, label]) => `<span class="ad-chip">${esc(label)}<button
        class="vx-icon" onclick="cvClear('${key}')" title="Without this filter" aria-label="Without this filter">✕</button></span>`).join('')}
        <button class="ms-link" onclick="cvResetPanel()">Reset</button></div>`;
}

// The bar's state while a field is typed in — not drawn anew, or the field
// would lose the focus under the cursor
function cvSyncBar() {
    const n = cvPanelChips().length;
    const btn = document.querySelector('#cvFilters .ad-panel-btn');
    if (btn) btn.innerHTML = `⚙ Filters${n ? ` <b>${n}</b>` : ''}`;
    const on = cvLengthIndex();
    document.querySelectorAll('#cvFilters .cv-lens .ms-chip').forEach(b => b.classList.toggle('on', Number(b.dataset.len) === on));
}

function cvMarkUnknown() {
    const unknown = new Set(_cv.unknown.map(cvFold));
    document.querySelectorAll('#cvFilters .cv-person').forEach(el =>
        el.classList.toggle('unknown', unknown.has(cvFold(el.dataset.name))));
}

// ── Changing them ───────────────────────────────────────────────────────────

// A filter changed by a click: the bar drawn anew, the list asked for
function cvChanged() {
    cvRenderFilters();
    cvReload();
}

function cvReload() {
    clearTimeout(_cv.later);
    _cv.later = null;
    _cv.finished = false;
    cvLoad(false);
    cvLoadPerson();
}

function cvPanel(open) {
    _cv.panel = open === undefined ? !_cv.panel : !!open;
    cvRenderFilters();
}

function cvSetLength(i) {
    const len = CV_LENGTHS[i];
    if (!len) return;
    _cv.f.min_sec = len[0];
    _cv.f.max_sec = len[1];
    cvChanged();
}

/* One kind at a time (a click on the one chosen takes it off), and "+ Group"
 * on top of any of them: three people or more in a detailed conversation. */
function cvSetKind(k) {
    const kinds = cvKinds(_cv.f.kind);
    let main = kinds.filter(x => x !== 'group');
    let group = kinds.includes('group');
    if (k === 'group') group = !group;
    else if (!k) main = [];
    else if (main.includes(k)) main = main.filter(x => x !== k);
    else if (Object.prototype.hasOwnProperty.call(CV_KIND_NAMES, k)) main = [k];
    _cv.f.kind = [...main, ...(group ? ['group'] : [])].join(',');
    cvMonoSpeakers();
    cvChanged();
}

// A monologue has one voice: the tab's "two speakers at least" would leave
// only the ones where somebody else says a little. Lowered while a
// monologue is asked for — and back when it no longer is.
function cvMonoSpeakers() {
    const mono = cvKinds(_cv.f.kind).includes('monologue');
    if (mono && _cv.f.speakers_min === CV_NO_FILTERS().speakers_min) {
        _cv.f.speakers_min = 0;
        _cv.spkAuto = true;
    } else if (!mono && _cv.spkAuto) {
        if (!_cv.f.speakers_min) _cv.f.speakers_min = CV_NO_FILTERS().speakers_min;
        _cv.spkAuto = false;
    }
}

function cvSetSentences(v) {
    _cv.f.sentences = v === 'longer' || v === 'shorter' ? v : '';
    cvChanged();
}

function cvSetSort(v) {
    if (!CV_SORTS.some(([k]) => k === v)) return;
    _cv.f.sort = v;
    cvReload();
}

function cvSetMatch(v) {
    _cv.f.match = v === 'any' ? 'any' : 'all';
    cvChanged();
}

function cvSetCopies(fold) {
    _cv.f.copies = fold ? 'fold' : 'all';
    cvSyncBar();
    cvReload();
}

function cvPersonKey(ev) {
    if (ev.key !== 'Enter') return;
    ev.preventDefault();
    cvAddPerson(ev.target.value);
}

// The display form of a typed name; null when nobody has it, undefined while
// the list is not there (the server then says it: people_unknown)
function cvKnownName(typed) {
    if (!Array.isArray(_cv.people)) return undefined;
    const k = cvFold(typed);
    const hit = _cv.people.find(p => cvFold(p.name) === k);
    return hit ? String(hit.name) : null;
}

function cvAddPerson(text) {
    const typed = String(text ?? '').trim();
    if (!typed) return;
    const known = cvKnownName(typed);
    if (known === null && _cv.hintFor !== cvFold(typed)) {
        // A typo most of the time. Enter again looks anyway: the count may
        // not have reached their recordings yet
        _cv.hint = `Nobody named “${typed}” takes part in a counted recording — Enter again to look anyway`;
        _cv.hintFor = cvFold(typed);
        cvRenderFilters();
        cvFocusPeople(typed);
        return;
    }
    cvAddName(known || typed);
    cvFocusPeople('');
}

function cvAddName(name) {
    _cv.hint = '';
    _cv.hintFor = '';
    if (!_cv.f.people.some(p => cvFold(p) === cvFold(name))) _cv.f.people.push(String(name));
    cvChanged();
}

function cvFocusPeople(value) {
    const el = document.getElementById('cvPersonIn');
    if (!el) return;
    el.value = value;
    if (typeof el.focus === 'function') el.focus();
}

function cvPersonDrop(i) {
    const f = _cv.f;
    if (i < 0 || i >= f.people.length) return;
    f.people.splice(i, 1);
    // "Any of them" with one person left would be a choice nobody sees
    if (f.people.length < 2) f.match = 'all';
    if (!f.people.length && f.sort === 'together') f.sort = 'longest';
    _cv.hint = '';
    _cv.hintFor = '';
    cvChanged();
}

// "Talks most with": the partner joins the filter
function cvPartner(i) {
    const p = ((_cv.person && _cv.person.d && _cv.person.d.partners) || [])[i];
    if (p && p.name) cvAddName(p.name);
}

/* A typed value: kept at once, asked for half a second after the last key
 * (Enter, a date or a list: at once) — a request per keystroke would only
 * race the next one. */
function cvField(key, raw, now) {
    const f = _cv.f;
    const n = cvPositive(raw);
    switch (key) {
    case 'min_min': f.min_sec = Math.round(n * 60); break;
    case 'max_min': f.max_sec = Math.round(n * 60); break;
    case 'questions_pct': f.questions_min = Math.min(1, n / 100); break;
    case 'speakers_min': case 'speakers_max': f[key] = Math.round(n); _cv.spkAuto = false; break;
    case 'wps_min': case 'wps_max': case 'rate_min': case 'rate_max': f[key] = n; break;
    case 'q': case 'folder': f[key] = String(raw ?? '').trim(); break;
    case 'language': case 'date_from': case 'date_to': f[key] = String(raw ?? '').trim(); now = true; break;
    default: return;
    }
    cvSyncBar();
    clearTimeout(_cv.later);
    _cv.later = null;
    if (now) cvReload();
    else _cv.later = setTimeout(cvReload, 500);
}

function cvFieldKey(ev, key) {
    if (ev.key !== 'Enter') return;
    ev.preventDefault();
    cvField(key, ev.target.value, true);
}

function cvClear(key) {
    const f = _cv.f, D = CV_NO_FILTERS();
    if (key === 'length') { f.min_sec = 0; f.max_sec = 0; }
    else if (key === 'wps') { f.wps_min = 0; f.wps_max = 0; }
    else if (key === 'rate') { f.rate_min = 0; f.rate_max = 0; }
    else if (key === 'speakers') { f.speakers_min = D.speakers_min; f.speakers_max = 0; _cv.spkAuto = false; }
    else if (key === 'dates') { f.date_from = ''; f.date_to = ''; }
    else if (key !== 'people' && Object.prototype.hasOwnProperty.call(f, key)) f[key] = D[key];
    cvChanged();
}

// The panel's filters back to none — the bar's own (people, length chip,
// kind, sentences, order) stay
function cvResetPanel() {
    const f = _cv.f;
    const preset = cvLengthIndex() >= 0;
    _cv.f = { ...CV_NO_FILTERS(), people: f.people, match: f.match, kind: f.kind, sentences: f.sentences,
              sort: f.sort, min_sec: preset ? f.min_sec : 0, max_sec: preset ? f.max_sec : 0 };
    _cv.spkAuto = false;
    cvMonoSpeakers();
    cvChanged();
}

// Every filter but the people
function cvResetAll() {
    _cv.f = { ...CV_NO_FILTERS(), people: _cv.f.people, match: _cv.f.match };
    _cv.spkAuto = false;
    cvChanged();
}

// ↻ — the background count adds to the person's card as much as to the
// list: both are asked for again
function cvRefresh() {
    _cv.person = null;
    cvReload();
    cvLoadStatus();
}

function cvMore() {
    cvLoad(true);
}

function cvMoreBusy() {
    const btn = document.querySelector('#cvList .cv-more');
    if (btn) { btn.disabled = true; btn.textContent = 'Loading …'; }
}

// ── The list ────────────────────────────────────────────────────────────────

function cvRenderList() {
    const box = document.getElementById('cvList');
    if (!box) return;
    if (_cv.off) { box.innerHTML = cvOffHtml(); return; }
    if (_cv.error) { box.innerHTML = `<div class="sv-empty ms-err">${esc(_cv.error)}</div>`; return; }
    const L = _cv.list;
    if (!L) { box.innerHTML = '<div class="sv-empty">Looking for conversations …</div>'; return; }
    if (!L.results.length) { _cv.emptyWhy = cvEmptyWhy(); box.innerHTML = cvEmptyHtml(); return; }
    const shown = L.results.length;
    const more = L.next < L.total
        ? `<button class="vx-btn am-more cv-more" onclick="cvMore()">Show more (${shown.toLocaleString()} of ${L.total.toLocaleString()})</button>` : '';
    box.innerHTML = `<div class="cv-rows">${cvHeadHtml()}${L.results.map(cvRowHtml).join('')}</div>${more}`;
}

function cvOffHtml() {
    return `<div class="ms-empty"><div class="ms-empty-title">🗣 Conversation statistics are switched off</div>
        <p>Switched on, every transcript is counted — words per sentence, speaker changes, who talks how much — and
           conversations can be found by who took part and how they talked.</p>
        <button class="vx-btn primary" onclick="openSettings('search#conversations')">Open the settings</button></div>`;
}

// Why a list is empty: no transcripts at all, nothing counted yet (and no
// count going — the info line follows one), or the filters
function cvEmptyWhy() {
    const st = _cv.status || {}, c = st.counts || {};
    if (c.transcripts === 0) return 'library';
    if (c.counted === 0 && !(st.run && st.run.running)) return 'uncounted';
    return 'filters';
}

function cvEmptyHtml() {
    const f = _cv.f, why = cvEmptyWhy();
    if (why === 'library') {
        return `<div class="ms-empty"><div class="ms-empty-title">🗣 No transcripts yet</div>
            <p>Every transcript is counted when it is written — its conversations show up here then.</p></div>`;
    }
    if (why === 'uncounted') {
        return `<div class="ms-empty"><div class="ms-empty-title">🗣 Nothing is counted yet</div>
            <p>The statistics of a transcript are counted when it is written; the ones transcribed before need one run
               over the library — minutes of one CPU core.</p>
            <button class="vx-btn primary" onclick="cvRunStart(false)">Count now</button></div>`;
    }
    const tips = [];
    if (f.people.length > 1 && f.match === 'all') tips.push('“Any of them” instead of “All together”');
    if (f.people.length > 1) tips.push('fewer people');
    if (f.min_sec || f.max_sec) tips.push('another length');
    if (f.kind) tips.push('another kind');
    if (f.sentences) tips.push('any sentence length');
    if (cvPanelChips().length) tips.push('fewer filters under ⚙');
    const say = tips.length
        ? `More turn up with ${tips.slice(0, -1).join(', ')}${tips.length > 1 ? ' or ' : ''}${tips[tips.length - 1]}.` : '';
    const set = f.min_sec || f.max_sec || f.kind || f.sentences || cvPanelChips().length;
    return `<div class="sv-empty">No conversation fits these filters.${say ? `<br>${esc(say)}` : ''}${
        set ? ' <button class="ms-link" onclick="cvResetAll()">Reset the filters</button>' : ''}</div>`;
}

function cvHeadHtml() {
    const h = (label, tip, cls = '') => `<span class="cv-c${cls}" title="${esc(tip)}">${label}</span>`;
    return `<div class="cv-row cv-head" aria-hidden="true">
        ${h('Kind', 'What kind of conversation — point at a badge for why')}
        ${h('Recording', 'The file, and the folder it lies in')}
        ${h('Stretch', 'Where in the recording these people talk together — or the whole recording')}
        ${h('Length', 'How long the conversation lasts', ' cv-num')}
        ${h('People', 'Who takes part, with their share of the talk')}
        ${h('Sentences', 'Words per sentence — compared with the library’s conversations in the same language', ' cv-num')}
        ${h('Speaker changes', 'Per minute of speech — compared with the library’s conversations in the same language', ' cv-num')}
        ${h('Questions', 'The share of the sentences that are questions', ' cv-num')}
        ${h('Longest monologue', 'The longest stretch one voice talks without a change — a click opens it there', ' cv-num')}
        ${h('Date', 'Recorded — the date in the file name, else the file’s own')}
        <span class="cv-c"></span>
    </div>`;
}

/* One conversation. Every value comes from the server and goes through esc;
 * the handlers carry the row's number only, never a name or a path. */
function cvRowHtml(r, i) {
    const kind = String(r.kind || '');
    const why = (Array.isArray(r.reasons) ? r.reasons : []).map(x => String(x ?? '')).filter(Boolean);
    const kindTip = [cvKindName(kind) || 'Kind', ...(why.length ? why : [CV_KIND_TIPS[kind] || ''])].filter(Boolean).join('\n');
    const path = String(r.abs_path || r.filename || '');
    const copies = Math.round(Number(r.copies) || 0);
    const rec = `<span class="cv-name" title="${esc(path)}">${esc(cvBase(r.filename || r.abs_path))}</span>
        <small class="cv-folder" title="${esc(path)}">${esc(cvFolder(r.abs_path || r.filename))}</small>${copies > 1
        ? `<small class="cv-copies" title="The same recording in ${copies} files — shown once">+${copies - 1} cop${copies === 2 ? 'y' : 'ies'}</small>` : ''}`;
    const when = r.whole
        ? `<span>${cvClock(r.recording_sec || r.duration_sec)}</span><small>the whole recording</small>`
        : `<span>${cvClock(r.start_sec)}–${cvClock(r.end_sec)}</span><small>of ${cvClock(r.recording_sec)}</small>`;
    const wps = cvFig(r.wps);
    const than = cvThan(r.wps_pct, 'longer', 'shorter');
    const theirs = cvTheirs();
    const sentences = !wps
        ? '<span class="cv-dim" data-tip="No sentence length\nThis transcript has no sentence punctuation — its sentences can’t be told apart">—</span>'
        : theirs
            ? `<span><b>${wps}</b> words</span><small title="${esc(than ? `Against the library’s conversations: ${than}` : '')}">${
                esc(cvTheirsWord(r.wps, theirs))}</small>`
            : `<span><b>${wps}</b> words</span><small>${esc(than)}</small>`;
    // Questions are sentences that end in "?": a transcript without
    // sentence punctuation (no words per sentence either) has a share of
    // 0 % that only says nothing could be told — a dash, as for its sentences
    const questions = wps
        ? `<span>${cvPctOr(r.questions_share)}</span>`
        : '<span class="cv-dim" data-tip="No share of questions\nThis transcript has no sentence punctuation — its questions can’t be told apart">—</span>';
    const rate = cvFig(r.turn_rate);
    const changes = rate
        ? `<span><b>${rate}</b> /min</span><small>${esc(cvThan(r.rate_pct, 'more', 'fewer'))}</small>`
        : '<span class="cv-dim" data-tip="No speaker changes counted\nLess than a minute of speech with a known speaker">—</span>';
    const mono = Number(r.longest_turn_sec) > 0 && r.longest_turn_start !== null && r.longest_turn_start !== undefined
        ? `<button class="cv-seek" onclick="event.stopPropagation(); cvMonologue(${i})" title="Open it there">${esc(fmtDuration(r.longest_turn_sec))}</button>
           <small>${r.longest_turn_name ? `${esc(r.longest_turn_name)} · ` : ''}${cvClock(r.longest_turn_start)}</small>`
        : '<span class="cv-dim">—</span>';
    const dateTip = r.date_source === 'name' ? 'Recorded\nThe date in the file name' : 'The file’s date\nThe name holds no date';
    return `<div class="cv-row" role="button" tabindex="0" data-keyclick onclick="cvOpenRow(${i})">
        <span class="cv-c cv-kinds"><span class="cv-kind" data-tip="${esc(kindTip)}">${esc(cvKindName(kind) || '—')}</span>${
            r.group ? `<span class="cv-tag" data-tip="${esc(`Group\n${CV_KIND_TIPS.group}`)}">Group</span>` : ''}</span>
        <span class="cv-c cv-rec">${rec}</span>
        <span class="cv-c cv-when"><small class="cv-lab">Stretch</small>${when}</span>
        <span class="cv-c cv-num cv-len"><small class="cv-lab">Length</small><span>${esc(fmtDuration(r.duration_sec))}</span></span>
        <span class="cv-c cv-ppl">${cvPeopleCell(r)}</span>
        <span class="cv-c cv-num cv-wps"><small class="cv-lab">Sentences</small>${sentences}</span>
        <span class="cv-c cv-num cv-rate"><small class="cv-lab">Speaker changes</small>${changes}</span>
        <span class="cv-c cv-num cv-q"><small class="cv-lab">Questions</small>${questions}</span>
        <span class="cv-c cv-num cv-mono"><small class="cv-lab">Longest monologue</small>${mono}</span>
        <span class="cv-c cv-date"><small class="cv-lab">Date</small>${r.date
            ? `<span data-tip="${esc(dateTip)}">${esc(r.date)}</span>` : '<span class="cv-dim">—</span>'}</span>
        <span class="cv-c cv-btns"><button class="vx-btn" onclick="event.stopPropagation(); cvPlayRow(${i})"
                title="Play it from there — the search stays open">▶</button><button class="vx-btn"
                onclick="event.stopPropagation(); cvOpenRow(${i})" title="Open the transcript there">open ›</button></span>
    </div>`;
}

// Who takes part: a pill each (their colour, their share), the unnamed
// voices counted, and their shares as one thin bar
function cvPeopleCell(r) {
    const people = (Array.isArray(r.people) ? r.people : []).filter(p => p && p.name);
    const pills = people.slice(0, CV_PILLS).map(p => {
        const wps = cvFig(p.wps);
        const tip = `${p.name}\n${fmtDuration(p.talk_sec)} of the talk · ${cvPct(p.share)}${wps ? `\n${wps} words per sentence` : ''}`;
        return `<span class="cv-pill" data-tip="${esc(tip)}"><i style="background:${cvColour(p.name)}"></i>${esc(p.name)} <b>${cvPct(p.share)}</b></span>`;
    });
    if (people.length > CV_PILLS) {
        const rest = people.slice(CV_PILLS);
        pills.push(`<span class="cv-pill cv-more-pill" data-tip="${esc(rest.map(p => `${p.name} · ${cvPct(p.share)}`).join('\n'))}">+${rest.length}</span>`);
    }
    const unnamed = Math.round(Number(r.unnamed) || 0);
    if (unnamed > 0) {
        pills.push(`<span class="cv-pill cv-unnamed" data-tip="${esc(`Without a name\n${plural(unnamed, 'voice')} nobody has named yet`)}">+${unnamed} unnamed</span>`);
    }
    if (!pills.length) return '<span class="cv-dim">—</span>';
    const bar = people.length ? `<span class="cv-share">${people.map(p => `<i style="width:${
        (Math.max(0, Math.min(1, Number(p.share) || 0)) * 100).toFixed(1)}%;background:${cvColour(p.name)}"></i>`).join('')}</span>` : '';
    return `<span class="cv-pills">${pills.join('')}</span>${bar}`;
}

function cvRow(i) {
    return ((_cv.list && _cv.list.results) || [])[i];
}

function cvOpenRow(i) {
    const r = cvRow(i);
    if (r) svGoto(r.filename, Math.max(0, Number(r.start_sec) || 0), r.file_dir);
}

function cvPlayRow(i) {
    const r = cvRow(i);
    if (r) svPlay(r.filename, Math.max(0, Number(r.start_sec) || 0), r.file_dir);
}

function cvMonologue(i) {
    const r = cvRow(i);
    if (r) svGoto(r.filename, Math.max(0, Number(r.longest_turn_start) || 0), r.file_dir);
}

// ── The person card ─────────────────────────────────────────────────────────

function cvRenderPerson() {
    const box = document.getElementById('cvPerson');
    if (box) box.innerHTML = cvPersonHtml();
}

function cvPersonHtml() {
    const P = _cv.person;
    if (!P || _cv.off || _cv.f.people.length !== 1) return '';
    const dot = `<i class="cv-dot" style="background:${cvColour(P.name)}"></i>`;
    if (P.error) return `<div class="cv-pc"><div class="cv-pc-head">${dot}<b>${esc(P.name)}</b><span class="ms-err">${esc(P.error)}</span></div></div>`;
    if (!P.d) return `<div class="cv-pc"><div class="cv-pc-head">${dot}<b>${esc(P.name)}</b><span class="cv-dim">Loading …</span></div></div>`;
    const d = P.d;
    const name = String(d.name || P.name);
    if (!d.known) {
        return `<div class="cv-pc"><div class="cv-pc-head">${dot}<b>${esc(name)}</b>
            <span class="cv-dim">is no name of the voice library, nor typed in a transcript</span></div></div>`;
    }
    const n = (x) => Math.round(Number(x) || 0).toLocaleString();
    const tile = (big, label, tip) => `<div class="cv-tile"${tip ? ` title="${esc(tip)}"` : ''}><b>${esc(big)}</b><span>${esc(label)}</span></div>`;
    const recs = Math.round(Number(d.recordings) || 0), convs = Math.round(Number(d.conversations) || 0);
    const tiles = [
        tile(fmtDuration(d.talk_sec), 'of talk', 'Everything they say in the counted recordings'),
        tile(n(recs), recs === 1 ? 'recording' : 'recordings', 'The recordings they take part in'),
        tile(n(convs), convs === 1 ? 'conversation' : 'conversations', 'The recordings they take part in with at least one other voice'),
        tile(cvFig(d.wps) || '—', 'words per sentence', 'Their own sentences'),
        tile(cvPctOr(d.questions_share), 'of their sentences are questions'),
    ].join('');
    // Numbered by their place in the answer: cvPartner(i) looks them up there
    const partners = (Array.isArray(d.partners) ? d.partners : []).slice(0, 12).map((p, i) => [p, i])
        .filter(([p]) => p && p.name).map(([p, i]) =>
        `<button class="cv-partner" onclick="cvPartner(${i})" title="${esc(`Add ${p.name} to the filter — ${n(p.recordings)} recordings together, ${fmtDuration(p.together_sec)} of speech`)}"><i
            style="background:${cvColour(p.name)}"></i>${esc(p.name)} <small>${n(p.recordings)}</small></button>`).join('');
    const kinds = Object.entries(d.kinds && typeof d.kinds === 'object' ? d.kinds : {})
        .map(([k, c]) => [k, Math.round(Number(c) || 0)]).filter(([, c]) => c > 0).sort((a, b) => b[1] - a[1]);
    const top = Math.max(1, ...kinds.map(([, c]) => c));
    const bars = kinds.map(([k, c]) => `<div class="cv-kbar" title="${esc(CV_KIND_TIPS[k] || '')}"><span>${esc(cvKindName(k))}</span>
        <span class="cv-kbar-track"><i style="width:${Math.max(2, c / top * 100).toFixed(1)}%"></i></span><b>${n(c)}</b></div>`).join('');
    const first = d.first_date ? String(d.first_date) : '', last = d.last_date ? String(d.last_date) : '';
    const when = first ? (last && last !== first ? `${first} – ${last}` : first) : '';
    return `<div class="cv-pc">
        <div class="cv-pc-head">${dot}<b>${esc(name)}</b>${when ? `<span class="cv-dim">${esc(when)}</span>` : ''}</div>
        <div class="cv-pc-body">
            <div class="cv-pc-tiles">${tiles}</div>
            ${partners ? `<div class="cv-pc-part"><span class="cv-flabel">Talks most with</span>${partners}</div>` : ''}
            ${bars ? `<div class="cv-pc-kinds"><span class="cv-flabel">Their conversations</span>${bars}</div>` : ''}
        </div></div>`;
}

// ── The background count ────────────────────────────────────────────────────

async function cvRunStart(recount) {
    if (recount && !confirm('Count the conversation statistics of every transcript again?\n\n'
            + 'It runs in the background — minutes of one CPU core for a large library. Until it is done, '
            + 'some numbers are still the old ones.')) return;
    try { await api('/api/conversations/run/start', { method: 'POST', body: { recount: !!recount } }); }
    catch (e) { if (typeof Toast !== 'undefined') Toast.error(e.message); return; }
    _cv.finished = false;
    await cvLoadStatus();
}

async function cvRunStop() {
    try { await api('/api/conversations/run/stop', { method: 'POST', body: {} }); }
    catch (e) { if (typeof Toast !== 'undefined') Toast.error(e.message); }
    await cvLoadStatus();
}

// ── Keys and tooltips ───────────────────────────────────────────────────────

/* Esc in the tab closes the ⚙ panel first, then (the search view's own
 * handler) the workspace. Wired on the first opening — after voices.js has
 * wired its own, so the name suggestions under the people field close
 * before the panel does. */
function cvWireKeys() {
    if (_cv.keys || typeof document === 'undefined' || typeof document.addEventListener !== 'function') return;
    _cv.keys = true;
    document.addEventListener('keydown', cvEscKey, true);
}

function cvEscKey(ev) {
    if (ev.key !== 'Escape' || !_cv.panel || !cvVisible()) return;
    // A settings page above the search closes on its own Esc
    if (document.querySelector('.settings-overlay.visible')) return;
    if (typeof VX_SUGGEST !== 'undefined' && VX_SUGGEST.el && !VX_SUGGEST.el.hidden && VX_SUGGEST.items.length) return;
    ev.preventDefault();
    ev.stopImmediatePropagation();
    cvPanel(false);
}

// The badges' reasons and the people's shares as the statistics' tooltip
// (statsview.js) — it lies above the search workspace
function cvWireTips() {
    if (_cv.tips || typeof stvWireTips !== 'function') return;
    const box = document.getElementById('cvResults');
    if (!box) return;
    _cv.tips = true;
    stvWireTips(box);
}
