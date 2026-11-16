// ── Same audio: the duplicates (api/audio.py, whisper_tm/audio/duplicates.py) ──
//
// What the audio analysis found by itself, at a glance and drawn:
//
//   tiles      how many recordings exist more than once, the space their
//              extra copies take, how many contain others
//   folders    the folders that hold the same recordings: two coverage bars
//              facing each other, so a folder that is a copy of another (a
//              backup, a game's second install) stands out
//   groups     each copy as its waveform, on the timeline the group shares —
//              the same shapes one above the other are the proof
//   wholes     a recording with every recording it contains, each a block at
//              the second it begins there (a stream and the clips cut from it)
//
// Narrowed by many filters (the ⚙ panel): letters of a path, the person
// speaking, video or sound, the format, the length, the recording date, the
// language, transcribed or not, the number of copies, across folders or in
// one; for recordings holding others the holding one and the parts each by
// their folder — the clips of one folder where they are in the streams of
// another.
//
// Asked for in so many words ("find duplicate audio files", "duplicates in
// Archive") in the top bar or in Find (amDupIntent), by Ask's
// find_duplicates, by the file view's "≋ n copies", or by this tab's views.

// The filters (api/audio.py _filters, audio/duplicates.py Filters)
const AD_NO_FILTERS = () => ({
    q: '',                      // letters of a path (any recording of a card)
    qWhole: '', qPart: '',      // the holding recording's, a part's path
    person: '',                 // speaks in it (the voice map's names)
    media: '',                  // video | audio
    ext: [],                    // formats
    minSec: 0, maxSec: 0,
    since: '', until: '',       // the recording date, YYYY-MM-DD
    lang: '', transcribed: '',  // the transcript's language; yes | no
    minCopies: 0, spread: '',   // across | within
    minParts: 0,
});

const _ad = {
    kind: 'same',               // same | contains
    f: AD_NO_FILTERS(),
    panel: false,               // the ⚙ panel is open
    allExt: false,              // every format in the panel, the rare ones too
    sort: { same: 'space', contains: 'parts' },
    pair: null,                 // {id, label}: one folder pair of the overview
    file: '',                   // only the duplicates of one recording
    overview: null,
    list: null,                 // {kind, items: groups | wholes, total}
    seq: 0,
    levels: new Map(),          // recording id → Uint8Array (0…255 over -60…0 dBFS)
    wanted: new Set(),          // ids whose waveform is on its way
    allFolders: false,          // all folder pairs asked for (200), not the first 12
    moreFolders: false,         // the 12 shown, not the first 6
    open: new Set(),            // cards shown with all their copies or parts
    // The rows the buttons and canvases point at: "g3.1" (group 3, copy 1),
    // "w2" (whole 2), "w2.5" (its part 5) — a key stays the same when a card
    // opens all its copies, so the preview's buttons keep pointing right
    items: new Map(),
    ovKey: '',                  // the filters the overview was loaded for
    fresh: false,               // the next load works the duplicates out anew (↻)
    asked: '',                  // the top bar's words that opened the view (adAsWords)
    resize: null,
};

const AD_PAGE = 40;
const AD_FOLDERS_SHOWN = 6;
const AD_COPIES_SHOWN = 6;
const AD_PARTS_SHOWN = 8;
const AD_LENGTHS = [[0, 'Any length'], [10, '≥ 10 s'], [60, '≥ 1 min'], [600, '≥ 10 min']];
const AD_SORTS = {
    same: [['space', 'Most space'], ['copies', 'Most copies'], ['length', 'Longest'], ['new', 'Newest found']],
    contains: [['parts', 'Most parts'], ['length', 'Longest'], ['new', 'Newest found']],
};

// ── Asked for in words ─────────────────────────────────────────────────────

/* Whether a query asks for the duplicates rather than for something said or
 * a moment: {kind: 'same' | 'contains', q: the folder or name it narrows to}
 * or null. The top bar asks this before anything else, so only a request in
 * so many words counts — "duplicate audio files", "duplicates in Archive",
 * "videos that contain other videos" —, never a word that merely occurs in
 * a description ("he says it twice", "a compilation of goals", "how to
 * duplicate a file"): Find's language model recognizes the other phrasings
 * (find/plan.py, "asks_for"), and the view offers the words' search when a
 * query landed there wrongly (adAsWords).
 * The German words are data, matched against German queries. */
const AD_NOUN = '(?:audio ?files?|files?|recordings?|videos?|audios?|sounds?|songs?|tracks?|clips?|media|' +
    'voice messages?|mp3s?|dateien|datei|audiodateien|aufnahmen|aufnahme|tonaufnahmen|lieder|sprachnachrichten)';
// Between "duplicate" and the noun only a word of the same kind ("duplicate
// audio files"), so the verb ("duplicate a file") stays a description
const AD_MID = '(?:(?:audio|video|sound|media|mp3|wav|ton|video-?|audio-?)\\s*)?';
// A request opens with the duplicates themselves, or with a few words that ask
const AD_ASK = '^(?:(?:please|bitte|find|search|show|list|get|give|me|mir|all|alle|my|meine|the|die|any|finde|finden|such|suche|zeig|zeige|liste|gib)\\s+)*';
const AD_SAME = new RegExp(
    // "find duplicates", "Duplikate in Archive" — the noun alone, then the end or a place
    AD_ASK + '(?:duplicates|dupes|duplikate|dubletten)(?=\\s*$|\\s+(?:in|inside|under|within|from|of|im|unter|aus|von|innerhalb)(?:\\s|$))' +
    // "duplicate audio files", "doppelte Videos", "identical recordings"
    '|\\b(?:duplicate[ds]?|identical|doppelte[nrs]?|mehrfache[nrs]?|identische[nrs]?)\\s+' + AD_MID + AD_NOUN + '\\b' +
    // "recordings that exist twice" — not "twice as long", "doppelt so groß"
    '|\\b' + AD_NOUN + '\\s+(?:that|which|die)?\\s*(?:exist|are|sind)\\s+(?:twice|more than once|doppelt|mehrfach)\\b(?!\\s+(?:as|so|wie)\\b)', 'i');
const AD_CONTAINS = new RegExp(
    // "videos that contain other videos", "recordings contained in others"
    '\\b' + AD_NOUN + '\\s+(?:that\\s+|which\\s+)?contain(?:s)?\\s+(?:an)?others?\\s+' + AD_NOUN + '\\b' +
    '|\\b' + AD_NOUN + '\\s+(?:that\\s+are\\s+|which\\s+are\\s+)?contained\\s+in\\s+(?:an)?others?\\b' +
    // "Videos, die andere Videos enthalten", "Aufnahmen, die in anderen enthalten sind"
    '|\\b' + AD_NOUN + ',?\\s+(?:die\\s+)?(?:sind\\s+)?(?:in\\s+anderen|andere)\\s+(?:' + AD_NOUN + '\\s+)?enthalten\\b' +
    '|\\b' + AD_NOUN + '\\s+enth(?:ä|ae)lt\\s+andere|\\b' + AD_NOUN + '\\s+enthalten\\s+andere', 'i');
// What follows "in …" narrows it to a folder or a name
const AD_SCOPE = /\s(?:in|inside|under|within|from|of|im|unter|aus|von|innerhalb)\s+(.+)$/i;
const AD_SCOPE_LEAD = /^(?:(?:the|my|this|that|der|die|das|dem|den|meinem|meinen|meiner)\s+)?(?:(?:folders?|directory|directories|ordner[n]?|verzeichnis(?:se[n]?)?)\s+)?/i;
const AD_NO_SCOPE = /^(?:(?:the|my|der|die|meiner)\s+)?(?:library|collection|everything|all|it|them|bibliothek|sammlung|allem|allen)$/i;

function amDupIntent(text) {
    const s = String(text || '').trim();
    if (!s || s.length > 200) return null;
    // A file or folder name ("duplicates.mp3", "dupes/2024") is the file search's
    if (!/\s/.test(s) && (/[\\/*]/.test(s) || /\.[a-z0-9]{2,4}$/i.test(s))) return null;
    // A keyword only in quotes is a word to search for
    const bare = s.replace(/["„“”»«][^"„“”»«]*["„“”»«]/g, ' ').trim();
    const contains = AD_CONTAINS.test(bare);
    if (!contains && !AD_SAME.test(bare)) return null;
    // What follows "in …" once the request's own words are out
    const rest = ' ' + s.replace(new RegExp(AD_SAME.source, 'gi'), ' ').replace(new RegExp(AD_CONTAINS.source, 'gi'), ' ');
    let q = '';
    const m = rest.match(AD_SCOPE);
    if (m) {
        q = m[1].replace(/["„“”»«]/g, '').replace(AD_SCOPE_LEAD, '').replace(/[?!.]+$/, '')
            .replace(/\s+(?:folders?|directory|ordner|verzeichnis)$/i, '').trim();
        if (AD_NO_SCOPE.test(q) || new RegExp('^' + AD_NOUN + '$', 'i').test(q)) q = '';
    }
    return { kind: contains ? 'contains' : 'same', q };
}

// The top bar's query was routed here (searchview.js svRouteQuery)
function amIncoming(text) {
    const want = amDupIntent(text);
    if (!want) return;
    adSet({ kind: want.kind, q: want.q });
    _ad.asked = String(text || '').trim();
}

// The query wasn't meant for the duplicates: the words, searched as said
function adAsWords() {
    const words = _ad.asked;
    _ad.asked = '';
    if (!words) return;
    if (typeof svState !== 'undefined') svState.routed = words;
    const input = document.getElementById('svQuery');
    if (input) input.value = words;
    svTab('text');
}

function adSet(o) {
    _ad.seq++;                         // a load still on its way is of the old filters
    _ad.asked = '';
    _ad.kind = o.kind === 'contains' ? 'contains' : 'same';
    _ad.f = { ...AD_NO_FILTERS(), q: String(o.q || ''), minSec: Number(o.minSec) || 0,
              person: String(o.person || '') };
    _ad.pair = null;
    _ad.file = o.file || '';
    _ad.overview = null;
    _ad.list = null;
    _ad.open = new Set();
    _ad.allFolders = false;
    _ad.moreFolders = false;
    if (typeof _am !== 'undefined') _am.view = _ad.kind;
}

/* The duplicates in the search view's tab — from Find, Ask's card, the file
 * view: {kind, q, minSec, person, file}. */
function amShowDuplicates(o) {
    adSet(o || {});
    const keep = document.getElementById('svQuery')?.value || '';
    if (typeof svState !== 'undefined') svState.routed = keep;
    const visible = document.getElementById('svOverlay')?.classList.contains('visible');
    if (visible && typeof svTab === 'function' && svState.tab !== 'audio') { svTab('audio'); return; }
    svShowTab('audio');
    if (visible) { if (typeof amOpen === 'function') amOpen(); return; }
    openSearchView(keep);          // its svRun() opens this tab: amOpen()
}

// ── Loading ─────────────────────────────────────────────────────────────────

function adQuery(extra) {
    const f = _ad.f, p = new URLSearchParams();
    const put = (k, v) => {
        if (Array.isArray(v) ? v.length : (v !== '' && v !== null && v !== undefined && v !== 0)) {
            p.set(k, Array.isArray(v) ? v.join(',') : String(v));
        }
    };
    put('q', f.q); put('q_whole', f.qWhole); put('q_part', f.qPart); put('person', f.person);
    put('media', f.media); put('ext', f.ext); put('min_sec', f.minSec); put('max_sec', f.maxSec);
    put('since', f.since); put('until', f.until); put('lang', f.lang); put('transcribed', f.transcribed);
    put('min_copies', f.minCopies); put('spread', f.spread); put('min_parts', f.minParts);
    for (const [k, v] of Object.entries(extra || {})) if (v !== '' && v !== null && v !== undefined) p.set(k, String(v));
    return p.toString();
}

// How many of the panel's filters are set (the bar's own two not counted)
function adPanelCount() {
    const f = _ad.f;
    return [f.qWhole, f.qPart, f.person, f.media, f.ext.length, f.maxSec, f.since, f.until, f.lang,
            f.transcribed, f.minCopies, f.spread, f.minParts].filter(Boolean).length;
}

// The tab's view "The same recording" or "Contained in another"
async function adOpen(kind) {
    if (kind) _ad.kind = kind;
    if (_ad.overview && _ad.list && _ad.list.kind === _ad.kind) {
        _ad.seq++;                     // a load of the other view, still on its way, is dropped
        adRender();
        return;
    }
    await adLoad(false);
}

async function adLoad(more) {
    const box = document.getElementById('amResults');
    if (!box) return;
    const seq = ++_ad.seq;
    const kind = _ad.kind;
    const have = more && _ad.list && _ad.list.kind === kind ? _ad.list : null;
    if (!more) box.innerHTML = '<div class="sv-empty">Loading …</div>';
    const list = adQuery({ kind, sort: _ad.sort[kind], pair: kind === 'same' ? _ad.pair?.id : '',
                           file: _ad.file, offset: have ? have.items.length : 0, limit: AD_PAGE });
    // The tiles and folders follow the filters, not the order or the page
    const ovKey = adQuery({ folders: _ad.allFolders ? 200 : 12 });
    let ov = _ad.overview && _ad.ovKey === ovKey ? _ad.overview : null, page;
    // The overview first when it is to be worked out anew: the list then
    // comes from the same new state
    const fresh = _ad.fresh;
    _ad.fresh = false;
    try {
        if (!ov && fresh) ov = await api('/api/audio/overview?' + ovKey + '&fresh=1');
        [ov, page] = await Promise.all([
            ov || api('/api/audio/overview?' + ovKey),
            api('/api/audio/duplicates?' + list),
        ]);
    } catch (e) {
        if (seq === _ad.seq && !more) box.innerHTML = `<div class="sv-empty ms-err">${esc(e.message)}</div>`;
        return;
    }
    if (seq !== _ad.seq || kind !== _ad.kind) return;
    let items = kind === 'same' ? page.groups : page.wholes;
    if (have) {
        // While the run finds more, the order moves between pages: a card
        // already shown is not shown twice
        const seen = new Set(have.items.map(adCardKey));
        items = have.items.concat(items.filter(x => !seen.has(adCardKey(x))));
    } else {
        // A new list: the preview's buttons and the opened cards were the old one's
        _ad.open = new Set();
        adResetPreview();
    }
    _ad.overview = ov;
    _ad.ovKey = ovKey;
    _ad.list = { kind, items, total: page.total || 0 };
    if (typeof amRenderViews === 'function') amRenderViews();
    if (_am.view === kind) adRender();
}

function adCardKey(x) {
    return x.whole ? 'w' + x.whole.id : 'g' + Math.min(...(x.files || []).map(f => f.id));
}

function adResetPreview() {
    const pv = document.getElementById('amPreview');
    if (!pv) return;
    if (typeof amPausePreview === 'function') amPausePreview();
    pv.innerHTML = adPreviewHint();
}

// Only the tiles and folders anew (all folder pairs): the list stays as it is
async function adLoadOverview() {
    const seq = _ad.seq;
    const ovKey = adQuery({ folders: _ad.allFolders ? 200 : 12 });
    let ov;
    try { ov = await api('/api/audio/overview?' + ovKey); }
    catch (e) { if (typeof Toast !== 'undefined') Toast.error(e.message); return; }
    if (seq !== _ad.seq) return;
    _ad.overview = ov;
    _ad.ovKey = ovKey;
    adRender();
}

// ── The view ────────────────────────────────────────────────────────────────

function adRender() {
    const box = document.getElementById('amResults');
    const ov = _ad.overview, list = _ad.list;
    if (!box || !ov || !list) return;
    _ad.items = new Map();
    const pv = document.getElementById('amPreview');
    if (pv && !pv.querySelector('audio')) pv.innerHTML = adPreviewHint();
    // One recording's (the file view's chip): its copies, not the library's tiles
    box.innerHTML = `<div class="ad">
        ${_ad.file ? '' : adTilesHtml(ov)}
        ${adFiltersHtml()}
        ${_ad.kind === 'same' && !_ad.file ? adFoldersHtml(ov) : ''}
        <section class="ad-list">${adListHtml(list)}</section>
    </div>`;
    adWaves();
}

function adPreviewHint() {
    return `<div class="sv-empty">${_ad.kind === 'same'
        ? 'Each copy is drawn as its waveform, on the timeline the copies share. Click one to hear it from there — and the same moment in the other copies.'
        : 'Each recording that holds others, with every one it holds where it begins. Click a block or the waveform to hear it.'}
        <br><br>Found by the sound, not by the names: also re-encoded, in other formats or folders.</div>`;
}

function adTile(big, label, sub, onclick, title) {
    const tag = onclick ? 'button' : 'div';
    return `<${tag} class="ad-tile${onclick ? ' link' : ''}"${onclick ? ` onclick="${onclick}"` : ''}${title ? ` title="${esc(title)}"` : ''}>
        <b>${big}</b><span>${label}</span>${sub ? `<small>${sub}</small>` : ''}</${tag}>`;
}

function adTilesHtml(ov) {
    const s = ov.same || {}, c = ov.contains || {};
    const n = (x) => Number(x || 0).toLocaleString();
    const twice = s.groups === 1 ? 'recording exists more than once' : 'recordings exist more than once';
    const holds = c.wholes === 1 ? 'recording contains others' : 'recordings contain others';
    const inside = c.parts === 1 ? 'recording is inside another' : 'recordings are inside another';
    const tiles = _ad.kind === 'same' ? [
        adTile(n(s.groups), twice, `${n(s.files)} files — ${n(s.extra)} of them ${s.extra === 1 ? 'an extra copy' : 'extra copies'}`),
        adTile(amBytes(s.extra_bytes), 'in the extra copies', 'every copy but the largest of each kind',
               '', 'A video and a sound file with its sound both count as kept: the picture is not in the sound'),
        adTile(n(c.wholes), holds, `${n(c.parts)} ${c.parts === 1 ? 'is' : 'are'} inside another →`,
               "amView('contains')", 'Show them'),
    ] : [
        adTile(n(c.wholes), holds, `at ${n(c.pairs)} place${c.pairs === 1 ? '' : 's'}`),
        adTile(n(c.parts), inside, 'a clip, a cut, a song in a video'),
        adTile(n(s.groups), twice, `${amBytes(s.extra_bytes)} in extra copies →`, "amView('same')", 'Show them'),
    ];
    const run = (_am.info || {}).run || {};
    const note = run.running
        ? `<div class="ad-note">⟳ Among ${n(ov.analyzed)} recording${ov.analyzed === 1 ? '' : 's'} analyzed so far — the library run goes on, and more turn up as it does.</div>`
        : '';
    return `<div class="ad-tiles">${tiles.join('')}</div>${note}`;
}

function adFiltersHtml() {
    const lens = AD_LENGTHS.map(([sec, label]) =>
        `<button class="ms-chip${_ad.f.minSec === sec ? ' on' : ''}" onclick="adSetLength(${sec})">${label}</button>`).join('');
    const nf = adPanelCount();
    const sorts = AD_SORTS[_ad.kind].map(([k, label]) =>
        `<option value="${k}"${_ad.sort[_ad.kind] === k ? ' selected' : ''}>${label}</option>`).join('');
    const only = [];
    if (_ad.pair) only.push(`<span class="ad-only">Only ${esc(_ad.pair.label)} <button class="vx-icon" onclick="adClearPair()" title="All folders">✕</button></span>`);
    if (_ad.file) only.push(`<span class="ad-only">Only <b>${esc(String(_ad.file).split('/').pop())}</b> <button class="vx-icon" onclick="adClearFile()" title="All recordings">✕</button></span>`);
    const asked = _ad.asked
        ? `<div class="ad-asked">Opened for “${esc(_ad.asked)}” — <button class="ms-link" onclick="adAsWords()">search these words instead</button></div>`
        : '';
    return `${asked}<div class="ad-filters">
        <input type="search" class="ad-q" id="adFilter" value="${esc(_ad.f.q)}" autocomplete="off" spellcheck="false"
               placeholder="Only a folder or name — e.g. Archive, Phone, .mp3 (Enter)" onkeydown="adFilterKey(event)"
               oninput="if (!this.value) adSetFilter('')">
        <span class="ad-lens">${lens}</span>
        <button class="vx-btn ad-panel-btn${_ad.panel ? ' on' : ''}" onclick="adPanel()" aria-expanded="${_ad.panel}"
                title="Narrow it down: person, video or sound, format, length, date, language, copies, folders">⚙ Filters${nf ? ` <b>${nf}</b>` : ''}</button>
        <select class="ad-sort" onchange="adSetSort(this.value)" title="Order">${sorts}</select>
        <button class="vx-btn ad-again" onclick="adRefresh()" title="Look again — the analysis finds more while it goes">↻</button>
        ${only.join('')}
    </div>${_ad.panel ? adPanelHtml() : adChipsHtml()}`;
}

// ── The ⚙ panel ─────────────────────────────────────────────────────────────

const AD_MAX_LENGTHS = [[0, 'Any'], [60, '1 min'], [600, '10 min'], [3600, '1 h']];
const AD_KINDS = [['', 'Any'], ['video', 'Video'], ['audio', 'Sound only']];
const AD_TRANSCRIBED = [['', 'Any'], ['yes', 'Transcribed'], ['no', 'Not transcribed']];
const AD_COPIES = [[0, 'Any'], [3, '3+'], [5, '5+'], [10, '10+']];
const AD_SPREAD = [['', 'Anywhere'], ['across', 'Across folders'], ['within', 'In one folder']];
const AD_PARTS = [[0, 'Any'], [2, '2+'], [5, '5+'], [10, '10+']];

// A row of chips that choose one value of a filter
function adChoice(key, options) {
    return options.map(([v, label], i) => `<button class="ms-chip${_ad.f[key] === v ? ' on' : ''}"
        onclick="adChoose('${key}', ${i})">${esc(label)}</button>`).join('');
}

function adPanelHtml() {
    const f = _ad.f, fc = (_ad.overview || {}).facets || {};
    const people = (fc.people || []).map(([name, n]) =>
        `<option value="${esc(name)}"${f.person === name ? ' selected' : ''}>${esc(name)} (${Number(n).toLocaleString()})</option>`).join('');
    const unknownPerson = f.person && !(fc.people || []).some(([name]) => name === f.person)
        ? `<option value="${esc(f.person)}" selected>${esc(f.person)}</option>` : '';
    const langs = (fc.lang || []).map(([l, n]) =>
        `<option value="${esc(l)}"${f.lang === l ? ' selected' : ''}>${esc(l)} (${Number(n).toLocaleString()})</option>`).join('');
    // The common formats; the rare ones behind "more" (a chosen one always shows)
    const extAll = fc.ext || [];
    const extShown = _ad.allExt ? extAll : extAll.filter(([e], i) => i < 8 || f.ext.includes(e));
    const exts = extShown.map(([e, n]) =>
        `<button class="ms-chip${f.ext.includes(e) ? ' on' : ''}" onclick="adToggleExt(${extAll.findIndex(x => x[0] === e)})">${esc(e)} <small>${Number(n).toLocaleString()}</small></button>`).join('')
        + (extShown.length < extAll.length ? `<button class="ms-link" onclick="_ad.allExt = true; adRender()">+ ${extAll.length - extShown.length} more</button>` : '');
    const days = fc.days || [];
    const row = (label, body, hint) => `<div class="ad-prow"><span class="ad-plabel"${hint ? ` title="${esc(hint)}"` : ''}>${label}</span>
        <span class="ad-pbody">${body}</span></div>`;
    const same = _ad.kind === 'same';
    return `<div class="ad-panel">
        ${row('Person', `<select class="ad-sel" onchange="adSetF('person', this.value)">
            <option value="">Anyone</option>${unknownPerson}${people}</select>`,
            'Speaks in one of the recordings — the names of the voice map')}
        ${row('Kind', adChoice('media', AD_KINDS))}
        ${exts ? row('Format', exts, 'Any of the chosen formats') : ''}
        ${row('Recorded', `<input type="date" class="ad-date" value="${esc(f.since)}"${days[0] ? ` min="${esc(days[0])}"` : ''}${days[1] ? ` max="${esc(days[1])}"` : ''}
                onchange="adSetF('since', this.value)" aria-label="Recorded from">
            <span class="ad-to">to</span>
            <input type="date" class="ad-date" value="${esc(f.until)}"${days[0] ? ` min="${esc(days[0])}"` : ''}${days[1] ? ` max="${esc(days[1])}"` : ''}
                onchange="adSetF('until', this.value)" aria-label="Recorded until">`,
            'The date in the file name (a copy\'s file date is often only when it was copied), else the file\'s own')}
        ${row('At most', adChoice('maxSec', AD_MAX_LENGTHS), 'The longest recording shown')}
        ${langs ? row('Language', `<select class="ad-sel" onchange="adSetF('lang', this.value)">
            <option value="">Any</option>${langs}</select>`, 'The language of the transcript') : ''}
        ${row('Transcript', adChoice('transcribed', AD_TRANSCRIBED))}
        ${same ? row('Copies', adChoice('minCopies', AD_COPIES)) + row('Where', adChoice('spread', AD_SPREAD),
            'Copies in different folders (a backup, a second install) or next to each other in one') : `
            ${row('Holding one in', `<input type="search" class="ad-q ad-q-small" value="${esc(f.qWhole)}" placeholder="a folder or name — e.g. vods"
                onkeydown="if (event.key === 'Enter') adSetF('qWhole', this.value)" onchange="adSetF('qWhole', this.value)">`,
                'Letters of the path of the recording that holds the others — a stream, a compilation')}
            ${row('Parts in', `<input type="search" class="ad-q ad-q-small" value="${esc(f.qPart)}" placeholder="a folder or name — e.g. clips"
                onkeydown="if (event.key === 'Enter') adSetF('qPart', this.value)" onchange="adSetF('qPart', this.value)">`,
                'Only these parts are shown, and only the recordings holding one of them')}
            ${row('Parts', adChoice('minParts', AD_PARTS))}`}
        <div class="ad-prow ad-pfoot"><button class="ms-link" onclick="adResetFilters()">Reset the filters</button>
            <button class="ms-link" onclick="adPanel()">Close</button></div>
    </div>`;
}

// The filters set, as chips that remove them — when the panel is closed
function adChipsHtml() {
    const f = _ad.f, out = [];
    const chip = (label, key) => out.push(`<span class="ad-chip">${label}<button class="vx-icon" onclick="adClearF('${key}')"
        title="Without this filter">✕</button></span>`);
    if (f.person) chip(`👤 ${esc(f.person)}`, 'person');
    if (f.media) chip(f.media === 'video' ? 'Video' : 'Sound only', 'media');
    if (f.ext.length) chip(esc(f.ext.join(', ')), 'ext');
    if (f.since || f.until) chip(`📅 ${esc(f.since || '…')} – ${esc(f.until || '…')}`, 'days');
    if (f.maxSec) chip(`≤ ${amClock(f.maxSec)}`, 'maxSec');
    if (f.lang) chip(`🗣 ${esc(f.lang)}`, 'lang');
    if (f.transcribed) chip(f.transcribed === 'yes' ? 'Transcribed' : 'Not transcribed', 'transcribed');
    if (f.minCopies) chip(`${f.minCopies}+ copies`, 'minCopies');
    if (f.spread) chip(f.spread === 'across' ? 'Across folders' : 'In one folder', 'spread');
    if (f.qWhole) chip(`Holding one in “${esc(f.qWhole)}”`, 'qWhole');
    if (f.qPart) chip(`Parts in “${esc(f.qPart)}”`, 'qPart');
    if (f.minParts) chip(`${f.minParts}+ parts`, 'minParts');
    return out.length ? `<div class="ad-chips">${out.join('')}<button class="ms-link" onclick="adResetFilters()">Reset</button></div>` : '';
}

function adPanel() {
    _ad.panel = !_ad.panel;
    adRender();
}

const AD_CHOICES = { media: AD_KINDS, maxSec: AD_MAX_LENGTHS, transcribed: AD_TRANSCRIBED,
                     minCopies: AD_COPIES, spread: AD_SPREAD, minParts: AD_PARTS };

function adChoose(key, i) {
    const opt = (AD_CHOICES[key] || [])[i];
    if (opt) adSetF(key, opt[0]);
}

function adSetF(key, value) {
    if (!(key in _ad.f)) return;
    const v = typeof _ad.f[key] === 'number' ? Number(value) || 0 : String(value || '').trim();
    if (JSON.stringify(_ad.f[key]) === JSON.stringify(v)) return;
    _ad.f[key] = v;
    _ad.pair = null;
    adReload();
}

function adToggleExt(i) {
    const e = (((_ad.overview || {}).facets || {}).ext || [])[i]?.[0];
    if (!e) return;
    const has = _ad.f.ext.includes(e);
    _ad.f.ext = has ? _ad.f.ext.filter(x => x !== e) : [..._ad.f.ext, e];
    _ad.pair = null;
    adReload();
}

function adClearF(key) {
    if (key === 'days') { _ad.f.since = ''; _ad.f.until = ''; }
    else if (key === 'ext') _ad.f.ext = [];
    else if (key in _ad.f) _ad.f[key] = typeof _ad.f[key] === 'number' ? 0 : '';
    adReload();
}

function adResetFilters() {
    _ad.f = { ...AD_NO_FILTERS(), q: _ad.f.q, minSec: _ad.f.minSec };
    _ad.pair = null;
    adReload();
}

function adShortFolder(path, names = 2) {
    const parts = String(path || '').split('/').filter(Boolean);
    if (!parts.length) return '/';
    return (parts.length > names ? '…/' : '') + parts.slice(-names).join('/');
}

function adFoldersHtml(ov) {
    const fs = ov.folders || [];
    if (!fs.length) return '';
    const n = (x) => Number(x || 0).toLocaleString();
    const pct = (a, b) => (b ? Math.min(100, Math.round(a / b * 100)) : 0);
    const shown = _ad.allFolders || _ad.moreFolders ? fs : fs.slice(0, AD_FOLDERS_SHOWN);
    const rows = shown.map((f, i) => {
        const on = _ad.pair && _ad.pair.id === f.id ? ' on' : '';
        if (f.within) {
            return `<button class="ad-pair within${on}" onclick="adPickPair(${i})" title="${esc(f.a)}">
                <span class="ad-within-ico">⟲</span>
                <span class="ad-pair-main"><b>${n(f.groups)}</b> recording${f.groups === 1 ? '' : 's'} twice or more inside one folder:
                    <span class="ad-folder">${esc(adShortFolder(f.a, 3))}</span></span>
                <small>${n(f.a_files)} of its ${n(f.a_total)} files · ${amBytes(f.a_bytes)}</small>
            </button>`;
        }
        // A folder inside the other is named by its own name
        const own = (path) => String(path || '').split('/').filter(Boolean).pop() || '/';
        const la = f.a_rest || own(f.a), lb = f.b_rest || own(f.b);
        const pa = pct(f.a_files, f.a_total), pb = pct(f.b_files, f.b_total);
        const whole = (p) => (p >= 95 ? ' whole' : '');
        return `<button class="ad-pair${on}" onclick="adPickPair(${i})">
            ${f.common && f.common !== '/' ? `<span class="ad-pair-common" title="${esc(f.common)}">${esc(adShortFolder(f.common, 3))}/</span>` : ''}
            <span class="ad-pair-row">
                <span class="ad-side a${whole(pa)}" title="${esc(f.a)}">
                    <span class="ad-side-name">${esc(adShortFolder(la, 2))}</span>
                    <span class="ad-cover"><i style="width:${pa}%"></i></span>
                    <small>${pa} % · ${n(f.a_files)} of ${n(f.a_total)} · ${amBytes(f.a_bytes)}</small>
                </span>
                <span class="ad-link" title="${n(f.groups)} ${f.groups === 1 ? 'recording is' : 'recordings are'} in both">
                    <b>${n(f.groups)}</b><small>in both</small>
                </span>
                <span class="ad-side b${whole(pb)}" title="${esc(f.b)}">
                    <span class="ad-side-name">${esc(adShortFolder(lb, 2))}</span>
                    <span class="ad-cover"><i style="width:${pb}%"></i></span>
                    <small>${pb} % · ${n(f.b_files)} of ${n(f.b_total)} · ${amBytes(f.b_bytes)}</small>
                </span>
            </span>
            ${f.layout > 1 ? `<span class="ad-pair-note">the same layout in ${n(f.layout)} subfolders — a copied folder</span>` : ''}
        </button>`;
    }).join('');
    const total = ov.folders_total || fs.length;
    const all = shown.length < fs.length
        ? `<button class="ms-link" onclick="adMoreFolders()">Show ${n(fs.length - shown.length)} more</button>`
        : total > fs.length && !_ad.allFolders
            ? `<button class="ms-link" onclick="adAllFolders()">Show all ${n(total)}</button>` : '';
    return `<section class="ad-folders">
        <h4>Where the copies are <small>${n(total)} folder pair${total === 1 ? '' : 's'} · the share of each folder that is also in the other · click one for its recordings</small> ${all}</h4>
        <div class="ad-pairs">${rows}</div>
    </section>`;
}

function adListHtml(list) {
    const n = (x) => Number(x || 0).toLocaleString();
    if (!list.items.length) {
        const why = _ad.f.q ? ` with “${esc(_ad.f.q)}” in the path` : '';
        const len = _ad.f.minSec ? ` of at least ${amClock(_ad.f.minSec)}` : '';
        const more = adPanelCount() ? ' that fits the filters' : '';
        const what = _ad.file
            ? (_ad.kind === 'same' ? 'It exists nowhere else' : 'It holds no other recording, and none holds it')
            : (_ad.kind === 'same' ? `No recording${len}${why}${more} exists twice`
                                   : `No recording${len}${why}${more} contains another one`);
        const reset = _ad.f.q || _ad.f.minSec || adPanelCount() || _ad.pair || _ad.file
            ? ` <button class="ms-link" onclick="adReset()">Show all</button>` : '';
        return `<div class="sv-empty">${what} — among the ${n((_ad.overview || {}).analyzed)} recording${(_ad.overview || {}).analyzed === 1 ? '' : 's'} analyzed so far.${reset}</div>`;
    }
    const cards = list.kind === 'same'
        ? list.items.map((g, i) => adGroupHtml(g, i)).join('')
        : list.items.map((w, i) => adWholeHtml(w, i)).join('');
    const shown = list.items.length;
    const more = shown < list.total
        ? `<button class="vx-btn am-more" onclick="adLoad(true)">Show more (${n(shown)} of ${n(list.total)})</button>` : '';
    const head = `<div class="ad-list-head">${n(list.total)} ${list.kind === 'same'
        ? `recording${list.total === 1 ? '' : 's'} with copies` : `recording${list.total === 1 ? '' : 's'} holding others`}</div>`;
    return head + cards + more;
}

// A recording's date and the people speaking in it, small
function adFacts(r) {
    const day = r.day ? `<span class="ad-day" title="Recorded (the date in the name, else the file's)">${esc(r.day)}</span>` : '';
    const people = (r.people || []).slice(0, 4).map(n => `<span class="ad-person" title="Speaks in it">${esc(n)}</span>`).join('');
    const more = (r.people || []).length > 4 ? `<span class="ad-person">+${r.people.length - 4}</span>` : '';
    return day + people + more;
}

function adExt(path, video) {
    const m = String(path || '').match(/\.([a-z0-9]{1,5})$/i);
    return `<span class="ad-ext${video ? ' video' : ''}" title="${video ? 'A video' : 'Sound only'}">${esc(m ? m[1].toUpperCase() : '?')}</span>`;
}

// One recording's row of buttons: hear it here, the main player, the transcript
function adButtons(k) {
    return `<span class="ad-btns">
        <button class="vx-btn" onclick="adHear('${k}', 0)" title="Hear it">▶</button>
        <button class="vx-btn" onclick="adOpenIn('${k}', false)" title="Play it in the player of the main window">⧉</button>
        <button class="vx-btn" onclick="adOpenIn('${k}', true)" title="Open its transcript">📄</button>
    </span>`;
}

function adGroupHtml(g, gi) {
    const files = g.files || [];
    const span = Math.max(...files.map(f => (Number(f.pos) || 0) + (Number(f.duration) || 0)), 0.1);
    const open = _ad.open.has(gi);
    const shown = open ? files : files.slice(0, AD_COPIES_SHOWN);
    const rows = shown.map((f, ci) => {
        const k = `g${gi}.${ci}`;
        _ad.items.set(k, { ...f, group: gi, copy: ci });
        const loud = amLoud(f);
        return `<div class="ad-copy${f.largest ? ' largest' : ''}">
            <span class="ad-copy-meta">
                <span class="ad-copy-top">${adExt(f.path, f.video)}<span class="ad-size">${amBytes(f.size)}</span>
                    ${f.largest && g.count > 1 ? '<span class="ad-keep" title="The largest copy of its kind — the one counted as kept">largest</span>' : ''}
                    ${loud ? `<span class="ad-loud">${loud}</span>` : ''}${adFacts(f)}</span>
                <span class="ms-row-name" title="${esc(f.path)}">${esc(f.name || '')}</span>
                <span class="ms-row-text" title="${esc(f.path)}">${esc(amFolder(f))}</span>
            </span>
            <canvas class="ad-wave" data-k="${k}" data-span="${span}" role="img" aria-label="Waveform of ${esc(f.name || '')}"
                    onclick="adWaveClick(event, this)" onmousemove="adHover(event, this)" onmouseleave="adHoverOut(this)"></canvas>
            ${adButtons(k)}
        </div>`;
    }).join('');
    const rest = files.length - shown.length + Math.max(0, (g.count || files.length) - files.length);
    const more = !open && rest > 0
        ? `<button class="ms-link ad-more-copies" onclick="adToggle(${gi})">+ ${rest.toLocaleString()} more cop${rest === 1 ? 'y' : 'ies'}</button>` : '';
    const extra = g.extra_bytes ? ` · <span class="ad-extra">${amBytes(g.extra_bytes)} in the extra cop${g.extra === 1 ? 'y' : 'ies'}</span>` : '';
    return `<div class="ad-card" data-g="${gi}">
        <div class="ad-card-head"><b>${(g.count || files.length).toLocaleString()} copies</b> · ${amClock(g.length)}${extra}</div>
        <div class="ad-copies">${rows}</div>${more}
    </div>`;
}

// Parts on lanes: a part that begins before the last one on a lane ends goes
// to the next lane (each block at least a sliver wide, so a short clip in a
// long stream is seen)
function adLanes(parts, dur) {
    const sliver = dur * 0.006;
    const ends = [];
    return parts.map(p => {
        const a = Number(p.at) || 0, b = a + Math.max(Number(p.matched) || Number(p.duration) || 0, sliver);
        let lane = ends.findIndex(e => e <= a);
        if (lane < 0) { lane = ends.length; ends.push(b); } else ends[lane] = b;
        return Math.min(lane, 5);
    });
}

function adWholeHtml(w, wi) {
    const whole = w.whole, parts = w.parts || [];
    const dur = Math.max(Number(whole.duration) || 0, 0.1);
    const wk = `w${wi}`;
    _ad.items.set(wk, { ...whole, pos: 0, whole: wi });
    const lanes = adLanes(parts, dur);
    const nLanes = Math.max(...lanes, 0) + 1;
    const ks = parts.map((p, pi) => {
        const k = `w${wi}.${pi}`;
        _ad.items.set(k, { ...p, pos: 0, part: pi, wholeK: wk });
        return k;
    });
    const blocks = parts.map((p, pi) => {
        const left = Math.max(0, Math.min(100, (Number(p.at) || 0) / dur * 100));
        const width = Math.max(0.6, Math.min(100 - left, (Number(p.matched) || Number(p.duration) || 0) / dur * 100));
        return `<span class="ad-block" style="left:${left}%;width:${width}%;top:${lanes[pi] * 9}px;background:var(--sp${pi % 8})"
                      title="${esc(p.name || '')} — from ${amClock(p.at)}, ${amClock(p.matched || p.duration)}"
                      onclick="adHearPart('${ks[pi]}')"></span>`;
    }).join('');
    const open = _ad.open.has(wi);
    const shown = open ? parts : parts.slice(0, AD_PARTS_SHOWN);
    const rows = shown.map((p, pi) => `<div class="ad-part">
            <span class="ad-dot" style="background:var(--sp${pi % 8})"></span>
            <button class="ms-link ad-part-at" onclick="adHearPart('${ks[pi]}')" title="Hear it where it begins in the whole">${amClock(p.at)}</button>
            <span class="ms-row-main">
                <span class="ms-row-name" title="${esc(p.path)}">${esc(p.name || '')}</span>
                <span class="ms-row-text" title="${esc(p.path)}">${esc(amFolder(p))}</span>
            </span>
            <span class="ad-facts">${adFacts(p)}</span>
            ${adExt(p.path, p.video)}<span class="ad-size">${amClock(p.duration)}</span>
            ${adButtons(ks[pi])}
        </div>`).join('');
    const rest = parts.length - shown.length;
    const more = rest > 0
        ? `<button class="ms-link ad-more-copies" onclick="adToggle(${wi})">+ ${rest.toLocaleString()} more</button>` : '';
    const ticks = [0, 0.25, 0.5, 0.75, 1].map(x => `<span style="left:${x * 100}%">${amClock(x * dur)}</span>`).join('');
    const covered = w.covered ? ` · ${amClock(w.covered)} of it` : '';
    // Parts left out by a filter: how many there are in all
    const all = w.parts_all && w.parts_all > parts.length ? ` of ${w.parts_all.toLocaleString()}` : '';
    return `<div class="ad-card ad-whole" data-w="${wi}">
        <div class="ad-card-head">${adExt(whole.path, whole.video)}
            <span class="ms-row-main"><span class="ms-row-name" title="${esc(whole.path)}">${esc(whole.name || '')}</span>
                <span class="ms-row-text" title="${esc(whole.path)}">${esc(amFolder(whole))}</span></span>
            <span class="ad-facts">${adFacts(whole)}</span>
            <span class="ad-holds">${amClock(dur)} · holds <b>${parts.length.toLocaleString()}</b>${all} recording${(w.parts_all || parts.length) === 1 ? '' : 's'}${covered}</span>
            ${adButtons(wk)}
        </div>
        <div class="ad-timeline">
            <canvas class="ad-wave big" data-k="${wk}" data-span="${dur}" role="img" aria-label="Waveform of ${esc(whole.name || '')}"
                    onclick="adWaveClick(event, this)" onmousemove="adHover(event, this)" onmouseleave="adHoverOut(this)"></canvas>
            <div class="ad-lanes" style="height:${nLanes * 9}px">${blocks}</div>
            <div class="ad-axis">${ticks}</div>
        </div>
        <div class="ad-parts">${rows}</div>${more}
    </div>`;
}

// ── Waveforms ───────────────────────────────────────────────────────────────

function adB64(s) {
    const bin = atob(s);
    const out = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
    return out;
}

// Draw what is there, ask for what is not (a page's worth at a time)
async function adWaves() {
    const canvases = [...document.querySelectorAll('#amResults canvas.ad-wave')];
    canvases.forEach(adDraw);
    adWatchSize();
    const want = [...new Set(canvases.map(cv => _ad.items.get(cv.dataset.k)?.id)
        .filter(id => id !== undefined && !_ad.levels.has(id) && !_ad.wanted.has(id)))];
    for (let i = 0; i < want.length; i += 100) {
        const ids = want.slice(i, i + 100);
        ids.forEach(id => _ad.wanted.add(id));
        let r;
        try { r = await api(`/api/audio/levels?n=600&ids=${ids.join(',')}`); }
        catch (e) { ids.forEach(id => _ad.wanted.delete(id)); continue; }
        for (const [id, b64] of Object.entries(r.levels || {})) _ad.levels.set(Number(id), adB64(b64));
        // a recording without a stored waveform is not asked for again
        ids.forEach(id => { if (!_ad.levels.has(id)) _ad.levels.set(id, null); _ad.wanted.delete(id); });
        document.querySelectorAll('#amResults canvas.ad-wave').forEach(cv => {
            if (ids.includes(_ad.items.get(cv.dataset.k)?.id)) adDraw(cv);
        });
    }
}

function adWatchSize() {
    if (_ad.resize || typeof ResizeObserver === 'undefined') return;
    const box = document.getElementById('amResults');
    if (!box) return;
    let t = null;
    _ad.resize = new ResizeObserver(() => {
        clearTimeout(t);
        t = setTimeout(() => document.querySelectorAll('#amResults canvas.ad-wave').forEach(adDraw), 120);
    });
    _ad.resize.observe(box);
}

/* A copy's waveform where it lies on the shared timeline (``span`` seconds):
 * where its sound runs a faint line, over it the levels — each recording
 * against its own loudest, so a quieter copy shows the same shape. */
function adDraw(cv) {
    const it = _ad.items.get(cv.dataset.k);
    if (!it || !cv.clientWidth) return;
    const dpr = window.devicePixelRatio || 1;
    const w = Math.max(1, Math.round(cv.clientWidth * dpr)), h = Math.max(1, Math.round(cv.clientHeight * dpr));
    if (cv.width !== w) cv.width = w;
    if (cv.height !== h) cv.height = h;
    const g = cv.getContext('2d');
    if (!g) return;
    g.clearRect(0, 0, w, h);
    const span = Number(cv.dataset.span) || Number(it.duration) || 1;
    const x0 = (Number(it.pos) || 0) / span * w;
    const x1 = Math.max(x0 + 1, ((Number(it.pos) || 0) + (Number(it.duration) || 0)) / span * w);
    const mid = h / 2;
    const color = getComputedStyle(cv).color;
    g.globalAlpha = 0.35;
    g.fillStyle = color;
    g.fillRect(x0, mid - dpr / 2, x1 - x0, dpr);
    g.globalAlpha = 1;
    const lv = _ad.levels.get(it.id);
    if (!lv || !lv.length) return;
    let top = 64;
    for (const v of lv) if (v > top) top = v;
    const per = (x1 - x0) / lv.length;
    const bar = Math.max(1, per > 3 ? per - 1 : per);
    for (let i = 0; i < lv.length; i++) {
        const v = lv[i];
        if (!v) continue;
        // a little contrast: loud stretches stand out from the merely audible
        const a = Math.max(0.5, Math.pow(v / top, 1.8) * (mid - dpr));
        g.fillRect(x0 + i * per, mid - a, bar, 2 * a);
    }
}

// ── Hearing and opening ─────────────────────────────────────────────────────

function adWaveClick(ev, cv) {
    const k = cv.dataset.k;
    const it = _ad.items.get(k);
    if (!it) return;
    const r = cv.getBoundingClientRect();
    const span = Number(cv.dataset.span) || Number(it.duration) || 1;
    const t = (ev.clientX - r.left) / Math.max(1, r.width) * span - (Number(it.pos) || 0);
    adHear(k, Math.max(0, Math.min(Number(it.duration) || 0, t)));
}

function adClip(path, t, len = 60) {
    const start = Math.max(0, Number(t) || 0);
    return `/api/clip/audio?file=${encodeURIComponent(path)}&start=${start.toFixed(2)}&end=${(start + len).toFixed(2)}`;
}

// The preview: this recording from ``t``, and for a copy the same moment in
// the other copies of its group
function adHear(k, t) {
    const it = _ad.items.get(k);
    const pv = document.getElementById('amPreview');
    if (!it || !pv) return;
    t = Math.max(0, Math.min(Number(t) || 0, Math.max(0, (Number(it.duration) || 0) - 0.5)));
    let kind = '', others = '';
    if (it.group !== undefined) {
        const sibs = [..._ad.items].map(([i, x]) => [x, i]).filter(([x, i]) => x.group === it.group && i !== k);
        const g = (_ad.list?.items || [])[it.group] || {};
        kind = `≋ Copy ${it.copy + 1} of ${(g.count || sibs.length + 1).toLocaleString()} · from ${amClock(t)}`;
        if (sibs.length) {
            others = `<div class="am-places"><b>The same moment in the other copies</b>${sibs.map(([x, i]) => {
                const there = t + (Number(it.pos) || 0) - (Number(x.pos) || 0);
                return `<div class="am-place"><span title="${esc(x.path)}">${adExt(x.path, x.video)} ${esc(x.name || '')}
                        <small>${esc(amFolder(x))}</small></span>
                    <button class="vx-btn" onclick="adHear('${i}', ${there.toFixed(2)})" title="Hear this copy there">▶ ${amClock(there)}</button></div>`;
            }).join('')}</div>`;
        }
    } else if (it.part !== undefined) {
        const whole = _ad.items.get(it.wholeK) || {};
        kind = `⊂ Inside ${esc(whole.name || '')} from ${amClock(it.at)}`;
        others = `<div class="am-place"><span>Where it sits in the whole</span>
            <button class="vx-btn" onclick="adHearPart('${k}')">▶ ${amClock(it.at)}</button></div>`;
    } else if (it.whole !== undefined) {
        const w = (_ad.list?.items || [])[it.whole] || {};
        kind = `⊂ Holds ${plural((w.parts || []).length, 'recording')} · from ${amClock(t)}`;
        const inside = (w.parts || []).map((p, pi) => [p, pi]).filter(([p]) => t >= p.at - 0.5 && t < p.at + (p.matched || p.duration || 0));
        if (inside.length) others = `<div class="am-places"><b>Playing here</b>${inside.map(([p, pi]) =>
            `<div class="am-place"><span class="ad-dot" style="background:var(--sp${pi % 8})"></span><span>${esc(p.name || '')}</span></div>`).join('')}</div>`;
    }
    pv.innerHTML = `
        <audio class="ms-audio" controls preload="auto" src="${esc(adClip(it.path, t))}"></audio>
        <div class="ms-pv-kind">${kind}</div>
        <div class="ms-pv-name">${esc(it.name || '')}</div>
        <div class="ms-pv-folder" title="${esc(it.path)}">${esc(amFolder(it))}</div>
        <div class="am-place">
            <button class="vx-btn" onclick="adOpenIn('${k}', false, ${t.toFixed(2)})" title="Play it in the player of the main window">▶ Player</button>
            <button class="vx-btn" onclick="adOpenIn('${k}', true, ${t.toFixed(2)})" title="Open the transcript there">📄 Transcript</button>
        </div>
        ${others}
        ${adStats(it)}`;
    const audio = pv.querySelector('audio');
    audio?.addEventListener('play', () => { try { svPlayer()?.pause(); } catch (e) { /* none */ } });
    audio?.addEventListener('error', () => {
        audio.insertAdjacentHTML('afterend', '<div class="ms-err ad-cant">It can\'t be played here — the file may be gone, '
            + 'or lie outside the folders this app may open.</div>');
    }, { once: true });
    try { audio?.play()?.catch(() => { /* a click is needed */ }); } catch (e) { /* not here */ }
    document.querySelectorAll('#amResults .ad-copy.active, #amResults .ad-part.active').forEach(el => el.classList.remove('active'));
    document.querySelector(`#amResults canvas[data-k="${k}"]`)?.closest('.ad-copy')?.classList.add('active');
}

// What the analysis knows of a recording, with its size and format
function adStats(it) {
    const ext = (String(it.path || '').match(/\.([a-z0-9]{1,5})$/i) || ['', '?'])[1].toUpperCase();
    const head = `<span>Size</span><b>${amBytes(it.size)}</b><span>Format</span><b>${esc(ext)}${it.video ? ' (video)' : ''}</b>`;
    const rest = amStats(it);
    return rest ? rest.replace('<div class="am-stats">', '<div class="am-stats">' + head)
        : `<div class="am-stats">${head}</div>`;
}

// A part where it begins in the whole that holds it
function adHearPart(k) {
    const it = _ad.items.get(k);
    if (!it || it.wholeK === undefined) return;
    adHear(it.wholeK, Number(it.at) || 0);
}

function adOpenIn(k, transcript, t) {
    const it = _ad.items.get(k);
    if (!it) return;
    if (typeof amPausePreview === 'function') amPausePreview();
    const at = Math.max(0, Number(t) || 0);
    if (transcript) svGoto(it.filename, at, it.file_dir);
    else svPlay(it.filename, at, it.file_dir);
}

// ── Filters ─────────────────────────────────────────────────────────────────

function adReload() {
    _ad.list = null;
    _ad.open = new Set();
    adLoad(false);
}

function adFilterKey(ev) {
    if (ev.key === 'Enter') { ev.preventDefault(); adSetFilter(ev.target.value); }
}

// The field takes letters of a path — or the request in words again
function adSetFilter(v) {
    v = String(v || '').trim();
    const want = amDupIntent(v);
    const q = want ? want.q : v;
    if (want && want.kind !== _ad.kind) { _ad.kind = want.kind; _am.view = want.kind; }
    if (q === _ad.f.q && !want) return;
    _ad.f.q = q;
    _ad.pair = null;
    if (typeof amRenderViews === 'function') amRenderViews();
    adReload();
}

function adSetLength(sec) {
    _ad.f.minSec = Number(sec) || 0;
    _ad.pair = null;
    adReload();
}

function adSetSort(v) {
    _ad.sort[_ad.kind] = v;
    _ad.list = null;
    _ad.open = new Set();
    adLoad(false);
}

function adPickPair(i) {
    const f = ((_ad.overview || {}).folders || [])[i];
    if (!f) return;
    if (_ad.pair && _ad.pair.id === f.id) { adClearPair(); return; }
    const label = f.within ? adShortFolder(f.a, 2)
        : `${adShortFolder(f.a_rest || f.a, 2)} ⟷ ${adShortFolder(f.b_rest || f.b, 2)}`;
    _ad.pair = { id: f.id, label };
    _ad.list = null;
    _ad.open = new Set();
    adLoad(false).then(() => document.querySelector('#amResults .ad-list')?.scrollIntoView({ block: 'start', behavior: 'smooth' }));
}

function adClearPair() {
    _ad.pair = null;
    _ad.list = null;
    adLoad(false);
}

function adClearFile() {
    _ad.file = '';
    adReload();
}

function adReset() {
    _ad.f = AD_NO_FILTERS();
    _ad.pair = null;
    _ad.file = '';
    adReload();
}

function adAllFolders() {
    _ad.allFolders = true;
    adLoadOverview();
}

// Worked out anew on the server: what the run found since
function adRefresh() {
    _ad.overview = null;
    _ad.list = null;
    _ad.fresh = true;
    adLoad(false);
}

function adMoreFolders() {
    _ad.moreFolders = true;
    adRender();
}

/* A line across a group's copies (or a whole's timeline) where the pointer
 * is, with the time: the same moment in every copy, one above the other. */
function adHover(ev, cv) {
    const holder = cv.closest('.ad-copies, .ad-timeline');
    if (!holder) return;
    let line = holder.querySelector('.ad-cursor');
    if (!line) {
        line = document.createElement('div');
        line.className = 'ad-cursor';
        line.appendChild(document.createElement('span'));
        holder.appendChild(line);
    }
    const r = cv.getBoundingClientRect(), hr = holder.getBoundingClientRect();
    const x = Math.max(0, Math.min(r.width, ev.clientX - r.left));
    line.style.left = `${r.left - hr.left + x}px`;
    line.firstChild.textContent = amClock(x / Math.max(1, r.width) * (Number(cv.dataset.span) || 0));
    line.hidden = false;
}

function adHoverOut(cv) {
    const line = cv.closest('.ad-copies, .ad-timeline')?.querySelector('.ad-cursor');
    if (line) line.hidden = true;
}

function adToggle(i) {
    if (_ad.open.has(i)) _ad.open.delete(i); else _ad.open.add(i);
    adRender();
}
