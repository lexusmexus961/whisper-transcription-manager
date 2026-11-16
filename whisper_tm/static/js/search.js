// ── The search field in the top bar ──
// While typing, its dropdown shows places to go to (with other words for
// them), the recordings where the words were said (one line per
// recording), and where Enter goes. Enter opens Find: the Text tab for one
// to three words, the Find tab for a longer description (svRouteQuery).
// The filters (file name, speaker, folder) are in Find's Text tab.
let searchTimeout;
let searchPanelOpen = false;
let lastSearchTerms = [];
let lastSearchMediaDir = '';
let _spActive = -1;      // the highlighted suggestion (keyboard)

/* Places and window tabs the field can go to. `words` are other words
   people use for them. */
const SP_PLACES = [
    {label: 'Add files', where: '', icon: 'i-upload', go: () => Shell.go('add'), words: 'add upload import download link playlist new'},
    {label: 'Home', where: '', icon: 'i-home', go: () => Shell.go('home'), words: 'home start overview'},
    {label: 'Library', where: '', icon: 'i-folder', go: () => Shell.go('library'), words: 'library folders statistics stats numbers overview'},
    {label: 'Queue', where: '', icon: 'i-queue', go: () => Shell.go('queue'), words: 'queue jobs waiting running transcribing progress pause'},
    {label: 'Find', where: '', icon: 'i-sparkle', go: () => Shell.go('find'), words: 'find search moment describe'},
    {label: 'Text', where: 'Find', icon: 'i-search', go: () => spOpenTab('text'), words: 'text words exact said transcript keyword'},
    {label: 'Images & sounds', where: 'Find', icon: 'i-music', go: () => spOpenTab('media'), words: 'images sounds pictures video frames noises music seen heard'},
    {label: 'Same audio', where: 'Find', icon: 'i-captions', go: () => spOpenTab('audio'), words: 'same audio duplicates copies snippet fingerprint'},
    {label: 'Conversations', where: 'Find', icon: 'i-conversation', go: () => spOpenTab('conversations'), words: 'conversations talks who with whom dialog'},
    {label: 'Files', where: 'Find', icon: 'i-folder', go: () => spOpenTab('files'), words: 'files names metadata file list'},
    {label: 'Deep search', where: 'Find', icon: 'i-search', go: () => spOpenTab('deep'), words: 'deep search thorough situation'},
    {label: 'Ask', where: '', icon: 'i-chat', go: () => Shell.go('ask'), words: 'ask chat assistant question llm ai'},
    {label: 'Analyses', where: '', icon: 'i-layers', go: () => Shell.go('analyses'), words: 'analyses analysis compare summarize together task'},
    {label: 'Voices', where: '', icon: 'i-voices', go: () => Shell.go('voices'), words: 'voices speakers people recognition'},
    {label: 'People', where: 'Voices', icon: 'i-user', go: () => spOpenVoices('people'), words: 'people persons speakers voices names'},
    {label: 'Name voices', where: 'Voices', icon: 'i-user-check', go: () => spOpenVoices('assign'), words: 'name voices assign who is this unknown speakers'},
    {label: 'Voice map', where: 'Voices', icon: 'i-grid', go: () => spOpenVoices('map'), words: 'voice map groups clusters speakers'},
    {label: 'Conversations', where: 'Voices', icon: 'i-conversation', go: () => spOpenVoices('conversations'), words: 'conversations who with whom person talks'},
    {label: 'Voice history', where: 'Voices', icon: 'i-history', go: () => spOpenVoices('history'), words: 'history decisions undo names'},
    {label: 'Voice dataset', where: 'Voices', icon: 'i-scissors', go: () => spOpenVoices('extract'), words: 'voice dataset extract clips training'},
    {label: 'Machine', where: '', icon: 'i-health', go: () => Shell.go('machine'), words: 'machine health gpu vram load cpu models server status'},
    {label: 'Load history', where: 'Machine', icon: 'i-cpu', go: () => { if (typeof openPerfView === 'function') openPerfView(); }, words: 'load history performance cpu ram gpu graph'},
    {label: 'Settings', where: '', icon: 'i-settings', go: () => Shell.go('settings'), words: 'settings options preferences config configuration'},
    {label: 'Help', where: 'Help & setup', icon: 'i-help', go: () => Shell.go('help'), words: 'help how guide keys shortcuts docs'},
    {label: 'Setup guide', where: 'Help & setup', icon: 'i-guide', go: () => { if (typeof openSetupGuide === 'function') openSetupGuide(); }, words: 'setup guide install first steps models'},
];

function spOpenTab(tab) {
    Shell.go('find');
    if (typeof svTab === 'function') svTab(tab);
}

function spOpenVoices(tab) {
    if (typeof vxTab === 'function') vxTab(tab);
    Shell.go('voices');
}

/* Places whose name or other words start with what was typed. */
function spPlaceHits(q) {
    const words = q.toLowerCase().split(/\s+/).filter(Boolean);
    if (!words.length) return [];
    const scored = [];
    for (const p of SP_PLACES) {
        const hay = (p.label + ' ' + p.words).toLowerCase().split(/[\s&]+/).filter(Boolean);
        if (!words.every(w => hay.some(h => h.startsWith(w)))) continue;
        const label = p.label.toLowerCase();
        scored.push([label.startsWith(words[0]) ? 0 : 1, p]);
    }
    return scored.sort((a, b) => a[0] - b[0]).slice(0, 5).map(x => x[1]);
}

function setupGlobalSearch() {
    const input = document.getElementById('globalSearch');
    const panel = document.getElementById('searchPanel');

    const open = () => { panel.classList.add('visible'); searchPanelOpen = true; };
    const close = () => { panel.classList.remove('visible'); searchPanelOpen = false; _spActive = -1; };

    input.addEventListener('focus', () => { open(); spRender(); });

    document.addEventListener('mousedown', (e) => {
        const box = document.getElementById('globalSearchBox');
        if (!box.contains(e.target)) close();
    });

    input.addEventListener('input', () => {
        open();
        _spActive = -1;
        spRender();          // places and the footer at once …
        clearTimeout(searchTimeout);
        searchTimeout = setTimeout(() => runGlobalSearch(), 300);   // … what was said a moment later
    });

    input.addEventListener('keydown', (e) => {
        const opts = [...document.querySelectorAll('#spResults .sp-opt')];
        if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
            if (!opts.length) return;
            e.preventDefault();
            _spActive = e.key === 'ArrowDown'
                ? (_spActive + 1) % opts.length
                : (_spActive <= 0 ? opts.length - 1 : _spActive - 1);
            opts.forEach((o, i) => o.classList.toggle('active', i === _spActive));
            opts[_spActive].scrollIntoView({block: 'nearest'});
            input.setAttribute('aria-activedescendant', opts[_spActive].id || '');
            return;
        }
        if (e.key === 'Enter') {
            e.preventDefault();
            clearTimeout(searchTimeout);
            if (_spActive >= 0 && opts[_spActive]) { opts[_spActive].click(); close(); input.blur(); return; }
            const q = input.value.trim();
            close();
            input.blur();
            if (typeof openSearchView === 'function') openSearchView(q);
            return;
        }
        if (e.key === 'Escape') { close(); input.blur(); }
    });

    // "/" puts the cursor in the field (not while typing elsewhere, and
    // not where a window has its own search under "/": the settings)
    document.addEventListener('keydown', (e) => {
        if (e.key !== '/' || e.ctrlKey || e.metaKey || e.altKey) return;
        const t = e.target;
        if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.tagName === 'SELECT' || t.isContentEditable)) return;
        if (document.getElementById('settingsOverlay')?.classList.contains('visible')) return;
        e.preventDefault();
        input.focus();
        input.select();
    });
}

/* Where Enter goes, and the other places to search the same words. */
function spFooter(q) {
    const foot = document.getElementById('spFoot');
    if (!foot) return;
    if (!q) {
        foot.innerHTML = '<span class="sp-foot-tip">A few words find where they were said ("quotes" = the exact word), a longer description finds a moment, a place name goes there.</span>';
        return;
    }
    const route = typeof svRouteQuery === 'function' ? svRouteQuery(q) : 'text';
    const name = {find: 'Find', text: 'Text', files: 'Files', audio: 'Same audio'}[route] || 'Text';
    const tabs = [['find', 'Find'], ['text', 'Text'], ['media', 'Images & sounds'], ['conversations', 'Conversations'], ['files', 'Files']]
        .filter(([t]) => t !== route);
    foot.innerHTML = `<span class="sp-foot-enter"><kbd>Enter</kbd> opens <b>Find · ${esc(name)}</b></span>`
        + `<span class="sp-foot-alt">Search instead in: `
        + tabs.map(([t, label]) => `<button type="button" class="sp-foot-btn" onmousedown="event.preventDefault()" onclick="spSearchIn('${t}')">${esc(label)}</button>`).join('')
        + `<button type="button" class="sp-foot-btn" onmousedown="event.preventDefault()" onclick="spAsk()">Ask</button></span>`;
}

function spClose() {
    document.getElementById('searchPanel')?.classList.remove('visible');
    searchPanelOpen = false;
    _spActive = -1;
    document.getElementById('globalSearch')?.blur();
}

function spSearchIn(tab) {
    const q = document.getElementById('globalSearch').value.trim();
    spClose();
    if (typeof openSearchView !== 'function') return;
    openSearchView(q);
    if (typeof svState !== 'undefined' && svState.tab !== tab && typeof svTab === 'function') svTab(tab);
}

function spAsk() {
    const q = document.getElementById('globalSearch').value.trim();
    spClose();
    Shell.go('ask');
    const box = document.getElementById('chatInput');
    if (box && q) { box.value = q; box.focus(); }
}

function spGo(i) {
    const p = SP_PLACES[i];
    spClose();
    if (p) p.go();
}

let _spSaid = {q: '', html: ''};   // the last "Said in transcripts" answer

/* Draw the dropdown: places now, what was said when its answer is in. */
function spRender() {
    const q = (document.getElementById('globalSearch')?.value || '').trim();
    const results = document.getElementById('spResults');
    if (!results) return;
    const places = q ? spPlaceHits(q) : SP_PLACES.filter(p => !p.where && p.label !== 'Add files').slice(0, 7);
    let html = '';
    if (places.length) {
        html += `<div class="sp-sec">${q ? 'Go to' : 'Places'}</div>`;
        html += places.map(p => {
            const i = SP_PLACES.indexOf(p);
            return `<button type="button" class="sp-opt sp-place" id="spOpt${i}" role="option" onmousedown="event.preventDefault()" onclick="spGo(${i})">`
                + `<svg class="ui-ico"><use href="#${p.icon}"/></svg><span class="sp-place-name">${esc(p.label)}</span>`
                + (p.where ? `<span class="sp-place-where">${esc(p.where)}</span>` : '') + `</button>`;
        }).join('');
    }
    if (q.length >= 2) html += _spSaid.q === q ? _spSaid.html : '<div class="sp-sec">Said in transcripts</div><div class="sr-loading">Searching …</div>';
    results.innerHTML = html;
    spFooter(q);
}

let _searchSeq = 0;  // Sequence guard: prevents a slower, older response
                     // from overwriting newer results

async function runGlobalSearch() {
    const q = document.getElementById('globalSearch').value.trim();
    if (q.length < 2) {
        // Counts as a newer request too: an answer still in flight for the
        // longer query must not refill the dropdown that was just emptied.
        ++_searchSeq;
        _spSaid = {q: '', html: ''};
        spRender();
        return;
    }
    const seq = ++_searchSeq;
    try {
        const r = await fetch(`/api/search?q=${encodeURIComponent(q)}&limit=60`);
        const data = await r.json();
        if (seq !== _searchSeq) return;  // Response is stale — discard
        lastSearchTerms = data.terms || q.toLowerCase().split(/\s+/);
        if (data.media_dir) lastSearchMediaDir = data.media_dir;
        _spSaid = {q, html: spSaidHtml(data)};
    } catch (e) {
        if (seq !== _searchSeq) return;
        _spSaid = {q, html: `<div class="sp-sec">Said in transcripts</div><div class="sr-empty">Search failed: ${esc(e.message)}</div>`};
    }
    spRender();
}

/* One line per recording: its first hit, the time and the speaker. */
function spSaidHtml(data) {
    const items = data.results || [];
    const terms = data.terms || lastSearchTerms;
    if (!items.length) return '<div class="sp-sec">Said in transcripts</div><div class="sr-empty">Not said in any transcript</div>';
    const first = new Map();
    for (const r of items) if (!first.has(r.filename)) first.set(r.filename, r);
    const shown = [...first.values()].slice(0, 6);
    const nFiles = Array.isArray(data.files) ? (data.files_on_page ?? data.files.length) : first.size;
    const more = Math.max(0, nFiles - shown.length);
    let html = `<div class="sp-sec">Said in transcripts <span class="sp-sec-n">${plural(nFiles, 'recording', 'recordings')}${data.truncated ? '+' : ''}</span></div>`;
    html += shown.map((r, k) => {
        const spName = r.speaker_name || r.speaker || '';
        const base = String(r.filename).split('/').pop();
        return `<button type="button" class="sp-opt sp-said" id="spSaid${k}" role="option" onmousedown="event.preventDefault()" `
            + `onclick="spClose();goToSearchResult('${escJs(r.filename)}', ${Number(r.start) || 0}, '${escJs(r.file_dir || '')}')" title="${esc(r.filename)}">`
            + `<span class="sp-said-file">${esc(base)}</span>`
            + `<span class="sp-said-text">${_hlHit(r, terms)}</span>`
            + `<span class="sp-said-at">${esc(r.start_fmt || '')}${spName ? ' · ' + esc(spName) : ''}</span></button>`;
    }).join('');
    if (more) html += `<button type="button" class="sp-more" onmousedown="event.preventDefault()" onclick="spSearchIn('text')">${plural(more, 'more recording', 'more recordings')} in Text …</button>`;
    return html;
}

function highlightTerms(text, terms) {
    if (!terms.length) return esc(text);
    // ONE combined regex pass on the raw text instead of sequential
    // replaces on the HTML: sequential replaces could let a later term
    // match inside an already-inserted <mark> tag and break the HTML.
    const sorted = [...terms].filter(Boolean).sort((a, b) => b.length - a.length);
    const escaped = sorted.map(t => t.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'));
    if (!escaped.length) return esc(text);
    const re = new RegExp(`(${escaped.join('|')})`, 'gi');
    // Split on the RAW TEXT, then escape each piece individually
    return String(text).split(re).map((part, i) =>
        i % 2 === 1 ? `<mark>${esc(part)}</mark>` : esc(part)
    ).join('');
}

// Highlight matches. Prefers what FTS5 itself has marked: the search knows
// exactly which tokens matched — including forms the index stores without
// diacritics. The earlier regex in the browser failed exactly on that:
// searching for "ware" (index form) while the text has "wäre" — no highlight.
// The DB delivers \x02/\x03 as markers; escape first, then swap, so that
// no HTML from the transcript leaks through.
function _hlHit(r, terms) {
    if (r.highlighted && r.highlighted.indexOf('\x02') !== -1) {
        return esc(r.highlighted)
            .split('\x02').join('<mark>')
            .split('\x03').join('</mark>');
    }
    return highlightTerms(r.text, terms);   // JSON search returns no highlighted
}


// Find a search match in the CURRENTLY loaded file list.
//
// Deliberately exact only. A basename and suffix comparison used to sit here
// as stage 3 and 4 — exactly the stopgaps that docs/PATHS.md removed from
// the backend, and with the same consequence: if the current folder has a
// file with the same name, IT got opened and the player jumped to a
// timestamp from a completely different file. No error, the result just
// looked plausible.
//
// The fallback is no loss: if nothing is found here, /api/navigate_to_file
// takes over — it resolves via identity and switches to the right directory
// if needed.
//
// fileDir is the root that `filename` belongs to (comes as file_dir with
// every match). If it differs from the current directory, the file cannot,
// by definition, be in the current list.
function _findFile(filename, fileDir) {
    // The currently displayed directory is in the path field (directory.js
    // writes it there after every scan).
    const current = (document.getElementById('dirInput')?.value || '').trim();
    if (fileDir && current && fileDir !== current) return null;
    const normalized = String(filename).replace(/\\/g, '/');
    return files.find(x => x.name === filename)
        || files.find(x => x.name.replace(/\\/g, '/') === normalized)
        || null;
}

async function goToSearchResult(filename, startTime, fileDir) {
    document.getElementById('searchPanel').classList.remove('visible');
    searchPanelOpen = false;

    const _d = typeof wtmDebug === 'function' ? (...a) => wtmDebug('Search', ...a) : () => {};
    _d('[Search] goToSearchResult:', filename, 'startTime:', startTime, 'fileDir:', fileDir);

    // ── 1. Already loaded? ────────────────────────────────────────
    // Pass fileDir along: the match can only be in files[] if it belongs
    // to the currently displayed directory.
    let f = _findFile(filename, fileDir);
    if (f) {
        _d('[Search] Found in current files[]:', f.name);
        await selectFile(f);
        setTimeout(() => seekTo(startTime), 500);
        return;
    }
    _d('[Search] Not in current files[]. Calling navigate_to_file…');

    // ── 2. Server does everything: find, switch, scan ─────────
    try {
        if (typeof Toast !== 'undefined') Toast.info('Navigating to file…');

        const debugOn = typeof _wxpDebugEnabled !== 'undefined' && _wxpDebugEnabled;
        const r = await fetch('/api/navigate_to_file', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                filename: filename,
                media_dir: fileDir || lastSearchMediaDir || '',
                debug: debugOn,
            })
        });
        const data = await r.json();

        _d('[Search] navigate_to_file response:', {
            found: data.found,
            directory: data.directory,
            filename_used: data.filename_used,
            file_name: data.file?.name,
            total_files: data.total_files,
            error: data.error,
        });

        if (data.debug_log) {
            data.debug_log.forEach(l => _d('[Server] ' + l));
        }

        if (data.found && data.file) {
            document.getElementById('dirInput').value = data.directory;
            if (typeof updateFavBtn === 'function') updateFavBtn();

            // The server may have switched the directory → reload the
            // paginated list + aggregates. The file itself comes along as a
            // ready-made entry — open it directly, without waiting for the
            // list window.
            if (typeof reloadFileList === 'function') reloadFileList();
            await selectFile(data.file);
            setTimeout(() => seekTo(startTime), 500);
            return;
        }

        const short = filename.replace(/\\/g, '/').split('/').pop();
        if (typeof Toast !== 'undefined') Toast.error('File not found: ' + short);

    } catch(e) {
        _d('[Search] Error:', e.message);
        if (typeof Toast !== 'undefined') Toast.error('Navigation failed: ' + e.message);
    }
}

// ── UI rework: the panel is a pure SEARCH dropdown again ──────────────────
// The earlier second file list (renderSpFileMatches + "▶ into the queue")
// and the quickfilter chip copy (_spQuickCounts/renderSpQuickFilters) have
// been removed: both already existed in the sidebar — with their own
// counting and filter logic, which had become wrong anyway with pagination.
// One filter, one list: on the left. Full text: here, or via Enter in the
// large search view.

// Populate the speaker dropdown server-side (the client list is only a
// window since pagination; /api/search/speakers knows all of them).
async function updateSearchSpeakers() {
    const select = document.getElementById('spSpeakerFilter');
    if (!select) return;
    const current = select.value;
    let speakers = [];
    try {
        const d = await fetch('/api/search/speakers').then(r => r.json());
        speakers = d.speakers || [];
    } catch (e) { return; }   // Dropdown keeps its old state
    let optHtml = '<option value="">All speakers</option>';
    speakers.forEach(s => {
        const name = typeof s === 'string' ? s : (s.name || s.speaker || '');
        if (!name) return;
        optHtml += `<option value="${esc(name)}" ${name === current ? 'selected' : ''}>${esc(name)}</option>`;
    });
    select.innerHTML = optHtml;
}

