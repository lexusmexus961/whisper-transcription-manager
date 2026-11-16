// ── Search view ──
// The Search workspace: hits, files, the map and Ask (the chat, chat.js) in
// the visual system of the voices workspace — a head bar with tabs, one
// query bar, toolbars, cards. The small panel in the
// topbar remains for the quick glance; as soon as someone presses Enter, the
// stage belongs here. Results are loaded page by page (10k+ hits
// written into the DOM at once would freeze the browser), but the numbers in the
// sidebar come from /api/search/facets and therefore always relate
// to ALL hits — not just what is currently loaded.

const SV_PAGE = 200;          // hits per load-more step
const SV_BUCKETS = 40;        // resolution of the time heatmap per file

const svState = {
    query: '',
    routed: '',      // the last top-bar query sent to a tab (svRouteQuery)
    file: '',        // free-text file filter from the input field
    pinned: '',      // file pinned via click in the sidebar
    speaker: '',
    offset: 0,
    total: 0,
    fileCount: null,   // from the facets: files across ALL hits
    facets: null,      // last facets response — data basis of the hit map
    loading: false,
    done: false,
    terms: [],
    files: [],       // hits in the filename/title of the running search
    seq: 0,
    // ── Tabs and view ────────────────────────────────────────────────
    mode: 'both',   // what is searched with: both | text | meaning
    plays: null,      // what is currently playing in the background
    filled: false,          // has the view already been filled once?
    tab: 'find',       // find | text | files | map | chat | deep | analysis | media | audio | conversations
    view: 'hits',// in the text tab: 'hits' (one line per hit)
                       // or 'recordings' (map per file, with hit band)
    hits: [],       // the loaded hits — data basis of the reading column
    selection: -1,       // index in `hits`; -1 = nothing selected
    copySel: '',       // id of the copy line shown in the reading column ('' = none)
    readSeq: 0,        // discards stale context fetches of the reading column
    meaningPending: false,  // the meaning list's request is still out
};

function setupSearchView() {
    const ov = document.getElementById('svOverlay');
    if (!ov) return;

    svMountSplitter();

    document.getElementById('svQuery').addEventListener('keydown', (e) => {
        if (e.key === 'Enter') { e.preventDefault(); svRun(); }
    });
    document.getElementById('svFileFilter').addEventListener('keydown', (e) => {
        if (e.key === 'Enter') { e.preventDefault(); svRun(); }
    });
    document.getElementById('svSpeaker').addEventListener('change', svRun);
    document.getElementById('svPerson')?.addEventListener('change', () => {
        // The "only their sentences" checkbox only makes sense with a
        // person selected — without one it would be a switch with no effect.
        svShowPersonBox();
        svRun();
    });
    document.getElementById('svPersonSegments')?.addEventListener('change', svRun);

    // Advanced filters: every change re-triggers the search (Enter on
    // number fields too — there, change only fires on blur).
    for (const id of ['svDateFrom', 'svDateTo', 'svDurationMin', 'svDurationMax',
                      'svWordMin', 'svWordMax', 'svTempoMin', 'svTempoMax',
                      'svSort', 'svDirection', 'svTranscribedOnly']) {
        const el = document.getElementById(id);
        if (!el) continue;
        el.addEventListener('change', svRun);
        el.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') { e.preventDefault(); svRun(); }
        });
    }
    document.getElementById('svSort')?.addEventListener('change', svSortSync);
    svSortSync();

    // Escape closes — but only when the search view is really on top.
    // The settings its tabs link to lie above it and close on their own
    // Escape; whether they were open is noted in the capture phase, before
    // their handler closes them.
    // Arrow keys page through the hits: a dense list is only worth
    // half as much if you have to click every line individually.
    const above = () => !!document.querySelector('.settings-overlay.visible');
    let onTop = true;
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') onTop = !above();
    }, true);
    document.addEventListener('keydown', (e) => {
        if (!ov.classList.contains('visible')) return;
        if (e.key === 'Escape') { if (onTop) closeSearchView(); return; }
        if (above() || svState.tab !== 'text') return;
        if (e.key !== 'ArrowDown' && e.key !== 'ArrowUp') return;
        // In the search field, arrow keys belong to the text. Exception: if
        // something is already selected, the user wants to keep paging without
        // clicking away first — so it only kicks in there with Alt held.
        const inField = /^(INPUT|SELECT|TEXTAREA)$/.test(document.activeElement?.tagName || '');
        if (inField && !e.altKey) return;
        e.preventDefault();
        svMoveSelection(e.key === 'ArrowDown' ? 1 : -1);
    });

    // Restore the remembered view (deliberately NOT the tab: where a new
    // query goes is decided by the query itself — svRouteQuery).
    try {
        const a = localStorage.getItem('svView');
        if (a === 'recordings' || a === 'hits') svState.view = a;
        const m = localStorage.getItem('svMode');
        if (['both', 'text', 'meaning'].includes(m)) svState.mode = m;
    } catch (e) { /* doesn't matter */ }
    svModeSync();
    svViewButtonSync();
    svClearRead();

    // The map computes in pixels — if the window changes, it has to redraw.
    let _kt = null;
    window.addEventListener('resize', () => {
        if (svState.tab !== 'map') return;
        clearTimeout(_kt);
        _kt = setTimeout(svMapDraw, 150);
    });

    // Load more on scroll: trigger early enough that the user never
    // sees the gap.
    // Both lists page: the text tab's hits (#svResults) and the file tab's
    // rows (#svFileList). Each listener only acts while its own tab is the
    // active one, so a scroll in the hidden list can't load the other's page.
    for (const [id, tab] of [['svResults', 'text'], ['svFileList', 'files']]) {
        const box = document.getElementById(id);
        if (!box) continue;
        box.addEventListener('scroll', () => {
            if (svState.tab !== tab) return;
            if (box.scrollTop + box.clientHeight >= box.scrollHeight - 400) svLoadMore();
        });
    }
}

function openSearchView(query) {
    const ov = document.getElementById('svOverlay');
    if (!ov) return;

    // Take over the filter from the small panel, so the switch is seamless
    const q = (query !== undefined && query !== null)
        ? query
        : (document.getElementById('globalSearch')?.value || '');
    document.getElementById('svQuery').value = q;
    document.getElementById('svFileFilter').value =
        document.getElementById('spFileFilter')?.value || '';
    /* Also reset the pinned file.
     *
     * `pinned` lives in the module and once survived closing the
     * view. All other filters are freshly taken over here from the small
     * panel — the pin was the only silent survivor. Anyone who closed the
     * search, reopened it later and typed something new was unknowingly
     * still searching only within a file they had clicked on half an hour
     * earlier. The only visible sign of it was a "(1)" in the
     * sidebar.
     */
    const previouslyPinned = svState.pinned;
    svState.pinned = '';

    // Set the selection first, then load the list — svSyncSpeakers is
    // asynchronous and receives the existing selection.
    svSetSpeaker(document.getElementById('spSpeakerFilter')?.value || '');
    svSyncSpeakers();
    svSyncPeople();
    svShowPersonBox();

    const panel = document.getElementById('searchPanel');
    if (panel) { panel.classList.remove('visible'); searchPanelOpen = false; }

    // A new query from the top bar goes where it belongs (svRouteQuery) —
    // the search below runs once, in that tab. The same query again
    // continues where it was left.
    const routed = q && q !== svState.routed ? svRouteQuery(q) : null;
    if (routed) {
        svState.routed = q;
        if (routed !== svState.tab) svShowTab(routed);
        if (routed === 'find' && typeof fdIncoming === 'function') fdIncoming(q);
        if (routed === 'audio' && typeof amIncoming === 'function') amIncoming(q);
    }

    ov.classList.add('visible');
    svTabChrome(svState.tab);
    document.getElementById('svQuery').focus();
    document.getElementById('svQuery').select();

    /* Continue where you left off.
     *
     * Anyone who closes the search and reopens it usually doesn't intend
     * to start over — they just wanted to briefly look at something else.
     * A new search is therefore only run if the query has changed or
     * there wasn't one yet.
     *
     * `svState.pinned` is reset above (the silent survivor described
     * there). When continuing, that's exactly backwards: the list being
     * restored IS the filtered one. That's why the pin is
     * restored here — and shown in the status line, so it doesn't
     * become invisible again.
     */
    const onward = svState.filled && q === svState.query && !!_svStand[svState.tab];
    if (onward) {
        svState.pinned = previouslyPinned;
        if (!svLoadState(svState.tab)) svRun();
        else svShowPin();
    } else {
        svRun();
    }
    svState.filled = true;
    // The deep-search tab asks the server only while it is in view — shown
    // again, it picks up where it paused (deepsearch.js)
    if (svState.tab === 'deep' && typeof dsOpen === 'function') dsOpen();
}

// Make an active file filter visible. This used to be the
// bug nobody saw; when continuing it comes back and therefore needs
// a sign.
function svShowPin() {
    const el = document.getElementById('svPinned');
    if (!el) return;
    if (svState.pinned) {
        el.style.display = '';
        el.title = svState.pinned;
        el.innerHTML = `only in <b>${esc(svShortName(svState.pinned))}</b>`
            + ' <span class="sv-detach" role="button" tabindex="0" data-keyclick title="Search all files again" onclick="svPinFile(svState.pinned)">✕</span>';
    } else {
        el.style.display = 'none';
    }
}

function closeSearchView() {
    // Save state BEFORE the view disappears — otherwise on
    // reopening there'd be nothing to continue from.
    svSaveState();
    document.getElementById('svOverlay')?.classList.remove('visible');
    svState.seq++;   // discard in-flight responses
    // Closed, the deep search stops asking (it runs on in the LLM queue)
    if (typeof dsLeave === 'function') dsLeave();
    // … and Find's preview clip stops (it plays inside the search)
    if (typeof fdLeave === 'function') fdLeave();
    if (typeof msLeave === 'function') msLeave();
    if (typeof amLeave === 'function') amLeave();
    if (typeof cvLeave === 'function') cvLeave();
    // Playback deliberately KEEPS running: the player belongs to the
    // main window, not the search. Anyone who wants to stop it uses the ✕ button
    // in the playback bar.
}

// Fill the speaker dropdown. Deliberately NOT mirrored from the small panel:
// that one only lists the speakers of the most recently scanned folder, but the search covers
// the entire collection. Without a scan the list would even be empty — and
// clicking a speaker row in the sidebar would go nowhere.
let _svSpeakersLoaded = false;

async function svSyncSpeakers() {
    const dst = document.getElementById('svSpeaker');
    if (!dst || _svSpeakersLoaded) return;
    try {
        const r = await fetch('/api/search/speakers');
        const data = await r.json();
        const cur = dst.value;
        dst.innerHTML = '<option value="">All speakers</option>'
            + (data.speakers || []).map(s =>
                `<option value="${esc(s)}">${esc(s)}</option>`).join('');
        dst.value = cur;
        _svSpeakersLoaded = true;
    } catch (e) { /* filter stays empty — search still works */ }
}

// Set the selection, even if the option is (still) missing: the sidebar knows
// speakers from the facets that don't need to appear in the dropdown.
function svSetSpeaker(speaker) {
    const sel = document.getElementById('svSpeaker');
    if (speaker && !Array.from(sel.options).some(o => o.value === speaker)) {
        sel.insertAdjacentHTML('beforeend',
            `<option value="${esc(speaker)}">${esc(speaker)}</option>`);
    }
    sel.value = speaker;
}

// ── Tabs ────────────────────────────────────────────────────────────────
//
// Four tasks that once shared one screen:
// searching the spoken text, browsing files, getting a spatial
// overview of the hits, and (in preparation) talking to a model about it. Each
// gets the full area, instead of sharing space with the others.
const SV_TABS = ['find', 'text', 'files', 'map', 'chat', 'analysis', 'deep', 'media', 'audio', 'conversations'];

/* State per tab.
 *
 * The text tab and the file tab query the same interface with
 * different parameters, but share `svState`. Anyone switching back and forth
 * switched back and forth therefore saw the file tab's numbers in the text tab:
 * "N hits in N files" (the whole library) above a hit list with far fewer
 * hits — and the block with the filename hits was gone,
 * because the file search had emptied `svState.files`.
 *
 * Instead of searching again when switching back (costs time and throws away selection and
 * scroll position), each tab remembers its state. It's only restored
 * when the query is still the same — otherwise a new search is run.
 */
const _svStand = {};

/* `t`: the tab to remember — the shown one by default. The meaning list
 * passes 'text' when its answer lands while another tab is shown: it can
 * take seconds, and the user may have moved on (to the map, Ask, …) in
 * the meantime — what the switch remembered then is the "running …" line,
 * and coming back would restore that over the finished list. */
function svSaveState(t = svState.tab) {
    if (t !== 'text' && t !== 'files') return;
    // Its list and parameters follow the tab shown: the files tab has a list
    // and a query of its own, so the hit list can't be read from there
    if (t !== svState.tab && svState.tab === 'files') return;
    const box = document.getElementById(t === 'files' ? 'svFileList' : 'svResults');
    if (!box) return;
    _svStand[t] = {
        key: svParams(),
        listing: box.innerHTML,
        scrollTop: box.scrollTop,
        read: document.getElementById('svRead')?.innerHTML || '',
        total: svState.total, fileCount: svState.fileCount,
        offset: svState.offset, done: svState.done, terms: svState.terms,
        hits: svState.hits, selection: svState.selection,
        files: svState.files, facets: svState.facets,
        view: svState.view,
        // A meaning list still on its way: there is nothing to restore yet
        pending: t === 'text' && svState.meaningPending,
        // The meaning list's own state (its copies, its weaker matches) —
        // the files tab's search clears it, coming back needs it again
        quick: t === 'text' ? _svQuick : null,
        copySel: svState.copySel,
    };
}

function svLoadState(name) {
    const s = _svStand[name];
    if (!s) return false;
    if (s.pending) return false;
    // `svParams()` reads the current tab — for the comparison it must
    // already have switched, so only the remembered key is here
    // compared against the now-valid one.
    if (s.key !== svParams()) return false;
    const box = svListBox();
    box.innerHTML = s.listing;
    // After setting innerHTML, otherwise the box has no height yet.
    requestAnimationFrame(() => { box.scrollTop = s.scrollTop || 0; });
    const read = document.getElementById('svRead');
    if (read && s.read) read.innerHTML = s.read;
    Object.assign(svState, {
        total: s.total, fileCount: s.fileCount, offset: s.offset,
        done: s.done, terms: s.terms, hits: s.hits,
        selection: s.selection, files: s.files, facets: s.facets,
        view: s.view, loading: false, copySel: s.copySel || '',
    });
    if (name === 'text') _svQuick = s.quick || null;
    svSetStatus(svState.total, 0);
    svViewButtonSync();
    // The meaning block may have finished while another tab was shown —
    // the restored listing holds it as it was then ("running …")
    if (name === 'text' && String(s.listing).includes('RAG search running')
            && typeof _svMeaningData !== 'undefined' && _svMeaningData && !_svMeaningData.running) {
        svRenderMeaning();
    }
    return true;
}

/* What a tab shows of the bars above it: the view switch (hits · by
 * recording · map) belongs to the text; the query bar and its filters to
 * everything but Find, Ask, Analyses and Deep search, which have fields of
 * their own. */
function svTabChrome(name) {
    const viewBtn = document.getElementById('svViewBtn');
    if (viewBtn) viewBtn.style.display = (name === 'text' || name === 'map') ? '' : 'none';
    // Same audio takes a snippet, not words: no query bar either — nor
    // Conversations, whose filters are people, lengths and kinds
    const asking = name === 'chat' || name === 'analysis' || name === 'deep' || name === 'find' || name === 'audio'
        || name === 'conversations';
    // The search's own count ("N recordings affected") says nothing about
    // a conversation
    const count = document.getElementById('svTabCount');
    if (count) count.style.visibility = (asking || name === 'files' || name === 'media') ? 'hidden' : '';
    for (const id of ['svQueryBar', 'svFilterBar']) {
        const el = document.getElementById(id);
        if (el) el.style.display = asking ? 'none' : '';
    }
    const adv = document.getElementById('svAdvBar');
    if (adv && asking) adv.style.display = 'none';
    // Images & sounds takes only the query: the filters, the modes, Ask and
    // the deep search belong to the text. Words / meaning say how the SPOKEN
    // text is searched — the files tab searches names, and the map shows
    // where the word hits are (it is drawn from the full-text index's
    // counts); Ask and the deep search have nothing to do with a file name.
    const media = name === 'media', files = name === 'files', map = name === 'map';
    const filters = document.getElementById('svFilterBar');
    if (filters && media) filters.style.display = 'none';
    for (const [sel, hide] of [['#svModeSeg', media || files || map],
                               ['#svQueryBar .sv-ask-btn', media || files],
                               ['#svDeepBtn', media || files],
                               // the query IS the file name there
                               ['#svFileFilter', files],
                               ['#svCountHits', files]]) {
        const el = document.querySelector(sel);
        if (el) el.style.display = hide ? 'none' : '';
    }
    const input = document.getElementById('svQuery');
    if (input) {
        input.dataset.textPlaceholder ??= input.placeholder;
        input.placeholder = media ? 'Describe what is seen or heard …  ("police at night", "gunshots")'
            : files ? 'A file or folder name …  (* = anything, "WA0" · "2024-05" · "vacation/*.mp4")'
            : input.dataset.textPlaceholder;
    }
    svViewButtonSync();
}

/* Where a query typed in the top bar goes. Enter there used to start
 * Find — a run of a minute with the language model — for every word, even
 * for a plain word search:
 *   the duplicates           → Same audio ("find duplicate audio files",
 *                                         "duplicates in Archive" — audiodups.js)
 *   a file or folder name   → Files   (an extension, a path, a wildcard,
 *                                      "VID-2024…", "PTT-…")
 *   quoted words, or ≤ 3     → Text    (the words as they were said)
 *   a description            → Find
 */
function svRouteQuery(q) {
    const s = (q || '').trim();
    if (!s) return null;
    if (typeof amDupIntent === 'function' && amDupIntent(s)) return 'audio';
    // (no "?": a question is a description, not a wildcard)
    if (/[\\/*]/.test(s) || /\.[a-z0-9]{2,4}$/i.test(s) || /^(VID|IMG|AUD|PTT|STK)-\d/i.test(s)) return 'files';
    if (/["„“”»«]/.test(s)) return 'text';
    return s.split(/\s+/).filter(Boolean).length <= 3 ? 'text' : 'find';
}

// Switch the panes and bars to a tab — no search started (svTab starts it)
function svShowTab(name) {
    svSaveState();
    const leaving = svState.tab;
    svState.tab = name;
    // The map is a view of the text's hits — the text tab stays lit
    const lit = name === 'map' ? 'text' : name;
    for (const t of SV_TABS) {
        const btn = document.getElementById('svTabBtn' + t[0].toUpperCase() + t.slice(1));
        btn?.classList.toggle('active', t === lit);
        if (btn) btn.ariaSelected = String(t === lit);
        document.getElementById('svPane' + t[0].toUpperCase() + t.slice(1))
            ?.classList.toggle('active', t === name);
    }
    // Find's preview and polling stop when its tab is left
    if (leaving === 'find' && name !== 'find' && typeof fdLeave === 'function') fdLeave();
    if (leaving === 'media' && name !== 'media' && typeof msLeave === 'function') msLeave();
    if (leaving === 'audio' && name !== 'audio' && typeof amLeave === 'function') amLeave();
    if (leaving === 'conversations' && name !== 'conversations' && typeof cvLeave === 'function') cvLeave();
    svTabChrome(name);
    svMode(name);
    document.getElementById('svAdvToggle')?.classList.remove('active');
    try { localStorage.setItem('svTab', name); } catch (e) { /* doesn't matter */ }
    return leaving;
}

/* One window, three places: Find with its tabs, Ask and Analyses on their
   own (their own title, no tab bar). */
const SV_MODES = {chat: ['ask', 'Ask', 'i-chat'], analysis: ['analyses', 'Analyses', 'i-layers']};
function svMode(tab) {
    const [mode, title, icon] = SV_MODES[tab] || ['find', 'Find', 'i-sparkle'];
    const ov = document.getElementById('svOverlay');
    if (ov) ov.dataset.mode = mode;
    const t = document.getElementById('svTitleText');
    if (t) t.textContent = title;
    document.querySelector('#svTitle use')?.setAttribute('href', '#' + icon);
    const tabs = document.getElementById('svTabs');
    if (tabs) tabs.hidden = mode !== 'find';
}

function svTab(name) {
    if (!SV_TABS.includes(name)) return;
    if (name === svState.tab) return;
    const leaving = svShowTab(name);

    // Tabs only load once you look at them. The map builds a tree from 500
    // files and computes a treemap — that doesn't belong in every search.
    if (name === 'text' || name === 'files') {
        if (!svLoadState(name)) svRun();
    }
    if (name === 'map') svMapFillTab();
    if (name === 'chat' && typeof chatOpen === 'function') chatOpen();
    if (name === 'analysis' && typeof anOpen === 'function') anOpen();
    // The deep search asks the server only while its tab is in view
    if (leaving === 'deep' && typeof dsLeave === 'function') dsLeave();
    if (name === 'deep' && typeof dsOpen === 'function') dsOpen();
    if (name === 'media' && typeof msOpen === 'function') msOpen();
    if (name === 'audio' && typeof amOpen === 'function') amOpen();
    if (name === 'conversations' && typeof cvOpen === 'function') cvOpen();
    if (name === 'find' && typeof fdOpen === 'function') fdOpen();
}

// Where the ACTIVE tab's hit list lives. The text tab has its
// own, the file tab a second — if both shared `#svResults`,
// every tab switch would overwrite the other's list.
function svListBox() {
    return document.getElementById(
        svState.tab === 'files' ? 'svFileList' : 'svResults');
}

/* What is searched — and why that's a real decision.
 *
 * The two paths do NOT search the same thing:
 *
 *   literal      full-text index over the whole library (every transcript)
 *   semantic     vector index over the EMBEDDED part — on a large library
 *                that can be well under 1% of the transcripts for a long time
 *
 * "Nothing found" in the second case therefore usually means "not
 * embedded yet" and not "never said". The toggle makes that
 * visible instead of silently blending the two paths.
 */
function svSetMode(m) {
    if (!['both', 'text', 'meaning'].includes(m)) return;
    svState.mode = m;
    try { localStorage.setItem('svMode', m); } catch (e) { /* doesn't matter */ }
    svModeSync();
    svRun();
}

// The segmented switch shows the mode in force.
function svModeSync() {
    document.querySelectorAll('#svModeSeg [data-mode]').forEach(b =>
        b.classList.toggle('active', b.dataset.mode === svState.mode));
}

// ── Run search ───────────────────────────────────────────────────────

function svRun() {
    // Images & sounds answers the query field with a search of its own
    if (svState.tab === 'media') { if (typeof msRun === 'function') msRun(); return; }
    if (svState.tab === 'audio') { if (typeof amOpen === 'function') amOpen(); return; }
    if (svState.tab === 'conversations') { if (typeof cvOpen === 'function') cvOpen(); return; }
    if (svState.tab === 'find') { if (typeof fdOpen === 'function') fdOpen(); return; }
    svState.query = document.getElementById('svQuery').value.trim();
    svState.file = document.getElementById('svFileFilter').value.trim();
    svState.speaker = document.getElementById('svSpeaker').value;
    svState.person = document.getElementById('svPerson')?.value || '';
    svState.personSegments =
        !!document.getElementById('svPersonSegments')?.checked;
    svState.offset = 0;
    svState.done = false;
    svState.total = 0;
    svState.fileCount = null;   // until facets arrive: show page count
    svState.loading = false;    // in-flight fetch is now stale
    // Discard facets of the PREVIOUS search — otherwise the map builds its
    // tree from foreign files while the header names the new search.
    svState.facets = null;
    svState.files = [];
    svState.hits = [];
    svState.selection = -1;
    svState.copySel = '';
    svState.meaningPending = false;
    _svQuick = null;        // the meaning list belongs to the previous search
    _svFilesOpen = false;   // new search → file block collapsed again
    svClearRead();

    const res = svListBox();
    if (svState.query.length < 2 && !svHasFilter()) {
        // The extra blocks disappear too. Without this, clearing the
        // search field left the semantic block of the PREVIOUS question standing — under
        // the prompt to finally enter something. The sequence number must
        // count toward that: otherwise a still-in-flight
        // answer to the old question would insert its block right back.
        ++svState.seq;
        _svMeaningData = null;
        svState.files = [];
        res.innerHTML = '<div class="sv-empty">Enter at least 2 characters '
            + '— or filter under "⚙ Filter &amp; Sort" by date, duration, '
            + 'word count or pace (also works without a search term).</div>';
        svSetStatus(0, 0);
        return;
    }

    res.innerHTML = '<div class="sv-empty">Searching…</div>';
    const seq = ++svState.seq;
    // Pure semantic mode doesn't even query the full-text search — the
    // vector hits ARE the list here, not an extra alongside it.
    if (svState.tab === 'text' && svState.mode === 'meaning') {
        svMeaningOnly(seq);
        return;
    }
    // The facets deliver hit count per file, runtime, and hit band —
    // the data basis for the recordings view AND for the hit map.
    // They're therefore still fetched, even though the sidebar is gone.
    // Facets (bands, honest file count, map data basis) belong
    // to the text search — the file tab has no text hits.
    if (svState.tab !== 'files' && svState.query.length >= 2) svLoadFacets(seq);
    svLoadMore(true, seq);
}

// ── Draggable splitter ────────────────────────────────────────────────
//
// The file list used to have a fixed 340px width. For deeply nested paths
// (`media/archive/channels/…`) that's too little; with a flat
// layout it wastes space the hits could use better. How
// wide it should be, only the person looking at it knows — so they set it.
const SV_SIDE_STD = 620;      // starting width of the hit list
const SV_SIDE_MIN = 280;      // below this nothing is left of the snippet
const SV_SIDE_MAX_SHARE = 0.6;   // never more than 60% of window width

function svSetSideWidth(px, remember) {
    const side = document.querySelector('#svOverlay .sv-list');
    if (!side) return;
    const max = Math.max(SV_SIDE_MIN, window.innerWidth * SV_SIDE_MAX_SHARE);
    const b = Math.round(Math.min(Math.max(px, SV_SIDE_MIN), max));
    side.style.width = b + 'px';
    if (remember) {
        try { localStorage.setItem('svSideWidth', String(b)); } catch (e) { /* doesn't matter */ }
    }
}

// After dragging, the browser still fires a `click` — on the
// common ancestor of the press and release points. If you drag the splitter
// far to the left, that's `.sv-overlay`, and its click handler closes
// the search: the hit list was gone just because someone wanted to
// narrow the file list. Reproduced in the browser before this was here.
//
// The guard lives for exactly one event cycle: the click follows immediately
// after the release. If it stayed around, it would swallow the next real
// click by the user.
function svSwallowClickAfterDrag() {
    const swallow = (ev) => { ev.stopPropagation(); ev.preventDefault(); };
    document.addEventListener('click', swallow, true);
    setTimeout(() => document.removeEventListener('click', swallow, true), 0);
}

function svMountSplitter() {
    const handle = document.getElementById('svSplitter');
    const side = document.querySelector('#svOverlay .sv-list');
    if (!handle || !side) return;

    let remembered = null;
    try { remembered = parseInt(localStorage.getItem('svSideWidth'), 10); } catch (e) { /* doesn't matter */ }
    if (remembered > 0) svSetSideWidth(remembered, false);

    // Pointer events instead of mouse*: so it also drags with pen and finger.
    //
    // The movement is tied to the DOCUMENT, not the handle. The obvious approach
    // via `handle.setPointerCapture(e.pointerId)` and `pointermove` on the handle
    // looked right but did nothing: after `pointerdown`, the handle received
    // not a single `pointermove` afterward (measured in the browser — the
    // pointer leaves the 5px-wide strip immediately, and the capture
    // didn't bring the events back here). On the document this
    // problem doesn't exist: every movement passes by there, no matter what it's over.
    handle.addEventListener('pointerdown', (e) => {
        e.preventDefault();
        handle.classList.add('drags');
        document.body.classList.add('sv-drag');
        const startX = e.clientX;
        const startB = side.getBoundingClientRect().width;

        let dragged = false;
        const move = (ev) => {
            dragged = true;
            svSetSideWidth(startB + (ev.clientX - startX), false);
        };
        const release = () => {
            document.removeEventListener('pointermove', move);
            document.removeEventListener('pointerup', release);
            document.removeEventListener('pointercancel', release);
            handle.classList.remove('drags');
            document.body.classList.remove('sv-drag');
            // Only write at the end, not on every pixel.
            svSetSideWidth(side.getBoundingClientRect().width, true);
            if (dragged) svSwallowClickAfterDrag();
        };
        document.addEventListener('pointermove', move);
        document.addEventListener('pointerup', release);
        document.addEventListener('pointercancel', release);
    });

    // Double-click restores the starting width — the way back
    // if you accidentally shrank the list.
    handle.addEventListener('dblclick', () => svSetSideWidth(SV_SIDE_STD, true));

    // Keyboard: the splitter is focusable, so it must also work without a mouse.
    handle.addEventListener('keydown', (e) => {
        const step = e.shiftKey ? 40 : 10;
        const now = side.getBoundingClientRect().width;
        if (e.key === 'ArrowLeft') { svSetSideWidth(now - step, true); e.preventDefault(); }
        else if (e.key === 'ArrowRight') { svSetSideWidth(now + step, true); e.preventDefault(); }
        else if (e.key === 'Home') { svSetSideWidth(SV_SIDE_STD, true); e.preventDefault(); }
    });

    // A smaller window must not leave the list wider than allowed.
    // While the search view is closed the overlay is display:none — the
    // list measures 0 then, and clamping that would shrink it to the minimum
    // (a resize with the view closed lost the chosen width).
    window.addEventListener('resize', () => {
        const width = side.getBoundingClientRect().width;
        if (width > 0) svSetSideWidth(width, false);
    });
}

// Is any metadata filter, file filter, or sort active?
function svHasFilter() {
    return svAdvParams() !== '' || !!(svState.pinned || svState.file)
        || !!svState.speaker;
}

function svParams() {
    /* A file pinned via click beats the free-text filter — otherwise
     * the sidebar selection and the input field would contradict each other.
     *
     * And in the file tab, the SEARCH TERM applies to the filename when
     * the filename field is empty.
     *
     * Before this, a file-name prefix typed into the query field showed the
     * whole library below it (N hits in N files). The tab discarded the query
     * (correctly: otherwise the server does a full-text search), but left it
     * standing in the field. You see your search term and a result that has
     * nothing to do with it; that there's a second, dedicated field for it
     * has to be found first. `file=<prefix>` finds the files, while
     * `q=<prefix>` in file mode returns none.
     *
     * The filename field still wins when filled — anyone typing
     * something there means it more precisely than the text above. */
    const file = svState.pinned
        || (svState.tab === 'files' ? svState.query : svState.file);
    const q = svState.tab === 'files' ? '' : svState.query;
    let p = `q=${encodeURIComponent(q)}`;
    if (file) p += `&file=${encodeURIComponent(file)}`;
    if (svState.speaker) p += `&speaker=${encodeURIComponent(svState.speaker)}`;
    p += svPersonParams();
    p += svAdvParams();
    return p;
}

/* The person filter — the same string for both hit list AND facets,
 * otherwise the sidebar would show different numbers than the list. */
function svPersonParams() {
    if (!svState.person) return '';
    let p = `&person=${encodeURIComponent(svState.person)}`;
    if (svState.personSegments) p += '&person_segments=1';
    return p;
}

/* Only show the "only their sentences" checkbox when a person is selected. */
function svShowPersonBox() {
    const box = document.getElementById('svPersonSegmentsBox');
    const person = document.getElementById('svPerson')?.value || '';
    if (box) box.style.display = person ? '' : 'none';
    if (!person) {
        const cb = document.getElementById('svPersonSegments');
        if (cb) cb.checked = false;
    }
}

/* The named people from speaker recognition.
 *
 * Deliberately a DIFFERENT list than `svSyncSpeakers()`: that one holds the
 * diarization identifiers (SPEAKER_00), which only apply within a single
 * recording. This one holds people who have been recognized across
 * the entire collection. */
let _svPeopleLoaded = false;

async function svSyncPeople() {
    const dst = document.getElementById('svPerson');
    if (!dst || _svPeopleLoaded) return;
    try {
        const r = await fetch('/api/speaker/people');
        const data = await r.json();
        const cur = dst.value;
        const people = (data.people || data.people || [])
            .map(p => typeof p === 'string' ? {name: p} : p)
            .filter(p => p && p.name);
        dst.innerHTML = '<option value="">All people</option>'
            + people.map(p => {
                const n = p.voices ?? p.voices ?? p.count;
                return `<option value="${esc(p.name)}">${esc(p.name)}`
                    + (n ? ` (${n})` : '') + '</option>';
            }).join('');
        dst.value = cur;
        _svPeopleLoaded = true;
    } catch (e) { /* filter stays empty — search still works */ }
}

// ── Advanced filters & sorting ────────────────────────────────────────
// Date (recording date), duration, word count, processing pace — plus
// sorting across the same metadata. The values go 1:1 as query
// parameters to /api/search and /api/search/facets, so the hit list and
// sidebar numbers match up.
function svAdvParams() {
    const val = (id) => (document.getElementById(id)?.value || '').trim();
    let p = '';
    const map = [
        ['svDateFrom', 'date_from', (v) => v],
        ['svDateTo', 'date_to', (v) => v],
        ['svDurationMin', 'duration_min', (v) => String(parseFloat(v) * 60)],
        ['svDurationMax', 'duration_max', (v) => String(parseFloat(v) * 60)],
        ['svWordMin', 'word_count_min', (v) => v],
        ['svWordMax', 'word_count_max', (v) => v],
        ['svTempoMin', 'tempo_min', (v) => v],
        ['svTempoMax', 'tempo_max', (v) => v],
    ];
    for (const [id, key, convert] of map) {
        const v = val(id);
        if (v !== '') p += `&${key}=${encodeURIComponent(convert(v))}`;
    }
    if (document.getElementById('svTranscribedOnly')?.checked) {
        p += '&transcribed_only=1';
    }
    // The file tab IS file mode. Previously it hung on the class of
    // a button in the filter bar — two truths about the same
    // state, one of which was invisible.
    if (svState.tab === 'files') p += '&mode=files';
    const sort = val('svSort');
    if (sort) p += `&sort=${encodeURIComponent(sort)}`
        + `&direction=${encodeURIComponent(val('svDirection') || 'desc')}`;
    return p;
}

// 📁 toggle: list the entire library (even without a search term and
// without a transcript). Clears the search field so file mode applies.
//
// DIVISION OF ROLES with the sidebar list (UI rework): the sidebar is the
// WORK LIST (status tabs, selection, enqueue/delete, paginated via
// /api/files). File mode here is the ANALYSIS BROWSER — it has the
// advanced value filters (duration/word count/date/pace/speaker), which the
// sidebar deliberately doesn't get. No actions, only viewing/jumping.
// The toggle has become a tab. The function remains because the
// "📁 All files" button in the filter bar still refers to it — it now leads
// there instead of setting a second, invisible mode.
function svToggleFileMode() {
    if (svState.tab === 'files') { svTab('text'); return; }
    document.getElementById('svQuery').value = '';
    // Sensible starting point: descending by duration — the long recordings
    // first, that's the most common "all files" intent.
    const sort = document.getElementById('svSort');
    if (sort && !sort.value) sort.value = 'duration';
    svSortSync();
    svTab('files');
}

// ── Content of the individual tabs ───────────────────────────────────────────

function svToggleAdv() {
    const bar = document.getElementById('svAdvBar');
    const isOpen = bar.style.display !== 'none';
    bar.style.display = isOpen ? 'none' : 'flex';
    document.getElementById('svAdvToggle')?.classList.toggle('active', !isOpen);
}

/* How many of the filters behind "Filters & order" are set — shown on
 * the button, so a closed bar doesn't hide that the list is filtered. */
function svAdvCountSync() {
    const el = document.getElementById('svAdvCount');
    if (!el) return;
    let n = 0;
    for (const id of ['svDateFrom', 'svDateTo', 'svDurationMin', 'svDurationMax',
                      'svWordMin', 'svWordMax', 'svTempoMin', 'svTempoMax', 'svSort']) {
        if ((document.getElementById(id)?.value || '').trim()) n++;
    }
    if (document.getElementById('svTranscribedOnly')?.checked) n++;
    el.textContent = n ? String(n) : '';
}

/* "File & time" reads front to back: the direction does nothing there —
 * it said "descending" over a list in ascending order. Shown only for the
 * orders it turns. */
function svSortSync() {
    const d = document.getElementById('svDirection');
    if (!d) return;
    d.style.display = document.getElementById('svSort')?.value ? '' : 'none';
}

function svAdvReset() {
    for (const id of ['svDateFrom', 'svDateTo', 'svDurationMin', 'svDurationMax',
                      'svWordMin', 'svWordMax', 'svTempoMin', 'svTempoMax',
                      'svSort']) {
        const el = document.getElementById(id);
        if (el) el.value = '';
    }
    const r = document.getElementById('svDirection');
    if (r) r.value = 'desc';
    const nt = document.getElementById('svTranscribedOnly');
    if (nt) nt.checked = false;
    svSortSync();
    svRun();
}

async function svLoadMore(reset, seq) {
    if (seq === undefined) seq = svState.seq;
    if (svState.loading || (svState.done && !reset)) return;
    // File mode (empty search term but filters set) is allowed —
    // the server then searches purely via metadata.
    if (svState.query.length < 2 && !svHasFilter()) return;
    svState.loading = true;

    try {
        const url = `/api/search?${svParams()}&limit=${SV_PAGE}&offset=${svState.offset}`;
        const r = await fetch(url);
        const data = await r.json();
        if (seq !== svState.seq) return;   // stale response — discard

        if (data.error) {
            svListBox().innerHTML =
                `<div class="sv-empty">${esc(data.error)}</div>`;
            svState.done = true;
            return;
        }

        if (data.terms) svState.terms = data.terms;
        else if (reset) svState.terms = svState.query.toLowerCase().split(/\s+/);
        svState.total = data.total || 0;

        const items = data.results || [];
        svState.offset += items.length;
        // No more incoming, or the server has nothing left: done.
        if (!items.length || svState.offset >= svState.total) svState.done = true;

        // Filename/title hits only on the first load — they're tied to the
        // query, not the page, and would come in duplicated on load-more.
        // Rendering happens AFTER the text hits: `svRenderHits(…, reset)` clears
        // `#svResults`, and the block lives INSIDE it (see
        // svRenderFiles) instead of as a neighbor beside it.
        if (reset) svState.files = data.files || [];

        // Not a single word hit, but there is a search term? Then the
        // input most likely meant a FILENAME ("ptt") — the
        // file search takes over, with the same filters.
        if (data.mode !== 'files' && reset && !items.length
                && svState.query.length >= 2) {
            svState.done = true;
            await svFileFallback(seq);
            // Also here — and especially here. This branch is exactly the
            // case "no spoken word matches", i.e. the one where
            // semantic search is the only thing that can still find something. In a
            // first version it only ran after the regular path
            // and was skipped by the `return` here: anyone searching for "someone
            // is upset about a technical problem" — the
            // textbook example for a vector search — never got to see it.
            // AFTER the fallback, because `#svResults` gets rewritten.
            if (svState.mode === 'both') svMeaningSearch(seq, 0);
            return;
        }

        if (data.mode === 'files') {
            svRenderFileList(items, reset);
        } else {
            svRenderHits(items, reset);
        }
        if (reset) {
            svRenderFiles(svState.files);
            // Only trigger semantic search now: it needs an embedding
            // (~2s) and must not hold up the literal hits.
            if (svState.mode === 'both') svMeaningSearch(seq, items.length);
        }
        svSetStatus(svState.total, data.files || 0);
    } catch (e) {
        if (seq !== svState.seq) return;
        svListBox().innerHTML =
            `<div class="sv-empty">Error: ${esc(e.message)}</div>`;
        svState.done = true;
    } finally {
        // ALWAYS release, not only for the current sequence: the lock belongs
        // to THIS call. If a new search had started meanwhile, it would
        // otherwise stay set forever and the results list would never load
        // more — while the sidebar kept updating, which hid it well.
        svState.loading = false;
    }
}

function svSetStatus(total, files) {
    // data.files from /api/search only counts the files of the loaded PAGE.
    // The honest number is in the facets — otherwise the header would show
    // "2 files" for 4500 hits, just because two happened to be on page 1.
    // Second safeguard: the status line must never throw under any
    // circumstances. It runs in the same try as rendering the hits — an exception
    // here would replace the complete hit list with an error message.
    const z = (v) => Number.isFinite(Number(v))
        ? Number(v).toLocaleString() : '–';
    // The files tab lists files: its total IS the file count (its "hits in"
    // part is hidden — a restored tab passes no count of its own)
    const fileCount = svState.tab === 'files' ? total
        : (svState.fileCount !== null && svState.fileCount !== undefined)
        ? svState.fileCount : files;
    document.getElementById('svTotal').textContent = z(total);
    document.getElementById('svFiles').textContent = z(fileCount);
    document.getElementById('svShown').textContent = z(svState.offset);
    // The words follow their numbers: "1 hit in 1 file"
    const noun = (id, n, one, many) => {
        const el = document.getElementById(id);
        if (el) el.textContent = Number(n) === 1 ? one : many;
    };
    noun('svTotalNoun', total, 'hit', 'hits');
    noun('svFilesNoun', fileCount, 'file', 'files');
    const badge = document.getElementById('svTabNText');
    if (badge && svState.tab === 'text') badge.textContent = Number(total) > 0 ? z(total) : '';
    if (typeof svAdvCountSync === 'function') svAdvCountSync();
}

// ── Hits in the filename / embedded title ───────────────────────────
//
// Full-text search searches the SPOKEN word. Anyone searching for a
// series title, a recording date, or the name of a downloaded
// video won't find anything there — the term is in the filename or in
// the file's FFprobe metadata. These hits therefore appear as their own
// block above the text hits.
// How many file hits are shown expanded? The remainder sits behind "show
// all". Showing all 40 expanded took, for a query like "2024", more than
// the visible area — and the text hits only started
// below that. Three lines say "there are also file hits here", without
// crowding out the answer to the actual search.
const SV_FILES_SHORT = 3;
let _svFilesOpen = false;

function svToggleFiles() {
    _svFilesOpen = !_svFilesOpen;
    svRenderFiles(svState.files || []);
}

function svRenderFiles(files) {
    const previous = document.getElementById('svFileHits');
    if (previous) previous.remove();
    if (!files || !files.length) return;

    const res = document.getElementById('svResults');
    const box = document.createElement('div');
    box.id = 'svFileHits';
    box.className = 'sv-files';

    const fmtDuration = (s) => {
        if (!s || s <= 0) return '';
        const h = Math.floor(s / 3600), m = Math.round((s % 3600) / 60);
        return h > 0 ? `${h}:${String(m).padStart(2, '0')}h` : `${m}min`;
    };

    const showThem = _svFilesOpen ? files : files.slice(0, SV_FILES_SHORT);
    const remainder = files.length - showThem.length;
    let html = `<div class="sv-files-head">
        <span>📄 ${files.length} `
             + `${files.length === 1 ? 'file' : 'files'} with matching `
             + `name or title</span>`
             + ((remainder > 0 || _svFilesOpen)
                ? `<button class="sv-files-more" onclick="svToggleFiles()">`
                  + (_svFilesOpen ? 'show less'
                                     : `show all ${files.length}`)
                  + `</button>`
                : '')
             + `</div><div class="sv-files-list${_svFilesOpen ? ' open' : ''}">`;
    for (const d of showThem) {
        const duration = fmtDuration(d.duration_sec);
        const marker = d.has_transcript
            ? '<span class="sv-d-tx" title="Transcript available">📝</span>'
            : '';
        // The title is the more deliberate label than a filename like
        // LmNoPqRsTuV.mp4 — so it comes first when there is one.
        const header = d.title ? esc(d.title) : esc(d.display_name);
        const bottom = d.title ? esc(d.filename) : '';
        html += `<div class="sv-d-item" title="${esc(d.filename)}"
                      onclick="goToSearchResult('${escJs(d.filename)}', 0, '${escJs(d.file_dir || '')}')">
            <span class="sv-d-icon">${d.is_video ? '🎬' : '🎵'}</span>
            <span class="sv-d-text">
                <span class="sv-d-title">${header}</span>
                ${bottom ? `<span class="sv-d-path">${bottom}</span>` : ''}
            </span>
            ${duration ? `<span class="sv-d-duration">${duration}</span>` : ''}
            ${marker}
            <!-- Bridge to the voice map. It sits with the FILE hits, not
                 with the text hits: the map knows voices per recording, an
                 individual text hit has no counterpart there. The
                 shared key of both views is the file. -->
            <span class="sv-d-map" role="button" tabindex="0" data-keyclick title="Voices of this recording on the map"
                  onclick="event.stopPropagation();mapForFile('${escJs(svAbsPath(d))}')">🗣</span>
        </div>`;
    }
    html += '</div>';
    box.innerHTML = html;
    // INSIDE, not beside it. An earlier version said
    //     res.parentNode.insertBefore(box, res);
    // — and `res.parentNode` is `.sv-body`, a flex container with ROW
    // direction. That made the block the third sibling next to the sidebar
    // and the hit list, taking away its width: for "2024" the
    // hit list was left with a 90px-narrow column on the right edge, where the
    // text hits wrapped letter by letter. As the first child of `#svResults`
    // it scrolls along with the hits and only costs its own height.
    res.insertBefore(box, res.firstChild);
}

// ══════════════════════════════════════════════════════════════════════════
//  Background playback
// ══════════════════════════════════════════════════════════════════════════
//
// Originally every playback click closed the search: `svGoto` called
// `closeSearchView()` and jumped into the transcript. Anyone wanting to
// listen through 200 hits had to reopen the search every time.
//
// The player (`#videoPlayer`) sits OUTSIDE this overlay, in the
// main window. It therefore keeps running while the search stays open
// above it — it just needs a remote control. The exact opposite of an earlier
// trap ("innerHTML eats the audio element"): there, a running
// media element was torn out of the document along with its sound because it was
// INSIDE. Here it's outside, and that's why this works.
let _svWgTimer = null;

function svPlayer() { return document.getElementById('videoPlayer'); }

/* A hit's recording as a full path — what the voice map locates by. The
 * hit's filename is relative to its root (file_dir); on its own it was
 * looked up under the folder that happens to be open and found nothing
 * ("no voices are on file for this recording"). */
function svAbsPath(d) {
    const f = String((d && d.filename) || '');
    if (!f || f.startsWith('/') || /^[A-Za-z]:[\\/]/.test(f) || !d.file_dir) return f;
    return String(d.file_dir).replace(/[\\/]+$/, '') + '/' + f;
}

async function svPlay(filename, start, fileDir) {
    // No `closeSearchView()` — that's the whole difference from `svGoto`.
    svState.plays = { filename, start, fileDir };
    const bar = document.getElementById('svPlayer');
    if (bar) {
        bar.style.display = 'flex';
        document.getElementById('svWgFile').textContent = svShortName(filename);
        document.getElementById('svWgFile').title = filename;
    }
    try {
        await goToSearchResult(filename, start, fileDir);
    } catch (e) {
        if (typeof Toast !== 'undefined') Toast.error('Playback: ' + e.message);
    }
    svWgAttach();
}

// Track the player's events. Attach only ONCE: otherwise every
// playback click adds another listener, and the bar gets rewritten several
// times on every time step.
function svWgAttach() {
    const v = svPlayer();
    if (!v || v.dataset.svWg === '1') { svWgSync(); return; }
    v.dataset.svWg = '1';
    ['play', 'pause', 'ended', 'timeupdate', 'loadedmetadata']
        .forEach(e => v.addEventListener(e, svWgSync));
    svWgSync();
}

function svWgSync() {
    const v = svPlayer();
    const bar = document.getElementById('svPlayer');
    if (!v || !bar || bar.style.display === 'none') return;
    document.getElementById('svWgButton').textContent = v.paused ? '▶' : '⏸';
    const duration = Number.isFinite(v.duration) ? v.duration : 0;
    document.getElementById('svWgTime').textContent =
        svFmt(v.currentTime) + (duration ? ' / ' + svFmt(duration) : '');
    const bar2 = document.getElementById('svWgBar');
    // Do NOT interfere while dragging — otherwise the slider snaps back under the
    // finger to the last reported position.
    if (duration > 0 && document.activeElement !== bar2) {
        bar2.value = Math.round(v.currentTime / duration * 1000);
    }
}

function svWgToggle() {
    const v = svPlayer();
    if (!v) return;
    if (v.paused) { const p = v.play(); if (p && p.catch) p.catch(() => {}); }
    else v.pause();
    svWgSync();
}

function svWgJump(value) {
    const v = svPlayer();
    if (!v || !Number.isFinite(v.duration) || !v.duration) return;
    v.currentTime = (Number(value) / 1000) * v.duration;
    svWgSync();
}

function svWgInTranscript() {
    const s = svState.plays;
    const v = svPlayer();
    // Carry over the CURRENT position, not the one you started at —
    // if you've listened ten minutes further, you want to continue there.
    const t = (v && Number.isFinite(v.currentTime)) ? v.currentTime : (s?.start || 0);
    if (s) svGoto(s.filename, t, s.fileDir);
}

function svWgStop() {
    const v = svPlayer();
    if (v) v.pause();
    svState.plays = null;
    const bar = document.getElementById('svPlayer');
    if (bar) bar.style.display = 'none';
}

// ── Pure semantic mode ─────────────────────────────────────────────────────
//
// Here the vector search isn't the extra next to the hit list, it IS
// the list itself. Its hits are brought into the shape of the full-text hits
// so that line, selection, reading column and jump work unchanged
// — the only difference is the value column in front.
//
// It asks for more than "the 50 nearest vectors" — the
// situation search. `rerank=1`: a cross-encoder, the reranker, reads the
// question and a passage TOGETHER and scores how well the passage shows what
// was asked, where two vectors could only say how close they lie (the note
// on SV_MEANING_K: nonsense came out as "similar" as real questions).
// `group=1`: copies of one passage in other folders — backups, often a
// large share of the candidates — fold into one hit.
//
// So the list is sorted by RELEVANCE, each line with a bar; what the reranker
// scored below the backend's threshold, or never got to, waits behind "show
// weaker matches"; copies sit folded in their hit. When the reranker can't
// run (the card is busy), the backend says so: the list is then sorted by
// similarity, as before, with a quiet note in place of the bars.
const SV_MEANING_LIST_K = 50;
const SV_QUICK_MIN = 0.05;    // the backend's rerank.quick_min_score, if it sends none
let _svQuick = null;          // the meaning list on show (svQuickPrepare)

async function svMeaningOnly(seq) {
    const res = document.getElementById('svResults');
    _svMeaningData = null;
    _svQuick = null;
    svState.meaningPending = true;
    document.getElementById('svMeaning')?.remove();
    res.innerHTML = '<div class="sv-empty">Meaning search running… (the question is '
        + 'embedded and the best passages are ranked by relevance — a few seconds)</div>';
    const url = '/api/rag/search?mode=vector&k=' + SV_MEANING_LIST_K
        + '&group=1&q=' + encodeURIComponent(svState.query);
    const hitKey = h => h ? `${h.filename}|${h.file_dir}|${h.start}` : '';
    const show = (d, first) => {
        // What the reading column shows, and whether the reader chose it: the
        // first list selects its top hit by itself — that choice follows the
        // ranking; a passage the reader picked stays picked
        const before = first ? null : svState.hits[svState.selection];
        const auto = !first && svState.selection === 0 && _svQuick
            && hitKey(before) === hitKey((_svQuick.strong.concat(_svQuick.weak))[0]);
        _svQuick = svQuickPrepare(d);
        _svQuick.ranking = first;
        const all = _svQuick.strong.concat(_svQuick.weak);
        svState.total = all.length;
        svState.fileCount = new Set(all.map(t => t.filename)).size;
        svState.offset = all.length;
        svState.done = true;      // one batch — there is no next page
        svState.terms = [];
        svState.meaningPending = false;
        if (first || auto || !before) {
            svRenderQuick(true);
        } else {
            // The reader's passage, at its place in the ranked list — or, gone
            // from the shown hits, still in the reading column but unmarked
            svState.selection = _svQuick.strong.findIndex(h => hitKey(h) === hitKey(before));
            svRenderQuick(false);
        }
        svSetStatus(svState.total, svState.fileCount);
    };
    try {
        // Two steps: the fused list right away (a second or two), then the
        // reranker's order — reading 120 passages takes it 10–15 s, and a
        // list to look at meanwhile beats an empty pane.
        const ranked = fetch(url + '&rerank=1').then(r => r.json());
        let early = null;
        try {
            early = await (await fetch(url)).json();
        } catch (e) { /* the ranked answer still comes */ }
        if (seq !== svState.seq) return;
        let done = false;
        if (early && early.ok) show(early, true);
        const d = await ranked.catch(e => ({ok: false, error: e.message}));
        if (seq !== svState.seq) return;
        if (!d.ok) {
            if (_svQuick && early && early.ok) {
                // The fused list stays; only the ranking failed
                _svQuick.ranking = false;
                _svQuick.note = d.error || 'Relevance ranking failed';
                svRenderQuick(false);
                done = true;
            } else {
                throw new Error(d.error || 'Semantic search not available');
            }
        }
        if (!done) show(d, false);
    } catch (e) {
        if (seq !== svState.seq) return;
        svState.meaningPending = false;
        res.innerHTML = `<div class="sv-empty">Semantic search not available: `
            + `${esc(e.message)}</div>`;
        svState.done = true;
        svSetStatus(0, 0);
    } finally {
        svState.loading = false;
    }
    // Answered while another tab is shown: that tab's switch remembered the
    // "running …" line — remember the finished list instead
    if (svState.tab !== 'text') svSaveState('text');
}

/* The backend's answer as list hits — sorted by relevance when the reranker
 * scored them, split at the backend's threshold. Pure: no DOM. */
function svQuickPrepare(d) {
    const reranked = d.reranked === true;
    const min = Number(d.min_score);
    const minScore = d.min_score != null && Number.isFinite(min) ? min : SV_QUICK_MIN;
    let hits = (d.results || []).map(svQuickHit).filter(h => h.filename);
    if (reranked) {
        // The backend sends this order already — sorting here as well keeps
        // the bars and the order from ever telling different stories. The
        // unscored ones (beyond the reranker's share) after the scored ones,
        // in the order they came.
        hits = hits.map((h, i) => ({h, i})).sort((a, b) => {
            const x = a.h.relevance, y = b.h.relevance;
            if (x == null || y == null) return (x == null) - (y == null) || a.i - b.i;
            return y - x || a.i - b.i;
        }).map(p => p.h);
    }
    const clear = h => !reranked || (h.relevance != null && h.relevance >= minScore);
    return {
        reranked,
        known: typeof d.reranked === 'boolean',    // an older server says nothing
        note: d.note ? String(d.note) : '',
        minScore,
        candidates: Number(d.candidates) || 0,
        // People the query names, with the passages they speak in — those
        // passages were candidates too (rag/speakers.py)
        voices: Object.entries(d.voices && typeof d.voices === 'object' ? d.voices : {})
            .filter(([, n]) => Number(n) > 0).map(([who, n]) => [String(who), Number(n)]),
        ms: Number(d.elapsed_ms) || 0,
        strong: hits.filter(clear),
        weak: hits.filter(h => !clear(h)),
        weakOpen: false,
        open: new Set(),      // hits (by list index) whose copies are unfolded
    };
}

function svQuickHit(x) {
    const rel = Number(x.relevance);
    const vs = Number(x.vscore);
    const sc = Number(x.score);
    return {
        filename: x.rel_path || x.media_path || '',
        file_dir: x.file_dir || '',
        start: Number(x.start_sec) || 0,
        start_fmt: svTime(x.start_sec),
        text: x.content || '',
        // The name the voice library knows for the label, else the label
        speaker: x.speaker_name || x.speaker || '',
        score: x.score,
        relevance: x.relevance != null && Number.isFinite(rel)
            ? Math.max(0, Math.min(1, rel)) : null,
        // Similarity: the cosine. The situation search (its hits carry
        // `relevance`) sends it apart as `vscore` — null for a passage found
        // by its words only — and its `score` is the reranker's or the
        // fusion's value. An older server's `score` is the cosine itself.
        similarity: x.vscore != null && Number.isFinite(vs) ? vs
            : (!Object.prototype.hasOwnProperty.call(x, 'relevance')
               && x.score != null && Number.isFinite(sc) ? sc : null),
        words: x.match === 'fts',       // found by its words, not by meaning
        copies: svCopyList(x),
    };
}

/* A hit's copies — the same passage in another folder (backups) — each a
 * place of its own: file, root, time. The hit itself is left out should the
 * server list it too. A copy without a path (its transcript is no longer in
 * the library) stays visible but can't be opened. Shared with the deep
 * search (deepsearch.js). */
function svCopyList(x) {
    const ident = (o, dir) => String(o.media_path || o.transcript_key
        || ((o.file_dir || dir || '') + '/' + (o.rel_path || '')));
    const seen = new Set([ident(x) + '|' + (Number(x.start_sec) || 0)]);
    const out = [];
    for (const c of (Array.isArray(x && x.copies) ? x.copies : [])) {
        if (!c) continue;
        const start = Number(c.start_sec) || 0;
        const filename = c.rel_path || '';
        const label = filename || c.media_path || c.transcript_key || '';
        const key = ident(c, x.file_dir) + '|' + start;
        if (!label || seen.has(key)) continue;
        seen.add(key);
        out.push({filename, label, file_dir: c.file_dir || x.file_dir || '',
                  start, start_fmt: svTime(start), dead: !filename});
    }
    return out;
}

/* 0..1 as a percentage — one decimal below 1 %, so a weak score doesn't
 * read as "0 %", nothing at all. */
function svPct(v) {
    const n = Number(v);
    if (v == null || !Number.isFinite(n)) return '–';
    const p = Math.max(0, Math.min(100, n * 100));
    return (p > 0 && p < 1 ? p.toFixed(1) : String(Math.round(p))) + '%';
}

/* The meaning list: a head that says what the order means, the clear
 * matches, the rest behind "show weaker matches". `reset`: a new answer —
 * the first hit opens; otherwise (copies unfolded, weaker ones shown) the
 * list keeps its place. */
function svRenderQuick(reset) {
    const res = document.getElementById('svResults');
    const q = _svQuick;
    if (!res || !q) return;
    const top = res.scrollTop;
    svState.hits = q.weakOpen ? q.strong.concat(q.weak) : q.strong.slice();
    let html = svQuickHead(q);
    if (!q.strong.length && !q.weak.length) {
        html += '<div class="sv-empty">The RAG index has nothing for this — it may cover '
            + 'only part of the library so far.</div>';
    } else if (!q.strong.length) {
        html += '<div class="sv-quick-none">No passage clearly shows this. '
            + '<button class="sv-files-more" onclick="dsStartFromQuery()" title="A deep search '
            + 'checks the whole library passage by passage (in the LLM queue, takes minutes)">'
            + '🔎 Deep search</button> searches more thoroughly.</div>';
    }
    q.strong.forEach((h, i) => { html += svQuickRowHtml(h, i, false); });
    if (q.weak.length) {
        html += '<div class="sv-weak-toggle"><button class="sv-files-more" onclick="svQuickWeak()" '
            + 'title="Scored below the threshold by the reranker, or beyond the passages it reads">'
            + (q.weakOpen ? 'hide the weaker matches'
                          : `show weaker matches (${q.weak.length.toLocaleString()})`)
            + '</button></div>';
        if (q.weakOpen) {
            q.weak.forEach((h, j) => { html += svQuickRowHtml(h, q.strong.length + j, true); });
        }
    }
    res.innerHTML = html;
    if (reset) {
        svState.selection = -1;
        svState.copySel = '';
        if (svState.hits.length) svSelect(0);
    } else {
        res.scrollTop = top;
    }
}

// The head says what the order means — and, without the reranker, why the
// order is the old one.
function svQuickHead(q) {
    const n = q.strong.length, w = q.weak.length;
    const copies = q.strong.concat(q.weak).reduce((s, h) => s + h.copies.length, 0);
    const voiced = (q.voices || []).map(([who, k]) => ` · with the ${k.toLocaleString()} `
        + `passage${k === 1 ? '' : 's'} ${esc(who)} speaks in`).join('');
    const folded = (copies
        ? ` · ${copies.toLocaleString()} ${copies === 1 ? 'copy' : 'copies'} folded in` : '')
        + voiced;
    let title;
    if (q.reranked) {
        const from = q.candidates ? ` from ${plural(q.candidates, 'candidate')}` : '';
        title = (n
            ? `🧠 ${n.toLocaleString()} passage${n === 1 ? '' : 's'} by relevance`
              + `<span class="sv-semantic-hint"> — ranked by the reranker${from}${folded}</span>`
            : '🧠 No passage clearly matches'
              + `<span class="sv-semantic-hint"> — ${w.toLocaleString()} weaker `
              + `match${w === 1 ? '' : 'es'} below${folded}</span>`)
            // Ranked, but from fewer candidates than usual (the meaning
            // half failed — full text only): the server says so
            + (q.note ? `<span class="sv-quick-note">${esc(q.note)}</span>` : '');
    } else if (q.ranking) {
        // The first, fused list — the reranker is still reading
        title = `🧠 the ${plural(n, 'closest passage')}`
            + '<span class="sv-quick-note sv-quick-ranking" title="The reranker reads the best '
            + 'passages together with the question and scores how well each one matches — the '
            + 'list re-sorts when it is done">ranking by relevance …</span>'
            + (folded ? `<span class="sv-semantic-hint">${folded}</span>` : '');
    } else if (q.known) {
        title = `🧠 the ${plural(n, 'closest passage')}`
            + '<span class="sv-quick-note" title="The reranker scores relevance — without it the '
            + 'list is in the fused order of meaning and wording">'
            + `${esc(q.note || 'Reranker busy — sorted by meaning and wording')}</span>`
            + (folded ? `<span class="sv-semantic-hint">${folded}</span>` : '');
    } else {
        title = `🧠 the ${plural(n, 'most similar spot')} in the RAG index`
            + '<span class="sv-semantic-hint"> — sorted by similarity, not by '
            + '"matches / doesn\'t match"</span>';
    }
    return `<div class="sv-semantic-head sv-quick-head"><span>${title}</span>`
        + `<span class="sv-semantic-hint">${q.ms.toLocaleString()} ms</span></div>`;
}

// One line of the meaning list: the value in front (relevance bar, or the
// similarity when nothing was ranked), then the usual time · file · text,
// and the copies' button at the end.
function svQuickRowHtml(h, i, weak) {
    const q = _svQuick;
    const open = !!(q && q.open.has(i));
    const n = h.copies.length;
    const copies = n
        ? `<button class="sv-copies-btn${open ? ' open' : ''}" onclick="event.stopPropagation();svQuickCopies(${i})"
                   title="The same passage in ${n} other ${n === 1 ? 'place' : 'places'} — copies, backups">+${n} ${n === 1 ? 'copy' : 'copies'}</button>`
        : '<span></span>';
    const active = i === svState.selection && !svState.copySel;
    return `<div class="sv-row with-score with-rel${weak ? ' weak' : ''}${active ? ' active' : ''}"
                 id="svZ${i}" onclick="svSelect(${i})"
                 ondblclick="svGoto('${escJs(h.filename)}', ${Number(h.start) || 0}, '${escJs(h.file_dir || '')}')"
                 title="${esc(h.filename)}">
        ${svQuickValue(h, q)}
        <span class="sv-z-time">${esc(h.start_fmt || '')}</span>
        <span class="sv-z-file">${esc(svShortName(h.filename))}</span>
        <span class="sv-z-text">${_hlHit(h, svState.terms)}</span>
        ${copies}
    </div>` + (n && open ? svCopiesHtml(h, i) : '');
}

function svQuickValue(h, q) {
    if (h.relevance != null) {
        const pct = svPct(h.relevance);
        return `<span class="sv-z-rel" title="Relevance ${esc(pct)} — how well the reranker thinks the passage shows what was asked">`
            + `<i><b style="width:${(h.relevance * 100).toFixed(1)}%"></b></i>${esc(pct)}</span>`;
    }
    if (q && q.reranked) {
        return '<span class="sv-z-rel none" title="Not ranked — the reranker reads only the best '
            + 'candidates"><i></i>–</span>';
    }
    if (h.similarity != null) {
        return `<span class="sv-z-score" title="Similarity">${h.similarity.toFixed(2)}</span>`;
    }
    if (h.words) {
        return '<span class="sv-z-score words" title="Found by its words (full text), not by meaning">words</span>';
    }
    return '<span></span>';
}

// The copies of hit `i`, one line each: click shows it in the reading
// column, double-click (or ↗) opens its transcript, ▶ plays it from there.
function svCopiesHtml(h, i) {
    return '<div class="sv-copies">' + h.copies.map((c, j) => {
        if (c.dead) {
            return `<div class="sv-copy dead" title="${esc(c.label)} — no longer in the library">
                <span class="sv-copy-mark">↳</span><span class="sv-z-time">${esc(c.start_fmt)}</span>
                <span class="sv-copy-path"><bdi>${esc(c.label)}</bdi></span>
                <span class="sv-copy-gone">not in the library</span></div>`;
        }
        const id = `svC${i}_${j}`;
        const f = escJs(c.filename), d = escJs(c.file_dir);
        return `<div class="sv-copy${svState.copySel === id ? ' active' : ''}" id="${id}"
                     onclick="svSelectCopy(${i}, ${j})" ondblclick="svGoto('${f}', ${Number(c.start) || 0}, '${d}')"
                     title="${esc(c.filename)}">
            <span class="sv-copy-mark">↳</span>
            <span class="sv-z-time">${esc(c.start_fmt)}</span>
            <span class="sv-copy-path"><bdi>${esc(c.filename)}</bdi></span>
            <button class="sv-copy-act" onclick="event.stopPropagation();svPlay('${f}', ${Number(c.start) || 0}, '${d}')"
                    title="Play this copy from ${esc(c.start_fmt)}">▶</button>
            <button class="sv-copy-act" onclick="event.stopPropagation();svGoto('${f}', ${Number(c.start) || 0}, '${d}')"
                    title="Open this copy in its transcript">↗</button>
        </div>`;
    }).join('') + '</div>';
}

// "+3 copies": fold the copies of hit `i` out or in.
function svQuickCopies(i) {
    const q = _svQuick;
    if (!q || !svState.hits[i]) return;
    if (q.open.has(i)) q.open.delete(i); else q.open.add(i);
    // A copy folded away can't stay the one shown — its hit takes over
    const shown = !q.open.has(i) && svState.copySel.startsWith(`svC${i}_`);
    if (shown) svState.copySel = '';
    svRenderQuick(false);
    if (shown) svSelect(i);
}

// "show weaker matches (N)": they join the list (and the arrow keys) below
// the clear ones — or leave it again.
function svQuickWeak() {
    const q = _svQuick;
    if (!q) return;
    q.weakOpen = !q.weakOpen;
    if (!q.weakOpen && svState.selection >= q.strong.length) {
        svState.selection = -1;
        svState.copySel = '';
        svClearRead();
    }
    svRenderQuick(false);
}

/* A copy in the reading column — the same passage, in a file of its own. It
 * reads like a hit: its context, ▶ from its time, open its transcript. */
function svSelectCopy(i, j) {
    const h = svState.hits[i];
    const c = h && h.copies && h.copies[j];
    if (!c || c.dead) return;
    document.getElementById('svZ' + svState.selection)?.classList.remove('active');
    if (svState.copySel) document.getElementById(svState.copySel)?.classList.remove('active');
    svState.selection = i;        // the arrow keys go on from its hit
    svState.copySel = `svC${i}_${j}`;
    document.getElementById(svState.copySel)?.classList.add('active');
    return svShowRead({...h, filename: c.filename, file_dir: c.file_dir,
                       start: c.start, start_fmt: c.start_fmt, copies: []});
}

// ── Semantically related spots (vector search) ───────────────────────────────────
//
// Full-text search finds WORDS. Anyone searching for "it's about money" won't
// find a spot where someone says "that was expensive" — not because it's
// missing, but because none of the searched words occur there. This gap is
// exactly what the vector index fills.
//
// That's why `mode=vector` runs here and NOT `hybrid`: the full-text part
// of `hybrid` would only repeat what's already shown above. What's added is
// what the literal search structurally cannot find.
//
// Three rules so this doesn't get in the way:
//   1. It runs ALONGSIDE — the literal hits are long since there when the
//      embedding (~2s, the server spins up on demand if needed) comes back.
//      It must never block and never pass an error up.
//   2. It sits COLLAPSED below the file hits — unless the
//      literal search found nothing. Then it's the only answer and expands
//      on its own.
//   3. It says HOW MUCH it even knows. The index so far only covers
//      part of the library; "no hits" would otherwise mean "doesn't
//      exist" where it should mean "not embedded yet".
/* How many semantically related spots the block shows.
 *
 * IMPORTANT, because it invites the question "why is it always 12?":
 * this number is a LIMIT, not a result. A vector search
 * always returns the k nearest neighbors — there's no point at which it
 * says "found nothing".
 *
 * A minimum similarity, on the other hand, does NOT help. Measured against a real
 * database, the best of twelve hits each time:
 *
 *     "an appointment is being arranged"          0.643
 *     "someone is upset about a problem"          0.524
 *     "it's about cooking and eating"             0.496
 *     "flrmpf glonk zzzz"          (nonsense)      0.527
 *     "complete nonsense xyzq"     (nonsense)      0.524
 *     "qwertz asdfgh yxcvbn"       (nonsense)      0.502
 *
 * The ranges overlap completely: any threshold that keeps out the nonsense
 * also throws out two of the three meaningful questions. That's why there's
 * no threshold here, just the value on each hit and a
 * label that says "most similar" rather than "matching". A real separation
 * would only be possible with a reranker (cross-encoder) that scores question and
 * spot together, instead of comparing two vectors.
 */
const SV_MEANING_K = 12;
let _svMeaningOpen = false;
let _svMeaningData = null;
let _svRagDown = false;     // last RAG call failed (e.g. no llama-server)
let _svRagDownAt = 0;       // when it failed — "Text + RAG" retries after a pause
const SV_RAG_RETRY_MS = 60000;

function svRagOptionSync() {
    for (const m of ['both', 'meaning']) {
        const b = document.querySelector(`#svModeSeg [data-mode="${m}"]`);
        if (!b) continue;
        b.classList.toggle('off', _svRagDown);
        b.dataset.off = _svRagDown ? 'meaning search not available' : '';
    }
}

function svToggleMeaning() {
    _svMeaningOpen = !_svMeaningOpen;
    svRenderMeaning();
}

async function svMeaningSearch(seq, wordHits) {
    document.getElementById('svMeaning')?.remove();
    _svMeaningData = null;
    _svMeaningOpen = false;
    if (svState.query.length < 2) return;
    // "Text + RAG" with RAG down would repeat the same notice on every
    // search — after the first failure it quietly searches text only.
    // Not for good though: llama-server may just have been starting —
    // after SV_RAG_RETRY_MS the next search tries again.
    if (_svRagDown && svState.mode === 'both'
            && Date.now() - _svRagDownAt < SV_RAG_RETRY_MS) return;
    // Without literal hits, the semantic search is the only answer — then
    // it's shown expanded, instead of hiding behind a button.
    _svMeaningOpen = !wordHits;

    _svMeaningData = { running: true };
    svRenderMeaning();
    try {
        const r = await fetch('/api/rag/search?mode=vector&k=' + SV_MEANING_K
            + '&q=' + encodeURIComponent(svState.query));
        const d = await r.json();
        if (seq !== svState.seq) return;    // search has long since moved on
        _svMeaningData = d.ok
            ? { hits: d.results || [], ms: d.elapsed_ms }
            : { error: d.error || 'Semantic search not available' };
    } catch (e) {
        if (seq !== svState.seq) return;
        _svMeaningData = { error: e.message };
    }
    _svRagDown = !!_svMeaningData.error;
    _svRagDownAt = _svRagDown ? Date.now() : 0;
    svRagOptionSync();
    svRenderMeaning();
}

function svRenderMeaning() {
    const previous = document.getElementById('svMeaning');
    if (previous) previous.remove();
    if (!_svMeaningData) return;
    const res = document.getElementById('svResults');
    if (!res) return;

    const box = document.createElement('div');
    box.id = 'svMeaning';
    box.className = 'sv-semantic';

    if (_svMeaningData.running) {
        box.innerHTML = '<div class="sv-semantic-head"><span>🧠 RAG search running…'
            + '</span></div>';
    } else if (_svMeaningData.error) {
        // No red alarm: the vector search is an EXTRA. Anyone who hasn't
        // set it up still has a complete full-text search.
        box.innerHTML = '<div class="sv-semantic-head"><span class="sv-semantic-off">'
            + `🧠 RAG not available — ${esc(_svMeaningData.error)}</span></div>`;
    } else {
        const t = _svMeaningData.hits;
        const header = t.length
            ? `🧠 the ${plural(t.length, 'most similar spot')} in the RAG index`
              + ` <span class="sv-semantic-hint">— without the searched words,`
              + ` sorted by similarity</span>`
            : '🧠 The RAG index has nothing';
        let html = `<div class="sv-semantic-head"><span>${header}</span>`
            + `<button class="sv-files-more" onclick="svToggleMeaning()">`
            + (_svMeaningOpen ? 'collapse' : 'show') + `</button></div>`;
        if (_svMeaningOpen) {
            html += '<div class="sv-semantic-list">';
            if (!t.length) {
                html += '<div class="sv-semantic-empty">The RAG index so far only covers '
                    + 'part of the library — what\'s missing here isn\'t '
                    + 'necessarily unsaid, it may simply '
                    + 'not be embedded yet.</div>';
            }
            for (const s of t) {
                // Without rel_path the UI doesn't know the file (transcript
                // removed from the DB, index not yet caught up) — then
                // the hit stays readable, but not clickable.
                const jumps = !!s.rel_path;
                const name = s.rel_path || s.media_path || '—';
                html += `<div class="sv-semantic-item${jumps ? '' : ' dead'}"
                         ${jumps ? `onclick="svGoto('${escJs(s.rel_path)}', ${Number(s.start_sec) || 0}, '${escJs(s.file_dir || '')}')"` : ''}
                         title="${esc(name)}">
                    <div class="sv-semantic-meta">
                        <span class="sv-semantic-score" title="Similarity">${(s.score || 0).toFixed(2)}</span>
                        <span class="sv-semantic-time">${svTime(s.start_sec)}</span>
                        <span class="sv-semantic-sp">${esc(s.speaker_name || s.speaker || '—')}</span>
                        <span class="sv-semantic-file">${esc(name.split('/').pop())}</span>
                    </div>
                    <div class="sv-semantic-text">${esc(s.content || '')}</div>
                </div>`;
            }
            html += '</div>';
        }
        box.innerHTML = html;
    }

    // Below the file hits, above the text hits. `insertBefore` instead of
    // `after()`: the same effect, but it's the method the file
    // already uses throughout.
    const fileHits = document.getElementById('svFileHits');
    res.insertBefore(box, fileHits ? fileHits.nextSibling : res.firstChild);
}

function svTime(sec) {
    const s = Math.max(0, Math.floor(sec || 0));
    const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60);
    return (h ? `${h}:${String(m).padStart(2, '0')}` : `${m}`)
        + ':' + String(s % 60).padStart(2, '0');
}

// ── Render results ────────────────────────────────────────────────────

let _svLastGroup = null;   // last file heading, so load-more doesn't
                           // create a duplicate group for the same file

// One LINE per hit, fixed grid: time, speaker, snippet. Originally
// every hit brought its whole paragraph along — for a common word
// only 7 of thousands of hits were visible at once, and because the paragraphs
// ranged from 2 to 61 words, there was no grid for the eye to
// follow along. The context now sits on the right in the
// reading column, for the one hit you're actually looking at.
function svRenderHits(items, reset) {
    const res = document.getElementById('svResults');
    if (reset) {
        res.innerHTML = '';
        _svLastGroup = null;
        svState.hits = [];
    }

    if (!items.length) {
        if (reset) res.insertAdjacentHTML('beforeend',
            '<div class="sv-empty">No hits</div>');
        else svAppendFooter();
        return;
    }

    const first = svState.hits.length;   // index of the first new hit
    svState.hits.push(...items);

    if (svState.view === 'recordings') {
        // The cards are built from the facets plus ALL loaded hits — so a
        // load-more page can't be appended, it has to redraw the whole
        // overview (appending used to repeat every card on each page).
        if (reset) svRenderRecordings(true);
        else svRedrawRecordings();
        return;
    }

    // FLAT list, no group header per file. The headers were in the first
    // attempt and cost half the area: for a common word the 200 loaded
    // hits were spread across well over a hundred files — fewer than two
    // hits per header, which nearly halved the hits visible at once.
    // Which recording it is now sits in the line itself and in full
    // in the reading column.
    let html = '';
    items.forEach((r, n) => { html += svRowHtml(r, first + n); });

    // Remove footer before appending, otherwise it slides into the middle
    document.getElementById('svFooter')?.remove();
    res.insertAdjacentHTML('beforeend', html);
    svAppendFooter();

    // On first load, open the first hit right away — otherwise
    // half the view sits empty and you have to guess that it does something.
    if (reset && svState.hits.length) svSelect(0);
}

// `withoutFile` in the recordings view: the name is already in the header of the
// card there, repeating it in every line would just be wasted width.
function svRowHtml(r, i, withoutFile) {
    const sp = r.speaker_name || r.speaker || '';
    return `<div class="sv-row${i === svState.selection ? ' active' : ''}${withoutFile ? ' narrow' : ''}${r.score != null ? ' with-score' : ''}"
                 id="svZ${i}" onclick="svSelect(${i})"
                 ondblclick="svGoto('${escJs(r.filename)}', ${Number(r.start) || 0}, '${escJs(r.file_dir || '')}')"
                 title="${esc(r.filename)}">
        ${r.score != null ? `<span class="sv-z-score" title="Similarity">${Number(r.score).toFixed(2)}</span>` : ''}
        <span class="sv-z-time">${esc(r.start_fmt || '')}</span>
        ${withoutFile
            ? `<span class="sv-z-sp">${esc(sp || '—')}</span>`
            : `<span class="sv-z-file">${esc(svShortName(r.filename))}</span>`}
        <span class="sv-z-text">${_hlHit(r, svState.terms)}</span>
    </div>`;
}

// The filename without the path in front. The full path is in the title and in
// the reading column — in the line it just eats the width the text needs.
function svShortName(path) {
    const parts = String(path || '').split('/');
    return parts[parts.length - 1] || path || '';
}

// ── Reading column ────────────────────────────────────────────────────────────

function svClearRead() {
    const el = document.getElementById('svRead');
    if (el) el.innerHTML = '<div class="sv-read-empty">Click a hit — it appears here '
        + 'in context.<div class="sv-read-keys"><kbd>↑</kbd> <kbd>↓</kbd> next hit '
        + '(<kbd>Alt</kbd> in a field) · double-click opens the transcript</div></div>';
}

async function svSelect(i) {
    const r = svState.hits[i];
    if (!r) return;
    document.getElementById('svZ' + svState.selection)?.classList.remove('active');
    if (svState.copySel) {
        document.getElementById(svState.copySel)?.classList.remove('active');
        svState.copySel = '';
    }
    svState.selection = i;
    const line = document.getElementById('svZ' + i);
    line?.classList.add('active');
    line?.scrollIntoView({ block: 'nearest' });
    return svShowRead(r);
}

// The reading column for one place: its passage in context, ▶ from its time.
async function svShowRead(r) {
    const read = document.getElementById('svRead');
    if (!read) return;
    const sp = r.speaker_name || r.speaker || '—';
    // A meaning hit says how relevant the reranker found it, and that the
    // same passage sits in other places too
    const extra = (r.relevance != null ? ` · relevance ${svPct(r.relevance)}` : '')
        + (r.copies && r.copies.length
            ? ` · ${r.copies.length} ${r.copies.length === 1 ? 'copy' : 'copies'}` : '');
    const header = `<div class="sv-read-head">
            <span class="sv-read-file" title="${esc(r.filename)}">
                <b>${esc(svShortName(r.filename))}</b>
                <i>${esc(r.filename.slice(0, Math.max(0, r.filename.length - svShortName(r.filename).length - 1)))}</i>
            </span>
            <span class="sv-read-markers">${esc(r.start_fmt || '')} · ${esc(sp)}${esc(extra)}</span>
            <button class="sv-read-btn"
                    onclick="svPlay('${escJs(r.filename)}', ${Number(r.start) || 0}, '${escJs(r.file_dir || '')}')">
                ▶ play from ${esc(r.start_fmt || '0:00')}</button>
        </div>`;
    read.innerHTML = header + '<div class="sv-read-text">'
        + `<p class="sv-abs core sv-abs-passage">${_hlHit(r, svState.terms)}</p>`
        + '<p class="sv-read-loading">loading context…</p></div>';

    // Load the surrounding context specifically. `readSeq` discards whatever
    // comes back too late: anyone paging through the list with arrow keys
    // triggers faster than the answers come back, and the last answer isn't
    // necessarily the one for the last selected line.
    const seq = ++svState.readSeq;
    try {
        const r2 = await fetch('/api/search/context?file='
            + encodeURIComponent(r.filename)
            + '&dir=' + encodeURIComponent(r.file_dir || '')
            + '&start=' + (r.start || 0) + '&n=3');
        const d = await r2.json();
        if (seq !== svState.readSeq) return;
        if (!d.segments) throw new Error(d.error || 'no context');
        // The hit keeps the marks the search set (whole words): re-marked
        // by the plain terms, "ich" lit up inside "mich" once the context came
        read.innerHTML = header + '<div class="sv-read-text">'
            + d.segments.map(s => `<p class="sv-abs ${s.is_hit ? 'core' : 'edge'}">
                    <span class="sv-abs-sp">${esc(s.speaker)}</span>${
                      !s.is_hit ? esc(s.text)
                      : s.text === r.text ? _hlHit(r, svState.terms)
                      : highlightTerms(s.text, svState.terms)
                    }</p>`).join('')
            + '</div>'
            + `<div class="sv-read-foot">
                 <span class="sv-read-pill" role="button" tabindex="0" data-keyclick onclick="mapForFile('${escJs(svAbsPath(r))}')">🗣 Voices of this recording</span>
                 <span class="sv-read-pill" role="button" tabindex="0" data-keyclick onclick="svPinFile('${escJs(r.filename)}')">search only in this file</span>
                 <span class="sv-read-pill" role="button" tabindex="0" data-keyclick onclick="svAskFile('${escJs(r.filename)}')">💬 ask about this recording</span>
                 <span class="sv-read-pill" role="button" tabindex="0" data-keyclick onclick="svGoto('${escJs(r.filename)}', ${Number(r.start) || 0}, '${escJs(r.file_dir || '')}')">↗ open in the transcript</span>
               </div>`;
    } catch (e) {
        if (seq !== svState.readSeq) return;
        // No drama: the hit itself is already there, only the
        // context is missing. The message says what's missing, instead of clearing the column.
        read.innerHTML = header + '<div class="sv-read-text">'
            + `<p class="sv-abs core sv-abs-passage">${_hlHit(r, svState.terms)}</p>`
            + `<p class="sv-read-loading">Context could not be loaded: ${esc(e.message)}</p></div>`;
    }
}

// Arrow keys in the list. Without this a dense list is only worth half as
// much — you want to page through, not click.
function svMoveSelection(step) {
    if (!svState.hits.length) return;
    const new_value = Math.max(0, Math.min(svState.hits.length - 1,
                                     (svState.selection < 0 ? 0 : svState.selection + step)));
    svSelect(new_value);
}

// ── "Recordings" view ───────────────────────────────────────────────────
//
// The same hits, but the recording is the unit: one card per file
// with the hit band over the runtime. Answers the question "which
// recording was that in?" first — the hit list answers it last.
//
// The bands come from the facets (`histogram`, 40 buckets over the
// runtime), not from the loaded hits: the facets know ALL
// hits of the file, the loaded page only the first 200.
function svToggleView() {
    svState.view = svState.view === 'hits' ? 'recordings' : 'hits';
    try { localStorage.setItem('svView', svState.view); } catch (e) { /* doesn't matter */ }
    svViewButtonSync();
    // The meaning list has no facets to build recording cards from — it
    // stays a list (with its bars, copies and weaker matches) in both views
    if (_svQuick) { svRenderQuick(false); return; }
    const res = document.getElementById('svResults');
    res.innerHTML = '';
    _svLastGroup = null;
    if (svState.view === 'recordings') {
        svRenderRecordings(true);
    } else {
        const all_items = svState.hits;
        svState.hits = [];
        svRenderHits(all_items, true);
    }
    svRenderFiles(svState.files);
    svRenderMeaning();
}

function svViewButtonSync() {
    const shown = svState.tab === 'map' ? 'map' : svState.view;
    document.querySelectorAll('#svViewBtn [data-view]').forEach(b =>
        b.classList.toggle('active', b.dataset.view === shown));
}

/* Hits · By recording · Map. The map has a pane of its own (it is drawn
 * from the facets, not the list), so it is reached as the 'map' tab — the
 * text tab stays lit; the two list views switch back to the text. */
function svSetView(v) {
    if (v === 'map') { svTab('map'); return; }
    if (svState.tab === 'map') svTab('text');
    if ((v === 'hits' || v === 'recordings') && v !== svState.view) svToggleView();
    else svViewButtonSync();
}

/* Redraws the recordings view in place: the cards plus the two blocks that
 * live inside `#svResults` too (filename hits, RAG block) — clearing the
 * list without re-inserting them wiped both. Keeps the scroll position, so a
 * load-more page or late facets don't throw the reader back to the top. */
function svRedrawRecordings() {
    const res = document.getElementById('svResults');
    const top = res ? res.scrollTop : 0;
    svRenderRecordings(true);
    svRenderFiles(svState.files);
    svRenderMeaning();
    if (res) res.scrollTop = top;
}

function svRenderRecordings(reset) {
    const res = document.getElementById('svResults');
    if (reset) res.innerHTML = '';
    document.getElementById('svFooter')?.remove();

    /* The data basis is the FACETS, not the loaded hits.
     *
     * The first attempt grouped the 200 loaded hits — and those come
     * sorted by path. The screen then showed "clip.mp4 · 1
     * hit" for a file with 31 hits, with five-second snippets at
     * the very top because one hit in five seconds gives the highest
     * "density". The facets know every affected file with its
     * true hit count and the band over the runtime; the loaded
     * hits only contribute the example lines.
     *
     * Sorting is by HIT COUNT. Density as the primary key sounds
     * fairer toward long recordings, but on real data leads exactly
     * to the picture above.
     */
    const files = (svState.facets?.files || []);
    if (!files.length) {
        res.insertAdjacentHTML('beforeend',
            '<div class="sv-empty">No recordings — the overview needs the '
            + 'search facets.</div>');
        return;
    }

    const perFile = new Map();
    svState.hits.forEach((r, i) => {
        if (!perFile.has(r.filename)) perFile.set(r.filename, []);
        perFile.get(r.filename).push(i);
    });

    let html = '';
    for (const f of files) {
        const idx = perFile.get(f.filename) || [];
        const example = idx.length ? svState.hits[idx[0]] : null;
        const duration = Number(f.duration) || 0;
        const hist = f.histogram || [];
        const peak = Math.max(1, ...hist);
        // Clicking a bar jumps to THIS spot in the recording — the
        // timeline is thus a control, not just a picture.
        const band = hist.length
            ? hist.map((v, i) => {
                const t = duration * i / hist.length;
                return `<i class="${v ? 'hot' : ''}" style="height:${v ? (20 + 80 * v / peak).toFixed(0) : 6}%"
                   title="${plural(v, 'hit')} at ${esc(svFmt(t))}"
                   onclick="event.stopPropagation();svGoto('${escJs(f.filename)}', ${t.toFixed(1)}, '${escJs(f.file_dir || '')}')"></i>`;
              }).join('')
            : '';
        html += `<div class="sv-rec">
            <div class="sv-rec-head" title="${esc(f.filename)}">
                <span class="sv-rec-name" onclick="svGoto('${escJs(f.filename)}', 0, '${escJs(f.file_dir || '')}')">${esc(svShortName(f.filename))}</span>
                ${duration ? `<span class="sv-badge">${esc(svFmt(duration))}</span>` : ''}
                <span class="sv-rec-count">${plural(f.hits, 'hit')}</span>
            </div>
            ${band ? `<div class="sv-rec-band">${band}</div>` : ''}
            ${idx.slice(0, 3).map(i => svRowHtml(svState.hits[i], i, true)).join('')}
            ${(!idx.length || idx.length < Number(f.hits || 0))
                ? `<div class="sv-rec-more" onclick="svPinFile('${escJs(f.filename)}')">${
                     idx.length ? 'all ' + Number(f.hits).toLocaleString() + ' hits of this recording'
                                : 'show hits of this recording'}</div>`
                : ''}
        </div>`;
    }

    // What's left out is named — otherwise the list reads as if
    // "these are all affected recordings".
    const f = svState.facets || {};
    if (f.files_truncated) {
        const remainder = Math.max(0, (f.file_count || 0) - files.length);
        html += `<div class="sv-more">… ${remainder.toLocaleString()} more `
              + 'recordings with hits are not listed here '
              + '(the facets deliver the 500 with the most hits)</div>';
    }
    res.insertAdjacentHTML('beforeend', html);
}

/* Fallback: the search term didn't match a spoken word → interpret it as a
 * FILENAME and run the metadata search.
 *
 * An earlier version said:
 *
 *     const name = svState.pinned || svState.file || svState.query;
 *
 * That's exactly backwards. The filter isn't a search query. If a
 * pinned file was in `pinned`, the fallback searched for files whose
 * name matches THIS file — and inevitably found exactly it. The screen
 * then showed:
 *
 *     No hit in the spoken text — but 1 file(s) whose name matches
 *     "media/archive/meeting_2024-03-01.mp4"
 *
 * for a search that had nothing to do with that name. The message looked
 * like a result, but was just the filter finding itself — while the small
 * search, which doesn't know about pinning, delivered hundreds of hits for
 * the same query.
 *
 * With a file filter set, there's nothing to search here: the field that
 * would be searched is already occupied. Then the view says WHY,
 * and offers to clear the filter.
 */
async function svFileFallback(seq) {
    const filter = svState.pinned || svState.file;
    if (filter) {
        const res = document.getElementById('svResults');
        res.innerHTML = `<div class="sv-empty">
            No hits for "${esc(svState.query)}" —
            the search is restricted to <b>${esc(filter)}</b>.
            <div style="margin-top:10px">
                <button class="btn" onclick="svClearFilters()">
                    Clear filter and search again
                </button>
            </div></div>`;
        svSetStatus(0, 0);
        return;
    }
    const name = svState.query;
    let p = `q=&file=${encodeURIComponent(name)}`;
    if (svState.speaker) p += `&speaker=${encodeURIComponent(svState.speaker)}`;
    p += svPersonParams();
    p += svAdvParams();
    try {
        const r = await fetch(`/api/search?${p}&limit=500&offset=0`);
        const data = await r.json();
        if (seq !== svState.seq) return;
        const res = document.getElementById('svResults');
        const items = data.results || [];
        if (!items.length) {
            res.innerHTML = '<div class="sv-empty">No hits — neither in the '
                + 'spoken text nor in the filename (check filters).</div>';
            svSetStatus(0, 0);
            return;
        }
        res.innerHTML = `<div class="sv-fallback-head">No hit in the
            spoken text — but <b>${data.total}</b> file${data.total === 1 ? '' : 's'} whose
            <b>name</b> matches "${esc(name)}" (filters applied):</div>`;
        _svLastGroup = null;
        _svLastFolder = null;
        svState.total = data.total || 0;
        svRenderFileList(items, false);
        svSetStatus(data.total || 0, data.files || 0);
    } catch (e) { /* main "No hits" message stays as-is */ }
}

// File mode: one line per file, under its folder — the folder once, as a
// slim head line, the line itself only the file's name. Every line with the
// whole path was hard to read — you don't always have to see the whole
// path. The values stand in columns, so they read down the list.
// Click opens the file at the beginning.
let _svLastFolder = null;   // the folder the list ends with — load-more continues it

function svRenderFileList(items, reset) {
    const res = svListBox();
    if (reset) { res.innerHTML = ''; _svLastGroup = null; _svLastFolder = null; }
    if (!items.length) {
        if (reset) res.innerHTML =
            '<div class="sv-empty">No file matches the filters.</div>';
        else svAppendFooter();
        return;
    }
    let html = '';
    items.forEach(r => {
        const path = String(r.filename || '');
        const cut = path.lastIndexOf('/');
        const folder = cut > 0 ? path.slice(0, cut) : '';
        const key = (r.file_dir || '') + '|' + folder;
        if (key !== _svLastFolder) {
            _svLastFolder = key;
            html += svFolderHead(folder, r.file_dir || '');
        }
        html += svFileLine(r, cut >= 0 ? path.slice(cut + 1) : path);
    });
    document.getElementById('svFooter')?.remove();
    res.insertAdjacentHTML('beforeend', html);
    svAppendFooter();
}

/* A folder as a head line: its last levels in full, the ones before
 * folded — the whole path is in the tooltip. */
function svFolderHead(folder, root) {
    const parts = folder ? folder.split('/') : [];
    const shown = parts.length > 4
        ? [parts[0], '…', ...parts.slice(-3)] : parts;
    const crumbs = shown.map((p, i) => i === shown.length - 1
        ? `<b>${esc(p)}</b>` : `<span>${esc(p)}</span>`).join('<i>/</i>');
    const full = (root ? root.replace(/\/+$/, '') + '/' : '') + folder;
    return `<div class="sv-fold" title="${esc(full || root)}">
        <svg class="ui-ico"><use href="#i-folder"/></svg>
        <span class="sv-fold-path">${crumbs || `<b>${esc(root || 'library')}</b>`}</span>
        ${folder ? `<button class="sv-fold-only" onclick="svOnlyFolder('${escJs(folder)}')"
                            title="List every file of this folder">all in this folder</button>` : ''}
    </div>`;
}

function svFileLine(r, name) {
    const date = r.file_date
        ? new Date(r.file_date * 1000).toLocaleDateString(undefined,
            {year: 'numeric', month: '2-digit', day: '2-digit'}) : '';
    const words = Number(r.file_word_count) > 0
        ? plural(r.file_word_count, 'word') : '';
    const tx = r.has_transcript !== false;
    const tip = [r.filename, r.file_rate ? `${r.file_rate}× real time` : '']
        .filter(Boolean).join(' · ');
    return `<div class="sv-fl" title="${esc(tip)}"
                 onclick="svGoto('${escJs(r.filename)}', 0, '${escJs(r.file_dir || '')}')">
        <i class="sv-fl-dot${tx ? ' tx' : ''}" title="${tx ? 'Transcribed' : 'Not transcribed yet'}"></i>
        <span class="sv-fl-name">${esc(name)}</span>
        <span class="sv-fl-col">${esc(words)}</span>
        <span class="sv-fl-col">${r.file_duration > 0 ? esc(svFmt(r.file_duration)) : ''}</span>
        <span class="sv-fl-col">${r.file_size ? esc(svFmtSize(r.file_size)) : ''}</span>
        <span class="sv-fl-col">${esc(date)}</span>
        <span class="sv-fl-state${tx ? ' tx' : ''}">${tx ? 'transcript' : 'no transcript'}</span>
        <span class="sv-fl-open">open ›</span>
    </div>`;
}

/* "all in this folder": the folder becomes the file filter. */
function svOnlyFolder(folder) {
    const input = document.getElementById('svFileFilter');
    if (input) input.value = folder + '/';
    svRun();
}


function svFmtSize(bytes) {
    if (bytes >= 1e9) return (bytes / 1e9).toFixed(1) + ' GB';
    if (bytes >= 1e6) return Math.round(bytes / 1e6) + ' MB';
    return Math.max(1, Math.round(bytes / 1e3)) + ' kB';
}

function svAppendFooter() {
    const res = svListBox();
    document.getElementById('svFooter')?.remove();
    const remainder = Math.max(0, svState.total - svState.offset);
    const txt = svState.done
        ? (svState.offset ? `All ${svState.total.toLocaleString()} hits loaded` : '')
        : `Loading more… (${remainder.toLocaleString()} left)`;
    if (txt) res.insertAdjacentHTML('beforeend',
        `<div class="sv-more" id="svFooter">${esc(txt)}</div>`);
}

// ── Facets & heatmap ────────────────────────────────────────────────────

async function svLoadFacets(seq) {
    try {
        const r = await fetch(`/api/search/facets?${svParams()}&buckets=${SV_BUCKETS}`);
        const data = await r.json();
        if (seq !== svState.seq) return;
        svRenderFacets(data);
    } catch (e) {
        if (seq !== svState.seq) return;
        // The facets are extra (honest file count, hit bands,
        // map data basis). If they fail, the hit list stays
        // fully intact — no error message over the search is warranted.
        svState.facets = null;
    }
}

/* The facets no longer have their own display.
 *
 * They once filled the left sidebar: one line per file with
 * hit count and hit band. The sidebar is gone —
 * the same information now lives in two better places: the
 * "recordings" view (band per file, large enough to click) and the
 * "hit map" tab (the whole folder tree at once).
 *
 * They're still fetched unchanged, because they deliver three things the
 * hit list doesn't have: the HONEST file count (the loaded page only
 * knows its own files), the hit count per file across all pages, and
 * the bands.
 */
function svRenderFacets(data) {
    svState.facets = data;
    /* `files` MUST be a list and `fileCount` a number.
     *
     * The old version bailed out early on `data.unavailable` and on an empty
     * list — when it was rebuilt into the tab, those two guards were lost,
     * and so `fileCount` could become undefined: a response without
     * `file_count` (error case, missing FTS database) then made the
     * next `svSetStatus` throw "Cannot read properties of undefined".
     * The catch in `svLoadMore` then wrote "Error: …" over the
     * completed hit list — a failure of the EXTRA query thus took
     * the actual answer down with it. */
    const files = Array.isArray(data.files) ? data.files : [];
    svState.fileCount = Number(data.file_count) || files.length || 0;
    svSetStatus(svState.total, svState.fileCount);
    const number = document.getElementById('svTabCount');
    if (number) {
        number.textContent = files.length
            ? `${plural(svState.fileCount, 'recording')} affected`
            : '';
    }
    // The recordings view relies on the bands — if they arrive only after
    // the hit list, it needs to redraw once.
    if (svState.tab === 'text' && svState.view === 'recordings'
            && svState.hits.length) {
        svRedrawRecordings();
    }
    if (svState.tab === 'map') svMapFillTab(true);
}

function svFmt(sec) {
    sec = Math.max(0, Math.round(sec || 0));
    const h = Math.floor(sec / 3600), m = Math.floor((sec % 3600) / 60), s = sec % 60;
    return h ? `${h}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`
             : `${m}:${String(s).padStart(2, '0')}`;
}

// Click on a file in the sidebar: filter, clicking again cancels it.
function svPinFile(filename) {
    svState.pinned = (svState.pinned === filename) ? '' : filename;
    svShowPin();
    svRun();
}

function svClearFilters() {
    svState.pinned = '';
    document.getElementById('svFileFilter').value = '';
    svSetSpeaker('');
    svRun();
}

// Jump into the transcript — closes the overlay so the player is visible.
function svGoto(filename, start, fileDir) {
    closeSearchView();
    goToSearchResult(filename, start, fileDir);
}

// ══════════════════════════════════════════════════════════════════════════
//  "Hit map" tab — the whole folder tree at once
// ══════════════════════════════════════════════════════════════════════════
//
// The old map always showed EXACTLY ONE level: click a folder, one level
// deeper, breadcrumb back. For a layout like
// `media/archive/channels/channel-a/downloaded_videos/…` that's
// five clicks before a single file even appears — and along the way
// you never see where the hits sit in the OTHER branches.
//
// This map draws the tree nested: each folder tile contains
// its children's tiles, as deep as the space allows. That's also
// the original WinDirStat principle the old map claims to follow —
// it had just left out the nesting.
//
// The depth is limited by AREA, not a counter: wherever a tile gets too
// small to make out anything, the nesting stops. A fixed
// depth value would cut off too early or too late depending on the layout.
const SV_MAP_MIN_W = 44;    // narrower than this is no longer subdivided
const SV_MAP_MIN_H = 30;
const SV_MAP_HEAD  = 16;    // height of a folder tile's name band

let _svMapTree = null;
let _svMapPath = [];        // node chain from the root — the zoom

async function svMapFillTab(onlyIfEmpty) {
    const pane = document.getElementById('svPaneMap');
    if (!pane) return;
    let files = svState.facets?.files || [];

    // The facets can be missing even though there are hits: a detour into
    // the file tab starts its own search, which discards them. Without
    // this catch-up, the map was then left empty and claimed there were
    // no hits — in the text tab they were still visible.
    if (!files.length && svState.query.length >= 2) {
        pane.innerHTML = '<div class="sv-empty">Building map…</div>';
        await svLoadFacets(svState.seq);
        // The response in turn calls back in here (svRenderFacets) — if it
        // already built the map, a second pass would be pure duplication.
        if (_svMapTree) return;
        files = svState.facets?.files || [];
    }
    if (!files.length) {
        pane.innerHTML = '<div class="sv-empty">Search for something first — '
            + 'the map shows where its word hits are in the folder tree.</div>';
        _svMapTree = null;
        _svMapPath = [];
        return;
    }
    if (onlyIfEmpty && _svMapTree) return;

    _svMapTree = (typeof smBuildTree === 'function') ? smBuildTree(files) : null;
    if (!_svMapTree) {
        pane.innerHTML = '<div class="sv-empty">Map not available.</div>';
        return;
    }
    _svMapPath = [_svMapTree];
    pane.innerHTML = `
        <div class="sv-map-bar">
            <span class="sv-map-crumb" id="svMapCrumb"></span>
            <span class="sv-map-info" id="svMapInfo"></span>
            <span class="vx-gap"></span>
            <span class="sv-map-legend" title="The brighter a tile, the more word hits in it — the full-text index's counts (the meaning search has none)">fewer
                <i></i> more word hits</span>
            <div class="vx-seg" role="group" aria-label="How deep the map goes">
                <button class="vx-seg-btn${_svMapDeep ? ' active' : ''}" data-deep="1"
                        onclick="svMapSetDeep(true)" title="Folders show what is inside them">All levels</button>
                <button class="vx-seg-btn${_svMapDeep ? '' : ' active'}" data-deep="0"
                        onclick="svMapSetDeep(false)" title="One level — a folder is one tile">One level</button>
            </div>
        </div>
        <div class="sv-map-area" id="svMapArea"></div>
        <div class="sv-map-hint">Click a folder to zoom in · a file opens its transcript</div>`;
    svMapDraw();

    // The area's size is only known after it's shown — for a tab that's
    // just becoming visible, clientWidth is still 0 in the same pass.
    requestAnimationFrame(() => svMapDraw());
}

function svMapDraw() {
    const area = document.getElementById('svMapArea');
    if (!area || !_svMapPath.length) return;
    const W = area.clientWidth, H = area.clientHeight;
    if (W < 20 || H < 20) return;

    const nodes = _svMapPath[_svMapPath.length - 1];
    const low = _svMapDeep;
    const childList = smChildren(nodes);
    // How bright is "most": with all levels drawn, the busiest single
    // recording — measured against the biggest folder, every file came out
    // equally dim; with one level, the biggest tile of that level
    const maxHits = Math.max(1, low ? svMapMostInAFile(nodes)
                                    : Math.max(...childList.map(k => k.hits)));
    const pieces = [];
    _svMapTiles = [];
    svMapLevel(childList, 0, 0, W, H, [], maxHits, low, pieces);
    area.innerHTML = pieces.join('');

    svMapDrawCrumb();

    const info = document.getElementById('svMapInfo');
    if (info) {
        const files = svMapCount(nodes);
        // The facets are capped to the files with the most hits. Without
        // a note, the map would report a smaller hit count than the header
        // above it — and you'd look for the bug in yourself. Only at the
        // top level: further down, the excerpt is intentional anyway.
        const f = svState.facets || {};
        const capped = f.files_truncated && _svMapPath.length === 1;
        const missing = capped
            ? Math.max(0, (f.file_count || 0) - (f.files || []).length) : 0;
        const missingHits = capped
            ? Math.max(0, (f.total || 0) - _svMapTree.hits) : 0;
        info.innerHTML = `<b>${nodes.hits.toLocaleString()}</b> hit${nodes.hits === 1 ? '' : 's'}`
            + ` in <b>${files.toLocaleString()}</b> file${files === 1 ? '' : 's'}`
            + (missing
                ? ` <span class="sv-map-lid">— ${plural(missing, 'more file')} `
                  + `with ${plural(missingHits, 'hit')} `
                  + `${missing === 1 ? 'is' : 'are'} not shown (facets cap)</span>`
                : '');
    }
}

// Breadcrumb: the way back out of the zoom. Without it, getting out of a deeply
// nested folder would only be possible by rebuilding the map.
function svMapDrawCrumb() {
    const el = document.getElementById('svMapCrumb');
    if (!el) return;
    el.innerHTML = _svMapPath.map((n, i) =>
        `<span class="sv-crumb${i === _svMapPath.length - 1 ? ' here' : ''}"
               onclick="svMapLevelBack(${i})">${esc(n.name)}</span>`)
        .join('<span class="sv-crumb-arrow">›</span>');
}

function svMapLevelBack(i) {
    if (i < 0 || i >= _svMapPath.length) return;
    _svMapPath = _svMapPath.slice(0, i + 1);
    svMapDraw();
}

function svMapCount(node) {
    if (node.type === 'file') return 1;
    let n = 0;
    for (const k of node.kids.values()) n += svMapCount(k);
    return n;
}

/* The most hits a single recording below `node` has. */
function svMapMostInAFile(node) {
    if (node.type === 'file') return node.hits;
    let most = 0;
    for (const k of node.kids.values()) most = Math.max(most, svMapMostInAFile(k));
    return most;
}

/* "All levels" or one: whether a folder shows what is inside it. */
let _svMapDeep = true;

function svMapSetDeep(deep) {
    _svMapDeep = !!deep;
    document.querySelectorAll('#svPaneMap .vx-seg-btn').forEach(b =>
        b.classList.toggle('active', (b.dataset.deep === '1') === _svMapDeep));
    svMapDraw();
}

/* How bright a tile is: the accent, mixed in by its share of the hits —
 * the root on a square-root scale, so a folder with a tenth of the hits
 * still reads as more than nothing. One hue for the whole map, like the
 * rest of the app; the folder's place says which one it is. */
function svMapHeat(hits, maxHits) {
    return Math.round(14 + 66 * Math.sqrt(Math.max(0, hits) / Math.max(1, maxHits)));
}

/* `trail`: the folder nodes between the current zoom node and this level.
 * Each folder tile registers its full node chain in `_svMapTiles`, and its
 * click zooms along exactly that chain. */
let _svMapTiles = [];

function svMapLevel(childList, x, y, w, h, trail, maxHits, low, out) {
    for (const st of smSquarify(childList, x, y, w, h)) {
        const n = st.node;
        const heat = svMapHeat(Math.min(n.hits, maxHits), maxHits);
        const box = `left:${st.x}px;top:${st.y}px;width:${st.w}px;height:${st.h}px;--heat:${heat}%`;

        if (n.type === 'file') {
            out.push(`<div class="sv-tile file${heat >= 56 ? ' hot' : ''}" title="${esc(n.rel)} · ${plural(n.hits, 'hit')}"
                 style="${box}"
                 onclick="svGoto('${escJs(n.rel)}', 0, '${escJs(n.dir || '')}')">
                 ${st.w > 52 && st.h > 18
                    ? `<span class="sv-tile-txt">${esc(svShortName(n.name))}</span>` : ''}
                 ${st.w > 52 && st.h > 34
                    ? `<span class="sv-tile-n">${n.hits.toLocaleString()}</span>` : ''}
            </div>`);
            continue;
        }

        // A folder that holds one folder only is one frame for the whole
        // chain — `archive / channels / channel-a` instead of three
        // name bands stacked on top of each other
        let deep = n;
        const chain = [n.name];
        const path = trail.concat([n]);              // node chain down to `deep`
        for (;;) {
            const kids = smChildren(deep);
            if (kids.length !== 1 || kids[0].type !== 'dir') break;
            deep = kids[0];
            chain.push(deep.name);
            path.push(deep);
        }
        const tile = _svMapTiles.push(path) - 1;

        // Folder. If another subdivision still fits, it's drawn —
        // otherwise the tile stays a solid block. Clicking zooms in
        // in both cases.
        const divisible = low && st.w >= SV_MAP_MIN_W
                        && st.h >= SV_MAP_MIN_H + SV_MAP_HEAD;
        const name = chain.join(' / ');
        out.push(`<div class="sv-tile folder${divisible ? ' open' : ''}${heat >= 56 ? ' hot' : ''}"
             title="${esc(name)} · ${plural(n.hits, 'hit')} — click to zoom in"
             style="${box}"
             onclick="event.stopPropagation();svMapInto(${tile})">
             ${st.w > 46
                ? `<span class="sv-tile-head">${esc(name)}<b>${n.hits.toLocaleString()}</b></span>` : ''}
        </div>`);

        if (divisible) {
            svMapLevel(smChildren(deep),
                         st.x + 2, st.y + SV_MAP_HEAD + 1,
                         st.w - 4, st.h - SV_MAP_HEAD - 3,
                         path, maxHits, low, out);
        }
    }
}

/* Click on a folder: ZOOM IN.
 *
 * The first attempt took over the folder name as a file filter and jumped into
 * the text tab. That answered a question nobody had asked — anyone
 * clicking a folder on a map wants to look at it.
 *
 * With the nested display several levels are on screen at once, and a name
 * is ambiguous (two folders named "2024" in different branches). Looking it
 * up by (depth, name) picked the first branch that had one — clicking
 * `B / 2024` zoomed into `A/2024`. The tile therefore carries its own node
 * chain from the current zoom node (`_svMapTiles`, rebuilt on every draw).
 */
function svMapInto(tile) {
    const path = _svMapTiles[tile];
    if (!Array.isArray(path) || !path.length) return;
    _svMapPath = _svMapPath.concat(path);
    svMapDraw();
}

// Clickable rows that aren't buttons (they hold buttons of their own): Enter
// and Space click them, as on a button. Each carries role="button",
// tabindex="0" and data-keyclick.
if (typeof document !== 'undefined' && document.addEventListener) {
    document.addEventListener('keydown', (e) => {
        const el = e.target;
        if (!el || !el.matches || !el.matches('[data-keyclick]')) return;
        if (e.key !== 'Enter' && e.key !== ' ') return;
        e.preventDefault();
        el.click();
    });
}
