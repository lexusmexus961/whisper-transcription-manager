/* ══════════════════════════════════════════════════════════════════════════
 *  REVIEW — listening through every voice of one person
 *
 *  A list to listen through every voice of a person with autoplay, to
 *  check the names — like the assignment's play all, optionally only the
 *  first seconds of each — with many filters.
 *
 *  Opened from a row of the People page. Left the filters, on top one
 *  player for the sequence (the browser's own, so right-click still saves
 *  the clip), below the list. The doubtful voices come first: the ones
 *  nearer to someone else, then the least fitting (core/voice_review.py
 *  says how fit is measured). Each voice can lose the name, go to the
 *  trash, or be named as someone else — one entry of the History each.
 *  The last one can be undone right in the head of the list: a toast per
 *  decision piled up over the list's own buttons when deciding quickly.
 *
 *  Keys while the list is open: Space play/pause · ←/→ previous/next ·
 *  N not this person · T trash · A name as … · Esc stop.
 * ══════════════════════════════════════════════════════════════════════════ */

const REVIEW = {
    person: null,
    filters: {},
    voices: [],             // what is loaded, from offset 0
    total: 0,
    counts: null,
    legacy: false,
    loading: false,
    seq: 0,
    pos: -1,                // the voice in the player
    running: false,         // the sequence goes on by itself
    per: 0,                 // seconds per voice, 0 = all of it
    auto: true,
    picked: new Set(),      // checked for a decision on many
    decided: new Map(),     // id → what was decided here
    last: null,             // the last decision: {decision, ids, text} — undone from the head
};

const REVIEW_PAGE = 100;
const REVIEW_PER = [0, 1, 2, 3, 5, 10];
const REVIEW_MEMO = 'whisperx.voices.review';

function reviewDefaults() {
    return {sort: 'fit', source: 'all', closer: false, fit_max: '', date_from: '',
            date_to: '', folder: '', file: '', label: '', min_speech: '', max_speech: '',
            spans: 'all', per_recording: false};
}

function reviewLoadMemo() {
    try {
        const m = JSON.parse(localStorage.getItem(REVIEW_MEMO) || 'null');
        if (m) {
            if (REVIEW_PER.includes(m.per)) REVIEW.per = m.per;
            if (typeof m.auto === 'boolean') REVIEW.auto = m.auto;
        }
    } catch (e) { /* a private window: the defaults */ }
}

function reviewSaveMemo() {
    try { localStorage.setItem(REVIEW_MEMO, JSON.stringify({per: REVIEW.per, auto: REVIEW.auto})); }
    catch (e) { /* not remembered */ }
}

/* ── Open and close ─────────────────────────────────────────────────────── */

function reviewOpen(name) {
    reviewLoadMemo();
    REVIEW.person = name;
    REVIEW.filters = reviewDefaults();
    REVIEW.voices = [];
    REVIEW.total = 0;
    REVIEW.pos = -1;
    REVIEW.running = false;
    REVIEW.picked.clear();
    REVIEW.decided.clear();
    const main = document.getElementById('peopleMain');
    const box = document.getElementById('peopleReview');
    if (!main || !box) return;
    main.hidden = true;
    box.hidden = false;
    box.innerHTML = reviewShellHtml();
    reviewWire();
    vxState('people', name);
    reviewLoad(true);
    // A step of its own: Back (Alt+Left) returns to all people
    if (typeof Shell !== 'undefined') Shell.step(reviewClose);
}

function reviewLeave() {
    if (typeof Shell !== 'undefined') Shell.leave(reviewClose); else reviewClose();
}

function reviewClose() {
    reviewStop();
    REVIEW.person = null;
    const main = document.getElementById('peopleMain');
    const box = document.getElementById('peopleReview');
    if (box) { box.hidden = true; box.innerHTML = ''; }
    if (main) main.hidden = false;
    if (typeof peopleLoad === 'function') peopleLoad();
}

function reviewIsOpen() {
    const box = document.getElementById('peopleReview');
    return !!(REVIEW.person && box && !box.hidden
              && document.getElementById('mapOverlay')?.classList.contains('visible')
              && typeof VX !== 'undefined' && VX.tab === 'people');
}

/* ── The page ───────────────────────────────────────────────────────────── */

function reviewShellHtml() {
    const f = REVIEW.filters;
    const name = esc(REVIEW.person);
    const per = REVIEW_PER.map(v => `<option value="${v}"${v === REVIEW.per ? ' selected' : ''}>${v ? v + ' s' : 'all of it'}</option>`).join('');
    return `<div class="vx-review">
    <div class="vx-x-form" id="rvForm">
        <div class="vx-form-section">
            <div class="vx-crumbs"><button class="vx-crumb" onclick="reviewLeave()" title="All people (Esc · Alt+←)">People</button><span aria-hidden="true">›</span><b>${name}</b></div>
        </div>
        <div class="vx-form-section">
            <div class="vx-form-title">Order</div>
            <div class="vx-row"><select id="rvSort">
                <option value="fit" title="Nearer to someone else first, then the least fitting">Doubtful first</option>
                <option value="date_desc">Newest recordings first</option>
                <option value="date_asc">Oldest recordings first</option>
                <option value="recording">By recording</option>
                <option value="speech_desc">Most speech first</option></select></div>
            <div class="vx-row check"><input type="checkbox" id="rvPerRec"><label for="rvPerRec">One voice per recording</label></div>
        </div>
        <div class="vx-form-section">
            <div class="vx-form-title">Which voices</div>
            <div class="vx-radio">
                <label><input type="radio" name="rvSource" value="all" checked><span><b>All</b> with the name</span></label>
                <label><input type="radio" name="rvSource" value="direct"><span><b>Named directly</b> — by you, on the map or in Assign</span></label>
                <label><input type="radio" name="rvSource" value="spread"><span><b>Named by spreading</b> — the machine's guess</span></label>
            </div>
            <div class="vx-row check"><input type="checkbox" id="rvCloser"><label for="rvCloser" id="rvCloserLabel">Only those nearer to someone else</label></div>
            <div class="vx-row"><label for="rvFit" title="How well the voice fits ${name}: its best similarity to voices named ${name} directly in other recordings">Fit at most</label>
                <input type="number" id="rvFit" step="0.05" min="0" max="1" placeholder="any"></div>
            <div class="vx-row" id="rvSpansRow" hidden><label title="A voice without stored speech spans has the window's first 30 s as its sample, which can hold other speakers">Speech spans</label>
                <select id="rvSpans"><option value="all">all</option><option value="with">stored</option><option value="without">missing</option></select></div>
        </div>
        <div class="vx-form-section">
            <div class="vx-form-title">When and where</div>
            <div class="vx-row"><label>Recorded from</label><input type="date" id="rvFrom"></div>
            <div class="vx-row"><label>to</label><input type="date" id="rvTo"></div>
            <div class="vx-row"><label title="The path starts with this">Folder</label><input type="text" id="rvFolder" placeholder="e.g. interviews/2024"></div>
            <div class="vx-row"><label>File name contains</label><input type="text" id="rvFile"></div>
            <div class="vx-row"><label>Speaker label</label><input type="text" id="rvLabel" placeholder="e.g. SPEAKER_07"></div>
            <div class="vx-row"><label>Speech (s)</label><input type="number" id="rvMin" min="0" step="1" placeholder="min"> –
                <input type="number" id="rvMax" min="0" step="1" placeholder="max"></div>
            <div class="vx-row"><button class="vx-btn" onclick="reviewResetFilters()">Reset the filters</button></div>
        </div>
    </div>
    <div class="vx-review-main">
        <div class="vx-review-head">
            <div class="vx-review-title"><b>${name}</b> <span class="map-small" id="rvCounts"></span></div>
            <div class="vx-review-player">
                <audio id="rvAudio" controls preload="none"></audio>
                <div class="vx-review-keys">
                    <button class="map-mini" onclick="reviewStep(-1)" title="Previous voice (←)">⏮</button>
                    <button class="map-mini" id="rvPlay" onclick="reviewToggle()" title="Play / pause the sequence (Space)">▶</button>
                    <button class="map-mini" onclick="reviewStep(1)" title="Next voice (→)">⏭</button>
                    <button class="map-mini" onclick="reviewStop()" title="Stop (Esc)">⏹</button>
                    <label class="map-small" title="How much of each voice is played before the next">per voice
                        <select id="rvPer">${per}</select></label>
                    <label class="map-small" title="Go on to the next voice by itself"><input type="checkbox" id="rvAuto"${REVIEW.auto ? ' checked' : ''}> Autoplay</label>
                    <span class="map-small vx-review-pos" id="rvPos"></span>
                </div>
                <div class="vx-review-now" id="rvNow"></div>
            </div>
            <div class="vx-review-last map-small" id="rvLast" hidden></div>
            <div class="vx-review-bulk" id="rvBulk" hidden></div>
        </div>
        <div class="vx-review-list" id="rvList"><div class="vx-empty">Loading …</div></div>
        <div class="map-small vx-review-keyhelp"><kbd>Space</kbd> play/pause · <kbd>←</kbd><kbd>→</kbd> previous/next ·
            <kbd>N</kbd> not ${name} · <kbd>T</kbd> trash · <kbd>A</kbd> name as … · <kbd>Esc</kbd> stop</div>
    </div>
    </div>`;
}

function reviewWire() {
    const form = document.getElementById('rvForm');
    let wait = null;
    const soon = ms => { clearTimeout(wait); wait = setTimeout(() => { reviewReadFilters(); reviewLoad(true); }, ms); };
    form.addEventListener('change', e => soon(e.target.type === 'text' ? 0 : 150));
    form.addEventListener('keydown', e => {
        if (e.key === 'Enter' && e.target.tagName === 'INPUT') { e.preventDefault(); soon(0); }
    });
    document.getElementById('rvSort').value = REVIEW.filters.sort;
    document.getElementById('rvPer').addEventListener('change', e => {
        REVIEW.per = Number(e.target.value) || 0;
        reviewSaveMemo();
    });
    document.getElementById('rvAuto').addEventListener('change', e => {
        REVIEW.auto = e.target.checked;
        reviewSaveMemo();
    });
    const audio = document.getElementById('rvAudio');
    audio.addEventListener('ended', () => {
        if (REVIEW.running && REVIEW.auto) reviewStep(1); else reviewButtons();
    });
    // Only the first seconds — through timeupdate, not a timer: a timer
    // keeps running while the sample is still loading
    audio.addEventListener('timeupdate', () => {
        if (!REVIEW.running || !REVIEW.per) return;
        if (audio.currentTime >= REVIEW.per) {
            audio.pause();
            if (REVIEW.auto) reviewStep(1); else { REVIEW.running = false; reviewButtons(); }
        }
    });
    audio.addEventListener('play', reviewButtons);
    audio.addEventListener('pause', reviewButtons);
    // Dragging the seek bar means: hear THIS voice — the sequence waits
    audio.addEventListener('seeking', () => {
        if (REVIEW.running && audio.currentTime > 0.5) { REVIEW.running = false; reviewButtons(); }
    });
}

function reviewReadFilters() {
    const v = id => (document.getElementById(id) || {}).value || '';
    const radio = name => document.querySelector(`input[name="${name}"]:checked`)?.value || 'all';
    REVIEW.filters = {
        sort: v('rvSort') || 'fit', source: radio('rvSource'),
        closer: !!document.getElementById('rvCloser')?.checked,
        per_recording: !!document.getElementById('rvPerRec')?.checked,
        fit_max: v('rvFit'), date_from: v('rvFrom'), date_to: v('rvTo'),
        folder: v('rvFolder').trim(), file: v('rvFile').trim(), label: v('rvLabel').trim(),
        min_speech: v('rvMin'), max_speech: v('rvMax'), spans: v('rvSpans') || 'all',
    };
}

function reviewResetFilters() {
    const keep = REVIEW.filters.sort;
    REVIEW.filters = {...reviewDefaults(), sort: keep};
    const form = document.getElementById('rvForm');
    form.querySelectorAll('input[type=text], input[type=number], input[type=date]').forEach(i => { i.value = ''; });
    form.querySelectorAll('input[type=checkbox]').forEach(i => { i.checked = false; });
    const all = form.querySelector('input[name="rvSource"][value="all"]');
    if (all) all.checked = true;
    const sp = document.getElementById('rvSpans');
    if (sp) sp.value = 'all';
    reviewLoad(true);
}

/* ── Loading ────────────────────────────────────────────────────────────── */

async function reviewLoad(reset) {
    if (!REVIEW.person) return;
    if (reset) {
        reviewStop();
        REVIEW.voices = [];
        REVIEW.pos = -1;
        REVIEW.picked.clear();
    }
    const mine = ++REVIEW.seq;
    REVIEW.loading = true;
    const p = new URLSearchParams({person: REVIEW.person, offset: REVIEW.voices.length, limit: REVIEW_PAGE});
    for (const [k, v] of Object.entries(REVIEW.filters)) {
        if (v === '' || v === false || v === 'all') continue;
        p.set(k, v === true ? '1' : v);
    }
    let d;
    try {
        const r = await fetch('/api/voices/review?' + p);
        d = await r.json();
        if (!r.ok || d.error) throw new Error(d.error || `HTTP ${r.status}`);
    } catch (e) {
        if (mine !== REVIEW.seq) return;
        REVIEW.loading = false;
        const list = document.getElementById('rvList');
        if (list) list.innerHTML = `<div class="vx-note err">The voices could not be loaded: ${esc(e.message || String(e))}</div>`;
        return;
    }
    if (mine !== REVIEW.seq) return;
    REVIEW.loading = false;
    REVIEW.total = d.total;
    REVIEW.counts = d.counts;
    REVIEW.legacy = d.legacy;
    REVIEW.voices = REVIEW.voices.concat(d.voices || []);
    reviewDrawCounts();
    reviewDrawList();
    reviewButtons();
}

function reviewDrawCounts() {
    const c = REVIEW.counts;
    const el = document.getElementById('rvCounts');
    if (!el || !c) return;
    el.textContent = `${vxN(REVIEW.total)} of ${vxN(c.voices)} voice${c.voices === 1 ? '' : 's'} · ${vxN(c.recordings)} recording${c.recordings === 1 ? '' : 's'} · `
        + `${vxN(c.direct)} named directly, ${vxN(c.spread)} by spreading`
        + (c.closer ? ` · ${vxN(c.closer)} nearer to someone else` : '');
    const cl = document.getElementById('rvCloserLabel');
    if (cl) cl.textContent = `Only those nearer to someone else${c.closer ? ` (${vxN(c.closer)})` : ''}`;
    const sp = document.getElementById('rvSpansRow');
    if (sp) sp.hidden = !REVIEW.legacy;
}

function reviewFitBar(v) {
    if (v.fit === null || v.fit === undefined) {
        return '<span class="map-small" title="No other recording with a voice named directly to compare with">fit —</span>';
    }
    const p = Math.max(0, Math.min(1, (v.fit - 0.4) / 0.6));
    const cls = v.fit >= 0.8 ? 'hi' : (v.fit < 0.65 ? 'lo' : '');
    return `<span class="vx-fit ${cls}" title="Best similarity to a voice named ${esc(REVIEW.person)} directly in another recording">
        <span class="vx-fit-bar"><span style="width:${Math.round(p * 100)}%"></span></span>${v.fit.toFixed(2)}</span>`;
}

function reviewRowHtml(v, j) {
    const done = REVIEW.decided.get(v.id);
    const now = j === REVIEW.pos;
    // Always eleven cells of fixed width — every row is its own grid, and
    // only fixed columns line up from row to row. What was decided takes
    // the place of "nearer to": a decided voice doesn't need the hint.
    const other = done ? `<span class="vx-tag undone" title="Decided here — the toast or the History undoes it">${esc(done)}</span>`
        : (v.other ? `<span class="vx-rv-other" title="Nearer to ${esc(v.other)}'s centroid (${v.other_sim}) than to ${esc(REVIEW.person)}'s (${v.own})">nearer to ${esc(v.other)}</span>` : '<span></span>');
    return `<div class="vx-rv-row${now ? ' now' : ''}${done ? ' done' : ''}" id="rv-${v.id}" data-j="${j}">
        <input type="checkbox" class="vx-rv-pick" ${REVIEW.picked.has(v.id) ? 'checked' : ''}
               onclick="event.stopPropagation();reviewPick(${v.id}, this.checked)" title="Choose — for a decision on many">
        <button class="map-mini" onclick="reviewPlayAt(${j})" title="Play from here">${now && REVIEW.running ? '⏸' : '▶'}</button>
        <span class="vx-rv-label">${esc(v.label)}</span>
        <span class="vx-rv-file" title="${esc(v.display)}">${esc(v.file)}</span>
        <span class="vx-rv-date map-small">${esc(v.date || '')}</span>
        <span class="vx-rv-speech map-small" title="Speech in this voice's window">${Math.round(v.speech)} s</span>
        <span class="vx-tag${v.direct ? '' : ' spread'}" title="${v.direct ? 'Named directly' : 'Named by spreading'}">${v.direct ? 'named' : 'spread'}</span>
        ${reviewFitBar(v)}
        ${other}
        ${v.spans ? '<span></span>' : '<span class="vx-rv-old" title="No stored speech spans: the sample is the window\'s first 30 s and can hold other speakers. Renew voices (queue editor → From library) makes it anew and keeps the name">no spans</span>'}
        <span class="vx-rv-acts">
            <button class="map-mini" onclick="reviewDecide([${v.id}], 'unname')" title="Not ${esc(REVIEW.person)} — remove the name (N)">✕</button>
            <button class="map-mini map-mini-danger" onclick="reviewDecide([${v.id}], 'trash')" title="To the trash (T)">🗑</button>
            <button class="map-mini" onclick="reviewNameAs([${v.id}], this)" title="Name as someone else (A)">↗</button>
            <button class="map-mini" onclick="mapShowVoices([${v.id}], '${escJs(REVIEW.person)} · ${escJs(v.label)}')" title="Show on the map">⌖</button>
        </span>
    </div>`;
}

function reviewDrawList() {
    const list = document.getElementById('rvList');
    if (!list) return;
    if (!REVIEW.voices.length) {
        list.innerHTML = `<div class="vx-empty">${REVIEW.total || !REVIEW.counts ? 'Loading …' : 'No voice matches the filters.'}</div>`;
        return;
    }
    const more = REVIEW.voices.length < REVIEW.total;
    list.innerHTML = REVIEW.voices.map(reviewRowHtml).join('')
        + (more ? `<div style="text-align:center;padding:12px"><button class="vx-btn" onclick="reviewLoad(false)">
            Load ${vxN(Math.min(REVIEW_PAGE, REVIEW.total - REVIEW.voices.length))} more (${vxN(REVIEW.voices.length)} of ${vxN(REVIEW.total)})</button></div>` : '');
    reviewDrawBulk();
}

function reviewRedrawRow(j) {
    const v = REVIEW.voices[j];
    const el = v && document.getElementById(`rv-${v.id}`);
    if (!el) return;
    const tmp = document.createElement('div');
    tmp.innerHTML = reviewRowHtml(v, j);
    el.replaceWith(tmp.firstElementChild);
}

/* ── The sequence ───────────────────────────────────────────────────────── */

function reviewPlayAt(j) {
    const audio = document.getElementById('rvAudio');
    if (!audio || j < 0) return;
    if (j >= REVIEW.voices.length) {
        // The end of what is loaded — the next page, then go on
        if (REVIEW.voices.length < REVIEW.total && !REVIEW.loading) {
            reviewLoad(false).then(() => { if (j < REVIEW.voices.length) reviewPlayAt(j); });
        } else {
            reviewStop();
        }
        return;
    }
    if (j === REVIEW.pos && !audio.paused && REVIEW.running) { reviewToggle(); return; }
    const old = REVIEW.pos;
    REVIEW.pos = j;
    REVIEW.running = true;
    const v = REVIEW.voices[j];
    if (audio.dataset.vid !== String(v.id)) {
        audio.dataset.vid = String(v.id);
        audio.src = v.url;
    } else {
        audio.currentTime = 0;
    }
    audio.play().catch(() => { /* a newer click took over */ });
    if (old >= 0) reviewRedrawRow(old);
    reviewRedrawRow(j);
    document.getElementById(`rv-${v.id}`)?.scrollIntoView({block: 'nearest'});
    reviewNow();
    reviewButtons();
    reviewWarm(j);
}

/* The samples are cut when asked for — asking for the next two ahead
 * puts them in the server's cache, so the sequence doesn't wait. */
function reviewWarm(j) {
    for (const k of [j + 1, j + 2]) {
        const v = REVIEW.voices[k];
        if (v && !v._warm) {
            v._warm = true;
            fetch(v.url).catch(() => { /* only a head start */ });
        }
    }
}

function reviewStep(step) {
    let j = REVIEW.pos < 0 ? 0 : REVIEW.pos + step;
    if (j < 0) j = 0;
    reviewPlayAt(j);
}

function reviewToggle() {
    const audio = document.getElementById('rvAudio');
    if (!audio) return;
    if (REVIEW.pos < 0) { reviewPlayAt(0); return; }
    if (REVIEW.running && !audio.paused) {
        REVIEW.running = false;
        audio.pause();
    } else {
        REVIEW.running = true;
        if (REVIEW.per && audio.currentTime >= REVIEW.per) audio.currentTime = 0;
        audio.play().catch(() => {});
    }
    reviewRedrawRow(REVIEW.pos);
    reviewButtons();
}

function reviewStop() {
    REVIEW.running = false;
    const audio = document.getElementById('rvAudio');
    if (audio) audio.pause();
    if (REVIEW.pos >= 0) reviewRedrawRow(REVIEW.pos);
    reviewButtons();
}

function reviewButtons() {
    const audio = document.getElementById('rvAudio');
    const play = document.getElementById('rvPlay');
    if (play) play.textContent = REVIEW.running && audio && !audio.paused ? '⏸' : '▶';
    const pos = document.getElementById('rvPos');
    if (pos) pos.textContent = REVIEW.pos >= 0 ? `${vxN(REVIEW.pos + 1)} / ${vxN(REVIEW.total)}` : `${vxN(REVIEW.total)} voice${REVIEW.total === 1 ? '' : 's'}`;
}

function reviewNow() {
    const el = document.getElementById('rvNow');
    const v = REVIEW.voices[REVIEW.pos];
    if (!el) return;
    if (!v) { el.innerHTML = ''; return; }
    el.innerHTML = `<span><b>${esc(v.label)}</b> · <span title="${esc(v.display)}">${esc(v.display)}</span>
        ${v.date ? ' · ' + esc(v.date) : ''} · ${Math.round(v.speech)} s speech · ${v.direct ? 'named directly' : 'named by spreading'}
        ${v.other ? ` · <span class="vx-rv-other">nearer to ${esc(v.other)}</span>` : ''}</span>
        <span class="vx-review-now-acts">
            <button class="vx-btn" onclick="reviewDecide([${v.id}], 'unname')">Not ${esc(REVIEW.person)} <kbd>N</kbd></button>
            <button class="vx-btn danger" onclick="reviewDecide([${v.id}], 'trash')">Trash <kbd>T</kbd></button>
            <button class="vx-btn" onclick="reviewNameAs([${v.id}], this)">Name as … <kbd>A</kbd></button>
        </span>`;
}

/* ── Deciding ───────────────────────────────────────────────────────────── */

function reviewPick(id, on) {
    if (on) REVIEW.picked.add(id); else REVIEW.picked.delete(id);
    reviewDrawBulk();
}

function reviewDrawBulk() {
    const el = document.getElementById('rvBulk');
    if (!el) return;
    const n = REVIEW.picked.size;
    el.hidden = !n;
    if (!n) { el.innerHTML = ''; return; }
    el.innerHTML = `<span><b>${vxN(n)}</b> chosen</span>
        <button class="vx-btn" onclick="reviewDecide([...REVIEW.picked], 'unname')">Not ${esc(REVIEW.person)}</button>
        <button class="vx-btn danger" onclick="reviewDecide([...REVIEW.picked], 'trash')">Trash</button>
        <button class="vx-btn" onclick="reviewNameAs([...REVIEW.picked], this)">Name as …</button>
        <button class="vx-btn" onclick="REVIEW.picked.clear();reviewDrawList()">Clear</button>`;
}

/* A small field to type the other name — with the known names offered. */
function reviewNameAs(ids, anchor) {
    document.getElementById('rvNamePop')?.remove();
    const pop = document.createElement('div');
    pop.className = 'vx-rv-namepop';
    pop.id = 'rvNamePop';
    pop.innerHTML = `<input type="text" id="rvNameAs" list="vxPeopleList" placeholder="Who is it?" autocomplete="off">
        <button class="vx-btn primary">Name</button>`;
    (anchor?.closest('.vx-rv-row, .vx-review-now, .vx-review-bulk') || document.getElementById('rvList')).appendChild(pop);
    const field = pop.querySelector('input');
    const go = () => {
        const to = field.value.trim();
        if (!to) return;
        pop.remove();
        reviewDecide(ids, 'name', to);
    };
    pop.querySelector('button').onclick = go;
    field.addEventListener('keydown', e => {
        if (e.key === 'Enter') { e.preventDefault(); go(); }
        if (e.key === 'Escape') { e.preventDefault(); e.stopPropagation(); pop.remove(); }
    });
    if (typeof vxPeopleNames === 'function') vxPeopleNames();
    field.focus();
}

async function reviewDecide(ids, action, to = '') {
    if (!ids.length) return;
    const before = REVIEW.voices[REVIEW.pos];
    let d;
    try {
        d = await vxJson('/api/voices/review/decide', {person: REVIEW.person, ids, action, to});
    } catch (e) {
        Toast.error('Not done: ' + (e.message || e));
        return;
    }
    const what = action === 'unname' ? `not ${REVIEW.person}` : (action === 'trash' ? 'trashed' : `→ ${to}`);
    for (const id of ids) {
        REVIEW.decided.set(id, what);
        REVIEW.picked.delete(id);
    }
    const n = d.changed || 0;
    REVIEW.last = {decision: d.decision, ids, text: `${vxN(n)} ${n === 1 ? 'voice' : 'voices'} ${action === 'unname'
        ? `no longer ${REVIEW.person}` : (action === 'trash' ? 'to the trash' : `named ${to}`)}`};
    reviewDrawLast();
    vxMapChanged();
    if (typeof historyRefresh === 'function') historyRefresh();
    if (typeof assignRecentLoad === 'function') assignRecentLoad();
    if (action === 'name' && typeof vxPeopleNames === 'function') vxPeopleNames(true);
    ids.forEach(id => {
        const j = REVIEW.voices.findIndex(v => v.id === id);
        if (j >= 0) reviewRedrawRow(j);
    });
    reviewDrawBulk();
    // Decided on the voice in the player: on to the next one
    if (before && ids.includes(before.id) && REVIEW.running) reviewStep(1);
}

function reviewDrawLast() {
    const el = document.getElementById('rvLast');
    if (!el) return;
    const l = REVIEW.last;
    el.hidden = !l;
    if (!l) { el.innerHTML = ''; return; }
    el.innerHTML = `Last: ${esc(l.text)}${l.decision
        ? ` <button class="vx-btn" onclick="reviewUndoLast()" title="Put it back — the History has every decision">Undo</button>` : ''}`;
}

async function reviewUndoLast() {
    const l = REVIEW.last;
    if (!l || !l.decision) return;
    const d = await vxUndo(l.decision);
    if (!d) return;
    REVIEW.last = null;
    for (const id of l.ids) REVIEW.decided.delete(id);
    l.ids.forEach(id => {
        const j = REVIEW.voices.findIndex(v => v.id === id);
        if (j >= 0) reviewRedrawRow(j);
    });
    reviewDrawLast();
}

/* ── Keys ───────────────────────────────────────────────────────────────── */

document.addEventListener('keydown', e => {
    if (!reviewIsOpen()) return;
    const t = e.target;
    const typing = t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.tagName === 'SELECT' || t.isContentEditable);
    if (e.key === 'Escape') {
        if (document.getElementById('rvNamePop')) { document.getElementById('rvNamePop').remove(); e.stopImmediatePropagation(); return; }
        if (REVIEW.running) { reviewStop(); e.preventDefault(); e.stopImmediatePropagation(); }
        return;
    }
    if (typing || e.ctrlKey || e.metaKey || e.altKey) return;
    const v = REVIEW.voices[REVIEW.pos];
    if (e.key === ' ') { e.preventDefault(); reviewToggle(); }
    else if (e.key === 'ArrowRight') { e.preventDefault(); reviewStep(1); }
    else if (e.key === 'ArrowLeft') { e.preventDefault(); reviewStep(-1); }
    else if ((e.key === 'n' || e.key === 'N') && v) { e.preventDefault(); reviewDecide([v.id], 'unname'); }
    else if ((e.key === 't' || e.key === 'T' || e.key === 'Delete') && v) { e.preventDefault(); reviewDecide([v.id], 'trash'); }
    else if ((e.key === 'a' || e.key === 'A') && v) {
        e.preventDefault();
        reviewNameAs([v.id], document.querySelector('#rvNow .vx-btn'));
    }
}, true);
