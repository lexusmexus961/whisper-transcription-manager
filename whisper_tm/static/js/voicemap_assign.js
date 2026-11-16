/* ══════════════════════════════════════════════════════════════════════════
 *  GUIDED ASSIGNMENT — the Assign tab
 *
 *  A guided path that covers most of the naming work in a few clicks.
 *
 *  The procedure is described at the top of `core/speaker_assignment.py`.
 *  This file is the path there, and the path is ONE question at a time:
 *
 *      "3.0 h · 12 speakers from 10 recordings"
 *      [samples]  [Who is this? ______]  [Apply]  [Trash]  [Skip]
 *
 *  Sorted by OPEN speaking time, not by size: a handful of clicks covers
 *  half the collection instead of thousands.
 *
 *  **Its own page.** It used to float over the map as a box you
 *  couldn't move, and the map under it reloaded after every decision —
 *  tens of MB on a large collection for a picture nobody looked at. Now a
 *  decision marks the map stale (`vxMapChanged`) and it reloads when shown.
 *
 *  **And it can go back.** Assigning many voices and noticing a bit later
 *  that one speaker should have been kept must be recoverable. The last
 *  decisions sit next to the question, each with Undo — and an undone
 *  decision comes back as the question it was, to be decided anew
 *  (`assignUndo`). The undo itself changes only what that click changed
 *  and nothing a later click decided (database/speaker.py).
 * ══════════════════════════════════════════════════════════════════════════ */

const ASSIGN_MEMO = 'assign.skipped';

const ASSIGN = {
    on: false,                   // the tab is on screen (assignOpen … assignClose)
    loaded: false,               // the remembered settings were read once
    fetchSeq: 0,                 // only the newest suggestion request may draw
    suggestion: null,
    running: false,
    /* Skipped ones, as the vector id of the group's largest voice.
     *
     * NOT as a group number: after every decision the groups are
     * re-clustered, and the numbers get reassigned. A remembered number
     * would afterward point to a different group — you'd blindly skip
     * one you've never seen. */
    skipped: new Set(),
    listens: false,              // the listeners on the player are already attached
    /* What the next suggestion is picked by — hours, speakers or
     * recordings. All three only count a group's OPEN portion. */
    sort_by: 'hours',
    progress: null,              // the last progress the server reported
    recent: [],                  // the last decisions, newest first
    open: new Set(),             // the decisions opened to their samples
    message: '',                 // what the last step did
    // Playing the samples one after another — the first N seconds each
    seq: {running: false, pos: 0, per: 3, auto: false, timer: null},
};

const ASSIGN_PER = [1, 2, 3, 5, 10, 0];       // seconds per sample; 0 = whole

const ASSIGN_BY = [
    ['hours',   'Hours',      'Where is the most open speaking time? The '
                             + 'fastest way to coverage.'],
    ['speaker',  'Speakers',   'Where do the most speakers cluster together? '
                             + 'Finds fragmented voices with many '
                             + 'short appearances.'],
    ['recordings', 'Recordings', 'Who appears in the most recordings? The '
                             + 'regulars — yields fewer hours per click, '
                             + 'but makes search across the collection '
                             + 'usable.'],
];

/* The skipped ones survive a reload, so the work can continue later.
 * The assignments themselves live in the database anyway. */
function assignLoadMemo() {
    try {
        const raw = localStorage.getItem(ASSIGN_MEMO);
        if (raw) ASSIGN.skipped = new Set(JSON.parse(raw));
        const n = localStorage.getItem(ASSIGN_MEMO + '.sort_by');
        if (ASSIGN_BY.some(x => x[0] === n)) ASSIGN.sort_by = n;
        const kept = localStorage.getItem(ASSIGN_MEMO + '.per');
        if (kept !== null && ASSIGN_PER.includes(Number(kept))) ASSIGN.seq.per = Number(kept);
        ASSIGN.seq.auto = localStorage.getItem(ASSIGN_MEMO + '.auto') === '1';
    } catch (e) { /* a broken entry is no reason to block the path */ }
}

/* Switch the sort order. The skipped ones stay put — they're a statement
 * about the GROUPS, not about the order in which they're asked about. */
function assignAfterSet(kind) {
    if (!ASSIGN_BY.some(x => x[0] === kind)) return;
    ASSIGN.sort_by = kind;
    try { localStorage.setItem(ASSIGN_MEMO + '.sort_by', kind); } catch (e) { /* full */ }
    assignFetch();
}

function assignSaveMemo() {
    try {
        localStorage.setItem(ASSIGN_MEMO, JSON.stringify([...ASSIGN.skipped]));
    } catch (e) { /* full storage must not stop assigning */ }
}

/* The tab left the screen (another tab, or the workspace closed — vxLeaveTab
 * in voices.js): the sequence and its 25 s timer stop, and a suggestion that
 * arrives later is kept but neither drawn into view nor autoplayed. */
function assignVisible() {
    return ASSIGN.on && (typeof VX === 'undefined' || VX.tab === 'assign')
        && !!document.getElementById('mapOverlay')?.classList.contains('visible');
}

/* The tab came on screen. A question already shown stays — switching to
 * the map and back must not throw away an asked-again question. */
function assignOpen() {
    const first = !ASSIGN.loaded;
    ASSIGN.loaded = true;
    ASSIGN.on = true;
    if (first) assignLoadMemo();
    if (typeof vxPeopleNames === 'function') vxPeopleNames();
    assignRecentLoad();
    // An asked-again question (from the History) is kept even on the first
    // open; otherwise the first open fetches, a later one keeps what's there.
    if (ASSIGN.suggestion && (ASSIGN.suggestion.asked || !first)) assignDraw(ASSIGN.suggestion);
    else assignFetch();
}

function assignClose() {
    ASSIGN.on = false;
    assignSeqStop();
    assignOnlyOnePlays(null);
}

async function assignFetch() {
    if (!ASSIGN.on) {
        // Off screen (e.g. Apply answered after the tab was left): the
        // question on hold is the one just decided — drop it, so the next
        // assignOpen asks the server afresh instead of showing it again.
        // An asked-again question (from the History) is not ours to drop.
        if (!(ASSIGN.suggestion && ASSIGN.suggestion.asked)) ASSIGN.suggestion = null;
        return;
    }
    // Only the newest request counts: Skip, then another "Ask by" before
    // the first answer is back must not show the older order's question.
    const mine = ++ASSIGN.fetchSeq;
    assignDraw({load_it: true});
    try {
        const p = new URLSearchParams({sort_by: ASSIGN.sort_by});
        const without = [...ASSIGN.skipped].join(',');
        if (without) p.set('without', without);
        const d = await (await fetch(
            '/api/speaker/assignment/suggestion?' + p)).json();
        if (mine !== ASSIGN.fetchSeq) return;
        if (d.error) throw new Error(d.error);
        if (d.progress) ASSIGN.progress = d.progress;
        ASSIGN.suggestion = d.done ? null : d;
        assignDraw(d);
    } catch (e) {
        if (mine !== ASSIGN.fetchSeq) return;
        assignDraw({error: String(e)});
    }
}

/* Apply, discard, or skip — all three end in the next question. After
 * every decision the progress visibly jumps: a click that decides thousands
 * of voices should not look like one that decides twelve. */
async function assignDecide(kind) {
    if (ASSIGN.running || !ASSIGN.suggestion) return;
    const v = ASSIGN.suggestion;
    // The next question shows different voices — a playback would keep
    // going over it and risk naming the wrong one.
    assignSeqStop();
    assignOnlyOnePlays(null);
    if (kind === 'skip') {
        /* An asked-again question is simply left open. A normal one is
         * remembered by its largest voice — the server would otherwise
         * return the same group, still the largest open one. */
        const anchor = (v.samples && v.samples[0] && v.samples[0].vector_id);
        if (anchor && !v.asked) {
            ASSIGN.skipped.add(anchor);
            assignSaveMemo();
        }
        ASSIGN.message = v.asked ? 'Left open.' : '';
        if (ASSIGN.suggestion === v) ASSIGN.suggestion = null;   // decided: never shown again
        assignFetch();
        return;
    }
    const body = {vector_ids: v.vector_ids};
    if (kind === 'trash') {
        body.trash = true;
    } else {
        const field_el = document.getElementById('assignName');
        const name = (field_el && field_el.value || '').trim();
        if (!name) { if (field_el) field_el.focus(); return; }
        body.name = name;
    }
    ASSIGN.running = true;
    assignDraw({...v, working: true});
    let d;
    try {
        const r = await fetch('/api/speaker/assignment/apply', {
            method: 'POST', headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(body)});
        d = await r.json();
        if (!r.ok || d.error) throw new Error(d.error || r.status);
    } catch (e) {
        ASSIGN.running = false;
        assignDraw({...v, error: String(e)});
        return;
    }
    ASSIGN.running = false;
    if (d.progress) ASSIGN.progress = d.progress;
    ASSIGN.message = kind === 'trash'
        ? `${vxN(d.discarded || 0)} voice${d.discarded === 1 ? '' : 's'} discarded.`
        : `${vxN(d.set_flag || 0)} voice${d.set_flag === 1 ? '' : 's'} named "${body.name}"`
          + (d.spread ? `, ${vxN(d.spread)} more by spreading.` : '.');
    // The data changed — the map is fetched again when it is shown.
    vxMapChanged();
    if (kind !== 'trash' && typeof vxPeopleNames === 'function') vxPeopleNames(true);
    if (typeof historyRefresh === 'function') historyRefresh();
    assignRecentLoad();
    // The question is decided — even if the tab was left meanwhile and
    // assignFetch therefore asks nothing now, coming back must ask anew.
    if (ASSIGN.suggestion === v) ASSIGN.suggestion = null;
    assignFetch();
}

/* Bring the skipped ones back. Without this the list would be a one-way
 * street — remembered across reloads, a misjudged group would never come
 * back. */
function assignSkippedBack() {
    ASSIGN.skipped.clear();
    assignSaveMemo();
    assignFetch();
}

// ── The last decisions, and going back ───────────────────────────────────

async function assignRecentLoad() {
    try {
        const d = await (await fetch('/api/voices/history?limit=8')).json();
        ASSIGN.recent = d.entries || [];
    } catch (e) {
        ASSIGN.recent = [];
    }
    assignDrawSide();
}

/* Undo a decision — and ask its question again.
 *
 * The voices the click decided come back open (its spread goes back too,
 * and nothing a later click decided is touched). They are shown as the
 * question they were, with the same samples, to be decided anew: named
 * after all, discarded after all, or left open. */
async function assignUndo(id) {
    let d;
    try {
        d = await vxJson(`/api/voices/history/${id}/undo`, {});
    } catch (e) {
        ASSIGN.message = 'Undo failed: ' + (e.message || e);
        assignDrawSide();
        return;
    }
    vxMapChanged();
    if (typeof historyRefresh === 'function') historyRefresh();
    if (d.progress) ASSIGN.progress = d.progress;
    const left = (d.left || []).map(([why, n]) => `${vxN(n)} ${why}`).join(', ');
    ASSIGN.message = `Undone: ${vxN(d.changed)} voice${d.changed === 1 ? '' : 's'} back`
        + (left ? ` (left alone: ${left})` : '') + '.';
    if (d.ask) {
        ASSIGN.fetchSeq++;           // a suggestion still on its way is outdated now
        assignSeqStop();
        assignOnlyOnePlays(null);
        ASSIGN.suggestion = {...d.ask, asked: d.ask.decided, done: false};
        assignDraw(ASSIGN.suggestion);
    }
    assignRecentLoad();
}

async function assignRedo(id) {
    try {
        const d = await vxJson(`/api/voices/history/${id}/redo`, {});
        if (d.progress) ASSIGN.progress = d.progress;
        ASSIGN.message = `Redone: ${vxN(d.changed)} voice${d.changed === 1 ? '' : 's'}.`;
    } catch (e) {
        ASSIGN.message = 'Redo failed: ' + (e.message || e);
    }
    vxMapChanged();
    if (typeof historyRefresh === 'function') historyRefresh();
    assignRecentLoad();
    assignFetch();
}

/* Listen — one browser player per sample (the default player):
 * right-click → "save audio as".
 * Eight players can play at once, so one starting stops all the others,
 * the map's own player and its sequence included — two overlapping voices
 * are useless exactly where the point is to compare them. */
function assignOnlyOnePlays(except_for) {
    document.querySelectorAll('#mapOverlay audio').forEach(a => {
        if (a !== except_for && !a.paused) a.pause();
    });
    // The row in the map also needs to stop — otherwise it would advance
    // to the next clip unasked while a sample is playing here.
    if (typeof graphRowStop === 'function') graphRowStop();
}

/* One listener for all players, including the ones built later. `play`
 * doesn't bubble, so this uses the capture phase. */
if (typeof document !== 'undefined' && !window.__assignSoundListening) {
    window.__assignSoundListening = true;
    document.addEventListener('play', e => {
        const el = e.target;
        if (el && el.tagName === 'AUDIO' && el.closest
                && el.closest('#mapOverlay')) {
            assignOnlyOnePlays(el);
            assignSeqFollow(el);
        }
    }, true);
}

// ── Drawing ──────────────────────────────────────────────────────────────

function assignDraw(d) {
    const el = document.getElementById('mapAssignBody');
    if (!el) return;
    assignDrawSide();
    if (d.load_it) {
        el.innerHTML = '<div class="vx-empty">Finding the group that decides the most open speech …</div>';
        vxState('assign', 'Computing the next question …');
        return;
    }
    if (d.error) {
        el.innerHTML = `<div class="vx-card"><div class="vx-note err">Error: ${esc(d.error)}</div>
            <div style="margin-top:10px"><button class="vx-btn" onclick="assignFetch()">Try again</button></div></div>`;
        return;
    }
    const n = ASSIGN.skipped.size;
    if (d.done || !d.samples) {
        vxState('assign', 'Nothing left to ask');
        el.innerHTML = `<div class="vx-card"><h3>${n
            ? 'Nothing left open besides the skipped ones.'
            : 'Done — nothing left open that occurs in more than one recording.'}</h3>
            <div class="vx-sub">${n ? `${n} skipped — <button class="vx-btn" onclick="assignSkippedBack()">ask them again</button>` : ''}</div>
            ${ASSIGN.message ? `<div class="vx-note" style="margin-top:10px">${esc(ASSIGN.message)}</div>` : ''}</div>`;
        return;
    }
    const again = d.asked ? `
        <div class="vx-again"><svg class="ui-ico" style="width:16px;height:16px"><use href="#i-undo"/></svg>
            <span>Asked again — you had ${d.asked.kind === 'trash' ? 'discarded' : `named`}
            ${d.asked.kind === 'trash' ? 'these voices' : `them "${esc(d.asked.name || '')}"`}
            ${d.asked.at ? `(${esc(vxWhen(d.asked.at))})` : ''}. Decide anew — or skip to leave them open.</span>
        </div>` : '';
    vxState('assign', `${d.speech_hours} h of speech · ${plural(d.speaker, 'speaker')} from ${plural(d.recordings, 'recording')}`);
    const shown = (d.samples || []).length;
    el.innerHTML = `${again}
        <div class="vx-q-head">
            <span class="vx-q-title">${d.speech_hours} h of speech · ${plural(d.speaker, 'speaker')}
                from ${plural(d.recordings, 'recording')}</span>
            <span class="vx-q-meta">One click decides ${vxN(d.vector_ids.length)} voice${d.vector_ids.length === 1 ? '' : 's'}.</span>
        </div>
        ${assignStatsHtml(d)}
        <div class="vx-seq" id="assignSeq">
            <button class="vx-btn" id="assignSeqPlay" onclick="assignSeqToggle()"
                    title="Play the samples one after another (Ctrl+P in the name field)">▶ Play all</button>
            <button class="vx-btn" onclick="assignSeqStep(-1)" title="Previous sample">⏮</button>
            <button class="vx-btn" onclick="assignSeqStep(1)" title="Next sample">⏭</button>
            <label class="vx-field" title="How long each sample plays in the row">each
                <select id="assignPer" onchange="assignSeqPer(this.value)">${ASSIGN_PER.map(x =>
                    `<option value="${x}"${x === ASSIGN.seq.per ? ' selected' : ''}>${x ? x + ' s' : 'whole'}</option>`).join('')}</select></label>
            <label class="vx-field" title="Start playing as soon as a new question is shown">
                <input type="checkbox" id="assignAuto" ${ASSIGN.seq.auto ? 'checked' : ''}
                       onchange="assignSeqAuto(this.checked)"> start with every question</label>
            <span class="map-small" id="assignSeqState">${shown} of ${plural(d.speaker, 'speaker')} ${shown === 1 ? 'has' : 'have'} a sample${
                shown < d.speaker ? ' — the ones with the most speech' : ''}</span>
        </div>
        <div id="assignSamples">${vxSamplesHtml(d.samples)}</div>
        ${d.hint ? `<div class="vx-note">${esc(d.hint)}</div>` : ''}
        <div class="vx-decide">
            <div class="vx-decide-row">
                <input type="text" id="assignName" placeholder="Who is this?" list="vxPeopleList"
                       autocomplete="off" onkeydown="assignKey(event)">
                <button class="vx-btn primary big" onclick="assignDecide('name')"
                        title="Name all these voices — the name spreads to the ones that sound alike">Apply</button>
                <button class="vx-btn danger big" onclick="assignDecide('trash')"
                        title="Not relevant — to the trash (the History brings it back)"><svg class="ui-ico"><use href="#i-trash"/></svg>Trash</button>
                <button class="vx-btn big" onclick="assignDecide('skip')"
                        title="${d.asked ? 'Leave these voices open' : 'Not now — ask the next group'}">${d.asked ? 'Leave open' : 'Skip'}</button>
            </div>
            <div class="vx-keys"><kbd>Enter</kbd> apply · <kbd>Del</kbd> in the empty field: trash ·
                <kbd>Shift</kbd>+<kbd>Enter</kbd> skip · <kbd>Ctrl</kbd>+<kbd>Z</kbd> in the empty field: undo the last decision ·
                <kbd>Ctrl</kbd>+<kbd>P</kbd> play all / pause · <kbd>↓</kbd> <kbd>Tab</kbd> pick a suggested name</div>
            ${d.working ? '<div class="vx-note">Applying …</div>' : ''}
            ${ASSIGN.message && !d.working ? `<div class="vx-note">${esc(ASSIGN.message)}</div>` : ''}
        </div>`;
    const field_el = document.getElementById('assignName');
    if (field_el && !d.working) field_el.focus();
    // Autoplay only on screen: an answer arriving after the tab or the
    // workspace was left would otherwise play with no visible control.
    if (!d.working && !d.error && ASSIGN.seq.auto && !ASSIGN.seq.running
        && assignVisible()) assignSeqStart(0);
}

/* What helps deciding, beside the samples: more statistics, and which
 * named people the voice group is closest to. */
function assignStatsHtml(d) {
    const near = d.nearest || [];
    const st = d.stats || {};
    const pct = x => `${Math.round((x || 0) * 100)} %`;
    const verdict = sim => sim >= 0.8 ? ['very alike', 'hi'] : sim >= 0.65 ? ['alike', 'mid'] : ['hardly', 'lo'];
    const nearHtml = near.length ? near.map(p => {
        const [word, cls] = verdict(p.sim);
        return `<div class="vx-near" onclick="assignTakeName(${esc(JSON.stringify(p.name))})"
                     title="Take this name into the field">
            <span class="vx-near-name">${esc(p.name)}</span>
            <span class="vx-near-bar"><i class="${cls}" style="width:${Math.max(3, Math.round(Math.max(0, p.sim) * 100))}%"></i></span>
            <span class="vx-near-sim ${cls}">${p.sim.toFixed(2)} · ${word}</span>
            <span class="vx-near-votes" title="Speakers of this group whose own nearest name it is">${vxN(p.votes)} / ${vxN(d.speaker)}</span>
        </div>`;
    }).join('') : '<div class="map-small">Nobody named yet — nothing to compare with.</div>';
    const when = st.first_date
        ? (st.first_date === st.last_date ? esc(st.first_date) : `${esc(st.first_date)} – ${esc(st.last_date)}`)
        : '—';
    return `<div class="vx-q-stats">
        <div class="vx-card vx-q-near"><h3>Sounds like</h3>
            <div class="vx-sub">The named people nearest to this group. 0.65 and more: a name would
                reach it by spreading. Click a name to take it.</div>${nearHtml}</div>
        <div class="vx-card"><h3>When</h3><div class="vx-big">${when}</div>
            <div class="vx-sub">${st.dated ? `${vxN(st.dated)} of ${vxN(d.recordings)} recording${d.recordings === 1 ? '' : 's'} ${st.dated === 1 ? 'carries a date in its' : 'carry a date in their'} name`
                : 'No recording carries a date in its name'}</div></div>
        <div class="vx-card"><h3>Where</h3>${(st.folders || []).map(f =>
            `<div class="vx-q-folder"><span title="${esc(f.folder)}">${esc(f.folder)}</span><b>${pct(f.share)}</b></div>`).join('')
            || '<div class="map-small">—</div>'}<div class="vx-sub">Share of the speech per folder</div></div>
        <div class="vx-card"><h3>Speech</h3>
            <div class="vx-q-folder"><span>per speaker, median</span><b>${esc(vxSpeech(st.speech_median_sec))}</b></div>
            <div class="vx-q-folder"><span>longest speaker</span><b>${esc(vxSpeech(st.speech_max_sec))}</b></div>
            <div class="vx-q-folder"><span>voice windows</span><b>${vxN(st.windows || d.vector_ids.length)}</b></div></div>
    </div>`;
}

function assignTakeName(name) {
    const f = document.getElementById('assignName');
    if (!f) return;
    f.value = name;
    f.focus();
}

// ── Playing the samples one after another ───────────────────────────────
//
// All of them one after another, or only the first seconds of each.
// The samples' own players stay —
// the row just starts one after the other and stops each after the chosen
// seconds. Playing one by hand while the row runs moves the row there.

function assignSeqAudios() {
    return [...document.querySelectorAll('#assignSamples audio')];
}

function assignSeqToggle() {
    if (ASSIGN.seq.running) assignSeqStop();
    else assignSeqStart(ASSIGN.seq.pos || 0);
}

function assignSeqStart(pos) {
    const all = assignSeqAudios();
    if (!all.length) return;
    ASSIGN.seq.running = true;
    assignSeqPlayAt(Math.min(Math.max(0, pos), all.length - 1));
}

function assignSeqPlayAt(pos) {
    const all = assignSeqAudios();
    clearTimeout(ASSIGN.seq.timer);
    if (pos >= all.length) { assignSeqStop(true); return; }
    ASSIGN.seq.pos = pos;
    const a = all[pos];
    all.forEach((x, i) => x.closest('.vx-sample')?.classList.toggle('playing', i === pos));
    // The next one loads while this one plays
    if (all[pos + 1]) all[pos + 1].preload = 'auto';
    a.closest('.vx-sample')?.scrollIntoView({block: 'nearest'});
    ASSIGN.seq.expect = a;
    try { a.currentTime = 0; } catch (e) { /* not loaded yet */ }
    const mine = () => ASSIGN.seq.running && ASSIGN.seq.expect === a;
    // A pause (stop, another sample) aborts the pending start — that is no
    // reason to go on. A sample that can't be played is skipped.
    a.play().catch(err => { if (mine() && err && err.name !== 'AbortError') assignSeqNext(); });
    // A sample is cut from its recording on request; one that doesn't
    // start within 25 s is skipped instead of holding the row
    ASSIGN.seq.timer = setTimeout(() => { if (mine() && a.currentTime === 0) assignSeqNext(); }, 25000);
    a.onended = () => { if (ASSIGN.seq.running && ASSIGN.seq.expect === a) assignSeqNext(); };
    a.ontimeupdate = () => {
        const per = ASSIGN.seq.per;
        if (ASSIGN.seq.running && ASSIGN.seq.expect === a && per && a.currentTime >= per) assignSeqNext();
    };
    assignSeqState();
}

function assignSeqNext() {
    const a = ASSIGN.seq.expect;
    if (a && !a.paused) a.pause();
    assignSeqPlayAt(ASSIGN.seq.pos + 1);
}

function assignSeqStep(d) {
    const n = assignSeqAudios().length;
    if (!n) return;
    const pos = Math.min(Math.max(0, (ASSIGN.seq.pos || 0) + d), n - 1);
    if (ASSIGN.seq.running) { const a = ASSIGN.seq.expect; if (a) a.pause(); assignSeqPlayAt(pos); }
    else { ASSIGN.seq.running = true; assignSeqPlayAt(pos); }
}

function assignSeqStop(finished = false) {
    clearTimeout(ASSIGN.seq.timer);
    const was = ASSIGN.seq.running;
    ASSIGN.seq.running = false;
    const a = ASSIGN.seq.expect;
    ASSIGN.seq.expect = null;
    if (a && !a.paused) a.pause();
    if (finished) ASSIGN.seq.pos = 0;
    document.querySelectorAll('#assignSamples .vx-sample.playing').forEach(x => x.classList.remove('playing'));
    if (was) assignSeqState(finished);
}

/* A sample started by hand while the row runs: the row goes on from there. */
function assignSeqFollow(el) {
    if (!ASSIGN.seq.running || el === ASSIGN.seq.expect) return;
    const i = assignSeqAudios().indexOf(el);
    if (i < 0) { assignSeqStop(); return; }
    // Taken over by hand: the row goes on from this sample, as it plays
    const old = ASSIGN.seq.expect;
    ASSIGN.seq.expect = null;
    if (old && !old.paused) old.pause();
    assignSeqPlayAt(i);
}

function assignSeqPer(v) {
    ASSIGN.seq.per = Number(v) || 0;
    try { localStorage.setItem(ASSIGN_MEMO + '.per', String(ASSIGN.seq.per)); } catch (e) { /* full */ }
}

function assignSeqAuto(on) {
    ASSIGN.seq.auto = !!on;
    try { localStorage.setItem(ASSIGN_MEMO + '.auto', on ? '1' : '0'); } catch (e) { /* full */ }
}

function assignSeqState(finished = false) {
    const b = document.getElementById('assignSeqPlay');
    if (b) b.textContent = ASSIGN.seq.running ? '⏸ Pause' : '▶ Play all';
    const st = document.getElementById('assignSeqState');
    const n = assignSeqAudios().length;
    if (st) st.textContent = ASSIGN.seq.running ? `Sample ${ASSIGN.seq.pos + 1} of ${n}`
        : finished ? (n === 1 ? 'The sample played' : `All ${n} samples played`) : `Stopped at sample ${ASSIGN.seq.pos + 1} of ${n}`;
}

/* Keys in the name field — one hand stays on the keyboard for a hundred
 * decisions in a row. Only an EMPTY field gives Del and Ctrl+Z a meaning
 * of their own; with text in it they edit the text as always. */
function assignKey(e) {
    const empty = !e.target.value.trim();
    if (e.key === 'Enter' && e.shiftKey) { e.preventDefault(); assignDecide('skip'); }
    else if (e.key === 'Enter') { e.preventDefault(); if (!empty) assignDecide('name'); }
    else if (e.key === 'Delete' && empty) { e.preventDefault(); assignDecide('trash'); }
    else if ((e.key === 'p' || e.key === 'P') && (e.ctrlKey || e.metaKey)) { e.preventDefault(); assignSeqToggle(); }
    else if ((e.key === 'z' || e.key === 'Z') && (e.ctrlKey || e.metaKey) && empty) {
        const last = ASSIGN.recent.find(x => !x.undone);
        if (last) { e.preventDefault(); assignUndo(last.id); }
    }
}

function assignDrawSide() {
    const el = document.getElementById('assignSide');
    if (!el) return;
    const f = ASSIGN.progress || {};
    const p = Math.round((f.share || 0) * 1000) / 10;
    const n = ASSIGN.skipped.size;
    el.innerHTML = `
        <div class="vx-side-title">Progress</div>
        <div class="vx-q-meta"><b class="vx-num" style="color:var(--tx-1);font-size:1.1rem">${p.toFixed(1)} %</b>
            of ${vxN(f.target_hours || 0)} h recurring speech</div>
        <div class="assign-bar"><div style="width:${p}%"></div></div>
        <div class="map-small">${vxN(f.open_hours || 0)} h still open ·
            ${vxN(f.one_off_hours || 0)} h occur only once and aren't asked about</div>
        <div class="vx-side-title">Ask by</div>
        <div class="vx-seg">${ASSIGN_BY.map(([w, l, t]) => `
            <button class="vx-seg-btn${ASSIGN.sort_by === w ? ' active' : ''}" title="${esc(t)}"
                    onclick="assignAfterSet('${w}')">${l}</button>`).join('')}</div>
        ${n ? `<div class="map-small" style="margin-top:8px">${n} skipped ·
            <button class="map-mini" onclick="assignSkippedBack()">ask again</button></div>` : ''}
        <div class="vx-side-title">Last decisions</div>
        <div class="vx-recent">${ASSIGN.recent.length ? ASSIGN.recent.map(assignRecentRow).join('')
            : '<div class="map-small">None yet.</div>'}</div>
        ${ASSIGN.recent.length ? '<div class="map-small" style="margin-top:4px">Click a decision to hear what it was.</div>' : ''}
        <div style="margin-top:10px"><button class="vx-btn" onclick="vxTab('history')">
            <svg class="ui-ico"><use href="#i-history"/></svg>All decisions</button></div>`;
}

function assignRecentRow(e) {
    const what = e.kind === 'trash' ? 'Trash'
        : e.kind === 'restore' ? 'Restored'
        : e.kind === 'unname' ? 'Name removed'
        : e.kind === 'rename' ? `Renamed to ${e.name || ''}`
        : (e.name || 'Named');
    const icon = {name: '✓', trash: '🗑', restore: '↩', unname: '✕', rename: '✎'}[e.kind] || '•';
    const action = e.undone
        ? `<button class="map-mini" onclick="assignRedo(${e.id})" title="Put it in force again">Redo</button>`
        : `<button class="map-mini" onclick="assignUndo(${e.id})"
                   title="Undo — the voices come back as a question">Undo</button>`;
    /* A click opens the decision to its samples, so discarded voices can
     * be previewed later. The History has them too, one tab away. */
    const open = ASSIGN.open.has(e.id);
    return `<div class="vx-recent-item">
        <div class="vx-recent-row${e.undone ? ' undone' : ''}" onclick="assignRecentToggle(${e.id})"
             title="${esc(`${vxN(e.voices)} voice${e.voices === 1 ? '' : 's'} · ${plural(e.speakers, 'speaker')} · ${plural(e.recordings, 'recording')} — click to listen`)}">
            <span class="vx-kind ${esc(e.kind)}">${icon}</span>
            <span class="vx-what">${esc(what)} · ${vxN(e.voices)}</span>
            <span class="vx-when">${esc(vxClock(e.at))}</span>
            <span onclick="event.stopPropagation()">${action}</span></div>
        ${open ? `<div class="vx-recent-samples">${vxSamplesHtml((e.samples || []).slice(0, 4))
            || '<div class="map-small">No samples.</div>'}</div>` : ''}
    </div>`;
}

function assignRecentToggle(id) {
    if (ASSIGN.open.has(id)) ASSIGN.open.delete(id); else ASSIGN.open.add(id);
    assignDrawSide();
}
