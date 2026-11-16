/* ══════════════════════════════════════════════════════════════════════════
 *  VOICES — the workspace around the voice map
 *
 *  Floating panels that can't be moved were replaced by a uniform layout
 *  split into tabs.
 *
 *  Five tabs, each a whole page instead of a box over the map:
 *
 *    Map      the tiles or the graph, the details of a voice docked at the
 *             side (resizable, can be hidden)
 *    Assign   the guided assignment — the question, the samples, and the
 *             last decisions with an undo
 *    People   the named people
 *    History  every decision on names and the trash, undo and redo
 *    Extract  one person's voice as clips with their sentences
 *
 *  **Each tab loads when it is opened.** The map is the heavy one (tens of MB of
 *  JSON with hundreds of thousands of voices, seconds to build). Before, the guided
 *  assignment floated over it and reloaded it after every decision — every
 *  click paid for a picture nobody was looking at. Now a decision only
 *  marks the map stale (`vxMapChanged`), and it reloads when it is shown.
 * ══════════════════════════════════════════════════════════════════════════ */

const VX = {
    tab: null,
    mapLoaded: false,      // fetched at least once since the page loaded
    mapStale: false,       // names or the trash changed since
    state: {},             // the head line per tab
    people: null,          // named people, for the name suggestions
};

const VX_TABS = ['people', 'assign', 'history', 'map', 'conversations', 'extract', 'help'];
const VX_MEMO = 'whisperx.voices.tab';
const VX_SIDE_MEMO = 'whisperx.voices.side';

function vxRememberedTab() {
    let t = null;
    try { t = localStorage.getItem(VX_MEMO); } catch (e) { /* private mode */ }
    return VX_TABS.includes(t) ? t : 'people';
}

/* Open the workspace — on the given tab, else where it was left. */
function vxOpen(tab) {
    const ov = document.getElementById('mapOverlay');
    if (!ov) return;
    const was = ov.classList.contains('visible');
    ov.classList.add('visible');
    vxSideRestore();
    vxTab(tab || (was && VX.tab) || vxRememberedTab());
}

function vxClose() {
    document.getElementById('mapOverlay')?.classList.remove('visible');
    vxLeaveTab(VX.tab);
    vxAudioOff();
    vxPopClose();
    if (typeof extractStopPolling === 'function') extractStopPolling();
}

/* What a tab keeps running in the background ends when it is left (or the
 * workspace closes): the Assign sequence and its timers, the Extract clip
 * editor's playback. Otherwise audio plays on a page nobody sees. The clip
 * editor itself stays open with its unsaved bounds and text — a glance at
 * another tab must not throw away work; its global keys only act while
 * the Extract tab is on screen (voices_clipedit.js). */
function vxLeaveTab(tab) {
    if (tab === 'conversations') vxReturnConversations();
    if (tab === 'assign' && typeof assignClose === 'function') assignClose();
    if (tab === 'extract' && typeof clipEditStop === 'function'
        && typeof CLIPEDIT !== 'undefined' && CLIPEDIT.i >= 0) clipEditStop();
}

/* Every player in the workspace stops — the map's own player included.
 * Otherwise a sample keeps playing invisibly after a tab switch. */
function vxAudioOff() {
    document.querySelectorAll('#mapOverlay audio').forEach(a => {
        if (!a.paused) a.pause();
    });
    if (typeof panelAudioOff === 'function') panelAudioOff();
}

function vxTab(name) {
    if (!VX_TABS.includes(name)) name = 'people';
    const prev = VX.tab;
    const changed = prev !== name;
    VX.tab = name;
    try { localStorage.setItem(VX_MEMO, name); } catch (e) { /* private mode */ }
    document.querySelectorAll('#vxTabs .vx-tab').forEach(b => {
        const on = b.dataset.tab === name;
        b.classList.toggle('active', on);
        b.setAttribute('aria-selected', on ? 'true' : 'false');
    });
    document.querySelectorAll('#mapOverlay .vx-page').forEach(p => {
        p.hidden = p.dataset.page !== name;
    });
    if (changed) { vxLeaveTab(prev); vxAudioOff(); }
    vxPopClose();
    const st = document.getElementById('mapState');
    if (st) st.textContent = VX.state[name] || '';
    if (name === 'map') vxShowMap();
    else if (name === 'assign' && typeof assignOpen === 'function') assignOpen();
    else if (name === 'people' && typeof peopleShow === 'function') peopleShow();
    else if (name === 'history' && typeof historyShow === 'function') historyShow();
    else if (name === 'extract' && typeof extractShow === 'function') {
        extractShow();
        // An editor kept open while another tab was on screen draws again
        if (typeof CLIPEDIT !== 'undefined' && CLIPEDIT.i >= 0 && typeof clipEditDraw === 'function')
            requestAnimationFrame(() => clipEditDraw());
    }
    else if (name === 'help' && typeof helpShow === 'function') helpShow();
    else if (name === 'conversations') vxBorrowConversations();
}

/* ── Conversations: Find's view, shown here ───────────────────────────────
 * One view and one state for both windows: while this tab is on screen,
 * Find's Conversations pane (and the remote control of the player it
 * plays into) moves into this page; leaving the tab or the window puts
 * them back where they came from. */
const VX_CV = {marks: null};

function vxConversationsHere() {
    return !!VX_CV.marks;
}

function vxBorrowConversations() {
    const page = document.getElementById('vxPageConversations');
    const pane = document.getElementById('svPaneConversations');
    const player = document.getElementById('svPlayer');
    if (!page || !pane) return;
    if (!VX_CV.marks) {
        VX_CV.marks = [pane, player].filter(Boolean).map(el => {
            const mark = document.createComment('moved to Voices');
            el.parentNode.insertBefore(mark, el);
            return [el, mark];
        });
        for (const [el] of VX_CV.marks) page.appendChild(el);
    }
    pane.classList.add('active');
    if (typeof cvOpen === 'function') cvOpen();
}

function vxReturnConversations() {
    if (!VX_CV.marks) return;
    // A conversation playing from here loses its remote control with the
    // tab: it stops rather than going on unseen
    if (typeof svState !== 'undefined' && svState.plays && typeof svWgStop === 'function') svWgStop();
    for (const [el, mark] of VX_CV.marks) {
        mark.parentNode.insertBefore(el, mark);
        mark.remove();
    }
    VX_CV.marks = null;
    const pane = document.getElementById('svPaneConversations');
    if (pane) pane.classList.toggle('active', typeof svState !== 'undefined' && svState.tab === 'conversations');
    if (typeof cvLeave === 'function') cvLeave();
}

/* The head line belongs to a tab: the map's counts don't describe the
 * history. Each tab writes its own, the head shows the one on screen. */
function vxState(tab, text) {
    VX.state[tab] = text || '';
    if (VX.tab === tab) {
        const st = document.getElementById('mapState');
        if (st) st.textContent = VX.state[tab];
    }
}

/* The map, when it comes on screen: wired once, fetched once — and again
 * only when something changed it meanwhile. A canvas that was hidden has
 * no size; only now can the tiles be laid out. */
function vxShowMap() {
    if (typeof mapWireUp !== 'function') return;
    mapWireUp();
    if (!VX.mapLoaded) {
        mapView(tileRememberedView());
        VX.mapLoaded = true;
        VX.mapStale = false;
        mapLoadGraph();
        return;
    }
    if (VX.mapStale) {
        VX.mapStale = false;
        mapLoadGraph();
        return;
    }
    if (typeof TILE !== 'undefined' && TILE.on) tileLayout(true);
    mapBuildGrid();
    mapRequest();
}

/* Names or the trash changed (a decision in another tab, an undo). The
 * map is fetched again — now if it is on screen, else when it next is. */
function vxMapChanged() {
    VX.people = null;
    // Names changed: the People page reloads the next time it shows
    if (typeof PEOPLE !== 'undefined') PEOPLE.stale = true;
    const visible = document.getElementById('mapOverlay')?.classList.contains('visible');
    if (VX.tab === 'map' && visible && VX.mapLoaded) {
        VX.mapStale = false;
        mapLoadGraph();
    } else if (VX.mapLoaded) {
        VX.mapStale = true;
    }
}

// ── The docked details: width and hiding ─────────────────────────────────

function vxSideRestore() {
    let m = {};
    try { m = JSON.parse(localStorage.getItem(VX_SIDE_MEMO) || '{}'); } catch (e) { /* none */ }
    const body = document.getElementById('mapBody');
    if (!body) return;
    if (m.width > 0) body.style.setProperty('--vx-side', `${Math.round(m.width)}px`);
    body.classList.toggle('side-closed', !!m.closed);
    vxSideButton();
}

function vxSideSave(patch) {
    let m = {};
    try { m = JSON.parse(localStorage.getItem(VX_SIDE_MEMO) || '{}'); } catch (e) { /* none */ }
    try { localStorage.setItem(VX_SIDE_MEMO, JSON.stringify({...m, ...patch})); } catch (e) { /* full */ }
}

function vxSideButton() {
    const body = document.getElementById('mapBody');
    const b = document.getElementById('mapSideBtn');
    if (b && body) b.textContent = body.classList.contains('side-closed') ? 'Details ⇤' : 'Details ⇥';
}

function vxSideToggle(open) {
    const body = document.getElementById('mapBody');
    if (!body) return;
    const closed = open === undefined ? !body.classList.contains('side-closed') : !open;
    body.classList.toggle('side-closed', closed);
    vxSideSave({closed});
    vxSideButton();
    vxMapResized();
}

/* The canvas changed its size: lay the tiles out again, redraw. */
function vxMapResized() {
    if (typeof TILE !== 'undefined' && TILE.on && typeof tileLayout === 'function') tileLayout(true);
    if (typeof mapBuildGrid === 'function') mapBuildGrid();
    if (typeof mapRequest === 'function') mapRequest();
}

function vxSplitterWire() {
    const sp = document.getElementById('mapSplitter');
    const body = document.getElementById('mapBody');
    if (!sp || !body || sp.dataset.wired) return;
    sp.dataset.wired = '1';
    let startX = 0, startW = 0, moving = false, relayout = 0;
    sp.addEventListener('mousedown', e => {
        const side = document.getElementById('mapSide');
        if (!side) return;
        moving = true;
        startX = e.clientX;
        startW = side.getBoundingClientRect().width;
        sp.classList.add('dragging');
        document.body.style.userSelect = 'none';
        e.preventDefault();
    });
    window.addEventListener('mousemove', e => {
        if (!moving) return;
        const w = Math.max(300, Math.min(window.innerWidth * 0.6, startW + (startX - e.clientX)));
        body.style.setProperty('--vx-side', `${Math.round(w)}px`);
        // One relayout per frame at most: vxMapResized() walks every voice
        // (hundreds of thousands on a large library) — once per mousemove
        // froze the tab.
        if (!relayout) relayout = requestAnimationFrame(() => { relayout = 0; vxMapResized(); });
    });
    window.addEventListener('mouseup', () => {
        if (!moving) return;
        moving = false;
        if (relayout) { cancelAnimationFrame(relayout); relayout = 0; }
        vxMapResized();
        sp.classList.remove('dragging');
        document.body.style.userSelect = '';
        const side = document.getElementById('mapSide');
        if (side) vxSideSave({width: side.getBoundingClientRect().width});
    });
}

// ── Menus under a toolbar button ─────────────────────────────────────────

function vxPopToggle(id, btn) {
    const pop = document.getElementById(id);
    if (!pop) return;
    const open = pop.hidden;
    vxPopClose();
    pop.hidden = !open;
    if (btn) btn.classList.toggle('active', open);
}

function vxPopClose() {
    document.querySelectorAll('#mapOverlay .vx-pop').forEach(p => { p.hidden = true; });
}

document.addEventListener('mousedown', e => {
    if (!e.target.closest || e.target.closest('.vx-pop-wrap')) return;
    vxPopClose();
});

// ── Shared formatting ────────────────────────────────────────────────────

const VX_MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug',
                   'Sep', 'Oct', 'Nov', 'Dec'];
const VX_DAYS = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];

function vxDate(iso) {
    const d = iso ? new Date(iso) : null;
    return d && !isNaN(d) ? d : null;
}

/* "23:17" — local time */
function vxClock(iso) {
    const d = vxDate(iso);
    if (!d) return '';
    return `${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}`;
}

/* "Today", "Yesterday", "Mon 28 Sep", "12 Aug 2025" */
function vxDay(iso) {
    const d = vxDate(iso);
    if (!d) return 'Unknown day';
    const start = x => new Date(x.getFullYear(), x.getMonth(), x.getDate()).getTime();
    const diff = Math.round((start(new Date()) - start(d)) / 86400000);
    if (diff === 0) return 'Today';
    if (diff === 1) return 'Yesterday';
    const base = `${d.getDate()} ${VX_MONTHS[d.getMonth()]}`;
    return diff < 7 ? `${VX_DAYS[d.getDay()]} ${base}`
        : (d.getFullYear() === new Date().getFullYear() ? base : `${base} ${d.getFullYear()}`);
}

function vxWhen(iso) {
    const day = vxDay(iso);
    return (day === 'Today' ? '' : day + ', ') + vxClock(iso);
}

/* Hours of speech in a readable unit */
function vxHours(h) {
    h = Number(h) || 0;
    if (h >= 10) return `${h.toFixed(0)} h`;
    if (h >= 1) return `${h.toFixed(1)} h`;
    const m = h * 60;
    return m >= 1 ? `${Math.round(m)} min` : `${Math.round(m * 60)} s`;
}

/* A path for a field that cuts it at its START (direction:rtl keeps the
 * end — the file's name — in sight). Without the marks the right-to-left
 * layout moves the leading slash to the end: "mnt/…/a.mp4/". */
function vxLtr(text) {
    return '\u200E' + String(text ?? '') + '\u200E';
}

/* Seconds of speech: "45 s", "18 min", "2.1 h" */
function vxSpeech(sec) {
    sec = Number(sec) || 0;
    if (sec < 90) return `${Math.round(sec)} s`;
    if (sec < 5400) return `${Math.round(sec / 60)} min`;
    return `${(sec / 3600).toFixed(1)} h`;
}

function vxN(n) {
    return (Number(n) || 0).toLocaleString('en');
}

/* One player per sample. `preload="none"`: eight clips of ~250 kB on mere
 * sight would be 2 MB per question, and most are never played. The
 * browser's own player — right-click saves the clip. */
function vxSamplesHtml(samples) {
    if (!samples || !samples.length) return '';
    return `<div class="vx-samples">${samples.map(p => `
        <div class="vx-sample">
            <div class="vx-sample-head" title="${esc(p.file || '')}">
                <b>${esc(p.label || '')}</b>
                <span class="vx-sample-file">${esc(vxLtr(String(p.file || '').split('/').pop()))}</span>
                <span class="vx-sample-sec">${esc(vxSpeech(p.speech_sec))}</span>
            </div>
            <audio controls preload="none" src="/api/speaker/sample/${Number(p.vector_id)}"></audio>
        </div>`).join('')}</div>`;
}

/* The named people — for the suggestions under every name field. */
async function vxPeopleNames(force = false) {
    if (VX.people && !force) return VX.people;
    try {
        const d = await (await fetch('/api/speaker/people')).json();
        VX.peopleInfo = (d.people || []).filter(p => p.voices > 0)
            .map(p => ({name: p.name, voices: p.voices}));
    } catch (e) {
        VX.peopleInfo = VX.peopleInfo || [];
    }
    VX.people = VX.peopleInfo.map(p => p.name);
    return VX.people;
}

/* ── Suggesting names while typing ───────────────────────────────────────
 *
 * Typing should suggest names right away so one can be picked quickly.
 * The browser's own list (a
 * `<datalist>`) opened as a loose, light box far from the field and only
 * matched the start of a name. This one sits under the field, matches
 * the start of the name first, then the start of any word, then anywhere,
 * and the person with the most voices comes first among equals.
 *
 * Every name field takes part by `list="vxPeopleList"` (or `data-names`):
 * ↓/↑ choose, Tab takes the chosen one (or the first), Enter takes the
 * chosen one and goes on as the field's own Enter (apply, save …) — with
 * nothing chosen, Enter keeps exactly what was typed, so a NEW name never
 * turns into an existing one by accident. Esc closes the list first. */
const VX_SUGGEST = {input: null, items: [], pos: -1, el: null};
const VX_SUGGEST_MAX = 8;

function vxFold(t) {
    return String(t || '').toLowerCase().normalize('NFKD').replace(/[̀-ͯ]/g, '');
}

function vxSuggestFor(text) {
    const q = vxFold(text).trim();
    const all = VX.peopleInfo || [];
    if (!q) return [];
    const ranked = [];
    for (const p of all) {
        const n = vxFold(p.name);
        const rank = n.startsWith(q) ? 0
            : n.split(/[\s_\-.]+/).some(w => w.startsWith(q)) ? 1
            : n.includes(q) ? 2 : -1;
        if (rank >= 0 && n !== q) ranked.push([rank, -p.voices, p]);
    }
    ranked.sort((a, b) => a[0] - b[0] || a[1] - b[1] || a[2].name.localeCompare(b[2].name));
    return ranked.slice(0, VX_SUGGEST_MAX).map(r => r[2]);
}

function vxSuggestMark(name, text) {
    const q = vxFold(text).trim();
    const i = q ? vxFold(name).indexOf(q) : -1;
    if (i < 0 || vxFold(name).length !== name.length) return esc(name);
    return `${esc(name.slice(0, i))}<b>${esc(name.slice(i, i + q.length))}</b>${esc(name.slice(i + q.length))}`;
}

function vxSuggestShow(input) {
    VX_SUGGEST.input = input;
    VX_SUGGEST.items = vxSuggestFor(input.value);
    VX_SUGGEST.pos = -1;
    vxSuggestDraw();
}

function vxSuggestDraw() {
    let el = VX_SUGGEST.el;
    const input = VX_SUGGEST.input;
    if (!input || !VX_SUGGEST.items.length || !document.body.contains(input)) {
        if (el) el.hidden = true;
        return;
    }
    if (!el) {
        el = VX_SUGGEST.el = document.createElement('div');
        el.className = 'vx-suggest';
        el.setAttribute('role', 'listbox');
        // Keep the focus in the field while picking with the mouse
        el.addEventListener('mousedown', e => {
            e.preventDefault();
            const row = e.target.closest('[data-i]');
            if (row) vxSuggestTake(Number(row.dataset.i));
        });
        document.body.appendChild(el);
    }
    const r = input.getBoundingClientRect();
    el.style.left = `${Math.round(r.left)}px`;
    el.style.top = `${Math.round(r.bottom + 4)}px`;
    el.style.minWidth = `${Math.round(Math.min(Math.max(r.width, 220), 420))}px`;
    el.innerHTML = VX_SUGGEST.items.map((p, i) => `
        <div class="vx-suggest-row${i === VX_SUGGEST.pos ? ' on' : ''}" data-i="${i}" role="option">
            <span>${vxSuggestMark(p.name, input.value)}</span>
            <span class="vx-suggest-n">${vxN(p.voices)}</span></div>`).join('')
        + `<div class="vx-suggest-keys"><kbd>↓</kbd><kbd>↑</kbd> choose · <kbd>Tab</kbd> take ·
            <kbd>Enter</kbd> ${VX_SUGGEST.pos >= 0 ? 'take and go on' : 'keep what is typed'}</div>`;
    el.hidden = false;
}

function vxSuggestClose() {
    VX_SUGGEST.items = [];
    VX_SUGGEST.pos = -1;
    if (VX_SUGGEST.el) VX_SUGGEST.el.hidden = true;
}

function vxSuggestTake(i) {
    const p = VX_SUGGEST.items[i];
    const input = VX_SUGGEST.input;
    if (!p || !input) return;
    input.value = p.name;
    input.dispatchEvent(new Event('input', {bubbles: true}));
    vxSuggestClose();
}

function vxNameField(el) {
    return el && el.tagName === 'INPUT'
        && (el.getAttribute('list') === 'vxPeopleList' || el.dataset.names !== undefined);
}

if (typeof document !== 'undefined' && !window.__vxSuggestWired) {
    window.__vxSuggestWired = true;
    document.addEventListener('focusin', e => {
        const el = e.target;
        if (!vxNameField(el)) return;
        // The browser's own list never opens again over ours
        if (el.getAttribute('list')) { el.removeAttribute('list'); el.dataset.names = ''; }
        el.setAttribute('autocomplete', 'off');
        if (!VX.people) vxPeopleNames();
    });
    document.addEventListener('input', e => {
        if (vxNameField(e.target) && e.isTrusted) vxSuggestShow(e.target);
    });
    document.addEventListener('focusout', e => {
        if (e.target === VX_SUGGEST.input) vxSuggestClose();
    });
    window.addEventListener('resize', vxSuggestClose);
    document.addEventListener('scroll', e => {
        if (VX_SUGGEST.el && !VX_SUGGEST.el.hidden && !VX_SUGGEST.el.contains(e.target)) vxSuggestClose();
    }, true);
    // Capture: before the field's own keys (Enter applies) and before Esc
    // closes the workspace
    document.addEventListener('keydown', e => {
        if (e.target !== VX_SUGGEST.input || !VX_SUGGEST.items.length
                || !VX_SUGGEST.el || VX_SUGGEST.el.hidden) return;
        const n = VX_SUGGEST.items.length;
        if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
            e.preventDefault();
            VX_SUGGEST.pos = e.key === 'ArrowDown'
                ? (VX_SUGGEST.pos + 1) % n
                : (VX_SUGGEST.pos <= 0 ? n - 1 : VX_SUGGEST.pos - 1);
            vxSuggestDraw();
        } else if (e.key === 'Tab' && !e.shiftKey) {
            e.preventDefault();
            vxSuggestTake(Math.max(0, VX_SUGGEST.pos));
        } else if (e.key === 'Enter' && !e.shiftKey) {
            // Taken — the field's own Enter then goes on with that name
            if (VX_SUGGEST.pos >= 0) vxSuggestTake(VX_SUGGEST.pos);
            else vxSuggestClose();
        } else if (e.key === 'Escape') {
            e.preventDefault();
            e.stopImmediatePropagation();
            vxSuggestClose();
        }
    }, true);
}

async function vxJson(url, body) {
    const r = await fetch(url, body === undefined ? {} : {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(body)});
    return mapResponse(r);
}

/* "Undo" right after a decision made on the map or on the People page —
 * the same undo as in the History, one click away. */
function vxUndoToast(decision, text) {
    if (!decision) { Toast.success(text); return; }
    Toast.info(`${text} — click to undo`, 9000, () => vxUndo(decision));
}

async function vxUndo(decision) {
    try {
        const d = await vxJson(`/api/voices/history/${decision}/undo`, {});
        Toast.success(`Undone — ${vxN(d.changed)} ${d.changed === 1 ? 'voice' : 'voices'} back`);
        vxMapChanged();
        if (typeof historyRefresh === 'function') historyRefresh();
        if (typeof assignRecentLoad === 'function') assignRecentLoad();
        if (typeof peopleRefresh === 'function') peopleRefresh();
        return d;
    } catch (e) {
        Toast.error('Undo failed: ' + (e.message || e));
        return null;
    }
}

// Esc: a menu first, then the workspace.
document.addEventListener('keydown', e => {
    if (e.key !== 'Escape') return;
    const ov = document.getElementById('mapOverlay');
    if (!ov || !ov.classList.contains('visible')) return;
    if ([...ov.querySelectorAll('.vx-pop')].some(p => !p.hidden)) {
        vxPopClose();
        e.stopImmediatePropagation();
        return;
    }
    // A chosen voice or a person's page closes before the workspace
    if (typeof Shell !== 'undefined' && Shell.undoStep()) { e.stopImmediatePropagation(); return; }
    vxClose();
});

document.getElementById('mapOverlay')?.addEventListener('click', e => {
    if (e.target.id === 'mapOverlay') vxClose();
});
vxSplitterWire();
