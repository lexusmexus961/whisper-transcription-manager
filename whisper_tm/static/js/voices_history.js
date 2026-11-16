/* ══════════════════════════════════════════════════════════════════════════
 *  HISTORY — every decision on names and the trash, with undo and redo
 *
 *  Discarded or named voices can be taken back or sorted in anew — e.g.
 *  after assigning many voices, a speaker that should have been kept —
 *  without damaging the rest of the voice network.
 *
 *  Each click is one entry: which voices, how many by spreading, when,
 *  from where. What was decided before the history existed is rebuilt from
 *  the times of the trash and of the names (marked "rebuilt").
 *
 *  **Undo changes only what the entry changed** — and only where no later
 *  decision changed the voice since; those are left alone and counted
 *  (database/speaker.py). After an undo, the freed voices can be decided
 *  anew right here: named after all, discarded after all, or left open.
 *  The entry stays, struck through, with Redo.
 * ══════════════════════════════════════════════════════════════════════════ */

const HISTORY = {
    built: false,
    kind: 'all',
    q: '',
    entries: [],
    more: false,
    loading: false,
    seq: 0,                 // newest load; older answers are dropped
    open: new Set(),        // expanded entries
    preview: {},            // id → what an undo would do
    previewing: new Set(),  // previews on their way
    epoch: 0,               // moves with every change: older previews are void
    asks: {},               // id → the question an undo left behind
    counts: null,
    stale: false,           // something decided elsewhere meanwhile
};

const HISTORY_KINDS = [
    ['all', 'All'], ['name', 'Names'], ['trash', 'Trash'], ['undone', 'Undone'],
];

function historyShow() {
    const page = document.getElementById('vxPageHistory');
    if (!page) return;
    if (!HISTORY.built) {
        HISTORY.built = true;
        page.innerHTML = `
            <div class="vx-toolbar">
                <div class="vx-seg" id="historyKinds">${HISTORY_KINDS.map(([k, l]) =>
                    `<button class="vx-seg-btn${k === HISTORY.kind ? ' active' : ''}" data-kind="${k}"
                             onclick="historyKind('${k}')">${l}</button>`).join('')}</div>
                <label class="vx-search"><svg class="ui-ico"><use href="#i-search"/></svg>
                    <input type="text" id="historyQ" placeholder="Find a name …"
                           oninput="historySearch(this.value)"></label>
                <span class="vx-gap"></span>
                <span class="map-small" id="historyCounts"></span>
            </div>
            <div class="vx-scroll"><div class="vx-list" id="historyList"></div></div>`;
        historyLoad(true);
        return;
    }
    if (HISTORY.stale) historyLoad(true);
}

/* Something was decided elsewhere (the map, Assign, People). Fetched again
 * now if the tab is on screen, else when it is next opened. */
function historyRefresh() {
    HISTORY.stale = true;
    HISTORY.preview = {};
    HISTORY.epoch++;
    if (typeof VX !== 'undefined' && VX.tab === 'history') historyLoad(true);
}

function historyKind(k) {
    HISTORY.kind = k;
    document.querySelectorAll('#historyKinds .vx-seg-btn').forEach(
        b => b.classList.toggle('active', b.dataset.kind === k));
    historyLoad(true);
}

let _historyTimer = null;
function historySearch(v) {
    HISTORY.q = v.trim();
    clearTimeout(_historyTimer);
    _historyTimer = setTimeout(() => historyLoad(true), 250);
}

async function historyLoad(reset) {
    // A new filter (reset) replaces a load in flight — its answer is
    // dropped below. Only "load more" waits for the running load, so the
    // same page isn't appended twice.
    if (HISTORY.loading && !reset) return;
    const seq = ++HISTORY.seq;
    HISTORY.loading = true;
    HISTORY.stale = false;
    const p = new URLSearchParams({limit: 40, kind: HISTORY.kind});
    if (HISTORY.q) p.set('q', HISTORY.q);
    if (!reset && HISTORY.entries.length) p.set('before', HISTORY.entries[HISTORY.entries.length - 1].id);
    if (reset) {
        const list = document.getElementById('historyList');
        if (list && !HISTORY.entries.length) list.innerHTML = '<div class="vx-empty">Loading …</div>';
    }
    try {
        const d = await (await fetch('/api/voices/history?' + p)).json();
        if (seq !== HISTORY.seq) return;
        HISTORY.entries = reset ? (d.entries || []) : HISTORY.entries.concat(d.entries || []);
        HISTORY.more = !!d.more;
        HISTORY.counts = d.counts || null;
    } catch (e) {
        if (seq !== HISTORY.seq) return;
        const list = document.getElementById('historyList');
        if (list) list.innerHTML = `<div class="vx-note err">History unavailable: ${esc(String(e))}</div>`;
        HISTORY.loading = false;
        return;
    }
    HISTORY.loading = false;
    historyDraw();
}

function historyTitle(e) {
    if (e.kind === 'trash') return 'Discarded';
    if (e.kind === 'restore') return 'Restored from the trash';
    if (e.kind === 'unname') return 'Name removed';
    if (e.kind === 'rename') return `Renamed to ${e.name || ''}`;
    return `Named ${e.name || ''}`;
}

const HISTORY_ORIGIN = {
    assignment: 'Assign', map: 'Map', history: 'History', people: 'People',
    human: 'Map', 'map-group': 'Map',
};

function historyDraw() {
    const list = document.getElementById('historyList');
    if (!list) return;
    const c = HISTORY.counts;
    const counts = document.getElementById('historyCounts');
    if (counts && c) {
        counts.textContent = `${vxN(c.decisions)} decision${c.decisions === 1 ? '' : 's'} · ${vxN(c.voices)} voice${c.voices === 1 ? '' : 's'}`
            + (c.undone ? ` · ${vxN(c.undone)} undone` : '');
    }
    vxState('history', c ? `${vxN(c.decisions)} decision${c.decisions === 1 ? '' : 's'}` : '');
    if (!HISTORY.entries.length) {
        list.innerHTML = `<div class="vx-empty">${HISTORY.q || HISTORY.kind !== 'all'
            ? 'Nothing matches.' : 'No decisions yet — name or discard voices on the Map or in Assign.'}</div>`;
        return;
    }
    let html = '', day = null;
    for (const e of HISTORY.entries) {
        const d = vxDay(e.at);
        if (d !== day) { day = d; html += `<div class="vx-day">${esc(d)}</div>`; }
        html += historyEntryHtml(e);
    }
    if (HISTORY.more) {
        html += `<div style="text-align:center;margin-top:12px">
            <button class="vx-btn" onclick="historyLoad(false)">Load older decisions</button></div>`;
    }
    list.innerHTML = html;
    HISTORY.open.forEach(id => historyPreviewLoad(id));
}

function historyEntryHtml(e) {
    const icon = {name: '✓', trash: '🗑', restore: '↩', unname: '✕', rename: '✎'}[e.kind] || '•';
    const direct = e.voices - (e.spread || 0);
    const meta = [
        `${vxN(e.voices)} ${e.voices === 1 ? 'voice' : 'voices'}`
            + (e.spread ? ` (${vxN(direct)} chosen + ${vxN(e.spread)} by spreading)` : ''),
        `${vxN(e.speakers)} ${e.speakers === 1 ? 'speaker' : 'speakers'}`,
        `${vxN(e.recordings)} ${e.recordings === 1 ? 'recording' : 'recordings'}`,
        vxHours(e.speech_hours),
        HISTORY_ORIGIN[e.origin] || e.origin || '',
    ].filter(Boolean).join(' · ');
    const tags = (e.undone ? `<span class="vx-tag undone">undone ${esc(vxClock(e.undone_at))}</span>` : '')
        + (e.reconstructed ? `<span class="vx-tag" title="Decided before the history existed — rebuilt from the times the names and the trash were written">rebuilt</span>` : '');
    const open = HISTORY.open.has(e.id);
    return `<div class="vx-entry${e.undone ? ' undone' : ''}" id="hist-${e.id}">
        <div class="vx-entry-row" onclick="historyToggle(${e.id})">
            <span class="vx-kind ${esc(e.kind)}" style="font-size:1rem">${icon}</span>
            <div class="vx-entry-main">
                <div class="vx-entry-title">${esc(historyTitle(e))}${tags}</div>
                <div class="vx-entry-meta">${esc(meta)}</div>
            </div>
            <span class="vx-entry-when">${esc(vxClock(e.at))}</span>
            ${e.undone
                ? `<button class="vx-btn" onclick="event.stopPropagation();historyRedo(${e.id})"
                           title="Put it in force again — as far as nothing was decided since">Redo</button>`
                : `<button class="vx-btn" onclick="event.stopPropagation();historyUndo(${e.id})"
                           title="Put back what this click changed — nothing else"><svg class="ui-ico"><use href="#i-undo"/></svg>Undo</button>`}
        </div>
        ${open ? `<div class="vx-entry-detail" id="hist-detail-${e.id}">${historyDetailHtml(e)}</div>` : ''}
    </div>`;
}

function historyDetailHtml(e) {
    const p = HISTORY.preview[e.id];
    const ask = HISTORY.asks[e.id];
    let info = '<div class="map-small" style="margin-top:8px">Checking what an undo would change …</div>';
    if (p) {
        const left = (p.left || []).map(([why, n]) => `${vxN(n)} ${esc(why)}`).join(', ');
        info = `<div class="vx-note" style="margin-top:8px">${p.redo
                ? `Redo would put ${vxN(p.moves)} voice${p.moves === 1 ? '' : 's'} back in force.`
                : `Undo would put back ${vxN(p.moves)} of ${vxN(e.voices)} voice${e.voices === 1 ? '' : 's'}.`}
            ${left ? ` Left alone: ${left} — a later decision owns them.` : ''}</div>`
            + (p.later_same_name && p.later_same_name.length
                ? `<div class="vx-note warn">Later namings of "${esc(e.name)}" spread against a centroid
                   this one was part of: ${p.later_same_name.map(x => `${vxWhen(x.at)} (${vxN(x.voices)} voice${x.voices === 1 ? '' : 's'})`).join(', ')}
                   — if this naming was wrong, look at them too.</div>` : '');
    }
    const decide = ask ? `
        <div class="vx-card" style="margin-top:10px">
            <div class="vx-sub">${vxN(ask.vector_ids.length)} ${ask.vector_ids.length === 1 ? 'voice is open again. Decide it' : 'voices are open again. Decide them'} anew:</div>
            <div class="vx-decide-row" style="margin-top:8px">
                <input type="text" id="histName-${e.id}" placeholder="Who is this?" list="vxPeopleList"
                       autocomplete="off" onkeydown="if(event.key==='Enter')historyDecide(${e.id}, 'name')">
                <button class="vx-btn primary" onclick="historyDecide(${e.id}, 'name')">Apply</button>
                <button class="vx-btn danger" onclick="historyDecide(${e.id}, 'trash')">Trash</button>
                <button class="vx-btn" onclick="historyDecide(${e.id}, 'open')">Leave open</button>
            </div>
        </div>` : '';
    return `${vxSamplesHtml(e.samples)}${info}${decide}
        <div class="vx-entry-actions">
            <button class="vx-btn" onclick="historyOnMap(${e.id})"
                    title="Light exactly these voices on the map">⌖ Show on the map</button>
            ${!e.undone && ['name', 'trash'].includes(e.kind)
                ? `<button class="vx-btn" onclick="historyUndo(${e.id}, true)"
                           title="Undo, and decide these voices anew on the Assign page">Undo and ask again</button>` : ''}
        </div>`;
}

function historyToggle(id) {
    if (HISTORY.open.has(id)) HISTORY.open.delete(id); else HISTORY.open.add(id);
    historyRedraw(id);
}

/* What an undo (or redo) of an open entry would do — fetched whenever an
 * open entry has none: when it opens, and again after an undo, a redo or
 * a reload dropped the old one. Before, only opening fetched it, and after
 * an undo the entry said "Checking …" for good. An answer that arrives after a change is dropped. */
async function historyPreviewLoad(id) {
    if (!HISTORY.open.has(id) || HISTORY.preview[id] || HISTORY.previewing.has(id)) return;
    if (!HISTORY.entries.some(x => x.id === id)) return;
    HISTORY.previewing.add(id);
    const epoch = HISTORY.epoch;
    let p;
    try {
        p = await (await fetch(`/api/voices/history/${id}`)).json();
    } catch (e) {
        p = {moves: 0, left: [[String(e), 0]]};
    }
    HISTORY.previewing.delete(id);
    if (epoch !== HISTORY.epoch) { historyPreviewLoad(id); return; }
    HISTORY.preview[id] = p;
    historyRedraw(id);
}

function historyRedraw(id) {
    const e = HISTORY.entries.find(x => x.id === id);
    const el = document.getElementById(`hist-${id}`);
    if (!e || !el) return;
    // Players inside the entry would lose their state — pause first
    el.querySelectorAll('audio').forEach(a => a.pause());
    el.outerHTML = historyEntryHtml(e);
    historyPreviewLoad(id);
}

async function historyUndo(id, askInAssign = false) {
    let d;
    try {
        d = await vxJson(`/api/voices/history/${id}/undo`, {});
    } catch (e) {
        Toast.error('Undo failed: ' + (e.message || e));
        return;
    }
    historyAfterChange(id, d);
    const left = (d.left || []).map(([why, n]) => `${vxN(n)} ${why}`).join(', ');
    Toast.success(`Undone — ${vxN(d.changed)} voice${d.changed === 1 ? '' : 's'} back` + (left ? ` (left alone: ${left})` : ''));
    if (d.ask) {
        if (askInAssign) {
            ASSIGN.suggestion = {...d.ask, asked: d.ask.decided, done: false};
            vxTab('assign');
            return;
        }
        HISTORY.asks[id] = d.ask;
        HISTORY.open.add(id);
        if (typeof vxPeopleNames === 'function') vxPeopleNames();
    }
    historyRedraw(id);
}

async function historyRedo(id) {
    let d;
    try {
        d = await vxJson(`/api/voices/history/${id}/redo`, {});
    } catch (e) {
        Toast.error('Redo failed: ' + (e.message || e));
        return;
    }
    delete HISTORY.asks[id];
    historyAfterChange(id, d);
    Toast.success(`Redone — ${vxN(d.changed)} voice${d.changed === 1 ? '' : 's'}`);
    historyRedraw(id);
}

function historyAfterChange(id, d) {
    const i = HISTORY.entries.findIndex(x => x.id === id);
    if (i >= 0 && d.entry) HISTORY.entries[i] = d.entry;
    HISTORY.epoch++;
    // The Assign page's progress bar follows an undo made here
    if (d.progress && typeof ASSIGN !== 'undefined') ASSIGN.progress = d.progress;
    delete HISTORY.preview[id];
    vxMapChanged();
    if (typeof assignRecentLoad === 'function') assignRecentLoad();
    if (typeof peopleRefresh === 'function') peopleRefresh();
}

/* Decide the voices an undo freed — the same apply as the Assign page, so a
 * name spreads the same way. */
async function historyDecide(id, kind) {
    const ask = HISTORY.asks[id];
    if (!ask) return;
    if (kind === 'open') {
        delete HISTORY.asks[id];
        historyRedraw(id);
        return;
    }
    const body = {vector_ids: ask.vector_ids};
    if (kind === 'trash') body.trash = true;
    else {
        const f = document.getElementById(`histName-${id}`);
        const name = (f && f.value || '').trim();
        if (!name) { if (f) f.focus(); return; }
        body.name = name;
    }
    try {
        const d = await vxJson('/api/speaker/assignment/apply', body);
        Toast.success(kind === 'trash' ? `${vxN(d.discarded)} voice${d.discarded === 1 ? '' : 's'} discarded`
            : `${vxN(d.set_flag)} voice${d.set_flag === 1 ? '' : 's'} named "${body.name}"`
              + (d.spread ? `, ${vxN(d.spread)} more by spreading` : ''));
    } catch (e) {
        Toast.error('Failed: ' + (e.message || e));
        return;
    }
    delete HISTORY.asks[id];
    vxMapChanged();
    if (typeof vxPeopleNames === 'function') vxPeopleNames(true);
    if (typeof assignRecentLoad === 'function') assignRecentLoad();
    historyLoad(true);
}

/* Exactly these voices on the map — discarded ones on the trash map. */
async function historyOnMap(id) {
    const e = HISTORY.entries.find(x => x.id === id);
    let d;
    try {
        d = await (await fetch(`/api/voices/history/${id}/voices`)).json();
    } catch (err) {
        Toast.error('Voices unavailable: ' + err);
        return;
    }
    const trash = !!(e && e.kind === 'trash' && !e.undone);
    mapShowVoices(d.ids || [], `${historyTitle(e || {})} · ${vxWhen(e && e.at)}`, trash);
}

/* Light a set of voices, given by id, on the map — pinned like a located
 * recording, so no minimum group size hides them. */
async function mapShowVoices(ids, title, trash = false) {
    vxOpen('map');
    if (!!GRAPH.trash !== !!trash) {
        GRAPH.trash = !!trash;
        graphTrashBar();
        mapLoadGraph();
    }
    GRAPH.locate = {file: '', name: title, label: '', t: undefined,
                    windows: ids.map(id => ({id, trashed: false}))};
    await mapWhenLoaded();
    mapLocateApply(true);
}
