/* ══════════════════════════════════════════════════════════════════════════
 *  PEOPLE — the named people, one row each
 *
 *  What is known about a person: how many voices carry the name (and how
 *  many got it by spreading), how much speech, in how many recordings,
 *  from when to when (the dates in the file names). From each row: listen
 *  through every voice of the person (voices_review.js), show on the map,
 *  the conversations they take part in (the search view's Conversations
 *  tab), extract the voice, rename — or merge into another name ("Alice
 *  Temp 1" into "Alice"), which is one decision in the History and can be
 *  undone.
 *
 *  How many voices have their speech spans stored is only a column where
 *  some voices lack them — otherwise the column would only puzzle.
 * ══════════════════════════════════════════════════════════════════════════ */

const PEOPLE = {
    built: false,
    rows: [],
    q: '',
    renaming: null,
    legacy: false,          // some voices lack stored speech spans
    stale: false,
};

function peopleShow() {
    const page = document.getElementById('vxPagePeople');
    if (!page) return;
    if (!PEOPLE.built) {
        PEOPLE.built = true;
        page.innerHTML = `
            <div id="peopleMain" class="vx-page-inner">
                <div class="vx-toolbar">
                    <label class="vx-search"><svg class="ui-ico"><use href="#i-search"/></svg>
                        <input type="text" id="peopleQ" placeholder="Find a person …"
                               oninput="PEOPLE.q=this.value.trim().toLowerCase();peopleDraw()"></label>
                    <span class="vx-gap"></span>
                    <span class="map-small" id="peopleCounts"></span>
                </div>
                <div class="vx-scroll"><div class="vx-list" id="peopleList"></div></div>
            </div>
            <div id="peopleReview" class="vx-page-inner" hidden></div>`;
        peopleLoad();
        return;
    }
    if (PEOPLE.stale && !(typeof REVIEW !== 'undefined' && REVIEW.person)) peopleLoad();
}

function peopleRefresh() {
    PEOPLE.stale = true;
    if (typeof VX !== 'undefined' && VX.tab === 'people') peopleLoad();
}

async function peopleLoad() {
    PEOPLE.stale = false;
    const list = document.getElementById('peopleList');
    if (list && !PEOPLE.rows.length) list.innerHTML = '<div class="vx-empty">Loading …</div>';
    try {
        const d = await (await fetch('/api/voices/people')).json();
        PEOPLE.rows = d.people || [];
    } catch (e) {
        if (list) list.innerHTML = `<div class="vx-note err">People unavailable: ${esc(String(e))}</div>`;
        return;
    }
    peopleDraw();
}

/* The same colour the map gives the name, where the map knows it. */
function peopleColour(name) {
    try {
        const idx = typeof graphGroupIndex === 'function' ? graphGroupIndex() : null;
        if (idx) {
            for (const e of idx.values()) if (e.g.name === name) return e.color;
        }
    } catch (e) { /* no map loaded */ }
    return GRAPH_COLORS[graphColourIndex(name)];
}

function peopleDraw() {
    const list = document.getElementById('peopleList');
    if (!list) return;
    const rows = PEOPLE.rows.filter(p => !PEOPLE.q || p.name.toLowerCase().includes(PEOPLE.q));
    const named = PEOPLE.rows.reduce((n, p) => n + p.voices, 0);
    const people = PEOPLE.rows.filter(p => p.voices).length;
    const counts = document.getElementById('peopleCounts');
    if (counts) counts.textContent = `${vxN(people)} ${people === 1 ? 'person' : 'people'} · ${vxN(named)} named voice${named === 1 ? '' : 's'}`;
    vxState('people', `${vxN(people)} ${people === 1 ? 'person' : 'people'}`);
    if (!rows.length) {
        list.innerHTML = `<div class="vx-empty">${PEOPLE.rows.length ? 'Nobody matches.'
            : 'Nobody named yet — name voices on the Map or in Assign.'}</div>`;
        return;
    }
    // Spans only where some voices lack them
    PEOPLE.legacy = PEOPLE.rows.some(p => p.voices && p.with_spans < p.voices);
    list.innerHTML = `<div class="vx-card${PEOPLE.legacy ? ' vx-people-legacy' : ''}" style="padding:4px 0">
        <div class="vx-people-row vx-people-head">
            <span>Person</span><span class="vx-num">Named voices</span><span class="vx-num">Speech</span>
            <span class="vx-num">Recordings</span>${PEOPLE.legacy ? '<span class="vx-num" title="Voices with stored speech spans">With spans</span>' : ''}
            <span>Recorded</span><span></span>
        </div>
        ${rows.map(peopleRowHtml).join('')}</div>`;
}

function peopleRowHtml(p) {
    const spans = p.voices ? Math.round(100 * p.with_spans / p.voices) : 0;
    const when = p.first_date
        ? (p.first_date.slice(0, 4) === p.last_date.slice(0, 4) ? p.first_date.slice(0, 4)
           : `${p.first_date.slice(0, 4)}–${p.last_date.slice(0, 4)}`)
        : '—';
    const key = escJs(p.name);
    const renaming = PEOPLE.renaming === p.name ? `
        <div class="vx-decide-row" style="padding:0 14px 12px">
            <input type="text" id="peopleNewName" value="${esc(p.name)}" list="vxPeopleList"
                   onkeydown="if(event.key==='Enter')peopleRename('${key}')">
            <button class="vx-btn primary" onclick="peopleRename('${key}')">Rename</button>
            <button class="vx-btn" onclick="PEOPLE.renaming=null;peopleDraw()">Cancel</button>
            <span class="map-small">An existing name merges the two — the History can undo it.</span>
        </div>` : '';
    return `<div class="vx-people-row" style="border-top:1px solid var(--border)">
            <span class="vx-people-name"><span class="map-point" style="background:${peopleColour(p.name)}"></span>
                <span title="${esc(p.name)}">${esc(p.name)}</span></span>
            <span class="vx-num" title="${vxN(p.spread)} of them by spreading">${vxN(p.voices)}
                <span class="map-small">${p.spread ? `(${vxN(p.spread)} spread)` : ''}</span></span>
            <span class="vx-num">${vxHours(p.speech_hours)}</span>
            <span class="vx-num">${vxN(p.recordings)}</span>
            ${PEOPLE.legacy ? `<span class="vx-num" title="Voices with stored speech spans — a voice without them has the window's first 30 s as its sample">${spans} %</span>` : ''}
            <span class="vx-muted">${esc(when)}</span>
            <span class="vx-people-actions">
                ${p.voices ? `<button class="vx-btn primary" onclick="reviewOpen('${key}')" title="Every voice of ${esc(p.name)} in a list — with autoplay, filters, and a way to correct each one">▶ Listen through</button>` : ''}
                <button class="vx-btn" onclick="peopleOnMap('${key}')" title="The map, filtered to this name">Map</button>
                <button class="vx-btn" onclick="peopleConversations('${key}')"
                        title="The conversations ${esc(p.name)} takes part in — who with, how long, how they talk">Conversations</button>
                <button class="vx-btn" onclick="peopleExtract('${key}')" title="Cut this voice out as clips with their sentences">Extract</button>
                <button class="vx-btn" onclick="PEOPLE.renaming='${key}';peopleDraw();document.getElementById('peopleNewName')?.select()"
                        title="Rename — or merge into another name">Rename</button>
            </span>
        </div>
        ${renaming}`;
}

function peopleOnMap(name) {
    vxOpen('map');
    const f = document.getElementById('mapFText');
    const w = document.getElementById('mapFWhere');
    if (f) f.value = name;
    if (w) w.value = 'person';
    GRAPH.filter.scope = 'person';
    mapWhenLoaded().then(() => graphFilterSet('text', name));
}

// The Conversations tab of this window, filtered to the person: who they
// talk with (the same view as Find's).
function peopleConversations(name) {
    if (typeof cvShowInVoices === 'function') cvShowInVoices({people: [name]});
}

function peopleExtract(name) {
    if (typeof extractFor === 'function') extractFor(name);
}

async function peopleRename(old) {
    const f = document.getElementById('peopleNewName');
    const to = (f && f.value || '').trim();
    if (!to || to === old) { PEOPLE.renaming = null; peopleDraw(); return; }
    const exists = PEOPLE.rows.some(p => p.name.toLowerCase() === to.toLowerCase()
                                         && p.name.toLowerCase() !== old.toLowerCase());
    if (exists && !confirm(`Merge all voices of "${old}" into "${to}"?\n\nThe History can undo it.`)) return;
    let d;
    try {
        d = await vxJson('/api/voices/people/rename', {from: old, to});
    } catch (e) {
        Toast.error('Rename failed: ' + (e.message || e));
        return;
    }
    PEOPLE.renaming = null;
    vxUndoToast(d.decision, d.merged ? `"${old}" merged into "${to}"` : `"${old}" is now "${to}"`);
    vxMapChanged();
    if (typeof historyRefresh === 'function') historyRefresh();
    if (typeof vxPeopleNames === 'function') vxPeopleNames(true);
    peopleLoad();
}
