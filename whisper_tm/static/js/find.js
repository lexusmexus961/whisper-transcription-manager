// ── Find (api/find.py, find/agent.py) — the search view's start tab ──────────
//
// One description in your own words — what is said, seen or heard, who is
// in it, what the file was called. The description is searched by meaning
// at once, the language model plans the rest, every search runs at once,
// and each fills a tile of its own as it finishes; "Best matches" brings
// them together — each recording once, its other moments behind "+n more
// moments" — and the language model checks the best ones (one per
// recording first, with the still when it sees pictures, with the facts
// the program knows) and orders them: may be it, not checked, rejects.
//
// A tile can be maximized (⤢; Esc or ⤡ goes back), and its ↗ opens the
// same search in its own tab. A click on a hit — or Enter on it; ↑ ↓ move
// through a list — shows it on the right and plays it there: a clip of
// the moment cut from the recording (from a few seconds before it, a
// passage to its end), what was said, why it was found — and opens it in
// the player or its transcript. "Ask about these results" carries the run
// into Ask.
//
// The same moment in several files (the .mp4, .mkv and .webm of one video,
// a backup copy) is one hit with "⧉ n suspected duplicates": the server
// folds them by what the files hold, never by their names (core/copies.py).
// They stay one click away — in Best matches below the hit, everywhere in
// the preview.
//
// A description typed in the top bar comes here (searchview.js routes a
// few keywords to the text instead); one run at a time — a new one stops
// the one before, on the server too. Polling pauses while the tab isn't in
// view, and the searches' lists are only sent when they changed (erev).

const FD_TILES = [
    { id: 'best', icon: '★', title: 'Best matches', hint: 'All searches together — each recording once, checked by the language model' },
    { id: 'words', icon: '💬', title: 'Said — word for word', hint: 'The full-text index, best match first: each phrase, then its words alone', tab: 'text' },
    { id: 'meaning', icon: '🧠', title: 'Said — by meaning', hint: 'The semantic search of the transcripts — with the description itself and with the plan’s sentence', tab: 'text' },
    { id: 'seen', icon: '🖼', title: 'Seen', hint: 'Images & sounds: stills, descriptions, clips', tab: 'media' },
    { id: 'heard', icon: '🔊', title: 'Heard', hint: 'Images & sounds: sounds', tab: 'media' },
    { id: 'files', icon: '📁', title: 'In the files named like that', hint: 'Every search again, inside the files whose name has what you remember — they also lift those files in Best matches', tab: 'files' },
    { id: 'people', icon: '👤', title: 'In what they say', hint: 'Every search again, in what the named people say (the voice library) — they also lift their recordings', tab: 'text' },
];
const FD_EXAMPLES = [
    'Alice tells a story at a birthday party and says "that\'s the best day ever" — a WhatsApp video',
    'a dog barks when the doorbell rings',
    'someone explains a recipe in the kitchen while water boils',
];
const FD_POLL_MS = 700;
// Seconds a preview clip starts before its moment (find/agent.CLIP_BEFORE_SEC)
const FD_CLIP_BEFORE = 4;
const FD_VIDEO_EXT = /\.(mp4|m4v|mkv|webm|avi|mov|wmv|flv|mpe?g|ts|m2ts|mts|3gp|ogv|vob)$/i;
// A judged moment below this is a reject (find/judge.REJECT_BELOW)
const FD_REJECT_BELOW = 4;

const _fd = {
    run: null,          // the state from the server
    id: null, rev: -1, erev: -1, timer: null, seq: 0, polling: false,
    max: null,          // the maximized tile
    // The hit on the right — by what it is, not by its row: the list
    // reorders while the language model checks, and a row number would
    // swap the clip being watched for another one
    pick: null,         // {tile, id, dup, item}
    open: new Set(),    // Best matches whose duplicates are unfolded (by id)
    more: new Set(),    // … and whose recording's other moments are
    sig: {},            // tile → what it last showed (no redraw without a change)
    shown: '',          // which tiles the grid holds
    lastQuery: '',
    incoming: null,     // a description from the top bar, to run when the tab opens
    ptok: 0,            // the polling loop that may go on (fdPoll)
    logOpen: null,      // "What it did" opened or closed by hand (null: open while live)
};

function fdId(it) { return `${it.key}|${Number(it.t) || 0}`; }

function fdClock(sec) {
    sec = Math.max(0, Math.floor(Number(sec) || 0));
    const h = Math.floor(sec / 3600), m = Math.floor((sec % 3600) / 60), s = sec % 60;
    return h ? `${h}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}` : `${m}:${String(s).padStart(2, '0')}`;
}

function fdTile(id) { return FD_TILES.find(t => t.id === id); }

// ── Opening the tab ─────────────────────────────────────────────────────────

// searchview.js hands over a description from the top bar
function fdIncoming(q) { _fd.incoming = (q || '').trim() || null; }

function fdOpen() {
    const ta = document.getElementById('fdQuery');
    if (!ta) return;
    if (!document.getElementById('fdGrid').children.length) fdRenderEmpty();
    // A description from the top bar replaces what the field held and runs —
    // a new one is never dropped
    const q = _fd.incoming;
    _fd.incoming = null;
    if (q && q !== _fd.lastQuery) { ta.value = q; fdRun(); return; }
    // Back to a run that went on while the tab wasn't in view
    if (_fd.id && _fd.run && !_fd.run.ended && !_fd.polling) fdPoll(_fd.seq);
    setTimeout(() => ta.focus(), 0);
}

// Leaving the tab or the search: the preview stops (the main player plays
// on) and so does the polling — the run goes on on the server
function fdLeave() {
    document.querySelectorAll('#fdPreview video, #fdPreview audio').forEach(m => { try { m.pause(); } catch (e) { /* gone */ } });
    clearTimeout(_fd.timer);
    _fd.polling = false;
    _fd.ptok++;         // a request still on its way ends its loop
}

function fdVisible() {
    const ov = document.getElementById('svOverlay');
    const pane = document.getElementById('svPaneFind');
    return !!(pane && pane.classList.contains('active') && (!ov || ov.classList.contains('visible')));
}

function fdRenderEmpty() {
    const grid = document.getElementById('fdGrid');
    const agent = document.getElementById('fdAgent');
    if (agent) agent.innerHTML = `<div class="fd-agent-line"><span class="fd-dot idle"></span>
        Describe the moment in your own words — the language model plans the searches, runs them all at once and checks the best.</div>
        <div class="fd-examples">${FD_EXAMPLES.map(x => `<button class="fd-example" onclick="fdTry(this)">${esc(x)}</button>`).join('')}</div>`;
    _fd.shown = '';
    _fd.sig = {};
    fdLayout(FD_TILES.slice(1).map(t => t.id));
    grid.innerHTML = FD_TILES.map(t => fdTileHtml(t, {status: 'idle'})).join('');
}

function fdTry(btn) {
    const ta = document.getElementById('fdQuery');
    ta.value = btn.textContent;
    fdRun();
}

function fdKey(e) {
    if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) { e.preventDefault(); fdRun(); }
}

// ── A run ───────────────────────────────────────────────────────────────────

async function fdRun() {
    const ta = document.getElementById('fdQuery');
    const q = (ta?.value || '').trim();
    if (q.length < 3) { ta?.focus(); return; }
    // The duplicates asked for in so many words: no moment to find
    // (other phrasings: the plan's "asks_for", fdApply)
    const dups = typeof amDupIntent === 'function' ? amDupIntent(q) : null;
    if (dups && typeof amShowDuplicates === 'function') { amShowDuplicates(dups); return; }
    const seq = ++_fd.seq;
    clearTimeout(_fd.timer);
    _fd.polling = false;
    _fd.lastQuery = q;
    _fd.max = null;
    _fd.pick = null;
    _fd.open = new Set();
    _fd.more = new Set();
    _fd.sig = {};
    _fd.shown = '';
    _fd.logOpen = null;
    document.getElementById('fdGrid')?.classList.remove('maxed');
    document.getElementById('fdPreview').innerHTML = '';
    document.getElementById('fdMain')?.classList.remove('previewing');
    fdButtons(true);
    let r;
    // The server stops the run before when this one starts
    try { r = await api('/api/find', { method: 'POST', body: { query: q } }); }
    catch (e) {
        fdButtons(false);
        document.getElementById('fdAgent').innerHTML = `<div class="fd-agent-line err">${esc(e.message)}</div>`;
        return;
    }
    if (seq !== _fd.seq) return;
    _fd.id = r.id;
    _fd.rev = -1;
    _fd.erev = -1;
    _fd.run = null;
    fdApply(r.state);
    fdPoll(seq);
}

/* One loop at a time: each call takes a new token, and an answer that comes
 * back to an older one ends there — leaving the tab and coming back before
 * the request returned started a second loop that polled at double rate. */
async function fdPoll(seq) {
    if (seq !== _fd.seq || !_fd.id) return;
    const tok = ++_fd.ptok;
    _fd.polling = true;
    let d = null, gone = false;
    try { d = await api(`/api/find/${_fd.id}?rev=${_fd.rev}&erev=${_fd.erev}`); }
    catch (e) { gone = e.status === 404; }
    if (seq !== _fd.seq || tok !== _fd.ptok) return;
    if (gone) { fdGone(); return; }
    if (d && !d.same) fdApply(d);
    if (_fd.run && _fd.run.ended) { _fd.polling = false; fdButtons(false); return; }
    // Out of view: the run goes on, the page asks again when it is back
    if (!fdVisible()) { _fd.polling = false; return; }
    _fd.timer = setTimeout(() => fdPoll(seq), FD_POLL_MS);
}

// A run started elsewhere (Ask's find_moments) shown here
function fdShowRun(id) {
    if (!id) return;
    const seq = ++_fd.seq;
    clearTimeout(_fd.timer);
    Object.assign(_fd, { id, rev: -1, erev: -1, run: null, pick: null, max: null, sig: {}, shown: '',
                         open: new Set(), more: new Set() });
    document.getElementById('fdPreview').innerHTML = '';
    document.getElementById('fdMain')?.classList.remove('previewing');
    fdPoll(seq).then(() => {
        const ta = document.getElementById('fdQuery');
        if (ta && _fd.run) { ta.value = _fd.run.query || ''; _fd.lastQuery = ta.value; }
        // A run that goes on can be stopped here too
        if (seq === _fd.seq && _fd.run) fdButtons(!_fd.run.ended);
    });
}

// The server doesn't know the run (it restarted) — no polling into the void
function fdGone() {
    _fd.polling = false;
    fdButtons(false);
    if (_fd.run) _fd.run = { ...(_fd.run), ended: _fd.run.ended || Date.now() / 1000, status: 'gone' };
    const box = document.getElementById('fdAgent');
    if (box) box.insertAdjacentHTML('afterbegin', '<div class="fd-agent-line err"><span class="fd-dot err"></span>'
        + 'This search is gone — the server was restarted. Run it again.</div>');
    fdRenderTiles();
}

async function fdStop() {
    if (!_fd.id) return;
    try { await api(`/api/find/${_fd.id}/stop`, { method: 'POST' }); } catch (e) { /* it ends anyway */ }
}

function fdButtons(running) {
    const go = document.getElementById('fdGo'), stop = document.getElementById('fdStop');
    if (go) go.disabled = running;
    if (stop) stop.hidden = !running;
}

function fdApply(state) {
    // Unchanged lists aren't sent again — the page keeps the ones it has
    if (state.engines_same && _fd.run) {
        for (const [n, e] of Object.entries(state.engines || {})) {
            const old = (_fd.run.engines || {})[n];
            if (old && old.items) e.items = old.items;
        }
    }
    const before = _fd.run;
    _fd.run = state;
    _fd.rev = state.rev;
    _fd.erev = state.erev ?? -1;
    fdRenderAgent();
    fdRenderTiles();
    if (_fd.pick) fdRenderPreview();
    // The plan says it asks for the duplicates, not a moment: they are
    // shown in Same audio — once, while this tab is in view
    if (state.status === 'duplicates' && before?.status !== 'duplicates' && fdVisible()
            && typeof amShowDuplicates === 'function') {
        const p = state.plan || {};
        amShowDuplicates({ kind: p.asks_for === 'contained' ? 'contains' : 'same',
                           q: (p.file_hints || [])[0] || '' });
    }
    // … or whole conversations — who took part, how long, how they talked
    // (find/agent.py's conversation_filters, by the API's parameter names):
    // shown in Conversations, once, while this tab is in view
    if (state.status === 'conversations' && before?.status !== 'conversations' && fdVisible()
            && typeof cvShow === 'function') {
        cvShow(state.conversation_filters || {});
    }
}

// The run's conversations once more — the agent line's button, for a run
// that handed over while the tab was not in view
function fdShowConversations() {
    if (_fd.run && typeof cvShow === 'function') cvShow(_fd.run.conversation_filters || {});
}

// A conversation in Ask with this run attached (find/agent.ask)
async function fdAsk() {
    if (!_fd.id) return;
    let d;
    try { d = await api(`/api/find/${_fd.id}/ask`, { method: 'POST' }); }
    catch (e) {
        if (typeof Toast !== 'undefined') Toast.error('No conversation: ' + e.message);
        return;
    }
    fdLeave();
    svTab('chat');
    if (typeof chatLoad === 'function') await chatLoad(d.conversation);
    if (typeof chatLoadList === 'function') chatLoadList();
    document.getElementById('chatInput')?.focus();
}

// ── The agent: its plan and what it does ────────────────────────────────────

const FD_STATUS = {
    planning: 'The language model plans the searches — the description is already searched by meaning',
    searching: 'Searching …',
    judging: 'Checking the best matches …',
    done: 'Done',
    stopped: 'Stopped — what was found stays',
    duplicates: 'This asks for recordings that exist more than once — Same audio shows them',
    conversations: 'This asks for whole conversations — Conversations shows them',
    failed: 'Failed',
    gone: 'Gone — run it again',
};

function fdRenderAgent() {
    const box = document.getElementById('fdAgent');
    const s = _fd.run;
    if (!box || !s) return;
    const p = s.plan;
    const eng = s.engines || {};
    const done = Object.values(eng).filter(e => ['done', 'skipped', 'failed', 'stopped'].includes(e.status)).length;
    let line = FD_STATUS[s.status] || s.status;
    if (s.status === 'searching') line = `Searching — ${done} of ${Object.keys(eng).length} done`;
    if (s.status === 'judging') line = `The language model checks the best matches — ${s.judged} of ${s.judge_total}`;
    if (s.ended) line += ` · ${Math.round(s.ended - s.started)} s`;
    const live = !s.ended;
    const chips = [];
    if (p) {
        for (const w of p.said_words || []) chips.push(['💬', `says “${w}”`, 'words']);
        if (p.said_meaning) chips.push(['🧠', p.said_meaning, 'meaning']);
        if (p.seen) chips.push(['🖼', p.seen, 'seen']);
        if (p.heard) chips.push(['🔊', p.heard, 'heard']);
        for (const n of p.people || []) chips.push(['👤', n, 'people']);
        for (const h of p.file_hints || []) chips.push(['📁', `name has “${h}”`, 'files']);
        if (p.media && p.media !== 'any') {
            chips.push([p.media === 'video' ? '🎞' : '🎙',
                (p.media === 'video' ? 'a video' : 'audio only') + (p.media_stated ? '' : ' (guessed)'), '']);
        }
    }
    // The searches the plan had nothing for — a line, not empty tiles
    const off = FD_TILES.filter(t => t.id !== 'best' && (eng[t.id] || {}).status === 'skipped')
        .map(t => `${t.icon} ${esc(fdTileTitle(t))} — ${esc(eng[t.id].note || 'not asked')}`);
    const log = (s.log || []).slice(-10).map(l => `<div><span class="fd-log-t">${l.t.toFixed(1)} s</span>${esc(l.text)}</div>`).join('');
    const canAsk = (s.best || []).length && s.status !== 'gone';
    box.innerHTML = `
        <div class="fd-agent-line">
            <span class="fd-dot ${s.status === 'failed' || s.status === 'gone' ? 'err' : live ? 'live' : 'ok'}"></span>
            <b>${esc(line)}</b>
            ${p ? `<span class="fd-summary">${esc(p.summary || '')}</span>` : ''}
            ${s.plan_source && s.plan_source !== 'llm' ? `<span class="fd-warn" title="${esc(s.plan_source)}">without the language model's plan</span>` : ''}
            ${s.error ? `<span class="fd-warn">${esc(s.error)}</span>` : ''}
            <span class="fd-gap"></span>
            ${s.status === 'conversations' && typeof cvShow === 'function' ? `<button class="vx-btn fd-ask-btn" onclick="fdShowConversations()"
                title="The conversations this asks for, in the Conversations tab">🗣 Show them</button>` : ''}
            ${canAsk ? `<button class="vx-btn fd-ask-btn" onclick="fdAsk()" title="A conversation in Ask about these results — they go along with every question">💬 Ask about these results</button>` : ''}
        </div>
        ${chips.length ? `<div class="fd-chips">${chips.map(([i, t, tile]) => tile
            ? `<button class="fd-chip fd-chip-${tile}" onclick="fdMax('${tile}')" title="Show this search">${i} ${esc(t)}</button>`
            : `<span class="fd-chip">${i} ${esc(t)}</span>`).join('')}</div>` : ''}
        ${off.length ? `<div class="fd-off">Not searched: ${off.join(' · ')}</div>` : ''}
        ${log ? `<details class="fd-log${live ? ' live' : ''}"${(_fd.logOpen ?? live) ? ' open' : ''}><summary onclick="_fd.logOpen = !this.parentNode.open">What it did</summary>${log}</details>` : ''}`;
}

// ── The tiles ───────────────────────────────────────────────────────────────

function fdTileState(id) {
    const s = _fd.run;
    if (!s) return { status: 'idle' };
    if (id === 'best') {
        const anyDone = Object.values(s.engines || {}).some(e => e.status === 'done' || (e.items || []).length);
        const status = s.status === 'judging' ? 'running' : s.ended ? 'done'
            : anyDone ? 'running' : 'waiting';
        return { status, items: s.best || [], count: fdBestGroups(s.best || []).length };
    }
    return (s.engines || {})[id] || { status: 'waiting' };
}

// A tile's title — the scoped searches name what they search in
function fdTileTitle(t) {
    const p = (_fd.run || {}).plan || {};
    if (t.id === 'files' && (p.file_hints || []).length) return `In files named “${p.file_hints.join('”, “')}”`;
    if (t.id === 'people' && (p.people || []).length) return `In what ${p.people.join(', ')} say${p.people.length === 1 ? 's' : ''}`;
    return t.title;
}

// The tiles the grid shows: the searches that run or ran — the ones the
// plan had nothing for are a line in the agent's bar
function fdShownTiles() {
    const eng = (_fd.run || {}).engines || {};
    if (!_fd.run) return FD_TILES.slice(1).map(t => t.id);
    return FD_TILES.slice(1).filter(t => (eng[t.id] || {}).status !== 'skipped').map(t => t.id);
}

// Rows of the grid for that many tiles: two columns beside Best matches,
// one when the preview narrows it (find.css)
function fdLayout(ids) {
    const grid = document.getElementById('fdGrid');
    if (!grid) return;
    const n = Math.max(1, ids.length);
    grid.style.setProperty('--fd-rows2', String(Math.ceil(n / 2)));
    grid.style.setProperty('--fd-rows1', String(n));
}

function fdTileHtml(t, st) {
    const items = st.items || [];
    const shown = t.id === 'best' ? st.count : items.length;
    const count = st.status === 'done' || (st.status === 'running' && shown) ? Number(shown || 0).toLocaleString() : '';
    const total = t.id !== 'best' && st.count != null && st.count !== items.length
        ? `${Number(st.count).toLocaleString()} found — the best ${items.length} shown` : '';
    const name = fdTileTitle(t);
    const title = t.id === 'best' && _fd.run && _fd.run.judge_total
        ? `${esc(name)} <span class="fd-tile-sub">${_fd.run.judged}/${_fd.run.judge_total} checked</span>` : esc(name);
    const ids = fdShownTiles();
    const wide = t.id !== 'best' && ids.length % 2 === 1 && ids[ids.length - 1] === t.id;
    return `<section class="fd-tile fd-tile-${t.id}${_fd.max === t.id ? ' max' : ''}${wide ? ' wide' : ''}" data-tile="${t.id}" data-state="${esc(st.status || "")}">
        <header class="fd-tile-head" ondblclick="fdMax('${t.id}')">
            <span class="fd-dot ${{running: 'live', done: 'ok', failed: 'err', skipped: 'off', stopped: 'off', waiting: 'wait'}[st.status] || 'idle'}"></span>
            <span class="fd-tile-icon">${t.icon}</span>
            <span class="fd-tile-title" title="${esc(t.hint)}">${title}</span>
            ${count ? `<span class="fd-tile-count"${total ? ` title="${esc(total)}"` : ''}>${count}</span>` : ''}
            ${st.ms ? `<span class="fd-tile-ms">${(st.ms / 1000).toFixed(1)} s</span>` : ''}
            <span class="fd-gap"></span>
            ${t.tab && st.status === 'done' ? `<button class="fd-tile-btn" onclick="fdOpenTab('${t.id}')" title="Open this search in its own tab">↗</button>` : ''}
            <button class="fd-tile-btn" onclick="fdMax('${t.id}')" title="${_fd.max === t.id ? 'Back to all tiles (Esc)' : 'Maximize'}">${_fd.max === t.id ? '⤡' : '⤢'}</button>
        </header>
        ${st.query ? `<div class="fd-tile-query" title="${esc(st.query)}">${esc(st.query)}</div>` : ''}
        <div class="fd-tile-body" role="listbox" aria-label="${esc(name)}">${fdTileBody(t.id, st)}</div>
    </section>`;
}

function fdSkeleton(n) {
    return `<div class="fd-skel">${'<div class="fd-skel-row"></div>'.repeat(n)}</div>`;
}

function fdTileBody(id, st) {
    if (st.status === 'idle') return `<div class="fd-quiet">${esc(fdTile(id).hint)}</div>`;
    if (st.status === 'waiting' || (st.status === 'running' && !(st.items || []).length)) return fdSkeleton(id === 'seen' ? 2 : 4);
    if (st.status === 'skipped' || st.status === 'stopped') return `<div class="fd-quiet">— ${esc(st.note || 'not asked')}</div>`;
    if (st.status === 'failed') return `<div class="fd-quiet err">${esc(st.error || 'failed')}</div>`;
    const items = st.items || [];
    const notes = [st.note, st.folded ? `${st.folded} suspected duplicate${st.folded === 1 ? '' : 's'} folded in` : '']
        .filter(Boolean);
    const note = notes.length ? `<div class="fd-note">${esc(notes.join(' · '))}</div>` : '';
    if (!items.length) return note + '<div class="fd-quiet">nothing found</div>';
    const max = _fd.max === id;
    if (id === 'best') return fdBestGroups(items.slice(0, max ? 60 : 40)).map((g, n) => fdBestRow(g.m, g.k, n, g.more)).join('');
    if (id === 'seen') return note + `<div class="fd-thumbs">${items.slice(0, max ? 60 : 18).map((it, k) => fdThumb(id, it, k)).join('')}</div>`;
    return note + items.slice(0, max ? 60 : 20).map((it, k) => fdRow(id, it, k)).join('');
}

function fdSel(tile, it, dup = null) {
    const p = _fd.pick;
    return p && p.tile === tile && p.id === fdId(it) && (p.dup || null) === dup ? ' sel' : '';
}

// "⧉ 3": the same moment in other files, folded into this hit
function fdDupBadge(it, tile, k) {
    const n = (it.dups || []).length;
    if (!n) return '';
    const list = it.dups.map(d => `${d.name} (${d.why})`).join('\n');
    const open = tile === 'best' && _fd.open.has(fdId(it));
    const tip = `${n} suspected duplicate${n === 1 ? '' : 's'} — the same moment in other files:\n${list}`;
    if (tile !== 'best') return `<span class="fd-dups-badge" title="${esc(tip)}">⧉ ${n}</span>`;
    return `<button class="fd-dups-badge${open ? ' open' : ''}" onclick="event.stopPropagation(); fdToggleDups(${k})"
        aria-expanded="${open ? 'true' : 'false'}" title="${esc(tip + '\n(click to show them)')}">⧉ ${n}</button>`;
}

function fdToggleDups(k) {
    const it = (fdTileState('best').items || [])[k];
    if (!it) return;
    const id = fdId(it);
    if (_fd.open.has(id)) _fd.open.delete(id); else _fd.open.add(id);
    _fd.sig.best = null;
    fdRenderTiles();
}

function fdSize(b) {
    b = Number(b) || 0;
    if (!b) return '';
    return typeof fmtBytes === 'function' ? fmtBytes(b) : `${Math.round(b / 1048576)} MB`;
}

function fdName(it) { return it.name || String(it.key || '').split('/').pop(); }

function fdFolder(it) {
    const key = String(it.key || '');
    return key.split('/').slice(0, -1).join('/');
}

// A row picks on click and on Enter/Space (the keyboard handler below).
// data-fid names the hit, not its place: the keyboard stays on it when the
// list reorders.
function fdPickAttrs(tile, k, dup = null, it = null, part = '') {
    const fid = it ? `${tile}|${fdId(it)}|${dup || ''}|${part}` : `${tile}|${k}`;
    return `role="option" tabindex="0" data-pick="${tile}" data-fid="${esc(fid)}" onclick="fdPick('${tile}', ${k}${dup ? `, '${escJs(dup)}'` : ''})"`;
}

function fdRow(tile, it, k) {
    const at = it.t != null ? ` · ${fdClock(it.t)}` : '';
    let text = '';
    if (tile === 'words') text = fdMark(it.text, (it.extra || {}).phrase);
    else if (tile === 'heard') text = esc((it.extra?.tags || []).join(', '));
    else text = esc(String(it.text || '').replace(/<[^>]+>/g, ''));
    const via = (it.extra || {}).via;
    const by = (tile === 'files' || tile === 'people') && Array.isArray(via) && via.length
        ? `<span class="fd-via" title="found inside by">${via.map(v => FD_EVIDENCE[v] || '').join('')}</span>` : '';
    const rel = (it.extra || {}).relevance;
    return `<div class="fd-row${fdSel(tile, it)}" ${fdPickAttrs(tile, k, null, it)} title="${esc(it.key || '')}">
        <span class="fd-row-head"><span class="fd-row-rank">${k + 1}</span><span class="fd-row-name">${esc(fdName(it))}${at}</span>
            ${by}${fdDupBadge(it, tile, k)}
            ${rel != null ? `<span class="fd-rel" style="--w:${Math.round(Math.max(0, Math.min(1, rel)) * 100)}%"></span>` : ''}</span>
        ${text ? `<span class="fd-row-text">${text}</span>` : ''}
    </div>`;
}

function fdThumb(tile, it, k) {
    return `<div class="fd-thumb${fdSel(tile, it)}" ${fdPickAttrs(tile, k, null, it)} title="${esc(fdName(it))} · ${fdClock(it.t)}">
        ${it.thumb ? `<img loading="lazy" src="${esc(it.thumb)}" alt="">` : '<span class="fd-noimg">🔊</span>'}
        <span class="fd-thumb-play" aria-hidden="true">▶</span>
        ${(it.dups || []).length ? `<span class="fd-thumb-dups" title="${plural(it.dups.length, 'suspected duplicate')} in other files">⧉ ${it.dups.length}</span>` : ''}
        <span class="fd-thumb-cap">${k + 1} · ${fdClock(it.t)}</span>
    </div>`;
}

const FD_EVIDENCE = { words: '💬', meaning: '🧠', seen: '🖼', heard: '🔊', files: '📁', people: '👤' };

function fdVerdict(m) {
    if (m.verdict == null) return '';
    const v = Number(m.verdict);
    const cls = v >= 6 ? 'yes' : v >= FD_REJECT_BELOW ? 'maybe' : 'no';
    const said = m.verdict_said != null ? `The language model said ${Number(m.verdict_said).toFixed(1)} of 9 — capped at ${v.toFixed(0)}`
        : `The language model's verdict: ${v.toFixed(1)} of 9`;
    const failed = (m.failed || []).length ? `\nDoesn't fit what you said: ${m.failed.join('; ')}` : '';
    return `<span class="fd-verdict ${cls}${failed ? ' capped' : ''}" title="${esc(said + failed)}">${Math.round(v)}</span>`;
}

/* One line per recording. The searches can find several moments of one
 * recording (three at most, copies counted in) — they aren't duplicates,
 * but in a list they look like them: the best one stands, the others are
 * "+2 more moments" under it. `[{m, k, more: [{m, k}]}]`, k the index in
 * the run's list. */
function fdBestGroups(items) {
    const first = new Map(), out = [];
    items.forEach((m, k) => {
        const rec = m.rec || m.key;
        const g = first.get(rec);
        if (g) { g.more.push({ m, k }); return; }
        const fresh = { m, k, more: [] };
        first.set(rec, fresh);
        out.push(fresh);
    });
    return out;
}

function fdToggleMore(k) {
    const it = (fdTileState('best').items || [])[k];
    if (!it) return;
    const id = fdId(it);
    if (_fd.more.has(id)) _fd.more.delete(id); else _fd.more.add(id);
    _fd.sig.best = null;
    fdRenderTiles();
}

function fdBestRow(m, k, n = k, more = []) {
    const s = _fd.run || {};
    const checking = s.status === 'judging' && m.verdict == null;
    const ev = Object.keys(m.engines || {}).map(e => `<span class="fd-ev" title="${esc(fdTile(e)?.title || e)}">${FD_EVIDENCE[e] || ''}</span>`).join('')
        + (m.boosts || []).map(b => `<span class="fd-ev boost" title="${b === 'files' ? 'the file name matches' : 'the person speaks in it'}">${b === 'files' ? '📁' : '👤'}</span>`).join('');
    const said = (m.engines?.words || m.engines?.meaning || m.engines?.heard || {}).text || '';
    const open = _fd.open.has(fdId(m)) && (m.dups || []).length;
    const openMore = _fd.more.has(fdId(m)) && more.length;
    const moreBadge = more.length ? `<button class="fd-more-badge${openMore ? ' open' : ''}"
        onclick="event.stopPropagation(); fdToggleMore(${k})" aria-expanded="${openMore ? 'true' : 'false'}"
        title="This recording has ${more.length} more moment${more.length === 1 ? '' : 's'} in the list (click to show)">+${more.length} more moment${more.length === 1 ? '' : 's'}</button>` : '';
    const reject = m.verdict != null && m.verdict < FD_REJECT_BELOW;
    return `<div class="fd-best${fdSel('best', m)}${reject ? ' dim' : ''}${open || openMore ? ' unfolded' : ''}" ${fdPickAttrs('best', k, null, m)}>
        <span class="fd-best-rank">${n + 1}</span>
        <span class="fd-best-pic">${m.thumb ? `<img loading="lazy" src="${esc(m.thumb)}" alt="">` : `<span>${m.video ? '🎞' : '🎙'}</span>`}<span class="fd-best-play" aria-hidden="true">▶</span></span>
        <span class="fd-best-main">
            <span class="fd-best-name">${esc(m.name)} <span class="fd-best-at">${fdClock(m.t)}</span></span>
            <span class="fd-best-why">${ev}${fdDupBadge(m, 'best', k)}${moreBadge}</span>
            ${m.reason ? `<span class="fd-best-reason">${esc(m.reason)}</span>` : said ? `<span class="fd-best-said">${esc(said.replace(/<[^>]+>/g, '').slice(0, 160))}</span>` : ''}
        </span>
        ${checking && k < 16 ? '<span class="fd-verdict wait" title="being checked">…</span>' : fdVerdict(m)}
    </div>${open ? fdDupRows(m, k) : ''}${openMore ? fdMoreRows(more) : ''}`;
}

// The folded duplicates of a best match, below it
function fdDupRows(m, k) {
    return `<div class="fd-dups" role="group" aria-label="Suspected duplicates">${m.dups.map(d => `
        <div class="fd-dup${fdSel('best', m, d.key)}" ${fdPickAttrs('best', k, d.key, m)} title="${esc(d.key)}">
            <span class="fd-dup-pic">${d.thumb ? `<img loading="lazy" src="${esc(d.thumb)}" alt="">` : (d.video ? '🎞' : '🎙')}</span>
            <span class="fd-dup-main">
                <span class="fd-dup-name">${esc(d.name)} <span class="fd-best-at">${fdClock(d.t)}</span></span>
                <span class="fd-dup-why">${esc(fdFolder(d))} · ${esc(d.why)}</span>
            </span>
        </div>`).join('')}</div>`;
}

// The recording's other moments, below its best one
function fdMoreRows(more) {
    const s = _fd.run || {};
    return `<div class="fd-dups fd-more" role="group" aria-label="More moments of this recording">${more.map(({ m, k }) => {
        const checking = s.status === 'judging' && m.verdict == null;
        return `
        <div class="fd-dup${fdSel('best', m)}" ${fdPickAttrs('best', k, null, m, 'head')} title="${esc(m.key)}">
            <span class="fd-dup-pic">${m.thumb ? `<img loading="lazy" src="${esc(m.thumb)}" alt="">` : (m.video ? '🎞' : '🎙')}</span>
            <span class="fd-dup-main">
                <span class="fd-dup-name">${fdClock(m.t)} <span class="fd-best-at">${esc(m.name)}</span></span>
                <span class="fd-dup-why">${esc(m.reason || Object.keys(m.engines || {}).map(e => fdTile(e)?.title || e).join(' · '))}</span>
            </span>
            ${checking ? '<span class="fd-verdict wait" title="being checked">…</span>' : fdVerdict(m)}
        </div>`;
    }).join('')}</div>`;
}

/* The words a hit was found by, marked. Matched on the plain text and
 * escaped piece by piece — matching the escaped text missed "that's"
 * (its ' had become &#39;) and could break an entity. The phrase may be a
 * list of stems ("best* | day"). */
function fdMark(text, phrase) {
    const raw = String(text || '').replace(/<[^>]+>/g, '');
    const words = String(phrase || '').toLowerCase().split(/[\s|·]+/)
        .map(w => w.replace(/\*+$/, '')).filter(w => w.length >= 3);
    if (!words.length) return esc(raw);
    const re = new RegExp('(' + [...new Set(words)].map(w => w.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('|') + ')', 'gi');
    let out = '', last = 0, m;
    while ((m = re.exec(raw)) !== null) {
        out += esc(raw.slice(last, m.index)) + '<mark>' + esc(m[0]) + '</mark>';
        last = m.index + m[0].length;
        if (!m[0].length) re.lastIndex++;
    }
    return out + esc(raw.slice(last));
}

function fdRenderTiles() {
    const grid = document.getElementById('fdGrid');
    if (!grid) return;
    grid.classList.toggle('maxed', !!_fd.max);
    const ids = fdShownTiles();
    const shown = ids.join(',');
    if (_fd.shown !== shown || !grid.children.length) {
        _fd.shown = shown;
        _fd.sig = {};
        fdLayout(ids);
        const tiles = FD_TILES.filter(t => t.id === 'best' || ids.includes(t.id));
        grid.innerHTML = tiles.map(t => fdTileHtml(t, fdTileState(t.id))).join('');
    }
    for (const t of FD_TILES) {
        if (t.id !== 'best' && !ids.includes(t.id)) continue;
        const st = fdTileState(t.id);
        const sig = JSON.stringify([st.status, st.count, (st.items || []).length, st.note, st.folded, _fd.max === t.id,
            fdTileTitle(t), (st.items || []).slice(0, 3).map(m => m.key),
            t.id === 'best' ? [_fd.run?.judged, _fd.run?.status, [..._fd.open], [..._fd.more],
                (st.items || []).slice(0, 40).map(m => [m.key, m.t, m.verdict, (m.dups || []).length])] : 0,
            _fd.pick && _fd.pick.tile === t.id ? [_fd.pick.id, _fd.pick.dup] : -1]);
        if (_fd.sig[t.id] === sig) continue;
        _fd.sig[t.id] = sig;
        const old = grid.querySelector(`[data-tile="${t.id}"]`);
        const body = old?.querySelector('.fd-tile-body');
        const scroll = body ? body.scrollTop : 0;
        const focused = old && old.contains(document.activeElement) ? document.activeElement.dataset.fid : null;
        const tmp = document.createElement('div');
        tmp.innerHTML = fdTileHtml(t, st);
        const fresh = tmp.firstElementChild;
        if (old) old.replaceWith(fresh); else grid.appendChild(fresh);
        const nb = fresh.querySelector('.fd-tile-body');
        if (nb) nb.scrollTop = scroll;
        // the keyboard stays where it was
        if (focused) [...fresh.querySelectorAll('[data-pick]')].find(el => el.dataset.fid === focused)?.focus();
    }
}

// The keyboard in the lists: Enter/Space pick, ↑ ↓ move
document.addEventListener('keydown', (e) => {
    const el = e.target;
    if (!el || !el.matches || !el.matches('#svPaneFind [data-pick]')) return;
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); el.click(); return; }
    if (e.key !== 'ArrowDown' && e.key !== 'ArrowUp') return;
    const list = [...(el.closest('.fd-tile-body')?.querySelectorAll('[data-pick]') || [])];
    const next = list[list.indexOf(el) + (e.key === 'ArrowDown' ? 1 : -1)];
    if (next) { e.preventDefault(); next.focus(); next.click(); }
});

// ── Maximize ────────────────────────────────────────────────────────────────

function fdMax(id) {
    _fd.max = _fd.max === id ? null : id;
    _fd.sig = {};
    fdRenderTiles();
}

document.addEventListener('keydown', (e) => {
    if (e.key !== 'Escape' || !_fd.max) return;
    const pane = document.getElementById('svPaneFind');
    if (!pane || !pane.classList.contains('active')) return;
    e.stopImmediatePropagation();
    e.preventDefault();
    fdMax(_fd.max);
}, true);

// The same search in its own tab
function fdOpenTab(id) {
    const p = (_fd.run || {}).plan || {};
    const q = document.getElementById('svQuery');
    const tile = fdTile(id);
    if (!tile || !tile.tab || typeof svTab !== 'function') return;
    if (id === 'words') q.value = (p.said_words || [])[0] || '';
    else if (id === 'meaning') q.value = p.said_meaning || _fd.lastQuery;
    else if (id === 'seen') q.value = p.seen || '';
    else if (id === 'heard') q.value = p.heard || '';
    // the files tab searches names: the remembered piece is its query
    else if (id === 'files') q.value = (p.file_hints || [])[0] || '';
    else if (id === 'people') {
        q.value = p.said_meaning || '';
        const sel = document.getElementById('svPerson');
        const want = ((p.people || [])[0] || '').toLowerCase();
        const opt = sel && [...sel.options].find(o => o.value && o.value.toLowerCase() === want);
        if (opt) sel.value = opt.value;
    }
    if (id === 'meaning' && typeof svSetMode === 'function' && svState.mode === 'text') {
        svState.mode = 'both';
        if (typeof svModeSync === 'function') svModeSync();    // the control shows it
    }
    fdLeave();
    // a tab switch runs its search itself; on the same tab it runs here
    if (svState.tab === tile.tab) svRun();
    else svTab(tile.tab);
}

// ── A hit, on the right ─────────────────────────────────────────────────────

/* The picked hit: looked up by what it is in the tile's current list (the
 * newest verdicts and duplicates), else the one picked — a re-fused list may
 * have folded it into another hit by now. `{it, parent}`: a duplicate comes
 * with the hit it was folded into. */
function fdPicked() {
    const p = _fd.pick;
    if (!p) return null;
    const items = fdTileState(p.tile).items || [];
    let it = items.find(x => fdId(x) === p.id);
    if (it) p.item = it; else it = p.item;
    if (!it) return null;
    if (p.dup) {
        const d = (it.dups || []).find(x => x.key === p.dup);
        if (d) return { it: d, parent: it };
    }
    return { it, parent: null };
}

function fdPick(tile, k, dup = null) {
    const it = (fdTileState(tile).items || [])[k];
    if (!it) return;
    _fd.pick = { tile, id: fdId(it), dup: dup || null, item: it };
    if (tile === 'best') {
        // a moment under its recording's best one: that one shows its others
        const g = fdBestGroups(fdTileState('best').items || []).find(x => x.more.some(o => o.k === k));
        if (g) _fd.more.add(fdId(g.m));
    }
    document.getElementById('fdMain')?.classList.add('previewing');
    fdRenderTiles();
    fdRenderPreview();
}

function fdPickDup(key) {
    const p = _fd.pick;
    if (!p) return;
    _fd.pick = { ...p, dup: key || null };
    if (p.tile === 'best' && key) _fd.open.add(p.id);
    _fd.sig[p.tile] = null;
    fdRenderTiles();
    fdRenderPreview();
}

function fdClosePreview() {
    document.querySelectorAll('#fdPreview video, #fdPreview audio').forEach(m => { try { m.pause(); } catch (e) { /* gone */ } });
    _fd.pick = null;
    document.getElementById('fdMain')?.classList.remove('previewing');
    const pv = document.getElementById('fdPreview');
    pv.innerHTML = '';
    delete pv.dataset.media;
    delete pv.dataset.info;
    fdRenderTiles();
}

function fdPlace(it) {
    // what the player and the transcript open: relative name + root, else the path split
    if (it.rel && it.dir) return { file: it.rel, dir: it.dir };
    const key = String(it.key || '');
    const i = key.lastIndexOf('/');
    return { file: key.slice(i + 1), dir: key.slice(0, i) };
}

function fdIsVideo(it) {
    if (typeof it.video === 'boolean') return it.video;
    return FD_VIDEO_EXT.test(String(it.key || ''));
}

// The clip of a moment (api/find.py → find/agent.cut_clip) and where it starts
function fdClipUrl(it) {
    const t = Math.max(0, Number(it.t) || 0);
    let u = `/api/find/clip?t=${t.toFixed(1)}&key=${encodeURIComponent(it.key)}`;
    const end = Number(it.end);
    if (it.end != null && Number.isFinite(end) && end > t) u += `&end=${end.toFixed(1)}`;
    return u;
}

function fdClipFrom(it) { return Math.max(0, (Number(it.t) || 0) - FD_CLIP_BEFORE); }

function fdRenderPreview() {
    const pv = document.getElementById('fdPreview');
    const got = fdPicked();
    if (!pv || !got) { if (pv) pv.innerHTML = ''; return; }
    const { it, parent } = got;
    // The player only when the hit changes — a verdict arriving must not
    // start the clip being watched over again
    const mediaKey = `${_fd.pick.tile}|${it.key}|${it.t}`;
    if (pv.dataset.media !== mediaKey) {
        pv.dataset.media = mediaKey;
        delete pv.dataset.info;
        pv.innerHTML = `
            <div class="fd-pv-head"><b id="fdPvName"></b><button class="vx-icon" onclick="fdClosePreview()" title="Close">✕</button></div>
            <div class="fd-pv-media" id="fdPvMedia"></div>
            <div class="fd-pv-clock" id="fdPvClock"></div>
            <div id="fdPvInfo"></div>`;
        fdRenderPlayer(it);
    }
    const best = _fd.pick.tile === 'best';
    const head = parent || it;
    const infoKey = JSON.stringify([mediaKey, head.verdict, head.reason, (head.dups || []).length, _fd.pick.dup, head.failed]);
    if (pv.dataset.info === infoKey) return;
    pv.dataset.info = infoKey;
    document.getElementById('fdPvName').textContent = fdName(it);
    const ev = best && !parent ? Object.entries(it.engines || {}).map(([n, e]) => {
        // a hit found inside a scope: which one ("In files named “-WA”", "In what Alice says")
        const where = e.scope && fdTile(e.scope) ? ` · ${fdTileTitle(fdTile(e.scope))}` : '';
        return `<div class="fd-pv-ev"><b>${FD_EVIDENCE[n] || ''} ${esc(fdTileTitle(fdTile(n) || { id: n, title: n }))}</b> <span>#${e.rank + 1} · ${fdClock(e.t)}${e.copy ? ` · in the copy ${esc(e.copy)}` : ''}${esc(where)}</span>
            ${e.text ? `<div>${esc(String(e.text).replace(/<[^>]+>/g, ''))}</div>` : ''}</div>`;
    }).join('') : '';
    const text = !best && !parent && it.text ? `<div class="fd-pv-said">${_fd.pick.tile === 'words' ? fdMark(it.text, (it.extra || {}).phrase) : esc(it.text)}</div>` : '';
    const dupOf = parent ? `<div class="fd-pv-dupof">⧉ Suspected duplicate of <a href="#" onclick="event.preventDefault(); fdPickDup(null)">${esc(fdName(parent))}</a> — ${esc(it.why || '')}${
        it.length ? ` · ${fdClock(it.length)}` : ''}${it.size ? ` · ${esc(fdSize(it.size))}` : ''}</div>` : '';
    const failed = best && (head.failed || []).length
        ? `<div class="fd-pv-failed" title="Checked by the program — the verdict is capped">Doesn't fit what you said: ${esc(head.failed.join('; '))}${
            head.verdict_said != null ? ` — the language model said ${Number(head.verdict_said).toFixed(1)}, capped at ${Number(head.verdict).toFixed(0)}` : ''}</div>` : '';
    const dups = (head.dups || []).length ? `
        <div class="fd-pv-label">Suspected duplicates · ${head.dups.length}</div>
        <div class="fd-pv-dups-hint">The same moment in other files — found by what they hold (length, words, picture), not by their names.</div>
        ${[head, ...head.dups].map(d => {
            const sel = (d === head ? !parent : parent && d.key === it.key) ? ' sel' : '';
            return `<button class="fd-pv-dup${sel}" onclick="fdPickDup(${d === head ? 'null' : `'${escJs(d.key)}'`})" title="${esc(d.key)}">
                <span class="fd-pv-dup-name">${esc(fdName(d))} <span class="fd-best-at">${fdClock(d.t)}</span></span>
                <span class="fd-pv-dup-where">${esc(fdFolder(d))}</span>
                <span class="fd-pv-dup-why">${d === head ? 'shown in the list' : esc(d.why)}${d.length ? ` · ${fdClock(d.length)}` : ''}${d.size ? ` · ${esc(fdSize(d.size))}` : ''}</span>
            </button>`;
        }).join('')}` : '';
    document.getElementById('fdPvInfo').innerHTML = `
        <div class="fd-pv-meta">${it.t != null ? 'at ' + fdClock(it.t) : ''} <span title="${esc(it.key || '')}">${esc(fdFolder(it))}</span></div>
        ${dupOf}
        ${best && head.reason ? `<div class="fd-pv-verdict">${fdVerdict(head)} ${esc(head.reason)}</div>` : ''}
        ${failed}
        <div class="fd-pv-actions">
            <button class="vx-btn" onclick="fdGo(false)" title="The whole recording in the player of the main window — from where the clip is">▶ Open in the player</button>
            <button class="vx-btn" onclick="fdGo(true)" title="Open the transcript at this moment">📄 Transcript</button>
        </div>
        ${dups}
        ${ev ? `<div class="fd-pv-evs"><div class="fd-pv-label">Why it was found</div>${ev}</div>` : ''}
        ${text}`;
}

/* The moment, playable right here: a clip cut from the recording — from a
 * few seconds before it, H.264 or AAC whatever the file is (.mkv, .avi,
 * .opus …). Cutting takes a moment the first time (the best ones are cut
 * while the language model checks); the still stands in meanwhile. The
 * clock under it counts the RECORDING's time. */
function fdRenderPlayer(it) {
    const box = document.getElementById('fdPvMedia');
    if (!box) return;
    const video = fdIsVideo(it);
    const src = fdClipUrl(it);
    const poster = it.thumb ? ` poster="${esc(it.thumb)}"` : '';
    box.innerHTML = video
        ? `<video class="fd-pv-video" src="${esc(src)}"${poster} controls autoplay playsinline preload="auto"></video>`
        : `${it.thumb ? `<img class="fd-pv-img" src="${esc(it.thumb)}" alt="">` : ''}<audio class="fd-pv-audio" src="${esc(src)}" controls autoplay preload="auto"></audio>`;
    box.insertAdjacentHTML('beforeend', '<div class="fd-pv-wait"><span class="fd-dot live"></span> Cutting the clip …</div>');
    const el = box.querySelector('video, audio');
    const clock = document.getElementById('fdPvClock');
    const from = fdClipFrom(it);
    const tick = () => {
        if (!clock) return;
        const now = from + (el.currentTime || 0);
        const len = Number.isFinite(el.duration) ? el.duration : 0;
        clock.innerHTML = `<span class="fd-pv-now">${fdClock(now)}</span> in the recording`
            + (len ? ` · the clip ${fdClock(from)}–${fdClock(from + len)}` : '');
    };
    const ready = () => { box.querySelector('.fd-pv-wait')?.remove(); tick(); };
    el.addEventListener('loadeddata', ready);
    el.addEventListener('timeupdate', tick);
    // One sound at a time: the main player pauses while the clip plays
    el.addEventListener('play', () => { try { svPlayer()?.pause(); } catch (e) { /* none */ } });
    el.addEventListener('error', () => {
        box.querySelector('.fd-pv-wait')?.remove();
        box.insertAdjacentHTML('beforeend', '<div class="fd-pv-fail">This moment can\'t be played here — the recording may be unreadable. Open it in the player.</div>');
        if (clock) clock.textContent = '';
    });
    tick();
}

function fdGo(transcript) {
    const got = fdPicked();
    if (!got) return;
    const { it } = got;
    const { file, dir } = fdPlace(it);
    // From where the clip is — the moment, or as far as it was watched
    const el = document.querySelector('#fdPvMedia video, #fdPvMedia audio');
    const watched = el && el.currentTime > 0.5 ? fdClipFrom(it) + el.currentTime : null;
    document.querySelectorAll('#fdPreview video, #fdPreview audio').forEach(m => { try { m.pause(); } catch (e) { /* gone */ } });
    const t = watched != null ? watched : Math.max(0, (Number(it.t) || 0) - 2);
    if (transcript) svGoto(file, t, dir);
    else svPlay(file, t, dir);
}
