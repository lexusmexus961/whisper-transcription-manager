/* ══════════════════════════════════════════════════════════════════════════
 *  DEEP SEARCH — "find everything" for one situation (/api/rag/deep)
 *
 *  *"Alice gets a compliment"*, *"someone apologizes for being late"* — situations the
 *  meaning search shows a few of, but not all of them, and not without junk
 *  in between. A deep search goes through the whole library for one:
 *
 *    plan        the language model turns the question into the situation,
 *                what counts and what doesn't, and keywords;
 *    candidates  passages close in meaning or with the keywords — copies of
 *                a recording in other folders fold into one;
 *    ranking     the reranker orders every candidate by the criteria;
 *    checking    the language model reads the best ones ONE AT A TIME and
 *                answers yes, unclear or no.
 *
 *  Only what it confirmed is listed; "unclear" is kept apart. It runs in the
 *  LLM queue, on the server — the page only asks how far it is, and only
 *  while somebody looks: with the search closed, another tab shown or the
 *  browser tab hidden it asks nothing, and shown again it goes on.
 * ══════════════════════════════════════════════════════════════════════════ */

const DS = {
    list: [],              // the deep searches kept on the server, newest first
    listHtml: '',          // the list as drawn — redrawn only when it changes
    listError: '',
    listCtrl: null,        // the list's request on its way — one at a time
    listAgain: false,      // asked for again while one was on its way
    listSeq: 0,
    listTimer: null,
    cur: null,             // the one shown — null: the form for a new one
    hits: [],              // its confirmed passages, in the order they came
    maybe: [],             // the ones the model could not decide
    since: 0,              // results received — the next poll asks from here on
    seq: 0,                // bumps when the shown one changes or polling pauses
    timer: null,           // the next poll
    inflight: null,        // the poll on its way (AbortController) — one at a time
    fails: 0,              // polls failed in a row
    retry: 0,              // the wait before the next try (ms)
    open: new Set(),       // 'hit:3' — results whose copies are unfolded
    unclearOpen: false,    // the "unclear" section, opened by hand
    hot: '',               // the result played last
    armed: '',             // the id a first click on delete armed …
    armedUntil: 0,         // … until then
    starting: false,       // a start on its way — a second click waits for it
    planHtml: '',          // the plan column as drawn
};

const DS_FAST = 1500;            // asking while it runs
const DS_SLOW = 4000;            // while it waits in the LLM queue
const DS_BACKOFF_MAX = 30000;    // the longest wait after failures in a row
const DS_TIMEOUT = 20000;        // a request without an answer by then is given up
const DS_LIST_EVERY = 8000;      // the list, while another one runs
const DS_MIN_QUERY = 3;
const DS_ARMED_MS = 4000;        // how long a first click on delete stays armed

const DS_STATUS = {
    queued:  {text: 'waiting in the LLM queue', cls: 'wait'},
    running: {text: 'running', cls: 'run'},
    done:    {text: 'done', cls: 'ok'},
    stopped: {text: 'stopped', cls: 'off'},
    error:   {text: 'failed', cls: 'err'},
};
const DS_STAGES = [
    {key: 'plan', label: 'Plan'},
    {key: 'candidates', label: 'Candidates'},
    {key: 'ranking', label: 'Ranking'},
    {key: 'checking', label: 'Checking'},
];

function dsEl(id) { return document.getElementById(id); }
function dsActive(s) { return !!s && (s.status === 'queued' || s.status === 'running'); }
function dsN(x) { return (Number(x) || 0).toLocaleString(); }

/* Somebody looks: the search is open, on this tab, and the browser tab is
 * shown. Everything that asks the server asks this first. */
function dsWanted() {
    return !!dsEl('svOverlay')?.classList.contains('visible')
        && typeof svState !== 'undefined' && svState.tab === 'deep'
        && document.visibilityState !== 'hidden';
}

// ── Showing and leaving the tab ──────────────────────────────────────────

/* From elsewhere — the LLM queue's row: the search workspace on this tab,
 * with that deep search. */
function openDeepSearch(id) {
    const ov = dsEl('svOverlay');
    if (!ov) return;
    ov.classList.add('visible');
    if (svState.tab !== 'deep') svTab('deep');
    else dsOpen();
    if (id && !(DS.cur && DS.cur.id === id)) dsLoad(id);
}

/* The tab is shown (svTab, or the search opened again on it). */
function dsOpen() {
    const field = dsEl('dsQuery');
    if (field && !String(field.value || '').trim()) {
        field.value = String(dsEl('svQuery')?.value || '').trim();
    }
    if (DS.cur) dsRender(); else dsShowForm();
    dsResume();
}

/* The tab is left or the search closed: nothing more is asked — the deep
 * search itself runs on in the LLM queue. */
function dsLeave() {
    dsPark();
}

/* The shown one's poll ends: no timer, the request on its way abandoned,
 * and its answer — should it still come — dropped (`seq`). */
function dsStopPoll() {
    clearTimeout(DS.timer);
    DS.timer = null;
    if (DS.inflight) { DS.inflight.abort(); DS.inflight = null; }
    DS.seq++;
}

/* Nothing is asked any more — neither the shown one nor the list. */
function dsPark() {
    dsStopPoll();
    clearTimeout(DS.listTimer);
    DS.listTimer = null;
    if (DS.listCtrl) { DS.listCtrl.abort(); DS.listCtrl = null; }
    DS.listAgain = false;
    DS.listSeq++;
}

/* Shown again: ask right away, then on as before. */
function dsResume() {
    if (!dsWanted()) return;
    if (DS.cur && !DS.inflight && !DS.timer && (!DS.cur.loaded || dsActive(DS.cur))) dsPoll();
    dsLoadList();
}

// A hidden browser tab asks for nothing; shown again, it picks up.
document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'hidden') dsPark();
    else dsResume();
});

// ── Starting one ─────────────────────────────────────────────────────────

/* "🔎 Deep search" next to the question in the query bar. */
function dsStartFromQuery() {
    const q = String(dsEl('svQuery')?.value || '').trim();
    if (q.length < DS_MIN_QUERY) {
        Toast.info('Describe the situation first — e.g. "Alice gets a compliment"');
        dsEl('svQuery')?.focus();
        return;
    }
    const field = dsEl('dsQuery');
    if (field) field.value = q;
    if (svState.tab !== 'deep') svTab('deep');
    return dsStart(q);
}

/* The tab's own form. */
function dsFormStart() {
    const q = String(dsEl('dsQuery')?.value || '').trim();
    if (q.length < DS_MIN_QUERY) {
        Toast.info('Describe the situation first — e.g. "Alice gets a compliment"');
        dsEl('dsQuery')?.focus();
        return;
    }
    return dsStart(q);
}

function dsFormKey(e) {
    if (e.key === 'Enter' && !e.isComposing) { e.preventDefault(); dsFormStart(); }
}

/* The same question once more — a new run, the old one stays in the list. */
function dsAgain() {
    if (DS.cur && DS.cur.query) return dsStart(DS.cur.query);
}

function dsSameQuery(a, b) {
    const norm = s => String(s || '').trim().toLowerCase().replace(/\s+/g, ' ');
    return norm(a) === norm(b);
}

async function dsStart(query) {
    if (DS.starting) return;
    // Being searched already: show that one instead of a second run of it
    const same = DS.list.find(s => dsActive(s) && dsSameQuery(s.query, query))
        || (dsActive(DS.cur) && dsSameQuery(DS.cur.query, query) ? DS.cur : null);
    if (same) {
        Toast.info('This is being searched already — here it is');
        if (!(DS.cur && DS.cur.id === same.id)) dsLoad(same.id);
        return;
    }
    DS.starting = true;
    dsSyncStart();
    let d = null;
    try {
        const r = await fetch('/api/rag/deep', {method: 'POST',
            headers: {'Content-Type': 'application/json'}, body: JSON.stringify({query})});
        d = await r.json().catch(() => ({}));
        if (!r.ok || d.error || !d.id) throw new Error(d.error || `HTTP ${r.status}`);
    } catch (e) {
        Toast.error('Deep search not started: ' + (e.message || e));
        d = null;
    }
    DS.starting = false;
    dsSyncStart();
    if (!d) return;
    Toast.success('🔎 The deep search is in the LLM queue — it runs on the server, you can close this');
    dsLoad(d.id, {query, status: d.status || 'queued'});
    dsLoadList();
}

function dsSyncStart() {
    for (const id of ['dsStartBtn', 'svDeepBtn']) {
        const b = dsEl(id);
        if (b) b.disabled = DS.starting;
    }
}

// ── One deep search ──────────────────────────────────────────────────────

/* Show one — what the list knows of it at once, the rest from the server. */
function dsLoad(id, known) {
    if (!id) return;
    dsStopPoll();
    const listed = DS.list.find(s => s.id === id) || {};
    DS.cur = Object.assign({}, listed, known || {}, {id, loaded: false});
    DS.hits = [];
    DS.maybe = [];
    DS.since = 0;
    DS.fails = 0;
    DS.retry = 0;
    DS.open = new Set();
    DS.hot = '';
    DS.unclearOpen = false;
    dsRender();
    dsRenderList();
    dsPoll();
}

function dsNew() {
    dsStopPoll();
    DS.cur = null;
    DS.hits = [];
    DS.maybe = [];
    dsShowForm();
    dsRenderList();
    dsEl('dsQuery')?.focus();
}

/* Ask how far the shown one is. One request at a time, from the results
 * received so far on (`since`); an answer for a search no longer shown is
 * dropped. While it runs every 1.5 s, while it waits in the queue every
 * 4 s, after failures longer and longer (2, 4, 8 … 30 s) — and not at all
 * while nobody looks. Each answer schedules the next. */
async function dsPoll() {
    clearTimeout(DS.timer);
    DS.timer = null;
    const cur = DS.cur;
    if (!cur || DS.inflight) return;
    if (!dsWanted()) return;            // shown again, dsResume asks
    const seq = DS.seq;
    const ctrl = DS.inflight = new AbortController();
    const kill = setTimeout(() => ctrl.abort(), DS_TIMEOUT);
    let d = null, error = null, gone = false;
    try {
        const r = await fetch(`/api/rag/deep/${encodeURIComponent(cur.id)}?since=${DS.since}`,
                              {signal: ctrl.signal});
        if (r.status === 404) {
            gone = true;
        } else {
            d = await r.json();
            if (!r.ok || !d || !d.status) throw new Error((d && d.error) || `HTTP ${r.status}`);
        }
    } catch (e) {
        error = e;
    }
    clearTimeout(kill);
    if (DS.inflight === ctrl) DS.inflight = null;
    if (seq !== DS.seq || DS.cur !== cur) return;   // another one shown, or paused
    if (gone) { dsGone(cur); return; }
    if (error) {
        DS.fails++;
        DS.retry = Math.min(DS_BACKOFF_MAX, 1000 * 2 ** DS.fails);
        dsRenderProgress();
        DS.timer = setTimeout(dsPoll, DS.retry);
        return;
    }
    DS.fails = 0;
    DS.retry = 0;
    if (dsApply(d)) DS.timer = setTimeout(dsPoll, 0);     // started over
    else if (dsActive(cur)) DS.timer = setTimeout(dsPoll, cur.status === 'running' ? DS_FAST : DS_SLOW);
}

/* An answer onto what is shown: the new results appended, the rest
 * replaced. Returns true when the server's results are not the ones held
 * (fewer than received — the search was rebuilt): then they are read anew. */
function dsApply(d) {
    const cur = DS.cur;
    const was = {status: cur.status, loaded: cur.loaded};
    if (d.next != null && Number(d.next) < DS.since) {
        DS.hits = [];
        DS.maybe = [];
        DS.since = 0;
        DS.open = new Set();
        Object.assign(cur, dsMeta(d), {loaded: false});
        dsRender();
        return true;
    }
    const before = {hit: DS.hits.length, maybe: DS.maybe.length};
    const fresh = Array.isArray(d.results) ? d.results : [];
    for (const x of fresh) {
        const r = dsResult(x);
        if (r.verdict === 'hit') DS.hits.push(r);
        else if (r.verdict === 'maybe') DS.maybe.push(r);
    }
    const next = Number(d.next);
    DS.since = d.next != null && Number.isFinite(next) ? next : DS.since + fresh.length;
    Object.assign(cur, dsMeta(d), {loaded: true});
    dsListPatch(cur);
    if (!was.loaded || was.status !== cur.status) {
        dsRender();
    } else {
        dsRenderProgress();
        dsAppend(before);
        dsRenderPlan();
    }
    dsRenderList();
    // Over while somebody watched: say so, and the list takes it up
    if (was.loaded && dsActive(was) && !dsActive(cur)) {
        const n = DS.hits.length;
        if (cur.status === 'done') Toast.success(`🔎 Deep search done: ${dsN(n)} hit${n === 1 ? '' : 's'}`);
        else if (cur.status === 'error') Toast.error('Deep search failed: ' + (cur.error || 'unknown error'));
        dsLoadList();
    }
    return false;
}

// What the shown one keeps of an answer — everything but the results.
function dsMeta(d) {
    const out = {};
    for (const k of ['query', 'status', 'stage', 'message', 'plan', 'progress', 'counts',
                     'created', 'started', 'finished', 'elapsed_sec', 'reranked', 'note',
                     'error', 'total_results']) {
        if (k in d) out[k] = d[k];
    }
    return out;
}

/* A result as the page holds it. Without `rel_path` its transcript is no
 * longer in the library: shown, but it can't be opened or played. */
function dsResult(x) {
    const raw = x.relevance ?? x.score;
    const score = Number(raw);
    return {
        verdict: x.verdict === 'hit' || x.verdict === 'maybe' ? x.verdict : '',
        filename: x.rel_path || '',
        label: x.rel_path || x.media_path || x.transcript_key || '',
        file_dir: x.file_dir || '',
        start: Number(x.start_sec) || 0,
        text: String(x.snippet ?? x.content ?? ''),
        // The name the voice library knows for the label, else the label
        speaker: x.speaker_name || x.speaker || '',
        score: raw != null && Number.isFinite(score) ? score : null,
        // Without a language model the best-ranked passages come back
        // unread — "unclear", and marked so
        unchecked: !!x.unchecked,
        copies: typeof svCopyList === 'function' ? svCopyList(x) : [],
    };
}

// The search is gone on the server (deleted elsewhere).
function dsGone(cur) {
    Toast.info('This deep search no longer exists');
    DS.list = DS.list.filter(s => s.id !== cur.id);
    DS.cur = null;
    DS.hits = [];
    DS.maybe = [];
    dsShowForm();
    dsRenderList();
}

// ── Drawing it ───────────────────────────────────────────────────────────

function dsShowForm() {
    const view = dsEl('dsView'), form = dsEl('dsForm');
    if (view) view.hidden = true;
    if (form) form.hidden = false;
    dsRenderPlan();
    dsSyncStart();
}

function dsRender() {
    const c = DS.cur;
    const view = dsEl('dsView'), form = dsEl('dsForm');
    if (!c || !view) { dsShowForm(); return; }
    if (form) form.hidden = true;
    view.hidden = false;
    const st = DS_STATUS[c.status] || {text: c.status || '', cls: ''};
    const active = dsActive(c);
    const armed = DS.armed === c.id && Date.now() < DS.armedUntil;
    view.innerHTML = `<div class="an-head ds-head">
            <h3 title="${esc(c.query || '')}">${c.query ? '“' + esc(c.query) + '”' : 'Deep search'}</h3>
            ${c.loaded && st.text ? `<span class="an-status ${st.cls}">${esc(st.text)}</span>` : ''}
            <span class="vx-gap"></span>
            ${active ? `<button class="vx-btn danger" id="dsStopBtn" onclick="dsStop()"
                         title="Stop searching — what was found so far is kept">Stop</button>` : ''}
            ${c.loaded && !active ? `<button class="vx-btn" onclick="dsAgain()"
                         title="A new deep search with the same question">↻ Search again</button>` : ''}
            <button class="vx-btn danger${armed ? ' armed' : ''}" id="dsDelBtn" data-label="Delete"
                    onclick="dsDeleteClick(this, '${escJs(c.id)}')"
                    title="Delete this deep search${active ? ' — it is stopped first' : ''}">${armed ? 'Delete?' : 'Delete'}</button>
        </div>
        <div class="ds-progress" id="dsProgress">${dsProgressHtml(c)}</div>
        <div class="ds-section">
            <div class="ds-section-title">Confirmed · <b id="dsNHits">${dsN(DS.hits.length)}</b>
                <span class="an-muted">— checked one by one: the situation happens here</span></div>
            <div class="ds-hits" id="dsHits">${dsRowsHtml('hit')}</div>
        </div>
        <details class="ds-unclear${DS.maybe.length ? '' : ' empty'}" id="dsUnclear"${DS.unclearOpen ? ' open' : ''}
                 ontoggle="DS.unclearOpen = this.open">
            <summary>Unclear · <b id="dsNMaybe">${dsN(DS.maybe.length)}</b>
                <span class="an-muted">— the model could not decide</span></summary>
            <div class="ds-hits" id="dsMaybe">${dsRowsHtml('maybe')}</div>
        </details>`;
    dsRenderPlan();
}

/* The steps (plan → candidates → ranking n/N → checking n/N), a bar for
 * the one under way, and the line with what was found and how long it took. */
function dsProgressHtml(c) {
    const retry = DS.fails
        ? `<span class="an-warn"> · no answer from the server — asking again in ${Math.round(DS.retry / 1000)} s</span>`
        : '';
    if (!c.loaded) return `<div class="ds-line an-muted">Loading …${retry}</div>`;
    const p = c.progress || {};
    const active = dsActive(c);
    const at = c.stage === 'done' ? DS_STAGES.length : DS_STAGES.findIndex(s => s.key === c.stage);
    const steps = DS_STAGES.map((s, k) => {
        let cls = '';
        if (c.stage === 'done' || (at >= 0 && k < at)) cls = 'done';
        else if (k === at && c.status !== 'queued') cls = active ? 'run' : (c.status === 'error' ? 'err' : 'off');
        let val = '', tip = '';
        // A new search counts 0 of everything: "Candidates 0" only once the
        // step is behind it — then it is a finding, before it is noise
        const counted = c.stage === 'done' || at > 1 || Number(p.groups) || Number(p.candidates);
        if (s.key === 'candidates' && counted && (p.groups != null || p.candidates != null)) {
            val = dsN(p.groups ?? p.candidates);
            if (p.groups != null && p.candidates != null) {
                tip = `${dsN(p.candidates)} passage${p.candidates === 1 ? '' : 's'} — ${dsN(p.groups)} once copies are folded together`;
            }
        } else if (s.key === 'ranking' && p.rank_total) {
            val = `${dsN(p.ranked)} / ${dsN(p.rank_total)}`;
        } else if (s.key === 'checking' && p.verify_total) {
            val = `${dsN(p.verified)} / ${dsN(p.verify_total)}`;
        }
        return `<li${cls ? ` class="${cls}"` : ''}${tip ? ` title="${esc(tip)}"` : ''}>${s.label}${val ? ` <b>${val}</b>` : ''}</li>`;
    }).join('');
    const share = c.stage === 'ranking' && p.rank_total ? (Number(p.ranked) || 0) / p.rank_total
        : c.stage === 'checking' && p.verify_total ? (Number(p.verified) || 0) / p.verify_total : null;
    const bar = active && share != null
        ? `<div class="an-bar ds-bar"><div style="width:${Math.max(2, Math.min(100, share * 100)).toFixed(1)}%"></div></div>`
        : '';
    const counts = c.counts || {};
    const hits = Number(counts.hits ?? DS.hits.length) || 0;
    const maybe = Number(counts.maybe ?? DS.maybe.length) || 0;
    const sec = dsElapsed(c);
    const facts = c.status === 'queued'
        ? ['Waiting in the LLM queue — it starts when the tasks before it are done']
        : [`<b>${dsN(hits)}</b> hit${hits === 1 ? '' : 's'}${active ? ' so far' : ''}`,
           maybe ? `${dsN(maybe)} unclear` : '',
           sec != null ? (active ? `${dsDuration(sec)} running` : `took ${dsDuration(sec)}`) : ''];
    return `<ol class="ds-steps"${c.message ? ` title="${esc(c.message)}"` : ''}>${steps}</ol>${bar}`
        + `<div class="ds-line">${facts.filter(Boolean).join(' · ')}${retry}</div>`
        + (c.note ? `<div class="ds-note">${esc(c.note)}</div>` : '')
        + (c.error ? `<div class="sv-msg-error">${esc(c.error)}</div>` : '');
}

function dsElapsed(c) {
    const e = Number(c.elapsed_sec);
    if (c.elapsed_sec != null && Number.isFinite(e)) return e;
    const s = Number(c.started);
    if (!c.started || !Number.isFinite(s)) return null;
    const f = Number(c.finished);
    return Math.max(0, (c.finished && Number.isFinite(f) ? f : Date.now() / 1000) - s);
}

function dsDuration(sec) {
    sec = Math.max(0, Math.round(Number(sec) || 0));
    if (sec < 60) return `${sec} s`;
    const m = Math.floor(sec / 60), s = sec % 60;
    if (m < 60) return s ? `${m} min ${s} s` : `${m} min`;
    return `${Math.floor(m / 60)} h ${m % 60} min`;
}

function dsRenderProgress() {
    const el = dsEl('dsProgress');
    if (el && DS.cur) el.innerHTML = dsProgressHtml(DS.cur);
}

function dsRowsHtml(kind) {
    const list = kind === 'maybe' ? DS.maybe : DS.hits;
    if (list.length) return list.map((r, i) => dsRowHtml(kind, i)).join('');
    if (kind === 'maybe') return '';
    const c = DS.cur;
    const text = !c || !c.loaded ? ''
        : dsActive(c) ? (c.stage === 'checking'
            ? 'None confirmed yet — they appear here as the model confirms them.'
            : 'Confirmed passages appear here once the checking starts.')
        : 'Nothing was confirmed.';
    return text ? `<div class="ds-none">${text}</div>` : '';
}

/* New results at the end of their lists — the rows already there stay as
 * they are (scroll position, unfolded copies). */
function dsAppend(before) {
    for (const kind of ['hit', 'maybe']) {
        const list = kind === 'maybe' ? DS.maybe : DS.hits;
        const box = dsEl(kind === 'maybe' ? 'dsMaybe' : 'dsHits');
        if (!box) continue;
        const from = before[kind];
        if (!list.length) {
            box.innerHTML = dsRowsHtml(kind);      // the waiting line follows the stage
            continue;
        }
        if (from === 0) box.innerHTML = '';
        if (list.length > from) {
            box.insertAdjacentHTML('beforeend',
                list.slice(from).map((r, k) => dsRowHtml(kind, from + k)).join(''));
        }
    }
    const nh = dsEl('dsNHits');
    if (nh) nh.textContent = dsN(DS.hits.length);
    const nm = dsEl('dsNMaybe');
    if (nm) nm.textContent = dsN(DS.maybe.length);
    dsEl('dsUnclear')?.classList.toggle('empty', !DS.maybe.length);
}

/* One result: a click plays the moment (the search stays open), a
 * double-click or ↗ opens the transcript there — as Ask's and the analyses'
 * passages do. A single click does not leave the search. */
function dsRowHtml(kind, i) {
    const r = (kind === 'maybe' ? DS.maybe : DS.hits)[i];
    if (!r) return '';
    const key = `${kind}:${i}`;
    const open = DS.open.has(key);
    const n = r.copies.length;
    const live = !!r.filename;
    const when = svTime(r.start);
    const score = r.score != null && DS.cur && DS.cur.reranked !== false
        ? `<span class="ds-hit-score" title="Reranker score — the order it was checked in">${esc(svPct(r.score))}</span>`
        : '';
    return `<div class="ds-hit${live ? '' : ' dead'}${DS.hot === key ? ' hot' : ''}" id="dsR-${kind}-${i}"
                 ${live ? `role="button" tabindex="0" data-keyclick onclick="dsPlay('${kind}', ${i})" ondblclick="dsGoto('${kind}', ${i})"` : ''}
                 title="${esc(r.label)}${live ? ' — click plays from ' + esc(when) + ', double-click opens the transcript' : ' — no longer in the library'}">
        <div class="ds-hit-head">
            ${live ? `<button class="ds-play" onclick="event.stopPropagation();dsPlay('${kind}', ${i})"
                              title="Play this moment — the search stays open">▶</button>` : ''}
            <span class="ds-hit-time">${esc(when)}</span>
            <span class="ds-hit-file">${esc(svShortName(r.label))}</span>
            ${r.speaker ? `<span class="ds-hit-sp">${esc(r.speaker)}</span>` : ''}
            ${r.unchecked ? `<span class="ds-hit-unchecked"
                               title="No language model was reachable — ranked by the reranker, not read">not checked</span>` : ''}
            <span class="vx-gap"></span>
            ${n ? `<button class="sv-copies-btn${open ? ' open' : ''}" onclick="event.stopPropagation();dsCopies('${kind}', ${i})"
                           title="The same passage in ${n} other ${n === 1 ? 'place' : 'places'} — copies, backups">+${n} ${n === 1 ? 'copy' : 'copies'}</button>` : ''}
            ${score}
            ${live ? `<button class="ds-open" onclick="event.stopPropagation();dsGoto('${kind}', ${i})"
                              title="Open the transcript at ${esc(when)}">↗</button>` : ''}
        </div>
        <div class="ds-hit-text">${esc(r.text)}</div>
        ${n && open ? dsCopiesHtml(kind, i, r) : ''}
    </div>`;
}

function dsCopiesHtml(kind, i, r) {
    return '<div class="sv-copies">' + r.copies.map((c, j) => c.dead
        ? `<div class="sv-copy dead" title="${esc(c.label)} — no longer in the library">
              <span class="sv-copy-mark">↳</span><span class="sv-z-time">${esc(c.start_fmt)}</span>
              <span class="sv-copy-path"><bdi>${esc(c.label)}</bdi></span>
              <span class="sv-copy-gone">not in the library</span></div>`
        : `<div class="sv-copy" onclick="event.stopPropagation();dsGoto('${kind}', ${i}, ${j})"
                title="${esc(c.filename)} — click opens this copy's transcript at ${esc(c.start_fmt)}">
              <span class="sv-copy-mark">↳</span><span class="sv-z-time">${esc(c.start_fmt)}</span>
              <span class="sv-copy-path"><bdi>${esc(c.filename)}</bdi></span>
              <button class="sv-copy-act" onclick="event.stopPropagation();dsPlay('${kind}', ${i}, ${j})"
                      title="Play this copy from ${esc(c.start_fmt)}">▶</button>
           </div>`).join('') + '</div>';
}

// "+3 copies" on a result: fold them out or in.
function dsCopies(kind, i) {
    const key = `${kind}:${i}`;
    if (DS.open.has(key)) DS.open.delete(key); else DS.open.add(key);
    const box = dsEl(kind === 'maybe' ? 'dsMaybe' : 'dsHits');
    if (box) box.innerHTML = dsRowsHtml(kind);
}

// The place a click means: the result itself, or copy `j` of it.
function dsTarget(kind, i, j) {
    const r = (kind === 'maybe' ? DS.maybe : DS.hits)[i];
    if (!r) return null;
    const t = (j == null || j < 0) ? r : r.copies[j];
    return t && t.filename ? t : null;
}

function dsPlay(kind, i, j) {
    const t = dsTarget(kind, i, j);
    if (!t) return;
    if (DS.hot) dsEl('dsR-' + DS.hot.replace(':', '-'))?.classList.remove('hot');
    DS.hot = `${kind}:${i}`;
    dsEl(`dsR-${kind}-${i}`)?.classList.add('hot');
    svPlay(t.filename, t.start, t.file_dir);
}

function dsGoto(kind, i, j) {
    const t = dsTarget(kind, i, j);
    if (t) svGoto(t.filename, t.start, t.file_dir);
}

/* The right column: the plan — or, before there is one, how it works. */
function dsRenderPlan() {
    const el = dsEl('dsPlan');
    if (!el) return;
    const html = dsPlanHtml(DS.cur);
    if (html === DS.planHtml && el.innerHTML) return;
    DS.planHtml = html;
    el.innerHTML = html;
}

function dsPlanHtml(c) {
    if (!c) {
        return `<div class="sv-src-title">How it works</div><div class="sv-src-empty ds-how">
            <p><b>1 · Plan</b> — the language model turns the question into the situation, what
               counts and what doesn't, and keywords.</p>
            <p><b>2 · Candidates</b> — passages close in meaning and passages with the keywords,
               from the whole RAG index; copies of a recording in other folders fold into one.</p>
            <p><b>3 · Ranking</b> — the reranker orders every candidate by the criteria.</p>
            <p><b>4 · Checking</b> — the language model reads the best ones one at a time and
               answers yes, unclear or no.</p>
            <p>Only what it confirmed is listed; unclear ones are kept apart. It runs in the LLM
               queue on the server — close this any time, the result is kept here.</p></div>`;
    }
    const p = c.plan;
    if (!p) {
        return `<div class="sv-src-title">The plan</div><div class="sv-src-empty">${dsActive(c) || !c.loaded
            ? 'The language model turns the question into the situation, criteria and keywords — they appear here.'
            : 'No plan was written.'}</div>`;
    }
    const list = v => (Array.isArray(v) ? v : (v ? [v] : [])).map(String).filter(Boolean);
    const text = v => Array.isArray(v) ? v.join(' ') : String(v || '');
    const chips = (v, cls) => list(v).map(k =>
        `<span class="ds-chip${cls ? ' ' + cls : ''}">${esc(k)}</span>`).join('');
    const row = (k, v) => v ? `<div class="ds-plan-k">${k}</div><div class="ds-plan-v">${v}</div>` : '';
    return `<div class="sv-src-title">The plan${p.source === 'fallback'
            ? ' <span class="an-warn" title="No language model was reachable — generic criteria and the words of the question">· generic</span>'
            : ''}</div>
        <div class="ds-plan">
            ${row('Situation', esc(text(p.situation)))}
            ${row('Counts', esc(text(p.criteria_yes)))}
            ${row('Does not count', esc(text(p.criteria_no)))}
            ${row('Must contain', chips(p.must_terms, 'must'))}
            ${row('Keywords', chips(p.keywords))}
        </div>`;
}

// ── Stop and delete ──────────────────────────────────────────────────────

async function dsStop() {
    const cur = DS.cur;
    if (!cur || !dsActive(cur)) return;
    const btn = dsEl('dsStopBtn');
    if (btn) btn.disabled = true;
    let d;
    try {
        const r = await fetch(`/api/rag/deep/${encodeURIComponent(cur.id)}/stop`, {method: 'POST'});
        d = await r.json().catch(() => ({}));
        if (!r.ok || d.error) throw new Error(d.error || `HTTP ${r.status}`);
    } catch (e) {
        Toast.error('Not stopped: ' + (e.message || e));
        if (btn) btn.disabled = false;
        return;
    }
    // `ok: false`: it had ended on its own meanwhile — the poll shows how
    Toast.info(d.ok === false ? 'It is not running any more'
                              : 'Stopping — what was found so far is kept');
    // Ask at once, so the view shows it stopped
    if (DS.cur === cur && !DS.inflight) dsPoll();
}

/* Two clicks — one stray click must not lose a search (the armed button,
 * as for summaries). */
function dsDeleteClick(btn, id) {
    if (!(DS.armed === id && Date.now() < DS.armedUntil)) {
        DS.armed = id;
        DS.armedUntil = Date.now() + DS_ARMED_MS;
        btn.classList.add('armed');
        btn.textContent = 'Delete?';
        setTimeout(() => {
            if (DS.armed !== id || Date.now() < DS.armedUntil) return;
            DS.armed = '';
            for (const b of [btn, dsEl('dsDelBtn')]) {
                if (b && b.classList.contains('armed')) {
                    b.classList.remove('armed');
                    b.textContent = b.dataset.label || '✕';
                }
            }
            dsRenderList();
        }, DS_ARMED_MS);
        return;
    }
    DS.armed = '';
    return dsDelete(id);
}

async function dsDelete(id) {
    try {
        const r = await fetch('/api/rag/deep/' + encodeURIComponent(id), {method: 'DELETE'});
        const d = await r.json().catch(() => ({}));
        // Gone already is what was wanted
        if (r.status !== 404 && (!r.ok || d.error)) throw new Error(d.error || `HTTP ${r.status}`);
    } catch (e) {
        Toast.error('Not deleted: ' + (e.message || e));
        return;
    }
    DS.list = DS.list.filter(s => s.id !== id);
    if (DS.cur && DS.cur.id === id) {
        dsStopPoll();
        DS.cur = null;
        DS.hits = [];
        DS.maybe = [];
        dsShowForm();
    }
    dsRenderList();
    dsLoadList();
}

// ── The list ─────────────────────────────────────────────────────────────

/* The kept deep searches. One request at a time — asked for again while one
 * is on its way, it asks once more after it; and again every few seconds
 * while one runs that is not the one shown (that one has its own poll). */
async function dsLoadList() {
    if (DS.listCtrl) { DS.listAgain = true; return; }
    clearTimeout(DS.listTimer);
    DS.listTimer = null;
    if (!dsWanted()) return;
    const seq = DS.listSeq;
    const ctrl = DS.listCtrl = new AbortController();
    let d = null, error = '';
    try {
        const r = await fetch('/api/rag/deep', {signal: ctrl.signal});
        d = await r.json();
        if (!r.ok || !d || d.error) throw new Error((d && d.error) || `HTTP ${r.status}`);
    } catch (e) {
        error = String(e.message || e) || 'no answer';
    }
    if (DS.listCtrl === ctrl) DS.listCtrl = null;
    if (seq !== DS.listSeq) return;
    if (!error) {
        DS.list = Array.isArray(d.searches) ? d.searches : [];
        DS.listError = '';
        if (DS.cur) dsListPatch(DS.cur);      // its own poll is the newer word on it
    } else {
        DS.listError = error;
    }
    dsRenderList();
    if (DS.listAgain) { DS.listAgain = false; dsLoadList(); return; }
    const others = DS.list.some(s => dsActive(s) && !(DS.cur && s.id === DS.cur.id));
    if (dsWanted() && (error || others)) {
        DS.listTimer = setTimeout(dsLoadList, error ? DS_BACKOFF_MAX : DS_LIST_EVERY);
    }
}

function dsListPatch(c) {
    if (!c || !c.loaded) return;
    const item = DS.list.find(s => s.id === c.id);
    if (item) {
        Object.assign(item, {status: c.status, stage: c.stage,
                             counts: c.counts || item.counts, query: c.query || item.query});
    }
}

function dsRenderList() {
    const el = dsEl('dsList');
    if (!el) return;
    const cur = DS.cur && DS.cur.id;
    let html = DS.list.map(s => {
        const st = DS_STATUS[s.status] || {text: s.status || '', cls: ''};
        const c = s.counts || {};
        const hits = Number(c.hits) || 0;
        const what = [`${dsN(hits)} hit${hits === 1 ? '' : 's'}`];
        if (Number(c.maybe)) what.push(`${dsN(c.maybe)} unclear`);
        if (st.text) what.push(st.text);
        const armed = DS.armed === s.id && Date.now() < DS.armedUntil;
        return `<div class="sv-chat-item an-item ds-item${s.id === cur ? ' active' : ''}" role="button" tabindex="0" data-keyclick
                     onclick="dsLoad('${escJs(s.id)}')" title="${esc(s.query || '')}">
            <span class="an-dot ${st.cls}"></span>
            <span class="sv-chat-item-title">${esc(s.query || '…')}<small>${esc(what.join(' · '))}</small></span>
            <span class="sv-chat-item-when">${esc(typeof chatWhen === 'function' ? chatWhen(s.created) : '')}</span>
            <button class="sv-chat-item-del${armed ? ' armed' : ''}" data-label="✕" title="Delete this deep search"
                    onclick="event.stopPropagation();dsDeleteClick(this, '${escJs(s.id)}')">${armed ? 'Delete?' : '✕'}</button>
        </div>`;
    }).join('');
    if (!DS.list.length) {
        html = DS.listError
            ? `<div class="sv-chat-empty-list an-bad">The list could not be loaded: ${esc(DS.listError)}</div>`
            : '<div class="sv-chat-empty-list">No deep searches yet. Describe a situation and press '
              + '🔎 Start the deep search — the result is kept here.</div>';
    }
    if (html !== DS.listHtml || !el.innerHTML) {
        DS.listHtml = html;
        el.innerHTML = html;
    }
    const busy = DS.list.filter(dsActive).length;
    const badge = dsEl('svTabNDeep');
    if (badge) badge.textContent = busy ? String(busy) : '';
}
