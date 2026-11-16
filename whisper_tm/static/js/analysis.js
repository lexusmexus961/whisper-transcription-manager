/* ══════════════════════════════════════════════════════════════════════════
 *  ANALYSES — one task over many recordings (core/analysis.py)
 *
 *  e.g. "over a set of support calls — list every question about
 *  pricing", or "three videos on the same topic, summarized together"
 *
 *    · the analyses on the left, kept on the server;
 *    · in the middle a new one — which recordings (the file list's
 *      selection, a folder, the files of a search), the task, how — or the
 *      one chosen: its answer with a chip for every passage it cites, the
 *      notes of each part, what it read;
 *    · on the right the passages the answer cites, each playable and
 *      openable in its transcript, as in Ask.
 *
 *  It runs in the LLM queue, on the server: the page only asks how far it is.
 * ══════════════════════════════════════════════════════════════════════════ */

const AN = {
    list: [],
    presets: [],
    cur: null,             // the analysis shown — null: the form for a new one
    timer: null,           // asks again while the one shown runs
    listTimer: null,       // the list, while any runs
    hot: '',               // the passage last played
    src: 'folder',         // where the set comes from: selection | folder | search
    preset: '',            // the template the task was started from
    selection: null,       // {base_dir, filenames} — taken from the file list
    search: null,          // {label, abs_paths, more, key} — collected from the search
    searchSeq: 0,          // the newest collection — older ones are dropped
    searchKey: null,       // svParams() of the collection in flight
    plan: null,
    planSeq: 0,
    planTimer: null,
    notesOpen: null,       // the notes, opened or closed by hand — else open until there's an answer
};

const AN_STATUS = {
    waiting:   {text: 'waiting for transcripts', cls: 'wait'},
    queued:    {text: 'waiting', cls: 'wait'},
    running:   {text: 'running', cls: 'run'},
    done:      {text: 'done', cls: 'ok'},
    error:     {text: 'failed', cls: 'err'},
    cancelled: {text: 'cancelled', cls: 'off'},
};
const AN_SEARCH_FILES = 20000;     // as many as one analysis takes

function anEl(id) { return document.getElementById(id); }

/* The search workspace, on this tab. */
function openAnalyses() {
    const ov = anEl('svOverlay');
    if (!ov) return;
    ov.classList.add('visible');
    if (svState.tab === 'analysis') anOpen();
    else svTab('analysis');
}

function anOpen() {
    anLoadList();
    // Files ticked in the file list are offered as "Selected in the file
    // list" however Analyses was opened (the rail, Home, a link) — not only
    // through the selection bar's button
    const ticked = [...(typeof selectedFiles !== 'undefined' ? selectedFiles : [])];
    if (ticked.length) {
        AN.selection = {base_dir: anEl('dirInput')?.value || '', filenames: ticked};
    }
    if (typeof _llmModels === 'function' && !anEl('anModel').options.length) {
        _llmModels('anModel', 'summary').catch(() => {});
    }
    // The search may have changed while another tab was shown — the
    // collected recordings then belong to the old one. Nothing is collected
    // here (up to 10 × 5000 hits on every visit): anSearchStale() marks it,
    // and the form, the source button or Start collect when it is needed.
    if (AN.cur) {
        anShowView();
        // Polling stops while another tab is shown — coming back restarts it.
        if (anBusy()) anPoll();
    } else {
        anShowForm();
    }
}

/* "🧠 Analyze" in the file list's selection bar. */
function analyzeSelection() {
    const names = [...(typeof selectedFiles !== 'undefined' ? selectedFiles : [])];
    if (!names.length) return;
    AN.selection = {base_dir: anEl('dirInput')?.value || '', filenames: names};
    AN.src = 'selection';
    AN.cur = null;
    openAnalyses();
    anShowForm();
}

/* An analysis from elsewhere — the LLM queue's row. */
function openAnalysis(id) {
    openAnalyses();
    anLoad(id);
}

// ── The list ─────────────────────────────────────────────────────────────

async function anLoadList() {
    try {
        const d = await (await fetch('/api/analyses')).json();
        AN.list = d.analyses || [];
        if (!AN.presets.length) { AN.presets = d.presets || []; anRenderPresets(); }
        AN.listError = '';
    } catch (e) {
        AN.listError = String(e.message || e) || 'no answer';
    }
    anRenderList();
    clearTimeout(AN.listTimer);
    if (AN.list.some(a => ['waiting', 'queued', 'running'].includes(a.status))) {
        AN.listTimer = setTimeout(() => { if (anVisible()) anLoadList(); }, 5000);
    }
}

/* In view: the tab shown, the search open, the browser tab visible. Closing
 * the search keeps the tab, so without this the polling would go on;
 * anOpen and the browser tab shown again pick it up. */
function anVisible() {
    const ov = document.getElementById('svOverlay');
    return !document.hidden && svState.tab === 'analysis' && (!ov || ov.classList.contains('visible'));
}

if (typeof document !== 'undefined' && document.addEventListener) {
    document.addEventListener('visibilitychange', () => {
        if (document.hidden || !anVisible()) return;
        anLoadList();
        if (AN.cur && anBusy()) anPoll();
    });
}

function anRenderList() {
    const el = anEl('anList');
    if (!el) return;
    const cur = AN.cur && AN.cur.id;
    el.innerHTML = AN.list.length ? AN.list.map(a => {
        const st = AN_STATUS[a.status] || {text: a.status, cls: ''};
        const n = (a.counts || {}).transcribed ?? (a.counts || {}).files;
        const what = a.status === 'running' ? `${a.progress || 0} %` : st.text;
        return `<div class="sv-chat-item an-item${a.id === cur ? ' active' : ''}" role="button" tabindex="0" data-keyclick onclick="anLoad('${escJs(a.id)}')"
                     title="${esc(a.task || a.title)}">
            <span class="an-dot ${st.cls}"></span>
            <span class="sv-chat-item-title">${esc(a.title)}<small>${n != null ? esc(plural(n, 'recording')) + ' · ' : ''}${esc(what)}</small></span>
            <span class="sv-chat-item-when">${esc(chatWhen(a.created))}</span>
            <button class="sv-chat-item-del" title="Delete this analysis"
                    onclick="event.stopPropagation();anDelete('${escJs(a.id)}')">✕</button>
        </div>`;
    }).join('') : AN.listError
        ? `<div class="sv-chat-empty-list bad">The analyses could not be loaded: ${esc(AN.listError)}</div>`
        : '<div class="sv-chat-empty-list">No analyses yet. Choose recordings and a task — the result is kept here.</div>';
    const busy = AN.list.filter(a => ['waiting', 'queued', 'running'].includes(a.status)).length;
    const badge = anEl('svTabNAnalysis');
    if (badge) badge.textContent = busy ? String(busy) : '';
}

// ── A new one ────────────────────────────────────────────────────────────

function anNew() {
    clearTimeout(AN.timer);
    AN.cur = null;
    anRenderList();
    anShowForm();
}

function anShowForm() {
    clearTimeout(AN.timer);
    anEl('anView').hidden = true;
    anEl('anForm').hidden = false;
    if (AN.src === 'selection' && !(AN.selection && AN.selection.filenames.length)) AN.src = 'folder';
    if (AN.src === 'folder' && !anEl('anFolder').value) anEl('anFolder').value = anEl('dirInput')?.value || '';
    // The folder's own name, not the root every path starts with
    if (typeof showPathEnd === 'function') showPathEnd(anEl('anFolder'));
    // The form is where the recordings are used: a search source collected
    // for another search than the one now set is collected anew.
    if (AN.src === 'search' && AN.search && anSearchStale()) anCollectSearchOnce();
    anSourceSync();
    anRenderSources();
    anPlanSoon(0);
}

function anRenderPresets() {
    const el = anEl('anPresets');
    if (!el) return;
    el.innerHTML = AN.presets.map(p =>
        `<button class="an-preset" onclick="anPreset('${escJs(p.key)}')" title="${esc(p.task)}">${esc(p.label)}</button>`).join('');
}

/* A template into the task field — "Find every place where …" with the
 * cursor at the gap. */
function anPreset(key) {
    const p = AN.presets.find(x => x.key === key);
    const ta = anEl('anTask');
    if (!p || !ta) return;
    ta.value = p.task;
    AN.preset = p.key;
    ta.focus();
    const gap = p.task.indexOf('…');
    if (gap >= 0) ta.setSelectionRange(gap, gap + 1);
    anStartSync();
}

function anSource(kind) {
    AN.src = kind;
    anSourceSync();
    if (kind === 'search' && anSearchStale()) {
        anCollectSearchOnce();
    } else {
        anPlanSoon(0);
    }
}

/* The collected recordings are for another search than the one now set. */
function anSearchStale() {
    return !AN.search || AN.search.key !== svParams();
}

/* Collect the current search — unless exactly this one is already being
 * collected (a second click, or the form shown again, waits for it). */
function anCollectSearchOnce() {
    if (AN.searchKey !== svParams()) anCollectSearch();
}

function anSourceSync() {
    const sel = AN.selection && AN.selection.filenames.length;
    const searching = !!((svState.query || '').trim() || (typeof svHasFilter === 'function' && svHasFilter()));
    document.querySelectorAll('#anSourceSeg .vx-seg-btn').forEach(b => {
        b.classList.toggle('active', b.dataset.src === AN.src);
        if (b.dataset.src === 'selection') b.disabled = !sel;
        if (b.dataset.src === 'search') b.disabled = !searching && !AN.search;
    });
    anEl('anSelCount').textContent = sel ? AN.selection.filenames.length.toLocaleString() : '';
    anEl('anSearchCount').textContent = AN.search ? AN.search.abs_paths.length.toLocaleString() : '';
    anEl('anSrcFolder').hidden = AN.src !== 'folder';
    const note = anEl('anSrcNote');
    if (AN.src === 'selection' && sel) {
        const names = AN.selection.filenames;
        note.innerHTML = `<b>${names.length.toLocaleString()}</b> selected in <span class="an-path">${esc(AN.selection.base_dir || '/')}</span>`
            + `<div class="an-names">${names.slice(0, 6).map(n => `<span>${esc(svShortName(n))}</span>`).join('')}`
            + `${names.length > 6 ? `<span>… ${(names.length - 6).toLocaleString()} more</span>` : ''}</div>`;
    } else if (AN.src === 'search') {
        note.innerHTML = AN.search
            ? `The recordings with hits for <b>${esc(AN.search.label)}</b>${AN.search.more ? ` — the first ${AN.search.abs_paths.length.toLocaleString()}` : ''}.`
            : 'Collecting the recordings of the search …';
    } else {
        note.innerHTML = '';
    }
}

/* The recordings the search finds — every page of hits, each recording once. */
async function anCollectSearch() {
    const q = (svState.query || '').trim();
    const label = q ? `"${q}"` : 'the filters';
    const key = svParams();
    const seq = ++AN.searchSeq;
    AN.searchKey = key;
    AN.search = null;
    anSourceSync();
    const seen = new Set();
    let more = false;
    try {
        for (let offset = 0; offset < 50000; offset += 5000) {
            const d = await (await fetch(`/api/search?${key}&limit=5000&offset=${offset}`)).json();
            if (seq !== AN.searchSeq) return;     // a newer search took over
            const items = d.results || [];
            for (const it of items) {
                const p = svAbsPath(it);
                if (p) seen.add(p);
            }
            if (seen.size >= AN_SEARCH_FILES) { more = true; break; }
            if (items.length < 5000) break;
        }
    } catch (e) {
        if (seq !== AN.searchSeq) return;
        Toast.error('The search could not be read: ' + (e.message || e));
    }
    if (seq !== AN.searchSeq) return;
    AN.searchKey = null;
    AN.search = {label, abs_paths: [...seen].slice(0, AN_SEARCH_FILES), more, key};
    if (AN.src === 'search') { anSourceSync(); anPlanSoon(0); }
}

function anUseOpenFolder() {
    anEl('anFolder').value = anEl('dirInput')?.value || '';
    if (typeof ppUpdateStars === 'function') ppUpdateStars();
    if (typeof showPathEnd === 'function') showPathEnd(anEl('anFolder'));
    anPlanSoon(0);
}

/* The path picker chose a folder (pathpicker.js, place 'an'). */
function anFolderChosen() {
    anPlanSoon(0);
}

function anSourceSpec() {
    if (AN.src === 'selection' && AN.selection) {
        return {kind: 'files', base_dir: AN.selection.base_dir, filenames: AN.selection.filenames};
    }
    if (AN.src === 'search') {
        return AN.search ? {kind: 'paths', abs_paths: AN.search.abs_paths, label: AN.search.label} : null;
    }
    const folder = (anEl('anFolder').value || '').trim();
    return folder ? {kind: 'folder', folder, recursive: anEl('anRecursive').checked} : null;
}

function anPlanSoon(ms = 400) {
    clearTimeout(AN.planTimer);
    AN.planTimer = setTimeout(anPlan, ms);
}

/* What it would read: counted on the server before anything runs. */
async function anPlan() {
    const box = anEl('anPlan');
    const source = anSourceSpec();
    const seq = ++AN.planSeq;
    if (!source) {
        AN.plan = null;
        box.innerHTML = '<span class="an-muted">Choose the recordings above.</span>';
        return anStartSync();
    }
    box.innerHTML = '<span class="an-muted">Counting …</span>';
    let d;
    try {
        const r = await fetch('/api/analyses/plan', {method: 'POST', headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({source, order: anEl('anOrder').value, speakers: anEl('anSpeakers').checked})});
        d = await r.json();
        if (!r.ok || d.error) throw new Error(d.error || r.status);
    } catch (e) {
        if (seq !== AN.planSeq) return;
        AN.plan = null;
        box.innerHTML = `<span class="an-bad">${esc(e.message || e)}</span>`;
        return anStartSync();
    }
    if (seq !== AN.planSeq) return;
    AN.plan = d;
    box.innerHTML = anPlanHtml(d);
    anStartSync();
}

function anPlanHtml(d) {
    const n = x => Number(x || 0).toLocaleString();
    if (!d.files) return '<span class="an-muted">No recordings there.</span>';
    if (d.too_large && !d.transcribed) {
        return `<span class="an-bad">${n(d.files)} files — more than one analysis takes (${n(d.limit_files)}). Choose a smaller set.</span>`;
    }
    const facts = [`<b>${n(d.transcribed)}</b> recording${d.transcribed === 1 ? '' : 's'} with a transcript`];
    if (d.missing) facts.push(`<span class="an-warn" title="Not transcribed yet — they are left out">${n(d.missing)} without one</span>`);
    if (d.empty) facts.push(`${n(d.empty)} without speech`);
    facts.push(`${n(d.words)} word${d.words === 1 ? '' : 's'}`);
    const how = d.single
        ? 'read in <b>one go</b>'
        : `read in <b>${n(d.parts)} part${d.parts === 1 ? '' : 's'}</b>, then one answer from their notes`;
    const time = d.estimate_sec ? ` · about ${anDuration(d.estimate_sec)}` : '';
    return `<div class="an-facts">${facts.join(' · ')}</div>`
        + `<div class="an-how">${how}${time}</div>`
        + (d.too_large ? `<div class="an-bad">${n(d.parts)} parts are more than one analysis reads (${n(d.max_parts)}) — choose a smaller set.</div>` : '')
        + (d.sample && d.sample.length ? `<div class="an-names">${d.sample.map(s => `<span>${esc(s)}</span>`).join('')}${d.transcribed > d.sample.length ? '<span>…</span>' : ''}</div>` : '');
}

function anDuration(sec) {
    sec = Math.round(sec || 0);
    if (sec < 60) return `${sec} s`;
    const m = Math.round(sec / 60);
    return m < 60 ? `${m} min` : `${Math.floor(m / 60)} h ${m % 60} min`;
}

function anStartSync() {
    const btn = anEl('anStartBtn');
    const note = anEl('anStartNote');
    const task = (anEl('anTask')?.value || '').trim();
    const p = AN.plan;
    const ready = !!(p && p.transcribed && !p.too_large && task);
    if (btn) btn.disabled = !ready;
    if (note) {
        note.textContent = !p || !p.transcribed ? 'Choose recordings that have a transcript.'
            : p.too_large ? 'The set is too large.'
            : !task ? 'Write the task — or start from a template.'
            : 'It runs in the LLM queue — you can close this.';
    }
}

async function anStart() {
    // Never analyse the recordings of an earlier search: collect the
    // current one first — its plan then enables Start again.
    if (AN.src === 'search' && anSearchStale()) {
        anCollectSearchOnce();
        anSourceSync();
        return;
    }
    const source = anSourceSpec();
    const task = (anEl('anTask').value || '').trim();
    if (!source || !task) return;
    const btn = anEl('anStartBtn');
    btn.disabled = true;
    let d;
    try {
        const r = await fetch('/api/analyses', {method: 'POST', headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({source, task, title: anEl('anTitle').value.trim(),
                                  order: anEl('anOrder').value, speakers: anEl('anSpeakers').checked,
                                  merge: anEl('anMerge').checked, model: anEl('anModel').value,
                                  preset: AN.preset})});
        d = await r.json();
        if (!r.ok || d.error) throw new Error(d.error || r.status);
    } catch (e) {
        Toast.error('Not started: ' + (e.message || e));
        btn.disabled = false;
        return;
    }
    Toast.success('🧠 The analysis is in the LLM queue');
    anEl('anTitle').value = '';
    AN.cur = d.analysis;
    anShowView();
    anLoadList();
    anPoll();
}

// ── One analysis ─────────────────────────────────────────────────────────

async function anLoad(id) {
    clearTimeout(AN.timer);
    try {
        const r = await fetch('/api/analyses/' + encodeURIComponent(id));
        if (!r.ok) throw new Error('not found');
        AN.cur = await r.json();
    } catch (e) {
        Toast.error('Analysis not found');
        return anLoadList();
    }
    AN.hot = '';
    AN.notesOpen = null;
    anShowView();
    anRenderList();
    if (anBusy()) AN.timer = setTimeout(anPoll, 2000);
}

function anBusy() {
    return !!(AN.cur && ['waiting', 'queued', 'running'].includes(AN.cur.status));
}

/* While it runs: ask again every two seconds; at the end the list too. */
async function anPoll() {
    clearTimeout(AN.timer);
    const cur = AN.cur;
    if (!cur) return;
    try {
        const d = await (await fetch('/api/analyses/' + encodeURIComponent(cur.id))).json();
        if (AN.cur !== cur || d.error) return;
        const was = cur.status;
        AN.cur = d;
        anRender();
        if (was !== d.status) anLoadList();
        if (was !== 'done' && d.status === 'done') Toast.success(`🧠 Analysis done: ${d.title}`);
    } catch (e) { /* the next poll tries again */ }
    if (anBusy() && anVisible()) AN.timer = setTimeout(anPoll, 2000);
}

function anShowView() {
    anEl('anForm').hidden = true;
    anEl('anView').hidden = false;
    anRender();
}

function anRender() {
    const el = anEl('anView');
    const a = AN.cur;
    if (!el || !a) return;
    const st = AN_STATUS[a.status] || {text: a.status, cls: ''};
    const c = a.counts || {};
    const n = x => Number(x || 0).toLocaleString();
    const opts = a.options || {};
    const order = {folder: 'folder by folder', name: 'by file name', dialogue: 'dialogue lines together'}[opts.order] || '';
    const meta = [
        `<b>${esc(a.label || '')}</b>`,
        c.transcribed != null ? `${n(c.transcribed)} recording${c.transcribed === 1 ? '' : 's'}` : `${n(c.files)} file${c.files === 1 ? '' : 's'}`,
        c.missing ? `<span class="an-warn">${n(c.missing)} without a transcript</span>` : '',
        order,
        esc(a.model_used || a.model || ''),
        a.duration ? `took ${anDuration(a.duration)}` : '',
        chatWhen(a.created),
    ].filter(Boolean).join(' · ');
    const busy = anBusy();
    const progress = busy ? `<div class="an-progress">
            <div class="an-bar"><div style="width:${Math.max(3, a.progress || 0)}%"></div></div>
            <div class="an-progress-row"><span>${esc(a.status === 'queued' ? 'Waiting in the LLM queue …'
                : a.status === 'waiting' ? 'Waiting for the transcripts — it starts by itself once they are there …'
                : (a.message || 'Running …'))}</span>
            <span class="vx-gap"></span><button class="vx-btn danger" onclick="anCancel()">Cancel</button></div></div>` : '';
    const live = a.status === 'running' && a.live_text
        ? `<div class="an-result live">${anMd(a.live_text)}<span class="sv-cursor"></span></div>` : '';
    const result = a.result ? `<div class="an-result">${anMd(a.result)}</div>` : '';
    // The notes: with an overview above them they are the findings, open;
    // not merged, they are the answer itself — not shown twice
    const notes = a.final_mode === 'notes' ? [] : (a.parts || []);
    const findings = a.final_mode === 'overview';
    const notesOpen = AN.notesOpen ?? (findings || !a.result);
    const notesHtml = notes.length ? `<details class="an-notes"${notesOpen ? ' open' : ''} ontoggle="AN.notesOpen=this.open">
            <summary>${findings ? 'Every finding, part by part' : 'Notes of each part'} · ${notes.length}${busy || findings ? '' : ` — what the answer was written from`}</summary>
            ${notes.map(p => `<div class="an-note${p.text ? '' : ' empty'}">
                <div class="an-note-head"><b>Part ${p.number} of ${p.total}</b>
                    <span title="${esc(p.from_value)} … ${esc(p.to_value)}">${esc(svShortName(p.from_value || ''))} … ${esc(svShortName(p.to_value || ''))}</span>
                    <span class="vx-gap"></span><span>${p.duration ? p.duration + ' s' : ''}</span></div>
                ${p.text ? `<div class="an-note-text">${anMd(p.text)}</div>` : '<div class="an-note-text an-muted">Nothing in this part for the task.</div>'}
            </div>`).join('')}</details>` : '';
    const actions = a.status === 'done' ? `<div class="an-actions">
            <a class="vx-btn" href="/api/analyses/${encodeURIComponent(a.id)}/export" download>⬇ Markdown</a>
            <button class="vx-btn" onclick="anAsk()" title="A conversation in Ask with this analysis attached">💬 Ask about it</button>
            <button class="vx-btn" onclick="anRerun(this)" title="Read the set again from the start — a folder with its new transcripts">↻ Run again</button>
            <span class="vx-gap"></span>
            <button class="vx-btn danger" onclick="anDelete('${escJs(a.id)}')">Delete</button></div>`
        : (!busy ? `<div class="an-actions"><button class="vx-btn" onclick="anRerun(this)">↻ Run again</button><span class="vx-gap"></span>
            <button class="vx-btn danger" onclick="anDelete('${escJs(a.id)}')">Delete</button></div>` : '');
    el.innerHTML = `<div class="an-head">
            <h3>${esc(a.title)}</h3><span class="an-status ${st.cls}">${esc(st.text)}</span>
        </div>
        <div class="an-task">${esc(a.task || '')}</div>
        <div class="an-meta">${meta}</div>
        ${progress}
        ${a.error ? `<div class="sv-msg-error">${esc(a.error)}</div>` : ''}
        ${result}${live}
        ${!result && !live && busy ? `<div class="an-muted an-wait">${a.status === 'queued' || a.status === 'waiting'
            ? 'The answer appears here once the LLM queue gets to it.'
            : (notes.length || a.parts_total) ? 'The answer is written once every part is read — the notes come in below.'
            : 'The model reads the recordings — the answer appears here as it is written.'}</div>` : ''}
        ${notesHtml}
        ${actions}`;
    anRenderSources();
}

// ── References: the passages the answer cites ───────────────────────────

function anRefs() { return (AN.cur && AN.cur.refs) || {}; }

/* Markdown; then every [S12] — also [S1, S2] — as a chip that plays. */
function anMd(text) {
    let html = typeof _renderMarkdown === 'function' ? _renderMarkdown(String(text || '')) : esc(text);
    const refs = anRefs();
    html = html.replace(CHAT_REFS_RE, (m, inner) =>
        inner.split(/\s*[,;]\s*/).map(k => {
            const r = refs[k];
            if (!r) return `<span class="sv-ref" title="Not among the passages">${esc(k)}</span>`;
            return `<button class="sv-ref${AN.hot === k ? ' hot' : ''}" data-ref="${esc(k)}"
                        onclick="anRefPlay('${k}')" ondblclick="anRefOpen('${k}')"
                        title="${esc(chatRefTitle(r))} — click plays, double-click opens the transcript">${esc(k)} ▶${r.short ? '' : ' ' + esc(svFmt(r.start))}</button>`;
        }).join(''));
    return typeof chatFoldRefs === 'function' ? chatFoldRefs(html) : html;
}

/* The passages in the order the answer cites them, then the notes. */
function anCited() {
    const a = AN.cur;
    if (!a) return [];
    const refs = anRefs();
    const out = [], seen = new Set();
    const texts = [a.result || a.live_text || '', ...(a.parts || []).map(p => p.text || '')];
    for (const t of texts) {
        for (const m of String(t).matchAll(CHAT_REFS_RE)) {
            for (const k of m[1].split(/\s*[,;]\s*/)) {
                if (refs[k] && !seen.has(k)) { seen.add(k); out.push(k); }
            }
        }
    }
    return out;
}

function anRenderSources() {
    const el = anEl('anSources');
    if (!el) return;
    if (!AN.cur) {
        el.innerHTML = `<div class="sv-src-title">How it works</div><div class="sv-src-empty">
            Every transcript of the set goes to the language model with the task. A large set is read in
            parts: each part's notes, with the passages they rest on, then one answer from all notes.<br><br>
            Every statement carries its passage — a chip that plays it and opens it in its transcript.
            The passages appear here.<br><br>
            <b>Dialogue order</b> puts the lines of a game dialogue together — games often keep each voice in a
            folder of its own, so the hero's lines and the other's lie apart.</div>`;
        return;
    }
    const cited = anCited();
    const refs = anRefs();
    if (!cited.length) {
        el.innerHTML = `<div class="sv-src-title">Passages</div><div class="sv-src-empty">${anBusy()
            ? 'The passages the notes and the answer cite appear here as they come.'
            : 'The answer cites no passage.'}</div>`;
        return;
    }
    el.innerHTML = `<div class="sv-src-title">Passages cited · ${cited.length.toLocaleString()}</div>` + cited.map(k => {
        const r = refs[k];
        return `<div class="sv-src${AN.hot === k ? ' hot' : ''}" id="anSrc-${esc(k)}" onclick="anRefPlay('${k}')">
            <div class="sv-src-head"><b>${esc(k)}</b><span class="sv-src-file" title="${esc(r.abs_path || r.file)}">${esc(svShortName(r.filename || r.file))}</span>
                <span>${r.short ? '' : esc(svFmt(r.start))}</span></div>
            ${chatSpeaker(r) ? `<div class="an-src-who">${esc(chatSpeaker(r))}</div>` : ''}
            <div class="sv-src-text">${esc(r.text || '')}</div>
            <div class="sv-src-acts">
                <button onclick="event.stopPropagation();anRefPlay('${k}')">▶ play</button>
                <button onclick="event.stopPropagation();anRefOpen('${k}')">↗ transcript</button>
            </div></div>`;
    }).join('');
}

function anRef(k) {
    const r = anRefs()[k];
    if (!r) return null;
    return {file: r.filename || r.file, start: Number(r.start) || 0, dir: r.file_dir || ''};
}

function anRefPlay(k) {
    const r = anRef(k);
    if (!r) return;
    AN.hot = k;
    svPlay(r.file, r.start, r.dir);
    document.querySelectorAll('#svPaneAnalysis .sv-ref').forEach(b => b.classList.toggle('hot', b.dataset.ref === k));
    document.querySelectorAll('#anSources .sv-src').forEach(b => b.classList.toggle('hot', b.id === 'anSrc-' + k));
    document.getElementById('anSrc-' + k)?.scrollIntoView({block: 'nearest'});
}

function anRefOpen(k) {
    const r = anRef(k);
    if (r) svGoto(r.file, r.start, r.dir);
}

// ── Actions ──────────────────────────────────────────────────────────────

async function anCancel() {
    if (!AN.cur) return;
    await fetch(`/api/analyses/${encodeURIComponent(AN.cur.id)}/cancel`, {method: 'POST'});
    Toast.info('Cancel requested — a part being read still finishes');
    anPoll();
}

async function anRerun(btn) {
    if (!AN.cur || AN.rerunning) return;
    AN.rerunning = true;
    if (btn) btn.disabled = true;
    let d;
    try {
        const r = await fetch(`/api/analyses/${encodeURIComponent(AN.cur.id)}/rerun`, {method: 'POST'});
        d = await r.json();
        if (!r.ok || d.error) throw new Error(d.error || r.status);
    } catch (e) {
        Toast.error('Not started: ' + (e.message || e));
        return;
    } finally {
        AN.rerunning = false;
        if (btn) btn.disabled = false;
    }
    AN.cur = d.analysis;
    anRender();
    anLoadList();
    anPoll();
}

async function anDelete(id) {
    const a = AN.list.find(x => x.id === id) || (AN.cur && AN.cur.id === id ? AN.cur : null);
    if (!confirm(`Delete the analysis "${a ? a.title : ''}"?`)) return;
    try {
        const r = await fetch('/api/analyses/' + encodeURIComponent(id), {method: 'DELETE'});
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
    } catch (e) {
        Toast.error('Not deleted: ' + (e.message || e));
        return;
    }
    if (AN.cur && AN.cur.id === id) { AN.cur = null; anShowForm(); }
    anLoadList();
}

async function anAsk() {
    if (!AN.cur) return;
    let d;
    try {
        const r = await fetch(`/api/analyses/${encodeURIComponent(AN.cur.id)}/ask`, {method: 'POST'});
        d = await r.json();
        if (!r.ok || d.error) throw new Error(d.error || r.status);
    } catch (e) {
        Toast.error('No conversation: ' + (e.message || e));
        return;
    }
    svTab('chat');
    await chatLoad(d.conversation);
    chatLoadList();
    document.getElementById('chatInput')?.focus();
}
