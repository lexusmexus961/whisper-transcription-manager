// ── LLM tools: correction (transparent review), translation,
//    transcript view switcher (original / translation / summaries)
//
// Modals in the toolbar design language (llm-* classes in app.css), full
// insight into every run (suggestions / discarded with reason / checked),
// translation tracks from the DB directly switchable in the transcript
// view. docs/LLM-PIPELINE.md.

let _corrFilename = null;
let _corrJobES = null;
let _corrJobId = null;
let _corrRun = null;          // current pending run (payload)
let _corrTab = 'suggestions';
let _transFilename = null;
let _transJobES = null;
let _transJobId = null;
let _corrTracker = null;   // llmQueueTrack handle
let _transTracker = null;

// View switcher state (applies to the currently open file)
let _txView = {mode: 'original', lang: '', tracks: [], map: null,
               showOriginal: false, sumCount: 0, sumPending: false};
let _sumPollTimer = null;

// ══════════════════════════════════════════════════════════════
//  Shared: job SSE, model list, word diff
// ══════════════════════════════════════════════════════════════

// Fills a model select and resolves to the /api/llm/status answer (null on
// failure). The first entry sends '' — then the task → provider assignment
// in the settings (llm.tasks) decides; a named model would pull its own
// provider along (llm_client.build_chat_payload). With `task`, the entry
// names the model that assignment picks; without (the folder dialog, whose
// task can change) it stays generic. It does not show the legacy
// llm.model there, which the assignment need not use at all.
function _llmModels(selectId, task) {
    return fetch('/api/llm/status').then(r => r.json()).then(st => {
        const sel = document.getElementById(selectId);
        if (!sel) return st;
        sel.innerHTML = '';
        const cur = (task && st.task_models?.[task]) || '';
        const def = document.createElement('option');
        def.value = ''; def.textContent = 'Default (task setting)' + (cur ? ': ' + cur : '');
        sel.appendChild(def);
        // The task's own model isn't listed a second time — the default runs it
        const inside = new Set(['', cur]);
        for (const [name, id] of Object.entries(st.models || {})) {
            if (typeof id !== 'string' || inside.has(id)) continue;
            inside.add(id);
            const o = document.createElement('option');
            o.value = id; o.textContent = name !== id ? `${name} (${id})` : name;
            sel.appendChild(o);
        }
        // Default models from the provider cards — a model set there
        // (e.g. gpt-4.1-nano) was otherwise not selectable in any dialog.
        for (const pm of (st.provider_models || [])) {
            if (!pm.model || inside.has(pm.model)) continue;
            inside.add(pm.model);
            const o = document.createElement('option');
            o.value = pm.model;
            o.textContent = `${pm.model} (${pm.providers})`;
            sel.appendChild(o);
        }
        return st;
    }).catch(() => null);
}

// LCS diff (words, CJK falls back to characters; >300 tokens = no markup)
function _diffHtml(old_value, new_value) {
    const tokenize = (s) => {
        const t = s.split(/(\s+)/).filter(x => x.length);
        return (t.length > 2) ? t : s.split('');
    };
    const a = tokenize(old_value), b = tokenize(new_value);
    if (a.length > 300 || b.length > 300) return {old_value: esc(old_value), new_value: esc(new_value)};
    const n = a.length, m = b.length;
    const dp = Array.from({length: n + 1}, () => new Uint16Array(m + 1));
    for (let i = n - 1; i >= 0; i--)
        for (let j = m - 1; j >= 0; j--)
            dp[i][j] = a[i] === b[j] ? dp[i+1][j+1] + 1 : Math.max(dp[i+1][j], dp[i][j+1]);
    let i = 0, j = 0;
    const outA = [], outB = [];
    while (i < n && j < m) {
        if (a[i] === b[j]) { outA.push(esc(a[i])); outB.push(esc(b[j])); i++; j++; }
        else if (dp[i+1][j] >= dp[i][j+1]) { outA.push(`<del>${esc(a[i])}</del>`); i++; }
        else { outB.push(`<ins>${esc(b[j])}</ins>`); j++; }
    }
    while (i < n) { outA.push(`<del>${esc(a[i])}</del>`); i++; }
    while (j < m) { outB.push(`<ins>${esc(b[j])}</ins>`); j++; }
    return {old_value: outA.join(''), new_value: outB.join('')};
}

function _segTime(id) {
    return activeTranscript?.segments?.[id]?.start_fmt || '';
}

// ══════════════════════════════════════════════════════════════
//  Correction modal
// ══════════════════════════════════════════════════════════════

function openCorrectionModal(filename) {
    _corrFilename = filename;
    _corrRun = null;
    _corrJobId = null;
    _corrTab = 'suggestions';
    document.getElementById('corrProgress').style.display = 'none';
    document.getElementById('corrResult').innerHTML = '';
    document.getElementById('corrHistory').innerHTML = '';
    // Step log belongs to the PREVIOUS run (possibly a different file) —
    // without clearing it, the dialog would show the sections of the
    // last-opened video after a cancel.
    document.getElementById('corrSteps').innerHTML = '';
    llmResetParts();
    document.getElementById('btnCorrStart').disabled = false;
    document.getElementById('btnCorrCancel').style.display = 'none';
    document.getElementById('correctionModal').classList.add('visible');
    // The threshold comes from the settings (llm.correction.score_threshold)
    // on every open — a fixed value in the markup would be sent with every
    // run, and the setting would never reach a run started from here.
    // The field starts empty: empty sends nothing and the server applies
    // the setting itself — until the answer arrives, or if it never does.
    const score = document.getElementById('corrScore');
    score.value = '';
    _llmModels('corrModel', 'correction').then(st => {
        const v = parseFloat(st?.correction_score_threshold);
        // Only an untouched field: a value typed meanwhile wins
        if (score.value === '' && !isNaN(v)) score.value = String(v);
    });
    _corrLoadState();
    // Is a correction already running for this file? Then attach to it —
    // a dialog closed in the meantime did not stop it.
    llmQueueOpen('correction', filename).then(e => { if (e) _corrTrack(e.id); });
}

function closeCorrectionModal() {
    // Only release the display — the job keeps running in the LLM queue.
    if (_corrTracker) { _corrTracker.stop(); _corrTracker = null; }
    document.getElementById('correctionModal').classList.remove('visible');
}

async function _corrLoadState() {
    try {
        const r = await fetch('/api/llm/edits/' + encodeURIComponent(_corrFilename));
        const d = await r.json();
        _corrRun = d.pending || null;
        _corrRenderResult();
        _corrShowHistory(d.history || []);
    } catch (e) {}
}

async function startCorrection() {
    const params = {
        only_marked: document.getElementById('corrOnlySelected').checked,
        model: document.getElementById('corrModel')?.value || '',
    };
    const thr = parseFloat(document.getElementById('corrScore').value);
    if (!isNaN(thr)) params.score_threshold = thr;
    try {
        const id = await llmQueueAdd('correction', _corrFilename, params, true);
        _corrTrack(id);
    } catch (e) {
        document.getElementById('corrProgress').querySelector('.llm-progress-text')
            .textContent = '❌ ' + e.message;
        document.getElementById('corrProgress').style.display = '';
    }
}

// Attaches to a queue entry (even one that was already running before
// the dialog was opened) — closing does NOT cancel anything.
function _corrTrack(id) {
    _corrJobId = id;
    const prog = document.getElementById('corrProgress');
    const bar = prog.querySelector('.llm-progress-fill');
    const text = prog.querySelector('.llm-progress-text');
    prog.style.display = '';
    // Fresh run = fresh display: the bar color may still be red from the
    // last error, and the step log only fills once the FIRST chunk is
    // done — until then the old content (possibly from the previous
    // video) would otherwise still be showing.
    bar.style.background = 'var(--accent)';
    bar.style.width = '3%';
    text.textContent = '';
    document.getElementById('corrSteps').innerHTML = '';
    llmResetParts();
    document.getElementById('btnCorrStart').disabled = true;
    document.getElementById('btnCorrCancel').style.display = '';
    if (_corrTracker) _corrTracker.stop();
    _corrAbortPending = false;
    document.getElementById('btnCorrCancel').disabled = false;
    _corrTracker = llmQueueTrack(id, (e) => {
        bar.style.width = Math.max(3, e.progress || 0) + '%';
        // A requested cancel stays visible — the tick would otherwise
        // immediately overwrite it again with "Chunk n/m".
        text.innerHTML = _corrAbortPending
            ? '⏹ Cancel requested — takes effect after the current section…'
            : (e.status === 'pending' ? llmWaitText(e) : esc(e.message || ''));
        // Live step log: which section found what
        const parts = e.partial_results || [];
        if (parts.length) {
            llmRenderParts(document.getElementById('corrSteps'), parts, 'q' + id);
        }
    }, (e) => {
        _corrCleanUp();
        const r = e.result || {};
        // Two cases where "Done" without explanation looks broken:
        if (r.confidence_data === false && r.only_marked !== false && typeof Toast !== 'undefined')
            Toast.info('No word confidences in the transcript — the entire transcript was checked from context');
        if (!r.run_id && !r.chunks) {
            document.getElementById('corrSteps').innerHTML =
                `<div class="llm-empty"><span class="large">✨</span>
                 Nothing to check: no segment is below the threshold.
                 Raise the threshold or deselect "Only flagged segments"
                 to have the entire transcript checked.</div>`;
        }
        _corrLoadState();
    }, (error) => {
        _corrCleanUp(true);
        text.textContent = '❌ ' + error;
        bar.style.width = '100%';
        bar.style.background = 'var(--red)';
    });
}

function _corrCleanUp(error) {
    document.getElementById('btnCorrStart').disabled = false;
    document.getElementById('btnCorrCancel').style.display = 'none';
    if (!error) document.getElementById('corrProgress').style.display = 'none';
    if (_corrTracker) { _corrTracker.stop(); _corrTracker = null; }
}

async function cancelCorrection() {
    if (!_corrJobId) return;
    if (!await llmQueueCancel(_corrJobId)) return;
    // The cancel only takes effect at the next chunk boundary — a
    // running section can take minutes. Without this feedback the
    // button looks dead.
    _corrAbortPending = true;
    const text = document.querySelector('#corrProgress .llm-progress-text');
    if (text) text.textContent = '⏹ Cancel requested — takes effect after the current section…';
    document.getElementById('btnCorrCancel').disabled = true;
}
let _corrAbortPending = false;
let _transAbortPending = false;

function corrTab(tab) { _corrTab = tab; _corrRenderResult(); }

function _corrRenderResult() {
    const el = document.getElementById('corrResult');
    if (!_corrRun) { el.innerHTML = ''; return; }
    const p = _corrRun.payload || {};
    const sugg = p.suggestions || [];
    // Old runs (before the transparency rebuild) have `rejected` as a NUMBER
    // and no `checked` — the tabs then end up empty/off accordingly.
    const disc = Array.isArray(p.discarded) ? p.discarded : [];
    const discOld = (!Array.isArray(p.discarded) && p.discarded) ? p.discarded : 0;
    const chk = p.checked || [];

    // ── Run card: what happened ──
    const chips = [
        `<span class="llm-chip">🏷 <b>${esc(p.model || _corrRun.model || '?')}</b></span>`,
        `<span class="llm-chip">Chunks <b>${p.chunks ?? '?'}</b></span>`,
        p.duration ? `<span class="llm-chip">⏱ <b>${p.duration}s</b></span>` : '',
        `<span class="llm-chip">Threshold <b>${(p.thresholds?.score ?? 0.4)}</b></span>`,
        `<span class="llm-chip ${sugg.length ? 'green' : ''}">✏ <b>${sugg.length}</b> suggestion${sugg.length === 1 ? '' : 's'}</span>`,
        `<span class="llm-chip ${(disc.length || discOld) ? 'orange' : ''}">🚫 <b>${disc.length || discOld}</b> discarded</span>`,
        chk.length ? `<span class="llm-chip">✔ <b>${chk.length}</b> checked</span>`
                  : (p.marked_segments ? `<span class="llm-chip">✔ <b>${p.marked_segments}</b> checked</span>` : ''),
        p.confirmed ? `<span class="llm-chip green" title="The LLM explicitly reported these segments as fine">👍 <b>${p.confirmed}</b> confirmed</span>` : '',
        p.chunks_failed ? `<span class="llm-chip red">⚠ <b>${p.chunks_failed}</b> chunk error${p.chunks_failed === 1 ? '' : 's'}</span>` : '',
    ].filter(Boolean).join(' ');

    const openCount = sugg.filter(v => (v.status || 'open') === 'open').length;
    const applied_count = sugg.filter(v => v.status === 'applied').length;
    const rejectedCount = sugg.filter(v => v.status === 'rejected').length;
    const progress = sugg.length
        ? `<span class="llm-chip ${openCount ? '' : 'green'}">
             <b>${applied_count}</b>/${sugg.length} applied${rejectedCount ? ` · ${rejectedCount} rejected` : ''}${openCount ? ` · ${openCount} pending` : ''}</span>`
        : '';
    const actions = openCount ? `
        <span style="flex:1"></span>
        <button class="llm-btn-primary" onclick="corrApply(${_corrRun.id}, true)">✅ Apply all ${openCount}</button>
        <button class="tx-act-btn tx-act-danger" onclick="corrDiscard(${_corrRun.id})">✕ Discard run</button>`
        : `<span style="flex:1"></span>
        ${applied_count ? `<button class="tx-act-btn" onclick="corrUndo(${_corrRun.id})">↩ Undo all</button>` : ''}
        <button class="tx-act-btn tx-act-danger" onclick="corrDiscard(${_corrRun.id})">✕ Close run</button>`;

    let html = `<div class="llm-group">
        <div class="llm-group-title">Last run — result</div>
        <div style="display:flex;gap:6px;flex-wrap:wrap;align-items:center">${chips}${progress}${actions}</div>
        <div class="llm-tabs">
            <button class="llm-tab ${_corrTab==='suggestions'?'active':''}" onclick="corrTab('suggestions')">✏ Suggestions<span class="n">${sugg.length}</span></button>
            <button class="llm-tab ${_corrTab==='discarded'?'active':''}" onclick="corrTab('discarded')">🚫 Discarded<span class="n">${disc.length || discOld}</span></button>
            <button class="llm-tab ${_corrTab==='checked'?'active':''}" onclick="corrTab('checked')">✔ Checked<span class="n">${chk.length}</span></button>
        </div>
        <div id="corrTabBody">`;

    if (_corrTab === 'suggestions') {
        html += sugg.length ? _compareHtml(sugg) : `<div class="llm-empty"><span class="large">✨</span>
            The LLM checked ${plural(chk.length, 'segment')} and found no
            need for correction${disc.length ? ` — validation sorted out ${plural(disc.length, 'candidate')} (tab "Discarded")` : ''}.</div>`;
    } else if (_corrTab === 'discarded') {
        html += disc.length ? disc.map(v => {
            const hasDiff = v.new_value && v.old_value && v.new_value !== v.old_value;
            const d = hasDiff ? _diffHtml(v.old_value, v.new_value) : null;
            return `<div class="llm-map rejected">
                <div class="llm-map-head">
                    <span class="seg-id">[${v.id ?? '?'}]</span><span>${esc(_segTime(v.id))}</span>
                    <span class="llm-chip red">${esc(v.rejected_because || '')}</span>
                    ${v.confidence ? `<span class="llm-chip">Confidence ${v.confidence.toFixed(2)}</span>` : ''}
                </div>
                ${hasDiff ? `<div class="llm-diff-old">${d.old_value}</div><div class="llm-diff-new">${d.new_value}</div>`
                          : `<div class="llm-diff-old">${esc(v.old_value || v.new_value || '')}</div>`}
                ${v.llm_reason ? `<div class="llm-reason">💡 LLM: ${esc(v.llm_reason)}</div>` : ''}
            </div>`;
        }).join('') : (discOld
            ? `<div class="llm-empty"><span class="large">🚫</span>
                ${plural(discOld, 'candidate')} ${discOld === 1 ? 'was' : 'were'} discarded. This run comes from
                an older version that did not store the details —
                a new run shows them in full.</div>`
            : `<div class="llm-empty"><span class="large">🚫</span>Nothing discarded — every LLM response passed validation.</div>`);
    } else {
        html += chk.length ? chk.map(g => `
            <div class="llm-map" style="padding:7px 12px">
                <div class="llm-map-head" style="margin-bottom:2px">
                    <span class="seg-id">[${g.id}]</span><span>${esc(_segTime(g.id))}</span>
                    ${g.confirmed ? `<span class="llm-chip green">👍 confirmed by LLM</span>` : ''}
                    ${g.uncertain_words ? `<span class="llm-chip orange">${plural(g.uncertain_words, 'uncertain word')}</span>` : ''}
                    ${g.avg_logprob != null ? `<span class="llm-chip">logprob ${g.avg_logprob}</span>` : ''}
                </div>
                <div style="font-size:0.82rem;color:var(--tx-2)">${esc(g.text || '')}</div>
            </div>`).join('')
            : (p.marked_segments
                ? `<div class="llm-empty"><span class="large">✔</span>
                    ${plural(p.marked_segments, 'segment')} ${p.marked_segments === 1 ? 'was' : 'were'} checked. This run
                    comes from an older version that did not store the
                    details — a new run lists them.</div>`
                : `<div class="llm-empty"><span class="large">✔</span>No flagged segments — nothing to check.</div>`);
    }
    html += '</div></div>';
    el.innerHTML = html;
}

// ══════════════════════════════════════════════════════════════
//  Compare view (Notepad++/Git style)
//  The ENTIRE transcript in two columns: original on the left, the
//  corrected version on the right. Line numbers on both sides, unchanged
//  lines dimmed, changed lines colored with word diff, approve/reject
//  in the line margin, jump navigation across the changes.
// ══════════════════════════════════════════════════════════════

let _cmpOnlyChanges = false;
let _cmpPos = -1;            // index into the change list (navigation)

function corrToggleCompare() {
    _cmpOnlyChanges = !_cmpOnlyChanges;
    _corrRenderResult();
}

function _cmpChangedIds() {
    return (_corrRun?.payload?.suggestions || []).map(v => v.id);
}

function corrJumpChange(direction) {
    const ids = _cmpChangedIds();
    if (!ids.length) return;
    _cmpPos = (_cmpPos + direction + ids.length) % ids.length;
    const el = document.querySelector(`.cmp-row[data-seg="${ids[_cmpPos]}"]`);
    if (!el) return;
    document.querySelectorAll('.cmp-row.active').forEach(z => z.classList.remove('active'));
    el.classList.add('active');
    el.scrollIntoView({block: 'center', behavior: 'smooth'});
}

function _compareHtml(sugg) {
    const segments = activeTranscript?.segments || [];
    const bySeg = {};
    sugg.forEach(v => { bySeg[v.id] = v; });
    const changedIds = new Set(sugg.map(v => v.id));

    const lines = [];
    segments.forEach((s, idx) => {
        const v = bySeg[idx];
        const old_value = (s.text || '').trim();
        // The WHOLE line plays the spot on click — hitting only the
        // narrow line number is hard to hit. The ✓/✕ buttons stop the
        // propagation.
        const seek = `onclick="seekTo(${Number(s.start) || 0})" title="▶ ${esc(s.start_fmt || 'play from here')}"`;
        if (!v) {
            if (_cmpOnlyChanges) return;
            const t = esc(old_value);
            lines.push(`<div class="cmp-row same" data-seg="${idx}" ${seek}>
                <div class="cmp-page left"><span class="cmp-nr">${idx}</span><span class="cmp-text">${t}</span></div>
                <div class="cmp-page right"><span class="cmp-nr">${idx}</span><span class="cmp-text">${t}</span></div>
            </div>`);
            return;
        }
        const st = v.status || 'open';
        const d = _diffHtml(v.old_value || old_value, v.new_value || '');
        const actions = st === 'open'
            ? `<button class="ok" title="Apply" onclick="event.stopPropagation();corrApproveOne(${idx})">✓</button>
               <button class="reject" title="Reject" onclick="event.stopPropagation();corrDismissOne(${idx})">✕</button>`
            : (st === 'applied' ? `<span class="done">✓ applied</span>`
                                  : `<span class="done reject">✕ rejected</span>`);
        lines.push(`<div class="cmp-row ${st === 'applied' ? 'done' : (st === 'rejected' ? 'rejected' : 'changed')}" data-seg="${idx}" ${seek}>
            <div class="cmp-page left"><span class="cmp-nr">${idx}</span><span class="cmp-text">${d.old_value}</span></div>
            <div class="cmp-page right"><span class="cmp-nr">${idx}</span><span class="cmp-text">${d.new_value}</span></div>
            <div class="cmp-active">${actions}</div>
        </div>`);
    });

    const count = changedIds.size;
    return `
    <div class="cmp-bar">
        <span class="llm-chip">${count} changed line${count === 1 ? '' : 's'}</span>
        <span class="llm-chip">${segments.length} line${segments.length === 1 ? '' : 's'} total</span>
        <button class="tx-act-btn" onclick="corrJumpChange(-1)" title="Previous change">⌃</button>
        <button class="tx-act-btn" onclick="corrJumpChange(1)" title="Next change">⌄</button>
        <label class="llm-check"><input type="checkbox" ${_cmpOnlyChanges ? 'checked' : ''}
            onchange="corrToggleCompare()"> Changes only</label>
    </div>
    <div class="cmp-head"><div>Original</div><div>Corrected</div></div>
    <div class="cmp-body" id="cmpBody">${lines.join('')}</div>`;
}

async function corrApproveOne(segId) {
    if (!_corrRun) return;
    try {
        const r = await fetch('/api/llm/edits/apply', {
            method: 'POST', headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({filename: _corrFilename, run_id: _corrRun.id,
                                  ids: [segId]})});
        const d = await r.json();
        if (!r.ok || d.error) throw new Error(d.error || 'Error');
        Toast.success(`Segment ${segId} applied`);
        // Only update the affected segment instead of reloading the
        // whole transcript (noticeable with hundreds of segments).
        _segUpdateText(segId);
        await _corrLoadState();
    } catch (e) { Toast.error(e.message); }
}

function _segUpdateText(segId) {
    const v = (_corrRun?.payload?.suggestions || []).find(x => x.id === segId);
    const seg = activeTranscript?.segments?.[segId];
    if (!v || !seg) return;
    seg.text = v.new_value;
    seg.edited = 1;
    const el = document.querySelector(`#segList .seg[data-idx="${segId}"] .txt`);
    if (el && _txView.mode === 'original') el.innerHTML = buildWordHtml(seg);
}

async function corrDismissOne(segId) {
    if (!_corrRun) return;
    try {
        const r = await fetch('/api/llm/edits/dismiss', {
            method: 'POST', headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({filename: _corrFilename, run_id: _corrRun.id,
                                  ids: [segId]})});
        const d = await r.json();
        if (!r.ok || d.error) throw new Error(d.error || 'Error');
        await _corrLoadState();
    } catch (e) { Toast.error(e.message); }
}

function _corrShowHistory(history) {
    const el = document.getElementById('corrHistory');
    if (!history.length) { el.innerHTML = ''; return; }
    el.innerHTML = `<div class="llm-group"><div class="llm-group-title">History</div>`
        + history.map(h => {
            const dt = h.created_at ? new Date(h.created_at * 1000).toLocaleString() : '';
            const n = (h.payload?.applied || []).length;
            const badge = h.status === 'applied' ? 'green' : (h.status === 'undone' ? 'orange' : '');
            const undo = h.status === 'applied'
                ? ` <button class="tx-act-btn" style="height:22px;font-size:0.68rem" onclick="corrUndo(${h.id})">↩ Undo</button>` : '';
            return `<div style="display:flex;gap:8px;align-items:center;font-size:0.76rem;color:var(--tx-2);margin:4px 0">
                <span>${esc(dt)}</span>
                <span class="llm-chip ${badge}">${esc(h.status)}</span>
                ${n ? `<span class="llm-chip">${plural(n, 'edit')}</span>` : ''}${undo}</div>`;
        }).join('') + '</div>';
}

async function corrApply(runId, all_items) {
    // all_items=true → all pending suggestions (backend takes everything open without ids)
    const body = {filename: _corrFilename, run_id: runId};
    if (!all_items) {
        const ids = [...document.querySelectorAll('.corrPick:checked')].map(x => parseInt(x.value, 10));
        if (!ids.length) { Toast.error('Nothing selected'); return; }
        body.ids = ids;
    }
    try {
        const r = await fetch('/api/llm/edits/apply', {
            method: 'POST', headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(body)});
        const d = await r.json();
        if (!r.ok || d.error) throw new Error(d.error || 'Error');
        Toast.success(`${plural(d.applied, 'correction')} applied`);
        await _corrLoadState();
        if (typeof loadTranscript === 'function') loadTranscript(_corrFilename);
    } catch (e) { Toast.error(e.message); }
}

async function corrDiscard(runId) {
    try {
        await fetch('/api/llm/edits/reject', {
            method: 'POST', headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({filename: _corrFilename, run_id: runId})});
        _corrLoadState();
    } catch (e) {}
}

async function corrUndo(runId) {
    try {
        const r = await fetch('/api/llm/edits/undo', {
            method: 'POST', headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({filename: _corrFilename, run_id: runId})});
        const d = await r.json();
        if (!r.ok || d.error) throw new Error(d.error || 'Error');
        Toast.success(`${plural(d.reverted, 'segment')} reset`
            + (d.skipped ? `, ${d.skipped} manually changed → left as is` : ''));
        _corrLoadState();
        if (typeof loadTranscript === 'function') loadTranscript(_corrFilename);
    } catch (e) { Toast.error(e.message); }
}

// ══════════════════════════════════════════════════════════════
//  Translation modal
// ══════════════════════════════════════════════════════════════

// ── The translation's target language ────────────────────────────────
// Both translation dialogs opened on whatever stood first in their list —
// German, for everybody. They open on the language chosen last, else on the
// browser's when the list has it, else on English.
const _LLM_TARGETS = {en: 'English', de: 'German', ja: 'Japanese', fr: 'French', es: 'Spanish'};

function llmPresetTarget(selectId, freeId) {
    const sel = document.getElementById(selectId), free = document.getElementById(freeId);
    if (!sel) return;
    let last = '';
    try { last = localStorage.getItem('llmTargetLanguage') || ''; } catch (e) { /* private mode */ }
    const has = v => [...sel.options].some(o => o.value === v);
    const browser = _LLM_TARGETS[(navigator.language || '').slice(0, 2).toLowerCase()];
    if (last && has(last)) sel.value = last;
    else if (last && free) { sel.value = '_free'; free.value = last; }
    else sel.value = browser && has(browser) ? browser : 'English';
    sel.dispatchEvent(new Event('change'));
}

function llmRememberTarget(lang) {
    try { if (lang) localStorage.setItem('llmTargetLanguage', lang); } catch (e) { /* private mode */ }
}

function openTranslateModal(filename) {
    _transFilename = filename;
    _transJobId = null;
    document.getElementById('transProgress').style.display = 'none';
    document.getElementById('transResult').innerHTML = '';
    document.getElementById('transSteps').innerHTML = '';
    llmResetParts();
    document.getElementById('btnTransStart').disabled = false;
    document.getElementById('btnTransCancel').style.display = 'none';
    document.getElementById('translateModal').classList.add('visible');
    llmPresetTarget('transLanguage', 'transLanguageFree');
    _llmModels('transModel', 'translation');
    _transLoadList();
    llmQueueOpen('translation', filename).then(e => { if (e) _transTrack(e.id); });
}

function closeTranslateModal() {
    if (_transTracker) { _transTracker.stop(); _transTracker = null; }
    document.getElementById('translateModal').classList.remove('visible');
}

async function _transLoadList() {
    const el = document.getElementById('transList');
    el.innerHTML = '';
    try {
        const r = await fetch('/api/translations/' + encodeURIComponent(_transFilename));
        const d = await r.json();
        const tracks = d.translations || [];
        if (!tracks.length) {
            el.innerHTML = `<div class="llm-empty" style="padding:12px"><span class="large">🌐</span>No translations for this file yet.</div>`;
            return;
        }
        el.innerHTML = tracks.map(t => {
            const dt = t.created_at ? new Date(t.created_at * 1000).toLocaleDateString() : '';
            const active = _txView.mode === 'translation' && _txView.lang === t.language;
            return `<div class="llm-track ${active ? 'active' : ''}">
                <span class="language">${esc(t.language)}</span>
                <span class="llm-chip">${t.segments ?? '?'} segment${t.segments === 1 ? '' : 's'}</span>
                ${t.model ? `<span class="llm-chip">🏷 ${esc(t.model)}</span>` : ''}
                ${dt ? `<span class="llm-chip">${dt}</span>` : ''}
                <span style="flex:1"></span>
                <button class="tx-act-btn" onclick="closeTranslateModal();switchTxView('translation','${escJs(t.language)}')" title="Open this translation in the transcript view"><span class="tb-ico">📄</span>In transcript</button>
                <button class="tx-act-btn" onclick="transSrtExport('${escJs(t.language)}')" title="Download as SRT (timings from the original)">SRT</button>
                <button class="tx-act-btn tx-act-danger" onclick="transDelete('${escJs(t.language)}')">✕</button>
            </div>`;
        }).join('');
    } catch (e) {}
}

async function startTranslation() {
    let lang = document.getElementById('transLanguage').value;
    if (lang === '_free') lang = (document.getElementById('transLanguageFree').value || '').trim();
    if (!lang) { Toast.error('Select target language'); return; }
    llmRememberTarget(lang);
    try {
        const id = await llmQueueAdd('translation', _transFilename, {
            language: lang,
            model: document.getElementById('transModel')?.value || '',
        }, true);
        _transTrack(id);
    } catch (e) {
        const prog = document.getElementById('transProgress');
        prog.style.display = '';
        prog.querySelector('.llm-progress-text').textContent = '❌ ' + e.message;
    }
}

function _transTrack(id) {
    _transJobId = id;
    const prog = document.getElementById('transProgress');
    const bar = prog.querySelector('.llm-progress-fill');
    const text = prog.querySelector('.llm-progress-text');
    prog.style.display = '';
    bar.style.background = 'var(--accent)';
    bar.style.width = '3%';
    text.textContent = '';
    document.getElementById('btnTransStart').disabled = true;
    document.getElementById('btnTransCancel').style.display = '';
    document.getElementById('transResult').innerHTML = '';
    document.getElementById('transSteps').innerHTML = '';
    llmResetParts();
    if (_transTracker) _transTracker.stop();
    _transAbortPending = false;
    document.getElementById('btnTransCancel').disabled = false;
    _transTracker = llmQueueTrack(id, (e) => {
        bar.style.width = Math.max(3, e.progress || 0) + '%';
        text.innerHTML = _transAbortPending
            ? '⏹ Cancel requested — takes effect after the current section…'
            : (e.status === 'pending' ? llmWaitText(e) : esc(e.message || ''));
        const parts = e.partial_results || [];
        if (parts.length) {
            llmRenderParts(document.getElementById('transSteps'), parts, 'q' + id);
        }
    }, (e) => {
        _transCleanUp();
        const result = e.result || {};
        const chips = [
            `<span class="llm-chip green">✅ <b>${result.segments}/${result.total ?? '?'}</b> translated</span>`,
            result.duration ? `<span class="llm-chip">⏱ <b>${Math.round(result.duration)}s</b></span>` : '',
            `<span class="llm-chip">Chunks <b>${result.chunks}</b></span>`,
            result.missing ? `<span class="llm-chip orange">⚠ <b>${result.missing}</b> skipped (source hallucination)</span>` : '',
            result.chunks_failed ? `<span class="llm-chip red">⚠ ${plural(result.chunks_failed, 'chunk error')}</span>` : '',
        ].filter(Boolean).join(' ');
        document.getElementById('transResult').innerHTML =
            `<div style="display:flex;gap:6px;flex-wrap:wrap;align-items:center;margin:8px 0">${chips}
             <button class="tx-act-btn" onclick="closeTranslateModal();switchTxView('translation','${escJs(result.language || '')}')"><span class="tb-ico">📄</span>View in transcript</button></div>`;
        _transLoadList();
        _txViewsLoadMeta();
    }, (error) => {
        _transCleanUp(true);
        text.textContent = '❌ ' + error;
        bar.style.width = '100%';
        bar.style.background = 'var(--red)';
    });
}

function _transCleanUp(error) {
    document.getElementById('btnTransStart').disabled = false;
    document.getElementById('btnTransCancel').style.display = 'none';
    if (!error) document.getElementById('transProgress').style.display = 'none';
    if (_transTracker) { _transTracker.stop(); _transTracker = null; }
}

async function cancelTranslation() {
    if (!_transJobId) return;
    if (!await llmQueueCancel(_transJobId)) return;
    _transAbortPending = true;
    const text = document.querySelector('#transProgress .llm-progress-text');
    if (text) text.textContent = '⏹ Cancel requested — takes effect after the current section…';
    document.getElementById('btnTransCancel').disabled = true;
}

async function _fetchTranslationTrack(lang, filename) {
    const fn = filename || _transFilename || activeFile;
    const r = await fetch('/api/translation/' + encodeURIComponent(lang) + '/'
        + encodeURIComponent(fn));
    const d = await r.json();
    if (d.error) throw new Error(d.error);
    return d.segments || [];
}

function transSrtExport(lang) {
    _fetchTranslationTrack(lang).then(track => {
        const segs = activeTranscript?.segments || [];
        const map = {};
        track.forEach(t => { map[t.id] = t.text; });
        const fmtT = (s) => {
            const h = Math.floor(s / 3600), m = Math.floor(s % 3600 / 60),
                  sec = Math.floor(s % 60), ms = Math.round((s % 1) * 1000);
            return `${String(h).padStart(2,'0')}:${String(m).padStart(2,'0')}:${String(sec).padStart(2,'0')},${String(ms).padStart(3,'0')}`;
        };
        let n = 0;
        const srt = segs.map((s, i) => {
            const text = map[i];
            if (!text) return null;
            n += 1;
            return `${n}\n${fmtT(s.start || 0)} --> ${fmtT(s.end || 0)}\n${text}\n`;
        }).filter(Boolean).join('\n');
        const blob = new Blob([srt], {type: 'text/plain;charset=utf-8'});
        const a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        const base = ((_transFilename || activeFile) || 'transcript').split('/').pop().replace(/\.[^.]+$/, '');
        a.download = `${base}.${lang}.srt`;
        a.click();
        URL.revokeObjectURL(a.href);
    }).catch(e => Toast.error(e.message));
}

async function transDelete(lang) {
    try {
        await fetch('/api/translations/delete', {
            method: 'POST', headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({filename: _transFilename, language: lang})});
        if (_txView.mode === 'translation' && _txView.lang === lang)
            switchTxView('original');
        _transLoadList();
        _txViewsLoadMeta();
    } catch (e) {}
}

// ══════════════════════════════════════════════════════════════
//  Transcript view switcher (original / translation / summary)
//  Hooked in by fileview.renderTranscript() (txViewsInit).
// ══════════════════════════════════════════════════════════════

// ``withoutOriginal``: the file no longer has a transcript, only derivatives.
// Then there is no segment list for a translation to attach to —
// in that case the view builds it from the translation itself.
function txViewsInit(filename, withoutOriginal = false) {
    // New file render: reset view to original, reload meta
    _txView = {mode: 'original', lang: '', tracks: [], map: null,
               showOriginal: _txView.showOriginal, sumCount: 0, sumPending: false,
               withoutOriginal: !!withoutOriginal};
    clearTimeout(_sumPollTimer);
    _txViewsLoadMeta();
}

// A summary the queue writes after the transcript (the dialog's
// "summarize"): the transcript shows when it is done, the summary follows
// about a minute later. /api/summaries says "pending" until it is saved;
// the view shows it coming and asks again until it is there — otherwise
// the view never learns of it and keeps saying 0.
function _txSumPoll(fn) {
    clearTimeout(_sumPollTimer);
    _sumPollTimer = setTimeout(async () => {
        if ((activeFile || _corrFilename) !== fn) return;     // another file is open
        let sr;
        try {
            sr = await fetch('/api/summaries/' + encodeURIComponent(fn)).then(r => r.json());
        } catch (e) {
            _txSumPoll(fn);
            return;
        }
        if ((activeFile || _corrFilename) !== fn) return;
        const before = _txView.sumCount;
        _txView.sumCount = (sr.summaries || []).length;
        _txView.sumPending = !!sr.pending;
        if (_txView.sumPending) {
            _txSumPoll(fn);
        } else if (_txView.sumCount > before) {
            Toast.success('🤖 The summary is ready');
        }
        if (_txView.mode === 'summary') _sumViewRender();
        else _txViewbarRender();
    }, 4000);
}

async function _txViewsLoadMeta() {
    const fn = activeFile || _corrFilename;
    if (!fn) return;
    try {
        const [tr, sr] = await Promise.all([
            fetch('/api/translations/' + encodeURIComponent(fn)).then(r => r.json()),
            fetch('/api/summaries/' + encodeURIComponent(fn)).then(r => r.json()),
        ]);
        // Another file was opened meanwhile: this answer belongs to the old
        // one and must not put its counts on the new file's tabs.
        if ((activeFile || _corrFilename) !== fn) return;
        _txView.tracks = tr.translations || [];
        _txView.sumCount = (sr.summaries || []).length;
        _txView.sumPending = !!sr.pending;
        if (_txView.sumPending) _txSumPoll(fn);
    } catch (e) {
        _txView.tracks = [];
        _txView.sumCount = 0;
    }
    // Without an original, starting on the transcript tab would be an empty
    // page — open whatever is actually there right away instead.
    if (_txView.withoutOriginal && _txView.mode === 'original') {
        if (_txView.tracks.length) {
            switchTxView('translation');
            return;
        }
        if (_txView.sumCount) {
            switchTxView('summary');
            return;
        }
    }
    _txViewbarRender();
    if (_txView.withoutOriginal) _txApplyView();
}

function _txViewbarRender() {
    const bar = document.getElementById('txViewBar');
    if (!bar) return;
    // The recording's tabs carry the counts now (recording.js); a
    // Translation tab already open follows the tracks that just arrived
    if (typeof Rec !== 'undefined') {
        Rec.renderTabs();
        if (Rec.current === 'translation') Rec.tab('translation', {quiet: true});
    }
    const m = _txView.mode;
    const langOpts = _txView.tracks.map(t =>
        `<option value="${esc(t.language)}" ${t.language === _txView.lang ? 'selected' : ''}>${esc(t.language)} (${t.segments ?? '?'})</option>`).join('');
    bar.innerHTML = `
        <button class="tx-view-tab ${m==='original'?'active':''}" onclick="switchTxView('original')">📄 Transcript</button>
        ${_txView.tracks.length ? `
        <button class="tx-view-tab ${m==='translation'?'active':''}" onclick="switchTxView('translation')">🌐 Translation<span class="n">${_txView.tracks.length}</span></button>
        ${m==='translation' ? `<select onchange="switchTxView('translation', this.value)">${langOpts}</select>` : ''}` : ''}
        <button class="tx-view-tab ${m==='summary'?'active':''}" onclick="switchTxView('summary')"
            ${_txView.sumPending ? 'title="A summary is being written — it appears here when it is done"' : ''}>🤖 Summaries<span class="n">${_txView.sumCount}${_txView.sumPending ? ' · ⏳' : ''}</span></button>
        ${m==='translation' && !_txView.withoutOriginal ? `<label class="view-opt"><input type="checkbox" ${_txView.showOriginal?'checked':''}
            onchange="_txView.showOriginal=this.checked;_txApplyTranslation()"> Show original</label>` : ''}
    `;
}

// Every switch gets a number: a translation track that arrives after the
// user went on — to another file, another tab or another language — is
// dropped instead of landing on what is shown now.
let _txSwitchSeq = 0;

async function switchTxView(mode, lang) {
    const seq = ++_txSwitchSeq;
    if (mode === 'translation') {
        lang = lang || _txView.lang || (_txView.tracks[0]?.language || '');
        if (!lang) { Toast.error('No translation available'); return; }
        const fn = activeFile;
        const view = _txView;
        try {
            const track = await _fetchTranslationTrack(lang, fn);
            if (seq !== _txSwitchSeq || activeFile !== fn || _txView !== view) return;
            const map = {};
            track.forEach(t => { map[t.id] = t.text; });
            _txView.mode = 'translation';
            _txView.lang = lang;
            _txView.map = map;
        } catch (e) {
            if (seq === _txSwitchSeq && activeFile === fn) Toast.error(e.message);
            return;
        }
    } else {
        _txView.mode = mode;
        if (mode === 'original') _txView.map = null;
    }
    _txViewbarRender();
    _txApplyView();
}

function _txApplyView() {
    const segList = document.getElementById('segList');
    const sumView = document.getElementById('sumView');
    const m = _txView.mode;
    if (segList) segList.style.display = (m === 'summary') ? 'none' : '';
    if (sumView) sumView.style.display = (m === 'summary') ? '' : 'none';
    if (m === 'summary') { _sumViewRender(); }
    else if (m === 'translation') { _txApplyTranslation(); }
    else { _txApplyOriginal(); }
    // Player subtitles follow the view
    if (typeof applySubtitleState === 'function') applySubtitleState();
}

function _txApplyOriginal() {
    if (_txView.withoutOriginal) {
        const segList = document.getElementById('segList');
        if (segList) {
            segList.innerHTML = `<div class="empty-state"><div class="icon">📝</div>
                <p>The original transcript has been deleted.</p>
                <p style="font-size:0.8rem;color:var(--tx-3)">Translations and
                summaries are preserved — switch above.</p></div>`;
        }
        return;
    }
    document.querySelectorAll('#segList .seg').forEach((el, idx) => {
        const seg = activeTranscript?.segments?.[idx];
        if (seg) el.querySelector('.txt').innerHTML = buildWordHtml(seg);
    });
}

function _txApplyTranslation() {
    const map = _txView.map || {};
    // Without an original there are no segment nodes for the translation
    // to attach to — here it IS the content. It has no timestamps (those
    // only live in the original), so just the text.
    if (_txView.withoutOriginal) {
        const segList = document.getElementById('segList');
        if (!segList) return;
        const ids = Object.keys(map).map(Number).sort((a, b) => a - b);
        segList.innerHTML = ids.length
            ? ids.map(id => `<div class="seg"><div><div class="txt">${esc(map[id])}</div></div></div>`).join('')
            : `<div class="empty-state"><p>This translation is empty.</p></div>`;
        return;
    }
    document.querySelectorAll('#segList .seg').forEach((el, idx) => {
        const seg = activeTranscript?.segments?.[idx];
        if (!seg) return;
        const trans = map[idx];
        let html;
        if (trans !== undefined) {
            html = esc(trans);
        } else {
            html = `<span class="txt-missing">${esc(seg.text || '')}</span>`;
        }
        if (_txView.showOriginal && trans !== undefined) {
            html += `<div class="txt-orig">${esc(seg.text || '')}</div>`;
        }
        el.querySelector('.txt').innerHTML = html;
    });
}

// Player subtitles: fileview asks here for the view text
function llmSubtitleOverride(idx) {
    if (_txView.mode !== 'translation' || !_txView.map) return null;
    return _txView.map[idx] ?? '';
}

// ── Summary view ──────────────────────────────────

let _sumViewSeq = 0;

async function _sumViewRender() {
    const el = document.getElementById('sumView');
    if (!el) return;
    const fn = activeFile;
    const seq = ++_sumViewSeq;
    // "Loading…" only when the view shows nothing of this file yet: the
    // pending-summary poll refreshes every 4 s, and a placeholder each time
    // threw the reader of an older summary back to the top.
    if (el._sumFile !== fn) {
        el._sumFile = fn;
        el._sumHtml = null;
        el.innerHTML = '<div class="llm-empty">Loading…</div>';
    }
    let summaries = [];
    let pending = false;
    try {
        const r = await fetch('/api/summaries/' + encodeURIComponent(fn));
        const d = await r.json();
        summaries = d.summaries || [];
        pending = !!d.pending;
    } catch (e) {}
    // Another file (or a newer render) took over while this was loading:
    // its tabs and view must not get this file's summaries.
    if (seq !== _sumViewSeq || activeFile !== fn) return;
    _txView.sumPending = pending;
    if (pending) _txSumPoll(fn);
    _txView.sumCount = summaries.length;
    _txViewbarRender();
    const pendingNote = _txView.sumPending
        ? `<div class="llm-empty" style="padding:10px 0">⏳ A summary is being written — it appears
           here when it is done.</div>` : '';

    const header = `<div style="display:flex;gap:8px;align-items:center;margin-bottom:12px">
        <span style="font-size:0.72rem;text-transform:uppercase;letter-spacing:0.6px;color:var(--tx-3);font-weight:600">
            ${summaries.length} saved summar${summaries.length === 1 ? 'y' : 'ies'}</span>
        <span style="flex:1"></span>
        <button class="llm-btn-primary" onclick="openSummarizeModal('${escJs(fn)}')">🤖 New summary</button></div>`;

    // Unchanged → leave the DOM alone (scroll position, an armed 🗑).
    const show = (html) => {
        if (el._sumHtml === html) return;
        el._sumHtml = html;
        el.innerHTML = html;
    };
    if (!summaries.length && pendingNote) {
        show(`<div class="sum-view">${header}${pendingNote}</div>`);
        return;
    }
    if (!summaries.length) {
        show(`<div class="sum-view">${header}
            <div class="llm-empty"><span class="large">🤖</span>
            No summary saved yet.<br>
            Create one via the button above — it is saved in the database
            and appears here.</div></div>`);
        return;
    }
    // Newest first, as in the dialog's Saved tab — a new summary was at
    // the bottom of a long page. The index stays the one the server knows.
    show(`<div class="sum-view">${header}${pendingNote}` + summaries.map((s, i) => ({s, i}))
            .reverse().map(({s, i}) => {
        const dt = s.timestamp ? new Date(s.timestamp * 1000).toLocaleString() : '';
        return `<div class="sum-view-map">
            <div class="sum-view-head">
                ${dt ? `<span class="llm-chip">📅 ${dt}</span>` : ''}
                ${s.model ? `<span class="llm-chip">🏷 ${esc(s.model)}</span>` : ''}
                ${s.prompt ? `<span class="llm-chip" title="${esc(s.prompt)}">💬 ${esc(s.prompt.slice(0, 40))}${s.prompt.length > 40 ? '…' : ''}</span>` : ''}
                <span style="flex:1"></span>
                <button class="tx-act-btn tx-act-danger sum-del" style="height:22px;font-size:0.68rem"
                    title="Delete this summary" aria-label="Delete this summary"
                    onclick="_sumViewDeleteClick(this, ${i}, ${Number(s.timestamp) || 'null'})">🗑</button>
            </div>
            <div class="sum-view-content">${typeof _renderMarkdown === 'function' ? _renderMarkdown(s.text || '') : esc(s.text || '')}</div>
        </div>`;
    }).join('') + '</div>');
}

// Two clicks to delete — one stray click must not lose a summary
function _sumViewDeleteClick(btn, index, stamp) {
    if (!btn.classList.contains('armed')) {
        btn.classList.add('armed');
        btn.textContent = 'Delete?';
        setTimeout(() => {
            if (!btn.isConnected) return;
            btn.classList.remove('armed');
            btn.textContent = '🗑';
        }, 4000);
        return;
    }
    _sumViewDelete(index, stamp);
}

async function _sumViewDelete(index, stamp) {
    try {
        // The timestamp makes the server refuse if the list changed since
        // it was shown — an index alone would then hit another summary
        const r = await fetch('/api/summaries/delete', {
            method: 'POST', headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({filename: activeFile, index, timestamp: stamp})});
        const d = await r.json().catch(() => ({}));
        if (!r.ok || d.error) Toast.error('Not deleted: ' + (d.error || `HTTP ${r.status}`));
    } catch (e) {
        Toast.error('Not deleted: ' + e.message);
    }
    _sumViewRender();
    _txViewbarRender();
}

// ══════════════════════════════════════════════════════════════
//  Shared display: job progress with expandable partial results
//  (map sections, chunk results).
//  Used by the summarize dialog and the queue view.
// ══════════════════════════════════════════════════════════════

// Expand state lives HERE, not in the DOM: the view gets rebuilt on
// polling, and an "open" state set in the DOM would be gone after 700 ms.
// It is kept PER TASK
// (key 'q' + queue id): the summary dialog and several "Steps & progress"
// boxes of other tasks can be on screen at once, and opening section 3 of
// one must not open section 3 of all of them.
const _partsState = new Map();     // key → {panel: bool, open: Set<index>}

function _partsOf(key) {
    let st = _partsState.get(key);
    if (!st) { st = {panel: false, open: new Set()}; _partsState.set(key, st); }
    return st;
}

// Several boxes can be in the DOM at once (summary dialog, correction and
// translation steps, every open "Steps & progress" in the LLM queue), so
// the clicked header finds its own box — no ids, which would repeat.
function llmPartsPanelToggle(head, key = '') {
    const st = _partsOf(key);
    st.panel = !st.panel;
    head?.closest('.llm-parts')?.classList.toggle('open', st.panel);
}

function llmPartToggle(i, head, key = '') {
    const st = _partsOf(key);
    if (st.open.has(i)) st.open.delete(i); else st.open.add(i);
    head?.closest('.llm-part')?.classList.toggle('open', st.open.has(i));
}

/* Forget one task's expand state. A dialog starting a new task needs no
 * reset (the new task has its own key) — called without a key it leaves
 * the other tasks' boxes, e.g. in the LLM queue tab, as they are. */
function llmResetParts(key) {
    if (key !== undefined) _partsState.delete(key);
}

function llmPartsHtml(parts, key = '') {
    if (!parts || !parts.length) return '';
    const st = _partsOf(key);
    const k = escJs(key);
    const lines = parts.map((t, i) => `
        <div class="llm-part ${st.open.has(i) ? 'open' : ''}" data-part="${i}">
            <div class="llm-part-head" onclick="llmPartToggle(${i}, this, '${k}')">
                <span class="arrow">▸</span>
                <b>Section ${t.number ?? (i + 1)}${t.total ? '/' + t.total : ''}</b>
                ${t.from_value ? `<span class="llm-chip">${esc(t.from_value)}–${esc(t.to_value)}</span>` : ''}
                ${(t.from_seg != null) ? `<span class="llm-chip">Segments ${t.from_seg}–${t.to_seg}</span>` : ''}
                ${t.tokens ? `<span class="llm-chip">${t.tokens} tok</span>` : ''}
                ${t.duration ? `<span class="llm-chip">⏱ ${t.duration}s</span>` : ''}
                ${t.metric ? `<span class="llm-chip green">${esc(t.metric)}</span>` : ''}
                <span style="flex:1"></span>
                <span style="color:var(--tx-3)">${plural((t.text || '').length, 'character')}</span>
            </div>
            <div class="llm-part-text${t.markdown ? ' md' : ''}">${
                t.markdown && typeof _renderMarkdown === 'function'
                    ? _renderMarkdown(t.text || '') : esc(t.text || '')}</div>
        </div>`).join('');
    return `<div class="llm-parts ${st.panel ? 'open' : ''}">
        <div class="llm-parts-head" onclick="llmPartsPanelToggle(this, '${k}')">
            <span class="arrow">▸</span> View intermediate results (${parts.length})
        </div>
        <div class="llm-parts-list">${lines}</div>
    </div>`;
}

// Render the step log into an element WITHOUT losing the scroll position
// on each poll tick: the tick came in ~1×/s and rebuilt everything via
// innerHTML — anyone reading an open section kept getting thrown back to
// the top. So: don't assign unchanged
// HTML at all, and if it did change (new chunk done), save the scroll
// positions first.
function llmRenderParts(el, parts, key = '') {
    if (!el) return;
    const html = llmPartsHtml(parts, key);
    if (el._partsHtml === html) return;
    const scroll = {};
    el.querySelectorAll('.llm-part').forEach(t => {
        const content = t.querySelector('.llm-part-text');
        if (content && content.scrollTop) scroll[t.dataset.part] = content.scrollTop;
    });
    el._partsHtml = html;
    el.innerHTML = html;
    for (const [num, top] of Object.entries(scroll)) {
        const content = el.querySelector(`.llm-part[data-part="${num}"] .llm-part-text`);
        if (content) content.scrollTop = top;
    }
}

// ══════════════════════════════════════════════════════════════
//  LLM queue: attach to running tasks
// ══════════════════════════════════════════════════════════════

async function llmQueueAdd(kind, filename, params, front) {
    const r = await fetch('/api/llm/queue/add', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({kind, filename, params: params || {}, front: !!front})});
    const d = await r.json();
    if (!r.ok || d.error) throw new Error(d.error || 'Error enqueuing');
    return d.ids[0];
}

async function llmQueueOpen(kind, filename) {
    try {
        const r = await fetch('/api/llm/queue/open?kind=' + encodeURIComponent(kind)
            + '&filename=' + encodeURIComponent(filename));
        return (await r.json()).entry || null;
    } catch (e) { return null; }
}

// Polls a queue entry until it's done.
// onTick(entry) is called on every change.
let _lqLastStatus = '';

// "Waiting…" alone says nothing. This line says WHAT FOR: which slot,
// how many are ahead, and what's currently blocking the queue.
function llmWaitText(e) {
    const parts = ['⏳ Waiting in the LLM queue'];
    if (e.queue_slot) parts.push(`Slot ${e.queue_slot} of ${e.pending}`);
    if (e.currently_running)
        parts.push(`currently running: <b>${esc((e.currently_running || '').split('/').pop())}</b>`);
    return parts.join(' · ')
        + ` <button class="tx-act-btn" style="height:20px;font-size:0.66rem;margin-left:6px"
             onclick="openLlmQueue()">⚙ Queue</button>`;
}

function llmQueueTrack(id, onTick, onDone, onError) {
    let stopped = false;
    let failures = 0;
    const tick = async () => {
        if (stopped) return;
        let e;
        try {
            const r = await fetch('/api/llm/queue/item/' + id);
            // The entry is gone for good (deleted, cleared): that is final.
            if (r.status === 404) {
                if (stopped) return;
                let msg = 'Entry not found';
                try { msg = (await r.json()).error || msg; } catch (_) { /* not JSON */ }
                onError?.(msg);
                return;
            }
            // Anything else that fails — network down, a 502 page while
            // the server restarts, non-JSON — is passing: the queue lives
            // on the server and resumes the task by itself. Giving up here
            // used to re-enable "Summarize" and invite a duplicate run.
            if (!r.ok) throw new Error('HTTP ' + r.status);
            e = await r.json();
            // An entry's own `error` field belongs to a failed task
            // (status 'error', handled below), not to the request.
            if (!e || !e.status) throw new Error((e && e.error) || 'No status');
        } catch (err) {
            if (stopped) return;
            failures++;
            setTimeout(tick, Math.min(15000, 1800 * 2 ** Math.min(failures - 1, 4)));
            return;
        }
        // stop() may have come while the request was in flight — a closed
        // (or meanwhile reopened for another file) dialog must not get it.
        if (stopped) return;
        failures = 0;
        _lqLastStatus = e.status;
        onTick?.(e);
        if (e.status === 'done') { onDone?.(e); return; }
        if (e.status === 'error') { onError?.(e.error || 'Failed', e); return; }
        if (e.status === 'cancelled') { onError?.('Cancelled', e); return; }
        // If the job is running, check more often: the text should
        // visibly grow. If it's only waiting in the queue, sluggish
        // polling is enough.
        setTimeout(tick, _lqLastStatus === 'running' ? 700 : 1800);
    };
    tick();
    return {stop: () => { stopped = true; }};
}

// ══════════════════════════════════════════════════════════════
//  Folder batch: all transcripts of a folder at once
//  (summarize / translate / correct)
// ══════════════════════════════════════════════════════════════

function openFolderLlmModal(kind) {
    document.getElementById('folderLlmKind').value = kind || 'summary';
    document.getElementById('folderLlmPreview').innerHTML = '';
    document.getElementById('folderLlmModal').classList.add('visible');
    llmPresetTarget('folderLlmLanguage', 'folderLlmLanguageFree');
    _folderLlmTypeSwitch();
    _llmModels('folderLlmModel');
    _checkFolderLlm();
}

function closeFolderLlmModal() {
    document.getElementById('folderLlmModal').classList.remove('visible');
}

function _folderLlmTypeSwitch() {
    const kind = document.getElementById('folderLlmKind').value;
    document.getElementById('folderLlmLanguageWrap').style.display =
        kind === 'translation' ? '' : 'none';
    document.getElementById('folderLlmPresetWrap').style.display =
        kind === 'summary' ? '' : 'none';
    if (kind === 'summary') _folderLlmPresets();
    _checkFolderLlm();
}

function _folderLlmPresets() {
    const sel = document.getElementById('folderLlmPreset');
    if (!sel || sel.options.length > 1) return;
    fetch('/api/llm/status').then(r => r.json()).then(st => {
        Object.keys(st.presets || {}).forEach(n => {
            const o = document.createElement('option');
            o.value = n; o.textContent = n;
            sel.appendChild(o);
        });
    }).catch(() => {});
}

function _folderLlmParams() {
    const kind = document.getElementById('folderLlmKind').value;
    const p = {model: document.getElementById('folderLlmModel')?.value || ''};
    if (kind === 'translation') {
        let l = document.getElementById('folderLlmLanguage').value;
        if (l === '_free') l = (document.getElementById('folderLlmLanguageFree').value || '').trim();
        p.language = l;
        llmRememberTarget(l);
    }
    if (kind === 'summary') {
        p.preset = document.getElementById('folderLlmPreset')?.value || '';
        p.save = true;
    }
    return p;
}

async function _checkFolderLlm() {
    const box = document.getElementById('folderLlmPreview');
    box.innerHTML = '<span style="font-size:0.8rem;color:var(--tx-3)">Checking…</span>';
    try {
        const r = await fetch('/api/llm/queue/folder', {
            method: 'POST', headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                kind: document.getElementById('folderLlmKind').value,
                dir: document.getElementById('folderLlmSub')?.value || '',
                recursive: document.getElementById('folderLlmRecursive').checked,
                skip_existing: document.getElementById('folderLlmSkip').checked,
                params: _folderLlmParams(),
                dry_run_only: true})});
        const d = await r.json();
        if (d.error) throw new Error(d.error);
        document.getElementById('btnFolderLlmStart').disabled = !d.to_enqueue;
        box.innerHTML = `
            <div style="display:flex;gap:6px;flex-wrap:wrap;align-items:center">
                <span class="llm-chip">${plural(d.found, 'transcript')} in the folder</span>
                <span class="llm-chip ${d.to_enqueue ? 'green' : ''}">${d.to_enqueue} will be enqueued</span>
                ${d.skipped ? `<span class="llm-chip orange">${d.skipped} skipped (already present)</span>` : ''}
                ${d.empty ? `<span class="llm-chip" title="Silence, music or a tone — nothing to work on">${d.empty} without speech, left out</span>` : ''}
            </div>
            ${d.examples?.length ? `<div style="font-size:0.74rem;color:var(--tx-3);margin-top:6px">
                e.g. ${d.examples.map(esc).join(' · ')}${d.to_enqueue > d.examples.length ? ' …' : ''}</div>` : ''}`;
    } catch (e) {
        box.innerHTML = `<span style="color:var(--red);font-size:0.8rem">❌ ${esc(e.message)}</span>`;
    }
}

async function startFolderLlm() {
    const btn = document.getElementById('btnFolderLlmStart');
    btn.disabled = true;
    try {
        const r = await fetch('/api/llm/queue/folder', {
            method: 'POST', headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                kind: document.getElementById('folderLlmKind').value,
                dir: document.getElementById('folderLlmSub')?.value || '',
                recursive: document.getElementById('folderLlmRecursive').checked,
                skip_existing: document.getElementById('folderLlmSkip').checked,
                params: _folderLlmParams()})});
        const d = await r.json();
        if (!r.ok || d.error) throw new Error(d.error || 'Error');
        Toast.success(`${plural(d.added, String(d.label || '').toLowerCase() + ' task')} enqueued`);
        openLlmQueue();
    } catch (e) {
        Toast.error(e.message);
        btn.disabled = false;
    }
}

// ══════════════════════════════════════════════════════════════
//  LLM queue view
// ══════════════════════════════════════════════════════════════

let _lqES = null;
let _lqBox = null;   // lives across renderFileList() rebuilds — see lqAttach

// The LLM queue is a sidebar tab next to ⏳ Queue (it used to be a modal):
// switching between the transcription queue and the LLM queue is one click
// in the same place.
function openLlmQueue() {
    // Called from inside the LLM dialogs ("⚙ Queue") too: they cover the
    // sidebar, so close whichever is open — the task keeps running server-side.
    for (const [id, close] of [['folderLlmModal', closeFolderLlmModal],
                               ['correctionModal', closeCorrectionModal],
                               ['translateModal', closeTranslateModal],
                               ['summarizeModal', window.closeSummarizeModal]]) {
        if (document.getElementById(id)?.classList.contains('visible') && close) close();
    }
    // The AI jobs of the Queue place
    if (typeof QueuePage !== 'undefined') QueuePage.open('llm');
    else setFileTab('llm', document.getElementById('tabLlmQueueBtn'));
}

// Called by renderFileList() while the LLM tab is active. The same element
// is re-attached every time — renderFileList runs on every queue tick, and
// a fresh element would collapse open detail boxes and lose their scroll.
function lqAttach(list) {
    if (!_lqBox) {
        _lqBox = document.createElement('div');
        _lqBox.id = 'llmQueueBody';
    }
    list.appendChild(_lqBox);
    if (!_lqES) {
        _lqRender();
        _lqES = new EventSource('/api/llm/queue/stream');
        _lqES.onmessage = (ev) => {
            try { _lqRender(JSON.parse(ev.data)); } catch (e) {}
        };
    }
}

// Leaving the tab: stop streaming (the badge keeps polling on its own).
function lqDetach() {
    if (_lqES) { _lqES.close(); _lqES = null; }
}

const LQ_KIND = { summary: 'Summary', translation: 'Translation', correction: 'Correction', analysis: 'Analysis',
                  deep_search: 'Deep search', media_describe: 'Describe pictures', best_moments: 'Best moments' };
// Row class (status strip color, shared with file rows) and status text.
const LQ_STATUS = {
    running:   { text: 'running',   cls: 'processing', stat: 's-run' },
    pending:   { text: 'waiting',   cls: 'queued',     stat: '' },
    done:      { text: 'done',      cls: 'done',       stat: 's-ok' },
    error:     { text: 'failed',    cls: 'qfail',      stat: 's-err' },
    cancelled: { text: 'cancelled', cls: 'qcancel',    stat: 's-cancel' },
};

async function _lqRender(snap) {
    if (!snap) {
        try { snap = await (await fetch('/api/llm/queue')).json(); }
        catch (e) { return; }
    }
    const box = _lqBox;
    if (!box) return;
    const c = snap.counts || {};
    // Same header as the transcription queue (listHead in filelist.js).
    const sum = [];
    if (c.running) sum.push(`<span class="lh-run">${c.running} running</span>`);
    sum.push(`${c.pending || 0} waiting`, `${c.done || 0} done`);
    if (c.error) sum.push(`<span class="lh-err">${c.error} failed</span>`);
    if (c.cancelled) sum.push(`${c.cancelled} cancelled`);
    const finished = (c.done || 0) + (c.error || 0) + (c.cancelled || 0);
    const tail = (finished ? lhBtn('Remove finished', "_lqClear('done')",
                                   'Remove done, failed and cancelled entries from the list', 'link') : '')
        + (c.pending ? lhBtn('Clear waiting', "_lqClear('pending')", 'Remove all waiting tasks', 'link danger') : '');
    // The language model didn't answer: the tasks wait for it (next try
    // after 30 s, then longer — up to 10 min); "Try now" brings it forward
    const w = snap.waiting;
    if (w) {
        const next = w.next_in >= 60 ? `${Math.round(w.next_in / 60)} min` : `${w.next_in} s`;
        sum.unshift(`<span class="lh-err" title="${esc(w.reason || '')}">no answer from the language model — next try in ${next}</span>`);
    }
    const header = listHead(
        `<span title="LLM tasks run here one after another — server-side, independent of the browser. A restart resumes interrupted tasks.">LLM queue</span>${snap.paused ? ' · ⏸ paused' : ''}${w ? ' · <span title="Waiting for the language model">⏳</span>' : ''}`,
        sum.join(' · ') + (tail ? `<span class="lh-tail">${tail}</span>` : ''),
        (w && !snap.paused ? lhBtn('↻ Try now', '_lqPause(false)', 'Try the language model again now') : '')
        + lhBtn(snap.paused ? '▶ Resume' : '⏸ Pause', `_lqPause(${!snap.paused})`,
              snap.paused ? 'Resume the LLM queue' : 'Pause after the running task')
        + lhBtn('📁 Folder', "openFolderLlmModal('summary')",
                'Summarize, translate, or correct all transcripts of a folder'));

    const line = (i) => {
        const st = LQ_STATUS[i.status] || { text: i.status, cls: '', stat: '' };
        const p = i.params || {};
        // The stored message is only meaningful while running (live step
        // text); for finished entries it can be stale — older rows still
        // carry the German wording of earlier versions — so the status wins.
        const statusText = i.status === 'running' && i.message ? i.message : st.text;
        const stats = [LQ_KIND[i.kind] || i.kind, p.preset, p.language && `→ ${p.language}`, p.model]
            .filter(Boolean).map(x => `<span class="fi-stat">${esc(x)}</span>`).join('')
            + `<span class="fi-stat ${st.stat}">${esc(statusText)}</span>`;
        const extra =
            (i.status === 'running' ? `<div class="fi-progress"><div style="width:${Math.max(3, i.progress || 0)}%"></div></div>` : '')
            // An analysis has a view of its own: its answer, notes and passages
            + (i.kind === 'analysis' && p.analysis_id
                ? `<button class="fi-link" onclick="event.stopPropagation();openAnalysis('${escJs(p.analysis_id)}')">↗ Open the analysis</button>` : '')
            // A deep search too: its plan, progress and confirmed passages
            + (i.kind === 'deep_search' && p.deep_id
                ? `<button class="fi-link" onclick="event.stopPropagation();openDeepSearch('${escJs(p.deep_id)}')">↗ Open the deep search</button>` : '')
            + (i.error ? `<div class="fi-error" title="${esc(i.error)}">${esc(i.error.slice(0, 200))}</div>` : '')
            + ((i.status === 'running' || i.status === 'done')
                ? `<button class="fi-link" onclick="event.stopPropagation();lqDetails(${i.id})">${_lqOpen.has(i.id) ? '▾' : '▸'} Steps &amp; progress</button>
                   <div id="lqDetail${i.id}" class="fi-detail"></div>` : '');
        const actions = (i.status === 'pending' || i.status === 'running')
            ? `<button class="del" onclick="event.stopPropagation();_lqCancel(${i.id})" title="Cancel">✕</button>`
            : ((i.status === 'error' || i.status === 'cancelled')   // server retries both
                ? `<button onclick="event.stopPropagation();_lqRetry(${i.id})" title="Run again">↻</button>` : '');
        return `<div class="file-item lq-item ${st.cls}">${workRowHtml({ path: i.filename, stats, extra, actions })}</div>`;
    };

    const items = snap.items || [];
    const html = header + (items.length
        ? items.map(line).join('')
        : listEmpty('The LLM queue is empty'));
    // The SSE stream ticks ~1×/s for running jobs. An unchanged rebuild
    // would collapse open detail boxes and discard scroll positions —
    // so: only touch it on a real change, restore open details afterward
    // (including scroll, then reload fresh).
    if (html === box._lastHtml) return;
    box._lastHtml = html;
    const details = {};
    _lqOpen.forEach(id => {
        const d = document.getElementById('lqDetail' + id);
        if (!d) return;
        const scroll = {};
        d.querySelectorAll('.llm-part-text, .llm-sum-result').forEach((el, k) => {
            if (el.scrollTop) scroll[k] = el.scrollTop;
        });
        details[id] = {html: d.innerHTML, scroll};
    });
    box.innerHTML = html;
    for (const [id, old_value] of Object.entries(details)) {
        const d = document.getElementById('lqDetail' + id);
        if (!d) { _lqOpen.delete(Number(id)); continue; }
        d.innerHTML = old_value.html;
        d.querySelectorAll('.llm-part-text, .llm-sum-result').forEach((el, k) => {
            if (old_value.scroll[k]) el.scrollTop = old_value.scroll[k];
        });
        _lqLoadDetail(Number(id));   // pull fresh data
    }
}

// One POST to the LLM queue's control endpoints. Returns the answer, or
// null after saying why — a failed cancel must not look like a done one
// (the button showed "Cancel requested" while the task ran on).
async function _lqPost(path, body, what) {
    try {
        const r = await fetch('/api/llm/queue/' + path, {method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(body)});
        let d = {};
        try { d = await r.json(); } catch (e) { /* not JSON (proxy page) */ }
        if (!r.ok || d.error) throw new Error(d.error || ('HTTP ' + r.status));
        return d;
    } catch (e) {
        if (typeof Toast !== 'undefined') Toast.error(`${what} failed: ${e.message}`);
        return null;
    }
}

// Cancel a task; true when the server took it. `ok: false` means the
// entry was no longer waiting or running — there is nothing to cancel.
async function llmQueueCancel(id) {
    const d = await _lqPost('cancel', {id}, 'Cancel');
    if (!d) return false;
    if (d.ok === false) {
        if (typeof Toast !== 'undefined') Toast.info('Nothing to cancel — the task has already ended');
        return false;
    }
    return true;
}

async function _lqCancel(id) {
    if (await llmQueueCancel(id) && typeof Toast !== 'undefined') {
        Toast.info('Cancel requested — a running section still finishes');
    }
    _lqRender();
}
async function _lqRetry(id) {
    const d = await _lqPost('retry', {id}, 'Retry');
    if (d && d.ok === false && typeof Toast !== 'undefined') {
        Toast.info('This task cannot be run again');
    }
    _lqRender();
}
async function _lqClear(what) {
    // Waiting tasks are work not done yet (a folder batch can be thousands)
    // — ask, like "Clear" on the transcription queue. Finished ones are
    // just list entries; their results stay.
    if (what === 'pending') {
        let n = 0;
        try { n = ((await (await fetch('/api/llm/queue')).json()).counts || {}).pending || 0; }
        catch (e) { /* ask without a number */ }
        if (!confirm(`Remove ${n ? n.toLocaleString() + ' ' : 'all '}waiting LLM `
                     + `task${n === 1 ? '' : 's'}?`)) return;
    }
    await _lqPost('clear', {what}, 'Clearing the LLM queue');
    _lqRender();
}
async function _lqPause(on) {
    await _lqPost('pause', {pause: on}, on ? 'Pausing the LLM queue' : 'Resuming the LLM queue');
    _lqRender();
}

// Counter on the 🤖 LLM chip next to the queue tab: running + waiting tasks
let llmQueueSnap = null;
async function llmQueueRefreshBadge() {
    try {
        const d = await (await fetch('/api/llm/queue')).json();
        llmQueueSnap = d;   // the Queue page's AI jobs read this answer
        if (typeof QueuePage !== 'undefined') QueuePage.llmChanged();
        const openJobs = (d.counts?.running || 0) + (d.counts?.pending || 0);
        const el = document.getElementById('llmQueueBadge');
        if (!el) return;
        el.textContent = openJobs;
        // The rail's Analyses badge: analysis jobs running or waiting
        el.dataset.analyses = (d.items || []).filter(i => i.kind === 'analysis'
            && (i.status === 'running' || i.status === 'pending')).length;
        // If something is running, the chip glows like an active tab
        document.getElementById('tabLlmQueueBtn')
            ?.classList.toggle('llm-active', openJobs > 0);
    } catch (e) {}
}
setInterval(llmQueueRefreshBadge, 5000);
document.addEventListener('DOMContentLoaded', llmQueueRefreshBadge);

// Reload and display steps + running text of a queue entry. This lets
// you follow along even without the file dialog. Which details are open
// lives in _lqOpen — the SSE rebuild of the list restores them from it.
const _lqOpen = new Set();

async function lqDetails(id) {
    const box = document.getElementById('lqDetail' + id);
    if (!box) return;
    const toggle = box.previousElementSibling;
    if (_lqOpen.has(id)) {
        _lqOpen.delete(id);
        box.innerHTML = ''; box._detailHtml = '';
        if (toggle) toggle.innerHTML = '▸ Steps &amp; progress';
        return;
    }
    _lqOpen.add(id);
    if (toggle) toggle.innerHTML = '▾ Steps &amp; progress';
    box.innerHTML = '<span style="font-size:0.75rem;color:var(--tx-3)">Loading…</span>';
    await _lqLoadDetail(id);
}

async function _lqLoadDetail(id) {
    const box = document.getElementById('lqDetail' + id);
    if (!box || !_lqOpen.has(id)) return;
    try {
        const e = await (await fetch('/api/llm/queue/item/' + id)).json();
        const parts = e.partial_results || [];
        const html =
            (e.partial_text
                ? `<div class="llm-sum-result" style="max-height:26vh;margin-top:6px">
                     ${typeof _renderMarkdown === 'function' ? _renderMarkdown(e.partial_text) : esc(e.partial_text)}
                     ${e.status === 'running' ? '<span class="sum-cursor">▌</span>' : ''}</div>`
                : '')
            // Short transcripts run in one pass: no parts, just the text
            // above — "no intermediate results yet" would then read as if
            // something were still missing.
            + (parts.length ? llmPartsHtml(parts, 'q' + id)
               : e.partial_text ? ''
               : `<div style="font-size:0.75rem;color:var(--tx-3);padding:6px">${
                     ['pending', 'running'].includes(e.status) ? 'No intermediate results yet.'
                                                               : 'No text was kept for this task.'}</div>`);
        if (html === box._detailHtml) return;
        box._detailHtml = html;
        // Save scroll positions (live text + open sections) across the
        // rebuild — otherwise the reading position jumps up ~1×/s.
        const scroll = {};
        box.querySelectorAll('.llm-part-text, .llm-sum-result').forEach((el, k) => {
            if (el.scrollTop) scroll[k] = el.scrollTop;
        });
        box.innerHTML = html;
        box.querySelectorAll('.llm-part-text, .llm-sum-result').forEach((el, k) => {
            if (scroll[k]) el.scrollTop = scroll[k];
        });
    } catch (err) {
        box.innerHTML = `<span style="color:var(--red);font-size:0.75rem">${esc(err.message)}</span>`;
    }
}
