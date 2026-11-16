// ══════════════════════════════════════════════════════════════════════════
//  Settings — the live parts inside the pages
// ══════════════════════════════════════════════════════════════════════════
//
// Status cards, the provider cards, template and vocabulary lists, the
// embedding server, index runs, backups, logs, the database explorer. Each
// draws into its own container of a page (settings_pages.js) and redraws
// only that container — never the page: a late answer must not throw away
// what someone typed in the meantime.

// Stop what runs in the background while the settings are closed
function _settingsToolsStop() {
    _ragStopPolling();
    _artStopPolling();
    if (typeof _mediaStopPolling === 'function') _mediaStopPolling();
}

// ── Folders: where each folder resolves to, and its state ────────────────

async function _sDirStates() {
    let info = {};
    try { info = await fetch('/api/settings/directories').then(r => r.json()); } catch (e) { return; }
    document.querySelectorAll('#settingsOverlay [data-path-state]').forEach(el => {
        const i = info[el.dataset.pathState];
        if (!i) { el.innerHTML = ''; return; }
        // Where the field resolves to — only worth a line when it isn't
        // what the field already says (empty = automatic, ~, a relative path)
        const typed = el.closest('.set-ctl')?.querySelector('input[data-key]')?.value.trim();
        const bits = el.dataset.noPath !== undefined || typed === i.path ? [] : [`→ <span class="mono">${esc(i.path)}</span>`];
        if (i.exists === false) bits.push('<span class="err">✕ doesn\'t exist</span>');
        else if (i.writable === false) bits.push('<span class="warn">⚠ read-only</span>');
        else if (i.exists) bits.push('<span class="ok">✓ writable</span>');
        if (i.disk_free_gb !== undefined) {
            const cls = i.disk_free_gb < 5 ? 'err' : i.disk_free_gb < 20 ? 'warn' : '';
            bits.push(`<span class="${cls}">${i.disk_free_gb} GB free of ${i.disk_total_gb} GB</span>`);
        }
        el.innerHTML = bits.join(' · ');
    });
}

// ── This machine (GPU & queue) ───────────────────────────────────────────

function _hwInfoRender() {
    const box = document.getElementById('hwInfo');
    const hw = _hwData;
    if (!box || !hw) return;
    const gpus = (hw.gpus || []).map(g => {
        const mem = g.memory_mb ? `${Math.round(g.memory_mb / 1024)} GB VRAM`
            : g.vendor === 'amd' && !hw.rocm_version ? 'not used — no ROCm (Setup guide → AMD GPU)'
            : 'VRAM not reported';
        return `<span>GPU ${g.index ?? ''}</span><span>${esc(g.name)} · ${mem}${g.driver ? ` · driver ${esc(g.driver)}` : ''}</span>`;
    }).join('');
    const tool = (ok, name, why) => `<span class="set-badge ${ok ? 'ok' : 'miss'}" title="${esc(why)}">${ok ? '✓' : '✕'} ${esc(name)}</span>`;
    box.innerHTML = `
        <div class="set-lines">
            <span>System</span><span>${esc(hw.os || '?')} ${esc(hw.arch || '')}</span>
            <span>CPU</span><span>${esc(hw.cpu || '?')} · ${hw.cpu_cores || '?'} core${hw.cpu_cores === 1 ? '' : 's'} / ${hw.cpu_threads || '?'} thread${hw.cpu_threads === 1 ? '' : 's'}</span>
            <span>RAM</span><span>${Math.round((hw.ram_total_mb || 0) / 1024)} GB</span>
            ${gpus || '<span>GPU</span><span>none found</span>'}
            <span>Runs on</span><span>${esc(String(hw.device || '?').toUpperCase())}${hw.cuda_version ? ` · CUDA ${esc(hw.cuda_version)}` : ''}${hw.rocm_version ? ` · ROCm ${esc(hw.rocm_version)}` : ''}</span>
        </div>
        <div class="set-badges" style="margin-top:12px">
            ${tool(hw.has_ffmpeg, 'FFmpeg' + (hw.ffmpeg_version ? ' ' + hw.ffmpeg_version : ''), 'Reads every audio and video format')}
            ${tool(hw.has_audio_separator, 'audio-separator', 'Vocal isolation')}
            ${tool(hw.has_ytdlp, 'yt-dlp', 'Link downloads')}
            ${tool(hw.has_fastmcp, 'fastmcp', 'MCP server')}
        </div>
        <div class="set-hint" style="margin-top:8px">As found when the server started — the load meters in the top bar are live.</div>`;
}

// ── Voice store (Speakers) ───────────────────────────────────────────────

async function _speakerStoreLoad() {
    const box = document.getElementById('speakerStore');
    if (!box) return;
    let S;
    try { S = await fetch('/api/speaker/statistics').then(r => r.json()); } catch (e) {
        box.innerHTML = `<div class="set-note err">Voice store not readable: ${esc(e.message)}</div>`;
        return;
    }
    if (!S || !S.path) { box.innerHTML = '<div class="set-hint">No voice store yet — it is created with the first voice vectors.</div>'; return; }
    const b = S.operating_point || {};
    const mb = (S.size_bytes || 0) / 1e6;
    const kpi = (label, value, sub = '') => `<div class="set-kpi"><div class="set-kpi-label">${label}</div>
        <div class="set-kpi-value">${value}</div>${sub ? `<div class="set-kpi-sub">${sub}</div>` : ''}</div>`;
    box.innerHTML = `
        <div class="set-kpis">
            ${kpi('Voice windows', (S.windows || 0).toLocaleString(), `from ${plural(S.files, 'file')}`)}
            ${kpi('Speech', `${(S.speech_hours || 0).toLocaleString()} h`)}
            ${kpi('Named people', (S.people || 0).toLocaleString(), plural(S.named_voices, 'voice'))}
            ${kpi('Size', `${mb.toFixed(mb < 10 ? 1 : 0)} MB`)}
        </div>
        ${S.revisions_mixed ? `<div class="set-note warn" style="margin-top:10px">⚠ Mixed versions: ${S.revision_separated} separated, ${plural(S.revision_raw, 'raw vector')}. They don't compare — the store splits into two halves that never find each other.</div>` : ''}
        <div class="set-lines" style="margin-top:12px">
            <span>File</span><span class="mono">${esc(S.path)}</span>
            <span>Model</span><span>${esc(b.model || '?')} — stored with every vector; a different model makes old vectors incomparable</span>
            <span>Window</span><span>${b.window_length_sec || '?'} s, speech of at least ${b.min_segment_sec || '?'} s per segment — measured, fixed</span>
        </div>`;
}

// ── Downloads: YouTube cookies (own endpoints — the content is a file
//    with credentials and doesn't belong in config.yaml) ──────────────────

async function _ytCookiesLoad() {
    const box = document.getElementById('ytCookieBox');
    if (!box) return;
    let d = {};
    try { d = await fetch('/api/youtube/cookies').then(r => r.json()); } catch (e) { /* shown below */ }
    const state = d.exists
        ? `<span class="set-dot ok"></span>Cookies stored — ${plural(d.lines, 'entry', 'entries')}, from ${new Date(d.changed * 1000).toLocaleString()}`
        : '<span class="set-dot"></span>No cookies — downloads run anonymously.';
    box.innerHTML = `
        <div class="set-result">${state}</div>
        <textarea class="set-textarea mono" id="ytCookieText" rows="5" style="margin-top:8px" spellcheck="false"
                  placeholder="# Netscape HTTP Cookie File&#10;.youtube.com&#9;TRUE&#9;/&#9;TRUE&#9;…"></textarea>
        <div class="set-actions" style="margin-top:8px">
            <button type="button" class="set-btn" onclick="_ytCookiesSave()">💾 Store cookies</button>
            ${d.exists ? '<button type="button" class="set-btn danger" onclick="_ytCookiesDelete()">🗑 Delete</button>' : ''}
        </div>
        <div class="set-hint" style="margin-top:6px">Paste the content of a cookies.txt (Netscape format), e.g. exported with the “Get cookies.txt” browser extension. Stored on the server, readable only by the server user.</div>`;
}

async function _ytCookiesSave() {
    const text = document.getElementById('ytCookieText')?.value.trim();
    if (!text) { _showToast('Paste the content of a cookies.txt first', true); return; }
    try {
        const r = await fetch('/api/youtube/cookies', {
            method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({text})});
        const d = await r.json().catch(() => ({}));
        if (!r.ok || d.error) throw new Error(d.error || `HTTP ${r.status}`);
        _showToast('Cookies stored');
    } catch (e) { _showToast('Not stored: ' + e.message, true); }
    _ytCookiesLoad();
}

async function _ytCookiesDelete() {
    if (!confirm('Delete the stored YouTube cookies?')) return;
    try {
        const r = await fetch('/api/youtube/cookies', {method: 'DELETE'});
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        _showToast('Cookies deleted');
    } catch (e) { _showToast('Not deleted: ' + e.message, true); }
    _ytCookiesLoad();
}

// ── Vocabularies (Whisper initial_prompt) ────────────────────────────────
//
// Kept in _settingsData and sent as a whole by Save. The counter uses the
// server's estimate (CJK ~1.5, Latin ~4 characters per token) against
// Whisper's budget of ~224 tokens.
const WHISPER_PROMPT_BUDGET = 224;

function _vocabTokens(text) {
    let cjk = 0;
    for (const ch of (text || '')) {
        const cp = ch.codePointAt(0);
        if ((cp >= 0x4E00 && cp <= 0x9FFF) || (cp >= 0x3400 && cp <= 0x4DBF)
            || (cp >= 0x3040 && cp <= 0x30FF) || (cp >= 0xAC00 && cp <= 0xD7AF)
            || (cp >= 0xFF00 && cp <= 0xFFEF)) cjk++;
    }
    const remainder = (text || '').length - cjk;
    return Math.max(text ? 1 : 0, Math.floor(cjk / 1.5 + remainder / 4));
}

function _vocabBudget(text) {
    const n = _vocabTokens(text);
    const over = n > WHISPER_PROMPT_BUDGET;
    return {text: `~${n} / ${WHISPER_PROMPT_BUDGET} tokens` + (over ? ' — too long, only the end counts' : ''), over};
}

function _renderVocabPresets(t) {
    const list = document.getElementById('vocabPresetsList');
    if (!list) return;
    const presets = t.vocab_presets || {};
    const names = Object.keys(presets);
    if (!names.length) {
        list.innerHTML = '<div class="set-hint">No vocabulary yet.</div>';
        return;
    }
    list.innerHTML = names.map((name, i) => {
        const b = _vocabBudget(presets[name]);
        return `<div class="set-item" data-vocab="${esc(name)}">
            <div class="set-item-head">
                <span class="set-item-name">${esc(name)}</span>
                <span class="set-item-meta${b.over ? ' over' : ''}" id="vocabTok${i}">${esc(b.text)}</span>
                <button type="button" class="set-btn sm danger" onclick="_removeVocabPreset(this)" title="Delete this vocabulary">✕</button>
            </div>
            <textarea class="set-textarea" rows="3" spellcheck="false"
                oninput="_updateVocabPreset(this, ${i})">${esc(presets[name])}</textarea>
        </div>`;
    }).join('');
}

function _vocabStore() {
    const cfg = _settingsData.config = _settingsData.config || {};
    const t = cfg.transcription = cfg.transcription || {};
    t.vocab_presets = t.vocab_presets || {};
    return t;
}

function _addVocabPreset() {
    const input = document.getElementById('vocabNewPresetName');
    const name = input?.value.trim();
    if (!name) return;
    const t = _vocabStore();
    if (name in t.vocab_presets) { _showToast('A vocabulary with that name exists', true); return; }
    t.vocab_presets[name] = '';
    input.value = '';
    _markDirty('transcription.vocab_presets');
    _renderVocabPresets(t);
}

function _removeVocabPreset(btn) {
    const name = btn.closest('[data-vocab]')?.dataset.vocab;
    const t = _vocabStore();
    if (name === undefined || !(name in t.vocab_presets)) return;
    if (t.vocab_presets[name].trim() && !confirm(`Delete the vocabulary “${name}”?`)) return;
    delete t.vocab_presets[name];
    _markDirty('transcription.vocab_presets');
    _renderVocabPresets(t);
}

function _updateVocabPreset(area, i) {
    const name = area.closest('[data-vocab]')?.dataset.vocab;
    const t = _vocabStore();
    if (name === undefined) return;
    t.vocab_presets[name] = area.value;
    const tok = document.getElementById('vocabTok' + i);
    if (tok) {
        const b = _vocabBudget(area.value);
        tok.textContent = b.text;
        tok.classList.toggle('over', b.over);
    }
    _markDirty('transcription.vocab_presets');
}

// ── Clean-up: trying the artifact patterns ───────────────────────────────
//
// The lines as typed, not the saved ones — on the server, so the answer is
// Python's regular expressions, the ones the filter runs, not the browser's.

let _artifactTimer = null;
let _artifactSeq = 0;

function _artifactTry() {
    clearTimeout(_artifactTimer);
    _artifactTimer = setTimeout(_artifactTryNow, 250);
}

async function _artifactTryNow() {
    const box = document.getElementById('artifactTryResult');
    const area = document.getElementById('artifactPatterns');
    if (!box || !area) return;
    const text = document.getElementById('artifactTryText')?.value || '';
    const seq = ++_artifactSeq;
    let d;
    try {
        const r = await fetch('/api/settings/artifact_test', {
            method: 'POST', headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({patterns: area.value.split('\n'), text})});
        d = await r.json();
    } catch (e) { return; }
    if (seq !== _artifactSeq) return;       // a newer try is on its way
    const lines = [];
    for (const e of d.errors || []) lines.push(`<div class="set-result err">✕ Not a valid pattern — ${esc(e)}</div>`);
    if (text.trim()) {
        lines.push(d.dropped
            ? `<div class="set-result warn">✕ Dropped — ${d.reason === 'no letter or digit' ? 'it holds no letter or digit'
                                                         : `matches ${_sCode(d.reason)}`}</div>`
            : '<div class="set-result ok">✓ Kept — no pattern matches the whole text</div>');
    }
    box.innerHTML = lines.join('');
}

// Typing in the pattern field tries again, like typing in the try field
document.addEventListener('input', e => {
    if (e.target?.id === 'artifactPatterns') _artifactTry();
});

// ── Clean-up: the patterns on the existing transcripts ───────────────────
//
// Dry run → remove what it found → undo; each a background run on the
// server (core/artifact_cleanup.py) that this polls while it is going.

let _artPoll = null;

function _artStopPolling() {
    if (_artPoll) { clearInterval(_artPoll); _artPoll = null; }
}

async function _artStatusLoad() {
    let s;
    try { s = await fetch('/api/artifacts/status').then(r => r.json()); } catch (e) { return; }
    _artRender(s);
    if (s.running && !_artPoll) _artPoll = setInterval(_artTick, 1000);
}

async function _artTick() {
    let s;
    try { s = await fetch('/api/artifacts/status').then(r => r.json()); } catch (e) { return; }
    _artRender(s);
    if (!s.running) _artStopPolling();
}

async function _artStart(kind, body, question) {
    if (question && !confirm(question)) return;
    try {
        const r = await fetch('/api/artifacts/' + kind, {
            method: 'POST', headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(body || {})});
        const d = await r.json().catch(() => ({}));
        if (!r.ok || d.error) throw new Error(d.error || `HTTP ${r.status}`);
    } catch (e) {
        _showToast('Not started: ' + e.message, true);
        return;
    }
    _artStatusLoad();
}

function _artDryRun() {
    const area = document.getElementById('artifactPatterns');
    _artStart('dry_run', area ? {patterns: area.value.split('\n')} : {});
}

function _artApply(runId, segments, transcripts) {
    _artStart('apply', {run_id: runId},
        `Remove ${plural(segments, 'segment')} from ${plural(transcripts, 'transcript')}?\n\n`
        + 'The database is backed up first (Database → Backups), and every removed segment is logged: “Undo” puts them back.');
}

function _artUndo(name) {
    _artStart('undo', {log: name}, 'Put the segments of this cleanup back into their transcripts?');
}

function _artStop() {
    fetch('/api/artifacts/stop', {method: 'POST'}).catch(() => {});
}

const _ART_KIND = {dry_run: 'Dry run', apply: 'Removing', undo: 'Undo'};

function _artRender(s) {
    const box = document.getElementById('artifactRun');
    if (!box) return;
    const dryBtn = document.getElementById('artDryRunBtn');
    const stopBtn = document.getElementById('artStopBtn');
    if (dryBtn) dryBtn.disabled = !!s.running;
    if (stopBtn) stopBtn.hidden = !s.running;
    const n = (x) => Number(x || 0).toLocaleString();
    const when = (t) => t ? new Date(t * 1000).toLocaleString() : '';
    const rows = (list) => `<div class="set-table-wrap" style="max-height:260px"><table class="set-table">${
        list.map(([a, b, c]) => `<tr><td style="max-width:none;white-space:normal">${esc(a)}</td>${c !== undefined
            ? `<td>${esc(c)}</td>` : ''}<td class="num">${n(b)}</td></tr>`).join('')}</table></div>`;
    const why = (w) => w === 'no letter or digit' ? 'no letter or digit' : _sCode(w);
    let html = '';
    if (s.running) {
        const pct = s.total ? Math.round(100 * s.done / s.total) : 0;
        html += `<div class="set-result">⟳ ${esc(_ART_KIND[s.kind] || s.kind)}: ${esc(s.phase || '')}${s.total ? ` — ${n(s.done)} / ${n(s.total)} (${pct} %)` : ''}</div>
            <div class="set-bar" style="margin:6px 0 12px"><span style="width:${pct}%"></span></div>`;
    } else if (s.error) {
        html += `<div class="set-note ${s.error === 'stopped' ? '' : 'err'}" style="margin-bottom:12px">${s.error === 'stopped'
            ? 'Stopped. Everything done so far is kept — and logged.' : '✕ ' + esc(s.error)}</div>`;
    }
    // Only the newest step speaks: a cleanup, or its undo, from before the
    // latest dry run is history (listed below), not news
    const d0 = s.dry_run;
    const a = s.applied && (!d0 || s.applied.run_id === d0.run_id) ? s.applied : null;
    const u = s.undone && s.undone.at >= Math.max(d0?.at || 0, a?.at || 0) ? s.undone : null;
    if (a && !s.running) {
        html += `<div class="set-note ok" style="margin-bottom:12px">✓ Removed <b>${n(a.segments)}</b> segment${a.segments === 1 ? '' : 's'} from <b>${n(a.transcripts)}</b> transcript${a.transcripts === 1 ? '' : 's'} (${esc(when(a.at))}).
            Backup first: <span class="mono">${esc(a.backup)}</span>${(a.left || []).length ? `<br>Left alone: ${a.left.map(([w, k]) => `${esc(w)} ${n(k)}`).join(' · ')}` : ''}</div>`;
    }
    if (u && !s.running) {
        html += `<div class="set-note ok" style="margin-bottom:12px">↶ Put back ${n(u.segments)} segment${u.segments === 1 ? '' : 's'} into ${n(u.transcripts)} transcript${u.transcripts === 1 ? '' : 's'}${(u.left || []).length ? ` — left alone: ${u.left.map(([w, k]) => `${esc(w)} ${n(k)}`).join(' · ')}` : ''}.</div>`;
    }
    const d = s.dry_run;
    if (d && !(s.running && s.kind === 'dry_run')) {
        const done = a && a.run_id === d.run_id;
        html += `<div class="set-kpis">
                <div class="set-kpi"><div class="set-kpi-label">${done ? 'Were' : 'Would be'} removed</div><div class="set-kpi-value">${n(d.segments)}</div><div class="set-kpi-sub">segment${d.segments === 1 ? '' : 's'}</div></div>
                <div class="set-kpi"><div class="set-kpi-label">In</div><div class="set-kpi-value">${n(d.transcripts)}</div><div class="set-kpi-sub">transcript${d.transcripts === 1 ? '' : 's'}</div></div>
                <div class="set-kpi"><div class="set-kpi-label">Left alone</div><div class="set-kpi-value">${n(d.skipped?.transcripts)}</div><div class="set-kpi-sub">${n(d.skipped?.segments)} segment${d.skipped?.segments === 1 ? '' : 's'}</div></div>
                <div class="set-kpi"><div class="set-kpi-label">Checked</div><div class="set-kpi-value">${n(d.scanned)}</div><div class="set-kpi-sub">segment${d.scanned === 1 ? '' : 's'}</div></div>
            </div>
            <div class="set-hint" style="margin:8px 0">Dry run of ${esc(when(d.at))} with ${(d.patterns || []).map(p => _sCode(p)).join(' ') || 'no pattern'} — and “no letter or digit”.</div>`;
        if (d.segments) {
            html += `<div class="set-lines" style="margin-bottom:10px">${(d.reasons || []).map(([w, k]) =>
                    `<span>${why(w)}</span><span>${n(k)} segment${k === 1 ? '' : 's'}</span>`).join('')}</div>
                <details class="set-more" open><summary>The most frequent texts (${Math.min((d.texts || []).length, 30)})</summary>
                    <div style="margin-top:6px">${rows(d.texts || [])}</div></details>
                <details class="set-more"><summary>The most affected transcripts (${(d.files || []).length}${d.transcripts > (d.files || []).length ? ' of ' + n(d.transcripts) : ''})</summary>
                    <div style="margin-top:6px">${rows(d.files || [])}</div></details>`;
        }
        if (d.skipped?.transcripts) {
            html += `<details class="set-more"><summary>Left alone: ${n(d.skipped.transcripts)} transcript${d.skipped.transcripts === 1 ? '' : 's'} with a translation or a correction run — they address segments by position, and removing one would shift them</summary>
                <div style="margin-top:6px">${rows((d.skipped.files || []).map(([f, k, w]) => [f, k, w]))}</div></details>`;
        }
        if (d.segments && !done) {
            html += `<div class="set-actions" style="margin-top:12px">
                <button type="button" class="set-btn danger" ${s.running ? 'disabled' : ''}
                    onclick="_artApply('${escJs(d.run_id)}', ${Number(d.segments)}, ${Number(d.transcripts)})">🧹 Remove ${d.segments === 1 ? 'this segment' : `these ${n(d.segments)} segments`}</button>
                <span class="set-hint">Backs the database up first; each transcript is checked again right before it is changed.</span></div>`;
        } else if (!d.segments) {
            html += '<div class="set-result ok">✓ Nothing to remove — no segment matches.</div>';
        }
    }
    const logs = s.logs || [];
    if (logs.length) {
        html += `<details class="set-more" style="margin-top:14px"${a || u ? ' open' : ''}><summary>Earlier cleanups (${logs.length})</summary>
            <div class="set-table-wrap" style="margin-top:6px"><table class="set-table">${logs.map(l => `<tr>
                <td>${esc(when(l.at))}</td>
                <td class="num">${n(l.segments)} segment${l.segments === 1 ? '' : 's'}</td><td class="num">${n(l.transcripts)} transcript${l.transcripts === 1 ? '' : 's'}</td>
                <td style="max-width:none;white-space:normal">${(l.patterns || []).map(p => _sCode(p)).join(' ')}</td>
                <td>${l.undone ? '<span class="set-hint">undone</span>'
                    : `<button type="button" class="set-btn sm" ${s.running ? 'disabled' : ''} onclick="_artUndo('${escJs(l.name)}')">↶ Undo</button>`}</td>
            </tr>`).join('')}</table></div></details>`;
    }
    box.innerHTML = html;
}

// ══════════════════════════════════════════════════════════════════════
//  Language models: provider cards (llama.cpp / Ollama / OpenAI / …)
// ══════════════════════════════════════════════════════════════════════

let _provData = {providers: [], types: [], tasks: {}};
// Embedding is deliberately absent: it has its own section (AI models →
// Embedding model) — a mapping row here would do nothing.
const _TASK_LABELS = {
    summary: '📝 Summaries',
    correction: '🪄 Correction',
    translation: '🌐 Translation',
    chat: '💬 Chat (Ask in the search)',
};

async function _provLoad() {
    try {
        _provData = await (await fetch('/api/llm/providers')).json();
    } catch (e) { _provData = {providers: [], types: [], tasks: {}}; }
    _provData.tasks = _provData.tasks || {};
    _provRender();
}

function _provTypeOptions(chosen) {
    return (_provData.types || []).map(t =>
        `<option value="${esc(t.id)}" ${t.id === chosen ? 'selected' : ''}>${esc(t.name)}</option>`).join('');
}

// Model chips of a card. OpenAI supplies 70+ models — the list shows a
// capped number and expands on request. The chosen default model is always
// visible, even if it would be far down alphabetically.
const _provModelsFor = new Set();
const _PROV_CHIP_MAX = 10;

function _provModelChips(i, p) {
    const all = p.models || [];
    const chip = (m) => `<button type="button" class="set-model ${m === p.default_model ? 'active' : ''}"
        onclick="_provField(${i},'default_model','${escJs(m)}');_provRender()"
        title="Use as the default model">${esc(m)}</button>`;
    if (all.length <= _PROV_CHIP_MAX || _provModelsFor.has(i)) {
        return all.map(chip).join('')
            + (all.length > _PROV_CHIP_MAX
                ? `<button type="button" class="set-model more" onclick="_provModelsToggle(${i})">▲ fewer</button>` : '');
    }
    let show = all.slice(0, _PROV_CHIP_MAX);
    if (p.default_model && all.includes(p.default_model) && !show.includes(p.default_model)) {
        show = [p.default_model, ...show.slice(0, _PROV_CHIP_MAX - 1)];
    }
    return show.map(chip).join('')
        + `<button type="button" class="set-model more" onclick="_provModelsToggle(${i})">+${all.length - show.length} more…</button>`;
}

function _provModelsToggle(i) {
    if (_provModelsFor.has(i)) _provModelsFor.delete(i);
    else _provModelsFor.add(i);
    _provRender();
}

// Last test result per provider — survives the redraw that another test
// (running in parallel) triggers when it finds models
const _provTestMsg = new WeakMap();

function _provRender() {
    const box = document.getElementById('provList');
    if (!box) return;
    const ps = _provData.providers || [];
    if (!ps.length) {
        box.innerHTML = '<div class="set-hint">No provider yet — “+ Add provider” starts with a llama.cpp template.</div>';
    } else {
        box.innerHTML = ps.map((p, i) => {
            const kind = (_provData.types || []).find(t => t.id === p.kind);
            const msg = _provTestMsg.get(p);
            return `<div class="set-prov${i === 0 ? ' first' : ''}">
                <div class="set-prov-head">
                    <input class="set-input" value="${esc(p.name)}" aria-label="Provider name"
                           onchange="_provField(${i},'name',this.value)">
                    ${i === 0 ? '<span class="set-tag jobs">Default</span>' : ''}
                    ${p.has_key ? '<span class="set-tag">🔑 key saved</span>' : ''}
                    <span style="flex:1"></span>
                    ${i > 0 ? `<button type="button" class="set-btn sm" onclick="_provUp(${i})" title="Make this the default">▲ Default</button>` : ''}
                    <button type="button" class="set-btn sm" onclick="_provTest(${i})">🔌 Test</button>
                    <button type="button" class="set-btn sm danger" onclick="_provRemove(${i})" title="Remove this provider">✕</button>
                </div>
                <div class="set-prov-grid">
                    <label>Type</label>
                    <select class="set-select" onchange="_provTypeSwitch(${i}, this.value)">${_provTypeOptions(p.kind)}</select>
                    <label>Address</label>
                    <input class="set-input mono" value="${esc(p.endpoint)}" placeholder="${esc(kind?.template || '')}"
                           onchange="_provField(${i},'endpoint',this.value)">
                    <label>API key</label>
                    <input class="set-input mono" type="password" autocomplete="new-password"
                           placeholder="${p.has_key ? '•••••• saved — type to replace' : 'empty = none'}"
                           onchange="_provField(${i},'api_key',this.value)">
                    <label>Default model</label>
                    <input class="set-input mono" value="${esc(p.default_model)}" placeholder="the server's own default"
                           onchange="_provField(${i},'default_model',this.value)">
                </div>
                ${kind?.hint ? `<div class="set-hint" style="margin-top:8px">💡 ${esc(kind.hint)}</div>` : ''}
                <div class="set-models">${_provModelChips(i, p)}</div>
                <div class="set-result ${msg?.cls || ''}" id="provTest${i}" style="margin-top:6px">${esc(msg?.text || '')}</div>
            </div>`;
        }).join('');
    }
    _provRenderTasks();
}

function _provRenderTasks() {
    const box = document.getElementById('provTasks');
    if (!box) return;
    const ps = _provData.providers || [];
    if (!ps.length) { box.innerHTML = '<div class="set-hint">Add a provider first.</div>'; return; }
    box.innerHTML = '<div class="set-tasks"><span></span><span class="set-th">Provider</span><span class="set-th">Model</span>'
        + Object.entries(_TASK_LABELS).map(([task, label]) => {
            const z = _provData.tasks[task] || {};
            const p = ps.find(x => x.id === z.provider) || ps[0];
            return `<span>${label}</span>
                <select class="set-select" onchange="_provTask('${task}','provider',this.value)" aria-label="${esc(label)} provider">
                    <option value="">Default (${esc(ps[0].name)})</option>
                    ${ps.map(x => `<option value="${esc(x.id)}" ${x.id === z.provider ? 'selected' : ''}>${esc(x.name)}</option>`).join('')}
                </select>
                <select class="set-select" onchange="_provTask('${task}','model',this.value)" aria-label="${esc(label)} model">
                    <option value="">Default model${p?.default_model ? ` (${esc(p.default_model)})` : ''}</option>
                    ${_sWith((p?.models || []).map(m => [m, m]), z.model).map(([m, l]) =>
                        `<option value="${esc(m)}" ${m === z.model ? 'selected' : ''}>${esc(l)}</option>`).join('')}
                </select>`;
        }).join('') + '</div>';
}

function _provField(i, field, value) {
    _provData.providers[i][field] = value;
    if (field === 'api_key' && value) _provData.providers[i].has_key = true;
    _markDirty('llm.providers');
}

function _provTask(task, field, value) {
    _provData.tasks[task] = _provData.tasks[task] || {};
    _provData.tasks[task][field] = value;
    // Another provider has other models: its model choice starts over
    if (field === 'provider') {
        _provData.tasks[task].model = '';
        _provRenderTasks();
    }
    _markDirty('llm.providers');
}

function _provTypeSwitch(i, kind) {
    const p = _provData.providers[i];
    const template = (_provData.types || []).find(t => t.id === kind)?.template || '';
    p.kind = kind;
    if (!p.endpoint && template) p.endpoint = template;
    _provRender();
    _markDirty('llm.providers');
}

function _provAdd() {
    const n = (_provData.providers || []).length + 1;
    _provData.providers.push({
        id: 'p' + Date.now().toString(36), name: 'Provider ' + n,
        kind: 'llamacpp', endpoint: 'http://localhost:8080/v1',
        models: [], default_model: '', active: true, has_key: false});
    _provRender();
    _markDirty('llm.providers');
}

function _provRemove(i) {
    const p = _provData.providers[i];
    if (!confirm(`Remove the provider “${p.name}”?${p.has_key ? ' Its saved API key goes with it.' : ''}`)) return;
    _provData.providers.splice(i, 1);
    // Tasks that named it fall back to the default
    for (const z of Object.values(_provData.tasks || {})) {
        if (z.provider === p.id) { z.provider = ''; z.model = ''; }
    }
    _provRender();
    _markDirty('llm.providers');
}

function _provUp(i) {
    const [p] = _provData.providers.splice(i, 1);
    _provData.providers.unshift(p);
    _provRender();
    _markDirty('llm.providers');
}

async function _provTest(i) {
    const p = _provData.providers[i];
    const show = (cls, text) => {
        _provTestMsg.set(p, {cls, text});
        const box = document.getElementById('provTest' + _provData.providers.indexOf(p));
        if (!box) return;
        box.className = 'set-result ' + cls;
        box.textContent = text;
    };
    show('', '⏳ Connecting…');
    try {
        const r = await fetch('/api/llm/providers/test', {
            method: 'POST', headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({endpoint: p.endpoint, api_key: p.api_key || '', id: p.id})});
        const d = await r.json();
        if (!d.reachable) { show('err', '✕ Not reachable: ' + (d.detail || '')); return; }
        // llama.cpp lists its model under two keys — the same id twice
        const models = [...new Set(d.models || [])];
        const msg = `✓ Reachable (${d.api_type})`
            + (d.context ? ` · context ${d.context.toLocaleString()} tokens` : '')
            + (models.length ? ` · ${models.length} model${models.length === 1 ? '' : 's'}` : '');
        _provTestMsg.set(p, {cls: 'ok', text: msg});
        if (models.length) {
            const before = JSON.stringify([p.models, p.default_model]);
            p.models = models;
            if (!p.default_model) p.default_model = models[0];
            if (JSON.stringify([p.models, p.default_model]) !== before) _markDirty('llm.providers');
            _provRender();
        }
        show('ok', msg);
    } catch (e) {
        show('err', '✕ ' + e.message);
    }
}

// Built-in correction rules — the grey placeholder, and the start of your own
let _corrRulesDefault = '';

async function _llmStatusLoad() {
    let st;
    try { st = await fetch('/api/llm/status').then(r => r.json()); } catch (e) { return; }
    _corrRulesDefault = st.correction_rules_default || '';
    const ta = document.getElementById('corrRulesTa');
    if (ta) ta.placeholder = _corrRulesDefault || 'Empty = the built-in rules';
    const span = document.getElementById('llmCtxDetected');
    if (span) {
        span.textContent = st.context_effective
            ? st.context_effective.toLocaleString() + ' tokens' + (st.context_size > 0 ? ' (set above)' : ' (reported by the server)')
            : st.configured ? 'not reported — 8,192 assumed' : '–';
    }
}

function _corrRulesTemplate() {
    const ta = document.getElementById('corrRulesTa');
    if (!ta || !_corrRulesDefault) return;
    if (ta.value.trim() && !confirm('The field isn\'t empty — replace it with the built-in rules?')) return;
    _sWrite(ta, _corrRulesDefault);
}

// One summary through the summary task, with the saved settings
async function _testLlmConnection() {
    const el = document.getElementById('llmTestResult');
    const set = (cls, text) => { el.className = 'set-result ' + cls; el.textContent = text; };
    if (_settingsUnsavedCount()) { set('warn', 'Save first — the test uses the saved settings.'); return; }
    set('', '⏳ Asking…');
    try {
        const st = await fetch('/api/llm/status').then(r => r.json());
        if (!st.configured) { set('err', '✕ No provider set up'); return; }
        const tr = await fetch('/api/summarize', {
            method: 'POST', headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                segments: [{text: 'Test', start: 0, end: 1, speaker: 'A', start_fmt: '0:00', end_fmt: '0:01'}],
                prompt: 'Reply only with the word OK', include_speakers: false}),
        });
        if (!tr.ok) {
            const err = await tr.json().catch(() => ({}));
            set('err', '✕ ' + (err.error || `HTTP ${tr.status}`));
            return;
        }
        // Tokens can be split across chunk boundaries: read it all first
        const reader = tr.body.getReader();
        const decoder = new TextDecoder();
        let acc = '';
        for (;;) {
            const {done, value} = await reader.read();
            if (done) break;
            acc += decoder.decode(value, {stream: true});
        }
        const err = acc.match(/"error"\s*:\s*"([^"]+)"/);
        if (err) set('err', '✕ ' + err[1]);
        else if (acc.includes('"text"')) {
            const task = _provData.tasks?.summary || {};
            const p = (_provData.providers || []).find(x => x.id === task.provider) || _provData.providers?.[0];
            set('ok', `✓ Answered — ${p ? p.name : 'default provider'}${task.model || p?.default_model ? ' · ' + (task.model || p.default_model) : ''}`);
        } else set('warn', '⚠ Answer without text');
        if (typeof refreshConnStatus === 'function') refreshConnStatus();
    } catch (e) {
        set('err', '✕ ' + e.message);
    }
}

// ── Summaries: templates and model shortcuts ─────────────────────────────

function _llmStore() {
    const cfg = _settingsData.config = _settingsData.config || {};
    return cfg.llm = cfg.llm || {};
}

// Templates unfolded for editing. Folded, a template is one line — its
// name and how it starts — so ten templates stay one screen, not five.
const _presetOpen = new Set();

function _renderLlmPresets(l) {
    const list = document.getElementById('llmPresetsList');
    if (!list) return;
    const presets = l.presets || {};
    const names = Object.keys(presets);
    if (!names.length) { list.innerHTML = '<div class="set-hint">No templates yet.</div>'; return; }
    list.innerHTML = names.map(name => {
        const start = String(presets[name] || '').trim().split('\n')[0] || '(empty)';
        return `<details class="set-item set-item-fold" data-preset="${esc(name)}"${_presetOpen.has(name) ? ' open' : ''}
                ontoggle="_presetToggle(this)">
            <summary class="set-item-head">
                <span class="set-item-name">${esc(name)}</span>
                <span class="set-item-preview">${esc(start)}</span>
                <button type="button" class="set-btn sm danger" onclick="event.preventDefault();_removeLlmPreset(this)" title="Delete this template">✕</button>
            </summary>
            <textarea class="set-textarea" rows="9" spellcheck="false"
                oninput="_updateLlmPreset(this)">${esc(presets[name])}</textarea>
        </details>`;
    }).join('');
}

function _presetToggle(el) {
    const name = el.dataset.preset;
    if (el.open) _presetOpen.add(name);
    else _presetOpen.delete(name);
}

function _addLlmPreset() {
    const input = document.getElementById('llmNewPresetName');
    const name = input?.value.trim();
    if (!name) return;
    const l = _llmStore();
    l.presets = l.presets || {};
    if (name in l.presets) { _showToast('A template with that name exists', true); return; }
    l.presets[name] = `Write the prompt for “${name}” here.`;
    _presetOpen.add(name);
    input.value = '';
    _markDirty('llm.presets');
    _renderLlmPresets(l);
}

function _removeLlmPreset(btn) {
    const name = btn.closest('[data-preset]')?.dataset.preset;
    const l = _llmStore();
    if (name === undefined || !l.presets || !(name in l.presets)) return;
    if (!confirm(`Delete the template “${name}”?`)) return;
    delete l.presets[name];
    _markDirty('llm.presets');
    _renderLlmPresets(l);
}

function _updateLlmPreset(area) {
    const name = area.closest('[data-preset]')?.dataset.preset;
    const l = _llmStore();
    if (name === undefined || !l.presets) return;
    l.presets[name] = area.value;
    _markDirty('llm.presets');
}

// The user config has its own copy of the built-in templates; improved
// versions only reach it through this. Templates of your own stay untouched.
async function _checkDefaultPresets() {
    const box = document.getElementById('llmStdPresets');
    if (!box) return;
    box.innerHTML = '<div class="set-hint">Comparing…</div>';
    try {
        const d = await (await fetch('/api/llm/standard_presets')).json();
        // Statuses from /api/llm/standard_presets: new · same · different
        const missing = d.presets.filter(p => p.status === 'new');
        const changed = d.presets.filter(p => p.status === 'different');
        if (!missing.length && !changed.length) {
            box.innerHTML = '<div class="set-note ok" style="margin-bottom:10px">✓ All built-in templates are up to date.</div>';
            return;
        }
        box.innerHTML = `<div class="set-note" style="margin-bottom:10px">
            ${missing.length ? `<div>🆕 <b>${missing.length} missing</b>: ${missing.map(p => esc(p.name)).join(', ')}</div>` : ''}
            ${changed.length ? `<div>✏️ <b>${changed.length} differ</b> from the built-in version (edited, or an older version): ${changed.map(p => esc(p.name)).join(', ')}</div>` : ''}
            ${d.custom.length ? `<div class="set-hint">Your own templates (${d.custom.map(esc).join(', ')}) stay untouched.</div>` : ''}
            <div class="set-actions" style="margin-top:8px">
                ${missing.length ? `<button type="button" class="set-btn sm" onclick="_applyDefaultPresets(false)">➕ Add the ${missing.length} missing</button>` : ''}
                ${changed.length ? `<button type="button" class="set-btn sm" onclick="_applyDefaultPresets(true)">↻ Reset the ${changed.length} to the built-in version</button>` : ''}
            </div></div>`;
    } catch (e) {
        box.innerHTML = `<div class="set-note err" style="margin-bottom:10px">${esc(e.message)}</div>`;
    }
}

async function _applyDefaultPresets(overwrite) {
    if (_settingsUnsavedCount() && !confirm('This reloads the settings — unsaved changes are lost. Continue?')) return;
    try {
        const r = await fetch('/api/llm/presets/merge', {
            method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({overwrite})});
        const d = await r.json();
        if (!r.ok || d.error) throw new Error(d.error || 'Error');
        const n = d.added.length + d.replaced.length;
        _showToast(`${n} template${n === 1 ? '' : 's'} updated (${d.total} in all)`);
        _settingsExtraDirty.clear();
        await _loadAllSettings();
        _checkDefaultPresets();
    } catch (e) { _showToast(e.message, true); }
}

function _renderLlmModels(l) {
    const list = document.getElementById('llmModelsList');
    if (!list) return;
    // Only strings: broken nested objects of an old save are left out
    const models = Object.entries(l.models || {}).filter(([, v]) => typeof v === 'string');
    if (!models.length) { list.innerHTML = '<div class="set-hint">No shortcuts — the dialog offers the task\'s model.</div>'; return; }
    list.innerHTML = models.map(([name, id]) => `<div class="set-item" data-model="${esc(name)}">
            <div class="set-item-head" style="margin:0">
                <span class="set-item-name">${esc(name)}</span>
                <span class="set-item-meta mono">${esc(id)}</span>
                <button type="button" class="set-btn sm danger" onclick="_removeLlmModel(this)" title="Remove">✕</button>
            </div></div>`).join('');
}

function _addLlmModel() {
    const name = document.getElementById('llmNewModelName')?.value.trim();
    const id = document.getElementById('llmNewModelId')?.value.trim() || name;
    if (!name) return;
    const l = _llmStore();
    l.models = l.models || {};
    l.models[name] = id;
    document.getElementById('llmNewModelName').value = '';
    document.getElementById('llmNewModelId').value = '';
    _markDirty('llm.models');
    _renderLlmModels(l);
}

function _removeLlmModel(btn) {
    const name = btn.closest('[data-model]')?.dataset.model;
    const l = _llmStore();
    if (name === undefined || !l.models) return;
    delete l.models[name];
    _markDirty('llm.models');
    _renderLlmModels(l);
}

// ── Embeddings: the server the app starts itself ─────────────────────────

let _embStatus = null;

async function _embLoadStatus() {
    const card = document.getElementById('embServerCard');
    if (!card) return;
    try {
        _embStatus = await fetch('/api/embedding/status').then(r => r.json());
    } catch (e) {
        card.innerHTML = `<div class="set-note err">State not available: ${esc(String(e))}</div>`;
        return;
    }
    const l = _embStatus.local_server || {};
    const running = l.reachable;
    const own = l.managed;
    let state;
    if (running && own) state = `<span class="set-dot ok"></span>Running (pid ${l.pid}, for ${_embDuration(l.running_for_sec)})`;
    // Something WE did not start answers on this port: used, not stopped
    else if (running) state = '<span class="set-dot warn"></span>Another server answers on this port — used, but not managed';
    else if (!l.possible) state = `<span class="set-dot err"></span>Can't be started — ${esc(l.hint || '')}`;
    else state = '<span class="set-dot"></span>Off — starts with the first embedding';
    const startBtn = document.getElementById('embBtnStart');
    const stopBtn = document.getElementById('embBtnStop');
    if (startBtn) {
        startBtn.disabled = !l.possible || running;
        startBtn.title = !l.possible ? (l.hint || 'Can\'t be started') : running ? 'Already running' : 'Start the embedding server now';
    }
    if (stopBtn) {
        stopBtn.disabled = !(running && own);
        stopBtn.title = running && !own ? 'Not started by this app — it is left alone'
            : running ? 'Stop the server and free its VRAM' : 'Not running';
    }
    card.innerHTML = `<div class="set-lines">
            <span>State</span><span>${state}</span>
            <span>Address</span><span class="mono">${esc(l.endpoint || '')}</span>
            <span>Program</span><span>${l.binary ? `<span class="mono">${esc(l.binary)}</span>` : '<i>llama-server not found</i>'}</span>
            <span>Model</span><span>${l.model ? `<span class="mono">${esc(l.model)}</span>` : '<i>not set</i>'}</span>
            ${own && l.idle_limit_sec ? `<span>Idle</span><span>${_embDuration(l.idle_sec)} of ${_embDuration(l.idle_limit_sec)}</span>` : ''}
        </div>`;
}

function _embDuration(s) {
    s = Math.round(s || 0);
    if (s < 60) return `${s} s`;
    if (s < 3600) return `${Math.floor(s / 60)} min ${s % 60} s`;
    return `${Math.floor(s / 3600)} h ${Math.floor((s % 3600) / 60)} min`;
}

async function _embServerStart(btn) {
    const label = btn.textContent;
    btn.disabled = true; btn.textContent = '⏳ Starting…';
    try {
        await _settingsSaveQuiet();
        const r = await fetch('/api/embedding/server/start', {method: 'POST'});
        const d = await r.json().catch(() => ({}));
        if (!r.ok || d.error) _showToast('Start failed: ' + (d.error || `HTTP ${r.status}`), true);
        else _showToast(`Server ready after ${d.duration_sec} s`);
    } catch (e) {
        _showToast('Start failed: ' + e.message, true);
    } finally {
        btn.disabled = false; btn.textContent = label; _embLoadStatus();
    }
}

async function _embServerStop(btn) {
    const label = btn.textContent;
    btn.disabled = true; btn.textContent = '⏳ Stopping…';
    try {
        const r = await fetch('/api/embedding/server/stop', {method: 'POST'});
        const d = await r.json().catch(() => ({}));
        _showToast(d.finished ? 'Server stopped, VRAM free' : 'No server started by this app was running');
    } catch (e) {
        _showToast(e.message, true);
    } finally {
        btn.disabled = false; btn.textContent = label; _embLoadStatus();
    }
}

// Does a vector come back? Many servers answer /v1/models but refuse
// /v1/embeddings without --embedding.
async function _testEmbeddingConnection() {
    const el = document.getElementById('embTestResult');
    if (!el) return;
    const set = (cls, text) => { el.className = 'set-result ' + cls; el.textContent = text; };
    set('', '⏳ Testing… (starting a server can take a while)');
    try {
        await _settingsSaveQuiet();
        const r = await fetch('/api/embedding/test', {
            method: 'POST', headers: {'Content-Type': 'application/json'}, body: '{}'});
        const d = await r.json().catch(() => ({}));
        if (d.ok) set('ok', `✓ ${d.dimensions} dimensions · ${d.duration_ms} ms` + (d.normalized ? ' · normalized' : ` · length ${d.length}`));
        else set('err', '✕ ' + (d.error || `HTTP ${r.status}`));
        if (typeof refreshConnStatus === 'function') refreshConnStatus();
    } catch (e) {
        set('err', '✕ ' + e.message);
    }
    _embLoadStatus();
}

async function _embBenchmark(btn) {
    const target = document.getElementById('embBenchResult');
    const label = btn.textContent;
    btn.disabled = true; btn.textContent = '⏳ Measuring…';
    target.innerHTML = '<div class="set-hint">Measuring on real chunks of your library — this takes a while.</div>';
    try {
        await _settingsSaveQuiet();
        const r = await fetch('/api/embedding/benchmark', {
            method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({sizes: [8, 32, 64]})});
        const d = await r.json().catch(() => ({}));
        if (!d.ok) { target.innerHTML = `<div class="set-note err">✕ ${esc(d.error || 'Measuring failed')}</div>`; return; }
        const rows = (d.results || []).map(x => x.error
            ? `<tr><td class="num">${x.batch}</td><td colspan="3" style="color:var(--red)">${esc(x.error)}</td></tr>`
            : `<tr><td class="num">${x.batch}</td><td class="num">${x.chunks_per_sec}</td>
                   <td class="num">${x.ms_per_request} ms</td><td class="num">${x.chunks} in ${x.seconds} s</td></tr>`).join('');
        target.innerHTML = `<div class="set-table-wrap"><table class="set-table">
                <tr><th class="num">Batch</th><th class="num">Chunks/s</th><th class="num">Per request</th><th class="num">Measured</th></tr>${rows}
            </table></div>
            <div class="set-result" style="margin-top:8px">Best: <b>${d.best_rate} chunks/s</b>${d.estimated_hours != null
                ? ` · still open: ~${plural(d.estimated_chunks, 'chunk')} → <b>~${d.estimated_hours} h</b>` : ''}</div>
            <div class="set-hint">Chunks average ${d.chunk_length_avg} characters · warm-up ${d.warmup_sec} s${d.estimated_hours != null ? ' · the time left is estimated from segments per chunk' : ''}</div>`;
    } catch (e) {
        target.innerHTML = `<div class="set-note err">✕ ${esc(e.message)}</div>`;
    } finally {
        btn.disabled = false; btn.textContent = label; _embLoadStatus();
    }
}

// The setup guide with a ready-made prompt for an AI of your choice
async function _embGuideToggle() {
    const el = document.getElementById('embGuide');
    if (!el) return;
    el.hidden = !el.hidden;
    if (el.hidden || el.dataset.loaded) return;
    el.innerHTML = '<div class="set-hint">Loading…</div>';
    let prompt = '';
    try { prompt = (await fetch('/api/embedding/guide').then(r => r.json())).prompt || ''; } catch (e) { /* empty */ }
    el.dataset.loaded = '1';
    el.innerHTML = `<div class="set-note" style="margin-bottom:10px">
            <p style="margin:0 0 6px"><b>You set up the server yourself — on purpose.</b> Which model fits the machine, llama.cpp or Ollama, whether it survives a reboot: decisions about your system. The text below carries this machine's values; give it to an AI of your choice and it walks you through installation, model choice and a test.</p>
            <p style="margin:0">Afterwards: enter the /v1 address here → 🧬 Test (it must report dimensions) → ⏱ Measure (how long the index will take) → Search → Semantic search → Embed.</p>
        </div>
        <div class="set-actions" style="margin-bottom:6px"><b style="font-size:0.78rem">Prompt for an AI</b>
            <button type="button" class="set-btn sm" onclick="_embCopyPrompt(this)">📋 Copy</button>
            <span class="set-hint">Your hardware is filled in</span></div>
        <textarea id="embPromptText" class="set-textarea mono" readonly rows="9">${esc(prompt)}</textarea>`;
}

function _embCopyPrompt(btn) {
    const ta = document.getElementById('embPromptText');
    if (ta) _sCopy(ta.value, btn);
}

// Copy a text; the button says whether it worked. Without HTTPS there is
// no clipboard API — and the app usually runs over http at home: the old
// way is the normal case, not a fallback.
function _sCopy(text, btn) {
    const done = (ok) => {
        if (!btn) return;
        btn.dataset.label = btn.dataset.label || btn.textContent;
        btn.textContent = ok ? '✓ Copied' : '✕ Not copied';
        clearTimeout(btn._copyTimer);
        btn._copyTimer = setTimeout(() => { btn.textContent = btn.dataset.label; }, 1500);
    };
    const old = () => {
        const ta = document.createElement('textarea');
        ta.value = String(text ?? '');
        ta.setAttribute('readonly', '');
        ta.style.cssText = 'position:fixed;top:-1000px;opacity:0';
        document.body.appendChild(ta);
        ta.select();
        let ok = false;
        try { ok = document.execCommand('copy'); } catch (e) { ok = false; }
        ta.remove();
        done(ok);
    };
    if (navigator.clipboard && window.isSecureContext) {
        navigator.clipboard.writeText(String(text ?? '')).then(() => done(true), old);
    } else {
        old();
    }
}

// ── Semantic search: the vector index ────────────────────────────────────

let _ragBuildSource = null;
let _ragPolling = null;

async function _loadRagStatus() {
    const card = document.getElementById('ragStatusCard');
    if (!card) return;
    try {
        const d = await fetch('/api/rag/status').then(r => r.json());
        const pdb = d.db || {};
        const st = pdb.stats || {};
        const emb = d.embedding || {};
        const src = d.source || {};
        const reach = !!pdb.reachable;
        const kpi = (label, value, sub = '') => `<div class="set-kpi"><div class="set-kpi-label">${label}</div>
            <div class="set-kpi-value">${value}</div>${sub ? `<div class="set-kpi-sub">${sub}</div>` : ''}</div>`;
        card.innerHTML = `<div class="set-lines">
                <span>Index file</span><span><span class="set-dot ${reach ? 'ok' : 'err'}"></span>${reach ? `<span class="mono">${esc(pdb.path || '')}</span>` : 'not readable' + (pdb.detail ? ' — ' + esc(pdb.detail) : '')}</span>
                <span>Embedding model</span><span><span class="set-dot ${emb.configured ? 'ok' : 'err'}"></span>${emb.configured ? esc(emb.model) : 'not set up' + (emb.detail ? ' — ' + esc(emb.detail) : '')} ${_sGo('embedding', 'AI models')}</span>
            </div>
            ${st.exists ? `<div class="set-kpis" style="margin-top:12px">
                ${kpi('Chunks', (st.chunks || 0).toLocaleString())}
                ${kpi('Transcripts', (st.transcripts || 0).toLocaleString(), `of ${(src.transcripts || 0).toLocaleString()} in the library`)}
                ${kpi('Size', _fmtBytesS(st.size_bytes || 0))}
                ${kpi('Dimensions', st.dimensions || 0, esc(st.vector_type || ''))}
            </div>` : reach ? '<div class="set-hint" style="margin-top:8px">No index yet — “Embed new transcripts” builds it.</div>' : ''}`;
        if (d.build && d.build.running) { _ragSetBuilding(true); _ragStartPolling(); }
    } catch (e) {
        card.innerHTML = `<div class="set-note err">Status not available: ${esc(e.message)}</div>`;
    }
    // Coverage separately — it takes a moment with many transcripts
    _loadRagCoverage();
}

async function _loadRagCoverage() {
    const card = document.getElementById('ragCoverageCard');
    const wrap = document.getElementById('ragPendingWrap');
    if (!card) return;
    // During a run the card belongs to the run: coverage is expensive and
    // would be stale by the time it arrives
    if (_ragBuildSource || _ragPolling) return;
    card.innerHTML = '<div class="set-hint">Computing coverage…</div>';
    if (wrap) wrap.innerHTML = '';
    try {
        const d = await fetch('/api/rag/coverage').then(r => r.json());
        if (d.error && !d.reachable) { card.innerHTML = `<div class="set-hint">Coverage not available: ${esc(d.error)}</div>`; return; }
        const total = d.total || 0, done = d.embedded || 0, open = d.pending || 0, stale = d.stale || 0;
        const pct = total > 0 ? Math.round(done / total * 100) : 0;
        const btn = document.getElementById('ragBuildBtn');
        if (btn && !_ragBuildSource) btn.textContent = open > 0 ? `🧠 Embed ${plural(open, 'transcript')}` : '🧠 Everything is embedded';
        card.innerHTML = _sCoverageHtml(done, total, pct, [
            `✓ ${done.toLocaleString()} embedded`, `⏳ ${open.toLocaleString()} open`,
            ...(stale ? [`↻ ${stale.toLocaleString()} changed since`] : [])], false, {unit: total === 1 ? 'transcript' : 'transcripts'});
        if (wrap && open > 0 && (d.pending_sample || []).length) {
            wrap.innerHTML = `<details class="set-more" style="margin-top:8px"><summary>Open transcripts (${open.toLocaleString()}${d.pending_truncated ? ', first 300' : ''})</summary>
                <div class="set-table-wrap" style="max-height:220px;margin-top:6px">${d.pending_sample.map(k =>
                    `<div class="set-log-row" title="${esc(k)}"><span class="set-log-meta">📄</span><span class="mono" style="font-size:0.72rem">${esc(String(k).split('/').pop())}</span></div>`).join('')}</div></details>`;
        }
    } catch (e) {
        card.innerHTML = `<div class="set-note err">Coverage not available: ${esc(e.message)}</div>`;
    }
}

/* How far an index is: "done / total", the share, a bar, the parts — one
 * design for the semantic search's index and for Images & sounds, so both
 * look alike. `unit` names what is counted, `what` what the share is of
 * ("embedded", "of 1,200 h indexed"); `pct` may carry a decimal. */
function _sCoverageHtml(done, total, pct, parts, running = false, {unit = '', what = 'embedded'} = {}) {
    const width = Math.max(0, Math.min(100, Number(pct) || 0));
    return `<div style="display:flex;justify-content:space-between;align-items:baseline;gap:12px;margin-bottom:6px">
            <span style="font-size:1.05rem;font-weight:600">${done.toLocaleString()} / ${total.toLocaleString()}${unit ? ` <span class="set-result" style="font-weight:400">${unit}</span>` : ''}</span>
            <span class="set-result">${running ? '⟳ ' : ''}${pct} % ${what}${running ? ' — running' : ''}</span></div>
        <div class="set-bar"><span style="width:${width}%"></span></div>
        <div class="set-result" style="display:flex;gap:14px;flex-wrap:wrap;margin-top:6px">${parts.map(p => `<span>${p}</span>`).join('')}</div>`;
}

function _ragSetBuilding(on) {
    const b = document.getElementById('ragBuildBtn');
    const s = document.getElementById('ragStopBtn');
    if (b) b.disabled = on;
    if (s) s.hidden = !on;
}

async function _ragBuild(rebuild, btn) {
    if (_ragBuildSource) { _showToast('A run is already going', true); return; }
    if (rebuild && !confirm('Rebuild the whole index? Every transcript is embedded again — with a large library that takes hours.')) return;
    const prog = document.getElementById('ragBuildProgress');
    const say = (cls, text) => { if (prog) { prog.className = 'set-result' + (cls ? ' ' + cls : ''); prog.textContent = text; } };
    say('', 'Saving the settings…');
    try { await _settingsSaveQuiet(); } catch (e) {
        say('', '');
        _showToast('Not started — saving failed: ' + e.message, true);
        return;
    }
    // The stream can't hand over an error answer — EventSource only sees
    // that it failed. So ask first whether a run can start at all; without
    // this the line stood at "Saving the settings…" for good.
    try {
        const st = await fetch('/api/rag/status').then(r => r.json());
        if (st.embedding && !st.embedding.configured) {
            say('err', `✕ Not started — no embedding model to embed with: ${st.embedding.detail || 'not set up'}. Set one up under AI models → Embedding model.`);
            return;
        }
        if (st.build?.running) {
            say('', '');
            _showToast('A run is already going', true);
            _ragSetBuilding(true);
            _ragStartPolling();
            return;
        }
    } catch (e) { /* the stream will tell */ }
    say('', '⟳ Starting…');
    _ragSetBuilding(true);
    let src;
    try { src = new EventSource(`/api/rag/build?${rebuild ? 'rebuild=1' : ''}`); }
    catch (e) { _ragSetBuilding(false); say('err', '✕ ' + e.message); return; }
    _ragBuildSource = src;
    let heard = false;
    src.onmessage = (ev) => {
        heard = true;
        let d; try { d = JSON.parse(ev.data); } catch (e) { return; }
        _ragRenderProgress(d, prog);
        // Terminal phases close the stream (else EventSource reconnects)
        if (['complete', 'error', 'closed'].includes(d.phase)) _ragBuildDone();
    };
    src.onerror = () => {
        if (!heard) say('err', '✕ The run didn\'t start — the index status above says why.');
        if (_ragBuildSource) _ragBuildDone();
    };
}

/* A run that was already going before this page was opened.
 *
 * `/api/rag/build` is a stream and a start command in one: a second call
 * gets 409 and nothing else. Without this, whoever opened the settings
 * after the start saw a card standing still while a whole library ran
 * through in the background. `/api/rag/build/progress` is information only. */
function _ragStartPolling() {
    if (_ragPolling || _ragBuildSource) return;
    const tick = async () => {
        try {
            const d = await fetch('/api/rag/build/progress').then(r => r.json());
            if (!d.running) { _ragStopPolling(); _ragBuildDone(); return; }
            if (d.progress) _ragRenderProgress(d.progress, document.getElementById('ragBuildProgress'));
        } catch (e) { /* next attempt in 2 s */ }
    };
    tick();
    _ragPolling = setInterval(tick, 2000);
}

function _ragStopPolling() {
    if (_ragPolling) { clearInterval(_ragPolling); _ragPolling = null; }
}

function _ragBuildDone() {
    _ragStopPolling();
    if (_ragBuildSource) { try { _ragBuildSource.close(); } catch (e) { /* closed */ } _ragBuildSource = null; }
    _ragSetBuilding(false);
    _loadRagStatus();
}

/* The coverage card during a run, from the run's own messages: `indexed`
 * were embedded in this run, `skipped` already were — together the current
 * state, without repeating the expensive coverage query. */
function _ragRenderProgress(d, prog) {
    const card = document.getElementById('ragCoverageCard');
    if (card && d && typeof d.total === 'number' && d.total) {
        const done = (Number(d.indexed) || 0) + (Number(d.skipped) || 0);
        const total = Number(d.total) || 0;
        card.innerHTML = _sCoverageHtml(done, total, total ? Math.round(done / total * 100) : 0, [
            `🧠 ${(Number(d.indexed) || 0).toLocaleString()} in this run`,
            `⏭ ${(Number(d.skipped) || 0).toLocaleString()} already there`,
            `⏳ ${Math.max(0, total - done).toLocaleString()} to go`,
            ...(d.chunks ? [`🧩 ${plural(d.chunks, 'chunk')}`] : []),
            ...(d.error_count ? [`⚠ ${plural(d.error_count, 'error')}`] : [])], true, {unit: total === 1 ? 'transcript' : 'transcripts'});
    }
    if (!prog) return;
    if (d.phase === 'error') { prog.className = 'set-result err'; prog.textContent = '✕ ' + (d.error || 'Error'); return; }
    if (d.phase === 'complete' && d.gave_up) {
        prog.className = 'set-result err';
        prog.textContent = `✕ ${d.gave_up} — ${d.indexed || 0} embedded before`;
        return;
    }
    if (d.phase === 'complete') {
        prog.className = 'set-result ok';
        prog.textContent = `✓ Done — ${d.indexed || 0} embedded, ${d.skipped || 0} skipped, ${plural(d.chunks, 'chunk')}`
            + (d.error_count ? `, ${plural(d.error_count, 'error')}` : '') + ` (${d.elapsed_sec || 0} s)`;
        return;
    }
    if (d.phase === 'closed' || d.phase === 'started') return;
    const bits = [];
    if (d.message) bits.push(d.message);
    if (typeof d.done === 'number' && d.total) bits.push(`${d.done}/${d.total} (${Math.round(100 * d.done / d.total)} %)`);
    if (d.current) bits.push(String(d.current).split('/').pop());
    prog.className = 'set-result';
    if (bits.length) prog.textContent = '⟳ ' + bits.join(' · ');
}

async function _ragBuildStop() {
    try { await fetch('/api/rag/build/stop', {method: 'POST'}); _showToast('Stopping after the current transcript'); }
    catch (e) { _showToast(e.message, true); }
}

async function _ragClear(btn) {
    if (!confirm('Empty the vector index? All vectors are deleted — the transcripts stay, and the index can be built again.')) return;
    const label = btn.textContent; btn.disabled = true; btn.textContent = '⏳ …';
    try {
        const d = await fetch('/api/rag/clear', {method: 'POST'}).then(r => r.json());
        if (d.ok) _showToast('Index emptied');
        else _showToast(d.error || 'Error', true);
    } catch (e) { _showToast(e.message, true); }
    finally { btn.disabled = false; btn.textContent = label; _loadRagStatus(); }
}

function _ragFmtTime(sec) {
    sec = Math.max(0, Math.floor(sec || 0));
    const h = Math.floor(sec / 3600), m = Math.floor((sec % 3600) / 60), s = sec % 60;
    const p = (n) => String(n).padStart(2, '0');
    return h > 0 ? `${h}:${p(m)}:${p(s)}` : `${m}:${p(s)}`;
}

async function _ragSearch() {
    const q = (document.getElementById('ragSearchInput').value || '').trim();
    const mode = document.getElementById('ragSearchMode').value;
    const box = document.getElementById('ragSearchResults');
    if (!q) { box.innerHTML = ''; return; }
    box.innerHTML = '<div class="set-hint">Searching…</div>';
    try {
        const d = await fetch(`/api/rag/search?q=${encodeURIComponent(q)}&mode=${mode}`).then(r => r.json());
        if (d.error) { box.innerHTML = `<div class="set-note err">${esc(d.error)}</div>`; return; }
        if (!d.results || !d.results.length) { box.innerHTML = `<div class="set-hint">Nothing found (${d.elapsed_ms} ms)</div>`; return; }
        box.innerHTML = `<div class="set-hint" style="margin-bottom:6px">${plural(d.count, 'hit')} · ${d.elapsed_ms} ms · ${esc(d.mode)}</div>
            <div class="set-items">${d.results.map(r => {
                const name = String(r.transcript_key || '').split('/').pop();
                const how = r.match === 'hybrid' ? 'hybrid' : r.match === 'vector' ? 'vector' : 'full text';
                let content = String(r.content || '');
                if (content.length > 300) content = content.slice(0, 300) + '…';
                return `<div class="set-item"><div class="set-item-head">
                        <span class="set-item-name" title="${esc(r.transcript_key || '')}">📄 ${esc(name)} · ${r.speaker ? esc(r.speaker) + ' · ' : ''}${_ragFmtTime(r.start_sec)}</span>
                        <span class="set-item-meta">${how} · ${(r.score || 0).toFixed(4)}</span></div>
                    <div class="set-result">${esc(content)}</div></div>`;
            }).join('')}</div>`;
    } catch (e) {
        box.innerHTML = `<div class="set-note err">${esc(e.message)}</div>`;
    }
}

// ── MCP server: state, access token, setting up a client ─────────────────

let _mcpAccess = null;           // GET /api/mcp/access
let _mcpTokenVisible = false;    // the token in plain text on screen
let _mcpSnippets = [];           // what each "Copy" copies — with the real token

async function _mcpAccessLoad() {
    const status = document.getElementById('mcpStatus');
    if (!status) return;
    try {
        const r = await fetch('/api/mcp/access');
        const d = await r.json().catch(() => ({}));
        if (!r.ok || d.error) throw new Error(d.error || `HTTP ${r.status}`);
        _mcpAccess = d;
    } catch (e) {
        status.innerHTML = `<div class="set-note err">State not available: ${esc(e.message)}</div>`;
        return;
    }
    _mcpRender();
}

// The address a client uses. Listening on every interface: this machine as
// the browser reaches it — the same host, another port.
function _mcpHost() {
    const h = String(_mcpAccess?.host || '127.0.0.1');
    if (['', '0.0.0.0', '::', '[::]'].includes(h)) return location.hostname || '127.0.0.1';
    return h === 'localhost' ? '127.0.0.1' : h;
}

function _mcpRender() {
    const a = _mcpAccess;
    if (!a) return;
    const host = _mcpHost();
    const svc = Object.fromEntries((a.services || []).map(s => [s.key, s]));
    const urlOf = (s) => s ? `http://${host}:${s.port}${s.path || ''}` : '';

    const status = document.getElementById('mcpStatus');
    if (status) {
        const state = (s) => s.listening
            ? `<span class="set-dot ok"></span>Running — <span class="mono">${esc(urlOf(s))}</span>`
            : s.enabled
                ? `<span class="set-dot err"></span>Switched on, but nothing answers on port ${s.port} — restart the app, or look for the error in its log`
                : '<span class="set-dot"></span>Off';
        status.innerHTML = `<div class="set-lines">
                <span>fastmcp</span><span>${_hwData?.has_fastmcp
                    ? `<span class="set-badge ok">✓ installed${_hwData.fastmcp_version ? ' ' + esc(_hwData.fastmcp_version) : ''}</span>`
                    : `<span class="set-badge miss">✕ missing</span> — without it the MCP server can't run: ${_sCode('pip install fastmcp')}`}</span>
                ${(a.services || []).map(s => `<span>${esc(s.name)}</span><span>${state(s)}</span>`).join('')}
                <span>Access</span><span>${a.required
                    ? '<span class="set-dot ok"></span>Only with the access token'
                    : '<span class="set-dot warn"></span>Open — anyone who reaches the ports'}</span>
            </div>`;
    }

    const tok = document.getElementById('mcpToken');
    if (tok && a.token_hidden) {
        tok.innerHTML = `<div class="set-hint">The access token is only shown in a browser on the server itself,
            or with the login switched on. Open these settings there to see, copy or renew it.</div>`;
    } else if (tok) {
        const token = String(a.token || '');
        tok.innerHTML = `<div class="set-token">
                <input class="set-input mono" readonly aria-label="Access token"
                       value="${esc(_mcpTokenVisible ? token : '•'.repeat(Math.min(token.length || 32, 43)))}"
                       ${_mcpTokenVisible ? 'onclick="this.select()"' : ''}>
                <button type="button" class="set-btn" onclick="_mcpTokenToggle()">${_mcpTokenVisible ? 'Hide' : 'Show'}</button>
                <button type="button" class="set-btn" onclick="_sCopy(_mcpAccess.token, this)">📋 Copy</button>
                <button type="button" class="set-btn danger" onclick="_mcpTokenRenew(this)">↻ New token</button>
            </div>
            <div class="set-hint" style="margin-top:6px">${a.source === 'config'
                ? `Your own token, from config.yaml (${_sCode('mcp.token')}); “New token” replaces it with a generated one.`
                : `Made by the app, kept in <span class="mono">${esc(a.file || '')}</span> — readable only by you. For a token of your own, set ${_sCode('mcp.token')} in config.yaml.`}
                A new token stops the old one at once: every client then needs the new one.</div>`;
    }

    const box = document.getElementById('mcpClients');
    if (!box) return;
    // Hidden token (another machine, no login): a placeholder keeps the
    // snippets complete — the user fills in the token from the server.
    const T = String(a.token || (a.token_hidden && a.required ? '<access token>' : ''));
    const bearer = a.required && T ? `Bearer ${T}` : '';
    const http = svc.http, oa = svc.openapi;
    const items = [
        {title: 'Claude Code', need: http,
         desc: 'One command in a terminal.',
         text: `claude mcp add --transport http whisper-tm ${urlOf(http)}` + (bearer ? ` --header "Authorization: ${bearer}"` : '')},
        {title: 'JSON configuration', need: http,
         desc: 'Cursor, LM Studio, a project\'s .mcp.json and other clients with a JSON file. Some call the outer key “servers”.',
         text: JSON.stringify({mcpServers: {'whisper-tm': {type: 'http', url: urlOf(http),
                               ...(bearer ? {headers: {Authorization: bearer}} : {})}}}, null, 2)},
        {title: 'Open WebUI', need: oa,
         desc: 'Settings → Tools → add a tool server: this address, authentication “Bearer”, the token as key.',
         text: `${urlOf(oa)}` + (bearer ? `\nBearer ${T}` : '')},
        {title: 'A client on this machine (stdio)',
         desc: 'The client starts the server itself — no port, no token. The paths are this installation\'s.',
         text: JSON.stringify({mcpServers: {'whisper-tm': {command: a.python || 'python',
                               args: ['-m', 'whisper_tm.mcp_server'],
                               env: {PYTHONPATH: a.package_root || '', ...(a.config ? {WHISPER_TM_CONFIG: a.config} : {})}}}}, null, 2)},
    ];
    _mcpSnippets = items.map(it => it.text);
    const shown = (text) => (_mcpTokenVisible || !T || a.token_hidden) ? text : text.split(T).join('•'.repeat(12));
    box.innerHTML = items.map((it, i) => `<div class="set-snippet">
            <div class="set-snippet-head">
                <div><b>${esc(it.title)}</b><div class="set-hint">${esc(it.desc)}</div></div>
                <button type="button" class="set-btn sm" onclick="_sCopy(_mcpSnippets[${i}], this)">📋 Copy</button>
            </div>
            ${it.need && !it.need.enabled ? `<div class="set-note warn">Needs “${esc(it.need.name)}”, which is off — switch it on under Network services.</div>` : ''}
            <pre class="set-code">${esc(shown(it.text))}</pre>
        </div>`).join('');
}

function _mcpTokenToggle() {
    _mcpTokenVisible = !_mcpTokenVisible;
    _mcpRender();
}

async function _mcpTokenRenew(btn) {
    if (!confirm('Make a new access token? The old one stops working at once — every client needs the new one.')) return;
    btn.disabled = true;
    try {
        const r = await fetch('/api/mcp/access/renew', {method: 'POST'});
        const d = await r.json().catch(() => ({}));
        if (!r.ok || !d.token) throw new Error(d.error || `HTTP ${r.status}`);
        _mcpAccess = {..._mcpAccess, token: d.token, source: 'file'};
        _mcpTokenVisible = true;
        _showToast('New token — give it to your clients');
    } catch (e) {
        _showToast('No new token: ' + e.message, true);
    } finally {
        btn.disabled = false;
        _mcpRender();
    }
}

// ── Notifications: the done sound ────────────────────────────────────────

async function _loadSoundList() {
    const sel = document.getElementById('settingsDoneSound');
    if (!sel) return;
    try {
        const d = await fetch('/api/sounds').then(r => r.json());
        sel.innerHTML = (d.sounds || []).map(s => `<option value="${esc(s.name)}" ${s.name === d.current ? 'selected' : ''}>
            ${esc(s.builtin ? 'Built-in sound' : s.name.replace(/^custom\//, ''))}</option>`).join('');
    } catch (e) { /* the list stays as it is */ }
}

function _soundUrl(name) {
    return name === 'done.mp3' ? '/static/done.mp3' : `/api/sounds/play/${encodeURIComponent(name)}`;
}

async function _selectDoneSound(name) {
    try {
        const r = await fetch('/api/sounds/select', {
            method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({name})});
        const d = await r.json().catch(() => ({}));
        if (!r.ok || d.ok === false) throw new Error(d.error || `HTTP ${r.status}`);
        // The page plays from a cached address: without this, the choice
        // only took effect after a reload
        if (typeof _customSoundUrl !== 'undefined') _customSoundUrl = _soundUrl(name);
        _showToast('Done sound chosen');
    } catch (e) {
        _showToast('Not saved: ' + e.message, true);
        _loadSoundList();
    }
}

function _previewDoneSound() {
    const name = document.getElementById('settingsDoneSound')?.value || 'done.mp3';
    const a = new Audio(_soundUrl(name));
    a.volume = 0.5;               // as loud as when a job finishes
    a.play().catch(() => {});
}

async function _uploadDoneSound(input) {
    const file = input.files?.[0];
    if (!file) return;
    const form = new FormData();
    form.append('file', file);
    try {
        const d = await fetch('/api/sounds/upload', {method: 'POST', body: form}).then(r => r.json());
        if (!d.ok) throw new Error(d.error || 'Upload failed');
        await _loadSoundList();
        const sel = document.getElementById('settingsDoneSound');
        if (sel) { sel.value = d.name; _selectDoneSound(d.name); }
    } catch (e) {
        _showToast(e.message, true);
    }
    input.value = '';
}

// The on/off switch is the bell in the top bar (state.js, per browser)
function _soundSwitchSync() {
    const box = document.getElementById('setSoundOn');
    if (box && typeof _soundMuted !== 'undefined') box.checked = !_soundMuted;
}

function _setSoundOn(on) {
    if (typeof _soundMuted !== 'undefined' && _soundMuted === on && typeof toggleDoneSound === 'function') toggleDoneSound();
    _soundSwitchSync();
}

// ── Database & backups ───────────────────────────────────────────────────

async function _dbStatusLoad() {
    const box = document.getElementById('dbStatus');
    if (!box) return;
    let s;
    try { s = await fetch('/api/db/stats').then(r => r.json()); } catch (e) {
        box.innerHTML = `<div class="set-note err">${esc(e.message)}</div>`;
        return;
    }
    const f = s.files || {}, t = s.transcripts || {};
    const kpi = (label, value, sub = '') => `<div class="set-kpi"><div class="set-kpi-label">${label}</div>
        <div class="set-kpi-value">${value}</div>${sub ? `<div class="set-kpi-sub">${sub}</div>` : ''}</div>`;
    box.innerHTML = (s.error ? `<div class="set-note err" style="margin-bottom:10px">${esc(s.error)}</div>` : '') + `
        <div class="set-kpis">
            ${kpi('Size', _fmtBytesS(s.db_size_bytes || 0))}
            ${kpi('Transcripts', (t.total || 0).toLocaleString())}
            ${kpi('Segments', (t.total_segments || 0).toLocaleString(), 'in the search index')}
            ${kpi('Words', (t.total_words || 0).toLocaleString())}
            ${kpi('Media files', (f.total_files || 0).toLocaleString(), `${(f.done || 0).toLocaleString()} done`)}
        </div>
        <div class="set-lines" style="margin-top:12px">
            <span>File</span><span class="mono">${esc(_settingsData?.db_path || '')}</span>
        </div>`;
}

/* The state of the backup folder: reachable, the next backup, the last.
 * Reachability first — an unmounted NAS is the likeliest reason nothing
 * arrives, and it should show without a look into the log. */
async function _loadBackupStatus() {
    const box = document.getElementById('backupStatusBox');
    if (!box) return;
    try {
        const d = await fetch('/api/db/backup_status').then(r => r.json());
        if (d.error) { box.textContent = 'Error: ' + d.error; return; }
        const btn = document.getElementById('btnBackupNow');
        if (btn) {
            btn.disabled = !!d.running || (d.path && d.reachable === false);
            btn.title = d.running ? 'A backup is running'
                : d.path && d.reachable === false ? 'The backup folder can\'t be reached'
                : d.path ? 'Into the backup folder' : 'Into backups/ next to the database — no backup folder set';
        }
        if (!d.path) {
            box.innerHTML = '<span class="set-dot warn"></span>No backup folder — backups only go next to the database, on the same disk.';
            return;
        }
        const parts = [d.reachable === false
            ? `<span class="set-dot err"></span>The backup folder can't be reached (${esc(d.path)})`
            : '<span class="set-dot ok"></span>Backup folder reachable'];
        parts.push(d.every_changes > 0
            ? `next backup in <b>${Math.max(0, d.every_changes - d.counter)}</b> change${d.every_changes - d.counter === 1 ? '' : 's'} (${d.counter} of ${d.every_changes})`
            : 'no backups after changes (0)');
        if (d.running) parts.push('⏳ a backup is running');
        if (d.last) {
            const when = d.last_ts ? new Date(d.last_ts * 1000).toLocaleString() : '';
            parts.push(`last: ${esc(d.last)} (${_fmtBytesS(d.last_size_bytes || 0)}${when ? ', ' + when : ''}) · ${d.backups} kept`);
        } else if (d.reachable) parts.push('no backup there yet');
        box.innerHTML = parts.join(' · ');
    } catch (e) {
        box.textContent = 'Status not available: ' + e.message;
    }
}

async function _dbAction(action, btn) {
    const confirmText = {
        rebuild: 'Rebuild the full-text index? Takes minutes with a large library; the search is incomplete meanwhile.',
        vacuum: 'Compact the database? It backs up first; the database is busy for minutes — queued jobs wait meanwhile.',
    }[action];
    if (confirmText && !confirm(confirmText)) return;
    const urls = {sync: '/api/db/sync', rebuild: '/api/db/rebuild_index', vacuum: '/api/db/vacuum', backup: '/api/db/backup'};
    const label = btn.textContent;
    btn.disabled = true;
    btn.textContent = '⏳ …';
    try {
        const r = await fetch(urls[action], {method: 'POST'});
        const d = await r.json().catch(() => ({}));
        if (!r.ok || d.error) { _showToast('Failed: ' + (d.error || `HTTP ${r.status}`), true); return; }
        if (action === 'sync') {
            _showToast(`Imported ${d.imported || 0}, skipped ${d.skipped || 0}`
                + (d.folders > 1 ? ` from ${d.folders} folders` : '')
                + (d.errors ? `, ${plural(Array.isArray(d.errors) ? d.errors.length : d.errors, 'error')}` : ''), !!d.errors?.length);
        } else if (action === 'backup') {
            _showToast(`Backed up: ${d.path || d.backup} (${_fmtBytesS(d.size_bytes || 0)})${d.backups ? ` · ${d.backups} kept` : ''}`);
            _loadBackupStatus();
        } else if (action === 'vacuum') {
            _showToast(`Compacted${d.saved_bytes > 0 ? ` — ${_fmtBytesS(d.saved_bytes)} freed` : ''}`);
        } else {
            _showToast('Done');
        }
        _dbStatusLoad();
    } catch (e) {
        _showToast('Failed: ' + e.message, true);
    } finally {
        btn.disabled = false;
        btn.textContent = label;
    }
}

async function _deleteDb(btn) {
    if (!confirm('Really delete the database?\n\nTranscripts, segments, search index, queue, summaries, translations, edits and job history in it are deleted. JSON transcripts on disk stay (in "Database only" mode there are none).\n\nA backup copy is made first; without one nothing is deleted.')) return;
    if (!confirm('Last chance — delete the database now?')) return;
    const label = btn.textContent;
    btn.disabled = true;
    btn.textContent = '⏳ …';
    try {
        const d = await fetch('/api/db/delete', {method: 'POST'}).then(r => r.json());
        if (d.ok) { _showToast('Database deleted and created empty — the backup: ' + (d.backup || '?')); _dbStatusLoad(); }
        else _showToast('Failed: ' + (d.error || 'unknown'), true);
    } catch (e) {
        _showToast('Failed: ' + e.message, true);
    } finally {
        btn.disabled = false;
        btn.textContent = label;
    }
}

// ── Diagnostics: job logs ────────────────────────────────────────────────

let _loadLogsSeq = 0;  // only the newest answer is drawn
async function _loadLogs() {
    const filter = document.getElementById('logExplorerFilter')?.value || '';
    const seq = ++_loadLogsSeq;
    const el = () => document.getElementById('logExplorerList');
    if (el()) el().innerHTML = '<div class="set-browser-msg">Loading…</div>';
    try {
        const d = await fetch(`/api/logs?q=${encodeURIComponent(filter)}`).then(r => r.json());
        if (seq !== _loadLogsSeq || !el()) return;
        const logs = d.logs || [];
        if (!logs.length) { el().innerHTML = '<div class="set-browser-msg">No logs found</div>'; return; }
        el().innerHTML = `<div class="set-log-row"><span class="set-log-meta" style="flex:1">${logs.length === 200 ? 'The newest 200' : logs.length} log${logs.length === 1 ? '' : 's'}</span>
                <button type="button" class="set-btn sm danger" onclick="_deleteAllLogs(this)">🗑 Delete all</button></div>`
            + logs.map(l => `<div class="set-log-row">
                <button type="button" class="set-log-name" onclick="_viewLog('${escJs(l.filename)}')" title="Open">${esc(l.filename)}</button>
                <span class="set-log-meta">${_fmtBytesS(l.size)} · ${new Date(l.modified * 1000).toLocaleString()}</span>
                <button type="button" class="set-btn sm" onclick="_deleteLog('${escJs(l.filename)}', this)" title="Delete">✕</button>
            </div>`).join('');
    } catch (e) {
        if (seq === _loadLogsSeq && el()) el().innerHTML = `<div class="set-browser-msg">Error: ${esc(e.message)}</div>`;
    }
}

async function _deleteLog(filename, btn) {
    try {
        const r = await fetch(`/api/logs/delete/${encodeURIComponent(filename)}`, {method: 'POST'});
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        btn.closest('.set-log-row')?.remove();
    } catch (e) {
        _showToast('Not deleted: ' + e.message, true);
    }
}

async function _deleteAllLogs(btn) {
    if (!confirm('Delete all job log files?')) return;
    btn.disabled = true;
    try {
        const d = await fetch('/api/logs/delete_all', {method: 'POST'}).then(r => r.json());
        _showToast(`${plural(d.deleted, 'log')} deleted`);
        _loadLogs();
    } catch (e) {
        _showToast('Failed: ' + e.message, true);
        btn.disabled = false;
    }
}

async function _viewLog(filename) {
    try {
        const r = await fetch(`/api/logs/view/${encodeURIComponent(filename)}`);
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        const d = await r.json();
        const w = window.open('', '_blank', 'width=900,height=650');
        if (!w) { _showToast('The browser blocked the window — allow pop-ups for this page', true); return; }
        const e = (s) => String(s).replace(/[<>&"']/g, c => ({'<': '&lt;', '>': '&gt;', '&': '&amp;', '"': '&quot;', "'": '&#39;'}[c]));
        w.document.write(`<html><head><title>${e(filename)}</title>
            <style>body{background:#0d0d14;color:#e4e4ee;font-family:ui-monospace, SFMono-Regular, SF Mono, Menlo, Consolas, Liberation Mono, DejaVu Sans Mono, monospace;
            font-size:12px;padding:16px;white-space:pre-wrap;line-height:1.6}</style></head>
            <body>${(d.lines || []).map(e).join('\n')}</body></html>`);
    } catch (e) {
        _showToast('Log not readable: ' + e.message, true);
    }
}

// ── Diagnostics: database explorer ───────────────────────────────────────

let _dbExpOffset = 0;
const _dbExpLimit = 50;
let _dbExpSeq = 0;  // only the newest answer is drawn

// The explorer opened: the chosen table's first rows. It stayed empty until
// another table was picked or Enter pressed in the search.
function _dbExplorerOpened() {
    const out = document.getElementById('dbExploreResult');
    if (out && !out.innerHTML.trim()) _loadDbExplorer();
}

function _loadDbExplorer() {
    _dbExpOffset = 0;
    _fetchDbExplorer();
}

function _dbExpPage(dir) {
    _dbExpOffset = Math.max(0, _dbExpOffset + dir * _dbExpLimit);
    _fetchDbExplorer();
}

async function _fetchDbExplorer() {
    const table = document.getElementById('dbExploreTable')?.value || 'transcripts';
    const search = document.getElementById('dbExploreSearch')?.value || '';
    const el = document.getElementById('dbExploreResult');
    const info = document.getElementById('dbExploreInfo');
    if (!el) return;
    const seq = ++_dbExpSeq;
    el.innerHTML = '<div class="set-browser-msg">Loading…</div>';
    try {
        const params = new URLSearchParams({table, limit: _dbExpLimit, offset: _dbExpOffset, search});
        const d = await fetch(`/api/db/explore?${params}`).then(r => r.json());
        if (seq !== _dbExpSeq) return;
        if (d.error) { el.innerHTML = `<div class="set-browser-msg">${esc(d.error)}</div>`; return; }
        const pages = Math.max(1, Math.ceil(d.total / _dbExpLimit));
        if (info) info.textContent = `${plural(d.total, 'row')} · page ${Math.floor(_dbExpOffset / _dbExpLimit) + 1} of ${pages.toLocaleString()}`;
        if (!d.rows || !d.rows.length) { el.innerHTML = '<div class="set-browser-msg">No rows</div>'; return; }
        const cols = d.columns || [];
        el.innerHTML = '<table class="set-table mono"><thead><tr>'
            + cols.map(c => `<th>${esc(c)}</th>`).join('') + '</tr></thead><tbody>'
            + d.rows.map(row => '<tr>' + cols.map(c => {
                let v = row[c];
                const full = esc(String(v ?? ''));
                if (v === null || v === undefined) v = '<span style="color:var(--tx-3)">null</span>';
                else if (typeof v === 'string' && v.length > 80) v = esc(v.slice(0, 77)) + '…';
                else if (typeof v === 'number' && !Number.isInteger(v)) v = v.toFixed(2);
                else v = esc(String(v));
                return `<td title="${full}">${v}</td>`;
            }).join('') + '</tr>').join('') + '</tbody></table>';
    } catch (e) {
        if (seq === _dbExpSeq) el.innerHTML = `<div class="set-browser-msg">Error: ${esc(e.message)}</div>`;
    }
}

// ── Diagnostics: debug output (this browser) ─────────────────────────────

// _wxpDebugEnabled lives in core.js; this switch only changes the browser
function _toggleDebugMode() {
    const on = !!document.getElementById('debugModeToggle')?.checked;
    _wxpDebugEnabled = on;
    try { localStorage.setItem('wtm_debug', on ? '1' : ''); } catch (e) { /* storage off */ }
    if (on) wtmDebug('Debug', 'Browser debug output on');
}

function _debugToggleSync() {
    const box = document.getElementById('debugModeToggle');
    if (box) box.checked = !!_wxpDebugEnabled;
}

// ── About ────────────────────────────────────────────────────────────────

function _aboutRender() {
    const box = document.getElementById('aboutInfo');
    if (!box || !_settingsData) return;
    const cfg = _settingsData.config || {};
    const v = _settingsData.version || {};
    const login = cfg.auth?.enabled;
    box.innerHTML = `<div class="set-lines">
            <span>Version</span><span>Whisper Transcription Manager ${esc(v.display || '?')}</span>
            <span>Configuration</span><span class="mono">${esc(_settingsData.config_path || 'built-in defaults — nothing saved yet')}</span>
            <span>Database</span><span class="mono">${esc(_settingsData.db_path || '')}</span>
            ${_settingsData.data_dir ? `<span>Data folder</span><span class="mono">${esc(_settingsData.data_dir)}</span>` : ''}
            <span>Server</span><span class="mono">${esc(cfg.server?.host || '')}:${esc(String(cfg.server?.port || ''))}</span>
            <span>Login</span><span>${login ? '<span class="set-dot ok"></span>on' : '<span class="set-dot warn"></span>off — anyone who reaches the server can use it'} <span class="set-hint">(auth in config.yaml)</span></span>
        </div>
        <div class="set-hint" style="margin-top:8px">Please include the version with an error report.</div>`;
}

// ── Danger zone ──────────────────────────────────────────────────────────

async function resetAllSettings() {
    if (!confirm('Reset the settings to their defaults?\n\nKept: folders and database location, server address, login and allowed folders, credentials, LLM providers, templates and vocabularies. Transcripts and the database are not touched.')) return;
    try {
        const d = await fetch('/api/settings/reset', {method: 'POST'}).then(r => r.json());
        if (!d.ok) throw new Error(d.error || 'unknown');
        _showToast('Settings reset — reloading…');
        _settingsExtraDirty.clear();
        setTimeout(() => location.reload(), 1200);
    } catch (e) {
        _showToast('Reset failed: ' + e.message, true);
    }
}

function resetBrowserData() {
    // The queue lives on the server and is not touched here
    if (!confirm('Forget what this browser remembers?\n\nFavourite folders, the done-sound switch, folded cards, panel sizes and other view choices. The queue, transcripts and server settings stay.')) return;
    try {
        // Everything this origin stored — the keys grew up with several
        // prefixes (wtm_, wtm, sv, favDirs, voice map memos, setFold:), and
        // a prefix filter used to miss most of them
        localStorage.clear();
        try { sessionStorage.clear(); } catch (e) { /* storage off */ }
        _showToast('Forgotten — reloading…');
        setTimeout(() => location.reload(), 800);
    } catch (e) {
        _showToast('Failed: ' + e.message, true);
    }
}

async function cleanWorkdir() {
    if (!confirm('Empty the work folder?\n\nThe temporary files of finished and cancelled jobs are deleted. While a job is running, nothing is deleted.')) return;
    try {
        const r = await fetch('/api/clean_workdir', {method: 'POST'});
        const d = await r.json().catch(() => ({}));
        if (!r.ok || d.ok === false) throw new Error(d.error || `HTTP ${r.status}`);
        _showToast(typeof d.removed === 'number' ? `Work folder emptied — ${plural(d.removed, 'entry', 'entries')} removed` : 'Work folder emptied');
    } catch (e) {
        _showToast('Not emptied: ' + e.message, true);
    }
}
