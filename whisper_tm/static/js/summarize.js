// ── LLM Summarize ──
// The dialog (index.html #summarizeModal): on the left what the LLM gets —
// the source, the template, an instruction for this run — on the right what
// comes back: the result, the exact LLM input, the saved summaries.
//
// A template is the complete system prompt (config llm.presets). It is
// always in view as it is sent; changing it marks it "edited" for this run,
// "Save as template" keeps it. In earlier versions a preset swapped the system
// prompt behind a closed toggle, so what was sent could not be seen.

let _sumFilename = null;
let _sumFullText = '';       // the run's result (markdown source)
let _sumTokenData = null;
let _sumLlmStatus = null;    // cached /api/llm/status
let _sumSources = [];        // [{language, segments}] — translation tracks
let _sumSourceSegs = null;   // segments of the active source (null = original)
let _sumTracker = null;      // llmQueueTrack handle
let _sumQueueId = null;
let _sumSaved = [];          // saved summaries of the open file
let _sumShown = -1;          // saved summary shown in the result (-1 = the run's)
let _sumTabNow = 'result';
let _sumDebugTimer = null;

// The template being worked on. Lives across closing and reopening (Esc
// must never throw an edited prompt away) and across files: a prompt is
// worked on once and used on many recordings.
const _sumTpl = {name: null, text: null, undo: null};
const SUM_DEFAULT = '__default__';   // settings' llm.system_prompt, when no template has its text
const SUM_PREFS = 'wtm_sumPrefs';    // per browser: last template, source options

function openSummarizeModal(filename) {
    _sumFilename = filename;
    _sumFullText = '';
    _sumTokenData = null;
    _sumSourceSegs = null;
    _sumShown = -1;
    _sumSaved = [];
    document.getElementById('sumFileName').textContent = filename || '';
    document.getElementById('sumFileName').title = filename || '';
    document.getElementById('sumError').innerHTML = '';
    document.getElementById('sumStats').innerHTML = '';
    document.getElementById('sumProgress').hidden = true;
    document.getElementById('btnSumStop').hidden = true;
    document.getElementById('btnSumSave').hidden = true;
    document.getElementById('sumTokenInfo').innerHTML = '';
    document.getElementById('sumSaveTpl').hidden = true;
    const go = document.getElementById('btnSumGo');
    go.disabled = false;
    go.textContent = '🤖 Summarize';
    _sumPrefsApply();
    _sumResultEmpty();
    _sumSavedRender();
    _sumTab('result');
    document.getElementById('summarizeModal').classList.add('visible');
    _loadLlmStatus();
    _loadSavedSummaries(filename);
    _loadSumSources(filename);
    _fetchTokenEstimate();
    // Is a summary already running for this file? Then attach to it.
    llmQueueOpen('summary', filename).then(e => {
        if (e) _sumFollow(e.id);
    });
}

function closeSummarizeModal() {
    // Only detach the display: the summary keeps running in the LLM
    // queue and saves itself.
    if (_sumTracker) { _sumTracker.stop(); _sumTracker = null; }
    _sumPrefsSave();
    document.getElementById('summarizeModal').classList.remove('visible');
    // A summary saved in the dialog shows in the open Summary tab: its list
    // was drawn before, and only the count on the tab followed
    if (typeof _txView !== 'undefined' && _txView.mode === 'summary'
        && typeof _sumViewRender === 'function') _sumViewRender();
}

// ── Remembered per browser: last template and source options ──

function _sumPrefsSave() {
    try {
        localStorage.setItem(SUM_PREFS, JSON.stringify({
            template: _sumTpl.name,
            speakers: document.getElementById('sumSpeakers').checked,
            timestamps: document.getElementById('sumTimestamps').checked,
            meta: [...document.querySelectorAll('#sumMeta .sum-toggle')]
                .filter(b => b.getAttribute('aria-pressed') === 'true')
                .map(b => b.dataset.meta),
        }));
    } catch (e) { /* private mode: defaults next time */ }
}

function _sumPrefs() {
    try { return JSON.parse(localStorage.getItem(SUM_PREFS) || 'null') || {}; }
    catch (e) { return {}; }
}

function _sumPrefsApply() {
    const p = _sumPrefs();
    if (typeof p.speakers === 'boolean')
        document.getElementById('sumSpeakers').checked = p.speakers;
    if (typeof p.timestamps === 'boolean')
        document.getElementById('sumTimestamps').checked = p.timestamps;
    if (Array.isArray(p.meta)) {
        document.querySelectorAll('#sumMeta .sum-toggle').forEach(b =>
            b.setAttribute('aria-pressed', String(p.meta.includes(b.dataset.meta))));
    }
}

// ══════════════════════════════════════════════════════════════
//  Source: original or a translation track
//  The LLM gets exactly the language version chosen here —
//  timings/speakers always come from the original.
// ══════════════════════════════════════════════════════════════

let _sumSourcesSeq = 0;

async function _loadSumSources(filename) {
    const sel = document.getElementById('sumSource');
    if (!sel) return;
    const seq = ++_sumSourcesSeq;
    const origLang = (activeTranscript?.language || '').toUpperCase();
    sel.innerHTML = `<option value="">📄 Original transcript${origLang ? ` (${origLang})` : ''}</option>`;
    try {
        const r = await fetch('/api/translations/' + encodeURIComponent(filename));
        const tracks = (await r.json()).translations || [];
        // The dialog was closed or reopened for another file meanwhile:
        // these tracks are not that file's.
        // (Or the same file twice: a second open already refilled the list.)
        if (seq !== _sumSourcesSeq || filename !== _sumFilename) return;
        _sumSources = tracks;
        _sumSources.forEach(t => {
            const o = document.createElement('option');
            o.value = t.language;
            o.textContent = `🌐 ${t.language} (${t.segments ?? '?'} segment${t.segments === 1 ? '' : 's'})`;
            sel.appendChild(o);
        });
    } catch (e) { if (seq === _sumSourcesSeq) _sumSources = []; }
}

async function _onSumSourceChange() {
    const lang = document.getElementById('sumSource').value;
    if (!lang) {
        _sumSourceSegs = null;
        _fetchTokenEstimate();
        _sumInputsChanged();
        return;
    }
    try {
        const r = await fetch('/api/translation/' + encodeURIComponent(lang)
            + '/' + encodeURIComponent(_sumFilename));
        const d = await r.json();
        if (d.error) throw new Error(d.error);
        // Lay the translated text over the original structure: timings and
        // speakers stay, only the text comes from the track.
        const map = {};
        (d.segments || []).forEach(t => { map[t.id] = t.text; });
        _sumSourceSegs = (activeTranscript?.segments || []).map((s, i) => ({
            ...s, text: map[i] ?? s.text, words: undefined,
        }));
        _fetchTokenEstimate();
        _sumInputsChanged();
    } catch (e) {
        Toast.error('Translation could not be loaded: ' + e.message);
        document.getElementById('sumSource').value = '';
        _sumSourceSegs = null;
    }
}

// Segments that actually go to the LLM
function _sumSegments() {
    return _sumSourceSegs || activeTranscript?.segments || [];
}

// The name the file view shows for a speaker label. The LLM gets it too:
// a speaker named in the file view stayed "SPEAKER_00" in the summary.
function _sumSpeakerName(label) {
    return label && typeof getSpeakerName === 'function' ? getSpeakerName(label) : label;
}

// What the estimate and the debug view POST: only the fields the server
// formats (text, speaker, times). The per-word arrays made a 24 h recording
// a multi-MB upload on every typing pause.
function _sumPostSegments() {
    return _sumSegments().map(s => ({
        start: s.start, end: s.end, start_fmt: s.start_fmt, end_fmt: s.end_fmt,
        speaker: _sumSpeakerName(s.speaker), text: s.text,
    }));
}

// Metadata toggles: one click, no hidden panel
document.addEventListener('click', e => {
    const b = e.target.closest('#sumMeta .sum-toggle');
    if (!b) return;
    b.setAttribute('aria-pressed', String(b.getAttribute('aria-pressed') !== 'true'));
    _sumInputsChanged();
});

// Ctrl+Enter anywhere in the dialog starts the summary
document.addEventListener('keydown', e => {
    if (e.key !== 'Enter' || !(e.ctrlKey || e.metaKey)) return;
    if (!document.getElementById('summarizeModal')?.classList.contains('visible')) return;
    e.preventDefault();
    if (!document.getElementById('btnSumGo').disabled) doSummarize();
});

// Anything the LLM input depends on changed
function _sumInputsChanged() {
    _updateTokenEstimate();
    _sumHints();
    if (_sumTabNow === 'input') {
        clearTimeout(_sumDebugTimer);
        _sumDebugTimer = setTimeout(_refreshDebug, 500);
    }
}

// ══════════════════════════════════════════════════════════════
//  Templates + models
// ══════════════════════════════════════════════════════════════

async function _loadLlmStatus() {
    try {
        const r = await fetch('/api/llm/status');
        _sumLlmStatus = await r.json();
    } catch(e) {
        _sumLlmStatus = {};
    }
    _populatePresets();
    _populateModels();
}

function _sumPresets() {
    const p = _sumLlmStatus?.presets;
    return (p && typeof p === 'object') ? p : {};
}

// The text a template name stands for
function _tplText(name) {
    if (name === SUM_DEFAULT) return _sumLlmStatus?.system_prompt || '';
    const t = _sumPresets()[name];
    return typeof t === 'string' ? t : '';
}

function _sumIsBuiltin(name) {
    return (_sumLlmStatus?.preset_builtin || []).includes(name);
}

function _populatePresets() {
    const sel = document.getElementById('sumPreset');
    const presets = _sumPresets();
    const names = Object.keys(presets).filter(n => typeof presets[n] === 'string');
    const builtin = (_sumLlmStatus?.preset_builtin || []).filter(n => names.includes(n));
    const own = names.filter(n => !builtin.includes(n))
                     .sort((a, b) => a.localeCompare(b));
    const defText = (_sumLlmStatus?.system_prompt || '').trim();
    const defIsTemplate = names.some(n => presets[n].trim() === defText);
    const opt = (value, label) =>
        `<option value="${esc(value)}">${esc(label)}</option>`;
    sel.innerHTML =
        (!defIsTemplate ? opt(SUM_DEFAULT, defText ? 'Default prompt (Settings)' : 'No system prompt') : '')
        + (builtin.length ? `<optgroup label="Built-in">${builtin.map(n => opt(n, n)).join('')}</optgroup>` : '')
        + (own.length ? `<optgroup label="Your templates">${own.map(n => opt(n, n)).join('')}</optgroup>` : '');

    const valid = n => n === SUM_DEFAULT ? !defIsTemplate : names.includes(n);
    const ta = document.getElementById('sumSysPrompt');
    if (_sumTpl.name !== null && valid(_sumTpl.name)) {
        // Back in the dialog: the template and the text as they were left
        sel.value = _sumTpl.name;
        ta.value = _sumTpl.text ?? _tplText(_sumTpl.name);
    } else {
        // First open: the last template used here, else the settings' default
        const last = _sumPrefs().template;
        const byDefault = names.find(n => presets[n].trim() === defText);
        const pick = valid(last) ? last
                   : byDefault ?? (!defIsTemplate ? SUM_DEFAULT : names[0] ?? SUM_DEFAULT);
        sel.value = pick;
        _sumTpl.name = pick;
        ta.value = _tplText(pick);
    }
    _sumTplRefresh();
    _updateTokenEstimate();
}

function _onPresetChange() {
    const sel = document.getElementById('sumPreset');
    const ta = document.getElementById('sumSysPrompt');
    // Switching away from an edited prompt keeps it one step back —
    // a mis-click must not cost the work in it
    _sumTpl.undo = _sumEdited() ? {name: _sumTpl.name, text: ta.value} : null;
    _sumTpl.name = sel.value;
    ta.value = _tplText(sel.value);
    _sumTplRefresh();
    _sumPrefsSave();
    _sumInputsChanged();
}

function _sumEdited() {
    const ta = document.getElementById('sumSysPrompt');
    return _sumTpl.name !== null && ta.value.trim() !== _tplText(_sumTpl.name).trim();
}

function _sumSysInput() {
    _sumTplRefresh();
    _sumInputsChanged();
}

function _sumResetPrompt() {
    document.getElementById('sumSysPrompt').value = _tplText(_sumTpl.name);
    _sumTplRefresh();
    _sumInputsChanged();
}

function _sumUndoSwitch() {
    const u = _sumTpl.undo;
    if (!u) return;
    const sel = document.getElementById('sumPreset');
    if ([...sel.options].some(o => o.value === u.name)) {
        sel.value = u.name;
        _sumTpl.name = u.name;
    }
    document.getElementById('sumSysPrompt').value = u.text;
    _sumTpl.undo = null;
    _sumTplRefresh();
    _sumInputsChanged();
}

// Edited badge, reset/undo buttons, the character/token count
function _sumTplRefresh() {
    const ta = document.getElementById('sumSysPrompt');
    const edited = _sumEdited();
    _sumTpl.text = ta.value;
    document.getElementById('sumEdited').hidden = !edited;
    document.getElementById('btnSumReset').hidden = !edited;
    document.getElementById('btnSumUndo').hidden = !_sumTpl.undo;
    const chars = ta.value.length;
    document.getElementById('sumSysCount').textContent = chars
        ? `${plural(chars, 'character')} · ~${plural(Math.ceil(chars / 4), 'token')}`
        : 'empty — the LLM gets no system prompt';
    _sumHints();
}

// A template that works with speaker labels or timestamps while that
// source option is off would get nothing to work with
function _sumHints() {
    const box = document.getElementById('sumHints');
    if (!box) return;
    const text = (document.getElementById('sumSysPrompt').value + '\n'
                  + document.getElementById('sumPrompt').value).toLowerCase();
    const hints = [];
    if (/speaker label/.test(text) && !document.getElementById('sumSpeakers').checked)
        hints.push(['sumSpeakers', 'The instruction works with speaker labels, but Speakers is off.']);
    if (/timestamp/.test(text) && !document.getElementById('sumTimestamps').checked)
        hints.push(['sumTimestamps', 'The instruction works with timestamps, but Timestamps is off.']);
    box.innerHTML = hints.map(([id, msg]) =>
        `<div class="sum-hint"><span>💡 ${msg}</span>
            <button class="tx-act-btn" onclick="_sumTurnOn('${id}')">Turn on</button></div>`).join('');
}

function _sumTurnOn(id) {
    document.getElementById(id).checked = true;
    _sumInputsChanged();
}

// ── Save as template ──

function _sumSaveTplOpen() {
    const row = document.getElementById('sumSaveTpl');
    const input = document.getElementById('sumTplName');
    const n = _sumTpl.name;
    // Your own template: updating it is the likely wish. A built-in one:
    // a new name, so the shipped version stays as it is.
    input.value = (n && n !== SUM_DEFAULT && !_sumIsBuiltin(n)) ? n : '';
    row.hidden = false;
    _sumTplNameInput();
    input.focus();
    input.select();
}

function _sumSaveTplClose() {
    document.getElementById('sumSaveTpl').hidden = true;
}

function _sumTplNameInput() {
    const name = document.getElementById('sumTplName').value.trim();
    const note = document.getElementById('sumTplNote');
    const btn = document.getElementById('btnSumTplSave');
    const exists = Object.prototype.hasOwnProperty.call(_sumPresets(), name);
    const empty = !document.getElementById('sumSysPrompt').value.trim();
    btn.disabled = !name || empty;
    btn.textContent = exists ? 'Replace' : 'Save';
    note.classList.toggle('warn', exists || empty);
    note.textContent = empty ? 'The system prompt is empty — there is nothing to keep.'
        : exists ? `“${name}” exists — saving replaces its text.`
        : 'Kept in the settings (LLM → Summarization) and listed under “Your templates”.';
}

async function _sumSaveTpl() {
    const name = document.getElementById('sumTplName').value.trim();
    const text = document.getElementById('sumSysPrompt').value;
    if (!name || !text.trim()) return;
    const btn = document.getElementById('btnSumTplSave');
    btn.disabled = true;
    try {
        const r = await fetch('/api/llm/presets/save', {
            method: 'POST', headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({name, text})});
        const d = await r.json().catch(() => ({}));
        if (!r.ok || d.error) throw new Error(d.error || `HTTP ${r.status}`);
        _sumLlmStatus = {..._sumLlmStatus, presets: d.presets};
        _sumTpl.name = name;
        _sumTpl.text = text;
        _sumTpl.undo = null;
        _populatePresets();
        _sumSaveTplClose();
        _sumPrefsSave();
        Toast.success(`Template “${name}” ${d.replaced ? 'updated' : 'saved'}`);
    } catch (e) {
        Toast.error('Template not saved: ' + e.message);
    } finally {
        btn.disabled = false;
    }
}

function _populateModels() {
    const sel = document.getElementById('sumModel');
    const models = _sumLlmStatus?.models || {};
    const taskModel = _sumLlmStatus?.task_models?.summary || '';
    const before = sel.value;
    sel.innerHTML = '';

    // The default sends '' — then the provider assigned to summaries in the
    // settings (llm.tasks) decides, like in the correction and translation
    // dialogs (_llmModels in llmtools.js). Earlier versions sent the legacy
    // llm.model by name, and a named model pulls ITS provider along
    // (llm_client.build_chat_payload): the summary assignment was never used
    // from here. Always present — without it, the first model below would be
    // the silent default, with the same effect.
    const def = document.createElement('option');
    def.value = '';
    def.textContent = 'Default (task setting)' + (taskModel ? ': ' + taskModel : '');
    sel.appendChild(def);

    // The task's own model isn't listed a second time — the default runs it
    const inside = new Set(['', taskModel]);
    for (const [name, modelId] of Object.entries(models)) {
        if (typeof modelId !== 'string') continue;  // Skip broken entries
        if (inside.has(modelId)) continue;
        inside.add(modelId);
        const opt = document.createElement('option');
        opt.value = modelId;
        opt.textContent = name !== modelId ? `${name} (${modelId})` : name;
        sel.appendChild(opt);
    }
    // Default models from the provider cards (see _llmModels in llmtools.js)
    for (const pm of (_sumLlmStatus?.provider_models || [])) {
        if (!pm.model || inside.has(pm.model)) continue;
        inside.add(pm.model);
        const opt = document.createElement('option');
        opt.value = pm.model;
        opt.textContent = `${pm.model} (${pm.providers})`;
        sel.appendChild(opt);
    }
    // Reopened: keep the model chosen before
    if (before && [...sel.options].some(o => o.value === before)) sel.value = before;
}

// ══════════════════════════════════════════════════════════════
//  Metadata
// ══════════════════════════════════════════════════════════════

function _sumMetaOn(key) {
    return document.querySelector(`#sumMeta .sum-toggle[data-meta="${key}"]`)
        ?.getAttribute('aria-pressed') === 'true';
}

function _gatherMetadata() {
    const d = activeTranscript;
    if (!d) return {};
    const meta = {};
    if (_sumMetaOn('filename'))
        meta.filename = d.filename || _sumFilename || activeFile || '';
    if (_sumMetaOn('duration'))
        meta.duration = d.total_duration || '';
    if (_sumMetaOn('language'))
        meta.language = d.language || '';
    if (_sumMetaOn('model'))
        meta.model = d.model || '';
    if (_sumMetaOn('date')) {
        // activeFile first (see fileview.js): d.filename refers to the
        // root at transcription time, not the current one.
        const f = (typeof fileByName === 'function' ? fileByName(activeFile) : null)
               || files.find(x => x.name === activeFile)
               || files.find(x => x.name === d.filename);
        if (f?.tx_date) meta.date = new Date(f.tx_date * 1000).toLocaleString();
        else if (d.processing_time) meta.date = d.processing_time;
    }
    if (_sumMetaOn('speakers') && d.speakers?.length)
        meta.speakers = d.speakers.map(_sumSpeakerName).join(', ');
    return meta;
}

// ══════════════════════════════════════════════════════════════
//  Token estimate
// ══════════════════════════════════════════════════════════════

async function _fetchTokenEstimate() {
    if (!_sumSegments().length) return;
    const segments = _sumPostSegments();
    try {
        const r = await fetch('/api/llm/estimate', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({segments})
        });
        if (!r.ok) return;
        _sumTokenData = await r.json();
        _updateTokenEstimate();
    } catch(e) {}
}

function _updateTokenEstimate() {
    const el = document.getElementById('sumTokenInfo');
    if (!_sumTokenData) { el.innerHTML = ''; return; }
    const d = _sumTokenData;
    const speakers = document.getElementById('sumSpeakers').checked;
    const timestamps = document.getElementById('sumTimestamps').checked;

    let transcriptTokens;
    if (speakers && timestamps) transcriptTokens = d.with_both;
    else if (speakers) transcriptTokens = d.with_speakers;
    else if (timestamps) transcriptTokens = d.with_timestamps;
    else transcriptTokens = d.text_only;

    const promptText = document.getElementById('sumPrompt').value || '';
    const sysText = document.getElementById('sumSysPrompt')?.value || '';
    const promptTokens = Math.max(0, Math.ceil(promptText.length / 4));
    const sysTokens = Math.max(0, Math.ceil(sysText.length / 4));
    const meta = _gatherMetadata();
    const metaTokens = Math.ceil(JSON.stringify(meta).length / 4);

    const totalInput = sysTokens + transcriptTokens + promptTokens + metaTokens;
    // context_effective: detected server context (or config override)
    const ctxSize = d.context_effective || d.context_size;
    const maxOut = d.max_tokens;
    const fmtK = (n) => n >= 1000 ? (n / 1000).toFixed(1) + 'k' : n.toString();

    let html = `<span class="llm-chip" title="System prompt ~${fmtK(sysTokens)} · Transcript ~${fmtK(transcriptTokens)} · Metadata ~${fmtK(metaTokens)} · Instruction ~${fmtK(promptTokens)}">📊 <b>~${fmtK(totalInput)}</b> tokens in</span>`;
    if (ctxSize > 0) {
        const inputPct = (totalInput / ctxSize * 100);
        const free = ctxSize - totalInput;
        const color = inputPct < 50 ? 'var(--green)' : inputPct < 80 ? 'var(--orange)' : 'var(--red)';
        html += `<span class="llm-token-bar"><span class="llm-token-fill" style="width:${Math.min(100, inputPct)}%;background:${color}"></span></span>`;
        html += `<span class="llm-chip" style="color:${color}">${inputPct.toFixed(0)}% of ${fmtK(ctxSize)}</span>`;
        if (totalInput > ctxSize) {
            // No longer an error — the core automatically chunks (map-reduce)
            html += `<span class="llm-chip orange" title="Too long for one call: the transcript is condensed in sections first, then the template runs over the notes">⚙ In sections (map-reduce)</span>`;
        } else if (free < 1000) {
            html += `<span class="llm-chip orange">⚠ little room left for the response (~${fmtK(free)})</span>`;
        }
    } else {
        html += `<span class="llm-chip">max ${fmtK(maxOut)} response tokens</span>`;
    }
    el.innerHTML = html;
}

// ══════════════════════════════════════════════════════════════
//  Tabs: result · LLM input · saved
// ══════════════════════════════════════════════════════════════

function _sumTab(name) {
    _sumTabNow = name;
    for (const [tab, pane, key] of [['sumTabResult', 'sumPaneResult', 'result'],
                                    ['sumTabInput', 'sumPaneInput', 'input'],
                                    ['sumTabSaved', 'sumPaneSaved', 'saved']]) {
        const on = key === name;
        document.getElementById(tab).classList.toggle('active', on);
        document.getElementById(tab).setAttribute('aria-selected', String(on));
        document.getElementById(pane).hidden = !on;
    }
    document.getElementById('btnSumCopy').hidden = name === 'saved';
    if (name === 'input') _refreshDebug();
}

// Copy what the tab shows: the summary as markdown, or the LLM input
function _sumCopy() {
    const text = _sumTabNow === 'input'
        ? document.getElementById('sumDebugContent').textContent
        : (_sumShown >= 0 ? (_sumSaved[_sumShown]?.text || '') : _sumFullText);
    if (!text) { Toast.info('Nothing to copy yet'); return; }
    _copyText(text);
}

function _copyText(text) {
    if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(() => {
            if (typeof Toast !== 'undefined') Toast.success('Copied');
        });
    } else {
        const ta = document.createElement('textarea');
        ta.value = text;
        ta.style.cssText = 'position:fixed;left:-9999px';
        document.body.appendChild(ta);
        ta.select();
        document.execCommand('copy');
        document.body.removeChild(ta);
        if (typeof Toast !== 'undefined') Toast.success('Copied');
    }
}

function _sumResultEmpty() {
    document.getElementById('sumViewing').hidden = true;
    document.getElementById('sumResult').innerHTML = `<div class="sum-empty">
        <span class="large">🤖</span>
        <div>The summary appears here.<br>
        Choose a template on the left, adjust it if you like,
        and press <b>Summarize</b> (Ctrl+Enter).</div></div>`;
}

// ══════════════════════════════════════════════════════════════
//  LLM input: exactly what is sent
// ══════════════════════════════════════════════════════════════

async function _refreshDebug() {
    const pre = document.getElementById('sumDebugContent');
    if (!pre.textContent) pre.textContent = 'Loading…';
    if (!_sumSegments().length) { pre.textContent = 'No transcript'; return; }
    const segments = _sumPostSegments();
    try {
        const r = await fetch('/api/llm/debug', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                prompt: document.getElementById('sumPrompt').value.trim(),
                // Always send — '' deliberately means "without a system prompt"
                system_prompt: document.getElementById('sumSysPrompt')?.value ?? '',
                model: document.getElementById('sumModel')?.value || undefined,
                include_speakers: document.getElementById('sumSpeakers').checked,
                include_timestamps: document.getElementById('sumTimestamps').checked,
                metadata: _gatherMetadata(),
                segments,
            })
        });
        const d = await r.json();
        let text = '';
        text += `═══ Endpoint ═══\n`;
        text += `Type: ${d.api_type}\nURL: ${d.api_url}\nModel: ${d.model}\n`;
        text += `Thinking: ${d.thinking}\nTemp: ${d.temperature}\nMax Tokens: ${d.max_tokens}\n`;
        text += `Input Tokens (est.): ~${d.estimated_input_tokens}\n`;
        text += `Context (detected): ${d.context_effective || '?'}\n`;
        if (d.chunk_plan) {
            text += d.chunk_plan.map_reduce
                ? `Processing: MAP-REDUCE in ${d.chunk_plan.chunks} sections `
                  + `(tokens per section: ${(d.chunk_plan.chunk_tokens || []).join(', ')})\n`
                : `Processing: ONE call (fits in the context)\n`;
        }
        const q = document.getElementById('sumSource')?.value;
        text += `Source: ${q ? q + ' (translation track)' : 'Original transcript'}\n`;
        text += `Transcript: ${d.transcript_chars} chars, ${d.transcript_lines} lines\n\n`;
        if (d.chunk_plan?.map_system) {
            text += `═══ Map step: system prompt of each section ═══\n`
                  + d.chunk_plan.map_system + '\n\n'
                  + `═══ Final step (below): the template over the sections' notes ═══\n\n`;
        }
        d.messages.forEach((m, i) => {
            text += `═══ Message ${i + 1}: ${m.role.toUpperCase()} (${m.content.length} chars) ═══\n`;
            text += m.content + '\n\n';
        });
        pre.textContent = text;
    } catch(e) {
        pre.textContent = 'Error: ' + e.message;
    }
}

// ══════════════════════════════════════════════════════════════
//  Saved summaries
// ══════════════════════════════════════════════════════════════

async function _loadSavedSummaries(filename) {
    try {
        const r = await fetch(`/api/summaries/${encodeURIComponent(filename)}`);
        if (!r.ok) return;
        const data = await r.json();
        if (filename !== _sumFilename) return;          // the dialog moved on
        _sumSaved = data.summaries || [];
    } catch(e) {
        return;
    }
    if (_sumShown >= _sumSaved.length) _sumShown = -1;
    _sumSavedRender();
}

function _sumSavedRender() {
    const n = _sumSaved.length;
    document.getElementById('sumSavedCount').textContent = n ? String(n) : '';
    const list = document.getElementById('sumSavedList');
    if (!n) {
        list.innerHTML = `<div class="sum-empty"><span class="large">💾</span>
            <div>No summary saved for this file yet.<br>Every run is saved by itself.</div></div>`;
        return;
    }
    // Newest first; the index stays the one the server knows
    list.innerHTML = _sumSaved.map((s, i) => ({s, i})).reverse().map(({s, i}) => {
        const when = s.timestamp ? new Date(s.timestamp * 1000).toLocaleString(undefined,
            {day: '2-digit', month: '2-digit', year: 'numeric', hour: '2-digit', minute: '2-digit'}) : '';
        const snip = (s.text || '').replace(/[#*_\`>|-]+/g, ' ').replace(/\s+/g, ' ').trim().slice(0, 160);
        return `<div class="sum-saved-item${i === _sumShown ? ' shown' : ''}" onclick="_sumShowSaved(${i})">
            <div class="sum-saved-main">
                <div class="sum-saved-title">${esc(s.prompt || 'Summary')}</div>
                <div class="sum-saved-sub">${esc([when, s.model, s.auto ? 'automatic' : ''].filter(Boolean).join(' · '))}</div>
                <div class="sum-saved-snip">${esc(snip)}</div>
            </div>
            <button class="tx-act-btn sum-del" onclick="event.stopPropagation(); _sumDeleteClick(this, ${i})"
                    title="Delete this summary" aria-label="Delete this summary">🗑</button>
        </div>`;
    }).join('');
}

function _sumShowSaved(i) {
    const s = _sumSaved[i];
    if (!s) return;
    _sumShown = i;
    const v = document.getElementById('sumViewing');
    const when = s.timestamp ? new Date(s.timestamp * 1000).toLocaleString() : '';
    v.innerHTML = `<span>💾 Saved: <b>${esc(s.prompt || 'Summary')}</b>${when ? ' · ' + esc(when) : ''}${s.model ? ' · ' + esc(s.model) : ''}</span>`
        + (_sumFullText ? `<span class="sum-grow"></span><button class="tx-act-btn" onclick="_sumShowRun()">← This run's result</button>` : '');
    v.hidden = false;
    document.getElementById('sumResult').innerHTML = _renderMarkdown(s.text || '');
    document.getElementById('btnSumSave').hidden = true;
    _sumSavedRender();
    _sumTab('result');
}

function _sumShowRun() {
    _sumShown = -1;
    document.getElementById('sumViewing').hidden = true;
    if (_sumFullText) document.getElementById('sumResult').innerHTML = _renderMarkdown(_sumFullText);
    else _sumResultEmpty();
    _sumSavedRender();
}

// Deleting takes two clicks — one stray click must not lose a summary
function _sumDeleteClick(btn, i) {
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
    _deleteSummary(_sumFilename, i);
}

async function _deleteSummary(filename, index) {
    const s = _sumSaved[index];
    try {
        const r = await fetch('/api/summaries/delete', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({filename, index, timestamp: s?.timestamp})
        });
        const d = await r.json().catch(() => ({}));
        if (!r.ok || d.error) Toast.error('Not deleted: ' + (d.error || `HTTP ${r.status}`));
        if (index === _sumShown) _sumShowRun();
        else if (_sumShown > index) _sumShown--;
    } catch(e) {
        Toast.error('Not deleted: ' + e.message);
    }
    await _loadSavedSummaries(filename);
    if (typeof _txViewsLoadMeta === 'function') _txViewsLoadMeta();
}

// ══════════════════════════════════════════════════════════════
//  Summarize (through the LLM queue) with stats
// ══════════════════════════════════════════════════════════════

function _sumShowError(msg, retry) {
    document.getElementById('sumError').innerHTML =
        `<div class="llm-sum-error"><span>❌</span><span style="flex:1">${esc(msg)}</span>
         ${retry ? `<button class="tx-act-btn" onclick="doSummarize()">↻ Retry</button>` : ''}</div>`;
}

function _sumDone(btn) {
    btn.disabled = false;
    btn.textContent = '🤖 Again';
    document.getElementById('btnSumStop').hidden = true;
    document.getElementById('sumProgress').hidden = true;
}

// What the saved summary is listed as: the template, marked when edited
function _sumLabel() {
    const n = _sumTpl.name;
    const base = n === SUM_DEFAULT || !n ? 'Default prompt' : n;
    const extra = document.getElementById('sumPrompt').value.trim();
    const source = document.getElementById('sumSource')?.value || '';
    return [base + (_sumEdited() ? ' (edited)' : ''),
            extra ? `+ “${extra.slice(0, 40)}${extra.length > 40 ? '…' : ''}”` : '',
            source ? 'Source: ' + source : '']
        .filter(Boolean).join(' · ');
}

async function doSummarize() {
    const btn = document.getElementById('btnSumGo');
    const prompt = document.getElementById('sumPrompt').value.trim();
    const sysPrompt = document.getElementById('sumSysPrompt')?.value ?? '';
    const preset = _sumTpl.name && _sumTpl.name !== SUM_DEFAULT ? _sumTpl.name : '';
    const params = {
        source: document.getElementById('sumSource')?.value || '',
        preset,
        system_prompt: sysPrompt,          // '' deliberately means WITHOUT
        prompt,
        model: document.getElementById('sumModel')?.value || '',
        include_speakers: document.getElementById('sumSpeakers').checked,
        include_timestamps: document.getElementById('sumTimestamps').checked,
        metadata: _gatherMetadata(),
        save: true,
        label: _sumLabel(),
    };
    _sumPrefsSave();
    document.getElementById('sumError').innerHTML = '';
    document.getElementById('sumStats').innerHTML = '';
    document.getElementById('btnSumSave').hidden = true;
    _sumShown = -1;
    document.getElementById('sumViewing').hidden = true;
    _sumTab('result');
    btn.disabled = true;
    btn.textContent = '⏳ Running…';
    try {
        _sumQueueId = await llmQueueAdd('summary', _sumFilename, params, true);
        _sumFollow(_sumQueueId);
    } catch (e) {
        _sumShowError(e.message, true);
        _sumDone(btn);
        btn.textContent = '🤖 Summarize';
    }
}

// Attaches to a queue entry. The dialog may close at any time —
// the summary keeps running server-side and gets saved.
function _sumFollow(id) {
    _sumQueueId = id;
    if (typeof llmResetParts === 'function') llmResetParts();
    const btn = document.getElementById('btnSumGo');
    const prog = document.getElementById('sumProgress');
    const bar = prog.querySelector('.llm-progress-fill');
    const text = prog.querySelector('.llm-progress-text');
    const resultDiv = document.getElementById('sumResult');
    const scroller = document.getElementById('sumPaneResult');
    btn.disabled = true;
    btn.textContent = '⏳ Running…';
    document.getElementById('btnSumStop').hidden = false;
    prog.hidden = false;
    bar.style.background = 'var(--accent)';
    _sumShown = -1;
    document.getElementById('sumViewing').hidden = true;
    resultDiv.innerHTML = '<span style="color:var(--tx-3)">Waiting for the LLM…</span>';
    resultDiv._lastHtml = '';

    if (_sumTracker) _sumTracker.stop();
    _sumTracker = llmQueueTrack(id, (e) => {
        bar.style.width = Math.max(3, e.progress || 0) + '%';
        text.innerHTML = e.status === 'pending'
            ? llmWaitText(e) : esc(e.message || '');
        if (_sumShown >= 0) return;          // a saved one is on screen
        // LIVE: the text as it is generated — plus the expandable
        // intermediate results of the map phase. At the end the same
        // text is there, just complete; a dialog closed in the
        // meantime resumes right here when reopened.
        const parts = e.partial_results || [];
        const header = e.partial_text
            ? ''
            : `<div style="color:var(--tx-3);font-size:0.82rem">⏳ ${esc(e.message || 'Waiting for the LLM…')}</div>`;
        if (e.partial_text || parts.length) {
            const html = header
                + (e.partial_text ? _renderMarkdown(e.partial_text)
                                + '<span class="sum-cursor">▌</span>' : '')
                + llmPartsHtml(parts, 'q' + id);
            // Unchanged → leave alone: every innerHTML assignment
            // throws away the scroll position of open sections.
            if (html === resultDiv._lastHtml) return;
            resultDiv._lastHtml = html;
            // Keep scrolling to the bottom as long as the user hasn't
            // scrolled up themselves; save the scroll positions of open
            // sections across the rebuild.
            const atTheEnd = scroller.scrollHeight - scroller.scrollTop
                           - scroller.clientHeight < 40;
            const scroll = {};
            resultDiv.querySelectorAll('.llm-part').forEach(t => {
                const content = t.querySelector('.llm-part-text');
                if (content && content.scrollTop) scroll[t.dataset.part] = content.scrollTop;
            });
            resultDiv.innerHTML = html;
            for (const [part, top] of Object.entries(scroll)) {
                const content = resultDiv.querySelector(`.llm-part[data-part="${part}"] .llm-part-text`);
                if (content) content.scrollTop = top;
            }
            if (atTheEnd) scroller.scrollTop = scroller.scrollHeight;
        }
    }, (e) => {
        const r = e.result || {};
        _sumFullText = r.text || '';
        _sumDone(btn);
        _sumShown = -1;
        document.getElementById('sumViewing').hidden = true;
        resultDiv.innerHTML = _renderMarkdown(_sumFullText)
            + llmPartsHtml(e.partial_results || [], 'q' + id);
        document.getElementById('sumStats').innerHTML = [
            r.source ? `<span class="llm-chip">🌐 Source: ${esc(r.source)}</span>` : '',
            `<span class="llm-chip">⏱ <b>${Math.round(r.duration || 0)}s</b></span>`,
            `<span class="llm-chip">📝 <b>${(r.chars || 0).toLocaleString()}</b> character${r.chars === 1 ? '' : 's'}</span>`,
            r.saved ? `<span class="llm-chip green">💾 saved</span>`
                    : `<span class="llm-chip orange">not saved</span>`,
            (e.partial_results || []).length
                ? `<span class="llm-chip">⚙ ${plural(e.partial_results.length, 'section')}</span>` : '',
        ].filter(Boolean).join(' ');
        // Saving failed on the server (DB and JSON): offer it by hand
        document.getElementById('btnSumSave').hidden = !!r.saved || !_sumFullText;
        _loadSavedSummaries(_sumFilename);
        if (typeof _txViewsLoadMeta === 'function') _txViewsLoadMeta();
    }, (error) => {
        _sumDone(btn);
        btn.textContent = '🤖 Summarize';
        if (!_sumFullText && _sumShown < 0) _sumResultEmpty();
        _sumShowError(error, true);
    });
}

async function stopSummarize() {
    if (!_sumQueueId) return;
    // Errors are reported by llmQueueCancel (llmtools.js); on success the
    // tracker sees the task end as "Cancelled".
    await llmQueueCancel(_sumQueueId);
}

// ══════════════════════════════════════════════════════════════
//  Save by hand — only when the server could not (see _sumFollow)
// ══════════════════════════════════════════════════════════════

async function saveSummary() {
    if (!_sumFullText || !_sumFilename) return;
    const btn = document.getElementById('btnSumSave');
    if (btn.disabled) return;             // double-click guard
    btn.disabled = true;
    btn.textContent = '💾 Saving…';
    try {
        const r = await fetch('/api/summaries/save', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                filename: _sumFilename,
                text: _sumFullText,
                prompt: _sumLabel(),
                model: document.getElementById('sumModel')?.value || '',
            })
        });
        const d = await r.json();
        if (!r.ok || d.error) throw new Error(d.error || 'Save failed');
        btn.textContent = '✅ Saved';
        setTimeout(() => {
            btn.hidden = true;
            btn.disabled = false;
            btn.textContent = '💾 Save';
        }, 1500);
        _loadSavedSummaries(_sumFilename);
        // Keep the counter in the 🤖 tab of the transcript view in sync
        if (typeof _txViewsLoadMeta === 'function') _txViewsLoadMeta();
    } catch(e) {
        btn.disabled = false;
        btn.textContent = '💾 Save';
        _sumShowError('Not saved: ' + e.message, false);
    }
}

// ══════════════════════════════════════════════════════════════
//  Markdown → HTML
// ══════════════════════════════════════════════════════════════

function _renderMarkdown(md) {
    if (!md) return '';

    // ── Phase 1: Block-level — extract tables before escaping ──
    // Parse table blocks separately (before esc(), so | doesn't break)
    const blocks = [];
    let blockIdx = 0;
    const PLACEHOLDER = '\x00TBL';

    // Detect tables: lines starting with |, with a separator line (|---|)
    md = md.replace(/((?:^\|.+\|[ \t]*\n)+)/gm, (tableBlock) => {
        const lines = tableBlock.trim().split('\n');
        if (lines.length < 2) return tableBlock;  // Min header + separator

        // Find the separator line (|---|---|)
        let sepIdx = -1;
        for (let i = 0; i < Math.min(3, lines.length); i++) {
            if (/^\|[\s:]*[-]+[\s:]*(\|[\s:]*[-]+[\s:]*)+\|?$/.test(lines[i].trim())) {
                sepIdx = i;
                break;
            }
        }
        if (sepIdx < 0) return tableBlock;  // No separator → no table

        const headerLines = lines.slice(0, sepIdx);
        const bodyLines = lines.slice(sepIdx + 1);

        const parseRow = (line) =>
            line.replace(/^\||\|$/g, '').split('|').map(c => c.trim());

        // Parse alignment from the separator
        const sepCells = parseRow(lines[sepIdx]);
        const aligns = sepCells.map(c => {
            if (c.startsWith(':') && c.endsWith(':')) return 'center';
            if (c.endsWith(':')) return 'right';
            return 'left';
        });

        let tableHtml = '<div class="sum-table-wrap"><table class="sum-table">';

        // Header
        if (headerLines.length > 0) {
            tableHtml += '<thead>';
            headerLines.forEach(line => {
                const cells = parseRow(line);
                tableHtml += '<tr>';
                cells.forEach((c, i) => {
                    const al = aligns[i] || 'left';
                    tableHtml += `<th style="text-align:${al}">${esc(c)}</th>`;
                });
                tableHtml += '</tr>';
            });
            tableHtml += '</thead>';
        }

        // Body
        if (bodyLines.length > 0) {
            tableHtml += '<tbody>';
            bodyLines.forEach(line => {
                if (!line.trim() || /^\|[\s-:|]*\|?$/.test(line.trim())) return;
                const cells = parseRow(line);
                tableHtml += '<tr>';
                cells.forEach((c, i) => {
                    const al = aligns[i] || 'left';
                    // Inline formatting in cells
                    let cellHtml = esc(c);
                    cellHtml = cellHtml.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
                    cellHtml = cellHtml.replace(/\*(.+?)\*/g, '<em>$1</em>');
                    cellHtml = cellHtml.replace(/`([^`]+)`/g, '<code style="background:var(--bg-2);padding:1px 4px;border-radius:2px;font-size:0.85em">$1</code>');
                    tableHtml += `<td style="text-align:${al}">${cellHtml}</td>`;
                });
                tableHtml += '</tr>';
            });
            tableHtml += '</tbody>';
        }

        tableHtml += '</table></div>';
        blocks.push(tableHtml);
        return `${PLACEHOLDER}${blockIdx++}\x00`;
    });

    // ── Phase 2: Escape remaining text ──
    let html = esc(md);

    // ── Phase 3: Inline + block formatting ──
    // Code blocks
    html = html.replace(/```(\w*)\n([\s\S]*?)```/g,
        '<pre style="background:var(--bg-2);padding:10px 14px;border-radius:6px;overflow-x:auto;font-size:0.82rem;margin:8px 0"><code>$2</code></pre>');
    html = html.replace(/`([^`]+)`/g,
        '<code style="background:var(--bg-2);padding:1px 5px;border-radius:3px;font-size:0.85em">$1</code>');

    // Headings
    html = html.replace(/^#### (.+)$/gm, '<h4 style="font-size:0.95rem;font-weight:600;margin:14px 0 6px;color:var(--tx-1)">$1</h4>');
    html = html.replace(/^### (.+)$/gm, '<h3 style="font-size:1.0rem;font-weight:600;margin:16px 0 8px;color:var(--tx-1)">$1</h3>');
    html = html.replace(/^## (.+)$/gm, '<h2 style="font-size:1.1rem;font-weight:700;margin:18px 0 8px;color:var(--tx-1)">$1</h2>');
    html = html.replace(/^# (.+)$/gm, '<h1 style="font-size:1.2rem;font-weight:700;margin:20px 0 10px;color:var(--tx-1)">$1</h1>');

    // Bold + Italic
    html = html.replace(/\*\*\*(.+?)\*\*\*/g, '<strong><em>$1</em></strong>');
    html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    html = html.replace(/\*(.+?)\*/g, '<em>$1</em>');

    // Lists — including indented (LLMs nest sub-items with leading
    // whitespace; without this, raw */- would show up in the text)
    const indent = (ind) =>
        20 + Math.min(ind.replace(/\t/g, '  ').length, 12) * 7;
    html = html.replace(/^([ \t]*)[-*+] (.+)$/gm, (m, ind, txt) =>
        `<li style="margin:2px 0;margin-left:${indent(ind)}px">${txt}</li>`);
    html = html.replace(/^([ \t]*)(\d+)\. (.+)$/gm, (m, ind, num, txt) =>
        `<li style="margin:2px 0;margin-left:${indent(ind)}px;list-style:decimal" value="${num}">${txt}</li>`);

    // HR
    html = html.replace(/^---$/gm, '<hr style="border:none;border-top:1px solid var(--border);margin:12px 0">');

    // Paragraphs
    html = html.replace(/\n\n/g, '</p><p style="margin:8px 0">');
    html = html.replace(/\n/g, '<br>');
    // Block elements bring their own spacing — the <br> from the line
    // break after them would double it
    html = html.replace(/<\/(h[1-4]|li|pre)><br>/g, '</$1>');

    // ── Phase 4: Restore table placeholders ──
    html = html.replace(/\x00TBL(\d+)\x00/g, (_, idx) => blocks[parseInt(idx)] || '');

    return '<p style="margin:0">' + html + '</p>';
}
