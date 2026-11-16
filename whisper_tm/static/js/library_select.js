// ── Queue editor, tab "From library" ─────────────────────────────────────
//
// Picks files from the WHOLE library by what is known about them and puts
// them into the queue as one job. The queue editor's own tab manages what
// is already queued; this one answers the question before: which files.
//
// Criteria are building blocks (LS_GROUPS), combined with AND on the
// server (core/library_select.py). The count follows every change after a
// short pause; queuing sends the count the user saw along — if the library
// changed in between, the server refuses instead of queuing something else.

const LS_GROUPS = [
    {title: 'Where', items: [
        {key: 'folders', label: 'In folders', type: 'paths'},
        {key: 'exclude_folders', label: 'Not in folders', type: 'paths'},
        {key: 'name', label: 'Path contains', type: 'text', hint: '* = anything'},
        {key: 'exclude_name', label: 'Path does not contain', type: 'text'},
        {key: 'extensions', label: 'Extension', type: 'multi', facet: 'extensions'},
    ]},
    {title: 'State', items: [
        {key: 'status', label: 'Status', type: 'checks',
         options: [['open', 'open'], ['done', 'transcribed'], ['skipped', 'skipped'],
                   ['no_audio', 'no audio'], ['error', 'error']]},
        {key: 'in_queue', label: 'Already in the queue', type: 'tri'},
        {key: 'failed_min', label: 'Failed in the queue at least', type: 'number', unit: '×'},
    ]},
    {title: 'Media', items: [
        {key: 'duration', label: 'Duration', type: 'range', unit: 'min', scale: 60},
        {key: 'size', label: 'Size', type: 'range', unit: 'MB', scale: 1e6},
        {key: 'video', label: 'Video', type: 'tri'},
    ]},
    {title: 'Transcript', items: [
        {key: 'transcribed', label: 'Transcribed', type: 'dates'},
        {key: 'models', label: 'Model', type: 'multi', facet: 'models'},
        {key: 'languages', label: 'Language', type: 'multi', facet: 'languages'},
        {key: 'vocal_models', label: 'Separation model', type: 'multi', facet: 'vocal_models'},
        {key: 'app_commits', label: 'App commit', type: 'multi', facet: 'app_commits'},
        {key: 'diarized', label: 'Diarized', type: 'tri'},
        {key: 'segments', label: 'Segments', type: 'range', unit: '', scale: 1},
        {key: 'words', label: 'Words', type: 'range', unit: '', scale: 1},
        {key: 'speakers', label: 'Speakers', type: 'range', unit: '', scale: 1},
    ]},
    {title: 'Voices', items: [
        {key: 'voices', label: 'Has voices', type: 'tri'},
        {key: 'voices_old', label: 'Has voices without stored speech spans', type: 'tri'},
        {key: 'voices_run', label: 'Voices-only run done (with or without a result)', type: 'tri'},
        {key: 'speech_for_voices', label: 'Enough speech for a voice (a segment of 4 s)',
         type: 'tri'},
    ]},
    {title: 'Derived', items: [
        {key: 'summary', label: 'Summary', type: 'tri'},
        {key: 'translation', label: 'Translation', type: 'tri'},
        {key: 'corrections', label: 'Correction runs', type: 'tri'},
    ]},
    {title: 'Formats', items: [
        {key: 'other_format_transcribed',
         label: 'Same title already transcribed in another format', type: 'tri'},
    ]},
];

// The four questions asked most — a click fills the criteria and the job.
const LS_PRESETS = [
    {label: '🗣 Voices missing, speakers known', job: 'voices',
     crit: {status: ['done'], voices: false, voices_run: false, speech_for_voices: true,
            diarized: true}},
    {label: '🗣 Never diarized, enough speech', job: 'voices',
     crit: {status: ['done'], voices: false, voices_run: false, speech_for_voices: true,
            diarized: false}},
    {label: '📝 Open, not queued yet', job: 'transcribe',
     crit: {status: ['open'], in_queue: false}},
    {label: '📝 Open, no other format done', job: 'transcribe',
     crit: {status: ['open'], in_queue: false, other_format_transcribed: false}},
];

// Rough speed from real runs: a transcription ran at 7.7× real
// time, a voices-only run at about 18× (90 min in 5 min). An estimate, and
// labelled as one.
const LS_SPEED = {transcribe: 7.7, retranscribe: 7.7, voices: 18, voices_refresh: 18};

let lsState = {crit: {}, offset: 0, limit: 50, last: null, seq: 0, facets: null,
               timer: null, busy: false};

function _lsEl(id) { return document.getElementById(id); }
function _lsJob() {
    const r = document.querySelector('input[name="lsJob"]:checked');
    return r ? r.value : 'transcribe';
}
const LS_JOB_LABEL = {transcribe: 'transcribe', voices: 'voices only',
                      voices_refresh: 'renew voices', retranscribe: 're-transcribe'};
function _lsJobLabel() { return LS_JOB_LABEL[_lsJob()] || _lsJob(); }
function _lsFront() {
    const r = document.querySelector('input[name="lsWhere"]:checked');
    return !!r && r.value === 'front';
}

// The header's always-visible button: with nothing in the queue there is
// nothing to edit there, so it opens where a queue gets filled.
/* The Queue place: its Jobs tab (queue_page.js). */
function openQueueEditorFromBar() {
    openQueueEditor();
    qeShowTab('jobs');
}

function qeShowTab(tab) {
    const panes = {jobs: ['qePaneJobs', 'qeTabJobs'], queue: ['qePaneQueue', 'qeTabQueue'],
                   library: ['qePaneLibrary', 'qeTabLibrary'], add: ['qePaneAdd', 'qeTabAdd']};
    if (tab !== 'add' && typeof ppCloseIn === 'function') ppCloseIn('qa');
    for (const [name, [pane, button]] of Object.entries(panes)) {
        if (!_lsEl(pane)) continue;
        _lsEl(pane).style.display = name === tab ? '' : 'none';
        _lsEl(button).classList.toggle('active', name === tab);
    }
    if (tab === 'library' && !lsState.facets) lsInit();
    if (tab === 'add' && typeof qaInit === 'function') qaInit();
    if (typeof QueuePage !== 'undefined') QueuePage.shown(tab === 'jobs');
}

// ↻ in the head reloads what the open tab shows — the queue, or the count
// of "From library" / "Add folder", which otherwise keeps the numbers of
// before the queue went on.
function qeReloadTab() {
    const shown = id => { const e = _lsEl(id); return !!e && e.style.display !== 'none'; };
    if (shown('qePaneJobs') && typeof QueuePage !== 'undefined') QueuePage.refresh();
    else if (shown('qePaneAdd') && typeof qaCount === 'function') qaCount();
    else if (shown('qePaneLibrary')) lsCount();
    else qeLoad();
}

async function lsInit() {
    _lsEl('lsPresets').innerHTML = LS_PRESETS.map((p, i) =>
        `<button class="tx-act-btn" onclick="lsPreset(${i})">${esc(p.label)}</button>`).join('')
        + `<button class="tx-act-btn" onclick="lsReset()">✕ Reset</button>`;
    try {
        lsState.facets = await (await fetch('/api/queue/library/facets')).json();
    } catch (e) {
        lsState.facets = {};
    }
    lsRenderCriteria();
    lsJobChanged();
    lsCount();                 // the library total, before any criterion
}

// ── Criteria ────────────────────────────────────────────────────────────

function _lsTri(item) {
    const v = lsState.crit[item.key];
    const opt = (val, text) => `<option value="${val}"${String(v) === val ? ' selected' : ''}>${text}</option>`;
    return `<select data-key="${item.key}" onchange="lsSet(this)">
        ${opt('undefined', 'any')}${opt('true', 'yes')}${opt('false', 'no')}</select>`;
}

function _lsControl(item) {
    const c = lsState.crit;
    switch (item.type) {
    case 'tri': return _lsTri(item);
    case 'text':
        return `<input type="text" data-key="${item.key}" value="${esc(c[item.key] || '')}"
                 placeholder="${esc(item.hint || '')}" oninput="lsSet(this)">`;
    case 'number':
        return `<input type="number" min="0" data-key="${item.key}" value="${c[item.key] ?? ''}"
                 oninput="lsSet(this)"> ${item.unit || ''}`;
    case 'paths':
        return `<textarea data-key="${item.key}" rows="2" placeholder="one absolute folder per line"
                 oninput="lsSet(this)">${esc((c[item.key] || []).join('\n'))}</textarea>`;
    case 'checks':
        return item.options.map(([val, text]) => `<label class="ls-check"><input type="checkbox"
                 data-key="${item.key}" value="${val}" onchange="lsSet(this)"
                 ${(c[item.key] || []).includes(val) ? 'checked' : ''}> ${text}</label>`).join('');
    case 'multi': {
        const vals = (lsState.facets && lsState.facets[item.facet]) || [];
        if (!vals.length) return '<span class="ls-none">none recorded yet</span>';
        const chosen = c[item.key] || [];
        return `<select multiple size="${Math.min(4, Math.max(2, vals.length))}"
                 data-key="${item.key}" onchange="lsSet(this)">${vals.map(f =>
            `<option value="${esc(f.value)}"${chosen.includes(f.value) ? ' selected' : ''}>
               ${esc(f.value)} (${f.count.toLocaleString()})</option>`).join('')}</select>`;
    }
    case 'range': {
        const lo = c[item.key + '_min'], hi = c[item.key + '_max'];
        const show = v => (v === undefined ? '' : +(v / item.scale).toFixed(2));
        return `<input type="number" min="0" step="any" data-key="${item.key}_min"
                 data-scale="${item.scale}" value="${show(lo)}" placeholder="from"
                 oninput="lsSet(this)"> – <input type="number" min="0" step="any"
                 data-key="${item.key}_max" data-scale="${item.scale}" value="${show(hi)}"
                 placeholder="to" oninput="lsSet(this)"> ${item.unit}`;
    }
    case 'dates':
        return `<input type="date" data-key="transcribed_from" value="${c.transcribed_from || ''}"
                 onchange="lsSet(this)"> – <input type="date" data-key="transcribed_to"
                 value="${c.transcribed_to || ''}" onchange="lsSet(this)">`;
    }
    return '';
}

function _lsActive(group) {
    return group.items.filter(it => {
        const keys = it.type === 'range' ? [it.key + '_min', it.key + '_max']
            : it.type === 'dates' ? ['transcribed_from', 'transcribed_to'] : [it.key];
        return keys.some(k => lsState.crit[k] !== undefined);
    }).length;
}

// A group's title with the number of its criteria in use
function _lsSummary(g) {
    const n = _lsActive(g);
    return `${esc(g.title)}${n ? ` <span class="llm-chip">${n}</span>` : ''}`;
}

function lsRenderCriteria() {
    _lsEl('lsCriteria').innerHTML = LS_GROUPS.map((g, i) => {
        const n = _lsActive(g);
        return `<details class="ls-group" data-group="${i}" ${n || g.title === 'State' ? 'open' : ''}>
            <summary>${_lsSummary(g)}</summary>
            ${g.items.map(it => `<div class="ls-row"><span class="ls-label">${esc(it.label)}</span>
                <span class="ls-ctl">${_lsControl(it)}</span></div>`).join('')}
        </details>`;
    }).join('');
}

// One handler for every control: the element says which key it sets.
function lsSet(el) {
    const key = el.dataset.key;
    const c = lsState.crit;
    let value;
    if (el.type === 'checkbox') {
        const set = new Set(c[key] || []);
        el.checked ? set.add(el.value) : set.delete(el.value);
        value = [...set];
    } else if (el.multiple) {
        value = [...el.selectedOptions].map(o => o.value);
    } else if (el.tagName === 'TEXTAREA') {
        value = el.value.split('\n').map(s => s.trim()).filter(Boolean);
    } else if (el.type === 'number') {
        value = el.value === '' ? undefined : Number(el.value) * Number(el.dataset.scale || 1);
    } else if (el.tagName === 'SELECT') {
        value = el.value === 'undefined' ? undefined : el.value === 'true';
    } else {
        value = el.value.trim() || undefined;
    }
    if (value === undefined || (Array.isArray(value) && !value.length)) delete c[key];
    else c[key] = value;
    // The group's number follows — it kept a preset's count after a change
    const box = el.closest && el.closest('.ls-group');
    const group = box && LS_GROUPS[Number(box.dataset.group)];
    if (group) box.querySelector('summary').innerHTML = _lsSummary(group);
    lsState.offset = 0;
    lsScheduleCount();
}

function lsPreset(i) {
    const p = LS_PRESETS[i];
    lsState.crit = JSON.parse(JSON.stringify(p.crit));
    lsState.offset = 0;
    const r = document.querySelector(`input[name="lsJob"][value="${p.job}"]`);
    if (r) r.checked = true;
    lsRenderCriteria();
    lsJobChanged();
    lsCount();
}

function lsReset() {
    lsState.crit = {};
    lsState.offset = 0;
    lsRenderCriteria();
    lsCount();
}

// ── Counting ────────────────────────────────────────────────────────────

function lsScheduleCount() {
    clearTimeout(lsState.timer);
    _lsEl('lsSummary').classList.add('ls-stale');
    lsState.timer = setTimeout(lsCount, 500);
}

async function lsCount() {
    const seq = ++lsState.seq;
    _lsEl('lsSummary').classList.add('ls-stale');
    let d;
    try {
        const r = await fetch('/api/queue/library/select', {method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({criteria: lsState.crit, offset: lsState.offset,
                                  limit: lsState.limit})});
        d = await r.json();
    } catch (e) {
        d = {error: 'Server not reachable'};
    }
    if (seq !== lsState.seq) return;          // a newer count is on its way
    _lsEl('lsSummary').classList.remove('ls-stale');
    if (d.error) {
        lsState.last = null;
        _lsEl('lsSummary').innerHTML = `<span class="ls-error">⚠ ${esc(d.error)}</span>`;
        _lsEl('lsList').innerHTML = '';
        lsRenderButton();
        return;
    }
    lsState.last = d;
    lsRenderResult();
}

function _lsHours(sec) {
    if (sec >= 3600) return `${(sec / 3600).toLocaleString(undefined, {maximumFractionDigits: 0})} h`;
    // Short clips read "0 min" otherwise
    return sec > 0 && sec < 60 ? `${Math.round(sec)} s` : `${Math.round(sec / 60)} min`;
}

function lsRenderResult() {
    const d = lsState.last;
    const gpu = d.audio_sec / (LS_SPEED[_lsJob()] || 7.7);
    _lsEl('lsSummary').innerHTML = `<b>${d.count.toLocaleString()}</b> file${d.count === 1 ? '' : 's'} ·
        ${_lsHours(d.audio_sec)} audio · roughly ${_lsHours(gpu)} of GPU time as
        “${_lsJobLabel()}”`;
    _lsEl('lsList').innerHTML = d.sample.length ? `<table class="ls-table"><thead><tr>
        <th>File</th><th>Length</th><th>Transcribed</th><th>Diarized</th>
        </tr></thead><tbody>${d.sample.map(s => `<tr title="${esc(s.abs_path)}">
          <td>${esc(s.name)}<div class="ls-dir">${esc(s.abs_path.slice(0, -s.name.length))}</div></td>
          <td>${_lsHours(s.duration_sec)}</td>
          <td>${s.transcribed ? new Date(s.transcribed_at * 1000).toLocaleDateString() : '—'}</td>
          <td>${s.diarized === true ? 'yes' : s.diarized === false ? 'no' : '—'}</td></tr>`).join('')}</tbody></table>`
        : '<div class="ls-empty">No file matches.</div>';
    const from = d.count ? d.offset + 1 : 0;
    _lsEl('lsPageInfo').textContent =
        `${from}–${Math.min(d.offset + d.limit, d.count)} of ${d.count.toLocaleString()}`;
    _lsEl('lsBtnPrev').disabled = d.offset === 0;
    _lsEl('lsBtnNext').disabled = d.offset + d.limit >= d.count;
    lsRenderButton();
}

function lsPage(delta) {
    if (!lsState.last) return;
    lsState.offset = Math.max(0, lsState.offset + delta * lsState.limit);
    lsCount();
}

// ── Queuing ─────────────────────────────────────────────────────────────

const LS_JOB_HINT = {
    transcribe: 'Files that already have a transcript are skipped.',
    voices: 'Diarization and voice vectors only — the transcript stays as it is. '
          + 'Files that already have voices are skipped, so the names given to them stay.',
    voices_refresh: 'Makes the voices of each file anew (with speech spans); the transcript '
                  + 'stays as it is. Names, trash and history go to the new voices of the same '
                  + 'people — where that is not clear for every named or discarded voice, the '
                  + 'file keeps its voices.',
    retranscribe: '⚠ Deletes and remakes each transcript: edits, speaker names and the '
                + 'names of its voices are lost.',
};

function lsJobChanged() {
    _lsEl('lsJobHint').textContent = LS_JOB_HINT[_lsJob()] || '';
    _lsEl('lsJobHint').classList.toggle('ls-danger', _lsJob() === 'retranscribe');
    if (lsState.last) lsRenderResult();
    else lsRenderButton();
}

function lsRenderButton() {
    const n = lsState.last ? lsState.last.count : 0;
    const b = _lsEl('lsBtnQueue');
    b.textContent = `Queue ${n.toLocaleString()} file${n === 1 ? '' : 's'}`
        + (_lsFront() ? ' at the front' : '');
    b.disabled = !n || lsState.busy;
    b.classList.toggle('tx-act-danger', _lsJob() === 'retranscribe');
}

function lsAskConfirm() {
    const d = lsState.last;
    if (!d || !d.count) return;
    _lsEl('lsConfirmText').innerHTML = `Queue <b>${d.count.toLocaleString()}</b> file${d.count === 1 ? '' : 's'}
        (${_lsHours(d.audio_sec)} audio) as <b>${esc(_lsJobLabel())}</b>${_lsFront()
            ? ' <b>at the front</b> — next up, before every waiting entry' : ''}?
        ${_lsJob() === 'retranscribe'   // the other hints already sit next to the radios
            ? `<div class="ls-danger">${esc(LS_JOB_HINT.retranscribe)}</div>` : ''}`;
    _lsEl('lsBtnConfirm').textContent = d.count === 1 ? 'Yes, queue it' : 'Yes, queue them';
    _lsEl('lsConfirm').style.display = '';
}

function lsCancelConfirm() { _lsEl('lsConfirm').style.display = 'none'; }

async function lsEnqueue() {
    const d = lsState.last;
    if (!d || lsState.busy) return;
    lsState.busy = true;
    _lsEl('lsBtnConfirm').disabled = true;
    lsRenderButton();
    let out;
    try {
        const r = await fetch('/api/queue/library/enqueue', {method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({criteria: lsState.crit, job: _lsJob(), front: _lsFront(),
                                  expected_count: d.count})});
        out = await r.json();
    } catch (e) {
        out = {error: 'Server not reachable'};
    }
    lsState.busy = false;
    _lsEl('lsBtnConfirm').disabled = false;
    lsCancelConfirm();
    if (out.error) {
        Toast.error(out.error);
    } else {
        const extra = [out.already_queued ? `${out.already_queued.toLocaleString()} were already queued` : '',
                       out.outside_allowed_roots ? `${out.outside_allowed_roots} outside the allowed folders` : '']
            .filter(Boolean).join(', ');
        const moved = out.moved_up ? `, ${out.moved_up.toLocaleString()} already waiting moved to the front` : '';
        Toast.success(`${plural(out.added, 'file')} queued as ${out.job}`
                      + `${out.front ? ' at the front' : ''}${moved}${extra ? ' — ' + extra : ''}.`);
        if (typeof refreshQueue === 'function') refreshQueue();
    }
    lsCount();
}
