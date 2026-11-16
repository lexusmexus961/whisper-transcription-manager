// ── Settings → Images & sounds: the live parts ─────────────────────────────
//
// The environment (installed? the model fetched? the service running?), what
// the index holds, how far it is (the same bar as the semantic search's
// index), what is left and what it takes, and the run — polled every two
// seconds while a run, an install or the first estimate goes
// (api/media_search.py; the estimate is kept on the server and never waited
// for).

let _msSetPoll = null;

function _msFmtHours(h) {
    h = Number(h) || 0;
    if (h >= 48) return `${(h / 24).toFixed(1)} days`;
    if (h >= 1) return `${h.toFixed(1)} h`;
    return `${Math.round(h * 60)} min`;
}

async function _mediaLoad() {
    const envBox = document.getElementById('msSetEnv');
    if (!envBox) { _mediaStopPolling(); return; }
    let env, idx;
    try {
        [env, idx] = await Promise.all([
            fetch('/api/media-search/environment').then(r => r.json()),
            fetch('/api/media-search/index').then(r => r.json()),
        ]);
    } catch (e) {
        envBox.innerHTML = `<div class="set-note err">State not available: ${esc(e.message)}</div>`;
        return;
    }
    _mediaRenderEnv(env);
    _mediaRenderIndex(idx);
    _mediaRenderDescribe(idx);
    // Also while the estimate is still being counted: it arrives by itself
    const busy = (env.install && env.install.running) || (idx.run && idx.run.running)
        || ['pending', 'running'].includes(idx.describe_run?.status) || (!idx.estimate && idx.estimate_computing);
    if (busy) _mediaStartPolling(); else _mediaStopPolling();
}

function _mediaRenderEnv(env) {
    const box = document.getElementById('msSetEnv');
    const detail = document.getElementById('msSetEnvDetail');
    const c = env.check || {};
    const inst = env.install || {};
    const svc = env.service || {};
    const ok = !!c.ok;
    const line = ok
        ? `<span class="set-dot ok"></span>Ready — ${esc(c.model || '')}`
        : `<span class="set-dot err"></span>${inst.running ? 'Installing …' : 'Not ready: ' + esc((c.problems || []).join(', ') || '?')}`;
    const service = svc.running
        ? `<span class="set-dot ok"></span>running (pid ${svc.pid}${svc.idle_sec != null ? `, idle ${svc.idle_sec} s` : ''})${(svc.loaded || []).length ? ' · ' + esc(svc.loaded.join(', ')) : ''}
            <button type="button" class="set-link" onclick="_mediaServiceStop()">stop</button>`
        : '<span class="set-dot"></span>not running — it starts with the first search';
    if (box) box.innerHTML = `<div class="set-lines">
            <span>Environment</span><span>${line} ${_sGo('media#media-running', 'Environment')}</span>
            <span>Service</span><span>${service}</span>
        </div>`;
    if (!detail) return;
    const pk = c.packages || {};
    const lines = (inst.lines || []).slice(-12);
    detail.innerHTML = `<div class="set-lines">
            <span>Folder</span><span class="mono">${esc(c.path || '')}</span>
            <span>Packages</span><span>${['torch', 'transformers', 'sentence_transformers', 'av'].map(n =>
                `${n} ${pk[n] ? esc(pk[n]) : '<b class="set-err-text">missing</b>'}`).join(' · ')}</span>
            <span>Model</span><span>${c.model_cached ? '<span class="set-dot ok"></span>downloaded' : '<span class="set-dot err"></span>not downloaded'}</span>
        </div>
        <div class="set-actions" style="margin-top:8px">
            <button type="button" class="set-btn${ok ? '' : ' primary'}" onclick="_mediaInstall(this)" ${inst.running ? 'disabled' : ''}>
                ${ok ? '🔧 Repair / update' : '⬇ Install'}</button>
            <span class="set-hint">${ok ? '' : 'Creates the environment (with uv: ~1 GB of new disk, packages linked from its cache) and downloads the model (1.5 GB).'}</span>
        </div>
        ${inst.running || lines.length ? `<pre class="set-log ms-install-log">${esc(lines.join('\n'))}</pre>` : ''}
        ${inst.ok === false ? `<div class="set-note err">Install failed: ${esc(inst.error || '')}</div>` : ''}`;
}

const _MS_SET_KINDS = [['image', 'stills', 'still'], ['video', 'clips', 'clip'],
    ['videoaudio', 'clips with sound', 'clip with sound'], ['audio', 'sounds', 'sound'],
    ['speech', 'spoken text', 'spoken text'], ['description', 'descriptions', 'description']];

// "1 moment", "2,048 moments"
function _msCount(x, word) {
    x = Number(x || 0);
    return `${x.toLocaleString()} ${word}${x === 1 ? '' : 's'}`;
}

// The length of a library: "1,250 h", "3.2 h", "12 min", "20 s"
function _msLength(h) {
    h = Number(h) || 0;
    if (h >= 10) return `${Math.round(h).toLocaleString()} h`;
    if (h >= 1) return `${h.toFixed(1)} h`;
    if (h * 60 >= 1) return `${Math.round(h * 60)} min`;
    return `${Math.round(h * 3600)} s`;
}

// A share as the page shows it: a decimal below 10 %
function _msPct(share) {
    const p = Math.max(0, Math.min(1, Number(share) || 0)) * 100;
    return p > 0 && p < 10 ? p.toFixed(1) : String(Math.round(p));
}

/* The index card, laid out like the semantic search's (Settings → Search):
 * the state lines, four numbers, how far it is as a bar, the buttons and
 * the run's line — then what is left and what it takes. */
function _mediaRenderIndex(d) {
    const box = document.getElementById('msSetIndex');
    const cov = document.getElementById('msSetCoverage');
    const est = document.getElementById('msSetEstimate');
    const runBox = document.getElementById('msSetRun');
    const c = d.counts || {};
    const k = c.kinds || {};
    const e = d.estimate || null;
    const run = d.run || {};
    const n = (x) => Number(x || 0).toLocaleString();
    const kpi = (label, value, sub = '') => `<div class="set-kpi"><div class="set-kpi-label">${label}</div>
        <div class="set-kpi-value">${value}</div>${sub ? `<div class="set-kpi-sub">${sub}</div>` : ''}</div>`;
    const vectors = _MS_SET_KINDS.reduce((a, [key]) => a + Number(k[key] || 0), 0);
    const thumbs = e ? Number(e.thumbs_have || 0) : 0;
    if (box) box.innerHTML = `<div class="set-kpis">
            ${kpi('Moments', n(c.moments), c.repeats ? `${n(c.repeats)} ${c.repeats === 1 ? 'repeat shares' : 'repeats share'} a still` : '')}
            ${kpi('Recordings', n((c.files || {}).done), e ? `of ${n(e.files)} in scope` : '')}
            ${kpi('Size', _fmtBytesS((c.bytes || 0) + thumbs), `index ${_fmtBytesS(c.bytes || 0)}${thumbs ? ' · stills ' + _fmtBytesS(thumbs) : ''}`)}
            ${kpi('Vectors', n(vectors), c.dims ? `${c.dims} numbers, 8-bit` : '')}
        </div>
        <div class="set-lines" style="margin-top:10px">
            <span>Kinds</span><span>${_MS_SET_KINDS.map(([key, label, one]) => `${n(k[key])} ${k[key] === 1 ? one : label}`).join(' · ')}</span>
            ${d.ram_bytes ? `<span>In RAM</span><span>${_fmtBytesS(d.ram_bytes)} — every vector as stored, for the search to score all of them</span>` : ''}
        </div>`;

    // How far it is — the bar by hours of recordings, the headline by recordings
    if (cov) {
        if (!e) {
            cov.innerHTML = d.estimate_error
                ? `<div class="set-note err">Estimate: ${esc(d.estimate_error)}</div>`
                : '<div class="set-hint">⟳ Counting what the scope holds …</div>';
        } else {
            const hours = Number(e.video_hours || 0) + Number(e.audio_hours || 0);
            const parts = [];
            if (run.running) {
                parts.push(`🖼 ${n(run.files_done)} in this run`);
                parts.push(`⏳ ${n(e.todo)} to go`);
                parts.push(`🧩 ${_msCount(run.moments_done, 'moment')} · ${_msCount(run.vectors, 'vector')}`);
                if (run.moments_per_hour && e.moments) {
                    parts.push(`⏱ ≈ ${_msFmtHours(e.moments / run.moments_per_hour)} left at ${n(run.moments_per_hour)} moments an hour`);
                }
                if (run.repeats) parts.push(`🔁 ${_msCount(run.repeats, 'repeat')}`);
            } else {
                parts.push(`✓ ${n(e.done)} complete`);
                parts.push(`⏳ ${n(e.todo)} to do${e.fills ? ` (${n(e.fills)} only get a new kind)` : ''}`);
                if (e.todo) parts.push(`⏱ ≈ ${_msFmtHours(e.gpu_hours)} of GPU time`);
            }
            if ((run.errors || []).length) parts.push(`⚠ ${run.errors.length} failed`);
            if (run.missing) parts.push(`📭 ${n(run.missing)} not reachable`);
            cov.innerHTML = _sCoverageHtml(e.done, e.files, _msPct(e.share_done), parts, !!run.running,
                {unit: e.files === 1 ? 'recording' : 'recordings', what: `of ${_msLength(hours)} indexed`});
        }
    }

    // What is left and what it takes
    if (est) {
        if (!e) est.innerHTML = '';
        else {
            const lay = e.layout || {};
            const need = Number(e.bytes || 0) + Number(e.thumb_bytes || 0);
            const room = e.free_gb == null ? null : Math.max(0, e.free_gb - (e.min_free_gb || 0)) * 1024 ** 3;
            const fits = room == null || need <= room;
            const kinds = Object.entries(e.vectors_by_kind || {}).sort((a, b) => b[1] - a[1])
                .map(([key, v]) => `${n(v)} ${(_MS_SET_KINDS.find(x => x[0] === key) || [key, key, key])[v === 1 ? 2 : 1]}`).join(' · ');
            est.innerHTML = `<div class="set-lines">
                    <span>In scope</span><span>${_msCount(e.files, 'recording')} — ${_msLength(e.video_hours)} of video, ${_msLength(e.audio_hours)} of audio${
                        (d.settings?.folders || []).length ? '' : ' · <b>the whole library</b>'}</span>
                    ${e.todo ? `<span>To do</span><span>${_msCount(e.todo, 'recording')} · ${_msCount(e.moments, 'moment')} → up to ${_msCount(e.vectors, 'vector')}${kinds ? ` (${kinds})` : ''} · ${_msCount(e.clip_frames, 'frame')} per clip</span>
                    <span>GPU time</span><span>≈ ${_msFmtHours(e.gpu_hours)} at the measured speed — less where repeats are skipped, more while the queue has the card</span>
                    <span>Disk</span><span${fits ? '' : ' class="set-err-text"'}>≈ ${_fmtBytesS(e.bytes)} of vectors${e.thumb_bytes ? ` + ${_fmtBytesS(e.thumb_bytes)} of stills` : ''}${
                        e.free_gb != null ? ` — ${e.free_gb} GB free; a run waits below ${e.min_free_gb} GB` : ''}${
                        fits ? '' : ' — <b>it doesn\'t fit</b>: free some space, choose folders, or put the index on a disk with room (Files below)'}</span>` : ''}
                    ${e.ram_bytes ? `<span>RAM</span><span${e.ram_heavy ? ' class="set-err-text"' : ''}>≈ ${_fmtBytesS(e.ram_bytes)} for searching once all of it is indexed${
                        e.ram_heavy ? ` — <b>more than half the machine's ${_fmtBytesS(e.ram_total)}</b>: choose folders or fewer kinds` : ''}</span>` : ''}
                </div>
                ${lay.differs ? `<div class="set-note warn" style="margin-top:10px">The index holds ${lay.dims} numbers per vector; the settings say ${lay.wanted_dims} — “Rebuild everything” applies them.${
                    lay.wanted_dims < 768 ? ' Measured on sounds, all 768 numbers find clearly more (mean AP 0.251 against 0.189 with 512).' : ' Measured on sounds, all 768 find clearly more than 512 (mean AP 0.251 against 0.189),.'}</div>` : ''}`;
        }
    }

    const stop = document.getElementById('msSetStop');
    const start = document.getElementById('msSetStart');
    if (stop) stop.hidden = !run.running;
    if (start) {
        start.disabled = !!run.running;
        start.textContent = run.running ? '🖼 Indexing …'
            : !e ? '🖼 Index' : e.todo ? `🖼 Index ${_msCount(e.todo, 'recording')}` : '✓ Everything in scope is indexed';
    }
    if (!runBox) return;
    if (run.running) {
        runBox.className = 'set-result' + (run.waiting === 'disk' ? ' err' : '');
        runBox.textContent = run.waiting ? run.message || '' : '⟳ ' + (run.message || '');
    } else if (run.ended) {
        runBox.className = 'set-result' + (run.error ? ' err' : ' ok');
        runBox.textContent = (run.error ? '✕ ' : '✓ ') + (run.message || '') + (run.moments_done ? ` — ${_msCount(run.moments_done, 'moment')}, ${_msCount(run.vectors, 'vector')}` : '')
            + (run.missing ? ` · ${_msCount(run.missing, 'recording')} not reachable` : '') + ((run.errors || []).length ? ` · ${run.errors.length} failed` : '');
    } else {
        runBox.textContent = '';
    }
}

function _mediaStartPolling() {
    if (_msSetPoll) return;
    // Settings pages stay in the page when closed: the card's own box being
    // out of sight (closed, or another page) ends the polling — otherwise it
    // would go on every 2 s for as long as an index run lasts. One request at a time.
    let busy = false;
    _msSetPoll = setInterval(async () => {
        const box = document.getElementById('msSetEnv');
        if (!box || box.offsetParent === null) { _mediaStopPolling(); return; }
        if (busy) return;
        busy = true;
        try { await _mediaLoad(); } finally { busy = false; }
    }, 2000);
}

function _mediaStopPolling() {
    if (_msSetPoll) { clearInterval(_msSetPoll); _msSetPoll = null; }
}

async function _mediaStart(rebuild, btn) {
    if (rebuild && !confirm('Rebuild the whole index? Every recording in scope is indexed again — with a large library that takes days.')) return;
    const runBox = document.getElementById('msSetRun');
    try { await _settingsSaveQuiet(); } catch (e) { _showToast('Not started — saving failed: ' + e.message, true); return; }
    try {
        const r = await fetch('/api/media-search/index/start', {method: 'POST', headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({rebuild})});
        const d = await r.json();
        if (!r.ok) { _showToast(d.error || 'Not started', true); return; }
        if (!d.started) _showToast('A run is already going', true);
        if (runBox) { runBox.className = 'set-result'; runBox.textContent = '⟳ Starting …'; }
    } catch (e) { _showToast(e.message, true); }
    _mediaLoad();
}

async function _mediaStop() {
    try { await fetch('/api/media-search/index/stop', {method: 'POST'}); _showToast('Stopping after the current request'); }
    catch (e) { _showToast(e.message, true); }
    _mediaLoad();
}

async function _mediaClear(btn) {
    if (!confirm('Empty the index of Images & sounds? Its vectors, thumbnails and preview clips are deleted — the recordings stay.')) return;
    const label = btn.textContent; btn.disabled = true; btn.textContent = '⏳ …';
    try {
        const r = await fetch('/api/media-search/index/clear', {method: 'POST'});
        const d = await r.json();
        if (r.ok) _showToast('Index emptied'); else _showToast(d.error || 'Error', true);
    } catch (e) { _showToast(e.message, true); }
    finally { btn.disabled = false; btn.textContent = label; _mediaLoad(); }
}

async function _mediaInstall(btn) {
    try { await _settingsSaveQuiet(); } catch (e) { _showToast('Saving failed: ' + e.message, true); return; }
    btn.disabled = true;
    try {
        const r = await fetch('/api/media-search/environment/install', {method: 'POST'});
        const d = await r.json();
        if (!r.ok) _showToast(d.error || 'Not started', true);
        else if (!d.started) _showToast('An install is already going', true);
    } catch (e) { _showToast(e.message, true); }
    _mediaLoad();
}

async function _mediaServiceStop() {
    try {
        const r = await fetch('/api/media-search/service/stop', {method: 'POST'});
        const d = await r.json();
        if (!r.ok) _showToast(d.error || 'Not stopped', true);
    } catch (e) { _showToast(e.message, true); }
    _mediaLoad();
}

function _mediaRenderDescribe(d) {
    const box = document.getElementById('msSetDescribe');
    if (!box) return;
    const c = d.descriptions || {};
    const run = d.describe_run;
    const left = Math.max(0, (c.stills || 0) - (c.described || 0));
    const pct = c.stills ? Math.round((c.described || 0) / c.stills * 100) : 0;
    box.innerHTML = `<div class="set-lines">
            <span>Described</span><span>${Number(c.described || 0).toLocaleString()} of ${_msCount(c.stills, 'still')} (${pct} %)${left ? ` · ${left.toLocaleString()} to go` : ''}</span>
            ${run ? `<span>Last run</span><span>${esc(run.status)}${run.status === 'running' ? ` · ${run.progress || 0} % · ${esc(run.message || '')}` : ''}${run.error ? ' — ' + esc(run.error) : ''}</span>` : ''}
        </div>`;
    const btn = document.getElementById('msSetDescribeBtn');
    if (btn) {
        const busy = run && ['pending', 'running'].includes(run.status);
        btn.disabled = busy || !left;
        btn.textContent = busy ? '📝 Describing …' : left ? `📝 Describe ${_msCount(left, 'still')}` : '✓ Every still is described';
    }
}

async function _mediaDescribe(btn) {
    try { await _settingsSaveQuiet(); } catch (e) { _showToast('Saving failed: ' + e.message, true); return; }
    btn.disabled = true;
    try {
        const r = await fetch('/api/media-search/describe-run', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: '{}'});
        const d = await r.json();
        if (!r.ok) _showToast(d.error || 'Not queued', true);
        else _showToast('Describing run queued in the LLM queue');
    } catch (e) { _showToast(e.message, true); }
    _mediaLoad();
}
