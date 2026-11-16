// ── Settings → Audio analysis: the live parts ──────────────────────────────
//
// What the index holds, what is left and what reading it takes, and the
// background run — polled every two seconds while it goes (api/audio.py).

let _amSetPoll = null;

function _amSetHours(h) {
    h = Number(h) || 0;
    if (h >= 48) return `${(h / 24).toFixed(1)} days`;
    if (h >= 1) return `${h.toFixed(1)} h`;
    return `${Math.round(h * 60)} min`;
}

// An amount of sound: hours, never days ("31,022 h", not "1292.6 days")
function _amSetSound(h) {
    h = Number(h) || 0;
    if (h >= 100) return `${Math.round(h).toLocaleString()} h`;
    if (h >= 1) return `${h.toFixed(1)} h`;
    return `${Math.round(h * 60)} min`;
}

function _amSetBytes(b) {
    b = Number(b) || 0;
    if (b >= 1e12) return `${(b / 1e12).toFixed(1)} TB`;
    if (b >= 1e9) return `${(b / 1e9).toFixed(1)} GB`;
    return `${Math.round(b / 1e6)} MB`;
}

async function _audioLoad(fresh = false) {
    const box = document.getElementById('amSetState');
    if (!box) { _audioStopPolling(); return; }
    let info, est = null;
    try {
        info = await fetch('/api/audio/info').then(r => r.json());
        if (info.enabled) est = await fetch('/api/audio/estimate' + (fresh ? '?fresh=1' : '')).then(r => r.json());
    } catch (e) {
        box.innerHTML = `<div class="set-note err">State not available: ${esc(e.message)}</div>`;
        return;
    }
    _audioRender(info, est);
    if (info.run && info.run.running) _audioStartPolling(); else _audioStopPolling();
}

function _audioRender(info, est) {
    const box = document.getElementById('amSetState');
    const runBox = document.getElementById('amSetRun');
    const estBox = document.getElementById('amSetEstimate');
    const c = info.counts || {};
    const run = info.run || {};
    if (!info.enabled) {
        box.innerHTML = '<div class="set-hint"><span class="set-dot"></span>Switched off — nothing is analyzed.</div>';
        if (runBox) runBox.textContent = '';
        if (estBox) estBox.textContent = '';
        return;
    }
    box.innerHTML = info.error
        ? `<div class="set-note err">The index can't be read: ${esc(info.error)}</div>`
        : `<div class="set-lines">
            <span>Analyzed</span><span><b>${Number(c.files || 0).toLocaleString()}</b> recording${c.files === 1 ? '' : 's'} · ${_amSetSound(c.hours)} of sound${
                c.errors ? ` · <span class="set-err-text">${Number(c.errors).toLocaleString()} unreadable</span>` : ''}</span>
            <span>Found</span><span>${plural(c.same, 'pair')} that ${c.same === 1 ? 'is' : 'are'} the same recording · ${
                Number(c.contains || 0).toLocaleString()} contained in another</span>
            <span>On disk</span><span>${_amSetBytes(c.disk_bytes)} — ${plural(c.entries, 'fingerprint')} in ${
                c.segments || 0} segment${c.segments === 1 ? '' : 's'}${c.pending ? `, ${Number(c.pending).toLocaleString()} still in memory` : ''}</span>
        </div>`;
    if (runBox) {
        if (run.running) {
            const pct = run.files_total ? Math.min(100, 100 * (run.files_done + (run.skipped || 0)) / run.files_total) : 0;
            const speed = run.speed ? ` · ${Math.round(run.speed).toLocaleString()}× real time` : '';
            const left = run.eta_sec ? ` · about ${_amSetHours(run.eta_sec / 3600)} left` : '';
            runBox.className = 'set-result';
            runBox.innerHTML = `<div class="set-bar" style="margin:6px 0 8px"><span style="width:${pct.toFixed(1)}%"></span></div>
                ⟳ ${esc(run.waiting ? 'Waiting: ' + run.waiting : run.message || '')}${speed}${left}
                ${(run.current || []).length ? `<div class="set-hint">Now: ${run.current.map(esc).join(', ')}</div>` : ''}
                ${run.last_error ? `<div class="set-hint set-err-text">Last unreadable: ${esc(run.last_error)}</div>` : ''}`;
        } else {
            runBox.className = 'set-result';
            runBox.innerHTML = run.message ? esc(run.message) : '';
        }
    }
    const start = document.getElementById('amSetStart');
    const stop = document.getElementById('amSetStop');
    if (start) start.hidden = !!run.running;
    if (stop) stop.hidden = !run.running;
    if (estBox) {
        // While a run goes, its own line says how far it is
        const reading = Number(est?.bytes || 0) / 100e6 / 3600;     // hours at 100 MB/s
        estBox.innerHTML = run.running || !est || est.error ? ''
            : est.files
                ? `<div class="set-hint">Left to analyze: <b>${Number(est.files).toLocaleString()}</b> recording${est.files === 1 ? '' : 's'}${
                    est.hours ? ` — ${_amSetSound(est.hours)} of sound` : ''}, ${_amSetBytes(est.bytes)} to read. One hour of
                    sound takes ~4 s of one core; reading the files is what takes the time — at 100 MB/s (a network share)
                    about ${_amSetHours(Math.max(reading, 1 / 60))}.</div>`
                : '<div class="set-hint">Every recording the library scan found is analyzed.</div>';
    }
}

function _audioStartPolling() {
    if (_amSetPoll) return;
    let busy = false;
    _amSetPoll = setInterval(async () => {
        const box = document.getElementById('amSetState');
        if (!box || box.offsetParent === null) { _audioStopPolling(); return; }
        if (busy) return;
        busy = true;
        try { await _audioLoad(); } finally { busy = false; }
    }, 2000);
}

function _audioStopPolling() {
    if (_amSetPoll) { clearInterval(_amSetPoll); _amSetPoll = null; }
}

async function _audioStart() {
    try { await _settingsSaveQuiet(); } catch (e) { _showToast('Not started — saving failed: ' + e.message, true); return; }
    try {
        const r = await fetch('/api/audio/run/start', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: '{}'});
        const d = await r.json();
        if (!r.ok) { _showToast(d.error || 'Not started', true); return; }
        if (!d.started) _showToast('A run is already going', true);
    } catch (e) { _showToast(e.message, true); }
    _audioLoad();
}

async function _audioStop() {
    try { await fetch('/api/audio/run/stop', {method: 'POST'}); _showToast('Stopping after the recordings in hand'); }
    catch (e) { _showToast(e.message, true); }
    _audioLoad();
}

async function _audioRetry() {
    try {
        const r = await fetch('/api/audio/retry-failed', {method: 'POST'});
        const d = await r.json();
        _showToast(r.ok ? `${plural(d.forgotten, 'unreadable recording')} will be tried again` : (d.error || 'Error'), !r.ok);
    } catch (e) { _showToast(e.message, true); }
    _audioLoad(true);
}

async function _audioClear(btn) {
    if (!confirm('Empty the audio analysis? Fingerprints, waveforms, loudness and the duplicates found are deleted — the recordings stay. Analyzing them again means reading them again.')) return;
    const label = btn.textContent; btn.disabled = true; btn.textContent = '⏳ …';
    try {
        const r = await fetch('/api/audio/clear', {method: 'POST'});
        const d = await r.json();
        if (r.ok) _showToast('Audio analysis emptied'); else _showToast(d.error || 'Error', true);
    } catch (e) { _showToast(e.message, true); }
    finally { btn.disabled = false; btn.textContent = label; _audioLoad(true); }
}
