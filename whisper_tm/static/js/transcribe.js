// ── Transcribe ──

function _gatherTxSettings() {
    // The options live in TX_OPTIONS (txoptions.js) — ONE line per
    // option instead of a listing here, one in batch.js and two in
    // upload.js. The four copies had already drifted apart: the
    // single-file dialog sent the speaker count as the string 'auto', the
    // batch dialog as null.
    return txCollect('opt');
}

let _isRetranscribe = false;

function startTranscribe(filename) {
    _isRetranscribe = false;
    pendingTxFile = filename;
    document.getElementById('modalTitle').textContent = `Transcription: ${filename}`;
    document.getElementById('modalBg').classList.add('visible');
    syncCheckboxUI();
}

// "Redo": the same transcription dialog as usual, just in redo mode. The
// earlier confirm() prompt is gone — the dialog IS the confirmation, and
// it's only submitted with "Transcribe".
function retranscribe(filename) {
    _isRetranscribe = true;
    pendingTxFile = filename;
    document.getElementById('modalTitle').textContent = `Retranscribe: ${filename}`;
    document.getElementById('modalBg').classList.add('visible');
    syncCheckboxUI();
}

function closeModal() {
    document.getElementById('modalBg').classList.remove('visible');
    _isRetranscribe = false;
}

// Without transcription AND without separation, the run would have nothing
// to do. Better to say so here than to queue a job that does nothing.
function _txIdle(p) {
    const model = document.getElementById(p + 'Model')?.value;
    const iso = document.getElementById(p + 'VocalIso')?.checked;
    if (model !== 'none' || iso) return false;
    const t = 'Without transcription and without vocal isolation there is nothing to do.';
    if (typeof Toast !== 'undefined') Toast.error(t); else alert(t);
    return true;
}

/**
 * Submit the dialog. `front` = "⤒ Next up" button: straight to the front of
 * the queue, instead of behind everything else waiting.
 */
async function confirmTranscribe(front = false) {
    if (_txIdle('opt')) return;
    // Save retranscribe mode BEFORE closeModal() resets _isRetranscribe
    const retranscribeMode = _isRetranscribe;
    closeModal();
    const filename = pendingTxFile;
    if (!filename) return;
    _isRetranscribe = false;

    // Always queue — never start directly. Whether the file runs right away
    // or waits is decided by the server worker. It also runs without an open WebUI.
    const settings = _gatherTxSettings();
    if (retranscribeMode) settings.retranscribe = true;
    // A redo comes from the file that's already open: the user wants the
    // result NOW, not behind a long queue of waiting files — so it's pushed to the
    // front (the server also resumes a paused queue for this).
    await enqueueFiles([filename], settings,
                       null, (front || retranscribeMode) ? {front: true} : {});
}

// Quick-queue, no modal — with the options last chosen in the modal
function quickQueue(filename) {
    if (isQueued(filename)) return;
    enqueueFiles([filename], _gatherTxSettings());
}

/**
 * "⤒ Next up": the same options as quickQueue, but right at the front — and
 * this also resumes a paused queue (server: /api/queue/add with
 * front).
 *
 * Deliberately WITHOUT the isQueued guard from quickQueue: if the file is
 * already sitting far back in the queue, "bring to front" is exactly the
 * intent — enqueue() then pulls it forward instead of discarding the call.
 */
function quickQueueFront(filename) {
    enqueueFiles([filename], _gatherTxSettings(), null, {front: true});
}
// ── Log Modal ──
/* `file` is new and the reason opening a file is fast again: the log
 * file's name is now only looked up HERE, no longer on every transcript
 * fetch. The lookup may have to scan a very large log directory on a
 * network share and can take tens of seconds — for nothing but the label on this button
 * (api/transcription.py: _find_log_for_file). */
// Every opening gets a number; closing bumps it too. Only the newest
// opening may write into the modal — the lookup takes up to a minute, and a
// late answer for file A used to replace the log of file B opened since.
let _logModalSeq = 0;

async function showLogModal(logFile, file) {
    const modal = document.getElementById('logModal');
    const content = document.getElementById('logModalContent');
    const dl = document.getElementById('logModalDownload');
    const seq = ++_logModalSeq;
    const stale = () => seq !== _logModalSeq || !modal.classList.contains('visible');
    modal.classList.add('visible');
    if (!logFile && file) {
        content.textContent = 'Searching for the log file … (the directory is large)';
        dl.style.display = 'none';
        try {
            const r = await fetch(
                `/api/transcript/${encodeURIComponent(file)}/logFile`);
            logFile = (await r.json()).log_file || '';
        } catch (e) { logFile = ''; }
        if (stale()) return;
    }
    if (!logFile) {
        content.textContent = 'No log exists for this file.';
        dl.style.display = 'none';
        return;
    }
    dl.style.display = '';
    dl.href = `/api/logs/download/${encodeURIComponent(logFile)}`;
    content.textContent = 'Loading…';
    try {
        const r = await fetch(`/api/logs/download/${encodeURIComponent(logFile)}`);
        if (!r.ok) throw new Error('Not found');
        const text = await r.text();
        if (stale()) return;
        // Colorize log lines
        content.innerHTML = text.split('\n').map(line => {
            if (line.includes('❌') || line.includes('ERROR'))
                return `<span style="color:var(--red)">${esc(line)}</span>`;
            if (line.includes('⚠'))
                return `<span style="color:var(--orange)">${esc(line)}</span>`;
            if (line.includes('✅') || line.includes('✓'))
                return `<span style="color:var(--green)">${esc(line)}</span>`;
            if (line.includes('⏱'))
                return `<span style="color:#5dade2">${esc(line)}</span>`;
            return esc(line);
        }).join('\n');
    } catch(e) {
        if (stale()) return;
        content.textContent = 'Error loading: ' + e.message;
    }
}
function closeLogModal() {
    _logModalSeq++;   // a lookup still running must not write into it anymore
    document.getElementById('logModal').classList.remove('visible');
}

// ── Log Viewer / Progress Map ──
// Two views of the same area: 'map' = graphical step map highlighting
// the running phase + duration per step; 'raw' = raw log.
let _logMode = localStorage.getItem('wtm_logMode') || 'map';
let _stepMapData = null;      // last SSE state (phase + step_timings)
let _stepMapTimer = null;      // live timer of the running phase

function setLogMode(mode) {
    _logMode = mode;
    try { localStorage.setItem('wtm_logMode', mode); } catch (e) {}
    document.getElementById('lmMap')?.classList.toggle('active', mode === 'map');
    document.getElementById('lmRaw')?.classList.toggle('active', mode === 'raw');
    _syncLogAreas();
}

function _syncLogAreas() {
    const isOpen = document.getElementById('logArrow')?.classList.contains('open');
    const viewer = document.getElementById('logViewer');
    const map = document.getElementById('stepMap');
    if (viewer) viewer.classList.toggle('visible', !!isOpen && _logMode === 'raw');
    if (map) map.classList.toggle('visible', !!isOpen && _logMode === 'map');
    if (isOpen && _logMode === 'map') renderStepMap();
    if (isOpen && _logMode === 'raw' && viewer) viewer.scrollTop = viewer.scrollHeight;
}

function toggleLogViewer() {
    const arrow = document.getElementById('logArrow');
    arrow.classList.toggle('open');
    _syncLogAreas();
}

/** Fed by the SSE handler on every tick. */
function updateStepMap(d) {
    if (!d || (!d.phase && !d.step_timings)) return;
    _stepMapData = {phase: d.phase, steps: d.step_timings || [],
                     status: d.status};
    if (document.getElementById('stepMap')?.classList.contains('visible')) {
        renderStepMap();
    }
}

function _fmtSecs(s) {
    if (s >= 90) return `${Math.floor(s / 60)}m ${Math.round(s % 60)}s`;
    return s >= 10 ? `${Math.round(s)}s` : `${s.toFixed(1)}s`;
}

function renderStepMap() {
    const el = document.getElementById('stepMap');
    if (!el) return;
    clearInterval(_stepMapTimer);
    if (!_stepMapData) {
        el.innerHTML = '<div class="sm-empty">No steps yet — starts with the next job.</div>';
        return;
    }
    const {phase, steps, status} = _stepMapData;
    let html = '';

    // Completed steps: top-level ones individually, groups (e.g. the
    // individual separation chunks) collapsed into one row with sum + count.
    const groups = {};
    const lines = [];
    for (const s of steps) {
        if (s.group) {
            if (!(s.group in groups)) {
                groups[s.group] = {n: 0, secs: 0};
                lines.push({group_key: s.group});
            }
            groups[s.group].n += 1;
            groups[s.group].secs += s.secs || 0;
        } else {
            lines.push({label: s.label, secs: s.secs || 0});
        }
    }
    for (const z of lines) {
        if (z.group_key) {
            const g = groups[z.group_key];
            html += `<div class="sm-row done"><span class="sm-ico">✓</span>
                <span class="sm-label">${esc(z.group_key)} <small>(${plural(g.n, 'step')})</small></span>
                <span class="sm-secs">${_fmtSecs(g.secs)}</span></div>`;
        } else {
            html += `<div class="sm-row done"><span class="sm-ico">✓</span>
                <span class="sm-label">${esc(z.label)}</span>
                <span class="sm-secs">${_fmtSecs(z.secs)}</span></div>`;
        }
    }

    // Running phase: highlighted, with live timer.
    if (phase && status !== 'done' && status !== 'error') {
        html += `<div class="sm-row active"><span class="sm-ico sm-pulse">▶</span>
            <span class="sm-label">${esc(phase.title)}
                <small>[${phase.index}/${phase.total}]</small></span>
            <span class="sm-secs" id="smLive">…</span></div>`;
    } else if (status === 'done') {
        html += `<div class="sm-row done"><span class="sm-ico">🏁</span>
            <span class="sm-label">Done</span><span class="sm-secs"></span></div>`;
    }
    el.innerHTML = html;

    if (phase && status !== 'done' && status !== 'error') {
        const start = phase.start * 1000;
        const tick = () => {
            const live = document.getElementById('smLive');
            if (live) live.textContent = _fmtSecs((Date.now() - start) / 1000);
        };
        tick();
        _stepMapTimer = setInterval(tick, 1000);
    }
}

function appendLogLines(lines) {
    if (!lines || !lines.length) return;
    logLines.push(...lines);
    const viewer = document.getElementById('logViewer');
    const countEl = document.getElementById('logCount');
    for (const line of lines) {
        const div = document.createElement('div');
        div.className = 'log-line';
        div.textContent = line;
        viewer.appendChild(div);
    }
    // Ring buffer: keep only the last MAX_LOG_LINES lines (bound DOM/memory)
    const MAX_LOG_LINES = 2000;
    if (logLines.length > MAX_LOG_LINES) {
        logLines.splice(0, logLines.length - MAX_LOG_LINES);
    }
    while (viewer.childElementCount > MAX_LOG_LINES) {
        viewer.removeChild(viewer.firstElementChild);
    }
    countEl.textContent = logLines.length;
    // Auto-scroll if visible
    if (viewer.classList.contains('visible')) {
        viewer.scrollTop = viewer.scrollHeight;
    }
}

function clearLogViewer() {
    logLines = [];
    const viewer = document.getElementById('logViewer');
    viewer.innerHTML = '';
    document.getElementById('logCount').textContent = '0';
    _stepMapData = null;
    clearInterval(_stepMapTimer);
    const map = document.getElementById('stepMap');
    if (map?.classList.contains('visible')) renderStepMap();
}

document.addEventListener('DOMContentLoaded', () => setLogMode(_logMode));

