// ══════════════════════════════════════════════════════════════════════════
//  Queue (client side)
//
//  Important: This module does NOT control processing — it only enqueues
//  and watches. The server has a permanently running worker that works
//  through the queue. If this tab is closed, it just keeps going.
//
//  Previously JS pushed the next job after each finished one
//  (processNextQueued) — that's why the queue stalled as soon as the browser
//  was closed. That logic no longer exists.
// ══════════════════════════════════════════════════════════════════════════

// ── Selection bar ─────────────────────────────────────────────────────────

function updateSelectionUI() {
    const bar = document.getElementById('selectionBar');
    const count = selectedFiles.size;
    if (count === 0) {
        bar.style.display = 'none';
        return;
    }
    bar.style.display = '';
    // Status per selection from selectionStatus (filled in when selecting —
    // also for rows outside the loaded window); fileByName as fallback.
    let pending = 0, doneCount = 0;
    for (const name of selectedFiles) {
        const f = fileByName(name);
        const status = f ? f.status : selectionStatus.get(name);
        if (status == null) continue;
        if (status !== 'done' && status !== 'skipped' && !(f && f._processing)) pending++;
        if (status === 'done' || status === 'partial') doneCount++;
    }

    document.getElementById('selCount').textContent = `${count} selected`;

    const txBtn = document.getElementById('btnTxSelected');
    // Short: the bar is as narrow as the sidebar, and a disabled
    // "▶ Transcribe selection" was cut off where nothing was left to do
    txBtn.textContent = pending > 0 ? `▶ Transcribe ${pending}` : 'Nothing open';
    txBtn.disabled = pending === 0;

    const delBtn = document.getElementById('btnDelSelected');
    document.getElementById('btnDelSelectedLabel').textContent = doneCount;
    delBtn.style.display = doneCount > 0 ? '' : 'none';
}

function clearSelection() {
    selectedFiles.clear();
    selectionStatus.clear();
    lastClickedFile = null;
    updateSelectionUI();
    syncSelectedClasses();   // only sync classes — no list rebuild
}

// Remembers whether the modal selection came from a selection (then clear it afterwards)
let _fromSelection = false;

function startSelectedBatch() {
    const selected = [...selectedFiles].filter(name => {
        const f = fileByName(name);
        const status = f ? f.status : selectionStatus.get(name);
        return status != null && status !== 'done' && status !== 'skipped'
            && !(f && f._processing);
    });
    if (selected.length === 0) return;

    modalFiles = selected;
    _fromSelection = true;

    document.getElementById('batchModalCount').textContent = selected.length;
    document.getElementById('batchModalFilter').textContent = '(Selection)';
    document.getElementById('batchModal').classList.add('visible');
    syncCheckboxUI();
}

function startBatch() {
    const bb = document.getElementById('btnBatch');

    // Quick filter "Error"/"Cancelled" (button shows "↩ … requeue"):
    // filtered retry instead of the batch dialog — the entries already
    // exist (with settings), they just need to go back to 'pending'. Only
    // while the queue isn't actively running — then the button is the
    // stop control. The server resolves the set (paginated list = only a
    // window).
    if ((activeTab === 'error' || activeTab === 'cancelled') && !queueRunning) {
        fetchFilteredNames()
            .then(d => { if (d.names.length) retryFilteredOutcomesByNames(d.names); })
            .catch(() => Toast.error('Filter resolution failed'));
        return;
    }

    // Paused → resume
    if (queuePaused) {
        bb.disabled = true;
        fetch('/api/queue/resume', {method: 'POST'})
            .then(r => { if (!r.ok) throw new Error('resume failed'); return r.json(); })
            .then(d => {
                // Starting an empty queue used to look exactly like one that
                // was starting up: nothing happened, and nobody said why.
                // Files must first be enqueued (+Q on the file or the batch
                // dialog).
                if (d && d.pending === 0 && typeof Toast !== 'undefined') {
                    Toast.info('The queue is empty — first enqueue files '
                             + 'with +Q or via "Batch"');
                }
            })
            .catch(() => {
                bb.disabled = false;
                if (typeof Toast !== 'undefined') Toast.error('Resume failed');
            });
        return;
    }

    // Running → stop. This only PAUSES: the queue is preserved,
    // it's only deleted via "Clear" (with confirmation).
    if (queueRunning) {
        bb.textContent = 'Stopping…';
        bb.disabled = true;
        fetch('/api/queue/stop', {method: 'POST'})
            .then(r => {
                if (!r.ok) throw new Error('stop failed');
                if (typeof Toast !== 'undefined') {
                    Toast.info('Queue paused — the queue is preserved.');
                }
            })
            .catch(() => {
                // The status stream corrects the button text on the next
                // tick anyway — just don't block permanently here.
                bb.disabled = false;
                if (typeof Toast !== 'undefined') Toast.error('Stop failed');
            });
        return;
    }

    openBatchModalForFiltered();
}

/** Open the batch dialog for the filtered open files. Separate path so
 *  enqueuing is ALSO reachable while the queue is running — the big
 *  button is then the stop control, and enqueuing must still be possible
 *  while the queue is running. */
async function openBatchModalForFiltered() {
    // "All filtered open" comes from the server (name list): in a status
    // tab the tab's own set, otherwise the open ones in the active filter.
    let names = [];
    try {
        const tab = (activeTab === 'all' || isWorkTab()) ? 'pending'
            : activeTab;
        const d = await fetchFilteredNames(tab);
        names = d.status
            ? d.names.filter((n, i) =>
                d.status[i] !== 'done' && d.status[i] !== 'skipped'
                && d.status[i] !== 'orphan')
            : d.names;
    } catch (e) {
        if (typeof Toast !== 'undefined') Toast.error('Filter resolution failed');
        return;
    }
    names = names.filter(n => !isQueued(n));
    if (!names.length) return;

    modalFiles = names;
    _fromSelection = false;

    document.getElementById('batchModalCount').textContent = modalFiles.length;
    document.getElementById('batchModalFilter').textContent =
        document.getElementById('fileFilter').value.trim() || '(all)';
    document.getElementById('batchModal').classList.add('visible');
    syncCheckboxUI();
}

function closeBatchModal() {
    document.getElementById('batchModal').classList.remove('visible');
}

// ── Enqueuing ─────────────────────────────────────────────────────────────

/**
 * Enqueue files with the server. The worker picks them up itself — nothing
 * is started or pushed here.
 */
async function enqueueFiles(filenames, settings, dirMap = null, opts = {}) {
    if (!filenames.length) return 0;
    try {
        const r = await fetch('/api/queue/add', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                filenames,
                base_dir: document.getElementById('dirInput').value,
                dir_map: dirMap || {},
                // front: upload/link download — to the FRONT of the queue;
                // a paused queue runs just these, the rest stays paused (server).
                ...(opts.front ? {front: true} : {}),
                ...settings,
            })
        });
        const d = await r.json();
        if (d.error) {
            if (typeof Toast !== 'undefined') Toast.error(d.error); else alert(d.error);
            return 0;
        }
        await refreshQueue();
        renderFileList();
        if (typeof Toast !== 'undefined' && d.added > 0) {
            // The server skips entries already waiting — that belongs in
            // the message, otherwise "500 of 10,000 enqueued" looks
            // like an error.
            const skip = d.skipped > 0
                ? ` (${d.skipped.toLocaleString()} skipped — already enqueued)` : '';
            // A paused queue doesn't move — the user needs to see that,
            // otherwise they wonder why nothing happens. Front-of-queue
            // entries run anyway: just them, the rest stays paused (server).
            if (d.run_now > 0) {
                // The name of the entry that runs: of several sent, the
                // others may have been skipped
                Toast.info(d.run_now === 1
                    ? `"${d.items?.[0]?.filename ?? filenames[0]}" runs now — the rest of the queue stays paused`
                    : `${d.run_now.toLocaleString()} run now — the rest of the queue stays paused`);
            } else if (queuePaused && !opts.front) {
                Toast.info(`${d.added.toLocaleString()} enqueued${skip} — the queue is paused. Click "Resume".`);
            } else if (d.moved_up === d.added && d.added > 0) {
                // Nothing newly enqueued, just reordered — "enqueued" would
                // be a lie here and the user would look for the new entry.
                Toast.info(d.added === 1
                    ? `"${filenames[0]}" was already in the queue — now at the very front`
                    : `${d.added.toLocaleString()} were already in the queue — now at the very front`);
            } else {
                const before = d.moved_up > 0
                    ? ` (${d.moved_up.toLocaleString()} of which moved to the front)` : '';
                Toast.info(d.added === 1 && !d.skipped
                    ? `"${filenames[0]}" enqueued`
                    : `${plural(d.added, 'file')} enqueued${before}${skip}`);
            }
        } else if (typeof Toast !== 'undefined' && d.skipped > 0) {
            Toast.info(`Nothing enqueued — all ${d.skipped.toLocaleString()} are already in the queue or finished`);
        }
        return d.added || 0;
    } catch (e) {
        if (typeof Toast !== 'undefined') Toast.error('Enqueuing failed: ' + e);
        return 0;
    }
}

function _batchModalSettings() {
    return txCollect('batch');
}

async function confirmBatch() {
    if (_txIdle('batch')) return;
    closeBatchModal();
    const toQueue = modalFiles.slice();
    if (!toQueue.length) return;

    await enqueueFiles(toQueue, _batchModalSettings(), queueDirMap);

    if (_fromSelection) {
        selectedFiles.clear();
        updateSelectionUI();
        _fromSelection = false;
    }
    modalFiles = [];
}

/** Enqueue a single file with the settings from the single-file modal. */
function addToQueue(filename) {
    if (isQueued(filename)) return;
    enqueueFiles([filename], _gatherTxSettings());
}

// ── Reordering / removing ────────────────────────────────────────────────

async function moveQueueItem(itemId, direction) {
    try {
        const r = await fetch('/api/queue/move', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({id: itemId, direction})
        });
        if (!r.ok) throw new Error('move failed');
        await refreshQueue();
        renderQueueList();
        renderFileList();
    } catch (e) {
        if (typeof Toast !== 'undefined') Toast.error('Move failed');
    }
}

/** Pull a waiting entry to the front of the queue.
 *  ▲ moves it one position — with many thousands waiting, only the jump helps. */
async function queueItemToTop(itemId) {
    try {
        const r = await fetch('/api/queue/move', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({id: itemId, top: true})
        });
        const d = await r.json();
        if (!r.ok) throw new Error(d.error || 'move failed');
        await refreshQueue();
        renderQueueList();
        renderFileList();
        // Paused stays paused (reordering is not a start command) —
        // otherwise the user waits in vain for "to the front" to start
        // running.
        if (typeof Toast !== 'undefined') {
            Toast.info(queuePaused
                ? '⤒ Moved to the front — the queue is paused. Click "Resume".'
                : '⤒ Moved to the front — up next');
        }
    } catch (e) {
        if (typeof Toast !== 'undefined') Toast.error('Moving to front failed: ' + e.message);
    }
}

async function removeFromQueue(itemId) {
    try {
        const r = await fetch('/api/queue/remove', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({id: itemId})
        });
        if (!r.ok) throw new Error('remove failed');
        await refreshQueue();
        renderQueueList();
        renderFileList();
    } catch (e) {
        if (typeof Toast !== 'undefined') Toast.error('Remove failed');
    }
}

/** Discard all waiting entries (the running job is left untouched). */
async function clearQueue() {
    if (!queuePending) return;
    if (!confirm(`Remove ${plural(queuePending, 'waiting entry', 'waiting entries')} from the queue?`)) return;
    await fetch('/api/queue/clear', {method: 'POST'}).catch(() => {});
    await refreshQueue();
    renderQueueList();
    renderFileList();
}

/**
 * Total abort: discard the running job AND delete all waiting entries.
 * The queue is empty afterwards. Unlike "Stop" (pauses, preserves everything).
 */
// Set by abortQueue: the server clears its abort flag once the worker is
// idle, often before the drain event reaches us — the label would then
// claim "Queue processed" for a queue the user just threw away.
let _queueAbortedByUser = false;

async function abortQueue() {
    if (!queueRunning && !queuePending) return;
    const details = [];
    // Queue lanes: name every running file — the abort hits all of them.
    const running = queueRunningItems.length ? queueRunningItems.map(r => r.filename)
        : (queueCurrentFile ? [queueCurrentFile] : []);
    if (running.length === 1) details.push(`the running file (${running[0]}) is aborted`);
    else if (running.length > 1) details.push(`the ${running.length} running files (${running.join(', ')}) are aborted`);
    if (queuePending) details.push(`${queuePending.toLocaleString()} waiting `
        + `${queuePending === 1 ? 'entry is' : 'entries are'} removed`);
    const msg = `Abort the queue completely?\n\n`
        + (details.length ? details.join(' and ').replace(/^./, c => c.toUpperCase()) + '.\n\n' : '')
        + 'This cannot be undone.';
    if (!confirm(msg)) return;

    const btn = document.getElementById('btnAbortQueue');
    if (btn) { btn.disabled = true; btn.textContent = 'Aborting…'; }
    try {
        _queueAbortedByUser = true;
        const r = await fetch('/api/queue/abort', {method: 'POST'});
        if (!r.ok) { _queueAbortedByUser = false; throw new Error('abort failed'); }
        const d = await r.json().catch(() => ({}));
        if (typeof Toast !== 'undefined') {
            Toast.info(`Aborted — queue cleared${d.removed ? ` (${d.removed} removed)` : ''}.`);
        }
        await refreshQueue();
        if (typeof renderQueueList === 'function') renderQueueList();
        if (typeof renderFileList === 'function') renderFileList();
    } catch(e) {
        if (typeof Toast !== 'undefined') Toast.error('Abort failed');
    } finally {
        if (btn) { btn.disabled = false; btn.textContent = '✕ Clear the entire queue'; }
    }
}

/**
 * Move failed and cancelled entries back into the queue.
 *
 * Without this path a failure was final: the entry stayed at 'error',
 * no longer showed up in any list, and could only be revived by
 * re-enqueuing the file.
 */
/** Filtered retry: re-enqueue the queue entries for the given file names
 *  (quick filter "Error"/"Cancelled") — only these. */
async function retryFilteredOutcomesByNames(names) {
    const ids = names.map(n => queueOutcomeOf(fileByName(n) || {name: n})?.id).filter(id => id != null);
    if (!ids.length) return;
    try {
        const r = await fetch('/api/queue/retry', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ids}),
        });
        const d = await r.json();
        if (!r.ok) throw new Error(d.error || `HTTP ${r.status}`);
        if (typeof Toast !== 'undefined') Toast.success(`↩ ${d.requeued} re-enqueued`);
        refreshQueue();
    } catch (e) {
        if (typeof Toast !== 'undefined') Toast.error('Re-enqueuing failed: ' + e.message);
    }
}

async function retryFailedQueue() {
    const n = queueRetryable;
    if (!n) return;
    const what = [queueFailed ? `${queueFailed} failed` : '',
                  queueCancelled ? `${queueCancelled} cancelled` : ''].filter(Boolean).join(' + ');
    if (!confirm(`Put ${what || n} ${n === 1 ? 'entry' : 'entries'} back into the queue?`)) return;
    try {
        const r = await fetch('/api/queue/retry', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: '{}',                    // without id = all
        });
        const d = await r.json().catch(() => ({}));
        if (!r.ok) {
            Toast.error(d.error || 'Re-enqueuing failed');
            return;
        }
        Toast.success(`${d.requeued} re-enqueued`);
    } catch (e) {
        Toast.error('Re-enqueuing failed');
        return;
    }
    await refreshQueue();
    renderQueueList();
}

// ── Applying server state ────────────────────────────────────────────────

// The server reports `running` as soon as the worker thread exists and
// `paused` whenever it is held — also with nothing to do. An EMPTY queue is
// idle for the interface: otherwise a server started with
// queue.autostart: false offered "▶ Resume queue" over an empty queue, and
// the click did nothing visible.
function _queueIdle(d) {
    return !d.current_file && !(d.pending > 0);
}

// The running entries of a payload. A server without queue lanes sends no
// running_items (or an empty list beside a current file) — then its one
// current file is the list. Keyed by identity, not current_item_id: only
// /api/queue sends that, the stream doesn't, and the key must not flip
// between the two.
function _runningItemsFrom(d) {
    const items = Array.isArray(d.running_items) ? d.running_items : [];
    if (items.length || !d.current_file) return items;
    return [{item_id: null, job_id: d.current_job_id || '',
             filename: d.current_file, abs: d.current_abs || '', dir: d.current_dir || '',
             progress: d.job_progress, message: d.job_message,
             job_elapsed_sec: d.job_elapsed_sec}];
}

// Which entries run — the change signal for the row marks, the sidebar
// counters and the Queue tab. Not their job ids: a row's ✕ looks its job up
// when clicked (fileactions.js: cancelRunningItem).
function _runningSig(items) {
    return items.map(runningKey).join(',');
}

function applyQueueState(d) {
    const idle = _queueIdle(d);
    queueRunning = !!d.running && !idle;
    queuePaused = !!d.paused && !idle;
    queueSolo = queuePaused ? (d.solo || 0) : 0;
    queueTotal = d.total || 0;
    queueProcessed = d.processed || 0;
    queueDone = d.done || 0;
    queueFailed = d.failed || 0;
    queueSkipped = d.skipped || 0;
    queueCancelled = d.cancelled || 0;
    queuePending = d.pending || 0;
    queueRetryable = d.retryable || 0;
    // A running file changed (started, finished, stopped): the sidebar
    // counters (Active/Open/Done) come from the server — fetch them anew,
    // or "Active 1" lingers after a stop. With queue lanes any of them,
    // not only the oldest.
    const running = _runningItemsFrom(d);
    const runningChanged = (d.current_file || null) !== queueCurrentFile
        || _runningSig(running) !== _runningSig(queueRunningItems);
    if (runningChanged && typeof loadServerStats === 'function') loadServerStats();
    queueCurrentFile = d.current_file || null;
    queueCurrentAbs = d.current_abs || null;
    queueCurrentDir = d.current_dir || null;
    queueRunningItems = running;
    queueRunningCount = typeof d.running_count === 'number' ? d.running_count : running.length;
    if (runningChanged && typeof refreshRunningPlaceholder === 'function') refreshRunningPlaceholder();
    if (Array.isArray(d.items)) queueItems = d.items;
    // Failed/cancelled → row marker + quick filter. Only arrives over the
    // wire on change; only re-render then.
    if (Array.isArray(d.failed_items)) queueFailedItems = d.failed_items;
    if (Array.isArray(d.failed_items) && setQueueOutcomes(d.failed_items)) {
        renderFileList();
    }
    _renderRetryBar();
    if (Array.isArray(d.system_warnings)) _renderSystemWarnings(d.system_warnings);
    // An open queue editor reloads debounced on a generation change.
    if (typeof qeGenerationSeen === 'function' && d.generation !== undefined) qeGenerationSeen(d.generation);
    // Its header chips (pending/running/errors) read the globals set above
    if (typeof qeRenderHead === 'function') qeRenderHead();
}

/** System-wide warnings (e.g. separation model fallback) — UNMISSABLE
 *  red banner at the top, stays until acknowledged. */
function _renderSystemWarnings(warnings) {
    let banner = document.getElementById('systemWarnBanner');
    if (!warnings.length) {
        if (banner) banner.remove();
        return;
    }
    if (!banner) {
        banner = document.createElement('div');
        banner.id = 'systemWarnBanner';
        banner.style.cssText =
            'position:fixed;top:0;left:0;right:0;z-index:9999;' +
            'background:#7f1d1d;color:#fecaca;border-bottom:3px solid #ef4444;' +
            'padding:10px 18px;font-size:0.95rem;font-weight:600;' +
            'display:flex;align-items:center;gap:14px;box-shadow:0 2px 12px rgba(0,0,0,.5)';
        document.body.prepend(banner);
    }
    const texts = warnings.map(w => esc(w.text)).join('<br>');
    banner.innerHTML = `<span style="flex:1">${texts}</span>
        <button onclick="fetch('/api/system_warnings/acknowledge',{method:'POST'})
                 .then(()=>document.getElementById('systemWarnBanner')?.remove())"
                style="background:#fecaca;color:#7f1d1d;border:none;border-radius:6px;
                       padding:6px 14px;font-weight:700;cursor:pointer">Got it</button>`;
}

/** Retry item in the ⋯ menu — and a dot on ⋯ — while something can be re-queued. */
function _renderRetryBar() {
    const bar = document.getElementById('batchRetry');
    if (!bar) return;
    const dot = document.getElementById('batchMoreDot');
    if (dot) dot.hidden = !queueRetryable;
    if (!queueRetryable) { bar.style.display = 'none'; return; }
    bar.style.display = '';
    const parts = [];
    if (queueFailed) parts.push(`${queueFailed} failed`);
    if (queueCancelled) parts.push(`${queueCancelled} cancelled`);
    document.getElementById('batchRetryLabel').textContent = parts.join(', ');
}

/** One-off fetch of the queue state (after own changes). */
async function refreshQueue() {
    try {
        const r = await fetch('/api/queue');
        if (!r.ok) return;
        applyQueueState(await r.json());
        updateCounts();
    } catch (e) { /* the stream catches up on the next tick */ }
}

// ── Persistent status stream ─────────────────────────────────────────────

let _queueReconnectTimer = null;

// Debounced library scan.
// A scan over a large collection takes seconds (several seconds at a few
// hundred thousand files) and re-renders the whole list afterward. Triggering one after
// EVERY finished file is a barrage for quickly-failing files (broken
// recording → error in <1s) that grinds server and tab to a halt.
// Multiple requests are therefore coalesced into a single scan.
//
// In normal operation this full scan is NO LONGER triggered during the
// queue — see _refreshOneFile(). It remains for the cases where the whole
// list can really change (queue drained, reconnect).
let _scanTimer = null;
let _scanRunning = false;
let _scanAgain = false;
const SCAN_DEBOUNCE_MS = 4000;
// More rows than this finishing in one tick: one full scan is cheaper than
// that many single reloads (queue lanes: at most 4 run at once).
const REFRESH_ONE_MAX = 4;

function _scheduleScan() {
    if (_scanRunning) { _scanAgain = true; return; }
    if (_scanTimer) return;              // already scheduled
    _scanTimer = setTimeout(() => {
        _scanTimer = null;
        _scanRunning = true;
        // `background`: The user hasn't touched anything — so the list
        // must not jump either (see reloadFileList).
        scanFiles({background: true}).finally(() => {
            _scanRunning = false;
            if (_scanAgain) { _scanAgain = false; _scheduleScan(); }
        });
    }, SCAN_DEBOUNCE_MS);
}

// Incremental list update for ONE finished file.
//
// A finished transcription changes exactly one row. Reloading the whole
// library for that used to be the reason for the constant "Loading from
// DB…": several seconds per scan on a large library, triggered after every
// file (every few seconds), and exactly when the queue worker is already busy writing
// to the DB. Now: a point query that replaces the entry in place.
//
// If that fails (file unknown to the DB, network error), the file falls
// back to the debounced full scan — the list must not stay stale just
// because a single update didn't work.
async function _refreshOneFile(name, abs) {
    if (!name) return;
    try {
        // `abs` lets the server say "not in the open folder" — the queue
        // runs across roots, and `name` is relative to the entry's root.
        const r = await fetch('/api/scan_entry/' + encodeURIComponent(name).replace(/%2F/g, '/')
                              + (abs ? '?abs=' + encodeURIComponent(abs) : ''));
        if (!r.ok) { _scheduleScan(); return; }
        const d = await r.json();
        if (d.outside) return;             // finished elsewhere: nothing to update here
        if (!d.file) { _scheduleScan(); return; }
        // The list's name for it — relative to the open folder, which
        // after a folder jump isn't the entry's root
        name = d.file.name || name;

        // Replace the row in the server buffer (re-renders the window
        // itself). If the file lies outside the window, an aggregate
        // refresh is enough — the row gets loaded fresh on scroll anyway.
        if (typeof updateLoadedEntry === 'function') updateLoadedEntry(name, d.file);
        if (typeof loadServerStats === 'function') loadServerStats();

        // Done transition WITHOUT tearing the UI apart: never switch the
        // view automatically. Instead:
        //  - File is currently OPEN → subtle banner over the transcript
        //    area ("show" loads the fresh transcript on click).
        //  - otherwise → success toast with click action. That way the
        //    file is also findable again if it disappeared from the
        //    ⟳-active tab ("gone through → search again" is no longer
        //    needed).
        if (d.file.status === 'done') {
            const base = name.split('/').pop();
            if (typeof activeFile !== 'undefined' && activeFile === name) {
                _showDoneBanner(d.file);
            } else if (typeof Toast !== 'undefined') {
                Toast.success(`Transcript finished: ${base} — show`, 12000,
                              () => selectFile(d.file));
            }
        }
    } catch (e) {
        _scheduleScan();
    }
}

// Banner "Transcript finished" over the open (old) view — deliberately
// NO auto-reload: whoever is currently reading or editing would otherwise
// lose their state. The banner disappears on click or with the next file
// switch (selectFile rebuilds the area).
function _showDoneBanner(entry) {
    const area = document.getElementById('transcriptArea');
    if (!area) return;
    document.getElementById('doneBanner')?.remove();
    const b = document.createElement('div');
    b.id = 'doneBanner';
    b.style.cssText = 'display:flex;align-items:center;gap:10px;padding:10px 14px;'
        + 'margin:8px;border:1px solid #22c55e;border-left:3px solid #22c55e;'
        + 'border-radius:8px;background:#22c55e18;font-size:0.85rem';
    b.innerHTML = '<span style="color:#22c55e;font-weight:700">✅ Transcript finished</span>'
        + '<span style="color:var(--tx-3);flex:1">This file was just transcribed.</span>'
        + '<button style="background:#22c55e;border:none;border-radius:6px;padding:5px 12px;'
        + 'cursor:pointer;font-family:inherit;font-size:0.8rem;color:#08130b;font-weight:700">'
        + 'Show</button>';
    b.querySelector('button').onclick = () => { b.remove(); selectFile(entry); };
    area.prepend(b);
}

/**
 * Connects to the queue stream and stays connected as long as the page is
 * open. If the connection drops, it reconnects — the server keeps running
 * unfazed in the meantime.
 */
function connectQueueSSE() {
    if (queueSSE) { queueSSE.close(); queueSSE = null; }
    if (_queueReconnectTimer) { clearTimeout(_queueReconnectTimer); _queueReconnectTimer = null; }

    let lastProcessed = -1;
    let wasRunning = false;
    // Signatures for change detection — see below why this is vital.
    let lastItemsSig = null;
    let lastStateSig = null;
    let lastRunningSig = null;
    // The previous tick's running entries (key → entry): the ones missing
    // now are the ones that just finished. null = unknown (fresh connection).
    let lastRunning = null;
    // A fresh connection attaches the live stream anew on its first tick
    watchedJobId = null;

    queueSSE = new EventSource('/api/queue/stream');

    queueSSE.onmessage = (ev) => {
        let d;
        try { d = JSON.parse(ev.data); } catch (e) { return; }

        applyQueueState(d);
        // The overlay's job before its progress is drawn (below): a new
        // job resets the overlay, and the same tick fills it again.
        syncWatchedJob();
        currentJobId = d.current_job_id || null;

        // ── Cheap: allowed to run every tick (a handful of DOM nodes) ──
        _renderQueueProgress(d);

        // ── Expensive: ONLY on a real change ────────────────────────────
        // This stream is now permanently connected and ticks every second.
        // renderFileList() filters and sorts the entire library (tens of
        // thousands of files) and rebuilds up to 2000 DOM nodes — running that
        // every second grinds the browser tab to a halt. This didn't show
        // up before because the stream only existed while a batch was
        // running.

        // Queue list: only if the entries have changed.
        // d.items is MISSING if the server saw no change — that's
        // "unchanged", NOT "empty". A missing items field must therefore
        // not clear the list.
        if (Array.isArray(d.items)) {
            const sig = d.items.map(i => i.id).join(',');
            if (sig !== lastItemsSig) {
                lastItemsSig = sig;
                renderQueueList();
            }
        }

        // Mark the running files in the sidebar: only on change. With
        // queue lanes every one of them, not just the oldest. From the list,
        // not current_file: a tick that has the list but no current file yet
        // would clear the marks, and the next ticks see no change to fix it.
        const runningSig = _runningSig(queueRunningItems);
        if (runningSig !== lastRunningSig) {
            // Via the server buffer: a running file can lie outside the
            // loaded window — then there's simply nothing to mark.
            const oldest = queueRunningItems.length ? queueRunningItems[0].filename : null;
            if (typeof setProcessingFile === 'function') setProcessingFile(oldest);
            lastRunningSig = runningSig;
        }

        // File list: only if something that it even displays has changed.
        const stateSig = [d.running, d.paused, d.pending, d.processed,
                          runningSig, lastItemsSig].join('|');
        if (stateSig !== lastStateSig) {
            lastStateSig = stateSig;
            renderFileList();
        }
        // The running rows' progress in the Queue tab: in place, every tick
        if (typeof updateRunningRows === 'function') updateRunningRows();

        // A file finished → reload ONLY its row instead of the whole
        // library. The finished ones are the entries that ran on the last
        // tick and don't anymore — with queue lanes not necessarily the
        // oldest. Every one that left gets its row reloaded (one that a
        // stop put back is simply shown as open again).
        //
        // The full scan has to run when that can't account for everything:
        // more finished than left (quickly-failing files run through between
        // two ticks), or more left than single reloads are worth. The first
        // tick of a connection has nothing to compare with and reloads nothing.
        const nowRunning = new Map(queueRunningItems.map(r => [runningKey(r), r]));
        const left = lastRunning
            ? [...lastRunning.keys()].filter(k => !nowRunning.has(k)).map(k => lastRunning.get(k))
            : [];
        const finished = lastProcessed >= 0 ? Math.max(0, d.processed - lastProcessed) : 0;
        if (left.length > REFRESH_ONE_MAX || finished > left.length) {
            _scheduleScan();
        } else {
            left.forEach(r => _refreshOneFile(r.filename, r.abs));
        }
        lastRunning = nowRunning;
        lastProcessed = d.processed;

        if (d.new_log?.length) appendLogLines(d.new_log);
        if (typeof updateStepMap === 'function') updateStepMap(d);

        // Queue just drained → finish up (the stream stays open)
        if (wasRunning && !d.running) {
            _onQueueDrained(d);
        }
        wasRunning = d.running;
    };

    queueSSE.onerror = () => {
        if (queueSSE) queueSSE.close();
        queueSSE = null;
        if (jobSSE) { jobSSE.close(); jobSSE = null; }
        if (_queueReconnectTimer) clearTimeout(_queueReconnectTimer);
        // Connection gone does NOT mean processing has stopped — just that
        // we're not watching right now. So just reconnect.
        _queueReconnectTimer = setTimeout(() => {
            _queueReconnectTimer = null;
            connectQueueSSE();
        }, 2000);
    };
}

// The running entry the progress overlay shows: the open file's when it
// runs, else the oldest. With queue lanes several run at once — the
// oldest's progress over another running file would be someone else's.
function _watchedRunningItem() {
    if (!queueRunningItems.length) return null;
    return (activeFile && activeFileObj
            && queueRunningItems.find(r => runningItemMatches(r, activeFileObj)))
        || queueRunningItems[0];
}

// Is the overlay the watched entry's? Yes when no file is open (it shows
// the queue's oldest) or the open file runs (then it IS the watched one).
// An open file that no longer runs keeps its own overlay.
function _overlayShowsWatched() {
    return !activeFile || isRunningFile(activeFileObj);
}

// The overlay shows a finished run's outcome (✅/❌ with "✕ Close").
function _overlayShowsOutcome() {
    return document.getElementById('btnCancelJob').onclick === _closeProgressOverlay;
}

// When the watched entry started: a file run again (retry) gets the same
// job id, and its new run must still be attached anew.
let _watchedStart = null;

// The overlay's ✕ re-armed from an outcome for another run is inert this
// long: a click meant for "✕ Close" must not cancel a file nobody chose.
const OVERLAY_REARM_INERT_MS = 1500;
let _overlayInert = false;
let _overlayInertTimer = null;

// The overlay's ✕ while it cancels (bound by _resetProgressChrome and
// _closeProgressOverlay): no effect in the inert moment.
function _overlayCancelClick() {
    if (_overlayInert) return;
    return cancelCurrentJob();
}

// The overlay's button was just re-armed from an outcome: inert for a
// moment, and with no file open it names the file it would cancel — the
// empty view doesn't say which file the overlay follows now.
function _overlayRearmInert(w) {
    const btn = document.getElementById('btnCancelJob');
    if (!activeFile && w) btn.textContent = `✕ Cancel ${_shortName(w.filename)}`;
    _overlayInert = true;
    btn.disabled = true;
    clearTimeout(_overlayInertTimer);
    _overlayInertTimer = setTimeout(() => {
        _overlayInertTimer = null;
        _overlayInert = false;
        // Live again — unless something else took the button meanwhile (an
        // outcome's "✕ Close" is live already, a cancel's "Stopping…" stays off)
        if (btn.onclick === _overlayCancelClick && btn.textContent !== 'Stopping…') btn.disabled = false;
    }, OVERLAY_REARM_INERT_MS);
}

// Attach the overlay's live stream (log, segments, done state) to the
// watched job. Every queue tick, and right when the open file changes
// (selectFile) — not a second late with the previous job's log on screen.
function syncWatchedJob() {
    const w = _watchedRunningItem();
    // Nothing runs: the next run of the same file has the same job id and
    // must still attach anew — also while an outcome is up (below)
    if (!w) watchedJobId = null;
    const visible = document.getElementById('progressOverlay').classList.contains('visible');
    if (activeFile && !isRunningFile(activeFileObj) && visible) {
        // The open file's run has ended and its overlay is still up. Its
        // outcome stays until closed: re-armed for another lane's file, the
        // click meant to close would cancel THAT file. A run that ended
        // without an outcome on screen leaves nothing to show — it goes.
        if (_overlayShowsOutcome()) return;
        _closeProgressOverlay();
    }
    // With no file open the overlay follows the queue, also away from an
    // outcome on screen — then its button is inert for a moment (below).
    const fromOutcome = visible && _overlayShowsOutcome();
    const jobId = (w && w.job_id) || null;
    const start = w ? (w.started_at ?? null) : null;
    let rearmed = false;
    if (jobId && (jobId !== watchedJobId || start !== _watchedStart)) {
        // New job → reset the overlay and attach the live stream (log +
        // segments). Without the reset it would still show "✅ Done!" and
        // the close button of the previous job.
        watchedJobId = jobId;
        _watchedStart = start;
        _resetProgressChrome();
        connectJobSSEForLive(jobId, w.filename, w.abs);
        // Its last known progress right away, not "Starting…" for a tick
        _paintWatchedProgress(w);
        rearmed = true;
    } else if (w && !jobId && (jobSSE || watchedJobId)) {
        // The watched file's lane hasn't made its job yet, but the overlay
        // still follows another one (its stream, or what it left there):
        // "Starting…" for this file, not the other's progress, log or outcome.
        if (jobSSE) { jobSSE.close(); jobSSE = null; }
        watchedJobId = null;
        _resetProgressChrome();
        rearmed = true;
    }
    // Re-armed right after an outcome, or again within that moment: inert
    if (rearmed && (fromOutcome || _overlayInert)) _overlayRearmInert(w);
}

function _baseName(path) {
    return String(path || '').split('/').pop();
}

// A file's name, short enough for a button
function _shortName(path, max = 32) {
    const base = _baseName(path);
    return base.length > max ? base.slice(0, max - 1) + '…' : base;
}

function _renderQueueProgress(d) {
    const prog = document.getElementById('batchProgress');
    const btn = document.getElementById('btnBatch');

    const abortBtn = document.getElementById('btnAbortQueue');

    const cancelItem = document.getElementById('btnCancelInline');
    if (!d.running || _queueIdle(d)) {
        // Idle: the button belongs to the file filter again — its label and
        // enabled state come from renderBatchButtons, not a blanket reset
        // (that used to re-enable "Transcribe" on the Done tab).
        if (abortBtn) abortBtn.style.display = 'none';
        if (cancelItem) cancelItem.style.display = 'none';
        if (typeof renderBatchButtons === 'function') renderBatchButtons();
        return;
    }
    // The running entry the overlay follows — and the ⋯ menu cancels
    const w = _watchedRunningItem();
    const running = queueRunningTotal();
    // Cancelling one file only makes sense while one runs and the queue
    // isn't cancelling them all anyway (stop, abort: the abort flag). A
    // queue that paused itself (VRAM) lets its running files finish — each
    // stays cancellable. With several running, the item names which one.
    if (cancelItem) {
        cancelItem.style.display = (running && !d.abort) ? '' : 'none';
        cancelItem.textContent = (queueRunningItems.length > 1 && w)
            ? `✕ Cancel ${_baseName(w.filename)}` : '✕ Cancel the current file';
    }

    prog.classList.add('visible');
    prog.classList.toggle('paused', !!d.paused);
    btn.disabled = false;
    // Offer a total abort as soon as something is running or waiting (also in the paused state)
    if (abortBtn) abortBtn.style.display = '';
    if (d.paused) {
        btn.className = 'btn-batch';
        btn.textContent = d.pending > 0
            ? `▶ Resume (${d.pending.toLocaleString()} waiting)`
            : '▶ Resume queue';
    } else {
        btn.className = 'btn-batch stop';
        btn.textContent = '⏸ Stop';
        btn.title = `Only pauses — the queue is fully preserved (${(d.pending || 0).toLocaleString()} waiting)`;
    }

    const pct = queueTotal > 0 ? (queueProcessed / queueTotal * 100) : 0;
    document.getElementById('batchProgFill').style.width = pct + '%';
    // Processed/total in the line; the breakdown in the tooltip — failed
    // ones get their own line (retry) below anyway.
    const count = document.getElementById('batchProgCount');
    count.textContent = `${queueProcessed.toLocaleString()}/${queueTotal.toLocaleString()}`;
    count.title = `${queueDone.toLocaleString()} done · ${queueFailed} failed · ${queueSkipped} skipped`;
    // No file name here — the running file shows in the ⏳ Queue tab. Only
    // a state the count alone doesn't tell.
    const label = document.getElementById('batchProgLabel');
    // Stop aborts the running files at their next checkpoint and puts them
    // back in the queue (the abort flag) — until then "Paused" next to a
    // moving bar looks broken. With queue lanes it says how many are still
    // winding down. A queue that paused itself after a VRAM failure sets
    // no abort flag: its running files are not cancelled but finish.
    label.textContent = !d.paused ? ''
        : !running ? 'Paused'
        : d.abort ? (running > 1 ? `Stopping ${running}…` : 'Stopping…')
        : `Paused — ${running} finishing`;

    _paintWatchedProgress(w);
}

// Progress of the watched file into the overlay (only if the user is also
// viewing it — otherwise the overlay would jump to a different file). The
// watched entry IS the viewed running file, so with queue lanes the numbers
// belong to the file underneath.
function _paintWatchedProgress(w) {
    if (!w || w.progress == null || !jobSSE) return;
    if (!_overlayShowsWatched()) return;
    document.getElementById('progressOverlay').classList.add('visible');
    document.getElementById('progSpin').style.display = '';
    document.getElementById('progSpin').classList.remove('stopped');
    document.getElementById('progBar').style.width = w.progress + '%';
    document.getElementById('progPct').textContent = w.progress + '%';
    document.getElementById('progMsg').textContent = w.message || w.filename;
    if (w.job_elapsed_sec != null) {
        document.getElementById('progElapsed').textContent = _fmtElapsed(w.job_elapsed_sec);
    }
}

function _fmtElapsed(e) {
    if (e < 60) return Math.round(e) + 's';
    if (e < 3600) return (e / 60).toFixed(1) + 'min';
    return (e / 3600).toFixed(1) + 'h';
}

function _onQueueDrained(d) {
    if (jobSSE) { jobSSE.close(); jobSSE = null; }
    currentJobId = null;
    watchedJobId = null;
    queueCurrentFile = null;
    queueCurrentAbs = null;
    queueCurrentDir = null;
    queueRunningItems = [];
    queueRunningCount = 0;
    files.forEach(f => f._processing = false);
    filesChanged();

    const prog = document.getElementById('batchProgress');
    const btn = document.getElementById('btnBatch');

    if (d.abort || _queueAbortedByUser) {
        _queueAbortedByUser = false;
        document.getElementById('batchProgLabel').textContent = 'Queue aborted';
    } else {
        document.getElementById('batchProgFill').style.width = '100%';
        document.getElementById('batchProgLabel').textContent = 'Queue processed';
        playDoneSound();
    }
    // The last poll was taken while the final file was still running —
    // without this the line keeps saying e.g. "1/2" for a finished queue.
    if (d.total) {
        document.getElementById('batchProgCount').textContent =
            `${(d.processed || 0).toLocaleString()}/${d.total.toLocaleString()}`;
    }

    btn.className = 'btn-batch';
    if (typeof renderBatchButtons === 'function') renderBatchButtons();
    document.getElementById('progressOverlay').classList.remove('visible');

    scanFiles();
    setTimeout(() => {
        prog.classList.remove('visible');
        document.getElementById('batchProgLabel').textContent = '';
        document.getElementById('batchProgCount').textContent = '';
        renderFileList();
    }, 3000);
}

// ── Live stream of the running job (log, segments, done state) ───────────

function connectJobSSEForLive(jobId, filename, abs) {
    if (jobSSE) { jobSSE.close(); jobSSE = null; }
    let liveStarted = false;
    // Is the user currently looking at THIS file? Via identity, not the
    // root-dependent name (state.js: isRunningFile).
    const watches = () => (abs && activeFileObj && activeFileObj.abs_path)
        ? activeFileObj.abs_path === abs : activeFile === filename;

    jobSSE = new EventSource(`/api/status/${jobId}`);
    jobSSE.onmessage = (ev) => {
        let d; try { d = JSON.parse(ev.data); } catch (e) { return; }

        if (d.new_segments?.length && watches()) {
            if (!liveStarted) {
                document.getElementById('liveBadge').style.display = 'flex';
                document.getElementById('transcriptArea').innerHTML =
                    `<div style="padding:14px 20px;font-size:0.9rem;font-weight:600">${esc(filename)}</div>` +
                    '<div class="tx-search"><input type="text" disabled placeholder="Live preview – speakers are assigned at the end"></div><div id="segList" data-live="1"></div>';
                liveStarted = true;
            }
            const container = document.getElementById('segList');
            if (container) {
                d.new_segments.forEach(s => {
                    const div = document.createElement('div');
                    div.className = 'seg';
                    div.innerHTML = `
                        <div class="ts">${esc(s.start_fmt)} → ${esc(s.end_fmt)}</div>
                        <div><div class="txt">${esc(s.text)}</div></div>`;
                    container.appendChild(div);
                });
                container.parentElement.scrollTop = container.parentElement.scrollHeight;
            }
        }

        if (d.new_log?.length) appendLogLines(d.new_log);
        if (typeof updateStepMap === 'function') updateStepMap(d);

        if (d.status === 'done') {
            jobSSE.close();
            jobSSE = null;
            document.getElementById('progBar').style.width = '100%';
            document.getElementById('progMsg').textContent = '✅ Done!';
            document.getElementById('progSpin').style.display = 'none';
            document.getElementById('progSpin').classList.add('stopped');
            playDoneSound();
            document.getElementById('progPct').textContent = d.result?.processing_time || '';
            document.getElementById('progElapsed').textContent = '';
            const cancelBtn = document.getElementById('btnCancelJob');
            cancelBtn.textContent = '✕ Close';
            cancelBtn.disabled = false;
            cancelBtn.onclick = _closeProgressOverlay;
            document.getElementById('liveBadge').style.display = 'none';
            if (d.log_file) {
                const dl = document.getElementById('logDownloadLink');
                dl.href = `/api/logs/download/${encodeURIComponent(d.log_file)}`;
                dl.style.display = 'inline';
            }
            _applyDoneResult(filename, d.result);
            updateCounts();
            renderFileList();
            if (watches()) {
                if (d.result) {
                    activeTranscript = d.result;
                    activeSpeaker = null;
                    renderTranscript();
                } else {
                    loadTranscript(filename);
                }
                // Close the overlay so the transcript becomes visible
                setTimeout(() => {
                    const overlay = document.getElementById('progressOverlay');
                    if (overlay.classList.contains('visible')) _closeProgressOverlay();
                }, 2000);
            }
        } else if (d.status === 'error') {
            jobSSE.close();
            jobSSE = null;
            document.getElementById('liveBadge').style.display = 'none';
            if (d.will_retry) {
                // Not the end of the file: the queue takes it up again (out of
                // GPU memory — with less VRAM, once the card has room) and its
                // stream follows the new attempt like any running file. A ❌
                // here would read as a failed upload.
                document.getElementById('progMsg').textContent =
                    '🔁 Out of GPU memory — the queue tries again with less VRAM…';
                return;
            }
            document.getElementById('progMsg').textContent = '❌ ' + (d.error || d.message);
            document.getElementById('progSpin').style.display = 'none';
            document.getElementById('progSpin').classList.add('stopped');
            if (d.log_file) {
                const dl = document.getElementById('logDownloadLink');
                dl.href = `/api/logs/download/${encodeURIComponent(d.log_file)}`;
                dl.style.display = 'inline';
            }
            const cancelBtn = document.getElementById('btnCancelJob');
            cancelBtn.textContent = '✕ Close';
            cancelBtn.disabled = false;
            cancelBtn.onclick = _closeProgressOverlay;
            const f = fileByName(filename);
            if (f) { f._processing = false; filesChanged(); }
            renderFileList();
        }
    };

    jobSSE.onerror = async (e) => {
        // Don't give up immediately on transient drops (proxy timeout,
        // standby) — let the EventSource reconnect do its job as long as
        // the job is running.
        const src = e.target;
        try {
            const r = await fetch('/api/active_jobs');
            const aj = await r.json();
            if ((aj.jobs || []).some(j => j.jobId === jobId)) return;
        } catch (err) { return; }
        src.close();
        if (jobSSE === src) jobSSE = null;
    };
}

function _closeProgressOverlay() {
    document.getElementById('progressOverlay').classList.remove('visible');
    const cb = document.getElementById('btnCancelJob');
    cb.onclick = _overlayCancelClick;
    cb.textContent = '✕ Cancel';
}

/**
 * Bring the overlay back to its initial state when a new job starts.
 * Deliberately does NOT make the overlay visible — _renderQueueProgress
 * decides that (only if the user is viewing the running file).
 */
function _resetProgressChrome() {
    document.getElementById('liveBadge').style.display = 'none';
    document.getElementById('progBar').style.width = '0%';
    document.getElementById('progMsg').textContent = 'Starting…';
    document.getElementById('progPct').textContent = '0%';
    document.getElementById('progElapsed').textContent = '';
    document.getElementById('progSpin').style.display = '';
    document.getElementById('progSpin').classList.remove('stopped');
    const cancelBtn = document.getElementById('btnCancelJob');
    cancelBtn.textContent = '✕ Cancel';
    cancelBtn.disabled = false;
    cancelBtn.onclick = _overlayCancelClick;
    // Clear ONLY the live preview. #segList is the same ID that
    // fileview.js uses for a normal transcript — without this check,
    // starting the next queue job would delete the transcript the user is
    // currently reading, forcing them to click the file again.
    const segList = document.getElementById('segList');
    if (segList && segList.dataset.live === '1') segList.innerHTML = '';
    document.getElementById('logDownloadLink').style.display = 'none';
    clearLogViewer();
}

function _applyDoneResult(filename, result) {
    const f = fileByName(filename);
    if (!f) return;
    filesChanged();
    f.status = 'done';
    f._processing = false;
    if (!result) return;
    f.processing_time = result.processing_time || '';
    f.audio_duration = result.total_duration || '';
    f.tx_date = Date.now() / 1000;
    f.tx_warning_count = result.warning_count || 0;
    f.tx_word_count = result.segments
        ? result.segments.reduce((n, s) => n + (s.text ? s.text.trim().split(/\s+/).filter(Boolean).length : 0), 0)
        : 0;
    f.tx_seg_count = result.segments ? result.segments.length : 0;
}

// ── Queue row below the progress bar ─────────────────────────────────────
// The earlier mini-queue here at the bottom is gone with no replacement:
// first the 200-item preview with sorting (redundant with the ⏳ queue
// tab), and finally even the bare counter — the stop button right above it
// and the sidebar's ⏳ chip already carried that (too much duplicated
// info). The function stays as a no-op because several places call it.
function renderQueueList() {}


// ── ⋯ menu in the sidebar footer ──────────────────────────────────────
function toggleBatchMenu(ev) {
    ev?.stopPropagation();
    const menu = document.getElementById('batchMenu');
    if (!menu) return;
    if (!menu.hidden) { closeBatchMenu(); return; }
    // Shown when they apply: open files to select, finished AI jobs to clear
    const open = document.getElementById('btnSelectOpen');
    if (open) open.style.display = (typeof srvStats !== 'undefined' && srvStats && srvStats.pending) ? '' : 'none';
    const fin = document.getElementById('btnClearFinishedAi');
    if (fin) {
        const c = (typeof llmQueueSnap !== 'undefined' && llmQueueSnap && llmQueueSnap.counts) || {};
        fin.style.display = (c.done || 0) + (c.error || 0) + (c.cancelled || 0) ? '' : 'none';
    }
    // "Nothing to do" only when every item is hidden
    const any = [...menu.querySelectorAll('.bm-item')].some(b => b.style.display !== 'none');
    menu.querySelector('.bm-empty').style.display = any ? 'none' : '';
    menu.hidden = false;
    document.getElementById('btnBatchMore')?.setAttribute('aria-expanded', 'true');
}

function closeBatchMenu() {
    const menu = document.getElementById('batchMenu');
    if (menu) menu.hidden = true;
    document.getElementById('btnBatchMore')?.setAttribute('aria-expanded', 'false');
}

document.addEventListener('click', e => {
    if (!e.target.closest?.('.bb-more')) closeBatchMenu();
});
document.addEventListener('keydown', e => { if (e.key === 'Escape') closeBatchMenu(); });


/* ⋯ → Select all open: every file of the open folder without a transcript
   (the server's Open filter, whatever the list shows now). */
async function selectAllOpen() {
    try {
        const d = await fetchFilteredNames('pending');
        d.names.forEach((n, i) => {
            selectedFiles.add(n);
            if (d.status) selectionStatus.set(n, d.status[i]);
        });
        updateSelectionUI();
        syncSelectedClasses();
        if (typeof Toast !== 'undefined') Toast.info(`${plural(d.names.length, 'open file')} selected`);
    } catch (e) {
        if (typeof Toast !== 'undefined') Toast.error('Could not select them: ' + (e.message || e));
    }
}

/* 📝 in the selection bar: one summary per selected recording, as AI jobs
   (the default summary; the summary tab of a recording writes others). */
async function summarizeSelected() {
    const statusOf = n => selectionStatus.get(n)
        || (typeof fileByName === 'function' ? fileByName(n)?.status : '') || 'done';
    const names = [...selectedFiles].filter(n => statusOf(n) === 'done');
    const skipped = selectedFiles.size - names.length;
    if (!names.length) {
        if (typeof Toast !== 'undefined') Toast.info('Only transcribed recordings can be summarized');
        return;
    }
    try {
        const r = await fetch('/api/llm/queue/add', {method: 'POST', headers: {'Content-Type': 'application/json'},
                                                     body: JSON.stringify({kind: 'summary', filenames: names})});
        const d = await r.json();
        if (!r.ok) throw new Error(d.error || `HTTP ${r.status}`);
        if (typeof Toast !== 'undefined') {
            Toast.success(`${plural(d.added || 0, 'summary', 'summaries')} queued`
                + (skipped ? ` — ${plural(skipped, 'recording')} without a transcript left out` : ''));
        }
        if (typeof llmQueueRefreshBadge === 'function') llmQueueRefreshBadge();
    } catch (e) {
        if (typeof Toast !== 'undefined') Toast.error('Summaries not queued: ' + (e.message || e));
    }
}
