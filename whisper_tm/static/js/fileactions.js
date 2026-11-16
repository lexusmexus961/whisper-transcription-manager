// ── File actions (cancel, skiplist, download) ──
//  Named separation.js until the dialogs were merged: the dedicated
//  separation dialog is gone, separation now runs via the normal
//  transcription dialog ("no transcription") and the queue.

// The overlay's ✕ and the ⋯ menu: cancel the WATCHED running entry — the
// file the overlay shows and the ⋯ item names (the open file's, else the
// oldest; batch.js) — never simply the oldest (currentJobId). Its lane may
// not have made the job yet: then the queue entry itself is cancelled.
async function cancelCurrentJob() {
    const w = (typeof _watchedRunningItem === 'function') ? _watchedRunningItem() : null;
    // currentJobId only without a running list (a page state from before it)
    const jobId = w ? (w.job_id || null) : currentJobId;
    const itemId = (w && !w.job_id) ? w.item_id : null;
    if (!jobId && itemId == null) return;
    // The overlay's button says so only when the overlay is this file's live
    // run — from the ⋯ menu, an outcome on screen (✅/❌ with "✕ Close")
    // stays as it is, closable.
    const btn = (typeof _overlayShowsWatched !== 'function'
                 || (_overlayShowsWatched() && !_overlayShowsOutcome()))
        ? document.getElementById('btnCancelJob') : null;
    const orig = btn ? btn.textContent : '';
    if (btn) { btn.textContent = 'Stopping…'; btn.disabled = true; }
    // Only this file is cancelled — the queue then continues with the next
    // entry. Use "Stop queue" to clear the whole queue.
    const ok = jobId ? await cancelJob(jobId) : await cancelQueueItem(itemId);
    if (!ok && btn) {
        // Failed → don't permanently block the button
        btn.textContent = orig;
        btn.disabled = false;
    }
}

// Cancel ONE job by its id — the other running files keep going.
async function cancelJob(jobId, btn) {
    if (!jobId) return false;
    if (btn) btn.disabled = true;
    try {
        const r = await fetch(`/api/cancel/${encodeURIComponent(jobId)}`, {method: 'POST'});
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return true;
    } catch(e) {
        if (btn) btn.disabled = false;
        if (typeof Toast !== 'undefined') Toast.error('Cancel failed');
        return false;
    }
}

// ✕ on a running row of the Queue tab: cancel THAT row's file. The job is
// looked up on click — the row may have been drawn before its lane made
// the job (then the queue entry itself is cancelled), or the file may
// have finished since.
function cancelRunningItem(key, btn) {
    const items = (typeof queueRunningItems !== 'undefined') ? queueRunningItems : [];
    const r = items.find(x => runningKey(x) === key);
    if (!r) {
        if (typeof Toast !== 'undefined') Toast.info('This file is no longer running');
        return Promise.resolve(false);
    }
    if (r.job_id) return cancelJob(r.job_id, btn);
    return cancelQueueItem(r.item_id, btn);
}

async function cancelQueueItem(itemId, btn) {
    if (itemId == null) return false;
    if (btn) btn.disabled = true;
    try {
        const r = await fetch('/api/queue/cancel', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({item_id: itemId})
        });
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return true;
    } catch(e) {
        if (btn) btn.disabled = false;
        if (typeof Toast !== 'undefined') Toast.error('Cancel failed');
        return false;
    }
}

async function unskipFile(filename) {
    try {
        const r = await fetch('/api/unskip', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({filename})
        });
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
    } catch(e) {
        if (typeof Toast !== 'undefined') Toast.error('Could not remove from skiplist');
        return;
    }
    const f = fileByName(filename);
    if (f) { f.status = 'pending'; f.skip_reason = ''; filesChanged(); }
    updateCounts();
    renderFileList();
    if (activeFile === filename) selectFile(f || {name:filename, status:'pending'});
}

function downloadOriginal(filename) {
    const a = document.createElement('a');
    a.href = `/api/download/${encodeURIComponent(filename)}`;
    a.download = '';
    document.body.appendChild(a);
    a.click();
    a.remove();
}
