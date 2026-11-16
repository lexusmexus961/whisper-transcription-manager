// ── Quick filter (shared definition for sidebar AND search panel) ──
// One filter state, two surfaces: the chips in the sidebar and in the
// search dropdown show and set the same activeTab. hideEmpty chips
// only appear once there is something to show (errors/cancellations).
const QUICK_FILTERS = [
    {key: 'all',        label: 'All'},
    {key: 'pending',    label: '○ Open'},
    {key: 'done',       label: '✓ Done'},
    {key: 'processing', label: '⟳ Active',      hideEmpty: true},
    {key: 'error',      label: '✕ Error',       hideEmpty: true, cls: 'qf-err'},
    {key: 'cancelled',  label: '⊘ Cancelled',   hideEmpty: true, cls: 'qf-cancel'},
    {key: 'orphan',     label: '👻 Orphaned',   hideEmpty: true, cls: 'qf-orphan'},
];

// A file's tab membership has been decided by the SERVER since the UI
// rework (_media_page_filter in database/__init__.py) — the earlier
// client copy fileMatchesTab has been removed so there is exactly ONE
// definition and counters never diverge from the list.

// Queue and LLM are work lists, not file filters: the sidebar shows them
// instead of the file list, and the paginated file query pauses.
function isWorkTab(tab = activeTab) {
    return tab === 'queue' || tab === 'llm';
}

// ── Tab filter ──
function setFileTab(tab, btn) {
    activeTab = tab;
    // Both surfaces mirror the same state (btn parameter is kept for
    // existing onclick calls, but is no longer needed).
    document.querySelectorAll('.sidebar .qf-chip[data-filter]').forEach(b =>
        b.classList.toggle('active', b.dataset.filter === tab));
    // Work lists have their own header — the file filter and the sort bar
    // do not apply to them and are hidden (CSS: .sidebar.work-tab).
    document.querySelector('.sidebar')?.classList.toggle('work-tab', isWorkTab(tab));
    // The LLM list streams only while it is on screen.
    if (tab !== 'llm' && typeof lqDetach === 'function') lqDetach();
    // The progress panel follows the tab: the queue's time left while its
    // list is open, the folder's otherwise (stats.js)
    if (typeof updateCounts === 'function') updateCounts();
    if (tab === 'queue' && typeof loadQueueEta === 'function') loadQueueEta(true);
    if (isWorkTab(tab)) { renderFileList(); renderBatchButtons(); return; }
    reloadFileList();     // tab is now a server filter (paginated list)
}

// ── Sorting ──
function setSort(key, btn) {
    if (sortKey === key) {
        sortAsc = !sortAsc;
    } else {
        sortKey = key;
        sortAsc = key === 'name'; // name default asc, remainder default desc
    }
    document.querySelectorAll('.sort-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    // Update all arrows
    document.getElementById('arrowName').textContent = sortKey === 'name' ? (sortAsc ? '↑' : '↓') : '·';
    document.getElementById('arrowSize').textContent = sortKey === 'size' ? (sortAsc ? '↑' : '↓') : '·';
    document.getElementById('arrowDate').textContent = sortKey === 'modified' ? (sortAsc ? '↑' : '↓') : '·';
    document.getElementById('arrowTxDate').textContent = sortKey === 'tx_date' ? (sortAsc ? '↑' : '↓') : '·';
    document.getElementById('arrowWords').textContent = sortKey === 'tx_word_count' ? (sortAsc ? '↑' : '↓') : '·';
    reloadFileList();     // sorting runs server-side (index order scan)
}

// ── Server-side page buffer of the main list (UI rework 5.1) ────────────────────────
// Filter + sort no longer run in the browser over the full list (with a
// large library that's seconds of loading + expensive sort per render), but in SQL:
// /api/files returns PAGES in list order. The client keeps a sparse buffer
// index → entry; virtual scrolling renders from it and fetches missing
// pages as it scrolls. `files` (state.js) remains a dense list of the
// LOADED entries — fileByName & co. keep working, but now only see the
// window, not the library.
const SRV_PAGE = 500;
let srvList = {
    total: 0,
    entries: [],          // sparse: list index → entry
    pages: new Set(),     // loaded OR currently requested pages
    inflight: 0,          // running requests of the CURRENT sequence
    seq: 0,               // sequence guard against overtaking responses
    failures: new Map(),  // page → failed attempts in a row (retry backoff)
    // Where to jump back to after a BACKGROUND rebuild.
    // `null` = no one is bound to the end of the list, top is correct.
    desiredTop: null,
};

function _listQuery() {
    const q = document.getElementById('fileFilter').value.trim();
    return `tab=${encodeURIComponent(isWorkTab() ? 'all' : activeTab)}`
        + `&sort=${encodeURIComponent(sortKey)}`
        + `&dir=${sortAsc ? 'asc' : 'desc'}`
        + (q ? `&q=${encodeURIComponent(q)}` : '')
        + (typeof FilterPanel !== 'undefined' ? FilterPanel.query() : '');
}

// Rebuild the dense files list from the sparse buffer (for fileByName,
// selection displays, search speaker etc.). Order = list order.
function _rebuildLoadedFiles() {
    files = srvList.entries.filter(Boolean);
    if (queueCurrentFile) {
        files.forEach(f => { f._processing = isRunningFile(f); });
    }
    filesChanged();
}

// Fetch one page into the buffer. Also runs while scrolling — hence the
// sequence guard: after reloadFileList() old responses are worthless.
async function _srvFetchPage(page) {
    if (srvList.pages.has(page)) return;
    srvList.pages.add(page);
    srvList.inflight++;
    let counts = true;                     // does this request still count in inflight?
    const seq = srvList.seq;
    try {
        const r = await fetch(`/api/files?offset=${page * SRV_PAGE}`
            + `&limit=${SRV_PAGE}&${_listQuery()}`);
        const d = await r.json();
        if (seq !== srvList.seq) return;          // reloaded in the meantime
        if (d.error || !r.ok) throw new Error(d.error || `HTTP ${r.status}`);
        srvList.failures.delete(page);
        srvList.total = d.total;
        (d.files || []).forEach((f, i) => {
            srvList.entries[page * SRV_PAGE + i] = f;
        });
        srvList.inflight--; counts = false;          // before rendering: the
        // loading state is over, otherwise renderFileList would paint "Loading…" again
        _rebuildLoadedFiles();
        _buildOffsets();
        // An empty result has no window to render — then the hint text
        // is needed, otherwise the old "Loading…" node stays put.
        if (!srvList.total) renderFileList();
        else _renderWindow();
        renderBatchButtons();
    } catch (e) {
        if (seq === srvList.seq) {
            if (counts) { srvList.inflight--; counts = false; }
            // Retry with a backoff (1 s, 2 s, 4 s … 30 s). The page stays
            // marked as requested until then: freeing it right away let the
            // re-render below ask for the same page at once — with the
            // network down or /api/files failing, a tight fetch → fail →
            // re-render loop.
            const n = (srvList.failures.get(page) || 0) + 1;
            srvList.failures.set(page, n);
            setTimeout(() => {
                if (seq !== srvList.seq) return;   // reloaded: a new list asks anew
                srvList.pages.delete(page);
                if (srvList.total) _renderWindow();  // still in view → asks again
                else _srvFetchPage(page);
            }, Math.min(30000, 1000 * 2 ** (n - 1)));
            renderFileList();                      // don't freeze "Loading…"
        }
        wtmDebug('Files', `Page ${page} failed: ${e.message}`);
    }
}

// Request missing pages for the row range [first, last).
function _srvEnsureRange(first, last) {
    const p0 = Math.floor(first / SRV_PAGE);
    const p1 = Math.floor(Math.max(first, last - 1) / SRV_PAGE);
    for (let p = p0; p <= p1; p++) {
        if (!srvList.pages.has(p)) _srvFetchPage(p);
    }
}

/* Reload the list from the server.
 *
 * `preserveScroll` distinguishes the two triggers, and the distinction
 * is the whole point:
 *
 * * **The user changes something** (tab, sorting, filter, directory) —
 *   this is a NEW list, and it starts at the top.
 * * **Data changes in the background** (a transcription finishes) —
 *   the user hasn't done anything there, and then nothing should move
 *   under their eyes either.
 *
 * Without this, a list scrolled down in "All" jumped back to the top every
 * time a transcript finished. The path there was long and subtle: queue stream → several files finish in one tick →
 * `_scheduleScan()` → `scanFiles()` → here. And here it said "does NOT
 * keep the scroll position" — correct for a filter change,
 * wrong for a background update.
 */
function reloadFileList({preserveScroll = false} = {}) {
    const listing = document.getElementById('fileList');
    srvList.desiredTop = (preserveScroll && listing) ? listing.scrollTop : null;
    srvList.seq++;
    srvList.total = 0;
    srvList.entries = [];
    srvList.pages.clear();
    srvList.failures.clear();
    srvList.inflight = 0;   // responses still running for the old sequence don't count
    if (!isWorkTab()) _srvFetchPage(0);
    if (typeof loadServerStats === 'function') loadServerStats();
    renderFileList();
}

// The LOADED entries in list order (dense). Note: this is only
// the window — for "all filtered" actions get the name list from the
// server (fetchFilteredNames), never this function.
function getLoadedFiles() {
    return srvList.entries.filter(Boolean);
}

// Full name list of the active filter from the server (with status) —
// basis for Ctrl+A, batch over all filtered, bulk delete, retry.
async function fetchFilteredNames(tabOverride) {
    const q = document.getElementById('fileFilter').value.trim();
    const tab = tabOverride || (isWorkTab() ? 'all' : activeTab);
    const r = await fetch(`/api/files/names?tab=${encodeURIComponent(tab)}`
        + `&sort=${encodeURIComponent(sortKey)}&dir=${sortAsc ? 'asc' : 'desc'}`
        + (q ? `&q=${encodeURIComponent(q)}` : '')
        + (typeof FilterPanel !== 'undefined' ? FilterPanel.query() : '') + '&with_status=1');
    const d = await r.json();
    if (d.error) throw new Error(d.error);
    return d;   // {names: [...], status: [...], total}
}

// A loaded entry has changed (e.g. transcript finished): replace it in
// place, without reloading the list. Returns false if the file is not
// in the window.
function updateLoadedEntry(name, new_value) {
    const idx = srvList.entries.findIndex(f => f && f.name === name);
    if (idx < 0) return false;
    const old_value = srvList.entries[idx];
    if (old_value._processing) new_value._processing = true;
    srvList.entries[idx] = new_value;
    _rebuildLoadedFiles();
    _buildOffsets();
    _renderWindow();
    return true;
}

// Set the _processing flag across the loaded entries: on every running file
// (queue lanes: several at once, isRunningFile). `name` is the oldest one —
// null when nothing runs.
function setProcessingFile(name) {
    let changed = false;
    srvList.entries.forEach(f => {
        if (!f) return;
        const expected = name ? isRunningFile(f) : false;
        if (!!f._processing !== expected) { f._processing = expected; changed = true; }
    });
    if (changed) filesChanged();
    return changed;
}

// ── Targeted DOM updates instead of a full rebuild ───────────────────────────────
// A click on a clip used to rebuild up to 2000 DOM nodes via renderFileList()
// and ran through filter+sort of the whole library —
// just to move the active marker.
function markActiveFile(name) {
    const list = document.getElementById('fileList');
    if (!list) return;
    list.querySelectorAll('.file-item.active').forEach(el => el.classList.remove('active'));
    if (name == null) return;
    const el = list.querySelector(`.file-item[data-name="${CSS.escape(name)}"]`);
    if (el) el.classList.add('active');
}

// Sync the .selected classes of the rendered rows with selectedFiles —
// Ctrl-/Shift-click and clearing the selection don't need a list rebuild.
function syncSelectedClasses() {
    document.querySelectorAll('#fileList .file-item[data-name]').forEach(el => {
        el.classList.toggle('selected', selectedFiles.has(el.getAttribute('data-name')));
    });
}


// ── Virtual scrolling ─────────────────────────────────────────────────────
// With a large library, NOT the whole (filtered) list may go into the DOM: that's
// tens of thousands of heavy nodes + closures per render → Firefox dies. Instead,
// only the rows visible in the viewport (± overscan) live in the DOM;
// above/below, two spacer divs hold the correct total height so the
// scrollbar is right. On 'scroll', only the visible window is rebuilt.
const ROW_OVERSCAN = 8;            // extra rows above/below the viewport
let _vh = { short: 40, tall: 54 }; // row heights (2 variants) — fallback until measured
let _vhMeasured = false;            // true only after a real (visible) measurement
let _vOffsets = null;              // prefix sum of row tops (len+1)
let _vTotalH = 0;                  // total height of all rows in px
let _vScrollBound = false;         // bind scroll/resize listener only once

// Row height depends ONLY on whether the metadata_title row exists (all
// other fields are single-line, nowrap+ellipsis). So exactly two heights,
// which we measure once via an invisible sample (robust against font/zoom).
function _measureRowHeights(list) {
    if (_vhMeasured) return;   // already really measured → do nothing
    const sample = (withMeta) => {
        const d = document.createElement('div');
        d.className = 'file-item pending';
        d.style.cssText = 'visibility:hidden;position:absolute;left:-9999px;top:0;width:'
            + (list.clientWidth || 300) + 'px';
        d.innerHTML = `<div class="fi-strip"></div><div class="fi-body">
            <div class="fi-row-top"><div class="fi-name">x</div></div>
            ${withMeta ? '<div class="fi-meta-title">m</div>' : ''}
            <div class="fi-stats"><span class="fi-stat">1w</span></div></div>`;
        list.appendChild(d);
        const h = d.getBoundingClientRect().height;
        list.removeChild(d);
        return Math.round(h);
    };
    const s = sample(false), t = sample(true);
    // Only cache if the measurement was real (panel visible → height > 0).
    // Otherwise use the fallback and try again on the next render.
    if (s > 0 && t > 0) { _vh.short = s; _vh.tall = t; _vhMeasured = true; }
}

function _buildOffsets() {
    // Over the SERVER total, not the loaded rows: the spacers must hold
    // the full list height so the scrollbar is correct.
    // Not-yet-loaded rows count with the short height (on loading they
    // are rebuilt — a minimal jump, but no flying blind).
    const n = srvList.total;
    const off = new Float64Array(n + 1);
    let acc = 0;
    for (let i = 0; i < n; i++) {
        off[i] = acc;
        const f = srvList.entries[i];
        acc += (f && f.metadata_title) ? _vh.tall : _vh.short;
    }
    off[n] = acc;
    _vOffsets = off;
    _vTotalH = acc;
}

// First row index whose bottom edge > y (binary search over the prefix sum).
function _indexAtY(y) {
    const off = _vOffsets;
    let lo = 0, hi = off.length - 1;
    while (lo < hi) {
        const mid = (lo + hi) >> 1;
        if (off[mid + 1] <= y) lo = mid + 1; else hi = mid;
    }
    return lo;
}

// (Re)render only the visible window. Without an argument, the current
// scroll position applies (scroll handler). With targetTop, jumps there
// after the rebuild — needed because a previous innerHTML='' resets
// scrollTop to 0.
function _renderWindow(targetTop) {
    if (isWorkTab() || !_vOffsets) return;
    const list = document.getElementById('fileList');
    if (!list) return;
    /* The remembered position after a background rebuild.
     *
     * Here and not in `reloadFileList`, because the path there goes
     * through two stages: first `renderFileList` clears the list (which
     * pins the browser's scrollTop to 0), and only once the first page
     * has arrived from the server does it have a height again to jump
     * INTO — and this second call comes from `_srvFetchPage` with no
     * target at all. `_renderWindow` is the only place both
     * paths must go through. */
    if (targetTop == null && srvList.desiredTop != null && srvList.total) {
        targetTop = srvList.desiredTop;
        srvList.desiredTop = null;
    }
    const viewH = list.clientHeight || 600;
    // Clamp to a valid range: after filtering/deleting, the old scroll
    // position can be past the (now shorter) end of the list → otherwise
    // you'd land in the empty spacer.
    const maxTop = Math.max(0, _vTotalH - viewH);
    const scrollTop = Math.min((targetTop != null) ? targetTop : list.scrollTop, maxTop);
    if (targetTop != null) targetTop = scrollTop;
    const first = Math.max(0, _indexAtY(scrollTop) - ROW_OVERSCAN);
    const last = Math.min(srvList.total, _indexAtY(scrollTop + viewH) + ROW_OVERSCAN + 1);

    const queued = queuedKeys();
    const ask = document.createDocumentFragment();

    const topSpacer = document.createElement('div');
    topSpacer.style.height = _vOffsets[first] + 'px';
    ask.appendChild(topSpacer);

    for (let i = first; i < last; i++) {
        const f = srvList.entries[i];
        if (f) {
            ask.appendChild(_buildFileRow(f, queued));
        } else {
            // Row not yet loaded: placeholder at row height; the
            // missing page has already been requested below.
            const ph = document.createElement('div');
            ph.className = 'file-item';
            ph.style.cssText = `height:${_vh.short}px;opacity:0.35;display:flex;`
                + 'align-items:center;padding:0 12px;color:var(--tx-3);font-size:0.72rem';
            ph.textContent = '…';
            ask.appendChild(ph);
        }
    }

    const botSpacer = document.createElement('div');
    botSpacer.style.height = Math.max(0, _vTotalH - _vOffsets[last]) + 'px';
    ask.appendChild(botSpacer);

    list.innerHTML = '';
    list.appendChild(ask);

    // Now the container has its full height (spacer) again → jump possible.
    if (targetTop != null && list.scrollTop !== targetTop) list.scrollTop = targetTop;

    // Reload missing pages for the visible window (pattern: svLoadMore)
    _srvEnsureRange(first, last);
}

// Build a single file row as a DOM element (extracted from the former forEach
// so _renderWindow can reuse it per visible row).
function _buildFileRow(f, queued) {
        const isDone = f.status === 'done';
        const hasWarnings = isDone && (f.tx_warning_count || 0) > 0;
        const isEmpty = isDone && (f.tx_word_count || 0) === 0;
        const isPartial = f.status === 'partial';
        const isSkipped = f.status === 'skipped';
        const isOrphan = f.status === 'orphan';
        const isProcessing = f._processing;
        const inQueue = queuedIn(queued, f);
        // Queue outcome (error/cancel) colors non-finished rows — a
        // file with a finished transcript only shows it as an info chip below.
        const outcome = (!isProcessing && !inQueue) ? queueOutcomeOf(f) : null;
        const outcomeCls = outcome && !isDone ? (outcome.status === 'error' ? 'qfail' : 'qcancel') : '';
        const cls = (isProcessing ? 'processing' : (isDone ? (isEmpty ? 'done empty' : (hasWarnings ? 'done warned' : 'done')) : (isOrphan ? 'orphan' : (isSkipped ? 'skipped' : (isPartial ? 'partial' : (inQueue ? 'queued' : 'pending')))))) + (outcomeCls ? ' ' + outcomeCls : '');
        const icon = isProcessing ? '⟳' : (isDone ? (isEmpty ? '∅' : (hasWarnings ? '⚠' : '✓')) : (isOrphan ? '👻' : (isSkipped ? '⊘' : (isPartial ? '⚠' : (inQueue ? '⏳' : (outcome ? (outcome.status === 'error' ? '✕' : '⊘') : '○'))))));
        const active = activeFile === f.name ? ' active' : '';
        const sel = selectedFiles.has(f.name) ? ' selected' : '';
        const div = document.createElement('div');
        div.className = `file-item ${cls}${active}${sel}`;
        div.setAttribute('data-name', f.name);
        // Reachable by keyboard: Tab to the list, ↑/↓ between rows, Enter or
        // Space opens (the data-keyclick handler, searchview.js)
        div.setAttribute('role', 'button');
        div.tabIndex = 0;
        div.setAttribute('data-keyclick', '');
        div.setAttribute('aria-label', f.display_name || f.name);
        div.onkeydown = (ev) => {
            if (ev.key !== 'ArrowDown' && ev.key !== 'ArrowUp') return;
            const rows = [...document.querySelectorAll('#fileList .file-item[data-name]')];
            const next = rows[rows.indexOf(div) + (ev.key === 'ArrowDown' ? 1 : -1)];
            if (next) { ev.preventDefault(); next.focus(); next.scrollIntoView({block: 'nearest'}); }
        };
        div.onmousedown = (ev) => {
            // Prevent browser text selection on shift-click
            if (ev.shiftKey) ev.preventDefault();
        };
        div.onclick = (ev) => {
            if (ev.shiftKey) {
                ev.preventDefault();
                window.getSelection()?.removeAllRanges();
                if (lastClickedFile) {
                    // Range over the server buffer's list INDICES.
                    // If unloaded rows (gaps) lie between the endpoints,
                    // the server resolves the names.
                    const idxA = srvList.entries.findIndex(x => x && x.name === lastClickedFile);
                    const idxB = srvList.entries.findIndex(x => x && x.name === f.name);
                    if (idxA >= 0 && idxB >= 0) {
                        const from = Math.min(idxA, idxB);
                        const to = Math.max(idxA, idxB);
                        let gap = false;
                        for (let j = from; j <= to; j++) {
                            const e = srvList.entries[j];
                            if (e) { selectedFiles.add(e.name); selectionStatus.set(e.name, e.status); }
                            else gap = true;
                        }
                        if (gap) {
                            fetchFilteredNames().then(d => {
                                const a = d.names.indexOf(lastClickedFile);
                                const b = d.names.indexOf(f.name);
                                if (a < 0 || b < 0) return;
                                for (let j = Math.min(a, b); j <= Math.max(a, b); j++) {
                                    selectedFiles.add(d.names[j]);
                                    if (d.status) selectionStatus.set(d.names[j], d.status[j]);
                                }
                                updateSelectionUI();
                                syncSelectedClasses();
                            }).catch(() => {});
                        }
                    }
                } else {
                    selectedFiles.add(f.name);
                    selectionStatus.set(f.name, f.status);
                    lastClickedFile = f.name;
                }
                updateSelectionUI();
                syncSelectedClasses();   // no full rebuild for a class update
            } else if (ev.ctrlKey || ev.metaKey) {
                ev.preventDefault();
                if (selectedFiles.has(f.name)) {
                    selectedFiles.delete(f.name);
                    selectionStatus.delete(f.name);
                } else {
                    selectedFiles.add(f.name);
                    selectionStatus.set(f.name, f.status);
                }
                lastClickedFile = f.name;
                updateSelectionUI();
                syncSelectedClasses();   // no full rebuild for a class update
            } else {
                lastClickedFile = f.name;
                selectFile(f);
            }
        };

        const dt = new Date(f.modified * 1000);
        const dateStr = dt.toLocaleDateString(undefined, {day:'2-digit',month:'2-digit',year:'2-digit'});
        const displayName = f.display_name || f.name;
        const metaTitle = f.metadata_title ? `<div class="fi-meta-title" title="${esc(f.metadata_title)}">${esc(f.metadata_title)}</div>` : '';
        const subfolderTag = f.subfolder
            ? `<span class="fi-subfolder" title="${esc(f.subfolder)}">${esc(f.subfolder)}</span>`
            : '';

        // Build compact stat items
        let stats = '';
        stats += `<span class="fi-stat" title="File size">${esc(f.size_fmt)}</span>`;

        // Duration: from transcript, cached index metadata, or legacy scan
        const _fmtDurSec = (ds) => {
            const dh = Math.floor(ds/3600), dm = Math.floor((ds%3600)/60), dss = Math.floor(ds%60);
            return dh > 0 ? `${dh}:${String(dm).padStart(2,'0')}:${String(dss).padStart(2,'0')}` : `${dm}:${String(dss).padStart(2,'0')}`;
        };
        if (isDone && f.audio_duration) {
            stats += `<span class="fi-stat s-dur" title="Audio duration">${esc(f.audio_duration)}</span>`;
        } else if (f.media_duration_sec) {
            // Cached from the DB (index) — no FFprobe per display
            stats += `<span class="fi-stat s-dur" title="Media duration (indexed)">${_fmtDurSec(f.media_duration_sec)}</span>`;
        }

        if (isDone && f.tx_word_count) {
            stats += `<span class="fi-stat s-words" title="${plural(f.tx_word_count, 'word')}">${f.tx_word_count.toLocaleString()}w</span>`;
        }
        if (isDone && f.processing_time) {
            stats += `<span class="fi-stat s-proc" title="Processing time">${esc(f.processing_time)}</span>`;
        }
        if (hasWarnings) {
            stats += `<span class="fi-stat s-warn" title="${plural(f.tx_warning_count, 'warning')} during processing">⚠${f.tx_warning_count}</span>`;
        }
        if (isSkipped) {
            stats += `<span class="fi-stat s-err" title="${f.skip_reason === 'no_audio' ? 'No audio track detected' : 'Manually skipped'}">${f.skip_reason === 'no_audio' ? 'No audio' : 'Skipped'}</span>`;
        }
        if (isEmpty) {
            stats += `<span class="fi-stat s-err" title="Transcript contains 0 words">∅ Empty</span>`;
        }
        if (isOrphan) {
            stats += `<span class="fi-stat s-orphan" title="Media file gone — the transcript still exists. View with a click, delete via 🗑.">👻 File missing</span>`;
        }
        if (f.has_deriv) {
            stats += `<span class="fi-stat s-orphan" title="No transcript, but summaries/translations still exist. View with a click.">🤖 Derivatives</span>`;
        }
        if (outcome) {
            stats += outcome.status === 'error'
                ? `<span class="fi-stat s-err" title="${esc(outcome.error || 'Failed')}">✕ Failed</span>`
                : `<span class="fi-stat s-cancel" title="Processing was cancelled — can be re-queued via ↩">⊘ Cancelled</span>`;
        }

        // Actions: +Q always visible when busy, TX on hover, ⏳ if queued
        let queueIndicator = '';
        let hoverActions = '';
        if (isOrphan) {
            // File gone → neither +Q nor TX; only the transcript can be deleted
            hoverActions = `<button class="del" onclick="event.stopPropagation();deleteTx('${escJs(f.name)}')" title="Delete orphaned transcript">🗑</button>`;
        } else if (!isDone && !isProcessing && !isSkipped) {
            // ⤒ appears in BOTH cases: a not-yet-queued file gets
            // added immediately at the front, an already-pending one is
            // moved up (server: enqueue with front=true).
            const front = `<button class="fi-q-btn fi-q-top" onclick="event.stopPropagation();quickQueueFront('${escJs(f.name)}')" title="Transcribe next — straight to the front of the queue">⤒</button>`;
            if (inQueue) {
                queueIndicator = `<span class="fi-q-badge" title="In queue">⏳</span>${front}`;
            } else {
                // +Q is always available: the queue accepts files at any time,
                // regardless of whether something is currently running.
                queueIndicator = `<button class="fi-q-btn" onclick="event.stopPropagation();quickQueue('${escJs(f.name)}')" title="Add to queue">+Q</button>${front}`;
            }
            hoverActions = `<button onclick="event.stopPropagation();startTranscribe('${escJs(f.name)}')" title="Transcribe (with options)">TX</button>`;
        }
        if (isDone || isPartial) {
            hoverActions += `<button class="del" onclick="event.stopPropagation();deleteTx('${escJs(f.name)}')" title="Delete transcript">🗑</button>`;
        } else if (f.has_deriv) {
            // No transcript anymore, but derivatives exist — otherwise there
            // would be no way to get rid of them again.
            hoverActions += `<button class="del" onclick="event.stopPropagation();deleteTx('${escJs(f.name)}')" title="Delete summaries/translations">🗑</button>`;
        }

        // Rows with the always-visible +Q/⤒ keep their hover actions in the
        // flow right behind them: laid over the right edge (fi-actions) they
        // covered exactly those buttons.
        const inlineActions = (queueIndicator && hoverActions)
            ? `<span class="fi-inline-actions">${hoverActions}</span>` : '';
        const overlayActions = inlineActions ? '' : hoverActions;

        div.innerHTML = `
            <div class="fi-strip"></div>
            <div class="fi-body">
                <div class="fi-row-top">
                    ${subfolderTag}
                    <div class="fi-name" title="${esc(f.name)}">${esc(displayName)}</div>
                    ${queueIndicator}
                    ${inlineActions}
                </div>
                ${metaTitle}
                <div class="fi-stats">${stats}</div>
            </div>
            <div class="fi-actions">${overlayActions}</div>
        `;
        return div;
}

/* Badges for a queue entry from its settings.
 *
 * Empty settings = configured defaults (server: _normalize) — hence the
 * actual defaults instead of the old hard-coded fallback "turbo · auto".
 * Diarization gets a badge too, as it's the most expensive step. Shared by the queue tab and the queue editor.
 */
function queueBadges(s) {
    s = s || {};
    const d = (typeof _txDefaults !== 'undefined' && _txDefaults) ? _txDefaults : {};
    // A voices-only entry runs no Whisper: model and language would read as
    // a transcription that never happens. The server fixes separation,
    // diarization and vectors for it (core/queue._normalize).
    if (s.job === 'voices') {
        const v = [s.refresh ? '🗣 Renew voices, names kept' : '🗣 Voices only'];
        if (s.numSpeakers && s.numSpeakers !== 'auto') v.push(`👥 ${s.numSpeakers}`);
        if (s.low_vram) v.push('🧠 Spare VRAM');
        if (s.oom_attempts) v.push(`🔁 OOM ×${s.oom_attempts}`);
        return v;
    }
    const parts = [String(s.model || '').toLowerCase() === 'none' ? '🎤 Separation only'
                   : (s.model || d.model || 'Default'),
                   s.language || d.language || 'auto'];
    if (s.vocal_isolation) parts.push('🎤VI');
    if (s.diarize) parts.push('👥Speakers');
    if (s.speaker_vectors && s.diarize) parts.push('🗣Vectors');
    if (s.retranscribe) parts.push('↻ Re-transcription');
    if (s.summarize) parts.push('🤖 Summary');
    if (s.low_vram) parts.push('🧠 Spare VRAM');
    if (s.oom_attempts) parts.push(`🔁 OOM ×${s.oom_attempts}` + (s.batch_size ? ` · Batch ${s.batch_size}` : ''));
    if (s.vocab_preset) parts.push('📖 ' + s.vocab_preset);
    return parts;
}

// ── Shared sidebar building blocks (queue tab, LLM tab) ─────────────────
// One header and one row anatomy for every list in the sidebar, so the
// file list, the transcription queue and the LLM queue read as one place.

// Header strip: title + actions on top, one summary line below.
function listHead(title, summary, actions = '') {
    return `<div class="list-head">
        <div class="lh-top"><span class="lh-title">${title}</span>${actions}</div>
        ${summary ? `<div class="lh-sum">${summary}</div>` : ''}
    </div>`;
}

function lhBtn(label, onclick, title, cls = '', id = '') {
    return `<button class="lh-btn ${cls}"${id ? ` id="${id}"` : ''} onclick="${onclick}" title="${esc(title)}">${label}</button>`;
}

// Row = the file row's anatomy: status strip (color from the row's class),
// name with an optional trailing badge, folder, stats line (fi-stat items,
// "·"-separated by CSS), anything extra, and hover actions.
function workRowHtml({ path = '', badge = '', stats = '', extra = '', actions = '' }) {
    const full = String(path || '');
    const i = full.lastIndexOf('/');
    const name = i >= 0 ? full.slice(i + 1) : full;
    const folder = i >= 0 ? full.slice(0, i) : '';
    return `<div class="fi-strip"></div>
        <div class="fi-body">
            <div class="fi-row-top">
                <div class="fi-name" title="${esc(full)}">${esc(name)}</div>${badge}
            </div>
            ${folder ? `<div class="fi-folder" title="${esc(full)}">${esc(folder)}</div>` : ''}
            ${stats ? `<div class="fi-stats">${stats}</div>` : ''}
            ${extra}
        </div>
        ${actions ? `<div class="fi-actions">${actions}</div>` : ''}`;
}

function listEmpty(text) {
    return `<div class="list-empty">${text}</div>`;
}

/** Two lines for a queue entry: name large, folder small. */
function _queueRows(relPath) {
    const full = String(relPath || '');
    const i = full.lastIndexOf('/');
    const name = i >= 0 ? full.slice(i + 1) : full;
    const folder = i >= 0 ? full.slice(0, i) : '';
    const ellipsis = 'white-space:nowrap;overflow:hidden;text-overflow:ellipsis';
    return `
        <div style="font-size:0.78rem;${ellipsis}" title="${esc(full)}">${esc(name)}</div>
        ${folder ? `<div style="font-size:0.6rem;color:var(--tx-3);opacity:0.75;${ellipsis}"
                        title="${esc(full)}">${esc(folder)}</div>` : ''}`;
}

/**
 * Jump from the queue to the file.
 *
 * Don't just search files[]: the list is a paginated window
 * over the CURRENT directory, but a queue entry may come from a
 * completely different root. goToSearchResult tries locally first and
 * otherwise lets /api/navigate_to_file resolve it via identity — that
 * switches directory if needed (docs/PATHS.md).
 */
function openQueueFile(filename, dir) {
    if (!filename) return;
    if (typeof goToSearchResult === 'function') {
        goToSearchResult(filename, 0, dir || '');
        return;
    }
    const f = files.find(x => x.name === filename);
    if (f) selectFile(f);
}

// The stats line of a running row in the Queue tab: its job's progress and
// phase — "Processing…" until the lane has made the job.
function runningRowLabel(r) {
    const parts = [typeof r.progress === 'number' ? `⟳ ${r.progress}%` : '⟳ Processing…'];
    const ph = r.phase;
    if (ph && ph.title) parts.push(ph.total ? `${ph.title} (${ph.index}/${ph.total})` : ph.title);
    return parts.join(' · ');
}

// The running rows' progress, in place on every queue tick. A rebuild of
// the Queue tab every second would swallow a click on ✕ mid-hover.
function updateRunningRows() {
    if (activeTab !== 'queue' || typeof queueRunningItems === 'undefined') return;
    const list = document.getElementById('fileList');
    if (!list) return;
    const byKey = new Map(queueRunningItems.map(r => [runningKey(r), r]));
    list.querySelectorAll('.file-item[data-run]').forEach(el => {
        const r = byKey.get(el.getAttribute('data-run'));
        const s = el.querySelector('.s-run');
        if (r && s) s.textContent = runningRowLabel(r);
    });
}

function renderFileList() {
    updateCounts();
    const list = document.getElementById('fileList');
    const scrollTop = list.scrollTop;  // preserve scroll position
    list.innerHTML = '';

    // LLM tab: the LLM queue (llmtools.js keeps the element alive across
    // these rebuilds, so open detail boxes survive queue ticks).
    if (activeTab === 'llm') {
        if (typeof lqAttach === 'function') lqAttach(list);
        list.scrollTop = scrollTop;
        return;
    }

    // Queue tab: shows the server state (running file + queue).
    // There is only ONE queue now — single and batch transcriptions land
    // in the same server-side list.
    if (activeTab === 'queue') {
        const waiting = queuePending || queueItems.length;
        const sum = [`${waiting.toLocaleString()} waiting`];
        if (queueDone) sum.push(`${queueDone.toLocaleString()} done`);
        if (queueFailed) sum.push(`<span class="lh-err">${queueFailed} failed</span>`);
        const editable = queueItems.length || queuePending || queueRetryable;
        const head = document.createElement('div');
        // ✎ Edit: the queue editor (queue_editor.js) — filter, mark, change
        // settings, remove, reorder; here because this is where you notice
        // the preview only shows 200 rows.
        head.innerHTML = listHead(
            `Queue${queuePaused ? (queueSolo ? ` · ⏸ paused · new one${queueSolo > 1 ? 's run' : ' runs'}`
                                      : ' · ⏸ paused') : ''}`,
            sum.join(' · '),
            (editable ? lhBtn('✎ Edit', 'openQueueEditor()',
                              'Edit the queue: filter, mark, change settings, remove, reorder',
                              '', 'btnQueueEditor') : '')
            + (queueItems.length ? lhBtn('Clear', 'clearQueue()', 'Remove all pending entries', 'danger') : ''));
        list.appendChild(head.firstElementChild);

        // Queue lanes: every running entry, oldest first (state.js)
        const running = (typeof queueRunningItems !== 'undefined') ? queueRunningItems : [];

        if (!queueCurrentFile && !running.length && !queueItems.length) {
            list.insertAdjacentHTML('beforeend', listEmpty('No files in the queue'));
            list.scrollTop = scrollTop;
            return;
        }

        // Running files: one row each, with its own progress and its own ✕
        // that cancels THAT file — the other lanes keep going.
        running.forEach(r => {
            const key = runningKey(r);
            const div = document.createElement('div');
            div.className = 'file-item processing';
            div.setAttribute('data-run', key);
            div.innerHTML = workRowHtml({
                path: r.filename,
                stats: `<span class="fi-stat s-run">${esc(runningRowLabel(r))}</span>`,
                actions: `<button class="del" onclick="event.stopPropagation();cancelRunningItem('${escJs(key)}', this)"
                                  title="Cancel this file — the queue keeps running">✕</button>`,
            });
            div.onclick = () => openQueueFile(r.filename, r.dir || '');
            list.appendChild(div);
        });

        // The running file when the list names none: the snapshot reads the
        // oldest entry and the list one after the other, a finish can fall
        // in between (a server without lanes gets its list from batch.js).
        if (queueCurrentFile && !running.length) {
            const div = document.createElement('div');
            div.className = 'file-item processing';
            div.innerHTML = workRowHtml({
                path: queueCurrentFile,
                stats: '<span class="fi-stat s-run">⟳ Processing…</span>',
                actions: `<button class="del" onclick="event.stopPropagation();cancelCurrentJob()"
                                  title="Cancel this file — the queue keeps running">✕</button>`,
            });
            div.onclick = () => openQueueFile(queueCurrentFile, queueCurrentDir || '');
            list.appendChild(div);
        }

        // Pending entries — position as the badge, settings as the stats
        // line, reorder/remove as hover actions like on a file row.
        queueItems.forEach((item, idx) => {
            const div = document.createElement('div');
            div.className = 'file-item queued';
            const top = idx === 0;
            const bottom = idx === queueItems.length - 1;
            const act = (label, call, title, off, cls = '') =>
                `<button class="${cls}" onclick="event.stopPropagation();${call}" title="${title}"${off ? ' disabled' : ''}>${label}</button>`;
            div.innerHTML = workRowHtml({
                path: item.filename,
                badge: `<span class="fi-pos">#${idx + 1}</span>`,
                stats: queueBadges(item.settings || {}).map(x => `<span class="fi-stat">${esc(x)}</span>`).join(''),
                actions: act('⤒', `queueItemToTop(${item.id})`, 'Straight to the top — next in line', top)
                    + act('▲', `moveQueueItem(${item.id},-1)`, 'Move up', top)
                    + act('▼', `moveQueueItem(${item.id},1)`, 'Move down', bottom)
                    + act('✕', `removeFromQueue(${item.id})`, 'Remove', false, 'del'),
            });
            div.onclick = () => openQueueFile(item.filename, item.dir || '');
            list.appendChild(div);
        });

        // The item list is a 200-item preview (server intentionally caps it) —
        // for large queues, say how many are really still waiting.
        if (queuePending > queueItems.length) {
            const more = document.createElement('div');
            more.className = 'list-more';
            more.textContent = `… and ${(queuePending - queueItems.length).toLocaleString()} more in the queue`;
            more.title = 'Show all pending entries in the queue editor';
            more.onclick = () => openQueueEditor();
            list.appendChild(more);
        }

        list.scrollTop = scrollTop;
        return;
    }

    renderBatchButtons();

    // ── Virtual scrolling: measure heights, build offsets, render window ──
    _measureRowHeights(list);
    _buildOffsets();

    // Bind scroll/resize once: per event only ONE rAF-throttled
    // re-render of the visible window (no full rebuild).
    if (!_vScrollBound) {
        _vScrollBound = true;
        let raf = 0;
        const onView = () => {
            if (raf) return;
            raf = requestAnimationFrame(() => { raf = 0; _renderWindow(); });
        };
        list.addEventListener('scroll', onView, { passive: true });
        window.addEventListener('resize', onView);
    }

    if (!srvList.total) {
        // "Loading…" ONLY as long as a request is actually in flight. It used
        // to depend on srvList.pages — but that set also contains already
        // ANSWERED pages, so an empty result would stay in the loading
        // state forever (e.g. the "⟳ Active" tab right after a yt download).
        const q = document.getElementById('fileFilter').value.trim();
        const empty = (activeTab !== 'all' || q)
            ? 'No files in this filter'
            : 'No files — 🔄 for a filesystem scan';
        list.innerHTML = listEmpty(srvList.inflight ? 'Loading…' : empty);
        return;
    }

    _renderWindow(scrollTop);  // build the window for the remembered scroll position
}

// ── Batch/delete buttons below the list ──────────────────────────────────
// Counters come from the server: srvList.total is the size of the ACTIVE
// filter (tab + name filter), srvStats delivers the tab counters for the
// whole directory. With an active name filter, the action buttons show the
// filtered total from the /api/files/counts fields of the page response —
// the ACTION itself is resolved exactly by the server anyway (fetchFilteredNames).
function renderBatchButtons() {
    const btn = document.getElementById('btnBatch');
    const delBtn = document.getElementById('btnDelFiltered');
    if (!btn || !delBtn) return;
    const stats = (typeof srvStats !== 'undefined' && srvStats) ? srvStats : null;
    const q = document.getElementById('fileFilter').value.trim();

    // Pending/deletable in the active filter: without a name filter from
    // the directory aggregates; with a name filter approximated via the
    // active tab (exact if you're directly in the matching tab).
    let pendingCount = 0, doneCount = 0;
    if (activeTab === 'pending') pendingCount = srvList.total;
    else if (activeTab === 'done') doneCount = srvList.total;
    else if (activeTab === 'orphan') doneCount = srvList.total;
    else if (!q && stats) {
        pendingCount = stats.pending || 0;
        doneCount = (stats.done || 0) + (stats.orphan || 0);
    } else {
        // Name filter in a mixed tab: loaded window as an
        // approximation (actions are resolved exactly server-side).
        const loaded = getLoadedFiles();
        pendingCount = loaded.filter(f => f.status !== 'done' && f.status !== 'skipped' && f.status !== 'orphan').length;
        doneCount = loaded.filter(f => f.status === 'done' || f.status === 'orphan').length;
    }

    // Quick filter "Error"/"Cancelled": as long as the queue isn't actively
    // running (then the button stays the stop control), the matching
    // action is the filtered retry — original settings are preserved.
    const retryTab = (activeTab === 'error' || activeTab === 'cancelled')
        && srvList.total > 0 && !queueRunning;

    // "＋" button: queueing remains reachable even while the queue is
    // running/paused (btnBatch is then the stop/resume control).
    // It queues the ACTIVE filter — and shows the HONEST number:
    // "open" includes the already-pending ones; without the subtraction the
    // button promised every open file even when most of them were already
    // in the queue, and in the error tab it still offered "all open" anyway.
    const addBtn = document.getElementById('btnBatchAdd');
    if (addBtn) {
        // Only the waiting entries: the server's open count already leaves
        // out every file with a running entry (_processing_in) — subtracting
        // those again made the button hide or undercount (by up to the lanes).
        const inQueue = queuePending;
        const errorTab = activeTab === 'error' || activeTab === 'cancelled';
        const addable = errorTab ? srvList.total
                                  : Math.max(0, pendingCount - inQueue);
        const showThem = (queueRunning || queuePaused) && addable > 0 && !retryTab
            && !isWorkTab();
        addBtn.style.display = showThem ? '' : 'none';
        if (showThem) {
            addBtn.textContent = errorTab
                ? `＋ Queue ${addable.toLocaleString()} ${activeTab === 'error' ? 'failed' : 'cancelled'} file${addable === 1 ? '' : 's'}`
                : `＋ Queue ${plural(addable, 'open file')}`;
            addBtn.title = errorTab
                ? `Queue ${addable.toLocaleString()} ${activeTab === 'error' ? 'failed' : 'cancelled'} file${addable === 1 ? '' : 's'} with new settings`
                : `Queue ${plural(addable, 'open file')} of the active filter (already-pending ones don't count)`;
        }
    }
    if (retryTab) {
        btn.textContent = `↩ Re-queue ${srvList.total}`;
        btn.disabled = false;
        btn.className = 'btn-batch';
    } else if (queuePaused) {
        // Paused: the queue is stopped but remains fully intact.
        btn.textContent = queuePending > 0
            ? `▶ Resume (${queuePending.toLocaleString()} waiting)`
            : '▶ Resume queue';
        btn.disabled = false;
        btn.className = 'btn-batch';
    } else if (!queueRunning) {
        btn.textContent = pendingCount > 0
            ? `▶ Transcribe ${pendingCount.toLocaleString()} open`
            : 'Nothing open here';
        btn.disabled = pendingCount === 0;
        btn.className = 'btn-batch';
    } else {
        // Running: stop only pauses — deletion goes via "Clear".
        // The waiting count sits right above in the ⏳ Queue segment — in
        // the button it pushed the label out of a four-button row.
        btn.textContent = '⏸ Stop';
        btn.title = `Only pauses — the queue remains fully intact (${queuePending.toLocaleString()} waiting)`;
        btn.className = 'btn-batch stop';
        btn.disabled = false;
    }
    // File actions refer to the file filter — not offered on the work tabs.
    delBtn.style.display = doneCount > 0 && !isWorkTab() ? '' : 'none';
    // On the Queue and LLM tabs the button is only the queue's own control
    // (stop / resume) — "transcribe the filter" refers to a file list that
    // is not on screen there.
    btn.style.display = isWorkTab() && !retryTab && !queueRunning && !queuePaused ? 'none' : '';
    delBtn.textContent = `🗑 Delete ${doneCount.toLocaleString()} transcript${doneCount !== 1 ? 's' : ''}`;
    delBtn.title = `Delete ${doneCount.toLocaleString()} transcript${doneCount !== 1 ? 's' : ''}`;
}

// Name filter: debounced server reload (the filter runs in SQL, no longer
// over a client-side haystack field).
let _fileFilterTimer = null;
document.getElementById('fileFilter').addEventListener('input', () => {
    clearTimeout(_fileFilterTimer);
    _fileFilterTimer = setTimeout(() => reloadFileList(), 300);
});

