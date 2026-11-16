// ── Favorite Directories (Storage) ──
// The UI for it lives in the path picker (pathpicker.js).
// The upload directory always belongs in the favorites: that's where
// uploads and link downloads put their files, and the dialog jumps there
// after fetching. The path comes from the SERVER
// (uploadsDir, from settings → folders) — never built locally;
// root+/uploads is only the fallback while the answer is still pending.
function ensureUploadsFav() {
    let target = (typeof uploadsDir === 'string' && uploadsDir) ? uploadsDir : null;
    if (!target) {
        const root = (document.getElementById('dirInput')?.value || '')
            .trim().replace(/\/+$/, '');
        if (!root) return null;
        target = root.endsWith('/uploads') ? root : root + '/uploads';
    }
    let favs = loadFavDirs();
    // Self-healing: if the upload directory has changed (config switch,
    // new default), replace the favorite that was AUTOMATICALLY added
    // earlier instead of leaving a dead entry — clicking it used to
    // land in "outside the allowed directories" (403).
    let oldAuto = null;
    try { oldAuto = localStorage.getItem('wtmAutoUploadFav'); } catch(e) {}
    if (oldAuto && oldAuto !== target) {
        favs = favs.filter(f => f !== oldAuto);
    }
    if (!favs.includes(target)) {
        favs.unshift(target);
    }
    saveFavDirs(favs);
    try { localStorage.setItem('wtmAutoUploadFav', target); } catch(e) {}
    updateFavBtn();
    return target;
}

function loadFavDirs() {
    try {
        const v = JSON.parse(localStorage.getItem('favDirs') || '[]');
        return Array.isArray(v) ? v : [];
    } catch(e) { return []; }
}
function saveFavDirs(arr) {
    try { localStorage.setItem('favDirs', JSON.stringify(arr)); } catch(e) {}
}
// A long path shows its end — the folder that is open — while its field
// isn't being edited: the beginning is the same root every time, and the
// field showed only that
function showPathEnd(input) {
    if (input && document.activeElement !== input) input.scrollLeft = input.scrollWidth;
}
// …and again once a path field is left: typed and chosen with Enter, it
// kept showing where the typing had scrolled it
document.addEventListener('focusout', e => {
    const input = e.target;
    if (input && input.matches && input.matches('.path-field input')) {
        setTimeout(() => showPathEnd(input), 0);
    }
});

// Updates the favorite star in the path field (called from app.js/search.js)
function updateFavBtn() {
    const btn = document.getElementById('btnFavStar');
    if (!btn) return;
    showPathEnd(document.getElementById('dirInput'));
    const path = (document.getElementById('dirInput')?.value || '').trim();
    const isFav = loadFavDirs().includes(path);
    btn.textContent = isFav ? '★' : '☆';
    btn.classList.toggle('active', isFav);
    btn.title = isFav ? 'Remove favorite' : 'Save as favorite';
    const item = document.getElementById('bmFavFolder');
    if (item) item.textContent = isFav ? '★ Favorite folder (remove)' : '☆ Favorite folder';
}

// ── Directory change ──
// opts.chosen: somebody picked this folder in the path field — the server
// keeps it as the library when it is another one than the saved (a jump the
// page makes by itself, e.g. to the upload folder, is not kept).
async function changeDir(opts = {}) {
    if (typeof closePathPicker === 'function') closePathPicker();
    const input = document.getElementById('dirInput');
    const status = document.getElementById('dirStatus');
    const path = input.value.trim();
    if (!path) return;

    status.textContent = 'Loading…';
    status.className = 'dir-status';

    try {
        const r = await fetch('/api/set_directory', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({path, chosen: opts.chosen === true})
        });
        // Parse defensively: a non-JSON response (proxy error page or
        // similar) should produce a readable message, not a JSON.parse error.
        const data = await r.json().catch(() => ({error: `HTTP ${r.status}`}));
        if (data.error) {
            status.textContent = data.error;
            status.className = 'dir-status err';
        } else {
            status.textContent = '✓';
            status.className = 'dir-status ok';
            input.value = data.directory;
            updateFavBtn();
            // Only reset the view if the queue isn't currently processing anything
            if (!queueRunning) {
                activeFile = null;
                activeTranscript = null;
                document.getElementById('transcriptArea').innerHTML =
                    '<div class="empty-state"><div class="icon">📂</div><p>Select a file</p></div>';
                document.getElementById('playerBar').classList.remove('visible');
                document.getElementById('resizeHandle').classList.remove('visible');
            }
            await scanFiles();
            setTimeout(() => { status.textContent = ''; }, 2000);
        }
    } catch(e) {
        status.textContent = 'Error: ' + e.message;
        status.className = 'dir-status err';
    }
}

// ── Auto-walk (filesystem walk on open on/off) ──
// Off = file list comes ONLY from the DB (instant, no directory traversal).
// The 🔄 button then triggers the walk manually when needed.
function autoWalkEnabled() {
    try { return localStorage.getItem('wtmAutoWalk') !== '0'; } catch(e) { return true; }
}
function onAutoWalkToggle() {
    const on = document.getElementById('autoWalkToggle')?.checked ?? true;
    try { localStorage.setItem('wtmAutoWalk', on ? '1' : '0'); } catch(e) {}
    scanFiles();
}
// Restore checkbox state on load
document.addEventListener('DOMContentLoaded', () => {
    const cb = document.getElementById('autoWalkToggle');
    if (cb) cb.checked = autoWalkEnabled();
});

// Manual refresh: force a walk — independent of the auto-walk toggle
// ── ⋯ folder menu (scan on open, index) ──
function toggleFolderMenu(ev) {
    ev?.stopPropagation();
    const menu = document.getElementById('folderMenu');
    if (!menu) return;
    menu.hidden = !menu.hidden;
    document.getElementById('btnFolderMore')?.setAttribute('aria-expanded', String(!menu.hidden));
}

function closeFolderMenu() {
    const menu = document.getElementById('folderMenu');
    if (menu) menu.hidden = true;
    document.getElementById('btnFolderMore')?.setAttribute('aria-expanded', 'false');
}

document.addEventListener('click', e => {
    if (!e.target.closest?.('.dir-actions .bb-more')) closeFolderMenu();
});
document.addEventListener('keydown', e => { if (e.key === 'Escape') closeFolderMenu(); });

function refreshWalk() {
    scanFiles({forceWalk: true});
}

// ── File scanning ──
// Origin suffix of the status line (" · DB"), remembered from the last scan.
let _dirStatusSource = '';
let _lastOrphanCount = -1;   // last reported orphan count (toast debouncing)

/**
 * Rebuilds the status line "N files · M done · DB" from the current files
 * array. Called after every incremental single update so the done counter
 * keeps up without having to rescan the whole library.
 *
 * @param {boolean} force  true only from within scanFiles (that's what first
 *   sets the ok state). Without force, another state — error, limit
 *   notice, "no DB entries" — is not overwritten.
 */
function renderDirStatusCounts(force = false) {
    const status = document.getElementById('dirStatus');
    if (!status) return;
    if (!force && status.className !== 'dir-status ok') return;

    // Numbers come from the server aggregates (the client list is only
    // a window since pagination was introduced).
    // The counts are in the sidebar (filter segments, progress panel) —
    // the dir bar only speaks up for transient states and errors.
    status.textContent = '';
    status.title = _dirStatusSource ? 'List loaded from the database' : 'List from a fresh filesystem scan';
    status.className = 'dir-status ok';
}

let dirStatsData = null;

let _scanSeq = 0;  // #211: sequence token against superseded scan responses

// ── Walk progress chip (dir bar) ──
// An exact percentage is impossible during the walk (total is only known
// at the end). Two estimators: expected file count from the DB (last
// walk's count — accurate on re-walks) and otherwise finished/known
// folders (converges toward 100%). The processing phase afterward has
// exact numbers.
function _updateWalkChip(d) {
    const el = document.getElementById('walkChip');
    if (!el) return;
    let pct = null, text = '';
    if (d.phase === 'walk' && d.waiting) {
        // NAS outage: walk is paused and waiting for the mount to come back
        const mins = Math.floor((d.wait_secs || 0) / 60);
        text = `⏸ Waiting for network drive… (attempt ${d.wait_attempt || 1}`
            + (mins > 0 ? `, for ${mins}min` : '') + ')';
        el.title = (d.wait_reason || 'Network drive unreachable')
            + `\nSo far: ${plural(d.files, 'file')} · ${plural(d.dirs_done, 'folder')}`
            + '\nThe scan will resume automatically once the drive is back.';
        el.textContent = text;
        el.style.background = 'var(--bg-3)';
        el.classList.add('visible', 'waiting');
        return;
    }
    el.classList.remove('waiting');
    if (d.phase === 'walk') {
        const files = d.files || 0;
        // If the DB estimate has overflowed (found more than expected),
        // it would stay stuck at "99%" — switch to the folder estimate then.
        if (d.expected_files > 0 && files < d.expected_files) {
            pct = Math.min(99, files / d.expected_files * 100);
        } else if (d.dirs_known > 0) {
            pct = Math.min(99, (d.dirs_done || 0) / d.dirs_known * 100);
        }
        text = `🔍 ${plural(files, 'file')} · ${plural(d.dirs_done, 'folder')}`;
        el.title = 'Filesystem walk running'
            + (d.expected_files > 0
                ? ` — expecting ~${plural(d.expected_files, 'file')} (DB count from last walk)`
                : ' — progress estimated from finished/discovered folders')
            + (d.errors ? `\n⚠ ${plural(d.errors, 'folder')} unreadable` : '');
    } else if (d.phase === 'db') {
        text = '🗄 Loading DB…';
        el.title = 'Status & metadata are being loaded from the database';
    } else if (d.phase === 'process') {
        pct = d.total > 0 ? (d.done / d.total * 100) : 0;
        text = `⚙ ${(d.done || 0).toLocaleString()}/${(d.total || 0).toLocaleString()}`;
        el.title = 'Files are being processed (status, size, metadata)';
    } else {
        return;  // heartbeat etc.
    }
    el.textContent = text + (pct != null ? ` · ${Math.round(pct)}%` : '');
    // Fill level as background gradient = mini progress bar inside the chip
    el.style.background = pct != null
        ? `linear-gradient(to right, color-mix(in srgb, var(--accent) 30%, transparent) ${pct}%, var(--bg-3) ${pct}%)`
        : 'var(--bg-3)';
    el.classList.add('visible');
}

function _hideWalkChip() {
    const el = document.getElementById('walkChip');
    if (el) { el.classList.remove('visible', 'waiting'); el.textContent = ''; el.style.background = ''; }
}

// Walk as SSE stream: progress events → chip, complete event → result.
// IMPORTANT: on onerror do NOT allow the EventSource auto-reconnect — every
// reconnect would start a NEW full scan server-side. Abort instead; the
// caller falls back to the classic fetch.
function _fetchScanStream(seq) {
    return new Promise((resolve, reject) => {
        const src = new EventSource('/api/scan_stream');
        let settled = false;
        let lastEvt = Date.now();
        let wd = null;
        const finish = (fn, arg) => {
            if (settled) return;
            settled = true;
            if (wd) clearInterval(wd);
            try { src.close(); } catch(e) {}
            fn(arg);
        };
        // Watchdog: the server sends a heartbeat event at least every 10s —
        // 45s of silence therefore really means a dead connection, not a long walk.
        wd = setInterval(() => {
            if (Date.now() - lastEvt > 45000) finish(reject, new Error('Scan stream timeout'));
        }, 5000);
        src.onmessage = (ev) => {
            lastEvt = Date.now();
            if (seq !== _scanSeq) { finish(reject, new Error('superseded')); return; }
            let d; try { d = JSON.parse(ev.data); } catch(e) { return; }
            if (d.error) {
                // Scan error reported by the SERVER (e.g. network drive dead):
                // resolve as a result, do NOT reject — the reject path starts
                // the fetch fallback, and thus a second full scan that is
                // just as doomed to fail.
                finish(resolve, {error: d.error, files: [], unchosen: !!d.unchosen});
                return;
            }
            if (d.complete) { finish(resolve, d); return; }
            if (d.phase) _updateWalkChip(d);
        };
        src.onerror = () => finish(reject, new Error('Scan stream interrupted'));
    });
}

async function scanFiles(opts = {}) {
    // Walk only if auto-walk is on OR explicitly requested (🔄 button);
    // otherwise the list comes from the DB in milliseconds. The scan itself
    // always runs recursively/ultra (server default).
    const walk = opts.forceWalk || autoWalkEnabled();
    const status = document.getElementById('dirStatus');
    const spinner = document.getElementById('scanSpinner');

    wtmDebug('Scan', `scanFiles(walk=${walk})`);
    // #211: Only the latest scanFiles response may set the state — otherwise
    // a slower earlier response would overwrite newer status (done→pending)
    // and lose _processing flags.
    const seq = ++_scanSeq;
    spinner.style.display = 'inline-block';
    status.textContent = walk ? 'Scanning…' : 'Loading from DB…';
    status.className = 'dir-status';

    try {
        if (walk) {
            // Walk reconciles filesystem ↔ DB (with live progress); the
            // DISPLAY then loads paginated from the DB (/api/files).
            // Show what the DB already knows while the walk runs: on a NAS
            // it takes minutes, and until then the list read "No files —
            // 🔄 for a filesystem scan" although the counters showed thousands.
            if (!srvList.total) reloadFileList();
            let data;
            try {
                data = await _fetchScanStream(seq);
            } catch(e) {
                if (seq !== _scanSeq) return;         // a newer scan takes over
                wtmDebug('Scan', 'Stream failed, falling back to fetch: ' + e.message);
                const r = await fetch('/api/scan?walk=true');
                data = await r.json();
            }
            if (seq !== _scanSeq) return;  // #211: a newer scan response takes precedence

            if (data.error && data.unchosen) {
                // A first start without the default folder: a library to
                // choose, not an error
                status.textContent = 'Choose your library: type or pick its folder in the path field above';
                status.className = 'dir-status';
                _hideWalkChip();
                spinner.style.display = 'none';
                return;
            }
            if (data.error) {
                // A failed scan (e.g. network drive dead) must NOT replace
                // the currently displayed list with an empty one.
                status.textContent = data.error;
                status.className = 'dir-status err';
                if (typeof Toast !== 'undefined') Toast.error(data.error);
                _hideWalkChip();
                spinner.style.display = 'none';
                return;
            }

            wtmDebug('Scan', `Walk done: total=${data.total}, dir=${data.directory}, done=${data.done}`);
            if (data.directory) {
                document.getElementById('dirInput').value = data.directory;
                showPathEnd(document.getElementById('dirInput'));
            }
            // Index coverage → permanent chip in the dir bar (in index.js)
            if (typeof onScanIndexInfo === 'function') {
                onScanIndexInfo(data.index, data.directory);
            }
            // Found orphaned transcripts (file gone, transcript still there):
            // report once per changed state, don't nag on every scan.
            if (data.orphans > 0 && data.orphans !== _lastOrphanCount) {
                _lastOrphanCount = data.orphans;
                if (typeof Toast !== 'undefined') {
                    Toast.info(`👻 ${plural(data.orphans, 'orphaned transcript')} found `
                             + `(media file missing) — view via the "Orphaned" `
                             + `filter and delete if needed`);
                }
            } else if (!data.orphans) {
                _lastOrphanCount = 0;
            }
            _dirStatusSource = '';
            status.className = 'dir-status ok';
        } else {
            _dirStatusSource = ' · DB';
            status.className = 'dir-status ok';
        }
        if (seq !== _scanSeq) return;

        // Display: freshly load the paginated list + aggregates. The
        // counters in the status line are written by loadServerStats()
        // after the response.
        //
        // The scroll position stays put if the scan ran in the BACKGROUND
        // (triggered by a finished transcription). Otherwise the list would
        // jump to the top under the user without them having done anything.
        reloadFileList({preserveScroll: !!opts.background});
        if (typeof loadServerStats === 'function') loadServerStats(true);
    } catch(e) {
        if (seq !== _scanSeq) return;  // #211: a stale scan silently aborts
        status.textContent = 'Error: ' + e.message;
        status.className = 'dir-status err';
    }

    _hideWalkChip();
    spinner.style.display = 'none';
    // Update speaker dropdown for search (server-side)
    if (typeof updateSearchSpeakers === 'function') updateSearchSpeakers();
}
