// ══════════════════════════════════════════════════════════════════════
//  Media indexing (FFprobe metadata → DB)
//  - Permanent index chip in the dir bar (instead of popup + modal):
//    shows coverage (indexed/total) and, during a run, the live
//    progress with ETA.
//  - 📇 button starts/stops the indexing.
//  - Index statistics in the sidebar widget.
// ══════════════════════════════════════════════════════════════════════

let _scanIndexInfo = null;              // {total, indexed, unindexed, indexed_pct}
let _indexSSE = null;                   // running index connection
let _indexStartTs = 0;
let _indexWatchdog = null;              // watchdog against a dead SSE connection (no complete/error)
// This long without an event → treat as an error.
//
// 60 s was too tight: the indexer used to report only after walk AND
// DB load, so on a very large library that's only after several minutes.
// The watchdog closed the stream too early and reported a "timeout",
// while the server kept running invisibly. The indexer now sends
// heartbeats during both phases (scanner/indexer.py); this value is
// the second line of defense in case a phase does block for longer
// after all (dead NAS mount).
const _INDEX_WATCHDOG_MS = 180000;

// Called after every scan (from directory.js) — updates the chip.
function onScanIndexInfo(info, directory) {
    _scanIndexInfo = info || null;
    // While an index run is in progress, the chip shows the live progress;
    // the scan status would just overwrite it.
    if (!_indexSSE) _renderIndexChip();
}

// ── Permanent index chip (dir bar) ─────────────────────────────────────
// The chip only shows a running index; the coverage itself sits where the
// action is — on the "Index metadata" item of the ⋯ folder menu — and in
// the sidebar's progress details.
function _renderIndexChip() {
    const el = document.getElementById('indexChip');
    if (el) {
        el.textContent = '';
        el.className = 'index-chip';
    }
    _setIndexButton(false);
}

function _setIndexChipProgress(text, title) {
    const el = document.getElementById('indexChip');
    if (!el) return;
    el.textContent = text;
    el.className = 'index-chip busy';
    if (title) el.title = title;
}

function _setIndexButton(running) {
    const btn = document.getElementById('btnIndex');
    if (!btn) return;
    const missing = _scanIndexInfo ? _scanIndexInfo.unindexed : 0;
    btn.textContent = running ? '⏹ Stop indexing'
        : `📇 Index metadata${missing ? ` · ${missing.toLocaleString()} missing` : ''}`;
    btn.title = running
        ? 'Stop indexing — anything already indexed stays saved'
        : missing ? `${plural(missing, 'file')} without stored metadata (duration, resolution, codecs)`
                  : 'All files indexed — nothing missing';
    btn.classList.toggle('active', running);
}

// ── Watchdog against a dead SSE connection ──────────────────────────────
// Restarted on EVERY incoming event and only fires if the stream simply
// goes silent without complete/error (e.g. connection dead). This way a
// normally running, long index run is NOT falsely aborted.
function _armIndexWatchdog() {
    _clearIndexWatchdog();
    _indexWatchdog = setTimeout(() => {
        _indexWatchdog = null;
        if (!_indexSSE) return;
        try { _indexSSE.close(); } catch(e) {}
        _indexSSE = null;
        _indexError('Timeout — no response from the index process');
    }, _INDEX_WATCHDOG_MS);
}

function _clearIndexWatchdog() {
    if (_indexWatchdog) { clearTimeout(_indexWatchdog); _indexWatchdog = null; }
}

// ── Button: start/stop ──────────────────────────────────────────────────
function toggleIndexing() {
    if (_indexSSE) cancelIndexing();
    else startIndexing(false);
}

// ── Start indexing ───────────────────────────────────────────────────────
function startIndexing(force = false) {
    if (_indexSSE) { _indexSSE.close(); _indexSSE = null; }

    _indexStartTs = performance.now();
    _setIndexButton(true);
    _setIndexChipProgress('📇 Starting…', 'Indexing running');

    const url = `/api/index?recursive=true&force=${force ? 'true' : 'false'}`;
    const src = new EventSource(url);
    _indexSSE = src;
    _armIndexWatchdog();

    src.onmessage = (ev) => {
        _armIndexWatchdog();   // event received → restart watchdog
        let d; try { d = JSON.parse(ev.data); } catch(e) { return; }
        if (d.error) {
            _clearIndexWatchdog();
            src.close(); _indexSSE = null;
            _indexError(d.error);
            return;
        }
        if (d.complete) {
            _clearIndexWatchdog();
            src.close(); _indexSSE = null;
            _indexDone(d);
            return;
        }
        _updateIndexProgress(d);
    };
    src.onerror = () => {
        // The stream also ends with onerror on normal completion; only treat
        // it as an error if we did NOT end cleanly via {complete:true}.
        if (_indexSSE) {
            _clearIndexWatchdog();
            _indexSSE = null;
            src.close();
            _indexError('Connection to the index process lost');
        }
    };
}

function cancelIndexing() {
    _clearIndexWatchdog();
    if (_indexSSE) { _indexSSE.close(); _indexSSE = null; }
    _renderIndexChip();
    _setIndexButton(false);
    if (typeof Toast !== 'undefined') Toast.info('Indexing stopped — anything already indexed stays saved');
    // Progress is persisted; the next run picks up from there.
    // Re-scan updates the chip + file list with the new state.
    if (typeof scanFiles === 'function') scanFiles();
}

// ── Progress in the chip ─────────────────────────────────────────────────
function _updateIndexProgress(d) {
    const total = d.total || 0;
    const done = d.done || 0;
    // Preparation phases: the indexer doesn't know the total yet and
    // sends total=0. Without this branch the chip would show "📇 0/0" for minutes.
    if (d.phase === 'collect' || d.phase === 'db') {
        _setIndexChipProgress(
            d.phase === 'collect' ? '📇 Searching files…' : '📇 Loading index state…',
            'Indexing is being prepared' + (d.current ? `\n▸ ${d.current}` : ''));
        return;
    }
    if (d.phase === 'transcripts') {
        _setIndexChipProgress('📇 Transcripts…',
            'Discovering existing transcripts'
            + (d.current ? `\n▸ ${d.current}` : ''));
        return;
    }
    let s = `📇 ${done.toLocaleString()}/${total.toLocaleString()}`;
    if (d.eta_sec != null) s += ` · ~${_fmtEta(d.eta_sec)}`;
    const rate = _fmtRate(d.rate_sec, d.sec_per_file);
    let title = 'Indexing running';
    if (d.skipped) title += ` · ${d.skipped.toLocaleString()} skipped`;
    if (d.failed) title += ` · ${plural(d.failed, 'error')}`;
    if (rate) title += ` · ${rate}`;
    if (d.current) title += `\n▸ ${d.current}`;
    _setIndexChipProgress(s, title);
}

function _indexDone(d) {
    // The run is over: a reload from the database (no walk, the default)
    // never reports the coverage back, and the chip stayed "Transcripts…"
    _renderIndexChip();
    _setIndexButton(false);
    const took = d.elapsed_sec || ((performance.now() - _indexStartTs) / 1000);
    if (typeof Toast !== 'undefined') {
        if (d.db_error) {
            Toast.error('Index finished, but DB write error: ' + d.db_error);
        } else {
            Toast.success(`Index done: ${(d.indexed||0).toLocaleString()} indexed · `
                + `${(d.skipped||0).toLocaleString()} skipped`
                + (d.failed ? ` · ${plural(d.failed, 'error')}` : '')
                + (d.healed_transcripts ? ` · ${plural(d.healed_transcripts, 'transcript')} discovered` : '')
                + ` · ${_fmtEta(took)}`);
        }
    }
    // Update sidebar statistics + chip
    // Re-scan loads the new metadata and sets the chip via onScanIndexInfo
    if (typeof scanFiles === 'function') scanFiles();
}

function _indexError(msg) {
    _setIndexButton(false);
    _renderIndexChip();  // back to the coverage state
    if (typeof Toast !== 'undefined') Toast.error('Indexing failed: ' + String(msg));
}

function _fmtEta(s) {
    if (s == null) return '?';
    if (s < 60) return Math.round(s) + 's';
    if (s < 3600) return Math.floor(s/60) + 'min ' + Math.round(s%60) + 's';
    return Math.floor(s/3600) + 'h ' + Math.floor((s%3600)/60) + 'min';
}

// Throughput compact: at ≥1 file/s as "N/s", below that as time per file
// (more meaningful when each file takes several seconds — e.g. NAS).
function _fmtRate(rateSec, secPerFile) {
    if (rateSec && rateSec >= 1) return rateSec.toFixed(1) + '/s';
    if (secPerFile && secPerFile > 0) return secPerFile.toFixed(1) + 's/file';
    return '';
}


