// ── State ──
// Since the UI rework (5.1, pagination) `files` no longer holds the whole
// library, only the currently LOADED rows of the server buffer
// (srvList in filelist.js). fileByName() can thus return null even though
// the file exists — whoever needs the full filter set gets it via
// fetchFilteredNames() (server).
let files = [];

// ── Data revision of the file list ─────────────────────────────────────────
// EVERY mutation of `files` (new list, status/processing flags) must
// call filesChanged() — render caches (updateCounts et al.) depend on it.
let _filesRev = 0;
let _filesByName = null;   // Map name → file (lazy, invalidated via filesChanged)
function filesChanged() {
    _filesRev++;
    _filesByName = null;
}
// O(1) lookup instead of files.find(...) — updateSelectionUI & co. would
// otherwise run O(selection × library).
function fileByName(name) {
    if (!_filesByName) _filesByName = new Map(files.map(f => [f.name, f]));
    return _filesByName.get(name);
}
let activeFile = null;
// The entry selectFile() was last opened with. Needed for jumps
// from outside the loaded list (queue click, search hit): files[] is
// only a window over the current directory and is only reloaded
// asynchronously on a jump — until then the view wouldn't find its own file.
let activeFileObj = null;
let activeTranscript = null;
let activeSpeaker = null;
let pendingTxFile = null;
let currentJobId = null;
// Resolved upload directory from the server (/api/settings/directories) —
// target of all uploads/link downloads. NEVER assemble client-side.
let uploadsDir = null;

let selectedFiles = new Set();
// Status per selected name (filled on selection — from the row or the
// server name list). Needed because fileByName can no longer know
// selected items outside the window with a paginated list.
let selectionStatus = new Map();
let lastClickedFile = null;  // For shift-click range selection
let activeTab = 'all';
let sortKey = 'name';
let sortAsc = true;
let followPlayback = false;
let logLines = [];
let wordScoreVisible = false;

// ── Queue (mirror of the server) ───────────────────────────────────
// The queue lives server-side (table queue_items) and is processed by a
// permanently running worker. Everything here is ONLY a copy of the
// last server state and controls nothing — that's exactly why processing
// keeps going when this tab is closed.
// (The queue used to live in localStorage and JS had to push every job.)
let queueItems = [];          // pending entries: {id, filename, settings, dir}
let queueRunning = false;     // worker is currently processing or entries are waiting
let queuePaused = false;
let queueSolo = 0;            // entries running although the queue is paused (only the new ones)
let queueCurrentFile = null;
let queueCurrentAbs = null;   // canonical abs_path of the running file (identity)
let queueCurrentDir = null;   // root of the running entry
// Queue lanes: EVERY running entry, oldest first — the server's
// running_items ({item_id, job_id, filename, abs, dir, started_at} plus the
// job's progress, message, job_elapsed_sec, phase once the job exists).
// queueCurrent* above stay the OLDEST one, for the places that only ask
// "is anything running". From a server without lanes this is built from
// current_* (batch.js: _runningItemsFrom).
let queueRunningItems = [];
let queueRunningCount = 0;
// The queue's failed and cancelled entries ({filename, dir, error, status, …}),
// as the server last sent them — Home's Failed card reads them.
let queueFailedItems = [];
// The job the progress overlay and its live stream follow: the open file's
// when it runs, else the oldest (batch.js: syncWatchedJob). currentJobId
// stays the oldest.
let watchedJobId = null;

// Does running entry r belong to file f? Via abs_path — the name is
// relative to the currently open root and no longer matches the queue name
// after a folder jump (clicking "Processing…" once showed no progress
// because the browser had jumped into the file's folder).
function runningItemMatches(r, f) {
    if (!r || !f) return false;
    if (r.abs && f.abs_path) return f.abs_path === r.abs;
    return !!r.filename && f.name === r.filename;
}

// Is f a running file? With queue lanes several run at once — any of them.
function isRunningFile(f) {
    if (!f) return false;
    if (queueRunningItems.length) return queueRunningItems.some(r => runningItemMatches(r, f));
    if (queueCurrentAbs && f.abs_path) return f.abs_path === queueCurrentAbs;
    return !!queueCurrentFile && f.name === queueCurrentFile;
}

// One running entry's stable key (rows, change detection): the queue id,
// or the identity from a server that sends none.
function runningKey(r) {
    return r.item_id != null ? String(r.item_id) : `${r.filename}|${r.abs || ''}`;
}

// How many entries run right now — "in queue" is pending + this.
function queueRunningTotal() {
    return Math.max(queueRunningCount || 0, queueRunningItems.length, queueCurrentFile ? 1 : 0);
}
let queueTotal = 0;
let queueProcessed = 0;
let queueDone = 0;
let queueFailed = 0;
let queueSkipped = 0;
let queueCancelled = 0;
let queuePending = 0;         // entries still pending (for the total-cancel)
let queueRetryable = 0;       // failed + cancelled → can be re-queued via retry
let queueSSE = null;          // permanent status stream
let jobSSE = null;            // live log/segments of the running job
let queueDirMap = {};
let modalFiles = [];          // files available for selection in the batch modal
// Transcription defaults from /api/settings — ONLY for display: entries
// without their own settings run server-side with these values (_normalize).
// Previously the queue list hard-coded "turbo · auto" for these (old default).
let _txDefaults = {};

// Is file f waiting in the queue? Via abs_path, as runningItemMatches —
// after a folder jump (the upload dialog opens the uploads folder) the
// list's name is not the queue's, and a queued upload showed "+Q".
function isQueuedFile(f) {
    if (!f) return false;
    return queueItems.some(q => (q.abs && f.abs_path) ? q.abs === f.abs_path : q.filename === f.name);
}

function isQueued(filename) {
    const f = (typeof fileByName === 'function' && fileByName(filename)) || {name: filename};
    return isQueuedFile(f);
}

// The waiting entries as a set of keys for many rows at once: abs_path
// where the entry has one, else the name (queuedIn below)
function queuedKeys() {
    return new Set(queueItems.map(q => q.abs || q.filename));
}

function queuedIn(keys, f) {
    return !!f && ((!!f.abs_path && keys.has(f.abs_path)) || keys.has(f.name));
}

// The failed/cancelled outcome of file f, by the same identity
function queueOutcomeOf(f) {
    if (!f) return null;
    return (f.abs_path && queueOutcomes.get(f.abs_path)) || queueOutcomes.get(f.name) || null;
}

// ── Queue outcome per file (quick filter "Error"/"Cancelled") ──────────
// From the server's failed_items: filename → {id, status, error}. The
// file list uses this to mark rows (✕/⊘), the quick filters count and
// filter by it, and the filtered retry thus knows the entry IDs.
let queueOutcomes = new Map();
let _queueOutcomeRev = 0;     // feeds into the filter cache key

// Returns true if something actually changed (caller then re-renders).
function setQueueOutcomes(items) {
    const new_value = new Map();
    (items || []).forEach(i => {
        if (i.status === 'error' || i.status === 'cancelled') {
            new_value.set(i.abs || i.filename, {id: i.id, status: i.status, error: i.error || ''});
        }
    });
    if (new_value.size === queueOutcomes.size) {
        let same = true;
        for (const [name, o] of new_value) {
            const old_value = queueOutcomes.get(name);
            if (!old_value || old_value.id !== o.id || old_value.status !== o.status) { same = false; break; }
        }
        if (same) return false;
    }
    queueOutcomes = new_value;
    _queueOutcomeRev++;
    return true;
}

const SP = ['sp0','sp1','sp2','sp3','sp4','sp5','sp6','sp7'];
// The same palette as CSS values, for marks drawn in JS (pie, bars, lanes).
const SPEAKER_COLORS = SP.map(c => `var(--${c})`);
const BL = ['bl0','bl1','bl2','bl3','bl4','bl5','bl6','bl7'];

// ── Done Sound ──
let _soundMuted = false;

let _customSoundUrl = null; // Cached custom sound URL

// Preload the custom sound URL so the FIRST done sound already plays
// the configured sound (previously it always played the default the first time).
function preloadDoneSound() {
    if (_customSoundUrl !== null) return;
    _customSoundUrl = '/static/done.mp3'; // sensible default right away
    fetch('/api/sounds').then(r => r.ok ? r.json() : null).then(d => {
        if (d && d.current && d.current !== 'done.mp3') {
            _customSoundUrl = `/api/sounds/play/${encodeURIComponent(d.current)}`;
        }
    }).catch(() => {});
}

function playDoneSound() {
    if (_soundMuted) return;
    try {
        if (_customSoundUrl === null) preloadDoneSound();
        const a = new Audio(_customSoundUrl);
        a.volume = 0.5;
        const p = a.play();
        if (p) p.catch(() => console.log('Sound blocked by browser'));
    } catch(e) {}
}

function toggleDoneSound() {
    _soundMuted = !_soundMuted;
    try { localStorage.setItem('wtm_soundMuted', _soundMuted ? '1' : ''); } catch(e) {}
    // The class picks the bell icon (app.css); textContent would wipe the icons
    const btn = document.getElementById('btnSoundToggle');
    btn.classList.toggle('muted', _soundMuted);
    btn.title = _soundMuted ? 'Enable done sound' : 'Disable done sound';
}

// A top-bar icon inside text (help, setup guide): the very symbol of the
// button it points to (sprite in index.html), so it is found at a glance.
function tbIcon(name) {
    return `<svg class="ui-ico ui-ico-inline" aria-hidden="true"><use href="#i-${name}"/></svg>`;
}

// ── Utilities ──
function esc(s) {
    // Pure string escaping instead of the text-node trick: the DOM approach
    // does NOT escape quotes — in attribute contexts (title="${esc(x)}")
    // a filename like: x" onmouseover="… could break out of the
    // attribute (XSS).
    return String(s ?? '')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}
/** The file name in a Content-Disposition header: the UTF-8 form
 *  (filename*=UTF-8''…) first, then the plain one, quoted or not. */
function dispositionName(header) {
    const cd = String(header ?? '');
    const star = /filename\*\s*=\s*UTF-8''([^;]+)/i.exec(cd);
    if (star) {
        try { return decodeURIComponent(star[1].trim()); } catch (e) { /* the plain form */ }
    }
    const plain = /filename\s*=\s*(?:"((?:[^"\\]|\\.)*)"|([^;]+))/i.exec(cd);
    if (!plain) return '';
    return plain[1] !== undefined ? plain[1].replace(/\\(.)/g, '$1') : plain[2].trim();
}
function escJs(s) {
    // Safe for onclick="fn('${escJs(s)}')" — survives HTML decode → JS eval.
    // & < > MUST be encoded too: a filename like &#39; would otherwise
    // become a real ' when the onclick attribute is HTML-decoded, breaking
    // out of the JS string (XSS). Encode & BEFORE < >, otherwise the &
    // in &lt;/&gt; gets double-encoded.
    // Line breaks MUST be handled: under ext4/xfs \n is allowed in a
    // filename, and a raw newline in a single-quoted JS string is a
    // SyntaxError. That would kill the entire handler — +Q, TX, 🗑 and
    // "Transcribe" simply did nothing on this line, with no error.
    // U+2028 and U+2029 also count as line terminators in JS.
    return String(s)
        .replace(/\\/g, '\\\\')
        .replace(/'/g, '\\x27')
        .replace(/"/g, '\\x22')
        .replace(/\n/g, '\\n')
        .replace(/\r/g, '\\r')
        .replace(/\u2028/g, '\\u2028')
        .replace(/\u2029/g, '\\u2029')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');
}
