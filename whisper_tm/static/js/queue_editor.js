// ══════════════════════════════════════════════════════════════════════════
//  Queue editor — edit the queue instead of just viewing it
//
//  Purpose: change the settings of every queued entry at once (e.g. turn
//  on low-VRAM mode) or of individually selected ones. A plain preview
//  with a few buttons per row offers no way to touch the settings of
//  thousands of pending entries.
//
//  Building blocks:
//    • Filter (status, root, folder breadcrumbs, search, setting) —
//      server-side, with hit count BEFORE every action.
//    • Folder view (GET /api/queue/editor/folders): tens of thousands of
//      entries are a tree, not a list.
//    • Two-layer selection: explicitly marked ids (survive paging and
//      filter changes) OR virtual "all hits" with a deselection list.
//    • Every action — including retrying failed entries — runs as a
//      dry run first; from 20 hits (remove: always) a confirmation
//      strip appears with count, roots and a sample; remove above
//      200 only after typing the number. The real run sends `expected` —
//      if the count has changed, the server responds 409, it is
//      recounted and the strip is shown again, never written blindly.
//    • Change settings via the familiar option cards (prefix 'qe')
//      with a mode bar per group: unchanged / set to / default.
//      Only groups in "set" or "default" mode go into the request —
//      everything else stays per entry as it is.
//
//  qeTarget() is the ONLY place that builds the target part of a body — and
//  the real run sends the target frozen during the DRY RUN (qePending.target),
//  never a freshly built one: "Yes" matches what the strip described, even
//  if the filter has since changed. Any filter/selection change clears the
//  strip anyway.
// ══════════════════════════════════════════════════════════════════════════

const QE_PAGE_SIZES = [200, 500, 1000];
const QE_INSTANT_UP_TO = 20;          // settings/move up to here without a strip
const QE_COUNT_FROM = 200;            // remove from here on: type the number
const QE_ROOT_ALL = '*';        // dropdown value "all roots" ('' is the media root!)

let qeFilter = {status: ['pending'], dir: null, prefix: '', recursive: true, q: '', name: '', setting: null};
let qePage = {offset: 0, limit: 200, total: 0, items: [], processing: null, processing_items: [],
              roots: [], generation: -1};
let qeFolderData = null;            // last /folders response
let qeSelection = new Set();           // explicitly marked ids
let qeAll = false;                  // virtual "all hits marked"
let qeDeselected = new Set();        // ids unchecked in qeAll mode (→ except_ids)
let qeScope = 'all_items';                // 'all_items' | 'selected'
let qeAnchor = null;                  // last-clicked id (shift range)
let qeRootChoice = null;             // root chosen in the dropdown (null = all); folder rows
                                     // only temporarily narrow dir, ⌂ all restores this
let qeLoadNr = 0;                    // running number of the /items request — late responses expire
let qeSession = 0;                   // counts open/close: a late dry-run response from an
                                     // earlier session must no longer trigger anything
let qeFolderNr = 0;                  // ditto for /folders
let qeModeState = {};               // group → 'keep' | 'setTo' | 'standard'
let qePending = null;             // {kind, extra, dry} during confirmation
let qeRunning = false;               // request in flight → buttons locked
let qeGenTimer = null;               // pending reload for qeGenerationSeen
let qeGenLoading = false;            // that reload is still in flight
let qeGenRetryAt = 0;                // after a failed reload: not before this (ms)
let qeGenBackoff = 0;                // current wait after failures (ms)
let qeSearchTimer = null;             // 300 ms debounce
let qeListHtml = '';                // last innerHTML state
let qeServerMissing = false;           // 404 on the editor routes (older server)

const QE_GROUPS = [
    {group_nr: 'model',             keys: ['model'],                          anchor: 'qeModel', hull: '.txo-model'},
    {group_nr: 'language',          keys: ['language'],                       anchor: 'qeLang'},
    {group_nr: 'vocab_preset',      keys: ['vocab_preset'],                   anchor: 'qeVocabPreset'},
    {group_nr: 'speaker',          keys: ['diarize', 'numSpeakers'],         anchor: 'qeDiarize'},
    {group_nr: 'speaker_vectors', keys: ['speaker_vectors'],              anchor: 'qeSpeakerVectors'},
    {group_nr: 'word_timestamps',   keys: ['word_timestamps'],                anchor: 'qeWordTimestamps'},
    {group_nr: 'export_alongside',  keys: ['export_alongside'],               anchor: 'qeExportAlongside'},
    {group_nr: 'audio',             keys: ['vocal_isolation', 'vocal_model'], anchor: 'qeVocalIso'},
    {group_nr: 'save_audio',        keys: ['save_audio_alongside'],           anchor: 'qeSaveAudioAlongside'},
    {group_nr: 'low_vram',          keys: ['low_vram'],                       anchor: 'qeLowVram'},
    {group_nr: 'summarize',         keys: ['summarize', 'summarize_preset', 'summarize_model'], anchor: 'qeSummarize'},
];

const QE_KEY_LABEL = {
    model: 'Model', language: 'Language', vocab_preset: 'Vocabulary', diarize: 'Speakers',
    numSpeakers: 'Speaker count', speaker_vectors: 'Voice vectors',
    word_timestamps: 'Word timestamps', export_alongside: 'SRT alongside',
    vocal_isolation: 'Vocal Isolation', vocal_model: 'Separation model',
    save_audio_alongside: 'Audio alongside', low_vram: 'Spare VRAM', summarize: 'Summary',
    summarize_preset: 'Preset', summarize_model: 'LLM model', retranscribe: 'Re-transcription',
    batch_size: 'WhisperX batch', oom_attempts: 'OOM attempts',
};

// ── Helpers ────────────────────────────────────────────────────────────

function _qeEl(id) { return document.getElementById(id); }

function _qeToast(kind, text, duration) {
    if (typeof Toast !== 'undefined' && Toast && typeof Toast[kind] === 'function') Toast[kind](text, duration);
}

function _qeValueText(v) {
    if (v === null || v === undefined) return 'Default';
    if (v === true) return 'on';
    if (v === false) return 'off';
    if (v === '') return '—';
    return String(v);
}

function qePatchText(patch) {
    return Object.entries(patch || {}).map(([k, v]) =>
        `${QE_KEY_LABEL[k] || k} → ${_qeValueText(v)}`).join(', ');
}

function _qeShort(dir) {
    if (!dir) return '(media root)';
    const parts = dir.split('/').filter(Boolean);
    return parts.length > 2 ? '…/' + parts.slice(-2).join('/') : dir;
}

function _qeVisible(id) {
    const el = _qeEl(id);
    return !!(el && el.classList && el.classList.contains('visible'));
}

// ── Open / close ──────────────────────────────────────────────────────

function openQueueEditor(default_val) {
    qeFilter = {status: ['pending'], dir: null, prefix: '', recursive: true, q: '', name: '', setting: null,
                ...(default_val || {})};
    if (qeFilter.prefix && !qeFilter.prefix.endsWith('/')) qeFilter.prefix += '/';
    qeRootChoice = (qeFilter.dir === undefined) ? null : qeFilter.dir;
    qePage.offset = 0;
    // Selections do NOT survive closing: the ids from before may long since
    // have been processed, and "Spare VRAM (3)" would then be counting ghosts.
    qeAll = false; qeDeselected.clear(); qeSelection.clear(); qeAnchor = null;
    qeScope = 'all_items';
    qePending = null; qeRunning = false; qeSession++;
    const st = _qeEl('qeStatus'); if (st) st.value = qeFilter.status[0];
    const su = _qeEl('qeSearch'); if (su) su.value = qeFilter.q || '';
    const nm = _qeEl('qeName'); if (nm) nm.value = qeFilter.name || '';
    const rk = _qeEl('qeRecursive'); if (rk) rk.checked = qeFilter.recursive !== false;
    const sf = _qeEl('qeSettingFilter'); if (sf) sf.value = '';
    const be = _qeEl('qeConfirmation'); if (be) be.style.display = 'none';
    _qeEl('queueEditorModal')?.classList.add('visible');
    // Opened with a filter (or from a list's "Edit"): the bulk editor
    if (typeof qeShowTab === 'function') qeShowTab('queue');
    qeLoad();
    qeLoadFolders();
}

function closeQueueEditor() {
    // The folder picker of "Add folder" would stay open behind the dialog
    if (typeof ppCloseIn === 'function') ppCloseIn('qa');
    _qeEl('qeSettingsModal')?.classList.remove('visible');
    _qeEl('queueEditorModal')?.classList.remove('visible');
    if (typeof QueuePage !== 'undefined') QueuePage.shown(false);
    qeCancelConfirmation();
    qeRunning = false; qeSession++;
}

// ── Loading ──────────────────────────────────────────────────────────────

function qeQuery() {
    const p = new URLSearchParams();
    p.set('status', (qeFilter.status || ['pending']).join(','));
    if (qeFilter.dir !== null && qeFilter.dir !== undefined) p.set('dir', qeFilter.dir);
    if (qeFilter.prefix) p.set('prefix', qeFilter.prefix);
    if (qeFilter.recursive === false) p.set('recursive', '0');
    if (qeFilter.q) p.set('q', qeFilter.q);
    if (qeFilter.name) p.set('name', qeFilter.name);
    if (qeFilter.setting) {
        p.set('setting_key', qeFilter.setting.key);
        const v = qeFilter.setting.value;
        p.set('setting_value', v === null || v === undefined ? 'null' : String(v));
    }
    return p.toString();
}

/* core.js' api() appends the HTTP status to the error; the text is "Not
 * found" (app.py 404 handler), never "HTTP 404". */
function _qeIs404(e) {
    return !!e && (e.status === 404 || /^HTTP 404$/.test(String(e.message || '')));
}

function qeServerOutdated() {
    qeServerMissing = true;
    const h = _qeEl('qeHint');
    if (h) { h.textContent = 'The server is still running an older version without this editor route — restart needed'; h.style.display = ''; }
    qePage = {...qePage, total: 0, items: [], processing: null, processing_items: [], roots: []};
    qeRenderList(); qeRenderScope(); qeRenderActions();
}

async function qeLoad({keepScroll = true} = {}) {
    const listing = _qeEl('qeList');
    const scroll = keepScroll && listing ? listing.scrollTop : 0;
    const nr = ++qeLoadNr;
    let d;
    try {
        d = await api(`/api/queue/editor/items?${qeQuery()}&offset=${qePage.offset}&limit=${qePage.limit}`);
    } catch (e) {
        if (nr !== qeLoadNr) return;
        if (_qeIs404(e)) { qeServerOutdated(); return false; }
        _qeToast('error', 'Could not load the queue: ' + (e && e.message ? e.message : e));
        return false;
    }
    // A slow older response (broad search) must not overwrite a newer one —
    // otherwise the button would show "(25,000)" for the "(8,000)" filter.
    if (nr !== qeLoadNr) return;
    qeServerMissing = false;
    const h = _qeEl('qeHint'); if (h) h.style.display = 'none';
    qePage = {offset: d.offset || 0, limit: d.limit || qePage.limit, total: d.total || 0,
               items: d.items || [], processing: d.processing || null,
               // Queue lanes: every running entry of the filter. A server
               // without lanes sends only the one 'processing'.
               processing_items: Array.isArray(d.processing_items) ? d.processing_items
                   : (d.processing ? [d.processing] : []),
               roots: d.roots || [],
               generation: (d.generation === undefined ? -1 : d.generation)};
    // Paging past the end (e.g. after removal): go back to the last page
    if (qePage.offset >= qePage.total && qePage.total > 0) {
        qePage.offset = Math.max(0, Math.floor((qePage.total - 1) / qePage.limit) * qePage.limit);
        return qeLoad({keepScroll});
    }
    qeRenderHead(); qeRenderList(); qeRenderScope(); qeRenderActions();
    if (listing && keepScroll) listing.scrollTop = scroll;
    return true;
}

async function qeLoadFolders() {
    const nr = ++qeFolderNr;
    let d = null;
    try {
        d = await api(`/api/queue/editor/folders?${qeQuery()}`);
    } catch (e) {
        if (nr !== qeFolderNr) return;
        if (_qeIs404(e)) { qeServerOutdated(); return; }
    }
    if (nr !== qeFolderNr) return;
    qeFolderData = d;
    qeRenderFolders();
}

// ── Rendering ──────────────────────────────────────────────────────────────

function qeRenderHead() {
    const w = _qeEl('qeChipPending');
    if (w) w.innerHTML = `⏳ <b>${(typeof queuePending === 'number' ? queuePending : 0).toLocaleString()}</b> pending`;
    const l = _qeEl('qeChipRunning');
    if (l) {
        // Queue lanes: several can run at once — the chip says how many
        const n = (typeof queueRunningTotal === 'function') ? queueRunningTotal()
            : ((typeof queueCurrentFile !== 'undefined' && queueCurrentFile) ? 1 : 0);
        l.style.display = n ? '' : 'none';
        l.textContent = n > 1 ? `⟳ ${n} running` : '⟳ running';
    }
    const f = _qeEl('qeChipError');
    const r = (typeof queueRetryable === 'number' ? queueRetryable : 0);
    if (f) { f.innerHTML = `⚠ <b>${r.toLocaleString()}</b> error/cancelled`; f.style.display = r ? '' : 'none'; }
    const t = _qeEl('qeHits'); if (t) t.textContent = (qePage.total || 0).toLocaleString();
    const tn = _qeEl('qeHitsNoun'); if (tn) tn.textContent = qePage.total === 1 ? 'hit' : 'hits';
}

function qeRenderFolders() {
    const box = _qeEl('qeFolder');
    const crumbs = _qeEl('qeCrumbs');
    const d = qeFolderData;
    // Breadcrumbs: ⌂ all › a › b
    if (crumbs) {
        let html = `<span class="qe-crumb" data-prefix="" onclick="qeCrumb('')">⌂ all</span>`;
        const segs = (qeFilter.prefix || '').split('/').filter(Boolean);
        let acc = '';
        for (const s of segs) {
            acc += s + '/';
            html += ` › <span class="qe-crumb" data-prefix="${esc(acc)}" onclick="qeCrumb('${escJs(acc)}')">${esc(s)}</span>`;
        }
        crumbs.innerHTML = html;
    }
    const onlyHere = qeFilter.recursive === false;
    const rk = _qeEl('qeRecursive'); if (rk) rk.checked = !onlyHere;
    if (!box) return;
    if (!d) { box.innerHTML = '<div class="llm-empty">No folder data</div>'; return; }
    const multipleRoots = (d.roots || []).length > 1;
    let html = '';
    if (d.files_here > 0 || onlyHere) {
        html += `<div class="qe-folder-row qe-folder-here${onlyHere ? ' active' : ''}" onclick="qeOnlyHere()" title="${onlyHere ? 'Back to including subfolders' : 'Only the files directly in this folder (without subfolders)'}">${onlyHere ? '☑' : '·'} directly here <span class="qe-count">${(d.files_here || 0).toLocaleString()}</span></div>`;
    }
    for (const f of (d.folders || [])) {
        html += `<div class="qe-folder-row" data-prefix="${esc(f.prefix)}" data-dir="${esc(f.dir || '')}"
            onclick="qePickFolder('${escJs(f.prefix)}', '${escJs(f.dir || '')}')">▸ ${esc(f.name)}${multipleRoots ? ` <span class="qe-root">${esc(_qeShort(f.dir))}</span>` : ''}<span class="qe-count">${(f.count || 0).toLocaleString()}</span></div>`;
    }
    if (d.truncated) html += `<div class="qe-folder-row qe-folder-here">… more folders (only the ${(d.folders || []).length} largest shown)</div>`;
    if (!html) html = '<div class="llm-empty">No subfolders</div>';
    box.innerHTML = html;
    // Root selection: roots come from the server counted ONLY by status and
    // therefore remain when a root is selected. Value '' is the media
    // root — "all" therefore needs its own value.
    const qeRootEl = _qeEl('qeRoot');
    if (qeRootEl) {
        let roots = (d.roots || []).map(r => ({dir: r.dir || '', count: r.count || 0}));
        const dirActive = qeFilter.dir !== null && qeFilter.dir !== undefined;
        // A set root that has nothing (anymore) in this status must still
        // remain SELECTABLE — otherwise it filters invisibly (empty select).
        if (dirActive && !roots.some(r => r.dir === qeFilter.dir)) roots = [...roots, {dir: qeFilter.dir, count: 0}];
        if (roots.length > 1) {
            qeRootEl.style.display = '';
            qeRootEl.innerHTML = `<option value="${QE_ROOT_ALL}">all roots</option>` + roots.map(r =>
                `<option value="${esc(r.dir)}">${esc(r.dir || ((d.media_dir || '') + ' (media root)'))} (${r.count.toLocaleString()})</option>`).join('');
            qeRootEl.value = dirActive ? qeFilter.dir : QE_ROOT_ALL;
        } else {
            qeRootEl.style.display = 'none';          // one root: nothing to choose
        }
    }
}

function qeIsSelected(id) {
    return qeAll ? !qeDeselected.has(id) : qeSelection.has(id);
}

function qeRenderList() {
    const box = _qeEl('qeList');
    const items = qePage.items || [];
    const multipleRoots = (qePage.roots || []).length > 1;
    const badges = (s) => (typeof queueBadges === 'function') ? queueBadges(s || {}) : [];
    const lines = (typeof _queueRows === 'function') ? _queueRows : (x => esc(x));
    let html = '';
    if (qeFilter.status[0] === 'pending' && qePage.offset === 0) {
        for (const p of _qeRunning()) {
            html += `<div class="qe-row qe-running" data-id="${p.id}">⟳ <div class="qe-name">${lines(p.filename)}</div><span class="qe-status">running — not editable</span></div>`;
        }
    }
    items.forEach((it, i) => {
        const selected = qeIsSelected(it.id);
        const st = it.status || 'pending';
        const err = it.error || '';
        html += `<div class="qe-row${selected ? ' selected' : ''}" data-id="${it.id}" onclick="qeRowClick(${it.id}, event)">`
            + `<input type="checkbox" class="qe-check" ${selected ? 'checked' : ''} onclick="event.stopPropagation(); qeRowClick(${it.id}, event)">`
            + `<span class="qe-pos">${qePage.offset + i + 1}.</span>`
            + `<div class="qe-name">${lines(it.filename)}${(it.dir && multipleRoots) ? `<div class="qe-root">${esc(it.dir)}</div>` : ''}</div>`
            + `<div class="qe-badge">${esc(badges(it.settings).join(' · '))}</div>`
            + (st !== 'pending' ? `<span class="qe-status qe-status-${esc(st)}">${esc(st)}</span><span class="qe-error" title="${esc(err)}">${esc(err.slice(0, 80))}</span>` : '')
            // The server went down while it ran: it runs alone next, and the
            // second time is booked as an error (queue.requeue_stale)
            + (st === 'pending' && it.interrupted ? `<span class="qe-status qe-status-interrupted" title="The server went down while this file ran — it runs alone next, and should that happen again it is booked as an error">⚡ interrupted</span>` : '')
            + `<button class="tx-act-btn qe-row-open" onclick="event.stopPropagation(); closeQueueEditor(); openQueueFile('${escJs(it.filename)}', '${escJs(it.dir || '')}')" title="Open file in the library (closes the editor)">↗</button>`
            + `</div>`;
    });
    if (!items.length && !html) html = '<div class="llm-empty">No entries for this filter</div>';
    if (box && html !== qeListHtml) { box.innerHTML = html; qeListHtml = html; }
    const info = _qeEl('qePageInfo');
    const from_value = qePage.total ? qePage.offset + 1 : 0;
    const to_value = Math.min(qePage.offset + items.length, qePage.total);
    if (info) info.textContent = `${from_value.toLocaleString()}–${to_value.toLocaleString()} of ${(qePage.total || 0).toLocaleString()}`;
    const z = _qeEl('qeBtnPrev'); if (z) z.disabled = qePage.offset <= 0;
    const w = _qeEl('qeBtnNext'); if (w) w.disabled = qePage.offset + qePage.limit >= qePage.total;
    const sg = _qeEl('qePageSize'); if (sg) sg.value = String(qePage.limit);
}

/* After a click, only sync checkmarks/selection — no rebuild. */
function qeSyncTicks() {
    const box = _qeEl('qeList');
    if (!box || typeof box.querySelectorAll !== 'function') return;
    for (const z of box.querySelectorAll('.qe-row[data-id]')) {
        if (z.classList.contains('qe-running')) continue;
        const id = Number(z.dataset.id);
        const m = qeIsSelected(id);
        z.classList.toggle('selected', m);
        const h = z.querySelector ? z.querySelector('.qe-check') : null;
        if (h) h.checked = m;
    }
    // qeListHtml stays the last BUILT state; the next reload builds
    // fresh with the current selections anyway.
}

function qeSelectedCount() {
    return qeAll ? Math.max(0, (qePage.total || 0) - qeDeselected.size) : qeSelection.size;
}

function qeCount() {
    return qeScope === 'all_items' ? (qePage.total || 0) : qeSelectedCount();
}

function qeRenderScope() {
    const k = qeSelectedCount();
    const m = _qeEl('qeSelected'); if (m) m.textContent = k.toLocaleString();
    const a = _qeEl('qeScopeAllLabel'); if (a) a.textContent = `all hits (${(qePage.total || 0).toLocaleString()})`;
    const b = _qeEl('qeScopeSelectedLabel'); if (b) b.textContent = `marked only (${k.toLocaleString()})`;
    const ra = _qeEl('qeScopeAll'); if (ra) ra.checked = qeScope === 'all_items';
    const rm = _qeEl('qeScopeSelected'); if (rm) { rm.checked = qeScope === 'selected'; rm.disabled = (k === 0); }
}

function qeFilterIsDefault() {
    const f = qeFilter;
    return f.status.length === 1 && f.status[0] === 'pending' && (f.dir === null || f.dir === undefined)
        && !f.prefix && !f.q && !f.name && !f.setting && f.recursive !== false;
}

/* THE only place that builds the target part of the body. */
function qeTarget() {
    const filterPart = () => (qeFilterIsDefault() ? {all: true} : {filter: {...qeFilter}});
    if (qeScope === 'all_items') return filterPart();
    if (qeAll) return {...filterPart(), except_ids: [...qeDeselected].sort((a, b) => a - b)};
    return {ids: [...qeSelection].sort((a, b) => a - b)};
}

function qeRenderActions() {
    const n = qeCount();
    const isLocked = (n === 0 || qeRunning || qeServerMissing);
    const pending = qeFilter.status[0] === 'pending';
    const setIt = (id, text) => {
        const el = _qeEl(id); if (!el) return;
        el.textContent = `${text} (${n.toLocaleString()})`;
        el.disabled = isLocked;
        el.style.display = pending ? '' : 'none';
    };
    setIt('qeBtnSettings', '⚙ Change settings…');
    setIt('qeBtnLowVram', '🧠 Turn on Spare VRAM');
    setIt('qeBtnTop', '⤒ To front');
    setIt('qeBtnBottom', '⤓ To back');
    setIt('qeBtnRemove', '✕ Remove');
    const r = _qeEl('qeBtnRetry');
    if (r) {
        r.style.display = ['error', 'cancelled'].includes(qeFilter.status[0]) ? '' : 'none';
        r.textContent = `↩ Re-queue (${n.toLocaleString()})`;
        r.disabled = isLocked;
        r.title = 'Re-queue the shown or marked entries — dry run first, then the number';
    }
}

// ── Filter ───────────────────────────────────────────────────────────────

function qeSetFilter(part) {
    qeCancelConfirmation();               // the strip described the OLD target
    Object.assign(qeFilter, part);
    const rk = _qeEl('qeRecursive'); if (rk) rk.checked = qeFilter.recursive !== false;
    qePage.offset = 0;
    qeAll = false; qeDeselected.clear();
    if (qeSelection.size === 0) qeScope = 'all_items';
    else _qeToast('info', `Selection of ${plural(qeSelection.size, 'entry', 'entries')} is kept`, 2000);
    qeLoad(); qeLoadFolders();
}

function qeStatusChanged(v) {
    // ids belong to ONE status view: an entry marked under "Failed"
    // is not a hit under "Pending" and must not count there.
    // The tree differs per status: folder closed, root = dropdown choice.
    qeSelection.clear(); qeAnchor = null;
    qeSetFilter({status: [v || 'pending'], prefix: '', dir: qeRootChoice});
}
function qeRootChanged(v) {
    qeRootChoice = (v === QE_ROOT_ALL || v === null || v === undefined) ? null : v;
    qeSetFilter({dir: qeRootChoice});
}
function qeRecursiveChanged(b) { qeSetFilter({recursive: !!b}); }
function qeOnlyHere() { qeSetFilter({recursive: qeFilter.recursive === false}); }
function qePickFolder(prefix, dir) { qeSetFilter({prefix: prefix || '', dir: (dir === undefined ? null : dir)}); }
function qeCrumb(prefix) {
    // Back to the root of the tree: the root taken along via a folder row
    // is dropped, the one chosen in the dropdown remains.
    qeSetFilter(prefix ? {prefix} : {prefix: '', dir: qeRootChoice});
}

/* Both text fields — the path and the file name alone ("interview" as a
 * name, not every file in an interview folder) — narrow together. */
function qeSearchChanged() {
    clearTimeout(qeSearchTimer);
    qeSearchTimer = setTimeout(() => {
        const text = id => (_qeEl(id)?.value || '').trim();
        qeSetFilter({q: text('qeSearch'), name: text('qeName')});
    }, 300);
}

function qeSettingFilterChanged(v) {
    if (!v) { qeSetFilter({setting: null}); return; }
    const i = v.indexOf('=');
    const key = v.slice(0, i), raw = v.slice(i + 1);
    const value = raw === 'true' ? true : raw === 'false' ? false : raw === 'null' ? null : raw;
    qeSetFilter({setting: {key, value}});
}

function qeSwitchPage(delta) {
    const new_value = qePage.offset + delta * qePage.limit;
    if (new_value < 0 || new_value >= qePage.total) return;
    qePage.offset = new_value;
    qeLoad({keepScroll: false});
}

function qeSetPageSize(n) {
    const g = Number(n);
    qePage.limit = QE_PAGE_SIZES.includes(g) ? g : 200;
    qePage.offset = 0;
    qeLoad({keepScroll: false});
}

// ── Selection ──────────────────────────────────────────────────────────────

function _qeToggle(id) {
    if (qeAll) { if (qeDeselected.has(id)) qeDeselected.delete(id); else qeDeselected.add(id); }
    else { if (qeSelection.has(id)) qeSelection.delete(id); else qeSelection.add(id); }
}

function _qeSelect(id) {
    if (qeAll) qeDeselected.delete(id); else qeSelection.add(id);
}

function _qeScopeAfterClick() {
    const k = qeSelectedCount();
    if (qeScope === 'all_items' && k > 0) qeScope = 'selected';
    if (k === 0) qeScope = 'all_items';
}

// The running rows of this page (queue lanes: several at once)
function _qeRunning() {
    const runs = qePage.processing_items || [];
    return runs.length ? runs : (qePage.processing ? [qePage.processing] : []);
}

function qeRowClick(id, ev) {
    ev = ev || {};
    if (_qeRunning().some(p => p.id === id)) return;
    const items = qePage.items || [];
    if (ev.shiftKey && qeAnchor !== null) {
        const a = items.findIndex(x => x.id === qeAnchor);
        const b = items.findIndex(x => x.id === id);
        if (a >= 0 && b >= 0) {
            const [from_value, to_value] = a < b ? [a, b] : [b, a];
            for (let i = from_value; i <= to_value; i++) _qeSelect(items[i].id);
        } else {
            _qeSelect(id);
        }
    } else if (ev.ctrlKey || ev.metaKey) {
        _qeToggle(id);
    } else {
        _qeToggle(id);
        qeAnchor = id;
    }
    _qeScopeAfterClick();
    qeCancelConfirmation();
    qeSyncTicks(); qeRenderScope(); qeRenderActions();
}

function qeSelectAll() {
    qeAll = true; qeDeselected.clear(); qeSelection.clear(); qeScope = 'selected';
    qeCancelConfirmation();
    qeSyncTicks(); qeRenderScope(); qeRenderActions();
}

function qeClearSelection() {
    qeAll = false; qeDeselected.clear(); qeSelection.clear(); qeScope = 'all_items';
    qeSyncTicks(); qeRenderScope(); qeRenderActions();
}

function qeSetScope(s) {
    qeScope = (s === 'selected' && qeSelectedCount() > 0) ? 'selected' : 'all_items';
    qeCancelConfirmation();
    qeRenderScope(); qeRenderActions();
}

// ── Actions: dry run → strip → real run ──────────────────────────

const QE_VERB = {
    settings: {run: 'Changing', inf: 'change', done: '✎ Settings changed'},
    remove:   {run: 'Removing', inf: 'remove', done: '🧹 Removed'},
    move_top: {run: 'Moving to front', inf: 'move to front', done: '⤒ Moved to front'},
    move_bottom: {run: 'Moving to back', inf: 'move to back', done: '⤓ Moved to back'},
    retry:    {run: 'Re-queueing:', inf: 're-queue', done: '↩ Re-queued'},
};

function _qeVerb(kind, extra) {
    return QE_VERB[kind === 'move' ? `move_${(extra && extra.to) || 'top'}` : kind] || QE_VERB.settings;
}

/* Dry run → (strip →) real run. Returns true if the dry run was accepted —
 * the settings dialog only closes then.
 * `stripe` forces the strip even under 20 hits (after a 409:
 * the count has changed, the user should see it, not us). */
async function qeAction(kind, extra, {stripe = false, prefix = ''} = {}) {
    if (qeRunning) return false;
    extra = extra || {};
    const n = qeCount();
    if (n === 0) { _qeToast('warn', 'No entries in the target'); return false; }
    qeCancelConfirmation();               // an old strip no longer applies from here on
    const target = qeTarget();                   // FROZEN: this is exactly what the dry run checks
    const session = qeSession;
    qeRunning = true; qeRenderActions();
    let dry;
    try {
        dry = await api(`/api/queue/editor/${kind}`, {method: 'POST', body: {...target, ...extra, dry_run: true}});
    } catch (e) {
        if (session !== qeSession) return false;   // editor meanwhile closed/reopened
        qeRunning = false; qeRenderActions();
        if (_qeIs404(e)) { qeServerOutdated(); return false; }
        _qeToast('error', (e && e.message) ? e.message : String(e));
        return false;
    }
    if (session !== qeSession) return false;
    qeRunning = false; qeRenderActions();
    if ((dry.affected || 0) === 0) {
        // Marked entries that have meanwhile been processed/removed: nothing
        // to do — and don't sell "changed — 0 entries" as a success.
        _qeToast('warn', `${n === 1 ? 'The entry is no longer' : `None of the ${n.toLocaleString()} entries are still`} in the target — list reloaded`);
        qeClearSelection();
        qeLoad(); qeLoadFolders();
        return false;
    }
    qePending = {kind, extra, dry, target};
    // Go straight through only if the dry run confirms EXACTLY what the
    // button promised: few hits, no ghosts among the marked, no 409.
    const ghosts = _qeGhosts(dry);
    if (!stripe && kind !== 'remove' && (dry.affected || 0) <= QE_INSTANT_UP_TO && !ghosts) {
        return qeConfirm();
    }
    qeShowConfirmation(dry, kind, extra, prefix);
    return true;
}

/* Marked entries that are neither a hit nor "currently running" — processed,
 * removed, or in a different status. */
function _qeGhosts(dry) {
    if (!dry || !dry.requested) return 0;
    return Math.max(0, dry.requested - (dry.affected || 0) - (dry.skipped_processing || 0));
}

function _qeNoun(kind, target, n) {
    const entries = n === 1 ? 'entry' : 'entries';
    if (kind !== 'retry') return `pending ${entries}`;
    const st = (target && target.filter && target.filter.status) || [];
    if (st.length === 1 && st[0] === 'error') return `failed ${entries}`;
    if (st.length === 1 && st[0] === 'cancelled') return `cancelled ${entries}`;
    return `failed/cancelled ${entries}`;
}

function _qeHints(d) {
    // hint/ignored_keys from the server response — also on the immediate
    // path, where no strip could show them.
    const parts = [];
    if (d && d.hint) parts.push(`Hint: ${d.hint}`);
    if (d && d.ignored_keys && d.ignored_keys.length) parts.push(`Ignored: ${d.ignored_keys.join(', ')}`);
    return parts;
}

function qeShowConfirmation(dry, kind, extra, prefix) {
    const box = _qeEl('qeConfirmation');
    if (!box) return;
    const v = _qeVerb(kind, extra);
    const target = (qePending && qePending.target) || qeTarget();
    const parts = [];
    if (prefix) parts.push(prefix);
    parts.push(`${v.run} ${(dry.affected || 0).toLocaleString()} ${_qeNoun(kind, target, dry.affected || 0)}`);
    const ghosts = _qeGhosts(dry);
    if (ghosts) parts.push(`${ghosts.toLocaleString()} of the marked entries ${ghosts === 1 ? 'is' : 'are'} no longer in the target`);
    if (dry.skipped_processing) parts.push(`${dry.skipped_processing} currently running and left untouched`);
    // Only worth a word when the entries span several library roots
    const roots = (dry.roots || []).length;
    if (roots > 1) parts.push(`from ${roots} library roots`);
    if (target.all && !(target.except_ids && target.except_ids.length)) parts.push('Affects ALL pending entries');
    if (kind === 'settings' && dry.patch) parts.push(`Change: ${qePatchText(dry.patch)}`);
    parts.push(..._qeHints(dry));
    const t = _qeEl('qeConfirmText'); if (t) t.textContent = parts.join(' · ');
    const s = _qeEl('qeConfirmSample');
    if (s) s.textContent = (dry.sample && dry.sample.length) ? 'e.g. ' + dry.sample.join(', ') : '';
    const number = _qeEl('qeCountInput');
    const ok = _qeEl('qeBtnConfirm');
    const needsCount = (kind === 'remove' && (dry.affected || 0) >= QE_COUNT_FROM);
    if (number) { number.style.display = needsCount ? '' : 'none'; number.value = ''; }
    if (ok) { ok.disabled = needsCount; ok.textContent = `Yes, ${v.inf} (${(dry.affected || 0).toLocaleString()})`; }
    if (box.classList && box.classList.toggle) box.classList.toggle('danger', kind === 'remove');
    box.style.display = '';
    if (needsCount && number && typeof number.focus === 'function') number.focus();
}

function qeCountChecked() {
    const number = _qeEl('qeCountInput'), ok = _qeEl('qeBtnConfirm');
    if (!number || !ok || !qePending) return;
    ok.disabled = (String(number.value || '').replace(/[^0-9]/g, '') !== String(qePending.dry.affected || 0));
}

function qeCancelConfirmation() {
    const box = _qeEl('qeConfirmation'); if (box) box.style.display = 'none';
    qePending = null;
}

/* Real run with the target frozen during the dry run. Returns true on
 * success (even if a new strip is shown after a 409), false on
 * an error — the settings dialog then stays open. */
async function qeConfirm() {
    if (!qePending || qeRunning) return false;
    const {kind, extra, dry, target} = qePending;
    const v = _qeVerb(kind, extra);
    const session = qeSession;
    qeRunning = true; qeRenderActions();
    let d;
    try {
        d = await api(`/api/queue/editor/${kind}`, {method: 'POST',
            body: {...(target || qeTarget()), ...extra, dry_run: false, expected: dry.affected}});
    } catch (e) {
        if (session !== qeSession) return false;
        qeRunning = false; qeRenderActions();
        const msg = (e && e.message) ? e.message : String(e);
        // A 409 or a "HitCount…" message — see tests/test_queue_editor_ui.py
        // and core/queue.py::HitCountChanged.
        if (e && e.status === 409 || /^HitCount/.test(msg)) {
            qeCancelConfirmation();
            // ONE new dry run, no write attempt — and ALWAYS with a strip:
            // otherwise it would loop dry→real→409→dry→real… without the user.
            return qeAction(kind, extra, {stripe: true, prefix: 'The queue has changed — count re-checked'});
        }
        qeCancelConfirmation();
        _qeToast('error', msg);
        return false;
    }
    qeRunning = false;
    qeCancelConfirmation();
    _qeToast('success', `${v.done} — ${plural(d.affected, 'entry', 'entries')}`
        + (d.skipped_processing ? ` (${d.skipped_processing} running and left untouched)` : ''));
    for (const h of _qeHints(d)) _qeToast('info', h, 6000);
    if (kind === 'move' && typeof queuePaused !== 'undefined' && queuePaused) _qeToast('info', 'The queue remains paused');
    // After removal/retry the marked entries are gone; after ghosts among
    // the marked ones the list is no longer accurate — either way: clear selection.
    if (kind === 'remove' || kind === 'retry' || _qeGhosts(d)) qeClearSelection();
    await qeLoad();
    await qeLoadFolders();
    if (typeof refreshQueue === 'function') await refreshQueue();
    if (typeof renderQueueList === 'function') renderQueueList();
    if (typeof renderFileList === 'function') renderFileList();
    qeRenderActions();
    return true;
}

/* Retry is an editor action like any other: same target
 * (qeTarget), dry run, number, expected — /api/queue/editor/retry matches
 * exactly the shown status, not a blanket error AND cancelled. */
function qeRetry() { return qeAction('retry', {}); }

// ── Live reload on generation change (from applyQueueState) ──────────

// Throttled, not debounced: the status stream ticks about once a second,
// so a timer reset on every tick could keep the reload from ever running
// while the queue works. One reload at a time; ticks during the wait or
// the load are absorbed — the next tick after it compares again against
// the generation the load brought. A FAILED load brings no generation, so
// every tick would retry (and toast) once a second: after a failure the
// reloads back off (5 s, doubling up to 60 s) until one succeeds.
function qeGenerationSeen(gen) {
    if (!_qeVisible('queueEditorModal')) return;
    if (gen === qePage.generation) return;
    if (qeGenTimer || qeGenLoading) return;
    if (Date.now() < qeGenRetryAt) return;
    qeGenTimer = setTimeout(async () => {
        qeGenTimer = null;
        qeGenLoading = true;
        try {
            const [ok] = await Promise.all([qeLoad(), qeLoadFolders()]);
            if (ok === false) {
                qeGenBackoff = Math.min(60000, qeGenBackoff ? qeGenBackoff * 2 : 5000);
                qeGenRetryAt = Date.now() + qeGenBackoff;
            } else if (ok) {
                qeGenBackoff = 0;
                qeGenRetryAt = 0;
            }
        } finally { qeGenLoading = false; }
    }, 1000);
}

// ── Settings dialog: mode bars per group ─────────────────────────

function qeBuildModeBars() {
    const missing = [];
    for (const g of QE_GROUPS) {
        const anchor = _qeEl(g.anchor);
        if (!anchor || typeof anchor.closest !== 'function') { missing.push(g.group_nr); continue; }
        const hull = anchor.closest(g.hull || '.field');
        if (!hull) { missing.push(g.group_nr); continue; }
        if (hull.dataset && hull.dataset.group_nr) {          // shell shared with another group
            missing.push(`${g.group_nr} (shell belongs to ${hull.dataset.group_nr})`); continue;
        }
        hull.classList.add('qe-group', 'qe-inactive');
        if (hull.dataset) hull.dataset.group_nr = g.group_nr;
        // Bar INSIDE at the start of the shell: an area hidden via visibility
        // (e.g. "Audio alongside" without Vocal Isolation) thus hides
        // its bar along with it.
        if (typeof hull.insertAdjacentHTML === 'function') {
            hull.insertAdjacentHTML('afterbegin',
                `<div class="qe-mode" data-group="${g.group_nr}"><select class="qe-mode-choice" data-group="${g.group_nr}" onchange="qeModeChanged(this)">`
                + `<option value="keep">— leave unchanged</option><option value="setTo">set to →</option>`
                + `<option value="standard">↺ Default (remove value)</option></select></div>`);
        }
        qeModeState[g.group_nr] = 'keep';
        _qeToggleGroup(hull, 'keep');
    }
    // Loudly, not silently: a group without a bar would be permanently
    // unchangeable from the editor — and no one would see it (txoptions.js markup).
    if (missing.length) console.warn('Queue editor: no mode bar for ' + missing.join(', ')
        + ' — anchor or shell (.field/.txo-model) not found in renderTxOptions');
    return missing;
}

function _qeToggleGroup(hull, mode) {
    if (!hull) return;
    if (hull.classList) hull.classList.toggle('qe-inactive', mode !== 'setTo');
    if (typeof hull.querySelectorAll !== 'function') return;
    for (const el of hull.querySelectorAll('input, select')) {
        if (el.classList && el.classList.contains('qe-mode-choice')) continue;
        el.disabled = (mode !== 'setTo');
    }
}

function qeModeChanged(sel) {
    const g = sel && sel.dataset ? sel.dataset.group : null;
    if (!g) return;
    const mode = ['keep', 'setTo', 'standard'].includes(sel.value) ? sel.value : 'keep';
    qeModeState[g] = mode;
    const hull = (typeof sel.closest === 'function') ? sel.closest('.qe-group') : null;
    _qeToggleGroup(hull, mode);
    qsRenderSummary();
}

function qeSettingsOpen() {
    const n = qeCount();
    if (n === 0) { _qeToast('warn', 'No entries in the target'); return; }
    const a = _qeEl('qsCount'); if (a) a.textContent = n.toLocaleString();
    const an = _qeEl('qsCountNoun'); if (an) an.textContent = n === 1 ? 'entry' : 'entries';
    for (const g of QE_GROUPS) qeModeState[g.group_nr] = 'keep';
    const mount = _qeEl('qeTxOptions');
    if (mount && typeof mount.querySelectorAll === 'function') {
        for (const sel of mount.querySelectorAll('.qe-mode-choice')) {
            sel.value = 'keep';
            _qeToggleGroup(typeof sel.closest === 'function' ? sel.closest('.qe-group') : null, 'keep');
        }
    }
    qsRenderSummary();
    _qeEl('qeSettingsModal')?.classList.add('visible');
}

function qeSettingsClose() {
    _qeEl('qeSettingsModal')?.classList.remove('visible');
}

/* Patch from the mode bars: only "set"/"default" groups, nothing else. */
function qeCollect() {
    const patch = {};
    const all_items = txCollect('qe');
    for (const g of QE_GROUPS) {
        const mode = qeModeState[g.group_nr] || 'keep';
        if (mode === 'keep') continue;
        if (mode === 'standard') { for (const k of g.keys) patch[k] = null; continue; }
        if (g.group_nr === 'summarize') {
            const on = !!(document.getElementById('qeSummarize') || {}).checked;
            patch.summarize = on;
            if (on) {
                if (all_items.summarize_preset !== undefined) patch.summarize_preset = all_items.summarize_preset;
                if (all_items.summarize_model !== undefined) patch.summarize_model = all_items.summarize_model;
            }
            continue;
        }
        for (const k of g.keys) {
            let v = all_items[k];
            if (k === 'numSpeakers' && (v === null || v === undefined)) v = 'auto';   // null would mean "default"
            if (v === undefined) continue;
            patch[k] = v;
        }
    }
    return patch;
}

function qsRenderSummary() {
    const patch = qeCollect();
    const n = Object.keys(patch).length;
    const z = _qeEl('qsSummary');
    if (z) z.textContent = n ? `${n} change${n === 1 ? '' : 's'}: ${qePatchText(patch)}` : 'No change selected';
    const b = _qeEl('qsBtnApply');
    if (b) { b.disabled = (n === 0); b.textContent = `Apply to ${qeCount().toLocaleString()}`; }
}

async function qeSettingsApply() {
    const patch = qeCollect();
    if (!Object.keys(patch).length) { _qeToast('warn', 'No group set to "set" or "default"'); return; }
    // Only close the dialog once the dry run is accepted: if the server
    // rejects the patch (400), the chosen modes remain.
    const ok = await qeAction('settings', {settings: patch});
    if (ok) qeSettingsClose();
}

// ── Keyboard ─────────────────────────────────────────────────────────────

function qeKeyboard(ev) {
    if (!_qeVisible('queueEditorModal')) return;
    if (ev.key === 'Escape') {
        const lsConfirm = _qeEl('lsConfirm');
        const qaConfirm = _qeEl('qaConfirm');
        if (_qeVisible('qeSettingsModal')) qeSettingsClose();
        else if (_qeEl('qeConfirmation') && _qeEl('qeConfirmation').style.display !== 'none') qeCancelConfirmation();
        else if (lsConfirm && lsConfirm.style.display !== 'none') lsCancelConfirm();
        else if (qaConfirm && qaConfirm.style.display !== 'none') qaCancelConfirm();
        else closeQueueEditor();
        if (typeof ev.stopImmediatePropagation === 'function') ev.stopImmediatePropagation();
        return;
    }
    const tag = (document.activeElement && document.activeElement.tagName) || '';
    // Ctrl+A marks queue entries — only while the queue tab is showing:
    // on "From library" or "Add folder" it would mark entries nobody sees.
    const queueTab = !_qeEl('qePaneQueue') || _qeEl('qePaneQueue').style.display !== 'none';
    if ((ev.ctrlKey || ev.metaKey) && ev.key === 'a' && queueTab
            && !['INPUT', 'TEXTAREA', 'SELECT'].includes(tag)) {
        if (typeof ev.preventDefault === 'function') ev.preventDefault();
        if (typeof ev.stopImmediatePropagation === 'function') ev.stopImmediatePropagation();
        qeSelectAll();
    }
}

// Registered before app.js' DOMContentLoaded listener (script order):
// Esc in the editor must not also clear the sidebar selection.
document.addEventListener('keydown', qeKeyboard);

// Install mode bars once renderTxOptions('qe') has run
// (txoptions.js also hooks into DOMContentLoaded — and runs before this).
document.addEventListener('DOMContentLoaded', () => {
    try { qeBuildModeBars(); } catch (e) { console.warn('Queue editor: mode bars', e); }
    // Inside the settings dialog, feed every change into the summary
    const mount = _qeEl('qeTxOptions');
    if (mount && typeof mount.addEventListener === 'function') {
        mount.addEventListener('change', () => { if (_qeVisible('qeSettingsModal')) qsRenderSummary(); });
    }
});
