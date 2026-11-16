// ── Add dialog: file OR link, ONE options block ───────────────────────────
// The source (file/link) is a visible switch; WHAT happens is decided
// solely by the options from the shared template
// (txoptions.js, upload mode): transcription=model → enqueue,
// transcription=none + vocal isolation → separation only,
// transcription=none → store only. Applies identically to file and link.

let _upSource = 'file';    // 'file' | 'link'

// ── Link collector ─────────────────────────────────────────────────────
// Several videos result in ONE job. Before this, the dialog had to be
// opened, filled in and submitted once per video — for ten videos that
// meant ten jobs, ten progress indicators and ten times cleaning up.
let _ytLinks = [];
const YT_MAX = 50;          // same limit as MAX_YT_LINKS in the server

/** How two links are recognized as the same video — the same rule as
 *  `dedup_key` in core/ytdl.py (that one is authoritative).
 *
 *  Without it the list showed twenty entries of which the server loaded
 *  nineteen — you'd then go looking for the vanished row in the result.
 *  The time anchor (t=) is meaningless for a download, a playlist
 *  (list=) is NOT. */
function _ytKey(u) {
    try {
        const z = new URL(u);
        const host = z.hostname.replace(/^www\./, '').toLowerCase();
        if (z.searchParams.has('list')) return u;
        if (host === 'youtu.be' && z.pathname.length > 1)
            return 'yt:' + z.pathname.replace(/^\//, '');
        if (/^(m\.|music\.)?youtube\.com$/.test(host)) {
            const v = z.searchParams.get('v');
            if (v) return 'yt:' + v;
            const m = z.pathname.match(/^\/(shorts|live|embed)\/([^/?]+)/);
            if (m) return 'yt:' + m[2];
        }
    } catch (e) { /* not a valid URL — then the text counts */ }
    return u;
}

function ytAddLink() {
    const el = document.getElementById('ytUrl');
    const raw = (el.value || '').trim();
    if (!raw) { el.focus(); return; }
    // Split multi-line input (from the clipboard) right away —
    // whoever pasted ten links shouldn't have to add them one by one.
    const newOnes = raw.split(/[\s,]+/).map(s => s.trim()).filter(Boolean);
    const bad = newOnes.filter(u => !/^https?:\/\//i.test(u));
    if (bad.length) {
        if (typeof Toast !== 'undefined')
            Toast.warn('That is not a link: ' + bad[0].slice(0, 60));
        return;
    }
    let along = 0;
    const already = new Set(_ytLinks.map(_ytKey));
    for (const u of newOnes) {
        if (already.has(_ytKey(u))) continue;   // silently skip duplicates
        already.add(_ytKey(u));
        if (_ytLinks.length >= YT_MAX) {
            if (typeof Toast !== 'undefined')
                Toast.warn(`At most ${YT_MAX} links per job`);
            break;
        }
        _ytLinks.push(u); along++;
    }
    if (!along && newOnes.length && typeof Toast !== 'undefined')
        Toast.info('Already in the list');
    el.value = '';
    el.focus();
    _ytDrawList();
}

function ytLinkRemove(i) {
    _ytLinks.splice(i, 1);
    _ytDrawList();
}

function _ytDrawList() {
    const box = document.getElementById('ytList');
    if (!box) return;
    if (!_ytLinks.length) { box.innerHTML = ''; _updateUploadGo(); return; }
    box.innerHTML = _ytLinks.map((u, i) =>
        '<div class="ytl-row">'
        + `<span class="ytl-nr">${i + 1}.</span>`
        + `<span class="ytl-url" title="${esc(u)}">${esc(u)}</span>`
        + `<button type="button" class="ytl-gone" title="Remove" `
        + `onclick="ytLinkRemove(${i})">✕</button>`
        + '</div>').join('')
        + `<div class="yt-sum">${_ytLinks.length} link`
        + `${_ytLinks.length === 1 ? '' : 's'} — one job</div>`;
    _updateUploadGo();
}

/** The links for the download: collected ones plus whatever is still in
 *  the field.
 *
 *  Including the field matters: whoever types a URL and presses
 *  "Download" right away, without clicking ＋ first, would otherwise have
 *  enqueued nothing and get a message instead of a download. */
function _ytAllLinks() {
    const el = document.getElementById('ytUrl');
    const urls = (el?.value || '').trim().split(/[\s,]+/)
        .map(s => s.trim()).filter(Boolean);
    const all_items = _ytLinks.slice();
    const already = new Set(all_items.map(_ytKey));
    for (const u of urls) {
        if (already.has(_ytKey(u))) continue;
        already.add(_ytKey(u));
        all_items.push(u);
    }
    return all_items;
}

function setUploadSource(q) {
    _upSource = q;
    document.getElementById('srcBtnFile').classList.toggle('active', q === 'file');
    document.getElementById('srcBtnLink').classList.toggle('active', q === 'link');
    document.getElementById('uploadFileSection').style.display = q === 'file' ? '' : 'none';
    document.getElementById('uploadYoutubeSection').style.display = q === 'link' ? '' : 'none';
    _updateUploadGo();
}

// Button label = source + chosen processing. The user sees exactly what
// will happen before clicking.
function _updateUploadGo() {
    const btn = document.getElementById('btnUploadGo');
    if (!btn || _ytJobId) return;   // during a download it's the cancel button
    const model = document.getElementById('upModel')?.value || 'turbo';
    const vi = document.getElementById('upVocalIso')?.checked;
    // With multiple links, show the count in the button: it's the one
    // number you want to check before clicking.
    const n = _upSource === 'link' ? _ytAllLinks().length : 0;
    const verb = _upSource === 'link'
        ? (n > 1 ? `⬇ Download ${n} videos` : '⬇ Download')
        : '⬆ Upload';
    if (model !== 'none') btn.textContent = `${verb} & Transcribe`;
    else if (vi) btn.textContent = `${verb} & Separate Audio`;
    else btn.textContent = _upSource === 'link'
        ? (n > 1 ? `⬇ Only download ${n} videos` : '⬇ Download only')
        : '⬆ Upload only';
}

// txoptions.js reports model changes; also listens to the VI checkbox separately.
window._txoOnChange = (p) => { if (p === 'up') _updateUploadGo(); };
document.addEventListener('DOMContentLoaded', () => {
    document.getElementById('upVocalIso')?.addEventListener('change', _updateUploadGo);
    // Typing counts too: the button states the count, which includes the
    // row not yet added in the field (_ytAllLinks).
    document.getElementById('ytUrl')?.addEventListener('input', _updateUploadGo);
    _updateUploadGo();
});

function openUpload() {
    document.getElementById('uploadModal').classList.add('visible');
    setUploadSource(_upSource);
    // Ensure the favorite at the latest here (in case the init fetch of
    // the upload directory was still running when the app started)
    if (typeof ensureUploadsFav === 'function') ensureUploadsFav();
}
function closeUpload() {
    document.getElementById('uploadModal').classList.remove('visible');
    const drop = document.getElementById('uploadDrop');
    drop.classList.remove('has-file');
    document.getElementById('uploadDropContent').innerHTML =
        '<div class="upload-drop-icon">📂</div>' +
        '<div class="upload-drop-text">Drag a file here or click</div>' +
        '<div class="upload-drop-hint">Audio & Video</div>';
    document.getElementById('uploadInput').value = '';
    const ytEl = document.getElementById('ytUrl');
    if (ytEl) ytEl.value = '';
    _ytLinks = [];
    _ytDrawList();
}

function onUploadFileSelected() {
    const input = document.getElementById('uploadInput');
    if (input.files.length) showDroppedFile(input.files[0]);
}

function showDroppedFile(file) {
    const drop = document.getElementById('uploadDrop');
    const content = document.getElementById('uploadDropContent');
    drop.classList.add('has-file');
    const sz = file.size > 1e9 ? (file.size/1e9).toFixed(1)+' GB'
             : file.size > 1e6 ? (file.size/1e6).toFixed(0)+' MB'
             : (file.size/1e3).toFixed(0)+' KB';
    const icon = file.type.startsWith('video/') ? '🎬' : '🎵';
    content.innerHTML =
        `<div class="upload-drop-file">` +
        `<span class="udf-icon">${icon}</span>` +
        `<span class="udf-name" title="${esc(file.name)}">${esc(file.name)}</span>` +
        `<span class="udf-size">${sz}</span>` +
        `</div>`;
}

// Drag-and-drop
(function() {
    let dragCounter = 0;
    document.addEventListener('DOMContentLoaded', () => {
        const drop = document.getElementById('uploadDrop');
        // Remembers whether the modal was opened automatically by a file
        // drag — only then do we close it again on drag-away without a drop.
        let autoOpenedByDrag = false;
        drop.addEventListener('dragenter', (e) => {
            e.preventDefault(); e.stopPropagation();
            dragCounter++;
            drop.classList.add('dragover');
        });
        drop.addEventListener('dragleave', (e) => {
            e.preventDefault(); e.stopPropagation();
            dragCounter--;
            if (dragCounter <= 0) { drop.classList.remove('dragover'); dragCounter = 0; }
        });
        drop.addEventListener('dragover', (e) => {
            e.preventDefault(); e.stopPropagation();
        });
        drop.addEventListener('drop', (e) => {
            e.preventDefault(); e.stopPropagation();
            drop.classList.remove('dragover');
            dragCounter = 0;
            autoOpenedByDrag = false;
            if (e.dataTransfer.files.length) {
                const input = document.getElementById('uploadInput');
                input.files = e.dataTransfer.files;
                showDroppedFile(e.dataTransfer.files[0]);
            }
        });
        document.body.addEventListener('dragenter', (e) => {
            if (!(e.dataTransfer && e.dataTransfer.types.includes('Files'))) return;
            if (!document.getElementById('uploadModal').classList.contains('visible')) {
                openUpload();
                autoOpenedByDrag = true;
            }
        });
        // Closes a modal that was opened by drag and is still empty, when
        // the drag leaves the window (relatedTarget === null) or ends. On a
        // switch between child elements relatedTarget is set → no
        // flickering from nested dragenter/dragleave.
        document.body.addEventListener('dragleave', (e) => {
            if (e.relatedTarget) return;
            if (autoOpenedByDrag && !drop.classList.contains('has-file')) closeUpload();
            autoOpenedByDrag = false;
        });
        document.body.addEventListener('dragend', () => {
            if (autoOpenedByDrag && !drop.classList.contains('has-file')) closeUpload();
            autoOpenedByDrag = false;
        });
    });
})();

// UUID helper (crypto.randomUUID is not available over HTTP without TLS)
function _genFileId() {
    if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
        return crypto.randomUUID();
    }
    return 'xxxxxxxxxxxx'.replace(/x/g, () => Math.floor(Math.random() * 16).toString(16));
}

// Collect settings from the upload modal
function _gatherUploadTxSettings() {
    return txCollect('up');
}

// ══════════════════════════════════════════════════════════════
//  Main dispatcher
// ══════════════════════════════════════════════════════════════

let _uploadInFlight = false;

async function doUpload() {
    // Double-click protection: lock the button right away, before any
    // await runs (otherwise two parallel uploads of the same file).
    if (_uploadInFlight) return;
    _uploadInFlight = true;
    const goBtn = document.getElementById('btnUploadGo');
    if (goBtn) goBtn.disabled = true;
    try {
        if (_upSource === 'link') return await doYoutubeDownload();
        return await doFileUpload();
    } finally {
        _uploadInFlight = false;
        // Every path unlocks the button again — before, only the file upload
        // did, so after a link download (or an empty "Upload"/"Download"
        // click) it stayed disabled until a page reload.
        if (goBtn) goBtn.disabled = false;
    }
}

// After upload/download: what should happen to a file (from the options)
async function _processFile(filename, settings) {
    // Transcribing and "separate only" take the same path: enqueue and let
    // the server worker do it. Previously "no transcription + vocal
    // isolation" branched off into its own separation dialog with its own
    // endpoint — that's gone, the queue can do both.
    if (settings.model !== 'none' || settings.vocal_isolation) {
        await _startUploadTranscription(filename, settings);
    } else {
        await _jumpToUploads(filename);   // stored only
    }
}

// ══════════════════════════════════════════════════════════════
//  File upload → media_dir/uploads/{filename}
// ══════════════════════════════════════════════════════════════

async function doFileUpload() {
    const input = document.getElementById('uploadInput');
    if (!input.files.length) {
        if (typeof Toast !== 'undefined') Toast.warn('Please select a file');
        return;
    }
    const file = input.files[0];
    const fileId = _genFileId();
    // The piece size comes from the server (server.upload_chunk_mb), so a
    // proxy's request limit is one setting, not a patched page
    let CHUNK = 32 * 1024 * 1024;
    try {
        const lim = await fetch('/api/upload_limits').then(r => r.ok ? r.json() : null);
        if (lim && Number(lim.chunk_bytes) > 0) CHUNK = Number(lim.chunk_bytes);
    } catch (e) { /* keep the default */ }
    const tc = Math.ceil(file.size / CHUNK);
    const settings = _gatherUploadTxSettings();

    document.getElementById('uploadProg').style.display = 'block';
    document.getElementById('btnUploadGo').disabled = true;
    const bar = document.getElementById('uploadBar');
    const info = document.getElementById('uploadInfo');

    let uploadedFilename = null;
    let finished = null;      // complete response (even without rel_path)
    try {
        // Up to 2 retries per chunk with a short backoff — a single
        // transient network error shouldn't abort a multi-GB upload
        // entirely. Resending is idempotent (same chunkIndex, same bytes).
        // Once retries are exhausted the loop aborts cleanly.
        const MAX_CHUNK_TRIES = 3;
        for (let i = 0; i < tc; i++) {
            const chunk = file.slice(i * CHUNK, (i+1) * CHUNK);
            let data = null;
            for (let attempt = 1; attempt <= MAX_CHUNK_TRIES; attempt++) {
                try {
                    const fd = new FormData();
                    fd.append('chunk', chunk);
                    fd.append('fileId', fileId);
                    fd.append('chunkIndex', i);
                    fd.append('totalChunks', tc);
                    fd.append('fileName', file.name);
                    const r = await fetch('/api/upload_chunk', {method:'POST', body:fd});
                    data = await r.json().catch(() => ({}));
                    // Check for errors per chunk — otherwise the loop keeps
                    // going on HTTP 500, or a network exception left the
                    // button disabled forever (modal only usable via reload)
                    if (!r.ok || data.error) {
                        throw new Error(data.error || `HTTP ${r.status} at chunk ${i+1}/${tc}`);
                    }
                    break; // chunk transferred successfully
                } catch(err) {
                    if (attempt >= MAX_CHUNK_TRIES) {
                        throw new Error(`${err.message} (after ${MAX_CHUNK_TRIES} attempts)`);
                    }
                    info.textContent = `Chunk ${i+1}/${tc} — retry ${attempt}/${MAX_CHUNK_TRIES - 1}…`;
                    await new Promise(res => setTimeout(res, 500 * attempt)); // short backoff
                }
            }
            bar.style.width = ((i+1)/tc*100) + '%';
            info.textContent = `Chunk ${i+1}/${tc}`;
            // Last chunk: server returns the final filename.
            // filename = rel_path to the library; null if the upload
            // directory is outside the media directory.
            if (data.status === 'complete') {
                finished = data;
                uploadedFilename = data.filename || null;
            }
        }
    } catch(e) {
        if (typeof Toast !== 'undefined') Toast.error('Upload failed: ' + e.message);
        info.textContent = '❌ ' + e.message;
        return;
    } finally {
        document.getElementById('uploadProg').style.display = 'none';
        document.getElementById('btnUploadGo').disabled = false;
    }

    closeUpload();

    if (!finished) {
        if (typeof Toast !== 'undefined') Toast.error('Upload failed');
        return;
    }
    if (!uploadedFilename) {
        // Configured upload directory is outside the library:
        // file is there, but not listable/enqueueable.
        if (typeof Toast !== 'undefined') {
            Toast.info(`File stored in ${finished.directory} — `
                + 'is outside the media directory and therefore does not '
                + 'appear in the list');
        }
        return;
    }

    await _processFile(uploadedFilename, settings);
}

// ══════════════════════════════════════════════════════════════
//  Link/playlist download (yt-dlp) → media_dir/uploads/
//
//  The server creates a background job (core/ytdl.py) and works through
//  all videos in sequence; progress via the job SSE channel
//  /api/status/<id>, cancellable. After that, the SAME options apply as
//  for the file upload: transcription or "separate only" chosen → all
//  loaded files get enqueued (including the whole playlist), "none" →
//  they just sit in uploads/.
// ══════════════════════════════════════════════════════════════

let _ytJobId = null;
let _ytSSE = null;

function cancelYoutubeDownload() {
    if (_ytJobId) fetch('/api/cancel/' + encodeURIComponent(_ytJobId), {method: 'POST'});
}

// Progress chip in the directory bar (instead of progress in the modal —
// that closes on start; the ✕ in the chip IS the cancel button).
function _ytChip(text, running_flag, pct) {
    const el = document.getElementById('ytChip');
    if (!el) return;
    if (text == null) {
        el.classList.remove('visible');
        el.innerHTML = '';
        el.style.background = '';
        return;
    }
    el.innerHTML = esc(text) + (running_flag
        ? ' <span onclick="cancelYoutubeDownload()" title="Cancel download"'
        + ' style="cursor:pointer;color:var(--red);font-weight:700;margin-left:4px">✕</span>'
        : '');
    // Fill level as a background gradient — the same mini bar as the
    // walk chip (_updateWalkChip).
    el.style.background = (pct != null && running_flag)
        ? `linear-gradient(to right, color-mix(in srgb, var(--accent) 30%, transparent) ${pct}%, var(--bg-3) ${pct}%)`
        : 'var(--bg-3)';
    el.classList.add('visible');
}

async function doYoutubeDownload() {
    const urls = _ytAllLinks();
    if (!urls.length) {
        if (typeof Toast !== 'undefined') Toast.warn('Please enter a URL');
        return;
    }
    if (_ytJobId) {
        if (typeof Toast !== 'undefined') Toast.warn('A download is already running — cancel or wait for it first');
        return;
    }
    const settings = _gatherUploadTxSettings();

    let jobId;
    try {
        const r = await fetch('/api/youtube/download', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                urls,
                // Switch in the dialog overrides the setting for THIS download
                audio_only: !!document.getElementById('ytAudioOnly')?.checked,
            }),
        });
        const data = await r.json();
        if (data.error) throw new Error(data.error);
        jobId = data.jobId;
    } catch (e) {
        if (typeof Toast !== 'undefined') Toast.error(e.message);
        return;
    }

    // Job runs server-side → close the dialog immediately and jump to the
    // upload directory. Progress and cancel are handled by the ⬇ chip in
    // the directory bar — previously the modal had two "Cancel" buttons
    // side by side.
    _ytJobId = jobId;
    closeUpload();
    _updateUploadGo();
    const inLibrary = await _jumpToUploads(null);
    // Do NOT switch to the "⟳ Active" tab: it only shows what the queue is
    // currently processing — freshly downloaded files are "pending" and
    // NEVER show up there. The list therefore stayed empty or stuck on
    // "Loading…". So make sure the active tab can show the new files.
    if (inLibrary && typeof setFileTab === 'function'
        && activeTab !== 'all' && activeTab !== 'pending') {
        setFileTab('all');
    }
    _ytChip(urls.length > 1
        ? `⬇ Starting download (${urls.length} links)…` : '⬇ Starting download…', true);
    let _seen = 0;   // highest video number seen in progress

    const done = async (ok, d) => {
        if (_ytSSE) { _ytSSE.close(); _ytSSE = null; }
        _ytJobId = null;
        if (!ok) {
            _ytChip(`❌ ${d.error || 'Download failed'}`, false);
            setTimeout(() => { if (!_ytJobId) _ytChip(null); }, 15000);
            if (typeof Toast !== 'undefined') Toast.error(d.error || 'Download failed');
            // Partial successes are already in the DB (the worker registers
            // each file immediately) — refreshing the list/counters is enough.
            reloadFileList();
            if (typeof loadServerStats === 'function') loadServerStats(true);
            return;
        }
        const res = d.result || {};
        // After the jump, the upload directory is the root → entries are
        // named by their basename.
        const files = (res.files || []).map(x => x.split('/').pop());
        _ytChip(`✅ ${res.ok}/${res.total} downloaded`
            + (res.error?.length ? ` · ${plural(res.error.length, 'error')}` : ''), false);
        setTimeout(() => { if (!_ytJobId) _ytChip(null); }, 10000);
        if (typeof Toast !== 'undefined') {
            (res.error || []).slice(0, 3).forEach(f =>
                Toast.error(`✕ ${f.title}: ${f.error}`));
        }
        if (!res.in_library && res.ok > 0) {
            // Upload directory is outside the media directory
            if (typeof Toast !== 'undefined') {
                Toast.info(`Files are in ${res.directory} — outside `
                    + 'the media directory, so they do not appear in the list');
            }
            return;
        }
        reloadFileList();
        if (typeof loadServerStats === 'function') loadServerStats(true);
        if ((settings.model !== 'none' || settings.vocal_isolation) && files.length) {
            // Transcription and/or "separate only" chosen: enqueue ALL loaded
            // files with the dialog options (queue starts up if needed). The
            // queue does both — the same path as _processFile for a file
            // upload. Transcription goes AT THE FRONT, the whole playlist
            // included, as it always did; "separate only" goes to the front
            // for a single link only — a playlist of separation jobs must not
            // push everything already waiting back.
            const front = settings.model !== 'none' || files.length === 1;
            const n = await enqueueFiles(files, _uploadQueueSettings(settings),
                                         null, {front});
            if (n && typeof Toast !== 'undefined') Toast.success(`▶ ${n} enqueued`);
        }
        if (files.length === 1) {
            try {
                const r = await fetch('/api/scan_entry/' + encodeURIComponent(files[0]));
                const e = await r.json();
                if (e.file) { activeFile = e.file.name; selectFile(e.file); }
            } catch (err) { /* row appears along with the list */ }
        }
    };

    _ytSSE = new EventSource('/api/status/' + encodeURIComponent(jobId));
    _ytSSE.onmessage = (ev) => {
        let d; try { d = JSON.parse(ev.data); } catch (e) { return; }
        if (d.message && _ytJobId) {
            _ytChip(d.message, true, d.progress);
            // The worker registers EVERY finished file in the DB right away.
            // If the number in the progress jumps forward, the previous file
            // is done → refresh the list, instead of waiting for the end of
            // the playlist (for a single video that's handled by completion).
            const m = /^⬇ (\d+)\//.exec(d.message);
            if (m && +m[1] > _seen) {
                const before = _seen;
                _seen = +m[1];
                if (before > 0) {
                    reloadFileList();
                    if (typeof loadServerStats === 'function') loadServerStats(true);
                }
            }
        }
        if (d.status === 'done') done(true, d);
        else if (d.status === 'error' || d.status === 'cancelled') done(false, d);
    };
    _ytSSE.onerror = () => {
        // Connection lost doesn't mean the job is lost — the browser
        // reconnects; only give up if the job no longer exists.
        fetch('/api/active_jobs').then(r => r.json()).then(jobs => {
            if (!(jobs.jobs || jobs || []).some?.(j => j.jobId === _ytJobId)) {
                done(false, {error: 'Connection to job lost'});
            }
        }).catch(() => {});
    };
}

// ══════════════════════════════════════════════════════════════
//  Start transcription + show result
//  filename = relative path to media_dir (e.g. "uploads/my_video.mp3")
// ══════════════════════════════════════════════════════════════

// Dialog settings → queue settings (same shape enqueueFiles expects)
function _uploadQueueSettings(s) {
    // This function used to enumerate ALL options a fourth time. If it
    // was forgotten when adding an option, the option worked everywhere
    // except on the upload/link path — silently. Now it just passes
    // through what the dialog delivered.
    const out = {...s};
    if (out.numSpeakers === 'auto') out.numSpeakers = null;
    // Only include summarization if chosen (otherwise no key at all)
    if (!s.summarize) {
        delete out.summarize;
        delete out.summarize_preset;
        delete out.summarize_model;
    }
    return out;
}


async function _startUploadTranscription(filename, settings) {
    // The uploaded file is only enqueued. The remainder — starting, reporting
    // progress, picking up the next file — is done by the server worker;
    // the overlay is fed by the persistent queue stream (batch.js). This
    // way an upload transcription survives closing the tab too.
    // Enqueue BEFORE the jump: the rel_path "uploads/…" refers to the
    // still-current root (enqueueFiles takes base_dir from the path field).
    // front=true: freshly uploaded files should be up NEXT — not waiting
    // behind an ongoing full run; a paused queue starts up.
    await enqueueFiles([filename], _uploadQueueSettings(settings), null, {front: true});
    // ALWAYS jump — even if enqueueing fails (file already pending, etc.):
    // the file is in uploads/, that's where you want to go.
    await _jumpToUploads(filename);
}

// After upload/download: jump to the upload directory
// and optionally open the fresh file. NO full walk needed anymore — the
// server registers stored files in the DB immediately.
async function _jumpToUploads(selectName) {
    const root = (document.getElementById('dirInput').value || '')
        .trim().replace(/\/+$/, '');
    const target = (typeof ensureUploadsFav === 'function' && ensureUploadsFav()) || root;
    // Also works from within a subfolder: the upload directory is allowed
    // as a root as soon as it lies under an allowed root — previously the
    // view stayed put if it wasn't under the CURRENTLY open one, and the
    // fresh file was never enqueued. Whether the jump is allowed is
    // decided by /api/set_directory.
    if (root !== target) {
        document.getElementById('dirInput').value = target;
        await changeDir();          // set_directory + reload (DB-only)
        const st = document.getElementById('dirStatus');
        if (st && st.classList.contains('err')) {   // rejected (403) — restore input, view stays
            document.getElementById('dirInput').value = root;
            return false;
        }
    } else {
        reloadFileList();
        if (typeof loadServerStats === 'function') loadServerStats(true);
    }
    if (!selectName) return true;
    // After the switch, the upload directory is the root → the entry is
    // named only by its basename.
    const base = selectName.split('/').pop();
    try {
        const r = await fetch('/api/scan_entry/' + encodeURIComponent(base));
        const d = await r.json();
        if (d.file) { activeFile = d.file.name; selectFile(d.file); }
    } catch (e) { /* row will appear along with the list at the latest */ }
    return true;
}
