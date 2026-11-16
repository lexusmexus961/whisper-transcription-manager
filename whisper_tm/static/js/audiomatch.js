// ── Same audio (api/audio.py, whisper_tm/audio) ───────────────────────────
//
// Finds every recording that holds a snippet — by its fingerprints, not by
// what it sounds like: the same recording, also re-encoded, quieter or with
// noise over it, at the exact second. (Things that only sound ALIKE are
// Images & sounds'.) The snippet is
//
//   a stretch    cut in the clip editor ("≋ Same audio" next to Preview)
//   a file       📎 a sound or a video of your own (its first five minutes)
//
// The other two views — the tab opens on the first — are what the analysis
// found by itself, without a snippet (audiodups.js): recordings that are
// the same, and recordings that contain another one (a compilation, a
// longer cut) — each pair found when the second of the two was analyzed.

const _am = {
    info: null, seq: 0,
    query: null,            // {type:'range', file, start, end} | {type:'file', file: File}
    result: null,           // the last answer of a search
    view: 'same',           // matches | same | contains (the last two: audiodups.js)
    active: -1,             // the hit in the preview
    timer: null,
};

function amClock(sec) {
    const neg = sec < 0;
    sec = Math.abs(Math.floor(Number(sec) || 0));
    const h = Math.floor(sec / 3600), m = Math.floor((sec % 3600) / 60), s = sec % 60;
    const out = h ? `${h}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}` : `${m}:${String(s).padStart(2, '0')}`;
    return (neg ? '−' : '') + out;
}

function amHours(h) {
    h = Number(h) || 0;
    return h >= 100 ? `${Math.round(h).toLocaleString()} h` : h >= 1 ? `${h.toFixed(1)} h` : `${Math.round(h * 60)} min`;
}

function amBytes(b) {
    b = Number(b) || 0;
    if (b >= 1e12) return `${(b / 1e12).toFixed(1)} TB`;
    if (b >= 1e9) return `${(b / 1e9).toFixed(1)} GB`;
    if (b >= 1e7) return `${Math.round(b / 1e6)} MB`;
    if (b >= 1e6) return `${(b / 1e6).toFixed(1)} MB`;
    // a game's sound effect is a few kB — not "0 MB"
    return `${b > 0 ? Math.max(1, Math.round(b / 1e3)) : 0} kB`;
}

// The folder a recording lies in. Not its file_dir: that is the root it
// opens under — for a transcribed recording the library's root, not its folder
function amFolder(f) {
    const p = String((f && f.path) || '');
    const cut = Math.max(p.lastIndexOf('/'), p.lastIndexOf('\\'));
    return cut > 0 ? p.slice(0, cut) : String((f && f.file_dir) || '');
}

function amSpan(sec) {
    sec = Math.round(Number(sec) || 0);
    if (sec < 120) return `${sec} s`;
    if (sec < 7200) return `${Math.round(sec / 60)} min`;
    if (sec < 2 * 86400) return `${Math.round(sec / 3600)} h`;
    return `${Math.round(sec / 86400)} days`;
}

// ── Opening the tab ─────────────────────────────────────────────────────────

async function amOpen() {
    await amLoadInfo();
    amRenderViews();
    amRun();
    amWatch();
}

function amLeave() {
    amPausePreview();
    if (_am.timer) { clearInterval(_am.timer); _am.timer = null; }
}

// While a background run goes, the bar follows it
function amWatch() {
    if (_am.timer) clearInterval(_am.timer);
    _am.timer = setInterval(() => {
        const pane = document.getElementById('svPaneAudio');
        if (!pane || !pane.classList.contains('active')
                || !document.getElementById('svOverlay')?.classList.contains('visible')) {
            clearInterval(_am.timer); _am.timer = null; return;
        }
        if (_am.info?.run?.running) amLoadInfo();
    }, 3000);
}

async function amLoadInfo() {
    try { _am.info = await api('/api/audio/info'); }
    catch (e) { _am.info = { error: e.message }; }
    amRenderInfo();
}

function amRenderInfo() {
    const el = document.getElementById('amInfo');
    if (!el) return;
    const i = _am.info || {};
    const c = i.counts || {};
    const run = i.run || {};
    let line;
    if (i.error) line = `<span class="ms-err">Not reachable: ${esc(i.error)}</span>`;
    else if (i.enabled === false) line = '<span class="ms-err">Audio analysis is switched off.</span>';
    else if (!c.files) line = 'Nothing analyzed yet.';
    else line = `<b>${Number(c.files).toLocaleString()}</b> recording${c.files === 1 ? '' : 's'} analyzed (${amHours(c.hours)})`
        + (c.same || c.contains ? ` · ${plural(c.same, 'pair')} the same, ${Number(c.contains || 0).toLocaleString()} contained` : '');
    if (run.running) {
        const left = run.eta_sec ? ` · about ${amSpan(run.eta_sec)} left` : '';
        line += ` · <span class="ms-building">⟳ ${esc(run.waiting ? 'Waiting: ' + run.waiting : run.message || 'analyzing …')}${left}</span>
                 <button class="ms-link" onclick="amRunStop()">Stop</button>`;
    }
    el.innerHTML = `${line} <button class="ms-link" onclick="openSettings('audio')">Settings →</button>`;
}

function amRenderViews() {
    const box = document.getElementById('amViews');
    if (!box) return;
    // The duplicates' counts once their overview is in (groups, not pairs)
    const ov = typeof _ad !== 'undefined' ? _ad.overview : null;
    const views = [
        ['same', '≋ The same recording', ov ? ov.same.groups : null],
        ['contains', '⊂ Contained in another', ov ? ov.contains.wholes : null],
        ['matches', 'Snippet matches', _am.result ? _am.result.hits.length : null],
    ];
    box.innerHTML = views.map(([k, label, n]) =>
        `<button class="ms-chip am-view${_am.view === k ? ' on' : ''}" onclick="amView('${k}')">${label}${
            n !== null && n !== undefined ? ` <small>${Number(n).toLocaleString()}</small>` : ''}</button>`).join('');
}

function amView(name) {
    _am.view = name;
    amRenderViews();
    amRun();
}

// The tab's contents: the search's hits, or the duplicates (audiodups.js)
function amRun() {
    if (_am.view === 'same' || _am.view === 'contains') {
        amRenderSource();
        if (_am.info?.enabled === false) { amRender(); return; }
        adOpen(_am.view);
        return;
    }
    if (_am.query && !_am.result) { amSearch(); return; }
    amRender();
}

// ── The snippet ─────────────────────────────────────────────────────────────

// From the clip editor: its stretch, searched in the search view's tab
function amFromClip() {
    if (typeof clipFile === 'undefined' || !clipFile) return;
    if (clipOut - clipIn < 1) { if (typeof Toast !== 'undefined') Toast.error('Choose at least one second'); return; }
    if (clipOut - clipIn > 300) { if (typeof Toast !== 'undefined') Toast.error('At most five minutes at a time'); return; }
    _am.query = { type: 'range', file: clipFile, start: clipIn, end: clipOut };
    _am.result = null;
    _am.view = 'matches';
    // The search opens on this tab, its text query left as it was
    const keep = document.getElementById('svQuery')?.value || '';
    svState.routed = keep;
    svShowTab('audio');
    openSearchView(keep);          // its svRun() opens this tab: amOpen()
}

function amPickFile(input) {
    const f = input.files && input.files[0];
    input.value = '';
    if (f) amUseFile(f);
}

function amUseFile(f) {
    _am.query = { type: 'file', file: f, name: f.name };
    _am.result = null;
    _am.view = 'matches';
    amRenderViews();
    amSearch();
}

function amDrop(ev) {
    ev.preventDefault();
    document.getElementById('svPaneAudio')?.classList.remove('am-dragging');
    const f = ev.dataTransfer?.files?.[0];
    if (f) amUseFile(f);
}

function amDragOver(ev, on) {
    if (!ev.dataTransfer?.types?.includes('Files')) return;
    ev.preventDefault();
    document.getElementById('svPaneAudio')?.classList.toggle('am-dragging', on);
}

function amRenderSource() {
    const el = document.getElementById('amSource');
    if (!el) return;
    const q = _am.query;
    const pick = `<label class="vx-btn am-pick" title="A sound or a video of your own — its first five minutes are searched">📎 A file …
        <input type="file" accept="audio/*,video/*" hidden onchange="amPickFile(this)"></label>`;
    if (!q) {
        el.innerHTML = `<span class="am-lead">Where else is a snippet?</span>${pick}<span class="am-hint">or drop one here — or cut a stretch in the clip editor (✂ under the player) and press <b>≋ Same audio</b></span>`;
        return;
    }
    const what = q.type === 'range'
        ? `<b>${amClock(q.start)}–${amClock(q.end)}</b> of <b title="${esc(q.file)}">${esc(String(q.file).split('/').pop())}</b> (${Math.round(q.end - q.start)} s)`
        : `<b>${esc(q.name)}</b>`;
    const r = _am.result;
    const took = r ? ` · ${plural(r.hashes, 'landmark')}, ${r.ms} ms` : '';
    el.innerHTML = `<span class="ms-src-ico">≋</span><span>Searching for ${what}${took}</span>${pick}`;
}

async function amSearch() {
    const box = document.getElementById('amResults');
    const q = _am.query;
    if (!box || !q) return;
    amRenderSource();
    const seq = ++_am.seq;
    box.innerHTML = `<div class="sv-empty">Searching …</div>`;
    let r;
    try {
        if (q.type === 'range') {
            r = await api('/api/audio/search', { method: 'POST', body: { file: q.file, start: q.start, end: q.end } });
        } else {
            const fd = new FormData();
            fd.append('file', q.file);
            r = await api('/api/audio/search-file', { method: 'POST', body: fd });
        }
    } catch (e) {
        if (seq === _am.seq) box.innerHTML = `<div class="sv-empty ms-err">${esc(e.message)}</div>`;
        return;
    }
    if (seq !== _am.seq) return;
    _am.result = r;
    _am.active = -1;
    amRenderSource();
    amRenderViews();
    amRender();
}

// ── The hits ────────────────────────────────────────────────────────────────

function amLoud(f) {
    if (!f || f.lufs === null || f.lufs === undefined) return '';
    return `${Number(f.lufs).toFixed(0)} LUFS`;
}

const AM_PREVIEW_EMPTY = '<div class="sv-empty">Click a recording to hear the place and open it.</div>';

function amRender() {
    const box = document.getElementById('amResults');
    if (!box) return;
    amRenderSource();
    const pv = document.getElementById('amPreview');
    if (pv && _am.active < 0) pv.innerHTML = AM_PREVIEW_EMPTY;
    const i = _am.info || {};
    if (i.enabled === false) {
        box.innerHTML = `<div class="ms-empty"><div class="ms-empty-title">≋ Same audio is switched off</div>
            <p>Switch it on under Settings → Audio analysis.</p>
            <button class="vx-btn primary" onclick="openSettings('audio')">Open the settings</button></div>`;
        return;
    }
    const r = _am.result;
    if (!r) {
        const c = i.counts || {};
        box.innerHTML = `<div class="ms-empty">
            <div class="ms-empty-title">≋ Find every recording that holds a snippet</div>
            <p>The same sound, also re-encoded, quieter or under noise — at the exact second: an intro, a
               jingle, a clip someone cut out of a longer recording. What only sounds alike is Images &amp; sounds'.</p>
            <p>${c.files ? `${plural(c.files, 'recording')} ${c.files === 1 ? 'is' : 'are'} analyzed. ` : ''}Recordings are analyzed while they
               are transcribed; the ones transcribed before need one run over the library
               (Settings → Audio analysis).</p></div>`;
        return;
    }
    if (!r.hits.length) {
        box.innerHTML = `<div class="sv-empty">No recording holds this snippet${
            (i.counts || {}).files ? ` (searched ${plural(i.counts.files, 'analyzed recording')})` : ''}.</div>`;
        return;
    }
    box.innerHTML = `<div class="ms-rows am-rows">${r.hits.map((h, n) => {
        const more = h.places.length > 1 ? ` · ${h.places.length} places` : '';
        const loud = amLoud(h);
        return `<button class="ms-row am-row${n === _am.active ? ' active' : ''}" data-n="${n}" onclick="amPick(${n})">
            <span class="am-strength ${h.strength}" title="${h.strength === 'strong' ? 'A sure match' : 'A match'} — ${h.score} landmarks in step">${h.strength === 'strong' ? '●●' : '●'}</span>
            <span class="ms-row-main">
                <span class="ms-row-name">${esc(h.name)}</span>
                <span class="ms-row-text" title="${esc(h.path)}">${esc(amFolder(h))}</span>
            </span>
            <span class="am-at" title="Where the snippet begins in this recording">at ${amClock(h.at)}${more}</span>
            <span class="ms-score" title="${loud ? 'Integrated loudness' : ''}">${amClock(h.duration)}${loud ? ' · ' + loud : ''}</span>
        </button>`;
    }).join('')}</div>
    <div class="ms-took">${plural(r.hits.length, 'recording')} · ${r.ms} ms</div>`;
    // Back from a duplicates list: the hit that was open is shown again
    if (_am.active >= 0 && _am.active < r.hits.length) amPick(_am.active);
}

function amPick(n) {
    const h = (_am.result?.hits || [])[n];
    const pv = document.getElementById('amPreview');
    if (!h || !pv) return;
    _am.active = n;
    document.querySelectorAll('#amResults .active').forEach(el => el.classList.remove('active'));
    document.querySelector(`#amResults [data-n="${n}"]`)?.classList.add('active');
    const len = _am.query?.type === 'range' ? Math.max(1, _am.query.end - _am.query.start)
        : Math.max(1, (h.q_end || 0) - (h.q_start || 0));
    const places = h.places.map((p, k) => `<div class="am-place">
            <span>at <b>${amClock(p.at)}</b> <small>${p.score} landmarks</small></span>
            <button class="vx-btn" onclick="amOpenAt(${n}, ${k}, false)" title="Play it in the player of the main window">▶ Player</button>
            <button class="vx-btn" onclick="amOpenAt(${n}, ${k}, true)" title="Open the transcript there">📄 Transcript</button>
        </div>`).join('');
    const start = Math.max(0, h.at);
    const sample = `/api/clip/audio?file=${encodeURIComponent(h.path)}&start=${start.toFixed(2)}&end=${(start + Math.min(len, 60)).toFixed(2)}`;
    pv.innerHTML = `
        <audio class="ms-audio" controls preload="none" src="${esc(sample)}"></audio>
        <div class="ms-pv-kind">≋ ${h.strength === 'strong' ? 'A sure match' : 'A match'} · ${h.score} landmarks in step</div>
        <div class="ms-pv-name">${esc(h.name)}</div>
        <div class="ms-pv-folder" title="${esc(h.path)}">${esc(amFolder(h))}</div>
        <div class="am-places"><b>${h.places.length > 1 ? `${h.places.length} places` : 'Where'}</b>${places}</div>
        ${amStats(h)}`;
    pv.querySelector('audio')?.addEventListener('play', () => {
        try { svPlayer()?.pause(); } catch (e) { /* none */ }
    });
}

function amStats(f) {
    const rows = [];
    if (f.duration) rows.push(['Length', amClock(f.duration)]);
    if (f.lufs !== null && f.lufs !== undefined) rows.push(['Loudness', `${Number(f.lufs).toFixed(1)} LUFS`]);
    if (f.lra !== null && f.lra !== undefined) rows.push(['Loudness range', `${Number(f.lra).toFixed(1)} LU`]);
    if (f.peak_db !== null && f.peak_db !== undefined) rows.push(['Peak', `${Number(f.peak_db).toFixed(1)} dBFS`]);
    if (f.silence !== null && f.silence !== undefined) rows.push(['Near silence', `${Math.round(f.silence * 100)} %`]);
    if (f.clipped) rows.push(['Clipped', `${(f.clipped * 100).toFixed(2)} % of the samples`]);
    if (!rows.length) return '';
    return `<div class="am-stats">${rows.map(([k, v]) => `<span>${k}</span><b>${esc(v)}</b>`).join('')}</div>`;
}

function amPausePreview() {
    document.querySelectorAll('#amPreview audio').forEach(m => { try { m.pause(); } catch (e) { /* gone */ } });
}

function amOpenAt(n, k, transcript) {
    const h = (_am.result?.hits || [])[n];
    if (!h) return;
    amPausePreview();
    const t = Math.max(0, (h.places[k] || h.places[0]).at);
    if (transcript) svGoto(h.filename, t, h.file_dir);
    else svPlay(h.filename, t, h.file_dir);
}

// ── Duplicates (audiodups.js) ───────────────────────────────────────────────

// From the file view's "≋ n copies": the duplicates of that recording
function amShowFile(path, same) {
    amShowDuplicates({ kind: same ? 'same' : 'contains', file: path });
}

// ── The background run, from the info bar ──────────────────────────────────

async function amRunStop() {
    try { await api('/api/audio/run/stop', { method: 'POST', body: {} }); }
    catch (e) { if (typeof Toast !== 'undefined') Toast.error(e.message); }
    amLoadInfo();
}
