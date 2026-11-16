// ── Export: one page for every way out ──────────────────────────────────
// Transcripts, subtitles, documents, video and audio — of the open
// recording ("This file") or of any set of recordings ("Multiple files":
// the library's folders on the left, the focused folder's recordings
// beside them). The server does the work (/api/export*, api/export.py):
// small text exports come back at once as the file; video, audio and big
// exports run as jobs ("Export · N files") with progress in the Queue and
// a download when done.
//
// Every way in opens this page with its scope: the recording header, the
// folder bar, the file list's selection footer and the Library.
// The options are remembered in the UI prefs (export_options); presets are
// built in (YouTube subtitles, Notes for the team) or saved by the user
// (export_presets: [{name, options: {fmt, opts, video, audio}}]).

const Export = (() => {
    const el = id => document.getElementById(id);
    const TEXT = ['txt', 'srt', 'docx', 'pdf', 'md', 'vtt', 'csv', 'json'];
    // The formats grid: two rows, the everyday ones first
    const GRID = [
        ['txt', 'Text', '.txt', 'The transcript as plain text'],
        ['srt', 'Subtitles', '.srt', 'Subtitles for video players and YouTube'],
        ['docx', 'Word', '.docx', 'A Word document'],
        ['video', 'Video', 'mp4 · webm · mkv', 'The video — converted, smaller, with subtitles burned in, or only some parts'],
        ['audio', 'Audio', 'mp3 · wav · flac …', 'The sound only — converted, mono, or only some speakers'],
        ['pdf', 'PDF', '.pdf', 'A PDF to print or send'],
        ['md', 'Markdown', '.md', 'Text with headings, for notes and wikis'],
        ['vtt', 'Web subtitles', '.vtt', 'Subtitles for web pages (WebVTT)'],
        ['csv', 'Table', '.csv', 'One line per row: start, end, speaker, text — for spreadsheets'],
        ['json', 'Data', '.json', 'Everything as data, for programs'],
    ];
    const VIDEO_FMT = [['original', 'Original'], ['mp4', 'MP4'], ['webm', 'WebM'], ['mkv', 'MKV']];
    const AUDIO_FMT = [['mp3', 'MP3'], ['wav', 'WAV'], ['flac', 'FLAC'], ['m4a', 'M4A'], ['opus', 'Opus'], ['track', 'Original track']];
    const OPT_DEFAULTS = {what: 'transcript', language: '', summary_on_top: false, speakers: [], part: 'whole',
                          lines_with: '', timestamps: 'line', names: true, paragraphs: false, unsure: false,
                          header: true, sub_cpl: 42, sub_lines: 2, sub_max: 7, packing: 'zip', keep_folders: false};
    const VIDEO_DEFAULTS = {format: 'mp4', size: 'original', subtitles: false, speakers: [], remove_pauses: false, with_transcript: false};
    const AUDIO_DEFAULTS = {format: 'mp3', quality: 'medium', sample_rate: 0, mono: false, speakers: [], remove_pauses: false};
    const PRESETS = [
        {name: 'YouTube subtitles', builtin: true, fmt: 'srt',
         opts: {names: false, sub_cpl: 42, sub_lines: 2, sub_max: 7, timestamps: 'line', what: 'transcript'}},
        {name: 'Notes for the team', builtin: true, fmt: 'docx',
         opts: {summary_on_top: true, paragraphs: true, timestamps: 'minute', names: true, header: true, what: 'transcript'}},
    ];
    const ADDED = [['', 'Any time'], ['today', 'Today'], ['yesterday', 'Yesterday'], ['7', 'Last 7 days'], ['30', 'Last 30 days']];

    let tab = 'one';
    let fmt = 'txt';
    let opts = {...OPT_DEFAULTS};
    let video = {...VIDEO_DEFAULTS};
    let audio = {...AUDIO_DEFAULTS};
    let one = null;              // {abs, name, speakers: [[label, name]], languages: []}
    // Multiple files
    let libRoot = '';            // the library root (absolute)
    let folder = '';             // the focused folder, relative to the root
    let sub = true, added = '', q = '';
    let whole = true;            // the folder box: the whole folder (with its filters)
    const picked = new Set();    // single recordings picked (abs paths)
    let listed = [];             // the focused folder's recordings as listed
    let listTotal = 0;
    let lastIdx = -1;
    const kids = new Map(), open = new Set(['']);
    let peopleHere = [];
    let count = null;            // the server's count of the current selection
    let preview = null;
    let seq = 0, prevSeq = 0, listSeq = 0;
    let jobs = [];
    let jobTimer = null;
    let restored = false;

    const visible = () => el('exportOverlay')?.classList.contains('visible');
    const isText = f => TEXT.includes(f);
    const media = () => fmt === 'video' || fmt === 'audio';

    /* ── Remembered options and presets ───────────────────────────────── */

    /* Stored options as the page can use them: the keys it knows, each of
       its default's kind. The server keeps them as free-form blocks — a
       damaged or hand-edited value (a number where a word belongs, markup
       in a field) must not break the page for everyone who opens it. */
    function fitted(defaults, stored) {
        const out = {...defaults};
        if (!stored || typeof stored !== 'object' || Array.isArray(stored)) return out;
        for (const [k, d] of Object.entries(defaults)) {
            const v = stored[k];
            if (Array.isArray(d)) {
                if (Array.isArray(v)) out[k] = v.filter(x => typeof x === 'string' && x.length <= 200).slice(0, 50);
            } else if (typeof v === typeof d
                       && (typeof v !== 'number' || Number.isFinite(v))
                       && (typeof v !== 'string' || v.length <= 200)) {
                out[k] = v;
            }
        }
        return out;
    }

    function restore() {
        if (restored || typeof Shell === 'undefined') return;
        restored = true;
        const last = Shell.pref('export_options');
        if (last && typeof last === 'object') {
            if (last.fmt && (GRID.some(g => g[0] === last.fmt))) fmt = last.fmt;
            opts = {...fitted(OPT_DEFAULTS, last.opts), speakers: []};
            video = {...fitted(VIDEO_DEFAULTS, last.video), speakers: []};
            audio = {...fitted(AUDIO_DEFAULTS, last.audio), speakers: []};
        }
    }

    let saveTimer = null;
    function remember() {
        if (typeof Shell === 'undefined') return;
        clearTimeout(saveTimer);
        // Speakers and the word filter belong to one export, not the next
        const keep = o => { const c = {...o}; delete c.speakers; delete c.lines_with; return c; };
        saveTimer = setTimeout(() => Shell.setPref('export_options', {fmt, opts: keep(opts), video: keep(video), audio: keep(audio)}), 400);
    }

    // Stored as {name, options: {...}}; used flat, as the built-in ones
    const ownPresets = () => ((typeof Shell !== 'undefined' && Array.isArray(Shell.pref('export_presets'))) ? Shell.pref('export_presets') : [])
        .map(p => ({name: p.name, ...(p.options || {})}));
    const storePresets = list => Shell.setPref('export_presets', list.map(p => {
        const {name, builtin, ...options} = p;
        return {name, options};
    }));
    const allPresets = () => [...PRESETS, ...ownPresets()];

    function applyPreset(i) {
        const p = allPresets()[Number(i)];
        if (!p) return;
        if (GRID.some(g => g[0] === p.fmt)) fmt = p.fmt;
        opts = {...fitted(OPT_DEFAULTS, p.opts), speakers: opts.speakers, lines_with: opts.lines_with};
        if (p.video) video = {...fitted(VIDEO_DEFAULTS, p.video), speakers: video.speakers};
        if (p.audio) audio = {...fitted(AUDIO_DEFAULTS, p.audio), speakers: audio.speakers};
        remember();
        draw();
        status(`Preset “${p.name}” applied`);
    }

    function savePreset() {
        const inp = el('exPresetName');
        const name = (inp?.value || '').trim().slice(0, 60);
        if (!name) { inp?.focus(); return; }
        const keep = o => { const c = {...o}; delete c.speakers; delete c.lines_with; return c; };
        const list = ownPresets().filter(p => p.name !== name);
        list.push({name, fmt, opts: keep(opts), video: keep(video), audio: keep(audio)});
        storePresets(list.slice(-20));
        drawPresets();
        status(`Saved as preset “${name}”`);
    }

    function deletePreset(name) {
        storePresets(ownPresets().filter(p => p.name !== name));
        drawPresets();
    }

    function drawPresets() {
        const box = el('exPresets');
        if (!box) return;
        const own = ownPresets();
        box.innerHTML = `<span class="ex-lbl">Presets</span>`
            + allPresets().map((p, i) => `<span class="ex-preset"><button class="ex-chip" onclick="Export.applyPreset(${i})" title="${esc(presetTitle(p))}">${esc(p.name)}</button>`
                + (p.builtin ? '' : `<button class="ex-chip-x" onclick="Export.deletePreset('${escJs(p.name)}')" title="Remove this preset" aria-label="Remove preset ${esc(p.name)}">✕</button>`) + `</span>`).join('')
            + `<span class="ex-save"><input id="exPresetName" placeholder="${own.length ? 'Name' : 'Save these settings as…'}" maxlength="60"
                 onkeydown="if(event.key==='Enter')Export.savePreset()" aria-label="Preset name">`
            + `<button class="ex-chip" onclick="Export.savePreset()" title="Keep the format and options under this name">Save as preset</button></span>`;
    }

    const presetTitle = p => `${(GRID.find(g => g[0] === p.fmt) || [, p.fmt])[1]} · ${summaryOf(p.fmt, {...OPT_DEFAULTS, ...(p.opts || {})}, {...VIDEO_DEFAULTS, ...(p.video || {})}, {...AUDIO_DEFAULTS, ...(p.audio || {})})}`;

    /* ── Opening ──────────────────────────────────────────────────────── */

    /* Open the page. scope: {file} — one recording (absolute path or one
       below the open folder); {files: [...]} — these; {folder} — a folder
       (absolute), its subfolders too; {library: rel} — a Library folder.
       Without a scope: the open recording, or the open folder. */
    function show(scope) {
        restore();
        scope = scope || {};
        picked.clear();
        whole = true;
        q = '';
        if (scope.file || (!scope.files && !scope.folder && scope.library === undefined && typeof activeFile !== 'undefined' && activeFile)) {
            setOne(scope.file || activeFileAbs());
            tab = 'one';
        } else {
            tab = 'many';
            if (scope.files && scope.files.length) {
                const absList = scope.files.map(toAbs);
                whole = false;
                absList.forEach(a => picked.add(a));
                folder = relOf(commonDir(absList));
            } else if (scope.library !== undefined) {
                folder = scope.library || '';
            } else {
                folder = relOf(scope.folder || openDir());
            }
            if (scope.added) added = scope.added;
        }
        if (typeof Shell !== 'undefined') Shell.go('export');
        else el('exportOverlay')?.classList.add('visible');
        pollJobs();
        draw();
        if (tab === 'many') loadTree().then(() => focusFolder(folder, {keep: true}));
    }

    function openDir() { return (el('dirInput')?.value || '').trim().replace(/\/+$/, ''); }
    function toAbs(p) {
        p = String(p || '');
        if (p.startsWith('/') || /^[A-Za-z]:/.test(p)) return p;
        const d = openDir();
        return d ? `${d}/${p}` : p;
    }
    function activeFileAbs() {
        if (typeof activeFileObj !== 'undefined' && activeFileObj && activeFileObj.abs_path) return activeFileObj.abs_path;
        return toAbs(typeof activeFile !== 'undefined' ? activeFile : '');
    }
    function commonDir(list) {
        let base = list.length ? list[0].split('/').slice(0, -1).join('/') : '';
        for (const p of list.slice(1)) while (base && !(p.startsWith(base + '/'))) base = base.split('/').slice(0, -1).join('/');
        return base;
    }
    function relOf(abs) {
        abs = String(abs || '').replace(/\/+$/, '');
        if (!libRoot) libRoot = (typeof Library !== 'undefined' && Library.root) || '';
        if (!libRoot || !abs || abs === libRoot) return '';
        return abs.startsWith(libRoot + '/') ? abs.slice(libRoot.length + 1) : '';
    }

    function setOne(abs) {
        const name = String(abs).split('/').pop();
        const sameAsOpen = typeof activeTranscript !== 'undefined' && activeTranscript && abs === activeFileAbs();
        let speakers = [];
        if (sameAsOpen) {
            const labels = [...new Set((activeTranscript.segments || []).map(s => s.speaker).filter(s => s && s !== '—' && s !== '-'))].sort();
            speakers = labels.map(l => [l, typeof getSpeakerName === 'function' ? getSpeakerName(l) : l]);
        }
        one = {abs, name, speakers, languages: []};
        if (sameAsOpen) {
            fetch('/api/translations?file=' + encodeURIComponent(abs))
                .then(r => r.ok ? r.json() : null).then(d => {
                    if (!d || !one || one.abs !== abs) return;
                    one.languages = (d.translations || []).map(x => x.language).filter(Boolean);
                    if (visible()) drawOptions();
                }).catch(() => {});
        }
    }

    function close() {
        el('exportOverlay')?.classList.remove('visible');
        if (typeof Shell !== 'undefined') Shell.detect();
    }

    function setTab(t) {
        if (t === tab) return;
        tab = t === 'many' ? 'many' : 'one';
        if (tab === 'one' && !one && typeof activeFile !== 'undefined' && activeFile) setOne(activeFileAbs());
        if (tab === 'many' && !kids.has('')) loadTree().then(() => focusFolder(folder || relOf(openDir()), {keep: true}));
        draw();
        if (typeof Shell !== 'undefined') Shell.detect();
    }

    /* ── Multiple files: the tree and the folder's recordings ─────────── */

    async function level(path) {
        const r = await fetch('/api/library/tree?path=' + encodeURIComponent(path));
        const d = await r.json();
        if (!r.ok) throw new Error(d.error || `HTTP ${r.status}`);
        libRoot = d.root || libRoot;
        kids.set(path, d.children || []);
        return d;
    }

    async function loadTree() {
        try { if (!kids.has('')) await level(''); }
        catch (e) { const t = el('exTree'); if (t) t.innerHTML = `<div class="ex-empty">${esc(e.message)}</div>`; }
    }

    async function focusFolder(path, o) {
        folder = path || '';
        if (!(o && o.keep)) { picked.clear(); whole = true; lastIdx = -1; }
        // Open the way down
        const parts = folder ? folder.split('/') : [];
        let at = '';
        for (const p of parts) {
            if (!kids.has(at)) { try { await level(at); } catch (e) { break; } }
            open.add(at);
            at = at ? `${at}/${p}` : p;
        }
        drawTree();
        loadList();
        loadPeople();
        refreshCount();
    }

    async function toggleNode(path, ev) {
        ev && ev.stopPropagation();
        if (open.has(path)) { open.delete(path); drawTree(); return; }
        open.add(path);
        if (!kids.has(path)) { try { await level(path); } catch (e) { /* stays closed */ } }
        drawTree();
    }

    function drawTree() {
        const box = el('exTree');
        if (!box) return;
        const rows = [];
        const walk = (path, depth) => {
            for (const c of kids.get(path) || []) {
                const isOpen = open.has(c.path);
                rows.push(`<div class="ex-node${c.path === folder ? ' on' : ''}" style="padding-left:${8 + depth * 14}px" role="treeitem"
                    aria-selected="${c.path === folder}" onclick="Export.focusFolder('${escJs(c.path)}')" title="${esc(c.path)}">`
                    + (c.has_children ? `<button class="ex-tw" onclick="Export.toggleNode('${escJs(c.path)}', event)" aria-label="${isOpen ? 'Close' : 'Open'} ${esc(c.name)}">${isOpen ? '▾' : '▸'}</button>` : '<span class="ex-tw"></span>')
                    + `<span class="ex-nn">${esc(c.name)}</span><span class="ex-nc">${(c.counts?.files || 0).toLocaleString()}</span></div>`);
                if (isOpen && c.has_children) walk(c.path, depth + 1);
            }
        };
        const rootName = libRoot ? libRoot.split('/').pop() : 'Library';
        rows.push(`<div class="ex-node${folder === '' ? ' on' : ''}" style="padding-left:8px" role="treeitem" onclick="Export.focusFolder('')"><span class="ex-tw"></span><span class="ex-nn">${esc(rootName)}</span></div>`);
        walk('', 1);
        box.innerHTML = rows.join('');
    }

    const folderAbs = () => libRoot ? (folder ? `${libRoot}/${folder}` : libRoot) : '';

    let listTimer = null;
    function loadList(delay) {
        clearTimeout(listTimer);
        listTimer = setTimeout(async () => {
            const s = ++listSeq;
            const box = el('exFiles');
            if (!folderAbs()) return;
            try {
                const r = await fetch('/api/export/files', {method: 'POST', headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({folders: [{path: folderAbs(), subfolders: sub}], added, q})});
                const d = await r.json();
                if (s !== listSeq) return;
                if (!r.ok) throw new Error(d.error || `HTTP ${r.status}`);
                listed = d.files || [];
                listTotal = d.total || 0;
                drawList();
                // The whole folder under a name filter counts what is listed
                if (whole && q) refreshCount();
            } catch (e) {
                if (s === listSeq && box) box.innerHTML = `<div class="ex-empty">${esc(e.message)}</div>`;
            }
        }, delay || 0);
    }

    function drawList() {
        const box = el('exFiles'), head = el('exFilesHead');
        if (!box) return;
        const name = folder ? folder.split('/').pop() : (libRoot ? libRoot.split('/').pop() : 'Library');
        if (head) head.innerHTML = `<label class="ex-all" title="Take the whole folder — with the filters below, however many recordings it holds">
                <input type="checkbox" ${whole ? 'checked' : ''} onchange="Export.wholeFolder(this.checked)"> All of <b>${esc(name)}</b></label>
            <span class="ex-gap"></span>
            <button class="ex-mini" onclick="Export.pickAll()" title="Pick every recording listed">All</button>
            <button class="ex-mini" onclick="Export.pickNone()" title="Pick none">None</button>`;
        if (!listed.length) { box.innerHTML = `<div class="ex-empty">${q || added ? 'No recordings match the filters.' : 'No recordings in this folder.'}</div>`; return; }
        box.innerHTML = listed.map((f, i) => {
            const on = whole || picked.has(f.abs);
            const st = f.status === 'done' ? '' : `<span class="ex-st">${f.status === 'pending' ? 'not transcribed' : esc(f.status)}</span>`;
            return `<div class="ex-file${on ? ' on' : ''}" role="option" aria-selected="${on}" onclick="Export.pick(${i}, event)" title="${esc(f.rel)}">`
                + `<span class="ex-fi">${f.video ? '🎬' : '🎵'}</span><span class="ex-fn">${esc(f.rel)}</span>${st}</div>`;
        }).join('') + (listTotal > listed.length ? `<div class="ex-more">and ${(listTotal - listed.length).toLocaleString()} more — “All of” takes them too</div>` : '');
    }

    function pick(i, ev) {
        const f = listed[i];
        if (!f) return;
        if (whole) { whole = false; picked.clear(); if (ev.ctrlKey || ev.metaKey) listed.forEach(x => picked.add(x.abs)); }
        if (ev.shiftKey && lastIdx >= 0) {
            const [a, b] = [Math.min(lastIdx, i), Math.max(lastIdx, i)];
            if (!(ev.ctrlKey || ev.metaKey)) picked.clear();
            for (let k = a; k <= b; k++) picked.add(listed[k].abs);
        } else if (ev.ctrlKey || ev.metaKey) {
            picked.has(f.abs) ? picked.delete(f.abs) : picked.add(f.abs);
            lastIdx = i;
        } else {
            picked.clear();
            picked.add(f.abs);
            lastIdx = i;
        }
        drawList();
        refreshCount();
    }

    function wholeFolder(on) { whole = !!on; if (whole) picked.clear(); drawList(); refreshCount(); }
    function pickAll() { whole = false; picked.clear(); listed.forEach(f => picked.add(f.abs)); drawList(); refreshCount(); }
    function pickNone() { whole = false; picked.clear(); drawList(); refreshCount(); }

    function setFilter(k, v) {
        if (k === 'q') q = String(v || '');
        if (k === 'added') added = String(v || '');
        if (k === 'sub') sub = !!v;
        loadList(k === 'q' ? 250 : 0);
        if (!(whole && q)) refreshCount(k === 'q' ? 250 : 0);
    }

    async function loadPeople() {
        peopleHere = [];
        if (!folderAbs()) return;
        try {
            const r = await fetch('/api/library/speakers?folder=' + encodeURIComponent(folderAbs()));
            const d = await r.json();
            peopleHere = (d.people || []).map(p => p.name).filter(Boolean).slice(0, 40);
        } catch (e) { /* no suggestions */ }
        if (visible()) drawOptions();
    }

    /* ── The selection as the server takes it ─────────────────────────── */

    function selection() {
        if (tab === 'one') return one ? {files: [one.abs]} : null;
        if (whole) return folderAbs() ? {folders: [{path: folderAbs(), subfolders: sub}], added, ...(q ? {} : {})} : null;
        return picked.size ? {files: [...picked]} : null;
    }

    function body() {
        const b = {...selection(), format: fmt, options: opts};
        if (fmt === 'video') b.video = video;
        if (fmt === 'audio') b.audio = audio;
        return b;
    }

    let countTimer = null;
    function refreshCount(delay) {
        clearTimeout(countTimer);
        countTimer = setTimeout(async () => {
            const s = ++seq;
            const sel = selection();
            if (!sel) { count = {count: 0, transcribed: 0, videos: 0, size: 0}; drawFoot(); refreshPreview(); return; }
            // A name filter over the whole folder: those listed
            const useSel = (tab === 'many' && whole && q) ? {files: listed.map(f => f.abs)} : sel;
            try {
                const r = await fetch('/api/export/count', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(useSel)});
                const d = await r.json();
                if (s !== seq) return;
                count = r.ok ? d : {error: d.error || `HTTP ${r.status}`};
            } catch (e) { if (s === seq) count = {error: e.message}; }
            drawFoot();
            refreshPreview();
        }, delay || 0);
    }

    let prevTimer = null;
    function refreshPreview() {
        clearTimeout(prevTimer);
        prevTimer = setTimeout(async () => {
            const s = ++prevSeq;
            const box = el('exPreview'), head = el('exPrevHead');
            if (!box) return;
            const sel = selection();
            if (!sel) { box.textContent = ''; if (head) head.textContent = 'Preview'; preview = null; drawFoot(); return; }
            const b = body();
            if (tab === 'many' && whole && q) { delete b.folders; b.files = listed.map(f => f.abs); }
            try {
                const r = await fetch('/api/export/preview', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(b)});
                const d = await r.json();
                if (s !== prevSeq) return;
                if (!r.ok) throw new Error(d.error || `HTTP ${r.status}`);
                preview = d;
                if (head) head.innerHTML = `Preview${d.name ? ` · <b>${esc(d.name)}</b>` : ''}${d.note ? ` <span class="ex-note">${esc(d.note)}</span>` : ''}`;
                box.textContent = d.text ? d.text + (d.truncated ? '\n…' : '') : (d.note || 'Nothing to show.');
                box.classList.toggle('ex-dim', !d.text);
            } catch (e) {
                if (s === prevSeq) { box.textContent = e.message; box.classList.add('ex-dim'); preview = null; }
            }
            drawFoot();
        }, 150);
    }

    /* ── Drawing ──────────────────────────────────────────────────────── */

    function draw() {
        if (!el('exportOverlay')) return;
        document.querySelectorAll('#exTabs .vx-tab').forEach(b => {
            const on = b.dataset.tab === tab;
            b.classList.toggle('active', on);
            b.setAttribute('aria-selected', on ? 'true' : 'false');
        });
        el('exOne').hidden = tab !== 'one';
        el('exMany').hidden = tab !== 'many';
        drawOne();
        if (tab === 'many') { drawTree(); drawList(); drawManyFilters(); }
        drawPresets();
        drawFormats();
        drawSub();
        drawOptions();
        refreshCount();
    }

    function drawOne() {
        const box = el('exOne');
        if (!box || tab !== 'one') return;
        if (!one) {
            box.innerHTML = `<div class="ex-empty">No recording is open. Open one from the file list, or pick recordings under <button class="ex-link" onclick="Export.setTab('many')">Multiple files</button>.</div>`;
            return;
        }
        const ppl = speakerItems().map(x => x[1]);
        box.innerHTML = `<div class="ex-onecard"><div class="ex-onename">${esc(one.name)}</div>`
            + `<div class="ex-onepath">${esc(one.abs)}</div>`
            + (ppl.length ? `<div class="ex-oneline">${esc(ppl.join(' · '))}</div>` : '')
            + `<div class="ex-oneline"><button class="ex-link" onclick="Export.setTab('many')">Export more recordings at once →</button></div></div>`;
    }

    function drawManyFilters() {
        const box = el('exManyFilters');
        if (!box) return;
        box.innerHTML = `<input class="ex-q" id="exQ" type="search" placeholder="Filter by name" value="${esc(q)}" oninput="Export.setFilter('q', this.value)" aria-label="Filter recordings by name">`
            + `<select id="exAdded" onchange="Export.setFilter('added', this.value)" aria-label="Added" title="Only recordings that came into the library then">${ADDED.map(([v, t]) => `<option value="${v}"${v === added ? ' selected' : ''}>${t === 'Any time' ? 'Added any time' : 'Added ' + t.toLowerCase()}</option>`).join('')}</select>`
            + `<label class="ex-ck" title="The recordings in its subfolders too"><input type="checkbox" ${sub ? 'checked' : ''} onchange="Export.setFilter('sub', this.checked)"> Subfolders</label>`;
    }

    function drawFormats() {
        const box = el('exFormats');
        if (!box) return;
        box.innerHTML = GRID.map(([k, label, ext, title]) =>
            `<button class="ex-fmt${fmt === k ? ' on' : ''}" role="radio" aria-checked="${fmt === k}" onclick="Export.setFormat('${k}')" title="${esc(title)}">`
            + `<span class="ex-fl">${esc(label)}</span><span class="ex-fe">${esc(ext)}</span></button>`).join('');
    }

    function drawSub() {
        const box = el('exSub');
        if (!box) return;
        if (!media()) { box.hidden = true; box.innerHTML = ''; return; }
        box.hidden = false;
        const o = fmt === 'video' ? video : audio;
        const list = fmt === 'video' ? VIDEO_FMT : AUDIO_FMT;
        box.innerHTML = `<span class="ex-lbl">${fmt === 'video' ? 'Video as' : 'Audio as'}</span>`
            + list.map(([k, t]) => `<button class="ex-chip${o.format === k ? ' on' : ''}" onclick="Export.setMedia('format', '${k}')">${t}</button>`).join('');
    }

    function sel(name, val, choices, onchange, title) {
        return `<select ${title ? `title="${esc(title)}"` : ''} aria-label="${esc(name)}" onchange="${onchange}">`
            + choices.map(([v, t]) => `<option value="${esc(String(v))}"${String(v) === String(val) ? ' selected' : ''}>${esc(t)}</option>`).join('') + `</select>`;
    }
    const ck = (label, on, onchange, title) =>
        `<label class="ex-ck"${title ? ` title="${esc(title)}"` : ''}><input type="checkbox" ${on ? 'checked' : ''} onchange="${onchange}"> ${esc(label)}</label>`;

    /* [value, shown] per person: a named voice by its name (labels that share
     * a name are one item), an unnamed one by its label, shown as "Speaker N"
     * the way the export writes it (SPEAKER_00 → Speaker 1). */
    const speakerItems = () => {
        const seen = new Set(), out = [];
        for (const [l, n] of (one ? one.speakers : [])) {
            const m = /^SPEAKER_(\d+)$/.exec(l);
            const item = n && n !== l ? [n, n] : [l, m ? `Speaker ${Number(m[1]) + 1}` : l];
            if (seen.has(item[0])) continue;
            seen.add(item[0]);
            out.push(item);
        }
        return out;
    };

    function speakerPicker(cur, kind) {
        // [value, shown]: an unnamed speaker goes by its label, shown as "Speaker N"
        const items = tab === 'one' && one ? speakerItems() : peopleHere.map(n => [n, n]);
        const names = items.map(x => x[0]);
        const set = new Set(cur || []);
        const chips = items.map(([n, shown]) => `<button class="ex-chip${set.has(n) ? ' on' : ''}" onclick="Export.toggleSpeaker('${kind}', '${escJs(n)}')" aria-pressed="${set.has(n)}">${esc(shown)}</button>`).join('');
        const extra = [...set].filter(n => !names.includes(n))
            .map(n => `<button class="ex-chip on" onclick="Export.toggleSpeaker('${kind}', '${escJs(n)}')">${esc(n)} ✕</button>`).join('');
        return `<div class="ex-row"><span class="ex-k">Only these speakers</span><span class="ex-chips">${chips}${extra}`
            + `<input class="ex-addname" placeholder="${names.length ? 'another name' : 'a name'} ↵" onkeydown="if(event.key==='Enter'&&this.value.trim()){Export.toggleSpeaker('${kind}', this.value.trim());}" aria-label="Add a speaker name">`
            + `${set.size ? `<button class="ex-link" onclick="Export.clearSpeakers('${kind}')">everyone</button>` : '<span class="ex-hint">none picked: everyone</span>'}</span></div>`;
    }

    function drawOptions() {
        const body = el('exOptsBody'), sum = el('exOptsSum');
        if (!body) return;
        if (sum) sum.innerHTML = `<span class="ex-lbl">Options</span><span class="ex-sumtext">${esc(summaryOf(fmt, opts, video, audio))}</span>`;
        const rows = [];
        const subs = fmt === 'srt' || fmt === 'vtt';
        if (fmt === 'video') {
            rows.push(`<div class="ex-row"><span class="ex-k">Size</span>${sel('Size', video.size, [['original', 'As it is'], ['1080', '1080p'], ['720', '720p'], ['480', '480p']], "Export.setMedia('size', this.value)")}</div>`);
            rows.push(`<div class="ex-row"><span class="ex-k">Subtitles</span>${ck('Burned into the picture', video.subtitles, "Export.setMedia('subtitles', this.checked)")}${ck('As a file beside the video (.srt)', video.with_transcript, "Export.setMedia('with_transcript', this.checked)")}</div>`);
            rows.push(speakerPicker(video.speakers, 'video'));
            rows.push(`<div class="ex-row"><span class="ex-k">Pauses</span>${ck('Cut out the pauses between what is said', video.remove_pauses, "Export.setMedia('remove_pauses', this.checked)")}</div>`);
        } else if (fmt === 'audio') {
            rows.push(`<div class="ex-row"><span class="ex-k">Quality</span>${sel('Quality', audio.quality, [['low', 'Small (96 kbit/s)'], ['medium', 'Good (160 kbit/s)'], ['high', 'High (256 kbit/s)']], "Export.setMedia('quality', this.value)", 'For MP3, M4A and Opus')}`
                + sel('Sample rate', audio.sample_rate, [[0, 'Sample rate as it is'], [16000, '16 kHz'], [22050, '22 kHz'], [44100, '44.1 kHz'], [48000, '48 kHz']], "Export.setMedia('sample_rate', Number(this.value))")
                + ck('Mono', audio.mono, "Export.setMedia('mono', this.checked)") + `</div>`);
            rows.push(speakerPicker(audio.speakers, 'audio'));
            rows.push(`<div class="ex-row"><span class="ex-k">Pauses</span>${ck('Cut out the pauses between what is said', audio.remove_pauses, "Export.setMedia('remove_pauses', this.checked)")}</div>`);
        } else {
            const langs = tab === 'one' && one ? one.languages : [];
            rows.push(`<div class="ex-row"><span class="ex-k">What</span>${sel('What', opts.what, [['transcript', 'The transcript'], ['translation', 'The translation'], ['both', 'Both, line under line'], ['summary', 'Only the summary']], "Export.setOpt('what', this.value)")}`
                + (opts.what === 'translation' || opts.what === 'both'
                    ? (langs.length ? sel('Language', opts.language, [['', 'First language there is'], ...langs.map(l => [l, l.toUpperCase()])], "Export.setOpt('language', this.value)")
                                    : `<input class="ex-small" value="${esc(opts.language)}" placeholder="language, e.g. de" onchange="Export.setOpt('language', this.value.trim())" aria-label="Translation language">`) : '')
                + (opts.what !== 'summary' ? ck('Summary on top', opts.summary_on_top, "Export.setOpt('summary_on_top', this.checked)") : '') + `</div>`);
            if (opts.what !== 'summary') {
                rows.push(speakerPicker(opts.speakers, 'text'));
                rows.push(`<div class="ex-row"><span class="ex-k">Part</span>${sel('Part', opts.part, [['whole', 'The whole recording'], ['cuts', 'Only the clip cuts']], "Export.setOpt('part', this.value)", 'The cuts made in the recording’s Clips tab')}`
                    + `<input class="ex-small ex-wide" value="${esc(opts.lines_with)}" placeholder="only lines with this word" onchange="Export.setOpt('lines_with', this.value)" aria-label="Only lines with this word"></div>`);
                if (!subs) {
                    rows.push(`<div class="ex-row"><span class="ex-k">Times</span>${sel('Timestamps', opts.timestamps, [['line', 'At every line'], ['minute', 'Once a minute'], ['none', 'None']], "Export.setOpt('timestamps', this.value)")}`
                        + ck('Names', opts.names, "Export.setOpt('names', this.checked)") + ck('Paragraphs', opts.paragraphs, "Export.setOpt('paragraphs', this.checked)", "One speaker's lines in a row as one paragraph")
                        + ck('Mark unsure words', opts.unsure, "Export.setOpt('unsure', this.checked)", 'Words the model was unsure of, [like this?]')
                        + ck('Header', opts.header, "Export.setOpt('header', this.checked)", "The recording's name, length and language on top") + `</div>`);
                } else {
                    rows.push(`<div class="ex-row"><span class="ex-k">Subtitles</span>`
                        + `<label class="ex-num">at most <input type="number" min="10" max="120" value="${Number(opts.sub_cpl) || 42}" onchange="Export.setOpt('sub_cpl', Number(this.value))" aria-label="Characters per line"> characters a line</label>`
                        + `<label class="ex-num"><input type="number" min="1" max="4" value="${Number(opts.sub_lines) || 2}" onchange="Export.setOpt('sub_lines', Number(this.value))" aria-label="Lines per subtitle"> lines</label>`
                        + `<label class="ex-num">at most <input type="number" min="1" max="30" step="0.5" value="${Number(opts.sub_max) || 7}" onchange="Export.setOpt('sub_max', Number(this.value))" aria-label="Longest on screen in seconds"> s on screen</label>`
                        + ck('Names', opts.names, "Export.setOpt('names', this.checked)") + `</div>`);
                }
            }
            if (tab === 'many') {
                const canOne = !subs;
                rows.push(`<div class="ex-row"><span class="ex-k">Packing</span>${sel('Packing', canOne ? opts.packing : 'zip', canOne ? [['zip', 'A ZIP with a file each'], ['one', 'All in one document']] : [['zip', 'A ZIP with a file each']], "Export.setOpt('packing', this.value)")}`
                    + (opts.packing === 'zip' || !canOne ? ck('Keep the folders', opts.keep_folders, "Export.setOpt('keep_folders', this.checked)", 'In the ZIP, the subfolders as in the library') : '') + `</div>`);
            }
        }
        body.innerHTML = rows.join('');
    }

    function summaryOf(f, o, v, a) {
        if (f === 'video') {
            return [v.format === 'original' ? 'unchanged' : v.format.toUpperCase(), v.size !== 'original' && v.size + 'p',
                    v.subtitles && 'subtitles burned in', v.with_transcript && '+ .srt', (v.speakers || []).length && 'only ' + v.speakers.join(', '),
                    v.remove_pauses && 'no pauses'].filter(Boolean).join(' · ');
        }
        if (f === 'audio') {
            return [a.format === 'track' ? 'original track' : a.format.toUpperCase(), ['mp3', 'm4a', 'opus'].includes(a.format) && {low: 'small', medium: 'good', high: 'high'}[a.quality] + ' quality',
                    a.sample_rate && (a.sample_rate / 1000) + ' kHz', a.mono && 'mono', (a.speakers || []).length && 'only ' + a.speakers.join(', '),
                    a.remove_pauses && 'no pauses'].filter(Boolean).join(' · ');
        }
        const subs = f === 'srt' || f === 'vtt';
        const what = {transcript: 'Transcript', translation: 'Translation', both: 'Transcript + translation', summary: 'Summary only'}[o.what];
        return [what, o.what !== 'summary' && o.summary_on_top && 'summary on top', (o.speakers || []).length && 'only ' + o.speakers.join(', '),
                o.part === 'cuts' && 'clip cuts only', o.lines_with && `lines with “${o.lines_with}”`,
                !subs && o.what !== 'summary' && {line: 'times at every line', minute: 'a time each minute', none: 'no times'}[o.timestamps],
                o.what !== 'summary' && (o.names ? 'names' : 'no names'), !subs && o.paragraphs && 'paragraphs', !subs && o.unsure && 'unsure words marked',
                subs && `${o.sub_cpl} chars × ${o.sub_lines} lines, ≤ ${o.sub_max} s`].filter(Boolean).join(' · ');
    }

    function nameAndSize() {
        const c = count || {};
        const n = tab === 'one' ? 1 : (c.count || 0);
        const stem = (tab === 'one' && one ? one.name : (c.names && c.names.length === 1 ? c.names[0].split('/').pop() : '')).replace(/\.[^.]+$/, '');
        let name, size = 0;
        if (media()) {
            const o = fmt === 'video' ? video : audio;
            const ext = o.format === 'original' || o.format === 'track' ? '' : o.format;
            name = n === 1 && stem ? `${stem}${ext ? '.' + ext : ''}` : 'export.zip';
            if ((o.format === 'original' && fmt === 'video' && video.size === 'original' && !video.subtitles && !(video.speakers || []).length && !video.remove_pauses)) size = c.size || 0;
        } else {
            const ext = fmt;
            const oneDoc = tab === 'many' && opts.packing === 'one' && !['srt', 'vtt'].includes(fmt);
            name = n === 1 && stem ? `${stem}.${ext}` : oneDoc ? `transcripts.${ext}` : 'transcripts.zip';
            if (preview && preview.text) size = (preview.size || new Blob([preview.text]).size) * Math.max(1, c.transcribed || 1) * (fmt === 'docx' ? 0.5 : fmt === 'pdf' ? 1.3 : 1);
        }
        return {name, size, n};
    }

    function drawFoot() {
        const info = el('exCount'), go = el('exGo'), cp = el('exCopy');
        if (!info || !go) return;
        const c = count || {};
        if (c.error) {
            info.innerHTML = `<span class="ex-err">${esc(c.error)}</span>`;
            go.disabled = true;
            if (cp) cp.hidden = true;
            return;
        }
        if (c.too_many) {
            info.innerHTML = `<b>${plural(c.count, 'recording')}</b> · <span class="ex-err">more than ${(c.max || 0).toLocaleString()} at once — choose a folder further down, or fewer recordings</span>`;
            go.disabled = true;
            if (cp) cp.hidden = true;
            drawJobs();
            return;
        }
        const ns = nameAndSize();
        const parts = [];
        if (tab === 'many') {
            parts.push(`<b>${plural(c.count || 0, 'recording')}</b>`);
            if (!media() && c.count) parts.push(c.transcribed === c.count ? 'all transcribed' : `${(c.transcribed || 0).toLocaleString()} transcribed — the others are skipped`);
            if (fmt === 'video' && c.count) parts.push(c.videos === c.count ? 'all videos' : `${(c.videos || 0).toLocaleString()} videos — sound-only ones are skipped`);
        } else if (one && !media() && c.transcribed === 0) {
            parts.push('<span class="ex-err">not transcribed yet</span>');
        } else if (one && fmt === 'video' && c.videos === 0) {
            parts.push('<span class="ex-err">this recording has no video</span>');
        }
        const runsAsJob = media() || (c.count || 0) > 300;
        if (runsAsJob && (c.count || 0) > 0) parts.push('runs in the background — see the Queue');
        info.innerHTML = parts.join(' · ');
        const usable = tab === 'one' ? !!one && (media() ? (fmt !== 'video' || c.videos !== 0) : c.transcribed !== 0)
                                     : (c.count || 0) > 0 && (media() || c.transcribed > 0);
        go.disabled = !usable;
        go.innerHTML = `<svg class="ui-ico"><use href="#i-download"/></svg> ${runsAsJob ? 'Start export' : 'Download'} <span class="ex-gn">${esc(ns.name)}${ns.size ? ' · ' + (runsAsJob ? '' : '≈ ') + fmtBytes(ns.size) : ''}</span>`;
        if (cp) cp.hidden = !(ns.n === 1 && isText(fmt) && !['docx', 'pdf'].includes(fmt) && usable);
        drawJobs();
    }

    /* ── Changing things ──────────────────────────────────────────────── */

    function setFormat(f) {
        if (!GRID.some(g => g[0] === f)) return;
        fmt = f;
        remember();
        drawFormats(); drawSub(); drawOptions(); drawFoot(); refreshPreview();
    }

    function setOpt(k, v) {
        opts = {...opts, [k]: v};
        remember();
        drawOptions(); drawFoot(); refreshPreview();
    }

    function setMedia(k, v) {
        if (fmt === 'video') video = {...video, [k]: v};
        else audio = {...audio, [k]: v};
        remember();
        drawSub(); drawOptions(); drawFoot(); refreshPreview();
    }

    function toggleSpeaker(kind, name) {
        const o = kind === 'video' ? video : kind === 'audio' ? audio : opts;
        const s = new Set(o.speakers || []);
        s.has(name) ? s.delete(name) : s.add(name);
        if (kind === 'video') video = {...video, speakers: [...s]};
        else if (kind === 'audio') audio = {...audio, speakers: [...s]};
        else opts = {...opts, speakers: [...s]};
        drawOptions(); drawFoot(); refreshPreview();
    }

    function clearSpeakers(kind) {
        if (kind === 'video') video = {...video, speakers: []};
        else if (kind === 'audio') audio = {...audio, speakers: []};
        else opts = {...opts, speakers: []};
        drawOptions(); drawFoot(); refreshPreview();
    }

    function status(text, bad) {
        const s = el('exStatus');
        if (!s) return;
        s.textContent = text || '';
        s.classList.toggle('ex-err', !!bad);
        clearTimeout(status._t);
        if (text) status._t = setTimeout(() => { s.textContent = ''; }, 6000);
    }

    /* ── Running it ───────────────────────────────────────────────────── */

    function finalBody() {
        const b = body();
        if (tab === 'many' && whole && q) { delete b.folders; delete b.added; b.files = listed.map(f => f.abs); }
        return b;
    }

    async function run() {
        const go = el('exGo');
        if (!go || go.disabled) return;
        go.disabled = true;
        status('Exporting…');
        try {
            const r = await fetch('/api/export', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(finalBody())});
            if (r.status === 202) {
                const d = await r.json();
                jobs = [d.job, ...jobs.filter(j => j.id !== d.job.id)];
                status(`Started: ${d.job.label}. It runs in the background — the download appears here and in the Queue.`);
                pollJobs();
                drawJobs();
                return;
            }
            if (!r.ok) {
                const d = await r.json().catch(() => ({}));
                throw new Error(d.error || `HTTP ${r.status}`);
            }
            const blob = await r.blob();
            const name = dispositionName(r.headers.get('Content-Disposition')) || 'export';
            save(blob, name);
            const skipped = Number(r.headers.get('X-Export-Skipped') || 0);
            status(`Downloaded ${name} (${fmtBytes(blob.size)})${skipped ? ` — ${plural(skipped, 'recording')} without a transcript skipped` : ''}`);
        } catch (e) {
            status(e.message || String(e), true);
        } finally {
            drawFoot();
        }
    }

    function save(blob, name) {
        const a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        a.download = name;
        document.body.appendChild(a);
        a.click();
        a.remove();
        setTimeout(() => URL.revokeObjectURL(a.href), 30000);
    }

    async function copy() {
        try {
            const r = await fetch('/api/export', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(finalBody())});
            if (!r.ok) { const d = await r.json().catch(() => ({})); throw new Error(d.error || `HTTP ${r.status}`); }
            const text = await r.text();
            await navigator.clipboard.writeText(text);
            status(`Copied (${plural(text.length, 'character')})`);
        } catch (e) {
            status('Could not copy: ' + (e.message || e), true);
        }
    }

    /* ── Jobs ─────────────────────────────────────────────────────────── */

    async function pollJobs() {
        clearTimeout(jobTimer);
        try {
            const r = await fetch('/api/export/jobs');
            if (r.ok) jobs = (await r.json()).jobs || [];
        } catch (e) { /* next time */ }
        drawJobs();
        if (typeof QueuePage !== 'undefined' && QueuePage.visible) QueuePage.llmChanged();
        const active = jobs.some(j => j.status === 'waiting' || j.status === 'running');
        if (active) jobTimer = setTimeout(pollJobs, 1500);
    }

    function drawJobs() {
        const box = el('exJobs');
        if (!box) return;
        const recent = jobs.filter(j => j.status !== 'cancelled').slice(0, 4);
        box.hidden = !recent.length;
        box.innerHTML = recent.map(j => jobRow(j, 'ex')).join('');
    }

    /* One job as a row (here and in the Queue). */
    function jobRow(j, where) {
        const pct = Math.round(j.progress || 0);
        let meta = '', acts = '';
        if (j.status === 'running' || j.status === 'waiting') {
            meta = j.status === 'waiting' ? '<span class="qj-msg">waiting for a free slot</span>'
                : `<span class="qj-bar-mini"><span style="width:${pct}%"></span></span><span class="qj-n">${pct} %</span><span class="qj-msg">${esc(j.message || '')}</span>`;
            acts = `<button class="qj-act" onclick="Export.cancelJob('${escJs(j.id)}')" title="Stop it">✕</button>`;
        } else if (j.status === 'done') {
            meta = `<span class="qj-n">${esc(j.result)} · ${fmtBytes(j.size || 0)}</span>`
                + (j.skipped && j.skipped.length ? `<span class="qj-err" title="${esc(j.skipped.map(s => s.join(': ')).join('\n'))}">${plural(j.skipped.length, 'skipped')}</span>` : '');
            acts = `<a class="qj-act ex-dl" href="/api/export/jobs/${encodeURIComponent(j.id)}/download" download title="Download ${esc(j.result)}">⬇</a>`
                + `<button class="qj-act" onclick="Export.removeJob('${escJs(j.id)}')" title="Remove it and its file">🗑</button>`;
        } else {
            meta = `<span class="qj-err">${esc(j.error || j.status)}</span>`;
            acts = `<button class="qj-act" onclick="Export.removeJob('${escJs(j.id)}')" title="Remove">🗑</button>`;
        }
        const sec = {running: 'running', waiting: 'waiting', done: 'done'}[j.status] || 'failed';
        const clip = j.kind === 'clip';
        return `<div class="qj-row qj-${sec}${where === 'ex' ? ' ex-job' : ''}"><span class="qj-kind" title="${clip ? 'Clip' : 'Export'}">${clip ? '🎬' : '⬇'}</span>`
            + `<span class="qj-what">${clip ? 'Clip' : 'Export'}</span><span class="qj-name qj-plain">${esc(j.label.replace(/^(Export|Clip) · /, ''))}</span>`
            + meta + `<span class="qj-acts">${acts}</span></div>`;
    }

    async function cancelJob(id) {
        await fetch(`/api/export/jobs/${encodeURIComponent(id)}/cancel`, {method: 'POST'}).catch(() => {});
        pollJobs();
    }

    async function removeJob(id) {
        await fetch(`/api/export/jobs/${encodeURIComponent(id)}`, {method: 'DELETE'}).catch(() => {});
        pollJobs();
    }

    document.addEventListener('keydown', e => {
        if (e.key === 'Escape' && visible() && !e.defaultPrevented) {
            const t = e.target;
            if (t && t.closest && t.closest('#exportOverlay input, #exportOverlay select') && t.value) return;
            close();
        }
    });
    document.addEventListener('DOMContentLoaded', () => { setTimeout(pollJobs, 1500); });

    return {show, close, setTab, focusFolder, toggleNode, pick, wholeFolder, pickAll, pickNone, setFilter,
            setFormat, setOpt, setMedia, toggleSpeaker, clearSpeakers, applyPreset, savePreset, deletePreset,
            run, copy, cancelJob, removeJob, pollJobs, jobRow, summaryOf,
            get jobs() { return jobs; }, get tab() { return tab; }, get visible() { return visible(); }};
})();

/* For the shell's closing of windows */
function closeExport() { document.getElementById('exportOverlay')?.classList.remove('visible'); }
