// ── Clips: a small video editor in a recording's Clips tab ───────────────
// Several cuts on the waveform timeline (the clip editor of clip.js, with
// its zoom and IN/OUT points, is the timeline here), a list of them to
// edit, play, reorder or drop; cuts picked from the transcript's lines,
// 30 s from the playhead, or suggested by the language model (Best
// moments). Then the frame (16:9, vertical 9:16 or square 1:1 at a fixed
// crop position — no face tracking), the options (pauses out, subtitles
// burned in, soft fades, one clip or one each) and "Make the clip": a job
// in the Queue (api/clips.py), with the download when done.
//
// The cuts are kept on the server per recording (/api/clips), so they
// survive a reload and Export's "Only the clip cuts" uses them.

const Clips = (() => {
    const el = id => document.getElementById(id);
    let abs = '';                // the recording shown
    let st = null;               // {cuts, settings, made}
    let sel = -1;                // the cut in the timeline's IN/OUT
    let loadSeq = 0;
    let saveTimer = null;
    let picking = null;          // {a: null | seconds} while lines are picked in the transcript
    let bestId = 0, bestTimer = null;

    const fmt = s => {
        s = Math.max(0, Number(s) || 0);
        const h = Math.floor(s / 3600), m = Math.floor(s % 3600 / 60), x = s % 60;
        const sec = x.toFixed(1).padStart(4, '0');
        return h ? `${h}:${String(m).padStart(2, '0')}:${sec}` : `${m}:${sec}`;
    };
    const dur = s => { s = Math.round(Number(s) || 0); return s >= 60 ? `${Math.floor(s / 60)} min ${s % 60} s` : `${s} s`; };
    const parse = str => {
        const p = String(str || '').trim().split(':').map(Number);
        if (!p.length || p.some(n => !Number.isFinite(n))) return null;
        return p.reduce((acc, n) => acc * 60 + n, 0);
    };
    const length = () => (typeof activeTranscript !== 'undefined' && activeTranscript && activeTranscript.total_duration_sec) || 0;
    const isVideo = () => {
        const v = el('videoPlayer');
        return !!(v && v.videoWidth) || /\.(mp4|mkv|mov|webm|avi|m4v|ts|mts|wmv|flv|mpg|mpeg)$/i.test(abs);
    };
    function currentAbs() {
        if (typeof activeFileObj !== 'undefined' && activeFileObj && activeFileObj.abs_path) return activeFileObj.abs_path;
        return typeof filePath === 'function' && typeof activeFile !== 'undefined' && activeFile ? filePath({name: activeFile}) : '';
    }
    const pane = () => document.querySelector('#transcriptArea .rv-pane[data-pane="clips"]');

    /* ── Loading and saving ───────────────────────────────────────────── */

    async function load() {
        const a = currentAbs();
        if (!a) return;
        if (a !== abs) { abs = a; st = null; sel = -1; }
        const s = ++loadSeq;
        try {
            const r = await fetch('/api/clips?file=' + encodeURIComponent(a));
            const d = await r.json();
            if (s !== loadSeq || a !== abs) return;
            if (!r.ok) throw new Error(d.error || `HTTP ${r.status}`);
            st = d;
        } catch (e) {
            if (s === loadSeq) st = st || {cuts: [], settings: {format: 'orig', crop: 0.5, pauses: false, pause_min: 0.8, subs: false, fade: true, stitch: 'one'}, made: [], error: e.message};
        }
        draw();
    }

    function save(now) {
        clearTimeout(saveTimer);
        const a = abs, body = {file: a, cuts: st.cuts, settings: st.settings};
        const go = async () => {
            try {
                const r = await fetch('/api/clips', {method: 'PUT', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)});
                const d = await r.json();
                if (!r.ok) throw new Error(d.error || `HTTP ${r.status}`);
                if (a === abs && st) { st.cuts = d.cuts; st.made = d.made; }
            } catch (e) {
                if (typeof Toast !== 'undefined') Toast.error('The cuts could not be saved: ' + e.message);
            }
        };
        if (now) return go();
        saveTimer = setTimeout(go, 300);
    }

    function changed(redraw = true) {
        if (redraw) draw();
        save();
        if (typeof Rec !== 'undefined' && Rec.renderTabs) try { Rec.renderTabs(); } catch (e) { /* the badge waits */ }
    }

    /* ── Showing ──────────────────────────────────────────────────────── */

    /* The Clips tab was opened (recording.js). */
    function show() {
        const p = pane();
        if (!p) return;
        if (!p.querySelector('.cl-wrap')) p.innerHTML = skeleton();
        // The timeline: the clip editor, moved into its card
        const slot = el('clTimeline'), ed = el('clipEditor');
        if (slot && ed && ed.parentElement !== slot) slot.appendChild(ed);
        if (ed && !ed.classList.contains('visible') && typeof openClipEditor === 'function') openClipEditor();
        if (currentAbs() !== abs || !st) load(); else draw();
    }

    function skeleton() {
        return `<div class="cl-wrap">
            <section class="cl-card">
                <div class="cl-head"><b class="cl-h">Cuts</b><span class="cl-sum" id="clSum"></span><span class="cl-gap"></span>
                    <button class="tb-btn" onclick="Clips.pickLines()" title="Click the first and the last line of a part in the transcript">Pick lines in the transcript</button>
                    <button class="tb-btn" id="clAdd30" onclick="Clips.add30()" title="A cut of 30 seconds from where the player is">Add 30 s from the playhead</button>
                    <button class="tb-btn" id="clBest" onclick="Clips.best()" title="The language model suggests three short moments — keep or drop each">✨ Best moments (AI)</button>
                </div>
                <div id="clTimeline" class="cl-timeline"></div>
                <div class="cl-addrow"><button class="tb-btn" id="clTake" onclick="Clips.takeRange()"></button><span class="cl-hint" id="clTakeHint"></span></div>
                <div id="clList" class="cl-list"></div>
            </section>
            <div class="cl-cols">
                <section class="cl-card" id="clFrame"></section>
                <section class="cl-card" id="clOpts"></section>
            </div>
            <section class="cl-card" id="clMade" hidden></section>
        </div>`;
    }

    function draw() {
        const p = pane();
        if (!p || p.hidden || !st) return;
        if (!p.querySelector('.cl-wrap')) { show(); return; }
        const c = st.cuts, s = st.settings;
        const total = c.reduce((n, x) => n + x.end - x.start, 0);
        el('clSum').textContent = c.length ? `${c.length} ${c.length === 1 ? 'cut' : 'cuts'} · ${dur(total)} of ${fmt(length()).replace(/\.\d$/, '')}`
                                           : 'Mark the parts you want: in the transcript, on the timeline, or from the playhead';
        const v = el('videoPlayer');
        el('clAdd30').textContent = `Add 30 s from the playhead (${fmt(v ? v.currentTime : 0).replace(/\.\d$/, '')})`;
        const take = el('clTake');
        take.textContent = sel >= 0 && c[sel] ? `✓ Use IN–OUT for cut ${sel + 1}` : '+ Add IN–OUT as a cut';
        el('clTakeHint').textContent = sel >= 0 && c[sel] ? 'Cut ' + (sel + 1) + ' is on the timeline — move IN and OUT, then use them. ' : 'Set IN and OUT on the timeline (click, or ⟨ ⟩ beside a line), then add them. ';
        drawBoxes();
        drawList();
        drawFrame();
        drawOpts();
        drawMade();
        if (st.error) el('clSum').textContent = st.error;
    }

    /* The cuts as boxes on the timeline. */
    function drawBoxes() {
        const inner = el('clipTimelineInner'), L = length();
        if (!inner || !L) return;
        inner.querySelectorAll('.cl-box').forEach(b => b.remove());
        st.cuts.forEach((x, i) => {
            const b = document.createElement('div');
            b.className = 'cl-box' + (i === sel ? ' on' : '') + (x.ai ? ' ai' : '');
            b.style.left = (x.start / L * 100).toFixed(3) + '%';
            b.style.width = Math.max(0.15, (x.end - x.start) / L * 100).toFixed(3) + '%';
            b.title = `Cut ${i + 1}: ${fmt(x.start)} – ${fmt(x.end)}${x.title ? ' · ' + x.title : ''} (click to edit it)`;
            b.innerHTML = `<b>${i + 1}</b>`;
            b.addEventListener('mousedown', e => { if (e.button === 0) { e.stopPropagation(); e.preventDefault(); edit(i); } });
            inner.appendChild(b);
        });
    }

    function firstWords(x) {
        const segs = (typeof activeTranscript !== 'undefined' && activeTranscript && activeTranscript.segments) || [];
        const s = segs.find(g => g.end > x.start + 0.05 && g.start < x.end);
        return s ? (s.text || '').trim().slice(0, 90) : '';
    }

    function drawList() {
        const box = el('clList');
        if (!st.cuts.length) { box.innerHTML = ''; return; }
        box.innerHTML = st.cuts.map((x, i) => `<div class="cl-cut${i === sel ? ' on' : ''}">
            <b class="cl-n">${i + 1}</b>
            <input class="cl-t" value="${fmt(x.start)}" aria-label="Start of cut ${i + 1}" onchange="Clips.setTime(${i}, 'start', this.value)">
            <span class="cl-dash">–</span>
            <input class="cl-t" value="${fmt(x.end)}" aria-label="End of cut ${i + 1}" onchange="Clips.setTime(${i}, 'end', this.value)">
            <span class="cl-len">${dur(x.end - x.start)}</span>
            <span class="cl-words">${x.ai ? '<span class="cl-ai" title="Suggested by the language model">AI</span> ' : ''}${x.title ? `<b>${esc(x.title)}</b> · ` : ''}${esc(firstWords(x))}</span>
            <span class="cl-acts">
                <button class="cl-act" onclick="Clips.play(${i})" title="Play this cut" aria-label="Play cut ${i + 1}">▶</button>
                <button class="cl-act" onclick="Clips.edit(${i})" title="Show it on the timeline to move IN and OUT" aria-label="Edit cut ${i + 1}">✎</button>
                <button class="cl-act" onclick="Clips.move(${i}, -1)" ${i === 0 ? 'disabled' : ''} title="Earlier in the clip" aria-label="Move cut ${i + 1} up">↑</button>
                <button class="cl-act" onclick="Clips.move(${i}, 1)" ${i === st.cuts.length - 1 ? 'disabled' : ''} title="Later in the clip" aria-label="Move cut ${i + 1} down">↓</button>
                <button class="cl-act cl-del" onclick="Clips.drop(${i})" title="Drop this cut" aria-label="Drop cut ${i + 1}">✕</button>
            </span></div>`).join('');
    }

    function drawFrame() {
        const box = el('clFrame'), s = st.settings;
        const video = isVideo();
        const R = {orig: [16, 9], vert: [9, 16], square: [1, 1]}[s.format];
        const seg = (v, t) => `<button class="ex-chip${s.format === v ? ' on' : ''}" onclick="Clips.set('format', '${v}')" ${video ? '' : 'disabled'}>${t}</button>`;
        let frame = '';
        if (video) {
            const sub = s.subs ? `<b class="cl-subline">${esc((firstWords(st.cuts[0] || {start: 0, end: 1e9}) || 'Subtitles appear here').slice(0, 44))}</b>` : '';
            if (s.format === 'orig') frame = `<div class="cl-frame">${sub}</div>`;
            else {
                const w = R[0] / R[1] * 9 / 16 * 100;
                frame = `<div class="cl-frame"><i class="cl-crop" style="width:${w.toFixed(1)}%;left:${(s.crop * (100 - w)).toFixed(1)}%"><span>${s.format === 'vert' ? '1080 × 1920' : '1080 × 1080'}</span>${sub}</i></div>
                    <label class="cl-row"><span class="cl-k">Frame position</span><input type="range" min="0" max="100" value="${Math.round(s.crop * 100)}" style="flex:1"
                        oninput="Clips.crop(this.value)" onchange="Clips.crop(this.value, true)" aria-label="Frame position"></label>`;
            }
        }
        box.innerHTML = `<div class="cl-head"><b class="cl-h">Format</b></div>
            <div class="cl-row">${seg('orig', 'Original 16:9')}${seg('vert', 'Vertical 9:16')}${seg('square', 'Square 1:1')}</div>
            ${video ? frame : '<div class="cl-hint">This recording is sound only — its clip is sound too (M4A).</div>'}`;
        const fr = box.querySelector('.cl-frame');
        const v = el('videoPlayer');
        if (fr && v && v.videoWidth) fr.style.aspectRatio = `${v.videoWidth} / ${v.videoHeight}`;
    }

    function outLength() {
        return st.cuts.reduce((n, x) => n + x.end - x.start, 0);
    }

    function drawOpts() {
        const box = el('clOpts'), s = st.settings, c = st.cuts, video = isVideo();
        const ck = (k, label, title) => `<label class="ex-ck"${title ? ` title="${esc(title)}"` : ''}><input type="checkbox" ${s[k] ? 'checked' : ''} onchange="Clips.set('${k}', this.checked)"> ${label}</label>`;
        const n = s.stitch === 'one' || c.length < 2 ? 1 : c.length;
        const size = !video ? 'sound only' : s.format === 'vert' ? '1080 × 1920' : s.format === 'square' ? '1080 × 1080' : 'as recorded';
        box.innerHTML = `<div class="cl-head"><b class="cl-h">Options</b></div>
            <div class="cl-row">${ck('pauses', 'Remove pauses longer than', 'Silence inside the cuts is left out; a little air stays')}
                <select onchange="Clips.set('pause_min', Number(this.value))" aria-label="Shortest pause removed">${[0.5, 0.8, 1.5, 3].map(v => `<option value="${v}"${s.pause_min === v ? ' selected' : ''}>${v} s</option>`).join('')}</select></div>
            <div class="cl-row">${video ? ck('subs', 'Burn in subtitles', "The transcript's lines in the picture") : '<span class="cl-hint">No subtitles: sound only</span>'}</div>
            <div class="cl-row">${ck('fade', 'Soft fade in and out of each cut')}</div>
            <div class="cl-row"><span class="cl-k">Cuts become</span>
                <button class="ex-chip${s.stitch === 'one' ? ' on' : ''}" onclick="Clips.set('stitch', 'one')">One clip</button>
                <button class="ex-chip${s.stitch === 'each' ? ' on' : ''}" onclick="Clips.set('stitch', 'each')">One clip each</button></div>
            <div class="cl-sumline">${c.length ? `<b>${n === 1 ? '1 clip' : n + ' clips'}</b> · ${dur(outLength())}${s.pauses ? ' at most (pauses out)' : ''} · ${size}` : '<span class="cl-hint">No cuts yet</span>'}</div>
            <div class="cl-row">
                <button class="ex-go" id="clMake" onclick="Clips.make()" ${c.length ? '' : 'disabled'}>🎬 Make the clip</button>
                <button class="tb-btn" onclick="Clips.dataset()" title="Voices → Extract: the speech of chosen people as audio files for a voice dataset">Audio for the voice dataset</button>
            </div>`;
    }

    function drawMade() {
        const box = el('clMade');
        const made = st.made || [];
        box.hidden = !made.length;
        if (!made.length) return;
        const jobs = typeof Export !== 'undefined' ? Export.jobs : [];
        box.innerHTML = `<div class="cl-head"><b class="cl-h">Made from this recording</b></div>` + made.map(m => {
            const j = jobs.find(x => x.id === m.job);
            const fmtName = m.format === 'vert' ? 'vertical' : m.format === 'square' ? 'square' : m.format === 'audio' ? 'sound' : '16:9';
            const dl = j && j.status === 'done' ? `<a class="tb-btn" href="/api/export/jobs/${encodeURIComponent(j.id)}/download" download>⬇ ${esc(j.result)} · ${fmtBytes(j.size || 0)}</a>`
                : '<span class="cl-hint">the download is kept for a day</span>';
            return `<div class="cl-made">🎬 ${m.n === 1 ? '1 clip' : m.n + ' clips'} · ${dur(m.len)} · ${fmtName} <span class="cl-hint">· ${esc(new Date((m.at || 0) * 1000).toLocaleString())}</span> ${dl}</div>`;
        }).join('');
    }

    /* ── Changing the cuts ────────────────────────────────────────────── */

    function addCut(a, b, extra) {
        const L = length();
        a = Math.max(0, a); b = L ? Math.min(L, b) : b;
        if (b - a < 0.2) { if (typeof Toast !== 'undefined') Toast.warn('A cut is at least 0.2 seconds long'); return false; }
        st.cuts.push({start: +a.toFixed(3), end: +b.toFixed(3), ...(extra || {})});
        sel = -1;
        changed();
        return true;
    }

    function takeRange() {
        if (!st || typeof clipIn === 'undefined') return;
        if (sel >= 0 && st.cuts[sel]) {
            st.cuts[sel] = {...st.cuts[sel], start: +Math.min(clipIn, clipOut).toFixed(3), end: +Math.max(clipIn, clipOut).toFixed(3)};
            sel = -1;
            changed();
        } else addCut(Math.min(clipIn, clipOut), Math.max(clipIn, clipOut));
    }

    function add30() {
        const v = el('videoPlayer');
        const t = v ? v.currentTime : 0;
        addCut(t, t + 30);
    }

    function setTime(i, k, value) {
        const x = st.cuts[i], v = parse(value);
        if (!x || v === null) { draw(); return; }
        x[k] = Math.max(0, Math.min(length() || v, v));
        if (x.end < x.start) [x.start, x.end] = [x.end, x.start];
        if (x.end - x.start < 0.2) x.end = x.start + 0.2;
        changed();
    }

    function edit(i) {
        const x = st.cuts[i];
        if (!x) return;
        sel = i;
        if (typeof setClipPoint === 'function') { setClipPoint('in', x.start); setClipPoint('out', x.end); }
        if (typeof clipZoomToRegion === 'function') clipZoomToRegion();
        draw();
    }

    function play(i) {
        const x = st.cuts[i];
        if (!x || typeof setClipPoint !== 'function') return;
        sel = i;
        setClipPoint('in', x.start); setClipPoint('out', x.end);
        if (typeof previewClip === 'function') previewClip();
        draw();
    }

    function move(i, d) {
        const j = i + d;
        if (j < 0 || j >= st.cuts.length) return;
        [st.cuts[i], st.cuts[j]] = [st.cuts[j], st.cuts[i]];
        if (sel === i) sel = j; else if (sel === j) sel = i;
        changed();
    }

    function drop(i) {
        st.cuts.splice(i, 1);
        if (sel === i) sel = -1; else if (sel > i) sel--;
        changed();
    }

    function set(k, v) {
        st.settings = {...st.settings, [k]: v};
        changed();
    }

    function crop(v, done) {
        st.settings.crop = Math.max(0, Math.min(1, Number(v) / 100));
        const i = el('clFrame')?.querySelector('.cl-crop');
        const R = {vert: [9, 16], square: [1, 1]}[st.settings.format];
        if (i && R) { const w = R[0] / R[1] * 9 / 16 * 100; i.style.left = (st.settings.crop * (100 - w)).toFixed(1) + '%'; }
        if (done) save();
    }

    /* ── Lines picked in the transcript ───────────────────────────────── */

    function pickLines() {
        picking = {a: null};
        if (typeof Rec !== 'undefined') Rec.tab('transcript');
        banner();
    }

    function banner() {
        const area = el('transcriptArea');
        let b = el('clPickBar');
        if (!picking) { b?.remove(); return; }
        if (!b && area) {
            b = document.createElement('div');
            b.id = 'clPickBar';
            b.className = 'cl-pickbar';
            area.prepend(b);
        }
        if (!b) return;
        const n = st ? st.cuts.length : 0;
        b.innerHTML = picking.a === null
            ? `✂ Click the line the cut <b>starts</b> with${picking.added ? ` · cut ${n} added` : ''}<span class="cl-gap"></span><button class="tb-btn" onclick="Clips.backToClips()">${picking.added ? 'Back to Clips' : 'Cancel'}</button>`
            : `✂ Starts at ${fmt(picking.a)} — now click the line it <b>ends</b> with<span class="cl-gap"></span><button class="tb-btn" onclick="Clips.backToClips()">Cancel</button>`;
    }

    function backToClips() {
        picking = null;
        banner();
        if (typeof Rec !== 'undefined') Rec.tab('clips');
    }

    // A line clicked while picking: its start, then its end (capture phase,
    // before the line's own click seeks the player)
    document.addEventListener('click', e => {
        if (!picking) return;
        const seg = e.target.closest && e.target.closest('#transcriptArea .seg[data-start]');
        if (!seg || e.target.closest('button, input, textarea, a')) return;
        const a = Number(seg.dataset.start), b = Number(seg.dataset.end);
        if (!Number.isFinite(a)) return;
        if (picking.a === null) {
            picking.a = a;
        } else {
            const start = Math.min(picking.a, a), end = Math.max(b, picking.a + 0.2);
            if (st && addCut(start, end)) picking = {a: null, added: true};
            else picking.a = null;
        }
        banner();
    }, true);

    /* ── Best moments, making the clip ────────────────────────────────── */

    async function best() {
        const btn = el('clBest');
        if (bestId) return;
        try {
            const r = await fetch('/api/clips/best', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({file: abs})});
            const d = await r.json();
            if (!r.ok) throw new Error(d.error || `HTTP ${r.status}`);
            bestId = d.id;
            if (btn) { btn.disabled = true; btn.textContent = '✨ Looking for moments…'; }
            if (typeof llmQueueRefreshBadge === 'function') llmQueueRefreshBadge();
            pollBest(abs);
        } catch (e) {
            if (typeof Toast !== 'undefined') Toast.error('Best moments: ' + e.message);
        }
    }

    async function pollBest(forAbs) {
        clearTimeout(bestTimer);
        try {
            const r = await fetch('/api/llm/queue/item/' + bestId);
            const d = await r.json();
            if (d.status === 'pending' || d.status === 'running') {
                const btn = el('clBest');
                if (btn && forAbs === abs) btn.textContent = d.status === 'pending' ? '✨ Waiting in the AI queue…' : `✨ Looking… ${Math.round(d.progress || 0)} %`;
                bestTimer = setTimeout(() => pollBest(forAbs), 1500);
                return;
            }
            bestId = 0;
            if (d.status === 'done') {
                const n = ((d.result || {}).moments || []).length;
                if (typeof Toast !== 'undefined') Toast.success(`${n} ${n === 1 ? 'moment' : 'moments'} suggested — keep or drop each`);
                if (forAbs === abs) await load();
            } else if (typeof Toast !== 'undefined') {
                Toast.error('Best moments: ' + (d.error || d.status));
            }
        } catch (e) {
            bestId = 0;
        }
        const btn = el('clBest');
        if (btn) { btn.disabled = false; btn.textContent = '✨ Best moments (AI)'; }
    }

    async function make() {
        const btn = el('clMake');
        if (!st || !st.cuts.length) return;
        if (btn) btn.disabled = true;
        try {
            await save(true);
            const r = await fetch('/api/clips/render', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({file: abs})});
            const d = await r.json();
            if (!r.ok) throw new Error(d.error || `HTTP ${r.status}`);
            if (typeof Export !== 'undefined') Export.pollJobs();
            if (typeof Toast !== 'undefined') Toast.success('Clip started — it runs in the background; the download is in the Queue and here', 6000,
                () => { if (typeof QueuePage !== 'undefined') QueuePage.open(); });
            followJob(d.job.id);
        } catch (e) {
            if (typeof Toast !== 'undefined') Toast.error('The clip could not be started: ' + e.message);
        } finally {
            if (btn) btn.disabled = false;
        }
    }

    /* Until the job is done: then the made list shows its download. */
    function followJob(id) {
        const a = abs;
        const tick = async () => {
            if (typeof Export === 'undefined') return;
            await Export.pollJobs();
            const j = Export.jobs.find(x => x.id === id);
            if (j && (j.status === 'waiting' || j.status === 'running')) { setTimeout(tick, 1500); return; }
            if (a === abs) load();
            if (j && j.status === 'error' && typeof Toast !== 'undefined') Toast.error('The clip failed: ' + (j.error || ''));
        };
        setTimeout(tick, 1200);
    }

    function dataset() {
        if (typeof Shell !== 'undefined') Shell.go('voices');
        if (typeof vxTab === 'function') vxTab('extract');
    }

    /* The open recording changed: forget the old one's cuts. */
    function fileChanged() {
        picking = null;
        banner();
        if (currentAbs() !== abs) { abs = ''; st = null; sel = -1; }
    }

    const count = () => (st && abs === currentAbs() ? st.cuts.length : 0);

    return {show, load, draw, takeRange, add30, setTime, edit, play, move, drop, set, crop, pickLines, backToClips,
            best, make, dataset, fileChanged, count, get state() { return st; }};
})();
