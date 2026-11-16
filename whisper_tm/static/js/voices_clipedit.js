/* ══════════════════════════════════════════════════════════════════════════
 *  CLIP EDITOR — trim an extracted clip by hand, in memory
 *
 *  Sometimes an extracted clip still carries half a word at its edge; this
 *  editor lets it be adjusted by hand before the clips are downloaded as a
 *  ZIP, much like the clip generator.
 *
 *  The clip and a few seconds around it are fetched ONCE as WAV and
 *  decoded into the browser's memory (Web Audio). From then on drawing,
 *  playing any part of it and moving the bounds cost nothing — no request
 *  per listen, which is what makes going through dozens of clips bearable.
 *
 *  What makes half words findable:
 *
 *  * the waveform, with the stretches quiet enough to be a PAUSE shaded —
 *    the same rule the export's automatic cut uses (core/voice_extract.py,
 *    `snap_bounds`): near the recording's own floor, measured in 20 ms
 *    frames; dragging a bound near one snaps to it (Alt: no snap);
 *  * listening to just the START or just the END of the clip — that is
 *    where the half words are;
 *  * "Cut at pauses": both bounds to the quietest moment within a quarter
 *    second, where that is a pause.
 *
 *  Saving sends the bounds, the text and the state (keep / drop) to the
 *  server, which keeps them for the person: the next preview — after a
 *  filter changed — has them too, and the export cuts the clip as edited.
 * ══════════════════════════════════════════════════════════════════════════ */

const CLIPEDIT = {
    i: -1,                 // the clip being edited (index in the run)
    clip: null,
    ctx: null,             // one AudioContext for the page
    buf: null,             // the clip with context, decoded
    from: 0,               // where the buffer starts in the recording
    a: 0, b: 0,            // the bounds being edited (seconds in the recording)
    energy: null, quiet: 0, pauses: [], peak: 1,
    src: null, playAt: 0, playFrom: 0, playTo: 0, raf: 0,
    drag: null,
    dirty: false,
};

const CLIPEDIT_FRAME = 0.02;     // as SNAP_FRAME in core/voice_extract.py
const CLIPEDIT_DIP = 0.25;       // as SNAP_DIP
const CLIPEDIT_FLOOR_MAX = 0.6;  // as SNAP_FLOOR_MAX
const CLIPEDIT_REACH = 0.25;     // "cut at pauses" looks this far either way
const CLIPEDIT_PAD = 2.5;        // seconds of context around the clip

function clipEditAudio() {
    if (!CLIPEDIT.ctx) {
        const AC = window.AudioContext || window.webkitAudioContext;
        CLIPEDIT.ctx = new AC();
    }
    return CLIPEDIT.ctx;
}

async function clipEditOpen(i) {
    if (CLIPEDIT.i === i) { clipEditClose(); return; }
    clipEditClose();
    const run = EXTRACT.run;
    const c = (EXTRACT.page || []).find(x => x.i === i);
    const box = document.getElementById(`xEdit-${i}`);
    if (!run || !c || !box) return;
    Object.assign(CLIPEDIT, {i, clip: c, a: c.start, b: c.end, buf: null, dirty: false,
                             energy: null, pauses: []});
    box.hidden = false;
    box.innerHTML = '<div class="map-small">Loading the clip and its surroundings …</div>';
    document.getElementById(`xClip-${i}`)?.classList.add('editing');
    let data, from;
    try {
        const r = await fetch(`/api/voices/extract/context/${encodeURIComponent(run.run_id)}/${i}?pad=${CLIPEDIT_PAD}`);
        if (!r.ok) throw new Error((await r.json().catch(() => ({}))).error || r.status);
        from = Number(r.headers.get('X-Clip-From')) || 0;
        data = await r.arrayBuffer();
        CLIPEDIT.buf = await clipEditAudio().decodeAudioData(data);
    } catch (e) {
        if (CLIPEDIT.i === i) box.innerHTML = `<div class="vx-note err">Could not load the clip: ${esc(e.message || String(e))}</div>`;
        return;
    }
    if (CLIPEDIT.i !== i) return;          // another clip was opened meanwhile
    CLIPEDIT.from = from;
    clipEditAnalyse();
    box.innerHTML = clipEditHtml(c);
    clipEditWire();
    clipEditDraw();
    clipEditLabels();
}

function clipEditClose() {
    clipEditStop();
    if (CLIPEDIT.i >= 0) {
        const box = document.getElementById(`xEdit-${CLIPEDIT.i}`);
        if (box) { box.hidden = true; box.innerHTML = ''; }
        document.getElementById(`xClip-${CLIPEDIT.i}`)?.classList.remove('editing');
    }
    CLIPEDIT.i = -1;
    CLIPEDIT.clip = null;
    CLIPEDIT.buf = null;
}

/* Loudness per 20 ms frame, and the stretches quiet enough to cut in. */
function clipEditAnalyse() {
    const buf = CLIPEDIT.buf, c = CLIPEDIT.clip;
    const x = buf.getChannelData(0), rate = buf.sampleRate;
    const n = Math.max(1, Math.round(rate * CLIPEDIT_FRAME));
    const frames = Math.floor(x.length / n);
    const e = new Float32Array(frames);
    for (let f = 0; f < frames; f++) {
        let s = 0;
        for (let k = f * n; k < (f + 1) * n; k++) s += x[k] * x[k];
        e[f] = Math.sqrt(s / n);
    }
    // A pause sits near the recording's own floor, a quarter of the way up
    // to the voice — the export's rule (core/voice_extract.py, SNAP_DIP):
    // a fixed share of the voice found none in a stream whose background
    // noise sits at 40 % of it.
    const i0 = Math.max(0, Math.floor((c.s_start - CLIPEDIT.from) / CLIPEDIT_FRAME));
    const i1 = Math.min(frames, Math.floor((c.s_end - CLIPEDIT.from) / CLIPEDIT_FRAME));
    const inside = Array.from(e.slice(i0, Math.max(i0 + 1, i1))).sort((p, q) => p - q);
    const voice = inside.length ? inside[Math.floor(inside.length / 2)] : 0;
    const all = Array.from(e).sort((p, q) => p - q);
    const floor = all.length ? all[Math.floor(all.length * 0.1)] : 0;
    CLIPEDIT.energy = e;
    // The drawing is scaled to the loudest moment: a quiet stream would
    // otherwise be a flat line in which no word can be told apart.
    let peak = 0;
    for (let k = 0; k < x.length; k++) { const v = Math.abs(x[k]); if (v > peak) peak = v; }
    CLIPEDIT.peak = Math.max(peak, 1e-3);
    CLIPEDIT.quiet = voice <= 0 || floor > CLIPEDIT_FLOOR_MAX * voice
        ? -1                                   // floor and voice don't part: no pauses
        : Math.max(floor + CLIPEDIT_DIP * (voice - floor), 1e-4);
    const pauses = [];
    let start = -1;
    for (let f = 0; f <= frames; f++) {
        const q = f < frames && e[f] <= CLIPEDIT.quiet;
        if (q && start < 0) start = f;
        if (!q && start >= 0) {
            pauses.push([CLIPEDIT.from + start * CLIPEDIT_FRAME, CLIPEDIT.from + f * CLIPEDIT_FRAME]);
            start = -1;
        }
    }
    CLIPEDIT.pauses = pauses;
}

/* The quietest moment within `reach` of t, if it is a pause — else null. */
function clipEditQuietest(t, reach = CLIPEDIT_REACH) {
    const e = CLIPEDIT.energy;
    if (!e) return null;
    const f0 = Math.max(0, Math.floor((t - reach - CLIPEDIT.from) / CLIPEDIT_FRAME));
    const f1 = Math.min(e.length - 1, Math.floor((t + reach - CLIPEDIT.from) / CLIPEDIT_FRAME));
    let best = -1;
    for (let f = f0; f <= f1; f++) if (best < 0 || e[f] < e[best]) best = f;
    if (best < 0 || e[best] > CLIPEDIT.quiet) return null;
    return CLIPEDIT.from + (best + 0.5) * CLIPEDIT_FRAME;
}

function clipEditHtml(c) {
    return `
        <canvas class="vx-wave" id="xWave" height="130" title="Drag the bounds · click to listen from there · Alt while dragging: no snapping"></canvas>
        <div class="vx-edit-row">
            <span class="vx-edit-bound">⟨ <b id="xEditA"></b>
                <button class="map-mini" onclick="clipEditNudge('a', -0.05)" title="Start 50 ms earlier (←)">−</button>
                <button class="map-mini" onclick="clipEditNudge('a', 0.05)" title="Start 50 ms later (→)">+</button></span>
            <span class="vx-edit-len" id="xEditLen"></span>
            <span class="vx-edit-bound"><button class="map-mini" onclick="clipEditNudge('b', -0.05)" title="End 50 ms earlier (Shift+←)">−</button>
                <button class="map-mini" onclick="clipEditNudge('b', 0.05)" title="End 50 ms later (Shift+→)">+</button>
                <b id="xEditB"></b> ⟩</span>
        </div>
        <div class="vx-edit-row">
            <button class="vx-btn" onclick="clipEditPlay('clip')" title="The clip (Space)">▶ Clip</button>
            <button class="vx-btn" onclick="clipEditPlay('start')" title="Its first second and a half — where half words hide">▶ Start</button>
            <button class="vx-btn" onclick="clipEditPlay('end')" title="Its last second and a half">▶ End</button>
            <button class="vx-btn" onclick="clipEditPlay('context')" title="With a second and a half around it">▶ With context</button>
            <button class="vx-btn" onclick="clipEditStop()">■</button>
            <span class="vx-gap"></span>
            <button class="vx-btn" onclick="clipEditSnap()" title="Both bounds to the quietest moment within a quarter second, where that is a pause (S)">Cut at pauses</button>
            <button class="vx-btn" onclick="clipEditResetBounds()" title="The bounds as the sentence gave them">Original bounds</button>
        </div>
        ${c.words && c.words.length ? `<div class="vx-edit-row vx-edit-words" id="xEditWords"
            title="How sure the alignment is of each word — the one being said lights up while it plays">${extractTextHtml(c)}</div>` : ''}
        <div class="vx-edit-row">
            <input type="text" id="xEditText" value="${esc(c.text)}" title="The text that goes with the clip into the dataset"
                   oninput="CLIPEDIT.dirty=true" style="flex:1">
        </div>
        <div class="vx-edit-row">
            <button class="vx-btn primary" onclick="clipEditSave('kept', true)" title="Keep it as it is now and go to the next clip (Enter)">✓ Keep &amp; next</button>
            <button class="vx-btn danger" onclick="clipEditSave('dropped', true)" title="Leave it out of the export and go to the next clip (Del)">✕ Drop &amp; next</button>
            <button class="vx-btn" onclick="clipEditSave(null, false)" title="Save the bounds and the text, keep the editor open">Save</button>
            <a class="vx-btn" id="xEditWav" title="Download this clip as WAV — as saved">⤓ WAV</a>
            <span class="vx-gap"></span>
            <span class="map-small"><kbd>Space</kbd> play · <kbd>←</kbd><kbd>→</kbd> start · <kbd>Shift</kbd>+<kbd>←</kbd><kbd>→</kbd> end · <kbd>S</kbd> cut at pauses · <kbd>Enter</kbd> keep · <kbd>Del</kbd> drop · <kbd>Esc</kbd> close</span>
            <button class="vx-btn" onclick="clipEditClose()">Close</button>
        </div>
        <div class="map-small" id="xEditMsg"></div>`;
}

function clipEditLabels() {
    const f = t => mapClockTime(t) + '.' + String(Math.round((t % 1) * 100)).padStart(2, '0');
    const set = (id, v) => { const e = document.getElementById(id); if (e) e.textContent = v; };
    set('xEditA', f(CLIPEDIT.a));
    set('xEditB', f(CLIPEDIT.b));
    set('xEditLen', `${(CLIPEDIT.b - CLIPEDIT.a).toFixed(2)} s`
        + (CLIPEDIT.dirty ? ' · not saved' : ''));
    const w = document.getElementById('xEditWav');
    if (w && EXTRACT.run) w.href = `/api/voices/extract/clip/${encodeURIComponent(EXTRACT.run.run_id)}/${CLIPEDIT.i}?format=wav`;
}

// ── Drawing ──────────────────────────────────────────────────────────────

function clipEditX(t, w) {
    return (t - CLIPEDIT.from) / CLIPEDIT.buf.duration * w;
}

function clipEditT(x, w) {
    return CLIPEDIT.from + x / w * CLIPEDIT.buf.duration;
}

function clipEditDraw(playhead = null) {
    const cv = document.getElementById('xWave');
    if (!cv || !CLIPEDIT.buf) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = cv.clientWidth, h = cv.clientHeight;
    // Hidden (another tab is on screen): no size to draw into — sizing the
    // canvas to 0 here would leave it blank when the tab comes back
    if (!w || !h) return;
    if (cv.width !== Math.round(w * dpr)) { cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr); }
    const g = cv.getContext('2d');
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, w, h);
    const css = getComputedStyle(document.documentElement);
    const accent = css.getPropertyValue('--accent').trim() || '#3b82f6';
    const ax = clipEditX(CLIPEDIT.a, w), bx = clipEditX(CLIPEDIT.b, w);
    // the clip
    g.fillStyle = 'rgba(59,130,246,0.12)';
    g.fillRect(ax, 0, bx - ax, h);
    // pauses: where a cut splits no word
    g.fillStyle = 'rgba(34,197,94,0.16)';
    for (const [p0, p1] of CLIPEDIT.pauses) {
        g.fillRect(clipEditX(p0, w), h - 10, Math.max(1, clipEditX(p1, w) - clipEditX(p0, w)), 10);
    }
    // the waveform, min/max per pixel
    const x = CLIPEDIT.buf.getChannelData(0);
    const per = x.length / w, mid = (h - 12) / 2, gain = 0.95 / CLIPEDIT.peak;
    for (let px = 0; px < w; px++) {
        let lo = 1, hi = -1;
        const k0 = Math.floor(px * per), k1 = Math.min(x.length, Math.floor((px + 1) * per));
        for (let k = k0; k < k1; k++) { const v = x[k]; if (v < lo) lo = v; if (v > hi) hi = v; }
        if (hi < lo) continue;
        g.fillStyle = px >= ax && px <= bx ? accent : '#5b6478';
        g.fillRect(px, mid - hi * mid * gain, 1, Math.max(1, (hi - lo) * mid * gain));
    }
    // the sentence's own bounds, for reference
    g.strokeStyle = 'rgba(255,255,255,0.35)';
    g.setLineDash([3, 4]);
    for (const t of [CLIPEDIT.clip.s_start, CLIPEDIT.clip.s_end]) {
        const sx = clipEditX(t, w) + 0.5;
        g.beginPath(); g.moveTo(sx, 0); g.lineTo(sx, h - 12); g.stroke();
    }
    g.setLineDash([]);
    // the bounds, with a tab to grab
    g.fillStyle = '#ffd24a';
    for (const bxx of [ax, bx]) {
        g.fillRect(bxx - 1, 0, 2, h);
        g.fillRect(bxx - 5, 0, 10, 12);
    }
    if (playhead !== null) {
        g.fillStyle = '#fff';
        g.fillRect(clipEditX(playhead, w) - 0.5, 0, 1.5, h);
    }
}

// ── Listening ────────────────────────────────────────────────────────────

function clipEditPlay(kind) {
    if (!CLIPEDIT.buf) return;
    clipEditStop();
    const span = 1.5;
    let t0 = CLIPEDIT.a, t1 = CLIPEDIT.b;
    if (kind === 'start') t1 = Math.min(CLIPEDIT.b, CLIPEDIT.a + span);
    else if (kind === 'end') t0 = Math.max(CLIPEDIT.a, CLIPEDIT.b - span);
    else if (kind === 'context') { t0 = CLIPEDIT.a - span; t1 = CLIPEDIT.b + span; }
    else if (typeof kind === 'number') { t0 = kind; t1 = Math.max(kind + 0.5, CLIPEDIT.b); }
    const lo = Math.max(0, t0 - CLIPEDIT.from), hi = Math.min(CLIPEDIT.buf.duration, t1 - CLIPEDIT.from);
    if (hi <= lo) return;
    const ctx = clipEditAudio();
    if (ctx.state === 'suspended') ctx.resume();
    // The other players of the workspace stop — one voice at a time
    document.querySelectorAll('#mapOverlay audio').forEach(a => { if (!a.paused) a.pause(); });
    const src = ctx.createBufferSource();
    src.buffer = CLIPEDIT.buf;
    src.connect(ctx.destination);
    src.start(0, lo, hi - lo);
    Object.assign(CLIPEDIT, {src, playAt: ctx.currentTime, playFrom: CLIPEDIT.from + lo,
                             playTo: CLIPEDIT.from + hi});
    src.onended = () => { if (CLIPEDIT.src === src) clipEditStop(); };
    const tick = () => {
        if (CLIPEDIT.src !== src) return;
        const t = CLIPEDIT.playFrom + (ctx.currentTime - CLIPEDIT.playAt);
        clipEditDraw(t);
        if (CLIPEDIT.clip && CLIPEDIT.clip.words) {
            extractWordsAt(document.getElementById('xEditWords'), CLIPEDIT.clip.words, t);
        }
        CLIPEDIT.raf = requestAnimationFrame(tick);
    };
    tick();
}

function clipEditStop() {
    if (CLIPEDIT.src) {
        const s = CLIPEDIT.src;
        CLIPEDIT.src = null;
        try { s.stop(); } catch (e) { /* already ended */ }
    }
    cancelAnimationFrame(CLIPEDIT.raf);
    clipEditDraw();
    if (CLIPEDIT.clip && CLIPEDIT.clip.words) {
        extractWordsAt(document.getElementById('xEditWords'), CLIPEDIT.clip.words, null);
    }
}

// ── Moving the bounds ────────────────────────────────────────────────────

function clipEditSet(which, t, snap = false) {
    const lo = CLIPEDIT.from, hi = CLIPEDIT.from + (CLIPEDIT.buf ? CLIPEDIT.buf.duration : 0);
    t = Math.min(hi, Math.max(lo, t));
    if (snap) {
        // magnetic: a pause within 40 ms pulls the bound to its quietest point
        const q = clipEditQuietest(t, 0.04);
        if (q !== null) t = q;
    }
    if (which === 'a') CLIPEDIT.a = Math.min(t, CLIPEDIT.b - 0.2);
    else CLIPEDIT.b = Math.max(t, CLIPEDIT.a + 0.2);
    CLIPEDIT.dirty = true;
    clipEditDraw();
    clipEditLabels();
}

function clipEditNudge(which, d) {
    clipEditSet(which, (which === 'a' ? CLIPEDIT.a : CLIPEDIT.b) + d);
}

/* Both bounds to the quietest moment within a quarter second, where that
 * is a pause — the export's automatic cut, applied to the bounds as they
 * stand now. Says which bound found none. */
function clipEditSnap() {
    const a = clipEditQuietest(CLIPEDIT.a), b = clipEditQuietest(CLIPEDIT.b);
    if (a !== null && (b === null || b - a >= 0.2)) CLIPEDIT.a = a;
    if (b !== null && b - CLIPEDIT.a >= 0.2) CLIPEDIT.b = b;
    CLIPEDIT.dirty = true;
    clipEditDraw();
    clipEditLabels();
    const msg = document.getElementById('xEditMsg');
    if (msg) {
        msg.textContent = a === null && b === null ? 'No pause near either bound — the voice runs on there.'
            : (a === null ? 'No pause near the start.' : (b === null ? 'No pause near the end.' : 'Both bounds sit in a pause now.'));
    }
}

function clipEditResetBounds() {
    CLIPEDIT.a = CLIPEDIT.clip.orig_start;
    CLIPEDIT.b = CLIPEDIT.clip.orig_end;
    const t = document.getElementById('xEditText');
    if (t) t.value = CLIPEDIT.clip.orig_text;
    CLIPEDIT.dirty = true;
    clipEditDraw();
    clipEditLabels();
}

function clipEditWire() {
    const cv = document.getElementById('xWave');
    if (!cv) return;
    const at = e => {
        const r = cv.getBoundingClientRect();
        return {x: e.clientX - r.left, w: r.width};
    };
    cv.addEventListener('mousedown', e => {
        const {x, w} = at(e);
        const ax = clipEditX(CLIPEDIT.a, w), bx = clipEditX(CLIPEDIT.b, w);
        if (Math.abs(x - ax) <= 8 || Math.abs(x - bx) <= 8) {
            CLIPEDIT.drag = Math.abs(x - ax) <= Math.abs(x - bx) ? 'a' : 'b';
            e.preventDefault();
        } else {
            clipEditPlay(clipEditT(x, w));
        }
    });
    cv.addEventListener('mousemove', e => {
        const {x, w} = at(e);
        if (CLIPEDIT.drag) { clipEditSet(CLIPEDIT.drag, clipEditT(x, w), !e.altKey); return; }
        const near = [CLIPEDIT.a, CLIPEDIT.b].some(t => Math.abs(clipEditX(t, w) - x) <= 8);
        cv.style.cursor = near ? 'ew-resize' : 'pointer';
    });
}

// One listener for every editor: a drag ends wherever the button comes up.
window.addEventListener('mouseup', () => { CLIPEDIT.drag = null; });

// ── Saving ───────────────────────────────────────────────────────────────

async function clipEditSave(state, next) {
    const i = CLIPEDIT.i;
    if (i < 0 || !EXTRACT.run) return;
    const text = (document.getElementById('xEditText')?.value || '').trim();
    const body = {start: Math.round(CLIPEDIT.a * 1000) / 1000, end: Math.round(CLIPEDIT.b * 1000) / 1000, text};
    if (state !== null) body.state = state;
    let d;
    try {
        d = await vxJson(`/api/voices/extract/edit/${encodeURIComponent(EXTRACT.run.run_id)}/${i}`, body);
    } catch (e) {
        const msg = document.getElementById('xEditMsg');
        if (msg) msg.textContent = 'Not saved: ' + (e.message || e);
        return;
    }
    CLIPEDIT.dirty = false;
    extractClipUpdated(d.clip, d.totals);
    if (next) {
        const k = (EXTRACT.page || []).findIndex(x => x.i === i);
        const after = (EXTRACT.page || [])[k + 1];
        clipEditClose();
        if (after) clipEditOpen(after.i);
        else extractPageNext(true);
    } else {
        CLIPEDIT.clip = d.clip;
        clipEditLabels();
        const msg = document.getElementById('xEditMsg');
        if (msg) msg.textContent = 'Saved.';
    }
}

// Keys while the editor is open — in the capture phase, so Esc closes the
// editor and not the whole workspace (voices.js).
document.addEventListener('keydown', e => {
    if (CLIPEDIT.i < 0) return;
    // Only while the editor is on screen: on another tab (People → Review)
    // or with the workspace closed, Delete/Space/Esc belong to whatever is
    // shown — before, Delete trashed the reviewed voice AND dropped the
    // hidden clip. vxLeaveTab (voices.js) closes the editor on leaving.
    if ((typeof VX !== 'undefined' && VX.tab !== 'extract')
        || !document.getElementById('mapOverlay')?.classList.contains('visible')) return;
    // Esc closes the editor even while its clip is still loading — before,
    // it fell through to the workspace and closed all of it.
    if (e.key === 'Escape') {
        clipEditClose();
        e.preventDefault();
        e.stopImmediatePropagation();
        return;
    }
    if (!document.getElementById('xWave')) return;
    const typing = e.target && (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT'
                                || e.target.tagName === 'TEXTAREA');
    let done = true;
    if (typing && e.key === 'Enter' && e.target.id === 'xEditText') clipEditSave('kept', true);
    else if (typing) done = false;
    else if (e.key === ' ') { if (CLIPEDIT.src) clipEditStop(); else clipEditPlay('clip'); }
    else if (e.key === 'ArrowLeft') clipEditNudge(e.shiftKey ? 'b' : 'a', -0.05);
    else if (e.key === 'ArrowRight') clipEditNudge(e.shiftKey ? 'b' : 'a', 0.05);
    else if (e.key === 's' || e.key === 'S') clipEditSnap();
    else if (e.key === 'Enter') clipEditSave('kept', true);
    else if (e.key === 'Delete') clipEditSave('dropped', true);
    else done = false;
    if (done) { e.preventDefault(); e.stopImmediatePropagation(); }
}, true);
