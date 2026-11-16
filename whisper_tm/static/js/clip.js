// ── Clip editor: the timeline of the Clips tab (clips.js) ──
// The waveform with zoom, the playhead and the IN/OUT points of the cut
// being made or edited; clips.js keeps the cuts and makes the clip.
let clipIn = 0;
let clipOut = 10;
let clipTotalDur = 0;
let clipZoom = 1;
let clipViewCenter = 0.5; // 0..1 center of viewport in timeline
let clipPreviewInterval = null; // Handle for the previewClip interval (prevents stacked intervals)
let clipWaveSeq = 0;            // the newest waveform request — older answers are dropped
let clipWaveTimer = null;
let clipWaveShown = null;       // {file, start, end, n} the canvas shows, complete
let clipWaveOff = null;         // a file without a readable sound — not asked again until reopened
let clipFile = null;            // the file the editor was opened on — its points belong to it

// Logarithmic zoom: slider 0→100 maps to zoom 1×→500×
const ZOOM_MAX = 500;
function sliderToZoom(v) { return v <= 0 ? 1 : Math.round(Math.exp(v / 100 * Math.log(ZOOM_MAX))); }
function zoomToSlider(z) { return z <= 1 ? 0 : Math.round(Math.log(z) / Math.log(ZOOM_MAX) * 100); }

function openClipEditor() {
    if (!activeFile || !activeTranscript) return;
    clipTotalDur = activeTranscript.total_duration_sec || 0;
    if (!clipTotalDur) return;
    clipFile = activeFile;

    const v = document.getElementById('videoPlayer');
    const cur = v.currentTime || 0;
    clipIn = Math.max(0, cur - 5);
    clipOut = Math.min(clipTotalDur, cur + 15);
    clipZoom = 1;
    clipViewCenter = (clipIn + clipOut) / 2 / clipTotalDur;
    document.getElementById('clipZoom').value = 0;
    document.getElementById('clipZoomVal').textContent = '1×';

    clipWaveReset();
    updateClipUI();
    drawClipSegBars();
    document.getElementById('clipEditor').classList.add('visible');
}

// A file switch parks the open editor of the file left (its in/out points,
// zoom and view) instead of discarding them; coming back to that file
// opens the editor again exactly as it was. The points are never carried
// over to ANOTHER file — they belong to the one they were set on.
const clipParked = new Map();          // file → {clipIn, clipOut, clipZoom, clipViewCenter}
const CLIP_PARKED_MAX = 20;

function clipParkForSwitch() {
    const ed = document.getElementById('clipEditor');
    if (!ed || !ed.classList.contains('visible')) return;
    if (clipFile) {
        clipParked.delete(clipFile);
        clipParked.set(clipFile, {clipIn, clipOut, clipZoom, clipViewCenter});
        while (clipParked.size > CLIP_PARKED_MAX) clipParked.delete(clipParked.keys().next().value);
    }
    closeClipEditor();
}

/* The transcript of `file` is on screen (fileview.js loadTranscript): an
 * editor parked for it opens again with its points. */
function clipRestoreFor(file) {
    const k = clipParked.get(file);
    if (!k || file !== activeFile || !activeTranscript) return;
    clipParked.delete(file);
    clipTotalDur = activeTranscript.total_duration_sec || 0;
    if (!clipTotalDur) return;
    clipFile = file;
    clipIn = Math.min(Math.max(0, k.clipIn), clipTotalDur);
    clipOut = Math.min(Math.max(clipIn, k.clipOut), clipTotalDur);
    clipZoom = k.clipZoom || 1;
    clipViewCenter = k.clipViewCenter;
    document.getElementById('clipZoom').value = zoomToSlider(clipZoom);
    document.getElementById('clipZoomVal').textContent = clipZoom + '×';
    clipWaveReset();
    document.getElementById('clipEditor').classList.add('visible');
    updateClipUI();
    drawClipSegBars();
}

function closeClipEditor() {
    clipFile = null;
    document.getElementById('clipEditor').classList.remove('visible');
    clipWaveHide();
    // Stop the preview interval when closing
    if (clipPreviewInterval) { clearInterval(clipPreviewInterval); clipPreviewInterval = null; }
}

function setClipPoint(point, time) {
    if (!document.getElementById('clipEditor').classList.contains('visible')) {
        if (!activeFile || !activeTranscript) return;
        clipTotalDur = activeTranscript.total_duration_sec || 0;
        if (!clipTotalDur) return;
        clipFile = activeFile;
        clipZoom = 1;
        document.getElementById('clipZoom').value = 0;
        document.getElementById('clipZoomVal').textContent = '1×';
        clipWaveReset();
        document.getElementById('clipEditor').classList.add('visible');
        drawClipSegBars();
    }
    if (point === 'in') {
        clipIn = Math.max(0, time);
        if (clipIn >= clipOut) clipOut = Math.min(clipTotalDur, clipIn + 5);
    } else {
        clipOut = Math.min(clipTotalDur, time);
        if (clipOut <= clipIn) clipIn = Math.max(0, clipOut - 5);
    }
    updateClipUI();
}

function nudgeClip(point, delta) {
    if (point === 'in') {
        clipIn = Math.max(0, Math.min(clipOut - 0.1, clipIn + delta));
    } else {
        clipOut = Math.min(clipTotalDur, Math.max(clipIn + 0.1, clipOut + delta));
    }
    updateClipUI();
}

function fmtClipTime(sec) {
    const h = Math.floor(sec / 3600);
    const m = Math.floor((sec % 3600) / 60);
    const s = Math.floor(sec % 60);
    const ms = Math.floor((sec % 1) * 10);
    return `${h}:${String(m).padStart(2,'0')}:${String(s).padStart(2,'0')}.${ms}`;
}

function parseClipTime(str) {
    const parts = str.trim().split(':');
    if (parts.length === 3) return parseFloat(parts[0]) * 3600 + parseFloat(parts[1]) * 60 + parseFloat(parts[2]);
    if (parts.length === 2) return parseFloat(parts[0]) * 60 + parseFloat(parts[1]);
    // No `|| 0`: invalid/incomplete input yields NaN so the isNaN guard in onClipTimeInput kicks in
    return parseFloat(str);
}

function onClipTimeInput() {
    const inVal = parseClipTime(document.getElementById('clipIn').value);
    const outVal = parseClipTime(document.getElementById('clipOut').value);
    if (!isNaN(inVal) && !isNaN(outVal) && inVal < outVal) {
        clipIn = Math.max(0, inVal);
        clipOut = Math.min(clipTotalDur, outVal);
        updateClipRegion();
        updateClipDuration();
    }
}

function updateClipUI() {
    document.getElementById('clipIn').value = fmtClipTime(clipIn);
    document.getElementById('clipOut').value = fmtClipTime(clipOut);
    updateClipInner();
    updateClipRegion();
    updateClipDuration();
    updateClipZoomHint();
}

function onClipZoom() {
    const slider = parseFloat(document.getElementById('clipZoom').value);
    clipZoom = sliderToZoom(slider);
    document.getElementById('clipZoomVal').textContent = clipZoom + '×';
    // Keep view centered on selection
    clipViewCenter = (clipIn + clipOut) / 2 / clipTotalDur;
    updateClipInner();
    updateClipZoomHint();
}

function clipZoomToRegion() {
    if (!clipTotalDur) return;
    const regionFrac = (clipOut - clipIn) / clipTotalDur;
    clipZoom = Math.min(ZOOM_MAX, Math.max(1, Math.round(0.6 / regionFrac)));
    clipViewCenter = (clipIn + clipOut) / 2 / clipTotalDur;
    document.getElementById('clipZoom').value = zoomToSlider(clipZoom);
    document.getElementById('clipZoomVal').textContent = clipZoom + '×';
    updateClipInner();
    updateClipZoomHint();
}

function clipZoomToPlayhead() {
    if (!clipTotalDur) return;
    const v = document.getElementById('videoPlayer');
    const cur = v.currentTime || 0;
    clipZoom = Math.min(ZOOM_MAX, Math.max(8, Math.round(clipTotalDur / 120)));
    clipViewCenter = cur / clipTotalDur;
    document.getElementById('clipZoom').value = zoomToSlider(clipZoom);
    document.getElementById('clipZoomVal').textContent = clipZoom + '×';
    updateClipInner();
    updateClipZoomHint();
}

/* Where the visible window begins, as a fraction of the file — the bar and
 * the waveform's request from the same calculation, or they drift apart. */
function clipViewLeftFrac() {
    // Center viewport on clipViewCenter
    const viewWidthFrac = 1 / clipZoom;
    const leftFrac = clipViewCenter - viewWidthFrac / 2;
    return Math.max(0, Math.min(1 - viewWidthFrac, leftFrac));
}

function updateClipInner() {
    const inner = document.getElementById('clipTimelineInner');
    const widthPct = clipZoom * 100;
    inner.style.width = widthPct + '%';
    inner.style.left = (-clipViewLeftFrac() * widthPct) + '%';
    clipWaveSoon();
}

function updateClipZoomHint() {
    const hint = document.getElementById('clipZoomHint');
    if (clipZoom <= 1) {
        hint.textContent = 'Entire file';
    } else {
        const viewSec = clipTotalDur / clipZoom;
        const h = Math.floor(viewSec / 3600);
        const m = Math.floor((viewSec % 3600) / 60);
        const s = Math.round(viewSec % 60);
        let txt = 'Visible: ';
        if (h > 0) txt += `${h}h `;
        if (m > 0) txt += `${m}m `;
        if (viewSec < 60) txt += `${s}s`;
        hint.textContent = txt.trim();
    }
}

function updateClipRegion() {
    if (!clipTotalDur) return;
    const region = document.getElementById('clipRegion');
    region.style.left = (clipIn / clipTotalDur * 100) + '%';
    region.style.width = ((clipOut - clipIn) / clipTotalDur * 100) + '%';
}

function updateClipDuration() {
    const dur = clipOut - clipIn;
    const m = Math.floor(dur / 60);
    const s = (dur % 60).toFixed(1);
    document.getElementById('clipDuration').textContent = m > 0 ? `${m}m ${s}s` : `${s}s`;
}

function drawClipSegBars() {
    const container = document.getElementById('clipSegBars');
    if (!container) return;
    container.innerHTML = '';
    if (!activeTranscript || !clipTotalDur) return;
    (activeTranscript.segments || []).forEach(seg => {
        const bar = document.createElement('div');
        bar.className = 'clip-seg-bar';
        bar.style.left = (seg.start / clipTotalDur * 100) + '%';
        bar.style.width = Math.max(0.05, (seg.end - seg.start) / clipTotalDur * 100) + '%';
        container.appendChild(bar);
    });
}

// ── Waveform ──
/* The loudness of the visible stretch, behind the speech bars.
 *
 * The waveform makes cutting easier, but it must never block, even on a
 * 24 h recording. Nothing here waits: the server decodes a short window on the
 * spot and reads a long recording in the background (core/waveform.py) —
 * until then the answer is partly empty and `pending`, and the canvas fills
 * from the left, asked again every second.
 *
 * One canvas, as wide as the window: the bar it sits in is up to 500 times
 * wider than the screen, a canvas that wide does not exist. It is placed in
 * the bar at the stretch it shows, so it moves and stretches with it while
 * the next one loads. */
function clipWaveSoon(delay = 150) {
    clearTimeout(clipWaveTimer);
    clipWaveTimer = setTimeout(clipWaveLoad, delay);
}

function clipWaveHide() {
    clipWaveSeq++;
    clearTimeout(clipWaveTimer);
    clipWaveShown = null;
    const cv = document.getElementById('clipWave');
    if (cv) cv.hidden = true;
    clipWaveState('');
}

/* Opening the editor: the last file's waveform goes, and a file that had
 * none is asked once more. */
function clipWaveReset() {
    clipWaveOff = null;
    clipWaveHide();
}

function clipWaveState(text) {
    const el = document.getElementById('clipWaveState');
    if (!el) return;
    el.textContent = text;
    el.hidden = !text;
}

async function clipWaveLoad() {
    const tl = document.getElementById('clipTimeline');
    if (!document.getElementById('clipEditor').classList.contains('visible')
        || !tl || !activeFile || !clipTotalDur) return;
    const left = clipViewLeftFrac();
    const start = left * clipTotalDur, end = (left + 1 / clipZoom) * clipTotalDur;
    const n = Math.max(16, Math.min(4096, Math.round(tl.clientWidth * (window.devicePixelRatio || 1))));
    const file = activeFile;
    if (clipWaveOff === file) return;
    const s = clipWaveShown;
    // A marker moved, the window didn't: nothing new to ask for
    if (s && s.file === file && s.n === n && Math.abs(s.start - start) < 1e-3
        && Math.abs(s.end - end) < 1e-3) return;
    const seq = ++clipWaveSeq;
    let d;
    try {
        const r = await fetch(`/api/clip/waveform?file=${encodeURIComponent(file)}`
                              + `&start=${start.toFixed(3)}&end=${end.toFixed(3)}&n=${n}`);
        d = await r.json();
        if (!r.ok || d.error) throw new Error(d.error || `HTTP ${r.status}`);
    } catch (e) {
        // No sound, or none to read: the bar as it was, without one — and
        // not asked again for every marker moved
        if (seq === clipWaveSeq) { clipWaveHide(); clipWaveOff = file; }
        return;
    }
    if (seq !== clipWaveSeq || file !== activeFile) return;   // a newer window, another file
    clipWaveDraw(d);
    clipWaveShown = d.pending ? null : {file, start, end, n};
    clipWaveState(d.pending ? `〰 ${Math.round((d.progress || 0) * 100)} %` : '');
    if (d.pending) clipWaveTimer = setTimeout(clipWaveLoad, 1000);
}

function clipWaveDraw(d) {
    const cv = document.getElementById('clipWave');
    const tl = document.getElementById('clipTimeline');
    if (!cv || !tl) return;
    const peaks = d.peaks || [];
    cv.width = peaks.length;
    cv.height = Math.max(1, Math.round(tl.clientHeight * (window.devicePixelRatio || 1)));
    cv.style.left = (d.start / clipTotalDur * 100) + '%';
    cv.style.width = ((d.end - d.start) / clipTotalDur * 100) + '%';
    const g = cv.getContext('2d');
    g.clearRect(0, 0, cv.width, cv.height);
    // A value is decibels above -60 (0…255); the loudest in view fills the
    // height, so a quiet recording still shows where it speaks
    let top = 64;
    for (const v of peaks) if (v > top) top = v;
    g.fillStyle = getComputedStyle(cv).color;
    const mid = cv.height / 2;
    peaks.forEach((v, x) => {
        if (!v) return;                                   // silence, or not read yet
        const h = Math.max(0.5, v / top * mid);
        g.fillRect(x, mid - h, 1, 2 * h);
    });
    cv.hidden = false;
}

// Timeline interaction: click/drag to set IN/OUT, scroll to pan when zoomed
document.addEventListener('DOMContentLoaded', () => {
    const tl = document.getElementById('clipTimeline');
    let isDragging = false;
    let dragPoint = null;

    function tlTimeFromEvent(e) {
        const rect = tl.getBoundingClientRect();
        const inner = document.getElementById('clipTimelineInner');
        const innerRect = inner.getBoundingClientRect();
        const pct = (e.clientX - innerRect.left) / innerRect.width;
        return Math.max(0, Math.min(clipTotalDur, pct * clipTotalDur));
    }

    tl.addEventListener('mousedown', (e) => {
        if (!clipTotalDur) return;
        const time = tlTimeFromEvent(e);

        /* Middle mouse button: jump the playback position instead of dragging IN/OUT.
         *
         * The left button is already taken for setting the markers and
         * stays that way — while trimming you want both, but not on the
         * same button.
         *
         * `preventDefault` here is not cosmetic: without it the browser
         * starts its autoscroll mode on the middle button, and the
         * timeline slides out from under the pointer. */
        if (e.button === 1) {
            const v = document.getElementById('videoPlayer');
            if (v) {
                v.currentTime = time;
                updateClipPlayhead();
            }
            e.preventDefault();
            return;
        }
        if (e.button !== 0) return;      // right button belongs to the context menu

        const distIn = Math.abs(time - clipIn);
        const distOut = Math.abs(time - clipOut);
        dragPoint = distIn < distOut ? 'in' : 'out';
        if (dragPoint === 'in') clipIn = Math.max(0, Math.min(clipOut - 0.1, time));
        else clipOut = Math.min(clipTotalDur, Math.max(clipIn + 0.1, time));
        isDragging = true;
        updateClipUI();
        e.preventDefault();
    });

    document.addEventListener('mousemove', (e) => {
        if (!isDragging || !dragPoint) return;
        const time = tlTimeFromEvent(e);
        if (dragPoint === 'in') clipIn = Math.max(0, Math.min(clipOut - 0.1, time));
        else clipOut = Math.min(clipTotalDur, Math.max(clipIn + 0.1, time));
        updateClipUI();
    });

    document.addEventListener('mouseup', () => {
        isDragging = false;
        dragPoint = null;
    });

    // Scroll wheel on timeline: pan (normal), zoom (Ctrl+wheel)
    tl.addEventListener('wheel', (e) => {
        e.preventDefault();
        if (e.ctrlKey || e.metaKey) {
            // Zoom in/out
            const dir = e.deltaY < 0 ? 1.3 : 0.77;
            clipZoom = Math.max(1, Math.min(ZOOM_MAX, Math.round(clipZoom * dir)));
            document.getElementById('clipZoom').value = zoomToSlider(clipZoom);
            document.getElementById('clipZoomVal').textContent = clipZoom + '×';
            updateClipInner();
            updateClipZoomHint();
        } else {
            if (clipZoom <= 1) return;
            const delta = e.deltaY > 0 ? 0.05 : -0.05;
            clipViewCenter = Math.max(0, Math.min(1, clipViewCenter + delta));
            updateClipInner();
        }
    }, {passive: false});

    // Another width, another number of buckets
    window.addEventListener('resize', () => clipWaveSoon(300));
});

/* The playhead marker — as a function, not just inline in the interval.
 *
 * The interval runs every 100 ms; but someone who jumps via the middle
 * mouse button wants to see the marker there IMMEDIATELY, not just on the
 * next tick. Two places with the same calculation would be the usual
 * source of drift, hence one function for both. */
function updateClipPlayhead() {
    if (!clipTotalDur) return;
    const v = document.getElementById('videoPlayer');
    const ph = document.getElementById('clipPlayhead');
    if (!v || !ph) return;
    ph.style.left = (v.currentTime / clipTotalDur * 100) + '%';
}

setInterval(() => {
    if (!document.getElementById('clipEditor').classList.contains('visible')) return;
    updateClipPlayhead();
}, 100);

function previewClip() {
    const v = document.getElementById('videoPlayer');
    // Stop the previous preview interval so repeated clicks don't stack intervals
    if (clipPreviewInterval) { clearInterval(clipPreviewInterval); clipPreviewInterval = null; }
    v.currentTime = clipIn;
    v.play();
    clipPreviewInterval = setInterval(() => {
        if (v.currentTime >= clipOut || v.paused) {
            v.pause();
            clearInterval(clipPreviewInterval);
            clipPreviewInterval = null;
        }
    }, 50);
}

