// ── Performance view ─────────────────────────────────────────────────────
// Opens from the load meters in the top bar. Four small multiples on one
// shared time axis — CPU %, GPU %, RAM, VRAM — because percent and gigabytes
// don't share a y-axis. Data: /api/system/history (core/metrics.py).
//
// Reading and zooming:
//   · range presets in one row above the charts
//   · drag across any chart to zoom into that span (refetched at higher
//     resolution), double-click or "Reset zoom" to go back
//   · one crosshair across all four charts, one tooltip with every value
//   · CPU/GPU: the line is the bucket mean, the light band reaches up to
//     the bucket's peak — short spikes stay visible in a 30-day view
//   · a data table below, so no value is hover-only
//   · VRAM by process (core/vram_procs.py), on the GPU WhisperX runs on
//     only: the VRAM chart stacks who held the memory — this app's
//     processes in cool colours, every other program in warm ones, the
//     driver's own share in grey —, the chart below it follows each of this
//     app's processes on its own: a line for the mean and a band from the
//     lowest to the highest reading, so a swinging diarization shows as a
//     wide band; a live list shows the processes on the card right now
//     (/api/system/vram)
//   · Live (the first range button): the raw readings of the last 90 s —
//     one per second while the view is open — from /api/system/live, which
//     answers only what the view hasn't got yet. The charts follow the
//     clock: every animation frame moves the drawn marks to the left, so
//     nothing jumps once per reading, and a new reading slides in at the
//     right edge. No mean/peak bands there: every point is one reading.
//     Leaving it — another range, closing the view, a hidden tab — stops
//     its requests and its frames.

const PV_RANGES = [['1h', 3600], ['6h', 21600], ['24h', 86400],
                   ['3d', 259200], ['7d', 604800], ['30d', 2592000]];
const PV_H = 112;                              // plot height per chart
const PV_M = { l: 46, r: 14, t: 8, b: 6 };    // margins inside each SVG
const PV_GB = mb => mb / 1024;
const PV_SERIES = [
    { key: 'cpu', peak: 'cpu_max', label: 'CPU', unit: '%', pct: true },
    { key: 'gpu', peak: 'gpu_max', label: 'GPU', unit: '%', pct: true },
    { key: 'ram_mb', total: 'ram_total_mb', label: 'RAM', unit: 'GB' },
    { key: 'vram_mb', total: 'vram_total_mb', label: 'VRAM', unit: 'GB' },
];

// Live: the last 90 s of raw readings (core/metrics.py's live buffer).
const PV_LIVE_SPAN = 90;          // seconds on screen
const PV_LIVE_DELAY = 2;          // the right edge trails the clock: a reading slides in, it doesn't pop up
const PV_LIVE_KEEP = 100;         // seconds of readings kept: the span, the delay, room to scroll
const PV_LIVE_AHEAD = 8;          // grid drawn past the right edge, for the scroll until the next draw
const PV_LIVE_REDRAW = 5;         // drawn again after this long without news, so the axis never runs dry
const PV_LIVE_GAP = 2;            // the sampler's regular step: a line breaks after three missed ones
const PV_LIVE_SOURCES = 6;        // sources named one by one per kind, as in the history
const PV_LIVE_TIMEOUT = 5000;     // one request's limit (ms)
const PV_LIVE_BACKOFF = 15000;    // longest pause after failed requests (ms)

const pv = { range: 86400, zoom: null, data: null, timer: null, drag: null,
             width: 0, loading: false, procs: null, procsTimer: null, loadSeq: 0,
             liveOn: false, lv: null };

// Sources of VRAM: this app's roles in fixed blues, other programs in
// clearly different colours — warm ones first —, the driver's share in
// grey. A program keeps the colour it got first, in the chart and the list
// alike (by name it was two near-identical oranges for two programs).
const PV_OWN_COLOURS = { 'own:server': '#3b82f6', 'own:diarization': '#34d399',
                         'own:embedding': '#a78bfa', 'own:other': '#93c5fd' };
const PV_OWN_MORE = ['#22d3ee', '#818cf8', '#60a5fa', '#c4b5fd'];
const PV_FOREIGN = ['#f59e0b', '#ec4899', '#ef4444', '#84cc16', '#d95926', '#fb7185',
                    '#eab308', '#f97316'];
const PV_DRIVER = '#606078';
const pvColours = {};

function pvSourceColour(key, own) {
    if (own && PV_OWN_COLOURS[key]) return PV_OWN_COLOURS[key];
    if (key === 'foreign:other') return '#a16207';
    if (!pvColours[key]) {
        const pool = own ? PV_OWN_MORE : PV_FOREIGN;
        const used = new Set(Object.values(pvColours));
        pvColours[key] = pool.find(c => !used.has(c))
            || pool[Object.keys(pvColours).length % pool.length];
    }
    return pvColours[key];
}

/* The load history is the Machine window's "Load history" tab. */
function openPerfView() {
    if (typeof Machine !== 'undefined') Machine.open('load');
    else pvStart();
}

// On screen: the Machine window's tab
let _pvOn = false;
function pvShown() { return _pvOn; }

/* Start loading, the live loop and the process list (the tab came into view). */
function pvStart() {
    _pvOn = true;
    if (pv.liveOn) pvLiveStart(); else pvLoad();
    pvLoadProcs();
    clearInterval(pv.timer);
    clearInterval(pv.procsTimer);
    // Live while not zoomed: the newest bucket keeps moving. (The Live
    // range has a loop of its own.)
    pv.timer = setInterval(() => { if (!pv.liveOn && !pv.zoom && !pv.drag) pvLoad(); }, 30000);
    // The process list is "now" whatever the charts show.
    pv.procsTimer = setInterval(pvLoadProcs, 3000);
}

function closePerfView() {
    pvStop();
}

/* Stop everything that follows the load (the tab left the screen). */
function pvStop() {
    _pvOn = false;
    clearInterval(pv.timer);
    clearInterval(pv.procsTimer);
    pv.timer = null;
    pv.procsTimer = null;
    pvLiveStop();
    pvHideHover();
}

async function pvLoadProcs() {
    // One request at a time: a slow answer let them stack up every 3 s,
    // and an older one could overwrite a newer one
    if (pv.procsBusy) return;
    pv.procsBusy = true;
    try {
        const r = await fetch('/api/system/vram');
        pv.procs = await r.json();
    } catch (e) {
        pv.procs = { error: String(e.message || e) };
    } finally {
        pv.procsBusy = false;
    }
    const box = $('pvProcs');
    if (box) box.innerHTML = pvProcsInner();
}

function pvSpan() {
    if (pv.liveOn && pv.lv?.end != null) return [pv.lv.end - PV_LIVE_SPAN, pv.lv.end];
    if (pv.zoom) return pv.zoom;
    const end = Date.now() / 1000;
    return [end - pv.range, end];
}

async function pvLoad() {
    const body = $('perfBody');
    if (!body) return;
    const [start, end] = pvSpan();
    pv.width = Math.max(320, (body.clientWidth || 900) - 40);
    const points = Math.min(1500, Math.round((pv.width - PV_M.l - PV_M.r) / 2));
    // Refetch keeps the frame: the old render stays, dimmed.
    // Only the newest request may land: a slow 7-day answer arriving after
    // a click on 1 h would otherwise be drawn against the 1-hour axis.
    const seq = ++pv.loadSeq;
    body.classList.add('loading');
    let data;
    try {
        const r = await fetch(`/api/system/history?start=${start}&end=${end}&points=${points}`);
        data = await r.json();
    } catch (e) {
        data = { error: String(e.message || e) };
    }
    if (seq !== pv.loadSeq) return;
    body.classList.remove('loading');
    pv.data = data;
    pvRender();
}

function pvSetRange(sec) {
    if (sec === 'live') {
        pv.liveOn = true;
        pv.zoom = null;
        if (!pv.lv) pvLiveStart();
        return;
    }
    if (pv.liveOn) {
        pv.liveOn = false;
        pvLiveStop();
        pv.data = null;            // the live readings are not this range's
    }
    pv.range = sec;
    pv.zoom = null;
    pvLoad();
}

function pvResetZoom() {
    if (!pv.zoom) return;
    pv.zoom = null;
    pvLoad();
}


// ── Formatting ───────────────────────────────────────────────────────────

function pvFmt(s, v) {
    if (v == null) return '—';
    return s.pct ? `${Math.round(v)}%` : `${PV_GB(v).toFixed(1)} GB`;
}

// Memory where a process may hold a few MB: MB below one GB.
function pvMB(mb) {
    if (mb == null) return '—';
    return mb >= 1024 ? `${PV_GB(mb).toFixed(1)} GB` : `${Math.round(mb)} MB`;
}

function pvTime(t, withDate) {
    const d = new Date(t * 1000);
    const hm = d.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' });
    return withDate ? `${d.toLocaleDateString(undefined, { day: '2-digit', month: 'short' })} ${hm}` : hm;
}

// To the second — the live view's readings are a second apart.
function pvClock(t) {
    return new Date(t * 1000).toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit', second: '2-digit' });
}

// One x-axis label: the date for day steps, the time otherwise. Live,
// where a step is seconds: the full minute as the time, the seconds
// between as ":15".
function pvTickLabel(t, step) {
    const d = new Date(t * 1000);
    if (step < 60) return d.getSeconds() ? ':' + String(d.getSeconds()).padStart(2, '0') : pvTime(t);
    if (step >= 86400) return d.toLocaleDateString(undefined, { day: '2-digit', month: 'short' });
    return pvTime(t, d.getHours() === 0 && d.getMinutes() === 0);
}

// Time ticks: a clean step giving roughly five to eight ticks.
function pvTicks(start, end) {
    const steps = [300, 600, 900, 1800, 3600, 7200, 10800, 21600, 43200,
                   86400, 172800, 345600, 604800];
    const span = end - start;
    // About one label per 110 px: on a phone "05:00 AM05:10 AM…" ran together
    const most = Math.max(2, Math.min(8, Math.floor((pv.width || 900) / 110)));
    const step = steps.find(s => span / s <= most) || 604800;
    const offset = new Date().getTimezoneOffset() * 60;   // align to local midnight/hours
    const out = [];
    for (let t = Math.ceil((start - offset) / step) * step + offset; t <= end; t += step) out.push(t);
    return { ticks: out, step };
}


// ── Rendering ────────────────────────────────────────────────────────────

function pvRender() {
    if (pv.liveOn) { pvLiveRender(); return; }
    const body = $('perfBody');
    if (!body) return;
    const d = pv.data;
    const [start, end] = pvSpan();
    const zoomed = !!pv.zoom;
    let html = `<div class="pv-controls">
        ${pvRangeButtons(zoomed)}
        ${zoomed ? `<button class="pv-reset" onclick="pvResetZoom()">↺ Reset zoom</button>` : ''}
        <span class="pv-span">${pvTime(start, true)} – ${pvTime(end, true)}</span>
        <span class="pv-legend"><i class="pv-key-line"></i>mean <i class="pv-key-band"></i>peak</span>
    </div>`;
    if (!d || d.error) {
        body.innerHTML = html + `<div class="pv-empty">${d ? esc(d.error) : 'Loading …'}</div>`;
        return;
    }
    const pts = d.points || [];
    if (!pts.length) {
        body.innerHTML = html + `<div class="pv-empty">No data in this range yet. The server records
            the load every 30 seconds from its start on (kept ${d.retention_days} days).</div>`;
        return;
    }
    if (d.first && d.first > start + d.bucket) {
        html += `<div class="pv-note">History starts ${pvTime(d.first, true)} — recording began with this version.</div>`;
    }
    const { ticks, step } = pvTicks(start, end);
    const own = pvOwnKeys(d);
    html += PV_SERIES.map((s, i) => pvChart(s, pts, d, start, end, ticks, step,
                                             i === PV_SERIES.length - 1 && !own.length)).join('');
    if (own.length) html += pvOwnChart(own, pts, d, start, end, ticks, step);
    html += `<p class="pv-hint">Drag across a chart to zoom in · double-click to zoom out ·
        ${d.bucket >= 60 ? `one point = ${Math.round(d.bucket / 60)} min` : `one point = ${Math.round(d.bucket)} s`}</p>`;
    html += `<section class="pv-procs" id="pvProcs" aria-live="polite">${pvProcsInner()}</section>`;
    html += pvTable(pts);
    body.innerHTML = html;
    pvWire();
}

// The range presets: Live first, then the history.
function pvRangeButtons(zoomed) {
    return `<div class="pv-seg" role="group" aria-label="Time range">
            <button class="pv-live-btn${pv.liveOn ? ' active' : ''}" onclick="pvSetRange('live')"
                title="The last 90 seconds, every reading, scrolling with the clock">Live 90s</button>
            ${PV_RANGES.map(([l, s]) => `<button class="${!pv.liveOn && !zoomed && pv.range === s ? 'active' : ''}"
                onclick="pvSetRange(${s})">${l}</button>`).join('')}
        </div>`;
}

// Live: the marks in a group the animation frames move to the left,
// clipped to the plot — nothing runs into the axis labels or past the edge.
function pvScrolling(id, marks, H) {
    return `<defs><clipPath id="${id}"><rect x="${PV_M.l}" y="0" width="${pv.width - PV_M.l - PV_M.r}"
            height="${H}"/></clipPath></defs><g clip-path="url(#${id})"><g class="pv-scroll">${marks}</g></g>`;
}

// Break the line where the server was off (gap wider than three buckets).
function pvRuns(pts, key, bucket) {
    const runs = [];
    let run = [];
    for (let i = 0; i < pts.length; i++) {
        const p = pts[i];
        const gap = i > 0 && p.t - pts[i - 1].t > bucket * 3;
        if (p[key] == null || gap) { if (run.length) runs.push(run); run = []; }
        if (p[key] != null) run.push(p);
    }
    if (run.length) runs.push(run);
    return runs;
}

function pvChart(s, pts, d, start, end, ticks, step, last) {
    const live = pv.liveOn;
    const W = pv.width, H = PV_H + PV_M.t + PV_M.b + (last ? 18 : 0);
    const pw = W - PV_M.l - PV_M.r;
    const totals = pts.map(p => p[s.total]).filter(v => v);
    const ymax = s.pct ? 100 : (Math.max(...totals, 1));
    const x = t => PV_M.l + (t - start) / (end - start) * pw;
    const y = v => PV_M.t + PV_H - Math.min(v, ymax) / ymax * PV_H;
    const yTicks = s.pct ? [0, 50, 100] : [0, ymax / 2, ymax];
    let svg = '';
    // Recessive hairline grid: y ticks, then x ticks
    yTicks.forEach(v => {
        svg += `<line class="pv-grid" x1="${PV_M.l}" x2="${W - PV_M.r}" y1="${y(v)}" y2="${y(v)}"/>`
             + `<text class="pv-ytick" x="${PV_M.l - 6}" y="${y(v) + 3}">${s.pct ? v + '%' : PV_GB(v).toFixed(0) + ' GB'}</text>`;
    });
    // Everything that moves with time: x ticks, then the marks
    let marks = '';
    ticks.forEach(t => {
        marks += `<line class="pv-grid" x1="${x(t)}" x2="${x(t)}" y1="${PV_M.t}" y2="${PV_M.t + PV_H}"/>`;
        if (last) marks += `<text class="pv-xtick" x="${x(t)}" y="${PV_M.t + PV_H + 15}">${pvTickLabel(t, step)}</text>`;
    });
    const stack = s.key === 'vram_mb' ? pvStack(pts, d) : null;
    if (stack) marks += pvStackMarks(stack, pts, d, x, y);
    for (const run of pvRuns(pts, s.key, d.bucket)) {
        const line = run.map((p, k) => `${k ? 'L' : 'M'}${x(p.t).toFixed(1)},${y(p[s.key]).toFixed(1)}`).join('');
        if (stack) {
            // The stack is the area; the total stays the line on top
        } else if (s.peak && !live) {
            // Band from the mean up to the peak: spikes survive the averaging.
            // (Live has raw readings — nothing averaged, no band.)
            const top = run.map(p => `${x(p.t).toFixed(1)},${y(p[s.peak] ?? p[s.key]).toFixed(1)}`);
            const bottom = run.slice().reverse().map(p => `${x(p.t).toFixed(1)},${y(p[s.key]).toFixed(1)}`);
            marks += `<polygon class="pv-band" points="${top.concat(bottom).join(' ')}"/>`;
        } else {
            marks += `<path class="pv-area" d="${line}L${x(run[run.length - 1].t).toFixed(1)},${y(0)}L${x(run[0].t).toFixed(1)},${y(0)}Z"/>`;
        }
        marks += run.length > 1 ? `<path class="pv-line" d="${line}"/>`
                                : `<circle class="pv-dot" cx="${x(run[0].t)}" cy="${y(run[0][s.key])}" r="3"/>`;
    }
    svg += live ? pvScrolling(`pvClip-${s.key}`, marks, H) : marks;
    // Hover and drag layer (above the marks)
    svg += `<line class="pv-cross" y1="${PV_M.t}" y2="${PV_M.t + PV_H}" visibility="hidden"/>`
         + `<circle class="pv-hdot" r="4" visibility="hidden"/>`
         + `<rect class="pv-sel" y="${PV_M.t}" height="${PV_H}" width="0" visibility="hidden"/>`
         + `<rect class="pv-hit" x="${PV_M.l}" y="${PV_M.t}" width="${pw}" height="${PV_H}"/>`;
    // Head figure: live the current value; zoomed into the past, the mean
    // of the visible span — the value at the span's end is not "now".
    const lastP = pts[pts.length - 1];
    let now;
    if (pv.zoom) {
        const vals = pts.map(p => p[s.key]).filter(v => v != null);
        now = vals.length ? `<span class="pv-dim">mean </span>${pvFmt(s, vals.reduce((a, b) => a + b, 0) / vals.length)}` : '—';
    } else {
        now = lastP && end - lastP.t < d.bucket * 3 ? pvFmt(s, lastP[s.key]) : '—';
    }
    const cap = s.total && lastP?.[s.total] ? ` of ${PV_GB(lastP[s.total]).toFixed(0)} GB` : '';
    return `<div class="pv-chart" data-series="${s.key}">
        <div class="pv-chart-head"><span class="pv-chart-title">${s.label}</span>
            <span class="pv-chart-now">${now}<span class="pv-dim">${cap}</span></span>
            ${stack ? pvStackLegend(stack) : ''}</div>
        <svg width="${W}" height="${H}" data-start="${start}" data-end="${end}" data-ymax="${ymax}">${svg}</svg>
    </div>`;
}


// ── This app's processes, each on its own ───────────────────────────────

// The app's sources the history has, in its order: the server first.
function pvOwnKeys(d) {
    return (d?.vram_by?.keys || []).filter(k => k.own);
}

function pvOwnChart(keys, pts, d, start, end, ticks, step) {
    const live = pv.liveOn;
    const W = pv.width, H = PV_H + PV_M.t + PV_M.b + 18;
    const pw = W - PV_M.l - PV_M.r;
    const by = d.vram_by;
    // The scale follows the highest reading, not the card: a worker
    // swinging between 0.4 and 1.2 GB must not be a flat line under 12 GB.
    let top = 0;
    keys.forEach(k => (by.high[k.key] || by.series[k.key] || []).forEach(v => { if (v > top) top = v; }));
    const ymax = Math.max(1024, Math.ceil(top / 1024) * 1024);
    const x = t => PV_M.l + (t - start) / (end - start) * pw;
    const y = v => PV_M.t + PV_H - Math.min(v, ymax) / ymax * PV_H;
    let svg = '';
    [0, ymax / 2, ymax].forEach(v => {
        svg += `<line class="pv-grid" x1="${PV_M.l}" x2="${W - PV_M.r}" y1="${y(v)}" y2="${y(v)}"/>`
             + `<text class="pv-ytick" x="${PV_M.l - 6}" y="${y(v) + 3}">${PV_GB(v).toFixed(v && ymax < 4096 ? 1 : 0)} GB</text>`;
    });
    let marks = '';
    ticks.forEach(t => {
        marks += `<line class="pv-grid" x1="${x(t)}" x2="${x(t)}" y1="${PV_M.t}" y2="${PV_M.t + PV_H}"/>`;
        marks += `<text class="pv-xtick" x="${x(t)}" y="${PV_M.t + PV_H + 15}">${pvTickLabel(t, step)}</text>`;
    });
    // Stretches of points that have the split, broken where the server was off
    const runs = [];
    let run = [];
    pts.forEach((p, i) => {
        const gap = run.length && p.t - pts[run[run.length - 1]].t > d.bucket * 3;
        if (p.vram_own_mb == null || gap) { if (run.length) runs.push(run); run = []; }
        if (p.vram_own_mb != null) run.push(i);
    });
    if (run.length) runs.push(run);
    // Bands first, lines on top. A band only where both ends are known:
    // rows from before the lowest reading was recorded have none.
    keys.forEach(k => {
        const lo = by.low[k.key], hi = by.high[k.key];
        if (!lo || !hi) return;
        const colour = pvSourceColour(k.key, true);
        const bandRuns = [];
        runs.forEach(r => {
            let part = [];
            r.forEach(i => {
                if (lo[i] == null || hi[i] == null) { if (part.length) bandRuns.push(part); part = []; }
                else part.push(i);
            });
            if (part.length) bandRuns.push(part);
        });
        bandRuns.forEach(r => {
            if (r.length < 2) return;
            const upper = r.map(i => `${x(pts[i].t).toFixed(1)},${y(hi[i]).toFixed(1)}`);
            const lower = r.slice().reverse().map(i => `${x(pts[i].t).toFixed(1)},${y(lo[i]).toFixed(1)}`);
            marks += `<polygon class="pv-swing" style="fill:${colour}" points="${upper.concat(lower).join(' ')}"/>`
                // Thin edges: where two bands overlap, each still shows its reach
                 + `<polyline class="pv-swing-edge" style="stroke:${colour}" points="${upper.join(' ')}"/>`
                 + `<polyline class="pv-swing-edge" style="stroke:${colour}" points="${lower.join(' ')}"/>`;
        });
    });
    keys.forEach(k => {
        const mean = by.series[k.key] || [];
        const colour = pvSourceColour(k.key, true);
        runs.forEach(r => {
            if (r.length === 1) {
                marks += `<circle class="pv-dot" style="fill:${colour}" cx="${x(pts[r[0]].t)}" cy="${y(mean[r[0]] ?? 0)}" r="3"/>`;
                return;
            }
            const line = r.map((i, n) => `${n ? 'L' : 'M'}${x(pts[i].t).toFixed(1)},${y(mean[i] ?? 0).toFixed(1)}`).join('');
            marks += `<path class="pv-line" style="stroke:${colour}" d="${line}"/>`;
        });
    });
    svg += live ? pvScrolling('pvClip-own', marks, H) : marks;
    svg += `<line class="pv-cross" y1="${PV_M.t}" y2="${PV_M.t + PV_H}" visibility="hidden"/>`
         + `<rect class="pv-sel" y="${PV_M.t}" height="${PV_H}" width="0" visibility="hidden"/>`
         + `<rect class="pv-hit" x="${PV_M.l}" y="${PV_M.t}" width="${pw}" height="${PV_H}"/>`;
    // Legend with the newest values: mean, and the swing as low–high
    const last = pts.length - 1;
    const legend = keys.map(k => {
        const m = by.series[k.key]?.[last], lo = by.low[k.key]?.[last], hi = by.high[k.key]?.[last];
        const now = m != null && !pv.zoom ? ` <b>${pvMB(m)}</b>${lo != null && hi != null && hi - lo >= 1
            ? `<span class="pv-dim"> ${pvMB(lo)}–${pvMB(hi)}</span>` : ''}` : '';
        return `<span class="pv-src"><i style="background:${pvSourceColour(k.key, true)}"></i>${esc(k.label)}${now}</span>`;
    }).join('');
    return `<div class="pv-chart" data-series="vram_own">
        <div class="pv-chart-head"><span class="pv-chart-title">VRAM · this app, by process</span>
            <span class="pv-chart-now pv-dim">${live ? 'line: each reading' : 'line: mean · band: lowest to highest reading'}</span>
            <span class="pv-srcs">${legend}</span></div>
        <svg width="${W}" height="${H}" data-start="${start}" data-end="${end}" data-ymax="${ymax}">${svg}</svg>
    </div>`;
}


// ── VRAM by source ───────────────────────────────────────────────────────

// Per point the stacked extent of every source; `null` where the history
// has no split (rows from before it was recorded).
function pvStack(pts, d) {
    const by = d.vram_by;
    if (!by || !by.keys || !by.keys.length) return null;
    const layers = by.keys.map(k => ({ key: k.key, label: k.label, own: k.own,
                                       colour: pvSourceColour(k.key, k.own) }));
    layers.push({ key: '_driver', label: 'Driver, no process', own: null, colour: PV_DRIVER });
    const lo = layers.map(() => new Array(pts.length).fill(null));
    const hi = layers.map(() => new Array(pts.length).fill(null));
    let any = false;
    pts.forEach((p, i) => {
        if (p.vram_own_mb == null || p.vram_mb == null) return;
        let sum = 0;
        by.keys.forEach((k, j) => {
            const v = (by.series[k.key] || [])[i] || 0;
            lo[j][i] = sum; sum += v; hi[j][i] = sum;
        });
        const j = layers.length - 1;
        lo[j][i] = sum; hi[j][i] = Math.max(sum, p.vram_mb);
        any = true;
    });
    return any ? { layers, lo, hi } : null;
}

function pvStackMarks(stack, pts, d, x, y) {
    let svg = '';
    stack.layers.forEach((layer, j) => {
        // One polygon per stretch of known points without a gap
        let run = [];
        const flush = () => {
            if (run.length > 1) {
                const top = run.map(i => `${x(pts[i].t).toFixed(1)},${y(stack.hi[j][i]).toFixed(1)}`);
                const bottom = run.slice().reverse().map(i => `${x(pts[i].t).toFixed(1)},${y(stack.lo[j][i]).toFixed(1)}`);
                svg += `<polygon class="pv-stack" style="fill:${layer.colour}" points="${top.concat(bottom).join(' ')}"/>`;
            }
            run = [];
        };
        pts.forEach((p, i) => {
            const gap = run.length && p.t - pts[run[run.length - 1]].t > d.bucket * 3;
            if (stack.hi[j][i] == null || gap) flush();
            if (stack.hi[j][i] != null) run.push(i);
        });
        flush();
    });
    return svg;
}

function pvStackLegend(stack) {
    const item = l => `<span class="pv-src"><i style="background:${l.colour}"></i>${esc(l.label)}</span>`;
    const own = stack.layers.filter(l => l.own === true);
    const foreign = stack.layers.filter(l => l.own === false);
    const driver = stack.layers.filter(l => l.own === null);
    return `<span class="pv-srcs">
        ${own.length ? `<span class="pv-src-group">This app</span>${own.map(item).join('')}` : ''}
        ${foreign.length ? `<span class="pv-src-group pv-foreign">Other programs</span>${foreign.map(item).join('')}` : ''}
        ${driver.map(item).join('')}</span>`;
}

// The live list: what is on the card right now.
function pvProcsInner() {
    const v = pv.procs;
    const head = `<div class="pv-procs-head"><span class="pv-chart-title">VRAM now, by process</span>`;
    if (!v) return head + `</div><p class="pv-note">Loading …</p>`;
    if (v.error) return head + `</div><p class="pv-note">${esc(v.error)}</p>`;
    if (!v.available) {
        return head + `</div><p class="pv-note">Not available: the memory of each process comes from the
            NVIDIA driver (NVML or nvidia-smi).</p>`;
    }
    const total = v.total_mb || v.used_mb || 1;
    const me = (v.procs.find(p => p.role === 'server') || {}).user;
    const rows = v.procs.map(p => ({ ...p, colour: pvSourceColour(p.key, p.own) }));
    if (v.unattributed_mb > 0) {
        rows.push({ label: 'Driver, no process', mb: v.unattributed_mb, own: null, colour: PV_DRIVER });
    }
    // One bar for the card: every source by its share, then what is free
    const bar = rows.map(r => r.mb > 0
        ? `<i style="width:${(r.mb / total * 100).toFixed(2)}%;background:${r.colour}"
              title="${esc(r.label)} · ${pvMB(r.mb)}"></i>` : '').join('');
    const kind = r => r.own === true ? '<span class="pv-badge own">this app</span>'
                    : r.own === false ? '<span class="pv-badge foreign">other program</span>' : '';
    const details = r => [r.pid ? 'PID ' + r.pid : '',
        r.user && me && r.user !== me ? esc(r.user) : '',
        r.type ? (r.type === 'G' ? 'graphics' : r.type === 'C+G' ? 'compute + graphics' : 'compute') : '']
        .filter(Boolean).join(' · ');
    const list = rows.map(r => `<tr class="${r.own === false ? 'is-foreign' : ''}">
        <td><i class="pv-swatch" style="background:${r.colour}"></i>${esc(r.label)} ${kind(r)}
            ${r.models && r.models.length ? `<div class="pv-proc-models">${r.models.map(esc).join(' · ')}</div>` : ''}
            ${details(r) ? `<div class="pv-proc-sub">${details(r)}</div>` : ''}</td>
        <td class="pv-num">${pvMB(r.mb)}</td>
        <td class="pv-num pv-dim">${r.mb != null ? Math.round(r.mb / total * 100) + '%' : ''}</td>
        <td class="pv-dim pv-proc-details">${details(r)}</td>
    </tr>`).join('');
    return head + `<span class="pv-chart-now">${pvMB(v.used_mb)}<span class="pv-dim"> of ${pvMB(v.total_mb)}
            · this app ${pvMB(v.own_mb)} · other programs ${pvMB(v.foreign_mb)}</span></span>
            ${v.name ? `<span class="pv-gpu" title="Only this card is listed — the one WhisperX runs on. Other cards (e.g. the one driving the display) are not counted.">on ${esc(v.name)}</span>` : ''}</div>
        <div class="pv-card" role="img" aria-label="VRAM by process">${bar}</div>
        <table class="pv-table pv-proc-table"><thead><tr><th>Process</th><th>VRAM</th><th>Share</th>
            <th class="pv-proc-details">Details</th></tr></thead><tbody>${list}</tbody></table>`;
}

function pvTable(pts) {
    const rows = pts.slice().reverse().slice(0, 400).map(p => `<tr>
        <td>${pvTime(p.t, true)}</td>
        <td>${pvFmt(PV_SERIES[0], p.cpu)} <span class="pv-dim">/ ${pvFmt(PV_SERIES[0], p.cpu_max)}</span></td>
        <td>${pvFmt(PV_SERIES[1], p.gpu)} <span class="pv-dim">/ ${pvFmt(PV_SERIES[1], p.gpu_max)}</span></td>
        <td>${pvFmt(PV_SERIES[2], p.ram_mb)}</td>
        <td>${pvFmt(PV_SERIES[3], p.vram_mb)}${p.vram_own_mb != null
            ? ` <span class="pv-dim">(this app ${pvMB(p.vram_own_mb)} · other ${pvMB(p.vram_foreign_mb)})</span>` : ''}</td></tr>`).join('');
    return `<details class="pv-table-wrap"><summary>Data table (${plural(pts.length, 'point')}${pts.length > 400 ? ', newest 400 shown' : ''})</summary>
        <table class="pv-table"><thead><tr><th>Time</th><th>CPU mean / peak</th><th>GPU mean / peak</th>
        <th>RAM</th><th>VRAM</th></tr></thead><tbody>${rows}</tbody></table></details>`;
}


// ── Live: the last 90 s, scrolling ──────────────────────────────────────

function pvLiveStart() {
    pvLiveStop();
    if (!pvShown() || document.visibilityState === 'hidden') return;
    pv.loadSeq++;                  // a history answer still on its way must not land here
    pv.lv = { readings: [], sources: {}, skew: null, step: null, ctrl: null, timer: null,
              raf: 0, fails: 0, error: null, retry: 0, drawn: null, end: null,
              groups: [], svgs: [], hover: null, hoverP: null };
    pvLiveRender();
    pvLivePoll();
    pv.lv.raf = requestAnimationFrame(pvLiveFrame);
}

// Stops the requests and the frames — the choice of Live stays (pv.liveOn),
// so the view comes back live when it is opened or shown again.
function pvLiveStop() {
    const lv = pv.lv;
    if (!lv) return;
    pv.lv = null;                  // an answer still on its way sees it is not wanted
    clearTimeout(lv.timer);
    lv.ctrl?.abort();
    cancelAnimationFrame(lv.raf);
    pvHideHover();
}

// One request at a time; the next is scheduled when this one is done —
// a slow server never gets a pile of them.
async function pvLivePoll() {
    const lv = pv.lv;
    if (!lv || lv.ctrl) return;
    lv.timer = null;
    const ctrl = lv.ctrl = new AbortController();
    const kill = setTimeout(() => ctrl.abort(), PV_LIVE_TIMEOUT);
    const last = lv.readings[lv.readings.length - 1];
    const sent = Date.now();
    let d = null, error = null;
    try {
        const r = await fetch('/api/system/live' + (last ? `?since=${last.t}` : ''), { signal: ctrl.signal });
        d = await r.json();
        if (!r.ok) throw new Error(d?.error || `HTTP ${r.status}`);
    } catch (e) {
        error = e;
    }
    clearTimeout(kill);
    if (pv.lv !== lv) return;      // Live was left meanwhile
    lv.ctrl = null;
    if (error) {
        // Wait longer after every failure in a row: 2, 4, 8, then 15 s
        lv.fails++;
        lv.error = error.name === 'AbortError' ? 'no answer' : String(error.message || error);
        lv.retry = Math.min(PV_LIVE_BACKOFF, 1000 * 2 ** lv.fails);
        pvLiveStatus();
        lv.timer = setTimeout(pvLivePoll, lv.retry);
        return;
    }
    lv.fails = 0;
    lv.error = null;
    pvLiveMerge(lv, d, (sent + Date.now()) / 2000);
    pvLiveDraw();
    lv.timer = setTimeout(pvLivePoll, Math.min(2000, Math.max(1000, (d.step || 1) * 1000)));
}

// The answer's readings onto the ones the view has. `clientMid`: when the
// server answered, by this browser's clock — the readings carry the
// server's clock, and the axis has to follow that one.
function pvLiveMerge(lv, d, clientMid) {
    const skew = d.now - clientMid;
    lv.skew = lv.skew == null || Math.abs(skew - lv.skew) > 1 ? skew : lv.skew + (skew - lv.skew) * 0.2;
    lv.step = d.step;
    Object.assign(lv.sources, d.sources || {});
    let last = lv.readings.length ? lv.readings[lv.readings.length - 1].t : -Infinity;
    for (const r of d.readings || []) {
        if (r.t > last) { lv.readings.push(r); last = r.t; }
    }
    const keep = d.now - PV_LIVE_KEEP;
    const cut = lv.readings.findIndex(r => r.t >= keep);
    lv.readings.splice(0, cut < 0 ? lv.readings.length : cut);
}

// Where the right edge is now, on the server's clock.
function pvLiveEnd() {
    return Date.now() / 1000 + (pv.lv?.skew || 0) - PV_LIVE_DELAY;
}

// The readings in the history's shape, so the same charts draw them: per
// source a value per point — the largest of each kind by name, the rest
// of each kind summed, as core/metrics.py names them. No low/high: a
// single reading has no swing.
function pvLiveData(lv) {
    const pts = lv.readings;
    const isOwn = k => (lv.sources[k] ? !!lv.sources[k].own : k.startsWith('own:'));
    const peak = {};
    pts.forEach(p => Object.entries(p.procs || {}).forEach(([k, mb]) => { peak[k] = Math.max(peak[k] || 0, mb); }));
    const named = {};
    [true, false].forEach(own => Object.keys(peak).filter(k => isOwn(k) === own)
        .sort((a, b) => peak[b] - peak[a] || (a < b ? -1 : 1))
        .forEach((k, i) => { named[k] = i < PV_LIVE_SOURCES ? k : (own ? 'own:other' : 'foreign:other'); }));
    const series = {};
    pts.forEach((p, i) => Object.entries(p.procs || {}).forEach(([k, mb]) => {
        const v = series[named[k]] ||= new Array(pts.length).fill(null);
        v[i] = (v[i] || 0) + mb;
    }));
    // A reading that knows its processes held 0 of a source absent from it
    Object.values(series).forEach(v => pts.forEach((p, i) => { if (p.procs && v[i] == null) v[i] = 0; }));
    const meta = k => (lv.sources[k] ? [lv.sources[k].label, !!lv.sources[k].own]
        : k === 'own:other' ? ['Other (own)', true]
        : k === 'foreign:other' ? ['Other programs', false] : [k, isOwn(k)]);
    const top = k => Math.max(...series[k].map(v => v || 0));
    const keys = Object.keys(series)
        .sort((a, b) => (meta(b)[1] - meta(a)[1]) || (top(b) - top(a)) || (a < b ? -1 : 1))
        .map(k => ({ key: k, label: meta(k)[0], own: meta(k)[1] }));
    return { points: pts, bucket: PV_LIVE_GAP, vram_by: { keys, series, low: {}, high: {} } };
}

// Live ticks: a step of seconds giving about one label per 110 px, on the
// clock's own seconds — the grid moves along with the readings.
function pvLiveTicks(start, end) {
    const most = Math.max(2, Math.min(8, Math.floor((pv.width || 900) / 110)));
    const step = [5, 10, 15, 30, 60].find(s => PV_LIVE_SPAN / s <= most) || 60;
    const ticks = [];
    for (let t = Math.ceil(start / step) * step; t <= end; t += step) ticks.push(t);
    return { ticks, step };
}

function pvLiveStatusText() {
    const lv = pv.lv;
    if (!lv) return 'Paused';
    if (lv.error) {
        return `<span class="pv-live-err">No answer from the server (${esc(lv.error)}) —
            trying again in ${Math.round(lv.retry / 1000)} s</span>`;
    }
    const pace = lv.step == null ? 'connecting …'
        : lv.step <= 1 ? 'one reading per second' : `one reading every ${lv.step} s`;
    return `Last 90 s · ${pace}`;
}

function pvLiveStatus() {
    const el = $('pvLiveStatus');
    if (el) el.innerHTML = pvLiveStatusText();
}

// The frame: built once per start (and on resize). Every answer only
// redraws the charts, the status and the table rows — the buttons, the
// process list and the table's open state stay put, so a click is never
// lost to a redraw.
function pvLiveRender() {
    const body = $('perfBody');
    if (!body) return;
    if (!pvLiveWired) {
        pvLiveWired = true;
        body.addEventListener('pointermove', pvLivePointer);
        body.addEventListener('pointerleave', pvLivePointer);
    }
    pv.width = Math.max(320, (body.clientWidth || 900) - 40);
    body.classList.remove('loading');
    body.innerHTML = `<div class="pv-controls">
            ${pvRangeButtons(false)}
            <span class="pv-span" id="pvLiveStatus">${pvLiveStatusText()}</span>
        </div>
        <div class="pv-live-charts" id="pvLiveCharts"></div>
        <p class="pv-hint">Raw readings as measured, the newest on the right — nothing averaged, so no
            mean/peak bands · the other ranges show the stored history</p>
        <section class="pv-procs" id="pvProcs" aria-live="polite">${pvProcsInner()}</section>
        <details class="pv-table-wrap"><summary id="pvLiveSum">Data table</summary>
            <table class="pv-table"><thead><tr><th>Time</th><th>CPU</th><th>GPU</th><th>RAM</th><th>VRAM</th></tr></thead>
            <tbody id="pvLiveRows"></tbody></table></details>`;
    pvLiveDraw();
}

function pvLiveDraw() {
    const lv = pv.lv, box = $('pvLiveCharts');
    if (!lv || !box) return;
    const end = lv.drawn = lv.end = pvLiveEnd();
    const start = end - PV_LIVE_SPAN;
    const d = pv.data = pvLiveData(lv);
    pvLiveStatus();
    pvLiveRows(d.points.filter(p => p.t >= start));     // the 90 s on screen, not the margin
    lv.hoverP = null;              // the crosshair goes with the old charts
    if (!d.points.length) {
        box.innerHTML = `<div class="pv-empty">${lv.error ? 'No readings — the server does not answer.'
            : 'Waiting for the first readings …'}</div>`;
        lv.groups = lv.svgs = [];
        return;
    }
    const { ticks, step } = pvLiveTicks(start - 10, end + PV_LIVE_AHEAD);
    const own = pvOwnKeys(d);
    let html = PV_SERIES.map((s, i) => pvChart(s, d.points, d, start, end, ticks, step,
                                               i === PV_SERIES.length - 1 && !own.length)).join('');
    if (own.length) html += pvOwnChart(own, d.points, d, start, end, ticks, step);
    box.innerHTML = html;
    lv.groups = [...box.querySelectorAll('.pv-scroll')];
    lv.svgs = [...box.querySelectorAll('svg')];
    pvWire();
}

// Every animation frame: the drawn marks move left by the time passed
// since they were drawn — the axis follows the clock, smoothly.
function pvLiveFrame() {
    const lv = pv.lv;
    if (!lv) return;
    lv.raf = requestAnimationFrame(pvLiveFrame);
    if (lv.drawn == null) return;
    const end = pvLiveEnd();
    // Nothing new for a while (a slow or a gone server): draw again, so the
    // grid keeps up and the heads stop showing a value that is no more
    if (end - lv.drawn > PV_LIVE_REDRAW) { pvLiveDraw(); return; }
    lv.end = end;
    const shift = -(end - lv.drawn) / PV_LIVE_SPAN * (pv.width - PV_M.l - PV_M.r);
    const transform = `translate(${shift.toFixed(2)},0)`;
    lv.groups.forEach(g => g.setAttribute('transform', transform));
    lv.svgs.forEach(svg => { svg.dataset.start = end - PV_LIVE_SPAN; svg.dataset.end = end; });
    if (lv.hover) pvLiveHover(false);
}

// The reading under the pointer — of those already on screen.
function pvLiveNearest(t) {
    const pts = pv.data?.points || [];
    let p = pvNearest(t), i = pts.indexOf(p);
    while (p && p.t > pv.lv.end && i > 0) p = pts[--i];
    return p && p.t <= pv.lv.end ? p : null;
}

// Live hover: one listener on the view's body, which outlives the redraws.
// The charts under the pointer are replaced with every answer, and a
// replaced chart never hears that the pointer left it.
let pvLiveWired = false;

function pvLivePointer(e) {
    const lv = pv.lv;
    if (!lv) return;
    if (e.type === 'pointermove' && e.target?.classList?.contains('pv-hit')) {
        lv.hover = { x: e.clientX, y: e.clientY };
        pvLiveHover(true);
    } else if (lv.hover) {
        lv.hover = lv.hoverP = null;
        pvHideHover();
    }
}

// `moved`: the pointer moved (the tooltip follows it); otherwise a frame
// moved the charts, and the tooltip is rebuilt only for another reading.
function pvLiveHover(moved) {
    const lv = pv.lv, h = lv?.hover;
    if (!h) return;
    // Still over a chart? A redraw swaps the charts under the pointer
    // without the old ones ever hearing that it left.
    const el = document.elementFromPoint?.(h.x, h.y);
    if (!el?.classList?.contains('pv-hit')) {
        lv.hover = lv.hoverP = null;
        pvHideHover();
        return;
    }
    const p = pvLiveNearest(pvTimeAt(el.ownerSVGElement, h.x));
    if (!p) return;
    if (moved || p !== lv.hoverP) { lv.hoverP = p; pvShowHover(p, h.x, h.y); }
    else pvCrossAt(p);
}

function pvLiveRows(pts) {
    const sum = $('pvLiveSum'), rows = $('pvLiveRows');
    if (sum) sum.textContent = `Data table (${plural(pts.length, 'reading')})`;
    if (!rows) return;
    rows.innerHTML = pts.slice().reverse().map(p => `<tr><td>${pvClock(p.t)}</td>
        <td>${pvFmt(PV_SERIES[0], p.cpu)}</td><td>${pvFmt(PV_SERIES[1], p.gpu)}</td>
        <td>${pvFmt(PV_SERIES[2], p.ram_mb)}</td>
        <td>${pvFmt(PV_SERIES[3], p.vram_mb)}${p.vram_own_mb != null
            ? ` <span class="pv-dim">(this app ${pvMB(p.vram_own_mb)} · other ${pvMB(p.vram_foreign_mb)})</span>` : ''}</td></tr>`).join('');
}


// ── Hover, drag-to-zoom ──────────────────────────────────────────────────

function pvTimeAt(svg, clientX) {
    const r = svg.getBoundingClientRect();
    const start = +svg.dataset.start, end = +svg.dataset.end;
    const pw = r.width - PV_M.l - PV_M.r;
    const px = Math.max(0, Math.min(pw, clientX - r.left - PV_M.l));
    return start + px / pw * (end - start);
}

function pvNearest(t) {
    const pts = pv.data?.points || [];
    if (!pts.length) return null;
    let lo = 0, hi = pts.length - 1;
    while (lo < hi) {
        const mid = (lo + hi) >> 1;
        if (pts[mid].t < t) lo = mid + 1; else hi = mid;
    }
    const cand = [pts[lo], pts[lo - 1]].filter(Boolean);
    return cand.sort((a, b) => Math.abs(a.t - t) - Math.abs(b.t - t))[0];
}

// The crosshair at a point in every chart (and the dot on a single line).
function pvCrossAt(p) {
    const [start, end] = pvSpan();
    document.querySelectorAll('#perfBody .pv-chart').forEach(ch => {
        const svg = ch.querySelector('svg');
        const s = PV_SERIES.find(x => x.key === ch.dataset.series);
        const pw = svg.getBoundingClientRect().width - PV_M.l - PV_M.r;
        const cx = PV_M.l + (p.t - start) / (end - start) * pw;
        const cross = svg.querySelector('.pv-cross');
        cross.setAttribute('x1', cx); cross.setAttribute('x2', cx);
        cross.setAttribute('visibility', 'visible');
        const dot = svg.querySelector('.pv-hdot');
        if (!s || !dot) return;             // several lines: the crosshair says it
        if (p[s.key] != null) {
            const ymax = +svg.dataset.ymax;
            dot.setAttribute('cx', cx);
            dot.setAttribute('cy', PV_M.t + PV_H - Math.min(p[s.key], ymax) / ymax * PV_H);
            dot.setAttribute('visibility', 'visible');
        } else {
            dot.setAttribute('visibility', 'hidden');
        }
    });
}

function pvShowHover(p, clientX, clientY) {
    pvCrossAt(p);
    // One tooltip, every series: values lead, labels follow.
    const tip = $('perfTip');
    tip.innerHTML = `<div class="pv-tip-time">${pv.liveOn ? pvClock(p.t) : pvTime(p.t, true)}</div>` + PV_SERIES.map(s => {
        const peak = s.peak && p[s.peak] != null ? ` <span class="pv-dim">peak ${pvFmt(s, p[s.peak])}</span>` : '';
        return `<div class="pv-tip-row"><i class="pv-key-line"></i><b>${pvFmt(s, p[s.key])}</b>
            <span class="pv-tip-label">${s.label}</span>${peak}</div>`
            + (s.key === 'vram_mb' ? pvTipSources(p) : '');
    }).join('');
    tip.hidden = false;
    const w = tip.offsetWidth, h = tip.offsetHeight;
    const left = clientX + 16 + w > window.innerWidth ? clientX - 16 - w : clientX + 16;
    tip.style.left = left + 'px';
    tip.style.top = Math.max(8, Math.min(window.innerHeight - h - 8, clientY - h / 2)) + 'px';
}

// Who held the VRAM at this point, largest first.
function pvTipSources(p) {
    const d = pv.data;
    const stack = d && pvStack(d.points || [], d);
    const i = (d?.points || []).indexOf(p);
    if (!stack || i < 0 || stack.hi[0][i] == null) return '';
    return stack.layers
        .map((l, j) => ({ l, v: stack.hi[j][i] - stack.lo[j][i] }))
        .filter(e => e.v >= 1)
        .sort((a, b) => b.v - a.v)
        .map(e => {
            const lo = d.vram_by.low?.[e.l.key]?.[i], hi = d.vram_by.high?.[e.l.key]?.[i];
            const swing = lo != null && hi != null && hi - lo >= 1
                ? ` <span class="pv-dim">${pvMB(lo)}–${pvMB(hi)}</span>` : '';
            return `<div class="pv-tip-row pv-tip-sub"><i class="pv-swatch" style="background:${e.l.colour}"></i>
                <b>${pvMB(e.v)}</b><span class="pv-tip-label">${esc(e.l.label)}</span>${swing}</div>`;
        }).join('');
}

function pvHideHover() {
    document.querySelectorAll('#perfBody .pv-cross, #perfBody .pv-hdot')
        .forEach(el => el.setAttribute('visibility', 'hidden'));
    const tip = $('perfTip');
    if (tip) tip.hidden = true;
}

function pvShowSel(t0, t1) {
    const [start, end] = pvSpan();
    document.querySelectorAll('#perfBody .pv-chart svg').forEach(svg => {
        const pw = svg.getBoundingClientRect().width - PV_M.l - PV_M.r;
        const xa = PV_M.l + (Math.min(t0, t1) - start) / (end - start) * pw;
        const xb = PV_M.l + (Math.max(t0, t1) - start) / (end - start) * pw;
        const sel = svg.querySelector('.pv-sel');
        sel.setAttribute('x', xa);
        sel.setAttribute('width', Math.max(0, xb - xa));
        sel.setAttribute('visibility', 'visible');
    });
}

function pvWire() {
    document.querySelectorAll('#perfBody .pv-hit').forEach(hit => {
        const svg = hit.ownerSVGElement;
        hit.addEventListener('pointerdown', e => {
            if (pv.liveOn) return;           // live: nothing to zoom into, it moves
            hit.setPointerCapture(e.pointerId);
            pv.drag = { t0: pvTimeAt(svg, e.clientX), x0: e.clientX };
        });
        hit.addEventListener('pointermove', e => {
            if (pv.lv) return;               // live: the body's listener (pvLivePointer)
            const t = pvTimeAt(svg, e.clientX);
            if (pv.drag) { pvShowSel(pv.drag.t0, t); }
            const p = pvNearest(t);
            if (p) pvShowHover(p, e.clientX, e.clientY);
        });
        hit.addEventListener('pointerup', e => {
            const drag = pv.drag;
            pv.drag = null;
            if (!drag) return;
            const t1 = pvTimeAt(svg, e.clientX);
            // A click is not a zoom; only a real drag (and at least 2 buckets) is.
            if (Math.abs(e.clientX - drag.x0) > 8
                    && Math.abs(t1 - drag.t0) >= Math.max(60, (pv.data?.bucket || 30) * 2)) {
                pv.zoom = [Math.min(drag.t0, t1), Math.max(drag.t0, t1)];
                pvHideHover();
                pvLoad();
            } else {
                document.querySelectorAll('#perfBody .pv-sel').forEach(el => el.setAttribute('visibility', 'hidden'));
            }
        });
        hit.addEventListener('pointerleave', () => { if (!pv.lv && !pv.drag) pvHideHover(); });
        hit.addEventListener('dblclick', () => pvResetZoom());
    });
}

// A hidden tab asks for nothing: the live loop and the process list pause
// (no requests, no frames); shown again, they start afresh.
document.addEventListener('visibilitychange', () => {
    if (!pvShown()) return;
    if (document.visibilityState === 'hidden') {
        pvLiveStop();
        clearInterval(pv.procsTimer);
        pv.procsTimer = null;
        return;
    }
    if (pv.liveOn && !pv.lv) pvLiveStart();
    if (!pv.procsTimer) {
        pvLoadProcs();
        pv.procsTimer = setInterval(pvLoadProcs, 3000);
    }
});

// Width follows the window; re-render (not refetch) on resize.
let _pvResize = null;
window.addEventListener('resize', () => {
    if (!pvShown()) return;
    clearTimeout(_pvResize);
    _pvResize = setTimeout(() => {
        pv.width = Math.max(320, ($('perfBody').clientWidth || 900) - 40);
        pvRender();
    }, 150);
});
