// ── Statistics view ──────────────────────────────────────────────────────
// One overlay for the open folder, any subfolder and the whole library —
// /api/stats returns the same shape for all three, so there is one
// renderer. The folder table drills down; the crumbs lead back; the scope
// switch in the header jumps to the whole library and back.
//
// Loading: the base statistics render first; the text analysis (a random
// segment sample, the expensive part) is requested at the same time and
// fills its section when it arrives — no button to press.
//
// Chart rules: one hue (the accent) for magnitudes, status colors only for
// states and always with a label, text in text tokens, hairline grid,
// hover tooltips on every mark — and every value also readable without
// hovering (labels, table).

// Abort before the browser waits forever. A large library's first count
// takes half a minute (the base numbers and the text analysis one after
// the other on the server); then the server keeps them for minutes.
const STV_TIMEOUT = 120000;

const stvState = {
    path: '',            // current scope ('' = the whole open folder)
    library: false,      // true = every root in the database, path ignored
    folder: '',          // the Library's chosen folder (absolute) — wins over path
    folderRel: '',       // the same, relative to the library root (the tree's path)
    rootName: '',        // name of the open folder, for the first crumb
    data: null,          // last response
    abort: null,         // AbortController of the running requests
};


// ── Open / close ─────────────────────────────────────────────────────────

function setupStatsView() {
    const ov = $('statsOverlay');
    if (!ov) return;
    $('stvClose')?.addEventListener('click', closeStatsView);
    $('stvReload')?.addEventListener('click', () => stvLoad(stvState.path, true));
    ov.addEventListener('click', e => { if (e.target === ov) closeStatsView(); });
    document.addEventListener('keydown', e => {
        if (e.key === 'Escape' && ov.classList.contains('visible')) closeStatsView();
    });
    // One tooltip for both statistics views (this one, and the recording's
    // Statistics tab: stats.js wires that box)
    stvWireTips($('statsBody'));
}

// Opens on the folder that is open in the list — the whole library is one
// click away in the header.
// The Library place: the tree on the left (library.js), the folder open in
// the file list chosen — or the whole library when asked for.
function openStatsView(path = '', { library = false } = {}) {
    const ov = $('statsOverlay');
    if (!ov) return;
    ov.classList.add('visible');
    if (library) { stvGo('', true); return; }
    if (typeof Library !== 'undefined') { Library.show(); return; }
    stvGo(path || '', library);
}

function stvGo(path, library = false) {
    stvState.library = !!library;
    if (library) stvState.folder = '';
    stvLoad(stvState.library ? '' : path);
}

/* The Library's tree chose a folder: its statistics. */
function stvGoFolder(abs, rel) {
    stvState.library = false;
    stvState.folder = abs || '';
    stvState.folderRel = rel || '';
    stvLoad('');
}

function stvUrl(path, extra = '') {
    const q = [];
    if (extra) q.push(extra);
    if (stvState.library) q.push('scope=library');
    else if (stvState.folder) q.push(`folder=${encodeURIComponent(stvState.folder)}`);
    else if (path) q.push(`path=${encodeURIComponent(path)}`);
    return '/api/stats' + (q.length ? '?' + q.join('&') : '');
}

function closeStatsView() {
    $('statsOverlay')?.classList.remove('visible');
    stvState.abort?.abort();
    stvState.abort = null;
    stvHideTip();
}


// ── Loading ──────────────────────────────────────────────────────────────

async function stvLoad(path, fresh = false) {
    stvState.path = path;
    stvHead();
    const body = $('statsBody');
    if (!body) return;
    // Refetch keeps the frame: the old render stays, dimmed, until the new one.
    if (fresh || !stvState.data) body.innerHTML = stvSkeleton();
    body.classList.add('loading');

    stvState.abort?.abort();
    const ctrl = new AbortController();
    stvState.abort = ctrl;
    const clock = setTimeout(() => ctrl.abort(), STV_TIMEOUT);

    // All at once: the text analysis and who speaks the most must not wait
    // for the base numbers.
    const textReq = fetch(stvUrl(path, 'text=1'), { signal: ctrl.signal })
        .then(r => r.json()).catch(e => ({ error: e.message || String(e) }));
    stvSpeakersAhead();
    try {
        const d = await (await fetch(stvUrl(path), { signal: ctrl.signal })).json();
        if (ctrl.signal.aborted || stvState.abort !== ctrl) return;
        if (d.error) { body.innerHTML = stvErrors(d.error); stvWire(); return; }
        stvState.data = d;
        const root = (d.scope?.root || '').replace(/\/+$/, '');
        stvState.rootName = root.slice(root.lastIndexOf('/') + 1) || root;
        stvHead();
        body.innerHTML = stvRender(d);
        body.scrollTop = 0;
        stvDrawHistory();
        stvWireHistory();
        stvWire();
        if (typeof Library !== 'undefined') Library.pills();
    } catch (e) {
        // Superseded by a newer load (drill-down, scope toggle) or the view
        // was closed: that load owns the body now — say nothing. Only our
        // own clock aborting a still-current request is a real timeout.
        if (stvState.abort !== ctrl) return;
        clearTimeout(clock);
        body.innerHTML = stvErrors(e.name === 'AbortError' ? 'Timed out' : (e.message || String(e)));
        stvWire();
        return;
    } finally {
        // The dim belongs to the newest load — a superseded one must not
        // lift it while that one is still pending.
        if (stvState.abort === ctrl) body.classList.remove('loading');
    }
    const t = await textReq;
    clearTimeout(clock);
    if (ctrl.signal.aborted || stvState.abort !== ctrl) return;
    stvState.abort = null;
    const target = $('stvText');
    if (target) target.innerHTML = t.error
        ? `<div class="stv-empty">Text analysis failed: ${esc(t.error)}</div>`
        : stvTextHtml(t.text);
}


// ── Header ───────────────────────────────────────────────────────────────

function stvHead() {
    const target = $('stvCrumbs');
    const seg = $('stvScope');
    if (seg) {
        seg.innerHTML = `<button class="${stvState.library ? '' : 'active'}" data-scope="folder"
                             title="The folder open in the list">Folder</button>`
                      + `<button class="${stvState.library ? 'active' : ''}" data-scope="library"
                             title="Every folder in the database">Whole library</button>`;
        seg.querySelectorAll('button').forEach(b => b.onclick = () => {
            const lib = b.dataset.scope === 'library';
            if (lib === stvState.library) return;
            // Back from the whole library: the tree's folder again
            if (!lib && typeof Library !== 'undefined') { stvState.library = false; Library.select(Library.rel); return; }
            stvGo('', lib);
        });
    }
    if (!target) return;
    // In the Library the tree and its crumbs say where we are
    if (stvState.folder) { target.innerHTML = ''; return; }
    if (stvState.library) {
        target.innerHTML = `<span class="stv-crumb current">All roots</span>`;
        return;
    }
    const parts = stvState.path ? stvState.path.split('/') : [];
    let html = `<button class="stv-crumb${parts.length ? '' : ' current'}" data-path="">`
             + `📂 ${esc(stvState.rootName || 'Open folder')}</button>`;
    let soFar = '';
    parts.forEach((t, i) => {
        soFar = soFar ? `${soFar}/${t}` : t;
        html += `<span class="stv-crumb-sep">›</span><button class="stv-crumb`
              + `${i === parts.length - 1 ? ' current' : ''}" data-path="${esc(soFar)}">${esc(t)}</button>`;
    });
    target.innerHTML = html;
    target.querySelectorAll('.stv-crumb:not(.current)').forEach(b =>
        b.addEventListener('click', () => stvGo(b.dataset.path)));
}

function stvWire() {
    const body = $('statsBody');
    body?.querySelectorAll('[data-drill]').forEach(el =>
        el.addEventListener('click', () => {
            // In the Library a subfolder is a folder of the tree
            if (stvState.folder && typeof Library !== 'undefined') {
                Library.show(stvState.folderRel ? `${stvState.folderRel}/${el.dataset.drill}` : el.dataset.drill);
                return;
            }
            stvGo(el.dataset.drill);
        }));
    body?.querySelectorAll('[data-open]').forEach(el =>
        el.addEventListener('click', () => {
            closeStatsView();
            if (typeof openQueueFile === 'function') openQueueFile(el.dataset.open, '');
        }));
    $('stvRetry')?.addEventListener('click', () => stvLoad(stvState.path, true));
}


// ── Formatting ───────────────────────────────────────────────────────────

// 123,456,789 → "123.5M"; exact number in the tooltip where it matters.
function stvCompact(n) {
    n = Number(n) || 0;
    if (n < 10000) return n.toLocaleString();
    if (n < 1e6) return (n / 1e3).toFixed(n < 1e5 ? 1 : 0) + 'k';
    if (n < 1e9) return (n / 1e6).toFixed(1) + 'M';
    return (n / 1e9).toFixed(1) + 'B';
}

// Durations from seconds to years: "7m 12s", "3h 05m", "12,345 h".
function stvDur(sec) {
    sec = Number(sec) || 0;
    if (sec < 48 * 3600) return fmtDuration(sec);
    return `${Math.round(sec / 3600).toLocaleString()} h`;
}

// A human anchor for very long totals: "≈ 1.5 years".
function stvDurHint(sec) {
    const days = (Number(sec) || 0) / 86400;
    if (days >= 365) return `≈ ${(days / 365).toFixed(1)} years`;
    if (days >= 2) return `≈ ${days.toFixed(0)} days`;
    return '';
}

function stvDate(tag, withWeekday) {
    const d = new Date(tag + 'T12:00:00');
    return d.toLocaleDateString(undefined, withWeekday
        ? { weekday: 'short', day: '2-digit', month: 'short' }
        : { day: '2-digit', month: 'short' });
}


// ── Tooltip (one for the whole view) ─────────────────────────────────────
// Marks carry data-tip; lines separated by "\n". Set via textContent —
// file and speaker names are data, never markup.

function stvWireTips(body) {
    const tip = $('stvTip');
    if (!body || !tip) return;
    body.addEventListener('pointerover', e => {
        const el = e.target.closest?.('[data-tip]');
        if (!el) return;
        const [head, ...rest] = el.dataset.tip.split('\n');
        tip.replaceChildren();
        const b = document.createElement('b');
        b.textContent = head;
        tip.append(b);
        rest.forEach(line => {
            const div = document.createElement('div');
            div.textContent = line;
            tip.append(div);
        });
        tip.hidden = false;
    });
    body.addEventListener('pointermove', e => {
        if (tip.hidden) return;
        if (!e.target.closest?.('[data-tip]')) { stvHideTip(); return; }
        const w = tip.offsetWidth, h = tip.offsetHeight;
        tip.style.left = (e.clientX + 14 + w > innerWidth ? e.clientX - 14 - w : e.clientX + 14) + 'px';
        tip.style.top = Math.max(8, e.clientY - h - 12) + 'px';
    });
    body.addEventListener('pointerleave', stvHideTip);
}

function stvHideTip() {
    const tip = $('stvTip');
    if (tip) tip.hidden = true;
}


// ── Building blocks ──────────────────────────────────────────────────────

function stvSkeleton() {
    return `<div class="stv-wait">Counting… a large library takes up to half a minute the first time.</div><div class="stv-grid">
        <div class="stv-card span-2 stv-skel" style="height:190px"></div>
        <div class="stv-card span-2 stv-skel" style="height:240px"></div>
        <div class="stv-card stv-skel" style="height:220px"></div>
        <div class="stv-card stv-skel" style="height:220px"></div></div>`;
}

function stvErrors(text) {
    return `<div class="stv-error">Statistics could not be loaded.<br>${esc(text)}
        <br><button class="stv-btn" id="stvRetry">Retry</button></div>`;
}

function stvCard(title, content, { span = 1, sub = '', id = '' } = {}) {
    return `<section class="stv-card${span > 1 ? ' span-' + span : ''}"${id ? ` id="${id}"` : ''}>
        <header class="stv-card-head"><h3>${esc(title)}</h3>${sub ? `<span class="stv-card-sub">${sub}</span>` : ''}</header>
        ${content}</section>`;
}

// Horizontal bars, one hue. `part` (optional) draws the done share inside
// the bar in full color, the rest as a lighter step of the same hue.
function stvBars(rows, { value = r => r.value, label = r => r.label, right, tip, part } = {}) {
    if (!rows.length) return `<div class="stv-empty">No data</div>`;
    const max = Math.max(...rows.map(value), 1);
    return `<div class="stv-bars">` + rows.map(r => {
        // A sliver keeps a small value visible — none for nothing ("0 0%" had one)
        const w = value(r) > 0 ? Math.max(1.5, value(r) / max * 100) : 0;
        const done = part ? Math.min(1, part(r)) * 100 : 100;
        return `<div class="stv-bar" data-tip="${esc(tip ? tip(r) : `${label(r)}\n${fmtNumber(value(r))}`)}">
            <span class="stv-bar-label">${esc(label(r))}</span>
            <span class="stv-bar-track"><span class="stv-bar-fill" style="width:${w.toFixed(1)}%">
                ${part ? `<span class="stv-bar-done" style="width:${done.toFixed(1)}%"></span>` : ''}</span></span>
            <span class="stv-bar-value">${right ? right(r) : stvCompact(value(r))}</span>
        </div>`;
    }).join('') + `</div>`;
}

function stvKpi(label, value, sub, tip = '') {
    return `<div class="stv-kpi"${tip ? ` data-tip="${esc(tip)}"` : ''}>
        <div class="stv-kpi-label">${esc(label)}</div>
        <div class="stv-kpi-value">${value}</div>
        ${sub ? `<div class="stv-kpi-sub">${sub}</div>` : ''}</div>`;
}


// ── Render ───────────────────────────────────────────────────────────────

function stvRender(d) {
    return `<div class="stv-grid">
        ${stvOverview(d)}
        ${stvState.folder ? stvAttention(d) + stvWhoSpeaks() : ''}
        ${stvHistory(d)}
        ${stvFolders(d)}
        ${stvLengths(d)}
        ${stvTypes(d)}
        ${stvModels(d)}
        ${stvQuality(d)}
        ${stvCard('Words & phrases', `<div id="stvText">${stvTextLoading()}</div>`,
                  { span: 2, sub: 'from a random sample of segments' })}
        ${stvRecords(d)}
        ${stvVector(d)}
    </div>${stvSystem(d)}`;
}

// ── The Library's own cards ─────────────────────────────────────────────

/* What needs doing in this folder, each with the way to do it. */
function stvAttention(d) {
    const p = d.progress || {}, m = d.media || {}, t = d.transcripts || {}, v = d.vector || {}, f = d.features || {};
    const items = [];
    const add = (n, what, acts) => { if (n > 0) items.push({n, what, acts}); };
    const btn = (label, call, title = '') => `<button class="stv-act" onclick="${call}"${title ? ` title="${esc(title)}"` : ''}>${label}</button>`;
    add(p.errors, 'failed', btn('Show them', "Library.showIn({status: 'error'})") + btn('Try all again', 'retryFailedQueue()', 'Every failed or cancelled file of the queue, not only these'));
    add(p.pending, 'not transcribed yet', btn('Transcribe', 'Library.transcribeOpen()'));
    add(p.without_audio, 'without an audio track', btn('Show them', "Library.showIn({has: ['no_audio']})"));
    add(Math.max(0, (p.skipped || 0) - (p.without_audio || 0)), 'skipped by hand', btn('Show them', "Library.showIn({has: ['skipped']})"));
    add(t.empty, 'transcripts without a word', btn('Show them', "Library.showIn({has: ['empty']})"));
    add(f.with_warning, 'with warnings', btn('Show them', "Library.showIn({has: ['warnings']})"));
    add(Math.max(0, (p.total || 0) - (m.indexed || 0)), 'without stored metadata (length, codecs)', btn('Index now', "Library.inList(() => toggleIndexing())", 'Opens the folder in the file list and indexes it'));
    if (v.active) add(Math.max(0, (v.total || 0) - (v.embedded || 0)), 'not searchable by meaning', btn('Embed now', 'stvEmbedVector(this)'));
    const body = items.length
        ? `<table class="stv-table stv-attn"><tbody>${items.map(i => `<tr><td class="num">${fmtNumber(i.n)}</td><td>${esc(i.what)}</td><td class="acts">${i.acts}</td></tr>`).join('')}</tbody></table>`
        : '<div class="stv-empty">Nothing — every file is transcribed, indexed and searchable.</div>';
    return stvCard('Needs attention', body, {span: 2});
}

/* Who speaks the most (named people, from the voice library) — asked
   for after the rest is drawn. */
function stvWhoSpeaks() {
    queueMicrotask(stvLoadSpeakers);
    return stvCard('Who speaks the most', '<div id="stvSpeakers" class="stv-empty">Loading …</div>',
                   {sub: 'named people, by their speech in this folder'});
}

/* Asked for together with the statistics (stvLoad), picked up when the
   card is drawn. */
let stvSpeakersReq = null;

function stvSpeakersAhead() {
    const folder = stvState.folder;
    if (!folder || stvState.library) { stvSpeakersReq = null; return; }
    if (stvSpeakersReq && stvSpeakersReq.folder === folder) return;
    stvSpeakersReq = {folder, p: fetch('/api/library/speakers?folder=' + encodeURIComponent(folder))
        .then(r => r.json())};
    stvSpeakersReq.p.catch(() => { if (stvSpeakersReq && stvSpeakersReq.folder === folder) stvSpeakersReq = null; });
}

async function stvLoadSpeakers() {
    const box = $('stvSpeakers'), folder = stvState.folder;
    if (!box || !folder) return;
    let d;
    try {
        const ahead = stvSpeakersReq && stvSpeakersReq.folder === folder ? stvSpeakersReq.p : null;
        stvSpeakersReq = null;        // the next load asks again (the server caches)
        d = await (ahead || fetch('/api/library/speakers?folder=' + encodeURIComponent(folder)).then(r => r.json()));
    }
    catch (e) { box.textContent = 'Unavailable'; return; }
    if (stvState.folder !== folder || !box.isConnected) return;
    const people = d.people || [];
    if (!people.length) { box.textContent = 'Nobody named speaks here yet — name voices under Voices.'; return; }
    box.className = '';
    box.innerHTML = stvBars(people.map(p => ({label: p.name, value: p.speech_sec,
        right: `${stvDur(p.speech_sec)} · ${plural(p.recordings, 'recording')}`})),
        {right: r => r.right, tip: r => `${r.label}\n${stvDur(r.value)} of speech`});
}

// Hero: the one figure the view leads with (share transcribed), the status
// split under it, and the four headline numbers.
function stvOverview(d) {
    const f = d.progress || {}, t = d.transcripts || {}, m = d.media || {};
    const total = f.total || 0;
    const states = [
        ['done', 'Transcribed', f.done],
        ['pending', 'Open', f.pending],
        ['skipped', 'Skipped', f.skipped],
        ['errors', 'Failed', f.errors],
    ].filter(s => (s[2] || 0) > 0);
    const split = states.map(s => `<span class="stv-split-seg ${s[0]}" style="flex:${s[2]}"
        data-tip="${esc(`${s[1]}\n${fmtNumber(s[2])} file${s[2] === 1 ? '' : 's'} · ${(s[2] / (total || 1) * 100).toFixed(1)}%`)}"></span>`).join('');
    const legend = states.map(s => `<span class="stv-legend-item"><i class="stv-dot ${s[0]}"></i>
        ${esc(s[1])} <b>${fmtNumber(s[2])}</b></span>`).join('');
    const speed = t.speed ? `${t.speed}× real-time` : '';
    return `<section class="stv-card span-2 stv-hero">
        <div class="stv-hero-main">
            <div class="stv-hero-label">Transcribed</div>
            <div class="stv-hero-value">${(f.pct || 0).toFixed(1)}<span>%</span></div>
            <div class="stv-hero-sub">${fmtNumber(f.done)} of ${fmtNumber(total)} file${total === 1 ? '' : 's'}</div>
            <div class="stv-split">${split}</div>
            <div class="stv-legend">${legend}</div>
        </div>
        <div class="stv-kpis">
            ${stvKpi('Audio transcribed', stvDur(t.duration), stvDurHint(t.duration),
                     `${fmtNumber(Math.round((t.duration || 0) / 3600))} hour${Math.round((t.duration || 0) / 3600) === 1 ? '' : 's'}\n${fmtNumber(t.total)} transcript${t.total === 1 ? '' : 's'}`)}
            ${stvKpi('Words', stvCompact(t.words),
                     // no rate below a minute of audio (the server sends 0)
                     t.wpm ? `${fmtNumber(t.wpm)} per minute of speech` : '',
                     `${fmtNumber(t.words)} word${t.words === 1 ? '' : 's'}\n${fmtNumber(t.segments)} segment${t.segments === 1 ? '' : 's'}`)}
            ${stvKpi('GPU time', stvDur(t.compute_time), speed,
                     `Processing time over all transcripts\naverage ${fmtDuration(t.avg_compute_time)} per file`)}
            ${stvKpi('Library size', fmtBytes(m.size), `${stvCompact(m.videos)} video · ${stvCompact(m.audio_only)} audio`,
                     `${fmtNumber(m.videos)} video file${m.videos === 1 ? '' : 's'}\n${fmtNumber(m.audio_only)} audio-only file${m.audio_only === 1 ? '' : 's'}`)}
        </div>
    </section>`;
}

// ── History: the whole timeline ──────────────────────────────────────────
// One card, two readings of the same daily series: a column chart with a
// 7-day average (range and metric switchable) and a contribution calendar
// — weeks as columns, weekdays as rows, like a commit overview. The server
// sends only days with activity; the gaps are filled here.

// A 7-day average below ten keeps a decimal: one transcript a week is 0.1, not 0
const stvAvg = v => fmtNumber(v >= 10 ? Math.round(v) : Math.round(v * 10) / 10);
const STV_METRICS = {
    cnt:      { label: 'Transcripts', fmt: stvAvg, axis: v => stvCompact(v) },
    // Not rounded on the axis: its middle tick of one hour read "1 h"
    duration: { label: 'Audio hours', fmt: v => stvDur(v), axis: v => `${stvCompact(v / 3600)} h` },
    words:    { label: 'Words',       fmt: stvAvg, axis: v => stvCompact(v) },
};
const STV_RANGES = [['30 d', 30], ['90 d', 90], ['1 y', 365], ['All', 0]];
stvState.metric = stvState.metric || 'cnt';
stvState.range = stvState.range ?? 90;

const stvDayKey = d => `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`;

// Gap-free list of days from `from` to today, zeros where nothing happened.
function stvDays(history, from) {
    const by = Object.fromEntries(history.map(h => [h.tag, h]));
    const out = [];
    const d = new Date(from + 'T12:00:00'), end = new Date();
    end.setHours(12, 0, 0, 0);
    for (; d <= end; d.setDate(d.getDate() + 1)) {
        const tag = stvDayKey(d);
        out.push(by[tag] || { tag, cnt: 0, duration: 0, words: 0 });
    }
    return out;
}

function stvStreaks(days) {
    let best = 0, run = 0, bestEnd = null;
    days.forEach(x => {
        run = x.cnt ? run + 1 : 0;
        if (run > best) { best = run; bestEnd = x.tag; }
    });
    // Current streak: today may still be empty — count from yesterday then
    let cur = 0;
    for (let i = days.length - 1; i >= 0; i--) {
        if (days[i].cnt) cur++;
        else if (i === days.length - 1) continue;
        else break;
    }
    return { best, bestEnd, cur };
}

function stvHistory(d) {
    const h = d.history || [];
    if (!h.length) return stvCard('History', `<div class="stv-empty">No transcripts finished yet in this scope.</div>`, { span: 2 });
    const days = stvDays(h, h[0].tag);
    const active = h.length;
    const best = h.reduce((b, x) => x.cnt > b.cnt ? x : b, h[0]);
    const st = stvStreaks(days);
    const total = h.reduce((s, x) => s + x.cnt, 0);
    const figs = `<div class="stv-hist-figs">
        ${stvKpi('Active days', fmtNumber(active), `of ${fmtNumber(days.length)} since ${esc(stvDate(h[0].tag))}`)}
        ${stvKpi('Best day', fmtNumber(best.cnt), esc(stvDate(best.tag, true)))}
        ${stvKpi('Longest streak', `${fmtNumber(st.best)} <small>day${st.best === 1 ? '' : 's'}</small>`, st.bestEnd ? `ended ${esc(stvDate(st.bestEnd))}` : '')}
        ${stvKpi('Current streak', `${fmtNumber(st.cur)} <small>day${st.cur === 1 ? '' : 's'}</small>`, `⌀ ${fmtNumber(Math.round(total / active))} per active day`)}
    </div>`;
    const metricSeg = Object.entries(STV_METRICS).map(([k, m]) =>
        `<button class="${stvState.metric === k ? 'active' : ''}" data-metric="${k}">${m.label}</button>`).join('');
    const rangeSeg = STV_RANGES.map(([l, n]) =>
        `<button class="${stvState.range === n ? 'active' : ''}" data-range="${n}">${l}</button>`).join('');
    return stvCard('History', figs + `
        <div class="stv-hist-controls">
            <div class="stv-seg" id="stvMetricSeg">${metricSeg}</div>
            <div class="stv-seg" id="stvRangeSeg">${rangeSeg}</div>
            <span class="stv-hist-legend"><i class="stv-key done"></i> per day <i class="stv-key-line"></i> 7-day average</span>
        </div>
        <div class="stv-chart" id="stvHistChart"></div>
        <div class="stv-sublabel" style="margin-top:20px">Calendar</div>
        <div class="stv-cal-wrap" id="stvCalendar"></div>`,
        { span: 2, sub: 'finished transcripts, day by day' });
}

function stvWireHistory() {
    $('stvMetricSeg')?.querySelectorAll('button').forEach(b => b.onclick = () => {
        stvState.metric = b.dataset.metric;
        $('stvMetricSeg').querySelectorAll('button').forEach(x => x.classList.toggle('active', x === b));
        stvDrawHistory();
    });
    $('stvRangeSeg')?.querySelectorAll('button').forEach(b => b.onclick = () => {
        stvState.range = +b.dataset.range;
        $('stvRangeSeg').querySelectorAll('button').forEach(x => x.classList.toggle('active', x === b));
        stvDrawHistory();
    });
}

function stvDrawHistory() {
    stvDrawHistChart();
    stvDrawCalendar();
}

function stvDrawHistChart() {
    const box = $('stvHistChart');
    const h = stvState.data?.history || [];
    if (!box || !h.length) return;
    const metric = STV_METRICS[stvState.metric];
    let days = stvDays(h, h[0].tag);
    if (stvState.range) {
        const from = new Date(); from.setDate(from.getDate() - stvState.range + 1);
        days = stvDays(h, stvDayKey(from));
    }
    const val = x => x[stvState.metric] || 0;
    // 7-day trailing average (over the full series, so the first days of a
    // range still average over real earlier days)
    const all = stvDays(h, h[0].tag);
    const idx = Object.fromEntries(all.map((x, i) => [x.tag, i]));
    const avg = days.map(x => {
        const i = idx[x.tag];
        if (i == null) return 0;
        const w = all.slice(Math.max(0, i - 6), i + 1);
        return w.reduce((s, y) => s + val(y), 0) / 7;
    });
    const W = Math.max(300, box.clientWidth), H = 190;
    const left = 52, right = 6, top = 10, bottom = 24;
    const pw = W - left - right, ph = H - top - bottom;
    // Round the scale in the unit shown (hours, not seconds) for clean ticks
    const unit = stvState.metric === 'duration' ? 3600 : 1;
    const max = Math.max(...days.map(val), ...avg, unit) / unit;
    const p = 10 ** Math.floor(Math.log10(max));
    const ymax = Math.ceil(max / p) * p * unit;
    const slot = pw / days.length;
    const bw = Math.max(1, Math.min(18, slot * 0.72));
    const y = v => top + ph * (1 - v / ymax);
    let svg = [0, ymax / 2, ymax].map(v =>
        `<line class="stv-grid-line" x1="${left}" x2="${W - right}" y1="${y(v)}" y2="${y(v)}"/>`
      + `<text class="stv-axis" x="${left - 8}" y="${y(v) + 4}" text-anchor="end">${metric.axis(v)}</text>`).join('');
    // Month ticks. A label only where the one before has room (11 px
    // mono): the first day's "Oct 23" ran into "Nov" right after it — it
    // gives way — and years of months ran into each other
    const ticks = [];   // [x, label, the first day's]
    days.forEach((x, i) => {
        if (!x.tag.endsWith('-01') && i) return;
        const cx = left + slot * i;
        svg += `<line class="stv-grid-line" x1="${cx}" x2="${cx}" y1="${top}" y2="${top + ph}"/>`;
        const label = new Date(x.tag + 'T12:00:00').toLocaleDateString(undefined,
            days.length > 400 ? { month: 'short', year: '2-digit' } : { month: 'short', day: i ? undefined : '2-digit' });
        const prev = ticks[ticks.length - 1];
        if (prev && cx - prev[0] < prev[1].length * 7 + 8) {
            if (!prev[2]) return;
            ticks.pop();
        }
        ticks.push([cx, label, i === 0]);
    });
    for (const [cx, label] of ticks)
        svg += `<text class="stv-axis" x="${cx + 3}" y="${H - 6}">${esc(label)}</text>`;
    days.forEach((x, i) => {
        const cx = left + slot * i + slot / 2;
        const v = val(x), hgt = v ? Math.max(2, ph * v / ymax) : 0;
        const r = Math.min(3, bw / 2, hgt);
        const tip = `${stvDate(x.tag, true)} ${x.tag.slice(0, 4)}\n${fmtNumber(x.cnt)} transcript${x.cnt === 1 ? '' : 's'} · `
                  + `${stvDur(x.duration)} of audio · ${stvCompact(x.words)} word${x.words === 1 ? '' : 's'}\n7-day average: ${metric.fmt(avg[i])}`;
        svg += `<g data-tip="${esc(tip)}"><rect class="stv-hit" x="${cx - slot / 2}" y="${top}" width="${slot}" height="${ph}"/>`
             + (hgt ? (bw > 4
                 ? `<path class="stv-col soft" d="M${cx - bw / 2},${top + ph} V${top + ph - hgt + r} q0,-${r} ${r},-${r} H${cx + bw / 2 - r} q${r},0 ${r},${r} V${top + ph} Z"/>`
                 : `<rect class="stv-col soft" x="${cx - bw / 2}" y="${top + ph - hgt}" width="${bw}" height="${hgt}"/>`) : '')
             + `</g>`;
    });
    const line = avg.map((v, i) => `${i ? 'L' : 'M'}${(left + slot * i + slot / 2).toFixed(1)},${y(v).toFixed(1)}`).join('');
    svg += `<path class="stv-avg" d="${line}"/>`;
    box.innerHTML = `<svg width="${W}" height="${H}">${svg}</svg>`;
}

// Contribution calendar: up to one year, weeks as columns (Monday first).
function stvDrawCalendar() {
    const box = $('stvCalendar');
    const h = stvState.data?.history || [];
    if (!box || !h.length) return;
    const val = x => x[stvState.metric] || 0;
    const metric = STV_METRICS[stvState.metric];
    const today = new Date(); today.setHours(12, 0, 0, 0);
    const first = new Date(h[0].tag + 'T12:00:00');
    const yearAgo = new Date(today); yearAgo.setDate(yearAgo.getDate() - 364);
    // At least 26 weeks so a young library still reads as a calendar
    const halfYear = new Date(today); halfYear.setDate(halfYear.getDate() - 7 * 26);
    let start = first > yearAgo ? first : yearAgo;
    if (start > halfYear) start = halfYear;
    start.setDate(start.getDate() - ((start.getDay() + 6) % 7));   // back to Monday
    const days = stvDays(h, stvDayKey(start));
    // Levels: quartiles of the active days (like a commit graph), so one
    // record day does not wash every other day into the lowest level.
    const vals = days.map(val).filter(v => v > 0).sort((a, b) => a - b);
    const q = f => vals.length ? vals[Math.min(vals.length - 1, Math.floor(vals.length * f))] : 0;
    const cuts = [q(0.25), q(0.5), q(0.75)];
    const level = v => !v ? 0 : v <= cuts[0] ? 1 : v <= cuts[1] ? 2 : v <= cuts[2] ? 3 : 4;
    const weeks = Math.ceil(days.length / 7);
    const left = 30, top = 18, gap = 3;
    // Cells grow with the card (13–28 px); a full year scrolls instead
    const cell = Math.max(13, Math.min(28, Math.floor((box.clientWidth - left) / weeks) - gap));
    const W = left + weeks * (cell + gap), H = top + 7 * (cell + gap);
    let svg = '';
    ['Mon', '', 'Wed', '', 'Fri', '', ''].forEach((l, r) => {
        if (l) svg += `<text class="stv-axis" x="0" y="${top + r * (cell + gap) + cell - 2}">${l}</text>`;
    });
    let lastMonth = -1;
    const months = [];   // [week, label]
    days.forEach((x, i) => {
        const w = Math.floor(i / 7), r = i % 7;
        const dt = new Date(x.tag + 'T12:00:00');
        // A month label where a month begins. One that would sit on top of
        // the previous label takes its place: that one was the partial
        // first month — "Apr" had pushed out "May"
        if (r === 0 && dt.getMonth() !== lastMonth) {
            lastMonth = dt.getMonth();
            if (months.length && w - months[months.length - 1][0] < 3) months.pop();
            months.push([w, dt.toLocaleDateString(undefined, { month: 'short' })]);
        }
        const future = dt > today;
        if (future) return;
        const tip = `${stvDate(x.tag, true)} ${x.tag.slice(0, 4)}\n` + (x.cnt
            ? `${fmtNumber(x.cnt)} transcript${x.cnt === 1 ? '' : 's'} · ${stvDur(x.duration)} · ${stvCompact(x.words)} word${x.words === 1 ? '' : 's'}`
            : 'nothing transcribed');
        svg += `<rect class="stv-cal l${level(val(x))}" x="${left + w * (cell + gap)}" y="${top + r * (cell + gap)}"
                      width="${cell}" height="${cell}" rx="3" data-tip="${esc(tip)}"/>`;
    });
    for (const [w, label] of months)
        svg += `<text class="stv-axis" x="${left + w * (cell + gap)}" y="11">${esc(label)}</text>`;
    const legend = `<div class="stv-cal-legend"><span>less</span>${[0, 1, 2, 3, 4].map(l =>
        `<svg width="13" height="13"><rect class="stv-cal l${l}" width="13" height="13" rx="3"/></svg>`).join('')}<span>more</span>
        <span class="stv-dim" style="margin-left:10px">${esc(metric.label.toLowerCase())} per day</span></div>`;
    box.innerHTML = `<div class="stv-cal-scroll"><svg width="${W}" height="${H}">${svg}</svg></div>${legend}`;
    // Newest weeks on screen when the year does not fit
    const sc = box.querySelector('.stv-cal-scroll');
    sc.scrollLeft = sc.scrollWidth;
}

let _stvResize = null;
window.addEventListener('resize', () => {
    if (!$('statsOverlay')?.classList.contains('visible')) return;
    clearTimeout(_stvResize);
    _stvResize = setTimeout(stvDrawHistory, 120);
});

function stvFolders(d) {
    const u = d.subfolders || [];
    // Only files directly in this folder: nothing to break down or click into
    if (!u.length || u.every(o => o.name === '.')) return '';
    const maxDur = Math.max(...u.map(o => o.duration || 0), 1);
    const rows = u.map(o => {
        const pct = o.total ? o.done / o.total * 100 : 0;
        const here = o.name === '.';
        const name = here ? '(files directly here)' : o.name;
        return `<tr class="${here ? '' : 'link'}"${here ? '' : ` data-drill="${esc(o.path)}"`}>
            <td class="name">${here ? '' : '<span class="stv-folder-ico">📁</span>'}${esc(name)}</td>
            <td class="num">${fmtNumber(o.total)}</td>
            <td class="prog" data-tip="${esc(`${name}\n${fmtNumber(o.done)} of ${fmtNumber(o.total)} transcribed`)}">
                <span class="stv-mini"><span style="width:${pct.toFixed(1)}%" class="${pct >= 99.95 ? 'full' : ''}"></span></span>
                <span class="stv-mini-val">${pct >= 99.95 ? '✓' : pct.toFixed(0) + '%'}</span></td>
            <td class="prog" data-tip="${esc(`${name}\n${fmtNumber(Math.round((o.duration || 0) / 3600))} h of audio`)}">
                <span class="stv-mini soft"><span style="width:${Math.max(o.duration ? 1 : 0, (o.duration || 0) / maxDur * 100).toFixed(1)}%"></span></span>
                <span class="stv-mini-val wide">${stvDur(o.duration)}</span></td>
            <td class="num">${stvCompact(o.words)}</td>
            <td class="num">${o.size ? fmtBytes(o.size) : ''}</td>
            ${stvVectorCell(o)}
        </tr>`;
    }).join('');
    const vec = u.some(o => o.vector_total != null);
    return stvCard(d.scope?.global ? 'Folders' : 'Subfolders', `<div class="stv-table-wrap"><table class="stv-table">
        <thead><tr><th>Name</th><th class="num">Files</th><th>Transcribed</th><th>Audio</th>
            <th class="num">Words</th><th class="num">Size</th>${vec ? '<th>Searchable</th>' : ''}</tr></thead>
        <tbody>${rows}</tbody></table></div>`,
        { span: 2, sub: 'click a folder to look inside' });
}

// RAG off: no column at all — an empty or 0% column would read as "nothing embedded".
function stvVectorCell(o) {
    if (o.vector_total == null) return '';
    if (!o.vector_total) return '<td class="num stv-dim">—</td>';
    const pct = o.vector_indexed / o.vector_total * 100;
    return `<td class="prog" data-tip="${esc(`${fmtNumber(o.vector_indexed)} of ${fmtNumber(o.vector_total)} embedded`)}">
        <span class="stv-mini"><span style="width:${pct.toFixed(0)}%"></span></span>
        <span class="stv-mini-val">${pct.toFixed(0)}%</span></td>`;
}

function stvLengths(d) {
    const b = d.duration_buckets || [];
    if (!b.length) return '';
    const total = b.reduce((s, x) => s + x.cnt, 0) || 1;
    return stvCard('Recording length', stvBars(b, {
        value: r => r.cnt,
        right: r => `${stvCompact(r.cnt)} <span class="stv-dim">${(r.cnt / total * 100).toFixed(0)}%</span>`,
        tip: r => `${r.label}\n${fmtNumber(r.cnt)} recording${r.cnt === 1 ? '' : 's'} · ${stvDur(r.duration)} of audio`,
    }) + stvLengthNote(b), { sub: 'transcripts by duration' });
}

// The one sentence the buckets tell: where the files are vs. where the hours are.
function stvLengthNote(b) {
    const files = b.reduce((s, x) => s + x.cnt, 0), secs = b.reduce((s, x) => s + (x.duration || 0), 0);
    if (!files || !secs) return '';
    const most = b.reduce((m, x) => x.cnt > m.cnt ? x : m, b[0]);
    const heavy = b.reduce((m, x) => (x.duration || 0) > (m.duration || 0) ? x : m, b[0]);
    const f = most.cnt / files * 100, h = (heavy.duration || 0) / secs * 100;
    return `<div class="stv-foot">${f.toFixed(0)}% of the files are ${esc(most.label)}`
        + (heavy === most ? ` and hold ${h.toFixed(0)}% of the audio.`
                          : `; ${h.toFixed(0)}% of the audio hours sit in recordings of ${esc(heavy.label)}.`) + `</div>`;
}

function stvTypes(d) {
    const x = (d.extensions || []).slice(0, 8);
    if (!x.length) return '';
    return stvCard('File types', stvBars(x, {
        value: r => r.cnt,
        label: r => r.name.replace(/^\./, ''),
        part: r => r.cnt ? (r.done || 0) / r.cnt : 0,
        right: r => `${stvCompact(r.cnt)} <span class="stv-dim">${fmtBytes(r.size)}</span>`,
        tip: r => `${r.name}\n${fmtNumber(r.cnt)} file${r.cnt === 1 ? '' : 's'} · ${fmtBytes(r.size)}\n${fmtNumber(r.done || 0)} transcribed (${r.cnt ? ((r.done || 0) / r.cnt * 100).toFixed(0) : 0}%)`,
    }) + `<div class="stv-foot"><i class="stv-key done"></i> transcribed <i class="stv-key rest"></i> open</div>`,
    { sub: 'files by format' });
}

function stvModels(d) {
    const models = d.models || [], langs = d.languages || [];
    const speedTip = r => `${r.name}\n${fmtNumber(r.cnt)} transcript${r.cnt === 1 ? '' : 's'} · ${stvDur(r.duration)} of audio`
        + (r.speed ? `\n${r.speed}× real-time` : '');
    return stvCard('Models & languages',
        `<div class="stv-sublabel">Model</div>` + stvBars(models, {
            value: r => r.cnt, label: r => r.name, tip: speedTip,
            right: r => `${stvCompact(r.cnt)}${r.speed ? ` <span class="stv-dim">${r.speed}×</span>` : ''}`,
        })
        + `<div class="stv-sublabel">Language</div>` + stvBars(langs.slice(0, 6), {
            value: r => r.cnt, label: r => (r.name || '?').toUpperCase(), tip: speedTip,
        }), { sub: 'what transcribed it' });
}

// Processing features as shares of all transcripts — one meter each.
function stvQuality(d) {
    const f = d.features || {}, t = d.transcripts || {};
    const n = t.total || 0;
    if (!n) return '';
    const row = (label, v, tip, warn) => {
        const pct = v / n * 100;
        return `<div class="stv-meter" data-tip="${esc(`${label}\n${fmtNumber(v)} of ${fmtNumber(n)} transcript${n === 1 ? '' : 's'} · ${pct.toFixed(1)}%\n${tip}`)}">
            <div class="stv-meter-top"><span>${esc(label)}</span><b>${pct < 1 && v ? '<1' : pct.toFixed(0)}%</b></div>
            <span class="stv-meter-track"><span class="${warn ? 'warn' : ''}" style="width:${Math.max(v ? 1 : 0, pct).toFixed(1)}%"></span></span>
            <div class="stv-meter-sub">${fmtNumber(v)} transcript${v === 1 ? '' : 's'}</div></div>`;
    };
    return stvCard('Processing', `<div class="stv-meters">
        ${row('Vocal isolation', f.vocal_iso || 0, 'Voice separated from music and noise before recognition')}
        ${row('Speaker diarization', f.diarized || 0, `average ${f.avg_speaker || 0} speaker${f.avg_speaker === 1 ? '' : 's'} where detected`)}
        ${row('With warnings', f.with_warning || 0, 'Something was retried or looked suspicious — see the incidents', true)}
    </div>
    <div class="stv-foot">Average file: ${fmtDuration(t.avg_duration)} of audio, ${fmtNumber(t.avg_segments)} segment${t.avg_segments === 1 ? '' : 's'},
        ${fmtNumber(t.avg_words)} word${t.avg_words === 1 ? '' : 's'}, ${fmtDuration(t.avg_compute_time)} to process.</div>`,
    { sub: 'share of transcripts' });
}

function stvRecords(d) {
    const r = d.records || [];
    if (!r.length) return '';
    const val = x => {
        if (x.column_name === 'duration_sec' || x.column_name === 'processing_secs') return stvDur(x.value);
        if (x.column_name === 'wpm') return `${fmtNumber(x.value)} <small>wpm</small>`;
        return stvCompact(x.value);
    };
    return stvCard('Records', `<div class="stv-records">` + r.map(x => {
        const name = (x.path || '').split('/').pop();
        return `<button class="stv-record" data-open="${esc(x.path)}" data-tip="${esc(`${x.title}\n${x.path}\nClick to open`)}">
            <span class="stv-record-title">${esc(x.title)}</span>
            <span class="stv-record-value">${val(x)}</span>
            <span class="stv-record-name">${esc(name)}</span></button>`;
    }).join('') + `</div>`, { span: 2, sub: 'click to open the file' });
}

function stvVector(d) {
    const v = d.vector || {};
    if (!v.active) {
        // "Not enabled" stood here while the search used a built index: the
        // switch behind it only shows the coverage
        return stvCard('Semantic search', `<div class="stv-muted-card">${v.hidden
            ? `The coverage isn't shown — switch on <b>Settings → Search → Semantic search →
               Show coverage</b>. The search uses what is embedded either way.`
            : `Not set up. With <b>Settings → Search</b> the transcripts get embedded — then the
               search also finds rephrasings, not only the exact words.`}</div>`, { span: 2 });
    }
    const g = v.total || 1;
    const never = (v.open_count || 0) - (v.stale || 0);
    return stvCard('Semantic search', `<div class="stv-vec">
        <div><div class="stv-hero-value small">${(v.pct || 0).toFixed(1)}<span>%</span></div>
             <div class="stv-kpi-sub">searchable</div></div>
        ${stvKpi('Embedded', fmtNumber(v.embedded), `of ${fmtNumber(v.total)} transcript${v.total === 1 ? '' : 's'}`)}
        ${stvKpi('Chunks', stvCompact(v.chunks), 'sections in the store')}
        ${stvKpi('Pending', fmtNumber(v.open_count), v.stale ? `${fmtNumber(v.stale)} stale` : 'never embedded')}
    </div>
    <div class="stv-split">
        ${v.embedded ? `<span class="stv-split-seg done" style="flex:${v.embedded / g}" data-tip="Embedded\n${fmtNumber(v.embedded)}"></span>` : ''}
        ${v.stale ? `<span class="stv-split-seg skipped" style="flex:${v.stale / g}" data-tip="Stale — will be re-embedded\n${fmtNumber(v.stale)}"></span>` : ''}
        ${never > 0 ? `<span class="stv-split-seg rest" style="flex:${never / g}" data-tip="Never embedded\n${fmtNumber(never)}"></span>` : ''}
    </div>
    ${v.open_count ? `<button class="stv-btn" onclick="stvEmbedVector(this)">🧠 Embed pending now</button>
        <span class="stv-dim" style="font-size:0.74rem;margin-left:8px">runs in the background; progress under Settings</span>`
        : `<div class="stv-foot" style="color:var(--green)">✓ Everything in this scope is searchable.</div>`}`,
    { span: 2 });
}

// Triggered here, tracked in the settings — one place draws the progress.
async function stvEmbedVector(btn) {
    const label = btn.textContent;
    btn.disabled = true; btn.textContent = '⏳ starting…';
    try {
        const r = await fetch('/api/rag/build?force=0');
        // A refused start answers with JSON (no embedding model, a run
        // already going) — say why, not just the status code
        if (!r.ok) throw new Error((await r.json().catch(() => ({}))).error || 'HTTP ' + r.status);
        Toast.info('Embedding running — progress under Settings → Search');
    } catch (e) {
        Toast.error('Failed to start: ' + e.message);
    } finally {
        btn.disabled = false; btn.textContent = label;
    }
}


// ── Text analysis ────────────────────────────────────────────────────────

function stvTextLoading() {
    return `<div class="stv-text-grid">
        <div class="stv-skel" style="height:260px;border-radius:8px"></div>
        <div class="stv-skel" style="height:260px;border-radius:8px"></div></div>
        <div class="stv-foot">Reading a sample of the segments …</div>`;
}

function stvTextHtml(t) {
    if (!t || !t.sample) return `<div class="stv-empty">No segments in this scope</div>`;
    const words = (t.top_words || []).slice(0, 24);
    const half = Math.ceil(words.length / 2);
    const wordBars = list => stvBars(list, {
        value: r => r.cnt, label: r => r.word,
        right: r => fmtNumber(r.cnt), tip: r => `${r.word}\n${fmtNumber(r.cnt)}× in the sample`,
    });
    const maxPhrase = Math.max(...(t.top_bigrams || []).map(b => b.cnt), 1);
    const phrases = (t.top_bigrams || []).length
        ? `<div class="stv-phrases">` + t.top_bigrams.map(b =>
            `<span class="stv-phrase" style="--w:${(0.35 + 0.65 * b.cnt / maxPhrase).toFixed(2)}"
                   data-tip="${esc(`${b.phrase}\n${fmtNumber(b.cnt)}× in the sample`)}">${esc(b.phrase)}<b>${fmtNumber(b.cnt)}</b></span>`
          ).join('') + `</div>`
        : `<div class="stv-empty">No recurring phrases</div>`;
    const spk = (t.top_speaker || []);
    const spkTotal = spk.reduce((s, x) => s + x.cnt, 0) || 1;
    const speakers = spk.length ? stvBars(spk.slice(0, 8), {
        value: r => r.cnt,
        label: r => (typeof getSpeakerName === 'function' ? getSpeakerName(r.name) : r.name),
        right: r => `${(r.cnt / spkTotal * 100).toFixed(0)}%`,
        tip: r => `${r.name}\n${fmtNumber(r.cnt)} segment${r.cnt === 1 ? '' : 's'} in the sample`,
    }) : '';
    return `<div class="stv-text-grid">
            <div><div class="stv-sublabel">Most frequent words</div>${wordBars(words.slice(0, half))}</div>
            <div><div class="stv-sublabel">&nbsp;</div>${wordBars(words.slice(half))}</div>
        </div>
        <div class="stv-sublabel" style="margin-top:18px">Recurring phrases</div>${phrases}
        ${speakers ? `<div class="stv-sublabel" style="margin-top:18px">Speaker labels in the sample</div>${speakers}` : ''}
        <div class="stv-foot">Sample of ${fmtNumber(t.sample)} segment${t.sample === 1 ? '' : 's'}, ${fmtNumber(t.unique)} distinct word${t.unique === 1 ? '' : 's'};
            filler and function words are left out. Counts refer to the sample, not to every transcript.</div>`;
}

function stvSystem(d) {
    const s = d.system || {};
    const parts = [
        s.cpu ? esc(s.cpu) : '',
        s.gpus?.length ? esc(s.gpus.join(', ')) : '',
        s.ram_mb ? `${fmtBytes(s.ram_mb * 1024 * 1024)} RAM` : '',
        s.db_bytes ? `database ${fmtBytes(s.db_bytes)}` : '',
    ].filter(Boolean);
    return parts.length ? `<div class="stv-system">${parts.join(' · ')}</div>` : '';
}

document.addEventListener('DOMContentLoaded', setupStatsView);
