// ── Hit map ──
// Treemap of the search results, modeled after WinDirStat: area = amount.
// There it's disk space, here it's the hit count — the question is the same
// ("where is most of it?") and the answer is built the same way as a result.
//
// Zooming goes through the folder structure down to the file:
//   Folder → … → File
// There the map ends and hands off to timeline + transcript: one more tile
// level would just repeat the timeline; what's wanted at that point is the
// text. Back via breadcrumb path, right-click, or backspace.

const smState = {
    root: null,        // tree built from the facets
    path: [],          // current path as a node chain from root
    kids: [],          // children of the current level — clicks go through
                       // the index into this array, not the name: two hits
                       // in the same second have the same timestamp and
                       // wouldn't be distinguishable by name.
    view: 'map',       // folder levels: 'map' = treemap, 'list' = list
    txMode: 'full',    // file level: 'full' = whole transcript, 'hits' = only hits
    // Map instead of an object literal: a file named "constructor" or
    // "toString" would otherwise hit an inherited prototype key, the guard
    // would think it was loaded, and the cache would return a function.
    hits: new Map(),   // cache: rel_path → loaded hits
    tx: new Map(),     // cache: rel_path → loaded transcript
    active: -1,        // bar highlighted in the timeline (-1 = none)
    capped: 0,       // how many files the cap left out
    total: 0,          // hit count across ALL files (including the capped ones)
    // Two separate counters: renderSeq discards outdated render passes,
    // seq discards outdated fetches. A shared counter doesn't work — the
    // fetch would increment it, and the waiting render would then consider
    // its OWN response outdated and never draw.
    renderSeq: 0,
    seq: 0,
};

function setupSearchMap() {
    const ov = document.getElementById('smOverlay');
    if (!ov) return;

    // Escape/Backspace: one level back, close only at the top level.
    document.addEventListener('keydown', (e) => {
        if (!ov.classList.contains('visible')) return;
        if (e.key === 'Escape' || e.key === 'Backspace') {
            e.preventDefault();
            e.stopPropagation();
            if (smState.path.length > 1) smUp();
            else closeSearchMap();
        }
    }, true);   // capture: before the search view's escape handler

    // Right-click on the map = one level back (like in file managers)
    document.getElementById('smCanvas').addEventListener('contextmenu', (e) => {
        e.preventDefault();
        if (smState.path.length > 1) smUp();
    });

    // Re-layout when the window is resized — a treemap in fixed pixels
    // would otherwise look wrong after a resize.
    if (typeof ResizeObserver !== 'undefined') {
        new ResizeObserver(() => {
            if (ov.classList.contains('visible')) smRender();
        }).observe(document.getElementById('smCanvas'));
    }
}

// ── Open / close ───────────────────────────────────────────────────────────

function closeSearchMap() {
    document.getElementById('smOverlay').classList.remove('visible');
    smState.seq++;
    smState.renderSeq++;
}

function smUp() {
    if (smState.path.length > 1) { smState.path.pop(); smRender(); }
}

function smGoLevel(i) {
    smState.path = smState.path.slice(0, i + 1);
    smRender();
}

// ── Tree from the facets ───────────────────────────────────────────────────

// A hit's full path: its root and its name relative to that root
function _smAbs(f) {
    const dir = String(f.file_dir || '').replace(/\\/g, '/').replace(/\/+$/, '');
    const name = String(f.filename).replace(/\\/g, '/');
    return dir ? `${dir}/${name}` : name;
}

// The folder every hit lies in ("" for "/"; null when a hit has no root)
function _smCommonDir(paths) {
    if (!paths.length || paths.some(a => !/^(\/|[A-Za-z]:\/)/.test(a))) return null;
    let common = paths[0].split('/').slice(0, -1);
    for (const a of paths) {
        const dirs = a.split('/').slice(0, -1);
        let i = 0;
        while (i < common.length && i < dirs.length && common[i] === dirs[i]) i++;
        common = common.slice(0, i);
    }
    return common.join('/');
}

function smBuildTree(files) {
    const root = { name: 'All hits', type: 'dir', hits: 0, kids: new Map() };
    // Folders from the one folder every hit lies in: a rel_path is relative
    // to the folder open when the file was transcribed, and the tree mixed
    // those bases — a file of "edge cases" sat at the top beside "samples"
    const common = _smCommonDir(files.map(_smAbs));
    files.forEach(f => {
        const a = _smAbs(f);
        const parts = (common !== null && a.startsWith(common + '/') ? a.slice(common.length + 1)
                       : String(f.filename).replace(/\\/g, '/')).split('/');
        const fname = parts.pop();
        let node = root;
        node.hits += Number(f.hits) || 0;
        parts.forEach(p => {
            if (!p) return;
            if (!node.kids.has(p)) {
                node.kids.set(p, { name: p, type: 'dir', hits: 0, kids: new Map() });
            }
            node = node.kids.get(p);
            node.hits += Number(f.hits) || 0;
        });
        node.kids.set(fname, {
            name: fname, type: 'file', hits: Number(f.hits) || 0,
            rel: f.filename, dir: f.file_dir || '',
            duration: f.duration || 0, histogram: f.histogram || [],
        });
    });
    return root;
}

// Children of the current node. Only folders have any — for a file the
// zoom ends, and the timeline and transcript take over there.
function smChildren(node) {
    if (node.type !== 'dir') return [];
    return [...node.kids.values()].sort((a, b) => b.hits - a.hits);
}

// ── Squarified treemap ──────────────────────────────────────────────────────
// Standard algorithm (Bruls/Huizing/van Wijk): fill rows so the tiles stay
// as square as possible. Long thin strips would otherwise be neither
// readable nor clickable.

function smSquarify(items, x, y, w, h) {
    const out = [];
    const total = items.reduce((s, i) => s + (Number(i.hits) || 0), 0);
    // `!(total > 0)` instead of `total <= 0`: NaN <= 0 is false and slipped
    // through, after which every tile's geometry was NaN — an empty map
    // with no error.
    if (!(total > 0) || !(w > 0) || !(h > 0)) return out;

    const scale = (w * h) / total;
    const remainder = items.map(i => ({ node: i, area: (Number(i.hits) || 0) * scale }));
    let rx = x, ry = y, rw = w, rh = h;

    while (remainder.length && rw > 0.5 && rh > 0.5) {
        const vertical = rw >= rh;          // row along the shorter side
        const side = vertical ? rh : rw;
        const row = [];
        let rowArea = 0, best = Infinity;

        while (remainder.length) {
            const cand = remainder[0];
            const newArea = rowArea + cand.area;
            const thick = newArea / side;
            // Worst aspect ratio of the row WITH the candidate
            let worst = 0;
            for (const it of [...row, cand]) {
                const len = thick > 0 ? it.area / thick : 0;
                if (len <= 0) { worst = Infinity; break; }
                worst = Math.max(worst, Math.max(thick / len, len / thick));
            }
            if (row.length && worst > best) break;   // gets worse → close the row
            row.push(remainder.shift());
            rowArea = newArea;
            best = worst;
        }

        const thick = rowArea / side;
        let off = vertical ? ry : rx;
        row.forEach(it => {
            const len = it.area / thick;
            if (vertical) out.push({ node: it.node, x: rx, y: off, w: thick, h: len });
            else out.push({ node: it.node, x: off, y: ry, w: len, h: thick });
            off += len;
        });
        if (vertical) { rx += thick; rw -= thick; }
        else { ry += thick; rh -= thick; }
    }
    return out;
}

// ── Rendering ───────────────────────────────────────────────────────────────

// Stable hue derived from the name: the same folder keeps its color while
// zooming, which keeps the map recognizable across levels.
// The golden angle at the end spreads the hues out: without it, similar
// names ("video01", "video02") land on almost the same hue and the map ends
// up monochrome.
function smHue(str) {
    let h = 2166136261;
    for (let i = 0; i < str.length; i++) {
        h ^= str.charCodeAt(i);
        h = Math.imul(h, 16777619);
    }
    return ((h >>> 0) % 1000) * 137.508 % 360;
}

// Light tiles need dark text. A fixed white label color would otherwise
// disappear completely on yellow/turquoise — so compute the perceived
// brightness and pick the text color from that.
function smIsLight(hue, sat, light) {
    const c = (1 - Math.abs(2 * light / 100 - 1)) * (sat / 100);
    const x = c * (1 - Math.abs((hue / 60) % 2 - 1));
    const m = light / 100 - c / 2;
    const seg = Math.floor(hue / 60) % 6;
    const [r, g, b] = [[c, x, 0], [x, c, 0], [0, c, x],
                       [0, x, c], [x, 0, c], [c, 0, x]][seg];
    return (0.299 * (r + m) + 0.587 * (g + m) + 0.114 * (b + m)) > 0.55;
}

async function smRender() {
    const node = smState.path[smState.path.length - 1];
    const canvas = document.getElementById('smCanvas');
    // Without a tree there's nothing to draw. Happens if the map was never
    // successfully opened (no hits) and is rendered anyway, e.g. via the
    // toggle button.
    if (!node) {
        canvas.innerHTML = '<div class="sm-note">No hits</div>';
        document.getElementById('smTimeline').style.display = 'none';
        document.getElementById('smInfo').textContent = '';
        smRenderCrumbs();
        return;
    }

    smRenderCrumbs();
    smRenderTimeline(node);
    smSyncViewBtn(node);

    // ── File level: timeline on top, transcript below ─────────────────
    // The zoom ends here. One more tile level would just repeat the
    // timeline; what you really want at this point is the text.
    if (node.type === 'file') {
        canvas.innerHTML = '<div class="sm-note">Loading transcript…</div>';
        const rseq = ++smState.renderSeq;
        await smEnsureFile(node);
        if (rseq !== smState.renderSeq) return;   // navigated away meanwhile
        smRenderTranscript(node);
        return;
    }

    // ── Folder level: treemap or list ──────────────────────────────────
    const kids = smChildren(node);
    smState.kids = kids;
    const sum = kids.reduce((s, k) => s + k.hits, 0);
    document.getElementById('smInfo').textContent =
        `${plural(kids.length, 'entry', 'entries')} · ${plural(sum, 'hit')}`;

    if (!kids.length) {
        canvas.innerHTML = '<div class="sm-note">Nothing to display</div>';
        return;
    }
    if (smState.view === 'list') smRenderList(kids, sum);
    else smRenderMap(kids, canvas);
}

// The toggle button means something different depending on the level: at
// the top it's map/list, inside a file it's whole transcript or only hits.
function smSyncViewBtn(node) {
    const btn = document.getElementById('smViewBtn');
    if (node.type === 'file') {
        btn.textContent = smState.txMode === 'full' ? '⌖ Hits only' : '☰ Whole transcript';
        btn.title = 'Toggle between the whole transcript and the hits';
    } else {
        btn.textContent = smState.view === 'map' ? '☰ List' : '⛶ Map';
        btn.title = 'Toggle between map and list';
    }
}

// Color of a tile/row — in one place, so map and list show the same mapping.
function smColor(n, maxHits) {
    const hue = smHue(n.name);
    // Brightness by hit share: the big chunks additionally stand out by
    // color, not just by area.
    const light = 28 + 26 * (n.hits / maxHits);
    return { hue, light, onLight: smIsLight(hue, 55, light) };
}

function smRenderMap(kids, canvas) {
    const W = canvas.clientWidth, H = canvas.clientHeight;
    if (W < 10 || H < 10) return;

    const cells = smSquarify(kids, 0, 0, W, H);
    const maxHits = Math.max(...kids.map(k => k.hits));
    const idx = new Map(kids.map((k, i) => [k, i]));

    let html = '';
    cells.forEach(c => {
        const n = c.node;
        const { hue, light, onLight } = smColor(n, maxHits);
        const small = c.w < 54 || c.h < 26;
        const tiny = c.w < 14 || c.h < 12;
        const label = tiny ? '' : `
            <span class="sm-cell-name">${esc(n.name)}</span>
            ${small ? '' : `<span class="sm-cell-n">${plural(n.hits, 'hit')}</span>`}`;
        html += `<div class="sm-cell${onLight ? ' sm-on-light' : ''}"
            style="left:${c.x}px;top:${c.y}px;width:${Math.max(0, c.w - 2)}px;height:${Math.max(0, c.h - 2)}px;background:hsl(${hue} 55% ${light}%)"
            title="${esc(smTitle(n))}"
            onclick="smPick(${idx.get(n)})">${label}</div>`;
    });
    canvas.innerHTML = html;
}

// List view: while zooming in, the map inevitably gets fine-grained, and
// eventually no label fits in the tile anymore. The list shows the same
// data with guaranteed-readable names and hit counts.
function smRenderList(kids, sum) {
    const canvas = document.getElementById('smCanvas');
    const maxHits = Math.max(...kids.map(k => k.hits));
    const rows = kids.map((n, i) => {
        const { hue, light } = smColor(n, maxHits);
        const pct = sum > 0 ? (100 * n.hits / sum) : 0;
        return `<div class="sm-row" onclick="smPick(${i})" title="${esc(smTitle(n))}">
            <span class="sm-row-dot" style="background:hsl(${hue} 55% ${light + 12}%)"></span>
            <span class="sm-row-name">${esc(n.name)}</span>
            <span class="sm-row-sub">${n.type === 'dir' ? 'Folder' : ''}</span>
            <span class="sm-row-bar"><span class="sm-row-fill"
                style="width:${(100 * n.hits / maxHits).toFixed(1)}%;background:hsl(${hue} 55% ${light + 12}%)"></span></span>
            <span class="sm-row-n">${n.hits}</span>
            <span class="sm-row-pct">${pct.toFixed(1)}%</span>
        </div>`;
    }).join('');
    canvas.innerHTML = `<div class="sm-list">${rows}</div>`;
}

// Bar chart above the map once you're inside a file: the temporal
// distribution is the actual question there ("where in the file?"), and a
// timeline is more honest for that than tiles — the treemap sorts by size
// and loses the ordering. Clicking a bar jumps to that section.
function smRenderTimeline(node) {
    const box = document.getElementById('smTimeline');
    if (node.type !== 'file' || !node.histogram.length) {
        box.style.display = 'none';
        box.innerHTML = '';
        return;
    }
    box.style.display = 'block';

    const h = node.histogram, n = h.length;
    const span = (node.duration || 0) / n;
    const peak = Math.max(1, ...h);
    const hue = smHue(node.name);

    const bars = h.map((v, i) => `<span
        class="sm-tl-bar${v ? '' : ' sm-tl-empty'}${i === smState.active ? ' active' : ''}"
        style="height:${v ? (8 + 92 * v / peak).toFixed(0) : 3}%;${v ? `background:hsl(${hue} 60% 52%)` : ''}"
        title="${svFmt(i * span)} – ${svFmt((i + 1) * span)} · ${plural(v, 'hit')}"
        onclick="smJumpTo(${i})"></span>`).join('');

    // Four axis ticks are enough for orientation; more gets cluttered with 40 bars
    const ticks = [0, 0.25, 0.5, 0.75, 1].map(f =>
        `<span>${esc(svFmt((node.duration || 0) * f))}</span>`).join('');

    box.innerHTML = `
        <div class="sm-tl-head">Time distribution in ${esc(node.name)} · click jumps to that spot in the transcript</div>
        <div class="sm-tl-bars">${bars}</div>
        <div class="sm-tl-axis">${ticks}</div>`;
}

function smTitle(n) {
    return `${n.name} · ${plural(n.hits, 'hit')}`;
}

function smRenderCrumbs() {
    const box = document.getElementById('smCrumbs');
    box.innerHTML = smState.path.map((n, i) => {
        const last = i === smState.path.length - 1;
        return `<span class="sm-crumb${last ? ' current' : ''}"
            onclick="smGoLevel(${i})">${esc(n.name)}</span>`;
    }).join('<span class="sm-crumb-sep">›</span>');
}

// Click: one level deeper — at the bottom level, jump into the transcript.
function smPick(i) {
    const target = smState.kids[i];
    if (!target) return;
    smState.active = -1;      // new file/folder: discard old highlight
    smState.path.push(target);
    smRender();
}

// Click on a bar in the timeline: scroll to the matching spot in the
// transcript and highlight it. Deliberately no further zoom — the text is
// already there below, a jump is the shorter answer than a new level.
function smJumpTo(i) {
    const node = smState.path[smState.path.length - 1];
    if (!node || node.type !== 'file') return;
    smState.active = i;
    smRenderTimeline(node);

    const span = (node.duration || 0) / (node.histogram.length || 1);
    const from = i * span, to = (i + 1) * span;
    const rows = document.querySelectorAll('#smCanvas .sm-seg');
    let first = null;
    rows.forEach(r => {
        const t = parseFloat(r.dataset.start);
        const inside = t >= from && t < to;
        r.classList.toggle('sm-seg-zone', inside);
        if (inside && !first) first = r;
    });
    if (first) first.scrollIntoView({ block: 'center', behavior: 'smooth' });
}

function smToggleView() {
    const node = smState.path[smState.path.length - 1];
    if (node && node.type === 'file') {
        smState.txMode = smState.txMode === 'full' ? 'hits' : 'full';
    } else {
        smState.view = smState.view === 'map' ? 'list' : 'map';
    }
    smRender();
}

// ── Transcript at the file level ─────────────────────────────────────────────

// Load the file's transcript and hits. Both are needed: the transcript
// provides the full text, the hits say which segments need highlighting
// (and provide the highlighting from FTS5 along with it).
async function smEnsureFile(node) {
    await Promise.all([
        smLoadHits(node),
        smLoadTranscript(node),
    ]);
}

async function smLoadTranscript(node) {
    if (smState.tx.has(node.rel)) return;
    try {
        const r = await fetch(`/api/transcript/${encodeURIComponent(node.rel)}`);
        const data = await r.json();
        smState.tx.set(node.rel, Array.isArray(data.segments) ? data : { segments: [] });
    } catch (e) {
        smState.tx.set(node.rel, { segments: [] });
    }
}

function smRenderTranscript(node) {
    const canvas = document.getElementById('smCanvas');
    const tx = smState.tx.get(node.rel) || { segments: [] };
    const hits = smState.hits.get(node.rel) || [];
    const names = tx.speaker_names || {};

    // Match hits by start time: search and transcript read the same
    // column, rounding is only against floating-point noise.
    const hitMap = new Map();
    hits.forEach(h => hitMap.set(Math.round(h.start * 1000), h));

    const onlyHits = smState.txMode === 'hits';
    const segs = tx.segments || [];
    const shown = onlyHits
        ? segs.filter(s => hitMap.has(Math.round((s.start || 0) * 1000)))
        : segs;

    document.getElementById('smInfo').textContent =
        `${plural(hits.length, 'hit')} · ${plural(segs.length, 'segment')}`
        + (onlyHits ? ' · hits only' : '');

    if (!shown.length) {
        canvas.innerHTML = `<div class="sm-note">${segs.length
            ? 'No hits in this transcript'
            : 'No transcript available'}</div>`;
        return;
    }

    const rows = shown.map(s => {
        const key = Math.round((s.start || 0) * 1000);
        const hit = hitMap.get(key);
        const sp = names[s.speaker] || s.speaker || '—';
        // For hits, use the highlighting from the search (which also knows
        // forms without diacritics), otherwise just the escaped text.
        const text = hit ? _hlHit(hit, svState.terms) : esc(s.text || '');
        // Force Number(): /api/transcript returns unchecked JSON from
        // disk. A "start" like "0);alert(1)//" would otherwise be executable.
        const start = Number(s.start) || 0;
        return `<div class="sm-seg${hit ? ' sm-seg-hit' : ''}" data-start="${start}"
            onclick="smOpenAt(${start})" title="Open in transcript">
            <span class="sm-seg-ts">${esc(s.start_fmt || svFmt(s.start))}</span>
            <span class="sm-seg-sp">${esc(sp)}</span>
            <span class="sm-seg-tx">${text}</span>
        </div>`;
    }).join('');

    canvas.innerHTML = `<div class="sm-tx">${rows}</div>`;
    smState.active = -1;

    // Jump to the first hit on opening. You're coming from a search —
    // landing at 0:00 up top and then having to scroll there would be an
    // unnecessary extra move when the hits only start deep into the file.
    const first = canvas.querySelector('.sm-seg-hit');
    if (first) first.scrollIntoView({ block: 'center' });
}

// Click on a segment: close the map and search view, open the file in the
// player and jump to that spot.
function smOpenAt(start) {
    const node = smState.path[smState.path.length - 1];
    if (!node || node.type !== 'file') return;
    closeSearchMap();
    closeSearchView();
    goToSearchResult(node.rel, start, node.dir || '');
}

// Load a file's hits — exactly matching the path, so the numbers match
// the tiles (file= alone searches as a substring, see file_exact).
async function smLoadHits(fileNode) {
    if (smState.hits.has(fileNode.rel)) return;
    const seq = ++smState.seq;
    let p = `q=${encodeURIComponent(svState.query)}`
        + `&file=${encodeURIComponent(fileNode.rel)}&file_exact=1&limit=5000`;
    if (svState.speaker) p += `&speaker=${encodeURIComponent(svState.speaker)}`;
    try {
        const r = await fetch(`/api/search?${p}`);
        const data = await r.json();
        if (seq !== smState.seq) return;
        smState.hits.set(fileNode.rel, data.results || []);
    } catch (e) {
        smState.hits.set(fileNode.rel, []);
    }
}
