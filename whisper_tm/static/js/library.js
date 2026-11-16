// ── Library: every folder with its numbers ──────────────────────────────
// The statistics window became the Library place: on the left the folder
// tree of the whole library (/api/library/tree, one level at a time, each
// folder with its transcribed share, failures, whether it is watched and
// its file count), on the right the chosen folder — its crumbs, what can be
// done with it, pills that jump to the sections, and the statistics of
// exactly that folder (/api/stats?folder=, statsview.js draws them).
// A click selects a folder, a double-click opens it in the file list.

const Library = (() => {
    let root = '';            // the library root (absolute)
    let rel = '';             // the chosen folder, relative to the root
    let node = null;          // its last answer
    let rootNode = null;      // the root's own answer (its counts, whether it is watched)
    const kids = new Map();   // rel → children (loaded levels)
    const open = new Set(['']);
    let filter = '';
    const el = id => document.getElementById(id);
    const pct = c => c && c.files ? Math.round(c.done / c.files * 100) : 0;

    async function level(path) {
        const r = await fetch('/api/library/tree?path=' + encodeURIComponent(path));
        const d = await r.json();
        if (!r.ok) throw new Error(d.error || `HTTP ${r.status}`);
        root = d.root || root;
        if (!path) rootNode = d;
        kids.set(path, d.children || []);
        return d;
    }

    /* The Library place, on a folder (relative to the library root), or on
       the folder the file list has open. */
    async function show(path) {
        try {
            if (!kids.has('')) await level('');
        } catch (e) {
            const t = el('lbNodes');
            if (t) t.innerHTML = `<div class="lb-empty">${esc(e.message || String(e))}</div>`;
        }
        // The open folder, once the tree's root is known (on the first open
        // it is not before level('') — the Library then opened on the whole
        // tree, the slowest view there is, instead of the open folder)
        if (path === undefined) path = openFolderRel();
        // Open every folder on the way down to the chosen one
        const parts = path ? path.split('/') : [];
        let at = '';
        for (const p of parts) {
            if (!kids.has(at)) { try { await level(at); } catch (e) { break; } }
            open.add(at);
            at = at ? `${at}/${p}` : p;
        }
        await select(path, {quiet: true});
    }

    function openFolderRel() {
        const dir = (el('dirInput')?.value || '').trim().replace(/\/+$/, '');
        if (!root || !dir) return '';
        if (dir === root) return '';
        return dir.startsWith(root + '/') ? dir.slice(root.length + 1) : '';
    }

    async function select(path, opts) {
        rel = path || '';
        try { node = await level(rel); } catch (e) { node = null; }
        drawTree();
        drawScope();
        if (typeof stvGoFolder === 'function') stvGoFolder(root + (rel ? '/' + rel : ''), rel);
    }

    async function toggle(path, ev) {
        ev && ev.stopPropagation();
        if (open.has(path)) { open.delete(path); drawTree(); return; }
        open.add(path);
        if (!kids.has(path)) { try { await level(path); } catch (e) { /* stays closed */ } }
        drawTree();
    }

    /* Double-click: the folder in the file list. */
    function openInList(path) {
        const abs = root + (path ? '/' + path : '');
        const input = el('dirInput');
        if (!input || typeof changeDir !== 'function') return;
        input.value = abs;
        changeDir();
        if (typeof Shell !== 'undefined') Shell.go('home');
    }

    function row(n, depth) {
        const c = n.counts || {};
        const isOpen = open.has(n.path);
        const p = pct(c);
        const eye = n.watched === 'self' ? '<span class="lb-eye" title="Watched by continuous scanning">👁</span>'
            : n.watched === 'parent' ? '<span class="lb-eye dim" title="Watched through a folder above it">👁</span>' : '';
        return `<div class="lb-row${n.path === rel ? ' on' : ''}" style="padding-left:${6 + depth * 14}px" data-path="${esc(n.path)}"`
            + ` onclick="Library.select('${escJs(n.path)}')" ondblclick="Library.openInList('${escJs(n.path)}')" title="${esc(n.path || n.name)} — double-click opens it in the file list">`
            + (n.has_children ? `<button class="lb-caret" onclick="Library.toggle('${escJs(n.path)}', event)" aria-label="${isOpen ? 'Close' : 'Open'}">${isOpen ? '▾' : '▸'}</button>` : '<span class="lb-caret"></span>')
            + `<span class="lb-name">${esc(n.name)}</span>${c.failed ? `<span class="lb-fail" title="${plural(c.failed, 'failed file')}"></span>` : ''}${eye}`
            + `<span class="lb-bar" title="${p} % transcribed"><span style="width:${p}%"></span></span>`
            + `<span class="lb-n">${(c.files || 0).toLocaleString()}</span></div>`;
    }

    function drawTree() {
        const box = el('lbNodes');
        if (!box) return;
        const out = [];
        const f = filter.toLowerCase();
        const walk = (path, depth) => {
            for (const n of kids.get(path) || []) {
                const match = !f || n.name.toLowerCase().includes(f) || n.path.toLowerCase().includes(f);
                if (match) out.push(row(n, depth));
                if ((open.has(n.path) || (f && kids.has(n.path))) && kids.has(n.path)) walk(n.path, depth + 1);
            }
        };
        const top = {name: root.split('/').pop() || root, path: '', counts: rootNode ? rootNode.counts : kidsTotal(),
                     has_children: true, watched: rootNode ? rootNode.watched : ''};
        out.push(row(top, 0));
        walk('', 1);
        box.innerHTML = out.join('') || '<div class="lb-empty">No folders</div>';
    }

    function kidsTotal() {
        const all = {files: 0, done: 0, open: 0, failed: 0, queued: 0};
        for (const n of kids.get('') || []) for (const k in all) all[k] += (n.counts || {})[k] || 0;
        return all;
    }

    function drawScope() {
        const box = el('lbScope');
        if (!box) return;
        const c = (node && node.counts) || {};
        const crumbs = (node && node.crumbs) || [{name: root.split('/').pop() || root, path: ''}];
        const isOpenFolder = openFolderRel() === rel && (el('dirInput')?.value || '').trim().replace(/\/+$/, '').startsWith(root);
        const watched = node && node.watched;
        box.innerHTML = `<div class="lb-crumbs">${crumbs.map((x, i) => i === crumbs.length - 1
                ? `<b>${esc(x.name)}</b>` : `<button class="lb-crumb" onclick="Library.select('${escJs(x.path)}')">${esc(x.name)}</button><span>›</span>`).join('')}</div>
            <div class="lb-line">${plural(c.files || 0, 'file')} · ${pct(c)} % transcribed${c.open ? ` · ${(c.open).toLocaleString()} open` : ''}${c.failed ? ` · <span class="lb-red">${(c.failed).toLocaleString()} failed</span>` : ''}${c.queued ? ` · ${(c.queued).toLocaleString()} waiting` : ''}${watched ? ` · 👁 watched${watched === 'parent' ? ' (through a folder above)' : ''}` : ''}</div>
            <div class="lb-acts">
                <button class="tb-btn" onclick="Library.openInList('${escJs(rel)}')"><span class="tb-label">Open in the file list</span></button>
                ${c.open ? `<button class="tb-btn primary" onclick="Library.transcribeOpen()"><span class="tb-label">Transcribe ${(c.open).toLocaleString()} open</span></button>` : ''}
                <button class="tb-btn" onclick="Library.exportHere()" title="Export this folder's recordings: text, subtitles, Word, PDF, video or audio"><span class="tb-label">Export</span></button>
                ${typeof Autoscan !== 'undefined' ? Autoscan.watchButton(root + (rel ? '/' + rel : ''), watched) : ''}
            </div>`;
    }

    /* The folder in the file list, then something to do there once its
       files are loaded. */
    function inList(then) {
        const abs = root + (rel ? '/' + rel : '');
        const same = (el('dirInput')?.value || '').trim().replace(/\/+$/, '') === abs;
        if (!same) openInList(rel);
        else if (typeof Shell !== 'undefined') Shell.go('home');
        if (then) setTimeout(then, same ? 0 : 900);
    }

    /* "Show them": the file list on this folder with the filter set. */
    function showIn(f) {
        inList(() => {
            if (typeof FilterPanel === 'undefined') return;
            FilterPanel.apply(f || {});
        });
    }

    /* Transcribe N open: the Queue's "Add a folder" with this folder. */
    function transcribeOpen() {
        if (typeof Shell !== 'undefined') Shell.go('queue');
        if (typeof qeShowTab === 'function') qeShowTab('add');
        const f = el('qaFolder');
        if (f) { f.value = root + (rel ? '/' + rel : ''); if (typeof qaCount === 'function') qaCount(); }
    }

    /* Export this folder: the export page on it, Multiple files. */
    function exportHere() {
        if (typeof Export !== 'undefined') Export.show({library: rel});
    }

    /* Pills that jump to the sections of the statistics below, the one in
       view lit. */
    let pillObs = null;
    function pills() {
        const nav = el('lbPills'), body = el('statsBody');
        if (!nav || !body) return;
        const cards = [...body.querySelectorAll('.stv-card')];
        cards.forEach((c, i) => { c.dataset.pill = String(i); });
        nav.innerHTML = cards.map((c, i) => {
            const h = c.querySelector('h3, .stv-card-title');
            const t = h ? (h.textContent || '').trim().split('\n')[0] : (i === 0 ? 'Overview' : '…');
            return `<button class="lb-pill" data-pill="${i}" onclick="Library.jump(${i})">${esc(t)}</button>`;
        }).join('');
        pillObs?.disconnect();
        pillObs = new IntersectionObserver(entries => {
            for (const e of entries) {
                if (!e.isIntersecting) continue;
                nav.querySelectorAll('.lb-pill').forEach(b => b.classList.toggle('on', b.dataset.pill === e.target.dataset.pill));
            }
        }, {root: body, rootMargin: '0px 0px -70% 0px'});
        cards.forEach(c => pillObs.observe(c));
    }

    function jump(i) {
        const c = el('statsBody')?.querySelector(`.stv-card[data-pill="${i}"]`);
        if (c) c.scrollIntoView({behavior: 'smooth', block: 'start'});
    }

    function setFilter(v) {
        filter = String(v || '').trim();
        drawTree();
    }

    function refresh() { kids.clear(); show(rel); }

    return {show, select, toggle, openInList, inList, showIn, transcribeOpen, exportHere, setFilter, refresh, pills, jump,
            get root() { return root; }, get rel() { return rel; }};
})();
