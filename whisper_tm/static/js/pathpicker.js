// ══════════════════════════════════════════════════════════════════════
//  Path Picker — unified folder selection
//  Breadcrumb navigation · live filter · favorites · keyboard
//  Replaces the old file browser + separate favorites dropdown.
//
//  One picker, two places: the file list's path field picks the working
//  folder, the queue editor's "Add folder" field the folder to queue,
//  both with the same explorer dropdown and favorites. A place names
//  its elements and what choosing a
//  folder does. One picker is open at a time; the rows rendered into it
//  call the pp* functions below, and those act on the open place.
// ══════════════════════════════════════════════════════════════════════

const PP_PLACES = {
    main: {input: 'dirInput', field: 'pathField', picker: 'pathPicker', crumbs: 'ppBreadcrumb',
           favs: 'ppFavs', list: 'ppList', hint: 'ppHint', star: 'btnFavStar',
           open: 'Open', openTitle: 'Open this folder', choose: () => changeDir({chosen: true})},
    qa:   {input: 'qaFolder', field: 'qaPathField', picker: 'qaPathPicker', crumbs: 'qaPpBreadcrumb',
           favs: 'qaPpFavs', list: 'qaPpList', hint: 'qaPpHint', star: 'qaFavStar',
           open: 'Choose', openTitle: 'Count this folder for the queue',
           choose: () => qaFolderChosen(),
           // In a dialog: one Esc closes the picker — not the dialog as well
           ownEsc: true},
    an:   {input: 'anFolder', field: 'anPathField', picker: 'anPathPicker', crumbs: 'anPpBreadcrumb',
           favs: 'anPpFavs', list: 'anPpList', hint: 'anPpHint', star: 'anFavStar',
           open: 'Choose', openTitle: 'Analyze the recordings of this folder',
           choose: () => anFolderChosen(),
           // In the search workspace: one Esc closes the picker, not the workspace
           ownEsc: true},
};
let _ppB = PP_PLACES.main;  // the place the picker serves (the open one, or the last)

let _ppOpen = false;
let _ppPath = '/';          // currently navigated directory in the picker
let _ppFolders = [];        // [{name, path}]
let _ppHighlight = -1;      // keyboard highlight index (in filtered list)
let _ppLoading = false;
let _ppReq = 0;             // sequence guard against superseding responses
let _ppOrig = '';           // the field's value on open — restored on cancel

function _pp(id) { return document.getElementById(id); }
function _ppPlace(name) { return PP_PLACES[name] || PP_PLACES.main; }
function _ppIn() { return _pp(_ppB.input); }

// Parent directory of a path
function _parentOf(p) {
    p = (p || '').replace(/\/+$/, '');
    if (!p || p === '') return '/';
    const idx = p.lastIndexOf('/');
    return idx <= 0 ? '/' : p.slice(0, idx);
}

// ── Open / close ───────────────────────────────────────────────────────
// `place`: 'main' (the default — the file list), 'qa' (Add folder) or 'an'
// (a new analysis).
function openPathPicker(place) {
    const b = _ppPlace(place);
    if (_ppOpen && b === _ppB) return;
    if (_ppOpen) closePathPicker(true);     // the other place's picker
    _ppB = b;
    _ppOpen = true;
    // Remember the field's original value — reset on cancel/close without selection
    _ppOrig = _ppIn().value;
    _pp(b.picker).style.display = 'flex';
    _pp(b.field).classList.add('focused');
    // Start: current directory from the input (or its existence)
    const cur = _ppIn().value.trim() || _ppPath || '/';
    ppNavigate(cur);
    // Click outside closes it
    setTimeout(() => document.addEventListener('mousedown', _ppOutside), 0);
}
// restore=true → cancel/close without selection: discard the browsed value,
// restore the field's original value. Without an argument (selection confirmed
// or changeDir takes over the value) the current field content is kept.
function closePathPicker(restore) {
    if (!_ppOpen) return;
    _ppOpen = false;
    // A navigation still on its way belongs to the closed picker: its late
    // answer must not write the old folder back into the path field (Enter
    // already opened what the field held).
    _ppReq++;
    _ppLoading = false;
    _pp(_ppB.picker).style.display = 'none';
    _pp(_ppB.field).classList.remove('focused');
    _ppHighlight = -1;
    document.removeEventListener('mousedown', _ppOutside);
    if (restore) {
        const inp = _ppIn();
        if (inp) inp.value = _ppOrig;
        _ppStars();
    }
}
// Close the picker if it serves this place — its dialog closes or hides it.
function ppCloseIn(place) {
    if (_ppOpen && _ppB === _ppPlace(place)) closePathPicker(true);
}
function togglePathPicker(ev, place) {
    if (ev) ev.stopPropagation();
    const b = _ppPlace(place);
    if (_ppOpen && _ppB === b) { closePathPicker(true); _pp(b.input).blur(); }
    else { _pp(b.input).focus(); openPathPicker(place); }
}
function _ppOutside(e) {
    const field = _pp(_ppB.field), picker = _pp(_ppB.picker);
    if (field && !field.contains(e.target) && picker && !picker.contains(e.target)) {
        closePathPicker(true);
    }
}

// ── Navigation ─────────────────────────────────────────────────────────
async function ppNavigate(path) {
    const seq = ++_ppReq;
    _ppLoading = true;
    _ppHighlight = -1;
    const list = _pp(_ppB.list);
    if (list) list.innerHTML = '<div class="pp-loading">⟳ Loading…</div>';
    try {
        const r = await fetch(`/api/browse?path=${encodeURIComponent(path)}`);
        const d = await r.json();
        if (seq !== _ppReq) return;             // discard stale response
        _ppPath = d.path || path;
        // Sync the input with the navigation (like an address bar) —
        // otherwise the filter logic picks up the old path as a filter fragment.
        const inp = _ppIn();
        if (inp) inp.value = _ppPath;
        _ppStars();
        // filter out ".." — navigation goes through the breadcrumb
        _ppFolders = (d.dirs || []).filter(x => x.name !== '..');
        _ppLoading = false;
        _ppRenderBreadcrumb();
        _ppRenderFavs();
        _ppRenderList(d.error);
        _ppUpdateHint();
    } catch(e) {
        if (seq !== _ppReq) return;
        _ppLoading = false;
        if (list) list.innerHTML = `<div class="pp-empty">Error: ${esc(e.message)}</div>`;
    }
}

function _ppRenderBreadcrumb() {
    const bc = _pp(_ppB.crumbs);
    if (!bc) return;
    const parts = _ppPath.split('/').filter(Boolean);
    let html = `<span class="pp-crumb" onclick="ppNavigate('/')" title="Root">/</span>`;
    let acc = '';
    parts.forEach((seg, i) => {
        acc += '/' + seg;
        const p = acc;
        html += `<span class="pp-crumb-sep">›</span>`;
        html += `<span class="pp-crumb${i === parts.length - 1 ? ' current' : ''}" onclick="ppNavigate('${escJs(p)}')" title="${esc(p)}">${esc(seg)}</span>`;
    });
    // Up button — always there, disabled at the root, so the crumbs don't jump
    html = `<button class="pp-up" ${parts.length ? `onclick="ppNavigate('${escJs(_parentOf(_ppPath))}')"` : 'disabled'}
                    title="Up one level">↑</button>` + html;
    bc.innerHTML = html;
}

function _ppRenderFavs() {
    const el = _pp(_ppB.favs);
    if (!el) return;
    const favs = loadFavDirs();
    if (!favs.length) { el.innerHTML = ''; el.style.display = 'none'; return; }
    el.style.display = 'block';
    el.innerHTML = `<div class="pp-section">Favorites</div><div class="pp-fav-row">` + favs.map(p => `
        <span class="pp-fav-chip" onclick="ppOpen('${escJs(p)}')" title="${esc(_ppB.open)} ${esc(p)}">
            <span class="pp-fav-star">★</span>
            <span class="pp-fav-name">${esc(p.split('/').filter(Boolean).pop() || p)}</span>
            <span class="pp-fav-x" role="button" onclick="event.stopPropagation();ppRemoveFav('${escJs(p)}')"
                  title="Remove from favorites">✕</span>
        </span>`).join('') + `</div>`;
}

function _ppFiltered() {
    const q = _ppIn().value.trim().toLowerCase();
    // Only filter when the input is NOT simply the current directory
    const curNorm = _ppPath.replace(/\/+$/, '').toLowerCase();
    const typed = q.replace(/\/+$/, '');
    if (!typed || typed === curNorm) return _ppFolders;
    // Filter after the last "/" (basename part), otherwise the whole string
    const ask = typed.includes('/') ? typed.slice(typed.lastIndexOf('/') + 1) : typed;
    if (!ask) return _ppFolders;
    return _ppFolders.filter(f => f.name.toLowerCase().includes(ask));
}

function _ppRenderList(errMsg) {
    const list = _pp(_ppB.list);
    if (!list) return;
    const favs = loadFavDirs();
    const folders = _ppFiltered();
    const head = `<div class="pp-section">Folders</div>`;
    if (!folders.length) {
        list.innerHTML = head + `<div class="pp-empty">${errMsg ? esc(errMsg) : 'No subfolders here'}</div>`;
        return;
    }
    list.innerHTML = head + folders.map((f, i) => {
        const isFav = favs.includes(f.path);
        return `<div class="pp-item${i === _ppHighlight ? ' hl' : ''}" data-idx="${i}"
                     onclick="ppNavigate('${escJs(f.path)}')" ondblclick="ppOpen('${escJs(f.path)}')">
            <span class="pp-item-icon">📁</span>
            <span class="pp-item-name">${esc(f.name)}</span>
            <button class="pp-item-star${isFav ? ' on' : ''}" title="${isFav ? 'Remove favorite' : 'Favorite'}"
                    onclick="event.stopPropagation();ppToggleFav('${escJs(f.path)}')">${isFav ? '★' : '☆'}</button>
            <button class="pp-item-open" title="${esc(_ppB.openTitle)}"
                    onclick="event.stopPropagation();ppOpen('${escJs(f.path)}')">${esc(_ppB.open)}</button>
        </div>`;
    }).join('');
}

function _ppUpdateHint() {
    const h = _pp(_ppB.hint);
    if (h) h.textContent = _ppPath;
}

// ── Selection ──────────────────────────────────────────────────────────
// ppNavigate = navigate into (browse); ppOpen = choose the folder — as the
// working folder (file list) or as the folder to queue (Add folder)
function ppOpen(path) {
    const b = _ppB;
    _pp(b.input).value = path;
    closePathPicker();
    _pp(b.input).blur();
    b.choose();
}
function ppChooseCurrent() { ppOpen(_ppPath); }

// ⌂ in the path field: open the library root. Asked from the server —
// /api/browse without a path answers with the first allowed directory.
// Not the folder the page was loaded with: the media directory moves along
// with every folder change, so that is only the working folder of then.
async function ppHome(ev) {
    if (ev) ev.stopPropagation();
    try {
        const r = await fetch('/api/browse');
        const d = await r.json().catch(() => ({}));
        if (!d.path) throw new Error(d.error || `HTTP ${r.status}`);
        // ⌂ sits in the file list's field — it opens there
        if (_ppB !== PP_PLACES.main) { ppCloseIn('qa'); _ppB = PP_PLACES.main; }
        ppOpen(d.path);
    } catch (e) {
        const status = _pp('dirStatus');
        if (status) {
            status.textContent = 'Library root not available: ' + e.message;
            status.className = 'dir-status err';
        }
    }
}

// ── Favorites ──────────────────────────────────────────────────────────
// Shared by both places: a folder starred in one is a favorite in the other.
function ppToggleFav(path) {
    const favs = loadFavDirs();
    const i = favs.indexOf(path);
    if (i >= 0) favs.splice(i, 1); else favs.push(path);
    saveFavDirs(favs);
    _ppRenderFavs();
    _ppRenderList();
    _ppStars();
}
function ppRemoveFav(path) {
    const favs = loadFavDirs().filter(p => p !== path);
    saveFavDirs(favs);
    _ppRenderFavs();
    _ppRenderList();
    _ppStars();
}
function toggleCurrentFav(ev, place) {
    if (ev) ev.stopPropagation();
    const path = _pp(_ppPlace(place).input).value.trim();
    if (!path) return;
    ppToggleFav(path);
}

// The ☆ of every path field: ★ while its path is a favorite. The file
// list's belongs to directory.js (updateFavBtn), the others are drawn here.
function _ppStars() {
    if (typeof updateFavBtn === 'function') updateFavBtn();
    const favs = loadFavDirs();
    for (const b of Object.values(PP_PLACES)) {
        if (b === PP_PLACES.main) continue;
        const btn = _pp(b.star), inp = _pp(b.input);
        if (!btn || !inp) continue;
        if (typeof showPathEnd === 'function') showPathEnd(inp);
        const on = favs.includes(inp.value.trim());
        btn.textContent = on ? '★' : '☆';
        btn.classList.toggle('active', on);
        btn.title = on ? 'Remove favorite' : 'Save as favorite';
    }
}
function ppUpdateStars() { _ppStars(); }

// ── Input / keyboard ───────────────────────────────────────────────────
function onPathInput(place) {
    const b = _ppPlace(place);
    if (!_ppOpen || _ppB !== b) openPathPicker(place);
    const val = _pp(b.input).value;
    // If the typed path ends with "/", navigate there directly
    if (val.endsWith('/') && val.length > 1) {
        const target = val.replace(/\/+$/, '') || '/';
        if (target.toLowerCase() !== _ppPath.replace(/\/+$/, '').toLowerCase()) {
            ppNavigate(target);
            return;
        }
    }
    _ppHighlight = -1;
    _ppRenderList();
    _ppStars();
}

function onPathKey(ev, place) {
    const b = _ppPlace(place);
    const open = _ppOpen && _ppB === b;
    if (ev.key === 'Escape') {
        if (open && b.ownEsc) ev.stopPropagation();
        closePathPicker(true);
        return;
    }
    if (!open && ev.key !== 'Enter') return;
    const folders = _ppFiltered();
    if (ev.key === 'ArrowDown') {
        ev.preventDefault();
        _ppHighlight = Math.min(_ppHighlight + 1, folders.length - 1);
        _ppRenderList(); _ppScrollToHighlight();
    } else if (ev.key === 'ArrowUp') {
        ev.preventDefault();
        _ppHighlight = Math.max(_ppHighlight - 1, -1);
        _ppRenderList(); _ppScrollToHighlight();
    } else if (ev.key === 'Enter') {
        ev.preventDefault();
        if (open && _ppHighlight >= 0 && folders[_ppHighlight]) {
            // Navigate into the highlighted folder (Tab would open it)
            ppNavigate(folders[_ppHighlight].path);
        } else {
            // Open the typed path directly
            closePathPicker();
            b.choose();
        }
    } else if (ev.key === 'Tab' && open && _ppHighlight >= 0 && folders[_ppHighlight]) {
        ev.preventDefault();
        ppOpen(folders[_ppHighlight].path);
    }
}

function _ppScrollToHighlight() {
    const list = _pp(_ppB.list);
    const el = list && list.querySelector ? list.querySelector('.pp-item.hl') : null;
    if (el) el.scrollIntoView({block: 'nearest'});
}

// Esc closes the picker wherever the focus is inside it — onPathKey only
// sees keys typed into the path field, not on the ↑ button or a folder row.
// In a dialog it stops there: the dialog's own Esc (queue_editor.js,
// registered after this file) would close the dialog as well.
document.addEventListener('keydown', e => {
    if (e.key === 'Escape' && _ppOpen && e.target !== _ppIn()) {
        const b = _ppB;
        closePathPicker(true);
        _pp(b.input)?.blur();
        if (b.ownEsc) e.stopImmediatePropagation();
    }
});
