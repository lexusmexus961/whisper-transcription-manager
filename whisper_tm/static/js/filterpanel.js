// ── The file list's filters and its sort menu ───────────────────────────
// Beside the name filter: a filter button that opens a panel (status
// beyond All/Open/Done, type, length, added, people, language, what a
// recording has, subfolder) and a sort menu. Active filters show as chips
// under the field, each with ×, and "Clear all". The filters go to the
// server with every page (/api/files: f_type, f_len_min …), so paging,
// counts and "select all" stay right.

const FilterPanel = (() => {
    const f = {type: '', len: '', added: '', people: [], language: '', has: [], sub: ''};
    const LENGTHS = {short: ['Under 1 min', 0, 60], mid: ['1–10 min', 60, 600], long: ['10–60 min', 600, 3600],
                     hour: ['Over 1 h', 3600, null]};
    const ADDED = {1: 'Today', 7: 'Last 7 days', 30: 'Last 30 days'};
    const HAS = {transcript: 'A transcript', summary: 'A summary', translation: 'A translation', warnings: 'Warnings',
                 empty: 'An empty transcript', no_audio: 'No audio track', skipped: 'Skipped by hand'};
    // Status beyond the segment's All · Open · Done (the server's tabs)
    const STATUS = {processing: ['Running', 'tabProcessing'], error: ['Failed', 'tabError'],
                    cancelled: ['Cancelled', 'tabCancelled'], orphan: ['Orphaned (media gone)', 'tabOrphan'],
                    unembedded: ['Not searchable by meaning', null]};
    const SORTS = [['name', 'Name'], ['size', 'Size'], ['modified', 'Date'], ['tx_date', 'Transcribed'], ['tx_word_count', 'Words']];
    let peopleNames = null;
    const el = id => document.getElementById(id);

    /* The query string for the server (appended to the list's own). */
    function query() {
        const p = [];
        if (f.type) p.push('f_type=' + f.type);
        if (f.len && LENGTHS[f.len]) {
            const [, lo, hi] = LENGTHS[f.len];
            if (lo) p.push('f_len_min=' + lo);
            if (hi) p.push('f_len_max=' + hi);
        }
        if (f.added) p.push('f_added=' + f.added);
        if (f.language) p.push('f_lang=' + encodeURIComponent(f.language));
        if (f.has.length) p.push('f_has=' + f.has.join(','));
        if (f.sub) p.push('f_sub=' + encodeURIComponent(f.sub));
        if (f.people.length) p.push('f_people=' + encodeURIComponent(f.people.join(',')));
        return p.length ? '&' + p.join('&') : '';
    }

    function statusTab() { return typeof activeTab !== 'undefined' && STATUS[activeTab] ? activeTab : ''; }

    /* The active filters as chips: [label, how to remove it]. */
    function active() {
        const out = [];
        const st = statusTab();
        if (st) out.push([`Status: ${STATUS[st][0]}`, "FilterPanel.clearStatus()"]);
        if (f.type) out.push([f.type === 'video' ? 'Video' : 'Audio only', "FilterPanel.set('type','')"]);
        if (f.len) out.push([LENGTHS[f.len][0], "FilterPanel.set('len','')"]);
        if (f.added) out.push([`Added: ${ADDED[f.added]}`, "FilterPanel.set('added','')"]);
        for (const n of f.people) out.push([`With ${n}`, `FilterPanel.removePerson('${escJs(n)}')`]);
        if (f.language) out.push([`Language: ${f.language}`, "FilterPanel.set('language','')"]);
        for (const h of f.has) out.push([/^(No|Skipped)/.test(HAS[h]) ? HAS[h] : `Has ${HAS[h].replace(/^An? /, '').toLowerCase()}`, `FilterPanel.toggleHas('${h}')`]);
        if (f.sub) out.push([`In ${f.sub}`, "FilterPanel.set('sub','')"]);
        return out;
    }

    function changed() {
        renderChips();
        if (!el('flPanel').hidden) renderPanel();
        if (typeof reloadFileList === 'function') reloadFileList();
    }

    function set(key, value) { f[key] = value; changed(); }
    function toggleHas(h) { f.has = f.has.includes(h) ? f.has.filter(x => x !== h) : [...f.has, h]; changed(); }
    function addPerson(name) {
        name = String(name || '').trim();
        if (!name || f.people.includes(name)) return;
        f.people = [...f.people, name];
        changed();
    }
    function removePerson(name) { f.people = f.people.filter(n => n !== name); changed(); }
    function clearStatus() { if (typeof setFileTab === 'function') setFileTab('all'); renderChips(); if (!el('flPanel').hidden) renderPanel(); }
    function setStatus(tab) {
        if (typeof setFileTab === 'function') setFileTab(tab || 'all');
        renderChips();
        renderPanel();
    }

    /* Several at once (the Library's "Show them"): everything else cleared. */
    function apply(next) {
        Object.assign(f, {type: '', len: '', added: '', people: [], language: '', has: [], sub: ''});
        for (const k of Object.keys(f)) if (next && next[k] !== undefined) f[k] = next[k];
        if (typeof setFileTab === 'function') setFileTab(next && next.status ? next.status : 'all');
        changed();
    }

    function clearAll() {
        Object.assign(f, {type: '', len: '', added: '', people: [], language: '', has: [], sub: ''});
        if (statusTab() && typeof setFileTab === 'function') setFileTab('all');
        changed();
    }

    function renderChips() {
        const box = el('flChips'), n = el('flFilterN');
        const list = active();
        if (n) { n.hidden = !list.length; n.textContent = list.length; }
        el('flFilterBtn')?.classList.toggle('on', list.length > 0);
        if (!box) return;
        box.hidden = !list.length;
        box.innerHTML = list.map(([label, rm]) =>
            `<span class="fl-chip">${esc(label)}<button type="button" onclick="${rm}" title="Remove this filter" aria-label="Remove ${esc(label)}">×</button></span>`).join('')
            + (list.length > 1 ? `<button type="button" class="fl-clear" onclick="FilterPanel.clearAll()">Clear all</button>` : '');
    }

    /* ── The panel ───────────────────────────────────────────────────── */

    const radio = (group, value, label, on, call) =>
        `<button type="button" class="fl-opt${on ? ' on' : ''}" onclick="${call}" aria-pressed="${on}">${esc(label)}</button>`;

    function renderPanel() {
        const p = el('flPanel');
        if (!p) return;
        const st = statusTab();
        const count = id => { const c = id && el(id); return c ? ` ${c.textContent}` : ''; };
        const sec = (title, body) => `<div class="fl-sec"><div class="fl-sec-t">${title}</div><div class="fl-sec-b">${body}</div></div>`;
        p.innerHTML = sec('Status',
                radio('st', '', 'All / Open / Done above', !st, "FilterPanel.setStatus('')")
                + Object.entries(STATUS).map(([k, [label, id]]) => radio('st', k, label + count(id), st === k, `FilterPanel.setStatus('${k}')`)).join(''))
            + sec('Type', radio('t', '', 'Any', !f.type, "FilterPanel.set('type','')")
                + radio('t', 'audio', 'Audio only', f.type === 'audio', "FilterPanel.set('type','audio')")
                + radio('t', 'video', 'Video', f.type === 'video', "FilterPanel.set('type','video')"))
            + sec('Length', radio('l', '', 'Any', !f.len, "FilterPanel.set('len','')")
                + Object.entries(LENGTHS).map(([k, [label]]) => radio('l', k, label, f.len === k, `FilterPanel.set('len','${k}')`)).join(''))
            + sec('Added', radio('a', '', 'Any time', !f.added, "FilterPanel.set('added','')")
                + Object.entries(ADDED).map(([k, label]) => radio('a', k, label, String(f.added) === k, `FilterPanel.set('added',${k})`)).join(''))
            + sec('People', `${f.people.map(n => `<span class="fl-chip">${esc(n)}<button type="button" onclick="FilterPanel.removePerson('${escJs(n)}')">×</button></span>`).join('')}`
                + `<input type="text" id="flPerson" list="flPeopleList" placeholder="Speaks in it … (Enter)" onfocus="FilterPanel.loadPeople()"`
                + ` onkeydown="if(event.key==='Enter'){FilterPanel.addPerson(this.value);this.value=''}">`
                + `<datalist id="flPeopleList">${(peopleNames || []).map(n => `<option value="${esc(n)}">`).join('')}</datalist>`)
            + sec('Language', `<input type="text" id="flLang" value="${esc(f.language)}" placeholder="Code, e.g. en" maxlength="12"`
                + ` onchange="FilterPanel.set('language', this.value.trim())">`)
            + sec('Has', Object.entries(HAS).map(([k, label]) =>
                `<label class="fl-check"><input type="checkbox"${f.has.includes(k) ? ' checked' : ''} onchange="FilterPanel.toggleHas('${k}')"> ${esc(label)}</label>`).join(''))
            + sec('Subfolder', `<input type="text" id="flSub" value="${esc(f.sub)}" placeholder="e.g. recordings/2024" onchange="FilterPanel.set('sub', this.value.trim())">`)
            + `<div class="fl-foot"><button type="button" class="fl-clear" onclick="FilterPanel.clearAll()">Clear all</button>`
            + `<button type="button" class="fl-done" onclick="FilterPanel.toggle(false)">Done</button></div>`;
    }

    async function loadPeople() {
        if (peopleNames) return;
        try {
            const d = await (await fetch('/api/voices/people')).json();
            peopleNames = (d.people || []).map(p => p.name).filter(Boolean);
            const dl = el('flPeopleList');
            if (dl) dl.innerHTML = peopleNames.map(n => `<option value="${esc(n)}">`).join('');
        } catch (e) { peopleNames = []; }
    }

    /* Panels open right under the filter row (the sidebar is a flex
       column: an absolute box without a top landed above it). */
    function placeUnderRow(box) {
        const row = document.querySelector('.fl-row'), side = box.offsetParent || box.parentElement;
        if (!row || !side) return;
        box.style.top = Math.round(row.getBoundingClientRect().bottom - side.getBoundingClientRect().top + 4) + 'px';
    }

    function toggle(on) {
        const p = el('flPanel');
        if (!p) return;
        const show = on === undefined ? p.hidden : !!on;
        if (show) closeSort();       // one of the two at a time
        p.hidden = !show;
        if (show) placeUnderRow(p);
        el('flFilterBtn')?.setAttribute('aria-expanded', String(show));
        if (show) renderPanel();
    }

    /* ── Sort menu ───────────────────────────────────────────────────── */

    function sortLabel() {
        const name = (SORTS.find(([k]) => k === sortKey) || SORTS[0])[1];
        const l = el('flSortLabel');
        if (l) l.textContent = `${name} ${sortAsc ? '↑' : '↓'}`;
    }

    function sortMenu() {
        const m = el('flSortMenu');
        if (!m) return;
        if (!m.hidden) { closeSort(); return; }
        toggle(false);
        m.innerHTML = SORTS.map(([k, label]) =>
            `<button type="button" class="fl-mi${sortKey === k ? ' on' : ''}" onclick="FilterPanel.sortBy('${k}')">${esc(label)}${sortKey === k ? `<span>${sortAsc ? 'ascending ↑' : 'descending ↓'}</span>` : ''}</button>`).join('')
            + `<div class="fl-mi-sep"></div><button type="button" class="fl-mi" onclick="FilterPanel.sortBy(sortKey)">Reverse the order</button>`;
        m.hidden = false;
        placeUnderRow(m);
        el('flSortBtn')?.setAttribute('aria-expanded', 'true');
    }

    function closeSort() {
        const m = el('flSortMenu');
        if (m) m.hidden = true;
        el('flSortBtn')?.setAttribute('aria-expanded', 'false');
    }

    /* The same key again reverses it (as the old arrow buttons did). */
    function sortBy(key) {
        const btn = [...document.querySelectorAll('.sort-btn')].find(b => (b.getAttribute('onclick') || '').includes(`'${key}'`));
        if (typeof setSort === 'function' && btn) setSort(key, btn);
        sortLabel();
        closeSort();
    }

    function init() {
        // The segment (All · Open · Done) changes the status too: the chips follow
        if (typeof setFileTab === 'function') {
            const orig = setFileTab;
            window.setFileTab = function (...a) { const r = orig.apply(this, a); renderChips(); return r; };
        }
        renderChips();
        sortLabel();
        // A click outside closes the panel and the menu; Esc too
        document.addEventListener('mousedown', e => {
            if (!e.target.closest('#flPanel, #flFilterBtn')) toggle(false);
            if (!e.target.closest('#flSortMenu, #flSortBtn')) closeSort();
        });
        document.addEventListener('keydown', e => {
            if (e.key !== 'Escape') return;
            const open = !el('flPanel')?.hidden || !el('flSortMenu')?.hidden;
            if (!open) return;
            toggle(false);
            closeSort();
            e.stopImmediatePropagation();
        }, true);
    }

    document.addEventListener('DOMContentLoaded', init);
    return {query, apply, set, toggleHas, addPerson, removePerson, setStatus, clearStatus, clearAll, toggle, sortMenu,
            sortBy, loadPeople, active, renderChips, get filters() { return JSON.parse(JSON.stringify(f)); }};
})();
