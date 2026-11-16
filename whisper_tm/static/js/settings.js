// ══════════════════════════════════════════════════════════════════════════
//  Settings — shell, rows, saving and the settings search
// ══════════════════════════════════════════════════════════════════════════
//
// Three files:
//   settings.js        this one: opening and closing, the navigation, how a
//                      setting row is drawn, what Save collects, the search
//   settings_pages.js  the pages: which settings, in which cards, with which
//                      description
//   settings_tools.js  the live parts inside the pages: status cards,
//                      provider cards, template lists, index runs, logs …
//
// A row belongs to exactly one config key (`data-key`, a dotted path such as
// `hardware.low_vram`). Its control carries `data-type` (bool, num, str,
// list, lines) so a value keeps its type on the way to config.yaml — a
// <select> hands out strings, and the string "false" is true in Python.
// `data-initial` is the value as drawn: a row whose control reads something
// else is unsaved, and Save sends only those.

let _settingsData = null;   // GET /api/settings: config, defaults, dirs, version …
let _hwData = null;         // GET /api/hardware
let _activeSettingsTab = 'directories';
// Editors that are not a row (provider cards, templates, vocabularies):
// they note themselves here when something changed, by config key.
const _settingsExtraDirty = new Set();

// Names other parts of the app open, and pages that became a section of
// another page ('page#section')
const _SETTINGS_ALIASES = {vocal_isolation: 'vocal', downloads: 'youtube', notifications: 'ui',
                           logs: 'diagnostics', folders: 'directories',
                           llm: 'models#llm', embedding: 'models#embedding', rag: 'search#semantic',
                           summaries: 'tasks#summaries', correction: 'tasks#correction',
                           translation: 'tasks#translation', ai_clients: 'mcp',
                           // the sections of the Search page, by their own name
                           keyword: 'search#keyword', semantic: 'search#semantic', ask: 'search#ask',
                           conversations: 'search#conversations', wordstats: 'search#wordstats'};

// 'embedding' → {page: 'models', section: 'embedding'}
function _settingsTarget(id) {
    const [name, section = ''] = String(_SETTINGS_ALIASES[id] || id || '').split('#');
    const pages = typeof SETTINGS_PAGES !== 'undefined' ? SETTINGS_PAGES : [];
    const page = pages.some(p => p.id === name) ? name : (pages[0]?.id || 'directories');
    return {page, section: page === name ? section : ''};
}

function _settingsPageId(id) {
    return _settingsTarget(id).page;
}

// The page icon, from the sprite in index.html
function _sIcon(name, cls = '') {
    return `<svg class="ui-ico${cls ? ' ' + cls : ''}" aria-hidden="true"><use href="#i-${esc(name)}"/></svg>`;
}

function _settingsPage(id) {
    return (typeof SETTINGS_PAGES !== 'undefined' ? SETTINGS_PAGES : []).find(p => p.id === id);
}

// ── Open, close, load ────────────────────────────────────────────────────

let _settingsOpenSection = '';   // where openSettings(page) lands once drawn

function openSettings(page) {
    document.getElementById('settingsOverlay').classList.add('visible');
    if (page) {
        const to = _settingsTarget(page);
        _activeSettingsTab = to.page;
        _settingsOpenSection = to.section;
    }
    _settingsExtraDirty.clear();
    _settingsSearchReset();
    _loadAllSettings();
}

function closeSettings() {
    if (_settingsUnsavedCount() && !confirm('Discard unsaved changes?')) return;
    document.getElementById('settingsOverlay').classList.remove('visible');
    _sBrowseClose();
    if (typeof _settingsToolsStop === 'function') _settingsToolsStop();
}

// Throw away what was typed and draw the saved state again
function discardSettings() {
    _settingsExtraDirty.clear();
    _loadAllSettings();
}

async function _loadAllSettings() {
    try {
        const [settings, hw] = await Promise.all([
            fetch('/api/settings').then(r => r.json()),
            fetch('/api/hardware').then(r => r.json()).catch(() => ({})),
        ]);
        _settingsData = settings;
        _hwData = hw;
        _renderSettings();
    } catch (e) {
        console.error('Settings load failed:', e);
        _showToast('Settings could not be loaded', true);
    }
}

// ── Navigation ───────────────────────────────────────────────────────────

// Open a page — from the navigation or from elsewhere in the app, also
// straight at a section ('models#embedding', or an old page name that
// became one). During a search this ends the search and jumps to the
// page's first hit.
function setSettingsTab(tab) {
    const to = _settingsTarget(tab);
    if (_ssTerms(_ssQuery).length) { _settingsSearchOpen(to.page); return; }
    _showSettingsTab(to.page);
    const content = document.getElementById('settingsContent');
    if (content) content.scrollTop = 0;
    if (to.section) _settingsJump('set-sec-' + to.section);
}

function _showSettingsTab(tab) {
    _activeSettingsTab = tab;
    document.querySelectorAll('#settingsOverlay .set-nav-item').forEach(el => {
        const on = el.dataset.tab === tab;
        el.classList.toggle('active', on);
        // On a phone the navigation is one row: bring the open page into it
        if (on) el.scrollIntoView?.({block: 'nearest', inline: 'nearest'});
    });
    document.querySelectorAll('#settingsOverlay .set-page').forEach(el =>
        el.classList.toggle('active', el.id === 'ss-' + tab));
    _settingsSubNav(tab);
    const page = _settingsPage(tab);
    if (page?.open) {
        try { page.open(); } catch (e) { console.error('Settings page open:', e); }
    }
}

function _settingsNavRender() {
    const nav = document.getElementById('settingsNavItems');
    if (!nav || typeof SETTINGS_PAGES === 'undefined') return;
    let group = null;
    nav.innerHTML = SETTINGS_PAGES.map(p => {
        const head = p.group !== group ? `<div class="set-nav-group">${esc(p.group)}</div>` : '';
        group = p.group;
        return head + `<button type="button" class="set-nav-item" data-tab="${p.id}"
                onclick="setSettingsTab('${p.id}')" title="${esc(p.hint || p.title)}">
                <span class="set-nav-icon">${_sIcon(p.icon)}</span>
                <span class="set-nav-label">${esc(p.nav || p.title)}</span>
                <span class="set-nav-count"></span></button>
            <div class="set-nav-sub" data-sub="${p.id}" hidden></div>`;
    }).join('');
}

/* The open page's sections under its name in the navigation — or its
 * cards, when it has no sections: what is on the page at a glance, one
 * click to each. The one in view is marked while scrolling. */
function _settingsSubNav(tab) {
    document.querySelectorAll('#settingsOverlay .set-nav-sub').forEach(box => {
        const mine = box.dataset.sub === tab;
        box.hidden = !mine;
        if (!mine) { box.innerHTML = ''; return; }
        const page = document.getElementById('ss-' + tab);
        if (!page) return;
        let marks = [...page.querySelectorAll(':scope > .set-section')];
        if (!marks.length) marks = [...page.querySelectorAll(':scope > .set-card')].filter(c => !c.hidden);
        box.innerHTML = marks.length < 2 ? '' : marks.map(el => {
            const title = el.querySelector('h3')?.textContent.trim() || '';
            return `<button type="button" class="set-nav-subitem" data-target="${esc(el.id)}"
                onclick="_settingsJump('${escJs(el.id)}')">${esc(title)}</button>`;
        }).join('');
    });
    _settingsSpy();
}

// Scroll a section or card to the top of the page — a folded card opens
// for this visit (its remembered state stays)
function _settingsJump(id) {
    const el = document.getElementById(id);
    const content = document.getElementById('settingsContent');
    if (!el || !content) return;
    if (el.classList.contains('set-card')) el.classList.remove('folded');
    const top = el.getBoundingClientRect().top - content.getBoundingClientRect().top;
    content.scrollTop += top - 12;
    _settingsSpy();
}

// Which section or card of the open page is in view: the last one whose
// top has passed the upper edge
function _settingsSpy() {
    const content = document.getElementById('settingsContent');
    const box = document.querySelector(`#settingsOverlay .set-nav-sub[data-sub="${_activeSettingsTab}"]`);
    if (!content || !box) return;
    const items = [...box.querySelectorAll('.set-nav-subitem')];
    if (!items.length) return;
    const edge = content.getBoundingClientRect().top + 90;
    let current = items[0];
    for (const it of items) {
        const el = document.getElementById(it.dataset.target);
        if (el && el.getBoundingClientRect().top <= edge) current = it;
    }
    // At the very bottom the last one counts, however short it is
    if (content.scrollTop + content.clientHeight >= content.scrollHeight - 4) current = items[items.length - 1];
    items.forEach(it => it.classList.toggle('current', it === current));
}

// ── Drawing ──────────────────────────────────────────────────────────────

function _renderSettings() {
    if (!_settingsData || typeof SETTINGS_PAGES === 'undefined') return;
    const content = document.getElementById('settingsContent');
    if (!content) return;
    const scroll = content.scrollTop;
    _settingsNavRender();
    _settingsNotes();
    const cfg = _settingsData.config || {};
    for (const p of SETTINGS_PAGES) {
        let sec = document.getElementById('ss-' + p.id);
        if (!sec) {
            sec = document.createElement('section');
            sec.className = 'set-page';
            sec.id = 'ss-' + p.id;
            content.appendChild(sec);
        }
        let body;
        try { body = p.render(cfg); } catch (e) {
            console.error('Settings render error:', p.id, e);
            body = `<div class="set-note err">This page could not be drawn: ${esc(e.message)}</div>`;
        }
        sec.innerHTML = `<header class="set-page-head"><h2>${_sIcon(p.icon, 'set-page-icon')}${esc(p.title)}</h2>`
            + (p.intro ? `<p>${p.intro}</p>` : '') + '</header>' + body;
        // Every card can be jumped to from the navigation
        sec.querySelectorAll(':scope > .set-card').forEach((card, i) => {
            if (!card.id) card.id = `set-card-${p.id}-${i}`;
        });
    }
    _sAfterRender(content);
    for (const p of SETTINGS_PAGES) {
        if (!p.after) continue;
        try { p.after(cfg); } catch (e) { console.error('Settings page after:', p.id, e); }
    }
    _ssWatch();
    _showSettingsTab(_settingsPageId(_activeSettingsTab));
    _sRefreshDirty();
    if (!content.dataset.spy) {
        content.dataset.spy = '1';
        content.addEventListener('scroll', _settingsSpy, {passive: true});
    }
    if (_ssTerms(_ssQuery).length) _applySettingsSearch();
    else if (_settingsOpenSection) {
        content.scrollTop = 0;
        _settingsJump('set-sec-' + _settingsOpenSection);
    } else content.scrollTop = scroll;
    _settingsOpenSection = '';
}

// What config.validate() says about the saved values
function _settingsNotes() {
    const el = document.getElementById('settingsNotes');
    if (!el) return;
    const warnings = _settingsData?.warnings || [];
    el.innerHTML = warnings.map(w => `<div class="set-note warn">⚠ ${esc(w)}</div>`).join('');
}

// Value of a config key: saved config, else `fallback`
function _cfgVal(dotPath, fallback = '') {
    let node = _settingsData?.config;
    for (const k of String(dotPath).split('.')) {
        if (node == null || typeof node !== 'object') return fallback;
        node = node[k];
    }
    return node ?? fallback;
}

// Default of a config key (config.default.yaml), undefined if it has none
function _defVal(dotPath) {
    let node = _settingsData?.defaults;
    for (const k of String(dotPath).split('.')) {
        if (node == null || typeof node !== 'object') return undefined;
        node = node[k];
    }
    return node ?? undefined;
}

const _S_TAGS = {
    jobs: ['jobs', 'New jobs', 'The start value for new transcriptions — the transcription dialog can change it per job, and queued jobs keep the value they were queued with'],
    restart: ['restart', 'Restart', 'Read when the server starts — takes effect after a restart'],
    worker: ['restart', 'Worker restart', 'Takes effect when the diarization worker starts again — it restarts by itself after a cancel, a crash or its memory limit, or with the server'],
    mcp: ['restart', 'MCP restart', 'Read by the MCP server when it starts — restart it (or this app, if the app starts it)'],
    secret: ['secret', 'Secret', 'Never sent back to the browser; leave the field as it is to keep the saved value'],
};

function _sTags(tags) {
    return (tags || []).map(t => {
        const [cls, text, title] = _S_TAGS[t] || ['', t, ''];
        return `<span class="set-tag ${cls}" title="${esc(title)}">${esc(text)}</span>`;
    }).join('');
}

/* A section of a page that covers more than one topic: a heading above its
 * cards. The navigation lists a page's sections; the search hides them and
 * shows the matching cards only. */
function _sSection({id, title, sub = ''}) {
    return `<div class="set-section" id="set-sec-${esc(id)}">
        <h3>${title}</h3>${sub ? `<p>${sub}</p>` : ''}</div>`;
}

/* A card: title, optional line below it, optional actions on the right,
 * then rows and blocks. `fold` makes it foldable (`true` = starts folded,
 * `false` = starts open); what the user chose is remembered. */
function _sCard({title, sub = '', actions = '', body = '', id = '', fold, cls = '', hidden = false,
                 onopen = ''}) {
    const foldable = fold !== undefined;
    return `<section class="set-card${foldable ? ' foldable' : ''}${cls ? ' ' + cls : ''}"${id ? ` id="${id}"` : ''}
                ${foldable ? `data-fold="${fold ? 1 : 0}"` : ''}${onopen ? ` data-onopen="${onopen}"` : ''}${hidden ? ' hidden' : ''}>
        <div class="set-card-head"${foldable ? ' onclick="_sFoldToggle(event, this)"' : ''}>
            ${foldable ? '<span class="set-fold">▼</span>' : ''}
            <div class="set-card-titles"><h3>${title}</h3>${sub ? `<div class="set-card-sub">${sub}</div>` : ''}</div>
            ${actions ? `<div class="set-card-actions">${actions}</div>` : ''}
        </div>
        <div class="set-card-body">${body}</div>
    </section>`;
}

// A free block inside a card (status, list, editor): searched like a row
function _sBlock(html, {id = '', skip = false, cls = ''} = {}) {
    return `<div class="set-block${cls ? ' ' + cls : ''}"${id ? ` id="${id}"` : ''}${skip ? ' data-search-skip' : ''}>${html}</div>`;
}

/* One setting.
 *
 *   key      config key; the row reads and writes it
 *   type     toggle · number · text · secret · select · seg · textarea ·
 *            list (comma list) · lines (one per line) · checks · dir · static
 *   label    the name — short, the thing it controls
 *   desc     one or two sentences: what it does, when to change it
 *   more     the details: measurements, examples, the why (folded)
 *   tags     'jobs' · 'restart' · 'worker' · 'mcp' · 'secret'
 *   unit, min, max, step       number
 *   options  [[value, label], …]  select, seg, checks — value keeps its type
 *   wide     control below the text, full width (paths, long text)
 */
function _sRow(f) {
    if (f.show === false) return '';
    const value = f.value !== undefined ? f.value
        : (f.key ? _cfgVal(f.key, f.fallback ?? '') : '');
    const def = f.key ? (f.def !== undefined ? f.def : _defVal(f.key)) : undefined;
    const wide = f.wide || ['textarea', 'lines', 'dir'].includes(f.type);
    const reset = f.key && def !== undefined && f.reset !== false && f.type !== 'static'
        ? `<button type="button" class="set-reset" onclick="_sResetRow(this)"
               title="Back to the default: ${esc(_sShow(def))}">↺</button>`
        : '<span class="set-reset" aria-hidden="true"></span>';
    return `<div class="set-row${wide ? ' wide' : ''}"${f.key ? ` data-row="${esc(f.key)}"` : ''}${f.id ? ` id="${f.id}"` : ''}${f.hidden ? ' hidden' : ''}
                ${def !== undefined ? ` data-def="${esc(JSON.stringify(def))}"` : ''}>
        <div class="set-text">
            <div class="set-label"${f.key ? ` title="config.yaml: ${esc(f.key)}"` : ''}>${f.label}${_sTags(f.tags)}</div>
            ${f.desc ? `<div class="set-desc">${f.desc}</div>` : ''}
            ${f.more ? `<details class="set-more"><summary>Details</summary><div class="set-more-body">${f.more}</div></details>` : ''}
        </div>
        <div class="set-ctl">${_sControl(f, value)}${reset}</div>
    </div>`;
}

// A default as a person reads it
function _sShow(v) {
    if (v === true) return 'on';
    if (v === false) return 'off';
    if (Array.isArray(v)) return v.length ? v.join(', ') : '(empty)';
    if (v === '' || v == null) return '(empty)';
    return String(v);
}

function _sTypeOf(value) {
    if (typeof value === 'boolean') return 'bool';
    if (typeof value === 'number') return 'num';
    return 'str';
}

function _sControl(f, value) {
    const key = f.key ? ` data-key="${esc(f.key)}"` : '';
    const id = f.ctlId ? ` id="${f.ctlId}"` : '';
    const ph = f.placeholder !== undefined ? ` placeholder="${esc(f.placeholder)}"` : '';
    const on = f.onchange ? ` onchange="${f.onchange}"` : '';
    const mono = f.mono === false ? '' : ' mono';
    switch (f.type) {
        case 'toggle':
            return `<label class="set-switch"><input type="checkbox" role="switch"${key}${id} data-type="bool"
                        ${_sTruthy(value) ? 'checked' : ''}${on} aria-label="${esc(_sPlain(f.label))}"></label>`;
        case 'number': {
            const attrs = ['min', 'max', 'step'].filter(a => f[a] !== undefined)
                .map(a => ` ${a}="${f[a]}"`).join('');
            return `<span class="set-num${f.unit ? ' has-unit' : ''}"><input type="number"${key}${id} data-type="num"
                        value="${esc(value === '' ? '' : String(value))}"${attrs}${ph}${on}
                        aria-label="${esc(_sPlain(f.label))}">${f.unit ? `<span class="set-unit">${esc(f.unit)}</span>` : ''}</span>`;
        }
        case 'secret':
            return `<input class="set-input mono" type="password" autocomplete="new-password"${key}${id} data-type="str"
                        value="${esc(value)}"${ph}${on} aria-label="${esc(_sPlain(f.label))}">`;
        case 'select': {
            const t = f.valueType || _sTypeOf(f.options?.[0]?.[0]);
            const cur = _sCoerce(value, t);
            return `<select class="set-select"${key}${id} data-type="${t}"${on} aria-label="${esc(_sPlain(f.label))}">
                ${(f.options || []).map(([v, label]) =>
                    `<option value="${esc(String(v))}" ${String(v) === String(cur) ? 'selected' : ''}>${esc(label)}</option>`).join('')}
            </select>`;
        }
        case 'seg': {
            const t = f.valueType || _sTypeOf(f.options?.[0]?.[0]);
            const cur = _sCoerce(value, t);
            return `<div class="set-seg" role="radiogroup" aria-label="${esc(_sPlain(f.label))}">
                ${(f.options || []).map(([v, label]) =>
                    `<button type="button" role="radio" data-value="${esc(String(v))}"
                        class="${String(v) === String(cur) ? 'active' : ''}" aria-checked="${String(v) === String(cur)}"
                        onclick="_sSeg(this)">${esc(label)}</button>`).join('')}
                <input type="hidden"${key}${id} data-type="${t}" value="${esc(String(cur))}"${on}>
            </div>`;
        }
        case 'textarea':
            return `<textarea class="set-textarea${mono}"${key}${id} data-type="str" rows="${f.rows || 4}"${ph}${on}
                        spellcheck="false" aria-label="${esc(_sPlain(f.label))}">${esc(value)}</textarea>`;
        case 'lines':
            return `<textarea class="set-textarea mono"${key}${id} data-type="lines" rows="${f.rows || 4}"${ph}${on}
                        spellcheck="false" aria-label="${esc(_sPlain(f.label))}">${esc(_arrToStr(value, '', '\n'))}</textarea>`;
        case 'list':
            return `<input class="set-input mono"${key}${id} data-type="list" value="${esc(_arrToStr(value))}"${ph}${on}
                        aria-label="${esc(_sPlain(f.label))}">`;
        case 'checks': {
            // `invert`: the saved list holds the UNticked options (e.g. the
            // built-in phrases that are switched off)
            const have = new Set(Array.isArray(value) ? value : _arrToStr(value).split(',').map(s => s.trim()));
            return `<div class="set-checks"${f.invert ? ' data-invert' : ''}>
                ${(f.options || []).map(([v, label]) => `<label><input type="checkbox" value="${esc(v)}"
                    ${have.has(v) !== !!f.invert ? 'checked' : ''} onchange="_sChecks(this)">${esc(label)}</label>`).join('')}
                <input type="hidden"${key}${id} data-type="list" value="${esc([...have].filter(Boolean).join('\n'))}"${on}>
            </div>`;
        }
        case 'dir':
            return `<div style="width:100%">
                <div class="set-path">
                    <input class="set-input mono"${key}${id} data-type="str" value="${esc(value)}"${ph}${on}
                           spellcheck="false" aria-label="${esc(_sPlain(f.label))}">
                    <button type="button" class="set-btn" onclick="_sBrowse(this)" title="Choose a folder">📁 Browse</button>
                </div>
                <div class="set-path-state" data-path-state="${esc(f.dir || '')}"></div>
                <div class="set-browser" hidden></div>
            </div>`;
        case 'static':
            return `<span class="set-result">${value}</span>`;
        default:
            return `<input class="set-input${mono}"${key}${id} data-type="str" value="${esc(value)}"${ph}${on}
                        spellcheck="false" aria-label="${esc(_sPlain(f.label))}">`;
    }
}

function _sPlain(html) { return String(html ?? '').replace(/<[^>]*>/g, '').trim(); }

function _sTruthy(v) {
    if (typeof v === 'string') return ['true', '1', 'yes', 'on'].includes(v.trim().toLowerCase());
    return !!v;
}

function _sCoerce(v, type) {
    if (type === 'bool') return _sTruthy(v);
    if (type === 'num') { const n = Number(v); return Number.isFinite(n) ? n : v; }
    return v ?? '';
}

// ── Reading and writing a control ────────────────────────────────────────

// The typed value of a control — undefined means "don't save" (an empty or
// invalid number would otherwise become 0 and silently cripple a limit).
// Empty is a value only where the default is empty: "automatic".
function _sRead(el) {
    if (el.type === 'checkbox') return !!el.checked;
    const t = el.dataset?.type || (el.type === 'number' ? 'num' : 'str');
    const raw = String(el.value ?? '');
    if (t === 'num') {
        if (raw.trim() === '') return el.closest?.('.set-row')?.dataset.def === '""' ? '' : undefined;
        let n = Number(raw);
        if (!Number.isFinite(n)) return undefined;
        if (el.min !== undefined && el.min !== '' && n < Number(el.min)) n = Number(el.min);
        if (el.max !== undefined && el.max !== '' && n > Number(el.max)) n = Number(el.max);
        return n;
    }
    if (t === 'bool') return raw === 'true';
    // A hidden check-list value is one per line — options may hold commas
    if (t === 'list') return raw.split(el.type === 'hidden' ? '\n' : ',').map(s => s.trim()).filter(Boolean);
    if (t === 'lines') return raw.split('\n').map(s => s.trim()).filter(Boolean);
    return el.tagName === 'INPUT' ? raw.trim() : raw;
}

function _sWrite(el, value) {
    const t = el.dataset?.type;
    if (el.type === 'checkbox') el.checked = _sTruthy(value);
    else if (t === 'list') el.value = Array.isArray(value) ? value.join(el.type === 'hidden' ? '\n' : ', ') : String(value ?? '');
    else if (t === 'lines') el.value = Array.isArray(value) ? value.join('\n') : String(value ?? '');
    else el.value = value == null ? '' : String(value);
    // Controls that mirror a hidden value
    const seg = el.closest?.('.set-seg');
    if (seg) seg.querySelectorAll('button').forEach(b => {
        const on = b.dataset.value === el.value;
        b.classList.toggle('active', on);
        b.setAttribute('aria-checked', String(on));
    });
    const checks = el.closest?.('.set-checks');
    if (checks) {
        const have = new Set(el.value.split('\n').map(s => s.trim()));
        const invert = checks.hasAttribute('data-invert');
        checks.querySelectorAll('input[type=checkbox]').forEach(c => { c.checked = have.has(c.value) !== invert; });
    }
    el.dispatchEvent?.(new Event('change', {bubbles: true}));
}

function _sSeg(btn) {
    const input = btn.closest('.set-seg')?.querySelector('input[type=hidden]');
    if (input) _sWrite(input, btn.dataset.value);
}

function _sChecks(box) {
    const wrap = box.closest('.set-checks');
    const input = wrap?.querySelector('input[type=hidden]');
    if (!input) return;
    const invert = wrap.hasAttribute('data-invert');
    input.value = [...wrap.querySelectorAll('input[type=checkbox]')]
        .filter(c => c.checked !== invert).map(c => c.value).join('\n');
    input.dispatchEvent(new Event('change', {bubbles: true}));
}

function _sResetRow(btn) {
    const row = btn.closest('.set-row');
    const el = row?.querySelector('[data-key]');
    if (!el || row.dataset.def === undefined) return;
    let def;
    try { def = JSON.parse(row.dataset.def); } catch (e) { return; }
    _sWrite(el, def);
}

function _sChanged(el) {
    if (el.dataset?.initial === undefined) return true;
    return JSON.stringify(_sRead(el) ?? null) !== el.dataset.initial;
}

// After drawing: note every control's value, mark rows that differ from
// the default, wire the foldable cards
function _sAfterRender(root) {
    root.querySelectorAll('[data-key]').forEach(el => {
        el.dataset.initial = JSON.stringify(_sRead(el) ?? null);
    });
    root.querySelectorAll('.set-row[data-def]').forEach(row => {
        const el = row.querySelector('[data-key]');
        if (!el) return;
        let def;
        try { def = JSON.parse(row.dataset.def); } catch (e) { return; }
        row.classList.toggle('custom', !_sSameAsDefault(el, def));
    });
    root.querySelectorAll('.set-card[data-fold]').forEach(card => {
        const key = _sFoldKey(card);
        let folded = card.dataset.fold === '1';
        try {
            const kept = localStorage.getItem(key);
            if (kept !== null) folded = kept === '1';
        } catch (e) { /* storage off */ }
        card.classList.toggle('folded', folded);
        if (!folded) _sCardOpened(card);
    });
}

// A foldable card shown open: its `onopen` function (a global's name) fills
// what it only loads on request
function _sCardOpened(card) {
    const fn = card.dataset.onopen;
    if (fn && typeof window[fn] === 'function') window[fn]();
}

// Is the drawn value the default? Compared the way the control reads it
function _sSameAsDefault(el, def) {
    const t = el.dataset.type;
    let d = def;
    if (t === 'bool') d = _sTruthy(def);
    else if (t === 'num') d = def === '' ? '' : Number(def);
    else if (t === 'list' || t === 'lines') d = Array.isArray(def) ? def : _arrToStr(def).split(',').map(s => s.trim()).filter(Boolean);
    else d = def == null ? '' : String(def);
    return JSON.stringify(d) === el.dataset.initial;
}

// By the card's own id or its title — not by the numbered id every card
// gets for the navigation, which shifts when a card is added
function _sFoldKey(card) {
    const own = card.id && !card.id.startsWith('set-card-') ? card.id : '';
    return 'setFold:' + (card.closest('.set-page')?.id || '') + ':'
        + (own || card.querySelector('h3')?.textContent.trim().slice(0, 60) || '');
}

function _sFoldToggle(ev, head) {
    // Controls inside the head (Test, Guide …) do not fold the card
    if (ev.target.closest('button, input, select, a, textarea, label')) return;
    // During a search the hit cards are held open
    if (_ssTerms(_ssQuery).length) return;
    const card = head.closest('.set-card');
    const folded = card.classList.toggle('folded');
    try { localStorage.setItem(_sFoldKey(card), folded ? '1' : '0'); } catch (e) { /* storage off */ }
    if (!folded) _sCardOpened(card);
}

// ── Unsaved changes ──────────────────────────────────────────────────────

// Mark a change in an editor that is not a row (key = the config key it
// writes). Rows notice their own changes.
function _markDirty(key = 'other') {
    _settingsExtraDirty.add(key);
    _sRefreshDirty();
}

function _settingsUnsavedCount() {
    let n = _settingsExtraDirty.size;
    document.querySelectorAll('#settingsOverlay .set-row [data-key]').forEach(el => {
        if (_sChanged(el)) n++;
    });
    return n;
}

let _sDirtyQueued = false;
function _sRefreshDirty() {
    if (_sDirtyQueued) return;
    _sDirtyQueued = true;
    Promise.resolve().then(() => {
        _sDirtyQueued = false;
        const pages = new Set();
        let n = 0;
        document.querySelectorAll('#settingsOverlay .set-row').forEach(row => {
            const el = row.querySelector('[data-key]');
            const dirty = !!el && _sChanged(el);
            row.classList.toggle('dirty', dirty);
            if (dirty) { n++; pages.add(row.closest('.set-page')?.id); }
        });
        for (const key of _settingsExtraDirty) {
            n++;
            const page = SETTINGS_PAGES.find(p => (p.owns || []).some(k => key.startsWith(k)));
            if (page) pages.add('ss-' + page.id);
        }
        document.querySelectorAll('#settingsOverlay .set-nav-item').forEach(el =>
            el.classList.toggle('dirty', pages.has('ss-' + el.dataset.tab)));
        const label = document.getElementById('settingsUnsaved');
        if (label) {
            label.hidden = !n;
            label.textContent = `${n} unsaved ${n === 1 ? 'change' : 'changes'}`;
        }
        const save = document.getElementById('settingsSave');
        if (save) save.disabled = !n;
        const discard = document.getElementById('settingsDiscard');
        if (discard) discard.hidden = !n;
    });
}

// Every edit in the dialog updates the unsaved marks
['input', 'change'].forEach(type => document.addEventListener(type, e => {
    if (e.target?.closest?.('#settingsOverlay') && e.target.closest('.set-row')) _sRefreshDirty();
}));

// ── Save ─────────────────────────────────────────────────────────────────

/* What Save sends: the rows whose value changed (a control without
 * `data-initial` counts as changed), plus the editors that noted a change.
 *
 * Only inside the settings dialog: the file list and the queue editor carry
 * data attributes too — a document-wide query once picked up the editor's
 * folder rows (`data-dir` on a <div>) and every save threw. */
function _collectFormValues() {
    const result = {};
    const scope = document.getElementById('settingsOverlay') || document;
    scope.querySelectorAll('[data-key]').forEach(el => {
        if (!_sChanged(el)) return;
        const value = _sRead(el);
        if (value === undefined) return;
        _sPut(result, el.dataset.key, value);
    });
    const llm = _settingsData?.config?.llm;
    if (llm && _settingsExtraDirty.has('llm.presets')) _sPut(result, 'llm.presets', llm.presets || {});
    if (llm && _settingsExtraDirty.has('llm.models')) _sPut(result, 'llm.models', llm.models || {});
    const tx = _settingsData?.config?.transcription;
    if (tx && _settingsExtraDirty.has('transcription.vocab_presets')) {
        _sPut(result, 'transcription.vocab_presets', tx.vocab_presets || {});
    }
    // Providers and their tasks live in the provider cards. The API keys are
    // never sent back to the browser: a key goes out only when one was typed.
    if (_settingsExtraDirty.has('llm.providers') && typeof _provData !== 'undefined'
            && _provData?.providers) {
        _sPut(result, 'llm.providers', _provData.providers.map(p => ({
            id: p.id, name: p.name, kind: p.kind, endpoint: p.endpoint,
            ...(p.api_key ? {api_key: p.api_key} : {}),
            models: p.models || [], default_model: p.default_model || '',
            active: p.active !== false,
        })));
        _sPut(result, 'llm.tasks', _provData.tasks || {});
    }
    return result;
}

function _sPut(obj, dotted, value) {
    const keys = dotted.split('.');
    let node = obj;
    for (let i = 0; i < keys.length - 1; i++) {
        if (!node[keys[i]] || typeof node[keys[i]] !== 'object') node[keys[i]] = {};
        node = node[keys[i]];
    }
    node[keys[keys.length - 1]] = value;
}

function _sFlatKeys(obj, prefix = '') {
    return Object.entries(obj).flatMap(([k, v]) =>
        v && typeof v === 'object' && !Array.isArray(v) && !['presets', 'models', 'vocab_presets', 'tasks'].includes(k)
            ? _sFlatKeys(v, prefix + k + '.') : [prefix + k]);
}

async function _sPutSettings(updates) {
    const r = await fetch('/api/settings?save=true', {
        method: 'PUT',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(updates),
    });
    const d = await r.json().catch(() => ({}));
    // An error response is not a success: the unsaved marks stay, and the
    // form is not redrawn over what was typed
    if (!r.ok || d.error) throw new Error(d.error || `HTTP ${r.status}`);
    return d;
}

async function saveSettings() {
    const updates = _collectFormValues();
    const keys = _sFlatKeys(updates);
    if (!keys.length) { _showToast('Nothing changed'); return; }
    // Which of them only take effect after a restart — named in the toast
    const later = keys.filter(k => {
        const row = document.querySelector(`#settingsOverlay .set-row[data-row="${CSS.escape(k)}"]`);
        return row?.querySelector('.set-tag.restart');
    });
    const btn = document.getElementById('settingsSave');
    if (btn) btn.disabled = true;
    try {
        const d = await _sPutSettings(updates);
        _settingsExtraDirty.clear();
        if (d.errors && d.errors.length) {
            _showToast('Saved, but not everything: ' + d.errors.join(' · '), true);
        } else if (later.length) {
            _showToast(`Saved — ${later.length === 1 ? 'one setting takes' : later.length + ' settings take'} effect after a restart`);
        } else {
            _showToast('Settings saved');
        }
        await _loadAllSettings();
        // The transcription dialogs take their start values once, when the
        // page loads — without this a changed default needed a reload
        if (typeof _applySettingsDefaults === 'function') _applySettingsDefaults();
    } catch (e) {
        _showToast('Save failed: ' + e.message, true);
        _sRefreshDirty();
    }
}

/* Save before a tool runs with the values on screen (an embedding test, an
 * index run). Without a redraw: the rows just take what was sent as their
 * saved state. */
async function _settingsSaveQuiet() {
    const updates = _collectFormValues();
    if (!_sFlatKeys(updates).length) return;
    await _sPutSettings(updates);
    document.querySelectorAll('#settingsOverlay [data-key]').forEach(el => {
        const v = _sRead(el);
        el.dataset.initial = JSON.stringify(v ?? null);
        if (v !== undefined && _settingsData?.config) _sPut(_settingsData.config, el.dataset.key, v);
    });
    _settingsExtraDirty.clear();
    _sRefreshDirty();
}

// ── Choosing a folder ────────────────────────────────────────────────────
//
// A small browser under the path field (/api/browse). It stays inside the
// allowed trees (security.allowed_roots) — a folder outside them, like the
// work folder on the local SSD, is typed.

function _sBrowse(btn) {
    const wrap = btn.closest('.set-ctl');
    const box = wrap?.querySelector('.set-browser');
    const input = wrap?.querySelector('input[data-key]');
    if (!box || !input) return;
    if (!box.hidden) { box.hidden = true; return; }
    _sBrowseClose();
    box.hidden = false;
    _sBrowseTo(box, input, input.value.trim() || input.placeholder.replace(/^Automatic: /, ''));
}

async function _sBrowseTo(box, input, path, note = '') {
    // Only the newest listing of this box may land: clicking folder A and
    // then B on a slow NAS must not end on A (and "Use this folder" pick A).
    const seq = box._seq = (box._seq || 0) + 1;
    box.innerHTML = '<div class="set-browser-msg">Loading…</div>';
    let d;
    try {
        const r = await fetch('/api/browse?path=' + encodeURIComponent(path || ''));
        d = await r.json();
        if (seq !== box._seq) return;
        if (!r.ok && !d.dirs) throw new Error(d.error || `HTTP ${r.status}`);
    } catch (e) {
        if (seq !== box._seq) return;
        box.innerHTML = `<div class="set-browser-msg">${esc(e.message)} — type the path into the field instead.</div>`;
        return;
    }
    // A folder outside the allowed trees (a work folder on a local SSD, say)
    // can't be listed: the answer names where browsing may start instead —
    // show that listing, with the reason on top
    if (d.error && !(d.dirs || []).length && d.path && d.path !== path && !note) {
        _sBrowseTo(box, input, d.path, `${path} can't be browsed: ${d.error} Pick a folder below, or type the path into the field.`);
        return;
    }
    box._path = d.path;
    box._input = input;
    box.innerHTML = `${note ? `<div class="set-browser-msg warn">${esc(note)}</div>` : ''}
        <div class="set-browser-head">
            <span class="set-browser-path" title="${esc(d.path)}">${esc(d.path)}</span>
            <button type="button" class="set-btn sm primary" onclick="_sBrowsePick(this)">Use this folder</button>
            <button type="button" class="set-btn sm" onclick="_sBrowseClose()">Cancel</button>
        </div>
        <div class="set-browser-list">
            ${(d.dirs || []).map(x => `<button type="button" data-path="${esc(x.path)}"
                onclick="_sBrowseTo(this.closest('.set-browser'), this.closest('.set-browser')._input, this.dataset.path)">
                ${x.name === '..' ? '↑ ..' : '📁 ' + esc(x.name)}</button>`).join('')
              || '<div class="set-browser-msg">No subfolders</div>'}
            ${d.error ? `<div class="set-browser-msg">${esc(d.error)}</div>` : ''}
        </div>`;
}

function _sBrowsePick(btn) {
    const box = btn.closest('.set-browser');
    if (!box?._input || !box._path) return;
    box._input.value = box._path;
    box._input.dispatchEvent(new Event('change', {bubbles: true}));
    box.hidden = true;
}

function _sBrowseClose() {
    document.querySelectorAll('#settingsOverlay .set-browser').forEach(b => { b.hidden = true; });
}

// ── Settings search ──────────────────────────────────────────────────────
//
// Filters all pages at once: what does not match is hidden, the hits of
// every page appear one below the other, each under its page heading.
// Nothing is re-rendered — the filter only sets classes on the existing
// rows, so a value changed in a hit is saved as usual, and clearing the
// search leaves every page exactly as it was.
//
// Searched per row or block: the visible text (name, description, select
// options) and the config keys (`hardware.low_vram`), with the page and
// card title as context — see _ssPick for how they combine. Query words
// match at the start of a word (_ssHas); `_` and `-` count as spaces, so
// "low vram", "low-vram" and "low_vram" are the same query, and "lowvram"
// matches too.
//
// Not searched: the folded details under a description, whatever the page
// hides itself (both stay hidden), and live data marked data-search-skip
// (log list, DB explorer, test-search results) — a file name in the DB
// explorer is not a setting.

let _ssQuery = '';
let _ssObserver = null;
let _ssScrollBack = 0;      // where the open page was when the search began

const _SS_SKIP = '.set-more, [data-search-skip]';

function _ssNorm(s) {
    return String(s ?? '').toLowerCase().replace(/[_-]+/g, ' ').replace(/\s+/g, ' ').trim();
}

// The query as the words that all have to match
function _ssTerms(query) {
    return [...new Set(_ssNorm(query).split(' ').filter(Boolean))];
}

const _SS_WORD_CHAR = /[\p{L}\p{N}]/u;

// A text prepared for _ssHas: its words run together, and where each starts
function _ssHay(text) {
    const hay = _ssNorm(text);
    let joined = '';
    const starts = [];
    for (const w of hay.split(/[^\p{L}\p{N}]+/u)) {
        if (!w) continue;
        starts.push(joined.length);
        joined += w;
    }
    return {hay, joined, starts};
}

// A query word has to begin a word of the text — "port" finds Port, not
// export or support; "log" finds Logs, not dialog. Written together it may
// run on into the following words ("lowvram" → Low-VRAM). A query word
// with punctuation ("0.5", "hardware.low") is looked up as it is.
function _ssHas(h, term) {
    if (/[^\p{L}\p{N}]/u.test(term)) return h.hay.includes(term);
    return h.starts.some(p => h.joined.startsWith(term, p));
}

// Which units (rows/blocks) of a card are hits — the best of three tiers:
//   1. units that contain every query word themselves;
//   2. else units that contain at least one, the rest being in their
//      headings (page + card title): "huggingface token" finds the
//      token row of the HuggingFace card;
//   3. else the whole card, when its headings alone contain every word.
// Tier 1 keeps a heading word from pulling in its neighbours: in "llm
// timeout" every LLM row has "llm" in its config key, but only the rows
// that also say "timeout" are hits.
function _ssPick(context, unitTexts, terms) {
    const ctx = _ssHay(context);
    const own = unitTexts.map(_ssHay);
    const full = own.map(h => terms.every(t => _ssHas(h, t)));
    if (full.some(Boolean)) return full;
    const mixed = own.map(h => terms.some(t => _ssHas(h, t))
                            && terms.every(t => _ssHas(h, t) || _ssHas(ctx, t)));
    if (mixed.some(Boolean)) return mixed;
    const whole = terms.every(t => _ssHas(ctx, t));
    return own.map(() => whole);
}

// Hidden by the page itself (`hidden` or inline display:none), up to `stop`
// — e.g. the fields of the embedding mode that is not selected. The search
// never reveals those.
function _ssHiddenByPage(el, stop) {
    for (let k = el; k && k !== stop; k = k.parentElement) {
        if (k.hidden || k.style?.display === 'none') return true;
    }
    return false;
}

// Searchable text of a row or block, plus its config keys
function _ssText(el) {
    const parts = [];
    const walker = document.createTreeWalker(el, NodeFilter.SHOW_TEXT);
    for (let n = walker.nextNode(); n; n = walker.nextNode()) {
        const p = n.parentElement;
        if (!p || p.closest(_SS_SKIP) || _ssHiddenByPage(p, el)) continue;
        parts.push(n.nodeValue);
    }
    el.querySelectorAll('[data-key]').forEach(c => parts.push(c.dataset.key));
    return parts.join(' ');
}

function settingsSearch(value) {
    const content = document.getElementById('settingsContent');
    const was = _ssTerms(_ssQuery).length > 0;
    if (!was && content) _ssScrollBack = content.scrollTop;
    _ssQuery = String(value ?? '');
    const clear = document.getElementById('settingsSearchClear');
    if (clear) clear.hidden = !_ssQuery;
    _ssWatch();
    _applySettingsSearch();
    if (!content) return;
    // New results start at the top; a cleared search returns to where the
    // open page was
    if (_ssTerms(_ssQuery).length) content.scrollTop = 0;
    else if (was) content.scrollTop = _ssScrollBack;
}

function _settingsSearchReset() {
    const input = document.getElementById('settingsSearch');
    if (input) input.value = '';
    settingsSearch('');
}

function _applySettingsSearch() {
    const content = document.getElementById('settingsContent');
    if (!content) return;
    content.querySelectorAll('.ss-hit, .ss-miss, .ss-match').forEach(el =>
        el.classList.remove('ss-hit', 'ss-miss', 'ss-match'));
    const nav = document.querySelectorAll('#settingsOverlay .set-nav-item');
    const terms = _ssTerms(_ssQuery);
    document.getElementById('settingsNavItems')?.classList.toggle('searching', terms.length > 0);
    if (!terms.length) {
        content.classList.remove('searching');
        nav.forEach(el => {
            el.classList.remove('ss-none');
            const badge = el.querySelector('.set-nav-count');
            if (badge) badge.textContent = '';
        });
        _ssNote(content, '');
        _ssHighlight(null, []);
        return;
    }
    content.classList.add('searching');
    const counts = {};
    let total = 0;
    content.querySelectorAll(':scope > .set-page').forEach(sec => {
        const tab = sec.id.replace(/^ss-/, '');
        const page = _settingsPage(tab);
        let hits = 0;
        sec.querySelectorAll(':scope > .set-card').forEach(card => {
            if (_ssHiddenByPage(card, sec)) return;
            const title = card.querySelector(':scope > .set-card-head');
            const body = card.querySelector(':scope > .set-card-body');
            if (!body) return;
            const context = `${page?.nav || ''} ${page?.title || tab} ${title ? title.textContent : ''}`;
            const units = [...body.children].filter(u => !_ssHiddenByPage(u, body));
            const picks = _ssPick(context, units.map(_ssText), terms);
            let cardHits = 0;
            units.forEach((unit, i) => {
                unit.classList.add(picks[i] ? 'ss-match' : 'ss-miss');
                if (picks[i]) cardHits++;
            });
            card.classList.add(cardHits ? 'ss-hit' : 'ss-miss');
            hits += cardHits;
        });
        if (hits) sec.classList.add('ss-hit');
        counts[tab] = hits;
        total += hits;
    });
    _ssHeads(content, counts);
    nav.forEach(el => {
        const n = counts[el.dataset.tab] || 0;
        const badge = el.querySelector('.set-nav-count');
        if (badge) badge.textContent = n ? String(n) : '';
        el.classList.toggle('ss-none', !n);
    });
    const q = _ssQuery.trim();
    _ssNote(content, total ? '' : `Nothing matches “${q}”. The search covers names, descriptions and config keys — try a shorter or a different word.`);
    _ssHighlight(content, terms);
}

// Heading above each page's hits; clicking it opens the page
function _ssHeads(content, counts) {
    content.querySelectorAll(':scope > .set-page').forEach(sec => {
        const tab = sec.id.replace(/^ss-/, '');
        const page = _settingsPage(tab);
        let head = sec.previousElementSibling;
        if (!head?.classList.contains('ss-head')) {
            head = document.createElement('button');
            head.type = 'button';
            head.className = 'ss-head';
            head.dataset.tab = tab;
            head.title = 'Open this page (ends the search)';
            head.innerHTML = `<span>${page ? _sIcon(page.icon) + esc(page.title) : esc(tab)}</span>`
                           + '<span class="ss-head-open">Open page →</span>';
            head.addEventListener('click', () => _settingsSearchOpen(tab));
            sec.before(head);
        }
        head.classList.toggle('ss-hit', !!counts[tab]);
    });
}

function _ssNote(content, text) {
    let note = content.querySelector(':scope > .ss-note');
    if (!note) {
        if (!text) return;
        note = document.createElement('div');
        note.className = 'ss-note';
        content.prepend(note);
    }
    note.textContent = text;
    note.hidden = !text;
}

// Marks the query words in the hits — CSS Custom Highlight API, so the
// DOM stays untouched (browsers without it simply show no marks)
function _ssHighlight(content, terms) {
    if (typeof CSS === 'undefined' || !CSS.highlights || typeof Highlight === 'undefined') return;
    if (!content || !terms.length) { CSS.highlights.delete('settings-search'); return; }
    const ranges = [];
    const scopes = content.querySelectorAll('.ss-head.ss-hit, .set-card.ss-hit > .set-card-head h3, .ss-match');
    for (const el of scopes) {
        const walker = document.createTreeWalker(el, NodeFilter.SHOW_TEXT);
        for (let n = walker.nextNode(); n && ranges.length < 2000; n = walker.nextNode()) {
            const p = n.parentElement;
            if (!p || p.closest('option, ' + _SS_SKIP)) continue;
            const text = n.nodeValue.toLowerCase();
            for (const t of terms) {
                const anywhere = /[^\p{L}\p{N}]/u.test(t);   // same rule as _ssHas
                for (let i = text.indexOf(t); i !== -1; i = text.indexOf(t, i + t.length)) {
                    if (!anywhere && i > 0 && _SS_WORD_CHAR.test(text[i - 1])) continue;
                    const r = document.createRange();
                    r.setStart(n, i);
                    r.setEnd(n, i + t.length);
                    ranges.push(r);
                }
            }
        }
    }
    CSS.highlights.set('settings-search', new Highlight(...ranges));
}

// Pages redraw parts of themselves (late data, polling, a provider added);
// new rows come without the filter's classes, so filter again. The
// observer hands over all mutations of one redraw in a single call, and
// the filter itself only changes classes — no childList change, no loop.
// (Not via requestAnimationFrame: that never fires in a background tab,
// and the stale filter would show whole pages until the tab is seen.)
function _ssWatch() {
    if (typeof MutationObserver === 'undefined') return;
    const content = document.getElementById('settingsContent');
    if (!content) return;
    if (!_ssObserver) {
        _ssObserver = new MutationObserver(() => {
            if (_ssTerms(_ssQuery).length) _applySettingsSearch();
        });
    }
    content.querySelectorAll(':scope > .set-page').forEach(sec => {
        if (sec.dataset.watched) return;
        sec.dataset.watched = '1';
        _ssObserver.observe(sec, {childList: true, subtree: true});
    });
}

// Leave the search for a page and land on its first hit
function _settingsSearchOpen(tab) {
    const first = document.querySelector(`#ss-${tab} .ss-match`);
    _settingsSearchReset();
    _showSettingsTab(tab);
    if (!first?.isConnected) return;
    // Opened for this visit only — the remembered folded state stays
    first.closest('.set-card')?.classList.remove('folded');
    first.scrollIntoView({block: 'center'});
    first.classList.remove('ss-flash');
    void first.offsetWidth;          // restart the animation
    first.classList.add('ss-flash');
    setTimeout(() => first.classList.remove('ss-flash'), 1800);
}

function _settingsSearchKey(e) {
    if (e.key === 'Escape') {
        if (!e.target.value) return;          // empty: Esc closes the settings
        e.preventDefault();
        e.stopPropagation();                  // only clear the search
        _settingsSearchReset();
    } else if (e.key === 'Enter') {
        // Straight to the first hit's control
        e.preventDefault();
        const first = document.querySelector('#settingsOverlay .set-page.ss-hit .ss-match');
        if (!first) return;
        const control = first.querySelector('input:not([type=hidden]), select, textarea, button');
        first.scrollIntoView({block: 'center'});
        control?.focus({preventScroll: true});
    }
}

// "/" or Ctrl+F while the settings are open jumps into the search. Ctrl+F
// inside the search field falls through to the browser's own find.
document.addEventListener('keydown', e => {
    if (!document.getElementById('settingsOverlay')?.classList.contains('visible')) return;
    const input = document.getElementById('settingsSearch');
    if (!input || document.activeElement === input) return;
    const typing = e.target.closest?.('input, textarea, select, [contenteditable="true"]');
    const find = (e.ctrlKey || e.metaKey) && !e.altKey && !e.shiftKey && String(e.key).toLowerCase() === 'f';
    const slash = e.key === '/' && !typing && !e.ctrlKey && !e.metaKey && !e.altKey;
    if (!find && !slash) return;
    e.preventDefault();
    input.focus();
    input.select();
});

// Esc: an open folder browser first, then the dialog
document.addEventListener('keydown', e => {
    if (e.key !== 'Escape' || !document.getElementById('settingsOverlay')?.classList.contains('visible')) return;
    const open = document.querySelector('#settingsOverlay .set-browser:not([hidden])');
    if (open) { open.hidden = true; return; }
    closeSettings();
});

// ── Helpers ──────────────────────────────────────────────────────────────

// Historically its own name (_fmtBytesS), because the now-removed
// dashboard.js also defined a global _fmtBytes and the two would overwrite
// each other depending on load order. The canonical formatter today is
// fmtBytes() in core.js; this variant stays for the rounding used here.
function _fmtBytesS(b) {
    b = Number(b);
    if (!Number.isFinite(b) || b <= 0) return '0 B';
    if (b < 1024) return b + ' B';
    if (b < 1024 * 1024) return (b / 1024).toFixed(1) + ' KB';
    if (b < 1024 * 1024 * 1024) return (b / 1024 / 1024).toFixed(1) + ' MB';
    if (b < 1024 * 1024 * 1024 * 1024) return (b / 1024 / 1024 / 1024).toFixed(2) + ' GB';
    return (b / 1024 / 1024 / 1024 / 1024).toFixed(2) + ' TB';
}

// Safe array-to-string: handles string, array, null, undefined
function _arrToStr(val, fallback, sep) {
    sep = sep || ', ';
    if (!val) return fallback || '';
    if (Array.isArray(val)) return val.join(sep);
    if (typeof val === 'string') return val;
    return String(val);
}

function _showToast(msg, isError) {
    let el = document.getElementById('settingsToast');
    if (!el) {
        el = document.createElement('div');
        el.id = 'settingsToast';
        el.className = 's-toast';
        document.body.appendChild(el);
    }
    el.textContent = msg;
    el.classList.toggle('error', !!isError);
    el.classList.add('visible');
    clearTimeout(el._timer);
    el._timer = setTimeout(() => el.classList.remove('visible'), isError ? 6000 : 3500);
}
