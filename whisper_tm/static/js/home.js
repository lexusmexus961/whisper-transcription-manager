// ── Home: what is going on, and every place ─────────────────────────────
// Shown over the main area when no recording is open, or when Home is
// chosen in the rail (the open recording stays loaded; Alt+Left goes back
// to it). Live cards first (running now, just finished, failed, recently
// opened), then every place and window tab in groups that fold; which
// groups are open is a UI preference on the server.
//
// Data: the queue's state as the app keeps it (batch.js, over its stream),
// the LLM queue the badge already polls, /api/files for what finished
// last, and the UI preferences for pins and recently opened. Nothing here
// polls on its own while Home is hidden.

const Home = (() => {
    let shown = false;
    let wanted = false;          // chosen in the rail although a recording is open
    let finished = null;         // last answer of "just finished"
    let finishedAt = 0;
    let summaries = {};          // place → short live text
    let summariesAt = 0;
    let drawTimer = null;
    let lastHtml = '';

    /* Window tabs a pin or a group card can open. */
    const TABS = {
        'find:find': ['Find', 'Find', 'i-sparkle', () => openFindTab('find')],
        'find:text': ['Text', 'Find', 'i-search', () => openFindTab('text')],
        'find:media': ['Images & sounds', 'Find', 'i-music', () => openFindTab('media')],
        'find:audio': ['Same audio', 'Find', 'i-captions', () => openFindTab('audio')],
        'find:conversations': ['Conversations', 'Find', 'i-conversation', () => openFindTab('conversations')],
        'find:files': ['Files', 'Find', 'i-folder', () => openFindTab('files')],
        'find:deep': ['Deep search', 'Find', 'i-search', () => openFindTab('deep')],
        'voices:people': ['People', 'Voices', 'i-user', () => openVoicesTab('people')],
        'voices:assign': ['Name voices', 'Voices', 'i-user-check', () => openVoicesTab('assign')],
        'voices:map': ['Voice map', 'Voices', 'i-grid', () => openVoicesTab('map')],
        'voices:history': ['History', 'Voices', 'i-history', () => openVoicesTab('history')],
        'voices:conversations': ['Conversations', 'Voices', 'i-conversation', () => openVoicesTab('conversations')],
        'voices:extract': ['Voice dataset', 'Voices', 'i-scissors', () => openVoicesTab('extract')],
        'voices:help': ['How it works', 'Voices', 'i-help', () => openVoicesTab('help')],
        library: ['Library', '', 'i-folder', () => Shell.go('library')],
        queue: ['Queue', '', 'i-queue', () => Shell.go('queue')],
        find: ['Find', '', 'i-sparkle', () => Shell.go('find')],
        ask: ['Ask', '', 'i-chat', () => Shell.go('ask')],
        analyses: ['Analyses', '', 'i-layers', () => Shell.go('analyses')],
        voices: ['Voices', '', 'i-voices', () => Shell.go('voices')],
        machine: ['Machine', '', 'i-health', () => Shell.go('machine')],
        'machine:load': ['Load history', 'Machine', 'i-cpu', () => { if (typeof openPerfView === 'function') openPerfView(); }],
        settings: ['Settings', '', 'i-settings', () => Shell.go('settings')],
        help: ['Help & setup', '', 'i-help', () => Shell.go('help')],
        'help:setup': ['Setup guide', 'Help & setup', 'i-guide', () => { if (typeof openSetupGuide === 'function') openSetupGuide(); }],
    };
    const DEFAULT_PINS = ['voices:map', 'voices:extract', 'find:audio', 'find:conversations'];

    const GROUPS = [
        ['work', 'Work', ['library', 'queue']],
        ['find', 'Find', ['find', 'find:text', 'find:media', 'find:audio', 'find:conversations',
                          'find:files', 'find:deep', 'ask', 'analyses']],
        ['voices', 'Voices', ['voices:people', 'voices:assign', 'voices:history', 'voices:map',
                              'voices:conversations', 'voices:extract', 'voices:help']],
        ['system', 'System', ['machine', 'machine:load', 'settings', 'help', 'help:setup']],
    ];

    function openFindTab(tab) {
        Shell.go('find');
        if (typeof svState !== 'undefined' && svState.tab !== tab && typeof svTab === 'function') svTab(tab);
    }

    function openVoicesTab(tab) {
        if (typeof vxTab === 'function') vxTab(tab);
        Shell.go('voices');
    }

    function open(id) {
        const t = TABS[id];
        if (t) t[3]();
    }

    const el = () => document.getElementById('homeView');
    const ago = sec => {
        const s = Math.max(0, Date.now() / 1000 - sec);
        if (s < 60) return 'just now';
        if (s < 3600) return Math.round(s / 60) + ' min ago';
        if (s < 86400) return Math.round(s / 3600) + ' h ago';
        return Math.round(s / 86400) + ' d ago';
    };
    const base = p => String(p || '').split('/').pop();

    /* ── Visibility ──────────────────────────────────────────────────── */

    function shouldShow() {
        return wanted || !(typeof activeFile !== 'undefined' && activeFile);
    }

    function sync() {
        const v = el();
        if (!v) return;
        const on = shouldShow();
        v.hidden = !on;
        if (on && !shown) refresh();
        shown = on;
    }

    /* Home was chosen in the rail. */
    function show() { wanted = true; sync(); }
    /* A recording was opened, or the place is the recording again. */
    function hide() { wanted = false; sync(); }
    function visible() { return !!el() && !el().hidden; }

    /* ── Data ────────────────────────────────────────────────────────── */

    async function loadFinished(force) {
        if (!force && finished && Date.now() - finishedAt < 15000) return;
        finishedAt = Date.now();
        try {
            const r = await fetch('/api/files?sort=tx_date&dir=desc&limit=4&tab=done');
            const d = await r.json();
            if (r.ok) finished = {dir: d.directory || '', files: d.files || [], total: d.total || 0};
        } catch (e) { /* keeps the last answer */ }
        draw();
    }

    async function loadSummaries() {
        if (Date.now() - summariesAt < 60000) return;
        summariesAt = Date.now();
        const get = url => fetch(url).then(r => r.ok ? r.json() : null).catch(() => null);
        const [people, analyses, chats] = await Promise.all([
            get('/api/voices/people'), get('/api/analyses'), get('/api/chat/conversations')]);
        if (people) summaries['voices:people'] = plural((people.people || []).length, 'person', 'people');
        if (analyses) summaries.analyses = plural((analyses.analyses || []).length, 'analysis', 'analyses');
        if (chats) summaries.ask = plural((chats.conversations || []).length, 'chat', 'chats');
        draw();
    }

    function refresh() {
        loadFinished(true);
        loadSummaries();
        draw();
    }

    /* Something changed (queue state, preferences): redraw soon. */
    function changed(finishedToo) {
        if (!visible()) return;
        if (finishedToo) loadFinished(true);
        clearTimeout(drawTimer);
        drawTimer = setTimeout(draw, 150);
    }

    /* ── Drawing ─────────────────────────────────────────────────────── */

    function card(title, body, foot) {
        return `<section class="home-card"><h3>${title}</h3><div class="home-card-body">${body}</div>`
            + (foot ? `<div class="home-card-foot">${foot}</div>` : '') + `</section>`;
    }

    function cardRunning() {
        const items = typeof queueRunningItems !== 'undefined' ? queueRunningItems : [];
        const waiting = typeof queuePending !== 'undefined' ? queuePending : 0;
        const paused = typeof queuePaused !== 'undefined' && queuePaused;
        const llm = document.getElementById('llmQueueBadge');
        const llmOpen = parseInt(llm?.textContent, 10) || 0;
        let body = items.length ? items.slice(0, 3).map(it => {
            const pct = Math.round(Number(it.progress) || 0);
            return `<button type="button" class="home-row" onclick="Home.openFile('${escJs(it.filename || '')}', '${escJs(it.dir || '')}')" title="${esc(it.filename || '')}">`
                + `<span class="home-row-name">${esc(base(it.filename))}</span>`
                + `<span class="home-bar"><span style="width:${pct}%"></span></span><span class="home-row-n">${pct} %</span></button>`;
        }).join('') : `<p class="home-quiet">${paused ? 'The queue is paused.' : 'Nothing is being transcribed.'}</p>`;
        const bits = [];
        if (waiting) bits.push(plural(waiting, 'file', 'files') + ' waiting');
        if (paused && items.length) bits.push('paused');
        if (llmOpen) bits.push(plural(llmOpen, 'AI job', 'AI jobs'));
        if (bits.length) body += `<p class="home-note">${esc(bits.join(' · '))}</p>`;
        return card('Running now', body, `<button type="button" class="home-link" onclick="Shell.go('queue')">Open the queue</button>`);
    }

    function cardFinished() {
        if (!finished) return card('Just finished', '<p class="home-quiet">Loading …</p>');
        if (!finished.files.length) return card('Just finished', '<p class="home-quiet">Nothing transcribed in this folder yet.</p>');
        const body = finished.files.map(f =>
            `<button type="button" class="home-row" onclick="Home.openFile('${escJs(f.name)}', '')" title="${esc(f.name)}">`
            + `<span class="home-row-name">${esc(base(f.name))}</span>`
            + `<span class="home-row-n">${f.tx_date ? esc(ago(f.tx_date)) : ''}</span></button>`).join('');
        return card('Just finished', body);
    }

    function cardFailed() {
        const failed = (typeof queueFailedItems !== 'undefined' ? queueFailedItems : [])
            .filter(i => i.status === 'error' || i.status === 'failed');
        if (!failed.length) return card('Failed', '<p class="home-quiet">Nothing failed.</p>');
        const body = failed.slice(0, 4).map(i =>
            `<button type="button" class="home-row home-row-two" onclick="Home.openFile('${escJs(i.filename || '')}', '${escJs(i.dir || '')}')" title="${esc(i.filename || '')}">`
            + `<span class="home-row-name">${esc(base(i.filename))}</span>`
            + `<span class="home-row-why">${esc(i.error || 'failed')}</span></button>`).join('')
            + (failed.length > 4 ? `<p class="home-note">and ${failed.length - 4} more</p>` : '');
        const retry = typeof queueRetryable !== 'undefined' && queueRetryable
            ? `<button type="button" class="home-link" onclick="retryFailedQueue()">Try again</button>` : '';
        return card(`Failed <span class="home-count">${failed.length}</span>`, body, retry);
    }

    function cardRecent() {
        const recent = (Shell.pref('recent') || []).slice(0, 4);
        if (!recent.length) return card('Recently opened', '<p class="home-quiet">Recordings you open show up here.</p>');
        const body = recent.map(r =>
            `<button type="button" class="home-row" onclick="Home.openPath('${escJs(r.path)}')" title="${esc(r.path)}">`
            + `<span class="home-row-name">${esc(base(r.path))}</span>`
            + `<span class="home-row-n">${r.at ? esc(ago(r.at)) : ''}</span></button>`).join('');
        return card('Recently opened', body);
    }

    function tile(id, small) {
        const t = TABS[id];
        if (!t) return '';
        const sum = summary(id);
        // Window tabs can be pinned: a pin on hover, or right-click
        const canPin = typeof Pins !== 'undefined' && Pins.pinnable(id);
        const pinned = canPin && Pins.has(id);
        const pin = canPin && !small
            ? `<span class="home-pin${pinned ? ' on' : ''}" role="button" title="${pinned ? 'Unpin from the rail' : 'Pin to the rail'}"`
              + ` onclick="event.stopPropagation();Pins.toggle('${escJs(id)}')"><svg class="ui-ico"><use href="#i-pin"/></svg></span>` : '';
        return `<button type="button" class="home-tile${small ? ' home-tile-pin' : ''}" data-open="${esc(id)}" onclick="Home.open('${escJs(id)}')"`
            + (canPin ? ` oncontextmenu="event.preventDefault();Pins.toggle('${escJs(id)}')"` : '') + `>`
            + `<svg class="ui-ico"><use href="#${t[2]}"/></svg><span class="home-tile-name">${esc(t[0])}</span>`
            + (t[1] ? `<span class="home-tile-where">${esc(t[1])}</span>` : '')
            + (sum ? `<span class="home-tile-sum">${esc(sum)}</span>` : '') + pin + `</button>`;
    }

    function summary(id) {
        if (summaries[id]) return summaries[id];
        if (id === 'library' && typeof srvStats !== 'undefined' && srvStats)
            return `${(srvStats.done || 0).toLocaleString()} of ${plural(srvStats.total || 0, 'file')} transcribed`;
        if (id === 'queue') {
            const w = typeof queuePending !== 'undefined' ? queuePending : 0;
            const r = typeof queueRunningCount !== 'undefined' ? queueRunningCount : 0;
            const p = typeof queuePaused !== 'undefined' && queuePaused;
            return [r ? r + ' running' : '', w ? w + ' waiting' : '', p ? 'paused' : ''].filter(Boolean).join(' · ') || 'empty';
        }
        if (id === 'machine') {
            const dot = document.getElementById('railMachineDot');
            return dot?.title || '';
        }
        return '';
    }

    function draw() {
        const v = el();
        if (!v || v.hidden) return;
        const pins = typeof Pins !== 'undefined' ? Pins.list() : DEFAULT_PINS;
        const open = Shell.pref('home_groups') || {};
        let html = '<div class="home-inner">';
        if (pins.length) html += `<div class="home-pins" aria-label="Pinned">${pins.map(p => tile(p, true)).join('')}</div>`;
        html += `<div class="home-cards">${cardRunning()}${cardFinished()}${cardFailed()}${cardRecent()}</div>`;
        for (const [key, title, ids] of GROUPS) {
            const isOpen = open[key] !== false;
            html += `<section class="home-group${isOpen ? ' open' : ''}">`
                + `<button type="button" class="home-group-head" aria-expanded="${isOpen}" onclick="Home.toggle('${key}')">`
                + `<span class="home-caret" aria-hidden="true">${isOpen ? '▾' : '▸'}</span>${esc(title)}</button>`
                + (isOpen ? `<div class="home-tiles">${ids.map(id => tile(id)).join('')}</div>` : '')
                + `</section>`;
        }
        html += '</div>';
        // Live updates arrive often (every queue message): the page is only
        // rebuilt when something on it changed — a rebuild between press and
        // release would swallow the click.
        if (html === lastHtml) return;
        lastHtml = html;
        const top = v.scrollTop;
        v.innerHTML = html;
        v.scrollTop = top;
    }

    function toggle(key) {
        const open = {...(Shell.pref('home_groups') || {})};
        open[key] = open[key] === false;
        Shell.setPref('home_groups', open);
        draw();
    }

    /* A recording from a card: in the open folder it opens at once, else
       through the search's path (which switches the folder). */
    function openFile(name, dir) {
        if (!name) return;
        const f = typeof files !== 'undefined' ? files.find(x => x.name === name) : null;
        if (f && typeof selectFile === 'function') { selectFile(f); return; }
        if (typeof goToSearchResult === 'function') goToSearchResult(name, 0, dir || '');
    }

    function openPath(abs) {
        const here = (document.getElementById('dirInput')?.value || '').trim().replace(/\/+$/, '');
        if (here && abs.startsWith(here + '/')) { openFile(abs.slice(here.length + 1), here); return; }
        const cut = abs.lastIndexOf('/');
        if (cut > 0 && typeof goToSearchResult === 'function') goToSearchResult(abs.slice(cut + 1), 0, abs.slice(0, cut));
    }

    function init() {
        const v = el();
        if (!v) return;
        if (typeof Shell !== 'undefined') Shell.onPrefs(() => changed(false));
        // The queue's state changes (running, waiting, failed): the cards follow
        if (typeof applyQueueState === 'function') {
            const orig = applyQueueState;
            let lastRunning = '';
            window.applyQueueState = function (...a) {
                const r = orig.apply(this, a);
                const sig = (typeof queueRunningItems !== 'undefined' ? queueRunningItems : []).map(i => i.filename).join('|');
                changed(sig !== lastRunning);   // a file finished or started: "just finished" too
                lastRunning = sig;
                return r;
            };
        }
        sync();
    }

    document.addEventListener('DOMContentLoaded', init);
    return {show, hide, sync, visible, refresh, redraw: draw, toggle, open, openFile, openPath, TABS, GROUPS,
            get wanted() { return wanted; }};
})();
