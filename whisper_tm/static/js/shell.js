// ── The shell: the rail's places, which one is open, and its badges ──────
// Every place opens through Shell.go(name). Which place is lit is read from
// what is actually on screen (the windows' "visible" class and the search
// view's tab), so a window opened any other way — a link, a toast action,
// another view's button — lights the right place too.
//
// History: every move (a place, a window's tab, an open recording, a step
// inside a view) becomes a browser history entry of its own, so Alt+Left
// and the mouse's back button walk back through them. The address stays
// the same; the entries live here, the browser only keeps their number.

const Shell = (() => {
    const WINDOWS = ['svOverlay', 'mapOverlay', 'settingsOverlay', 'healthOverlay',
                     'helpOverlay', 'setupOverlay', 'statsOverlay', 'queueEditorModal', 'uploadModal', 'exportOverlay'];
    const vis = id => document.getElementById(id)?.classList.contains('visible');
    let current = 'home';

    /* Close every window, so a place opens on its own. */
    function closeAll(except) {
        const close = {
            svOverlay: 'closeSearchView', mapOverlay: 'vxClose', settingsOverlay: 'closeSettings',
            healthOverlay: 'closeHealthView', helpOverlay: 'closeHelp',
            setupOverlay: 'closeSetupGuide', statsOverlay: 'closeStatsView', queueEditorModal: 'closeQueueEditor', uploadModal: 'closeUpload',
            exportOverlay: 'closeExport',
        };
        for (const id of WINDOWS) {
            if (id === except || !vis(id)) continue;
            const fn = window[close[id]];
            try { if (typeof fn === 'function') fn(); } catch (e) { console.error(e); }
        }
    }

    function searchTab(tab) {
        const ov = document.getElementById('svOverlay');
        if (!ov) return;
        if (!ov.classList.contains('visible')) {
            if (tab === 'chat' && typeof openAsk === 'function') { openAsk(); return; }
            if (typeof openSearchView === 'function') openSearchView();
        }
        if (typeof svState !== 'undefined' && svState.tab !== tab && typeof svTab === 'function') svTab(tab);
    }

    const PLACES = {
        add: () => { closeAll('uploadModal'); if (typeof openUpload === 'function') openUpload(); },
        home: () => { closeAll(); if (typeof Home !== 'undefined') Home.show(); },
        // The open recording, no window over it (lights no rail item)
        file: () => { closeAll(); if (typeof Home !== 'undefined') Home.hide(); },
        library: () => { closeAll('statsOverlay'); if (typeof openStatsView === 'function') openStatsView(); },
        queue: () => { closeAll('queueEditorModal'); if (typeof openQueueEditorFromBar === 'function') openQueueEditorFromBar(); },
        find: () => { closeAll('svOverlay'); searchTab(['chat', 'analysis'].includes(typeof svState !== 'undefined' ? svState.tab : '') ? 'find' : (svState?.tab || 'find')); },
        ask: () => { closeAll('svOverlay'); searchTab('chat'); },
        analyses: () => { closeAll('svOverlay'); searchTab('analysis'); },
        voices: () => { closeAll('mapOverlay'); if (typeof vxOpen === 'function') vxOpen(); },
        machine: () => { closeAll('healthOverlay'); if (typeof openHealthView === 'function') openHealthView(); },
        settings: () => { closeAll('settingsOverlay'); if (typeof openSettings === 'function') openSettings(); },
        help: () => { closeAll('helpOverlay'); if (typeof openHelp === 'function') openHelp(); },
        // The export page (export.js opens it with its scope; lights no rail item)
        export: () => { closeAll('exportOverlay'); document.getElementById('exportOverlay')?.classList.add('visible'); },
    };

    function go(name) {
        const fn = PLACES[name];
        if (!fn) return;
        try { fn(); } catch (e) { console.error(e); }
        detect();
    }

    function quiet() {
        document.querySelectorAll('#videoPlayer, #transcriptArea audio, #transcriptArea video, #clipEditor audio, #clipEditor video')
            .forEach(m => { try { if (!m.paused) m.pause(); } catch (e) { /* not a player */ } });
    }

    /* Which place is on screen: the topmost visible window decides. */
    function detect() {
        if (typeof Home !== 'undefined') Home.sync();
        let place = typeof Home !== 'undefined' && !Home.visible() ? 'file' : 'home';
        if (vis('exportOverlay')) place = 'export';
        else if (vis('settingsOverlay')) place = 'settings';
        else if (vis('helpOverlay') || vis('setupOverlay')) place = 'help';
        else if (vis('healthOverlay')) place = 'machine';
        else if (vis('mapOverlay')) place = 'voices';
        else if (vis('svOverlay')) {
            const tab = typeof svState !== 'undefined' ? svState.tab : 'find';
            place = tab === 'chat' ? 'ask' : tab === 'analysis' ? 'analyses' : 'find';
        } else if (vis('statsOverlay')) place = 'library';
        else if (vis('queueEditorModal')) place = 'queue';
        else if (vis('uploadModal')) place = 'add';
        // Leaving a recording stops its sound: nothing plays on behind
        // another place where no player is in sight
        if (current === 'file' && place !== 'file') quiet();
        current = place;
        record();
        document.querySelectorAll('#rail .rail-item[data-place]').forEach(b => {
            const on = b.dataset.place === place;
            b.classList.toggle('on', on);
            b.setAttribute('aria-current', on ? 'page' : 'false');
        });
        // A pinned tab that is open lights its pin instead (pins.js)
        if (typeof Pins !== 'undefined') Pins.sync();
    }


    /* ── History ─────────────────────────────────────────────────────── */
    const stack = [];        // entries: {place, tab, file, back}
    let at = -1;             // the entry on screen
    let restoring = false;   // true while an entry is being put back

    function tabOf(place) {
        if (['find', 'ask', 'analyses'].includes(place)) return typeof svState !== 'undefined' ? svState.tab : '';
        if (place === 'voices') return typeof VX !== 'undefined' ? VX.tab : '';
        if (place === 'export') return typeof Export !== 'undefined' ? Export.tab : '';
        return '';
    }

    function here() {
        return {place: current, tab: tabOf(current),
                file: typeof activeFile !== 'undefined' ? activeFile : null};
    }

    const same = (a, b) => a && b && a.place === b.place && a.tab === b.tab && a.file === b.file;

    function push(entry) {
        stack.length = at + 1;           // a new move drops the forward entries
        stack.push(entry);
        at = stack.length - 1;
        try { history.pushState({wtm: at}, '', location.href); } catch (e) { /* stays in memory */ }
    }

    /* Note where we are now (called after every move). */
    function record() {
        if (restoring) return;
        const e = here();
        if (at < 0) {
            stack.push(e); at = 0;
            try { history.replaceState({wtm: 0}, '', location.href); } catch (err) { /* no history */ }
            return;
        }
        if (same(stack[at], e)) return;
        push(e);
    }

    /* A step inside a view (a person's page, a drill-down): Back undoes it
       by calling back(). */
    function step(back) {
        if (restoring || typeof back !== 'function') return;
        record();
        push({...here(), back});
    }

    /* A view's own close control for a step: through the history when
       that step is the newest entry (so Back and the control agree),
       otherwise just undo it. */
    function leave(back) {
        if (stack[at] && stack[at].back === back && at > 0) { history.back(); return; }
        try { back(); } catch (e) { console.error(e); }
    }

    /* Esc inside a window: a step of this place (a chosen voice, a
       person's page) is undone first; false = nothing to undo. */
    function undoStep() {
        const e = stack[at];
        if (!e || !e.back || e.place !== current || at < 1) return false;
        history.back();
        return true;
    }

    async function restore(target) {
        const leaving = stack[at], e = stack[target];
        if (!e) return;
        restoring = true;
        try {
            // Leaving a step backwards: the view undoes it itself
            if (leaving && leaving.back && target < at) {
                try { leaving.back(); } catch (err) { console.error(err); }
            }
            if (e.place !== current || e.tab !== tabOf(current)) {
                if (e.place !== current) (PLACES[e.place] || PLACES.home)();
                if (e.tab && e.tab !== tabOf(e.place)) {
                    if (['find', 'ask', 'analyses'].includes(e.place) && typeof svTab === 'function') svTab(e.tab);
                    else if (e.place === 'voices' && typeof vxTab === 'function') vxTab(e.tab);
                    else if (e.place === 'export' && typeof Export !== 'undefined') Export.setTab(e.tab);
                }
            }
            if (e.file && e.file !== activeFile) {
                // Only a recording of the file list as it is now — going
                // back never switches the folder (that is shared by everyone)
                const f = (typeof files !== 'undefined' ? files : []).find(x => x.name === e.file);
                if (f && typeof selectFile === 'function') await selectFile(f);
            }
            at = target;
        } finally {
            // The entry now holds what is really on screen (a recording
            // that could not be put back stays open) — no new entry
            detect();
            const now = here();
            if (stack[at]) Object.assign(stack[at], {place: now.place, tab: now.tab, file: now.file});
            restoring = false;
        }
    }

    function onPop(ev) {
        const n = ev.state && typeof ev.state.wtm === 'number' ? ev.state.wtm : null;
        if (n === null || n === at || !stack[n]) return;
        restore(n);
    }

    function back() { try { history.back(); } catch (e) { /* nowhere to go */ } }

    /* Alt+Left / Alt+Right: the page steps itself (and stops the browser's
       own shortcut, so one press is one step). */
    function onKey(ev) {
        if (!ev.altKey || ev.ctrlKey || ev.metaKey || ev.shiftKey) return;
        if (ev.key !== 'ArrowLeft' && ev.key !== 'ArrowRight') return;
        ev.preventDefault();
        try { ev.key === 'ArrowLeft' ? history.back() : history.forward(); } catch (e) { /* nowhere */ }
    }

    /* ── UI preferences, kept on the server (/api/ui/prefs) ─────────────
       Loaded once; a change is sent a moment later, several at once.
       The browser's own copy of the fold only avoids a jump on load. */
    let prefs = {};
    let prefsReady = null;
    const prefsDirty = {};
    let prefsTimer = null;
    const prefsListeners = [];

    function loadPrefs() {
        prefsReady = fetch('/api/ui/prefs', {cache: 'no-cache'})
            .then(r => r.ok ? r.json() : {})
            .catch(() => ({}))
            .then(d => { prefs = d && typeof d === 'object' ? d : {}; prefsListeners.forEach(f => f(prefs)); return prefs; });
        return prefsReady;
    }

    function pref(key) { return prefs[key]; }

    function setPref(key, value) {
        prefs[key] = value;
        prefsDirty[key] = value;
        clearTimeout(prefsTimer);
        prefsTimer = setTimeout(flushPrefs, 400);
    }

    async function flushPrefs() {
        const body = {...prefsDirty};
        for (const k of Object.keys(prefsDirty)) delete prefsDirty[k];
        if (!Object.keys(body).length) return;
        try {
            const r = await fetch('/api/ui/prefs', {method: 'PUT', headers: {'Content-Type': 'application/json'},
                                                    body: JSON.stringify(body)});
            if (!r.ok) {
                const e = await r.json().catch(() => ({}));
                if (typeof Toast !== 'undefined') Toast.error('Not saved: ' + (e.error || r.status));
            }
        } catch (e) { /* offline: kept for this visit */ }
    }

    function onPrefs(fn) { prefsListeners.push(fn); if (prefsReady) prefsReady.then(fn); }

    /* A recording was opened: note it under "Recently opened". */
    function noteRecent() {
        const dir = (document.getElementById('dirInput')?.value || '').trim().replace(/\/+$/, '');
        const name = typeof activeFile !== 'undefined' ? activeFile : null;
        if (!dir || !name) return;
        fetch('/api/ui/recent', {method: 'POST', headers: {'Content-Type': 'application/json'},
                                 body: JSON.stringify({path: dir + '/' + name})})
            .then(r => r.ok ? r.json() : null)
            .then(d => { if (d) { prefs.recent = d.recent; prefsListeners.forEach(f => f(prefs)); } })
            .catch(() => { /* not noted */ });
    }

    /* "Icons only" */
    function fold(force, keep) {
        const on = force === undefined ? !document.body.classList.contains('rail-folded') : !!force;
        document.body.classList.toggle('rail-folded', on);
        const b = document.getElementById('railFold');
        if (b) { b.title = on ? 'Show the words' : 'Show icons only'; b.firstChild.textContent = on ? '» ' : '« '; }
        try { localStorage.setItem('wtmRailFolded', on ? '1' : '0'); } catch (e) { /* first paint only */ }
        if (!keep) setPref('rail_folded', on);
    }

    /* Badges, read from the counters the app already keeps up to date
       (no timers of their own). */
    let healthState = '';   // '' ok · 'warn' · 'err', from Shell.health()
    let healthWhy = '';
    function badges() {
        const q = document.getElementById('tabQueue'), llm = document.getElementById('llmQueueBadge');
        const n = (parseInt(q?.textContent, 10) || 0) + (parseInt(llm?.textContent, 10) || 0);
        setBadge('railBadgeQueue', n);
        setBadge('railBadgeAnalyses', parseInt(llm?.dataset.analyses, 10) || 0);
        const dot = document.getElementById('railMachineDot');
        if (dot) {
            const down = [['connLlmDot', 'The language model'], ['connEmbDot', 'The embedding server']]
                .filter(([id]) => /\berr\b/.test(document.getElementById(id)?.className || ''))
                .map(([, name]) => name);
            const connErr = down.length > 0;
            const err = healthState === 'err';
            const warn = !err && (healthState === 'warn' || connErr);
            dot.classList.toggle('err', err);
            dot.classList.toggle('warn', warn);
            const why = [healthWhy, connErr ? (down.length > 1 ? 'The language model and the embedding server do not answer'
                                                : down[0] + ' does not answer') : '']
                .filter(Boolean).join(' · ') || 'Everything is fine';
            dot.title = why;
            dot.closest('.rail-item')?.setAttribute('aria-description', why);
        }
    }

    function setBadge(id, n) {
        const b = document.getElementById(id);
        if (!b) return;
        b.hidden = !n;
        b.textContent = n > 999 ? Math.round(n / 1000) + 'k' : String(n);
    }

    /* The overall state from /api/health/detail (the app's 30 s health
       poll hands it over; null = the server did not answer). */
    function health(d) {
        if (!d) { healthState = 'err'; healthWhy = 'The server does not answer'; badges(); return; }
        const why = [];
        let st = '';
        if (d.db && d.db.ok === false) { st = 'err'; why.push('Database problem'); }
        if (d.nas && d.nas.ok === false) { st = 'err'; why.push('Library folder not reachable'); }
        if ((d.warnings || []).length) { st = 'err'; why.push(...d.warnings.slice(0, 3).map(String)); }
        if (!st && d.gpu && d.gpu.available === false) { st = 'warn'; why.push('No GPU detected'); }
        healthState = st;
        healthWhy = why.join(' · ');
        badges();
    }

    function init() {
        try { if (localStorage.getItem('wtmRailFolded') === '1') fold(true, true); } catch (e) { /* default: words */ }
        loadPrefs().then(p => { if (typeof p.rail_folded === 'boolean') fold(p.rail_folded, true); });
        const mo = new MutationObserver(detect);
        for (const id of WINDOWS) {
            const el = document.getElementById(id);
            if (el) mo.observe(el, {attributes: true, attributeFilter: ['class']});
        }
        // The search view's tabs change the place too (Find / Ask / Analyses)
        if (typeof svShowTab === 'function') {
            const orig = svShowTab;
            window.svShowTab = function (...a) { const r = orig.apply(this, a); detect(); return r; };
        }
        if (typeof vxTab === 'function') {
            const orig = vxTab;
            window.vxTab = function (...a) { const r = orig.apply(this, a); detect(); return r; };
        }
        if (typeof selectFile === 'function') {
            const orig = selectFile;
            window.selectFile = async function (...a) {
                const before = typeof activeFile !== 'undefined' ? activeFile : null;
                const r = await orig.apply(this, a);
                if (typeof Home !== 'undefined') Home.hide();
                detect();
                if (activeFile && activeFile !== before) noteRecent();
                return r;
            };
        }
        addEventListener('popstate', onPop);
        document.addEventListener('keydown', onKey, true);
        const bo = new MutationObserver(badges);
        for (const id of ['tabQueue', 'llmQueueBadge', 'connLlmDot', 'connEmbDot']) {
            const el = document.getElementById(id);
            if (el) bo.observe(el, {attributes: true, childList: true, characterData: true, subtree: true});
        }
        detect(); badges();
    }

    document.addEventListener('DOMContentLoaded', init);
    return {go, fold, detect, closeAll, health, step, leave, undoStep, back,
            pref, setPref, onPrefs, flushPrefs, get prefs() { return prefs; },
            get current() { return current; }, get history() { return {at, size: stack.length}; }};
})();
