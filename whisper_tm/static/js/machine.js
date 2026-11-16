// ── Machine: health, load, models and connections ───────────────────────
// One window (the former health check) with three tabs: Health (health.js
// draws it), Load history (the former performance view, perfview.js — its
// body lives in this window now) and Models & connections: the speech
// model, the language model, the embedding server and the images & sounds
// service — whether each answers, in plain words, with the way to fix it.

const Machine = (() => {
    const TABS = ['health', 'load', 'models'];
    let current = 'health';
    let modelsSeq = 0;

    function remembered() {
        try {
            const p = typeof Shell !== 'undefined' ? Shell.pref('last_tab') : null;
            if (p && TABS.includes(p.machine)) return p.machine;
        } catch (e) { /* default */ }
        return 'health';
    }

    /* The Machine place on a tab. */
    function open(t) {
        if (typeof Shell !== 'undefined') {
            Shell.go('machine');
            tab(t || current);
        } else if (typeof openHealthView === 'function') openHealthView(t);
    }

    function tab(t) {
        if (!TABS.includes(t)) t = 'health';
        const was = current;
        current = t;
        document.querySelectorAll('#mcTabs .vx-tab').forEach(b => {
            const on = b.dataset.tab === t;
            b.classList.toggle('active', on);
            b.setAttribute('aria-selected', on ? 'true' : 'false');
        });
        document.querySelectorAll('#healthOverlay .mc-page').forEach(p => { p.hidden = p.dataset.page !== t; });
        // The load history follows the load only while its tab is shown
        if (t === 'load' && typeof pvStart === 'function' && !(typeof pvShown === 'function' && pvShown())) pvStart();
        if (t !== 'load' && was === 'load' && typeof pvStop === 'function') pvStop();
        if (t === 'models') loadModels();
        if (typeof Shell !== 'undefined' && was !== t) {
            const lt = {...(Shell.pref('last_tab') || {}), machine: t};
            Shell.setPref('last_tab', lt);
            Shell.detect();
        }
    }

    function refresh() {
        if (current === 'health' && typeof _hcRefresh === 'function') _hcRefresh();
        else if (current === 'load' && typeof pvLoad === 'function') pvLoad();
        else if (current === 'models') loadModels();
    }

    /* ── Models & connections ─────────────────────────────────────────── */

    const card = (state, title, value, lines, fix) =>
        `<div class="hc-card ${state}"><div class="hc-card-head">${title}</div>`
        + `<div class="hc-card-big">${value}</div>`
        + `<div class="hc-card-lines">${lines.filter(Boolean).map(l => `<div>${l}</div>`).join('')}</div>`
        + (fix ? `<div class="mc-fix">${fix}</div>` : '') + `</div>`;

    const settingsLink = (section, text) =>
        `<button class="mc-link" onclick="openSettings('${escJs(section)}')">${esc(text)} →</button>`;

    function connCard(title, c, section) {
        if (!c) return card('neutral', title, 'unknown', ['No answer from the server']);
        if (!c.configured) {
            return card('warn', title, 'not set up', [esc(c.detail || 'No server or model is configured.')],
                        settingsLink(section, 'Set it up in Settings'));
        }
        const where = [c.model && `Model: ${esc(c.model)}`, c.endpoint && `At: <span class="mc-mono">${esc(c.endpoint)}</span>`];
        if (c.reachable === false) {
            return card('err', title, 'does not answer',
                        [...where, esc(c.detail || 'The server did not answer. Is it running, and is the address right?')],
                        settingsLink(section, 'Check the address in Settings'));
        }
        return card('ok', title, c.reachable ? 'answers' : 'set up', [...where, c.reachable == null ? 'Asked the next time it is needed.' : '']);
    }

    async function loadModels() {
        const box = document.getElementById('mcModels');
        if (!box) return;
        const seq = ++modelsSeq;
        if (!box.innerHTML) box.innerHTML = '<div class="hc-grid"><div class="hc-card neutral"><div class="hc-card-big">Loading …</div></div></div>';
        const get = url => fetch(url).then(r => r.ok ? r.json() : null).catch(() => null);
        const [conn, settings, health, media] = await Promise.all([
            get('/api/connection/status'), get('/api/settings'), get('/api/health/detail'), get('/api/media-search/info')]);
        if (seq !== modelsSeq) return;
        const cfg = settings && settings.config || {};
        const tx = cfg.transcription || {};
        const gpu = health && health.gpu || {};
        const resident = (gpu.resident_models || []);
        const speech = card(gpu.available === false ? 'warn' : 'ok', '🎙 Speech model',
            esc(tx.default_model || 'not set'),
            [tx.default_language ? `Language: ${esc(tx.default_language)}` : 'Language: detected per file',
             resident.length ? `Loaded now: ${resident.map(esc).join(', ')}` : 'Not loaded now — the next transcription loads it',
             gpu.available === false ? 'No GPU found: transcription runs on the CPU, much slower.' : ''],
            settingsLink('transcription', 'Transcription settings'));
        let mediaCard;
        if (!media) mediaCard = card('neutral', '🖼 Images & sounds', 'unknown', ['No answer from the server']);
        else if (!media.enabled) mediaCard = card('neutral', '🖼 Images & sounds', 'off', ['Searching what is seen or heard is switched off.'], settingsLink('media', 'Switch it on in Settings'));
        else if (media.environment_ok === false) mediaCard = card('warn', '🖼 Images & sounds', 'not installed',
            [esc((media.environment_problems || []).join(' · ') || 'Its environment is missing.'), media.model ? `Model: ${esc(media.model)}` : ''],
            settingsLink('media', 'How to install it'));
        else mediaCard = card('ok', '🖼 Images & sounds', media.indexer && media.indexer.running ? 'indexing' : 'ready',
            [media.model ? `Model: ${esc(media.model)}` : '', `${((media.counts || {}).moments || 0).toLocaleString()} moments indexed`]);
        box.innerHTML = `<div class="hc-grid">${speech}`
            + connCard('🧠 Language model', conn && conn.llm, 'llm')
            + connCard('🔎 Embeddings (search by meaning)', conn && conn.embedding, 'embedding')
            + mediaCard + `</div>`;
    }

    function init() {
        current = 'health';
        if (typeof Shell !== 'undefined') Shell.onPrefs(() => { current = remembered(); });
    }

    document.addEventListener('DOMContentLoaded', init);
    return {open, tab, refresh, loadModels, TABS, get last() { return current; }, get current() { return current; }};
})();
