/* ══════════════════════════════════════════════════════════════════════════
 *  The connection to the Whisper Transcription Manager server
 *
 *  The page (content.js) and the popup ask here; this is the one place that
 *  speaks to the server — POST /api/ingest with the links and a mode, GET
 *  /api/ingest/<id> for where each video stands (core/ingest.py). The
 *  batches sent are remembered (the popup lists them); every half minute
 *  the running ones are asked again — the number still working stands on
 *  the toolbar icon, and a finished batch says so in a notification.
 *
 *  Right-click on a YouTube link (or on a watch page): transcribe or
 *  summarize that video directly.
 *
 *  The side panel (sidepanel.js) asks here too: the forms' choices and the
 *  defaults (/api/ingest/options, /defaults), a video's transcript, and the
 *  chat about it (/api/ingest/ask, /api/chat/…). The page and the popup
 *  open the panel through here — Chrome lets an extension open it only
 *  during the click that asked, so that happens before anything is awaited.
 * ══════════════════════════════════════════════════════════════════════════ */

const DEFAULT_SERVER = 'http://localhost:9860';
const KEEP = 25;                     // batches remembered
const TIMEOUT = 20000;               // ms a server answer may take — a hung server ends the wait

async function server() {
    const {server} = await chrome.storage.sync.get('server');
    return String(server || DEFAULT_SERVER).trim().replace(/\/+$/, '');
}

/* The server's access token (auth.api_token), for a server with its login
 * on. Kept in this browser only (storage.local, not synced) and never handed
 * out: the page and the panel learn only whether one is set. */
async function token() {
    const {token} = await chrome.storage.local.get('token');
    return String(token || '').trim();
}

async function api(path, init) {
    const base = await server();
    const tok = await token();
    let r;
    const signal = typeof AbortSignal !== 'undefined' && AbortSignal.timeout ? AbortSignal.timeout(TIMEOUT) : undefined;
    const headers = {'Content-Type': 'application/json', ...(tok ? {'X-API-Token': tok} : {})};
    try {
        r = await fetch(base + path, {headers, signal, ...(init || {})});
    } catch (e) {
        if (e && (e.name === 'TimeoutError' || e.name === 'AbortError')) {
            throw new Error(`The server at ${base} didn't answer in time`);
        }
        throw new Error(`The server at ${base} can't be reached`);
    }
    const d = await r.json().catch(() => ({}));
    if (r.status === 401) {
        throw new Error(tok ? 'The server refused the access token — check it under ⚙'
            : 'The server has its login on — enter its access token (auth.api_token) under ⚙');
    }
    if (!r.ok || d.error) throw new Error(d.error || `The server answered ${r.status}`);
    return d;
}

// ── The batches sent ─────────────────────────────────────────────────────

async function remembered() {
    const {batches} = await chrome.storage.local.get('batches');
    return Array.isArray(batches) ? batches : [];
}

/* Every change to the list, one after the other: two read-change-write
 * rounds at once (the half-minute check, a send from a page) lost the
 * batch one of them added. */
let writing = Promise.resolve();
function serialized(fn) {
    const p = writing.then(fn, fn);
    writing = p.catch(() => {});
    return p;
}

/* A batch's latest state kept — whoever asked (the check, a page's bar,
 * the popup). The one that turns from working to finished says so once,
 * here: a page polling every few seconds saw it first, and the check then
 * found nothing working any more to tell about. */
function remember(batch, extra) {
    if (!batch || !batch.id) return Promise.resolve();
    return serialized(async () => {
        const list = await remembered();
        const i = list.findIndex(b => b.id === batch.id);
        const before = i >= 0 ? list[i] : null;
        const row = {...(before || {}), ...(extra || {}), id: batch.id, last: batch, at: Date.now()};
        const finished = !!before && !before.notified && before.last && before.last.status === 'working'
            && batch.status && batch.status !== 'working';
        if (finished) row.notified = true;
        if (i >= 0) list.splice(i, 1);
        list.unshift(row);
        await chrome.storage.local.set({batches: list.slice(0, KEEP)});
        if (finished) notify(batch);
        return row;
    });
}

function forget(id) {
    return serialized(async () => {
        const list = (await remembered()).filter(b => b.id !== id);
        await chrome.storage.local.set({batches: list});
    });
}

/* Videos of a batch no longer listed — ``urls`` left out: the whole batch.
 * Kept here too: a server that lost the batch (or can't be reached) shows
 * them no more either. A batch without a video left goes from the list. */
function hide(id, urls) {
    return serialized(async () => {
        const list = await remembered();
        const row = list.find(b => b.id === id);
        if (!row) return;
        row.hidden = [...new Set([...(row.hidden || []), ...(urls || [])])];
        const items = (row.last && row.last.items) || [];
        const left = urls ? items.filter(it => !row.hidden.includes(it.url)) : [];
        await chrome.storage.local.set({batches: left.length ? list : list.filter(b => b.id !== id)});
    });
}

/* Taken out of a batch (the side panel's ✕): not fetched, not transcribed
 * if it still waits — the server says what it stopped. */
async function drop(id, urls) {
    let stopped = null;
    try {
        stopped = await api(`/api/ingest/${encodeURIComponent(id)}/drop`, {method: 'POST',
            body: JSON.stringify(urls ? {urls} : {})});
        await remember(await api('/api/ingest/' + encodeURIComponent(id)));
    } catch (e) {
        // The server lost it, or can't be reached: off the list all the same
    }
    await hide(id, urls);
    refresh();
    return {ok: true, ...(stopped || {})};
}

/* The list emptied — ``all``: the working batches too, taken out (their
 * videos not fetched yet are not fetched); else only the finished ones. */
async function clearSent(all) {
    const rows = await remembered();
    const working = rows.filter(r => r.last && r.last.status === 'working');
    if (all) await Promise.all(working.map(r => drop(r.id)));
    await serialized(async () => {
        const list = all ? [] : (await remembered()).filter(r => r.last && r.last.status === 'working');
        await chrome.storage.local.set({batches: list});
    });
    refresh();
    return {ok: true, left: all ? 0 : working.length};
}

/* The running batches asked again: the badge, and a word when one is done
 * (remember says it). One round at a time — asked during a round, exactly
 * one more follows it: a batch sent just then counts on the badge at once,
 * not half a minute later. */
let refreshing = null, refreshAgain = false;
function refresh() {
    if (refreshing) {
        refreshAgain = true;
        return refreshing;
    }
    refreshing = (async () => {
        do {
            refreshAgain = false;
            await refreshNow();
        } while (refreshAgain);
    })().finally(() => { refreshing = null; });
    return refreshing;
}

async function refreshNow() {
    const list = await remembered();
    let working = 0;
    for (const row of list) {
        if (!row.last || row.last.status !== 'working') continue;
        let b;
        try {
            b = await api('/api/ingest/' + encodeURIComponent(row.id));
        } catch (e) {
            continue;
        }
        await remember(b);
        if (b.status === 'working') {
            working += (b.items || []).filter(it => !['transcribed', 'summarized', 'error'].includes(it.state)).length;
        }
    }
    chrome.action.setBadgeText({text: working ? String(working) : ''});
    chrome.action.setBadgeBackgroundColor({color: '#2563eb'});
}

/* A path in the app — never another address: "/x" yes; "//host",
 * "/\\host", "@host", "javascript:" no (put behind the server's address,
 * "@evil.example" made http://localhost:9860@evil.example). */
function appPath(p) {
    p = String(p ?? '');
    return /^\/(?![\/\\])[^\u0000-\u001f\u007f]*$/.test(p) ? p : '/';
}

function notify(b) {
    const items = b.items || [];
    const failed = items.filter(it => it.state === 'error').length;
    const what = {transcribe: 'transcribed', summarize: 'summarized', analyze: 'analyzed'}[b.mode] || 'done';
    const n = items.length - failed;
    chrome.notifications.create('wtm-' + b.id, {
        type: 'basic', iconUrl: 'icons/icon128.png', title: 'Whisper Transcription Manager',
        message: `${n} video${n === 1 ? '' : 's'} ${what}` + (failed ? ` · ${failed} failed` : '')
            + (items.length === 1 && items[0].title ? `: ${items[0].title}` : ''),
    });
}

chrome.notifications.onClicked.addListener(async id => {
    const bid = id.replace(/^wtm-/, '');
    const row = (await remembered()).find(b => b.id === bid);
    const b = row && row.last;
    const path = b && b.analysis ? b.analysis.open
        : (b && b.items && b.items.length === 1 && b.items[0].open) || '/';
    chrome.tabs.create({url: (await server()) + appPath(path)});
});

chrome.alarms.create('wtm-refresh', {periodInMinutes: 0.5});
chrome.alarms.onAlarm.addListener(a => { if (a.name === 'wtm-refresh') refresh(); });

// ── Questions from the page and the popup ───────────────────────────────

async function handle(msg) {
    switch (msg.type) {
    case 'ingest': {
        // options: how (the side panel's form) — left out, the defaults;
        // again: one more summary of a video summarized before
        const d = await api('/api/ingest', {method: 'POST', body: JSON.stringify({
            urls: msg.urls || [], mode: msg.mode || 'transcribe', task: msg.task || '',
            title: msg.title || '', ...(msg.options ? {options: msg.options} : {}),
            again: !!msg.again})});
        await remember(d.batch);
        refresh();
        return {batch: d.batch};
    }
    case 'batch': {
        const b = await api('/api/ingest/' + encodeURIComponent(msg.id));
        await remember(b);
        return {batch: b};
    }
    case 'cancel':
        return api(`/api/ingest/${encodeURIComponent(msg.id)}/cancel`, {method: 'POST'});
    case 'known':
        return api('/api/ingest/known', {method: 'POST', body: JSON.stringify({
            videos: msg.videos || [], summaries: !!msg.summaries})});
    case 'recent':
        // Without the videos taken out (``hide``)
        return {batches: (await remembered()).map(r => !r.hidden || !r.last ? r
            : {...r, last: {...r.last, items: (r.last.items || []).filter(it => !r.hidden.includes(it.url))}})};
    case 'selection':
        // A page's selection changed — for the side panel, which hears it itself
        return {ok: true};
    case 'forget':
        await forget(msg.id);
        return {ok: true};
    case 'drop':
        return drop(String(msg.id || ''), Array.isArray(msg.urls) ? msg.urls : undefined);
    case 'clear-sent':
        return clearSent(!!msg.all);
    case 'open':
        await chrome.tabs.create({url: (await server()) + appPath(msg.path || '/')});
        return {ok: true};
    case 'server':
        return {server: await server(), default: DEFAULT_SERVER, token: !!(await token())};
    case 'set-server': {
        const value = String(msg.server || '').trim().replace(/\/+$/, '');
        if (value && !/^https?:\/\/[^/\s]+$/.test(value)) throw new Error('An address like http://192.0.2.10:9860');
        await chrome.storage.sync.set({server: value});
        // An empty field keeps the saved token ('forget-token' removes it)
        const tok = String(msg.token || '').trim();
        if (tok) await chrome.storage.local.set({token: tok});
        return {server: await server(), token: !!(await token())};
    }
    case 'forget-token':
        await chrome.storage.local.set({token: ''});
        return {token: false};
    case 'inject':
        return {ok: (await inject(msg.tabId)) > 0};
    case 'health': {
        const d = await api('/api/health');
        return {ok: d.status === 'ok', server: await server()};
    }
    // ── The side panel ──
    case 'options':
        return api('/api/ingest/options');
    case 'set-defaults':
        return api('/api/ingest/defaults', {method: 'POST', body: JSON.stringify(msg.defaults || {})});
    case 'transcript':
        return api('/api/ingest/transcript', {method: 'POST', body: JSON.stringify({
            url: msg.url || '', title: msg.title || ''})});
    case 'ask':
        return api('/api/ingest/ask', {method: 'POST', body: JSON.stringify({
            url: msg.url || '', title: msg.title || '', new: !!msg.new})});
    case 'chat-conv':
        return api('/api/chat/conversations/' + encodeURIComponent(msg.id || ''));
    case 'chat-send':
        return api('/api/chat/send', {method: 'POST', body: JSON.stringify({
            text: msg.text || '', conversation: msg.conversation || ''})});
    case 'chat-run':
        return api(`/api/chat/runs/${encodeURIComponent(msg.id || '')}?after=${Number(msg.after) || 0}`);
    case 'chat-stop':
        return api(`/api/chat/runs/${encodeURIComponent(msg.id || '')}/stop`, {method: 'POST'});
    default:
        throw new Error('Unknown request: ' + msg.type);
    }
}

/* What the helper in a web page may ask — not the server's address, the
 * defaults, the chat, or the list of what was sent: those are the popup's
 * and the side panel's. */
const FROM_PAGES = new Set(['panel', 'ingest', 'batch', 'cancel', 'known', 'open', 'selection']);

function fromExtensionPage(sender) {
    const own = chrome.runtime.getURL ? chrome.runtime.getURL('') : '';
    return !!(sender && own && String(sender.url || '').startsWith(own));
}

chrome.runtime.onMessage.addListener((msg, sender, reply) => {
    const done = p => p.then(reply, e => reply({error: String((e && e.message) || e)}));
    msg = msg || {};
    if (sender && sender.id && chrome.runtime.id && sender.id !== chrome.runtime.id) {
        done(Promise.reject(new Error('Not from this extension')));
    } else if (sender && sender.tab && !fromExtensionPage(sender) && !FROM_PAGES.has(msg.type)) {
        done(Promise.reject(new Error('Not asked from a page: ' + msg.type)));
    } else if (msg.type === 'panel') {
        done(openPanel(msg.intent || {}, sender && sender.tab));
    } else {
        done(handle(msg));
    }
    return true;          // the answer comes later
});

/* The side panel, for the tab asked from, with what it should show (the
 * panel reads ``panelIntent``). Opened first: only during the click. */
function openPanel(intent, tab) {
    let opening;
    try {
        opening = chrome.sidePanel.open(tab && tab.id ? {tabId: tab.id} : {windowId: chrome.windows.WINDOW_ID_CURRENT});
    } catch (e) {
        return Promise.reject(new Error('This browser has no side panel for extensions (Chrome 116 or newer)'));
    }
    const stored = chrome.storage.session.set({panelIntent: {...intent, tabId: tab && tab.id, at: Date.now()}});
    return Promise.all([opening, stored]).then(() => ({ok: true}));
}

// ── The helper in a page ─────────────────────────────────────────────────

/* videos.js and content.js into a tab — or into every web page that is open.
 * A browser page (chrome://, the Web Store) refuses: nothing to do there. */
async function inject(tabId) {
    const tabs = tabId ? [{id: tabId}] : await chrome.tabs.query({url: ['http://*/*', 'https://*/*']});
    let ok = 0;
    for (const t of tabs) {
        try {
            await chrome.scripting.executeScript({target: {tabId: t.id}, files: ['videos.js', 'content.js']});
            ok++;
        } catch (e) { /* a page that takes no extension */ }
    }
    return ok;
}

// ── Right-click ──────────────────────────────────────────────────────────

/* "Ask before sending" (the side panel's ⚙), read ahead: a right-click opens
 * the panel only while the click lasts — no time to read it then. */
const prefs = {askFirst: false, loaded: false};
const prefsReady = chrome.storage.sync.get('askFirst').then(d => {
    prefs.askFirst = !!d.askFirst;
    prefs.loaded = true;
}, () => { prefs.loaded = true; });
chrome.storage.onChanged.addListener((changes, area) => {
    if (area === 'sync' && changes.askFirst) prefs.askFirst = !!changes.askFirst.newValue;
});

const LINKS = ['*://*.youtube.com/watch*', '*://youtube.com/watch*', '*://youtu.be/*',
               '*://*.youtube.com/shorts/*', '*://*.youtube.com/live/*', '*://*.youtube.com/playlist*'];
const PAGES = ['*://*.youtube.com/watch*', '*://*.youtube.com/shorts/*', '*://*.youtube.com/playlist*'];

chrome.runtime.onInstalled.addListener(() => {
    // The tabs open already get the helper now — Chrome only brings it into
    // pages loaded after the install (or a reload of the extension)
    inject();
    // No parent item: Chrome puts the ones that apply under "Whisper Transcription Manager"
    // by itself, and shows nothing on a page without a YouTube video
    chrome.contextMenus.removeAll(() => {
        chrome.contextMenus.create({id: 'wtm-link-summarize', title: 'Summarize this video',
                                    contexts: ['link'], targetUrlPatterns: LINKS});
        chrome.contextMenus.create({id: 'wtm-link-transcribe', title: 'Transcribe this video',
                                    contexts: ['link'], targetUrlPatterns: LINKS});
        chrome.contextMenus.create({id: 'wtm-page-summarize', title: 'Summarize this video',
                                    contexts: ['page'], documentUrlPatterns: PAGES});
        chrome.contextMenus.create({id: 'wtm-page-transcribe', title: 'Transcribe this video',
                                    contexts: ['page'], documentUrlPatterns: PAGES});
        chrome.contextMenus.create({id: 'wtm-link-choose', title: 'Choose how… (side panel)',
                                    contexts: ['link'], targetUrlPatterns: LINKS});
        chrome.contextMenus.create({id: 'wtm-page-choose', title: 'Choose how… (side panel)',
                                    contexts: ['page'], documentUrlPatterns: PAGES});
    });
    refresh();
});

chrome.contextMenus.onClicked.addListener(async (info, tab) => {
    const m = String(info.menuItemId).match(/^wtm-(link|page)-(summarize|transcribe|choose)$/);
    if (!m) return;
    const url = m[1] === 'link' ? info.linkUrl : (info.pageUrl || (tab && tab.url));
    if (!url) return;
    // The form in the side panel — opened at once, while the click lasts
    if (m[2] === 'choose' || (prefs.loaded && prefs.askFirst)) {
        openPanel({form: m[2] === 'choose' ? 'summarize' : m[2], url}, tab).catch(e => warn(e));
        return;
    }
    if (!prefs.loaded) {
        // A worker that just woke up: the setting is read now — the panel may
        // then refuse to open, which says so
        await prefsReady;
        if (prefs.askFirst) {
            openPanel({form: m[2], url}, tab).catch(() =>
                warn(new Error('click the extension icon → ▤ to choose how')));
            return;
        }
    }
    try {
        const {batch} = await handle({type: 'ingest', mode: m[2], urls: [url]});
        // The page shows it in its bar, where it can
        if (tab && tab.id) chrome.tabs.sendMessage(tab.id, {type: 'follow', batch}, () => void chrome.runtime.lastError);
    } catch (e) {
        warn(e);
    }
});

function warn(e) {
    chrome.notifications.create({type: 'basic', iconUrl: 'icons/icon128.png', title: 'Whisper Transcription Manager',
                                 message: 'Not sent: ' + ((e && e.message) || e)});
}
