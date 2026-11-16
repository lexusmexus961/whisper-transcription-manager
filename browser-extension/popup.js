/* ══════════════════════════════════════════════════════════════════════════
 *  The toolbar popup: the page's video, the others, what was sent, the server
 *
 *  The video a watch page plays comes first, with its picture and what it is up
 *  to — summarize or transcribe it from here, read its summary. Every other
 *  video of the page is folded away below, with a filter — the same
 *  selection as the ⊕ behind their titles. Below that, every batch sent with
 *  each video's state, its summary and a link into Whisper Transcription Manager; the
 *  server's address behind ⚙.
 *
 *  For reading, asking and choosing exactly how: the side panel (▤). With
 *  "ask before sending" on (sidepanel.js), Summarize and Transcribe open its
 *  form instead of sending. esc, STATE, md and the rest: ui.js.
 * ══════════════════════════════════════════════════════════════════════════ */

const $ = id => document.getElementById(id);
const P = {tab: null, videos: [], main: null, open: new Set(), timer: 0, sumOpen: false, askFirst: false};

/* The side panel, opened from this click — Chrome allows it only during
 * one — with what it should show; the popup closes. */
function openPanel(intent) {
    try {
        chrome.sidePanel.open({windowId: P.tab ? P.tab.windowId : undefined});
    } catch (e) {
        showError('This browser has no side panel for extensions (Chrome 116 or newer).');
        return;
    }
    chrome.storage.session.set({panelIntent: {...(intent || {}), tabId: P.tab && P.tab.id, at: Date.now()}})
        .finally(() => window.close());
}

const page = msg => new Promise(resolve => {
    if (!P.tab) return resolve({error: 'no tab'});
    chrome.tabs.sendMessage(P.tab.id, msg, r =>
        resolve(chrome.runtime.lastError ? {error: chrome.runtime.lastError.message} : (r || {})));
});

function showError(text) {
    $('err').textContent = text || '';
    $('err').hidden = !text;
}

// ── The server ───────────────────────────────────────────────────────────

async function health() {
    const d = await ask({type: 'health'});
    const dot = $('dot');
    dot.className = 'dot ' + (d.ok ? 'ok' : 'bad');
    dot.title = d.ok ? `Connected · ${d.server}` : (d.error || 'Not reachable');
    if (!d.ok) $('settings').hidden = false;
    const s = await ask({type: 'server'});
    $('server').value = s.server || '';
    $('server').placeholder = s.default || '';
    $('token').value = '';
    $('token').placeholder = s.token ? 'Access token saved — type to replace'
        : 'Access token — only with the login on';
    $('btnForgetToken').hidden = !s.token;
    $('serverNote').textContent = d.ok ? 'Connected.'
        : `${d.error || 'Not reachable'} — the address Whisper Transcription Manager is reached at from this browser.`;
}

$('btnSettings').addEventListener('click', () => { $('settings').hidden = !$('settings').hidden; });
$('btnPanel').addEventListener('click', () => openPanel({}));
$('btnSave').addEventListener('click', async () => {
    const d = await ask({type: 'set-server', server: $('server').value, token: $('token').value});
    if (d.error) { $('serverNote').textContent = d.error; return; }
    await health();
    loadRecent();
});
$('btnForgetToken').addEventListener('click', async () => {
    await ask({type: 'forget-token'});
    health();
});

// ── The page's videos ────────────────────────────────────────────────────

async function loadVideos() {
    [P.tab] = await chrome.tabs.query({active: true, currentWindow: true});
    let d = await page({type: 'list'});
    if (d.error && P.tab) {
        // A page open from before the extension was installed (or reloaded)
        // has no helper yet — it gets one now, then it is asked again
        const injected = await ask({type: 'inject', tabId: P.tab.id});
        if (injected.ok) d = await page({type: 'list'});
    }
    const box = $('videos');
    if (d.error) {
        P.videos = [];
        P.main = null;
        $('acts').hidden = true;
        $('mainBox').hidden = true;
        $('others').open = true;
        box.innerHTML = '<div class="note">This page takes no extension (a browser page or the Web Store) — open the page with the videos.</div>';
        return;
    }
    P.main = d.main || null;
    P.videos = (d.videos || []).filter(v => !v.main);
    renderMain();
    renderVideos();
}

function renderMain() {
    const m = P.main;
    $('mainBox').hidden = !m;
    if (!m) return;
    const id = (m.url.match(/[?&]v=([\w-]{11})/) || [])[1] || '';
    $('mainThumb').src = thumb(id, 'mqdefault');
    $('mainTitle').textContent = m.title || 'This video';
    const st = m.state ? STATE[m.state] || [m.state, ''] : null;
    $('mainState').innerHTML = st ? `<span class="chip ${st[1]}">${esc(st[0])}</span>`
        : '<span class="note">Not in Whisper Transcription Manager yet.</span>';
    const working = WORKING.includes(m.state);
    for (const b of $('mainActs').querySelectorAll('button[data-main]')) {
        b.hidden = working || (b.dataset.main === 'summarize' ? m.state === 'summarized'
                                                              : ['transcribed', 'summarized'].includes(m.state));
    }
    // Options: also for one summarized — once more, another template
    $('mainOptions').hidden = working;
    $('mainRead').hidden = !['transcribed', 'summarized'].includes(m.state);
    $('mainSum').hidden = !m.summary;
    $('mainSum').textContent = P.sumOpen ? 'hide the summary' : 'show the summary';
    $('mainOpen').hidden = !m.open;
    $('mainSummary').hidden = !(m.summary && P.sumOpen);
    $('mainSummary').innerHTML = m.summary ? md(m.summary) : '';
}

$('mainActs').addEventListener('click', async e => {
    const b = e.target.closest('button');
    if (!b) return;
    if (b.dataset.main && P.askFirst) {
        openPanel({form: b.dataset.main, url: P.main.url, title: P.main.title});
    } else if (b.id === 'mainOptions') {
        openPanel({form: 'summarize', url: P.main.url, title: P.main.title});
    } else if (b.id === 'mainRead') {
        openPanel({tab: 'summary', url: P.main.url, title: P.main.title});
    } else if (b.dataset.main) {
        showError('');
        b.disabled = true;
        const d = await page({type: 'main-send', mode: b.dataset.main});
        b.disabled = false;
        if (d.error) showError('Not sent: ' + d.error);
        await loadVideos();
        await loadRecent();
    } else if (b.id === 'mainSum') {
        P.sumOpen = !P.sumOpen;
        renderMain();
    } else if (b.id === 'mainOpen' && P.main && P.main.open) {
        ask({type: 'open', path: P.main.open});
    }
});

function renderVideos() {
    const box = $('videos');
    const others = $('others');
    const n = P.videos.length;
    $('pageCount').textContent = n ? String(n) : '';
    // Folded away where the page plays a video of its own; open elsewhere
    if (!others.dataset.touched) others.open = !P.main;
    if (!n) {
        box.innerHTML = `<div class="note">${P.main ? 'No other videos on this page.' : 'No YouTube videos on this page.'}</div>`;
    } else {
        const q = ($('filter').value || '').trim().toLowerCase();
        const shown = P.videos.filter(v => !q || (v.title || v.url).toLowerCase().includes(q));
        box.innerHTML = shown.map(v => {
            const k = v.known ? STATE[v.known.state] || [v.known.state, 'ok'] : null;
            return `<label class="v" title="${esc(v.title || v.url)}"><input type="checkbox" data-k="${esc(v.key)}"${v.selected ? ' checked' : ''}>
                ${v.id ? `<img src="${esc(thumb(v.id))}" alt="">` : ''}
                <span class="t">${v.kind === 'playlist' ? '▤ ' : ''}${esc(v.title || v.url)}</span>
                ${k ? `<span class="chip ok">${esc(k[0])}</span>` : ''}</label>`;
        }).join('') || '<div class="note">No title matches.</div>';
    }
    $('filter').hidden = n < 8;
    $('btnAll').hidden = n < 2;
    const q = ($('filter').value || '').trim().toLowerCase();
    $('btnAll').textContent = pickAllLabel(P.videos.filter(v => !q || (v.title || v.url).toLowerCase().includes(q)),
                                           n, !!q);
    const selected = P.videos.filter(v => v.selected).length;
    $('pickHint').textContent = selected ? `${selected} selected` : (P.main ? 'tick to add' : '');
    $('acts').hidden = !selected;
}

// Folded or unfolded by hand stays so — the click, not the toggle event: that
// one comes later, and also when the list is folded by the code
$('others').querySelector('summary').addEventListener('click', () => { $('others').dataset.touched = '1'; });
$('filter').addEventListener('input', renderVideos);

$('videos').addEventListener('change', async e => {
    const cb = e.target.closest('input[data-k]');
    if (!cb) return;
    await page({type: 'toggle', key: cb.dataset.k, on: cb.checked});
    const v = P.videos.find(x => x.key === cb.dataset.k);
    if (v) v.selected = cb.checked;
    renderVideos();
});

$('btnAll').addEventListener('click', async () => {
    const q = ($('filter').value || '').trim().toLowerCase();
    const shown = P.videos.filter(v => !q || (v.title || v.url).toLowerCase().includes(q));
    const all = !shown.every(v => v.selected);
    for (const v of shown) {
        await page({type: 'toggle', key: v.key, on: all});
        v.selected = all;
    }
    renderVideos();
});

async function sendSelected(mode, task) {
    showError('');
    const d = await page({type: 'send', mode, task});
    if (d.error || !d.batch) {
        // The page's helper answers with the server's word when there is one
        showError(/Receiving end does not exist/.test(d.error || '')
            ? 'Not sent: this page has no Whisper TM helper — reload the page and try again.'
            : d.reason || ('Not sent: ' + (d.error || 'no answer from the server')
                           + (/can't be reached|didn't answer|no answer/.test(d.error || 'no answer')
                               ? ' — check the address behind ⚙.' : '')));
        return;
    }
    $('analyzeBox').hidden = true;
    $('task').value = '';
    await loadVideos();
    await loadRecent();
}

$('acts').addEventListener('click', e => {
    const b = e.target.closest('button[data-mode], #btnOptions');
    if (!b) return;
    const picked = P.videos.filter(v => v.selected);
    if (b.id === 'btnOptions' || P.askFirst) {
        openPanel({form: b.dataset.mode || 'summarize', urls: picked.map(v => v.url),
                   titles: picked.map(v => v.title || '')});
    } else {
        sendSelected(b.dataset.mode, '');
    }
});
$('btnAnalyze').addEventListener('click', () => {
    $('analyzeBox').hidden = false;
    $('task').focus();
});
$('btnBack').addEventListener('click', () => { $('analyzeBox').hidden = true; });
$('btnStart').addEventListener('click', () => {
    const task = $('task').value.trim();
    if (!task) { showError('Write the task first.'); return; }
    sendSelected('analyze', task);
});

// ── What was sent ────────────────────────────────────────────────────────

async function loadRecent() {
    const d = await ask({type: 'recent'});
    const rows = d.batches || [];
    // The ones still working are asked again — each one now
    await Promise.all(rows.filter(r => r.last && r.last.status === 'working').slice(0, 5)
        .map(async r => {
            const x = await ask({type: 'batch', id: r.id});
            if (x.batch) r.last = x.batch;
        }));
    renderRecent(rows);
    clearTimeout(P.timer);
    // Asked again while something works — the page's video too: its last
    // step can come in after the batch says it is done
    if (rows.some(r => r.last && r.last.status === 'working') || (P.main && WORKING.includes(P.main.state))) {
        P.timer = setTimeout(async () => { await loadVideos(); await loadRecent(); }, 3000);
    }
}

function renderRecent(rows) {
    const box = $('recent');
    if (!rows.length) {
        box.innerHTML = '<div class="note">Nothing sent yet. On YouTube, use ✦ Whisper TM next to Share, or the ⊕ behind any title.</div>';
        return;
    }
    box.innerHTML = rows.slice(0, 12).map(r => {
        const b = r.last || {};
        const what = {transcribe: 'Transcribe', summarize: 'Summarize', analyze: 'Analyze'}[b.mode] || b.mode;
        const items = (b.items || []).map((it, i) => {
            const st = STATE[it.state] || [it.state, ''];
            const key = b.id + ':' + i;
            const open = P.open.has(key);
            const pct = it.state === 'downloading' && it.progress ? ` ${it.progress} %` : '';
            return `<div class="it"><span class="t" title="${esc(it.title || it.url)}">${esc(it.title || it.url)}</span>
                <span class="chip ${st[1]}" title="${esc(it.error || '')}">${esc(st[0] + pct)}</span>
                ${it.summary ? `<button class="l" data-sum="${esc(key)}">${open ? 'hide' : 'summary'}</button>` : ''}
                ${it.open ? `<button class="l" data-open="${esc(it.open)}">open ↗</button>` : ''}</div>
                ${it.summary && open ? `<div class="sum" data-url="${esc(it.url || '')}">${md(it.summary)}</div>` : ''}`;
        }).join('');
        const a = b.analysis;
        const an = a ? `<div class="it"><span class="t">Analysis: ${esc(a.title || '')}</span>
            <span class="chip ${a.status === 'done' ? 'ok' : a.status === 'error' ? 'err' : 'run'}">${esc(
                {waiting: 'waiting for transcripts', queued: 'in the LLM queue', running: (a.progress || 0) + ' %',
                 done: 'done', error: 'failed', cancelled: 'cancelled'}[a.status] || a.status)}</span>
            <button class="l" data-open="${esc(a.open || '')}">open ↗</button></div>` : '';
        return `<div class="batch"><div class="bh"><b>${esc(what)}</b> · ${(b.items || []).length} video${(b.items || []).length === 1 ? '' : 's'} · ${esc(when(r.at))}
                <span class="gap"></span>${b.status === 'working' ? '<span class="chip run">working</span>' : ''}
                <button class="l" data-forget="${esc(b.id)}" title="Remove from this list">✕</button></div>
            ${items}${an}${b.error ? `<div class="err">${esc(b.error)}</div>` : ''}</div>`;
    }).join('');
}

/* A time in a summary plays the video from there */
document.addEventListener('click', e => {
    const a = e.target.closest('a.time');
    if (!a) return;
    e.preventDefault();
    const box = a.closest('[data-url]');
    const url = box ? box.dataset.url : (P.main && P.main.url) || '';
    playAt(url, Number(a.dataset.t) || 0, P.tab, P.main ? videoIdOf(P.main.url) : '');
});

$('recent').addEventListener('click', async e => {
    const el = e.target.closest('button');
    if (!el) return;
    if (el.dataset.sum) {
        if (P.open.has(el.dataset.sum)) P.open.delete(el.dataset.sum); else P.open.add(el.dataset.sum);
        renderRecent((await ask({type: 'recent'})).batches || []);
    } else if (el.dataset.open) {
        ask({type: 'open', path: el.dataset.open});
    } else if (el.dataset.forget) {
        await ask({type: 'forget', id: el.dataset.forget});
        loadRecent();
    }
});

$('btnApp').addEventListener('click', () => ask({type: 'open', path: '/'}));

chrome.storage.sync.get('askFirst').then(d => { P.askFirst = !!d.askFirst; }, () => {});
health();
loadVideos();
loadRecent();
