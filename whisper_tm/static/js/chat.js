/* ══════════════════════════════════════════════════════════════════════════
 *  ASK — the chat tab of the search
 *
 *  A chat with the connected LLM, built on the app's own MCP server, to
 *  go along with the search.
 *
 *  The model looks things up itself through the app's MCP tools (search,
 *  transcripts, who speaks where …) — server side, core/chat.py. Here:
 *
 *    · the conversations on the left, kept on the server;
 *    · the conversation in the middle — the answer as it is written, each
 *      tool call as a small line between the texts, where the model made
 *      it (click: what it asked and found);
 *    · every quote carries a reference [S12]; it becomes a chip that plays
 *      the passage in the background player, and on the right the passages
 *      the answer quoted, each playable and openable in its transcript.
 *
 *  "Ask" in the query bar hands the search over: the term and its filters
 *  go along as the scope, so a question picks up where the search stood.
 * ══════════════════════════════════════════════════════════════════════════ */

const CHAT = {
    conv: null,            // the conversation shown (turns, refs) — null: a new one
    run: null,             // {id, events: Map n → event, done} while answering
    list: [],
    ready: null,
    timer: null,
    hot: '',               // the reference last played
    opened: new Set(),     // tool steps opened to their detail
    drawn: null,           // what the finished turns on screen were drawn from (chatRender)
    thinkOpen: new Set(),  // "Thinking" blocks opened by hand (turn time:round)
    sending: false,        // a question on its way to the server
    listError: '',
};

const CHAT_EXAMPLES = [
    'Who talks about money most often, and when?',
    'Find the passages where someone mentions a birthday and quote them.',
    'Which recordings from 2024 are about games, and who speaks in them?',
];

function chatOpen() {
    chatLoadList();
    chatReadiness();
    chatScopeSync();
    chatRender();
    chatPollResume();
    setTimeout(() => document.getElementById('chatInput')?.focus(), 30);
}

/* The rail's "Ask": the search workspace, straight on the chat. */
function openAsk() {
    const ov = document.getElementById('svOverlay');
    if (!ov) return;
    ov.classList.add('visible');
    const scope = document.getElementById('chatScope');
    if (scope) scope.checked = false;
    if (svState.tab === 'chat') chatOpen();
    else svTab('chat');
}

/* "Ask" next to the search field: the term and the filters go along. */
function svAskAbout() {
    const q = (document.getElementById('svQuery')?.value || '').trim();
    const scope = document.getElementById('chatScope');
    if (scope) scope.checked = !!(q || svHasFilter());
    svTab('chat');
    const input = document.getElementById('chatInput');
    if (input && q && !input.value.trim()) {
        input.value = `What is said about "${q}"?`;
        input.select();
    }
}

/* "Ask about this recording" in the passage view. */
function svAskFile(filename) {
    const scope = document.getElementById('chatScope');
    if (scope) scope.checked = false;
    chatNew();
    svTab('chat');
    const input = document.getElementById('chatInput');
    if (input) {
        input.value = `Summarise what is talked about in "${filename}" — with the passages.`;
        input.focus();
    }
}

// ── What the search shows, as text for the model ────────────────────────

function chatScopeText() {
    const parts = [];
    const q = (svState.query || document.getElementById('svQuery')?.value || '').trim();
    if (q) parts.push(`search term "${q}"`);
    const file = svState.pinned || svState.file;
    if (file) parts.push(`only files matching "${file}"`);
    if (svState.speaker) parts.push(`speaker label ${svState.speaker}`);
    if (svState.person) parts.push(`person ${svState.person}` + (svState.personSegments ? ' (only their sentences)' : ''));
    const v = id => (document.getElementById(id)?.value || '').trim();
    if (v('svDateFrom') || v('svDateTo')) parts.push(`recorded ${v('svDateFrom') || '…'} to ${v('svDateTo') || '…'}`);
    if (v('svDurationMin') || v('svDurationMax')) parts.push(`duration ${v('svDurationMin') || '0'}–${v('svDurationMax') || '∞'} min`);
    return parts.join(', ');
}

function chatScopeSync() {
    const t = chatScopeText();
    const box = document.getElementById('chatScopeBox');
    const el = document.getElementById('chatScopeText');
    if (box) box.hidden = !t;
    if (el) { el.textContent = t ? '— ' + t : ''; el.title = t; }
    if (!t) { const c = document.getElementById('chatScope'); if (c) c.checked = false; }
}

// ── The conversations ────────────────────────────────────────────────────

async function chatLoadList() {
    try {
        const d = await (await fetch('/api/chat/conversations')).json();
        CHAT.list = d.conversations || [];
        CHAT.listError = '';
    } catch (e) {
        CHAT.listError = String(e.message || e) || 'no answer';
    }
    chatRenderList();
}

function chatRenderList() {
    const el = document.getElementById('chatList');
    if (!el) return;
    const cur = CHAT.conv && CHAT.conv.id;
    el.innerHTML = CHAT.list.length ? CHAT.list.map(c => `
        <div class="sv-chat-item${c.id === cur ? ' active' : ''}" role="button" tabindex="0" data-keyclick onclick="chatLoad('${escJs(c.id)}')"
             title="${esc(c.title)}">
            <span class="sv-chat-item-title">${esc(c.title)}</span>
            <span class="sv-chat-item-when">${esc(chatWhen(c.updated))}</span>
            <button class="sv-chat-item-del" title="Delete this conversation"
                    onclick="event.stopPropagation();chatDelete('${escJs(c.id)}')">✕</button>
        </div>`).join('')
        : CHAT.listError
            ? `<div class="sv-chat-empty-list bad">The conversations could not be loaded: ${esc(CHAT.listError)}</div>`
            : '<div class="sv-chat-empty-list">No conversations yet. Ask something below — it is kept here.</div>';
}

function chatWhen(t) {
    if (!t) return '';
    const d = new Date(t * 1000), now = new Date();
    return d.toDateString() === now.toDateString()
        ? d.toLocaleTimeString([], {hour: '2-digit', minute: '2-digit'})
        : d.toLocaleDateString([], {day: '2-digit', month: '2-digit'});
}

async function chatReadiness() {
    const el = document.getElementById('chatReady');
    if (!el) return;
    try {
        const d = await (await fetch('/api/chat/status')).json();
        CHAT.ready = d;
        el.innerHTML = d.problem
            ? `<span class="bad">${esc(d.problem)}</span>`
            : `<b>${esc(d.model || 'model')}</b> · ${(d.tools || []).length} tools of the MCP server`
              + `<br>${d.actions ? 'may start work (transcribe, summarise …)' : 'reads and searches only'}`
              + ` · <a href="#" onclick="event.preventDefault();openSettings('search')">settings</a>`;
    } catch (e) {
        el.textContent = 'Status not available.';
    }
}

function chatNew() {
    if (CHAT.run && !CHAT.run.done) return;
    CHAT.conv = null;
    CHAT.run = null;
    CHAT.hot = '';
    chatRender();
    chatRenderList();
    document.getElementById('chatInput')?.focus();
}

/* Returns false when it can't: an answer is still being written. "Ask about
 * these results" and "Ask about it" would otherwise land on the old
 * conversation without a word — now they say why. */
async function chatLoad(id) {
    if (CHAT.run && !CHAT.run.done) {
        Toast.info('An answer is still being written — wait for it or stop it, then open the conversation in the list.');
        chatLoadList();
        return false;
    }
    try {
        const r = await fetch('/api/chat/conversations/' + encodeURIComponent(id));
        if (!r.ok) throw new Error('not found');
        CHAT.conv = await r.json();
    } catch (e) {
        Toast.error('Conversation not found');
        return chatLoadList();
    }
    CHAT.run = null;
    chatRender(true);
    chatRenderList();
    return true;
}

async function chatDelete(id) {
    if (!confirm('Delete this conversation?')) return;
    try {
        const r = await fetch('/api/chat/conversations/' + encodeURIComponent(id), {method: 'DELETE'});
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
    } catch (e) {
        Toast.error('Not deleted: ' + (e.message || e));
        return;
    }
    if (CHAT.conv && CHAT.conv.id === id) chatNew();
    chatLoadList();
}

// ── Asking ───────────────────────────────────────────────────────────────

function chatKey(e) {
    if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) {
        e.preventDefault();
        chatSend();
    }
}

async function chatSend(textArg) {
    const input = document.getElementById('chatInput');
    const text = (textArg || input?.value || '').trim();
    if (!text || CHAT.sending || (CHAT.run && !CHAT.run.done)) return;
    const scope = document.getElementById('chatScope')?.checked ? chatScopeText() : '';
    // Enter twice (or Enter and a click) before the server answered started
    // two runs, and the first one's answer was lost
    CHAT.sending = true;
    const btn = document.getElementById('chatSend');
    if (btn) btn.disabled = true;
    let d;
    try {
        const r = await fetch('/api/chat/send', {
            method: 'POST', headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({text, conversation: CHAT.conv ? CHAT.conv.id : '',
                                  scope: scope ? 'The search shows: ' + scope : ''})});
        d = await r.json();
        if (!r.ok || d.error) throw new Error(d.error || r.status);
    } catch (e) {
        Toast.error('Not asked: ' + (e.message || e));
        return;
    } finally {
        CHAT.sending = false;
        if (btn) btn.disabled = false;
    }
    if (input) input.value = '';
    if (!CHAT.conv || CHAT.conv.id !== d.conversation) {
        CHAT.conv = {id: d.conversation, title: text.slice(0, 80), turns: [], refs: {}};
    }
    CHAT.conv.turns.push({role: 'user', text, at: Date.now() / 1000});
    CHAT.run = {id: d.run, events: new Map(), done: false, after: 0};
    chatButtons();
    chatRender(true);
    chatLoadList();
    chatPoll();
}

/* Asks while the answer is in view; out of view (the search closed, another
 * tab, a hidden browser tab) it pauses, and chatOpen or the browser tab shown
 * again picks it up. A failed request waits longer each time (to 8 s). */
function chatVisible() {
    const ov = document.getElementById('svOverlay');
    return !document.hidden && svState.tab === 'chat' && (!ov || ov.classList.contains('visible'));
}

function chatPollResume() {
    if (CHAT.run && !CHAT.run.done && !CHAT.timer && !CHAT.polling) chatPoll();
}

if (typeof document !== 'undefined' && document.addEventListener) {
    document.addEventListener('visibilitychange', () => { if (!document.hidden) chatPollResume(); });
}

async function chatPoll() {
    clearTimeout(CHAT.timer);
    CHAT.timer = null;
    const run = CHAT.run;
    if (!run || run.done) return;
    if (!chatVisible()) return;
    CHAT.polling = true;
    let failed = false;
    try {
        const r = await fetch(`/api/chat/runs/${run.id}?after=${run.after}`);
        const d = await r.json();
        if (CHAT.run !== run) return;
        // The server no longer knows the run (it was restarted: 404
        // "unknown run") or refuses it — that answer never finishes. Polling
        // on would keep "Answering …" up and Ask, New and the list blocked
        // until a reload; end it here with what was written so far.
        if (!r.ok || d.error) {
            chatEndRun(run, d.error || `HTTP ${r.status}`);
            return;
        }
        for (const e of d.events || []) {
            run.events.set(e.n, e);
            run.after = Math.max(run.after, e.n);
        }
        if (d.done) {
            run.done = true;
            const end = [...run.events.values()].find(e => e.type === 'done');
            if (end) {
                CHAT.conv.turns.push(end.turn);
                CHAT.conv.refs = end.refs || CHAT.conv.refs;
            }
        }
    } catch (e) { failed = true; /* the next poll tries again */ }
    finally { CHAT.polling = false; }
    run.fails = failed ? (run.fails || 0) + 1 : 0;
    chatRender();
    chatButtons();
    if (!run.done) CHAT.timer = setTimeout(chatPoll, failed ? Math.min(8000, 400 * 2 ** run.fails) : 400);
    else chatLoadList();
}

/* A run that can't be followed any further: its turn so far, with the
 * reason, becomes part of the conversation, and the buttons are free again. */
function chatEndRun(run, message) {
    const turn = chatLiveTurn() || {role: 'assistant', text: ''};
    delete turn.live;
    turn.error = 'The answer was lost: ' + message;
    run.done = true;
    if (CHAT.conv) CHAT.conv.turns.push(turn);
    chatRender();
    chatButtons();
    chatLoadList();
}

async function chatStop() {
    if (!CHAT.run || CHAT.run.done) return;
    try { await fetch(`/api/chat/runs/${CHAT.run.id}/stop`, {method: 'POST'}); } catch (e) { /* the run ends anyway */ }
}

function chatButtons() {
    const busy = !!(CHAT.run && !CHAT.run.done);
    const send = document.getElementById('chatSend');
    const stop = document.getElementById('chatStop');
    if (send) { send.disabled = busy; send.textContent = busy ? 'Answering …' : 'Ask'; }
    if (stop) stop.hidden = !busy;
}

// ── Drawing ──────────────────────────────────────────────────────────────

/* The turn being answered, from the run's events. */
function chatLiveTurn() {
    const run = CHAT.run;
    if (!run || run.done) return null;
    const turn = {role: 'assistant', text: '', rounds: [], thinking: [], steps: [], live: true};
    const steps = new Map();
    for (const e of [...run.events.values()].sort((a, b) => a.n - b.n)) {
        if (e.type === 'text') {
            turn.rounds[e.round] = (turn.rounds[e.round] || '') + e.text;
        } else if (e.type === 'thinking') {
            turn.thinking[e.round] = (turn.thinking[e.round] || '') + e.text;
        } else if (e.type === 'tool') {
            const st = {id: e.id, name: e.name, args: e.args, round: e.round, running: true};
            steps.set(e.id, st);
            turn.steps.push(st);
        } else if (e.type === 'tool_done') {
            const st = steps.get(e.id);
            if (st) Object.assign(st, {running: false, summary: e.summary, ms: e.ms, error: e.error});
            if (e.refs && CHAT.conv) CHAT.conv.refs = e.refs;
        } else if (e.type === 'error') {
            turn.error = e.message;
        }
    }
    turn.text = Array.from(turn.rounds, x => x || '').filter(Boolean).join('\n\n');
    return turn;
}

/* Draw the thread anew — a player that has started stays the same element:
 * the thread is redrawn on every poll of a running answer, and a new copy
 * would cut the passage off mid-sentence (innerHTML eats a running audio
 * element). Put back within the same task, it never stops. */
function chatSwap(el, html) {
    const keep = new Map();
    el.querySelectorAll('audio[data-key], video[data-key]').forEach(a => {
        if (!a.paused || a.currentTime > 0) keep.set(a.dataset.key, a);
    });
    el.innerHTML = html;
    if (!keep.size) return;
    el.querySelectorAll('audio[data-key], video[data-key]').forEach(a => {
        const old = keep.get(a.dataset.key);
        if (old) { a.replaceWith(old); keep.delete(a.dataset.key); }
    });
}

/* Replaces only the nodes from `first` on (the live turn) and keeps a
 * started player, as chatSwap does for the whole thread. */
function chatSwapTail(el, first, html) {
    const keep = new Map();
    const old = Array.from(el.children).slice(first);
    for (const n of old) {
        n.querySelectorAll('audio[data-key], video[data-key]').forEach(a => {
            if (!a.paused || a.currentTime > 0) keep.set(a.dataset.key, a);
        });
    }
    old.forEach(n => n.remove());
    if (html) el.insertAdjacentHTML('beforeend', html);
    if (!keep.size) return;
    Array.from(el.children).slice(first).forEach(n =>
        n.querySelectorAll('audio[data-key], video[data-key]').forEach(a => {
            const was = keep.get(a.dataset.key);
            if (was) { a.replaceWith(was); keep.delete(a.dataset.key); }
        }));
}

function chatRender(toBottom) {
    const el = document.getElementById('chatThread');
    if (!el) return;
    const nearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 120;
    const turns = (CHAT.conv && CHAT.conv.turns) || [];
    const live = chatLiveTurn();
    // What the finished turns look like depends on these only. While an
    // answer streams (a poll every 400 ms) they stay the same — then only
    // the live turn is redrawn: rebuilding the whole thread re-rendered the
    // markdown of every earlier answer 2.5× a second and dropped any text
    // selection in them.
    const doneKey = [turns.length, CHAT.hot, [...CHAT.opened].join('|')].join('\u0001');
    const same = CHAT.drawn && CHAT.drawn.conv === CHAT.conv && CHAT.drawn.key === doneKey
        && CHAT.drawn.refs === (CHAT.conv && CHAT.conv.refs) && CHAT.drawn.el === el
        && el.children.length >= turns.length;
    if (turns.length && same) {
        chatSwapTail(el, turns.length, live ? chatTurnHtml(live) : '');
        if (toBottom || nearBottom) el.scrollTop = el.scrollHeight;
        chatRenderSources();
        return;
    }
    CHAT.drawn = turns.length
        ? {conv: CHAT.conv, key: doneKey, refs: CHAT.conv && CHAT.conv.refs, el} : null;
    if (!turns.length && !live) {
        el.innerHTML = `<div class="sv-chat-welcome">
            <h3>Ask the library</h3>
            The assistant searches the transcripts itself — words, meaning, who speaks where, who talked with whom —
            and answers with the passages it found. Every quote can be played and opened in its
            transcript.
            <div class="sv-chat-examples">${CHAT_EXAMPLES.map(x =>
                `<button class="sv-chat-example" onclick="chatSend(this.textContent)">${esc(x)}</button>`).join('')}</div>
        </div>`;
    } else {
        chatSwap(el, turns.map(t => chatTurnHtml(t)).join('') + (live ? chatTurnHtml(live) : ''));
    }
    if (toBottom || nearBottom) el.scrollTop = el.scrollHeight;
    chatRenderSources();
}

/* An answer in the order it happened: per round its thinking and its text,
   then the tool calls that round made. An answer kept without rounds shows
   its calls first, as it was saved. */
function chatRounds(t) {
    const steps = t.steps || [];
    if (!t.rounds) return [{steps}, {text: t.text || ''}];
    const thinking = t.thinking || [];
    const last = Math.max(t.rounds.length, thinking.length,
                          ...steps.map(s => (s.round || 0) + 1)) - 1;
    const out = [];
    for (let r = 0; r <= last; r++) {
        out.push({thinking: thinking[r] || '', text: t.rounds[r] || '',
                  steps: steps.filter(s => (s.round || 0) === r)});
    }
    return out;
}

// A "Thinking" block stays open across the redraws of a running answer
function chatThinkToggle(key, summary) {
    if (summary.parentNode.open) CHAT.thinkOpen.delete(key); else CHAT.thinkOpen.add(key);
}

function chatTurnHtml(t) {
    if (t.role === 'user') {
        return `<div class="sv-msg user"><div class="sv-msg-who">You</div>
            <div class="sv-msg-body">${esc(t.text)}</div></div>`;
    }
    const rounds = chatRounds(t);
    // Live: the cursor sits where the model writes next — after its last
    // text, or under the last calls once they are back
    const busy = (t.steps || []).some(s => s.running);
    const tail = rounds[rounds.length - 1] || {};
    const cursorAfter = t.live && !busy && !t.error
        ? ((tail.steps || []).length ? 'calls' : 'text') : '';
    const body = rounds.map((p, i) => {
        const lastOne = i === rounds.length - 1;
        const cursor = lastOne && cursorAfter === 'text' ? '<span class="sv-cursor"></span>' : '';
        const tk = `${t.at || ''}:${i}`;
        return (p.thinking ? `<details class="sv-thinking"${CHAT.thinkOpen.has(tk) ? ' open' : ''}><summary
                    onclick="chatThinkToggle('${escJs(tk)}', this)">Thinking</summary><div>${esc(p.thinking)}</div></details>` : '')
            + (p.text || cursor ? `<div class="sv-msg-body">${p.text ? chatMd(p.text, t.live && lastOne) : ''}${cursor}</div>` : '')
            + ((p.steps || []).length ? `<div class="sv-steps">${p.steps.map(s => chatStepHtml(t, s)).join('')}</div>` : '');
    }).join('') + (cursorAfter === 'calls' ? '<div class="sv-msg-body"><span class="sv-cursor"></span></div>' : '');
    const waiting = t.live && !t.text && !(t.steps || []).length
        ? '<div class="sv-msg-note">Waiting for the model …</div>' : '';
    return `<div class="sv-msg assistant"><div class="sv-msg-who">Assistant</div>
        ${body}${waiting}
        ${t.error ? `<div class="sv-msg-error">${esc(t.error)}</div>` : ''}
        ${t.stopped ? '<div class="sv-msg-note">Stopped.</div>' : ''}</div>`;
}

/* One tool call: what it is, what it asked, what it found. */
function chatStepHtml(t, s) {
    const args = Object.entries(s.args || {}).map(([k, v]) =>
        `${k}: ${typeof v === 'string' ? v : JSON.stringify(v)}`).join(', ');
    const key = (t.at || 'live') + ':' + s.id;
    const open = CHAT.opened.has(key);
    return `<div class="sv-step${s.running ? ' run' : ''}${s.error ? ' err' : ''}" role="button" tabindex="0" data-keyclick
                 onclick="chatStepToggle('${escJs(key)}')" title="What it asked and found">
                <b>${esc(chatToolLabel(s.name))}</b>
                <span class="sv-step-args">${esc(args)}</span>
                ${s.running ? '' : `<span>· ${esc(s.summary || '')}${s.ms != null ? ` · ${(s.ms / 1000).toFixed(1)} s` : ''}</span>`}
            </div>${open ? `<div class="sv-step-detail">${esc(s.name)}(${esc(JSON.stringify(s.args || {}, null, 1))})\n→ ${esc(s.summary || '…')}</div>` : ''}${chatMomentsHtml(s)}${chatDupsHtml(s)}${chatConvsHtml(s)}`;
}

/* What find_moments found, as cards under its step: the still, the
 * language model's verdict, the reason — a click plays it like a [S3]
 * chip; "Show in Find" opens the whole run with its tiles (find.js). */
function chatMomentsHtml(s) {
    const ms = s.moments || [];
    if (!ms.length) return '';
    const refs = (CHAT.conv && CHAT.conv.refs) || {};
    const cards = ms.map(m => {
        const r = refs[m.ref];
        if (!r) return '';
        const pic = m.video && r.abs_path
            ? `<img loading="lazy" src="/api/find/frame?t=${Math.floor(Number(r.start) || 0)}&key=${encodeURIComponent(r.abs_path)}" alt="">`
            : `<span>${m.video ? '🎞' : '🎙'}</span>`;
        const v = m.verdict != null ? `<span class="fd-verdict ${m.verdict >= 6 ? 'yes' : m.verdict >= 4 ? 'maybe' : 'no'}">${Math.round(m.verdict)}</span>` : '';
        return `<div class="chat-moment" role="button" tabindex="0" data-keyclick onclick="chatRefPlay('${escJs(m.ref)}')"
                     title="${esc(r.abs_path || r.file)}">
            <span class="chat-moment-pic">${pic}</span>
            <span class="chat-moment-main"><b>${esc(m.ref)}</b> ${esc(svShortName(r.filename || r.file))} <span class="fd-best-at">${esc(svFmt(r.start))}</span>
                ${m.reason ? `<span class="chat-moment-reason">${esc(m.reason)}</span>` : ''}</span>
            ${v}
        </div>`;
    }).join('');
    const more = s.find_run && typeof fdShowRun === 'function'
        ? `<button class="vx-btn chat-moment-open" onclick="svTab('find'); fdShowRun('${escJs(s.find_run)}')" title="The whole run — every search's tile, the duplicates, the clips">★ Show in Find</button>` : '';
    return `<div class="chat-moments">${cards}${more}</div>`;
}

/* What find_duplicates found: a button that shows them drawn — the groups
 * with their waveforms, the folders that hold the same recordings
 * (audiodups.js). */
function chatDupsHtml(s) {
    const d = s.duplicates;
    if (!d || s.error || typeof amShowDuplicates !== 'function') return '';
    const kind = d.kind === 'contains' ? 'contains' : 'same';
    return `<div class="chat-moments"><button class="vx-btn chat-moment-open"
        onclick="amShowDuplicates({kind: '${kind}', q: '${escJs(d.q || '')}', minSec: ${Number(d.min_sec) || 0}, person: '${escJs(d.person || '')}'})"
        title="The groups with their waveforms, and the folders that hold the same recordings">≋ Show them in Same audio</button></div>`;
}

/* What find_conversations found: a button that shows them in the search
 * view's Conversations tab (conversations.js) — with the filters the
 * model searched with, by the API's parameter names. Strings go through
 * escJs, numbers through Number(): the model chose them. The step names
 * the speakers minimum it counted with (mcp_server.conversation_query:
 * two, none for a monologue) — the tab's own default when it is missing,
 * so nothing is added here. */
function chatConvsHtml(s) {
    const c = s.conversations;
    if (!c || typeof c !== 'object' || s.error || typeof cvShow !== 'function') return '';
    const f = c.filters && typeof c.filters === 'object' ? c.filters : {};
    const str = (v) => `'${escJs(String(v))}'`;
    const parts = [];
    const people = (Array.isArray(f.people) ? f.people : String(f.people || '').split(','))
        .map(x => String(x ?? '').trim()).filter(Boolean);
    if (people.length) parts.push(`people: [${people.map(str).join(', ')}]`);
    for (const k of ['match', 'kind', 'sentences', 'language', 'q', 'folder', 'date_from', 'date_to', 'sort', 'copies']) {
        const v = Array.isArray(f[k]) ? f[k].join(',') : f[k];
        if (v !== undefined && v !== null && v !== '') parts.push(`${k}: ${str(v)}`);
    }
    for (const k of ['min_sec', 'max_sec', 'wps_min', 'wps_max', 'rate_min', 'rate_max', 'speakers_min',
                     'speakers_max', 'questions_min']) {
        if (f[k] !== undefined && f[k] !== null && f[k] !== '' && Number.isFinite(Number(f[k]))) {
            parts.push(`${k}: ${Number(f[k])}`);
        }
    }
    return `<div class="chat-moments"><button class="vx-btn chat-moment-open"
        onclick="cvShow({${parts.join(', ')}})"
        title="Every one of them, with who took part, the sentence length and the speaker changes">🗣 Show them in Conversations</button></div>`;
}

const CHAT_TOOL_LABELS = {
    search: '🔎 searched', semantic_search: '🧠 searched by meaning', find_moments: '✨ found moments',
    find_duplicates: '≋ looked for duplicates',
    find_conversations: '🗣 looked for conversations', conversation_stats: '🗣 conversation statistics',
    person_stats: '👤 how a person talks',
    transcript: '📄 read',
    search_distribution: '📊 counted', search_plan: '📋 planned', list_files: '📁 listed files',
    file_info: 'ℹ️ looked at', library_stats: '📊 library', who_speaks: '🗣 who speaks',
    where_person_speaks: '🗣 where speaks', list_people: '👥 people', speaker_stats: '👥 voices',
    existing_summaries: '📝 summaries', existing_translation: '🌐 translation',
    rag_status: '🧠 index', queue_status: '⏳ queue', job_status: '⏳ job', incidents: '⚠ incidents',
};

function chatToolLabel(name) {
    return CHAT_TOOL_LABELS[name] || name;
}

function chatStepToggle(key) {
    if (CHAT.opened.has(key)) CHAT.opened.delete(key); else CHAT.opened.add(key);
    chatRender();
}

/* Players the model writes, as in Open WebUI: an <audio> tag around the
 * clip address of a tool result (whatever host it names), around a
 * reference — <audio>S12</audio> — or with a src. A reference closed by
 * </audio> is one even when the model garbled the opening tag
 * (e.g. <mudio>S1</audio>). */
const CHAT_AUDIO_RE = /<(?:audio\b[^>]*?(?:\bsrc\s*=\s*["']([^"']+)["'][^>]*)?|[a-z]{2,8}(?=>\s*\[?S\d+\]?\s*<\/audio>))>([\s\S]*?)<\/audio>/gi;
const CHAT_REFS_RE = /\[((?:S\d+)(?:\s*[,;]\s*S\d+)*)\]/g;
const CHAT_LEAD_IN = 2;        // a player starts a little before the moment …
const CHAT_CLIP_SEC = 30;      // … and lasts as long as the MCP server's samples

/* A player or passage no tool result named: the model wrote the moment
 * itself — it may be right, the player lets you hear whether it is. A model
 * that answers without looking anything up can invent dozens of "hits"
 * with times and quotes of its own. */
const CHAT_UNSURE = '<span class="sv-unsure" title="No tool result named this moment — '
    + 'the model wrote it itself. Listen before you trust it.">not from a search</span>';

/* What a player block means: a reference, or a clip of a file. A clip the
 * conversation knows as a passage becomes that passage — it knows the
 * recording's root. */
function chatAudioWhat(src, inner) {
    const text = String(src || inner || '').trim();
    const named = text.match(/^\[?(S\d+)\]?$/);
    if (named) return {ref: named[1]};
    const url = (text.match(/<source[^>]*\bsrc\s*=\s*["']([^"']+)/i) || [])[1]
        || (text.match(/\S+/) || [])[0] || '';
    let u;
    try { u = new URL(url.replace(/&amp;/g, '&'), location.origin); } catch (e) { return null; }
    if (!/\/api\/clip\/audio$/.test(u.pathname)) return null;
    const file = u.searchParams.get('file') || u.searchParams.get('filename') || '';
    const start = Number(u.searchParams.get('start'));
    const end = Number(u.searchParams.get('end'));
    if (!file || !Number.isFinite(start)) return null;
    const ref = chatRefAt(file, start + CHAT_LEAD_IN) || chatRefAt(file, start);
    if (ref) return {ref};
    return {file, start, end: Number.isFinite(end) && end > start ? end : start + CHAT_CLIP_SEC,
            key: `${file}@${start}`};
}

/* The passage of the conversation at this file and moment. */
function chatRefAt(file, start) {
    const refs = (CHAT.conv && CHAT.conv.refs) || {};
    let best = '', gap = 2.5;
    for (const [k, r] of Object.entries(refs)) {
        if (r.file !== file && r.filename !== file && r.abs_path !== file) continue;
        const d = Math.abs((Number(r.start) || 0) - start);
        if (d < gap) { best = k; gap = d; }
    }
    return best;
}

/* The clip address of a passage — its recording by full path, so it plays
 * whichever folder is open. */
function chatClipUrl(file, start, end) {
    return `/api/clip/audio?file=${encodeURIComponent(file)}`
        + `&start=${Math.max(0, start).toFixed(1)}&end=${end.toFixed(1)}`;
}

function chatPlayerHtml(a) {
    const refs = (CHAT.conv && CHAT.conv.refs) || {};
    const r = a.ref ? refs[a.ref] : null;
    if (a.ref && !r) return `<span class="sv-ref" title="Not among the passages">${esc(a.ref)}</span>`;
    const file = r ? (r.abs_path || r.filename || r.file) : a.file;
    const from = r ? Math.max(0, (Number(r.start) || 0) - CHAT_LEAD_IN) : a.start;
    const to = r ? from + CHAT_CLIP_SEC : a.end;
    const url = chatClipUrl(file, from, to);
    const name = svShortName(r ? (r.filename || r.file) : a.file);
    const at = r ? Number(r.start) || 0 : a.start;
    const open = r ? `chatRefOpen('${a.ref}')` : `chatClipOpen('${escJs(a.file)}', ${a.start})`;
    return `<div class="sv-audio${r && CHAT.hot === a.ref ? ' hot' : ''}${r ? '' : ' unsure'}"${r ? ` data-ref="${esc(a.ref)}"` : ''}>
        <div class="sv-audio-head">
            ${r ? `<b>${esc(a.ref)}</b>` : '<b>♪</b>'}
            <span class="sv-audio-file" title="${esc(file)}">${esc(name)}</span>
            ${r ? '' : CHAT_UNSURE}
            <span class="sv-audio-at">${esc(svFmt(at))}</span>
            ${r && chatSpeaker(r) ? `<span class="sv-audio-sp">${esc(chatSpeaker(r))}</span>` : ''}
            <button onclick="${open}" title="Open this moment in its transcript">↗ transcript</button>
        </div>
        ${r && r.text ? `<div class="sv-audio-text">${esc(r.text)}</div>` : ''}
        ${chatVideoOf(r) ? chatVideoHtml(r)
            : `<audio controls preload="none" data-key="${esc(url)}" src="${esc(url)}"></audio>`}
    </div>`;
}

/* A passage of a video plays as video, not as a sound-only player. Find's
 * clip and still: cut on the first play, the still before. */
const CHAT_VIDEO_RE = /\.(mp4|m4v|mkv|webm|mov|avi|wmv|flv|mpe?g|ts|3gp)$/i;

function chatVideoOf(r) {
    return !!(r && r.abs_path && CHAT_VIDEO_RE.test(r.abs_path));
}

function chatVideoHtml(r) {
    const t = Math.max(0, Number(r.start) || 0);
    const key = encodeURIComponent(r.abs_path);
    const end = Number(r.end) > t ? `&end=${Number(r.end).toFixed(1)}` : '';
    const clip = `/api/find/clip?key=${key}&t=${t.toFixed(1)}${end}`;
    return `<video class="sv-audio-video" controls preload="none" data-key="${esc(clip)}"
                   poster="/api/find/frame?key=${key}&t=${Math.floor(t)}" src="${esc(clip)}"></video>`;
}

/* Markdown; then every [S12] — also [S1, S2] — as a chip that plays, and
 * every player block as a player. While an answer is being written, a
 * player tag that isn't closed yet waits out of sight. */
function chatMd(text, live) {
    let md = String(text || '');
    if (live) md = md.replace(/<audio\b(?![\s\S]*<\/audio>)[\s\S]*$/i, '');
    const players = [];
    md = md.replace(CHAT_AUDIO_RE, (m, src, inner) => {
        const a = chatAudioWhat(src, inner);
        if (!a) return m;
        players.push(a);
        return `\n\n\x00AUD${players.length - 1}\x00\n\n`;
    });
    let html = typeof _renderMarkdown === 'function' ? _renderMarkdown(md) : esc(md);
    const refs = (CHAT.conv && CHAT.conv.refs) || {};
    html = html.replace(CHAT_REFS_RE, (m, inner) =>
        inner.split(/\s*[,;]\s*/).map(k => {
            const r = refs[k];
            if (!r) return `<span class="sv-ref" title="Not among the passages">${esc(k)}</span>`;
            return `<button class="sv-ref${CHAT.hot === k ? ' hot' : ''}" data-ref="${esc(k)}"
                        onclick="chatRefPlay('${k}')" ondblclick="chatRefOpen('${k}')"
                        title="${esc(chatRefTitle(r))} — click plays, double-click opens the transcript">${esc(k)} ▶ ${esc(svFmt(r.start))}</button>`;
        }).join(''));
    return chatFoldRefs(html).replace(/(?:<p[^>]*>\s*)?\x00AUD(\d+)\x00(?:\s*<\/p>)?/g,
        (m, i) => chatPlayerHtml(players[Number(i)]));
}

/* A long run of passage buttons folded after its first few: "[S2][S3] …
 * [S24]" would make three lines of buttons under one sentence. The rest
 * opens with a click; a run holding the passage that plays stays open. */
const CHAT_REF_RUN_RE = /(?:<button class="sv-ref[^>]*>[^<]*<\/button>\s*){7,}/g;
const CHAT_REF_RUN_SHOWN = 4;

function chatFoldRefs(html) {
    return html.replace(CHAT_REF_RUN_RE, run => {
        const chips = run.match(/<button class="sv-ref[^>]*>[^<]*<\/button>/g);
        const open = /class="sv-ref hot"/.test(run) ? ' open' : '';
        const rest = chips.length - CHAT_REF_RUN_SHOWN;
        return `<span class="sv-ref-run${open}">${chips.slice(0, CHAT_REF_RUN_SHOWN).join('')}`
            + `<span class="sv-ref-more">${chips.slice(CHAT_REF_RUN_SHOWN).join('')}</span>`
            + `<button class="sv-ref-fold" onclick="this.parentNode.classList.add('open')"
                       title="Show the other ${rest} passages">+${rest} more</button></span>`;
    });
}

function chatRefTitle(r) {
    const sp = chatSpeaker(r);
    return `${svShortName(r.filename || r.file)} · ${svFmt(r.start)}${sp ? ' · ' + sp : ''}`;
}

/* A passage's speaker — "—" is a transcript without speaker labels. */
function chatSpeaker(r) {
    const sp = String((r && r.speaker) || '').trim();
    return sp === '—' || sp === '-' ? '' : sp;
}

/* What the answers present, in the order they present it: the passages
 * they quote and the players they show — each once. */
function chatPresented() {
    const refs = (CHAT.conv && CHAT.conv.refs) || {};
    const texts = ((CHAT.conv && CHAT.conv.turns) || [])
        .filter(t => t.role === 'assistant').map(t => t.text || '');
    const live = chatLiveTurn();
    if (live) texts.push(live.text);
    const out = [], seen = new Set();
    const both = new RegExp(`${CHAT_AUDIO_RE.source}|${CHAT_REFS_RE.source}`, 'gi');
    const take = (key, item) => { if (!seen.has(key)) { seen.add(key); out.push(item); } };
    for (const t of texts) {
        for (const m of t.matchAll(both)) {
            if (m[3]) {
                for (const k of m[3].split(/\s*[,;]\s*/)) if (refs[k]) take(k, {ref: k});
                continue;
            }
            const a = chatAudioWhat(m[1], m[2]);
            if (a && a.ref) { if (refs[a.ref]) take(a.ref, {ref: a.ref}); }
            else if (a) take(a.key, a);
        }
    }
    return out;
}

/* The passages the conversation presented, in the order it presented them. */
function chatRenderSources() {
    const el = document.getElementById('chatSources');
    if (!el) return;
    const refs = (CHAT.conv && CHAT.conv.refs) || {};
    const shown = chatPresented();
    const found = Object.keys(refs).length;
    if (!shown.length) {
        el.innerHTML = `<div class="sv-src-title">Passages</div><div class="sv-src-empty">${found
            ? `${plural(found, 'passage')} found so far — the ones the answer quotes or plays appear here.`
            : 'The passages an answer quotes or plays appear here — each one plays and opens in its transcript.'}</div>`;
        return;
    }
    el.innerHTML = `<div class="sv-src-title">Passages in the answers · ${shown.length}</div>` + shown.map(a => {
        if (!a.ref) {
            return `<div class="sv-src unsure" onclick="chatClipPlay('${escJs(a.file)}', ${a.start})">
                <div class="sv-src-head"><b>♪</b><span class="sv-src-file" title="${esc(a.file)}">${esc(svShortName(a.file))}</span>
                    <span>${esc(svFmt(a.start))}</span></div>
                <div class="sv-src-text">${CHAT_UNSURE}</div>
                <div class="sv-src-acts">
                    <button onclick="event.stopPropagation();chatClipPlay('${escJs(a.file)}', ${a.start})">▶ play</button>
                    <button onclick="event.stopPropagation();chatClipOpen('${escJs(a.file)}', ${a.start})">↗ transcript</button>
                </div></div>`;
        }
        const k = a.ref, r = refs[k];
        return `<div class="sv-src${CHAT.hot === k ? ' hot' : ''}" id="chatSrc-${esc(k)}" onclick="chatRefPlay('${k}')">
            <div class="sv-src-head"><b>${esc(k)}</b><span class="sv-src-file" title="${esc(r.abs_path || r.file)}">${esc(svShortName(r.filename || r.file))}</span>
                <span>${esc(svFmt(r.start))}</span></div>
            <div class="sv-src-text">${esc(r.text || '')}</div>
            <div class="sv-src-acts">
                <button onclick="event.stopPropagation();chatRefPlay('${k}')">▶ play</button>
                <button onclick="event.stopPropagation();chatRefOpen('${k}')">↗ transcript</button>
                ${r.abs_path ? `<button onclick="event.stopPropagation();mapForFile('${escJs(r.abs_path)}')">🗣 voices</button>` : ''}
            </div></div>`;
    }).join('');
}

function chatRef(k) {
    const r = ((CHAT.conv && CHAT.conv.refs) || {})[k];
    if (!r) return null;
    return {file: r.filename || r.file, start: Number(r.start) || 0, dir: r.file_dir || ''};
}

function chatRefPlay(k) {
    const r = chatRef(k);
    if (!r) return;
    CHAT.hot = k;
    svPlay(r.file, r.start, r.dir);
    document.querySelectorAll('#svPaneChat .sv-ref, #svPaneChat .sv-audio').forEach(b =>
        b.classList.toggle('hot', b.dataset.ref === k));
    document.querySelectorAll('#chatSources .sv-src').forEach(b => b.classList.toggle('hot', b.id === 'chatSrc-' + k));
    document.getElementById('chatSrc-' + k)?.scrollIntoView({block: 'nearest'});
}

function chatRefOpen(k) {
    const r = chatRef(k);
    if (r) svGoto(r.file, r.start, r.dir);
}

/* A clip that isn't among the passages: a full path splits into its
 * folder and name, a relative one is the open folder's. */
function chatClipWhere(file) {
    const f = String(file || '');
    // Absolute: "/…", or a Windows drive or share ("C:/…", "\\server\…")
    const abs = /^(\/|[A-Za-z]:[\\/]|\\\\)/.test(f);
    const cut = Math.max(f.lastIndexOf('/'), f.lastIndexOf('\\'));
    if (!abs || cut <= 0) return {file: f, dir: ''};
    let dir = f.slice(0, cut);
    if (/^[A-Za-z]:$/.test(dir)) dir += '/';     // "C:" alone is no folder
    return {file: f.slice(cut + 1), dir};
}

function chatClipPlay(file, start) {
    const w = chatClipWhere(file);
    svPlay(w.file, Number(start) + CHAT_LEAD_IN, w.dir);
}

function chatClipOpen(file, start) {
    const w = chatClipWhere(file);
    svGoto(w.file, Number(start) + CHAT_LEAD_IN, w.dir);
}
