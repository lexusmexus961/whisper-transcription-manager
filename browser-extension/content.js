/* ══════════════════════════════════════════════════════════════════════════
 *  On the page: select YouTube videos, send them to Whisper Transcription Manager
 *
 *  The main video can be picked right where it plays, without opening the
 *  popup. So, on YouTube:
 *
 *    · the video a watch page plays gets a button in YouTube's own action
 *      row, next to Share — ✦ Whisper TM ▾ with Summarize, Transcribe, Add to
 *      selection. It shows the progress itself, and the summary unfolds in
 *      a panel under the title;
 *    · every other video — search results, the home page, the suggestions
 *      beside the player, a playlist — gets a small ⊕ behind its title: one
 *      click selects it (✓), a green ring says it is in the library already.
 *
 *  Elsewhere: a ⊕ behind every link to a YouTube video, and a badge on an
 *  embedded player when the mouse is over it. With videos selected, a bar at
 *  the bottom left offers what to do with them — summarize, transcribe,
 *  analyze together — and then shows where each one stands.
 *
 *  To read and ask, and to choose exactly how: the side panel
 *  (sidepanel.js) — "Options…" and "Read & ask" open it with this video,
 *  and with "ask before sending" on (its ⚙) Summarize and Transcribe open
 *  its form instead of sending. A time clicked there plays the video here.
 *
 *  Our elements carry closed shadow roots, so the page's styles and ours never
 *  meet — and the page's scripts can't reach in: not read which of its videos
 *  are in the library, not press our buttons (clicks a script makes are
 *  ignored too). The server is spoken to through background.js only.
 *
 *  Tests and the live test harness: with ``globalThis.WTM_TEST = true`` set in
 *  this script's own world before it runs (a web page can't — it lives in
 *  another world), ``globalThis.__wxpTest`` reaches the closed shadow roots
 *  (``shadow(el)``, ``root()``) and the state, and clicks made by a script
 *  count. Without the flag neither exists.
 * ══════════════════════════════════════════════════════════════════════════ */

(() => {
    if (window.top !== window) return;
    // A content script from before the extension was reloaded stays in the
    // page, cut off from it: a new one takes over, the old one steps aside
    const before = globalThis.__wxpContent;
    if (before && before.alive()) return;
    if (before) before.retire();
    let retired = false;

    const ON_YOUTUBE = /(^|\.)youtube\.com$/.test(location.hostname);
    const TEST = globalThis.WTM_TEST === true;

    /* Closed shadow roots, kept here — ``el.shadowRoot`` is null for them. */
    const SHADOWS = new WeakMap();
    function shadow(el) {
        const sh = el.attachShadow({mode: 'closed'});
        SHADOWS.set(el, sh);
        return sh;
    }
    /* A click by the user — not one a script made. Asked only where a click
     * sends to the server or opens something: a script's click there is let
     * through untouched (not stopped, not swallowed). Selecting, unfolding,
     * the menu — keyboard helpers (Vimium, Surfingkeys) click with scripts,
     * and those work. */
    const real = e => !!(e && (e.isTrusted || TEST));

    const S = {
        videos: new Map(),        // key → {key, kind, id, url, title, els}
        byEl: new WeakMap(),      // element → key
        selected: new Map(),      // key → {key, kind, url, title}
        known: new Map(),         // key → {state, open} — in the library already
        asked: new Set(),         // keys whose library state was asked
        hover: null,              // {key, el} — the badge on an embedded player
        hideTimer: 0,
        batch: null,              // the batch the bar follows
        pollTimer: 0,
        mode: 'select',           // select · analyze · batch
        open: new Set(),          // items whose summary is unfolded
        listOpen: false,          // the bar lists the selection, not just its pictures
        titles: new Map(),        // url → the page's title, until the server knows it
        error: '',
    };

    // "Ask before sending" (the side panel's ⚙): Summarize and Transcribe open
    // its form instead. "Library marks on other sites": off YouTube, the ✓ of
    // videos in the library needs their links sent to the server — the
    // user's choice, not the default (every YouTube link on every page went
    // out, over the network and often as plain http)
    const PREFS = {askFirst: false, marksEverywhere: false};
    try {
        chrome.storage.sync.get(['askFirst', 'marksEverywhere']).then(d => {
            PREFS.askFirst = !!d.askFirst;
            PREFS.marksEverywhere = !!d.marksEverywhere;
            if (PREFS.marksEverywhere) askKnown();
        }, () => {});
        chrome.storage.onChanged.addListener((changes, area) => {
            if (area !== 'sync') return;
            if (changes.askFirst) PREFS.askFirst = !!changes.askFirst.newValue;
            if (changes.marksEverywhere) {
                PREFS.marksEverywhere = !!changes.marksEverywhere.newValue;
                if (PREFS.marksEverywhere) askKnown();
            }
        });
    } catch (e) { /* the extension was reloaded */ }

    // The video a watch page plays
    const M = {key: '', url: '', title: '', state: '', progress: 0, summary: '', open: '',
               batchId: '', timer: 0, menu: false, panel: false, error: ''};

    const STATE = {
        waiting: ['waiting', ''], resolving: ['looking up', ''], downloading: ['downloading', ''],
        queued: ['queued', ''], transcribing: ['transcribing', 'run'],
        transcribed: ['transcribed', 'ok'], summarizing: ['summarizing', 'run'],
        summarized: ['summarized', 'ok'], 'not queued': ['not queued', 'warn'],
        error: ['failed', 'err'], downloaded: ['downloaded', ''],
    };

    /* Markup set through a Trusted Types policy where the page enforces them
     * (YouTube does) — harmless where it doesn't. */
    let policy = null;
    function setHTML(el, html) {
        if (policy === null) {
            policy = false;
            try {
                if (window.trustedTypes && window.trustedTypes.createPolicy) {
                    policy = window.trustedTypes.createPolicy('whisper-tm', {createHTML: s => s});
                }
            } catch (e) { /* the page allows no new policy — plain markup then */ }
        }
        el.innerHTML = policy ? policy.createHTML(html) : html;
    }

    const esc = s => String(s ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;')
        .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
    const send = msg => new Promise(resolve => {
        try {
            chrome.runtime.sendMessage(msg, r => resolve(chrome.runtime.lastError
                ? {error: chrome.runtime.lastError.message} : (r || {})));
        } catch (e) {
            resolve({error: String(e.message || e)});   // the extension was reloaded
        }
    });
    /* Dark or light behind an element: the first background painted under it
     * decides — YouTube's theme, a dark blog on a light system, and so on. On
     * YouTube that is its app's background, whatever attributes say. */
    function dark(el) {
        if (ON_YOUTUBE) el = document.querySelector('ytd-app') || el;
        for (let e = el; e && e.nodeType === 1; e = e.parentElement || e.getRootNode().host || null) {
            const m = getComputedStyle(e).backgroundColor.match(/[\d.]+/g);
            if (m && m.length >= 3 && (m.length < 4 || Number(m[3]) >= 0.5)) {
                return 0.2126 * m[0] + 0.7152 * m[1] + 0.0722 * m[2] < 128;
            }
        }
        return matchMedia('(prefers-color-scheme: dark)').matches;
    }

    /* A summary's Markdown, the little of it a small box needs: headings,
     * bold, list items. Everything escaped first. */
    function md(text) {
        return esc(text).split('\n').map(line => {
            const h = line.match(/^#{1,6}\s+(.*)$/);
            if (h) return `<b>${h[1]}</b>`;
            return line.replace(/^\s*[-*]\s+/, '• ').replace(/\*\*(.+?)\*\*/g, '<b>$1</b>');
        }).join('\n');
    }

    function stateText(s) {
        return STATE[s] ? STATE[s][0] : s;
    }

    // ══ The overlay: a badge on embedded players, the bar ══════════════════

    const CSS = `
:host { all: initial; }
* { box-sizing: border-box; font-family: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }
/* The hidden attribute against the display rules below — without this, an
   empty badge sat in the top left corner of every page */
[hidden] { display: none !important; }
.badge {
    position: fixed; pointer-events: auto; display: inline-flex; align-items: center; gap: 5px;
    padding: 3px 9px 3px 7px; border-radius: 999px; border: 1px solid rgba(255,255,255,.35);
    background: rgba(17, 24, 39, .88); color: #fff; font-size: 12px; font-weight: 600;
    cursor: pointer; box-shadow: 0 2px 10px rgba(0,0,0,.35); line-height: 18px; user-select: none;
}
.badge:hover, .badge.on { background: #2563eb; border-color: #bfdbfe; }
.badge .k { font-size: 10.5px; font-weight: 500; padding: 0 6px; border-radius: 999px;
            background: rgba(34,197,94,.25); color: #bbf7d0; }
.mark { position: fixed; pointer-events: auto; width: 24px; height: 24px; border-radius: 50%;
        background: #2563eb; color: #fff; border: 2px solid #fff; font-size: 13px; font-weight: 700;
        display: flex; align-items: center; justify-content: center; cursor: pointer;
        box-shadow: 0 2px 8px rgba(0,0,0,.4); }
/* Bottom left: on YouTube the right side holds the suggestions and the
   miniplayer, elsewhere chat buttons */
.bar {
    position: fixed; left: 18px; bottom: 18px; width: 380px; max-height: 70vh; overflow: auto;
    pointer-events: auto; background: #111827; color: #e5e7eb; border: 1px solid #374151;
    border-radius: 14px; box-shadow: 0 16px 48px rgba(0,0,0,.5); font-size: 13px; line-height: 1.45;
}
.head { display: flex; align-items: center; gap: 8px; padding: 10px 12px; border-bottom: 1px solid #1f2937;
        position: sticky; top: 0; background: #111827; }
.logo { width: 18px; height: 18px; border-radius: 5px; background: #3b82f6; display: inline-flex; flex: none;
        align-items: center; justify-content: center; color: #fff; font-size: 11px; font-weight: 800; }
.title { font-weight: 700; flex: 1; }
.x { background: none; border: 0; color: #9ca3af; cursor: pointer; font-size: 15px; padding: 2px 4px; }
.x:hover { color: #fff; }
.body { padding: 10px 12px 12px; }
.acts { display: flex; flex-wrap: wrap; gap: 6px; }
button.b {
    font: inherit; font-weight: 600; font-size: 12.5px; padding: 6px 11px; border-radius: 8px; cursor: pointer;
    border: 1px solid #374151; background: #1f2937; color: #e5e7eb;
}
button.b:hover { border-color: #3b82f6; color: #fff; }
button.b.p { background: #2563eb; border-color: #2563eb; color: #fff; }
button.b.p:hover { filter: brightness(1.12); }
button.b:disabled { opacity: .5; cursor: default; }
.list { margin: 8px 0 0; display: flex; flex-direction: column; gap: 4px; }
.it { display: flex; align-items: center; gap: 8px; padding: 5px 0; border-top: 1px solid #1f2937; }
.it:first-child { border-top: 0; }
.it .t { flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: #d1d5db; }
.it img { width: 48px; height: 27px; object-fit: cover; border-radius: 4px; flex: none; background: #1f2937; }
.chip { font-size: 11px; padding: 1px 8px; border-radius: 999px; background: #1f2937; color: #9ca3af;
        border: 1px solid #374151; white-space: nowrap; }
.chip.run { color: #93c5fd; border-color: #1d4ed8; }
.chip.ok { color: #86efac; border-color: #166534; }
.chip.err { color: #fca5a5; border-color: #991b1b; }
.chip.warn { color: #fcd34d; border-color: #92400e; }
button.l { color: #93c5fd; cursor: pointer; background: none; border: 0; padding: 0; font: inherit;
           font-size: 12px; white-space: nowrap; }
button.l:hover { text-decoration: underline; }
.sum { margin: 2px 0 6px; padding: 8px 10px; border-radius: 8px; background: #0b1220; border: 1px solid #1f2937;
       color: #e5e7eb; white-space: pre-wrap; font-size: 12.5px; max-height: 240px; overflow: auto; }
textarea { width: 100%; min-height: 80px; resize: vertical; margin: 8px 0; padding: 8px 10px; border-radius: 8px;
           border: 1px solid #374151; background: #0b1220; color: #e5e7eb; font: inherit; font-size: 13px; outline: none; }
textarea:focus { border-color: #3b82f6; }
.note { color: #9ca3af; font-size: 12px; margin-top: 6px; }
.err { color: #fca5a5; font-size: 12px; margin-top: 6px; }
.an { margin-top: 8px; padding: 8px 10px; border-radius: 8px; border: 1px solid #1f2937; background: #0b1220;
      display: flex; align-items: center; gap: 8px; }
/* The selection as a row of pictures — the bar keeps its height */
.strip { display: flex; align-items: center; gap: 4px; margin-bottom: 8px; cursor: pointer; }
.strip img, .strip .ph { width: 48px; height: 27px; object-fit: cover; border-radius: 4px; flex: none; background: #1f2937; }
.strip .ph { display: inline-flex; align-items: center; justify-content: center; color: #9ca3af; font-size: 13px; }
.strip .more { color: #9ca3af; font-size: 12px; margin-left: 4px; white-space: nowrap; }
.head button.l { margin-right: 4px; }
`;

    const host = document.createElement('wtm-root');
    host.style.cssText = 'all: initial; position: fixed; inset: 0; pointer-events: none; z-index: 2147483646;';
    const root = shadow(host);
    setHTML(root, `<style>${CSS}</style><button class="badge" hidden></button><div class="marks"></div><div class="bar" hidden></div>`);
    (document.documentElement || document.body).appendChild(host);
    const badge = root.querySelector('.badge');
    const marks = root.querySelector('.marks');
    const bar = root.querySelector('.bar');
    globalThis.__wxpContent = {
        alive: () => { try { return !!(chrome.runtime && chrome.runtime.id); } catch (e) { return false; } },
        retire: () => {
            retired = true;
            host.remove();
            document.querySelectorAll('wtm-pick, wtm-action, wtm-panel, wtm-menu').forEach(e => e.remove());
        },
    };

    // ══ Our elements in the page: ⊕ behind a title, the action button, the panel

    /* Our elements inside the page start from `all: initial` — but take the
     * page's visibility along: `initial` is visible, and a ⊕ behind a link in
     * a menu hidden with visibility (a web app's dropdown menu) stood alone
     * in the page, where it was clicked. */
    /* What a ⊕ shows sits on its button, inside the closed root — never on
     * the element in the page, where the page could read it. */
    const PICK_CSS = `
:host { all: initial; visibility: inherit; display: inline-block; line-height: 0; }
button {
    all: unset; box-sizing: border-box; width: 22px; height: 22px; border-radius: 50%; cursor: pointer;
    display: inline-flex; align-items: center; justify-content: center;
    font: 700 14px/1 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
    color: var(--fg); background: var(--bg); border: 1.5px solid var(--line);
    transition: background .12s, color .12s, border-color .12s;
}
button.dark { --fg: #f1f1f1; --bg: rgba(255,255,255,.08); --line: rgba(255,255,255,.28); }
button.light { --fg: #0f0f0f; --bg: rgba(0,0,0,.04); --line: rgba(0,0,0,.25); }
button:hover { background: #2563eb; border-color: #2563eb; color: #fff; }
button.known { border-color: #22c55e; color: #22c55e; }
button.sel { background: #2563eb; border-color: #2563eb; color: #fff; }
`;

    const pickKey = new WeakMap();        // ⊕ → the key of its video
    let pickTheme = new WeakMap();        // ⊕ → dark · light, read once a round (off YouTube)
    let pageTheme = '';                   // on YouTube: its app's, read once a round
    const skipped = new Set();            // video links without a ⊕ because they are hidden — shown later, they get one

    /* The ⊕ behind a video's title (or a link elsewhere): one click selects it. */
    function makePick() {
        const el = document.createElement('wtm-pick');
        const sh = shadow(el);
        setHTML(sh, `<style>${PICK_CSS}</style><button aria-label="Select for Whisper Transcription Manager">+</button>`);
        const b = sh.querySelector('button');
        for (const t of ['mousedown', 'pointerdown', 'touchstart']) b.addEventListener(t, e => e.stopPropagation());
        b.addEventListener('click', e => {
            e.preventDefault();
            e.stopPropagation();
            if (pickKey.get(el)) toggle(pickKey.get(el));           // selecting only — sends nothing
        });
        return el;
    }

    function themeOf(el) {
        if (ON_YOUTUBE) return pageTheme || (pageTheme = dark(el) ? 'dark' : 'light');
        if (!pickTheme.has(el)) pickTheme.set(el, dark(el) ? 'dark' : 'light');
        return pickTheme.get(el);
    }

    /* Only what changed is written — a round that changes nothing changes
     * nothing in the page. */
    function paintPick(el) {
        const key = pickKey.get(el);
        const sh = SHADOWS.get(el);
        if (!key || !sh) return;
        const on = S.selected.has(key);
        const k = S.known.get(key);
        const b = sh.querySelector('button');
        const cls = [themeOf(el), on ? 'sel' : '', k && !on ? 'known' : ''].filter(Boolean).join(' ');
        const text = on || k ? '✓' : '+';
        const title = on ? 'Selected for Whisper Transcription Manager — click to unselect'
            : k ? `In Whisper Transcription Manager: ${stateText(k.state)} — click to select it anyway`
            : 'Select for Whisper Transcription Manager';
        if (b.className !== cls) b.className = cls;
        if (b.textContent !== text) b.textContent = text;
        if (b.title !== title) b.title = title;
    }

    /* A link the user edits (a mail being written, an editor): nothing of
     * ours goes in there — it became part of the text, and editors that
     * take foreign elements out again fought the next round. */
    function editable(el) {
        return !!(el.isContentEditable
            || (el.closest && el.closest('[contenteditable]:not([contenteditable="false"]), textarea, [role="textbox"]')));
    }

    /* A link the page shows — not one hidden with display: none, nor one
     * only for screen readers (clipped to a pixel). */
    function shown(el) {
        if (!el.getClientRects || !el.getClientRects().length) return false;
        const r = el.getBoundingClientRect();
        return r.width >= 2 && r.height >= 2;
    }

    /* The cards YouTube draws: a ⊕ as the first child of the title's heading,
     * floated to its right (a flex row puts it last) — outside the link, so a
     * click selects instead of opening the video. Cards are reused while
     * scrolling, so every round puts the right video's key on it. */
    function placePicks() {
        pageTheme = '';
        pickTheme = new WeakMap();
        skipped.clear();
        const cards = new Map();                          // card → key
        const links = [];                                 // [link, key] — read first, written after
        for (const v of S.videos.values()) {
            if (ON_YOUTUBE && v.key === M.key) continue;   // the watch page's own: the action button
            for (const el of v.els) {
                const card = ON_YOUTUBE && el.closest ? el.closest(WTM.CARDS) : null;
                if (card) {
                    if (!cards.has(card)) cards.set(card, v.key);
                } else if (!ON_YOUTUBE && el.tagName === 'A' && (el.textContent || '').trim() && !editable(el)) {
                    if (shown(el)) links.push([el, v.key]);
                    else skipped.add(el);
                }
            }
        }
        const placed = new Set();
        for (const [el, key] of links) {
            // A text link elsewhere: the ⊕ right behind it
            let p = el.nextElementSibling;
            if (!p || p.tagName !== 'WTM-PICK') {
                p = makePick();
                p.style.cssText = 'margin-left:4px;vertical-align:middle;';
                el.after(p);
            }
            pickKey.set(p, key);
            placed.add(p);
            paintPick(p);
        }
        for (const [card, key] of cards) {
            const h = card.querySelector('h3') || card.querySelector('#video-title');
            if (!h) continue;
            let p = h.querySelector(':scope > wtm-pick');
            if (!p) {
                p = makePick();
                p.style.cssText = 'float:right;order:99;flex:none;margin:2px 0 2px 8px;position:relative;z-index:2;';
                h.insertBefore(p, h.firstChild);
            }
            pickKey.set(p, key);
            placed.add(p);
            paintPick(p);
        }
        // A ⊕ whose link went away, changed or hid — or one of ours from
        // before a reload — goes too
        for (const p of document.querySelectorAll('wtm-pick')) if (!placed.has(p)) p.remove();
        watch();
    }

    function paintAllPicks() {
        document.querySelectorAll('wtm-pick').forEach(paintPick);
    }

    // ── The video a watch page plays ─────────────────────────────────────────

    const ACTION_CSS = `
:host { all: initial; visibility: inherit; display: inline-flex; position: relative; }
* { box-sizing: border-box; font-family: "Roboto", "Arial", system-ui, sans-serif; }
.main.dark { --fg: #f1f1f1; --bg: rgba(255,255,255,.1); --bg-h: rgba(255,255,255,.2);
             --line: rgba(255,255,255,.12); --div: rgba(255,255,255,.2); }
.main.light { --fg: #0f0f0f; --bg: rgba(0,0,0,.05); --bg-h: rgba(0,0,0,.1);
              --line: rgba(0,0,0,.1); --div: rgba(0,0,0,.1); }
.main {
    all: unset; display: inline-flex; align-items: center; gap: 6px; height: 36px; padding: 0 14px;
    border-radius: 18px; background: var(--bg); color: var(--fg); cursor: pointer; white-space: nowrap;
    font-size: 14px; font-weight: 500; line-height: 36px; position: relative;
}
.main:hover { background: var(--bg-h); }
.main .ic { color: #3b82f6; font-size: 15px; }
.main.ok .ic { color: #22c55e; }
.main .car { opacity: .7; font-size: 11px; }
/* Summarized: the button unfolds the summary, ▾ behind a divider opens the menu */
.main.split .car { opacity: .85; display: inline-flex; align-items: center; height: 24px;
                            padding: 0 2px 0 9px; margin-left: 3px; border-left: 1px solid var(--div); }
.prog { position: absolute; left: 14px; right: 14px; bottom: 5px; height: 2px; border-radius: 2px;
        background: var(--line); overflow: hidden; pointer-events: none; }
.prog div { height: 100%; background: #3b82f6; transition: width .4s; }
[hidden] { display: none !important; }
`;

    const MENU_CSS = `
:host { all: initial; position: fixed; top: 0; left: 0; z-index: 2147483647; }
:host([hidden]) { display: none !important; }
* { box-sizing: border-box; font-family: "Roboto", "Arial", system-ui, sans-serif; }
.menu.dark { --fg: #f1f1f1; --bg: rgba(255,255,255,.1); --menu: #282828; --line: rgba(255,255,255,.12); }
.menu.light { --fg: #0f0f0f; --bg: rgba(0,0,0,.05); --menu: #fff; --line: rgba(0,0,0,.1); }
.menu {
    min-width: 210px; padding: 8px 0; border-radius: 12px; background: var(--menu); color: var(--fg);
    box-shadow: 0 4px 32px rgba(0,0,0,.3); border: 1px solid var(--line);
}
.menu button { all: unset; display: flex; align-items: center; gap: 12px; width: 100%; box-sizing: border-box;
               padding: 0 16px; height: 36px; font-size: 14px; cursor: pointer; white-space: nowrap; }
.menu button:hover, .menu button:focus-visible { background: var(--bg); }
.menu .note { padding: 6px 16px 2px; font-size: 12px; opacity: .7; white-space: normal; max-width: 260px; }
`;

    const PANEL_CSS = `
:host { all: initial; visibility: inherit; display: block; margin: 12px 0 0; }
* { box-sizing: border-box; font-family: "Roboto", "Arial", system-ui, sans-serif; }
.card.dark { --fg: #f1f1f1; --fg2: #aaa; --bg: rgba(255,255,255,.1); --line: rgba(255,255,255,.12); }
.card.light { --fg: #0f0f0f; --fg2: #606060; --bg: rgba(0,0,0,.05); --line: rgba(0,0,0,.1); }
.card { border-radius: 12px; background: var(--bg); color: var(--fg); padding: 12px 14px; }
.head { display: flex; align-items: center; gap: 10px; font-size: 14px; font-weight: 500; margin-bottom: 8px; }
.head .ic { color: #22c55e; }
.head .gap { flex: 1; }
.head button { all: unset; cursor: pointer; font-size: 13px; color: #3ea6ff; }
.head button:hover { text-decoration: underline; }
.head .x { color: var(--fg2); font-size: 15px; }
.text { white-space: pre-wrap; font-size: 14px; line-height: 20px; max-height: 360px; overflow: auto; }
[hidden] { display: none !important; }
:host([hidden]) { display: none !important; }
`;

    let actionEl = null, panelEl = null, menuEl = null, menuFrame = 0;
    // The menu's and the bar's actions that send to the server or open something: the user's clicks only
    const SENDS = new Set(['summarize', 'transcribe', 'options', 'side', 'open', 'start-analyze', 'cancel']);
    const painted = {menu: null, text: null};   // what was written last — written again only when it changed

    function makeAction() {
        const el = document.createElement('wtm-action');
        el.style.cssText = 'margin-left:8px;flex:none;';
        const sh = shadow(el);
        setHTML(sh, `<style>${ACTION_CSS}</style>
            <button class="main" title="Whisper Transcription Manager" aria-haspopup="menu"><span class="ic">✦</span><span class="lbl">Whisper TM</span><span class="car">▾</span>
                <span class="prog" hidden><div></div></span></button>`);
        for (const t of ['mousedown', 'pointerdown', 'touchstart']) sh.addEventListener(t, e => e.stopPropagation());
        sh.querySelector('.main').addEventListener('click', e => {
            e.preventDefault();                   // the menu, the summary unfolded — nothing sent
            e.stopPropagation();
            if (M.state === 'summarized' && M.summary && !M.menu && !e.target.closest('.car')) {
                M.panel = !M.panel;               // the summary, unfolded or folded
                paintMain();
                return;
            }
            M.menu = !M.menu;
            paintMain();
        });
        return el;
    }

    /* The menu sits on top of the page, not in the action row: YouTube clips
     * that row (overflow hidden), and a menu inside it never showed. */
    function makeMenu() {
        const el = document.createElement('wtm-menu');
        el.hidden = true;
        const sh = shadow(el);
        painted.menu = null;
        setHTML(sh, `<style>${MENU_CSS}</style><div class="menu" role="menu"></div>`);
        for (const t of ['mousedown', 'pointerdown', 'touchstart']) sh.addEventListener(t, e => e.stopPropagation());
        sh.querySelector('.menu').addEventListener('click', e => {
            const b = e.target.closest('button[data-a]');
            if (!b) return;
            const a = b.dataset.a;
            if (SENDS.has(a) && !real(e)) return;
            e.preventDefault();
            e.stopPropagation();
            M.menu = false;
            const self = {url: M.url, title: M.title};
            if ((a === 'summarize' || a === 'transcribe') && PREFS.askFirst) openPanel({form: a, ...self});
            else if (a === 'summarize' || a === 'transcribe') mainSend(a);
            else if (a === 'options') openPanel({form: 'summarize', ...self});
            else if (a === 'side') openPanel({tab: 'summary', ...self});
            else if (a === 'select') toggle(M.key, true);
            else if (a === 'unselect') toggle(M.key, false);
            else if (a === 'open' && M.open) send({type: 'open', path: M.open});
            else if (a === 'panel') M.panel = !M.panel;
            paintMain();
        });
        (document.documentElement || document.body).appendChild(el);
        return el;
    }

    /* The side panel with what it should show — background.js opens it while
     * this click lasts (Chrome allows it only then), so it is asked at once. */
    function openPanel(intent) {
        send({type: 'panel', intent}).then(d => {
            if (!d.error) return;
            // An older Chrome, or the click didn't count: say where it is
            M.error = 'The side panel: click the Whisper Transcription Manager icon → ▤ (' + d.error + ')';
            S.error = M.error;
            paintMain();
            render();
        });
    }

    /* Under the button, right edges lined up — above it when there is no room
     * below. It follows the button while the page scrolls, and closes once the
     * button is out of sight. */
    function placeMenu() {
        if (!M.menu || !menuEl || !actionEl || !actionEl.isConnected) return false;
        const r = actionEl.getBoundingClientRect();
        if (r.bottom < 0 || r.top > innerHeight || !r.width) return false;
        const box = SHADOWS.get(menuEl).querySelector('.menu').getBoundingClientRect();
        const left = Math.max(8, Math.min(r.right - box.width, innerWidth - box.width - 8));
        const above = r.top - 6 - box.height;
        const top = r.bottom + 6 + box.height > innerHeight - 8 && above >= 8 ? above : r.bottom + 6;
        menuEl.style.left = Math.round(left) + 'px';
        menuEl.style.top = Math.round(top) + 'px';
        return true;
    }

    function trackMenu() {
        if (menuFrame) return;
        const step = () => {
            menuFrame = 0;
            if (!M.menu) return;
            if (!placeMenu()) {
                M.menu = false;
                paintMain();
                return;
            }
            menuFrame = requestAnimationFrame(step);
        };
        menuFrame = requestAnimationFrame(step);
    }

    function makePanel() {
        const el = document.createElement('wtm-panel');
        const sh = shadow(el);
        setHTML(sh, `<style>${PANEL_CSS}</style><div class="card">
            <div class="head"><span class="ic">✦</span><span>Summary · Whisper Transcription Manager</span><span class="gap"></span>
                <button class="ask" title="The side panel: read it, its transcript, ask about it">Ask about it ▤</button>
                <button class="open">Open in Whisper Transcription Manager ↗</button><button class="x" title="Close">✕</button></div>
            <div class="text"></div></div>`);
        sh.querySelector('.open').addEventListener('click', e => {
            if (!real(e)) return;
            e.stopPropagation();
            if (M.open) send({type: 'open', path: M.open});
        });
        sh.querySelector('.ask').addEventListener('click', e => {
            if (!real(e)) return;
            e.stopPropagation();
            openPanel({tab: 'ask', url: M.url, title: M.title});
        });
        sh.querySelector('.x').addEventListener('click', e => {
            e.stopPropagation();
            M.panel = false;
            paintMain();
        });
        return el;
    }

    /* The watch page's video: a new one resets what is known about it, and the
     * button and the panel go back where YouTube took them away. */
    function placeMain() {
        if (!ON_YOUTUBE) return;
        const v = location.pathname === '/watch' ? WTM.parse(location.href) : null;
        const key = v && v.kind === 'video' ? v.key : '';
        if (key !== M.key) {
            clearTimeout(M.timer);
            Object.assign(M, {key, url: v ? v.url : '', title: '', state: '', progress: 0, summary: '',
                              open: '', batchId: '', timer: 0, menu: false, panel: false, error: ''});
            if (key) askMain();
        }
        const meta = document.querySelector('ytd-watch-metadata');
        if (!key || !meta) {
            actionEl?.remove();
            panelEl?.remove();
            M.menu = false;
            if (menuEl) menuEl.hidden = true;
            return;
        }
        M.title = (S.videos.get(key) || {}).title || M.title;
        const menu = meta.querySelector('#actions ytd-menu-renderer');
        if (menu) {
            if (!actionEl) actionEl = makeAction();
            if (!menuEl || !menuEl.isConnected) menuEl = makeMenu();
            if (actionEl.parentElement !== menu) {
                const flexible = menu.querySelector(':scope > #flexible-item-buttons');
                menu.insertBefore(actionEl, flexible || menu.children[1] || null);
            }
        }
        const row = meta.querySelector('#top-row');
        if (row) {
            if (!panelEl) panelEl = makePanel();
            if (panelEl.previousElementSibling !== row) row.after(panelEl);
        }
        paintMain();
    }

    async function askMain() {
        const key = M.key;
        const d = await send({type: 'known', summaries: true,
                              videos: [{url: M.url, title: (S.videos.get(key) || {}).title || ''}]});
        if (M.key !== key || M.batchId) return;
        const row = (d.videos || [])[0];
        if (row && row.state) {
            M.state = row.state;
            M.open = row.open || '';
            M.summary = row.summary || '';
            S.known.set(key, {state: row.state, open: M.open});
        }
        paintMain();
    }

    async function mainSend(mode) {
        const key = M.key;
        M.error = '';
        M.state = 'waiting';
        M.progress = 0;
        paintMain();
        const d = await send({type: 'ingest', mode, urls: [M.url]});
        if (M.key !== key) {
            // Another video meanwhile (YouTube changes the page, not the
            // tab): the answer is not this one's
            return d.error || !d.batch ? {error: d.error || 'no answer'} : {batch: d.batch};
        }
        if (d.error || !d.batch) {
            M.state = '';
            M.error = d.error || 'no answer';
            M.menu = true;
            paintMain();
            return {error: M.error};
        }
        followMain(d.batch);
        return {batch: d.batch};
    }

    /* The button follows a batch: asked again three seconds after each
     * answer — never two questions at once, however slow the server. */
    function followMain(b) {
        clearTimeout(M.timer);
        M.timer = 0;
        M.batchId = b.id;
        if (takeMain(b)) M.timer = setTimeout(pollMain, 3000);
    }

    async function pollMain() {
        M.timer = 0;
        const key = M.key, id = M.batchId;
        if (!id || retired) return;
        const d = await send({type: 'batch', id});
        if (M.key !== key || M.batchId !== id || M.timer) return;    // another video, or followed anew
        if (!d.batch || takeMain(d.batch)) M.timer = setTimeout(pollMain, 3000);
    }

    /* The batch's word on this video — only its own item, never another's.
     * True while there is more to come. */
    function takeMain(b) {
        const it = (b.items || []).find(x => (WTM.parse(x.url) || {}).key === M.key);
        if (!it) return false;
        const was = M.state;
        M.state = it.state;
        M.progress = it.progress || 0;
        M.open = it.open || M.open;
        M.error = it.state === 'error' ? (it.error || 'failed') : '';
        if (it.summary) M.summary = it.summary;
        let more = true;
        if (['summarized', 'error'].includes(it.state) || (it.state === 'transcribed' && b.mode !== 'summarize')) {
            more = false;
            S.known.set(M.key, {state: it.state, open: M.open});
        } else if (b.status && b.status !== 'working') {
            more = false;                         // cancelled, or done without this one
        }
        if (it.state === 'summarized' && was !== 'summarized') M.panel = true;   // there it is
        paintMain();
        return more;
    }

    /* Written only where it changed: YouTube changes its page all the time,
     * and the summary under the title, written anew each round, lost the
     * text the user was selecting in it. */
    function paintMain() {
        if (!actionEl) return;
        const sh = SHADOWS.get(actionEl);
        const theme = dark(actionEl) ? 'dark' : 'light';
        const working = ['waiting', 'resolving', 'downloading', 'queued', 'transcribing', 'summarizing'].includes(M.state);
        const label = {
            waiting: 'Sending …', resolving: 'Looking up …', downloading: `Downloading ${M.progress || 0} %`,
            queued: 'Queued …', transcribing: 'Transcribing …', summarizing: 'Summarizing …',
            transcribed: 'Transcribed ✓', summarized: 'Summary ✓', error: 'Failed',
        }[M.state] || (S.selected.has(M.key) ? 'Selected ✓' : 'Whisper TM');
        const lbl = sh.querySelector('.lbl');
        if (lbl.textContent !== label) lbl.textContent = label;
        const split = M.state === 'summarized' && !!M.summary;
        const mainCls = ['main', theme, ['summarized', 'transcribed'].includes(M.state) ? 'ok' : '',
                         split ? 'split' : ''].filter(Boolean).join(' ');
        if (sh.querySelector('.main').className !== mainCls) sh.querySelector('.main').className = mainCls;
        sh.querySelector('.main').title = split ? (M.panel ? 'Hide the summary' : 'Show the summary') + ' — ▾ for more'
                                                : 'Whisper Transcription Manager';
        const prog = sh.querySelector('.prog');
        prog.hidden = !working;
        prog.firstElementChild.style.width = (M.state === 'downloading' ? Math.max(4, M.progress)
            : {waiting: 3, resolving: 6, queued: 30, transcribing: 60, summarizing: 85}[M.state] || 0) + '%';
        const selected = S.selected.has(M.key);
        const items = [];
        if (!working) {
            if (M.state !== 'summarized') items.push('<button data-a="summarize">⚡ Summarize</button>');
            if (!['transcribed', 'summarized'].includes(M.state)) items.push('<button data-a="transcribe">📝 Transcribe</button>');
            items.push(`<button data-a="options">⚙ ${M.state === 'summarized' ? 'Summarize again' : 'With options'} …</button>`);
        }
        items.push('<button data-a="side">▤ Read &amp; ask in the side panel</button>');
        if (M.summary) items.push(`<button data-a="panel">${M.panel ? '▴ Hide the summary' : '▾ Show the summary'}</button>`);
        items.push(selected ? '<button data-a="unselect">✓ Selected — unselect</button>'
                            : '<button data-a="select">＋ Add to selection</button>');
        if (M.open) items.push('<button data-a="open">↗ Open in Whisper Transcription Manager</button>');
        if (working) items.push('<div class="note">It goes on on the server — you can leave this page.</div>');
        if (M.error) items.push(`<div class="note">⚠ ${esc(M.error)}</div>`);
        sh.querySelector('.main').setAttribute('aria-expanded', String(M.menu));
        if (menuEl) {
            const box = SHADOWS.get(menuEl).querySelector('.menu');
            if (box.className !== 'menu ' + theme) box.className = 'menu ' + theme;
            const html = items.join('');
            if (painted.menu !== html) { painted.menu = html; setHTML(box, html); }
            menuEl.hidden = !M.menu;
            if (M.menu) {
                placeMenu();
                trackMenu();
            }
        }
        if (panelEl) {
            const psh = SHADOWS.get(panelEl);
            const card = psh.querySelector('.card');
            if (card.className !== 'card ' + theme) card.className = 'card ' + theme;
            panelEl.hidden = !(M.panel && M.summary);
            const html = md(M.summary || '');
            if (painted.text !== html) {
                painted.text = html;
                setHTML(psh.querySelector('.text'), html);
            }
        }
    }

    // A click anywhere else, or Escape, closes the menu
    document.addEventListener('click', e => {
        const path = e.composedPath();
        if (M.menu && actionEl && !path.includes(actionEl) && !path.includes(menuEl)) {
            M.menu = false;
            paintMain();
        }
    }, true);
    document.addEventListener('keydown', e => {
        if (M.menu && e.key === 'Escape') {
            M.menu = false;
            paintMain();
        }
    }, true);

    // ══ Finding the videos ════════════════════════════════════════════════

    let scanTimer = 0;
    function scanSoon() {
        if (scanTimer || retired) return;
        scanTimer = setTimeout(() => { scanTimer = 0; scan(); }, 600);
    }

    function scan() {
        if (retired) return;
        S.videos = WTM.scan(document, location);
        S.byEl = new WeakMap();
        for (const v of S.videos.values()) for (const el of v.els) if (el) S.byEl.set(el, v.key);
        for (const [k, sel] of S.selected) {                // a title found later
            const v = S.videos.get(k);
            if (v && !sel.title && v.title) sel.title = v.title;
        }
        placeMain();
        placePicks();
        askKnown();
        placeMarks();
    }

    /* Which of the page's videos are in the library already — asked once each:
     * on YouTube, and on other sites only when the user chose so (PREFS). */
    async function askKnown() {
        if (!ON_YOUTUBE && !PREFS.marksEverywhere) return;
        const fresh = [...S.videos.values()].filter(v => v.kind === 'video' && !S.asked.has(v.key)).slice(0, 200);
        if (!fresh.length) return;
        fresh.forEach(v => S.asked.add(v.key));
        // The title finds a download from before the ids were noted — on
        // YouTube only: elsewhere the page writes it, and a page asking for
        // made-up titles learned which files the library has
        const d = await send({type: 'known', videos: fresh.map(v => ({url: v.url, title: ON_YOUTUBE ? v.title : ''}))});
        for (const row of d.videos || []) {
            const v = WTM.parse(row.url);
            if (v && row.state) S.known.set(v.key, {state: row.state, open: row.open || ''});
        }
        paintAllPicks();
        if (S.hover) placeHover();
    }

    /* A round only for a change that can matter: links or players added,
     * removed, or pointed elsewhere — on YouTube also its cards and the
     * watch page's action row (it takes our button away when it redraws).
     * A clock ticking in a player, a chat's new line, our own ⊕: nothing. */
    const RELEVANT = 'a[href], iframe[src]'
        + (ON_YOUTUBE ? ', ytd-watch-metadata, #actions, ytd-menu-renderer, #top-row, ' + WTM.CARDS : '');
    const OURS = 'wtm-pick, wtm-action, wtm-panel, wtm-menu';
    const VIDEO_LINK = /youtu|^\/(watch|shorts|live|playlist)/;
    function has(n, sel) {
        return n.nodeType === 1 && (n.matches(sel) || !!n.querySelector(sel));
    }
    /* A hidden link's ⊕ waits: shown later (a menu opened, a tab, a <details>
     * unfolded — a class, a style, hidden or open changed on it or around
     * it), it gets one. */
    function reveals(target) {
        for (const a of skipped) if (a === target || (target.contains && target.contains(a))) return true;
        return false;
    }
    function matters(records) {
        for (const r of records) {
            if (r.type === 'attributes') {
                if (r.attributeName === 'href' || r.attributeName === 'src') {
                    if (VIDEO_LINK.test(r.target.getAttribute(r.attributeName) || '') || VIDEO_LINK.test(r.oldValue || '')) return true;
                } else if (skipped.size && reveals(r.target)) {
                    return true;
                }
                continue;
            }
            for (const n of r.addedNodes) {
                if (n.nodeType === 1 && !/^WTM-/.test(n.tagName) && has(n, RELEVANT)) return true;
            }
            for (const n of r.removedNodes) {
                if (n.nodeType === 1 && (has(n, RELEVANT) || has(n, OURS))) return true;
            }
        }
        return false;
    }
    const observer = new MutationObserver(records => {
        if (retired) observer.disconnect();
        else if (matters(records)) scanSoon();
    });
    /* class, style, hidden, open watched only while a hidden link waits —
     * never on YouTube, whose player changes its styles all the time. */
    let watching = null;
    function watch() {
        const reveal = !ON_YOUTUBE && skipped.size > 0;
        if (reveal === watching || retired) return;
        watching = reveal;
        observer.observe(document.documentElement, {childList: true, subtree: true, attributes: true,
            attributeFilter: reveal ? ['href', 'src', 'class', 'style', 'hidden', 'open'] : ['href', 'src'],
            attributeOldValue: true});
    }
    watch();
    window.addEventListener('yt-navigate-finish', scanSoon);
    // The system's dark mode switched: the colours read again
    try {
        const scheme = matchMedia('(prefers-color-scheme: dark)');
        if (scheme && scheme.addEventListener) {
            scheme.addEventListener('change', () => {
                if (retired) return;
                pageTheme = '';
                pickTheme = new WeakMap();
                paintAllPicks();
                paintMain();
            });
        }
    } catch (e) { /* no media queries here */ }
    scan();

    // ── The badge on an embedded player or a picture link (not on YouTube) ──

    /* Videos that have no ⊕ of their own: players and pictures off YouTube. */
    function badgeTarget(el) {
        if (ON_YOUTUBE || !el) return false;
        return el.tagName === 'IFRAME' || !(el.textContent || '').trim();
    }

    function videoAt(node) {
        for (let el = node, i = 0; el && i < 8; el = el.parentElement, i++) {
            const k = S.byEl.get(el);
            if (k && badgeTarget(el)) return {key: k, el};
        }
        return null;
    }

    document.addEventListener('mouseover', e => {
        if (retired) return;
        if (e.target === host) { clearTimeout(S.hideTimer); return; }
        const hit = videoAt(e.target);
        if (hit) {
            clearTimeout(S.hideTimer);
            S.hover = hit;
            placeHover();
        } else if (S.hover) {
            // Still over the same video — something drawn on top of it
            const r = S.hover.el.isConnected ? S.hover.el.getBoundingClientRect() : null;
            if (r && e.clientX >= r.left && e.clientX <= r.right
                    && e.clientY >= r.top && e.clientY <= r.bottom) return;
            clearTimeout(S.hideTimer);
            S.hideTimer = setTimeout(() => { S.hover = null; badge.hidden = true; }, 350);
        }
    }, true);

    function placeHover() {
        const h = S.hover;
        if (!h || !h.el.isConnected) { badge.hidden = true; return; }
        const r = h.el.getBoundingClientRect();
        if (r.width < 24 || r.height < 14) { badge.hidden = true; return; }
        const v = S.videos.get(h.key) || {};
        const on = S.selected.has(h.key);
        const k = S.known.get(h.key);
        badge.className = 'badge' + (on ? ' on' : '');
        setHTML(badge, (on ? '✓ Selected' : (v.kind === 'playlist' ? '＋ Playlist' : '＋ Whisper TM'))
            + (k ? `<span class="k">${esc(stateText(k.state))}</span>` : ''));
        badge.title = on ? 'Click to unselect' : 'Select this video for Whisper Transcription Manager';
        badge.style.left = Math.max(4, r.left + 6) + 'px';
        badge.style.top = Math.max(4, r.top + 6) + 'px';
        badge.hidden = false;
    }

    badge.addEventListener('mouseenter', () => clearTimeout(S.hideTimer));
    badge.addEventListener('click', e => {
        e.preventDefault();
        e.stopPropagation();
        if (S.hover) toggle(S.hover.key);                  // selecting only
    });

    function toggle(key, on) {
        const v = S.videos.get(key) || (key === M.key && M.url ? {kind: 'video', url: M.url, title: M.title} : null);
        const want = on === undefined ? !S.selected.has(key) : on;
        if (want && v) S.selected.set(key, {key, kind: v.kind, url: v.url, title: v.title});
        else S.selected.delete(key);
        if (S.mode === 'batch' && S.selected.size) S.mode = 'select';
        paintAllPicks();
        paintMain();
        placeHover();
        placeMarks();
        render();
    }

    /* A tick on a selected player or picture off YouTube — a ⊕ shows its own. */
    function placeMarks() {
        let html = '';
        for (const key of S.selected.keys()) {
            const v = S.videos.get(key);
            const els = v ? v.els.filter(badgeTarget) : [];
            const el = els.length ? WTM.anchor({els}) : null;
            if (!el) continue;
            const r = el.getBoundingClientRect();
            if (r.bottom < 0 || r.top > innerHeight) continue;
            if (S.hover && S.hover.key === key && !badge.hidden) continue;
            html += `<div class="mark" data-k="${esc(key)}" title="Selected — click to unselect"
                          style="left:${Math.max(4, r.left + 6)}px;top:${Math.max(4, r.top + 6)}px">✓</div>`;
        }
        setHTML(marks, html);
    }
    marks.addEventListener('click', e => {
        const m = e.target.closest('.mark');
        if (m) toggle(m.dataset.k, false);
    });

    let rafPending = false;
    const follow = () => {
        if (rafPending || retired) return;
        rafPending = true;
        requestAnimationFrame(() => { rafPending = false; placeMarks(); if (S.hover) placeHover(); });
    };
    window.addEventListener('scroll', follow, true);
    window.addEventListener('resize', follow);

    // ══ The bar: what to do with the selected videos ═══════════════════════

    const thumb = url => {
        const v = WTM.parse(url);
        return v && v.id ? `https://i.ytimg.com/vi/${v.id}/default.jpg` : '';
    };

    const STRIP = 6;          // pictures in the row before "+N"

    /* The selection changed: the side panel shows it at once (its "Selected"
     * box) — said once per change, not on every redraw of the bar. */
    let toldSelection = '';
    function tellSelection() {
        const now = [...S.selected.keys()].join(' ');
        if (now === toldSelection) return;
        toldSelection = now;
        send({type: 'selection', count: S.selected.size});
    }

    function render() {
        tellSelection();
        if (S.mode !== 'batch' && !S.selected.size) { bar.hidden = true; return; }
        bar.hidden = false;
        const n = S.selected.size;
        const listToggle = S.mode === 'select'
            ? `<button class="l" data-a="list">${S.listOpen ? 'hide the list' : 'show the list'}</button>` : '';
        const head = `<div class="head"><span class="logo">W</span>
            <span class="title">${S.mode === 'batch' ? esc(batchTitle()) : `${n} video${n === 1 ? '' : 's'} selected`}</span>
            ${listToggle}
            <button class="x" data-a="close" title="${S.mode === 'batch' ? 'Close — it goes on on the server' : 'Clear the selection'}">✕</button></div>`;
        let body = '';
        if (S.mode === 'select') {
            const sel = [...S.selected.values()];
            const strip = S.listOpen ? '' : `<div class="strip" data-a="list" title="Show the list">${sel.slice(0, STRIP).map(v =>
                    thumb(v.url) ? `<img src="${esc(thumb(v.url))}" alt="" title="${esc(v.title || v.url)}">`
                                 : `<span class="ph" title="${esc(v.title || v.url)}">${v.kind === 'playlist' ? '≡' : '▶'}</span>`).join('')}
                    ${sel.length > STRIP ? `<span class="more">+${sel.length - STRIP} more</span>` : ''}</div>`;
            const list = !S.listOpen ? '' : `<div class="list" style="margin:0 0 8px">${sel.map(v => `<div class="it">
                    ${thumb(v.url) ? `<img src="${esc(thumb(v.url))}" alt="">` : ''}
                    <span class="t" title="${esc(v.title || v.url)}">${esc(v.title || v.url)}</span>
                    ${S.known.get(v.key) ? `<span class="chip ok">${esc(stateText(S.known.get(v.key).state))}</span>` : ''}
                    <button class="x" data-a="drop" data-k="${esc(v.key)}" title="Unselect">✕</button></div>`).join('')}</div>`;
            body = `${strip}${list}<div class="acts">
                    <button class="b p" data-a="summarize" title="Download, transcribe, and summarize each one">Summarize</button>
                    <button class="b" data-a="transcribe" title="Download into the upload folder and transcribe — nothing more">Transcribe</button>
                    <button class="b" data-a="options" title="Choose how — the template, the language, the model … — in the side panel">Options…</button>
                    <button class="b" data-a="analyze" title="One task over all of them together — compare them, summarize them together, find …">Analyze…</button>
                </div>
                ${S.error ? `<div class="err">${esc(S.error)}</div>` : ''}`;
        } else if (S.mode === 'analyze') {
            body = `<div class="note">One task over the ${n} selected video${n === 1 ? '' : 's'} — once all are transcribed, the language model reads them together.</div>
                <textarea data-f="task" placeholder="e.g. Compare what they say about running models locally — where they agree, where they differ."></textarea>
                <div class="acts"><button class="b p" data-a="start-analyze">Start</button>
                <button class="b" data-a="back">Back</button></div>
                ${S.error ? `<div class="err">${esc(S.error)}</div>` : ''}`;
        } else {
            body = batchHtml();
        }
        const keep = bar.querySelector('textarea');
        const task = keep ? keep.value : '';
        setHTML(bar, head + `<div class="body">${body}</div>`);
        const ta = bar.querySelector('textarea');
        if (ta) { ta.value = task; if (!task) ta.focus(); }
    }

    function batchTitle() {
        const b = S.batch;
        if (!b) return 'Sending …';
        const n = (b.items || []).length;
        const what = {transcribe: 'Transcribing', summarize: 'Summarizing', analyze: 'Analyzing'}[b.mode] || '';
        return b.status === 'done' ? `Done · ${n} video${n === 1 ? '' : 's'}`
            : `${what} ${n} video${n === 1 ? '' : 's'} …`;
    }

    function batchHtml() {
        const b = S.batch;
        if (!b) return `<div class="note">Sending to Whisper Transcription Manager …</div>`;
        const items = (b.items || []).map((it, i) => {
            const st = STATE[it.state] || [it.state, ''];
            const pct = it.state === 'downloading' && it.progress ? ` ${it.progress} %` : '';
            const open = S.open.has(i);
            const title = it.title || S.titles.get(it.url) || it.url;
            return `<div class="it">
                    ${thumb(it.url) ? `<img src="${esc(thumb(it.url))}" alt="">` : ''}
                    <span class="t" title="${esc(title)}">${esc(title)}</span>
                    <span class="chip ${st[1]}" title="${esc(it.error || '')}">${esc(st[0] + pct)}</span>
                    ${it.summary ? `<button class="l" data-a="sum" data-i="${i}">${open ? 'hide' : 'summary'}</button>` : ''}
                    ${it.open ? `<button class="l" data-a="open" data-p="${esc(it.open)}" title="Open in Whisper Transcription Manager">open ↗</button>` : ''}
                </div>${it.summary && open ? `<div class="sum">${md(it.summary)}</div>` : ''}
                ${it.error && it.state === 'error' ? `<div class="err">${esc(it.error)}</div>` : ''}`;
        }).join('');
        const a = b.analysis;
        const an = a ? `<div class="an"><span class="t">Analysis: ${esc(a.title || '')}</span>
                <span class="chip ${a.status === 'done' ? 'ok' : a.status === 'error' ? 'err' : 'run'}">${esc(
                    {waiting: 'waiting for transcripts', queued: 'in the LLM queue', running: (a.progress || 0) + ' %',
                     done: 'done', error: 'failed', cancelled: 'cancelled'}[a.status] || a.status)}</span>
                <button class="l" data-a="open" data-p="${esc(a.open || '')}">open ↗</button></div>` : '';
        return `<div class="list">${items}</div>${an}
            ${b.error ? `<div class="err">${esc(b.error)}</div>` : ''}
            <div class="acts" style="margin-top:10px">
                ${b.status === 'working' && b.downloading ? '<button class="b" data-a="cancel">Stop downloading</button>' : ''}
                <button class="b" data-a="new">Select more</button></div>
            <div class="note">${b.status === 'done' ? 'Everything is in the library.' : 'It goes on on the server — you can close this or the tab.'}</div>`;
    }

    bar.addEventListener('click', async e => {
        const el = e.target.closest('[data-a]');
        if (!el) return;
        const a = el.dataset.a;
        if (SENDS.has(a) && !real(e)) return;
        S.error = '';
        if (a === 'close') {
            if (S.mode === 'batch') { stopPolling(); S.mode = 'select'; S.batch = null; }
            S.selected.clear();
            paintAllPicks();
            paintMain();
            placeMarks();
            render();
        } else if (a === 'drop') {
            toggle(el.dataset.k, false);
        } else if (a === 'list') {
            S.listOpen = !S.listOpen;
            render();
        } else if (a === 'options' || ((a === 'transcribe' || a === 'summarize') && PREFS.askFirst)) {
            const sel = [...S.selected.values()];
            openPanel({form: a === 'options' ? 'summarize' : a, urls: sel.map(v => v.url),
                       titles: sel.map(v => v.title || '')});
        } else if (a === 'transcribe' || a === 'summarize') {
            await ingest(a, '');
        } else if (a === 'analyze') {
            S.mode = 'analyze';
            render();
        } else if (a === 'back') {
            S.mode = 'select';
            render();
        } else if (a === 'start-analyze') {
            const task = (bar.querySelector('textarea')?.value || '').trim();
            if (!task) { S.error = 'Write the task first.'; render(); return; }
            await ingest('analyze', task);
        } else if (a === 'sum') {
            const i = Number(el.dataset.i);
            if (S.open.has(i)) S.open.delete(i); else S.open.add(i);
            render();
        } else if (a === 'open') {
            if (el.dataset.p) send({type: 'open', path: el.dataset.p});
        } else if (a === 'cancel') {
            if (S.batch) await send({type: 'cancel', id: S.batch.id});
            poll();
        } else if (a === 'new') {
            stopPolling();
            S.batch = null;
            S.mode = 'select';
            render();
        }
    });

    async function ingest(mode, task) {
        const items = [...S.selected.values()];
        if (!items.length) return;
        S.mode = 'batch';
        S.batch = null;
        S.open.clear();
        S.titles = new Map(items.map(v => [v.url, v.title]));
        render();
        const d = await send({type: 'ingest', mode, task, urls: items.map(v => v.url)});
        if (d.error || !d.batch) {
            S.mode = mode === 'analyze' ? 'analyze' : 'select';
            const err = d.error || 'no answer';
            // The address only where the server wasn't reached — a refusal
            // (no language model set up) says what it is itself
            S.error = 'Not sent: ' + err + (/can't be reached|didn't answer|no answer/.test(err)
                ? ' — is the server address right? (extension icon → ⚙)' : '');
            render();
            return;
        }
        S.batch = d.batch;
        if (M.key && items.some(v => v.key === M.key)) followMain(d.batch);   // the button follows too
        S.selected.clear();
        paintAllPicks();
        placeMarks();
        render();
        startPolling();
    }

    /* The bar follows its batch: asked again three seconds after each answer. */
    function startPolling() {
        stopPolling();
        S.pollTimer = setTimeout(poll, 3000);
    }

    function stopPolling() {
        clearTimeout(S.pollTimer);
        S.pollTimer = 0;
    }

    async function poll() {
        stopPolling();
        const b = S.batch;
        if (!b || retired) return;
        const d = await send({type: 'batch', id: b.id});
        if (S.batch !== b) return;              // closed, or another batch meanwhile
        if (!d.batch) { S.pollTimer = setTimeout(poll, 3000); return; }
        S.batch = d.batch;
        // What is in the library now shows on its ⊕ too
        for (const it of d.batch.items || []) {
            const v = WTM.parse(it.url);
            if (v && ['transcribed', 'summarized'].includes(it.state)) S.known.set(v.key, {state: it.state, open: it.open || ''});
        }
        paintAllPicks();
        if (S.mode === 'batch') render();
        if (d.batch.status === 'working') S.pollTimer = setTimeout(poll, 3000);
    }

    // ══ The popup asks ══════════════════════════════════════════════════════

    chrome.runtime.onMessage.addListener((msg, sender, reply) => {
        if (retired) return false;
        if (msg.type === 'list') {
            scan();
            // The watch page's own video first, the others in the page's order
            const all = [...S.videos.values()];
            all.sort((a, b) => (b.key === M.key) - (a.key === M.key));
            reply({main: M.key ? {key: M.key, url: M.url, title: M.title || (S.videos.get(M.key) || {}).title || '',
                                  state: M.state, summary: M.summary, open: M.open} : null,
                   // titleFromPage: the page wrote it (not YouTube) — shown, never matched against the library
                   videos: all.map(v => ({
                       key: v.key, kind: v.kind, id: v.id, url: v.url, title: v.title, main: v.key === M.key,
                       titleFromPage: !ON_YOUTUBE,
                       selected: S.selected.has(v.key), known: S.known.get(v.key) || null})),
                   // The selection in its order — also videos scrolled out of the page since
                   selection: [...S.selected.values()].map(v => ({key: v.key, kind: v.kind, url: v.url,
                                                                  title: v.title || ''}))});
        } else if (msg.type === 'toggle') {
            toggle(msg.key, !!msg.on);
            reply({ok: true});
        } else if (msg.type === 'unselect-all') {
            // The side panel's "clear the selection" — the watch page's own video too
            S.selected.clear();
            paintAllPicks();
            paintMain();
            placeMarks();
            render();
            reply({ok: true});
        } else if (msg.type === 'send') {
            ingest(msg.mode, msg.task || '').then(() => reply({ok: !!S.batch, batch: S.batch,
                                                                reason: S.error || ''}));
            return true;
        } else if (msg.type === 'main-send') {
            if (!M.key) { reply({error: 'This page plays no video'}); return false; }
            mainSend(msg.mode || 'summarize').then(reply);
            return true;
        } else if (msg.type === 'seek') {
            // A time clicked in the side panel: the video plays from there
            const v = document.querySelector('#movie_player video') || document.querySelector('video');
            if (v) {
                v.currentTime = Math.max(0, Number(msg.t) || 0);
                if (v.paused) v.play().catch(() => {});
            }
            reply({ok: !!v});
        } else if (msg.type === 'follow-main' && msg.batch) {
            // Sent from the side panel: the button shows how far it is
            if (M.key) followMain(msg.batch);
            reply({ok: true});
        } else if (msg.type === 'follow' && msg.batch) {
            if (msg.clear) {
                // The side panel sent the selection — it is not selected any more
                S.selected.clear();
                paintAllPicks();
                placeMarks();
            }
            S.batch = msg.batch;
            S.mode = 'batch';
            render();
            startPolling();
            if (M.key && (msg.batch.items || []).some(x => (WTM.parse(x.url) || {}).key === M.key)) followMain(msg.batch);
            reply({ok: true});
        }
        return false;
    });

    // Tests and the live test harness only (see the top): never without the flag
    if (TEST) {
        globalThis.__wxpTest = {shadow: el => SHADOWS.get(el) || null, root: () => root, scan,
                                state: S, main: M, matters, editable, shown};
    }
})();
