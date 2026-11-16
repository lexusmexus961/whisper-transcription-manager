/* ══════════════════════════════════════════════════════════════════════════
 *  Finding the YouTube videos of a page
 *
 *  Wherever there is a YouTube video: links (a search result, a post, a
 *  thumbnail on YouTube itself — every card of its home page, search and
 *  side bar), embedded players (youtube.com/embed, youtube-nocookie.com),
 *  the video of a watch page, and playlists. Each video once, under its id,
 *  with every element that shows it — the badge sits on the largest.
 *
 *  Plain functions on a document, so they can be tried without a browser
 *  (tests/test_browser_extension.py); content.js draws the selection on top.
 * ══════════════════════════════════════════════════════════════════════════ */

var WTM = globalThis.WTM || (globalThis.WTM = {});

/* A link to YouTube as one video (`watch?v=X` — also inside a playlist: the
 * thumbnail means that video) or one playlist; null for anything else. */
WTM.parse = function (href, base) {
    let u;
    try { u = new URL(href, base || 'https://www.youtube.com/'); } catch (e) { return null; }
    if (!/^https?:$/.test(u.protocol)) return null;
    const host = u.hostname.toLowerCase().replace(/^(www|m|music)\./, '');
    let id = '';
    if (host === 'youtube.com' || host === 'youtube-nocookie.com') {
        if (u.pathname === '/watch') {
            id = u.searchParams.get('v') || '';
        } else {
            const m = u.pathname.match(/^\/(shorts|live|embed|v)\/([\w-]+)/);
            if (m) id = m[2];
        }
        const list = u.searchParams.get('list') || '';
        if (!id && u.pathname === '/playlist' && /^[\w-]{10,}$/.test(list)) {
            return {key: 'list:' + list, kind: 'playlist', id: '',
                    url: 'https://www.youtube.com/playlist?list=' + list};
        }
    } else if (host === 'youtu.be') {
        const m = u.pathname.match(/^\/([\w-]+)/);
        if (m) id = m[1];
    } else {
        return null;
    }
    if (!/^[\w-]{11}$/.test(id)) return null;
    return {key: id, kind: 'video', id, url: 'https://www.youtube.com/watch?v=' + id};
};

/* The cards YouTube draws a video in — their title is the video's. */
WTM.CARDS = ['ytd-rich-item-renderer', 'ytd-video-renderer', 'ytd-compact-video-renderer',
             'ytd-grid-video-renderer', 'ytd-playlist-video-renderer', 'ytd-reel-item-renderer',
             'ytd-playlist-panel-video-renderer', 'yt-lockup-view-model', 'ytm-media-item',
             'ytd-playlist-renderer', 'ytd-radio-renderer'].join(', ');

// var, not const, like WTM above: background.js injects this file again into pages that have it already, and a
// second const in the same page stops the whole file at "Identifier has already been declared"
var WTM_GENERIC = /^(youtube( video)?( player)?|video|play|watch|thumbnail)?$/i;

function wtmText(s) {
    return String(s || '').replace(/\s+/g, ' ').trim();
}

/* What the page calls the video: the card's title, the link's own words,
 * the thumbnail's alt text, the player's title. */
WTM.title = function (el) {
    if (!el) return '';
    const tries = [];
    const card = el.closest ? el.closest(WTM.CARDS) : null;
    if (card && card.querySelector) {
        const t = card.querySelector('#video-title, #video-title-link, a#video-title, h3, .title');
        if (t) tries.push(t.getAttribute && t.getAttribute('title'), t.textContent);
    }
    tries.push(el.getAttribute && el.getAttribute('title'),
               el.getAttribute && el.getAttribute('aria-label'),
               el.textContent);
    const img = el.querySelector ? el.querySelector('img[alt]') : null;
    if (img) tries.push(img.getAttribute('alt'));
    for (const t of tries) {
        const s = wtmText(t);
        if (s && !WTM_GENERIC.test(s) && s.length > 2) return s.slice(0, 200);
    }
    return '';
};

/* Every video of the document: Map key → {key, kind, id, url, title, els}. */
WTM.scan = function (doc, loc) {
    doc = doc || document;
    loc = loc || (typeof location !== 'undefined' ? location : null);
    const found = new Map();
    const fromCard = new Set();          // keys whose title came from a card
    const add = (v, el, title) => {
        if (!v) return;
        let entry = found.get(v.key);
        if (!entry) {
            entry = {...v, title: '', els: []};
            found.set(v.key, entry);
        }
        if (el && !entry.els.includes(el)) entry.els.push(el);
        // A card's title beats any other: the end screen inside YouTube's
        // player comes first in the page and says "Title 5 hours, 26 minutes"
        const card = !!(el && el.closest && el.closest(WTM.CARDS));
        if (title && (!entry.title || (card && !fromCard.has(v.key)))) {
            entry.title = title;
            if (card) fromCard.add(v.key);
        }
    };
    const base = loc ? loc.href : undefined;
    for (const a of doc.querySelectorAll('a[href]')) {
        const href = a.getAttribute('href') || '';
        if (!/youtu/.test(href) && !/^\/(watch|shorts|live|playlist)/.test(href)) continue;
        const onYouTube = loc && /(^|\.)youtube\.com$/.test(loc.hostname || '');
        if (!onYouTube && !/youtu/.test(href)) continue;     // a relative link elsewhere
        add(WTM.parse(href, base), a, WTM.title(a));
    }
    for (const f of doc.querySelectorAll('iframe[src]')) {
        const v = WTM.parse(f.getAttribute('src') || '', base);
        add(v, f, WTM.title(f));
    }
    // A watch page: its own video, on the player — titled by the page's heading,
    // whatever its chapter links and time stamps said before
    if (loc && /(^|\.)youtube\.com$/.test(loc.hostname || '')) {
        const v = WTM.parse(loc.href);
        if (v && v.kind === 'video') {
            const player = doc.querySelector('#movie_player, ytd-player, video');
            const h1 = doc.querySelector('ytd-watch-metadata h1');
            const title = wtmText(h1 && h1.textContent)
                || wtmText((doc.title || '').replace(/^\(\d+\)\s*/, '').replace(/ - YouTube$/, ''));
            add(v, player, title);
            if (title) found.get(v.key).title = title;
        }
    }
    return found;
};

/* The element a video's badge goes on: the largest one that is visible. */
WTM.anchor = function (entry) {
    let best = null, area = 0;
    for (const el of entry.els) {
        if (!el || !el.isConnected) continue;
        const r = el.getBoundingClientRect();
        const a = r.width * r.height;
        if (a > area && r.width >= 24 && r.height >= 14) { best = el; area = a; }
    }
    return best;
};
