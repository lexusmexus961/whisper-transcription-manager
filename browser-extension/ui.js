/* ══════════════════════════════════════════════════════════════════════════
 *  What the popup and the side panel share
 *
 *  The states a video goes through, the server's answers (through
 *  background.js), pictures, and a summary's Markdown — rendered for
 *  reading, with every time in it ([0:04:12]) a link that plays the video
 *  from there and every reference of a chat answer ([S3]) a chip.
 * ══════════════════════════════════════════════════════════════════════════ */

const esc = s => String(s ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;')
    .replace(/>/g, '&gt;').replace(/"/g, '&quot;');

const STATE = {
    waiting: ['waiting', ''], resolving: ['looking up', ''], downloading: ['downloading', ''],
    queued: ['queued', ''], transcribing: ['transcribing', 'run'],
    transcribed: ['transcribed', 'ok'], summarizing: ['summarizing', 'run'],
    summarized: ['summarized', 'ok'], 'not queued': ['not queued', 'warn'],
    error: ['failed', 'err'], downloaded: ['downloaded', ''],
};
const WORKING = ['waiting', 'resolving', 'downloading', 'queued', 'transcribing', 'summarizing'];

/* What a list's "select all" ticks, said: the videos of this page (those the
 * filter shows) — never the batches sent. */
function pickAllLabel(shown, total, filtered) {
    if (shown.length && shown.every(v => v.selected)) {
        return filtered ? `unselect these ${shown.length}` : 'unselect all';
    }
    return filtered ? `select the ${shown.length} matching` : `select all ${total} videos on this page`;
}

const thumb = (id, size) => id ? `https://i.ytimg.com/vi/${encodeURIComponent(id)}/${size || 'default'}.jpg` : '';

/* The YouTube id of a watch link — "" for anything else. */
function videoIdOf(url) {
    const m = String(url || '').match(/[?&]v=([\w-]{11})/) || String(url || '').match(/youtu\.be\/([\w-]{11})/);
    return m ? m[1] : '';
}

/* A question to the server, through background.js. */
const ask = msg => new Promise(resolve => chrome.runtime.sendMessage(msg, r =>
    resolve(chrome.runtime.lastError ? {error: chrome.runtime.lastError.message} : (r || {}))));

function when(ms) {
    const d = new Date(ms);
    return d.toDateString() === new Date().toDateString()
        ? d.toLocaleTimeString([], {hour: '2-digit', minute: '2-digit'})
        : d.toLocaleDateString([], {day: '2-digit', month: '2-digit'});
}

/* "0:04:12" or "4:12" → seconds. */
function seconds(t) {
    return String(t).split(':').reduce((a, b) => a * 60 + Number(b), 0);
}

/* Seconds → "4:12", or "1:02:03" past an hour. */
function clock(sec) {
    sec = Math.max(0, Math.floor(Number(sec) || 0));
    const h = Math.floor(sec / 3600), m = Math.floor(sec / 60) % 60, s = sec % 60;
    return (h ? `${h}:${String(m).padStart(2, '0')}` : String(m)) + ':' + String(s).padStart(2, '0');
}

const TIME = /\[(\d{1,2}:\d{2}(?::\d{2})?)(?:\s*[–-]\s*\d{1,2}:\d{2}(?::\d{2})?)?\]/g;

/* What a line says inline: bold, italics, code, links — and times and
 * references. ``text`` is escaped already. */
function inline(text, refs) {
    return text
        .replace(/`([^`]+)`/g, '<code>$1</code>')
        .replace(/\*\*(.+?)\*\*/g, '<b>$1</b>')
        .replace(/(^|[\s(])\*([^*\s][^*]*?)\*(?=[\s).,;:!?]|$)/g, '$1<i>$2</i>')
        .replace(/\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>')
        .replace(TIME, (all, t) => `<a class="time" href="#" data-t="${seconds(t)}" title="Play from ${t}">${clock(seconds(t))}</a>`)
        .replace(/\[(S\d+)\]/g, (all, ref) => refs && refs[ref]
            ? `<a class="ref" href="#" data-ref="${ref}" title="${esc((refs[ref].text || '').slice(0, 160))}">${ref}</a>`
            : all);
}

/* A summary's or an answer's Markdown, for reading: headings, lists, quotes,
 * paragraphs. Everything is escaped first; what the model wrote never
 * becomes markup of its own. */
function md(text, refs) {
    const out = [];
    let list = '', para = [];
    const flush = () => {
        if (para.length) out.push(`<p>${para.map(l => inline(l, refs)).join('<br>')}</p>`);
        para = [];
        if (list) out.push(`</${list}>`);
        list = '';
    };
    for (const raw of esc(text || '').split('\n')) {
        const line = raw.trimEnd();
        let m;
        if (!line.trim()) { flush(); continue; }
        if ((m = line.match(/^#{1,6}\s+(.*)$/))) {
            flush();
            out.push(`<h4>${inline(m[1], refs)}</h4>`);
        } else if ((m = line.match(/^\s*[-*•]\s+(.*)$/)) || (m = line.match(/^\s*\d+[.)]\s+(.*)$/))) {
            const kind = /^\s*\d/.test(line) ? 'ol' : 'ul';
            if (para.length) { out.push(`<p>${para.map(l => inline(l, refs)).join('<br>')}</p>`); para = []; }
            if (list !== kind) {
                if (list) out.push(`</${list}>`);
                out.push(`<${kind}>`);
                list = kind;
            }
            out.push(`<li>${inline(m[1], refs)}</li>`);
        } else if ((m = line.match(/^&gt;\s?(.*)$/))) {
            flush();
            out.push(`<blockquote>${inline(m[1], refs)}</blockquote>`);
        } else if (/^(-{3,}|\*{3,}|_{3,})$/.test(line.trim())) {
            flush();
            out.push('<hr>');
        } else {
            if (list) { out.push(`</${list}>`); list = ''; }
            para.push(line);
        }
    }
    flush();
    return out.join('');
}

/* Play a video from a moment: in the tab that plays it already, else on
 * YouTube in a new tab. ``tab``: the tab looked at, ``playing``: the id of
 * the video it plays. */
function playAt(url, t, tab, playing) {
    const id = videoIdOf(url);
    if (tab && id && id === playing) {
        chrome.tabs.sendMessage(tab.id, {type: 'seek', t}, () => void chrome.runtime.lastError);
        return;
    }
    if (id) chrome.tabs.create({url: `https://www.youtube.com/watch?v=${id}&t=${Math.floor(t)}s`});
}
