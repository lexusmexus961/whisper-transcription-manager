/* ══════════════════════════════════════════════════════════════════════════
 *  CONTINUOUS SCANNING — the folder bar's "Auto" switch
 *
 *  An option for continuous scanning and automatic transcription of newly
 *  found files — off by default — with a top-bar button to switch it.
 *
 *  The work happens on the server (core/autoscan.py); this keeps the button
 *  in step: on/off, a round running, the last round, an error. A click
 *  switches and saves the setting — the same one as Library → Scanning.
 * ══════════════════════════════════════════════════════════════════════════ */

const AUTOSCAN = {status: null, timer: null, busy: false};

async function autoscanLoad() {
    try {
        AUTOSCAN.status = await (await fetch('/api/autoscan')).json();
    } catch (e) {
        AUTOSCAN.status = null;
    }
    autoscanRender();
    clearTimeout(AUTOSCAN.timer);
    // Faster while a round runs, then once a minute
    AUTOSCAN.timer = setTimeout(autoscanLoad, AUTOSCAN.status && AUTOSCAN.status.scanning ? 5000 : 60000);
}


function autoscanAgo(t) {
    if (!t) return '';
    const s = Math.max(0, Math.round(Date.now() / 1000 - t));
    return s < 90 ? `${s} s ago` : s < 5400 ? `${Math.round(s / 60)} min ago` : `${Math.round(s / 3600)} h ago`;
}

function autoscanSummary(st) {
    if (!st) return 'Continuous scanning: status not available';
    if (!st.enabled) {
        return 'Continuous scanning is off — click to watch the library for new files'
            + (st.transcribe ? ' and transcribe them' : '');
    }
    const lines = [`Continuous scanning is on — every ${st.interval_min} min`
        + (st.transcribe ? ', new files are transcribed' : ', new files join the library')];
    lines.push('Watching: ' + (st.folders || []).join(', '));
    if (st.scanning) lines.push('A round is running now …');
    else if (st.last_at) lines.push(`Last round ${autoscanAgo(st.last_at)}`
        + (st.next_at ? ` · next in ${Math.max(0, Math.round((st.next_at - Date.now() / 1000) / 60))} min` : ''));
    else lines.push('The first round takes note of what is there — only files that appear after it count as new.');
    const c = st.counts || {};
    if (c.queued || c.added || c.pending) {
        lines.push(`Found so far: ${(c.queued || 0)} queued, ${(c.added || 0)} added, ${(c.pending || 0)} waiting until copied`);
    }
    if (st.error) lines.push('Error: ' + st.error);
    lines.push('Click to switch off · settings: Library → Scanning');
    return lines.join('\n');
}

/* What continuous scanning watches, in a word for the button. */
function autoscanModeLabel(st) {
    if (!st || !st.enabled) return 'off';
    const n = (st.chosen || []).length;
    return n ? plural(n, 'folder') : 'library';
}

function autoscanRender() {
    const b = document.getElementById('btnAutoscan');
    if (!b) return;
    const st = AUTOSCAN.status;
    const on = !!(st && st.enabled);
    const label = b.querySelector('.tb-label');
    if (label) label.textContent = 'Auto · ' + autoscanModeLabel(st);
    b.classList.toggle('on', on);
    b.classList.toggle('scanning', !!(st && st.scanning));
    b.classList.toggle('err', !!(st && st.error));
    b.setAttribute('aria-pressed', on ? 'true' : 'false');
    b.title = autoscanSummary(st);
    // The settings card shows the same state
    const el = document.getElementById('setAutoscanState');
    if (el) el.textContent = autoscanSummary(st).split('\n').slice(0, -1).join(' · ');
}

if (typeof document !== 'undefined') {
    document.addEventListener('DOMContentLoaded', autoscanLoad);
}


// ── What it watches: off, the whole library, chosen folders ─────────────

async function autoscanSet(body) {
    try {
        const r = await fetch('/api/autoscan', {method: 'POST', headers: {'Content-Type': 'application/json'},
                                                 body: JSON.stringify(body)});
        const d = await r.json();
        if (!r.ok) throw new Error(d.error || `HTTP ${r.status}`);
        AUTOSCAN.status = d;
        autoscanRender();
        if (typeof Library !== 'undefined' && document.getElementById('statsOverlay')?.classList.contains('visible')) Library.refresh();
        return d;
    } catch (e) {
        if (typeof Toast !== 'undefined') Toast.error('Continuous scanning: ' + (e.message || e));
        return null;
    }
}

/* The folder bar's Auto button: a small menu. */
function autoscanMenu(ev) {
    ev && ev.stopPropagation();
    let m = document.getElementById('autoscanMenu');
    if (m && !m.hidden) { m.hidden = true; return; }
    const b = document.getElementById('btnAutoscan');
    if (!m) {
        m = document.createElement('div');
        m.id = 'autoscanMenu';
        m.className = 'bb-menu down as-menu';
        m.setAttribute('role', 'menu');
        b.parentElement.style.position = 'relative';
        b.insertAdjacentElement('afterend', m);
        document.addEventListener('click', e => { if (!e.target.closest('#autoscanMenu, #btnAutoscan')) m.hidden = true; });
    }
    const st = AUTOSCAN.status || {};
    const mode = st.mode || (st.enabled ? 'library' : 'off');
    const here = (document.getElementById('dirInput')?.value || '').trim().replace(/\/+$/, '');
    const chosen = st.chosen || [];
    const watchingHere = chosen.some(f => f.replace(/\/+$/, '') === here);
    const item = (label, call, on, title = '') => `<button class="bm-item${on ? ' on' : ''}" role="menuitemradio" aria-checked="${!!on}" onclick="document.getElementById('autoscanMenu').hidden=true;${call}"${title ? ` title="${esc(title)}"` : ''}>${on ? '● ' : '○ '}${esc(label)}</button>`;
    m.innerHTML = item('Off', "autoscanSet({mode: 'off'})", mode === 'off')
        + item('The whole library', "autoscanSet({mode: 'library'})", mode === 'library', 'Every folder of the library — on a big NAS a full round takes a while')
        + item(`Chosen folders${chosen.length ? ` (${chosen.length})` : ''}`, "autoscanSet({mode: 'folders'})", mode === 'folders', chosen.join('\n') || 'Watch a folder first')
        + '<div class="bm-sep"></div>'
        + (here ? (watchingHere
            ? `<button class="bm-item" onclick="document.getElementById('autoscanMenu').hidden=true;autoscanSet({unwatch: '${escJs(here)}'})">Stop watching this folder</button>`
            : `<button class="bm-item" onclick="document.getElementById('autoscanMenu').hidden=true;autoscanSet({watch: '${escJs(here)}'})" title="${esc(here)}">👁 Watch this folder</button>`) : '')
        + `<button class="bm-item" onclick="document.getElementById('autoscanMenu').hidden=true;openSettings('scanner')">Settings…</button>`;
    m.hidden = false;
}

/* The Library's button for the chosen folder. */
const Autoscan = {
    watchButton(abs, watched) {
        // escJs, as every other handler: JSON.stringify with only the quotes
        // replaced left "&" alone, and a folder named a&quot;+… ran as script
        const arg = `'${escJs(abs)}'`;
        if (watched === 'self') return `<button class="tb-btn" onclick="autoscanSet({unwatch: ${arg}})" title="New files here are no longer picked up by themselves"><span class="tb-label">👁 Stop watching</span></button>`;
        if (watched === 'parent') return `<button class="tb-btn" disabled title="A folder above it is watched — its new files are picked up already"><span class="tb-label">👁 Watched through a folder above</span></button>`;
        return `<button class="tb-btn" onclick="autoscanSet({watch: ${arg}})" title="New files in this folder and below are found and transcribed by themselves"><span class="tb-label">👁 Watch with auto-scan</span></button>`;
    },
};
