// ══════════════════════════════════════════════════════════════════════
//  Connection indicator (topbar): shows LLM and embedding status
//    green = configured & reachable
//    red   = configured, but not reachable
//    gray  = not configured
// ══════════════════════════════════════════════════════════════════════

let _connTimer = null;
let _connLast = null;

function _applyConnDot(dotId, pillId, info) {
    const dot = document.getElementById(dotId);
    const pill = document.getElementById(pillId);
    if (!dot || !pill) return;
    let state, title;
    if (info && info.reachable === false) {
        // BEFORE the configuration question. "Turned on, but the model file
        // is missing" reports `configured: false` — and thereby ended up in
        // the same gray state as "never set up". But the difference is
        // exactly what helps: gray means "set it up", red means "you did
        // set it up, but X is missing" — and names the X.
        state = 'err';
        title = `Not reachable${info.detail ? ' · ' + info.detail : ''}`
            + `\n${info.endpoint || ''}`;
    } else if (!info || !info.configured) {
        state = 'off';
        title = 'Not configured — click to set up';
    } else if (info.reachable === true) {
        state = 'on';
        title = `Connected${info.model ? ' · ' + info.model : ''}`
            + (info.detail ? ' · ' + info.detail : '')
            + `\n${info.endpoint || ''}`
            + `\nType: ${info.api_type === 'openai' ? 'OpenAI-compatible' : (info.api_type || '?')}`;
    } else if (info.ready) {
        // Own on-demand server: it's idle on purpose — after five minutes
        // without a request it frees its VRAM. That's normal operation,
        // not a failure. Orange ("checking…") would be a false signal here
        // that never resolves: it's only pinged once someone searches.
        state = 'on';
        title = `Ready${info.model ? ' · ' + info.model : ''}`
            + (info.detail ? '\n' + info.detail : '')
            + `\n${info.endpoint || ''}`;
    } else if (info.reachable === false) {
        state = 'err';
        title = `Not reachable${info.detail ? ' · ' + info.detail : ''}`
            + `\n${info.endpoint || ''}`;
    } else {
        // configured, but not (yet) checked
        state = 'unknown';
        title = `Configured${info.model ? ' · ' + info.model : ''} — checking…`;
    }
    dot.className = 'conn-dot ' + state;
    pill.title = title;
    pill.classList.toggle('configured', !!(info && info.configured));
}

async function refreshConnStatus() {
    try {
        const r = await fetch('/api/connection/status?sample=1');
        if (!r.ok) return;
        const d = await r.json();
        _connLast = d;
        _applyConnDot('connLlmDot', 'connLlm', d.llm);
        _applyConnDot('connEmbDot', 'connEmb', d.embedding);
    } catch(e) {
        // Network error → leave the indicator unchanged
    }
}

// Fast, network-free initial render (configuration status only),
// followed by the real sample refresh.
async function _initConnStatus() {
    try {
        const r = await fetch('/api/connection/status');  // without sample → fast
        if (r.ok) {
            const d = await r.json();
            _applyConnDot('connLlmDot', 'connLlm', d.llm);
            _applyConnDot('connEmbDot', 'connEmb', d.embedding);
        }
    } catch(e) {}
    refreshConnStatus();  // then actually ping
    if (_connTimer) clearInterval(_connTimer);
    _connTimer = setInterval(refreshConnStatus, 30000);  // every 30s
}

// Click on a pill → open settings on the matching tab
function openConnSettings(which) {
    if (typeof openSettings === 'function') openSettings(which === 'embedding' ? 'embedding' : 'llm');
}

document.addEventListener('DOMContentLoaded', () => {
    // Start delayed, so the first scan gets priority
    setTimeout(_initConnStatus, 1200);
});
