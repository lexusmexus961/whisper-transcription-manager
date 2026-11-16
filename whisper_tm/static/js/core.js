// ╔══════════════════════════════════════════════════════════════════╗
// ║  Whisper Transcription Manager — Frontend Core Utilities        ║
// ║                                                                  ║
// ║  Loaded FIRST (before state.js). Provides:                       ║
// ║    - Debug: Global debug logging system                          ║
// ║    - EventBus: Decoupled component communication                 ║
// ║    - Toast: User-facing notifications                            ║
// ║    - Global error handlers                                       ║
// ║    - Connection status indicator                                 ║
// ║    - API fetch wrapper with error handling                       ║
// ╚══════════════════════════════════════════════════════════════════╝

// ── Debug System ─────────────────────────────────────────────────────────
// Global debug logging to the browser console. Enable via Settings →
// Diagnostics → Debug output (stored as localStorage wtm_debug).
// Usage:  wtmDebug('context', 'message', {data});

let _wxpDebugEnabled = false;
try { _wxpDebugEnabled = !!localStorage.getItem('wtm_debug'); } catch(e) {}

function wtmDebug(context, ...args) {
    if (!_wxpDebugEnabled) return;
    const ts = new Date().toLocaleTimeString(undefined, {hour12: false, fractionalSecondDigits: 1});
    console.log(`%c[WTM ${ts}] %c${context}`, 'color:#f59e0b', 'color:#3b82f6', ...args);
}

// ── EventBus ─────────────────────────────────────────────────────────────
// Lightweight pub/sub for decoupled component communication.
// Usage:
//   EventBus.on('scan:complete', (data) => { ... });
//   EventBus.emit('scan:complete', {files: [...]});
//   EventBus.off('scan:complete', handler);

const EventBus = (() => {
    const _listeners = {};

    return {
        on(event, fn) {
            (_listeners[event] = _listeners[event] || []).push(fn);
        },
        off(event, fn) {
            const arr = _listeners[event];
            if (arr) _listeners[event] = arr.filter(f => f !== fn);
        },
        emit(event, ...args) {
            const arr = _listeners[event];
            if (arr) arr.forEach(fn => {
                try { fn(...args); } catch(e) { console.error(`EventBus[${event}]:`, e); }
            });
        },
        once(event, fn) {
            const wrapper = (...args) => { fn(...args); EventBus.off(event, wrapper); };
            EventBus.on(event, wrapper);
        },
    };
})();


// ── Toast Notifications ──────────────────────────────────────────────────
// Shows brief, non-blocking messages at the bottom of the screen.
// Usage:
//   Toast.success('Transcript saved');
//   Toast.error('Connection lost');
//   Toast.info('3 files added to queue');
//   Toast.warn('GPU memory low');

const Toast = (() => {
    let _container = null;
    const MAX_TOASTS = 5;
    const DEFAULT_DURATION = 4000;

    function _ensureContainer() {
        if (_container) return _container;
        _container = document.createElement('div');
        _container.id = 'toast-container';
        _container.style.cssText = `
            position: fixed; bottom: 20px; right: 20px; z-index: 10000;
            display: flex; flex-direction: column-reverse; gap: 8px;
            pointer-events: none; max-width: 420px;
        `;
        document.body.appendChild(_container);
        return _container;
    }

    function _show(message, type = 'info', duration = DEFAULT_DURATION, onClick = null) {
        const container = _ensureContainer();

        // Limit concurrent toasts
        while (container.children.length >= MAX_TOASTS) {
            container.removeChild(container.firstChild);
        }

        const colors = {
            success: { bg: '#22c55e20', border: '#22c55e', icon: '✓' },
            error:   { bg: '#ef444420', border: '#ef4444', icon: '✕' },
            warn:    { bg: '#f59e0b20', border: '#f59e0b', icon: '⚠' },
            info:    { bg: '#3b82f620', border: '#3b82f6', icon: 'ℹ' },
        };
        const c = colors[type] || colors.info;

        const el = document.createElement('div');
        el.style.cssText = `
            background: var(--bg-2, #13131c); border: 1px solid ${c.border};
            border-left: 3px solid ${c.border};
            border-radius: 8px; padding: 10px 14px; pointer-events: auto;
            font-family: var(--font-sans); font-size: 0.82rem;
            color: var(--tx-1, #e4e4ee); display: flex; align-items: center; gap: 8px;
            opacity: 0; transform: translateX(20px);
            transition: opacity 0.3s ease, transform 0.3s ease;
            box-shadow: 0 4px 16px #0004; cursor: pointer;
        `;
        // With onClick, the toast is an ACTION (e.g. "open finished file")
        // — the arrow shows it's clickable, the click both executes AND dismisses.
        el.innerHTML = `<span style="color:${c.border};font-size:1rem">${c.icon}</span>`
            + `<span>${_escHtml(message)}</span>`
            + (onClick ? `<span style="color:${c.border};font-weight:700;margin-left:2px">↗</span>` : '');
        el.addEventListener('click', () => {
            if (onClick) { try { onClick(); } catch (e) { /* Toast still dismisses */ } }
            _dismiss(el);
        });

        container.appendChild(el);

        // Animate in
        requestAnimationFrame(() => {
            el.style.opacity = '1';
            el.style.transform = 'translateX(0)';
        });

        // Auto-dismiss
        if (duration > 0) {
            setTimeout(() => _dismiss(el), duration);
        }

        return el;
    }

    function _dismiss(el) {
        el.style.opacity = '0';
        el.style.transform = 'translateX(20px)';
        setTimeout(() => {
            if (el.parentNode) el.parentNode.removeChild(el);
        }, 300);
    }

    function _escHtml(s) {
        // Quote-safe (like esc() in state.js) — toast text is technically
        // text context, but consistent escaping avoids surprises.
        return String(s ?? '')
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#39;');
    }

    return {
        success: (msg, dur, onClick) => _show(msg, 'success', dur, onClick),
        error:   (msg, dur, onClick) => _show(msg, 'error', dur || 6000, onClick),
        warn:    (msg, dur, onClick) => _show(msg, 'warn', dur, onClick),
        info:    (msg, dur, onClick) => _show(msg, 'info', dur, onClick),
    };
})();


// ── Global Error Handlers ────────────────────────────────────────────────

window.addEventListener('error', (e) => {
    console.error('Uncaught error:', e.error);
    // Don't show toast for script errors from extensions/3rd party
    if (e.filename && e.filename.includes('/static/js/')) {
        Toast.error(`Error: ${e.message}`);
    }
});

window.addEventListener('unhandledrejection', (e) => {
    console.error('Unhandled promise rejection:', e.reason);
    const msg = e.reason?.message || String(e.reason);
    // Filter out network errors (handled by connection status)
    //
    // Aborted media are also NOT an error: someone clicking the next audio
    // preview while the previous one is still loading aborts it — the browser
    // then reports "The fetching process for the media resource was aborted by
    // the user agent at the user's request", turning a perfectly normal user
    // action into a red error message. The real case (file corrupt, format
    // not playable) still reports itself via the element's own `error`
    // listener.
    const cancelled = e.reason?.name === 'AbortError'
        || msg.includes('aborted by the user agent')
        || msg.includes('interrupted by a new load request')
        || msg.includes('interrupted by a call to pause');
    if (!msg.includes('Failed to fetch') && !msg.includes('NetworkError')
            && !cancelled) {
        Toast.error(`Error: ${msg}`);
    }
});


// ── API Fetch Wrapper ────────────────────────────────────────────────────
// Usage:
//   const data = await api('/api/scan?recursive=true');
//   const data = await api('/api/transcribe', { method: 'POST', body: {...} });

async function api(url, options = {}) {
    const opts = {
        ...options,
        // Shallow-merge headers so a caller with its own headers doesn't lose
        // the default Content-Type (a deliberately set Content-Type — e.g.
        // for FormData/text — still wins).
        headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
    };
    if (opts.body && typeof opts.body === 'object' && !(opts.body instanceof FormData)) {
        opts.body = JSON.stringify(opts.body);
    }
    if (opts.body instanceof FormData) {
        delete opts.headers['Content-Type'];
    }

    const t0 = performance.now();
    wtmDebug('API', `→ ${opts.method || 'GET'} ${url}`);

    try {
        const resp = await fetch(url, opts);
        const dt = (performance.now() - t0).toFixed(0);
        if (!resp.ok) {
            let errMsg = `HTTP ${resp.status}`;
            try {
                const errData = await resp.json();
                errMsg = errData.error || errMsg;
            } catch(e) {}
            wtmDebug('API', `← ${resp.status} ${url} (${dt}ms) ERROR: ${errMsg}`);
            // Status on the error: the text is server prose ("Not found"),
            // the code is what a caller can reliably use to detect what happened.
            const err = new Error(errMsg);
            err.status = resp.status;
            throw err;
        }
        wtmDebug('API', `← ${resp.status} ${url} (${dt}ms)`);
        // 204/empty body: don't force JSON
        const text = await resp.text();
        if (!text) return {};
        try {
            return JSON.parse(text);
        } catch(e) {
            throw new Error('Invalid server response (not JSON)');
        }
    } catch(e) {
        if (e.name === 'TypeError' && (e.message.includes('fetch') || e.message.includes('NetworkError'))) {
            ConnectionStatus.setOffline();
            throw new Error('Server unreachable');
        }
        throw e;
    }
}


// ── DOM Helpers ──────────────────────────────────────────────────────────
// Short, safe access. $('id') = getElementById; $$('sel') = querySelector.
function $(id) { return document.getElementById(id); }
function $$(sel, root) { return (root || document).querySelector(sel); }


// ── Number Formatting ────────────────────────────────────────────────────
// The canonical formatters. Copies of these used to live in stats.js (three
// of them) and dashboard.js — with slightly different rounding, so the same
// file showed "1.4 GB" or "1,44 GB" depending on the view.

// Seconds as 1h 05m / 5m 30s / 42s. Deliberately without leading zeros on the
// largest unit — statistics are read, not typed.
function fmtDuration(sec) {
    sec = Math.max(0, Math.round(Number(sec) || 0));
    const h = Math.floor(sec / 3600);
    const m = Math.floor((sec % 3600) / 60);
    const s = sec % 60;
    if (h) return `${h}h ${String(m).padStart(2, '0')}m`;
    if (m) return `${m}m ${String(s).padStart(2, '0')}s`;
    return `${s}s`;
}

// Bytes in binary (1024-based), one decimal place from MB up. Below MB,
// decimal places are just noise.
function fmtBytes(bytes) {
    bytes = Number(bytes) || 0;
    const units = ['B', 'KB', 'MB', 'GB', 'TB', 'PB'];
    let i = 0;
    while (bytes >= 1024 && i < units.length - 1) { bytes /= 1024; i++; }
    return `${bytes.toFixed(i >= 2 ? 1 : 0)} ${units[i]}`;
}

// Thousands separator per the browser's locale (e.g. 1,234,567).
function fmtNumber(n) {
    return (Number(n) || 0).toLocaleString();
}

// A count with its noun: "1 window", "3 windows" — "1 errors" read wrong
function plural(n, one, many) {
    const k = Number(n) || 0;
    return `${k.toLocaleString()} ${k === 1 ? one : (many || one + 's')}`;
}

// A speed factor: "12×", "1.5×", "0.04×" — one decimal turned a slow run
// (a short file the model loading dominates) into "0.0×"
function fmtSpeed(x) {
    x = Number(x) || 0;
    return (x >= 10 ? x.toFixed(0) : x >= 0.1 ? x.toFixed(1) : x.toFixed(2)) + '×';
}


// ── Connection Status ────────────────────────────────────────────────────
// Shows a small indicator when connection to server is lost.
// Auto-reconnects and notifies when back online.

const ConnectionStatus = (() => {
    let _online = true;
    let _indicator = null;
    let _checkTimer = null;
    const CHECK_INTERVAL = 15000;

    function _ensureIndicator() {
        if (_indicator) return _indicator;
        _indicator = document.createElement('div');
        _indicator.id = 'connection-status';
        _indicator.style.cssText = `
            position: fixed; top: 8px; right: 16px; z-index: 10001;
            padding: 5px 12px; border-radius: 6px; font-size: 0.72rem;
            font-family: var(--font-sans); font-weight: 600;
            transition: opacity 0.3s, transform 0.3s;
            opacity: 0; transform: translateY(-10px);
            pointer-events: none;
        `;
        document.body.appendChild(_indicator);
        return _indicator;
    }

    function setOffline() {
        if (!_online) return;
        _online = false;
        const ind = _ensureIndicator();
        ind.textContent = '⚠ Offline';
        ind.style.background = 'var(--red, #ef4444)';
        ind.style.color = '#fff';
        ind.style.opacity = '1';
        ind.style.transform = 'translateY(0)';
        EventBus.emit('connection:offline');
        _startReconnect();
    }

    function setOnline() {
        if (_online) return;
        _online = true;
        const ind = _ensureIndicator();
        ind.textContent = '✓ Online';
        ind.style.background = 'var(--green, #22c55e)';
        ind.style.color = '#fff';
        ind.style.opacity = '1';
        ind.style.transform = 'translateY(0)';
        Toast.success('Connection restored');
        EventBus.emit('connection:online');

        // Fade out after 2s
        setTimeout(() => {
            ind.style.opacity = '0';
            ind.style.transform = 'translateY(-10px)';
        }, 2000);

        if (_checkTimer) {
            clearInterval(_checkTimer);
            _checkTimer = null;
        }
    }

    function _startReconnect() {
        if (_checkTimer) return;
        _checkTimer = setInterval(async () => {
            try {
                const resp = await fetch('/api/health', { method: 'GET', cache: 'no-cache' });
                if (resp.ok) setOnline();
            } catch(e) { /* still offline */ }
        }, CHECK_INTERVAL);
    }

    return { setOffline, setOnline, get isOnline() { return _online; } };
})();


// ── Esc closes the topmost dialog ────────────────────────────────────────
// Many dialogs registered their own Escape handler, some never did (the
// summary, the log). This is the fallback: if no handler closed anything,
// Esc presses the topmost open dialog's own Close/Cancel button — so its
// cleanup runs — or hides it. The visible set is noted in the capture
// phase and compared after all handlers ran, so one Esc never closes two
// dialogs.
(() => {
    const openDialogs = () => [...document.querySelectorAll('.modal-bg.visible')];
    let before = null;
    window.addEventListener('keydown', e => {
        if (e.key === 'Escape') before = openDialogs();
    }, true);
    window.addEventListener('keydown', e => {
        if (e.key !== 'Escape' || !before) return;
        const now = openDialogs();
        const handled = now.length !== before.length;
        before = null;
        if (handled || !now.length) return;
        const top = now[now.length - 1];
        const btn = [...top.querySelectorAll('button')].find(b =>
            b.offsetParent && /^(close|cancel|✕)$/i.test(b.textContent.trim()));
        if (btn) btn.click();
        else top.classList.remove('visible');
    });
})();
