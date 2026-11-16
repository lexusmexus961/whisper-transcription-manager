// ── Health check ─────────────────────────────────────────────────────
// A glance that answers "Is everything healthy?" — designed for
// long processing runs: GPU/VRAM, queue speed, error rate, DB, NAS,
// system warnings, prefilter balance. Data from ONE endpoint
// (/api/health/detail), auto-refresh every 5 s while the overlay is open.

let _hcTimer = null;
// One request at a time: when the server hangs (a stalled NAS mount),
// ticks must not pile up requests behind each other, and a late answer
// must not overwrite a newer one.
let _hcBusy = false;
const HC_TIMEOUT_MS = 8000;

/* The Machine window (machine.js), on its last tab or the one asked for. */
function openHealthView(tab) {
    document.getElementById('healthOverlay').classList.add('visible');
    _hcRefresh();
    if (!_hcTimer) _hcTimer = setInterval(_hcRefresh, 5000);
    if (typeof Machine !== 'undefined') Machine.tab(tab || Machine.last);
}

function closeHealthView() {
    document.getElementById('healthOverlay').classList.remove('visible');
    if (_hcTimer) { clearInterval(_hcTimer); _hcTimer = null; }
    if (typeof pvStop === 'function') pvStop();
}

document.addEventListener('keydown', e => {
    if (e.key === 'Escape'
        && document.getElementById('healthOverlay')?.classList.contains('visible')) {
        closeHealthView();
    }
});

// "Unload VRAM" in the top bar: frees every resident model right away
// instead of waiting for the idle unloader (hardware.model_idle_min) —
// e.g. before another application needs the same card. The server refuses while a
// transcription runs; the next job reloads the models by itself.
async function unloadVram(btn) {
    if (btn) btn.disabled = true;
    try {
        const r = await fetch('/api/vram/unload', { method: 'POST' });
        const d = await r.json().catch(() => ({}));
        if (!r.ok || !d.ok) {
            Toast.error(d.error || `Unload failed (HTTP ${r.status})`);
            return;
        }
        if (!d.unloaded) {
            Toast.info('No models loaded — VRAM is already free');
            return;
        }
        const used = d.vram_total_mb
            ? ` · now ${(d.vram_used_mb / 1024).toFixed(1)} / ${(d.vram_total_mb / 1024).toFixed(1)} GB in use`
            : '';
        Toast.success(`VRAM freed: ${(d.freed_mb / 1024).toFixed(1)} GB${used}`);
        if (document.getElementById('healthOverlay')?.classList.contains('visible')) _hcRefresh();
    } catch (e) {
        Toast.error('Unload failed: ' + (e.message || e));
    } finally {
        if (btn) btn.disabled = false;
    }
}

// ── Live load meters in the top bar ─────────────────────────────────────
// CPU, RAM, GPU, VRAM from /api/system/load every 2 s — only while the tab
// is visible, so a forgotten background tab costs the server nothing.
// A click on the meters opens the history (perfview.js).
const SL_INTERVAL = 2000;
let _slTimer = null;

function _slSet(id, pct, text, title) {
    const el = document.getElementById(id);
    if (!el) return;
    if (pct == null) { el.style.display = 'none'; return; }
    el.style.display = '';
    const p = Math.max(0, Math.min(100, pct));
    el.querySelector('.sl-fill').style.width = p.toFixed(0) + '%';
    el.querySelector('.sl-val').textContent = text;
    el.classList.toggle('high', p >= 75 && p < 92);
    el.classList.toggle('full', p >= 92);
    el.title = title;
}

async function _slTick() {
    try {
        const d = await (await fetch('/api/system/load')).json();
        const gb = mb => (mb / 1024).toFixed(1);
        _slSet('slCpu', d.cpu, `${d.cpu}%`, `CPU: ${d.cpu}% (all cores)`);
        _slSet('slRam', d.ram_total_mb ? d.ram_used_mb / d.ram_total_mb * 100 : null,
               `${gb(d.ram_used_mb)}G`, `RAM: ${gb(d.ram_used_mb)} of ${gb(d.ram_total_mb)} GB in use (whole system)`);
        const g = d.gpu;
        _slSet('slGpu', g && g.util != null ? g.util : null, g ? `${g.util}%` : '',
               g ? `GPU: ${g.util}% busy — ${g.name}` : '');
        _slSet('slVram', g && g.vram_total_mb ? g.vram_used_mb / g.vram_total_mb * 100 : null,
               g ? `${gb(g.vram_used_mb)}G` : '',
               g ? `VRAM: ${gb(g.vram_used_mb)} of ${gb(g.vram_total_mb)} GB in use (all processes) — ${g.name}` : '');
    } catch (e) { /* server gone — keep the last values */ }
}

function _slStart() {
    if (_slTimer || document.visibilityState !== 'visible') return;
    _slTick();
    _slTimer = setInterval(_slTick, SL_INTERVAL);
}

function _slStop() {
    if (_slTimer) { clearInterval(_slTimer); _slTimer = null; }
}

document.addEventListener('visibilitychange', () =>
    document.visibilityState === 'visible' ? _slStart() : _slStop());
document.addEventListener('DOMContentLoaded', _slStart);

const _hcFmtDuration = (s) => {
    if (s == null) return '—';
    const d = Math.floor(s / 86400), h = Math.floor((s % 86400) / 3600),
          m = Math.floor((s % 3600) / 60);
    if (d > 0) return `${d}d ${h}h`;
    if (h > 0) return `${h}h ${m}min`;
    return `${m}min`;
};
const _hcFmtGb = (mb) => (mb / 1024).toFixed(1) + ' GB';

// One tile: status color + title + big value + detail lines.
// A line cut off at the card's edge shows itself whole on hover (the
// parts are escaped already; only the markup comes off)
function _hcCard(status, title, value, lines) {
    return `<div class="hc-card ${status}">
        <div class="hc-card-head">${title}</div>
        <div class="hc-card-big">${value}</div>
        <div class="hc-card-lines">${(lines || []).map(z =>
            `<div title="${z.replace(/<[^>]*>/g, '').replace(/"/g, '&quot;')}">${z}</div>`).join('')}</div>
    </div>`;
}

async function _hcRefresh() {
    const body = document.getElementById('healthBody');
    if (!body || _hcBusy) return;
    _hcBusy = true;
    let d;
    try {
        const signal = typeof AbortSignal !== 'undefined' && AbortSignal.timeout
            ? AbortSignal.timeout(HC_TIMEOUT_MS) : undefined;
        const r = await fetch('/api/health/detail', { signal });
        d = await r.json();
        if (!r.ok) throw new Error(d.error || r.status);
    } catch (e) {
        _hcBusy = false;
        if (e && (e.name === 'TimeoutError' || e.name === 'AbortError')) {
            e = new Error(`no answer within ${HC_TIMEOUT_MS / 1000} s — the server or a mount hangs`);
        }
        body.innerHTML = `<div class="hc-card err" style="grid-column:1/-1">
            <div class="hc-card-head">Server</div>
            <div class="hc-card-big">unreachable</div>
            <div class="hc-card-lines"><div>${esc(String(e.message || e))}</div></div></div>`;
        return;
    }
    _hcBusy = false;
    // Closed while the request ran: leave the hidden view alone.
    if (!document.getElementById('healthOverlay')?.classList.contains('visible')) return;

    const maps = [];

    // ── GPU / VRAM ──
    const g = d.gpu || {};
    if (g.available) {
        const pct = g.vram_total_mb ? Math.round(g.vram_used_mb / g.vram_total_mb * 100) : 0;
        const status = pct > 92 ? 'warn' : 'ok';
        const models = (g.resident_models || []);
        maps.push(_hcCard(status, '🖥 GPU · ' + esc(g.name || ''),
            `${_hcFmtGb(g.vram_used_mb)} / ${_hcFmtGb(g.vram_total_mb)}`,
            [`<div class="hc-bar"><div style="width:${pct}%"></div></div>`,
             models.length
                 ? `Resident: ${models.map(esc).join(', ')}`
                 : 'No models loaded'
                   + (g.idle_min > 0 ? ` (Unloader: ${g.idle_min} min idle)` : ''),
            ]));
    } else {
        maps.push(_hcCard('warn', '🖥 GPU', 'none detected', []));
    }

    // ── Local LLM server (llama.cpp) ──
    // Only show when installed OR enabled — anyone using OpenAI/Ollama
    // doesn't need this card.
    // Always show: "not installed" is also information — and the
    // hint on how to add it belongs right here.
    const ll = d.llama || {};
    if (ll) {
        const st = ll.warnings?.length ? 'warn' : (ll.running ? 'ok' : 'neutral');
        const lines = [
            ll.model ? `Model: ${esc(ll.model)}${ll.model_gb ? ` (${ll.model_gb} GB)` : ''}` : '',
            `${plural(ll.threads, 'thread')} (${esc(ll.threads_reason || '')}) · context ${(ll.ctx_size || 0).toLocaleString()}`
              + ` · ubatch ${(ll.ubatch_size || 0).toLocaleString()}`
              + (ll.n_cpu_moe ? ` · ${plural(ll.n_cpu_moe, 'MoE layer')} on CPU` : ''),
            ll.vram_needed_mb ? `Estimated VRAM need ${_hcFmtGb(ll.vram_needed_mb)}`
                + (ll.vram_total_mb ? ` of ${_hcFmtGb(ll.vram_total_mb)}` : '') : '',
            ...(ll.warnings || []).map(w => `<span class="hc-red">⚠ ${esc(w)}</span>`),
            ...(ll.hints || []).map(h => `<span class="hc-hint">${esc(h)}</span>`),
        ].filter(Boolean);
        maps.push(_hcCard(ll.installed ? st : 'neutral',
            '🧠 Local LLM (llama.cpp)',
            ll.running ? '▶ running'
                      : (ll.installed ? 'installed, stopped' : 'not installed'),
            ll.installed ? lines
                : [(ll.hints || []).join(' ')
                   || 'Optional — the setup installs a local server (its llama option).']));
    }

    // ── Queue status + speed ──
    const q = d.queue || {};
    const t = q.speed || {};
    const qStatus = q.running ? 'ok' : (q.paused ? 'warn' : (q.pending > 0 ? 'warn' : 'ok'));
    const qValue = q.running ? '▶ running' : (q.paused ? '⏸ paused' : 'empty/done');
    const restH = (t.audio_h_per_h > 0 && q.pending > 0 && t.files_per_h > 0)
        ? q.pending / t.files_per_h : null;
    maps.push(_hcCard(qStatus, '📋 Queue', qValue, [
        `${(q.pending || 0).toLocaleString()} pending · ${(q.done || 0).toLocaleString()} done`
            + (q.error ? ` · <span class="hc-red">${plural(q.error, 'error')}</span>` : ''),
        // Queue lanes: the oldest running file, and how many run beside it
        q.current_file ? `⟳ ${esc(q.current_file)}`
            + (q.running_count > 1 ? ` +${q.running_count - 1} more` : '') : '',
    ].filter(Boolean)));

    const tStatus = !q.running ? 'neutral'
        : (t.x_realtime >= 15 ? 'ok' : (t.x_realtime > 0 ? 'warn' : 'neutral'));
    maps.push(_hcCard(tStatus, '⚡ Speed (last ' + (t.window_min || 60) + ' min)',
        t.x_realtime ? `${t.x_realtime}× realtime` : '—', [
        `${plural(t.files_per_h || 0, 'file')}/h · ${t.audio_h_per_h || 0} h audio per hour`,
        restH ? `≈ ${_hcFmtDuration(restH * 3600)} remaining at this speed` : '',
    ].filter(Boolean)));

    // ── Error rate ──
    const f = q.errors_24h || {};
    const fStatus = f.percent > 10 ? 'err' : (f.percent > 3 ? 'warn' : 'ok');
    maps.push(_hcCard(fStatus, '✕ Error rate (24 h)',
        (f.percent != null ? f.percent : 0) + ' %',
        [`${plural(f.error || 0, 'error')} out of ${plural((f.done || 0) + (f.error || 0), 'finished job')}`]));

    // ── DB ──
    const db = d.db || {};
    maps.push(_hcCard(db.ok ? 'ok' : 'err', '🗄 Database',
        db.ok ? 'reachable' : 'ERROR', [
        db.ok ? `${(db.size / 1e6).toFixed(0)} MB · schema v${db.schema_version}` : '',
        db.ok ? `Auto-backup: ${db.backup_counter}/${plural(db.backup_every, 'change')}` : '',
    ].filter(Boolean)));

    // ── NAS ──
    const n = d.nas || {};
    const nStatus = !n.ok ? 'err' : (n.latency_ms > 500 ? 'warn' : 'ok');
    maps.push(_hcCard(nStatus, '📡 Media directory',
        n.ok ? `${n.latency_ms} ms` : 'UNREACHABLE', [
        esc(n.path || ''),
        n.error ? esc(n.error) : '',
    ].filter(Boolean)));

    // ── Prefilter balance ──
    const v = d.prefilter || {};
    const vSum = (v.separate || 0) + (v.clean || 0) + (v.silence || 0);
    maps.push(_hcCard('neutral', '⚙ Prefilter (since server start)',
        vSum ? plural(vSum, 'chunk') : '—', [
        vSum ? `${(v.separate || 0).toLocaleString()} separated · `
               + `${(v.clean || 0).toLocaleString()} clean skipped · `
               + `${(v.silence || 0).toLocaleString()} silence` : 'No separation run yet',
    ]));

    // ── System warnings ──
    const w = d.warnings || [];
    maps.push(_hcCard(w.length ? 'err' : 'ok', '🔔 System warnings',
        w.length ? `${w.length} active` : 'none',
        w.slice(0, 3).map(x => esc(x.text || ''))));

    body.innerHTML = `<div class="hc-meta">Whisper Transcription Manager ${esc(d.version || '')}
        · Server running since ${_hcFmtDuration(d.uptime_s)}
        · As of ${new Date().toLocaleTimeString()}</div>
        <div class="hc-grid">${maps.join('')}</div>`;
}
