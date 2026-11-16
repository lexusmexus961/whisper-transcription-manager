// ── Live counters of the sidebar (server aggregates) + video statistics ────
// Since pagination (UI rework 5.1) the browser only ever sees one
// window of the list — the counters therefore come entirely from the server
// (/api/files/stats, ONE SQL pass over the directory). The large
// directory/global statistics still live in statsview.js (/api/stats).
let srvStats = null;        // last state from /api/files/stats
let _srvStatsTimer = null;  // throttle: at most 1 request per 3 s

function loadServerStats(immediately = false) {
    if (_srvStatsTimer && !immediately) return;   // already scheduled
    if (_srvStatsTimer) { clearTimeout(_srvStatsTimer); }
    _srvStatsTimer = setTimeout(async () => {
        _srvStatsTimer = null;
        try {
            const r = await fetch('/api/files/stats');
            const d = await r.json();
            // Do NOT check d.error: here that is the ERROR COUNT
            // (queue outcome), not an error message.
            if (r.ok && d && typeof d.total === 'number') {
                srvStats = d;
                // Index chip (📇) is likewise fed from the aggregates
                if (typeof _scanIndexInfo !== 'undefined') {
                    _scanIndexInfo = { total: d.total, indexed: d.indexed,
                                       unindexed: Math.max(0, d.total - d.indexed) };
                    // "Index metadata · N missing" in the ⋯ folder menu
                    if (typeof _setIndexButton === 'function'
                            && typeof _indexSSE !== 'undefined' && !_indexSSE) _setIndexButton(false);
                }
                updateCounts();
                if (typeof renderBatchButtons === 'function') renderBatchButtons();
                if (typeof renderDirStatusCounts === 'function') renderDirStatusCounts();
            }
        } catch (e) { /* server gone — counters stay put */ }
    }, immediately ? 0 : 3000);
}

// Segment counters are narrow (up to six segments in a 360 px sidebar):
// from 10,000 on they read "123.4k" — the exact number is in the tooltip.
function setCount(id, n) {
    const el = document.getElementById(id);
    if (!el) return;
    n = n || 0;
    el.textContent = n < 10000 ? n.toLocaleString()
        : n < 1e6 ? (n / 1000).toFixed(n < 100000 ? 1 : 0) + 'k'
        : (n / 1e6).toFixed(1) + 'M';
    el.title = n.toLocaleString();
}

function updateCounts() {
    // Guard: called from renderFileList among others — if the statistics DOM
    // is missing, do nothing instead of throwing on a null node access.
    // Guards on a sidebar chip: the former header with the same
    // three numbers no longer exists.
    if (!document.getElementById('tabAll')) return;
    // All counters from the server aggregates — zero state before the first response.
    const agg = srvStats || {};
    const total = agg.total || 0;
    const done = agg.done || 0;
    setCount('tabAll', total);
    setCount('tabDone', done);
    setCount('tabPending', agg.pending);
    setCount('tabProcessing', agg.processing);
    // Active chip only while something runs — like Error/Cancelled below.
    const procBtn = document.getElementById('tabProcessingBtn');
    if (procBtn) procBtn.style.display =
        ((agg.processing || 0) || activeTab === 'processing') ? '' : 'none';

    // Queue tab: pending entries + the files currently running (queue
    // lanes: several at once).
    // queuePending (server counter) instead of queueItems.length: the item list
    // is a PREVIEW capped at 200 — with a large queue the chip
    // would otherwise show "200".
    setCount('tabQueue', queuePending + queueRunningTotal());

    // Error/cancelled (queue outcome, scoped server-side to the current
    // directory). Chips only appear once there is something to show —
    // an empty error list doesn't need space in the sidebar.
    const qErr = agg.error || 0, qCancel = agg.cancelled || 0;
    const errBtn = document.getElementById('tabErrorBtn');
    const cancelBtn = document.getElementById('tabCancelledBtn');
    if (errBtn) {
        setCount('tabError', qErr);
        errBtn.style.display = (qErr || activeTab === 'error') ? '' : 'none';
    }
    if (cancelBtn) {
        setCount('tabCancelled', qCancel);
        cancelBtn.style.display = (qCancel || activeTab === 'cancelled') ? '' : 'none';
    }

    // Orphaned transcripts (file gone, transcript still there) — chip only when needed
    const orphanBtn = document.getElementById('tabOrphanBtn');
    if (orphanBtn) {
        setCount('tabOrphan', agg.orphan);
        orphanBtn.style.display = ((agg.orphan || 0) || activeTab === 'orphan') ? '' : 'none';
    }

    // The panel shows the queue while its list is open (click on ⏳ Queue),
    // otherwise the open folder.
    const queueMode = activeTab === 'queue';
    document.getElementById('dirStats')?.classList.toggle('queue-mode', queueMode);
    loadQueueEta();
    if (queueMode) { renderQueueStats(); return; }

    // Global dir stats
    const etaEl = document.getElementById('dsEta');
    if (!total) {
        document.getElementById('dsPctText').textContent = '0%';
        document.getElementById('dsBarFill').style.width = '0%';
        document.getElementById('dsCounts').innerHTML = '';
        document.getElementById('dsChips').innerHTML = '';
        if (etaEl) { etaEl.textContent = ''; etaEl.title = ''; }
        return;
    }

    const pct = total > 0 ? (done / total * 100) : 0;
    document.getElementById('dsPctText').textContent = pct.toFixed(0) + '%';
    document.getElementById('dsBarFill').style.width = pct.toFixed(1) + '%';

    const totalSize = agg.total_size || 0;
    const transcribedAudioSec = agg.transcribed_audio_sec || 0;
    const totalProcSec = agg.total_proc_sec || 0;

    // Same rules as the statistics view: from 48 h on whole hours with a
    // thousands separator ("12,345 h", not "12345h 26min").
    const fmtDur = (s) => {
        if (!s || s <= 0) return '0:00';
        const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60);
        if (h >= 48) return `${Math.round(s / 3600).toLocaleString()} h`;
        if (h > 0) return `${h}h ${m}min`;
        if (m > 0) return `${m}min`;
        return `${Math.round(s)}s`;
    };
    const fmtSize = (b) => {
        if (b > 1e12) return (b/1e12).toFixed(1) + ' TB';
        if (b > 1e9) return (b/1e9).toFixed(1) + ' GB';
        if (b > 1e6) return (b/1e6).toFixed(0) + ' MB';
        return (b/1e3).toFixed(0) + ' KB';
    };

    // Counts row
    const emptyCount = agg.empty_count || 0;
    const noAudioCount = agg.no_audio_count || 0;
    const manualSkipCount = (agg.skipped || 0) - noAudioCount;
    document.getElementById('dsCounts').innerHTML =
        `<span class="done-num" title="${done} transcribed">${done.toLocaleString()}</span>`
        + `<span class="total-num" title="${plural(total, 'file')} total">/ ${total.toLocaleString()}${total < 100000 ? ' done' : ''}</span>`;

    // Details (collapsed by default): exceptions first, then volume and pace.
    let chips = [];
    if (noAudioCount > 0) chips.push(`<span class="ds-chip red" title="${plural(noAudioCount, 'file')} without an audio track">⊘ ${noAudioCount.toLocaleString()} no audio</span>`);
    if (manualSkipCount > 0) chips.push(`<span class="ds-chip red" title="${manualSkipCount} manually skipped">⏭ ${manualSkipCount.toLocaleString()} skipped</span>`);
    if (emptyCount > 0) chips.push(`<span class="ds-chip red" title="${plural(emptyCount, 'transcript')} with 0 words (empty audio)">∅ ${emptyCount.toLocaleString()} empty</span>`);
    chips.push(`<span class="ds-chip" title="Total size of all files">${fmtSize(totalSize)}</span>`);
    if (transcribedAudioSec > 0) chips.push(`<span class="ds-chip blue" title="Total transcribed audio duration">🎵 ${fmtDur(transcribedAudioSec)}</span>`);
    if (totalProcSec > 0) chips.push(`<span class="ds-chip green" title="Total processing time">⚙ ${fmtDur(totalProcSec)}</span>`);
    if (transcribedAudioSec > 0 && totalProcSec > 0) {
        const speed = transcribedAudioSec / totalProcSec;
        chips.push(`<span class="ds-chip green" title="Average processing speed">⚡ ${fmtSpeed(speed)}</span>`);
    }

    // Words / segments
    const totalWords = agg.total_words || 0;
    const totalSegs = agg.total_segs || 0;
    if (totalWords > 0) {
        const fmtK = n => (typeof stvCompact === 'function' ? stvCompact(n) : n.toLocaleString());
        chips.push(`<span class="ds-chip" title="${plural(totalWords, 'word')} in ${plural(done, 'transcript')}">📝 ${fmtK(totalWords)} word${totalWords === 1 ? '' : 's'}</span>`);
    }
    if (totalSegs > 0) {
        chips.push(`<span class="ds-chip" title="${plural(totalSegs, 'segment')} total">💬 ${totalSegs.toLocaleString()} seg.</span>`);
    }

    // Total media duration from cached index metadata (media_duration_sec),
    // falling back to transcribed duration. The time left is the one figure
    // that stays visible when collapsed — it goes into the header, not a chip.
    let eta = '', etaTitle = '';
    const totalMediaDurSec = agg.total_media_dur_sec || 0;
    if (totalMediaDurSec > 0) {
        chips.push(`<span class="ds-chip accent" title="Total media duration (indexed)">🎬 ${fmtDur(totalMediaDurSec)}</span>`);
        // Share: only against what's still COMING (pending), not against
        // what was skipped.
        const pendingDurSec = agg.pending_media_dur_sec || 0;
        if (transcribedAudioSec > 0 && transcribedAudioSec + pendingDurSec > 0) {
            const durPct = (transcribedAudioSec / (transcribedAudioSec + pendingDurSec) * 100).toFixed(1);
            chips.push(`<span class="ds-chip" title="Share transcribed by duration (done ÷ done + pending; skipped doesn't count)">⏱ ${durPct}% by duration</span>`);
        }
    }
    // Time left: estimated server-side (core/eta.py) — a fixed cost per file
    // plus a share of the audio, at the pace of the last 30 days, and files
    // not indexed yet counted by their size. "+" only for files with
    // nothing to go by. It replaces "pending audio ÷ all-time speed", which
    // underestimated badly: it ignored a drop in pace, the fixed cost per
    // file, and files not indexed yet.
    const e = agg.eta;
    if (e && e.sec > 0) {
        eta = `~${_dsLeft(e.sec)} left${e.unknown ? '+' : ''}`;
        etaTitle = _libraryEtaTitle(e);
    }

    // Index coverage
    if (_scanIndexInfo && _scanIndexInfo.total > 0) {
        const ix = _scanIndexInfo;
        const iColor = ix.unindexed === 0 ? 'var(--green)' : 'var(--orange)';
        chips.push(`<span class="ds-chip" style="border-color:${iColor};color:${iColor}" title="Files with stored metadata in the DB">📇 ${ix.indexed.toLocaleString()}/${ix.total.toLocaleString()} indexed</span>`);
    }

    if (etaEl) { etaEl.textContent = eta; etaEl.title = etaTitle; }
    document.getElementById('dsChips').innerHTML = chips.join('');
    document.getElementById('dsChips').style.display = chips.length > 0 ? '' : 'none';

    // Indexing itself is started from the ⋯ folder menu in the dir bar —
    // the coverage chip above is the information, not a second button.
}

// ── Time left: shared formatting ───────────────────────────────────────

// Same rules as the statistics view: from 48 h on whole hours
function _dsDur(s) {
    if (!s || s <= 0) return '0:00';
    const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60);
    if (h >= 48) return `${Math.round(s / 3600).toLocaleString()} h`;
    if (h > 0) return `${h}h ${m}min`;
    if (m > 0) return `${m}min`;
    return `${Math.round(s)}s`;
}

// Short, for the header: from 10 h on the minutes are noise
function _dsLeft(s) {
    return s >= 36000 ? Math.round(s / 3600).toLocaleString() + 'h' : _dsDur(s);
}

// "2.0 s per file + audio ÷ 20.0" — the model in words
function _paceText(m) {
    if (!m || !m.b) return '';
    const ratio = `audio ÷ ${(1 / m.b).toFixed(1)}`;
    return m.rough ? ratio : `${m.a.toFixed(1)} s per file + ${ratio}`;
}

function _libraryEtaTitle(e) {
    const lines = [`Time left for ${plural(e.files, 'open file')}: ≈ ${_dsDur(e.sec)}`];
    if (e.known) lines.push(`• ${e.known.toLocaleString()} with a known duration — ${_dsDur(e.audio_sec)} of audio`);
    if (e.by_size) lines.push(`• ${e.by_size.toLocaleString()} not indexed yet — ≈ ${_dsDur(e.size_audio_sec)} by their file size (📇 Index metadata makes it exact)`);
    if (e.unknown) lines.push(`• ${e.unknown.toLocaleString()} with nothing to go by — not counted`);
    if (e.model) {
        const when = e.model.window_days ? `the last ${e.model.window_days} days` : 'all transcriptions';
        lines.push(`Pace of ${when}: ${_paceText(e.model)} (${plural(e.model.samples, 'transcription')})`);
    }
    return lines.join('\n');
}

// ── The queue in the progress panel ──────────────────────────────────────
// A click on ⏳ Queue turns the panel into the queue's: progress of the
// run, time left per kind of job, finish time, the last hour's pace. The
// estimate (core/eta.py) fits each kind from the run's own finished
// entries — voices-only runs cost a second or two per file plus a small
// share of the audio, transcriptions far more.

let queueEta = null;         // last /api/queue/eta answer
let _queueEtaAt = 0;
let _queueEtaBusy = false;

async function loadQueueEta(force = false) {
    // The server caches 20 s; outside the queue view only the ⏳ tooltip needs it
    const every = activeTab === 'queue' ? 20000 : 120000;
    if (_queueEtaBusy || (!force && Date.now() - _queueEtaAt < every)) return;
    _queueEtaBusy = true;
    try {
        const r = await fetch('/api/queue/eta');
        const d = await r.json();
        if (r.ok && d && d.ok) queueEta = d;
    } catch (e) { /* server gone — keep the last answer */ }
    finally { _queueEtaBusy = false; _queueEtaAt = Date.now(); }
    _queueTileTitle();
    if (activeTab === 'queue') renderQueueStats();
}

const _QUEUE_KINDS = {voices: ['🗣', 'voice'], transcription: ['🎙', 'transcription']};

function _finishText(ts) {
    const d = new Date(ts * 1000), now = new Date();
    const time = d.toLocaleTimeString([], {hour: '2-digit', minute: '2-digit'});
    const days = Math.round((new Date(d).setHours(0, 0, 0, 0) - new Date(now).setHours(0, 0, 0, 0)) / 86400000);
    if (days === 0) return `today ${time}`;
    if (days === 1) return `tomorrow ${time}`;
    return d.toLocaleDateString([], {weekday: 'short', day: 'numeric', month: 'short'}) + ' ' + time;
}

function _queueTileTitle() {
    const btn = document.getElementById('tabQueueBtn');
    if (!btn) return;
    const d = queueEta;
    let t = 'Files waiting in the transcription queue';
    if (d && d.sec != null && d.pending) {
        t += ` — ~${_dsLeft(d.sec)} left${d.partial ? '+' : ''}`
           + (d.paused ? ' (paused)' : d.finish_at ? `, done ${_finishText(d.finish_at)}` : '')
           + '. Click: the panel below shows the queue';
    }
    btn.title = t;
}

function renderQueueStats() {
    const pctEl = document.getElementById('dsPctText');
    const barEl = document.getElementById('dsBarFill');
    const countsEl = document.getElementById('dsCounts');
    const etaEl = document.getElementById('dsEta');
    const chipsEl = document.getElementById('dsChips');
    if (!pctEl || !barEl || !countsEl || !chipsEl) return;
    const d = queueEta;
    const set = (pct, counts, eta, etaTitle, chips) => {
        pctEl.textContent = pct;
        barEl.style.width = (parseFloat(pct) || 0) + '%';
        countsEl.innerHTML = counts;
        if (etaEl) { etaEl.textContent = eta; etaEl.title = etaTitle; }
        chipsEl.innerHTML = chips.join('');
        chipsEl.style.display = chips.length ? '' : 'none';
    };
    if (!d) { set('…', '<span class="total-num">⏳ Loading the queue…</span>', '', '', []); return; }
    const total = d.total || 0, done = d.processed || 0;
    if (!d.pending && !d.running) {
        set('—', `<span class="total-num">⏳ Queue is empty${total ? ` — ${total.toLocaleString()} done in the last run` : ''}</span>`, '', '', []);
        return;
    }
    const pct = total ? done / total * 100 : 0;
    const counts = `<span class="done-num" title="${done.toLocaleString()} of this run processed">⏳ ${done.toLocaleString()}</span>`
                 + `<span class="total-num">/ ${total.toLocaleString()} in the queue</span>`;
    let eta = '', etaTitle = '';
    if (d.sec != null) {
        eta = `~${_dsLeft(d.sec)} left${d.partial ? '+' : ''}`;
        etaTitle = `Time left for ${plural(d.pending, 'waiting entry', 'waiting entries')}: ≈ ${_dsDur(d.sec)}`
                 + (d.partial ? '\n+ a kind of job with no estimate yet — it comes once some of them are done' : '')
                 + (d.paused ? '\nThe queue is paused' : d.finish_at ? `\nDone ${_finishText(d.finish_at)}` : '');
    } else if (d.pending) {
        eta = 'estimating…';
        etaTitle = 'No estimate yet: it comes from this run\'s own finished entries';
    }
    const chips = [];
    for (const k of d.kinds || []) {
        const [icon, label] = _QUEUE_KINDS[k.kind] || ['•', k.kind];
        const left = k.sec != null ? ` · ~${_dsLeft(k.sec)}` : '';
        const from = k.source === 'library' ? 'the library\'s transcriptions (none in this run yet)'
                   : `the last ${k.model ? k.model.samples.toLocaleString() : 0} finished in this run`;
        chips.push(`<span class="ds-chip accent" title="${k.files.toLocaleString()} waiting, ${_dsDur(k.audio_sec + k.size_audio_sec)} of audio${k.model ? `\nPace: ${_paceText(k.model)} — from ${from}` : ''}">${icon} ${plural(k.files, label)}${left}</span>`);
    }
    if (d.paused) chips.push('<span class="ds-chip red" title="Resume the queue to continue">⏸ paused</span>');
    else if (d.finish_at) chips.push(`<span class="ds-chip green" title="If it keeps this pace">🏁 done ${_finishText(d.finish_at)}</span>`);
    const lh = d.last_hour || {};
    if (lh.files) chips.push(`<span class="ds-chip" title="Finished in the last 60 minutes">⚡ ${plural(lh.files, 'file')} · ${_dsDur(lh.audio_sec)} audio / h</span>`);
    for (const k of d.kinds || []) {
        if (k.by_size) chips.push(`<span class="ds-chip" title="Not indexed yet — their duration comes from the file size">📏 ${k.by_size.toLocaleString()} by file size</span>`);
    }
    if (d.errors) chips.push(`<span class="ds-chip red" title="Failed in this run — the Error chip above lists them">✗ ${d.errors.toLocaleString()} failed</span>`);
    set(pct >= 10 || pct === 0 ? pct.toFixed(0) + '%' : pct.toFixed(1) + '%', counts, eta, etaTitle, chips);
}

// Details of the progress panel: collapsed by default, remembered per browser.
const DS_EXPANDED_KEY = 'wtm_dirStatsExpanded';

function setDirStatsExpanded(open) {
    const det = document.getElementById('dsDetails');
    const btn = document.getElementById('dsToggle');
    if (!det || !btn) return;
    det.hidden = !open;
    btn.setAttribute('aria-expanded', open ? 'true' : 'false');
    document.getElementById('dirStats')?.classList.toggle('expanded', open);
}

function toggleDirStats() {
    const open = document.getElementById('dsDetails')?.hidden;
    setDirStatsExpanded(!!open);
    try { localStorage.setItem(DS_EXPANDED_KEY, open ? '1' : '0'); } catch (e) { /* storage off */ }
}

document.addEventListener('DOMContentLoaded', () => {
    let open = false;
    try { open = localStorage.getItem(DS_EXPANDED_KEY) === '1'; } catch (e) { /* storage off */ }
    setDirStatsExpanded(open);
});

// ── Transcript statistics ────────────────────────────────────────────────
// The 📊 view of ONE transcript, in the same building blocks as the library
// statistics (statsview.js/.css): who speaks when (one lane per speaker,
// click to jump), the speakers side by side, how the file was processed,
// and what is said. Everything is computed from the loaded transcript —
// no request — except the Conversation card, which shows what the server
// counted (txsConvLoad).

/* The per-recording statistics into a box — the recording's Statistics
   tab. ``beforeSeek``: what to do before a number jumps into the
   recording. */
function txsRenderInto(body, beforeSeek) {
    const d = activeTranscript;
    if (!d || !body) return;
    const fm = (typeof fileByName === 'function' ? fileByName(activeFile) : null)
            || files.find(f => f.name === activeFile) || {};
    const s = txsCompute(d);
    body.innerHTML = `<div class="stv-grid">
        ${txsHero(d, fm, s)}
        ${txsConvCard()}
        ${txsTimelineCard(s)}
        ${txsSpeakers(s)}
        ${txsRhythm(s)}
        ${txsProcessing(d, s)}
        ${txsFile(d, fm)}
        ${txsWords(s)}
    </div>`;
    body.scrollTop = 0;
    txsDrawTimeline(s);
    body.querySelectorAll('[data-seek]').forEach(el => el.addEventListener('click', () => {
        if (beforeSeek) beforeSeek();
        if (typeof seekTo === 'function') seekTo(+el.dataset.seek);
    }));
    body._stats = s;
    // One tooltip for the statistics (statsview.js), wired once per box
    if (!body.dataset.tips && typeof stvWireTips === 'function') { stvWireTips(body); body.dataset.tips = '1'; }
    txsConvLoad(txsConvPath());
}

/* Before a jump into the recording: the statistics' tooltip goes. */
function closeTxStats() {
    if (typeof stvHideTip === 'function') stvHideTip();
}

// Everything derived from the segments, once.
function txsCompute(d) {
    const segs = (d.segments || []).filter(x => x.end > x.start);
    // The stored duration can be a little shorter than the last segment's
    // end — take whichever is longer, or segments fall off the timeline.
    const total = Math.max(d.total_duration_sec || 0, segs.reduce((m, x) => Math.max(m, x.end), 0));
    const order = (d.speakers && d.speakers.length) ? d.speakers
        : [...new Set(segs.map(x => x.speaker || '—'))];
    const idx = Object.fromEntries(order.map((sp, i) => [sp, i]));
    const wordsOf = t => (t || '').split(/\s+/).filter(Boolean).length;
    const per = {};
    let speech = 0, words = 0, longestPause = { dur: 0, at: 0 }, longestTurn = { dur: 0 };
    let turns = 0, prevSp = null, turnStart = 0, turnEnd = 0, turnSp = null;
    segs.forEach((x, i) => {
        const sp = x.speaker || '—';
        const dur = x.end - x.start, w = wordsOf(x.text);
        const p = per[sp] ??= { id: sp, dur: 0, segs: 0, words: 0, first: x.start, last: x.end, turns: 0 };
        p.dur += dur; p.segs++; p.words += w; p.last = x.end;
        speech += dur; words += w;
        if (i > 0) {
            const gap = x.start - segs[i - 1].end;
            if (gap > longestPause.dur) longestPause = { dur: gap, at: segs[i - 1].end };
        }
        if (sp !== prevSp) {
            if (turnSp !== null && turnEnd - turnStart > longestTurn.dur)
                longestTurn = { dur: turnEnd - turnStart, sp: turnSp, at: turnStart };
            turns++; p.turns++; turnStart = x.start; turnSp = sp;
        }
        turnEnd = x.end; prevSp = sp;
    });
    if (turnSp !== null && turnEnd - turnStart > longestTurn.dur)
        longestTurn = { dur: turnEnd - turnStart, sp: turnSp, at: turnStart };
    const speakers = Object.values(per).sort((a, b) => b.dur - a.dur)
        .map(p => ({ ...p, color: SPEAKER_COLORS[(idx[p.id] ?? 0) % SPEAKER_COLORS.length],
                     name: p.id === '—' ? 'No speaker' : getSpeakerName(p.id) }));
    // Speech per minute-bin for the density strip
    const nBins = Math.max(1, Math.min(120, Math.ceil(total / 30)));
    const binSec = total / nBins || 1;
    const density = new Array(nBins).fill(0);
    segs.forEach(x => {
        // Spread each segment over the bins it covers
        const first = Math.min(nBins - 1, Math.floor(x.start / binSec));
        const last = Math.min(nBins - 1, Math.floor(x.end / binSec));
        for (let b = first; b <= last; b++) {
            const lo = Math.max(x.start, b * binSec);
            const hi = b === last ? x.end : Math.min(x.end, (b + 1) * binSec);
            if (hi > lo) density[b] += hi - lo;
        }
    });
    // Segment lengths
    const lenBuckets = [['< 2 s', 0, 2], ['2–5 s', 2, 5], ['5–10 s', 5, 10], ['10–20 s', 10, 20],
                        ['20–30 s', 20, 30], ['> 30 s', 30, Infinity]]
        .map(([label, lo, hi]) => ({ label, value: segs.filter(x => { const l = x.end - x.start; return l >= lo && l < hi; }).length }));
    // Words (client-side, same stopwords idea as the server: content words only).
    // Letters of every script — [a-zß-ž] left a Russian, Greek or Hindi
    // transcript without a single word
    const stop = TXS_STOP;
    const counts = {};
    const lettered = t => (t || '').toLowerCase().match(/[\p{L}\p{M}]+/gu) || [];
    segs.forEach(x => lettered(x.text).forEach(w => {
        if (w.length >= 3 && !stop.has(w)) counts[w] = (counts[w] || 0) + 1;
    }));
    const topWords = Object.entries(counts).sort((a, b) => b[1] - a[1]).slice(0, 16)
        .map(([word, cnt]) => ({ word, cnt }));
    const allTokens = segs.flatMap(x => lettered(x.text));
    const longestSeg = segs.reduce((b, x) => wordsOf(x.text) > (b.w || 0) ? { w: wordsOf(x.text), x } : b, {});
    return { segs, total, speech, words, speakers, turns, longestPause, longestTurn,
             density, binSec, lenBuckets, topWords, vocab: new Set(allTokens).size,
             tokens: allTokens.length, longestSeg };
}

// Function words and fillers the top list should skip (German + English).
const TXS_STOP = new Set(('der die das ein eine einen einem eines und oder aber ist sind war waren hat haben habe hab hast '
    + 'bin bist wird werden werde würde kann können muss soll will mit von für auf aus bei nach über unter durch als wie '
    + 'dass wenn weil auch noch schon dann nur nicht kein keine sehr mehr hier dort jetzt immer alle alles man sich mich '
    + 'dich dir mir uns euch ihm ihn ihr ihnen sie wir ich du er es mein dein sein dem den des diese dieser dieses was wer '
    + 'mal zum zur vom bis doch also eben halt gibt ganz gerade eigentlich nein ja beim einer so da zu in an am im denn wo '
    + 'einfach irgendwie vielleicht wirklich genau okay gut bisschen wieder gleich erst etwas nichts viel viele ähm äh hmm '
    + 'mhm naja ach oh nee ne jo hey vor gar kommt jeden damit dabei davon dazu natürlich ob ohne weiß glaube finde denke '
    + 'meine machen macht gemacht geht sagen sagt gesagt the and that this with for are was were have has had will would '
    + 'can not but from they them their there here been just about also more some than then when what which who how all '
    + 'you your she her his its our out did does these those yeah like get got know think really very now because '
    + 'see one lot before after able during still well want need going make way thing things something right back '
    + 'even first two over into only any may might must much many down off again same look let say said sure yes '
    + 'actually maybe gonna wanna kind sort okay alright through another sehen sagen schauen').split(' '));

function txsFmtClock(sec) {
    sec = Math.max(0, Math.round(sec));
    const h = Math.floor(sec / 3600), m = Math.floor(sec % 3600 / 60), s = sec % 60;
    return h ? `${h}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}` : `${m}:${String(s).padStart(2, '0')}`;
}

function txsHero(d, fm, s) {
    const pct = s.total ? s.speech / s.total * 100 : 0;
    const wpm = s.speech ? s.words / (s.speech / 60) : 0;
    const speed = d.processing_secs && s.total ? s.total / d.processing_secs : 0;
    return `<section class="stv-card span-2 stv-hero">
        <div class="stv-hero-main">
            <div class="stv-hero-label">Duration</div>
            <div class="stv-hero-value">${esc(d.total_duration || txsFmtClock(s.total))}</div>
            <div class="stv-hero-sub">${esc((d.language || '?').toUpperCase())} · ${esc(d.model || '—')}
                ${fm.size_fmt ? ` · ${esc(fm.size_fmt)}` : ''}</div>
            <div class="stv-split">
                <span class="stv-split-seg pending" style="flex:${s.speech}" data-tip="${esc(`Speech\n${fmtDuration(s.speech)} · ${pct.toFixed(0)}%`)}"></span>
                <span class="stv-split-seg rest" style="flex:${Math.max(0, s.total - s.speech)}" data-tip="${esc(`Silence and pauses\n${fmtDuration(Math.max(0, s.total - s.speech))} · ${(100 - pct).toFixed(0)}%`)}"></span>
            </div>
            <div class="stv-legend"><span class="stv-legend-item"><i class="stv-dot pending"></i>Speech <b>${pct.toFixed(0)}%</b></span>
                <span class="stv-legend-item"><i class="stv-dot" style="background:var(--bg-4)"></i>Pauses <b>${(100 - pct).toFixed(0)}%</b></span></div>
        </div>
        <div class="stv-kpis">
            ${stvKpi('Words', fmtNumber(s.words),
                     // no speech time (segments without length): no rate
                     wpm ? `${wpm.toFixed(0)} per minute of speech` : '')}
            ${stvKpi('Speakers', fmtNumber(s.speakers.filter(x => x.id !== '—').length || s.speakers.length),
                     `${plural(Math.max(0, s.turns - 1), 'change')} of speaker`)}
            ${stvKpi('Segments', fmtNumber(s.segs.length), s.segs.length ? `⌀ ${(s.speech / s.segs.length).toFixed(1)} s each` : '')}
            ${stvKpi('Processing', esc(d.processing_time || fmtDuration(d.processing_secs)),
                     speed ? `${fmtSpeed(speed)} real-time` : '')}
        </div>
    </section>`;
}

// Who speaks when: one lane per speaker, a density strip below.
function txsTimelineCard(s) {
    if (!s.segs.length) return '';
    const legend = s.speakers.slice(0, 8).map(p => `<span class="stv-legend-item">
        <i class="stv-dot" style="background:${p.color}"></i>${esc(p.name)}</span>`).join('');
    return stvCard('Who speaks when', `<div class="stv-legend" style="margin-bottom:10px">${legend}</div>
        <div class="txs-timeline" id="txsTimeline"></div>`,
        { span: 2, sub: 'click anywhere to jump there' });
}

function txsDrawTimeline(s) {
    const box = document.getElementById('txsTimeline');
    if (!box || !s.total) return;
    const lanes = s.speakers.slice(0, 8);
    const W = Math.max(320, box.clientWidth), laneH = 18, gap = 6, left = 120, right = 6;
    const densH = 34, axisH = 20;
    const H = lanes.length * (laneH + gap) + densH + axisH + 8;
    const pw = W - left - right;
    const x = t => left + t / s.total * pw;
    let svg = '';
    lanes.forEach((p, i) => {
        const y = i * (laneH + gap);
        svg += `<rect class="txs-lane" x="${left}" y="${y}" width="${pw}" height="${laneH}" rx="3"/>`;
        const name = p.name.length > 16 ? p.name.slice(0, 15) + '…' : p.name;
        svg += `<text class="stv-axis txs-lane-label" x="${left - 10}" y="${y + laneH / 2 + 4}" text-anchor="end">${esc(name)}</text>`;
        // Merge segments closer than a pixel — a 6-hour file has thousands
        let run = null;
        const flush = () => { if (run) svg += `<rect x="${run.a}" y="${y + 2}" width="${Math.max(1.5, run.b - run.a)}" height="${laneH - 4}" rx="2" style="fill:${p.color}"/>`; };
        s.segs.forEach(seg => {
            if ((seg.speaker || '—') !== p.id) return;
            const a = x(seg.start), b = x(seg.end);
            if (run && a - run.b < 1) run.b = Math.max(run.b, b);
            else { flush(); run = { a, b }; }
        });
        flush();
    });
    // Density: speech seconds per bin, one hue
    const dy = lanes.length * (laneH + gap) + 4;
    const maxD = Math.max(...s.density, 1);
    const bw = pw / s.density.length;
    svg += `<text class="stv-axis" x="${left - 10}" y="${dy + densH / 2 + 4}" text-anchor="end">speech</text>`;
    s.density.forEach((v, i) => {
        const h = v / maxD * densH;
        const from = i * s.binSec, to = (i + 1) * s.binSec;
        svg += `<g data-tip="${esc(`${txsFmtClock(from)} – ${txsFmtClock(to)}\n${Math.round(v / (to - from) * 100)}% speech`)}">
            <rect class="stv-hit" x="${left + i * bw}" y="${dy}" width="${bw}" height="${densH}"/>
            <rect class="stv-col" x="${left + i * bw + 0.5}" y="${dy + densH - h}" width="${Math.max(1, bw - 1)}" height="${h}"/></g>`;
    });
    // Time axis
    const step = [60, 300, 600, 900, 1800, 3600, 7200, 10800].find(st => s.total / st <= 8) || 21600;
    for (let t = 0; t <= s.total; t += step) {
        svg += `<line class="stv-grid-line" x1="${x(t)}" x2="${x(t)}" y1="0" y2="${dy + densH}"/>`
             + `<text class="stv-axis" x="${x(t)}" y="${H - 4}" text-anchor="middle">${txsFmtClock(t)}</text>`;
    }
    svg += `<line class="txs-cursor" y1="0" y2="${dy + densH}" visibility="hidden"/>`;
    box.innerHTML = `<svg width="${W}" height="${H}">${svg}</svg><div class="txs-time" hidden></div>`;
    const svgEl = box.querySelector('svg'), cursor = box.querySelector('.txs-cursor'), label = box.querySelector('.txs-time');
    const tAt = e => {
        const r = svgEl.getBoundingClientRect();
        const px = e.clientX - r.left;
        return px < left ? null : Math.min(s.total, Math.max(0, (px - left) / pw * s.total));
    };
    svgEl.addEventListener('pointermove', e => {
        const t = tAt(e);
        if (t == null) { cursor.setAttribute('visibility', 'hidden'); label.hidden = true; return; }
        cursor.setAttribute('x1', x(t)); cursor.setAttribute('x2', x(t));
        cursor.setAttribute('visibility', 'visible');
        label.hidden = false;
        label.textContent = txsFmtClock(t);
        label.style.left = `${x(t)}px`;
    });
    svgEl.addEventListener('pointerleave', () => { cursor.setAttribute('visibility', 'hidden'); label.hidden = true; });
    svgEl.addEventListener('click', e => {
        const t = tAt(e);
        if (t == null) return;
        closeTxStats();
        if (typeof seekTo === 'function') seekTo(t);
    });
}

function txsSpeakers(s) {
    const list = s.speakers;
    if (!list.length) return '';
    const max = Math.max(...list.map(p => p.dur), 1);
    const rows = list.map(p => {
        const pct = s.speech ? p.dur / s.speech * 100 : 0;
        const wpm = p.dur ? p.words / (p.dur / 60) : 0;
        return `<tr>
            <td class="name"><i class="stv-dot" style="background:${p.color};margin-right:8px"></i>${esc(p.name)}</td>
            <td class="prog" data-tip="${esc(`${p.name}\n${fmtDuration(p.dur)} of speech · ${pct.toFixed(1)}%`)}">
                <span class="stv-mini"><span style="width:${(p.dur / max * 100).toFixed(1)}%;background:${p.color}"></span></span>
                <span class="stv-mini-val">${pct.toFixed(0)}%</span></td>
            <td class="num">${fmtDuration(p.dur)}</td>
            <td class="num">${fmtNumber(p.words)}</td>
            <td class="num">${wpm.toFixed(0)}</td>
            <td class="num">${fmtNumber(p.turns)}</td>
            <td class="num"><button class="txs-link" data-seek="${p.first}" title="Jump to the first time">${txsFmtClock(p.first)}</button></td>
        </tr>`;
    }).join('');
    return stvCard('Speakers', `<div class="stv-table-wrap"><table class="stv-table">
        <thead><tr><th>Speaker</th><th>Share of speech</th><th class="num">Time</th><th class="num">Words</th>
            <th class="num">wpm</th><th class="num">Turns</th><th class="num">First heard</th></tr></thead>
        <tbody>${rows}</tbody></table></div>`, { span: 2 });
}

function txsRhythm(s) {
    if (!s.segs.length) return '';
    const facts = [
        s.longestTurn.dur ? `<button class="txs-fact" data-seek="${s.longestTurn.at}">
            <span>Longest monologue</span><b>${fmtDuration(s.longestTurn.dur)}</b>
            <small>${esc(getSpeakerName(s.longestTurn.sp))} at ${txsFmtClock(s.longestTurn.at)}</small></button>` : '',
        s.longestPause.dur ? `<button class="txs-fact" data-seek="${s.longestPause.at}">
            <span>Longest pause</span><b>${fmtDuration(s.longestPause.dur)}</b>
            <small>at ${txsFmtClock(s.longestPause.at)}</small></button>` : '',
        s.longestSeg.w ? `<button class="txs-fact" data-seek="${s.longestSeg.x.start}">
            <span>Longest segment</span><b>${fmtNumber(s.longestSeg.w)} word${s.longestSeg.w === 1 ? '' : 's'}</b>
            <small>at ${txsFmtClock(s.longestSeg.x.start)}</small></button>` : '',
    ].join('');
    return stvCard('Rhythm', `<div class="txs-facts">${facts}</div>
        <div class="stv-sublabel" style="margin-top:16px">Segment length</div>
        ${stvBars(s.lenBuckets, { right: r => fmtNumber(r.value), tip: r => `${r.label}\n${fmtNumber(r.value)} segment${r.value === 1 ? '' : 's'}` })}`,
        { sub: 'turns, pauses, segment lengths' });
}

function txsProcessing(d, s) {
    const steps = (d.step_timings || []).filter(t => !t.group && t.secs > 0);
    const total = d.processing_secs || steps.reduce((a, t) => a + t.secs, 0);
    const phaseOf = label => /vocal|separation/i.test(label) ? 'Vocal isolation'
        : /align/i.test(label) ? 'Alignment'
        : /diari|speaker|voice/i.test(label) ? 'Speakers'
        : /transcri|model loaded/i.test(label) ? 'Transcription'
        : 'Preparation';
    const phases = {};
    steps.forEach(t => {
        const ph = phases[phaseOf(t.label)] ??= { label: phaseOf(t.label), secs: 0, steps: [] };
        ph.steps.push({ ...t });
        // "Separation" is part of "Vocal Isolation" (the step's total) — the
        // phase is its longest step, not the sum, or it counts twice.
        ph.secs = ph.label === 'Vocal isolation' ? Math.max(ph.secs, t.secs) : ph.secs + t.secs;
    });
    const rows = Object.values(phases).sort((a, b) => b.secs - a.secs);
    const chips = [
        d.vocal_isolation ? `🎤 ${esc(d.vocal_model || 'vocal isolation')}${d.vocal_two_pass ? ' · 2 passes' : ''}` : '',
        d.total_chunks ? `${fmtNumber(d.total_chunks)} chunk${d.total_chunks === 1 ? '' : 's'}` : '',
        d.warning_count ? `⚠ ${fmtNumber(d.warning_count)} warning${d.warning_count === 1 ? '' : 's'}` : '',
    ].filter(Boolean).map(c => `<span class="txs-chip">${c}</span>`).join('');
    return stvCard('Processing', (rows.length ? stvBars(rows, {
            value: r => r.secs, label: r => r.label,
            right: r => `${fmtDuration(r.secs)} <span class="stv-dim">${total ? (r.secs / total * 100).toFixed(0) : 0}%</span>`,
            tip: r => `${r.label} · ${fmtDuration(r.secs)}\n` + r.steps.map(t => `${t.label}: ${fmtDuration(t.secs)}`).join('\n'),
        }) : `<div class="stv-empty">No step timings recorded for this file.</div>`)
        + (chips ? `<div class="txs-chips">${chips}</div>` : ''),
        { sub: `${esc(d.processing_time || fmtDuration(total))} in total` });
}

function txsFile(d, fm) {
    const row = (k, v) => v ? `<div class="txs-kv"><span>${k}</span><b>${v}</b></div>` : '';
    const res = fm.media_width && fm.media_height ? `${fm.media_width} × ${fm.media_height}` : '';
    return stvCard('File', `<div class="txs-kvs">
        ${row('Format', esc((fm.ext || '').replace(/^\./, '').toUpperCase()))}
        ${row('Size', esc(fm.size_fmt || ''))}
        ${row('Resolution', esc(res))}
        ${row('Video codec', esc(fm.media_video_codec || ''))}
        ${row('Audio codec', esc(fm.media_audio_codec || ''))}
        ${row('Bitrate', fm.media_bitrate ? `${Math.round(fm.media_bitrate / 1000).toLocaleString()} kbit/s` : '')}
        ${row('Language', esc((d.language || '').toUpperCase()))}
        ${row('Model', esc(d.model || ''))}
        ${row('Transcribed', fm.tx_date ? esc(new Date(fm.tx_date * 1000).toLocaleString()) : '')}
        ${row('App version', esc(fm.tx_app_version || ''))}
    </div>`, { span: 2, sub: esc(fm.subfolder || '') });
}

function txsWords(s) {
    if (!s.topWords.length) return '';
    const half = Math.ceil(s.topWords.length / 2);
    const bars = list => stvBars(list, { value: r => r.cnt, label: r => r.word, right: r => fmtNumber(r.cnt),
                                          tip: r => `${r.word}\n${fmtNumber(r.cnt)}×` });
    return stvCard('Words', `<div class="stv-inline-figs">
            <span><b>${fmtNumber(s.words)}</b> word${s.words === 1 ? '' : 's'}</span>
            <span><b>${fmtNumber(s.vocab)}</b> different one${s.vocab === 1 ? '' : 's'}</span>
            <span>lexical variety <b>${s.tokens ? (s.vocab / s.tokens * 100).toFixed(0) : 0}%</b></span></div>
        <div class="stv-text-grid"><div>${bars(s.topWords.slice(0, half))}</div><div>${bars(s.topWords.slice(half))}</div></div>
        <div class="stv-foot">Most frequent content words — filler and function words are left out.</div>`,
        { span: 2 });
}

// ── The conversation (api/conversations.py) ─────────────────────────────
// What the server counted for this recording: its kind and why, words per
// sentence, speaker changes, who talked when — the numbers the
// Conversations tab finds conversations by, with the names of the voice
// library. Not the figures of the cards around it: those count segments
// and diarization labels, these sentences and people. Filled when the
// answer comes; a file switch in between drops it.

let _txsConvSeq = 0;

function txsConvCard() {
    return stvCard('Conversation', '<div class="txs-conv" id="txsConv"><div class="stv-empty">Loading …</div></div>',
        { span: 2, sub: 'sentences, speaker changes and people as the Conversations tab counts them' });
}

// The open transcript's full path — what the server knows a recording by
function txsConvPath() {
    if (!activeFile) return '';
    return typeof filePath === 'function' ? filePath({ name: activeFile }) : String(activeFile);
}

async function txsConvLoad(path) {
    const box = document.getElementById('txsConv');
    if (!box) return;
    const seq = ++_txsConvSeq;
    box.dataset.file = path || '';
    if (!path) { box.innerHTML = '<div class="stv-empty">No file is open.</div>'; return; }
    let d;
    try { d = await api('/api/conversations/file?file=' + encodeURIComponent(path)); }
    catch (e) {
        if (seq !== _txsConvSeq || document.getElementById('txsConv') !== box) return;
        box.innerHTML = `<div class="stv-empty">${e.status === 503 && /switched off/i.test(String(e.message || ''))
            ? 'Conversation statistics are switched off — Settings → Search → Conversations.'
            : e.status === 404 ? 'No conversation statistics: the library holds no transcript of this file.'
            : esc(e.message)}</div>`;
        return;
    }
    if (seq !== _txsConvSeq || document.getElementById('txsConv') !== box || box.dataset.file !== path) return;
    box._conv = d;
    box.innerHTML = txsConvHtml(d);
    box.querySelectorAll('[data-seek]').forEach(el => el.addEventListener('click', () => {
        closeTxStats();
        if (typeof seekTo === 'function') seekTo(+el.dataset.seek);
    }));
}

function txsConvHtml(d) {
    if (!d || !d.counted) {
        return '<div class="stv-empty">Not counted yet — the background count gets to it, or start it in the Find window’s Conversations tab.</div>';
    }
    const s = d.summary && typeof d.summary === 'object' ? d.summary : {};
    const fig = (x) => (x === null || x === undefined || x === '' || !Number.isFinite(Number(x)) ? '' : Number(x).toFixed(1));
    const pct = (x) => `${Math.round(Math.max(0, Number(x) || 0) * 100)}%`;
    const pctOr = (x) => (x === null || x === undefined || x === '' ? '—' : pct(x));
    // "longer than 78 %" of the conversations compared with — under half the
    // other way round, which says the same and reads better
    const than = (p, up, down) => {
        if (p === null || p === undefined || p === '' || !Number.isFinite(Number(p))) return '';
        const v = Math.round(Number(p));
        return v >= 50 ? `${up} than ${v}% of conversations` : `${down} than ${100 - v}% of conversations`;
    };
    const kindName = typeof cvKindName === 'function' ? cvKindName(d.kind) : String(d.kind || '');
    const colour = (name) => (typeof cvColour === 'function' ? cvColour(name) : 'var(--accent)');
    const speakers = Array.isArray(d.speakers) ? d.speakers : [];
    const why = (Array.isArray(d.reasons) ? d.reasons : []).map(r => `<li>${esc(r)}</li>`).join('');
    const head = `<div class="txs-conv-head"><span class="txs-conv-kind">${esc(kindName || '—')}</span>${
        d.group ? '<span class="txs-conv-tag" title="Three people or more take part">Group</span>' : ''}${
        why ? `<ul class="txs-conv-why">${why}</ul>` : ''}</div>`;
    const wps = fig(s.wps), rate = fig(s.turn_rate);
    const turn = Number(s.mean_turn_sec) > 0 ? fmtDuration(s.mean_turn_sec) : '';
    const longest = Number(s.longest_turn_sec) > 0;
    const who = longest ? s.longest_turn_name || (speakers.find(x => x && x.label === s.longest_turn_label) || {}).name || '' : '';
    const kpis = [
        stvKpi('Words per sentence', wps || '—', esc(wps ? than(s.wps_pct, 'longer', 'shorter') : 'no sentence punctuation'),
               'Words per sentence\nSentences end at . ! ? — compared with the library’s conversations in the same language'),
        stvKpi('Speaker changes', rate ? `${rate}<small> /min</small>` : '—', esc(rate ? than(s.rate_pct, 'more', 'fewer') : 'under a minute of known speakers'),
               'Speaker changes per minute of speech\nCounted between the speakers the diarization told apart — one person it split in two counts as two'),
        stvKpi('Average turn', esc(turn || '—'), esc(turn ? than(s.turn_pct, 'longer', 'shorter') : ''),
               'Average turn\nSpeech with a known speaker per speaker change'),
        // Without sentence punctuation each segment counts as a sentence and
        // no question can be told apart: "0 % of 800 sentences" would be wrong
        'punctuated' in s && !s.punctuated
            ? stvKpi('Questions', '—', 'no sentence punctuation', 'Questions\nSentences that end in “?” — this transcript has no sentence punctuation')
            : stvKpi('Questions', pctOr(s.questions_share), `${fmtNumber(s.questions)} of ${plural(s.sentences, 'sentence')}`),
        longest ? `<button class="stv-kpi txs-conv-jump" data-seek="${Number(s.longest_turn_start) || 0}" title="Jump there">
            <div class="stv-kpi-label">Longest monologue</div>
            <div class="stv-kpi-value">${esc(fmtDuration(s.longest_turn_sec))}</div>
            <div class="stv-kpi-sub">${who ? `${esc(who)} · ` : ''}at ${esc(txsFmtClock(Number(s.longest_turn_start) || 0))}</div></button>`
            : stvKpi('Longest monologue', '—', ''),
    ].join('');
    const people = (Array.isArray(d.people) ? d.people : []).filter(p => p && p.name);
    const unnamed = speakers.filter(x => x && !x.name && (Number(x.share) || 0) >= 0.05).length;
    const bars = people.length
        ? stvBars(people, { value: p => Number(p.share) || 0, label: p => String(p.name),
                            right: p => `${pct(p.share)} <span class="stv-dim">${esc(fmtDuration(p.talk_sec))}</span>`,
                            tip: p => `${p.name}\n${fmtDuration(p.talk_sec)} of the talk · ${pct(p.share)}` })
        : speakers.length
            ? '<div class="stv-empty">Nobody in it has a name yet — names come from the voice map or from the transcript.</div>'
            // Not diarized: there are no voices to give a name to
            : '<div class="stv-empty">Its speakers were not told apart — there is nobody to name.</div>';
    const parts = (Array.isArray(d.parts) ? d.parts : []).filter(p => p && typeof p === 'object').map(p => {
        const names = (Array.isArray(p.people) ? p.people : []).map(n =>
            `<span><i class="stv-dot" style="background:${colour(n)}"></i>${esc(n)}</span>`).join('');
        const more = Number(p.unnamed) > 0 ? `<span class="stv-dim">+ ${Math.round(Number(p.unnamed))} unnamed</span>` : '';
        return `<div class="txs-conv-part">
            <span><button class="txs-link" data-seek="${Number(p.start_sec) || 0}" title="Jump there">${esc(txsFmtClock(Number(p.start_sec) || 0))}</button><span
                class="txs-conv-end"> – ${esc(txsFmtClock(Number(p.end_sec) || 0))}</span></span>
            <span class="txs-conv-names">${names || (more ? '' : '<span class="stv-dim">nobody named</span>')}${more}</span></div>`;
    }).join('');
    const more = people.length && typeof cvShow === 'function'
        ? `<button class="stv-btn" onclick="txsConvMore()" title="The Conversations tab of the search, with ${esc(people.length === 1 ? people[0].name : 'all of them together')}">🗣 More conversations with ${
            people.length === 1 ? esc(people[0].name) : 'these people'}</button>` : '';
    // Counted before the transcript last changed (or by older rules): shown
    // as they are, the background count renews them
    const stale = d.stale ? '<div class="stv-foot" style="margin:0 0 10px">Counted before the transcript last changed — the background count renews these numbers.</div>' : '';
    return `${head}${stale}<div class="txs-conv-kpis">${kpis}</div>
        <div class="txs-conv-cols">
            <div><div class="stv-sublabel">People</div>${bars}${unnamed
                ? `<div class="stv-foot">+ ${plural(unnamed, 'voice')} without a name</div>` : ''}</div>
            <div><div class="stv-sublabel">Who talked when</div>${parts
                ? `<div class="txs-conv-parts">${parts}</div>`
                : '<div class="stv-empty">Shown for recordings longer than ten minutes with speakers told apart.</div>'}</div>
        </div>${more}`;
}

// "More conversations with these people": the search's tab, with them all
function txsConvMore() {
    const d = document.getElementById('txsConv')?._conv;
    const names = ((d && d.people) || []).map(p => p && p.name).filter(Boolean);
    if (!names.length || typeof cvShow !== 'function') return;
    closeTxStats();
    cvShow({ people: names });
}

window.addEventListener('resize', () => {
    const box = document.getElementById('rvStats');
    if (box && box._stats && box.offsetParent) txsDrawTimeline(box._stats);
});

