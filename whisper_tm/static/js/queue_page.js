// ── Queue · Jobs: transcriptions and AI jobs in one list ────────────────
// The Queue place opens on this tab of the queue window. Sections Running,
// Waiting, Failed and Done; chips All · Transcriptions · AI jobs; each queue
// paused and resumed by its own button (nothing here resumes by itself);
// waiting transcriptions move up, down or to the front, take their
// settings or leave the queue; failed ones run again; AI jobs are stopped
// or run again as in the old LLM tab. The other tabs of the window edit
// many entries at once and add from the library or a folder.
//
// Data: the transcription queue's state as the app keeps it (batch.js) and
// the entries per status from /api/queue/editor/items — fetched when the
// tab is shown and when the queue's generation changes; the AI jobs from
// the LLM queue answer the badge polls anyway (llmtools.js); exports
// ("Export · N files") from export.js, which follows them while they run.

const QueuePage = (() => {
    let kind = 'all';            // all · tx · llm
    let visible = false;
    let tx = {pending: null, failed: null, done: null};
    let loadSeq = 0;
    let lastGen = null;
    let timer = null;

    const MAX_WAITING = 60, MAX_FAILED = 40, MAX_DONE = 15;
    const base = p => String(p || '').split('/').pop();
    const el = id => document.getElementById(id);

    /* The Queue place, on Jobs, showing one kind (or all). */
    function open(which) {
        if (which) kind = which === 'tx' || which === 'llm' ? which : 'all';
        if (typeof Shell !== 'undefined') Shell.go('queue');
        else if (typeof openQueueEditorFromBar === 'function') openQueueEditorFromBar();
        draw();
    }

    function shown(on) {
        visible = !!on;
        if (visible) { refresh(); if (typeof Export !== 'undefined') Export.pollJobs(); if (typeof loadQueueEta === 'function') loadQueueEta(true).then(draw); }
    }

    async function refresh() {
        const seq = ++loadSeq;
        const get = st => fetch(`/api/queue/editor/items?status=${st}&limit=${st === 'done' ? MAX_DONE : st === 'pending' ? MAX_WAITING : MAX_FAILED}`)
            .then(r => r.ok ? r.json() : null).catch(() => null);
        const [pending, failed, done] = await Promise.all([get('pending'), get('error,cancelled'), get('done')]);
        if (seq !== loadSeq) return;
        tx = {pending, failed, done};
        if (typeof llmQueueRefreshBadge === 'function' && !llmQueueSnap) llmQueueRefreshBadge();
        draw();
    }

    /* The queue's state changed: a new generation means entries came,
       went or moved — reload them (debounced); otherwise redraw progress. */
    function queueChanged(gen) {
        if (!visible) return;
        if (gen !== undefined && gen !== lastGen) {
            lastGen = gen;
            clearTimeout(timer);
            timer = setTimeout(refresh, 400);
            return;
        }
        draw();
    }

    function llmChanged() { if (visible) draw(); }

    function setKind(k) { kind = k; draw(); }

    /* ── Actions ─────────────────────────────────────────────────────── */

    async function post(url, body, okText) {
        try {
            const r = await fetch(url, {method: 'POST', headers: {'Content-Type': 'application/json'},
                                        body: JSON.stringify(body || {})});
            const d = await r.json().catch(() => ({}));
            if (!r.ok) throw new Error(d.error || `HTTP ${r.status}`);
            if (okText && typeof Toast !== 'undefined') Toast.success(okText);
            return d;
        } catch (e) {
            if (typeof Toast !== 'undefined') Toast.error(String(e.message || e));
            return null;
        } finally {
            refresh();
            if (typeof refreshQueue === 'function') refreshQueue();
        }
    }

    const move = (id, direction) => post('/api/queue/move', {id, direction});
    const toFront = id => post('/api/queue/move', {id, top: true}, 'Next up');
    const remove = id => post('/api/queue/remove', {id}, 'Taken out of the queue');
    const cancel = id => post('/api/queue/cancel', {item_id: id}, 'Cancelled');
    const retry = id => post('/api/queue/retry', {ids: [id]}, 'Back in the queue');

    /* One entry's transcription settings: the bulk editor's dialog with
       just this entry chosen. */
    function settings(id) {
        if (typeof qeShowTab !== 'function') return;
        qeShowTab('queue');
        qeAll = false;
        qeDeselected.clear();
        qeSelection.clear();
        qeSelection.add(id);
        qeScope = 'selected';
        if (typeof qeRenderScope === 'function') qeRenderScope();
        if (typeof qeSettingsOpen === 'function') qeSettingsOpen();
    }

    async function pauseTx(on) {
        await post(on ? '/api/queue/stop' : '/api/queue/resume', {},
                   on ? 'Transcriptions pause after the running file' : 'Transcriptions resume');
        if (typeof loadQueueEta === 'function') { await loadQueueEta(true); draw(); }
    }

    async function pauseLlm(on) {
        if (typeof _lqPause === 'function') await _lqPause(on);
        if (typeof llmQueueRefreshBadge === 'function') await llmQueueRefreshBadge();
    }

    async function llmCancel(id) {
        if (typeof _lqCancel === 'function') await _lqCancel(id);
        if (typeof llmQueueRefreshBadge === 'function') llmQueueRefreshBadge();
    }

    async function llmRetry(id) {
        if (typeof _lqRetry === 'function') await _lqRetry(id);
        if (typeof llmQueueRefreshBadge === 'function') llmQueueRefreshBadge();
    }

    function openFile(name, dir) {
        if (typeof Home !== 'undefined') Home.openFile(name, dir || '');
    }

    /* ── Drawing ─────────────────────────────────────────────────────── */

    function settingsText(s) {
        s = s || {};
        return [s.job === 'voices' ? 'voices only' : '', s.model, s.language, s.diarize ? 'speakers' : '',
                s.retranscribe ? 'again' : ''].filter(Boolean).join(' · ');
    }

    function txRow(i, section) {
        const name = i.filename || '';
        const open = `onclick="QueuePage.openFile('${escJs(name)}', '${escJs(i.dir || '')}')"`;
        let meta = '', acts = '';
        if (section === 'running') {
            const pct = Math.round(Number(i.progress) || 0);
            meta = `<span class="qj-bar-mini"><span style="width:${pct}%"></span></span><span class="qj-n">${pct} %</span>`
                + (i.message ? `<span class="qj-msg">${esc(i.message)}</span>` : '');
            acts = `<button class="qj-act" onclick="QueuePage.cancel(${Number(i.item_id ?? i.id)})" title="Cancel this transcription">✕</button>`;
        } else if (section === 'waiting') {
            meta = `<span class="qj-msg">${esc(settingsText(i.settings))}</span>`;
            acts = `<button class="qj-act" onclick="QueuePage.toFront(${Number(i.id)})" title="Next up">⤒</button>`
                + `<button class="qj-act" onclick="QueuePage.move(${Number(i.id)}, -1)" title="One up">▲</button>`
                + `<button class="qj-act" onclick="QueuePage.move(${Number(i.id)}, 1)" title="One down">▼</button>`
                + `<button class="qj-act" onclick="QueuePage.settings(${Number(i.id)})" title="This entry's transcription settings">⚙</button>`
                + `<button class="qj-act" onclick="QueuePage.remove(${Number(i.id)})" title="Take it out of the queue">✕</button>`;
        } else if (section === 'failed') {
            meta = `<span class="qj-err">${esc(i.status === 'cancelled' ? 'cancelled' : (i.error || 'failed'))}</span>`;
            acts = `<button class="qj-act" onclick="QueuePage.retry(${Number(i.id)})" title="Run it again">↻</button>`;
        } else {
            meta = `<span class="qj-n">${i.finished_at ? esc(new Date(i.finished_at * 1000).toLocaleString()) : ''}</span>`;
        }
        return `<div class="qj-row qj-${section}"><span class="qj-kind" title="Transcription">🎙</span>`
            + `<button class="qj-name" ${open} title="${esc(name)}">${esc(base(name))}</button>${meta}`
            + `<span class="qj-acts">${acts}</span></div>`;
    }

    function llmRow(i, section) {
        const kindName = (typeof LQ_KIND !== 'undefined' && LQ_KIND[i.kind]) || i.kind;
        const p = i.params || {};
        let meta = '', acts = '';
        if (section === 'running') {
            const pct = Math.round(Number(i.progress) || 0);
            meta = `<span class="qj-bar-mini"><span style="width:${pct}%"></span></span><span class="qj-n">${pct} %</span>`
                + (i.message ? `<span class="qj-msg">${esc(i.message)}</span>` : '');
        } else if (section === 'failed') {
            meta = `<span class="qj-err">${esc(i.status === 'cancelled' ? 'cancelled' : (i.error || 'failed').slice(0, 200))}</span>`;
        } else if (section === 'done') {
            meta = `<span class="qj-n">${i.finished_at ? esc(new Date(i.finished_at * 1000).toLocaleString()) : ''}</span>`;
        } else {
            meta = `<span class="qj-msg">${esc([p.preset, p.language && '→ ' + p.language, p.model].filter(Boolean).join(' · '))}</span>`;
        }
        if (section === 'running' || section === 'waiting')
            acts = `<button class="qj-act" onclick="QueuePage.llmCancel(${Number(i.id)})" title="Stop this job">✕</button>`;
        if (section === 'failed')
            acts = `<button class="qj-act" onclick="QueuePage.llmRetry(${Number(i.id)})" title="Run it again">↻</button>`;
        const link = i.kind === 'analysis' && p.analysis_id
            ? `<button class="qj-act" onclick="openAnalysis('${escJs(p.analysis_id)}')" title="Open the analysis">↗</button>`
            : i.kind === 'deep_search' && p.deep_id
                ? `<button class="qj-act" onclick="openDeepSearch('${escJs(p.deep_id)}')" title="Open the deep search">↗</button>` : '';
        const name = i.filename || p.label || kindName;
        return `<div class="qj-row qj-${section}"><span class="qj-kind" title="AI job">🤖</span>`
            + `<span class="qj-what">${esc(kindName)}</span>`
            + (i.filename ? `<button class="qj-name" onclick="QueuePage.openFile('${escJs(i.filename)}', '')" title="${esc(i.filename)}">${esc(base(i.filename))}</button>`
                          : `<span class="qj-name qj-plain">${esc(name)}</span>`)
            + meta + `<span class="qj-acts">${link}${acts}</span></div>`;
    }

    function section(title, rows, total, extra) {
        const more = total > rows.length ? `<div class="qj-more">and ${(total - rows.length).toLocaleString()} more${extra || ''}</div>` : '';
        return `<section class="qj-sec"><h4>${esc(title)} <span class="qj-count">${total.toLocaleString()}</span></h4>`
            + (rows.length ? rows.join('') : '<div class="qj-empty">None</div>') + more + `</section>`;
    }

    function draw() {
        const bar = el('qjBar'), list = el('qjList');
        if (!bar || !list || !visible) return;
        const showTx = kind !== 'llm', showLlm = kind !== 'tx';
        const llm = typeof llmQueueSnap !== 'undefined' && llmQueueSnap ? llmQueueSnap : {items: [], counts: {}};
        const li = st => (llm.items || []).filter(i => st.includes(i.status));
        // Paused also while nothing waits (the app calls an idle queue
        // neither running nor paused; the time estimate knows)
        const txPaused = (typeof queuePaused !== 'undefined' && queuePaused)
            || !!(typeof queueEta !== 'undefined' && queueEta && queueEta.paused);
        const txRunning = (typeof queueRunningItems !== 'undefined' ? queueRunningItems : []);
        const eta = typeof queueEta !== 'undefined' && queueEta && queueEta.sec != null && queueEta.pending
            ? `~${typeof _dsLeft === 'function' ? _dsLeft(queueEta.sec) : Math.round(queueEta.sec / 60) + ' min'} left`
              + (queueEta.paused ? ' (paused)' : queueEta.finish_at && typeof _finishText === 'function' ? ` · done ${_finishText(queueEta.finish_at)}` : '')
            : '';
        const chip = (k, label) => `<button class="qj-chip${kind === k ? ' on' : ''}" onclick="QueuePage.setKind('${k}')">${label}</button>`;
        bar.innerHTML = `<span class="qj-chips">${chip('all', 'All')}${chip('tx', 'Transcriptions')}${chip('llm', 'AI jobs')}</span>`
            + (showTx ? `<span class="qj-ctl"><b>Transcriptions</b> ${txPaused ? '<span class="qj-paused">paused</span>' : ''}`
                + `<button class="qj-btn" onclick="QueuePage.pauseTx(${!txPaused})">${txPaused ? '▶ Resume' : '⏸ Pause'}</button>`
                + (eta ? `<span class="qj-eta">${esc(eta)}</span>` : '') + `</span>` : '')
            + (showLlm ? `<span class="qj-ctl"><b>AI jobs</b> ${llm.paused ? '<span class="qj-paused">paused</span>' : ''}`
                + `<button class="qj-btn" onclick="QueuePage.pauseLlm(${!llm.paused})">${llm.paused ? '▶ Resume' : '⏸ Pause'}</button>`
                + (llm.waiting ? `<span class="qj-err" title="${esc(llm.waiting.reason || '')}">the language model does not answer</span>` : '') + `</span>` : '')
            + `<span class="qj-gap"></span>`;

        const parts = [];
        // Exports (export.js) show with everything, not under one kind
        const ex = kind === 'all' && typeof Export !== 'undefined' ? Export.jobs : [];
        const exRows = st => ex.filter(j => st.includes(j.status)).map(j => Export.jobRow(j, 'queue'));
        // Running
        const run = [...(showTx ? txRunning.map(i => txRow(i, 'running')) : []),
                     ...(showLlm ? li(['running']).map(i => llmRow(i, 'running')) : []), ...exRows(['running'])];
        parts.push(section('Running', run, run.length));
        // Waiting
        const wTx = showTx && tx.pending ? (tx.pending.items || []).map(i => txRow(i, 'waiting')) : [];
        const wLlm = showLlm ? li(['pending']).map(i => llmRow(i, 'waiting')) : [];
        const wEx = exRows(['waiting']);
        const wTotal = (showTx ? (tx.pending?.total ?? wTx.length) : 0) + wLlm.length + wEx.length;
        parts.push(section('Waiting', [...wTx, ...wLlm, ...wEx], wTotal,
            showTx ? ' — <button class="qj-link" onclick="qeShowTab(\'queue\')">edit them in bulk</button>' : ''));
        // Failed
        const fTx = showTx && tx.failed ? (tx.failed.items || []).map(i => txRow(i, 'failed')) : [];
        const fLlm = showLlm ? li(['error', 'cancelled']).map(i => llmRow(i, 'failed')) : [];
        const fEx = exRows(['error']);
        const fTotal = (showTx ? (tx.failed?.total ?? fTx.length) : 0) + fLlm.length + fEx.length;
        const retryAll = showTx && typeof queueRetryable !== 'undefined' && queueRetryable
            ? `<button class="qj-btn" onclick="retryFailedQueue()">↻ All transcriptions again</button>` : '';
        parts.push(section('Failed', [...fTx, ...fLlm, ...fEx], fTotal).replace('</h4>', retryAll + '</h4>'));
        // Done: the newest of both, together
        const dTx = showTx && tx.done ? (tx.done.items || []).map(i => [i.finished_at || 0, txRow(i, 'done')]) : [];
        const dLlm = showLlm ? li(['done']).map(i => [i.finished_at || 0, llmRow(i, 'done')]) : [];
        const dEx = ex.filter(j => j.status === 'done').map(j => [j.finished || 0, Export.jobRow(j, 'queue')]);
        const done = [...dTx, ...dLlm, ...dEx].sort((a, b) => b[0] - a[0]).slice(0, MAX_DONE).map(x => x[1]);
        const dTotal = (showTx ? (tx.done?.total ?? dTx.length) : 0) + dLlm.length + dEx.length;
        const llmFinished = showLlm ? li(['done', 'error', 'cancelled']).length : 0;
        const clearAi = llmFinished ? `<button class="qj-btn" onclick="_lqClear('done').then(() => llmQueueRefreshBadge())" title="Remove done, failed and cancelled AI jobs from the list">🧹 Remove finished AI jobs</button>` : '';
        parts.push(section('Done', done, dTotal).replace('</h4>', clearAi + '</h4>'));
        list.innerHTML = parts.join('');
    }

    function init() {
        // The queue's stream: progress redraws, a new generation reloads
        if (typeof applyQueueState === 'function') {
            const orig = window.applyQueueState;
            window.applyQueueState = function (d, ...rest) {
                const r = orig.call(this, d, ...rest);
                queueChanged(d && d.generation);
                return r;
            };
        }
    }

    document.addEventListener('DOMContentLoaded', init);
    return {open, shown, refresh, queueChanged, llmChanged, setKind, move, toFront, remove, cancel, retry,
            settings, pauseTx, pauseLlm, llmCancel, llmRetry, openFile,
            get kind() { return kind; }, get visible() { return visible; }};
})();
