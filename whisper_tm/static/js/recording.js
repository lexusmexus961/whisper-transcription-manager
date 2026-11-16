// ── An open recording: header, who-speaks-when lanes, tabs ───────────────
// fileview.renderTranscript() builds the panes; this module draws the
// header (the name, one quiet line, Export, ⋯), the lanes under the player
// (one per speaker — a click seeks), and the tab bar: Transcript, Summary,
// Translation, Speakers, Clips, Statistics, Details, Analyses. Tabs that
// do not fit go into "More"; the last tab is remembered (UI preference);
// arriving from a search hit always opens Transcript.

const Rec = (() => {
    const TABS = [
        ['transcript', 'Transcript', 'i-captions'],
        ['summary', 'Summary', 'i-book'],
        ['translation', 'Translation', 'i-chat'],
        ['speakers', 'Speakers', 'i-voices'],
        ['clips', 'Clips', 'i-scissors'],
        ['statistics', 'Statistics', 'i-layers'],
        ['details', 'Details', 'i-info'],
        ['analyses', 'Analyses', 'i-sparkle'],
    ];
    let current = 'transcript';
    let forceTranscript = false;
    let moreOpen = false;
    const el = id => document.getElementById(id);

    function remembered() {
        try {
            const p = typeof Shell !== 'undefined' ? Shell.pref('last_tab') : null;
            if (p && TABS.some(t => t[0] === p.recording)) return p.recording;
        } catch (e) { /* default */ }
        return 'transcript';
    }

    /* ── Header ───────────────────────────────────────────────────────── */

    const fmtLen = s => {
        s = Number(s) || 0;
        if (!s) return '';
        const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), x = Math.floor(s % 60);
        return h ? `${h}:${String(m).padStart(2, '0')}:${String(x).padStart(2, '0')}` : `${m}:${String(x).padStart(2, '0')}`;
    };

    function headerHtml(f, d, fn) {
        const name = (f && (f.display_name || f.name)) || (activeFile || '').split('/').pop();
        const len = d.total_duration || fmtLen(d.total_duration_sec || (f && f.media_duration_sec));
        const speakers = (d.speakers || []).filter(s => s && s !== '—' && s !== '-').length;
        const lang = d.language || (f && f.tx_lang) || '';
        const when = f && f.tx_date ? new Date(f.tx_date * 1000).toLocaleDateString(undefined, {day: 'numeric', month: 'short', year: 'numeric'}) : '';
        const bits = [len, speakers ? plural(speakers, 'speaker') : '', lang ? lang.toUpperCase() : '', when ? 'transcribed ' + when : ''].filter(Boolean);
        const warn = d.warning_count || (f && f.tx_warning_count) || 0;
        const log = `showLogModal('${d.log_file ? escJs(d.log_file) : ''}', '${fn}')`;
        return `<header class="rv-head">
            <div class="rv-names">
                <h2 class="rv-title" title="${esc(activeFile || name)}">${esc(name)}</h2>
                <div class="rv-line">${bits.map(esc).join(' · ')}${warn
                    ? ` · <button class="rv-warn" onclick="${log}" title="What went wrong or was unusual while transcribing — the log">⚠ ${plural(warn, 'warning')}</button>` : ''}</div>
            </div>
            <div class="rv-acts">
                <button class="tb-btn rv-export" onclick="Export.show({file: activeFileObj && activeFileObj.abs_path || '${fn}'})" title="Export: text, subtitles, Word, PDF, the video or the sound — with the options for it"><svg class="ui-ico"><use href="#i-download"/></svg><span class="tb-label">Export</span></button>
                <div class="bb-more rv-more">
                    <button class="tb-btn" onclick="Rec.menu('rvMoreMenu', event)" aria-haspopup="menu" title="More: the voice map, the log, delete">⋯</button>
                    <div class="bb-menu down" id="rvMoreMenu" role="menu" hidden>
                        <button class="bm-item" onclick="Rec.closeMenus();downloadOriginal('${fn}')">⬇ Download the original file</button>
                        <button class="bm-item" onclick="Rec.closeMenus();mapForFile('${fn}')">🗣 On the voice map</button>
                        <button class="bm-item" onclick="Rec.closeMenus();${log}">📋 Log</button>
                        <button class="bm-item danger" onclick="Rec.closeMenus();deleteTx('${fn}')">🗑 Delete…</button>
                    </div>
                </div>
            </div>
        </header>`;
    }

    function menu(id, ev) {
        ev && ev.stopPropagation();
        const m = el(id);
        if (!m) return;
        const open = m.hidden;
        closeMenus();
        m.hidden = !open;
    }

    function closeMenus() {
        for (const id of ['rvExportMenu', 'rvMoreMenu', 'rvTabMore']) { const m = el(id); if (m) m.hidden = true; }
    }

    /* ── Before a transcript exists: open, waiting, running, failed ───── */

    function stateHtml(f) {
        const fn = escJs(f.name);
        const name = f.display_name || String(f.name).split('/').pop();
        const outcome = typeof queueOutcomeOf === 'function' ? queueOutcomeOf(f) : null;
        const queued = typeof isQueuedFile === 'function' && isQueuedFile(f);
        const running = typeof queueCurrentFile !== 'undefined' && queueCurrentFile && typeof isRunningFile === 'function' && isRunningFile(f);
        const len = f.audio_duration || fmtLen(f.media_duration_sec);
        let what, body;
        const btn = (label, call, cls = '', title = '') => `<button class="tb-btn ${cls}" onclick="${call}"${title ? ` title="${esc(title)}"` : ''}><span class="tb-label">${label}</span></button>`;
        if (running) {
            what = 'being transcribed now';
            body = `<p>Progress, steps and the log are in the bar above (expand “Details”). The text appears here as the speech recognition delivers it.</p>`;
        } else if (queued) {
            const items = typeof queueItems !== 'undefined' ? queueItems : [];
            const i = items.findIndex(q => (q.abs && f.abs_path) ? q.abs === f.abs_path : q.filename === f.name);
            const item = items[i];
            what = 'waiting in the queue';
            body = `<p>${i >= 0 ? `Number <b>${i + 1}</b> of ${items.length.toLocaleString()} waiting.` : 'In the queue.'}</p>
                <div class="rv-state-acts">${btn('⤒ Move to the top', `quickQueueFront('${fn}')`)}
                ${item && item.id != null ? btn('✕ Take it out', `QueuePage.remove(${Number(item.id)})`) : ''}
                ${btn('Open the queue', "Shell.go('queue')")}</div>`;
        } else if (outcome && (outcome.status === 'error' || outcome.status === 'cancelled')) {
            what = outcome.status === 'error' ? 'transcription failed' : 'transcription cancelled';
            body = `<p class="rv-state-why">${esc(outcome.error || (outcome.status === 'cancelled' ? 'It was cancelled.' : 'It failed.'))}</p>
                <div class="rv-state-acts">${btn('↻ Try again', `retryFilteredOutcomesByNames(['${fn}'])`, 'primary')}
                ${btn('📋 The log', `showLogModal('', '${fn}')`)}
                ${btn('Settings…', `startTranscribe('${fn}')`, '', 'Transcribe it with other settings')}</div>`;
        } else if (f.status === 'skipped') {
            what = f.skip_reason === 'no_audio' ? 'no audio track' : 'skipped by hand';
            body = `<p>${f.skip_reason === 'no_audio' ? 'No audio track was found in this file.' : 'This file was skipped by hand.'}</p>
                <div class="rv-state-acts">${btn('↩ Reactivate', `unskipFile('${fn}')`)}${btn('▶ Transcribe anyway', `startTranscribe('${fn}')`)}</div>`;
        } else {
            what = f.status === 'partial' ? 'transcription incomplete' : 'not transcribed yet';
            body = `<p>Speech to text with speaker recognition. The dialog also offers “voices only” and no transcription at all.</p>
                <div class="rv-state-acts">${btn('▶ Transcribe', `quickQueueFront('${fn}')`, 'primary', 'Up next, with the default settings')}
                ${btn('Settings…', `startTranscribe('${fn}')`, '', 'Choose the model, the language and more first')}
                ${btn('+ Add to the queue', `addToQueue('${fn}')`, '', 'At the end of the queue')}
                ${btn('⬇ Download the original', `downloadOriginal('${fn}')`)}</div>`;
        }
        return `<header class="rv-head"><div class="rv-names"><h2 class="rv-title" title="${esc(f.name)}">${esc(name)}</h2>
            <div class="rv-line">${[len, what].filter(Boolean).map(esc).join(' · ')}</div></div></header>
            <div class="rv-state"${running || queued ? ' id="txRunningPlaceholder"' : ''}>${body}</div>
            ${detailsHtml(f, null)}`;
    }

    /* ── Details: everything known about the file and its transcription ── */

    function detailsHtml(f, d) {
        f = f || {};
        const st = (d && d.settings) || {};
        const when = ts => ts ? new Date(ts * 1000).toLocaleString(undefined, {day: 'numeric', month: 'short', year: 'numeric', hour: '2-digit', minute: '2-digit'}) : '';
        const yes = v => v === true ? 'yes' : v === false ? 'no' : '';
        const row = (k, v) => v === '' || v === null || v === undefined ? '' : `<tr><th>${esc(k)}</th><td>${v}</td></tr>`;
        const path = typeof filePath === 'function' ? filePath(f) : (f.abs_path || f.name || '');
        const speed = f.audio_duration_sec && f.processing_secs > 0 && typeof fmtSpeed === 'function' ? fmtSpeed(f.audio_duration_sec / f.processing_secs) : '';
        const res = f.width && f.height ? `${f.width} × ${f.height}` : '';
        const media = [
            row('Format', esc((f.ext || '').replace(/^\./, '').toUpperCase() + (f.is_video ? ' video' : ' audio'))),
            row('Size', esc(f.size_fmt || '')),
            row('Length', esc(f.audio_duration || (d && d.total_duration) || '')),
            row('Resolution', esc(res)),
            row('Codecs', esc([f.video_codec, f.audio_codec].filter(Boolean).join(' · '))),
            row('Bitrate', f.bitrate ? esc(Math.round(f.bitrate / 1000) + ' kbit/s') : ''),
            row('Loudness and copies', `<span class="fi-audio" data-file="${esc(path)}"></span>`),
            row('Added', esc(when(f.created_at || f.added_at))),
            row('Modified', esc(when(f.modified))),
            f.indexed === false ? row('Metadata', '<span class="rv-quiet">not indexed yet — folder ⋯ → Index metadata</span>') : '',
        ].join('');
        const tx = (d && d.segments) ? [
            row('Model', esc(st.model || d.model || f.tx_model || '')),
            row('Language', esc(((d.language || f.tx_lang || '') + (st.language_requested && st.language_requested !== d.language ? ` (asked for ${st.language_requested})` : '')).toUpperCase())),
            row('Vocal isolation', esc(d.vocal_isolation ? (d.vocal_model || 'on') : 'off')),
            row('Speakers told apart', esc(yes(st.diarized ?? st.diarize))),
            row('Word timestamps', esc(yes(st.word_timestamps))),
            row('Voice vectors', esc(yes(st.speaker_vectors))),
            row('Transcribed', esc(when(f.tx_date))),
            row('Processing', esc([d.processing_time || f.processing_time, speed && speed + ' real time'].filter(Boolean).join(' · '))),
            row('Segments · words · characters', esc([f.tx_seg_count, f.tx_word_count, f.tx_char_count].map(x => (x || 0).toLocaleString()).join(' · '))),
            row('App version', st.app_version || d.app_version ? esc('v' + (st.app_version || d.app_version)) : ''),
        ].join('') : '';
        return `<div class="file-info-panel rv-details">
            <section><h4>File and media</h4><table class="rv-dl">${media}</table></section>
            ${tx ? `<section><h4>Transcription</h4><table class="rv-dl">${tx}</table></section>` : ''}
            <section><h4>Where it is</h4><table class="rv-dl">
                <tr><th>Path</th><td><span class="rv-path">${esc(path)}</span>
                    <button class="fi-chip fi-path" onclick="fiCopyPath(this)" data-path="${esc(path)}" title="Copy the path — open it in your file manager from there"><small>Path</small>📋 copy</button></td></tr>
                <tr><th>Incidents</th><td><button class="fi-chip fi-incidents" onclick="fiShowIncidents(this)" data-file="${esc(path)}"
                    title="Warnings and errors from all runs of this file"><small>Incidents</small><span class="fi-incidents-n">…</span></button></td></tr>
            </table></section>
        </div>`;
    }

    /* The two values the server is asked for afterwards. */
    function detailsFill(f) {
        const path = typeof filePath === 'function' ? filePath(f || {}) : '';
        if (typeof fiAudio === 'function') fiAudio(path);
        if (typeof fiCountIncidents === 'function') fiCountIncidents(path);
    }

    /* ── Tabs ─────────────────────────────────────────────────────────── */

    function count(tab) {
        if (typeof _txView === 'undefined') return '';
        if (tab === 'summary') return _txView.sumCount ? String(_txView.sumCount) + (_txView.sumPending ? ' ⏳' : '') : (_txView.sumPending ? '⏳' : '');
        if (tab === 'translation') return _txView.tracks && _txView.tracks.length ? String(_txView.tracks.length) : '';
        if (tab === 'clips' && typeof Clips !== 'undefined') return Clips.count() ? String(Clips.count()) : '';
        return '';
    }

    function tabsHtml() {
        return `<nav class="rv-tabs vx-tabs" id="rvTabs" role="tablist" aria-label="This recording">`
            + TABS.map(([id, label, icon]) => {
                const n = count(id);
                return `<button class="vx-tab rv-tab" data-tab="${id}" role="tab" onclick="Rec.tab('${id}')">`
                    + `<svg class="ui-ico"><use href="#${icon}"/></svg>${label}${n ? ` <span class="vx-count">${esc(n)}</span>` : ''}</button>`;
            }).join('')
            + `<div class="bb-more rv-tab-more-wrap" id="rvTabMoreWrap" hidden><button class="vx-tab" onclick="Rec.menu('rvTabMore', event)" aria-haspopup="menu">More ▾</button>`
            + `<div class="bb-menu down" id="rvTabMore" role="menu" hidden></div></div></nav>`;
    }

    /* Counts on the tabs changed (translations, summaries). */
    function renderTabs() {
        document.querySelectorAll('#rvTabs .rv-tab').forEach(b => {
            const n = count(b.dataset.tab);
            const label = TABS.find(t => t[0] === b.dataset.tab)[1];
            b.innerHTML = `<svg class="ui-ico"><use href="#${TABS.find(t => t[0] === b.dataset.tab)[2]}"/></svg>${label}${n ? ` <span class="vx-count">${esc(n)}</span>` : ''}`;
        });
        fit();
        mark();
    }

    /* Tabs that don't fit move into More: hidden from the end (never the
       one shown) until the bar fits with the More button in it. */
    function fit() {
        const bar = el('rvTabs'), wrap = el('rvTabMoreWrap'), menuBox = el('rvTabMore');
        if (!bar || !wrap || !menuBox) return;
        const tabs = [...bar.querySelectorAll('.rv-tab')];
        tabs.forEach(t => { t.hidden = false; });
        wrap.hidden = true;
        if (bar.scrollWidth <= bar.clientWidth + 1) { menuBox.innerHTML = ''; return; }
        wrap.hidden = false;
        const hidden = [];
        for (let i = tabs.length - 1; i >= 0 && bar.scrollWidth > bar.clientWidth + 1; i--) {
            if (tabs[i].dataset.tab === current) continue;
            tabs[i].hidden = true;
            hidden.unshift(tabs[i].dataset.tab);
        }
        menuBox.innerHTML = hidden.map(id => {
            const [, label] = TABS.find(t => t[0] === id);
            return `<button class="bm-item" onclick="Rec.closeMenus();Rec.tab('${id}')">${label}</button>`;
        }).join('');
        wrap.hidden = !hidden.length;
    }

    function mark() {
        document.querySelectorAll('#rvTabs .rv-tab').forEach(b => {
            const on = b.dataset.tab === current;
            b.classList.toggle('active', on);
            b.setAttribute('aria-selected', on ? 'true' : 'false');
        });
    }

    /* Show a tab. The transcript and the translation share the line list
       (a translation is shown on the lines, with their times). */
    function tab(t, opts) {
        if (!TABS.some(x => x[0] === t)) t = 'transcript';
        const was = current;
        current = t;
        mark();
        const pane = t === 'transcript' || (t === 'translation' && hasTranslation()) ? 'text' : t;
        document.querySelectorAll('#transcriptArea .rv-pane').forEach(p => { p.hidden = p.dataset.pane !== pane; });
        const tools = el('rvTxTools'), trTools = el('rvTrTools');
        if (tools) tools.hidden = t !== 'transcript';
        if (trTools) trTools.hidden = t !== 'translation';
        if (typeof switchTxView === 'function' && typeof _txView !== 'undefined') {
            if (t === 'transcript' && _txView.mode !== 'original') switchTxView('original');
            if (t === 'translation' && hasTranslation() && _txView.mode !== 'translation') switchTxView('translation');
            if (t === 'summary') {
                // Already the summary view: read the saved ones again — one
                // may have been added since the list was drawn
                if (_txView.mode !== 'summary') switchTxView('summary');
                else if (typeof _sumViewRender === 'function') _sumViewRender();
            }
        }
        if (t === 'translation') renderTranslationTools();
        if (t === 'statistics' && typeof txsRenderInto === 'function') txsRenderInto(el('rvStats'));
        if (t === 'clips') showClips();
        if (t === 'speakers') showSpeakers();
        if (t === 'analyses' && typeof RecAnalyses !== 'undefined') RecAnalyses.load();
        fit();
        if (!(opts && opts.quiet) && was !== t && typeof Shell !== 'undefined') {
            Shell.setPref('last_tab', {...(Shell.pref('last_tab') || {}), recording: t});
        }
    }

    function hasTranslation() {
        return typeof _txView !== 'undefined' && _txView.tracks && _txView.tracks.length > 0;
    }

    /* The translation tab's own row: the languages there are, the
       original beside it, and a new translation. */
    function renderTranslationTools() {
        const box = el('rvTrTools'), empty = el('rvTrEmpty');
        const fn = escJs(activeFile || '');
        const tracks = typeof _txView !== 'undefined' ? (_txView.tracks || []) : [];
        if (box) {
            box.innerHTML = tracks.map(tr => `<button class="fl-opt${_txView.lang === tr.language ? ' on' : ''}" onclick="switchTxView('translation','${escJs(tr.language)}')">${esc(tr.language)}</button>`).join('')
                + (tracks.length ? `<label class="view-opt"><input type="checkbox" ${_txView.showOriginal ? 'checked' : ''} onchange="_txView.showOriginal=this.checked;_txApplyTranslation()"> Original beside it</label>` : '')
                + `<span class="rv-gap"></span><button class="tb-btn" onclick="openTranslateModal('${fn}')"><svg class="ui-ico"><use href="#i-chat"/></svg><span class="tb-label">Translate into …</span></button>`;
        }
        if (empty) empty.innerHTML = `<div class="rv-empty"><p>No translation yet.</p>`
            + `<button class="tb-btn" onclick="openTranslateModal('${fn}')"><svg class="ui-ico"><use href="#i-chat"/></svg><span class="tb-label">Translate into …</span></button></div>`;
    }

    /* ── Speakers: the recognition, and per voice its numbers ────────── */

    async function showSpeakers() {
        const box = el('speakerRecognition');
        if (box && activeFile && box.dataset.loadedFor !== activeFile && typeof loadSpeakerRecognition === 'function') {
            box.dataset.loadedFor = activeFile;
            loadSpeakerRecognition(activeFile);
        }
        speakerExtras();
        closestBelow();
    }

    /* Words, pace, turns and when first heard, on each voice's row. */
    function speakerExtras() {
        const d = activeTranscript;
        if (!d) return;
        const per = {};
        let prev = null;
        for (const s of d.segments || []) {
            const k = s.speaker;
            if (!k) continue;
            const p = per[k] || (per[k] = {words: 0, sec: 0, turns: 0, first: null});
            p.words += String(s.text || '').split(/\s+/).filter(Boolean).length;
            p.sec += Math.max(0, (Number(s.end) || 0) - (Number(s.start) || 0));
            if (prev !== k) p.turns++;
            if (p.first === null) p.first = Number(s.start) || 0;
            prev = k;
        }
        document.querySelectorAll('#speakerStats .sp-stat-row').forEach(row => {
            const p = per[row.dataset.sp];
            if (!p) return;
            let x = row.querySelector('.rv-sp-extra');
            if (!x) { x = document.createElement('div'); x.className = 'rv-sp-extra'; row.appendChild(x); }
            const wpm = p.sec > 0 ? Math.round(p.words / (p.sec / 60)) : 0;
            x.innerHTML = `${plural(p.words, 'word')} · ${wpm} words/min · ${plural(p.turns, 'turn')}`
                + ` · first heard <button class="rv-time" onclick="seekTo(${p.first})">${esc(fmtLen(p.first) || '0:00')}</button>`;
        });
    }

    /* Voices no named person reaches: the closest one below the
       threshold, with a way to take it. */
    async function closestBelow() {
        const fn = activeFile;
        if (!fn) return;
        const q = t => fetch(`/api/speaker/recording?file=${encodeURIComponent(fn)}${t ? '&threshold=' + t : ''}`)
            .then(r => r.ok ? r.json() : null).catch(() => null);
        const [main, low] = await Promise.all([q(null), q(0.4)]);
        if (activeFile !== fn || !main || !low || !main.windows) return;
        const named = new Set((main.people || []).flatMap(p => p.labels || []));
        const best = {};
        for (const p of low.people || []) {
            for (const label of p.labels || []) {
                if (named.has(label)) continue;
                const v = Number(p.centroid_value ?? p.best_value) || 0;
                if (!best[label] || v > best[label].v) best[label] = {person: p.person, v};
            }
        }
        document.querySelectorAll('#speakerStats .sp-stat-row').forEach(row => {
            const label = row.dataset.sp;
            row.querySelector('.rv-sp-near')?.remove();
            const b = best[label];
            if (named.has(label)) return;
            const x = document.createElement('div');
            x.className = 'rv-sp-near';
            x.innerHTML = b
                ? `Not recognized — closest named voice: <b>${esc(b.person)}</b> ${b.v.toFixed(2)}`
                  + (b.v < Number(main.threshold) ? ` (below ${Number(main.threshold).toFixed(2)})` : '')
                  + ` <button class="fl-opt" onclick="Rec.itIs('${escJs(label)}', '${escJs(b.person)}')">It is ${esc(b.person)}</button>`
                : `Not recognized — no named voice comes close`;
            x.innerHTML += ` <button class="fl-opt" onclick="mapForSpeaker('${escJs(fn)}', '${escJs(label)}')">Name this voice</button>`;
            row.appendChild(x);
        });
    }

    /* "It is X": the name goes into the voice's name field; Save names keeps it. */
    function itIs(label, person) {
        const input = [...document.querySelectorAll('#speakerStats .sp-stat-name')].find(i => i.dataset.speakerId === label);
        if (!input) return;
        input.value = person;
        if (typeof onSpeakerRename === 'function') onSpeakerRename(input);
        if (typeof Toast !== 'undefined') Toast.info(`${label} → ${person} — “Save names” keeps it`);
    }

    /* The clip editor sits in the Clips tab while it is shown; before
       the area is rebuilt (another file, a reload of the transcript) it
       goes back beside it, or it would be thrown away with the panes. */
    function parkClip() {
        const area = el('transcriptArea'), ed = el('clipEditor');
        if (area && ed && area.contains(ed)) area.parentElement.insertBefore(ed, area.nextSibling);
    }

    /* The Clips tab: clips.js draws it, the clip editor is its timeline. */
    function showClips() {
        if (typeof Clips !== 'undefined') Clips.show();
    }

    /* A link that opens this recording at this moment (the ?open=&t=
       link the browser extension uses too). */
    async function copyLink(t) {
        const dir = (el('dirInput')?.value || '').trim().replace(/\/+$/, '');
        if (!dir || !activeFile) return;
        const url = `${location.origin}${location.pathname}?open=${encodeURIComponent(dir + '/' + activeFile)}&t=${Math.floor(Number(t) || 0)}`;
        try {
            await navigator.clipboard.writeText(url);
            if (typeof Toast !== 'undefined') Toast.success('Link to this moment copied');
        } catch (e) {
            // Without the clipboard API (a plain http address): the old way
            const ta = document.createElement('textarea');
            ta.value = url;
            document.body.appendChild(ta);
            ta.select();
            const ok = document.execCommand && document.execCommand('copy');
            ta.remove();
            if (typeof Toast !== 'undefined') ok ? Toast.success('Link to this moment copied') : Toast.info(url);
        }
    }

    /* The speaker chips above the lines: one lit at a time. */
    function spChip(btn) {
        const on = btn && btn.dataset.sp && typeof activeSpeaker !== 'undefined' && activeSpeaker === btn.dataset.sp;
        document.querySelectorAll('#rvSpChips .fl-opt:not(.rv-sp-more)').forEach(b =>
            b.classList.toggle('on', on ? b === btn : !b.dataset.sp));
    }

    /* Called after renderTranscript() put the panes in place. */
    function rendered() {
        const t = forceTranscript ? 'transcript' : remembered();
        forceTranscript = false;
        tab(t, {quiet: true});
        lanes();
        requestAnimationFrame(fit);
    }

    /* ── Who speaks when ───────────────────────────────────────────────── */

    /* One lane per person, not per label: the voices the diarization told
     * apart but that carry the same name (two labels both named after one
     * person) share a lane. The lanes go by speaking time, the longest
     * first; with many speakers only the first few show until "all" is
     * asked for (remembered). Labels that never speak get no lane. */
    const LANES_FEW = 6;

    function lanesAll() {
        try { return localStorage.getItem('rv_lanes_all') === '1'; } catch (e) { return false; }
    }

    function lanesToggle(ev) {
        ev?.stopPropagation();
        try { localStorage.setItem('rv_lanes_all', lanesAll() ? '0' : '1'); } catch (e) { /* stays as is */ }
        lanes();
        playhead();
        requestAnimationFrame(fit);
    }

    function lanes() {
        const box = el('rvLanes');
        const d = typeof activeTranscript !== 'undefined' ? activeTranscript : null;
        if (!box) return;
        const segs = d && d.segments || [];
        const total = Number(d && d.total_duration_sec) || (segs.length ? Number(segs[segs.length - 1].end) || 0 : 0);
        const speakers = (d && d.speakers || []).filter(s => s && s !== '—' && s !== '-');
        if (!total || !speakers.length) { box.hidden = true; box.innerHTML = ''; return; }
        const colors = typeof SPEAKER_COLORS !== 'undefined' ? SPEAKER_COLORS : ['#60a5fa'];
        const byName = new Map();
        speakers.forEach((s, i) => {
            const name = typeof getSpeakerName === 'function' ? (getSpeakerName(s) || s) : s;
            if (!byName.has(name)) byName.set(name, {name, color: colors[i % colors.length], labels: [], segs: [], sec: 0});
            byName.get(name).labels.push(s);
        });
        const of = new Map();
        for (const g of byName.values()) for (const l of g.labels) of.set(l, g);
        for (const x of segs) {
            const g = of.get(x.speaker);
            if (!g) continue;
            const a = Math.max(0, Number(x.start) || 0), b = Math.max(a, Number(x.end) || 0);
            g.segs.push([a, b]);
            g.sec += b - a;
        }
        const groups = [...byName.values()].filter(g => g.segs.length).sort((a, b) => b.sec - a.sec);
        if (!groups.length) { box.hidden = true; box.innerHTML = ''; return; }
        const all = lanesAll() || groups.length <= LANES_FEW + 1;
        const shown = all ? groups : groups.slice(0, LANES_FEW);
        box.hidden = false;
        const fmt = sec => { const m = Math.floor(sec / 60), r = Math.round(sec % 60); return m ? `${m} min ${r} s` : `${r} s`; };
        let html = shown.map(g => {
            const bars = g.segs.map(([a, b]) =>
                `<span style="left:${(a / total * 100).toFixed(3)}%;width:${Math.max(0.15, (b - a) / total * 100).toFixed(3)}%;background:${g.color}"></span>`).join('');
            const tip = `${g.name} · speaks ${fmt(g.sec)}` + (g.labels.length > 1 ? ` · ${g.labels.length} voices with this name` : '') + ' — click to jump there';
            return `<div class="rv-lane" title="${esc(tip)}"><span class="rv-lane-name" style="color:${g.color}">${esc(g.name)}</span>`
                + `<div class="rv-lane-track" data-total="${total}">${bars}</div></div>`;
        }).join('');
        if (groups.length > LANES_FEW + 1) {
            html += all
                ? `<button type="button" class="rv-lanes-more" onclick="Rec.lanesToggle(event)">Show the ${LANES_FEW} who speak most ▴</button>`
                : `<button type="button" class="rv-lanes-more" onclick="Rec.lanesToggle(event)">${groups.length - LANES_FEW} more speakers ▾</button>`;
        }
        box.innerHTML = html + `<div class="rv-playhead" id="rvPlayhead"></div>`;
    }

    function laneClick(e) {
        const track = e.target.closest('.rv-lane-track');
        if (!track) return;
        const r = track.getBoundingClientRect();
        const t = Math.max(0, Math.min(1, (e.clientX - r.left) / r.width)) * Number(track.dataset.total || 0);
        if (typeof seekTo === 'function') seekTo(t);
        const v = el('videoPlayer');
        if (v && v.paused && typeof v.play === 'function') v.play().catch(() => {});
    }

    function playhead() {
        const v = el('videoPlayer'), ph = el('rvPlayhead'), track = document.querySelector('#rvLanes .rv-lane-track');
        if (!v || !ph || !track) return;
        const total = Number(track.dataset.total || 0);
        if (!total) return;
        const box = el('rvLanes').getBoundingClientRect(), r = track.getBoundingClientRect();
        ph.style.left = (r.left - box.left + Math.min(1, (v.currentTime || 0) / total) * r.width) + 'px';
    }

    function init() {
        // Whatever rebuilds the area (a new file, an error, a deletion):
        // a clip editor that was in its Clips tab goes back beside it, whole
        const area = el('transcriptArea'), ed = el('clipEditor');
        if (area && ed) {
            new MutationObserver(() => {
                if (!ed.isConnected) area.parentElement.insertBefore(ed, area.nextSibling);
            }).observe(area, {childList: true});
        }
        el('rvLanes')?.addEventListener('click', laneClick);
        el('videoPlayer')?.addEventListener('timeupdate', playhead);
        document.addEventListener('click', e => { if (!e.target.closest('.rv-acts, .rv-tab-more-wrap')) closeMenus(); });
        window.addEventListener('resize', () => { fit(); playhead(); });
        // A search hit opens the Transcript tab, on its line
        if (typeof goToSearchResult === 'function') {
            const orig = goToSearchResult;
            window.goToSearchResult = function (...a) { forceTranscript = true; return orig.apply(this, a); };
        }
        if (typeof Shell !== 'undefined') Shell.onPrefs(() => { if (!activeTranscript) current = remembered(); });
    }

    document.addEventListener('DOMContentLoaded', init);
    return {parkClip, headerHtml, stateHtml, detailsHtml, detailsFill, itIs, tabsHtml, renderTabs, tab, rendered, menu, closeMenus, lanes, lanesToggle, spChip, copyLink, TABS,
            renderTranslationTools, get current() { return current; },
            set forceTranscript(v) { forceTranscript = !!v; }};
})();

// ── A recording's Analyses tab ──────────────────────────────────────────
// The analyses that read this recording (/api/analyses/for — built from
// each analysis's list of files), and a new one with it already chosen.
const RecAnalyses = (() => {
    let seq = 0;

    async function load() {
        const box = document.getElementById('rvAnalyses');
        const fn = activeFile;
        if (!box || !fn) return;
        const my = ++seq;
        box.innerHTML = '<div class="rv-empty"><p>Loading …</p></div>';
        let list = [];
        try {
            const d = await (await fetch('/api/analyses/for?file=' + encodeURIComponent(fn))).json();
            list = d.analyses || [];
        } catch (e) { /* shows as none */ }
        if (my !== seq || activeFile !== fn) return;
        const when = t => t ? new Date(t * 1000).toLocaleDateString(undefined, {day: 'numeric', month: 'short', year: 'numeric'}) : '';
        const rows = list.map(a => `<button type="button" class="rv-an-row" onclick="openAnalysis('${escJs(a.id)}')" title="${esc(a.task || a.title || '')}">`
            + `<span class="rv-an-title">${esc(a.title || a.task || 'Analysis')}</span>`
            + `<span class="rv-an-meta">${esc([a.counts && a.counts.transcribed ? plural(a.counts.transcribed, 'recording') : '', a.status, when(a.created)].filter(Boolean).join(' · '))}</span></button>`).join('');
        box.innerHTML = `<div class="rv-an">
            <div class="rv-an-head"><span>${list.length ? plural(list.length, 'analysis', 'analyses') + ' read this recording' : 'No analysis has read this recording yet.'}</span>
                <button class="tb-btn" onclick="RecAnalyses.newWith()"><svg class="ui-ico"><use href="#i-layers"/></svg><span class="tb-label">New analysis with it</span></button></div>
            ${rows}</div>`;
    }

    /* Analyses, its form open with just this recording chosen. */
    function newWith() {
        if (!activeFile || typeof AN === 'undefined') return;
        AN.selection = {base_dir: document.getElementById('dirInput')?.value || '', filenames: [activeFile]};
        AN.src = 'selection';
        AN.cur = null;
        if (typeof openAnalyses === 'function') openAnalyses();
        if (typeof anShowForm === 'function') anShowForm();
    }

    return {load, newWith};
})();
