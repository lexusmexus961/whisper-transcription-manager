// ── File Info Panel Helper ──
// (The info panel's chips became the Details tab — recording.js: detailsHtml)

/* The audio analysis' chips of a recording: its loudness, and "≋ n" when
 * it exists elsewhere too — the same recording, or inside a longer one (or
 * holding others). A click lists them in the search view's tab. */
async function fiAudio(path) {
    const box = [...document.querySelectorAll('.fi-audio')].find(el => el.dataset.file === path);
    if (!box) return;
    let d;
    try { d = await api('/api/audio/file?file=' + encodeURIComponent(path)); } catch (e) { return; }
    if (!d || !d.analyzed || box.dataset.file !== path) return;
    const f = d.file || {};
    const out = [];
    if (f.lufs !== null && f.lufs !== undefined) {
        const more = [`Integrated loudness ${Number(f.lufs).toFixed(1)} LUFS (EBU R128)`];
        if (f.lra !== null && f.lra !== undefined) more.push(`Loudness range ${Number(f.lra).toFixed(1)} LU`);
        if (f.peak_db !== null && f.peak_db !== undefined) more.push(`Peak ${Number(f.peak_db).toFixed(1)} dBFS`);
        if (f.silence !== null && f.silence !== undefined) more.push(`Near silence ${Math.round(f.silence * 100)} %`);
        if (f.clipped) more.push(`Clipped ${(f.clipped * 100).toFixed(2)} % of the samples`);
        out.push(`<span class="fi-chip fi-hover" title="${esc(more.join('\n'))}"><small>Loudness</small>${Number(f.lufs).toFixed(0)} LUFS</span>`);
    }
    const n = (d.copies || 0) + (d.contains || 0) + (d.contained_in || 0);
    if (n) {
        const what = [d.copies ? `the same recording ${d.copies}× elsewhere` : '',
                      d.contained_in ? `inside ${d.contained_in} longer one${d.contained_in === 1 ? '' : 's'}` : '',
                      d.contains ? `holds ${d.contains} other${d.contains === 1 ? '' : 's'}` : ''].filter(Boolean).join(', ');
        out.push(`<button class="fi-chip orange" onclick="amShowFile(this.closest('.fi-audio').dataset.file, ${d.copies ? 1 : 0})"
            title="${esc(what[0].toUpperCase() + what.slice(1))} — show them"><small>≋</small>${n} ${n === 1 ? 'copy' : 'copies'}</button>`);
    }
    box.innerHTML = out.join('');
}

/* The full path of a file.
 *
 * `f.name` is the path RELATIVE to the root currently being resolved
 * against (docs/PATHS.md) — that's in the directory field. Both together
 * give the path you can paste into a terminal or file manager. If the
 * field is missing, at least the relative path remains;
 * half a path is more useful than none at all.
 */
function filePath(f) {
    const root = (document.getElementById('dirInput')?.value || '').replace(/\/+$/, '');
    const rel = f.name || f.display_name || '';
    return root ? `${root}/${rel}` : rel;
}

/* Write the incident count into the chip (one query per opened file). */
async function fiCountIncidents(path) {
    const btns = [...document.querySelectorAll('.fi-incidents')].filter(b => b.dataset.file === path);
    if (!btns.length) return;
    try {
        const r = await fetch(`/api/incidents?file=${encodeURIComponent(path)}&limit=1`);
        const d = await r.json();
        const n = d.total || 0;
        for (const b of btns) {
            const z = b.querySelector('.fi-incidents-n');
            if (z) z.textContent = n ? String(n) : 'none';
            b.classList.toggle('has-incidents', n > 0);
            b.dataset.total = String(n);
        }
    } catch (e) {
        for (const b of btns) { const z = b.querySelector('.fi-incidents-n'); if (z) z.textContent = '?'; }
    }
}

/* Expand and collapse the list under the details. */
async function fiShowIncidents(btn) {
    const panel = btn.closest('.file-info-panel');
    if (!panel) return;
    let box = panel.querySelector('.fi-incidents-list');
    if (box) { box.remove(); btn.classList.remove('open'); return; }
    box = document.createElement('div');
    box.className = 'fi-incidents-list';
    box.innerHTML = '<div class="fi-incident empty">loading …</div>';
    panel.appendChild(box);
    btn.classList.add('open');
    const path = btn.dataset.file || '';
    try {
        const r = await fetch(`/api/incidents?file=${encodeURIComponent(path)}&limit=200`);
        const d = await r.json();
        box.innerHTML = fiIncidentsHtml(d);
    } catch (e) {
        box.innerHTML = `<div class="fi-incident empty">Could not load incidents (${esc(String(e))})</div>`;
    }
}

const FI_INCIDENT_ICON = {warning: '⚠', error: '❌', oom: '🔁', retry: '🔁', system: '🛈'};

function fiIncidentsHtml(d) {
    const items = d.items || [];
    if (!items.length) return '<div class="fi-incident empty">No incidents — all runs of this file completed without warning.</div>';
    const header = `<div class="fi-incident header">${plural(d.total, 'incident')}` +
        (d.events && d.events !== d.total ? ` (${d.events} events)` : '') +
        Object.entries(d.kinds || {}).map(([a, n]) => ` · ${FI_INCIDENT_ICON[a] || ''} ${n} ${a}`).join('') + '</div>';
    return header + items.map(v => {
        const t = new Date((v.timestamp || 0) * 1000);
        const when = isNaN(t) ? '' : t.toLocaleString(undefined, {dateStyle: 'short', timeStyle: 'short'});
        return `<div class="fi-incident ${esc(v.kind)}">` +
            `<span class="fi-incident-time">${esc(when)}</span>` +
            `<span class="fi-incident-art" title="${esc(v.kind)}">${FI_INCIDENT_ICON[v.kind] || '•'}</span>` +
            (v.step ? `<span class="fi-incident-step">${esc(v.step)}</span>` : '') +
            `<span class="fi-incident-text">${esc(v.text)}${v.count > 1 ? ` <b>×${v.count}</b>` : ''}</span>` +
            `</div>`;
    }).join('');
}

function fiCopyPath(btn) {
    const path = btn.dataset.path || '';
    const done = () => {
        const old_value = btn.innerHTML;
        btn.innerHTML = '<small>Path</small>✅ copied';
        setTimeout(() => { btn.innerHTML = old_value; }, 1500);
    };
    // Without HTTPS there is no `navigator.clipboard` — and the app usually
    // runs on a local network over plain http. The old way isn't a fallback here,
    // it's the normal case (same spot as in settings.js).
    if (navigator.clipboard && window.isSecureContext) {
        navigator.clipboard.writeText(path).then(done, () => _copyOldPath(path, done));
    } else {
        _copyOldPath(path, done);
    }
}

function _copyOldPath(text, done) {
    const ta = document.createElement('textarea');
    ta.value = text;
    ta.style.position = 'fixed';
    ta.style.opacity = '0';
    document.body.appendChild(ta);
    ta.select();
    try { document.execCommand('copy'); } catch (e) {}
    document.body.removeChild(ta);
    done();
}


// ── Select file ──
async function selectFile(f) {
    // The clip editor may sit in the Clips tab: out of the area before it
    // is rebuilt below
    if (typeof Rec !== 'undefined') Rec.parkClip();
    // Clear multi-selection on normal click
    if (selectedFiles.size > 0) {
        selectedFiles.clear();
        updateSelectionUI();
        syncSelectedClasses();
    }
    const switched = activeFile !== f.name;
    activeFile = f.name;
    activeFileObj = f;
    if (switched) {
        // The previous file's transcript must not outlive the switch: on a
        // load error (or a file without a transcript) it stayed active, and
        // clicking one of its segments seeked the NEW media to the old
        // file's times; export and the editors acted on it too.
        activeTranscript = null;
        // The clip editor holds the previous file's in/out points, duration
        // and waveform — a cut taken from them would land on the new file.
        // It is parked for that file (clip.js) and opens again, points
        // intact, when its transcript is shown again.
        if (typeof clipParkForSwitch === 'function') clipParkForSwitch();
        if (typeof Clips !== 'undefined') Clips.fileChanged();
    }
    // ONLY move the active marker — the earlier renderFileList() call
    // rebuilt 2000 DOM nodes per click and ran filter+sort over the
    // entire library (multi-second clicks on a large library).
    markActiveFile(f.name);

    const area = document.getElementById('transcriptArea');
    const player = document.getElementById('playerBar');
    const video = document.getElementById('videoPlayer');

    // Hide progress/live preview when navigating away from the file
    // currently being processed …
    if (queueCurrentFile && !isRunningFile(f)) {
        document.getElementById('progressOverlay').classList.remove('visible');
        document.getElementById('liveBadge').style.display = 'none';
    }
    // … and show it again when returning to it
    if (queueCurrentFile && isRunningFile(f) && f.status !== 'done') {
        document.getElementById('progressOverlay').classList.add('visible');
    }
    // Queue lanes: the overlay follows the open file's job when it runs —
    // switch now, not a queue tick later with another file's log (batch.js)
    if (typeof syncWatchedJob === 'function') syncWatchedJob();

    // Show player (video and audio branches were identical → merged)
    player.classList.add('visible');
    player.classList.remove('audio-only');
    video.src = `/api/video/${encodeURIComponent(f.name)}`;
    document.getElementById('resizeHandle').classList.add('visible');
    // Audio has no picture — a 280 px black box is wasted space; shrink the
    // bar to its controls. Decided on the metadata, not the extension
    // (.mkv/.m4a can be either).
    video.addEventListener('loadedmetadata', () => {
        if (activeFileObj !== f) return;
        const audio = video.videoWidth === 0;
        player.classList.toggle('audio-only', audio);
        document.getElementById('resizeHandle').classList.toggle('visible', !audio);
    }, {once: true});

    if (f.status === 'done' || f.status === 'orphan') {
        // 'orphan': file gone, transcript there — viewing still works
        // (the player just stays black then).
        // The old file's segments stay clickable until the new transcript
        // is in — replace them with a loading state right away.
        if (switched) {
            area.innerHTML = '<div class="empty-state"><div class="icon">⏳</div><p>Loading transcript …</p></div>';
        }
        await loadTranscript(f.name);
    } else {
        // Before a transcript exists: open, waiting, running, failed,
        // skipped — each with what can be done now (recording.js)
        area.innerHTML = Rec.stateHtml(f) + (f.has_deriv ? buildDerivativesSection(f.name) : '');
        Rec.detailsFill(f);
        if (f.has_deriv) {
            activeTranscript = null;
            txViewsInit(f.name, true);
        }
    }
}

// No transcript, but summaries/translations still exist: deletion
// deliberately doesn't remove them (by design), so they must be
// reachable even without the original. Same containers as in
// renderTranscript — llmtools.js fills them via the same view switcher.
function buildDerivativesSection(name) {
    return `
        <div style="margin:16px 20px 0;padding:10px 12px;border-radius:8px;
                    background:var(--bg-1);border:1px solid var(--border);
                    font-size:0.8rem;color:var(--tx-2);display:flex;
                    align-items:center;gap:10px">
            <span style="flex:1">🤖 No original transcript — the derivatives
                below have been preserved. Re-transcribing
                will not overwrite them.</span>
            <button class="action-link" style="color:var(--red);flex:0 0 auto"
                    onclick="deleteTx('${escJs(name)}')"
                    title="Choose what should be deleted">🗑 Delete</button>
        </div>
        <div class="tx-viewbar" id="txViewBar"></div>
        <div id="segList"></div>
        <div id="sumView" style="display:none"></div>`;
}

// ── Load transcript ──
async function loadTranscript(filename) {
    // Race guard: on fast file switching, a late-arriving response must
    // not overwrite the file selected in the meantime.
    const requested = filename;
    // A transcript still active here belongs to this very file: selectFile()
    // drops it on every switch. That is a RELOAD (after Find & Replace, a
    // finished job, …) — a failed reload keeps the good view and only says
    // so; the area is taken over only when it showed another file before.
    const had = activeTranscript;
    const failed = (html, toast) => {
        if (had && activeTranscript === had) {
            if (typeof Toast !== 'undefined') Toast.error(`Reload failed — still showing the loaded version (${toast})`);
            return;
        }
        activeTranscript = null;
        document.getElementById('transcriptArea').innerHTML = html;
        if (toast && typeof Toast !== 'undefined') Toast.error('Failed to load transcript');
    };
    try {
        const r = await fetch(`/api/transcript/${encodeURIComponent(filename)}`);
        if (activeFile !== requested) return;  // user has clicked on
        if (!r.ok) {
            if (r.status === 404) {
                // Gone for real (deleted): the old view must not stay
                // active in export/highlight, reload or not.
                activeTranscript = null;
                document.getElementById('transcriptArea').innerHTML =
                    '<div class="empty-state"><div class="icon">📝</div><p>Transcript not found</p></div>';
            } else {
                failed(`<div class="empty-state"><div class="icon">⚠</div><p>Error loading (HTTP ${r.status})</p></div>`,
                       `HTTP ${r.status}`);
            }
            return;
        }
        const data = await r.json();
        if (activeFile !== requested) return;  // check again after await
        if (!data || data.error) {
            const msg = (data && data.error) || 'empty answer';
            failed(`<div class="empty-state"><div class="icon">⚠</div><p>${esc(msg)}</p></div>`, msg);
            return;
        }
        activeTranscript = data;
        // Safety net: the server could deliver an object without segments
        if (!Array.isArray(activeTranscript.segments)) {
            activeTranscript.segments = [];
        }
        activeSpeaker = null;
        wordScoreVisible = false;
        renderTranscript();
        applySubtitleState();   // persistent subtitle state (player)
        if (typeof clipRestoreFor === 'function') clipRestoreFor(requested);
    } catch(e) {
        console.error('loadTranscript error:', e);
        if (activeFile !== requested) return;
        // An error state instead of the previous file's view (whose segments
        // would seek this file's media to the wrong times)
        const msg = e.message || String(e);
        failed(`<div class="empty-state"><div class="icon">⚠</div><p>Failed to load the transcript: ${esc(msg)}</p></div>`, msg);
    }
}

let speakerNames = {};  // in-memory renames: "SPEAKER_00" → "Alice"
let speakerNamesDirty = false;

function getSpeakerName(id) {
    return speakerNames[id] || id;
}

function computeSpeakerStats(d) {
    const stats = {};
    (d.segments || []).forEach(s => {
        const sp = s.speaker || '—';
        if (!stats[sp]) stats[sp] = { dur: 0, count: 0 };
        stats[sp].dur += (s.end - s.start);
        stats[sp].count++;
    });
    const totalDur = Object.values(stats).reduce((a, b) => a + b.dur, 0);
    Object.values(stats).forEach(s => {
        s.pct = totalDur > 0 ? (s.dur / totalDur * 100) : 0;
    });
    return { stats, totalDur };
}

function buildPieChart(speakers, stats, spMap, size=120) {
    const r = size / 2;
    const colors = SPEAKER_COLORS;
    let cumAngle = -Math.PI / 2;
    let paths = '';

    speakers.forEach((sp, i) => {
        const s = stats[sp];
        if (!s || s.pct === 0) return;
        const angle = (s.pct / 100) * Math.PI * 2;
        const x1 = r + r * Math.cos(cumAngle);
        const y1 = r + r * Math.sin(cumAngle);
        cumAngle += angle;
        const x2 = r + r * Math.cos(cumAngle);
        const y2 = r + r * Math.sin(cumAngle);
        const large = angle > Math.PI ? 1 : 0;
        const ci = (spMap[sp] ?? i) % colors.length;

        if (s.pct >= 99.9) {
            paths += `<circle cx="${r}" cy="${r}" r="${r}" style="fill:${colors[ci]}"/>`;
        } else {
            paths += `<path d="M${r},${r} L${x1},${y1} A${r},${r} 0 ${large},1 ${x2},${y2} Z" style="fill:${colors[ci]}"/>`;
        }
    });

    return `<svg width="${size}" height="${size}" viewBox="0 0 ${size} ${size}">${paths}</svg>`;
}

function renderTranscript() {
    const d = activeTranscript;
    if (!d) return;
    const area = document.getElementById('transcriptArea');

    // Load saved speaker names from transcript
    if (d.speaker_names) {
        speakerNames = {...d.speaker_names};
    } else {
        speakerNames = {};
    }
    speakerNamesDirty = false;

    // Build speaker map
    const spMap = {};
    (d.speakers || []).forEach((s, i) => spMap[s] = i % SP.length);

    // If speakers not in transcript, derive from segments
    if (!d.speakers || d.speakers.length === 0) {
        const spkSet = new Set();
        (d.segments || []).forEach(s => { if (s.speaker) spkSet.add(s.speaker); });
        d.speakers = [...spkSet].sort();
        d.speakers.forEach((s, i) => spMap[s] = i % SP.length);
    }

    // Compute stats
    const { stats, totalDur } = computeSpeakerStats(d);

    // Get file object for info panel, patched with fresh data from the transcript
    // activeFile FIRST: the name comes from the current file list and is
    // therefore relative to the CURRENTLY open root. d.filename is in the
    // transcript and is relative to the root that applied at transcription
    // time — when browsing in a subfolder it no longer matches
    // (docs/PATHS.md: rel_path is a display value, not a key). The lookup
    // then came up empty and the stats panel stayed missing, even though
    // the transcript text appeared.
    // Last, the entry the file was opened with: if the jump comes from the
    // queue or a search hit, the file isn't in files[] yet (the list loads
    // asynchronously afterward). Without this fallback, the whole header
    // row was missing — no filename, no key data.
    const fileObj = (typeof fileByName === 'function'
                     ? fileByName(activeFile) : null)
                 || files.find(x => x.name === activeFile)
                 || files.find(x => x.name === d.filename)
                 || (activeFileObj && activeFileObj.name === activeFile
                     ? activeFileObj : null)
                 // Last resort: better a name alone than no header at all.
                 || (activeFile || d.filename
                     ? {name: activeFile || d.filename,
                        display_name: String(activeFile || d.filename)
                            .split('/').pop(),
                        status: 'done'}
                     : null);
    const allText = d.segments ? d.segments.map(s => s.text || '').join(' ') : '';
    const patchedFileObj = fileObj ? {
        ...fileObj,
        tx_model:          d.model          || fileObj.tx_model,
        tx_lang:           d.language       || fileObj.tx_lang,
        tx_vocal_iso:      d.vocal_isolation != null ? d.vocal_isolation : fileObj.tx_vocal_iso,
        tx_vocal_model:    d.vocal_model    ?? fileObj.tx_vocal_model,
        tx_app_version:    d.app_version    || fileObj.tx_app_version,
        tx_chunks:         d.total_chunks   != null ? d.total_chunks   : fileObj.tx_chunks,
        tx_seg_count:      d.segments       ? d.segments.length        : fileObj.tx_seg_count,
        tx_speaker_count:  d.speakers       ? d.speakers.length        : fileObj.tx_speaker_count,
        tx_warning_count:  d.warning_count  ?? fileObj.tx_warning_count,
        tx_word_count:     allText ? allText.split(/\s+/).filter(Boolean).length : fileObj.tx_word_count,
        tx_char_count:     allText ? allText.length                    : fileObj.tx_char_count,
        processing_time:   d.processing_time || fileObj.processing_time,
        processing_secs:   d.processing_secs ?? fileObj.processing_secs,
        tx_step_timings:   d.step_timings    ?? fileObj.tx_step_timings,
        audio_duration:    d.total_duration  || fileObj.audio_duration,
        audio_duration_sec: d.total_duration_sec ?? fileObj.audio_duration_sec,
    } : null;
    // (Its chips moved into the Details tab — recording.js: detailsHtml)
    // activeFile first: `d.filename` is the rel_path from the DB and
    // refers to the root last resolved against — after switching into a
    // subfolder, `config.media_dir` moves along with it, but the
    // rel_path doesn't (docs/PATHS.md). All endpoints here resolve the
    // name against `config.media_dir`; only activeFile does.
    const fn = escJs(activeFile||d.filename);

    const hasWordScores = (d.segments || []).some(x => x.words?.length);
    // The recording: header, tabs, and one pane per tab (recording.js).
    // The transcript and its translation share the line list below.
    const spHtml = `
        <div class="speaker-stats visible" id="speakerStats">
            <div class="speaker-stats-chart">
                ${buildPieChart(d.speakers || [], stats, spMap)}
            </div>
            <div class="speaker-stats-list">
                <div class="sp-filter-head">
                    <span id="spFilterHint">All speakers</span>
                    <button class="sp-only" id="spAllBtn" onclick="filterSpeaker(null, null)"
                            style="display:none">✕ Clear filter</button>
                </div>
                ${(d.speakers || []).map((s, i) => {
                    const st = stats[s] || {dur:0, pct:0, count:0};
                    const timeFmt = st.dur < 60 ? `${st.dur.toFixed(0)}s` : `${(st.dur/60).toFixed(1)}min`;
                    const colors = SPEAKER_COLORS;
                    const ci = (spMap[s] ?? i) % colors.length;
                    return `
                    <div class="sp-stat-row" data-sp="${esc(s)}">
                        <div class="dot" style="background:${colors[ci]}"></div>
                        <input class="sp-stat-name" value="${esc(getSpeakerName(s))}"
                               data-speaker-id="${esc(s)}"
                               onchange="onSpeakerRename(this)"
                               title="Click to rename – original: ${esc(s)}">
                        <div class="sp-stat-bar"><div class="sp-stat-bar-fill" style="width:${st.pct}%;background:${colors[ci]}"></div></div>
                        <div class="sp-stat-nums">${st.pct.toFixed(1)}% · ${timeFmt} · ${plural(st.count, 'segment')}</div>
                        <!-- Own button instead of clicking the whole row: the
                             name next to it is an input field for renaming —
                             whoever clicks there wants to type, not filter. -->
                        <button class="sp-only" onclick="filterSpeaker('${escJs(s)}', this)"
                                title="Show only this person's segments">this only</button>
                        <!-- And the same voice on the voice map. The button above
                             on the panel shows the whole RECORDING; this is about
                             exactly this one identity within it. -->
                        ${s && s !== '—' && s !== '-' ? `<button class="sp-only sp-map-btn" data-label="${esc(s)}"
                                onclick="mapForSpeaker('${escJs(activeFile || '')}', '${escJs(s)}')"
                                title="Highlight this voice on the voice map">🗣 Map</button>` : ''}
                    </div>`;
                }).join('')}
                <div class="sp-stat-actions">
                    <button class="btn-save-names" id="btnSaveNames" onclick="saveSpeakerNames()">
                        💾 Save names
                    </button>
                    <span class="save-status" id="saveNameStatus"></span>
                </div>
            </div>
            <!-- Bridge to the voice map: the FILE is the shared key of
                 both views — here is the text, there the voice. The button
                 filters the map to exactly this recording and jumps to the
                 first of its voices. -->
            <button class="tx-act-btn sp-map-btn" onclick="mapForFile('${escJs(activeFile || '')}')"
                    title="Show this recording on the voice map"><span class="tb-ico">🗣</span>Map</button>
            <!-- Recognition across file boundaries (speaker.js). SPEAKER_00
                 is only a cut marker WITHIN this file; who it actually is is
                 determined by comparing voice vectors against named people. -->
            <div class="sp-recognition" id="speakerRecognition"></div>
        </div>
    `;
    // One chip per person (labels that share a name filter together), the
    // ones who speak most first; with many, the rest fold behind "N more"
    const spPeople = (() => {
        const by = new Map();
        for (const x of (d.speakers || []).filter(x => x && x !== '—' && x !== '-')) {
            const n = getSpeakerName(x) || x;
            if (!by.has(n)) by.set(n, {label: x, name: n, sec: 0});
        }
        for (const sg of (d.segments || [])) {
            const g = by.get(getSpeakerName(sg.speaker) || sg.speaker);
            if (g) g.sec += Math.max(0, (Number(sg.end) || 0) - (Number(sg.start) || 0));
        }
        return [...by.values()].sort((a, b) => b.sec - a.sec);
    })();
    const SP_FEW = 10;
    const spChips = spPeople.length > 1
        ? `<div class="rv-spchips" id="rvSpChips"><button class="fl-opt on" data-sp="" onclick="filterSpeaker(null, null);Rec.spChip(this)">Everyone</button>`
          + spPeople.map((g, i) =>
              `<button class="fl-opt${i >= SP_FEW && spPeople.length > SP_FEW + 1 ? ' rv-spchip-x' : ''}" data-sp="${esc(g.label)}" onclick="filterSpeaker('${escJs(g.label)}', this);Rec.spChip(this)">${esc(g.name)}</button>`).join('')
          + (spPeople.length > SP_FEW + 1 ? `<button class="fl-opt rv-sp-more" onclick="this.parentElement.classList.toggle('all');this.textContent=this.parentElement.classList.contains('all')?'fewer ▴':'${spPeople.length - SP_FEW} more ▾'">${spPeople.length - SP_FEW} more ▾</button>` : '')
          + `</div>` : '';
    let html = Rec.headerHtml(patchedFileObj, d, fn) + Rec.tabsHtml() + `
        <div class="rv-pane" data-pane="text">
        <div class="tx-toolbar rv-tools" id="rvTxTools">
            <div class="tx-toolbar-row">
                <div class="tx-toolbar-actions">
                    <div class="tx-btn-group">
                        <div class="follow-toggle ${followPlayback ? 'active' : ''}" onclick="toggleFollow()" title="The transcript follows playback">
                            <span class="toggle-dot"></span>Follow
                        </div>
                        <button class="tx-act-btn" id="btnWordScore" onclick="toggleWordScore()"
                                ${hasWordScores ? '' : 'disabled'}
                                title="${hasWordScores ? 'Colour each word by how sure the model was'
                                                       : 'No word timestamps in this transcript — transcribe it again with “Per-word timestamps” to get them'}"><span class="tb-ico">🎯</span>Word confidence</button>
                    </div>
                    <div class="tx-divider"></div>
                    <div class="tx-btn-group" title="Change the transcript">
                        <button class="tx-act-btn" onclick="openCorrectionModal('${fn}')" title="Correct the transcript with the language model (guided by the word confidence, with review)"><span class="tb-ico" style="color:#8b5cf6">🪄</span>Correct</button>
                        <button class="tx-act-btn" onclick="retranscribe('${fn}')" title="Transcribe this file again (the dialog lets you change the settings)"><span class="tb-ico" style="color:var(--orange)">🔄</span>Transcribe again</button>
                        <button class="tx-act-btn" onclick="openFindReplace()" title="Find &amp; replace in the transcript"><span class="tb-ico" style="color:#3b82f6">✏️</span>Find &amp; replace</button>
                    </div>
                </div>
                <div class="tx-toolbar-search">
                    <input type="text" id="txSearch" placeholder="Find in the transcript …" oninput="filterSegments()">
                    <!-- The filter hides every segment without the words —
                         said here, or a long transcript looks like a few
                         lines and seems to contain only snippets -->
                    <span class="tx-filter-note" id="txFilterNote" hidden></span>
                </div>
            </div>
            ${spChips}
        </div>
        <div class="rv-tools rv-tr-tools" id="rvTrTools" hidden></div>
        <!-- The old view switcher's place: counts now sit on the tabs -->
        <div class="tx-viewbar" id="txViewBar" hidden></div>
        <div id="segList">
    `;


    (d.segments || []).forEach((s, idx) => {
        const si = spMap[s.speaker] ?? 0;
        // Without diarization every segment carries "—": no name line,
        // and nothing to look up on the voice map.
        const hasSpeaker = !!s.speaker && s.speaker !== '—' && s.speaker !== '-';
        // Force Number(): /api/transcript returns unvalidated JSON from
        // disk (the scanner finds .transcripts/*.json in every folder you
        // open — even on a network share). A "start" like
        // "0);alert(document.cookie)//" would otherwise land straight in
        // the onclick attributes below and get executed. searchmap.js:493
        // has long had the fix for exactly this case, it was missing here.
        // As a side effect this also fixes parseFloat(el.dataset.start) in
        // seekTo/filterSegments, which produced NaN for non-numeric values.
        const start = Number(s.start) || 0;
        const end = Number(s.end) || 0;
        html += `
            <div class="seg ${BL[si]}" data-idx="${idx}" data-speaker="${esc(s.speaker)}"
                 data-text="${esc((s.text||'').toLowerCase())}" data-start="${start}" data-end="${end}"
                 onclick="seekTo(${start})">
                <div class="ts">
                    ${esc(s.start_fmt || '')} → ${esc(s.end_fmt || '')}
                </div>
                <div>
                    ${hasSpeaker ? `<div class="spk ${SP[si]}">${esc(getSpeakerName(s.speaker))}</div>` : ''}
                    <div class="txt" id="seg-txt-${idx}">${buildWordHtml(s)}</div>
                </div>
                <!-- OWN COLUMN, no longer in the timestamp. In the
                     timestamp the edit and speaker-map buttons could not
                     be clicked — and that wasn't
                     because of the buttons: the time column is 110 px
                     wide and doesn't wrap, four buttons don't fit in
                     there. They overflowed, and the text field next to it
                     sits LATER in the tree and therefore lay on top. The
                     clicks went to the text, i.e. to the row's seekTo.
                     They were still visible though — the most annoying
                     case.

                     NO backticks in this comment: it sits inside a
                     template string, and a backtick character ends it.
                     Exactly that made the file view fail completely on
                     the first attempt.

                     In its own grid column nothing can overlap anymore,
                     no matter how many buttons get added. -->
                <span class="seg-clip-btns">
                    <button onclick="event.stopPropagation();setClipPoint('in',${start})" title="Set clip start here">⟨</button>
                    <button onclick="event.stopPropagation();setClipPoint('out',${end})" title="Set clip end here">⟩</button>
                    <button onclick="event.stopPropagation();editSegment(${idx})" title="Edit this segment" style="color:#3b82f6">✎</button>
                    <button onclick="event.stopPropagation();Rec.copyLink(${start})" title="Copy a link to this moment">🔗</button>
                    <!-- The reverse direction of "⤴ Position in recording" in
                         the map: from here to exactly the window this
                         sentence sits in. The speaker MUST be included —
                         SPEAKER_06 is only a cut marker within this
                         recording, and at the same second there are
                         windows for several identities. -->
                    ${hasSpeaker ? `<button onclick="event.stopPropagation();mapForSpot('${escJs(activeFile || '')}',${start},'${escJs(s.speaker || '')}')"
                            title="Show this position on the voice map" style="color:#22c55e">🗺</button>` : ''}
                </span>
            </div>`;
    });
    // Nothing said: the view stayed blank and read as still loading
    if (!(d.segments || []).some(s => String(s.text || '').trim())) {
        html += `<div class="llm-empty">No speech found — the transcript is empty (silence, music
            or a tone?). If someone does speak, re-transcribe it with a larger model.</div>`;
    }

    html += '</div></div>';
    // The other tabs' panes (recording.js shows one at a time)
    html += `<div class="rv-pane" data-pane="summary" hidden><div id="sumView" style="display:none"></div></div>
        <div class="rv-pane" data-pane="translation" hidden><div id="rvTrEmpty"></div></div>
        <div class="rv-pane" data-pane="speakers" hidden>${spHtml}</div>
        <div class="rv-pane" data-pane="clips" hidden></div>
        <div class="rv-pane" data-pane="statistics" hidden><div class="stv-body rv-stats" id="rvStats"></div></div>
        <div class="rv-pane" data-pane="details" hidden>${Rec.detailsHtml(patchedFileObj, d)}</div>
        <div class="rv-pane" data-pane="analyses" hidden><div id="rvAnalyses"></div></div>`;
    // The clip editor may sit in the old Clips pane: park it before the
    // panes are rebuilt, or it would be thrown away with them
    if (typeof Rec !== 'undefined') Rec.parkClip();
    area.innerHTML = html;
    // The transcript's language on its text: the right glyphs for Japanese
    // (not Chinese variants), hyphenation and screen readers. Direction is
    // per paragraph (CSS unicode-bidi: plaintext on the text).
    const lang = String(d.language || '').toLowerCase();
    if (/^[a-z]{2,3}$/.test(lang)) area.lang = lang; else area.removeAttribute('lang');

    // Initialize the view switcher (loads translation tracks + summary
    // count asynchronously afterward and renders the tab bar)
    if (typeof txViewsInit === 'function') txViewsInit(d.filename || activeFile);
    if (typeof Rec !== 'undefined') { Rec.detailsFill(patchedFileObj); Rec.rendered(); }
}


// ── Subtitle Overlay ──────────────────────────────────────────────────────
// The switch lives IN the player (Plyr bar, player.js) instead of in
// the action row. State, size and confidence coloring are
// persistent (localStorage) and survive file and page changes.
function _subOn()        { try { return localStorage.getItem('wtmSubOn') === '1'; } catch (e) { return false; } }
function _subSize()   { try { return Math.min(400, Math.max(60, parseInt(localStorage.getItem('wtmSubSize') || '100', 10) || 100)); } catch (e) { return 100; } }
function _subConfidence() { try { return localStorage.getItem('wtmSubConfidence') === '1'; } catch (e) { return false; } }

function applySubtitleState() {
    const overlay = document.getElementById('subtitleOverlay');
    if (!overlay) return;
    const on = _subOn();
    overlay.style.display = on ? 'block' : 'none';
    // Size relative to the base 0.88rem (100% = previous appearance)
    overlay.style.fontSize = (0.88 * _subSize() / 100).toFixed(3) + 'rem';
    if (!on) overlay.textContent = '';
    // Mirror the player button (only exists after Plyr is ready)
    document.getElementById('wtmSubBtn')?.classList.toggle('wtm-sub-active', on);
}

function toggleSubtitles() {
    try { localStorage.setItem('wtmSubOn', _subOn() ? '0' : '1'); } catch (e) {}
    applySubtitleState();
}

// Subtitle HTML of a segment — from the DATA, not from the transcript
// DOM: only this way can word confidence be colored independently of the
// transcript view's 🎯 switch (orange <0.85, red <0.6 — same as there).
function _subtitleHtml(seg) {
    if (!seg) return '';
    // The translation view has no word scores → never color it
    // (the text there comes from llmSubtitleOverride anyway).
    const translated = (typeof _txView !== 'undefined'
                        && _txView?.mode === 'translation');
    if (translated || !_subConfidence() || !seg.words || !seg.words.length) {
        return esc(seg.text || '');
    }
    return seg.words.map((w, i) => {
        const sc = w.score ?? 1;
        const cls = sc >= 0.85 ? '' : (sc >= 0.6 ? 'ws-mid' : 'ws-low');
        const raw = w.word || '';
        const leading = raw.startsWith(' ') ? ' ' : '';
        const display = esc(raw.trim());
        return leading + (cls ? `<span class="${cls}">${display}</span>` : display);
    }).join('');
}

// ── Word-Level Confidence ──
function buildWordHtml(s) {
    if (!s.words || !s.words.length || !wordScoreVisible) return esc(s.text || '');
    return s.words.map((w, i) => {
        const sc = w.score ?? 1;
        const cls = sc >= 0.85 ? '' : (sc >= 0.6 ? 'ws-mid' : 'ws-low');
        const raw = w.word || '';
        // Space: use embedded leading space if present, otherwise add before every word except first
        const leading = raw.startsWith(' ') ? ' ' : (i > 0 ? ' ' : '');
        const display = esc(raw.trim());
        return leading + (cls
            ? `<span class="${cls}" title="${sc.toFixed(2)}">${display}</span>`
            : display);
    }).join('');
}

function toggleWordScore() {
    const hasWords = activeTranscript?.segments?.some(s => s.words?.length);
    if (!hasWords) {
        const btn = document.getElementById('btnWordScore');
        if (btn) {
            const orig = btn.textContent;
            btn.textContent = '—';
            btn.style.opacity = '0.5';
            // Show inline hint next to button
            let hint = document.getElementById('wordScoreHint');
            if (!hint) {
                hint = document.createElement('span');
                hint.id = 'wordScoreHint';
                hint.style.cssText = 'font-size:0.65rem;color:var(--tx-3);white-space:nowrap;margin-left:2px';
                btn.insertAdjacentElement('afterend', hint);
            }
            hint.textContent = 'Word timestamps missing';
            setTimeout(() => {
                btn.textContent = orig;
                btn.style.opacity = '';
                if (hint) hint.textContent = '';
            }, 2500);
        }
        return;
    }
    wordScoreVisible = !wordScoreVisible;
    document.getElementById('btnWordScore')?.classList.toggle('active', wordScoreVisible);
    // Word confidence exists ONLY in the original — the translation has no
    // alignment scores. So when enabling it in the translation view, jump
    // back to the original instead of showing an empty coloring.
    if (wordScoreVisible && typeof _txView !== 'undefined'
            && _txView?.mode === 'translation') {
        if (typeof Toast !== 'undefined')
            Toast.success('Word confidence only exists in the original — back to the transcript');
        switchTxView('original');
        return;
    }
    if (typeof _txView !== 'undefined' && _txView?.mode === 'summary') return;
    document.querySelectorAll('.seg').forEach((el, idx) => {
        const seg = activeTranscript.segments?.[idx];
        if (seg) el.querySelector('.txt').innerHTML = buildWordHtml(seg);
    });
}

function onSpeakerRename(input) {
    speakerNameFromInput(input);
    speakerNamesChanged();
}

/* One rename field → speakerNames (state only, no DOM beyond the field). */
function speakerNameFromInput(input) {
    const id = input.dataset.speakerId;
    const newName = input.value.trim();
    if (!newName) {
        input.value = id;
        delete speakerNames[id];
    } else if (newName !== id) {
        speakerNames[id] = newName;
    } else {
        delete speakerNames[id];
    }
}

/* After one or more renames: mark Save dirty and redraw the names in the
 * pills and segment labels ONCE — a 24 h recording has tens of thousands
 * of segment labels, so "Apply names" must not do this per speaker. */
function speakerNamesChanged() {
    speakerNamesDirty = true;
    // Null guards: the panel elements only exist once the speaker panel has
    // been rendered (they can be missing on a partial re-render).
    document.getElementById('btnSaveNames')?.classList.add('dirty');
    const _snStatus = document.getElementById('saveNameStatus');
    if (_snStatus) _snStatus.textContent = '';

    // Update filter buttons and segment labels live
    document.querySelectorAll('.tx-sp-pill').forEach(btn => {
        const dot = btn.querySelector('.dot');
        if (!dot) return;
        // Find speaker ID from the onclick
        const match = btn.getAttribute('onclick')?.match(/filterSpeaker\('(.+?)'/);
        if (match) {
            const spId = match[1];
            const pctSpan = btn.querySelector('.sp-pct');
            const pctTxt = pctSpan ? pctSpan.outerHTML : '';
            btn.innerHTML = dot.outerHTML + esc(getSpeakerName(spId)) + pctTxt;
        }
    });
    document.querySelectorAll('.seg .spk').forEach(el => {
        const row = el.closest('.seg');
        if (!row) return;
        const spId = row.dataset.speaker;
        // Keep CSS class but update text
        const cls = el.className;
        el.textContent = getSpeakerName(spId);
        el.className = cls;
    });
}

async function saveSpeakerNames() {
    const btn = document.getElementById('btnSaveNames');
    const status = document.getElementById('saveNameStatus');
    if (!btn) return;
    const setStatus = (t) => { if (status) status.textContent = t; };
    btn.textContent = '💾 Saving…';
    try {
        const r = await fetch('/api/save_speaker_names', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                // activeFile first: the server resolves against the current
                // root (see renderTranscript)
                filename: activeFile || (activeTranscript && activeTranscript.filename),
                speaker_names: speakerNames
            })
        });
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        const data = await r.json();
        if (data.ok) {
            speakerNamesDirty = false;
            btn.classList.remove('dirty');
            btn.textContent = '💾 Save names';
            setStatus('✓ Saved');
            // Update local transcript
            if (activeTranscript) activeTranscript.speaker_names = {...speakerNames};
            setTimeout(() => setStatus(''), 3000);
        } else {
            // #218: reset button text on ok:false too, otherwise it hangs
            // permanently on "Saving…".
            btn.textContent = '💾 Save names';
            setStatus('✗ ' + (data.error || 'Error'));
        }
    } catch(e) {
        setStatus('✗ ' + e.message);
        btn.textContent = '💾 Save names';
    }
}

// ── Player seeking ──
function seekTo(seconds, scrollToSeg=true) {
    const v = document.getElementById('videoPlayer');
    if (!v) return;
    v.currentTime = Math.max(0, seconds - 1);
    const _p = v.play();
    if (_p && _p.catch) _p.catch(() => {});  // autoplay policy → no console rejection

    // Highlight closest segment and scroll into view
    document.querySelectorAll('.seg.playing').forEach(e => e.classList.remove('playing'));
    let bestSeg = null, bestDist = Infinity;
    document.querySelectorAll('.seg').forEach(el => {
        if (el.style.display === 'none') return;
        const d = Math.abs(parseFloat(el.dataset.start) - seconds);
        if (d < bestDist) { bestDist = d; bestSeg = el; }
    });
    if (bestSeg && bestDist < 2) {
        bestSeg.classList.add('playing');
        if (scrollToSeg) bestSeg.scrollIntoView({behavior:'smooth', block:'center'});
    }
}

// Auto-highlight current segment during playback.
//
// The tick used to walk EVERY .seg twice a second (querySelectorAll,
// parseFloat, classList.toggle) — on a 24 h recording that is ~25k segments,
// tens of thousands of DOM operations per tick, visible as jank while
// playing. Now the segment starts are read once per rendered list and the
// current one is found by binary search; only the previous and the new
// element are touched.
let _playSegs = null;        // {box, count, els, starts, sorted} of the rendered #segList
let _playActiveEl = null;    // the element marked 'playing' by the tick
let _playScrolledEl = null;  // Follow mode: the segment last scrolled to

function _playSegIndex() {
    const box = document.getElementById('segList');
    if (!box) return null;
    const c = _playSegs;
    // Rebuilt when the list is another one, grew (live preview) or was
    // replaced in place (translation view: same box, new children).
    if (!c || c.box !== box || c.count !== box.childElementCount
            || (c.els.length && !c.els[0].isConnected)) {
        const els = Array.from(box.querySelectorAll('.seg'));
        const starts = els.map(el => parseFloat(el.dataset.start));
        let sorted = true;
        for (let i = 1; i < starts.length && sorted; i++) {
            if (!(starts[i] >= starts[i - 1])) sorted = false;
        }
        _playSegs = {box, count: box.childElementCount, els, starts, sorted};
    }
    return _playSegs;
}

// The segment playing at time t: start <= t < start of the next one —
// the same rule as before, without touching the DOM.
function _playSegAt(c, t) {
    const {starts} = c;
    const n = starts.length;
    const live = i => t >= starts[i] && t < (i + 1 < n ? starts[i + 1] : Infinity);
    if (c.sorted) {
        let lo = 0, hi = n - 1, found = -1;
        while (lo <= hi) {
            const mid = (lo + hi) >> 1;
            if (starts[mid] <= t) { found = mid; lo = mid + 1; } else hi = mid - 1;
        }
        return found >= 0 && live(found) ? found : -1;
    }
    for (let i = n - 1; i >= 0; i--) if (live(i)) return i;   // the last match, as before
    return -1;
}

setInterval(() => {
    if (!activeTranscript) return;
    const v = document.getElementById('videoPlayer');
    if (!v || v.paused) return;  // null guard: player may not be in the DOM (yet)
    const t = v.currentTime;
    const c = _playSegIndex();
    const i = c ? _playSegAt(c, t) : -1;
    const activeSeg = i >= 0 ? c.els[i] : null;
    if (activeSeg !== _playActiveEl || (activeSeg && !activeSeg.classList.contains('playing'))) {
        // seekTo() marks a segment of its own — clear every mark, not only ours
        (c ? c.box : document).querySelectorAll('.seg.playing')
            .forEach(el => { if (el !== activeSeg) el.classList.remove('playing'); });
        if (activeSeg) activeSeg.classList.add('playing');
        _playActiveEl = activeSeg;
    }
    // Follow mode: scroll to the active segment — once per segment
    if (followPlayback && activeSeg && activeSeg !== _playScrolledEl) {
        activeSeg.scrollIntoView({behavior:'smooth', block:'center'});
        _playScrolledEl = activeSeg;
    }
    // Subtitle overlay — from the segment DATA (data-idx), so the
    // confidence coloring works independently of the transcript view.
    const overlay = document.getElementById('subtitleOverlay');
    if (overlay && overlay.style.display !== 'none') {
        if (activeSeg) {
            const idx = parseInt(activeSeg.dataset.idx, 10);
            const seg = activeTranscript?.segments?.[idx];
            // Active LLM translation wins (llmtools.js) — timings
            // remain those of the original, only the text is replaced.
            const trans = (typeof llmSubtitleOverride === 'function')
                ? llmSubtitleOverride(idx) : null;
            if (trans !== null) {
                overlay.textContent = trans;
            } else {
                overlay.innerHTML = seg ? _subtitleHtml(seg)
                    : esc(activeSeg.querySelector('.txt')?.textContent || '');
            }
        } else {
            overlay.textContent = '';
        }
    }
}, 500);

function toggleFollow() {
    followPlayback = !followPlayback;
    // Re-render toggle state in DOM
    document.querySelectorAll('.follow-toggle').forEach(el => {
        el.classList.toggle('active', followPlayback);
    });
}

// ── Resize handle: drag to resize player/transcript ──
document.addEventListener('DOMContentLoaded', () => {
    const handle = document.getElementById('resizeHandle');
    const player = document.getElementById('playerBar');
    const main = document.querySelector('.main');
    let dragging = false;
    let startY = 0;
    let startH = 0;

    handle.addEventListener('mousedown', (e) => {
        dragging = true;
        startY = e.clientY;
        startH = player.offsetHeight;
        document.body.style.cursor = 'ns-resize';
        document.body.style.userSelect = 'none';
        e.preventDefault();
    });

    document.addEventListener('mousemove', (e) => {
        if (!dragging) return;
        const delta = e.clientY - startY;
        const newH = Math.max(80, Math.min(main.offsetHeight - 150, startH + delta));
        player.style.height = newH + 'px';
    });

    document.addEventListener('mouseup', () => {
        if (!dragging) return;
        dragging = false;
        document.body.style.cursor = '';
        document.body.style.userSelect = '';
    });
});

// ── Filters ──
/* Reduce the transcript to one person.
 *
 * The function already existed — just nobody called it anymore. It comes
 * from an older UI with `.tx-sp-pill` buttons above the transcript;
 * those disappeared in a rework, but the function and its CSS remained,
 * leaving no way to filter by speaker.
 *
 * A second click on the same person clears the filter again — otherwise
 * you'd have to hunt for the way back while the list is already filtered.
 */
function filterSpeaker(speaker, btn) {
    activeSpeaker = (speaker && speaker === activeSpeaker) ? null : speaker;
    document.querySelectorAll('.tx-sp-pill').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.sp-stat-row').forEach(r =>
        r.classList.toggle('filtered', !!activeSpeaker && r.dataset.sp === activeSpeaker));
    document.querySelectorAll('.sp-only').forEach(b => b.classList.remove('active'));
    if (btn && activeSpeaker) btn.classList.add('active');
    filterSegments();
    txShowSpFilter();
}

// Make it visible that filtering is active — and how much falls away
// because of it. A silent filter is the same mistake as the invisible
// file pinning in the big search.
function txShowSpFilter() {
    const hint = document.getElementById('spFilterHint');
    const all_items = document.getElementById('spAllBtn');
    if (!hint || !all_items) return;
    if (!activeSpeaker) {
        hint.textContent = 'All speakers';
        all_items.style.display = 'none';
        return;
    }
    const segs = document.querySelectorAll('.seg');
    const visible = [...segs].filter(e => e.style.display !== 'none').length;
    hint.innerHTML = `only <b>${esc(getSpeakerName(activeSpeaker))}</b>`
        + ` — ${visible} of ${plural(segs.length, 'segment')}`;
    all_items.style.display = '';
}

function filterSegments() {
    const q = (document.getElementById('txSearch')?.value || '').toLowerCase();
    let shown = 0, all = 0;
    document.querySelectorAll('.seg').forEach(el => {
        // A person, not a label: voices that carry the same name count together
        const ms = !activeSpeaker || el.dataset.speaker === activeSpeaker
            || (typeof getSpeakerName === 'function' && getSpeakerName(el.dataset.speaker) === getSpeakerName(activeSpeaker));
        const mt = !q || el.dataset.text.includes(q);
        el.style.display = (ms && mt) ? '' : 'none';
        all++;
        if (ms && mt) shown++;
    });
    const note = document.getElementById('txFilterNote');
    if (note) {
        note.hidden = !q.trim();
        if (q.trim()) {
            note.innerHTML = `${shown.toLocaleString()} of ${plural(all, 'segment')}
                <button type="button" onclick="txClearFilter()" title="Show the whole transcript again">✕ show all</button>`;
        }
    }
    if (typeof txShowSpFilter === 'function') txShowSpFilter();
}

function txClearFilter() {
    const input = document.getElementById('txSearch');
    if (input) input.value = '';
    filterSegments();
}


// ── Export menu in the transcript toolbar ────────────────────────────────

/* The open file was shown as "being processed" and the queue has moved on
 * (finished, failed, stopped): show what it is now. The live job stream
 * normally does this, but a short run can finish before it attaches —
 * the view then kept the placeholder forever. */
async function refreshRunningPlaceholder() {
    if (!document.getElementById('txRunningPlaceholder') || !activeFileObj) return;
    const f = activeFileObj;
    try {
        const r = await fetch('/api/scan_entry/' + encodeURIComponent(f.name));
        const d = await r.json();
        if (d.file && activeFileObj === f && !isRunningFile(d.file)) selectFile(d.file);
    } catch (e) { /* the next click shows it anyway */ }
}
