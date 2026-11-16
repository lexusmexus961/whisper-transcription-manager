// ── Init ──
document.addEventListener('DOMContentLoaded', async () => {
    // Fetch queue state BEFORE the directory scan and attach to the stream:
    // the scan can take minutes with a large library on a busy network
    // share, and until then a freshly loaded tab would show "Queue 0" even
    // though the server was already working. Nothing is STARTED here: the
    // server worker processes the queue regardless.
    await refreshQueue();
    renderQueueList();
    connectQueueSSE();

    // Also BEFORE the scan: sets the dialog defaults and _txDefaults
    // (queue display) — does not depend on the file list.
    await _applySettingsDefaults();

    // ── Dialog wiring BEFORE the scan ──────────────────────────────
    //
    // If these listeners were placed AFTER `await scanFiles()`, every
    // checkbox in the transcription dialog would be a dead control while
    // the scan runs (which can take minutes, see above): Vocal Isolation
    // would not expand the separation model, the speaker count would stay
    // fixed, the warning for voice vectors would not appear — all without
    // reporting an error. The wiring doesn't need the scan — it only needs
    // the elements, and those exist since txoptions.js renders them.
// One behavior for all three dialogs (the blocks come from txoptions.js
// with consistently normalized IDs — the former per-modal special cases are gone).
TX_OPTION_PREFIXES.forEach(prefix => {
    const vocalIso = document.getElementById(prefix + 'VocalIso');
    const diarize = document.getElementById(prefix + 'Diarize');
    const speaker = document.getElementById(prefix + 'SpeakerVectors');
    if (vocalIso) vocalIso.addEventListener('change', function() {
        const o = document.getElementById(prefix + 'VocalIsoOpts');
        if (o) o.style.display = this.checked ? 'block' : 'none';
        syncSaveAudioVisibility(prefix);
        syncSpeakerVectors(prefix);
    });
    // Speaker count belongs to diarization — without it the field is meaningless
    if (diarize) diarize.addEventListener('change', function() {
        const f = document.getElementById(prefix + 'SpeakersField');
        if (f) f.style.display = this.checked ? '' : 'none';
        syncSpeakerVectors(prefix);
    });
    if (speaker) speaker.addEventListener('change',
                                            () => syncSpeakerVectors(prefix));
});

    // Sync all checkbox-dependent UI (handles browser autofill/restore)
    window.syncCheckboxUI = function() {
        for (const p of TX_OPTION_PREFIXES) {
            const cb = document.getElementById(p + 'VocalIso');
            const tgt = document.getElementById(p + 'VocalIsoOpts');
            if (cb && tgt) tgt.style.display = cb.checked ? 'block' : 'none';
            const dia = document.getElementById(p + 'Diarize');
            const spk = document.getElementById(p + 'SpeakersField');
            if (dia && spk) spk.style.display = dia.checked ? '' : 'none';
            if (typeof _txoModelChanged === 'function') _txoModelChanged(p);
            syncSaveAudioVisibility(p);
            syncSpeakerVectors(p);
        }
    };
    // Run once on load for browser-restored checkbox states
    syncCheckboxUI();

    // Wire up the search view controls BEFORE the scan.
    //
    // Behind `await scanFiles()` (which can take minutes, see above) the
    // full search could be opened (`openSearchView()` searches on its own)
    // while its controls weren't wired yet: Enter in the search field,
    // the speaker filter and the value filters would do nothing.
    //
    // They only attach to elements fixed in the HTML and need nothing from
    // the scan — up here they cost nothing and take effect immediately.
    setupGlobalSearch();
    setupSearchView();
    setupSearchMap();

    await scanFiles();
    // Fetch the resolved upload directory from the server (Settings →
    // Folders) and pin it into favorites
    try {
        const dirs = await fetch('/api/settings/directories').then(r => r.json());
        if (dirs.uploads?.path) uploadsDir = dirs.uploads.path;
    } catch(e) { /* favorite/jump then fall back to root+uploads */ }
    if (typeof ensureUploadsFav === 'function') ensureUploadsFav();
    updateFavBtn();
    preloadDoneSound();
    // Restore sound mute state
    try {
        if (localStorage.getItem('wtm_soundMuted')) {
            _soundMuted = true;
            const btn = document.getElementById('btnSoundToggle');
            btn.classList.add('muted');
            btn.title = 'Enable done sound';
        }
    } catch(e) {}

    renderFileList();

    // A link from outside — the browser extension's "open": the recording
    // (?open=<full path>&t=<seconds>) or an analysis (?analysis=<id>)
    _openFromLink();

    // Enter/input/focus of the path field are managed by the path picker
    // (pathpicker.js) via inline handlers (onkeydown/oninput/onfocus) — do
    // NOT bind them again here (otherwise Enter would trigger changeDir twice).
    updateFavBtn();

    // Escape to clear selection
    document.addEventListener('keydown', e => {
        if (e.key === 'Escape' && selectedFiles.size > 0) {
            clearSelection();
        }
        // Ctrl+A to select all filtered files (when not in input).
        // The set comes from the server — the client list is only a window into it.
        if ((e.ctrlKey || e.metaKey) && e.key === 'a' &&
            !['INPUT','TEXTAREA','SELECT'].includes(document.activeElement.tagName)) {
            e.preventDefault();
            fetchFilteredNames().then(d => {
                d.names.forEach((n, i) => {
                    selectedFiles.add(n);
                    if (d.status) selectionStatus.set(n, d.status[i]);
                });
                updateSelectionUI();
                syncSelectedClasses();   // only sync classes — no list rebuild
            }).catch(() => {});
        }
    });

    // Vocal Isolation toggle (single file modal)

// Show/hide the save-audio field: there is only processed audio to save
// when vocal isolation runs
function syncSaveAudioVisibility(prefix) {
    const vocalIso = document.getElementById(prefix + 'VocalIso')?.checked || false;
    const field = document.getElementById(prefix + 'SaveAudioField');
    if (field) field.style.display = vocalIso ? '' : 'none';
}

/* Voice vectors depend on two other options — and that needs to be VISIBLE.
 *
 * Without diarization there are no speaker segments to cut from: the
 * toggle would then be pointless. Instead of letting it silently do
 * nothing, it is disabled and states why.
 *
 * Without Vocal Isolation the vectors would be the RAW version, and that
 * must not happen — vectors with and without isolation would not match
 * each other. So the 30-second windows are then separated
 * individually; that delivers practically the same vector (cosine
 * similarity around 0.98-0.99) but costs time, and that too belongs in the dialog rather
 * than in a log.
 */
function syncSpeakerVectors(prefix) {
    const box  = document.getElementById(prefix + 'SpeakerVectors');
    const warn = document.getElementById(prefix + 'SpeakerVectorsWarn');
    if (!box || !warn) return;
    // In the queue editor (queue_editor.js) each group stands on its own:
    // "Set voice vectors" may apply to entries whose speaker toggle stays
    // unchanged. There the mode bar controls disabled — not the dependency
    // on the (meaningless) qeDiarize checkbox.
    if (typeof box.closest === 'function' && box.closest('.qe-group')) return;
    const diarize = document.getElementById(prefix + 'Diarize')?.checked || false;
    const vocal   = document.getElementById(prefix + 'VocalIso')?.checked || false;

    box.disabled = !diarize;
    box.parentElement.style.opacity = diarize ? '' : '0.45';

    if (!diarize) {
        warn.style.display = 'block';
        warn.innerHTML = '⚠ Requires <b>Detect speakers</b> — without '
            + 'speaker segments there is nothing to cut.';
    } else if (box.checked && !vocal) {
        warn.style.display = 'block';
        warn.innerHTML = 'Without Vocal Isolation, the vectors will have the '
            + '<b>30-second windows separated individually</b> — otherwise '
            + 'they would be the raw version and not comparable with the '
            + 'remainder of the library. The transcription remains untouched; it '
            + 'costs about one second per window.';
    } else {
        warn.style.display = 'none';
    }
}

    // ── Sidebar resize ──
    const sidebar = document.querySelector('.sidebar');
    const sidebarHandle = document.getElementById('sidebarResize');
    let sbDragging = false, sbStartX = 0, sbStartW = 0;

    if (sidebarHandle && sidebar) sidebarHandle.addEventListener('mousedown', (e) => {
        sbDragging = true;
        sbStartX = e.clientX;
        sbStartW = sidebar.offsetWidth;
        sidebarHandle.classList.add('dragging');
        document.body.style.cursor = 'col-resize';
        document.body.style.userSelect = 'none';
        e.preventDefault();
    });
    document.addEventListener('mousemove', (e) => {
        if (!sbDragging) return;
        const delta = e.clientX - sbStartX;
        const newW = Math.max(200, Math.min(800, sbStartW + delta));
        sidebar.style.width = newW + 'px';
    });
    document.addEventListener('mouseup', () => {
        if (!sbDragging) return;
        sbDragging = false;
        sidebarHandle.classList.remove('dragging');
        document.body.style.cursor = '';
        document.body.style.userSelect = '';
    });

    // ── EventBus: Key events ──
    // Emit events for other components to listen to
    EventBus.on('connection:online', () => {
        // Reload after a connection drop — but the user did nothing, so
        // their position in the list stays put.
        scanFiles({background: true});
    });

    // ── FileWatcher integration ──
    // Poll watcher status periodically; show toast when new files detected
    let _lastWatcherFiles = 0;
    async function _checkWatcher() {
        try {
            const resp = await fetch('/api/watcher/status');
            if (resp.ok) {
                const data = await resp.json();
                if (data.running && data.tracked_files !== _lastWatcherFiles) {
                    if (_lastWatcherFiles > 0 && data.tracked_files > _lastWatcherFiles) {
                        const diff = data.tracked_files - _lastWatcherFiles;
                        Toast.info(`${plural(diff, 'new file')} detected`);
                        EventBus.emit('watcher:files_changed', { count: data.tracked_files });
                    }
                    _lastWatcherFiles = data.tracked_files;
                }
            }
        } catch(e) { /* watcher not enabled or offline */ }
    }
    // Check every 60s (the actual watcher polls more frequently server-side)
    setInterval(_checkWatcher, 60000);
    _checkWatcher(); // Initial check

    // ── Health check (connection status + the rail's Machine dot) ──
    const _healthPoll = async () => {
        try {
            const resp = await fetch('/api/health/detail', { cache: 'no-cache' });
            if (resp.ok && !ConnectionStatus.isOnline) ConnectionStatus.setOnline();
            if (resp.ok && typeof Shell !== 'undefined') Shell.health(await resp.json());
        } catch(e) {
            ConnectionStatus.setOffline();
            if (typeof Shell !== 'undefined') Shell.health(null);
        }
    };
    setInterval(_healthPoll, 30000);
    _healthPoll();
});

// ── Populate vocal-isolation model dropdowns dynamically + annotated ──
// One source: /api/vocal_isolation_status returns the annotated catalog
// (group/label/description/quality/speed) from core/vocal_isolation.py.
// ALL four dropdowns are built from it — no more hardcoded <option>.
const VOCAL_SELECT_IDS = TX_OPTION_PREFIXES.map(p => p + 'VocalModel');

function _qualityMeter(n) {
    return n ? '●'.repeat(n) + '○'.repeat(5 - n) : '';
}

// `withDescription`: the detailed model description (SDR values, speed)
// now lives only in the separation dialog, where there's room for it. In
// the three options dialogs (single/add/batch) it made the menu
// unnecessarily long — there it remains as a tooltip on the <option>.
function _fillVocalSelect(el, catalog, groupOrder, defaultModel,
                          withDescription = true) {
    if (!el) return;
    // Group by group, group order from group_order (remainder at the end)
    const byGroup = {};
    for (const m of catalog) (byGroup[m.group] = byGroup[m.group] || []).push(m);
    const groups = [...groupOrder.filter(g => byGroup[g]),
                    ...Object.keys(byGroup).filter(g => !groupOrder.includes(g))];

    el.innerHTML = '';
    for (const g of groups) {
        const og = document.createElement('optgroup');
        og.label = g;
        for (const m of byGroup[g]) {
            const opt = document.createElement('option');
            opt.value = m.name;
            opt.textContent = m.label;
            if (m.desc) opt.title = m.desc;   // tooltip (browser-dependent)
            og.appendChild(opt);
        }
        el.appendChild(og);
    }
    if (defaultModel && catalog.some(m => m.name === defaultModel)) {
        el.value = defaultModel;
    }

    if (!withDescription) return;

    // Live description directly below the select — reliably visible,
    // regardless of whether the browser shows <option title> as a tooltip.
    let desc = el.nextElementSibling;
    if (!desc || !desc.classList.contains('vocal-model-desc')) {
        desc = document.createElement('div');
        desc.className = 'hint vocal-model-desc';
        desc.style.marginTop = '4px';
        el.insertAdjacentElement('afterend', desc);
    }
    const updateDesc = () => {
        const m = catalog.find(x => x.name === el.value);
        if (!m) { desc.innerHTML = ''; return; }
        const meter = [
            m.quality ? `Quality ${_qualityMeter(m.quality)}` : '',
            m.speed ? `Speed ${_qualityMeter(m.speed)}` : '',
        ].filter(Boolean).join('&nbsp;&nbsp;·&nbsp;&nbsp;');
        desc.innerHTML = (m.desc ? esc(m.desc) : '')
            + (meter ? `<div style="margin-top:2px;color:var(--tx-3)">${meter}</div>` : '');
    };
    el.addEventListener('change', updateDesc);
    updateDesc();
}

// ── Apply defaults from server settings to UI dropdowns ──
async function _applySettingsDefaults() {
    try {
        const r = await fetch('/api/vocal_isolation_status');
        if (!r.ok) return;
        const data = await r.json();
        const catalog = data.catalog || [];
        const groupOrder = data.group_order || [];

        if (catalog.length) {
            for (const id of VOCAL_SELECT_IDS) {
                // With description: since the removal of the separation
                // dialog, the transcription dialog is the only place where
                // a separation model is chosen — the quality/speed info
                // belongs where the decision is made.
                _fillVocalSelect(document.getElementById(id), catalog,
                                 groupOrder, data.default_model);
            }
        } else if (data.default_model) {
            // Fallback without a catalog: just set the default, if options exist
            for (const id of VOCAL_SELECT_IDS) {
                const el = document.getElementById(id);
                if (el && Array.from(el.options).some(o => o.value === data.default_model))
                    el.value = data.default_model;
            }
        }
    } catch(e) {
        // No problem — dropdowns stay empty, server uses its default
    }

    // Apply transcription defaults from the config to ALL three dialogs
    // (single/upload/batch). Previously "turbo"/"checked" was hardcoded in
    // the HTML — the settings under Transcription were thus ineffective
    // dummies for the dialogs.
    try {
        const r = await fetch('/api/settings');
        if (!r.ok) return;
        const answer = await r.json();
        const cfg = answer.config || {};
        const t = cfg.transcription || {};
        const vi = cfg.vocal_isolation || {};

        // Remember the effective defaults for the queue display (entries
        // without their own settings run with exactly these values).
        _txDefaults = { model: t.default_model || '',
                        language: t.default_language || '' };

        // Make the version visible (settings footer + logo tooltip +
        // console) — in case of an error you know exactly what was running.
        const v = answer.version;
        if (v && v.display) {
            const el = document.getElementById('appVersion');
            if (el) el.textContent = `Whisper Transcription Manager ${v.display}`;
            const logo = document.querySelector('.topbar .logo');
            if (logo) logo.title = `Whisper Transcription Manager ${v.display}`;
            console.log(`Whisper Transcription Manager ${v.display}`);
        }

        // Remains for the one toggle that does NOT belong to the
        // transcription options (ytAudioOnly further below). The dialog
        // options run through txPreset().
        const setCheck = (id, an) => {
            const el = document.getElementById(id);
            if (el && typeof an === 'boolean' && el.checked !== an) {
                el.checked = an;
                // fire change so dependent areas (VI options, save field)
                // show/hide
                el.dispatchEvent(new Event('change'));
            }
        };
        // Apply ALL configurable dialog defaults. Without pre-filling, the
        // toggles on the settings page are dummies: saving works fine, but
        // the dialog overwrites the value with the HTML default every time
        // it opens.
        //
        // WHICH options these are is, since the rework, no longer here but
        // in TX_OPTIONS (txoptions.js) — along with their IDs and where
        // they are collected. A new option therefore no longer needs an
        // entry at this spot.
        for (const prefix of TX_OPTION_PREFIXES) {
            txPreset(prefix, cfg);
            // Vocabulary presets (transcription.vocab_presets) into the
            // dialog dropdown — selection stays with the user (default:
            // none). This is a dynamic LIST, not pre-filling, and therefore
            // stays outside the table.
            const vpSel = document.getElementById(prefix + 'VocabPreset');
            if (vpSel) {
                const old_value = vpSel.value;
                vpSel.innerHTML = '<option value="">— no vocabulary —</option>'
                    + Object.keys(t.vocab_presets || {}).map(n =>
                        `<option value="${esc(n)}">${esc(n)}</option>`).join('');
                // What was picked stays; a fresh dialog starts with the
                // default vocabulary (transcription.default_vocabulary)
                const want = old_value || t.default_vocabulary || '';
                if (want && Array.from(vpSel.options).some(o => o.value === want))
                    vpSel.value = want;
            }
        }
        // "Audio only" in link mode: pre-fill the toggle from the setting
        // (the dialog may override it per download).
        setCheck('ytAudioOnly', !!(cfg.youtube || {}).audio_only);
    } catch(e) {
        // Dialogs then keep the HTML values; the server normalizes anyway
    }
}


/* ?open=<full path>&t=<seconds> opens that recording at that moment,
 * ?analysis=<id> that analysis, ?chat=<id> that conversation — the links
 * the browser extension gives (core/ingest.py). Taken out of the address
 * afterwards, so a reload doesn't jump again. */
function _openFromLink() {
    let q;
    try { q = new URLSearchParams(location.search); } catch (e) { return; }
    const abs = q.get('open') || '';
    const analysis = q.get('analysis') || '';
    const chat = q.get('chat') || '';
    if (!abs && !analysis && !chat) return;
    try { history.replaceState(null, '', location.pathname); } catch (e) { /* the jump still happens */ }
    if (analysis && typeof openAnalysis === 'function') {
        openAnalysis(analysis);
        return;
    }
    // A conversation — the browser extension's "go on asking in Whisper Transcription Manager"
    if (chat && typeof openAsk === 'function' && typeof chatLoad === 'function') {
        openAsk();
        chatLoad(chat);
        return;
    }
    const cut = abs.lastIndexOf('/');
    if (cut > 0 && typeof goToSearchResult === 'function') {
        const folder = abs.slice(0, cut);
        const name = abs.slice(cut + 1);
        // Opening switches the file list's folder — on the server, for
        // everyone using it. Any page can send a browser here with such a
        // link: inside the open folder it opens, elsewhere it asks first.
        const here = (document.getElementById('dirInput')?.value || '').replace(/\/+$/, '');
        const inside = !!here && (folder === here || folder.startsWith(here + '/'));
        if (!inside && !confirm(`Open "${name}"?\n\nA link asks for it. The file list switches to ${folder} — for everyone using this server.`)) {
            return;
        }
        goToSearchResult(name, Number(q.get('t')) || 0, folder);
    }
}
