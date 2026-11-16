// ══════════════════════════════════════════════════════════════════════════
//  Processing options — ONE template for all dialogs
//
//  renderTxOptions(prefix, opts) builds the options block into
//  `#${prefix}TxOptions` with normalized IDs (`${prefix}Model`,
//  `${prefix}VocalIsoOpts`, …). Used by:
//    - #modalBg   (transcribe single file)   → mode "normal"
//    - #batchModal (batch)                   → mode "normal"
//    - #uploadModal (bring in file/link)     → mode "upload"
//
//  The upload mode ADDITIONALLY has the transcription entry "none":
//  then it's only stored — or only separated, if Vocal Isolation is on.
//  This lets ONE options block replace the former separate
//  dialog modes ("& transcribe" / "& Voice Separation").
//
//  The fields are ordered in two groups: TRANSCRIPTION (fully hidden with
//  "none") and AUDIO PREPARATION (always applies).
// ══════════════════════════════════════════════════════════════════════════

const TX_OPTION_PREFIXES = ['opt', 'up', 'batch', 'qe', 'qa'];   // 'qe' = queue editor, 'qa' = its "Add folder" tab

// ══════════════════════════════════════════════════════════════════════════
//  THE TABLE — one row per option, instead of nine scattered spots
//
//  Previously each option name appeared verbatim in at least nine places: in
//  the markup here, in the preset (app.js), in THREE collector functions
//  (transcribe.js, batch.js, upload.js), in a FOURTH rewrite
//  (upload.js `_uploadQueueSettings`), in the endpoint's whitelist
//  (api/queue.py `_SETTING_KEYS`), in `_normalize()` and in the
//  worker signature.
//
//  Three of these omissions were SILENT and have already cost time:
//    • missing the key in `_SETTING_KEYS` silently drops it on the server;
//    • missing the preset makes the settings page a dummy —
//      the dialog overwrites the value every time it opens;
//    • missing `_uploadQueueSettings` makes the option work everywhere
//      EXCEPT the upload/link path.
//
//  Fields:
//    key   — the value's name in the queue JSON and on the server
//    field     — element ID suffix (`${prefix}${field}`)
//    kind   — 'toggle' (checkbox) or 'value' (select/input)
//    before   — function(cfg) → preset value; if missing, nothing is preset
//    empty  — fallback value if the element is missing (only for 'value')
//    auto  — 'auto' becomes null when collecting (speaker count)
//    setting — the config key `before` reads; "Make these my defaults"
//              writes the dialog's value back there (label: what the
//              toast calls it, toSetting: dialog value → config value)
// ══════════════════════════════════════════════════════════════════════════

const TX_OPTIONS = [
    {key: 'language',    field_el: 'Lang',  kind: 'value',
     before: c => (c.transcription || {}).default_language,
     setting: 'transcription.default_language', label: 'language'},
    {key: 'model',       field_el: 'Model', kind: 'value',
     before: c => (c.transcription || {}).default_model,
     // "none" (upload dialog: store or separate only) is no model to default to
     setting: 'transcription.default_model', label: 'model',
     toSetting: v => (v && v !== 'none' ? v : undefined)},
    // Speaker count: 0 in the config means "automatic" — then do NOT
    // preset it, otherwise the dialog would show a made-up number.
    {key: 'numSpeakers', field_el: 'Speakers', kind: 'value', auto: true,
     before: c => ((c.transcription || {}).default_num_speakers > 0
                ? String(c.transcription.default_num_speakers) : undefined),
     setting: 'transcription.default_num_speakers', label: 'number of speakers',
     toSetting: v => (v === 'auto' || v === '' || v == null ? 0 : Number(v))},
    {key: 'diarize',     field_el: 'Diarize', kind: 'toggle',
     before: c => !!(c.transcription || {}).default_diarize,
     setting: 'transcription.default_diarize', label: 'speaker detection'},
    {key: 'word_timestamps',  field_el: 'WordTimestamps', kind: 'toggle',
     before: c => !!(c.transcription || {}).word_timestamps,
     setting: 'transcription.word_timestamps', label: 'word timestamps'},
    {key: 'export_alongside', field_el: 'ExportAlongside', kind: 'toggle',
     before: c => !!(c.transcription || {}).export_alongside,
     setting: 'transcription.export_alongside', label: 'SRT alongside'},
    // The list itself is filled later (app.js), so the preset is applied
    // there too; the row only knows where the default lives
    {key: 'vocab_preset', field_el: 'VocabPreset', kind: 'value', empty: '',
     setting: 'transcription.default_vocabulary', label: 'vocabulary'},
    {key: 'vocal_isolation', field_el: 'VocalIso', kind: 'toggle',
     before: c => (c.vocal_isolation || {}).default_enabled !== false,
     setting: 'vocal_isolation.default_enabled', label: 'vocal isolation'},
    {key: 'vocal_model',  field_el: 'VocalModel', kind: 'value'},
    {key: 'save_audio_alongside', field_el: 'SaveAudioAlongside', kind: 'toggle',
     before: c => !!(c.vocal_isolation || {}).default_save_alongside,
     setting: 'vocal_isolation.default_save_alongside', label: 'saving the separated audio'},
    // Voice vectors: applies ONLY with diarization (without speaker segments
    // there's nothing to cut) and only writes separated versions.
    {key: 'speaker_vectors', field_el: 'SpeakerVectors', kind: 'toggle',
     before: c => (c.transcription || {}).speaker_vectors !== false,
     setting: 'transcription.speaker_vectors', label: 'voice vectors'},
    {key: 'low_vram',     field_el: 'LowVram', kind: 'toggle',
     before: c => !!(c.hardware || {}).low_vram,
     setting: 'hardware.low_vram', label: 'spare VRAM'},
];


/* Collect all options of a dialog.
 *
 * Replaces the three near-identical collectors. If an element is missing
 * (the upload dialog doesn't know every field), the value is simply left out
 * instead of becoming `undefined` — the server then falls back to its config
 * default, and that's exactly how `_SETTING_KEYS`/`_normalize` are built.
 */
function txCollect(p) {
    const out = {};
    for (const o of TX_OPTIONS) {
        const el = document.getElementById(p + o.field_el);
        if (o.kind === 'toggle') {
            if (el) out[o.key] = el.checked;
        } else {
            if (!el) {
                if (o.empty !== undefined) out[o.key] = o.empty;
                continue;
            }
            // 'auto' is not a number. The single-file dialog used to send
            // the string, the batch dialog null — same intent, two forms.
            out[o.key] = (o.auto && el.value === 'auto') ? null : el.value;
        }
    }
    return {...out, ..._sumSettings(p)};
}


/* Preset the dialogs from the config.
 *
 * Without this step, the switches on the settings page are dummies:
 * saving works fine, but the dialog overwrites the value every time it
 * opens with the HTML default. That was exactly the case before the
 * settings audit.
 */
function txPreset(p, cfg) {
    for (const o of TX_OPTIONS) {
        if (!o.before) continue;
        const value = o.before(cfg);
        if (value === undefined) continue;
        const el = document.getElementById(p + o.field_el);
        if (!el) continue;
        if (o.kind === 'toggle') {
            if (typeof value === 'boolean' && el.checked !== value) {
                el.checked = value;
                // fire change so dependent areas open/close
                el.dispatchEvent(new Event('change'));
            }
        } else if (value && Array.from(el.options || []).some(x => x.value === value)) {
            el.value = value;
        }
    }
}

/* ── "Make these my defaults" ─────────────────────────────────────────────
 *
 * The dialog's choices become the start values of every dialog from now
 * on: the same settings as under ⚙ Settings (Transcription, Vocal
 * isolation, GPU & queue, AI tasks), saved at once. Only what differs from
 * the saved values is sent, and the toast names it — a click that changes
 * nothing says so instead of pretending.
 */
async function txSaveDefaults(p, btn) {
    let cur;
    try {
        cur = (await fetch('/api/settings').then(r => r.json())).config || {};
    } catch (e) {
        Toast.error('Settings not readable: ' + e.message);
        return;
    }
    const saved = (path) => path.split('.').reduce(
        (node, k) => (node && typeof node === 'object' ? node[k] : undefined), cur);
    const updates = {};
    const changed = [];
    const put = (path, value, label) => {
        if (JSON.stringify(saved(path) ?? null) === JSON.stringify(value)) return;
        const keys = path.split('.');
        let node = updates;
        for (const k of keys.slice(0, -1)) node = node[k] = node[k] || {};
        node[keys[keys.length - 1]] = value;
        changed.push(label);
    };
    for (const o of TX_OPTIONS) {
        if (!o.setting) continue;
        const el = document.getElementById(p + o.field_el);
        if (!el) continue;
        let v = o.kind === 'toggle' ? el.checked : el.value;
        if (o.toSetting) v = o.toSetting(v);
        if (v !== undefined) put(o.setting, v, o.label);
    }
    const sum = document.getElementById(p + 'Summarize');
    if (sum) {
        put('llm.auto_summarize', sum.checked, 'summary afterwards');
        if (sum.checked) {
            put('llm.auto_summarize_preset', document.getElementById(p + 'SumPreset')?.value || '', 'summary template');
            put('llm.auto_summarize_model', document.getElementById(p + 'SumModel')?.value || '', 'summary model');
        }
    }
    if (!changed.length) {
        Toast.info('These already are your defaults');
        return;
    }
    if (btn) btn.disabled = true;
    try {
        const r = await fetch('/api/settings?save=true', {
            method: 'PUT', headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(updates)});
        const d = await r.json().catch(() => ({}));
        const errors = d.errors || [];
        if (!r.ok || d.error || errors.length) throw new Error(d.error || errors.join(' · ') || `HTTP ${r.status}`);
        Toast.success(`New defaults for every dialog: ${changed.join(', ')}`);
        // The other dialogs start from them too, from now on
        if (typeof _applySettingsDefaults === 'function') _applySettingsDefaults();
    } catch (e) {
        Toast.error('Defaults not saved: ' + e.message);
    } finally {
        if (btn) btn.disabled = false;
    }
}

/* ── Split into cards ─────────────────────────────────────────────────────
 *
 * Previously the panel was ONE long column in a 420-pixel-wide dialog:
 * fifteen options stacked on top of each other that you had to scroll
 * through, and the order followed how they were added rather than what
 * they belonged to — voice vectors, for example, sat far below the speech
 * recognition options they belong with.
 *
 * Now: the model at the top spanning the full width (it decides whether
 * the remainder even applies), below it cards in a grid that adapts to the
 * width. Related things stand together — above all the voice vectors,
 * which are a property of SPEAKER recognition and used to be under "Audio
 * preparation", three screens away from the checkbox without which they
 * don't work at all.
 *
 * The element IDs remain unchanged. They are the contract with
 * `TX_OPTIONS`, the collector functions and the server's whitelist —
 * `tests/test_options_collapsed.py` locks this down.
 */
const _TXO_MAP = (title, content, attr = '') =>
    `<section class="txo-map"${attr}>
        <div class="txo-map-head">${title}</div>
        ${content}
    </section>`;

const _TXO_TICKS = (id, color, text, hint, extra = '') =>
    `<div class="field">
        <label class="txo-check">
            <input type="checkbox" id="${id}" style="accent-color:${color}">
            ${text}
        </label>
        ${hint ? `<div class="hint">${hint}</div>` : ''}
        ${extra}
    </div>`;

function renderTxOptions(p, opts = {}) {
    const mount = document.getElementById(p + 'TxOptions');
    if (!mount) return;
    const upload = !!opts.upload;

    // ── Language & vocabulary ──────────────────────────────────────────────
    const language = `
        <div class="field">
            <label>Language</label>
            <select id="${p}Lang">
                <option value="auto">Detect automatically</option>
                <optgroup label="Common">
                    <option value="en">English</option>
                    <option value="de">German</option>
                    <option value="fr">French</option>
                    <option value="es">Spanish</option>
                    <option value="it">Italian</option>
                    <option value="pt">Portuguese</option>
                    <option value="nl">Dutch</option>
                    <option value="pl">Polish</option>
                    <option value="ru">Russian</option>
                    <option value="tr">Turkish</option>
                </optgroup>
                <optgroup label="More">
                    <option value="zh">Chinese</option>
                    <option value="ja">Japanese</option>
                    <option value="ko">Korean</option>
                    <option value="sv">Swedish</option>
                    <option value="da">Danish</option>
                    <option value="no">Norwegian</option>
                    <option value="fi">Finnish</option>
                    <option value="cs">Czech</option>
                    <option value="uk">Ukrainian</option>
                    <option value="ar">Arabic</option>
                    <option value="hi">Hindi</option>
                </optgroup>
            </select>
        </div>
        <div class="field">
            <label>📖 Vocabulary</label>
            <select id="${p}VocabPreset">
                <option value="">— no vocabulary —</option>
                <!-- Presets from transcription.vocab_presets (app.js) -->
            </select>
            <div class="hint">Proper nouns and technical terms that Whisper
                otherwise mangles. Manage under Settings&nbsp;→&nbsp;🎙&nbsp;Transcription.</div>
        </div>`;

    // ── Speakers: diarization, count AND voice vectors ────────────────
    //
    // The three belong together and are therefore placed together. The
    // voice vectors used to be under "Audio preparation" — far away from
    // the checkbox without which they can't work at all.
    const speaker = `
        <div class="field">
            <label class="txo-check">
                <input type="checkbox" id="${p}Diarize" checked
                       style="accent-color:var(--accent)">
                Detect speakers (diarization)
            </label>
            <div class="hint">Who speaks when. Without it, transcription
                runs noticeably faster.</div>
            <div class="field" id="${p}SpeakersField" style="margin:8px 0 0">
                <select id="${p}Speakers">
                    <option value="auto">Speaker count: automatic</option>
                    <option value="1">1 speaker</option><option value="2">2 speakers</option>
                    <option value="3">3 speakers</option><option value="4">4 speakers</option>
                    <option value="5">5 speakers</option><option value="6">6 speakers</option>
                    <option value="7">7 speakers</option><option value="8">8 speakers</option>
                </select>
                <div class="hint">An exact number improves the assignment.</div>
            </div>
        </div>
        <div class="field txo-detached">
            <label class="txo-check">
                <input type="checkbox" id="${p}SpeakerVectors"
                       style="accent-color:#8b5cf6">
                🗣 Record voice vectors
            </label>
            <div class="hint" id="${p}SpeakerVectorsHint">Remembers HOW the
                speakers sound — so it can later be answered who appeared in a
                recording and roughly where. Costs ~1&nbsp;s per
                30-second window.</div>
            <!-- Two conditions, and both are made VISIBLE in the dialog
                 instead of quietly taking effect (app.js: syncSpeakerVectors):
                 · without diarization there are no speaker segments to
                   cut — then the switch has no effect;
                 · without Vocal Isolation the vectors would be the RAW version.
                   Raw and separated are not comparable; a mixed
                   collection splits into two halves that never find each other.
                   The windows are therefore separated individually instead. -->
            <div id="${p}SpeakerVectorsWarn" class="txo-warning"
                 style="display:none"></div>
        </div>`;

    // ── Subtitles and files alongside the original ────────────────────────
    const subtitles =
        _TXO_TICKS(`${p}WordTimestamps`, '#3b82f6', '📝 Per-word timestamps',
            'For precise SRT subtitles; costs an extra '
            + 'alignment step.')
        + _TXO_TICKS(`${p}ExportAlongside`, '#22c55e', '📺 SRT alongside the video file',
            'For media servers (Plex/Jellyfin).');

    // ── Audio preparation ───────────────────────────────────────────────
    const audio = `
        <div class="field">
            <label class="txo-check">
                <input type="checkbox" id="${p}VocalIso" style="accent-color:#e67e22">
                🎤 Vocal Isolation — separate speech from music and noise
            </label>
            <div class="hint">Worthwhile for game audio, music, loud backgrounds.
                Clean speech recordings are detected by the pre-filter and it
                skips separation on its own — "on" costs almost nothing there.</div>
            <div id="${p}VocalIsoOpts" class="txo-subfield" style="display:none">
                <label>Separation model</label>
                <select id="${p}VocalModel" style="margin-top:4px">
                    <!-- filled dynamically from /api/vocal_isolation_status (app.js) -->
                </select>
            </div>
        </div>
        <div class="field" id="${p}SaveAudioField" style="display:none">
            <label class="txo-check">
                <input type="checkbox" id="${p}SaveAudioAlongside"
                       style="accent-color:#27ae60">
                💾 Save processed audio as MP3 alongside the source file
            </label>
        </div>`;

    // ── Performance ─────────────────────────────────────────────────────────
    const performance = _TXO_TICKS(`${p}LowVram`, '#06b6d4',
        '🧠 Spare VRAM — unload models after each step',
        'For runs that would otherwise fail due to GPU memory (large model + '
        + 'separation + alignment). Also clears the resident models from the '
        + 'previous stage upfront; somewhat slower.');

    mount.innerHTML = `
        <div class="txo-model">
            <label>Model</label>
            <select id="${p}Model" onchange="_txoModelChanged('${p}')">
                <option value="none">— no transcription —</option>
                <optgroup label="Recommended">
                    <option value="turbo">turbo — fast &amp; very good (~6GB VRAM)</option>
                    <option value="large-v3">large-v3 — maximum quality (~10GB VRAM)</option>
                    <option value="large-v2">large-v2 — proven (~10GB VRAM)</option>
                </optgroup>
                <optgroup label="Smaller / faster">
                    <option value="medium">medium (~5GB VRAM)</option>
                    <option value="small">small (~2GB VRAM)</option>
                    <option value="base">base (~1GB VRAM)</option>
                    <option value="tiny">tiny (~1GB VRAM)</option>
                </optgroup>
                <optgroup label="Distilled (faster, similar quality)">
                    <option value="distil-large-v3">distil-large-v3</option>
                    <option value="distil-large-v2">distil-large-v2</option>
                </optgroup>
                <optgroup label="English-only">
                    <option value="medium.en">medium.en</option>
                    <option value="small.en">small.en</option>
                    <option value="base.en">base.en</option>
                    <option value="tiny.en">tiny.en</option>
                    <option value="distil-medium.en">distil-medium.en</option>
                    <option value="distil-small.en">distil-small.en</option>
                </optgroup>
            </select>
            ${upload ? `<div class="hint">"none" only stores the file in the library
                (transcribing can be done later anytime via the list) — if
                Vocal Isolation is active, only the audio track is separated instead.</div>`
                     : `<div class="hint">"none" only separates the audio track: vocals and
                instrumental end up as MP3 alongside the source file. Everything that
                requires a transcript is then hidden.</div>`}
        </div>

        <div class="txo-raster">
            <!-- display:contents (in the stylesheet): the wrapper bundles everything
                 that requires a transcript, for showing/hiding in
                 _txoModelChanged — without occupying a grid cell itself.
                 Its cards remain independent cells. -->
            <div id="${p}TxOnly">
                ${_TXO_MAP('Language &amp; vocabulary', language)}
                ${_TXO_MAP('👥 Speakers', speaker)}
                ${_TXO_MAP('Subtitles &amp; files', subtitles)}
            </div>
            ${_TXO_MAP('🎚 Audio preparation', audio)}
            ${_TXO_MAP('⚙ Performance', performance)}
            ${_TXO_MAP('🤖 Summary', `
                <div class="field">
                    <label class="txo-check">
                        <input type="checkbox" id="${p}Summarize"
                               style="accent-color:#8b5cf6"
                               onchange="_txoSumToggle('${p}')">
                        Automatically summarize afterwards (LLM)
                    </label>
                    <div class="hint">Runs after transcription on the
                        server — the result is at the file's 🤖 button.</div>
                    <div id="${p}SumOpts" class="txo-subfield" style="display:none">
                        <label>Template</label>
                        <select id="${p}SumPreset" style="margin-top:4px"></select>
                        <label style="display:block;margin-top:8px">Model</label>
                        <select id="${p}SumModel" style="margin-top:4px"></select>
                    </div>
                </div>`, ` id="${p}SumGroup" style="display:none"`)}
        </div>`;
}

// ── Summary options (preset + model) ───────────────────────────
// ONE call to /api/llm/status fills all three dialogs. If no LLM is
// configured, the group disappears entirely: a checkbox that can't do
// anything is worse than no checkbox.
let _txoLlm = null;

async function loadSummaryOptions() {
    try {
        const r = await fetch('/api/llm/status');
        _txoLlm = r.ok ? await r.json() : null;
    } catch (e) {
        _txoLlm = null;
    }
    for (const p of TX_OPTION_PREFIXES) {
        _txoFillSumFields(p);
        _txoSumVisible(p);
        _txoSumPreset(p);
    }
}

// Preset from the settings (llm.auto_summarize*): checkbox + preset +
// model. Runs AFTER filling the selects — same response, no race.
function _txoSumPreset(p) {
    if (!_txoLlm) return;
    const tick_mark = document.getElementById(p + 'Summarize');
    if (tick_mark && !!_txoLlm.auto_summarize !== tick_mark.checked) {
        tick_mark.checked = !!_txoLlm.auto_summarize;
        _txoSumToggle(p);
    }
    const setIt = (id, value) => {
        const el = document.getElementById(id);
        if (el && value && Array.from(el.options).some(o => o.value === value))
            el.value = value;
    };
    setIt(p + 'SumPreset', _txoLlm.auto_summarize_preset);
    setIt(p + 'SumModel', _txoLlm.auto_summarize_model);
}

function _txoFillSumFields(p) {
    const preset = document.getElementById(p + 'SumPreset');
    const model = document.getElementById(p + 'SumModel');
    if (!preset || !model || !_txoLlm) return;

    // Presets are a flat dict {name: system prompt} (config → llm.presets)
    preset.innerHTML = '';
    for (const name of Object.keys(_txoLlm.presets || {})) {
        const o = document.createElement('option');
        o.value = name;
        o.textContent = name;
        preset.appendChild(o);
    }
    if (!preset.options.length) {
        preset.innerHTML = '<option value="">Default template (Settings → AI tasks)</option>';
    }

    // Models: the default first, then the quick selection. The default
    // sends '' so the provider assigned to summaries in the settings
    // (llm.tasks) decides — the server treats a missing model that way.
    // Previously it sent the legacy llm.model by name, and a named model
    // pulls ITS provider along (llm_client.build_chat_payload): automatic
    // summaries never used the summary assignment.
    model.innerHTML = '';
    const std = _txoLlm.task_models?.summary || '';
    const def = document.createElement('option');
    def.value = '';
    def.textContent = 'Default (task setting)' + (std ? ': ' + std : '');
    model.appendChild(def);
    for (const [name, id] of Object.entries(_txoLlm.models || {})) {
        // The task's own model isn't listed a second time — the default runs it
        if (typeof id !== 'string' || id === std) continue;
        const o = document.createElement('option');
        o.value = id;
        o.textContent = name !== id ? `${name} (${id})` : name;
        model.appendChild(o);
    }
}

// Only visible if an LLM is configured AND transcription happens at all —
// without a transcript there is nothing to summarize.
function _txoSumVisible(p) {
    const grp = document.getElementById(p + 'SumGroup');
    if (!grp) return;
    const model = document.getElementById(p + 'Model')?.value;
    grp.style.display = (_txoLlm && _txoLlm.configured && model !== 'none') ? '' : 'none';
}

// The three settings collectors (transcribe.js/batch.js/upload.js) get their
// summary part here — one place, one set of keys. If unchecked, the
// keys aren't included at all (the server fills in missing ones from the config).
function _sumSettings(p) {
    if (!document.getElementById(p + 'Summarize')?.checked) return {};
    return {
        summarize: true,
        summarize_preset: document.getElementById(p + 'SumPreset')?.value || '',
        summarize_model: document.getElementById(p + 'SumModel')?.value || '',
    };
}

function _txoSumToggle(p) {
    const on = document.getElementById(p + 'Summarize')?.checked;
    const opts = document.getElementById(p + 'SumOpts');
    if (opts) opts.style.display = on ? 'block' : 'none';
}

// Transcription "none": hide the purely transcription-related fields
// — audio preparation stays selectable and is then the entire job.
function _txoModelChanged(p) {
    const sel = document.getElementById(p + 'Model');
    const txOnly = document.getElementById(p + 'TxOnly');
    const isNone = sel.value === 'none';
    if (txOnly) txOnly.style.display = isNone ? 'none' : '';
    // Without transcription, separation is the only thing the run can still
    // do — so set the checkbox right away. In the upload dialog it stays
    // deselectable: "none + no separation" there means "just store".
    const iso = document.getElementById(p + 'VocalIso');
    if (isNone && iso && !iso.checked && p !== 'up') {
        iso.checked = true;
        iso.dispatchEvent(new Event('change'));
    }
    _txoSumVisible(p);   // no summary without transcription
    // Dialog-specific reaction (e.g. button label in the upload dialog)
    if (typeof window._txoOnChange === 'function') window._txoOnChange(p);
}

// Render before app.js' DOMContentLoaded (script order in index.html):
// _applySettingsDefaults and the change listeners expect finished elements.
document.addEventListener('DOMContentLoaded', () => {
    renderTxOptions('opt');
    renderTxOptions('batch');
    renderTxOptions('up', {upload: true});
    renderTxOptions('qe');            // queue editor: settings for the selection
    renderTxOptions('qa');            // queue editor, "Add folder": options of the new entries
    loadSummaryOptions();
});
