/* ══════════════════════════════════════════════════════════════════════════
 *  EXTRACT — one person's voice as clips with their sentences
 *
 *  All of one person's voices, or a part filtered by name, date and so
 *  on — as one audio file or single samples in a ZIP, with the transcript
 *  pieces, e.g. for TTS training.
 *
 *  Three columns: what to take (left), what that gives (middle — counts,
 *  years, how sure Whisper was, why sentences were left out, and the clips
 *  themselves, page by page, each one editable), and what to write (right
 *  — a dataset ZIP or one audio file with subtitles, and the exports made
 *  so far). How the voices and sentences are chosen is written at the top
 *  of `core/voice_extract.py`.
 *
 *  **The preview follows the filters.** It used to be computed only on a
 *  button, and a changed filter left the old preview standing — the page
 *  could show "named voices only" over a preview with recognized ones.
 *  Now a change computes it again; until then it is marked as out of date
 *  and can't be exported.
 *
 *  **Hand edits** (voices_clipedit.js) are kept on the server for the
 *  person, by sentence: a new preview has them again.
 *
 *  **Voices without stored speech spans** are no option in the form: they
 *  are rare, and a choice about them would only puzzle. Where a preview
 *  holds such voices, a note says so and offers to leave them out.
 *
 *  **Words** are coloured by how sure the alignment is, and the one being
 *  said is highlighted while a clip plays — where the transcript was made
 *  with word timestamps (core/voice_extract.clip_words).
 *
 *  **From conversations.** The search view's Conversations tab hands its
 *  filters over (✂ Extract there): only the sentences of the conversations
 *  they find — an hour where Alice, Bob and Carol talk in long sentences.
 *  The form shows them in words; they are changed in that tab, which the
 *  section opens.
 *
 *  **Where it goes.** The exports folder, to download from the list — or a
 *  folder of your own on the server, there also unpacked. The folder is checked while it is typed or picked; an export
 *  never replaces anything in it.
 * ══════════════════════════════════════════════════════════════════════════ */

const EXTRACT = {
    ready: null,           // the tab, built once
    defaults: null,
    run: null,             // the last preview
    busy: false,
    stale: false,          // a filter changed since the preview
    seq: 0,
    timer: null,           // status polling while an export runs
    status: null,
    page: [],              // the clips on screen
    offset: 0, total: 0, state: 'all',
    totals: null,          // what an export takes, with the hand edits
    dest: null,            // the checked folder of your own (/extract/dest)
    strict: false,         // leave out voices without speech spans
    conv: null,            // the conversation filters handed over (conversations.js)
};

const EXTRACT_PAGE = 50;

/* "1 clip", "500 clips" */
function extractClips(n) {
    return `${vxN(n)} clip${Number(n) === 1 ? '' : 's'}`;
}

/* Built once; a second call while the first is still loading waits for it
 * (the People page opens the tab and fills in a person right away). */
function extractShow() {
    if (!EXTRACT.ready) {
        EXTRACT.ready = extractBuild().catch(e => {
            EXTRACT.ready = null;
            const page = document.getElementById('vxPageExtract');
            if (page) page.innerHTML = `<div class="vx-note err">Extractor unavailable: ${esc(String(e))}</div>`;
            throw e;
        });
    }
    return EXTRACT.ready.then(() => extractPoll());
}

async function extractBuild() {
    const page = document.getElementById('vxPageExtract');
    if (!page) return;
    page.innerHTML = '<div class="vx-empty">Loading …</div>';
    EXTRACT.defaults = await (await fetch('/api/voices/extract/options')).json();
    page.innerHTML = extractShellHtml();
    extractWireForm();
    extractDestRestore();
    await extractPeople();
}

/* Open the tab for a person — from the People page: all of their voice,
 * not the conversations handed over before. */
async function extractFor(name) {
    vxOpen('extract');
    try { await extractShow(); } catch (e) { return; }
    EXTRACT.conv = null;
    extractDrawConv();
    extractPick(name);
    extractPreview();
}

/* Open the tab with conversation filters — from the search view's
 * Conversations tab: `filters` by the names of GET /api/conversations,
 * `person` the voice to take (the person chosen stays without one). */
async function extractFromConversations(filters, person) {
    vxOpen('extract');
    try { await extractShow(); } catch (e) { return; }
    EXTRACT.conv = filters && typeof filters === 'object' && Object.keys(filters).length ? filters : null;
    extractDrawConv();
    if (person) extractPick(person);
    EXTRACT.stale = !!EXTRACT.run;
    extractStaleMark();
    extractPreview();
}

function extractPick(name) {
    const sel = document.getElementById('xPerson');
    if (!sel || !name) return;
    if (![...sel.options].some(o => o.value === name)) {
        sel.insertAdjacentHTML('beforeend', `<option value="${esc(name)}">${esc(name)}</option>`);
    }
    if (sel.value !== name) { EXTRACT.run = null; EXTRACT.offset = 0; EXTRACT.state = 'all'; }
    sel.value = name;
}

/* The conversations the sentences come from — in words, with a way to
 * change them (in the Conversations tab) or to drop them. */
function extractConvHtml() {
    const f = EXTRACT.conv;
    if (!f) {
        return `<div class="map-small">Every recording of the voice. To take only some conversations — people
            together, about an hour, in long sentences — find them in the search view's Conversations tab and
            hand them over with ✂ Extract.</div>
            <button class="vx-btn" style="margin-top:6px" onclick="extractFindConversations()">🗣 Find conversations</button>`;
    }
    const words = typeof cvDescribe === 'function' ? cvDescribe(f) : [];
    return `<div class="ad-chips vx-x-conv">${(words.length ? words : ['Every conversation'])
            .map(w => `<span class="ad-chip">${esc(w)}</span>`).join('')}</div>
        <div class="vx-row" style="margin-top:6px">
            <button class="vx-btn" onclick="extractFindConversations()" title="Change them in the Conversations tab — ✂ Extract there brings them back">Change …</button>
            <button class="vx-btn" onclick="extractDropConv()" title="Every recording of the voice again">✕ Every recording</button></div>`;
}

function extractDrawConv() {
    const el = document.getElementById('xConv');
    if (el) el.innerHTML = extractConvHtml();
}

function extractDropConv() {
    EXTRACT.conv = null;
    extractDrawConv();
    EXTRACT.stale = !!EXTRACT.run;
    extractStaleMark();
    extractPreview();
}

/* The Conversations tab, with the filters handed over — or with the
 * person chosen here. This workspace lies above the search: it closes
 * first (the preview stays as it is for the way back). */
function extractFindConversations() {
    if (typeof cvShow !== 'function') return;
    const person = document.getElementById('xPerson')?.value || '';
    if (typeof vxClose === 'function') vxClose();
    cvShow(EXTRACT.conv || (person ? {people: [person]} : {}));
}

async function extractPeople() {
    const sel = document.getElementById('xPerson');
    if (!sel) return;
    let people = [];
    try {
        people = ((await (await fetch('/api/voices/people')).json()).people || [])
            .filter(p => p.voices > 0);
    } catch (e) { /* stays empty */ }
    const was = sel.value;
    sel.innerHTML = '<option value="">— choose a person —</option>'
        + people.map(p => `<option value="${esc(p.name)}">${esc(p.name)} · ${vxN(p.voices)} voice${p.voices === 1 ? '' : 's'} · ${vxHours(p.speech_hours)}</option>`).join('');
    if (was) sel.value = was;
}

function extractShellHtml() {
    const o = EXTRACT.defaults.defaults;
    const rates = EXTRACT.defaults.rates.map(r =>
        `<option value="${r}"${r === 22050 ? ' selected' : ''}>${(r / 1000).toFixed(r % 1000 ? 2 : 0)} kHz</option>`).join('');
    const num = (id, v, step, min, max, title = '') =>
        `<input type="number" id="${id}" value="${v}" step="${step}" min="${min}" max="${max}" title="${esc(title)}">`;
    return `<div class="vx-extract"><div class="vx-x-left">
    <div class="vx-x-form" id="xForm">
        <div class="vx-form-section">
            <div class="vx-form-title">Whose voice</div>
            <div class="vx-row"><select id="xPerson"></select></div>
            <div class="vx-radio">
                <label><input type="radio" name="xInclude" value="named"${o.include === 'named' ? ' checked' : ''}>
                    <span><b>Named voices only</b> — what you assigned (and its spread)</span></label>
                <label><input type="radio" name="xInclude" value="recognized"${o.include === 'recognized' ? ' checked' : ''}>
                    <span><b>Also recognized ones</b> — voices that sound like this person and like nobody else named</span></label>
            </div>
            <div class="vx-row"><label for="xThreshold" title="How alike a voice must be to one of the person's named voices">Similarity at least</label>
                <input type="range" id="xThreshold" min="0.6" max="0.95" step="0.01" value="${o.threshold}"
                       oninput="document.getElementById('xThresholdV').textContent=Number(this.value).toFixed(2)">
                <b id="xThresholdV" class="vx-num">${Number(o.threshold).toFixed(2)}</b></div>
            <button class="vx-btn" style="margin-top:8px" onclick="extractWholeRecordings()"
                    title="Export every recording this person speaks in, as it is — the filters below apply">
                <svg class="ui-ico"><use href="#i-download"/></svg>All recordings with this voice</button>
        </div>
        <div class="vx-form-section">
            <div class="vx-form-title">From conversations</div>
            <div id="xConv">${extractConvHtml()}</div>
        </div>
        <div class="vx-form-section">
            <div class="vx-form-title">When and where</div>
            <div class="vx-row"><label>Recorded from</label><input type="date" id="xFrom"></div>
            <div class="vx-row"><label>to</label><input type="date" id="xTo"></div>
            <div class="vx-row"><label title="The path starts with this">Folder</label><input type="text" id="xFolder" placeholder="e.g. interviews/2024"></div>
            <div class="vx-row"><label>File name contains</label><input type="text" id="xFile"></div>
            <div class="vx-row"><label>Language</label><input type="text" id="xLang" placeholder="any, or e.g. de"></div>
        </div>
        <div class="vx-form-section">
            <div class="vx-form-title">Sentences</div>
            <div class="vx-row"><label title="How sure Whisper was of the sentence: its mean token probability. Stored per sentence, not per word.">Whisper sure at least</label>
                ${num('xCert', o.min_certainty, 5, 0, 100, '0–100 %')}<span class="map-small">%</span></div>
            <div class="vx-row"><label>Text contains</label><input type="text" id="xText"></div>
            <div class="vx-row"><label>Text without</label><input type="text" id="xTextNot"></div>
            <div class="vx-row"><label>Length (s)</label>${num('xMin', o.min_sec, 0.5, 0.3, 60)} – ${num('xMax', o.max_sec, 0.5, 0.5, 120)}</div>
            <div class="vx-row"><label title="Characters per second — outside this the sentence's times are wrong">Speaking rate</label>${num('xRateMin', o.rate_min, 1, 0, 100)} – ${num('xRateMax', o.rate_max, 1, 1, 200)}</div>
            <div class="vx-row check"><input type="checkbox" id="xPhantoms"${o.skip_phantoms ? ' checked' : ''}><label for="xPhantoms">Leave out sentences marked as phantom</label></div>
            <div class="vx-row check"><input type="checkbox" id="xArtifacts"${o.skip_artifacts ? ' checked' : ''}><label for="xArtifacts">Leave out Whisper artifacts (Settings → Transcription)</label></div>
            <div class="vx-row"><label title="Neighbouring sentences closer than this become one clip (0 = off)">Join gaps up to (s)</label>${num('xJoin', o.join_gap, 0.1, 0, 2)}</div>
            <div class="vx-row"><label title="Extra audio before and after a sentence — never beyond the voice's speech span">Padding (s)</label>${num('xPad', o.padding, 0.05, 0, 1)}</div>
        </div>
        <div class="vx-form-section">
            <div class="vx-form-title">How much</div>
            <div class="vx-row"><label>Order</label><select id="xOrder">
                <option value="time">in time order</option>
                <option value="sure">Whisper surest first</option>
                <option value="best">best first (similarity, certainty)</option>
                <option value="spread">a little of every recording</option></select></div>
            <div class="vx-row"><label>Per recording at most</label>${num('xPerRec', o.per_recording, 1, 0, 100000, '0 = no limit')}</div>
            <div class="vx-row"><label>Hours at most</label>${num('xHours', o.max_hours, 0.5, 0, 10000, '0 = no limit')}</div>
            <div class="vx-row"><label>Clips at most</label>${num('xClips', o.max_clips, 100, 0, 10000000, '0 = no limit')}</div>
        </div>
    </div>
    <div class="vx-x-out">
        <div class="vx-form-section">
            <div class="vx-form-title">Write as</div>
            <div class="vx-radio">
                <label><input type="radio" name="xMode" value="dataset" checked onchange="extractModeUi()">
                    <span><b>Dataset (ZIP)</b> — one WAV per clip, metadata.csv (LJSpeech: <i>id|text|text</i>) and metadata.jsonl — what Coqui, Piper and most TTS trainers read</span></label>
                <label><input type="radio" name="xMode" value="single" onchange="extractModeUi()">
                    <span><b>One audio file</b> — all clips in a row, with subtitles (SRT) and a source index</span></label>
                <label><input type="radio" name="xMode" value="recordings" onchange="extractModeUi()">
                    <span><b>Whole recordings</b> — every file the person speaks in, as it is, with a list
                    (recordings.csv); the filters decide which</span></label>
            </div>
            <div class="vx-row" id="xFormatRow" hidden><label>Format</label><select id="xFormat" onchange="extractButtons()">
                <option value="flac">FLAC (lossless)</option><option value="wav">WAV</option>
                <option value="mp3">MP3 192 kbit/s</option><option value="opus">Opus 64 kbit/s</option></select></div>
            <div class="vx-row" id="xGapRow" hidden><label>Silence between (s)</label><input type="number" id="xGap" value="0.4" step="0.1" min="0" max="5" onchange="extractButtons()"></div>
            <div class="vx-row"><label>Sample rate</label><select id="xRate" onchange="extractButtons()">${rates}</select></div>
            <div class="vx-row check"><input type="checkbox" id="xNormalize" checked><label for="xNormalize">Even out the loudness</label></div>
            <div class="vx-row check" title="Clips you didn't trim by hand are cut at the quietest moment near their sentence's start and end — where that is a pause. Removes most half words."><input type="checkbox" id="xSnap" checked><label for="xSnap">Cut at the nearest pause (half words)</label></div>
            <div class="vx-row check"><input type="checkbox" id="xOnlyKept" onchange="extractButtons()" disabled><label for="xOnlyKept" id="xOnlyKeptLabel">Only the clips I kept</label></div>
            <div class="vx-row"><label>File name</label><input type="text" id="xName" placeholder="person-date"></div>
            <div class="vx-radio" id="xDestChoice">
                <label><input type="radio" name="xDest" value="exports" checked onchange="extractDestUi()">
                    <span><b>Download</b> — the browser saves it when it is done (also kept in the list below)</span></label>
                <label><input type="radio" name="xDest" value="folder" onchange="extractDestUi()">
                    <span><b>Into a folder of my choice</b> — on the server, e.g. where the TTS trainer reads</span></label>
            </div>
            <div id="xDestBox" class="vx-dest" hidden>
                <div class="vx-dest-field">
                    <input type="text" id="xDestPath" spellcheck="false" placeholder="/mnt/…"
                           oninput="extractDestCheckSoon()" onchange="extractDestCheckSoon(0)">
                    <button type="button" class="vx-btn" onclick="extractBrowse()" title="Choose the folder"><svg class="ui-ico"><use href="#i-folder"/></svg></button>
                </div>
                <div class="vx-browser" id="xBrowser" hidden></div>
                <div class="map-small vx-dest-info" id="xDestInfo"></div>
                <div class="vx-row check" id="xUnpackedRow"><input type="checkbox" id="xUnpacked" checked onchange="extractDestUi()"><label for="xUnpacked">Unpacked — a folder with the WAVs and the metadata, no ZIP</label></div>
            </div>
            <div class="map-small vx-consent">Use a person's voice only with their consent — a voice dataset can train a model that speaks like them.</div>
            <button class="vx-btn primary big" id="xExportBtn" style="width:100%" disabled onclick="extractExport()">Export</button>
            <div id="xJob" style="margin-top:10px"></div>
        </div>
        <div class="vx-form-section">
            <div class="vx-form-title">Exports</div>
            <div class="map-small" style="margin-bottom:6px">In <span id="xFolderOut">${esc(EXTRACT.defaults.folder)}</span></div>
            <div id="xFiles"></div>
        </div>
    </div></div>
    <div class="vx-x-preview" id="xPreview"><div class="vx-empty">Choose a person — the preview shows what the export will contain.</div></div>
    </div>`;
}

/* Every filter computes the preview again — selects, boxes and buttons at
 * once, typed fields when they are left (or Enter), the slider when it is
 * let go. Until the new preview is there, the old one says it is out of
 * date and the export waits for it. */
function extractWireForm() {
    const form = document.getElementById('xForm');
    if (!form) return;
    let wait = null;
    const soon = (ms) => {
        // Enter in a field fires both a key and a change — and a change can
        // leave the values as they were. The same filters, computed again,
        // would only cost time.
        if (extractKey() === EXTRACT.key) return;
        EXTRACT.stale = !!EXTRACT.run;
        extractStaleMark();
        clearTimeout(wait);
        wait = setTimeout(extractPreview, ms);
    };
    form.addEventListener('change', e => {
        if (e.target.id === 'xPerson') { EXTRACT.run = null; EXTRACT.offset = 0; EXTRACT.state = 'all'; }
        soon(e.target.type === 'text' ? 0 : 250);
    });
    form.addEventListener('keydown', e => {
        if (e.key === 'Enter' && e.target.tagName === 'INPUT') { e.preventDefault(); soon(0); }
    });
}

function extractStaleMark() {
    const box = document.getElementById('xPreview');
    if (box) box.classList.toggle('vx-stale', !!EXTRACT.stale);
    extractButtons();
}

function extractStrict(on) {
    EXTRACT.strict = !!on;
    EXTRACT.stale = !!EXTRACT.run;
    extractStaleMark();
    extractPreview();
}

function extractMode() {
    return document.querySelector('input[name="xMode"]:checked')?.value || 'dataset';
}

function extractModeUi() {
    const mode = extractMode();
    document.getElementById('xFormatRow').hidden = mode !== 'single';
    document.getElementById('xGapRow').hidden = mode !== 'single';
    // Whole recordings are copied as they are — nothing to cut or convert
    for (const id of ['xRate', 'xNormalize', 'xSnap']) {
        const row = document.getElementById(id)?.closest('.vx-row');
        if (row) row.hidden = mode === 'recordings';
    }
    const u = document.querySelector('label[for="xUnpacked"]');
    if (u) u.textContent = mode === 'recordings'
        ? 'Unpacked — a folder with the recordings and the list, no ZIP'
        : 'Unpacked — a folder with the WAVs and the metadata, no ZIP';
    extractDestUi();
}

/* Download every file the speaker is in — with all the filters still
 * applying. */
function extractWholeRecordings() {
    const r = document.querySelector('input[name="xMode"][value="recordings"]');
    if (r) { r.checked = true; extractModeUi(); }
    document.getElementById('xExportBtn')?.scrollIntoView({block: 'center', behavior: 'smooth'});
}

function extractOptions() {
    const v = id => (document.getElementById(id) || {}).value;
    const n = id => Number(v(id));
    const radio = name => document.querySelector(`input[name="${name}"]:checked`)?.value;
    return {
        include: radio('xInclude'), threshold: n('xThreshold'),
        clean: EXTRACT.strict ? 'strict' : 'best',
        date_from: v('xFrom') || '', date_to: v('xTo') || '',
        folder: (v('xFolder') || '').trim(), file_text: (v('xFile') || '').trim(),
        language: (v('xLang') || '').trim(),
        text: (v('xText') || '').trim(), text_not: (v('xTextNot') || '').trim(),
        min_sec: n('xMin'), max_sec: n('xMax'), rate_min: n('xRateMin'), rate_max: n('xRateMax'),
        min_certainty: n('xCert'),
        skip_phantoms: document.getElementById('xPhantoms').checked,
        skip_artifacts: document.getElementById('xArtifacts').checked,
        join_gap: n('xJoin'), padding: n('xPad'),
        order: v('xOrder'), per_recording: n('xPerRec'), max_hours: n('xHours'),
        max_clips: n('xClips'),
        conversations: EXTRACT.conv || {},
    };
}

/* What a preview is computed from — the person and every filter. */
function extractKey() {
    const person = document.getElementById('xPerson')?.value || '';
    return person ? JSON.stringify([person, extractOptions()]) : '';
}

async function extractPreview() {
    EXTRACT.key = extractKey();
    const person = document.getElementById('xPerson')?.value;
    const box = document.getElementById('xPreview');
    if (!box) return;
    if (!person) {
        EXTRACT.run = null;
        EXTRACT.stale = false;
        box.innerHTML = '<div class="vx-empty">Choose a person — the preview shows what the export will contain.</div>';
        extractStaleMark();
        return;
    }
    const mine = ++EXTRACT.seq;
    EXTRACT.busy = true;
    if (typeof clipEditClose === 'function') clipEditClose();
    if (!EXTRACT.run) {
        box.innerHTML = `<div class="vx-empty">Finding the voices and sentences of ${esc(person)} …</div>`;
    }
    extractStaleMark();
    let d;
    try {
        d = await vxJson('/api/voices/extract/preview', {person, options: extractOptions()});
    } catch (e) {
        if (mine !== EXTRACT.seq) return;
        EXTRACT.busy = false;
        EXTRACT.run = null;
        EXTRACT.stale = false;
        box.innerHTML = `<div class="vx-card"><div class="vx-note err">${esc(e.message || String(e))}</div></div>`;
        extractButtons();
        return;
    }
    if (mine !== EXTRACT.seq) return;            // a newer filter took over
    EXTRACT.busy = false;
    EXTRACT.stale = false;
    const samePerson = EXTRACT.run && EXTRACT.run.person === d.person;
    EXTRACT.run = d;
    if (!samePerson) { EXTRACT.offset = 0; EXTRACT.state = 'all'; }
    EXTRACT.offset = Math.min(EXTRACT.offset, Math.max(0, d.total - 1));
    EXTRACT.totals = null;
    extractDrawPreview();
    await extractLoadPage();
    extractStaleMark();
}

/* What the export takes — from the server's totals once there, which count
 * the hand edits (dropped clips out, kept ones counted). */
function extractTotals() {
    const r = EXTRACT.run;
    if (!r) return null;
    if (EXTRACT.totals) return EXTRACT.totals;
    return {clips: r.stats.clips, seconds: r.stats.seconds, kept: r.stats.kept || 0,
            kept_seconds: 0, dropped: 0, edited: r.stats.edited || 0};
}

/* What the export will write — the same estimate the server checks the
 * free space against (core/voice_extract.estimate_bytes). All of one
 * person can be hundreds of hours: 500 h are ~80 GB of WAV at 22 kHz. */
const EXTRACT_BYTES_PER_SEC = {mp3: 24000, opus: 8000};

function extractChosen() {
    const t = extractTotals();
    if (!t) return {clips: 0, seconds: 0};
    return document.getElementById('xOnlyKept')?.checked
        ? {clips: t.kept, seconds: t.kept_seconds, recordings: t.kept_recordings || 0,
           recordings_bytes: t.kept_recordings_bytes || 0}
        : {clips: t.clips, seconds: t.seconds, recordings: t.recordings || 0,
           recordings_bytes: t.recordings_bytes || 0};
}

function extractEstimate() {
    const c = extractChosen();
    if (!c.clips) return 0;
    if (extractMode() === 'recordings') return (c.recordings_bytes || 0) + 2e6;
    const radio = name => document.querySelector(`input[name="${name}"]:checked`)?.value;
    const rate = Number(document.getElementById('xRate')?.value) || 22050;
    if (radio('xMode') !== 'single') return c.seconds * rate * 2 * 1.01 + 44 * c.clips;
    const fmt = document.getElementById('xFormat')?.value || 'flac';
    const gap = Number(document.getElementById('xGap')?.value) || 0;
    const total = c.seconds + gap * c.clips;
    return total * (EXTRACT_BYTES_PER_SEC[fmt] || rate * 2 * (fmt === 'flac' ? 0.62 : 1));
}

function extractButtons() {
    const t = extractTotals();
    const kept = document.getElementById('xOnlyKept');
    if (kept) {
        kept.disabled = !t || !t.kept;
        if (kept.disabled) kept.checked = false;
        const l = document.getElementById('xOnlyKeptLabel');
        if (l) l.textContent = `Only the clips I kept${t && t.kept ? ` (${vxN(t.kept)} · ${vxHours(t.kept_seconds / 3600)})` : ''}`;
    }
    const b = document.getElementById('xExportBtn');
    const running = EXTRACT.status && EXTRACT.status.running;
    if (!b) return;
    const c = extractChosen();
    const own = extractDestMode() === 'folder';
    const d = EXTRACT.dest;
    const destBad = own && !(d && d.ok);
    // The server checks the same before it starts (start_export)
    const full = own && d && d.ok && d.free != null && extractEstimate() * 1.1 + 500e6 > d.free;
    b.disabled = !EXTRACT.run || !c.clips || running || EXTRACT.stale || EXTRACT.busy || destBad || full;
    b.textContent = EXTRACT.stale || EXTRACT.busy ? 'Updating the preview …'
        : !c.clips ? 'Export'
        : destBad ? (d ? 'Choose a folder that can be written' : 'Checking the folder …')
        : full ? `Not enough space in the folder — about ${fmtBytes(extractEstimate())} needed`
        : extractMode() === 'recordings'
            ? `${own ? 'Save' : 'Export'} ${vxN(c.recordings)} recording${c.recordings === 1 ? '' : 's'} · about ${fmtBytes(extractEstimate())}`
        : `${own ? 'Save' : 'Export'} ${extractClips(c.clips)} · ${vxHours(c.seconds / 3600)} · about ${fmtBytes(extractEstimate())}`;
}

// ── Where the export goes ────────────────────────────────────────────────
//
// The folder of your own is checked as it is typed or picked
// (/api/voices/extract/dest): it must exist, lie in the allowed folders
// and be writable — the export button waits for that. The choice is
// remembered in this browser for the next export.

const EXTRACT_DEST_KEY = 'wtmExtractDest';
let extractDestTimer = null;

function extractDestMode() {
    return document.querySelector('input[name="xDest"]:checked')?.value === 'folder' ? 'folder' : 'exports';
}

function extractDestRestore() {
    let saved = null;
    try { saved = JSON.parse(localStorage.getItem(EXTRACT_DEST_KEY) || 'null'); } catch (e) { /* none */ }
    if (saved) {
        const path = document.getElementById('xDestPath');
        if (path && typeof saved.path === 'string') path.value = saved.path;
        const un = document.getElementById('xUnpacked');
        if (un && typeof saved.unpacked === 'boolean') un.checked = saved.unpacked;
        const r = document.querySelector(`input[name="xDest"][value="${saved.mode === 'folder' ? 'folder' : 'exports'}"]`);
        if (r) r.checked = true;
    }
    extractDestUi();
}

function extractDestRemember() {
    try {
        localStorage.setItem(EXTRACT_DEST_KEY, JSON.stringify({
            mode: extractDestMode(),
            path: (document.getElementById('xDestPath')?.value || '').trim(),
            unpacked: !!document.getElementById('xUnpacked')?.checked}));
    } catch (e) { /* a private window: not remembered */ }
}

function extractDestUi() {
    const own = extractDestMode() === 'folder';
    const box = document.getElementById('xDestBox');
    if (box) box.hidden = !own;
    const single = document.querySelector('input[name="xMode"]:checked')?.value === 'single';
    const row = document.getElementById('xUnpackedRow');
    if (row) row.hidden = single;
    extractDestRemember();
    if (own) extractDestCheckSoon(0);
    else extractButtons();
}

function extractDestCheckSoon(ms = 400) {
    clearTimeout(extractDestTimer);
    EXTRACT.dest = null;                    // unknown until checked
    extractButtons();
    extractDestTimer = setTimeout(extractDestCheck, ms);
}

async function extractDestCheck() {
    const field = document.getElementById('xDestPath');
    const path = (field?.value || '').trim();
    extractDestRemember();
    let d;
    if (!path) {
        d = {ok: false, error: 'type the folder, or choose it with the folder button'};
    } else {
        try {
            d = await (await fetch('/api/voices/extract/dest?path=' + encodeURIComponent(path))).json();
        } catch (e) {
            d = {ok: false, error: String(e.message || e)};
        }
        if (path !== (field?.value || '').trim()) return;     // typed on meanwhile
    }
    EXTRACT.dest = d;
    const info = document.getElementById('xDestInfo');
    if (info) {
        const single = document.querySelector('input[name="xMode"]:checked')?.value === 'single';
        const what = single ? 'the audio file with its SRT and index go'
            : document.getElementById('xUnpacked')?.checked ? 'a new folder is made' : 'the ZIP goes';
        info.classList.toggle('ok', !!d.ok);
        info.classList.toggle('err', !d.ok);
        info.textContent = d.ok
            ? `✓ ${d.path}${d.free != null ? ` · ${fmtBytes(d.free)} free` : ''} — ${what} in it, nothing there is replaced`
            : `✕ ${d.error || 'not usable'}`;
    }
    extractButtons();
}

/* A small folder browser under the field — the allowed folders only, like
 * the one in Settings (/api/browse). */
async function extractBrowse(path, note = '') {
    const box = document.getElementById('xBrowser');
    if (!box) return;
    if (path === undefined) {
        if (!box.hidden) { box.hidden = true; return; }
        path = (document.getElementById('xDestPath')?.value || '').trim();
    }
    box.hidden = false;
    box.innerHTML = '<div class="map-small">Loading …</div>';
    let d;
    try {
        const r = await fetch('/api/browse?path=' + encodeURIComponent(path || ''));
        d = await r.json();
        if (!r.ok && !d.dirs) throw new Error(d.error || `HTTP ${r.status}`);
    } catch (e) {
        box.innerHTML = `<div class="vx-note err">${esc(e.message || String(e))} — type the path into the field instead.</div>`;
        return;
    }
    // Outside the allowed folders the answer names where browsing may start
    if (d.error && !(d.dirs || []).length && d.path && d.path !== path && !note) {
        extractBrowse(d.path, `${path} can't be browsed: ${d.error}`);
        return;
    }
    box._path = d.path;
    box.innerHTML = `${note ? `<div class="vx-note warn">${esc(note)}</div>` : ''}
        <div class="vx-browser-head">
            <span class="vx-browser-path" title="${esc(d.path)}"><bdi>${esc(d.path)}</bdi></span>
            <button type="button" class="vx-btn primary" onclick="extractBrowsePick()">Use this folder</button>
            <button type="button" class="vx-btn" onclick="document.getElementById('xBrowser').hidden = true">Cancel</button>
        </div>
        <div class="vx-browser-list">
            ${(d.dirs || []).map(x => `<button type="button" data-path="${esc(x.path)}" onclick="extractBrowse(this.dataset.path)">
                ${x.name === '..' ? '↑ ..' : '📁 ' + esc(x.name)}</button>`).join('')
              || '<div class="map-small">No subfolders</div>'}
            ${d.error ? `<div class="vx-note err">${esc(d.error)}</div>` : ''}
        </div>`;
}

function extractBrowsePick() {
    const box = document.getElementById('xBrowser');
    const field = document.getElementById('xDestPath');
    if (!box || !field || !box._path) return;
    field.value = box._path;
    box.hidden = true;
    extractDestCheckSoon(0);
}

function extractDrawPreview() {
    const box = document.getElementById('xPreview');
    const d = EXTRACT.run;
    if (!box || !d) return;
    const s = d.stats;
    vxState('extract', `${d.person}: ${extractClips(s.clips)} · ${vxHours(s.seconds / 3600)}`);
    const top = Math.max(1, ...s.years.map(y => y[1]));
    const stat = (v, l, t = '') => `<div class="vx-stat" title="${esc(t)}"><div class="vx-stat-v">${v}</div><div class="vx-stat-l">${l}</div></div>`;
    const bands = s.certainty || [];
    const bandTop = Math.max(1, ...bands.map(b => b[1]));
    // Voices without speech spans: only a topic where the preview has some
    const leftOut = (s.dropped.find(([why]) => why === 'voice without speech spans') || [0, 0])[1];
    const legacy = s.by_transcript > 0 || leftOut > 0;
    const conv = s.conversations && typeof s.conversations === 'object' ? s.conversations : null;
    const convNotes = [];
    if (conv && Array.isArray(conv.people_unknown) && conv.people_unknown.length) {
        convNotes.push(`${esc(conv.people_unknown.join(', '))}: no name of the voice library, nor typed in a transcript — no conversation together`);
    }
    if (conv && Array.isArray(conv.ignored) && conv.ignored.length) {
        convNotes.push(`Conversation filters not understood and left out: ${esc(conv.ignored.join(', '))}`);
    }
    box.innerHTML = `
        <div class="vx-stale-note">The filters changed — updating the preview …</div>
        <div class="vx-stats">
            ${stat(vxN(s.clips), s.clips === 1 ? 'clip' : 'clips', 'Sentences that pass every filter')}
            ${stat(vxHours(s.seconds / 3600), 'of audio')}
            ${stat(vxN(s.recordings), s.recordings === 1 ? 'recording' : 'recordings')}
            ${stat(vxN(s.windows_named), s.windows_named === 1 ? 'named voice' : 'named voices', 'Windows named as this person')}
            ${stat(vxN(s.windows_recognized), s.windows_recognized === 1 ? 'recognized voice' : 'recognized voices', 'Windows that sound like this person and like nobody else named')}
            ${legacy ? stat(vxN(s.by_transcript), 'without spans', 'Voices without stored speech spans: checked by the transcript instead') : ''}
            ${conv ? stat(vxN(conv.found), Number(conv.found) === 1 ? 'conversation' : 'conversations',
                `Found by the conversation filters, in ${vxN(conv.recordings)} recording${Number(conv.recordings) === 1 ? '' : 's'} — the clips come from them`) : ''}
        </div>
        ${convNotes.map(t => `<div class="vx-note err">${t}</div>`).join('')}
        ${legacy ? `<div class="vx-card vx-legacy-note"><label>
            <input type="checkbox" id="xStrict"${EXTRACT.strict ? ' checked' : ''} onchange="extractStrict(this.checked)">
            <span>${EXTRACT.strict
                ? `<b>Voices without speech spans are left out</b> — ${vxN(leftOut)} ${leftOut === 1 ? 'voice has' : 'voices have'} no stored speech spans.`
                : `<b>Leave out the ${extractClips(s.by_transcript)} from voices without speech spans</b> — for them,
                   whether only this voice speaks is checked by the transcript's overlap marker instead, and the cut is less exact.
                   <i>Renew voices</i> on their recordings makes them anew and keeps the names
                   (queue editor → From library → Has voices without stored speech spans).`}</span></label></div>` : ''}
        <div class="vx-x-cards">
        ${bands.length ? `<div class="vx-card"><h3>How sure Whisper was</h3>
            <div class="vx-sub" style="margin-bottom:6px">Per sentence — "Whisper sure at least" cuts the bottom off.</div>
            ${bands.map(([b, n]) => `<div class="vx-band"><span>${esc(b)}</span>
                <span class="vx-band-bar"><i style="width:${Math.max(1, Math.round(100 * n / bandTop))}%"></i></span>
                <b>${vxN(n)}</b></div>`).join('')}</div>` : ''}
        ${s.years.length > 1 ? `<div class="vx-card"><h3>By year</h3>
            <div class="vx-years">${s.years.map(([y, sec]) => `
                <div class="vx-year" title="${esc(y)}: ${vxHours(sec / 3600)}">
                    <div style="height:${Math.max(2, Math.round(70 * sec / top))}px"></div><span>${esc(y)}</span></div>`).join('')}</div></div>` : ''}
        ${s.dropped.length ? `<div class="vx-card"><h3>Left out</h3>
            <div class="vx-sub" style="margin-bottom:6px">Sentences (of ${vxN(s.sentences)}) — and whole voices, for folder, date and speech spans:</div>
            ${s.dropped.map(([why, n]) => `<div class="vx-drop"><span>${esc(why)}</span><b>${vxN(n)}</b></div>`).join('')}</div>` : ''}
        </div>
        <div class="vx-card vx-clips-card"><h3>Clips</h3>
            <div class="vx-sub">Cut from the recording when you play them. <b>✎</b> opens a clip to trim it by hand and correct its text;
                ✓ keeps it, ✕ leaves it out. What you change is kept for ${esc(d.person)} — a new preview has it too
                (until the server restarts). <button class="map-mini" onclick="extractForget()"
                title="Every clip of ${esc(d.person)} back to its sentence — nothing kept, nothing dropped">Forget my changes</button></div>
            <div class="vx-toolbar vx-clip-bar">
                <div class="vx-seg" id="xStates">${[['all', 'All'], ['open', 'Open'], ['edited', 'Edited'], ['kept', 'Kept'], ['dropped', 'Dropped']]
                    .map(([k, l]) => `<button class="vx-seg-btn${EXTRACT.state === k ? ' active' : ''}" data-state="${k}" onclick="extractState('${k}')">${l}</button>`).join('')}</div>
                <span class="vx-gap"></span>
                <span class="map-small" id="xPageInfo"></span>
                <button class="map-mini" onclick="extractPagePrev()" title="Previous page">◀</button>
                <button class="map-mini" onclick="extractPageNext(false)" title="Next page">▶</button>
            </div>
            <div class="map-small vx-words-note" id="xWordsNote"></div>
            <div id="xClipList"><div class="map-small">Loading the clips …</div></div>
        </div>
        ${s.top.length ? `<div class="vx-card"><h3>Recordings</h3>
            ${s.top.slice(0, 40).map(r => `<div class="vx-file"><span class="vx-file-name" title="${esc(r.display)}">${esc(r.display)}</span>
                <span class="map-small">${extractClips(r.clips)} · ${vxHours(r.seconds / 3600)}</span></div>`).join('')}
            ${s.top.length > 40 ? `<div class="map-small" style="margin-top:6px">… and ${vxN(s.recordings - 40)} more</div>` : ''}</div>` : ''}`;
}

// ── The clips, page by page ──────────────────────────────────────────────

async function extractLoadPage() {
    const r = EXTRACT.run;
    const list = document.getElementById('xClipList');
    if (!r || !list) return;
    // Only the newest page request may land: a double ▶ or a filter switch
    // while a page loads would otherwise show page N under the counter N+1.
    const seq = EXTRACT.pageSeq = (EXTRACT.pageSeq || 0) + 1;
    let d;
    try {
        d = await (await fetch(`/api/voices/extract/clips/${encodeURIComponent(r.run_id)}`
            + `?offset=${EXTRACT.offset}&limit=${EXTRACT_PAGE}&state=${EXTRACT.state}`)).json();
        if (d.error) throw new Error(d.error);
    } catch (e) {
        if (seq !== EXTRACT.pageSeq) return;
        list.innerHTML = `<div class="vx-note err">${esc(e.message || String(e))}</div>`;
        return;
    }
    if (seq !== EXTRACT.pageSeq) return;
    if (!EXTRACT.run || EXTRACT.run.run_id !== r.run_id) return;
    EXTRACT.page = d.clips;
    EXTRACT.total = d.total;
    EXTRACT.totals = d.totals;
    extractDrawPage();
    extractButtons();
}

function extractDrawPage() {
    const list = document.getElementById('xClipList');
    const info = document.getElementById('xPageInfo');
    if (info) {
        info.textContent = EXTRACT.total
            ? `${vxN(EXTRACT.offset + 1)}–${vxN(Math.min(EXTRACT.total, EXTRACT.offset + EXTRACT_PAGE))} of ${vxN(EXTRACT.total)}`
            : '';
    }
    if (!list) return;
    list.innerHTML = EXTRACT.page.length ? EXTRACT.page.map(extractClipHtml).join('')
        : `<div class="map-small">${EXTRACT.state === 'all' ? 'No clips.' : 'None here.'}</div>`;
    const note = document.getElementById('xWordsNote');
    if (note) {
        note.innerHTML = EXTRACT.page.some(c => c.words && c.words.length)
            ? `Words are coloured by how sure the alignment is — <span class="vx-w">plain</span> from 60 %,
               <span class="vx-w w-mid">amber</span> 35–60 %, <span class="vx-w w-lo">red</span> below;
               the word being said lights up while a clip plays.`
            : 'With word timestamps (Settings → Transcription) every word is coloured by how sure it is, and lights up while it plays.';
    }
}

/* The clip's text — word by word where the transcript has words, each
 * coloured by how sure the alignment is (0–1 per word). The bands mark
 * the doubtful few: most words score well above 0.6 — colouring
 * everything under 0.75 coloured a large share of all words. Below 0.6
 * and below 0.35 are only a small minority. */
function extractTextHtml(c) {
    if (!c.words || !c.words.length) return esc(c.text);
    return `<span class="vx-words">${c.words.map((w, k) => {
        const p = w.p;
        const none = p === null || p === undefined;
        const cls = none ? ' w-none' : (p < 0.35 ? ' w-lo' : (p < 0.6 ? ' w-mid' : ''));
        const tip = none ? 'no time for this word' : `${Math.round(p * 100)} % sure`;
        return `<span class="vx-w${cls}" data-k="${k}" title="${tip}">${esc(w.w)}</span>`;
    }).join(' ')}</span>`;
}

/* The word being said at ``t`` (seconds in the recording) lights up. */
function extractWordsAt(box, words, t) {
    if (!box || !words) return;
    let on = -1;
    if (t !== null) {
        for (let k = 0; k < words.length; k++) {
            const w = words[k];
            if (w.a !== null && w.a <= t && t < (w.b !== null ? w.b : w.a + 0.3)) { on = k; break; }
        }
    }
    box.querySelectorAll('.vx-w.on').forEach(e => { if (Number(e.dataset.k) !== on) e.classList.remove('on'); });
    if (on >= 0) box.querySelector(`.vx-w[data-k="${on}"]`)?.classList.add('on');
}

/* A clip's player starts at the clip's start: the time in the recording
 * is start + currentTime. One listener for all players, in the capture
 * phase — `play` doesn't bubble. */
if (typeof document !== 'undefined' && !window.__extractWordsListening) {
    window.__extractWordsListening = true;
    document.addEventListener('play', e => {
        const a = e.target;
        if (!a || a.tagName !== 'AUDIO' || !a.classList.contains('vx-clip-play')) return;
        const row = a.closest('.vx-clip');
        const i = row ? Number(row.id.replace('xClip-', '')) : -1;
        const c = (EXTRACT.page || []).find(x => x.i === i);
        if (!c || !c.words) return;
        const box = row.querySelector('.vx-words');
        const tick = () => {
            if (a.paused || a.ended || !a.isConnected) { extractWordsAt(box, c.words, null); return; }
            extractWordsAt(box, c.words, c.start + a.currentTime);
            requestAnimationFrame(tick);
        };
        tick();
    }, true);
}

function extractClipHead(c) {
    const rid = encodeURIComponent(EXTRACT.run.run_id);
    const badge = c.state === 'kept' ? '<span class="vx-tag kept">kept</span>'
        : (c.state === 'dropped' ? '<span class="vx-tag undone">dropped</span>' : '');
    // How sure Whisper was comes right after the length: the line ends in
    // the file name, and a long one is cut off at the right (the certainty
    // would sit behind it and never be visible)
    const sure = c.certainty === null || c.certainty === undefined ? ''
        : `<span class="vx-sure${c.certainty >= 90 ? ' hi' : (c.certainty < 75 ? ' lo' : '')}"
                 title="How sure Whisper was of this sentence">${c.certainty} % sure</span> · `;
    // The bounds belong in the address: a trimmed clip is a different cut
    const src = `/api/voices/extract/clip/${rid}/${c.i}?v=${Math.round(c.start * 1000)}-${Math.round(c.end * 1000)}`;
    return `<div class="vx-clip${c.state === 'dropped' ? ' dropped' : ''}${CLIPEDIT.i === c.i ? ' editing' : ''}" id="xClip-${c.i}">
            <audio class="vx-clip-play" controls preload="none" src="${src}"></audio>
            <div class="vx-clip-main">
                <div class="vx-clip-text">${extractTextHtml(c)}${badge}${c.edited ? '<span class="vx-tag">edited</span>' : ''}</div>
                <div class="vx-clip-meta" title="${esc(c.display)}">${(c.end - c.start).toFixed(1)} s · ${sure}
                    ${c.named ? 'named' : `recognized ${Number(c.sim).toFixed(2)}`} ·
                    ${c.clean_by === 'spans' ? 'spans' : 'transcript'} ·
                    ${c.date ? esc(c.date) + ' · ' : ''}${esc(String(c.display).split('/').pop())}</div>
            </div>
            <div class="vx-clip-acts">
                <button class="map-mini" onclick="clipEditOpen(${c.i})" title="Trim by hand, correct the text">✎</button>
                <button class="map-mini${c.state === 'kept' ? ' map-mini-on' : ''}" onclick="extractMark(${c.i}, '${c.state === 'kept' ? '' : 'kept'}')" title="Keep — for an export of the kept clips only">✓</button>
                <button class="map-mini map-mini-danger${c.state === 'dropped' ? ' map-mini-on' : ''}" onclick="extractMark(${c.i}, '${c.state === 'dropped' ? '' : 'dropped'}')" title="${c.state === 'dropped' ? 'Take it back in' : 'Leave it out of the export'}">✕</button>
                ${c.edited ? `<button class="map-mini" onclick="extractReset(${c.i})" title="Back to the bounds and the text of the sentence">↺</button>` : ''}
            </div>
        </div>`;
}

function extractClipHtml(c) {
    return extractClipHead(c) + `<div class="vx-clip-edit" id="xEdit-${c.i}" hidden></div>`;
}

/* A clip changed (the editor saved it, or ✓/✕/↺): its row and the counts.
 * Only the row's head is replaced — an open editor below it stays. */
function extractClipUpdated(clip, totals) {
    const k = EXTRACT.page.findIndex(x => x.i === clip.i);
    if (k >= 0) EXTRACT.page[k] = clip;
    if (totals) EXTRACT.totals = totals;
    const row = document.getElementById(`xClip-${clip.i}`);
    if (row) {
        const tmp = document.createElement('div');
        tmp.innerHTML = extractClipHead(clip);
        row.replaceWith(tmp.firstElementChild);
    }
    extractButtons();
}

async function extractMark(i, state) {
    if (!EXTRACT.run) return;
    try {
        const d = await vxJson(`/api/voices/extract/edit/${encodeURIComponent(EXTRACT.run.run_id)}/${i}`, {state});
        extractClipUpdated(d.clip, d.totals);
    } catch (e) {
        Toast.error('Not changed: ' + (e.message || e));
    }
}

async function extractReset(i) {
    if (!EXTRACT.run) return;
    if (CLIPEDIT.i === i) clipEditClose();
    try {
        const d = await vxJson(`/api/voices/extract/edit/${encodeURIComponent(EXTRACT.run.run_id)}/${i}`, {reset: true});
        extractClipUpdated(d.clip, d.totals);
    } catch (e) {
        Toast.error('Not reset: ' + (e.message || e));
    }
}

async function extractForget() {
    if (!EXTRACT.run) return;
    const t = extractTotals() || {};
    if ((t.edited || t.kept || t.dropped) && !confirm(
            `Forget every change to the clips of ${EXTRACT.run.person}?\n\n`
            + `${t.edited || 0} trimmed or corrected, ${t.kept || 0} kept, ${t.dropped || 0} dropped.`)) return;
    try {
        await vxJson('/api/voices/extract/forget', {person: EXTRACT.run.person});
    } catch (e) {
        Toast.error('Not forgotten: ' + (e.message || e));
        return;
    }
    EXTRACT.key = '';
    extractPreview();
}

function extractState(k) {
    EXTRACT.state = k;
    EXTRACT.offset = 0;
    document.querySelectorAll('#xStates .vx-seg-btn').forEach(b => b.classList.toggle('active', b.dataset.state === k));
    clipEditClose();
    extractLoadPage();
}

function extractPagePrev() {
    if (EXTRACT.offset <= 0) return;
    EXTRACT.offset = Math.max(0, EXTRACT.offset - EXTRACT_PAGE);
    clipEditClose();
    extractLoadPage();
}

/* The next page — with `openFirst`, straight into its first clip (the
 * editor's "keep & next" at the end of a page). */
async function extractPageNext(openFirst) {
    if (EXTRACT.offset + EXTRACT_PAGE >= EXTRACT.total) {
        if (openFirst) Toast.success('That was the last clip.');
        return;
    }
    EXTRACT.offset += EXTRACT_PAGE;
    clipEditClose();
    await extractLoadPage();
    document.getElementById('xClipList')?.scrollIntoView({block: 'start'});
    if (openFirst && EXTRACT.page.length) clipEditOpen(EXTRACT.page[0].i);
}

// ── The export ───────────────────────────────────────────────────────────

async function extractExport() {
    if (!EXTRACT.run || EXTRACT.stale) return;
    const bytes = extractEstimate();
    const own = extractDestMode() === 'folder';
    if (own && !(EXTRACT.dest && EXTRACT.dest.ok)) return;
    const folder = own ? EXTRACT.dest.path : EXTRACT.defaults.folder;
    // Ask before a big one: it runs for hours and fills gigabytes.
    if (bytes > 5 * 1024 ** 3 && !confirm(
            `This export writes about ${fmtBytes(bytes)} (${vxHours(extractChosen().seconds / 3600)} of audio) `
            + `into ${folder} and takes a long time.\n\n`
            + 'The limits under "How much" (hours, clips, per recording) make it smaller. Export anyway?')) return;
    const radio = name => document.querySelector(`input[name="${name}"]:checked`)?.value;
    const body = {
        run_id: EXTRACT.run.run_id, mode: radio('xMode'),
        format: document.getElementById('xFormat').value,
        sample_rate: Number(document.getElementById('xRate').value),
        normalize: document.getElementById('xNormalize').checked,
        snap: document.getElementById('xSnap').checked,
        only_kept: document.getElementById('xOnlyKept').checked,
        gap: Number(document.getElementById('xGap').value),
        name: document.getElementById('xName').value.trim(),
        dest: own ? EXTRACT.dest.path : '',
        unpacked: own && document.getElementById('xUnpacked').checked,
    };
    try {
        await vxJson('/api/voices/extract/export', body);
    } catch (e) {
        Toast.error('Export not started: ' + (e.message || e));
        return;
    }
    extractPoll();
}

async function extractStop() {
    try { await vxJson('/api/voices/extract/stop', {}); } catch (e) { /* reported by the status */ }
    extractPoll();
}

function extractStopPolling() {
    clearTimeout(EXTRACT.timer);
    EXTRACT.timer = null;
    if (typeof clipEditStop === 'function') clipEditStop();
}

/* The export's state, every second while it runs — and once on opening,
 * for the list of exports. */
async function extractPoll() {
    clearTimeout(EXTRACT.timer);
    let s;
    const visibleNow = () => document.getElementById('mapOverlay')?.classList.contains('visible');
    try {
        const r = await fetch('/api/voices/extract/status');
        s = await r.json();
        // s.error is the job's own outcome (e.g. 'stopped') — only a bad
        // HTTP status means the poll itself failed.
        if (!r.ok) throw new Error(r.status);
    } catch (e) {
        // A busy server, a 502 page or a restart: one failed poll must not
        // freeze the progress of an export that runs for hours — try again,
        // a little later each time.
        if (EXTRACT.status && EXTRACT.status.running && visibleNow()) {
            EXTRACT.pollFails = (EXTRACT.pollFails || 0) + 1;
            clearTimeout(EXTRACT.timer);
            EXTRACT.timer = setTimeout(extractPoll,
                Math.min(30000, 1000 * 2 ** Math.min(EXTRACT.pollFails, 5)));
        }
        return;
    }
    EXTRACT.pollFails = 0;
    const was = EXTRACT.status && EXTRACT.status.running;
    const mine = EXTRACT.status && EXTRACT.status.run_id && EXTRACT.status.run_id === s.run_id;
    EXTRACT.status = s;
    extractDrawJob();
    extractButtons();
    // Several callers start a poll; one chain of timers is enough.
    clearTimeout(EXTRACT.timer);
    if (s.running && visibleNow()) EXTRACT.timer = setTimeout(extractPoll, 1000);
    if (was && mine && !s.running && s.result) {
        Toast.success(`Export finished: ${s.result.recordings != null ? vxN(s.result.recordings) + (s.result.recordings === 1 ? " recording" : " recordings") : extractClips(s.result.clips)}`, 8000);
        // Chosen as a download: the browser saves it right away
        if (!s.result.dest && s.result.files && s.result.files.length) extractDownload(s.result.files[0]);
    }
}

function extractDrawJob() {
    const s = EXTRACT.status || {};
    const job = document.getElementById('xJob');
    if (job) {
        if (s.running) {
            const p = s.total ? Math.round(100 * s.done / s.total) : 0;
            const secs = (Date.now() / 1000) - (s.started || Date.now() / 1000);
            const eta = s.done ? secs / s.done * (s.total - s.done) : 0;
            job.innerHTML = `<div class="vx-bar"><div style="width:${p}%"></div></div>
                <div class="map-small" style="margin-top:5px">${vxN(s.done)} of ${s.mode === 'recordings'
                    ? `${vxN(s.total)} recording${s.total === 1 ? '' : 's'}` : extractClips(s.total)} · ${p} %
                    ${eta ? ` · about ${fmtDuration(eta)} left` : ''}</div>
                <div class="map-small" title="${esc(s.current || '')}" style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${esc(String(s.current || '').split('/').pop())}</div>
                <button class="vx-btn danger" style="margin-top:8px" onclick="extractStop()">Stop</button>`;
        } else if (s.error) {
            job.innerHTML = `<div class="vx-note ${s.error === 'stopped' ? 'warn' : 'err'}">${s.error === 'stopped'
                ? 'Stopped — nothing was left behind.' : 'Export failed: ' + esc(s.error)}</div>`;
        } else if (s.result) {
            const r = s.result;
            // Saved into a folder of your own: where, instead of a download
            const where = r.dest ? r.dest.replace(/\/+$/, '') + '/' + r.files[0] : '';
            job.innerHTML = `<div class="vx-note">Done: ${r.recordings != null ? `${vxN(r.recordings)} recording${r.recordings === 1 ? '' : 's'}`
                    : extractClips(r.clips)} · ${vxHours(r.seconds / 3600)}${r.recordings != null ? ' of the person' : ''} · ${fmtBytes(r.bytes)}</div>`
                + (!r.dest && r.files && r.files.length ? `<a class="vx-btn primary" style="margin-top:8px"
                        href="/api/voices/extract/download/${encodeURIComponent(r.files[0])}" download>
                        <svg class="ui-ico"><use href="#i-download"/></svg>Download ${esc(r.files[0])}</a>` : '')
                + (where ? `<div class="vx-note vx-saved">Saved as <b title="${esc(where)}">${esc(where)}</b>${r.files.length > 1
                        ? ` <span class="map-small">with ${r.files.slice(1).map(esc).join(', ')}</span>` : ''}
                        <button type="button" class="vx-btn" onclick="_copyText(this.dataset.path)" data-path="${esc(where)}" title="Copy the path">Copy path</button></div>` : '')
                + (r.left && r.left.length ? `<div class="vx-note warn">Left out while cutting: ${r.left.map(([w, n]) => `${vxN(n)} ${esc(w)}`).join(', ')}</div>` : '');
        } else {
            job.innerHTML = '';
        }
    }
    const files = document.getElementById('xFiles');
    if (files) {
        const list = s.files || [];
        files.innerHTML = list.length ? list.map(f => `
            <div class="vx-file">
                <span class="vx-file-name" title="${esc(f.name)}">${esc(f.name)}</span>
                <span class="map-small">${fmtBytes(f.bytes)}</span>
                <a class="vx-btn" href="/api/voices/extract/download/${encodeURIComponent(f.name)}"
                   title="Download"><svg class="ui-ico"><use href="#i-download"/></svg></a>
                <button class="vx-btn danger" onclick="extractDelete('${escJs(f.name)}')" title="Delete this export">✕</button>
            </div>`).join('') : '<div class="map-small">None yet.</div>';
    }
}

function extractDownload(name) {
    const a = document.createElement('a');
    a.href = `/api/voices/extract/download/${encodeURIComponent(name)}`;
    a.download = name;
    document.body.appendChild(a);
    a.click();
    a.remove();
}

async function extractDelete(name) {
    if (!confirm(`Delete ${name}?`)) return;
    try {
        await vxJson('/api/voices/extract/delete', {name});
    } catch (e) {
        Toast.error('Not deleted: ' + (e.message || e));
    }
    extractPoll();
}
