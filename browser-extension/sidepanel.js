/* ══════════════════════════════════════════════════════════════════════════
 *  The side panel: a video to read and ask about, and how it is done
 *
 *  A sidebar to read in comfort and send the LLM follow-up questions. It
 *  stays open beside the page and follows the tab: the video a watch page plays comes up with
 *
 *    · its summaries — Markdown for reading, every time in it ([0:04:12]) a
 *      link that plays the video from there;
 *    · a chat about it — its transcript goes along with every question
 *      (core/ingest.ask), the answers name the times;
 *    · its transcript — found in, every line played from.
 *
 *  The page's other videos and what was sent are tabs of their own; a title
 *  there shows that video here.
 *
 *  Defaults for summary and transcription, and an interactive mode to choose
 *  them per send: the form — the summary's template, the
 *  language it is written in, the Whisper model, language, speakers … —
 *  opens with "Options…", and for every send with "ask before sending" (⚙),
 *  where the defaults are set too. The page and the popup open the panel
 *  with what to show (``panelIntent`` in the session storage, background.js).
 * ══════════════════════════════════════════════════════════════════════════ */

const $ = id => document.getElementById(id);

const P = {
    tab: null,              // the tab looked at
    playing: '',            // the id of the video it plays
    pageVideos: [],         // its other videos
    allVideos: [],          // … and its own
    picked: [],             // what is selected on it, in its order
    askFirst: false,
    opts: null,             // the forms' choices and the defaults (/api/ingest/options)
    form: null,             // the form open: {mode, urls, titles, again}
    timer: 0,               // following a batch
    recentTimer: 0,
    open: new Set(),        // summaries unfolded in "Sent"
    clearAsked: 0,          // "clear all" waits for its second click
};

/* The video shown: the tab's, or one picked from the lists. */
let V = null;

/* The chat about it */
const C = {conv: null, run: null, error: '', timer: 0};

const page = msg => new Promise(resolve => {
    if (!P.tab) return resolve({error: 'no tab'});
    chrome.tabs.sendMessage(P.tab.id, msg, r =>
        resolve(chrome.runtime.lastError ? {error: chrome.runtime.lastError.message} : (r || {})));
});

const isIn = state => ['transcribed', 'summarized'].includes(state);
/* Its transcript is there — also while one more summary is made */
const hasText = () => !!V && (isIn(V.state) || !!(V.data && V.data.segments && V.data.segments.length));

function showError(text) {
    $('err').textContent = text || '';
    $('err').hidden = !text;
}

// ══ The server, ⚙ ═══════════════════════════════════════════════════════

async function health() {
    const d = await ask({type: 'health'});
    $('dot').className = 'dot ' + (d.ok ? 'ok' : 'bad');
    $('dot').title = d.ok ? `Connected · ${d.server}` : (d.error || 'Not reachable');
    const s = await ask({type: 'server'});
    $('server').value = s.server || '';
    $('server').placeholder = s.default || '';
    $('token').value = '';
    $('token').placeholder = s.token ? 'Access token saved — type to replace'
        : 'Access token — only with the login on';
    $('btnForgetToken').hidden = !s.token;
    $('serverNote').textContent = d.ok ? 'Connected.'
        : `${d.error || 'Not reachable'} — the address Whisper Transcription Manager is reached at from this browser.`;
    if (!d.ok) $('settings').hidden = false;
    return !!d.ok;
}

$('btnSettings').addEventListener('click', () => {
    $('settings').hidden = !$('settings').hidden;
    if (!$('settings').hidden) renderDefaults();
});
$('btnSave').addEventListener('click', async () => {
    const d = await ask({type: 'set-server', server: $('server').value, token: $('token').value});
    if (d.error) { $('serverNote').textContent = d.error; return; }
    if (await health()) {
        await loadOptions();
        renderDefaults();
        loadTab();
        loadRecent();
    }
});
$('btnForgetToken').addEventListener('click', async () => {
    await ask({type: 'forget-token'});
    health();
});
$('askFirst').addEventListener('change', () => {
    P.askFirst = $('askFirst').checked;
    chrome.storage.sync.set({askFirst: P.askFirst});
});
$('marksEverywhere').addEventListener('change', () => {
    chrome.storage.sync.set({marksEverywhere: $('marksEverywhere').checked});
});

async function loadOptions() {
    const d = await ask({type: 'options'});
    if (!d.error) P.opts = d;
    return P.opts;
}

function renderDefaults() {
    const box = $('defaultsForm');
    if (!P.opts) {
        box.innerHTML = '<div class="note">The server doesn’t answer — no defaults to show.</div>';
        return;
    }
    box.innerHTML = fieldsHtml(P.opts.defaults, {summary: true, transcription: true});
}

$('btnDefaults').addEventListener('click', async () => {
    const d = await ask({type: 'set-defaults', defaults: readFields($('defaultsForm'))});
    $('defaultsNote').textContent = d.error ? 'Not saved: ' + d.error : 'Saved.';
    if (!d.error && P.opts) P.opts.defaults = d.defaults;
    setTimeout(() => { $('defaultsNote').textContent = ''; }, 2500);
});

// ══ The form: how a video is done ═══════════════════════════════════════

const SPEAKERS = [['off', 'Off'], ['auto', 'Detect — how many: automatic'],
                  ...[1, 2, 3, 4, 5, 6, 7, 8].map(n => [String(n), `${n} speaker${n === 1 ? '' : 's'}`])];

function speakersOf(o) {
    if (o.diarize === false) return 'off';
    if (o.diarize === true) return o.numSpeakers > 0 ? String(o.numSpeakers) : 'auto';
    return '';
}

function fromSpeakers(v) {
    if (v === 'off') return {diarize: false, numSpeakers: null};
    if (v === 'auto') return {diarize: true, numSpeakers: 0};
    if (v) return {diarize: true, numSpeakers: Number(v)};
    return {diarize: null, numSpeakers: null};
}

const langName = code => code === 'auto' ? 'detect'
    : ((P.opts && P.opts.languages.find(l => l.id === code)) || {label: code}).label;

/* The template's own choice: an instruction written here instead of one of
 * the app's templates. */
const CUSTOM = '__custom__';

/* One choice: "" first — what it means is ``first``. */
function choice(key, value, first, options) {
    return `<select data-k="${key}"><option value="">${esc(first)}</option>${options.map(([v, label]) =>
        `<option value="${esc(v)}"${String(v) === String(value ?? '') ? ' selected' : ''}>${esc(label)}</option>`).join('')}</select>`;
}

/* The fields of the form and of ⚙'s defaults. ``o``: the options shown. */
function fieldsHtml(o, show) {
    const g = P.opts.general;
    const rows = [];
    if (show.summary) {
        rows.push('<div class="group">Summary</div>');
        const own = !!o.summary_prompt;
        rows.push(`<label>Template</label>${choice('summary_preset', own ? CUSTOM : o.summary_preset,
            `As in Whisper Transcription Manager (${g.summary_preset || 'its default'})`,
            [...P.opts.summary_presets.map(n => [n, n]), [CUSTOM, 'Custom…']])}`);
        rows.push(`<label class="own"${own ? '' : ' hidden'}>Instruction</label>
            <textarea class="own" data-k="summary_prompt" rows="4" maxlength="4000"${own ? '' : ' hidden'}
                placeholder="What the summary should be — e.g. “Only the tips and tricks, as a numbered list with the times”"
                >${esc(o.summary_prompt || '')}</textarea>`);
        rows.push(`<label>Written in</label>${choice('summary_language', o.summary_language,
            'The language of the video', P.opts.languages.map(l => [l.id, l.label]))}`);
        if (P.opts.summary_models.length) {
            rows.push(`<label>Model</label>${choice('summary_model', o.summary_model,
                `As in Whisper Transcription Manager (${g.summary_model || 'the summary task’s'})`,
                P.opts.summary_models.map(n => [n, n]))}`);
        }
    }
    if (show.transcription) {
        rows.push('<div class="group">Transcription</div>');
        rows.push(`<label>Whisper</label>${choice('model', o.model, `As in Whisper Transcription Manager (${g.model})`,
            P.opts.whisper_models.map(m => [m.id, m.label]))}`);
        rows.push(`<label>Language</label>${choice('language', o.language,
            `As in Whisper Transcription Manager (${langName(g.language)})`,
            [['auto', 'Detect automatically'], ...P.opts.languages.map(l => [l.id, l.label])])}`);
        rows.push(`<label>Speakers</label>${choice('speakers', speakersOf(o),
            `As in Whisper Transcription Manager (${g.diarize ? (g.numSpeakers ? g.numSpeakers + ' speakers' : 'detect') : 'off'})`,
            SPEAKERS)}`);
        rows.push(`<label>Vocal isolation</label>${choice('vocal_isolation',
            o.vocal_isolation === true ? 'on' : o.vocal_isolation === false ? 'off' : '',
            `As in Whisper Transcription Manager (${g.vocal_isolation ? 'on' : 'off'})`,
            [['on', 'On — music and noise taken out'], ['off', 'Off']])}`);
        rows.push(`<label>Download</label>${choice('audio_only',
            o.audio_only === true ? 'on' : o.audio_only === false ? 'off' : '',
            `As in Whisper Transcription Manager (${g.audio_only ? 'audio only' : 'video'})`,
            [['on', 'Audio only — smaller and faster'], ['off', 'Video and audio']])}`);
        if (P.opts.vocab_presets.length) {
            rows.push(`<label>Vocabulary</label>${choice('vocab_preset', o.vocab_preset,
                `As in Whisper Transcription Manager (${g.vocab_preset || 'none'})`, P.opts.vocab_presets.map(n => [n, n]))}`);
        }
    } else if (show.transcribedNote) {
        rows.push(`<div class="hint">${esc(show.transcribedNote)}</div>`);
    }
    return `<div class="fields">${rows.join('')}</div>`;
}

/* On, off, or "" — the general setting */
const SWITCHES = ['vocal_isolation', 'audio_only'];

/* The options the fields say — the keys of the server's options. */
function readFields(box) {
    const out = {};
    for (const s of box.querySelectorAll('select[data-k]')) {
        const k = s.dataset.k, v = s.value;
        if (k === 'speakers') Object.assign(out, fromSpeakers(v));
        else if (SWITCHES.includes(k)) out[k] = v === 'on' ? true : v === 'off' ? false : null;
        else out[k] = v;
    }
    // Custom: the instruction instead of a template — written, or none at all
    const t = box.querySelector('textarea[data-k="summary_prompt"]');
    if (t) {
        const own = out.summary_preset === CUSTOM;
        out.summary_prompt = own ? t.value.trim() : '';
        if (own) out.summary_preset = '';
    }
    return out;
}

/* Custom chosen: the field for the instruction shows, and takes the cursor. */
function ownInstruction(e) {
    const s = e.target;
    if (!s.matches || !s.matches('select[data-k="summary_preset"]')) return;
    const box = s.closest('.fields');
    const on = s.value === CUSTOM;
    box.querySelectorAll('.own').forEach(el => { el.hidden = !on; });
    if (on) box.querySelector('textarea.own').focus();
}
$('form').addEventListener('change', ownInstruction);
$('defaultsForm').addEventListener('change', ownInstruction);

/* The form for a send: one video or several, summarize or transcribe. */
async function openForm(f) {
    if (!P.opts && !(await loadOptions())) {
        showError('The server doesn’t answer — check the address behind ⚙.');
        return;
    }
    const one = f.urls.length === 1 && V && videoIdOf(f.urls[0]) === V.id ? V : null;
    // One summarized already: once more — another template, another language
    if (one && one.state === 'summarized') f = {...f, mode: 'summarize', again: true};
    if (one && one.state === 'transcribed') f = {...f, mode: 'summarize'};
    P.form = f;
    const what = f.urls.length === 1 ? (f.titles[0] || (one && one.title) || f.urls[0])
        : `${f.urls.length} videos`;
    const box = $('form');
    // Transcribed already: there is only the summary to choose
    const only = f.again ? 'Summarize again' : one && isIn(one.state) ? 'Summarize' : '';
    box.innerHTML = `<div class="head">${only || `<span class="seg">
            <button data-mode="summarize" class="${f.mode === 'summarize' ? 'on' : ''}">Summarize</button>
            <button data-mode="transcribe" class="${f.mode === 'transcribe' ? 'on' : ''}">Transcribe</button></span>`}
            <span class="note" style="flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap"
                  title="${esc(what)}">${esc(what)}</span></div>
        <div id="formFields">${fieldsHtml(P.opts.defaults, {
            summary: f.mode === 'summarize',
            transcription: !(one && isIn(one.state)),
            transcribedNote: 'Transcribed already — the summary uses its transcript.'})}</div>
        <label class="check"><input type="checkbox" id="remember"> Make these my defaults</label>
        <div class="acts"><button class="b p" id="formStart">${f.mode === 'transcribe' ? 'Transcribe' : 'Summarize'}</button>
            <button class="b" id="formCancel">Cancel</button></div>`;
    box.hidden = false;
    box.scrollIntoView({block: 'nearest'});
}

$('form').addEventListener('click', async e => {
    const b = e.target.closest('button');
    if (!b) return;
    const f = P.form;
    if (b.dataset.mode) {
        // Another mode: what was chosen so far stays
        const before = readFields($('form'));
        await openForm({...f, mode: b.dataset.mode});
        for (const s of $('form').querySelectorAll('select[data-k]')) {
            const k = s.dataset.k;
            if (k === 'speakers') s.value = speakersOf(before);
            else if (SWITCHES.includes(k)) s.value = before[k] === true ? 'on' : before[k] === false ? 'off' : '';
            else if (k === 'summary_preset' && before.summary_prompt) s.value = CUSTOM;
            else if (k in before) s.value = before[k] ?? '';
        }
        const t = $('form').querySelector('textarea.own');
        if (t && before.summary_prompt) {
            t.value = before.summary_prompt;
            $('form').querySelectorAll('.own').forEach(el => { el.hidden = false; });
        }
    } else if (b.id === 'formCancel') {
        $('form').hidden = true;
        P.form = null;
    } else if (b.id === 'formStart') {
        const options = readFields($('form'));
        if ($('form').querySelector('select[data-k="summary_preset"]')?.value === CUSTOM
                && !options.summary_prompt) {
            showError('Write the instruction for the summary first.');
            $('form').querySelector('textarea.own').focus();
            return;
        }
        if ($('remember').checked) {
            const d = await ask({type: 'set-defaults', defaults: options});
            if (!d.error && P.opts) P.opts.defaults = d.defaults;
        }
        b.disabled = true;
        const sent = await send(f.mode, f.urls, options, !!f.again, !!f.selection);
        b.disabled = false;
        if (sent) {
            $('form').hidden = true;
            P.form = null;
        }
    }
});

// ══ Sending ═════════════════════════════════════════════════════════════

/* To the server — ``options`` left out: the defaults. The page shows it
 * too: the button for its own video, the bar for the page's selection
 * (``selection``: these are it — sent, it empties). A video picked here
 * from Sent or Page is the panel's own: the page's selection stays. */
async function send(mode, urls, options, again, selection) {
    showError('');
    const d = await ask({type: 'ingest', mode, urls, options, again});
    if (d.error || !d.batch) {
        const err = d.error || 'no answer';
        showError('Not sent: ' + err + (/can't be reached|didn't answer|no answer/.test(err)
            ? ' — check the address behind ⚙.' : ''));
        return null;
    }
    const own = urls.length === 1 && videoIdOf(urls[0]) === P.playing;
    if (own) page({type: 'follow-main', batch: d.batch});
    else if (selection) page({type: 'follow', batch: d.batch, clear: true});
    if (V && urls.some(u => videoIdOf(u) === V.id)) follow(d.batch);
    loadRecent();
    return d.batch;
}

/* Summarize or Transcribe pressed: at once, or the form first. */
function sendOne(mode) {
    if (!V) return;
    if (P.askFirst) openForm({mode, urls: [V.url], titles: [V.title]});
    else send(mode, [V.url], undefined, false);
}

/* The video shown follows its batch: asked again three seconds after each
 * answer — an answer for a video no longer shown is dropped. */
function follow(b) {
    clearTimeout(P.timer);
    P.timer = 0;
    V.batch = b.id;
    take(b);
    if (b.status === 'working') P.timer = setTimeout(pollVideo, 3000);
}

async function pollVideo() {
    P.timer = 0;
    const want = V;
    if (!want || !want.batch) return;
    const id = want.batch;
    const d = await ask({type: 'batch', id});
    if (V !== want || V.batch !== id || P.timer) return;
    if (d.batch) take(d.batch);
    if (!d.batch || d.batch.status === 'working') P.timer = setTimeout(pollVideo, 3000);
}

/* The batch's word on the video shown — its own item only. */
function take(b) {
    const it = (b.items || []).find(x => videoIdOf(x.url) === V.id);
    if (!it) return;
    const was = V.state;
    V.state = it.state;
    V.progress = it.progress || 0;
    V.open = it.open || V.open;
    V.error = it.state === 'error' ? (it.error || 'Failed') : '';
    if (b.status !== 'working') {
        clearTimeout(P.timer);
        P.timer = 0;
    }
    // Transcribed, summarized: what it became
    if (isIn(it.state) && was !== it.state) loadData();
    renderVideo();
    renderPanes();
}

// ══ The video ═══════════════════════════════════════════════════════════

async function setVideo(v) {
    const id = videoIdOf(v.url);
    if (V && id && V.id === id) {
        if (v.title && v.title !== V.title && v.titleFromPage) V.match = false;
        V.title = v.title || V.title;
        if (v.state && !V.batch) V.state = v.state;
        renderVideo();
        return;
    }
    clearTimeout(P.timer);
    P.timer = 0;
    clearTimeout(C.timer);
    Object.assign(C, {conv: null, run: null, error: ''});
    $('form').hidden = true;
    // match: its title may find it in the library — not when a page other
    // than YouTube wrote it
    V = {url: v.url, id, title: v.title || '', state: v.state || '', open: v.open || '', abs: '',
         data: null, sum: 0, batch: '', progress: 0, error: '', match: !v.titleFromPage};
    renderVideo();
    renderPanes();
    if (!V.state) await known();
    if (isIn(V.state)) await loadData();
    if (tabNow() === 'ask') openChat(false);
}

/* The title the server may look the video up by. */
const matchTitle = () => V && V.match !== false ? V.title : '';

/* Where it stands in the library. */
async function known() {
    const want = V;
    const d = await ask({type: 'known', summaries: true, videos: [{url: V.url, title: matchTitle()}]});
    const row = (d.videos || [])[0];
    if (V !== want || !row) return;
    V.state = row.state || '';
    V.open = row.open || '';
    renderVideo();
}

/* Its transcript and summaries. */
async function loadData() {
    const want = V;
    const d = await ask({type: 'transcript', url: V.url, title: matchTitle()});
    if (V !== want) return;
    if (d.error) {
        V.data = null;
    } else {
        V.data = d;
        V.abs = d.abs_path;
        V.title = V.title || d.title;
        V.sum = 0;
    }
    renderVideo();
    renderPanes();
}

function renderVideo() {
    $('video').hidden = !V;
    $('noVideo').hidden = !!V;
    if (!V) {
        $('vacts').hidden = true;
        $('progress').hidden = true;
        return;
    }
    $('thumb').src = thumb(V.id, 'mqdefault');
    $('vtitle').textContent = V.title || V.url;
    const st = V.state ? (STATE[V.state] || [V.state, '']) : null;
    const pct = V.state === 'downloading' && V.progress ? ` ${V.progress} %` : '';
    $('vstate').innerHTML = (st ? `<span class="chip ${st[1]}">${esc(st[0] + pct)}</span>`
                                : '<span class="note">Not in Whisper Transcription Manager yet.</span>')
        + (V.open ? '<button class="l" data-open="video">open in Whisper Transcription Manager ↗</button>' : '')
        + (P.playing && V.id !== P.playing ? '<button class="l" data-back="1">← the tab’s video</button>' : '');
    const working = WORKING.includes(V.state);
    $('vacts').hidden = working;
    for (const b of $('vacts').querySelectorAll('button[data-send]')) {
        b.hidden = b.dataset.send === 'summarize' ? V.state === 'summarized' : isIn(V.state);
    }
    $('progress').hidden = !working;
    $('progress').firstElementChild.style.width = (V.state === 'downloading' ? Math.max(4, V.progress)
        : {waiting: 3, resolving: 6, queued: 30, transcribing: 60, summarizing: 85}[V.state] || 0) + '%';
    showError(V.error || '');
}

$('vacts').addEventListener('click', e => {
    const b = e.target.closest('button');
    if (!b || !V) return;
    if (b.dataset.send) sendOne(b.dataset.send);
    else if (b.id === 'btnForm') openForm({mode: 'summarize', urls: [V.url], titles: [V.title]});
});

// ══ The tabs ════════════════════════════════════════════════════════════

const tabNow = () => (document.querySelector('#tabs button.on') || {}).dataset?.tab || 'summary';

function showTab(name) {
    for (const b of document.querySelectorAll('#tabs button')) b.classList.toggle('on', b.dataset.tab === name);
    for (const p of document.querySelectorAll('.pane')) p.hidden = p.id !== 'pane-' + name;
    if (name === 'ask') openChat(false);
    if (name === 'sent') loadRecent();
    if (name === 'page') renderPage();
}

$('tabs').addEventListener('click', e => {
    const b = e.target.closest('button[data-tab]');
    if (b) showTab(b.dataset.tab);
});

function renderPanes() {
    renderSummary();
    renderTranscript();
    renderChat();
}

function placeholder(text, buttons) {
    return `<div class="placeholder">${esc(text)}${buttons ? `<div class="acts">${buttons}</div>` : ''}</div>`;
}

/* What can be done with the video: summarize it, transcribe it — not when it
 * is transcribed already — or choose how. */
function sendButtons() {
    return '<button class="b p" data-send="summarize">⚡ Summarize</button>'
        + (V && isIn(V.state) ? '' : '<button class="b" data-send="transcribe">📝 Transcribe</button>')
        + '<button class="b" data-form="summarize">Options…</button>';
}

// ── Summary ──────────────────────────────────────────────────────────────

function renderSummary() {
    const box = $('pane-summary');
    if (!V) {
        box.innerHTML = placeholder('No video yet — open one on YouTube, or pick one under Page or Sent.');
        return;
    }
    const sums = (V.data && V.data.summaries) || [];
    if (!sums.length) {
        box.innerHTML = WORKING.includes(V.state)
            ? placeholder('It is on its way — the summary shows here when it is there.')
            : placeholder(V.state === 'transcribed' ? 'Transcribed — no summary yet.'
                              : 'Not in Whisper Transcription Manager yet.', sendButtons());
        return;
    }
    const i = Math.min(V.sum, sums.length - 1);
    const s = sums[i];
    const name = x => `${x.name || 'Summary'} · ${when((x.at || 0) * 1000)}`;
    box.innerHTML = `<div class="sumhead">${sums.length > 1
            ? `<select id="sumPick" title="Its summaries, newest first">${sums.map((x, n) =>
                `<option value="${n}"${n === i ? ' selected' : ''}>${esc(name(x))}</option>`).join('')}</select>`
            : `<span class="note">${esc(name(s))}</span><span class="gap"></span>`}
            ${WORKING.includes(V.state) ? '<span class="chip run">a new one is on its way</span>'
                : '<button class="l" data-form="again" title="Another template, another language">another one…</button>'}</div>
        <div class="md">${md(s.text)}</div>`;
}

$('pane-summary').addEventListener('change', e => {
    if (e.target.id === 'sumPick') {
        V.sum = Number(e.target.value) || 0;
        renderSummary();
    }
});

// ── Asking ───────────────────────────────────────────────────────────────

async function openChat(fresh) {
    if (!hasText()) {
        renderChat();
        return;
    }
    if (C.conv && !fresh && C.conv.attached && C.conv.attached.id === V.abs) {
        renderChat();
        return;
    }
    const want = V;
    C.error = '';
    const d = await ask({type: 'ask', url: V.url, title: matchTitle(), new: !!fresh});
    if (V !== want) return;
    if (d.error) {
        C.conv = null;
        C.error = d.error;
        renderChat();
        return;
    }
    await loadConv(d.conversation);
}

/* A conversation, for the video shown when it was asked — another one
 * shown meanwhile keeps its own. */
async function loadConv(id) {
    const want = V;
    const d = await ask({type: 'chat-conv', id});
    if (V !== want) return;
    C.conv = d.error ? null : d;
    C.error = d.error || '';
    renderChat();
    scrollChat();
}

function stepsHtml(steps) {
    if (!steps || !steps.length) return '';
    return `<div class="steps">${steps.map(s => `🔎 ${esc(s.name)}${s.summary ? ' · ' + esc(s.summary) : ''}`)
        .join('<br>')}</div>`;
}

function renderChat() {
    const box = $('chat');
    const running = !!C.run;
    $('btnAsk').textContent = running ? 'Stop' : 'Ask';
    $('question').disabled = !C.conv;
    $('btnNewChat').hidden = !C.conv;
    $('btnChatApp').hidden = !C.conv;
    if (!V) {
        box.innerHTML = placeholder('No video yet.');
        return;
    }
    if (!hasText()) {
        box.innerHTML = placeholder(WORKING.includes(V.state)
            ? 'Questions need its transcript — it is on its way.'
            : 'Questions need its transcript — summarize or transcribe it first.',
            WORKING.includes(V.state) ? '' : sendButtons());
        return;
    }
    if (C.error) {
        box.innerHTML = placeholder('No conversation: ' + C.error);
        return;
    }
    if (!C.conv) {
        box.innerHTML = placeholder('…');
        return;
    }
    const refs = C.conv.refs || {};
    const html = (C.conv.turns || []).map(t => t.role === 'user'
        ? `<div class="msg user">${esc(t.text)}</div>`
        : `<div class="msg bot${t.error ? ' err' : ''}">${stepsHtml(t.steps)}${t.error
            ? esc('Failed: ' + t.error) : `<div class="md">${md(t.text || '', refs)}</div>`}</div>`);
    if (running) {
        const ev = C.run.events.filter(Boolean);
        const text = ev.filter(e => e.type === 'text').map(e => e.text).join('\n\n');
        const tools = ev.filter(e => e.type === 'tool').map(e => ({name: e.name,
            summary: (ev.find(x => x.type === 'tool_done' && x.id === e.id) || {}).summary || '…'}));
        const thinking = ev.length && ev[ev.length - 1].type === 'thinking';
        html.push(`<div class="msg user">${esc(C.run.question)}</div>`);
        html.push(`<div class="msg bot">${stepsHtml(tools)}${text ? `<div class="md">${md(text, refs)}</div>`
            : `<div class="thinking">${thinking ? 'Thinking …' : 'Reading the transcript …'}</div>`}</div>`);
    }
    box.innerHTML = html.join('');
}

function scrollChat() {
    const box = $('chat');
    box.scrollTop = box.scrollHeight;
}

async function askQuestion() {
    if (C.run) {
        ask({type: 'chat-stop', id: C.run.id});
        return;
    }
    const text = $('question').value.trim();
    if (!text || !C.conv) return;
    const d = await ask({type: 'chat-send', text, conversation: C.conv.id});
    if (d.error) {
        C.error = d.error;
        renderChat();
        return;
    }
    $('question').value = '';
    C.run = {id: d.run, question: text, events: [], after: 0};
    renderChat();
    scrollChat();
    pollRun();
}

async function pollRun() {
    const r = C.run;
    if (!r) return;
    const d = await ask({type: 'chat-run', id: r.id, after: r.after});
    if (C.run !== r) return;
    if (d.error) {
        C.run = null;
        C.error = d.error;
        renderChat();
        return;
    }
    for (const e of d.events || []) r.events[e.n] = e;   // the last one again: its text grew
    r.after = Math.max(0, r.events.length - 1);
    if (d.done) {
        C.run = null;
        await loadConv(C.conv.id);
        return;
    }
    const box = $('chat');
    const atEnd = box.scrollHeight - box.scrollTop - box.clientHeight < 40;
    renderChat();
    if (atEnd) scrollChat();
    C.timer = setTimeout(pollRun, 700);
}

$('btnAsk').addEventListener('click', askQuestion);
$('question').addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) {
        e.preventDefault();
        if (!C.run) askQuestion();          // Enter never stops an answer
    }
});
$('btnNewChat').addEventListener('click', () => { if (!C.run) openChat(true); });
$('btnChatApp').addEventListener('click', () => {
    if (C.conv) ask({type: 'open', path: '/?chat=' + encodeURIComponent(C.conv.id)});
});

// ── The transcript ───────────────────────────────────────────────────────

const speakerName = s => /^SPEAKER_\d+$/.test(s) ? 'Speaker ' + (Number(s.slice(8)) + 1) : s;

function renderTranscript() {
    const box = $('segments');
    const segs = (V && V.data && V.data.segments) || [];
    $('find').hidden = !segs.length;
    if (!segs.length) {
        $('findNote').textContent = '';
        box.innerHTML = placeholder(!V ? 'No video yet.' : WORKING.includes(V.state)
            ? 'It is on its way — the transcript shows here when it is there.'
            : 'Not transcribed yet.', V && !WORKING.includes(V.state) && !isIn(V.state)
            ? sendButtons() : '');
        return;
    }
    const q = $('find').value.trim().toLowerCase();
    const rows = q ? segs.filter(s => s.text.toLowerCase().includes(q)
                                      || speakerName(s.speaker || '').toLowerCase().includes(q)) : segs;
    $('findNote').textContent = q ? `${rows.length} of ${segs.length} lines` : '';
    box.innerHTML = rows.map(s => `<div class="sg"><a class="time" href="#" data-t="${s.start}">${clock(s.start)}</a>
        <div>${s.speaker ? `<span class="who">${esc(speakerName(s.speaker))}</span>` : ''}${
            highlight(s.text, q)}</div></div>`).join('');
}

/* A line with what was searched marked — found in the text itself, each
 * piece escaped after: searching "amp" marked the inside of "&amp;". */
function highlight(text, q) {
    text = String(text || '');
    const low = text.toLowerCase();
    if (!q || low.length !== text.length) return esc(text);
    let out = '', i = 0, j;
    while ((j = low.indexOf(q, i)) >= 0) {
        out += esc(text.slice(i, j)) + '<mark>' + esc(text.slice(j, j + q.length)) + '</mark>';
        i = j + q.length;
    }
    return out + esc(text.slice(i));
}

let findTimer = 0;
$('find').addEventListener('input', () => {
    clearTimeout(findTimer);
    findTimer = setTimeout(renderTranscript, 150);
});

// ══ Clicks anywhere: a time, a reference, a send, a link ═════════════════

document.addEventListener('click', e => {
    const a = e.target.closest('a.time, a.ref, button[data-send], button[data-form], button[data-open], button[data-back]');
    if (!a || a.closest('#vacts')) return;
    e.preventDefault();
    if (a.matches('a.time')) {
        const box = a.closest('[data-url]');
        playAt(box ? box.dataset.url : V && V.url, Number(a.dataset.t) || 0, P.tab, P.playing);
    } else if (a.matches('a.ref')) {
        // A moment a chat answer found with the tools: this video's plays here
        const ref = ((C.conv && C.conv.refs) || {})[a.dataset.ref];
        if (!ref) return;
        if (V && ref.abs_path && ref.abs_path === V.abs) playAt(V.url, ref.start || 0, P.tab, P.playing);
        else if (ref.abs_path) ask({type: 'open', path: '/?open=' + encodeURIComponent(ref.abs_path) + '&t=' + (Number(ref.start) || 0)});
    } else if (a.dataset.send) {
        sendOne(a.dataset.send);
    } else if (a.dataset.form && V) {
        openForm({mode: 'summarize', urls: [V.url], titles: [V.title]});
    } else if (a.dataset.open && V && V.open) {
        ask({type: 'open', path: V.open});
    } else if (a.dataset.back) {
        loadTab();
    }
});

// ══ The tab looked at ═══════════════════════════════════════════════════

/* The tab looked at — switching fast asks several times at once: only the
 * latest question's answers count, an older one's never shows its tab. */
let tabRound = 0;
async function loadTab() {
    const round = ++tabRound;
    const [tab] = await chrome.tabs.query({active: true, currentWindow: true});
    if (round !== tabRound) return;
    P.tab = tab || null;
    let d = await page({type: 'list'});
    if (round !== tabRound) return;
    if (d.error && P.tab) {
        // A page open from before the extension was installed: the helper now
        const injected = await ask({type: 'inject', tabId: P.tab.id});
        if (round !== tabRound) return;
        if (injected.ok) d = await page({type: 'list'});
        if (round !== tabRound) return;
    }
    if (d.error) {
        P.playing = '';
        takeList({});
        renderVideo();
        return;
    }
    P.playing = d.main ? videoIdOf(d.main.url) : '';
    takeList(d);
    if (d.main) await setVideo(d.main);
    else renderVideo();
}

chrome.tabs.onActivated.addListener(() => loadTab());
chrome.tabs.onUpdated.addListener((id, info) => {
    // Another video in the same tab (YouTube changes the address, not the page):
    // the helper sees it a moment later
    if (P.tab && id === P.tab.id && (info.url || info.status === 'complete')) setTimeout(loadTab, 900);
});

// ── The page's other videos ──────────────────────────────────────────────

/* The page's videos as its helper lists them — the Page tab and the
 * selection drawn from them. */
function takeList(d) {
    P.allVideos = d.videos || [];
    P.pageVideos = P.allVideos.filter(v => !v.main);
    // An older helper (a page open from before an update) lists no selection
    P.picked = (d.selection || P.allVideos.filter(v => v.selected)).map(v => ({...v, id: v.id || videoIdOf(v.url)}));
    renderPage();
}

/* The page's selection changed (its ⊕, its bar, the watch page's menu): the
 * list again, the video shown left alone. */
async function refreshSelection() {
    const tab = P.tab;
    const d = await page({type: 'list'});
    if (P.tab !== tab || d.error) return;
    takeList(d);
}

chrome.runtime.onMessage.addListener((msg, sender) => {
    if (msg && msg.type === 'selection' && sender && sender.tab && P.tab && sender.tab.id === P.tab.id) {
        refreshSelection();
    }
});

/* What is selected on the page — shown above the tabs, whichever is open:
 * each video with its ✕, and what can be done with them. */
function renderSelection() {
    const picked = P.picked;
    $('selBox').hidden = !picked.length;
    if (!picked.length) {
        $('analyzeBox').hidden = true;
        return;
    }
    $('selTitle').textContent = `${picked.length} video${picked.length === 1 ? '' : 's'} selected on this page`;
    $('selList').innerHTML = picked.map(v => `<div class="v">
            ${v.id ? `<img src="${esc(thumb(v.id))}" alt="" data-show="${esc(v.key)}">` : ''}
            <span class="t" data-show="${esc(v.key)}" title="Show it here — ${esc(v.title || v.url)}">${v.kind === 'playlist' ? '▤ ' : ''}${esc(v.title || v.url)}</span>
            <button class="l x" data-unpick="${esc(v.key)}" title="Unselect it">✕</button></div>`).join('');
}

function renderPage() {
    const n = P.pageVideos.length;
    $('pageCount').textContent = n ? String(n) : '';
    const box = $('videos');
    if (!n) {
        box.innerHTML = placeholder(P.tab ? 'No other YouTube videos on this page.' : 'No page.');
    } else {
        const q = ($('filter').value || '').trim().toLowerCase();
        const shown = P.pageVideos.filter(v => !q || (v.title || v.url).toLowerCase().includes(q));
        box.innerHTML = shown.map(v => {
            const k = v.known ? STATE[v.known.state] || [v.known.state, 'ok'] : null;
            return `<div class="v"><input type="checkbox" data-k="${esc(v.key)}"${v.selected ? ' checked' : ''}>
                ${v.id ? `<img src="${esc(thumb(v.id))}" alt="" data-show="${esc(v.key)}">` : ''}
                <span class="t" data-show="${esc(v.key)}" title="Show it here — ${esc(v.title || v.url)}">${v.kind === 'playlist' ? '▤ ' : ''}${esc(v.title || v.url)}</span>
                ${k ? `<span class="chip ok">${esc(k[0])}</span>` : ''}</div>`;
        }).join('') || placeholder('No title matches.');
    }
    $('filter').hidden = n < 8;
    $('btnAll').hidden = n < 2;
    const q = ($('filter').value || '').trim().toLowerCase();
    const shown = P.pageVideos.filter(v => !q || (v.title || v.url).toLowerCase().includes(q));
    $('btnAll').textContent = pickAllLabel(shown, n, !!q);
    const selected = P.pageVideos.filter(v => v.selected).length;
    $('selCount').textContent = selected ? `${selected} selected` : '';
    renderSelection();
}

$('filter').addEventListener('input', renderPage);

$('videos').addEventListener('change', async e => {
    const cb = e.target.closest('input[data-k]');
    if (!cb) return;
    await page({type: 'toggle', key: cb.dataset.k, on: cb.checked});
    const v = P.pageVideos.find(x => x.key === cb.dataset.k);
    if (v) v.selected = cb.checked;
    await refreshSelection();
});

function showPicked(e) {
    const el = e.target.closest('[data-show]');
    if (!el) return;
    const v = P.allVideos.find(x => x.key === el.dataset.show) || P.picked.find(x => x.key === el.dataset.show);
    if (v && v.kind === 'video') setVideo({url: v.url, title: v.title, titleFromPage: !!v.titleFromPage});
}
$('videos').addEventListener('click', showPicked);

$('selList').addEventListener('click', async e => {
    const x = e.target.closest('button[data-unpick]');
    if (!x) return showPicked(e);
    await page({type: 'toggle', key: x.dataset.unpick, on: false});
    const v = P.allVideos.find(y => y.key === x.dataset.unpick);
    if (v) v.selected = false;
    P.picked = P.picked.filter(y => y.key !== x.dataset.unpick);
    renderPage();
});

$('btnAll').addEventListener('click', async () => {
    const q = ($('filter').value || '').trim().toLowerCase();
    const shown = P.pageVideos.filter(v => !q || (v.title || v.url).toLowerCase().includes(q));
    const all = !shown.every(v => v.selected);
    for (const v of shown) {
        await page({type: 'toggle', key: v.key, on: all});
        v.selected = all;
    }
    await refreshSelection();
});

$('btnNone').addEventListener('click', async () => {
    await page({type: 'unselect-all'});
    for (const v of P.allVideos) v.selected = false;
    P.picked = [];
    renderPage();
});

$('acts').addEventListener('click', async e => {
    const b = e.target.closest('button');
    if (!b) return;
    const picked = P.picked;
    if (b.id === 'btnAnalyze') {
        $('analyzeBox').hidden = false;
        $('task').focus();
    } else if (b.id === 'btnOptions' || (b.dataset.mode && P.askFirst)) {
        openForm({mode: b.dataset.mode || 'summarize', urls: picked.map(v => v.url),
                  titles: picked.map(v => v.title || ''), selection: true});
    } else if (b.dataset.mode) {
        // Through the page: its bar shows them, its selection empties
        showError('');
        const d = await page({type: 'send', mode: b.dataset.mode, task: ''});
        if (d.error || !d.batch) showError(d.reason || 'Not sent: ' + (d.error || 'no answer'));
        await loadTab();
        loadRecent();
    }
});

$('btnBack').addEventListener('click', () => { $('analyzeBox').hidden = true; });
$('btnStart').addEventListener('click', async () => {
    const task = $('task').value.trim();
    if (!task) { showError('Write the task first.'); return; }
    const d = await page({type: 'send', mode: 'analyze', task});
    if (d.error || !d.batch) { showError(d.reason || 'Not sent: ' + (d.error || 'no answer')); return; }
    $('analyzeBox').hidden = true;
    $('task').value = '';
    await loadTab();
    showTab('sent');
});

// ── What was sent ────────────────────────────────────────────────────────

async function loadRecent() {
    const d = await ask({type: 'recent'});
    const rows = d.batches || [];
    await Promise.all(rows.filter(r => r.last && r.last.status === 'working').slice(0, 5).map(async r => {
        const x = await ask({type: 'batch', id: r.id});
        if (x.batch) r.last = x.batch;
    }));
    renderRecent(rows);
    clearTimeout(P.recentTimer);
    if (rows.some(r => r.last && r.last.status === 'working') && tabNow() === 'sent') {
        P.recentTimer = setTimeout(loadRecent, 3000);
    }
}

const isWorking = r => !!(r.last && r.last.status === 'working');

function renderRecent(rows) {
    const box = $('recent');
    const working = rows.filter(isWorking).length;
    $('btnClearDone').hidden = !working || working === rows.length;
    $('btnClearAll').hidden = !rows.length;
    if (!P.clearAsked) {
        $('btnClearAll').textContent = 'clear all';
        $('btnClearAll').title = working
            ? `Empty this list — ${working === 1 ? 'the batch still working is' : `the ${working} batches still working are`}`
              + ' taken out: what isn’t fetched or transcribed yet won’t be'
            : 'Empty this list — the transcripts stay';
    }
    if (!rows.length) {
        box.innerHTML = placeholder('Nothing sent yet.');
        return;
    }
    box.innerHTML = rows.slice(0, 25).map(r => {
        const b = r.last || {};
        const what = {transcribe: 'Transcribe', summarize: 'Summarize', analyze: 'Analyze'}[b.mode] || b.mode;
        const items = (b.items || []).map((it, i) => {
            const st = STATE[it.state] || [it.state, ''];
            const key = b.id + ':' + i;
            const open = P.open.has(key);
            const pct = it.state === 'downloading' && it.progress ? ` ${it.progress} %` : '';
            // Waiting for its turn: taken out, it isn't done at all
            const stops = ['waiting', 'resolving', 'downloading', 'queued'].includes(it.state);
            return `<div class="it"><span class="t" data-url="${esc(it.url || '')}" data-title="${esc(it.title || '')}"
                        title="Show it here — ${esc(it.title || it.url)}">${esc(it.title || it.url)}</span>
                <span class="chip ${st[1]}" title="${esc(it.error || '')}">${esc(st[0] + pct)}</span>
                ${it.summary ? `<button class="l" data-sum="${esc(key)}">${open ? 'hide' : 'summary'}</button>` : ''}
                <button class="l x" data-drop="${esc(b.id)}" data-url="${esc(it.url || '')}"
                        title="${stops ? 'Take it out — it won’t be fetched or transcribed' : 'Remove from this list — the transcript stays'}">✕</button></div>
                ${it.summary && open ? `<div class="md" data-url="${esc(it.url || '')}" style="margin:2px 0 8px">${md(it.summary)}</div>` : ''}`;
        }).join('');
        const a = b.analysis;
        const an = a ? `<div class="it"><span class="note" style="flex:1">Analysis: ${esc(a.title || '')}</span>
            <span class="chip ${a.status === 'done' ? 'ok' : a.status === 'error' ? 'err' : 'run'}">${esc(
                {waiting: 'waiting for transcripts', queued: 'in the LLM queue', running: (a.progress || 0) + ' %',
                 done: 'done', error: 'failed', cancelled: 'cancelled'}[a.status] || a.status)}</span>
            <button class="l" data-path="${esc(a.open || '')}">open ↗</button></div>` : '';
        return `<div class="batch"><div class="bh"><b>${esc(what)}</b> · ${(b.items || []).length} video${(b.items || []).length === 1 ? '' : 's'} · ${esc(when(r.at))}
                <span class="gap"></span>${b.status === 'working' ? '<span class="chip run">working</span>' : ''}
                <button class="l" data-forget="${esc(b.id)}" title="${b.status === 'working'
                    ? 'Take them all out — what isn’t fetched or transcribed yet won’t be'
                    : 'Remove from this list — the transcripts stay'}">✕</button></div>
            ${items}${an}${b.error ? `<div class="err">${esc(b.error)}</div>` : ''}</div>`;
    }).join('');
}

$('recent').addEventListener('click', async e => {
    const x = e.target.closest('button[data-drop]');
    if (x) {
        x.disabled = true;
        await ask({type: 'drop', id: x.dataset.drop, urls: [x.dataset.url]});
        loadRecent();
        return;
    }
    const t = e.target.closest('.t[data-url]');
    if (t) {
        setVideo({url: t.dataset.url, title: t.dataset.title});
        showTab('summary');
        return;
    }
    const el = e.target.closest('button');
    if (!el) return;
    if (el.dataset.sum) {
        if (P.open.has(el.dataset.sum)) P.open.delete(el.dataset.sum); else P.open.add(el.dataset.sum);
        renderRecent((await ask({type: 'recent'})).batches || []);
    } else if (el.dataset.path) {
        ask({type: 'open', path: el.dataset.path});
    } else if (el.dataset.forget) {
        // Still working: taken out on the server too, not only off the list
        const rows = (await ask({type: 'recent'})).batches || [];
        const row = rows.find(r => r.id === el.dataset.forget);
        await ask(row && isWorking(row) ? {type: 'drop', id: row.id} : {type: 'forget', id: el.dataset.forget});
        loadRecent();
    }
});

$('btnClearDone').addEventListener('click', async () => {
    await ask({type: 'clear-sent', all: false});
    loadRecent();
});

/* "clear all" with batches still working asks once more, on the button
 * itself — a second click within a few seconds takes them out. */
$('btnClearAll').addEventListener('click', async () => {
    const rows = (await ask({type: 'recent'})).batches || [];
    const working = rows.filter(isWorking).length;
    if (working && !P.clearAsked) {
        P.clearAsked = setTimeout(() => { P.clearAsked = 0; loadRecent(); }, 4000);
        $('btnClearAll').textContent = `stop ${working} working and clear?`;
        return;
    }
    clearTimeout(P.clearAsked);
    P.clearAsked = 0;
    await ask({type: 'clear-sent', all: true});
    loadRecent();
});

$('btnApp').addEventListener('click', () => ask({type: 'open', path: '/'}));

// ══ What the page and the popup ask the panel to show ═══════════════════

async function applyIntent(intent) {
    if (!intent || Date.now() - (intent.at || 0) > 60000) return;
    chrome.storage.session.remove('panelIntent');
    if (intent.url) await setVideo({url: intent.url, title: intent.title || ''});
    if (intent.form) {
        const urls = intent.urls || (intent.url ? [intent.url] : []);
        // urls: the page's selection (its bar, the popup's ticks) — a single url: one video
        if (urls.length) openForm({mode: intent.form, urls, titles: intent.titles || [intent.title || ''],
                                   selection: !!intent.urls});
    }
    if (intent.tab) showTab(intent.tab);
}

chrome.storage.onChanged.addListener((changes, area) => {
    if (area === 'session' && changes.panelIntent && changes.panelIntent.newValue) {
        applyIntent(changes.panelIntent.newValue);
    }
    if (area === 'sync' && changes.askFirst) {
        P.askFirst = !!changes.askFirst.newValue;
        $('askFirst').checked = P.askFirst;
    }
    if (area === 'sync' && changes.marksEverywhere) {
        $('marksEverywhere').checked = !!changes.marksEverywhere.newValue;
    }
});

(async () => {
    const {askFirst, marksEverywhere} = await chrome.storage.sync.get(
        ['askFirst', 'marksEverywhere']).catch(() => ({}));
    P.askFirst = !!askFirst;
    $('askFirst').checked = P.askFirst;
    $('marksEverywhere').checked = !!marksEverywhere;
    renderVideo();
    renderPanes();
    if (await health()) loadOptions();
    await loadTab();
    const {panelIntent} = await chrome.storage.session.get('panelIntent').catch(() => ({}));
    await applyIntent(panelIntent);
    loadRecent();
})();
