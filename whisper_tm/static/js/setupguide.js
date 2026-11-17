// ── Setup guide ───────────────────────────────────────────────────────────
// Opens by itself on a fresh installation (/api/setup/status → first_run)
// and from Help & setup (the rail), tab Setup, at any time. Seven pages:
// welcome, system check (quick checks + on-demand GPU test), speaker
// diarization (the Hugging Face token for the gated pyannote model), first
// steps, AMD (experimental, untested — says so on every surface), internal
// docs (paths, ports, commands, the Markdown files from docs/) and the API
// reference. Drawn in the settings' shell (settings.css): navigation on
// the left, a page head and cards on the right.

const SG_PAGES = [
    { id: 'welcome', icon: 'home', label: 'Welcome',
      intro: 'What this app does, and what this guide walks you through.' },
    { id: 'check', icon: 'health', label: 'System check',
      intro: 'Is everything installed, and does the GPU really compute?' },
    { id: 'speakers', icon: 'voices', label: 'Speaker diarization',
      intro: 'Telling speakers apart needs a free model from Hugging Face — set up once, with a token.' },
    { id: 'steps', icon: 'play', label: 'First steps',
      intro: 'From the library root to the first transcript — and what can be switched on later.' },
    { id: 'amd', icon: 'cpu', label: 'AMD GPU (experimental)', exp: true,
      intro: 'What is known about AMD cards — collected from public reports, not tested here.' },
    { id: 'docs', icon: 'book', label: 'Internal docs',
      intro: 'Where the data lives, which ports are open, the commands — and the design documents.' },
    { id: 'api', icon: 'code', label: 'API reference',
      intro: 'Every HTTP endpoint of this server, generated from the running code — routes, methods, parameters and descriptions. The Markdown version is made to be handed to an LLM or a script.' },
];

function sgIcon(name, cls = '') {
    return `<svg class="ui-ico${cls ? ' ' + cls : ''}" aria-hidden="true"><use href="#i-${name}"/></svg>`;
}

// A card of the guide — the same markup as a settings card
function sgCard(title, body, sub = '') {
    return `<section class="set-card">
        <div class="set-card-head"><div class="set-card-titles"><h3>${title}</h3>
            ${sub ? `<div class="set-card-sub">${sub}</div>` : ''}</div></div>
        <div class="set-card-body sg-body">${body}</div>
    </section>`;
}

const sgState = { page: 'welcome', check: null, checkLoading: false,
                  gpu: null, gpuRunning: false, docs: null, docsLoading: false, doc: null,
                  api: null, apiLoading: false, apiQuery: '',
                  hf: null, hfLoading: false, hfBusy: false, hfResult: null };

/* The guide is the Setup tab of Help & setup (help.js moves it there). */
function openSetupGuide(page = 'welcome') {
    if (typeof openHelp === 'function' && typeof Help !== 'undefined') {
        sgState.page = page;
        openHelp('setup');
    } else {
        $('setupOverlay')?.classList.add('visible');
    }
    sgShow(page);
}

function closeSetupGuide() {
    $('setupOverlay')?.classList.remove('visible');
    if (typeof closeHelp === 'function' && typeof Help !== 'undefined' && Help.current === 'setup') closeHelp();
}

// On screen: the Setup tab of Help & setup (or the guide's own overlay)
function sgShown() {
    if ($('setupOverlay')?.classList.contains('visible')) return true;
    return typeof Help !== 'undefined' && Help.current === 'setup'
        && !!$('helpOverlay')?.classList.contains('visible');
}

// "Done" marks the guide as seen: it will not open by itself again.
async function sgFinish() {
    try { await fetch('/api/setup/done', { method: 'POST' }); } catch (e) { /* offline */ }
    closeSetupGuide();
}

function sgShow(page) {
    sgState.page = page;
    const nav = $('sgNav');
    if (nav) {
        nav.innerHTML = SG_PAGES.map(p =>
            `<button type="button" class="set-nav-item${p.id === page ? ' active' : ''}${p.exp ? ' sg-nav-exp' : ''}"
                  onclick="sgShow('${p.id}')"><span class="set-nav-icon">${sgIcon(p.icon)}</span>
                  <span class="set-nav-label">${esc(p.label)}</span></button>`
        ).join('');
        nav.querySelector('.active')?.scrollIntoView?.({block: 'nearest', inline: 'nearest'});
    }
    const info = SG_PAGES.find(p => p.id === page);
    const body = $('sgContent');
    body.scrollTop = 0;
    body.innerHTML = `<header class="set-page-head"><h2>${sgIcon(info.icon, 'set-page-icon')}${esc(info.label)}</h2>
            <p>${esc(info.intro)}</p></header>`
        + ({
            welcome: sgWelcome, check: sgCheckPage, speakers: sgSpeakers, steps: sgSteps, amd: sgAmd,
            docs: sgDocsPage, api: sgApiPage,
        })[page]();
    // Load once per opening; the loading flags stop the re-render from
    // starting the same request again.
    if (page === 'check' && !sgState.check && !sgState.checkLoading) sgLoadCheck();
    if (page === 'docs' && !sgState.docs && !sgState.docsLoading) sgLoadDocs();
    if (page === 'speakers' && !sgState.hfLoading) sgLoadHf();
    if (page === 'steps') sgLoadLanguage();
    if (page === 'api') {
        if (!sgState.api && !sgState.apiLoading) sgLoadApi(); else sgApiRenderList();
    }
}

function sgRerender() {
    if (sgShown()) {
        const top = $('sgContent').scrollTop;
        sgShow(sgState.page);
        $('sgContent').scrollTop = top;
    }
}


// ── Pages ────────────────────────────────────────────────────────────────

function sgWelcome() {
    return `<div class="sg-lead">
        Whisper Transcription Manager transcribes audio and video — from a single file to a NAS
        with 100,000+ recordings — with speakers, timestamps, search and an
        optional language model for summaries, correction and translation.
    </div>
    ${sgCard('This guide', `<ol class="sg-list">
            <li><b>System check</b> — is everything installed, does the GPU really compute?</li>
            <li><b>Speaker diarization</b> — the Hugging Face token that unlocks telling speakers apart.</li>
            <li><b>First steps</b> — media folder, first transcription, optional features.</li>
            <li><b>AMD GPU</b> — what is known about AMD cards. <span class="sg-exp-tag">experimental</span></li>
            <li><b>Internal docs</b> — where data lives, ports, commands, design documents.</li>
            <li><b>API reference</b> — every HTTP endpoint, as a page and as Markdown for scripts and LLMs.</li>
        </ol>
        <p class="sg-note">You can reopen this guide any time: ${tbIcon('help')} <b>Help &amp; setup</b> on the left, tab ${tbIcon('guide')} <b>Setup</b>.
        The same check runs on the command line: <code>${esc(sgCmd('check'))} --gpu-test</code></p>`)}
    <div class="sg-actions">
        <button type="button" class="set-btn primary" onclick="sgShow('check')">Start the system check →</button>
        <button type="button" class="set-btn" onclick="sgFinish()">Skip — don't open automatically again</button>
    </div>`;
}

const SG_ICON = { ok: '✓', info: '·', warn: '⚠', fail: '✗' };

function sgCheckPage() {
    const c = sgState.check;
    let html = `<div class="sg-actions" style="margin-top:0">
        <button type="button" class="set-btn" onclick="sgLoadCheck()">↻ Check again</button>
        <button type="button" class="set-btn primary" id="sgGpuBtn" onclick="sgRunGpuTest()" ${sgState.gpuRunning ? 'disabled' : ''}>
            ${sgState.gpuRunning ? '⏳ Testing on the GPU …' : '⚡ Test the GPU for real'}</button>
        <span class="sg-note" style="margin:0">computes on the card and runs Whisper <i>tiny</i> on 5 s of audio
        (downloads ~75 MB once) — not while a transcription runs</span>
    </div>`;
    html += sgGpuResult();
    if (!c) return html + `<div class="sg-note">Checking …</div>`;
    if (c.error) return html + `<div class="sg-box fail">${esc(c.error)}</div>`;
    const verdict = { ok: ['ok', 'Everything needed is in place.'],
                      warn: ['warn', 'Works, with notes below.'],
                      fail: ['fail', 'Something essential is missing — see ✗ below.'] }[c.overall];
    html += `<div class="sg-box ${verdict[0]}">${SG_ICON[verdict[0]]} ${verdict[1]}</div>`;
    // One card per group, in the order the checks come
    const groups = [];
    for (const k of c.checks) {
        if (!groups.length || groups[groups.length - 1].name !== k.group) {
            groups.push({name: k.group, experimental: false, rows: []});
        }
        const g = groups[groups.length - 1];
        g.experimental = g.experimental || !!k.experimental;
        g.rows.push(`<div class="sg-row ${k.status}">
            <span class="sg-ico">${SG_ICON[k.status]}</span>
            <span class="sg-row-title">${esc(k.title)}</span>
            <span class="sg-row-detail">${esc(k.detail)}${k.hint && k.status !== 'ok'
                ? `<span class="sg-hint">→ ${esc(k.hint)}</span>` : ''}</span>
        </div>`);
    }
    for (const g of groups) {
        const exp = g.experimental ? ` <span class="sg-exp-tag">experimental · untested</span>` : '';
        html += sgCard(esc(g.name) + exp, g.rows.join(''));
    }
    return html;
}

function sgGpuResult() {
    const g = sgState.gpu;
    if (!g) return '';
    let html = '';
    if (g.error) html += `<div class="sg-box fail">✗ ${esc(g.error)}</div>`;
    for (const s of g.steps || []) {
        html += `<div class="sg-row ${s.status}"><span class="sg-ico">${SG_ICON[s.status]}</span>
            <span class="sg-row-title">${esc(s.step)}</span>
            <span class="sg-row-detail">${esc(s.detail)} <span class="sg-dim">(${s.secs} s)</span></span></div>`;
    }
    if ((g.steps || []).length) {
        html += g.ok ? `<div class="sg-box ok" style="margin-top:12px">✓ GPU transcription works.</div>`
                     : `<div class="sg-box fail" style="margin-top:12px">✗ GPU transcription does not work — the first ✗ is the cause.</div>`;
    }
    return sgCard('GPU test', html);
}

// The setup and start commands (/api/setup/status) — these until they are
// known
function sgCmd(key) {
    const linux = { setup_llama: './setup.sh --llama', start: './start.sh', start_llm: './start-llm.sh',
                    check: './start.sh --check' };
    return (sgState.commands || {})[key] || linux[key];
}

// In the Docker image: no setup or start scripts, no llama.cpp
function sgDocker() {
    return !!(sgState.commands || {}).docker;
}

// The command-line table: the start script's options — in Docker, the same
// options for a second container of the image
function sgCommands() {
    const docker = sgDocker();
    const run = docker ? 'docker compose run --rm whisper-tm' : sgCmd('start');
    const rows = [
        [docker ? esc(sgCmd('start')) : `make run</code> / <code>${esc(sgCmd('start'))}`, 'start the server'],
        [`${esc(run)} --check [--gpu-test]`, 'this system check on the command line'],
        docker ? ['docker compose logs -f', "the server's log"]
               : [`${esc(run)} --debug`, 'log everything, measure all timings, one log file per start'],
        [`${esc(run)} --show-roots`, 'media roots stored in the database and whether they exist here'],
        [`${esc(run)} --remap-path OLD NEW [--apply]`, 'after moving to another machine: translate stored paths'],
        [docker ? 'docker compose exec whisper-tm python -m whisper_tm.mcp_server --check'
                : './.venv/bin/python -m whisper_tm.mcp_server --check', 'can the MCP server reach the app?'],
    ];
    return `<table class="sg-table">${rows.map(([cmd, what]) =>
        `<tr><td><code>${cmd}</code></td><td>${what}</td></tr>`).join('')}</table>`;
}

async function sgLoadCheck() {
    if (sgState.checkLoading) return;
    sgState.checkLoading = true;
    sgState.check = null;
    sgRerender();
    try {
        const r = await fetch('/api/setup/check');
        sgState.check = await r.json();
    } catch (e) {
        sgState.check = { error: 'Check failed: ' + (e.message || e) };
    } finally {
        sgState.checkLoading = false;
    }
    sgRerender();
}

async function sgRunGpuTest() {
    sgState.gpuRunning = true;
    sgState.gpu = null;
    sgRerender();
    try {
        const r = await fetch('/api/setup/gpu_test', { method: 'POST' });
        sgState.gpu = await r.json();
    } catch (e) {
        sgState.gpu = { ok: false, error: 'Test failed: ' + (e.message || e) };
    } finally {
        sgState.gpuRunning = false;
        sgRerender();
    }
}

// ── Speaker diarization: the Hugging Face token ───────────────────────────
// The pyannote model is gated: an account accepts its conditions once, a
// token of that account downloads it, then the cache is enough. The check
// goes through the server (/api/setup/hf_token), which saves a token that
// works and downloads the model; the token never comes back.

const SG_HF_DEFAULTS = {
    repo: 'pyannote/speaker-diarization-community-1',
    repo_url: 'https://huggingface.co/pyannote/speaker-diarization-community-1',
    tokens_url: 'https://huggingface.co/settings/tokens',
    join_url: 'https://huggingface.co/join', model_mb: 32,
};

function sgSpeakers() {
    const h = { ...SG_HF_DEFAULTS, ...(sgState.hf || {}) };
    const link = (url, text) => `<a href="${esc(url)}" target="_blank" rel="noopener">${text}</a>`;
    return sgCard('Why a token', `<p>Who speaks when — the diarization — comes from pyannote's free model
            ${link(h.repo_url, esc(h.repo))}. Hugging Face hands it out only to signed-in
            accounts that accepted its conditions. With a token of such an account the app downloads it once
            (~${esc(String(h.model_mb))} MB); after that it loads from the cache, without the token.</p>
            <p class="sg-note">Without it everything else works — transcripts just have no speakers, and no voices are learned.</p>`)
    + `<div id="sgHfStatus">${sgHfStatusBox()}</div>`
    + sgCard('Set it up — once', `<ol class="sg-list">
            <li>Sign in at ${link('https://huggingface.co', 'huggingface.co')} — or ${link(h.join_url, 'create a free account')}.</li>
            <li>Open ${link(h.repo_url, esc(h.repo))} and accept the conditions. The form asks for a company or
                university and a use case; access is granted at once.</li>
            <li>Create an access token of type <b>Read</b> at ${link(h.tokens_url, 'huggingface.co/settings/tokens')}.
                A fine-grained token needs “Read access to contents of all public gated repos you can access”.</li>
            <li>Paste it here and check — a token that works is saved, and the model is downloaded right away:</li>
        </ol>
        <div class="set-token" style="margin-top:10px">
            <input type="password" id="sgHfToken" class="set-input mono" placeholder="hf_…" autocomplete="off"
                   spellcheck="false" aria-label="Hugging Face token"
                   onkeydown="if (event.key === 'Enter') sgHfCheck()">
            <button type="button" class="set-btn primary" id="sgHfBtn" onclick="sgHfCheck()">Check and save</button>
        </div>
        <div id="sgHfResult">${sgHfResultBox()}</div>
        <p class="sg-note">An empty field checks the token that is set already. The same on the command line:
            <code>whisper-tm --hf-token</code>. The token is kept in the config file (${tbIcon('settings')} Settings → Speakers).</p>`);
}

function sgHfStatusBox() {
    const h = sgState.hf;
    if (!h) return `<p class="sg-note">Looking at the token and the model …</p>`;
    if (h.error) return `<div class="sg-box fail">✗ ${esc(h.error)}</div>`;
    if (h.model_cached) {
        return `<div class="sg-box ok">✓ The model is downloaded — speakers can be told apart${
            h.token ? '' : ' (no token needed any more)'}.</div>`;
    }
    if (h.token) {
        return `<div class="sg-box ok">✓ A token is set${h.token === 'env' ? ' (environment variable HF_TOKEN)' : ''}
            — the model downloads with the first file with speakers. “Check and save” tests it now.</div>`;
    }
    return `<div class="sg-box warn">⚠ Not set up yet — transcripts get no speakers until it is.</div>`;
}

function sgHfResultBox() {
    const r = sgState.hfResult;
    if (sgState.hfBusy) {
        return `<p class="sg-note">⏳ Asking Hugging Face … (with access the model is downloaded too)</p>`;
    }
    if (!r) return '';
    if (r.error) return `<div class="sg-box fail" style="margin-top:12px">✗ ${esc(r.error)}</div>`;
    const cls = { ok: 'ok', gated: 'warn', offline: 'warn', no_token: 'warn' }[r.status] || 'fail';
    const ico = { ok: '✓', warn: '⚠', fail: '✗' }[cls];
    // The server's message names the page to go to — clickable, without the
    // sentence's full stop
    let text = esc(r.message || '').replace(/https:\/\/[^\s<"]*[^\s<".,)]/g,
        u => `<a href="${u}" target="_blank" rel="noopener">${u}</a>`);
    if (r.user) text += ` <span class="sg-dim">(account ${esc(r.user)})</span>`;
    if (r.saved) text += ' Token saved.';
    if (r.save_error) text += ' ' + esc(r.save_error);
    if (r.status === 'ok') {
        text += r.model_cached ? ' The model is downloaded — speakers can be told apart.'
                               : ' ' + esc(r.download_error || '');
    }
    if (r.status === 'no_token') text = 'Paste a token first.';
    return `<div class="sg-box ${cls}" style="margin-top:12px">${ico} ${text}</div>`;
}

async function sgLoadHf() {
    sgState.hfLoading = true;
    try {
        const r = await fetch('/api/setup/hf_token');
        sgState.hf = r.ok ? await r.json() : { error: `The server answered ${r.status}.` };
    } catch (e) {
        sgState.hf = { error: 'Could not ask the server: ' + (e.message || e) };
    } finally {
        sgState.hfLoading = false;
    }
    // Only the status box: a full redraw would empty the token field
    const box = $('sgHfStatus');
    if (box) box.innerHTML = sgHfStatusBox();
}

async function sgHfCheck() {
    if (sgState.hfBusy) return;
    const input = $('sgHfToken'), btn = $('sgHfBtn'), out = $('sgHfResult');
    sgState.hfBusy = true;
    if (btn) btn.disabled = true;
    if (out) out.innerHTML = sgHfResultBox();
    try {
        const r = await fetch('/api/setup/hf_token', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ token: (input?.value || '').trim() }),
        });
        sgState.hfResult = await r.json();
        if (sgState.hfResult.saved && input) input.value = '';
    } catch (e) {
        sgState.hfResult = { error: 'The check failed: ' + (e.message || e) };
    } finally {
        sgState.hfBusy = false;
        if (btn) btn.disabled = false;
        if ($('sgHfResult')) $('sgHfResult').innerHTML = sgHfResultBox();
    }
    sgLoadHf();
}

function sgSteps() {
    return sgCard('1 · Library root', `<p>The library lives under <b>one</b> folder, the library root: a local folder or a
        mounted NAS share. Every recording sits somewhere below it, at any depth, and the file list, the queue,
        the statistics and the search all scale from it. A folder outside the root is not part of the library,
        so choose the top folder that holds all your recordings, not one of its subfolders.</p>
        <p><code>./setup.sh</code> asks for it (or <code>--library PATH</code>); later it is the path field at the
        top (📂), and <code>directories.media</code> with <code>security.allowed_roots</code> in config.yaml.
        Moving it to another machine or mount: <code>--remap-path</code>.</p>`)
    + sgCard('2 · Work folder', `<p>Temporary audio, separated vocals and chunks go to the work folder. A local SSD makes long files
        much faster: <b>${tbIcon('settings')} Settings → Folders</b>.</p>`)
    + sgCard('3 · Language', `<p>Whisper detects the language of every recording by itself. If your library is in one
        language only, setting it skips the detection, which can be fooled by a quiet or music-heavy start.</p>
        <div class="set-row"><label for="sgLanguage">Language of the recordings</label>
            <select id="sgLanguage" class="set-input" onchange="sgSetLanguage(this.value)">${sgLanguageOptions()}</select>
            <span class="set-hint" id="sgLanguageSaved"></span></div>
        <p class="sg-note">Each job can still choose its own; later: ${tbIcon('settings')} Settings → Transcription.</p>`)
    + sgCard('4 · First transcription', `<p>Hover a file and click <b>+Q</b> — or, for a whole folder, <b>▶ Transcribe … open</b> at the
        bottom of the sidebar, or <b>${tbIcon('queue')} Queue</b> in the rail (tab <i>From the library</i>).
        The queue runs on the server — the browser can be closed.</p>`)
    + sgCard('5 · Optional', `<ul class="sg-list">
            <li><b>Speaker diarization</b> — needs a free Hugging Face token once:
                <a href="#" onclick="sgShow('speakers');return false">set it up</a>.</li>
            <li><b>Language model</b> — summaries, correction, translation: ${sgDocker()
                ? `the Docker image has no llama.cpp — run Ollama or a llama.cpp server on the host (listening on more than
                   127.0.0.1) and add it under Settings → AI models as <code>${esc((sgState.commands || {}).host_url || '')}</code>,
                   or add OpenAI/LM Studio there.`
                : `<code>${esc(sgCmd('setup_llama'))}</code> installs llama.cpp with a default model, or add
                   Ollama/OpenAI/LM Studio under Settings → AI models.`}</li>
            <li><b>Semantic search</b> — set up an embedding model (Settings → AI models), then build the index
                under Settings → Search → Semantic search.</li>
            <li><b>AI assistants</b> — Settings → MCP server lets Claude, Open WebUI and other clients use the library.</li>
            <li><b>Subtitles next to the videos</b> — tick "SRT alongside the video file" in the transcription dialog.</li>
        </ul>
        <p class="sg-note">More usage scenarios: ${tbIcon('help')} <b>Help &amp; setup</b> on the left, tab <b>How-tos</b>.</p>`)
    + `<div class="sg-actions"><button type="button" class="set-btn primary" onclick="sgFinish()">✓ Done — start working</button></div>`;
}

/* The language choices of Settings → Transcription, the saved one selected */
function sgLanguageOptions() {
    const langs = typeof _S_LANGUAGES !== 'undefined' ? _S_LANGUAGES : [['auto', 'Detect automatically']];
    const now = sgState.language || 'auto';
    return langs.map(([v, label]) => `<option value="${esc(v)}"${v === now ? ' selected' : ''}>${esc(label)}</option>`).join('');
}

async function sgLoadLanguage() {
    try {
        const d = await fetch('/api/settings').then(r => r.json());
        sgState.language = d?.config?.transcription?.default_language || 'auto';
        const sel = document.getElementById('sgLanguage');
        if (sel) sel.value = sgState.language;
    } catch (e) { /* the select keeps "Detect automatically" */ }
}

async function sgSetLanguage(value) {
    const note = document.getElementById('sgLanguageSaved');
    try {
        await _sPutSettings({transcription: {default_language: value}});
        sgState.language = value;
        if (typeof _applySettingsDefaults === 'function') _applySettingsDefaults();
        if (note) note.textContent = 'Saved';
    } catch (e) {
        if (note) note.textContent = 'Not saved: ' + e.message;
    }
}

function sgAmd() {
    return `<div class="sg-box warn sg-exp-banner">
        <b>🧪 Experimental — untested.</b> This project has <b>never transcribed on an AMD GPU</b>.
        Everything on this page is based on public reports as of September 2026. Treat it as a map, not as instructions that are known to work.
    </div>
    ${sgCard('What would run on AMD', `<table class="sg-table">
            <tr><th>Part</th><th>On AMD</th><th>Why</th></tr>
            <tr><td>Transcription (faster-whisper / CTranslate2)</td><td class="warn">⚠ build it yourself</td>
                <td>CTranslate2 supports AMD via ROCm/HIP since 4.7.0 (Feb 2025), but the pip package is NVIDIA-only.
                On Linux it has to be compiled with <code>-DWITH_HIP=ON</code> (reported: 20–40 min).</td></tr>
            <tr><td>Alignment, diarization, speaker vectors</td><td class="warn">⚠ likely</td>
                <td>Plain PyTorch — needs the ROCm build of PyTorch.</td></tr>
            <tr><td>Vocal isolation (Roformer, default)</td><td class="warn">⚠ likely</td><td>PyTorch.</td></tr>
            <tr><td>Vocal isolation (older .onnx MDX models)</td><td class="fail">✗ CPU only</td>
                <td>onnxruntime-gpu is built for CUDA.</td></tr>
            <tr><td>Language model (llama.cpp)</td><td class="ok">✓ works</td>
                <td>The setup downloads the Vulkan build, which runs on AMD.</td></tr>
            <tr><td>VRAM display, waiting for free VRAM, the figures of ${tbIcon('eject')} Unload VRAM</td><td class="fail">✗ missing</td>
                <td>The project reads VRAM through NVIDIA tools only.</td></tr>
        </table>`)}
    ${sgCard('Is my card supported?', `<p>ROCm officially lists these Radeon cards (Linux, ROCm 7.14): RX 7900 XTX/XT/GRE (gfx1100),
        RX 7800 XT / 7700 XT / 7700 (gfx1101), RX 9060 series (gfx1200), RX 9070 series (gfx1201).</p>
        <p>Other RDNA3 cards such as the <b>RX 7600 / 7600 XT (gfx1102)</b> are not on the list. They often run
        with <code>HSA_OVERRIDE_GFX_VERSION=11.0.0</code>, but there are reports of that breaking after driver
        updates, and official PyTorch ROCm wheels dropped gfx1102 at some point. Only a test on the real card
        answers it — the system check shows your gfx target once ROCm is installed.</p>`)}
    ${sgCard('Trying it — in a separate environment', `<ol class="sg-list">
            <li><b>Install ROCm</b> from your distribution or AMD (rocm.docs.amd.com). <code>rocminfo</code> must list
                your card's gfx target.</li>
            <li><b>A separate venv</b> — not the one this server runs from, so a failed attempt breaks nothing.</li>
            <li><b>PyTorch for ROCm</b> — pick the wheel matching your ROCm version on pytorch.org. The PyPI default
                wheel is CUDA; a report notes that a later <code>pip install</code> can silently swap the ROCm build
                back, so pin it with a constraints file.</li>
            <li><b>CTranslate2 with HIP</b> — build from source with <code>-DWITH_HIP=ON</code> (see the CTranslate2
                installation docs and the WhisperX ROCm guide below).</li>
            <li><b>Test</b> — in that venv run <code>python -m whisper_tm --check --gpu-test</code>. All three steps
                must show ✓; a common trap is that the card is detected but Whisper silently runs on the CPU —
                the test catches that.</li>
        </ol>
        <p class="sg-note"><code>./setup.sh --gpu amd</code> installs PyTorch's ROCm 6.4 build of the version WhisperX
        pins — untested, like everything on this page.</p>`)}
    ${sgCard('Sources', `<ul class="sg-list sg-links">
            <li><a href="https://github.com/OpenNMT/CTranslate2/releases" target="_blank" rel="noopener">CTranslate2 releases — v4.7.0 "Introduce AMD GPU support with ROCm HIP"</a></li>
            <li><a href="https://opennmt.net/CTranslate2/installation.html" target="_blank" rel="noopener">CTranslate2 installation (-DWITH_HIP=ON)</a></li>
            <li><a href="https://github.com/m-bain/whisperX/discussions/1364" target="_blank" rel="noopener">WhisperX on AMD GPU (ROCm 7.2) — installation guide, tested on RX 7900 XTX</a></li>
            <li><a href="https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html" target="_blank" rel="noopener">ROCm system requirements (supported GPUs)</a></li>
            <li><a href="https://github.com/ROCm/ROCm/issues/5555" target="_blank" rel="noopener">ROCm issue #5555 — RX 7600: override stopped working after updates</a></li>
        </ul>`)}`;
}

function sgDocsPage() {
    const c = sgState.check;
    const p = c?.paths || {}, ports = c?.ports || {};
    const row = (k, v) => `<tr><td>${esc(k)}</td><td><code>${esc(v || '—')}</code></td></tr>`;
    let html = sgCard('Where things live', `<table class="sg-table">
            ${row('Config file', p.config)}
            ${row('Database', p.database)}
            ${row('Media folder', p.media)}
            ${row('Work folder (temporary)', p.work)}
            ${row('Server logs (also llama.cpp, media service)', p.logs_debug)}
            ${row('Job logs', p.logs_jobs)}
            ${row('Design documents', p.docs)}
        </table>
        ${c ? '' : `<p class="sg-note">Paths load with the system check — <a href="#" onclick="sgLoadCheck();return false">load now</a>.</p>`}`)
    + sgCard('Processes and ports', `<table class="sg-table">
            ${row('Web interface + API', ports.web)}
            ${row('MCP over HTTP (mcp.autostart_http)', ports.mcp_http)}
            ${row('MCP OpenAPI (mcp.autostart_openapi)', ports.mcp_openapi)}
            ${row(sgDocker() ? 'llama.cpp language model (outside the Docker image)'
                             : `llama.cpp language model (${sgCmd('start_llm')})`, ports.llama)}
            ${row('Local embedding server', ports.embedding)}
        </table>`)
    + sgCard('Commands', sgCommands());
    let docs = '';
    if (!sgState.docs) docs += `<p class="sg-note">Loading …</p>`;
    else if (!sgState.docs.length) docs += `<p class="sg-note">No documents found (docs/ is not part of this installation).</p>`;
    else docs += `<div class="sg-doc-tabs">` + sgState.docs.map(d =>
        `<button type="button" class="set-btn sm${sgState.doc?.name === d.name ? ' primary' : ''}" onclick="sgOpenDoc('${escJs(d.name)}')">${esc(d.title)}</button>`
    ).join('') + `</div>`;
    if (sgState.doc) docs += `<div class="sg-md">${sgMarkdown(sgState.doc.text)}</div>`;
    return html + sgCard('Design documents (docs/)', docs);
}

async function sgLoadDocs() {
    if (sgState.docsLoading) return;
    sgState.docsLoading = true;
    try {
        const d = await (await fetch('/api/setup/docs')).json();
        sgState.docs = d.docs || [];
    } catch (e) {
        sgState.docs = [];
    } finally {
        sgState.docsLoading = false;
    }
    if (!sgState.check) { sgLoadCheck(); return; }   // paths/ports for the tables above
    sgRerender();
}

async function sgOpenDoc(name) {
    if (sgState.doc?.name === name) { sgState.doc = null; sgRerender(); return; }
    try {
        sgState.doc = await (await fetch('/api/setup/docs/' + encodeURIComponent(name))).json();
    } catch (e) {
        Toast.error('Could not load ' + name);
    }
    sgRerender();
}

// ── API reference (explorer over /api/docs) ────────────────────────────

function sgApiPage() {
    return `<div class="sg-actions" style="margin-top:6px">
        <a class="set-btn primary" href="/api/docs.md?download=1" download>⬇ Download Markdown</a>
        <a class="set-btn" href="/api/docs.md?download=1&brief=1" download
           title="First paragraph of each description only — a fraction of the size">⬇ Compact version</a>
        <a class="set-btn" href="/api/docs.md" target="_blank" rel="noopener">Open as text ↗</a>
    </div>
    <input type="search" id="sgApiSearch" class="sg-api-search" placeholder="Filter endpoints… (path, method, word)"
           value="${esc(sgState.apiQuery)}" oninput="sgState.apiQuery=this.value;sgApiRenderList()">
    <div id="sgApiList"><p class="sg-note">Loading …</p></div>`;
}

async function sgLoadApi() {
    sgState.apiLoading = true;
    try {
        sgState.api = await (await fetch('/api/docs')).json();
    } catch (e) {
        sgState.api = { groups: [], error: String(e.message || e) };
    } finally {
        sgState.apiLoading = false;
    }
    if (sgState.page === 'api') sgApiRenderList();
}

// Only the list re-renders while typing — the search field keeps its focus.
function sgApiRenderList() {
    const box = $('sgApiList');
    const a = sgState.api;
    if (!box || !a) return;
    if (a.error) { box.innerHTML = `<div class="sg-box fail">${esc(a.error)}</div>`; return; }
    const q = sgState.apiQuery.trim().toLowerCase();
    let shown = 0;
    const html = a.groups.map(g => {
        const eps = g.endpoints.filter(e => !q
            || `${e.methods.join(' ')} ${e.path} ${e.doc}`.toLowerCase().includes(q));
        if (!eps.length) return '';
        shown += eps.length;
        return sgCard(`${esc(g.name)} <span class="sg-dim">· ${eps.length}</span>`,
            eps.map(sgApiEndpoint).join(''), g.intro && !q ? esc(g.intro) : '');
    }).join('');
    box.innerHTML = `<p class="sg-note">${shown} of ${a.count} endpoints · generated ${esc(a.generated || '')}</p>`
        + (html || '<p class="sg-note">No endpoint matches.</p>');
}

function sgApiEndpoint(e) {
    const params = [];
    const codes = list => list.map(x => `<code>${esc(x)}</code>`).join(', ');
    if (e.path_params.length) params.push('Path: ' + codes(e.path_params.map(p => p.name)));
    if (e.query_params.length) params.push('Query: ' + codes(e.query_params));
    if (e.body_fields.length) params.push('Body: ' + codes(e.body_fields));
    const first = (e.doc || '').split(/\n\s*\n/)[0].replace(/\s+/g, ' ');
    // A plain GET without path parameters can be opened directly.
    const open = e.methods.includes('GET') && !e.path_params.length && !e.stream
        ? `<a class="sg-api-open" href="${esc(e.path)}" target="_blank" rel="noopener"
              onclick="event.stopPropagation()">open ↗</a>` : '';
    return `<details class="sg-api-ep">
        <summary>
            ${e.methods.map(m => `<span class="sg-api-m ${m.toLowerCase()}">${m}</span>`).join('')}
            <code class="sg-api-path">${esc(e.path)}</code>
            ${e.stream ? '<span class="sg-api-m">stream</span>' : ''}
            ${open}
            <span class="sg-api-first">${esc(first)}</span>
        </summary>
        <div class="sg-api-body">
            ${params.length ? `<div class="sg-api-params">${params.join(' · ')}</div>` : ''}
            <pre class="sg-api-doc">${esc(e.doc || 'No description.').replace(/\x60\x60([^\x60]+)\x60\x60/g, '<code>$1</code>')}</pre>
        </div>
    </details>`;
}

// Minimal Markdown → HTML for the design documents: headings, code blocks,
// inline code, bold, lists, paragraphs. Everything is escaped first, so the
// documents cannot inject markup.
function sgMarkdown(src) {
    const out = [];
    let inCode = false, list = false, para = [];
    const inline = s => esc(s)
        .replace(/`([^`]+)`/g, '<code>$1</code>')
        .replace(/\*\*([^*]+)\*\*/g, '<b>$1</b>');
    const flush = () => {
        if (para.length) { out.push(`<p>${inline(para.join(' '))}</p>`); para = []; }
        if (list) { out.push('</ul>'); list = false; }
    };
    for (const line of (src || '').split('\n')) {
        if (line.startsWith('```')) {
            if (inCode) { out.push('</code></pre>'); inCode = false; }
            else { flush(); out.push('<pre><code>'); inCode = true; }
            continue;
        }
        if (inCode) { out.push(esc(line) + '\n'); continue; }
        const h = line.match(/^(#{1,4})\s+(.*)/);
        if (h) { flush(); out.push(`<h${h[1].length + 2}>${inline(h[2])}</h${h[1].length + 2}>`); continue; }
        const li = line.match(/^\s*[-*]\s+(.*)/);
        if (li) {
            if (para.length) { out.push(`<p>${inline(para.join(' '))}</p>`); para = []; }
            if (!list) { out.push('<ul>'); list = true; }
            out.push(`<li>${inline(li[1])}</li>`);
            continue;
        }
        if (!line.trim()) { flush(); continue; }
        if (list) { out.push('</ul>'); list = false; }
        para.push(line.trim());
    }
    if (inCode) out.push('</code></pre>');
    flush();
    return out.join('');
}


// ── First run ────────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', async () => {
    document.addEventListener('keydown', e => {
        if (e.key === 'Escape' && $('setupOverlay')?.classList.contains('visible')) closeSetupGuide();
    });
    try {
        const s = await (await fetch('/api/setup/status')).json();
        sgState.commands = s.commands || {};
        if (s.first_run) openSetupGuide('welcome');
    } catch (e) { /* server not reachable — no guide */ }
});
