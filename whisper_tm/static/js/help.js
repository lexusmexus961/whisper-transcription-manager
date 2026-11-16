// ══════════════════════════════════════════════════════════════════════════
//  Help & setup — one window, four tabs: Where is what (built from the
//  layout itself: the rail, the windows' tabs, the bars), How-tos, Keys,
//  and Setup (the setup guide, setupguide.js, moved into its tab)
// ══════════════════════════════════════════════════════════════════════════

const Help = (() => {
    const TABS = ['where', 'howto', 'keys', 'setup'];
    let current = 'where';

    function tab(t) {
        if (!TABS.includes(t)) t = 'where';
        current = t;
        document.querySelectorAll('#helpTabs .vx-tab').forEach(b => {
            const on = b.dataset.tab === t;
            b.classList.toggle('active', on);
            b.setAttribute('aria-selected', on ? 'true' : 'false');
        });
        document.querySelectorAll('#helpOverlay .help-page').forEach(p => { p.hidden = p.dataset.page !== t; });
        if (t === 'where') renderWhere();
        if (t === 'setup' && typeof sgShow === 'function' && !document.getElementById('sgContent')?.innerHTML) {
            sgShow(typeof sgState !== 'undefined' ? sgState.page : 'welcome');
        }
    }

    const titleOf = el => (el && (el.getAttribute('title') || el.getAttribute('aria-label')) || '').trim();
    const labelOf = el => (el ? el.textContent : '').replace(/\s+/g, ' ').trim();

    /* Where is what, read off the page as it is now: every rail item, every
       window's tabs, the top bar and the folder bar, each with the words
       its own tooltip uses. A row opens what it names. */
    function renderWhere() {
        const box = document.getElementById('helpWhere');
        if (!box) return;
        const row = (name, where, what, open) => `<tr${open ? ` class="help-go" onclick="${open}" title="Open it"` : ''}>`
            + `<td>${name}</td><td class="help-where">${esc(where)}</td><td>${esc(what)}</td></tr>`;
        const table = rows => `<table class="help-table help-where-table"><tbody>${rows.join('')}</tbody></table>`;
        const parts = [];
        // The rail, by its headings
        const rail = [];
        let head = '';
        for (const el of document.querySelectorAll('#rail > .rail-head, #rail > .rail-item[data-place], #railPins > .rail-head, #railPins .rail-pin')) {
            if (el.classList.contains('rail-head')) { head = labelOf(el); continue; }
            const name = labelOf(el.querySelector('.rail-label')) || labelOf(el);
            const id = el.dataset.place || el.dataset.pin;
            const open = el.dataset.place ? `closeHelp();Shell.go('${escJs(id)}')` : `closeHelp();Pins.open('${escJs(id)}')`;
            rail.push(row(`${el.querySelector('svg') ? el.querySelector('svg').outerHTML : ''} <b>${esc(name)}</b>`,
                          'Rail' + (head ? ' · ' + head : ''), titleOf(el).replace(/^[^:]*:\s*/, ''), open));
        }
        parts.push(_helpCard('The rail', table(rail), 'On the left, always there. Pinned tabs of your own sit in Pinned; right-click a tab or a Home tile to pin it.'));
        // The windows and their tabs
        const windows = [
            ['Find', '#svTabs .vx-tab:not([hidden])', id => `closeHelp();Shell.go('find');svTab('${id.replace(/^svTabBtn/, '').replace(/^./, c => c.toLowerCase())}')`],
            ['Voices', '#vxTabs .vx-tab', (id, b) => `closeHelp();vxTab('${b.dataset.tab}');Shell.go('voices')`],
            ['Queue', '.qe-tabs .qe-tab', (id) => `closeHelp();Shell.go('queue');qeShowTab('${({qeTabJobs: 'jobs', qeTabQueue: 'queue', qeTabLibrary: 'library', qeTabAdd: 'add'})[id] || 'jobs'}')`],
            ['Machine', '#mcTabs .vx-tab', (id, b) => `closeHelp();Machine.open('${b.dataset.tab}')`],
            ['Help & setup', '#helpTabs .vx-tab', (id, b) => `Help.tab('${b.dataset.tab}')`],
        ];
        for (const [win, sel, open] of windows) {
            const rows = [...document.querySelectorAll(sel)].map(b =>
                row(`${b.querySelector('svg') ? b.querySelector('svg').outerHTML : ''} ${esc(labelOf(b).replace(/\s*\d+$/, ''))}`,
                    win + ' window · tab', titleOf(b), open(b.id, b)));
            if (rows.length) parts.push(_helpCard(`${esc(win)} window`, table(rows)));
        }
        // The bars
        const bars = [];
        const add = (el, where, name) => { if (el && (el.offsetParent || el.closest('.dir-actions'))) bars.push(row(name || esc(labelOf(el)), where, titleOf(el))); };
        add(document.getElementById('globalSearch'), 'Top bar', '<b>Search</b> <kbd>/</kbd>');
        add(document.getElementById('sysLoad'), 'Top bar', '<b>CPU · RAM · GPU · VRAM</b>');
        document.querySelectorAll('.topbar .tb-system > button').forEach(b => add(b, 'Top bar', `${b.querySelector('svg')?.outerHTML || ''} ${esc(b.getAttribute('aria-label') || labelOf(b))}`));
        add(document.getElementById('dirInput'), 'Folder bar', '<b>The open folder</b>');
        document.querySelectorAll('.dir-bar .dir-actions > button').forEach(b => add(b, 'Folder bar', `<b>${esc(labelOf(b)) || '⋯'}</b>`));
        document.querySelectorAll('#folderMenu .bm-item').forEach(b => bars.push(row(esc(labelOf(b)), 'Folder bar · ⋯', titleOf(b))));
        parts.push(_helpCard('The top bar and the folder bar', table(bars)));
        box.innerHTML = `<header class="set-page-head"><p>Where everything is — read off the app as it is laid out now. A row opens what it names.</p></header>` + parts.join('');
    }

    function init() {
        // The setup guide's pages and their navigation move into the Setup tab
        const main = document.querySelector('#setupOverlay .set-main');
        const page = document.getElementById('helpSetup');
        if (main && page && main.parentElement !== page) page.appendChild(main);
    }

    document.addEventListener('DOMContentLoaded', init);
    return {tab, renderWhere, TABS, get current() { return current; }};
})();

function openHelp(tab) {
    document.getElementById('helpOverlay').classList.add('visible');
    _renderHelp();
    Help.tab(tab || Help.current);
}

function closeHelp() {
    document.getElementById('helpOverlay').classList.remove('visible');
}

function _helpCard(title, body, sub = '') {
    return `<section class="set-card">
        <div class="set-card-head"><div class="set-card-titles"><h3>${title}</h3>
            ${sub ? `<div class="set-card-sub">${sub}</div>` : ''}</div></div>
        <div class="set-card-body" style="padding-bottom:16px">${body}</div>
    </section>`;
}

// One usage scenario: folded, the steps inside
function _helpScenario(title, steps, tip = '') {
    return `<details class="help-scenario">
        <summary>${title}</summary>
        <div class="help-scenario-body">
            <ol class="help-steps">${steps.map(s => `<li>${s}</li>`).join('')}</ol>
            ${tip ? `<p class="help-tip">💡 ${tip}</p>` : ''}
        </div>
    </details>`;
}

function _renderHelp() {
    const el = document.getElementById('helpContent');
    const rows = (list) => `<table class="help-table">${list.map(([a, b]) =>
        `<tr><td>${a}</td><td>${b}</td></tr>`).join('')}</table>`;
    const v = document.getElementById('appVersion')?.textContent || 'Whisper Transcription Manager';
    el.innerHTML = `
        <header class="set-page-head">
            <p>The first steps and the usual ways of working.</p></header>

        ${_helpCard('Quick start', `<ol class="help-steps">
            <li><b>Open a folder</b> — type a path in the folder field at the top (▾ to browse; ⋯ → Favorite folder keeps it) and press Enter</li>
            <li><b>Pick a file</b> — click it in the sidebar; the filter bar above the list narrows it down (All / Open / Done …)</li>
            <li><b>Transcribe</b> — hover a file and click <b>+Q</b> (or <b>⤒</b> to go first), or <b>▶ Transcribe … open</b> at the bottom of the sidebar for everything in the filter</li>
            <li><b>Read</b> — the transcript opens next to the player; click a line to jump there</li>
        </ol>`)}

        ${_helpCard('Typical scenarios', [
            _helpScenario('🎬 Transcribe a large NAS with thousands of files', [
                'Open the NAS folder (e.g. <code>/mnt/nas/videos</code>)',
                'Click <b>Open</b> in the filter bar to see only what is not transcribed yet',
                'Click <b>▶ Transcribe … open</b> at the bottom of the sidebar, choose model and options, start',
                'The queue runs on the server — close the browser whenever you like',
                `<b>${tbIcon('queue')} Queue</b> in the rail shows what runs, waits and failed; its tab <b>Edit in bulk</b> filters, reorders and changes settings`,
            ], `Put the work folder on a local SSD (${tbIcon('settings')} Settings → Folders) — the NAS only has to deliver each file once.`),
            _helpScenario('📺 Subtitles for Plex / Jellyfin', [
                'Start the transcription (single file or <b>▶ Transcribe … open</b>)',
                'In the dialog tick <b>SRT alongside the video file</b> — and <b>Per-word timestamps</b> for tighter timing',
                'The <code>.srt</code> lands next to the video; for existing transcripts use <b>SRT</b> in the transcript toolbar',
            ], 'Plex picks up <code>video.srt</code> when it sits next to <code>video.mp4</code>. How the cues are cut: Settings → Subtitles.'),
            _helpScenario('🎤 Speech over music, games or noise', [
                'In the transcription dialog keep <b>Vocal Isolation</b> on',
                'Tick <b>Save processed audio as MP3</b> to keep the separated voice next to the source',
            ], 'A pre-filter skips separation for chunks that are already clean, so leaving it on costs little.'),
            _helpScenario('✨ Find a moment you only remember vaguely', [
                'Type a description into the search at the top and press <b>Enter</b>: <i>“Alice tells a story at a birthday party and says that\'s the best day ever — a WhatsApp video”</i>. A description goes to <b>Find</b>; a few words or quoted words go to the text, a file name to <b>Files</b>',
                'The language model splits it into searches — word for word, by meaning, by the picture and the sound, inside the files whose name has what you remember, in what the named people say — and runs them at once, each in its own tile (⤢ maximizes one, ↗ opens it in its own tab)',
                '<b>Best matches</b> brings them together: each recording once (<b>+2 more moments</b> under it), copies of a recording folded in as <b>⧉ suspected duplicates</b> — found by length, words and picture, never by the name',
                'The language model checks the best ones with their stills and the facts the program knows (does the name have “-WA”, is it a video, does Alice speak in it) and sorts them; a click plays the moment right there, <b>💬 Ask about these results</b> continues in Ask',
            ], 'A search takes about a minute; the first hits come after a few seconds. One at a time — a new one stops the one before. In Ask, the assistant can run it too (“find the moment where …”).'),
            _helpScenario('🔍 Find a statement across all recordings', [
                'Type into the search at the top: <code>server migration</code> (all words, partial match)',
                'Exact word or phrase: <code>"good morning"</code> in quotation marks',
                'Press <b>Enter</b> for the Find window — a few words open the <b>Text</b> tab; filter by file, speaker or person there',
                'Click a hit to preview it, double-click to jump into the transcript at that point',
            ], 'With the semantic search set up (Settings → Search), the Find window also finds passages that say the same in other words.'),
            _helpScenario('🔎 Every place where something happens', [
                'Describe the situation in the Find window: <i>“Alice gets a compliment”</i>, <i>“someone apologizes for being late”</i>',
                '<b>Meaning</b> lists the closest passages by relevance, each with a bar; copies of a recording in other folders fold into one hit (<b>+3 copies</b>), weaker matches wait below',
                '<b>🔎 Deep search</b> next to the search field goes through the whole library: the language model plans the search and checks the likely passages one by one',
                'Speakers appear by name where the voice library or the transcript knows them (<b>[Alice]</b> instead of SPEAKER_04) — and a person named in the query also brings the passages they speak in, even where nobody says the name',
                'The <b>Deep search</b> tab shows the plan, how far it is and every confirmed passage as it is found — ▶ plays it, a click opens the transcript there; unclear ones are kept apart',
            ], 'A deep search runs in the LLM queue on the server and takes minutes — close the search any time, the result stays in the Deep search tab. The relevance ranking needs the graphics card; while it is busy, the meaning list is sorted by meaning and wording and says so.'),
            _helpScenario('🖼 Find a moment by what is seen or heard', [
                'Open Find, tab <b>Images &amp; sounds</b>, and describe it: <i>“police at night”</i>, <i>“a crowd”</i>, <i>“gunshots”</i>, <i>“rock music”</i>',
                'Stills, clips, clips with their sound, sounds, spoken text and the language model\'s descriptions each get their own list; a click plays the moment, <b>▶ In the player</b> and <b>📄 Transcript</b> open it in the main window',
                '<b>🔁 Similar</b> searches with a hit itself; <b>📎 Query with a file</b> with an image, a sound or a video of your own (optionally plus the text)',
                '<b>🔎 Check with the language model</b> lets a model that sees pictures judge the best stills one by one and sorts them by its verdict',
                '<b>💬 Spoken text</b> is the words of a moment\'s 10 s — what is said where something is seen or heard. What was said, with its context, is the Text tab\'s meaning search: the list takes the query there in one click',
            ], 'Set up and indexed under Settings → Images & sounds — its own environment, the index built as a guest on the GPU the queue works on, with a progress bar like the semantic search\'s. Sounds are asked in English for you (the language model translates); pictures understand other languages too.'),
            _helpScenario('💬 Ask the library a question', [
                `Click <b>${tbIcon('chat')} Ask</b> in the rail — or <b>Ask</b> next to the search field, which takes the search term and its filters along`,
                'Ask in your own words: <i>“Where does Carol talk about the move, and what does she say?”</i> — the assistant searches by itself (words, meaning, who speaks where, who talked with whom) and shows each step',
                'Every quote carries a reference like <b>S3 ▶ 1:02:21</b>: click plays the passage, double-click opens its transcript; on the right all quoted passages, each playable',
                'Follow-up questions keep the conversation; the conversations stay on the left',
            ], 'It uses the language model set under Settings → AI models (task “Chat”) and the tools of the app\'s MCP server. By default it only reads and searches — Settings → Search → Ask allows it to start work.'),
            _helpScenario('📡 New files transcribed by themselves', [
                '<b>Auto</b> in the folder bar switches continuous scanning on — the dot turns green',
                'The first round only notes what is there; a file that appears after that is added to the library and queued with the transcription defaults',
                'A file still being copied waits until it stopped changing; the button\'s tooltip says what was found',
            ], 'Folders, interval and “only add, don\'t transcribe” under Settings → Library → Scanning. Off by default.'),
            _helpScenario('🗣 Who is speaking — across recordings', [
                'Transcribe with <b>Detect speakers</b> and <b>Record voice vectors</b>',
                `Open <b>${tbIcon('voices')} Voices</b> in the rail — on the <b>Voice map</b> similar voices form groups`,
                'The <b>Name voices</b> tab walks you through the biggest unnamed groups first — <b>Undo</b> next to it brings a decision back as its question',
                'Every decision is in the <b>History</b> tab: undo or redo any of them, the rest stays as it is',
                'To check a person: <b>People</b> → <b>▶ Listen through</b> — every voice with autoplay, the doubtful ones first; <kbd>N</kbd> not this person, <kbd>T</kbd> trash, <kbd>A</kbd> name as …',
            ]),
            _helpScenario('👥 Find conversations by who took part and how they talked', [
                `Open <b>${tbIcon('sparkle')} Find</b> in the rail and its tab <b>${tbIcon('conversation')} Conversations</b> — or Voices → Conversations, where a person's <b>Conversations</b> button lands`,
                'Add people: type a name and press <kbd>Enter</kbd> — the voice library\'s names are suggested. With two or more, <b>All together</b> finds where they talk with each other (in a long recording only that stretch: <i>1:15:00–2:18:00 of 6:00:00</i>), <b>Any of them</b> every conversation one of them takes part in',
                'Pick a length (<b>About 1 h</b>, <b>Over 2 h</b> …) and a kind: <b>Detailed</b> (long sentences, long turns), <b>Back-and-forth</b> (many speaker changes), <b>Interview</b>, <b>Group</b> (three speakers or more), <b>Monologue</b>',
                '<b>Longer than usual</b> keeps the conversations whose sentences are longer than most — with people chosen, longer than usual for <i>them</i> (their own conversations), otherwise than most in your library in their language; <b>Filters</b> adds words per sentence, speaker changes, speakers, questions, language, folder and dates',
                'Each row shows who took part with their share, the words per sentence (<i>longer than 78 %</i>), speaker changes, questions and the longest monologue; ▶ plays it, <b>open ›</b> opens the transcript there',
                '<b>✂ Extract</b> opens Voices → Voice dataset with these conversations: one voice\'s sentences from just them, as a dataset or one audio file',
            ], 'Ask can do it in words: <i>“an hour-long conversation of Alice, Bob and Carol in long sentences”</i> — and Find opens this tab for such a request. A recording\'s <b>Statistics</b> tab shows its numbers in the card <b>Conversation</b>. Recordings transcribed before are counted once in the background — by themselves, or with <b>Count now</b> in the tab (Settings → Search → Conversations).'),
            _helpScenario('🎙 One person\'s voice as a dataset', [
                `Open <b>${tbIcon('voices')} Voices</b> → <b>Voice dataset</b> and choose the person`,
                'Narrow it down: named or also recognized voices, dates, folder, text, length — and <b>Whisper sure at least</b> (%) for clean sentences only',
                `Only from some conversations — an hour with two friends in long sentences, say: find them in the Find window's tab <b>${tbIcon('conversation')} Conversations</b> and press <b>✂ Extract</b> there`,
                'Check the clips: <b>✎</b> opens one as a waveform — drag the bounds or press <kbd>S</kbd> to cut at the pauses, fix the text, <kbd>Enter</kbd> keeps it, <kbd>Del</kbd> drops it. With word timestamps every word shows how sure it is',
                '<b>Export</b> writes a dataset (WAV + LJSpeech metadata) or one audio file with subtitles — to download, or straight into a folder of your choice, unpacked if you like. Use a person\'s voice only with their consent — a voice dataset can train a model that speaks like them.',
            ], 'Clips are whole sentences of the transcript, taken only where the voice is alone. Clips you didn\'t trim are cut at the nearest pause on export; your edits stay until the server restarts.'),
            _helpScenario('🤖 Let an AI assistant use the library', [
                'Settings → MCP server: switch on <b>MCP over HTTP</b> (or the OpenAPI service for Open WebUI) and restart the app',
                'Copy the ready-made set-up for your client from <b>Connect a client</b> — address and access token are filled in',
                'The assistant can now search, read transcripts, look up speakers and who talked with whom — and, if allowed, transcribe',
            ]),
            _helpScenario('✂️ Cut a clip', [
                'Open a transcribed video and click <b>✂ Clip</b> in the transcript toolbar',
                'Set start and end by clicking segments or typing times',
                'Preview, cut, download',
            ]),
        ].join(''))}

        ${_helpCard('Configuration', `
            <p class="help-text">All settings are stored in <code>config.yaml</code>. The file is created automatically and can be edited by hand or through ${tbIcon('settings')} Settings.</p>
            <p class="help-text">Where it is looked for, in this order:</p>
            <ol class="help-steps">
                <li>the <code>--config</code> argument</li>
                <li>the <code>WHISPER_TM_CONFIG</code> environment variable</li>
                <li><code>~/.config/whisper-tm/config.yaml</code></li>
                <li><code>./config.yaml</code> in the working folder (only when the one above doesn't exist)</li>
            </ol>`)}

        <div class="help-foot">${esc(v)} · Powered by WhisperX, pyannote, audio-separator</div>
    `;
    document.getElementById('helpKeys').innerHTML = `
        <header class="set-page-head"><p>Keys that work everywhere, and the ones of single views.</p></header>
        ${_helpCard('Everywhere', rows([
            ['<kbd>Alt</kbd>+<kbd>←</kbd> / <kbd>Alt</kbd>+<kbd>→</kbd>', 'Back / forward: places, tabs, recordings and steps like a person\'s page'],
            ['<kbd>/</kbd>', 'Put the cursor in the search field (in the settings: search every setting)'],
            ['<kbd>↓</kbd> <kbd>↑</kbd> <kbd>Enter</kbd>', 'In the search field\'s list: choose a place or a recording and open it; Enter alone opens Find'],
            ['<kbd>Esc</kbd>', 'Close the top layer: a menu, a dialog, a step inside a window, then the window'],
        ]))}
        ${_helpCard('The player and the file list', rows([
            ['<kbd>Space</kbd>', 'Play / pause (player focused)'],
            ['<kbd>←</kbd> <kbd>→</kbd>', 'Back / forward 5 s (player focused)'],
            ['<kbd>Ctrl+A</kbd>', 'Select every file of the current filter (file list)'],
            ['<kbd>Shift</kbd> / <kbd>Ctrl</kbd> + click', 'Select a range / single files'],
            ['<kbd>Ctrl+F</kbd>', 'In the settings: search every setting'],
        ]))}
        ${_helpCard('Voices', rows([
            ['Name voices, People → Listen through, name fields', 'Voices → <b>How it works</b> lists their keys'],
        ]))}`;
}

document.addEventListener('keydown', e => {
    if (e.key === 'Escape' && document.getElementById('helpOverlay')?.classList.contains('visible')) {
        closeHelp();
    }
});
