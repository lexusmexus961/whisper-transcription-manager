// ══════════════════════════════════════════════════════════════════════════
//  Settings — the pages
// ══════════════════════════════════════════════════════════════════════════
//
// Each page: an id (other parts of the app open it by that id), a group in
// the navigation, an icon from the sprite in index.html, a title, one line
// that says what the page is for, and render(cfg) → the cards. A page that
// covers more than one topic splits its cards into sections (_sSection);
// the navigation lists the sections of the open page — or its cards, when
// it has none. `after` runs once the page is in the DOM (live parts load
// there), `open` each time it is shown. `owns` names the config keys its
// editors write, for the unsaved mark in the navigation.
//
// How a setting is written (settings.js, _sRow): the name says what it
// controls, the description what it does and when to change it — in one or
// two sentences. Measurements, examples and the why go into `more`, which
// is folded. Tags: "New jobs" = a start value of the transcription dialog,
// "Restart" = read when the server starts.

// The transcription dialog's lists (txoptions.js) — the default can only be
// something the dialog offers
const _S_MODELS = [
    ['large-v3', 'large-v3 — best quality (~10 GB VRAM)'],
    ['large-v2', 'large-v2 — proven (~10 GB VRAM)'],
    ['turbo', 'turbo — fast, very good (~6 GB VRAM)'],
    ['distil-large-v3', 'distil-large-v3 — faster, similar quality'],
    ['distil-large-v2', 'distil-large-v2'],
    ['medium', 'medium (~5 GB VRAM)'],
    ['small', 'small (~2 GB VRAM)'],
    ['base', 'base (~1 GB VRAM)'],
    ['tiny', 'tiny (~1 GB VRAM)'],
    ['medium.en', 'medium.en — English only'],
    ['small.en', 'small.en — English only'],
    ['base.en', 'base.en — English only'],
    ['tiny.en', 'tiny.en — English only'],
    ['distil-medium.en', 'distil-medium.en — English only'],
    ['distil-small.en', 'distil-small.en — English only'],
];
const _S_LANGUAGES = [
    ['auto', 'Detect automatically'], ['en', 'English'], ['de', 'German'], ['fr', 'French'],
    ['es', 'Spanish'], ['it', 'Italian'], ['pt', 'Portuguese'], ['nl', 'Dutch'], ['pl', 'Polish'],
    ['ru', 'Russian'], ['tr', 'Turkish'], ['zh', 'Chinese'], ['ja', 'Japanese'], ['ko', 'Korean'],
    ['sv', 'Swedish'], ['da', 'Danish'], ['no', 'Norwegian'], ['fi', 'Finnish'], ['cs', 'Czech'],
    ['uk', 'Ukrainian'], ['ar', 'Arabic'], ['hi', 'Hindi'],
];

// A list with the saved value in it, even when the list doesn't know it — a
// select that can't show the value would save its first option instead
function _sWith(options, value) {
    if (value === '' || value == null || options.some(([v]) => String(v) === String(value))) return options;
    return [[value, `${value} (set in config.yaml)`], ...options];
}

function _sCode(s) { return `<code>${esc(s)}</code>`; }

// A button that opens another page (or a section of one) — `to` as for
// openSettings: 'embedding', 'search#semantic' …
function _sGo(to, label) {
    return `<button type="button" class="set-link" onclick="setSettingsTab('${to}')">${label} →</button>`;
}

// Images & sounds next to the semantic search: what each is for, so the
// difference between the two indexes is visible
function _mediaCompare() {
    const rows = [
        ['Finds', 'what was <b>said</b> — the meaning of a passage', 'what is <b>seen or heard</b> at a moment — also where nobody talks'],
        ['Reads', 'the transcripts', 'the recordings: pictures and sound'],
        ['A hit is', 'a passage of about a minute, with speaker names', 'a moment: a still every 20 s, 10 s of clip and of sound'],
        ['Model', 'the embedding model of AI models — text only', 'EmbeddingGemma 2 — text, pictures, video and sound in one space'],
        ['Used by', 'the meaning hits of the search, Ask, deep searches, AI clients (MCP)', 'the Find window\'s tab “Images &amp; sounds”'],
        ['Spoken text', 'the whole conversation, with its context — <b>the place to search what was said</b>', 'only the 10 s around a moment: what is said where something is seen or heard, and a way from a picture or a sound to words'],
    ];
    return `<table class="set-compare">
        <thead><tr><th></th><th>Semantic search ${_sGo('search#semantic', 'Search')}</th><th>Images &amp; sounds <span class="set-compare-here">this page</span></th></tr></thead>
        <tbody>${rows.map(r => `<tr><th>${r[0]}</th><td>${r[1]}</td><td>${r[2]}</td></tr>`).join('')}</tbody></table>
        <div class="set-hint" style="margin-top:8px">A question about what someone said belongs in the semantic search; “a police car at night”, “gunshots” or “more like this picture” in Images &amp; sounds.</div>`;
}

const SETTINGS_PAGES = [

// ── Library ──────────────────────────────────────────────────────────────

{
    id: 'directories', group: 'Library', icon: 'folder', title: 'Folders', hint: 'Library, transcripts, working folders',
    intro: 'Where the library is, where transcripts are kept and where the app puts its working files. An empty folder field means automatic — the path it resolves to is shown in grey.',
    render: (cfg) => {
        const d = cfg.directories || {};
        const dirs = _settingsData.dirs || {};
        const roots = cfg.security?.allowed_roots || [];
        const auto = (k) => `Automatic: ${dirs[k] || ''}`;
        const central = (d.transcript_mode || 'alongside') === 'central';
        return _sCard({
            title: 'Library',
            sub: 'The folder the file list shows. Switch folders with the path field at the top of the window — that is the one place for it.',
            body: _sRow({label: 'Open folder', type: 'static', wide: true,
                    value: `<div><div class="mono">${esc(dirs.media || '')}</div>
                        <div class="set-path-state" data-path-state="media" data-no-path></div></div>`})
                + _sRow({label: 'Allowed folder trees', type: 'static',
                    value: roots.length ? roots.map(r => `<span class="mono">${esc(r)}</span>`).join('<br>')
                                        : 'Only the folder the server started with',
                    desc: 'The app reads and writes only inside these trees — every folder you open, every path a request names. Changed in config.yaml only (' + _sCode('security.allowed_roots') + '), never from this page: a request that could widen it could switch the check off.'}),
        }) + _sCard({
            title: 'Transcripts',
            body: _sRow({key: 'directories.transcript_mode', type: 'seg', label: 'Where transcripts are saved',
                    options: [['alongside', 'Next to the media'], ['central', 'One folder'], ['database', 'Database only']],
                    onchange: '_sTranscriptMode(this.value)',
                    desc: 'The JSON file of a new transcript. The database is updated in every mode; switching doesn\'t move transcripts that exist.',
                    more: `<p><b>Next to the media</b> — a ${_sCode('.transcripts/')} folder beside each media file: ${_sCode('/videos/show/e1.mp4 → /videos/show/.transcripts/e1.mp4.json')}. Backed up together with the library; two files with the same name in different folders never collide.</p>
                           <p><b>One folder</b> — all transcripts in one place, the path flattened into the file name. For media on a read-only share.</p>
                           <p><b>Database only</b> — no JSON files at all; the fastest with very many files. The database is then the only copy: keep the backup folder (Database) set.</p>`})
                + _sRow({key: 'directories.transcripts', type: 'dir', dir: 'transcripts', label: 'Transcript folder',
                    hidden: !central, id: 'setRowTranscripts',
                    placeholder: `Automatic: ${esc((_settingsData.data_dir || '<data folder>') + '/transcripts')}`,
                    desc: 'Used in “One folder” mode only.'}),
        }) + _sCard({
            title: 'Working folders',
            body: _sRow({key: 'directories.work', type: 'dir', dir: 'work', label: 'Work folder', tags: ['restart'],
                    placeholder: auto('work'),
                    desc: 'Scratch space for running jobs: extracted audio, separated vocals, chunks. Emptied at every server start — so nothing of yours belongs in it. A fast local SSD makes long files much faster.'})
                + _sRow({key: 'directories.uploads', type: 'dir', dir: 'uploads', label: 'Uploads and downloads',
                    placeholder: 'Automatic: uploads/ in the open folder',
                    desc: 'Uploaded files and link downloads. Empty = an uploads/ folder inside the open media folder, following it when you switch — only there do new files show up in the file list.'})
                + _sRow({key: 'directories.clips', type: 'dir', dir: 'clips', label: 'Clips',
                    placeholder: auto('clips'),
                    desc: 'Cut clips and the short audio snippets of the voice map and search results.'})
                + _sRow({key: 'directories.logs', type: 'dir', dir: 'logs', label: 'Job logs', tags: ['restart'],
                    placeholder: auto('logs'),
                    desc: 'One log file per transcription job (Diagnostics → Job logs). The server\'s own log goes elsewhere (' + _sCode('logging.debug_dir') + ' in config.yaml).'}),
        });
    },
    after: () => _sDirStates(),
},

{
    id: 'scanner', group: 'Library', icon: 'drive', title: 'Scanning', hint: 'Which files count, what is skipped, network drives',
    intro: 'How the library is read from disk: which files count as media, what is skipped, and how a scan copes with a network drive that drops out.',
    render: (cfg) => _sCard({
        title: 'Continuous scanning',
        sub: 'The library watched for new files — and new files transcribed by themselves. The same switch as “Auto” in the folder bar.',
        body: _sBlock('<div class="set-hint" id="setAutoscanState">Loading…</div>', {skip: true})
            + _sRow({key: 'autoscan.enabled', type: 'toggle', label: 'Watch for new files',
                desc: 'Every few minutes the watched folders are checked for files that weren\'t there before. The first round only notes what is there: switching this on over a big library queues nothing that already existed.'})
            + _sRow({key: 'autoscan.transcribe', type: 'toggle', label: 'Transcribe new files',
                desc: 'A new file is queued at the end of the queue with the transcription defaults (Transcription). Off = it only joins the library. A file whose transcript already exists is never queued.'})
            + _sRow({key: 'autoscan.interval_min', type: 'number', unit: 'min', min: 1, label: 'Check every',
                desc: 'Minutes between two rounds. A round only reads folders whose time changed — one look per folder instead of reading the whole NAS; a round never starts before twice the last one\'s duration has passed.'})
            + _sRow({key: 'autoscan.settle_min', type: 'number', unit: 'min', min: 0, label: 'Wait until copied',
                desc: 'A new file must not change for this long before it is touched — a big video on the NAS is seen long before it is complete.'})
            + _sRow({key: 'autoscan.folders', type: 'lines', label: 'Folders to watch', rows: 3,
                placeholder: 'empty = the whole library',
                desc: 'One folder per line, inside the allowed folders. Empty = the whole library (' + _sCode('security.allowed_roots') + ').'})
            + _sRow({key: 'autoscan.full_every_h', type: 'number', unit: 'h', min: 1, label: 'Read everything every',
                desc: 'Every this many hours a round reads every folder, not only the changed ones — catches what a share didn\'t report.'}),
    }) + _sCard({
        title: 'What counts as media',
        body: _sRow({key: 'scanner.extra_video_extensions', type: 'list', label: 'Extra video extensions',
                placeholder: '.rmvb, .vob',
                desc: 'On top of the built-in ones (mp4, mkv, avi, mov, webm, ts, mpg, wmv, flv …), comma-separated.'})
            + _sRow({key: 'scanner.extra_audio_extensions', type: 'list', label: 'Extra audio extensions',
                placeholder: '.dts, .ac3',
                desc: 'On top of the built-in ones (mp3, wav, flac, ogg, m4a, opus, aac, amr …).'})
            + _sRow({key: 'scanner.ignore_patterns', type: 'lines', label: 'Skip these names', rows: 5,
                desc: 'Patterns for file and folder names, one per line (* and ? work). Matched against the name, not the whole path. Keep ' + _sCode('.*') + ' — it also hides NAS snapshot folders like .zfs.'}),
    }) + _sCard({
        title: 'Notice new files in the open folder', fold: true,
        sub: 'Older and simpler than continuous scanning above: a notice only, for the folder that is open.',
        body: _sRow({key: 'scanner.watch_enabled', type: 'toggle', label: 'Notice new files', tags: ['restart'],
                desc: 'Checks the open folder regularly and shows a notice when new files appear. It doesn\'t add them — a scan does that. Each round walks the whole tree, which is costly on a large NAS.'})
            + _sRow({key: 'scanner.watch_interval_sec', type: 'number', unit: 's', min: 5, label: 'Check every', tags: ['restart'],
                desc: 'Seconds between two rounds; a round never starts before twice the last walk\'s duration has passed.'}),
    }) + _sCard({
        title: 'Transcripts found on disk',
        sub: 'Transcript JSON files that are on disk but not in the database — made by another installation or copied in.',
        body: _sRow({key: 'database.auto_add_dirs', type: 'toggle', label: 'Import while scanning',
                desc: 'When a scan finds a new or changed .transcripts/ folder, its transcripts are imported into the database in the background.'})
            + _sRow({key: 'database.sync_on_start', type: 'toggle', label: 'Import at server start', tags: ['restart'],
                desc: 'At start, import the transcripts of the transcript folder (the central folder, or the open folder\'s .transcripts/).'}),
    }) + _sCard({
        title: 'Network drives', fold: true,
        sub: 'For a library on a NAS: how many reads run at once, and what happens when the share drops out in the middle of a scan.',
        body: _sRow({key: 'scanner.walk_workers', type: 'number', min: 1, max: 128, label: 'Parallel folder reads',
                desc: 'Folders read at the same time while files are discovered. Fewer = gentler on the NAS, more = faster. 1 = one at a time.'})
            + _sRow({key: 'scanner.scan_workers', type: 'number', min: 1, max: 256, label: 'Parallel file checks',
                desc: 'Files checked at the same time after discovery (size, date, transcript status). Raise it for a slow NAS.'})
            + _sRow({key: 'scanner.walk_wait_for_mount', type: 'toggle', label: 'Wait when the share drops out',
                desc: 'Pause the scan and continue once the drive is back; failed folders are read again. Off = the scan stops with an error.'})
            + _sRow({key: 'scanner.walk_wait_probe_sec', type: 'number', unit: 's', min: 1, label: 'Look for the drive every',
                desc: 'While waiting.'})
            + _sRow({key: 'scanner.walk_wait_max_min', type: 'number', unit: 'min', min: 0, label: 'Wait at most',
                desc: '0 = as long as it takes (the scan still ends when its browser tab closes).'})
            + _sRow({key: 'scanner.walk_max_error_streak', type: 'number', min: 0, label: 'Dropped out after',
                unit: 'errors', desc: 'This many failed folder reads in a row count as the drive being gone. 0 = off.'})
            + _sRow({key: 'scanner.walk_stall_timeout_sec', type: 'number', unit: 's', min: 0, label: '…or after silence of',
                desc: 'This long without a single folder answering counts as the drive being gone. 0 = off.'}),
    }),
    after: () => { if (typeof autoscanLoad === 'function') autoscanLoad(); },
},

{
    id: 'youtube', group: 'Library', icon: 'download', title: 'Downloads', hint: 'Links and playlists (yt-dlp)',
    intro: 'Links and playlists from the Add dialog, downloaded with yt-dlp into the uploads folder (Folders).',
    render: (cfg) => _sCard({
        title: 'Links and playlists',
        body: _sRow({label: 'yt-dlp', type: 'static',
                value: _hwData?.has_ytdlp ? '<span class="set-badge ok">✓ installed</span>' : '<span class="set-badge miss">✕ missing</span>',
                desc: _hwData?.has_ytdlp ? 'Downloads run on the server.' : 'Without yt-dlp on the server, links can\'t be downloaded: ' + _sCode('pip install yt-dlp') + '.'})
            + _sRow({key: 'youtube.audio_only', type: 'toggle', label: 'Audio only', tags: ['jobs'],
                desc: 'Download just the audio track (usually .m4a) instead of the video — much smaller when you only want the transcript. Pre-selects the switch in the Add dialog.'})
            + _sRow({key: 'youtube.extra_args', type: 'text', label: 'Extra yt-dlp options', wide: true,
                placeholder: '--limit-rate 5M',
                desc: 'Added to every download, written like a shell line — e.g. ' + _sCode('--limit-rate 5M') + ' or ' + _sCode('--sub-langs de --write-subs') + '.',
                more: '<p>Options that run programs or choose files (' + _sCode('--exec') + ', ' + _sCode('--output') + ', ' + _sCode('--ffmpeg-location') + ', ' + _sCode('--downloader') + ' …) are refused here: this page needs no login while the login is off, and such an option would let anyone on the network run commands on the server. If you really need one, write it into ' + _sCode('youtube.extra_args') + ' in config.yaml.</p>'}),
    }) + _sCard({
        title: 'Cookies', fold: true,
        sub: 'For age-restricted or private videos: yt-dlp logs in with the cookies of your browser.',
        body: _sBlock('<div id="ytCookieBox"><div class="set-hint">Loading…</div></div>'),
    }),
    after: () => _ytCookiesLoad(),
},

// ── Transcription ────────────────────────────────────────────────────────

{
    id: 'transcription', group: 'Transcription', icon: 'mic', title: 'Transcription', hint: 'Defaults, vocabularies, clean-up',
    intro: 'The start values of the transcription dialog, the vocabularies, and the clean-up every transcript runs through. A job keeps the values it was queued with.',
    owns: ['transcription.vocab_presets'],
    render: (cfg) => {
        const t = cfg.transcription || {};
        return _sCard({
            title: 'Defaults for new jobs',
            sub: 'Pre-selected in every transcription dialog — each job can change them there, and “☆ Make these my defaults” in the dialog saves its choices here.',
            body: _sRow({key: 'transcription.default_model', type: 'select', label: 'Model', tags: ['jobs'],
                    options: _sWith(_S_MODELS, t.default_model),
                    desc: 'The speech model (Whisper). Larger models are more accurate, slower and need more VRAM.'})
                + _sRow({key: 'transcription.default_language', type: 'select', label: 'Language', tags: ['jobs'],
                    options: _sWith(_S_LANGUAGES, t.default_language),
                    desc: 'Automatic lets Whisper detect it per file — less reliable on noisy recordings.'})
                + _sRow({key: 'transcription.default_diarize', type: 'toggle', label: 'Speaker diarization', tags: ['jobs'],
                    desc: 'Detect who speaks when. Noticeably slower; needs a Hugging Face token once (Speakers).'})
                + _sRow({key: 'transcription.default_num_speakers', type: 'number', min: 0, max: 8, label: 'Number of speakers', tags: ['jobs'],
                    desc: '0 = detect automatically. Set it only when nearly all recordings have the same number of speakers.'})
                + _sRow({key: 'transcription.word_timestamps', type: 'toggle', label: 'Word timestamps', tags: ['jobs'],
                    desc: 'A time for every word — needed for fine subtitle cues (Subtitles) and word highlighting. Adds an alignment step.'})
                + _sRow({key: 'transcription.default_vocabulary', type: 'select', label: 'Vocabulary', tags: ['jobs'],
                    options: _sWith([['', '— none —'], ...Object.keys(t.vocab_presets || {}).map(n => [n, n])], t.default_vocabulary),
                    desc: 'The vocabulary the dialog starts with (the list is below). Keep none unless nearly all recordings are about the same things — a vocabulary only helps where it fits.'})
                + _sRow({key: 'transcription.export_alongside', type: 'toggle', label: 'Subtitle files next to the media', tags: ['jobs'],
                    desc: 'Write the transcript as files next to the media file, for Plex or Jellyfin. While this is on, the files are also rewritten when you edit a transcript.'})
                + _sRow({key: 'transcription.export_alongside_formats', type: 'checks', label: 'Which files',
                    options: [['srt', 'SRT'], ['txt', 'TXT'], ['json', 'JSON']],
                    desc: 'None ticked = SRT.'}),
        }) + _sCard({
            title: 'Vocabularies',
            sub: 'How names and terms are spelled — Whisper reads the text as if it had been said just before the recording. Pick one per job in the transcription dialog.',
            body: _sBlock('<div id="vocabPresetsList" class="set-items"></div>'
                + `<div class="set-add"><input class="set-input" id="vocabNewPresetName" placeholder="Name, e.g. the series"
                        onkeydown="if(event.key==='Enter'){event.preventDefault();_addVocabPreset();}">
                    <button type="button" class="set-btn" onclick="_addVocabPreset()">+ Add vocabulary</button></div>`
                + `<details class="set-more"><summary>How to write one</summary><div class="set-more-body">
                    <p>Write a sentence that contains the terms — ${_sCode('The conversation is about Alice, Bob and ACME.')} — not a bare comma list: Whisper continues the style it is given, and on unclear audio a list comes back as a list.</p>
                    <p>Only the last ~224 tokens count; put the most important terms at the end. The counter shows an estimate.</p></div></details>`)
                + _sRow({key: 'transcription.vocab_no_repeat_ngram', type: 'number', min: 0, max: 20, label: 'Repetition brake: word groups',
                    desc: 'With a vocabulary, a group of this many words may not repeat — stops a transcript that runs away into a loop. 0 = off.',
                    more: '<p>Only while a vocabulary is active: a vocabulary makes Whisper more likely to repeat itself, and in its batch mode nothing else stops a loop. Not used without one — it could cut real repetitions of speech.</p>'})
                + _sRow({key: 'transcription.vocab_repetition_penalty', type: 'number', min: 1, max: 2, step: 0.05, label: 'Repetition brake: penalty',
                    desc: 'With a vocabulary, words already said are made less likely by this factor. 1.0 = off.'}),
        }) + _sCard({
            title: 'Clean-up',
            sub: 'Applied to every new transcript.',
            body: _sRow({key: 'transcription.filter_hallucinations', type: 'toggle', label: 'Drop Whisper artifacts',
                    desc: 'Removes segments Whisper invents on music or silence — a TV station\'s subtitle credit and the like: every segment that matches a pattern below, and every segment without a single letter or digit.'})
                + _sRow({key: 'transcription.hallucination_patterns', type: 'lines', label: 'Artifact patterns', rows: 5,
                    ctlId: 'artifactPatterns',
                    desc: 'One regular expression (regex) per line. A segment is dropped when its whole text matches one — case and the punctuation around it don\'t matter, ' + _sCode('.*') + ' stands for any text. Try a line in the field below.',
                    more: '<p>' + _sCode('Subtitles by.*') + ' drops every “Subtitles by …” credit; ' + _sCode('(Music ?)+') + ' drops “Music” and “Music Music Music”; ' + _sCode('Thank you\\.?') + ' drops a lone “Thank you.” — but also a real one, so keep patterns narrow.</p><p>Only a whole segment is dropped: a pattern never cuts words out of a longer segment. It applies to new transcripts — a file transcribed before keeps its segments until it is transcribed again.</p>'})
                + _sBlock(`<div class="set-actions">
                        <input class="set-input" id="artifactTryText" placeholder="Try a segment text, e.g. a line from a transcript…"
                               style="flex:1;min-width:260px" oninput="_artifactTry()" aria-label="Try a segment text">
                    </div>
                    <div id="artifactTryResult" class="set-result" data-search-skip style="margin-top:6px"></div>`, {cls: 'attach'})
                + _sRow({key: 'transcription.min_segment_chars', type: 'number', min: 0, unit: 'chars', label: 'Shortest segment',
                    desc: 'Drop segments shorter than this (punctuation not counted). Applies with the filter off, too. 0 = keep all.'})
                + _sRow({key: 'transcription.gain_below_db', type: 'number', max: 0, step: 1, unit: 'dBFS', label: 'Boost quiet recordings below',
                    desc: 'Recordings whose average level is below this get a plain gain to about −26 dB before transcription. 0 = off.',
                    more: '<p>Only a gain — no compression or loudness normalisation, both of which raise the noise in speech pauses. Whisper doesn\'t care about the level above roughly −40 dB; the boost helps real whisper and voice-note cases.</p>'}),
        }) + _sCard({
            title: 'Clean existing transcripts',
            sub: 'The artifact patterns on the transcripts already in the library: first a dry run that only counts and lists, then removing exactly what it found. The database is backed up first, and every removed segment is logged — “Undo” puts it back.',
            body: _sBlock(`<div class="set-actions">
                    <button type="button" class="set-btn" id="artDryRunBtn" onclick="_artDryRun()">🔍 Dry run</button>
                    <button type="button" class="set-btn" id="artStopBtn" onclick="_artStop()" hidden>⏹ Stop</button>
                    <span class="set-hint">With the patterns as they are in the field above — saved or not.</span>
                </div>
                <div id="artifactRun" data-search-skip style="margin-top:10px"></div>`),
        });
    },
    after: (cfg) => { _renderVocabPresets(cfg.transcription || {}); _artStatusLoad(); },
},

{
    id: 'subtitles', group: 'Transcription', icon: 'captions', title: 'Subtitles', hint: 'How subtitle cues are cut',
    intro: 'How a transcript with word timestamps is cut into subtitle cues. Takes effect when a file is transcribed — re-transcribe to change existing transcripts.',
    render: (cfg) => {
        const on = !!_cfgVal('transcription.word_timestamps', false);
        return (on ? '' : `<div class="set-note warn" style="margin-bottom:14px">Word timestamps are off for new jobs (Transcription → Defaults) — without them there are only coarse 20–30 s blocks, and nothing here applies.</div>`)
        + _sCard({
            title: 'Cue length',
            body: _sRow({key: 'transcription.max_segment_duration', type: 'number', unit: 's', step: 0.1, min: 0.5, label: 'Longest cue',
                    desc: 'A longer cue is split at a word boundary.'})
                + _sRow({key: 'transcription.max_segment_words', type: 'number', min: 1, label: 'Most words per cue',
                    desc: 'For languages with spaces between words.'})
                + _sRow({key: 'transcription.max_segment_duration_cjk', type: 'number', unit: 's', step: 0.1, min: 0.5, label: 'Longest cue — Japanese, Chinese, Korean',
                    desc: 'Set apart because these languages are denser: 1.0 gives short, quickly changing cues.'})
                + _sRow({key: 'transcription.max_segment_words_cjk', type: 'number', min: 1, unit: 'chars', label: 'Most characters per cue — Japanese, Chinese, Korean',
                    desc: 'About 4 characters per second of cue length read well: 1-s cues → 8, 3-s cues → 12–15.'}),
        }) + _sCard({
            title: 'Where cues split',
            body: _sRow({key: 'transcription.word_pause_threshold', type: 'number', unit: 's', step: 0.05, min: 0, label: 'Split at pauses from',
                    desc: 'Inside a cue that is over a limit, split at a speech pause at least this long (Japanese, Chinese, Korean: at most 0.15 s).'})
                + _sRow({key: 'transcription.split_natural_only', type: 'toggle', label: 'Only at natural boundaries',
                    desc: 'Split only at pauses and punctuation: the limits make splitting more eager instead of cutting a sentence in the middle. Some cues then run longer than the limits.'})
                + _sBlock(`<details class="set-more"><summary>How cues are cut — order, examples, proven combinations</summary><div class="set-more-body">
                    <p><b>Order.</b> A cue ends after a sentence end (. ! ? 。！？) always; after a comma (, ; 、) once it has some length; at the duration or word limit; at a speech pause.</p>
                    <p><b>Longest cue</b> 3.5 s: a 6-second sentence becomes “I'm sure it was him,” + “but I can't prove it”.</p>
                    <p><b>Most words</b> 8: “And then last night we quickly drove to the station to pick him up” (14 words) → two cues, split at a word.</p>
                    <p><b>Most characters</b> (CJK): 明日の会議は午後三時から始まる (15 characters in 2 s) stays one cue at 15, is split at 8.</p>
                    <p><b>Only at natural boundaries</b>, off: …会議の準備を始めるために集ま | る… — on: the cut waits for the next pause or punctuation mark (in testing 100 % of cue boundaries at punctuation).</p>
                    <p><b>Always:</b> punctuation stays with its sentence; ょ っ ー never start a cue; katakana words stay whole; cues under 0.4 s merge with a neighbour.</p>
                    <table class="set-table"><tr><th>Profile</th><th>Duration / characters (CJK)</th><th>Pause</th><th>Natural boundaries</th></tr>
                        <tr><td>Standard</td><td>3.0 s / 15</td><td>0.3</td><td>off</td></tr>
                        <tr><td>Fine cues</td><td>1.0 s / 8</td><td>0.15</td><td>off</td></tr>
                        <tr><td>Natural reading flow</td><td>1.0–2.0 s / 8–12</td><td>0.1–0.3</td><td>on</td></tr></table>
                </div></details>`),
        });
    },
},

{
    id: 'vocal', group: 'Transcription', icon: 'music', title: 'Vocal isolation', hint: 'Separating voices from music and noise',
    intro: 'Separates the voices from music and noise before transcription, with MelBand Roformer. Clean speech is recognized and skipped by itself.',
    render: (cfg) => _sCard({
        title: 'Defaults for new jobs',
        sub: 'Pre-selected in every transcription dialog.',
        body: _sRow({key: 'vocal_isolation.default_enabled', type: 'toggle', label: 'Vocal isolation', tags: ['jobs'],
                desc: 'A clear gain on music and noise, no harm on clean speech.',
                more: '<p>Measured: +23 points word recall on loud material, no loss on clean material. The model: MelBand Roformer (Kim) — the highest vocal quality measured here and fast, about 1.5 minutes per hour of audio on a 12 GB card. It is the only offered model; BS-Roformer only steps in if it can\'t be loaded. First use downloads it (~200 MB).</p>'})
            + _sRow({label: 'Model', type: 'static', value: 'MelBand Roformer (Kim)',
                desc: 'The only offered model; BS-Roformer only steps in when it can\'t be loaded.'})
            + _sRow({key: 'vocal_isolation.default_save_alongside', type: 'toggle', label: 'Keep the separated tracks', tags: ['jobs'],
                desc: 'Save ' + _sCode('<name>_vocals.mp3') + ' and ' + _sCode('<name>_instrumental.mp3') + ' next to the source file.'}),
    }) + _sCard({
        title: 'Speed',
        body: _sRow({key: 'vocal_isolation.prefilter', type: 'toggle', label: 'Skip clean stretches',
                desc: 'Each chunk is checked first: where the speech pauses are already silent, there is nothing to separate and the original is used. Saves about 40 % of the separation time. Off = always separate.'})
            + _sRow({key: 'vocal_isolation.use_autocast', type: 'toggle', label: 'Half precision (fp16)',
                desc: 'About 1.4× faster with the same result. Turn off only on a GPU without fast fp16. Applies when the separation model is loaded next.'}),
    }) + _sCard({
        title: 'Memory and quality', fold: true,
        body: _sRow({key: 'hardware.sep_chunk_duration_sec', type: 'number', unit: 's', min: 60, step: 60, label: 'Chunk length',
                desc: 'Audio is separated in chunks of this length, and the chunk is the RAM ceiling: 1800 s ≈ 4 GB, however long the file is. Lower it on a machine with little RAM.'})
            + _sRow({key: 'hardware.sep_overlap_sec', type: 'number', unit: 's', min: 0, max: 30, label: 'Chunk overlap',
                desc: 'Chunks overlap by this much and the overlap is cut off again, so no word is lost at a chunk border.'})
            + _sRow({key: 'vocal_isolation.mdxc_overlap', type: 'number', unit: 's', min: 1, max: 30, step: 1, label: 'Window step',
                desc: 'Step between separation windows (a window is 11 s long). Smaller = more overlap: slower, marginally smoother; 11 or more = no overlap, fastest. Applies when the separation model is loaded next.'}),
    }),
},

{
    id: 'speakers', group: 'Transcription', icon: 'voices', title: 'Speakers', hint: 'Diarization, voice vectors, voice map thresholds',
    intro: 'Who speaks when (diarization) and who it is (voice recognition): the diarization worker, the voice store and the thresholds of the voice map.',
    render: (cfg) => _sCard({
        title: 'Diarization',
        sub: 'pyannote, in a worker process of its own.',
        body: _sRow({key: 'transcription.hf_token', type: 'secret', label: 'Hugging Face token', tags: ['secret'],
                placeholder: 'hf_…',
                desc: 'Needed once, to download the gated diarization model. Empty = the ' + _sCode('HF_TOKEN') + ' environment variable, or the model downloaded earlier. The setup guide (Speaker diarization) checks a token and downloads the model.',
                more: '<p>The model, <a href="https://huggingface.co/pyannote/speaker-diarization-community-1" target="_blank" rel="noopener">pyannote/speaker-diarization-community-1</a>, is free but gated: accept its conditions on that page while signed in (access is granted at once), then create a token of type <b>Read</b> at <a href="https://huggingface.co/settings/tokens" target="_blank" rel="noopener">huggingface.co/settings/tokens</a>. A fine-grained token needs “Read access to contents of all public gated repos you can access”. Once the model is downloaded it loads from the cache without a token.</p>'})
            + _sRow({key: 'hardware.diarization_vram_cap_mb', type: 'number', unit: 'MB', min: 0, step: 256, label: 'VRAM limit', tags: ['worker'],
                desc: 'pyannote needs about 1.6 GB; without a limit cuDNN takes most of a free card as scratch space. A file that really needs more runs once without it. 0 = no limit.',
                more: '<p>Measured without the limit: 9.2 GB in one allocation for a voice note of a few seconds, the card at 11.9 of 12 GB. With it: the same segments, and faster (about 200 files in 66 s instead of 88 s).</p>'})
            + _sRow({key: 'hardware.diarization_worker_max_gb', type: 'number', unit: 'GB', min: 0, step: 0.5, label: 'Worker RAM limit',
                desc: 'The worker keeps its model loaded between files, and pyannote holds on to memory. Above this it restarts after the current file (about 5 s). 0 = no limit.'})
            + _sRow({key: 'hardware.diarization_deadline_factor', type: 'number', unit: '× length', min: 0, step: 0.1, label: 'Time limit',
                desc: 'A diarization running longer than this share of the audio length (at least 15 min) counts as hung and is stopped. 0 = no limit.',
                more: '<p>Measured on a 12 GB card: 1.9 h of speech in 92 s, 75× faster than real time. 0.5 therefore leaves 37 times that — the limit catches a hang, not a slow card.</p>'})
            + _sRow({key: 'hardware.diarization_skip_silence', type: 'toggle', label: 'Cut out long silences',
                desc: 'Diarize only the audio with sound in it and map the times back — only done when it saves at least 15 %.'})
            + _sRow({key: 'hardware.diarization_subprocess', type: 'toggle', label: 'Cancellable worker',
                desc: 'Only a separate process can be stopped when pyannote hangs, and frees the GPU completely. Turn off only where no child process can start.'}),
    }) + _sCard({
        title: 'Sentence check',
        sub: 'Each sentence measured on its own against the voices of its recording — catches people trading short sentences that the diarization kept under one speaker.',
        body: _sRow({key: 'speaker.check_sentences', type: 'toggle', label: 'Check each sentence\'s speaker', tags: ['jobs'],
                desc: 'Only with vocal isolation and voice vectors. Names play no part. Every move is kept with the transcript.'})
            + _sRow({key: 'speaker.check_min_sim', type: 'number', step: 0.01, min: 0.3, max: 0.99, label: 'Move a sentence from',
                desc: 'How similar a sentence must be to another voice of the recording to move there.'})
            + _sRow({key: 'speaker.check_margin', type: 'number', step: 0.01, min: 0, max: 0.5, label: 'Lead over its own speaker',
                desc: 'How far that voice must lead the sentence\'s own speaker. Higher = fewer moves, close calls stay as diarized.'})
            + _sRow({key: 'speaker.check_min_sec', type: 'number', step: 0.1, min: 0.5, max: 10, label: 'Shortest sentence checked (s)',
                desc: 'Below about a second a sentence\'s voice can\'t be told.'}),
    }) + _sCard({
        title: 'Voice vectors',
        sub: 'A voice fingerprint per 30-second speaker window — what the voice map and the “who was there” list are built from.',
        body: _sRow({key: 'transcription.speaker_vectors', type: 'toggle', label: 'Record voice vectors', tags: ['jobs'],
                desc: 'Only together with diarization. About 1 s per window.'})
            + _sRow({key: 'transcription.speaker_window_separation', type: 'toggle', label: 'Separate windows of files without vocal isolation',
                desc: 'Vectors of raw and separated audio don\'t compare. For a file transcribed without vocal isolation, each voice window is separated on its own instead (about 1 s per window). Off = such files get no voice vectors.'}),
    }) + _sCard({
        title: 'Voice map and assignment',
        sub: 'Similarities from 0 to 1. The defaults are measured on a real library — the material matters.',
        body: _sRow({key: 'speaker.threshold_grouping', type: 'number', step: 0.01, min: 0.3, max: 0.99, label: 'Voice map: starting threshold',
                desc: 'Where the voice map\'s slider starts: voices at least this similar share a group. Higher = more, smaller groups.',
                more: '<p>0.94: long recordings where all voices share one channel raise every similarity — at 0.71, about 40 % of all voices ended up in one group.</p>'})
            + _sRow({key: 'speaker.assignment_cut', type: 'number', step: 0.01, min: 0.3, max: 0.99, label: 'Assignment: grouping cut',
                desc: 'How the guided assignment groups speakers before asking you — not the voice map\'s slider. Higher = smaller, purer groups, more clicks.',
                more: '<p>0.71: largest group under 2 % of all speakers, 99.8 % purity against about a thousand manual assignments.</p>'})
            + _sRow({key: 'speaker.assignment_spread', type: 'number', step: 0.01, min: 0.3, max: 0.99, label: 'Assignment: pass names on from',
                desc: 'A name you give is passed on to other voices at least this similar. Lower = fewer clicks, more wrong names.',
                more: '<p>With a quarter of the voices named: 0.65 reaches 94 % of them, all correct; 0.60 reaches 95 %, but only ~97 % correct.</p>'})
            + _sRow({key: 'speaker.threshold_gallery', type: 'number', step: 0.01, min: 0.3, max: 0.99, label: 'Recognize named people from',
                desc: 'How similar a voice must be to a named person to be listed as them in a transcript. Lower finds more but adds wrong names — raise it as you name more people.',
                more: '<p>One recording is compared with everyone named, over many windows: hundreds of chances to cross a low threshold by accident. At 0.55, 47 % of recordings got at least one wrong name; at 0.75, 0.6 % — with 96 % of the right names found.</p>'}),
    }) + _sCard({
        title: 'Voice store',
        body: _sBlock('<div id="speakerStore"><div class="set-hint">Loading…</div></div>'),
    }),
    after: () => _speakerStoreLoad(),
},

{
    id: 'hardware', group: 'Transcription', icon: 'cpu', title: 'GPU & queue', hint: 'Device, VRAM, out of memory, queue start and lanes',
    intro: 'Which GPU runs the models, how its memory is shared with other programs, and how the queue starts.',
    render: (cfg) => {
        const hw = _hwData || {};
        return _sCard({
            title: 'This machine',
            // What the app chose for it (model, vocal isolation, files at once)
            body: _sBlock('<div id="hwInfo"></div>'
                + (hw.machine_defaults ? `<div class="set-hint">${esc(hw.machine_defaults)}</div>` : '')),
        }) + _sCard({
            title: 'Compute',
            body: _sRow({key: 'hardware.device', type: 'select', label: 'Device', tags: ['restart'],
                    options: [['auto', `Automatic (${hw.device || '?'} here)`], ['cuda', 'CUDA — NVIDIA'], ['rocm', 'ROCm — AMD (experimental)'],
                              ['cpu', 'CPU'], ['mps', 'MPS — Apple (runs on the CPU for now)']],
                    desc: 'Where Whisper, alignment and diarization run. Automatic uses CUDA or ROCm when present.'})
                + _sRow({key: 'hardware.compute_type', type: 'select', label: 'Precision',
                    options: [['auto', `Automatic (${hw.recommendations?.compute_type || '?'} here)`], ['float16', 'float16 — GPU'],
                              ['int8', 'int8 — small cards, CPU'], ['float32', 'float32 — CPU, most exact']],
                    desc: 'Of the Whisper model. Applies from the next job; unload the models (⏏ in the top bar) to free the old one.'})
                + _sRow({key: 'hardware.gpu_index', type: 'number', min: 0, max: 15, label: 'GPU number', tags: ['restart'],
                    desc: 'Which GPU (0 = the first) runs Whisper and is watched for free VRAM. Alignment, diarization and separation use the first GPU.'}),
        }) + _sCard({
            title: 'Sharing the GPU',
            sub: 'When other programs use the card too — a game, a browser, another AI tool.',
            body: _sRow({key: 'hardware.low_vram', type: 'toggle', label: 'Spare VRAM', tags: ['jobs'],
                    desc: 'Load each model only for its step and free it right after. On by itself for a card of up to 11 GB; for a busy card switch it on. Costs about 7 % speed.'})
                + _sRow({key: 'hardware.model_idle_min', type: 'number', unit: 'min', min: 0, label: 'Unload models when idle',
                    desc: 'Free the GPU after this long without work; the next job loads the models again in a few seconds. 0 = keep them.'})
                + _sRow({key: 'hardware.oom_min_free_mb', type: 'number', unit: 'MB', min: 0, step: 500, label: 'After out of memory, wait for',
                    desc: 'A file that fails with CUDA out of memory stays queued; the careful retry waits until this much VRAM is free. Keep it below the card\'s size. 0 = retry at once.'})
                + _sRow({key: 'hardware.oom_wait_min', type: 'number', unit: 'min', min: 0, label: '…for at most',
                    desc: 'If the VRAM doesn\'t come free in time, or the retry fails too, the queue pauses with a warning. Nothing is lost.'}),
        }) + _sCard({
            title: 'Queue',
            body: _sRow({key: 'queue.autostart', type: 'toggle', label: 'Start with the server', tags: ['restart'],
                    desc: 'Off = after a restart the queue waits, paused, until you press Resume. Nothing in it is lost.'})
                + _sRow({key: 'queue.lanes', type: 'number', min: 1, max: 4, label: 'Files at once', tags: ['restart'],
                    desc: 'Each model is still loaded once and works on one file at a time; the copying, decoding and writing around it overlap, so the GPU stops waiting. 3 measured fastest (short files ×1.8). 1 = one after the other.'})
                + _sRow({key: 'queue.long_file_sec', type: 'number', unit: 's', min: 0, step: 600, label: 'One long file at a time from',
                    desc: 'Files at least this long never run side by side — each holds its whole audio in memory. Short ones still pass. 0 = no limit.'}),
        }) + _sCard({
            title: 'Memory (RAM)', fold: true,
            body: _sRow({key: 'hardware.asr_block_max_sec', type: 'number', unit: 's', min: 60, step: 600, label: 'Whisper in one piece up to',
                    desc: 'Longer files are split into overlapping blocks. Lower it only when RAM is short — the audio takes about 0.23 GB per hour.'})
                + _sRow({key: 'hardware.server_memory_warning_gb', type: 'number', unit: 'GB', min: 0, step: 1, label: 'Warn when the server holds more than',
                    placeholder: '60 % RAM',
                    desc: 'Checked after every job (RAM and swap of the server process) — a warning in the log and the health check, nothing else. Empty = 60 % of this machine\'s memory, 0 = off.'}),
        });
    },
    after: () => _hwInfoRender(),
},

// ── Search & AI ──────────────────────────────────────────────────────────
//
// Four pages, one question each: how searching works (Search — the keyword
// search and the semantic search with its index), which models run (AI
// models — the language models and the embedding model, both servers you
// point the app at), what they do with a transcript (AI tasks), and how
// AI assistants get in (MCP server).

{
    id: 'search', group: 'Search & AI', icon: 'search', title: 'Search', hint: 'Keyword search, semantic search, conversations, word statistics',
    intro: 'Finding what was said. The keyword search finds the words you type; the semantic search finds passages by their meaning, even when they are worded differently; the conversation statistics find whole conversations by who took part and how they talked.',
    render: (cfg) => _sSection({id: 'keyword', title: 'Keyword search',
            sub: 'The search box in the top bar and the Find window.'})
    + _sCard({
        title: 'Results',
        body: _sRow({key: 'search.max_results', type: 'number', min: 1, max: 5000, label: 'Quick search results',
                desc: 'How many hits the search box in the top bar returns (at most 5000). The Find window loads more as you scroll.'})
            + _sRow({key: 'search.context_segments', type: 'number', min: 0, max: 10, label: 'Context around a hit',
                desc: 'Segments shown before and after each quick-search hit. 0 = none.'}),
    }) + _sCard({
        title: 'Typos',
        body: _sRow({key: 'search.fuzzy_enabled', type: 'toggle', label: 'Allow typos',
                desc: 'Also find words with a typo (words of four letters or more). About 30 ms more per search.'})
            + _sRow({key: 'search.fuzzy_threshold', type: 'number', min: 0.5, max: 1, step: 0.05, label: 'Typo strictness',
                desc: '0.5–1, higher = stricter. One wrong letter is always allowed; this sets from which word length two are (0.8 → from 10 letters, 0.5 → from 4).'}),
    }) + _sCard({
        title: 'Search index', fold: true,
        body: _sRow({key: 'database.fts_enabled', type: 'toggle', label: 'Use the full-text index',
                desc: 'The index is kept up to date either way. Off = the search reads the JSON files of the transcript folder instead — slow, and in “Database only” mode it finds nothing. Only for a broken index; rebuild it under Database.'}),
    })
    + _sSection({id: 'ask', title: 'Ask (chat)',
            sub: 'The Ask tab of the search: the language model (Language models → tasks → Chat) answers questions about the library and looks things up itself through the tools of the MCP server — search, transcripts, who speaks where, who talked with whom.'})
    + _sCard({
        title: 'What the chat may do',
        body: _sRow({key: 'chat.allow_actions', type: 'toggle', label: 'Allow actions',
                desc: 'Let the chat use tools that start work: transcribe, download from YouTube, summarize, translate, proofread, the queue (MCP server → Allow work must be on too). Off = it only reads and searches — a question can never start a long job. With it on, words in a transcript (or a video the extension attached) can lead the model to start such work.'})
            + _sRow({key: 'chat.thinking', type: 'toggle', label: 'Think before answering',
                desc: 'For models that can (Qwen3 and the like): slower, sometimes more careful. The thinking is shown folded above the answer.'})
            + _sRow({key: 'chat.temperature', type: 'number', min: 0, max: 2, step: 0.1, label: 'Temperature',
                desc: 'Low = sticks to what the tools found. 0.3 is a good value.'}),
    })
    + _sSection({id: 'semantic', title: 'Semantic search',
            sub: 'Finds what was SAID, by its meaning: passages of the transcripts — about a minute of talk, with speaker names — embedded by a large text model. The Find window shows them below the keyword hits, Ask answers from them, and AI clients ask it directly (MCP server). It needs an embedding model (AI models) and an index of the transcripts, built here. What is seen or heard in the recordings themselves, also where nobody talks: ' + _sGo('media', 'Images & sounds')})
    + _sCard({
        title: 'Index',
        sub: 'A SQLite file next to the database that can be rebuilt at any time.',
        body: _sBlock('<div id="ragStatusCard"><div class="set-hint">Loading…</div></div>', {skip: true})
            + _sBlock('<div id="ragCoverageCard"></div><div id="ragPendingWrap" data-search-skip></div>', {skip: true})
            + _sBlock(`<div class="set-actions">
                    <button type="button" class="set-btn primary" id="ragBuildBtn" onclick="_ragBuild(false, this)">🧠 Embed new transcripts</button>
                    <button type="button" class="set-btn" id="ragStopBtn" onclick="_ragBuildStop()" hidden>⏹ Stop</button>
                    <button type="button" class="set-btn" onclick="_ragBuild(true, this)">🔁 Rebuild everything</button>
                    <button type="button" class="set-btn" onclick="_loadRagStatus()">↻ Refresh</button>
                    <button type="button" class="set-btn danger" onclick="_ragClear(this)">🗑 Empty the index</button></div>
                <div class="set-hint">“Embed new transcripts” only does what is missing or changed, and can be stopped and continued. Saves the values of this page first.</div>
                <div id="ragBuildProgress" class="set-result" data-search-skip style="margin-top:8px"></div>`),
    }) + _sCard({
        title: 'Settings',
        body: _sRow({key: 'rag.enabled', type: 'toggle', label: 'Show coverage',
                desc: 'Show how much of the library is embedded, in the statistics. Search and indexing work either way.'})
            + _sRow({key: 'rag.rrf_k', type: 'number', min: 1, label: 'Hybrid search: rank fusion k',
                desc: 'How the hybrid search merges the semantic and the keyword ranking: higher = flatter. 60 is the usual value.'})
            + _sRow({key: 'rag.path', type: 'text', label: 'Index file', placeholder: 'rag.db next to the database', wide: true,
                desc: 'Empty = ' + _sCode('rag.db') + ' next to the main database. Download a copy under Database → Exports.'}),
    }) + _sCard({
        title: 'Chunking', fold: true,
        sub: 'How transcripts are cut into the pieces that get embedded. A change re-embeds the whole library on the next run.',
        body: _sRow({key: 'rag.chunk_max_chars', type: 'number', unit: 'chars', min: 100, step: 100, label: 'Chunk length',
                desc: 'Target length of a chunk (a soft limit). 1000–1500 balances context and precision.'})
            + _sRow({key: 'rag.chunk_min_chars', type: 'number', unit: 'chars', min: 0, step: 50, label: 'Shortest chunk',
                desc: 'A chunk doesn\'t end before this length.'})
            + _sRow({key: 'rag.chunk_overlap_segments', type: 'number', min: 0, max: 10, label: 'Overlap',
                desc: 'Segments repeated at the start of the next chunk.'})
            + _sRow({key: 'rag.chunk_on_speaker_change', type: 'toggle', label: 'End chunks at speaker changes',
                desc: 'Prefer to end a chunk where the speaker changes.'})
            + _sRow({key: 'rag.include_speaker_labels', type: 'toggle', label: 'Speaker names in chunks',
                desc: 'Each line starts with its speaker — helps questions like “what did Alice say about …”.'}),
    }) + _sCard({
        title: 'Try the search', fold: true,
        sub: 'With the saved settings — hybrid, semantic only or keyword only.',
        body: _sBlock(`<div class="set-actions">
                <input class="set-input" id="ragSearchInput" placeholder="A question or words…" style="flex:1;min-width:220px"
                       onkeydown="if(event.key==='Enter'){event.preventDefault();_ragSearch();}">
                <select class="set-select" id="ragSearchMode" style="width:auto" aria-label="Search mode">
                    <option value="hybrid">Hybrid (semantic + keyword)</option>
                    <option value="vector">Semantic only</option>
                    <option value="fts">Keyword only</option>
                </select>
                <button type="button" class="set-btn" onclick="_ragSearch()">🔎 Search</button></div>
            <div id="ragSearchResults" data-search-skip style="margin-top:10px"></div>`),
    })
    + _sSection({id: 'conversations', title: 'Conversations',
            sub: 'How people talk in every transcript — words per sentence, speaker changes, who talks how much, the longest monologue — counted from the segments alone, without a model. The Find window\'s Conversations tab, the Conversation card in a recording\'s Statistics tab, Ask and AI clients find conversations by them.'})
    + _sCard({
        title: 'Conversation statistics',
        body: _sRow({key: 'conversations.enabled', type: 'toggle', label: 'Conversation statistics',
                desc: 'Count them for every transcript and find conversations by them. Off = nothing is counted, and the Conversations tab says so.'})
            + _sRow({key: 'conversations.auto_run', type: 'toggle', label: 'Count in the background',
                desc: 'Count the transcripts that have no numbers yet — a minute after a start, then every half hour. A large library takes minutes of one CPU core; a new transcript is counted when it is written either way.'})
            + _sRow({key: 'conversations.min_share', type: 'number', min: 0, max: 0.5, step: 0.01, label: 'Takes part from',
                desc: 'A person takes part in a conversation when they speak at least this share of it (0.05 = 5 %) — less is a remark, not a part. Used when the Conversations tab, Ask or Find look for people.'}),
    })
    + _sSection({id: 'wordstats', title: 'Word statistics',
            sub: 'The most used words in the statistics of a folder or a transcript.'})
    + _sCard({
        title: 'Filler words',
        body: _sRow({key: 'search.stopword_languages', type: 'checks', label: 'Leave out filler words of',
                options: [['en', 'English'], ['de', 'German']],
                desc: 'Filler words — “and”, “the”, “because” and their German counterparts — don\'t count in the top words.'}),
    }),
    open: () => _loadRagStatus(),
},

{
    id: 'media', group: 'Search & AI', icon: 'play', title: 'Images & sounds', hint: 'Find moments by what is seen or heard',
    intro: 'Finds moments by what is SEEN or HEARD in the recordings themselves — a still every 20 s, 10 s clips, 10 s of sound and the words said there, all in one vector space (EmbeddingGemma 2). The Find window\'s tab “Images & sounds” asks it with a description, a hit (“Similar”) or a file of your own. The model runs in an environment of its own; the index is built here, as a guest on the GPU the queue works on.',
    render: (cfg) => _sSection({id: 'media-index', title: 'Index',
            sub: 'What it is for, how far it is, what is left and what that takes.'})
    + _sCard({
        title: 'Images & sounds or the semantic search?',
        sub: 'Two indexes for two questions — both are kept.',
        body: _sBlock(_mediaCompare()),
    })
    + _sCard({
        title: 'Index',
        sub: _sCode('media.db') + ' next to the database — made again from the recordings at any time.',
        body: _sRow({key: 'media_search.enabled', type: 'toggle', label: 'Images & sounds',
                desc: 'Off = the tab says so, nothing is indexed and no process starts.'})
            + _sBlock('<div id="msSetEnv"><div class="set-hint">Loading…</div></div><div id="msSetIndex" style="margin-top:12px"></div>', {skip: true})
            + _sBlock('<div id="msSetCoverage"></div>', {skip: true})
            + _sBlock(`<div class="set-actions">
                    <button type="button" class="set-btn primary" id="msSetStart" onclick="_mediaStart(false, this)">🖼 Index</button>
                    <button type="button" class="set-btn" id="msSetStop" onclick="_mediaStop()" hidden>⏹ Stop</button>
                    <button type="button" class="set-btn" onclick="_mediaStart(true, this)">🔁 Rebuild everything</button>
                    <button type="button" class="set-btn" onclick="_mediaLoad()">↻ Refresh</button>
                    <button type="button" class="set-btn danger" onclick="_mediaClear(this)">🗑 Empty the index</button></div>
                <div class="set-hint">“Index” does what is missing, newest recordings first, and can be stopped and continued — a kind switched on later is only added to the moments that are there. Saves the values of this page first.</div>
                <div id="msSetRun" class="set-result" data-search-skip style="margin-top:8px"></div>
                <div id="msSetEstimate" data-search-skip style="margin-top:12px"></div>`),
    }) + _sCard({
        title: 'What is indexed',
        sub: 'A kind switched on later is added to the moments that are there; the other values make a recording start over when “Index” runs.',
        body: _sRow({key: 'media_search.kinds', type: 'checks', label: 'Kinds',
                options: [['image', 'Stills'], ['video', 'Clips'], ['videoaudio', 'Clips with sound'], ['audio', 'Sounds'], ['speech', 'Spoken text']],
                desc: 'Stills and clips find what is seen, sounds what is heard; a clip with its sound is one vector of both. <b>Spoken text</b> is the words of the 10 s around a moment, in the same space: it shows what is said where something is seen or heard, and lets a picture or a sound find words. What was said — whole passages, with their context — is the semantic search\'s job (' + _sGo('search#semantic', 'Search') + ').',
                more: 'Cost per video moment at 4 frames: clips with sound ~43 % of the GPU time (a second clip), clips ~35 %, stills ~15 %, sounds ~5 %, spoken text ~1 %. Measured: with 4 frames a clip with its sound is cosine 0.92 to the clip alone — its own kind. Measured: weighing the sound list by the spoken words made it worse (mean AP 0.251 → 0.243 on 16 labeled sound kinds — 80 % of the sound windows have speech, lyrics included), so every kind stays a list of its own. A kind switched off stays in the index until “Rebuild everything”.'})
            + _sRow({key: 'media_search.folders', type: 'lines', label: 'Only these folders',
                placeholder: '/mnt/media/recordings', desc: 'One folder per line. Empty = the whole library — check the estimate above first: a large archive takes weeks of GPU time.'})
            + _sRow({key: 'media_search.auto_index', type: 'toggle', label: 'Index new recordings by itself',
                desc: 'Looks for new transcriptions in scope every ten minutes and indexes them.'})
            + _sRow({key: 'media_search.video_interval_sec', type: 'number', unit: 's', min: 2, max: 3600, label: 'A moment every',
                desc: 'One still, clip and sound per this many seconds of a video. Halving it doubles time and space.'})
            + _sRow({key: 'media_search.dedupe_similarity', type: 'number', min: 0, max: 1, step: 0.01, label: 'Skip repeats from',
                desc: 'A moment that looks like the last kept one (cosine at or above this) gets no still and no clip — a webcam stream over hours. Its sound still counts. 1 = keep every moment.'})
            + _sRow({key: 'media_search.clip_frames', type: 'select', label: 'Frames per clip',
                options: [[0, 'Automatic — 8 up to 500 hours of video, else 4'], [2, '2'], [4, '4'], [6, '6'], [8, '8'], [12, '12'], [16, '16']],
                desc: 'More frames see more of what happens, and cost proportionally more. Automatic decides at the first run and keeps it.',
                more: 'Measured against 16 frames: 8 frames cosine 0.99, 4 frames 0.98, 2 frames 0.965 — 4 frames ran at 5.5 clips/s on a 12 GB card, 8 at half that. Another value makes only the clips again.'})
            + _sRow({key: 'media_search.clip_sec', type: 'number', unit: 's', min: 2, max: 60, label: 'Clip length',
                desc: 'A clip runs from 30 % before its moment to 70 % after it.'})
            + _sRow({key: 'media_search.sound_window_sec', type: 'number', unit: 's', min: 2, max: 120, label: 'Sound window',
                desc: 'The sound around a video moment, and back to back through audio files.'})
            + _sRow({key: 'media_search.dims', type: 'select', label: 'Numbers per vector',
                options: [[768, '768 — all (8-bit numbers: 768 bytes a vector)'], [512, '512'], [256, '256'], [128, '128']],
                desc: 'Fixed when the index is first written; another value needs “Rebuild everything”. A search can still ask with fewer.',
                more: 'Measured on 16 labeled sound kinds: mean AP 0.251 with all 768 numbers, 0.189 with 512, 0.201 with 256. Stored as 8-bit numbers, all 768 take less disk than 512 as float16 did — the rounding moves a cosine by about 0.001.'}),
    })
    + _sSection({id: 'media-running', title: 'Running', sub: 'The environment, the model and how it shares the machine.'})
    + _sCard({
        title: 'Environment',
        sub: 'EmbeddingGemma 2 needs newer libraries than the app\'s own environment, so it runs in one of its own.',
        body: _sBlock('<div id="msSetEnvDetail"></div>', {skip: true})
            + _sRow({key: 'media_search.environment', type: 'dir', label: 'Environment folder', placeholder: 'media-env in the data folder',
                desc: 'Empty = ' + _sCode('media-env') + ' in the data folder. “Install” creates it here.'})
            + _sRow({key: 'media_search.model', type: 'text', label: 'Model', placeholder: 'google/embeddinggemma-2',
                desc: 'A Hugging Face id or a local folder. Another model needs “Rebuild everything”.'}),
    }) + _sCard({
        title: 'GPU, CPU and disk', fold: true,
        body: _sRow({key: 'media_search.index_device', type: 'seg', label: 'Index on', options: [['cuda', 'GPU'], ['cpu', 'CPU']],
                desc: 'The CPU indexes about ten times fewer moments an hour.'})
            + _sRow({key: 'media_search.index_while_queue_runs', type: 'toggle', label: 'Share the GPU with the queue',
                desc: 'On = index between the queue\'s steps (the GPU gate gives turns; Whisper never runs next to it). Off = only while no transcription runs.'})
            + _sRow({key: 'media_search.vram_need_mb', type: 'number', unit: 'MB', min: 0, step: 100, label: 'VRAM a turn needs',
                desc: 'A turn starts next to the queue\'s other steps only with this much free. Measured 1.6–2.1 GB.'})
            + _sRow({key: 'media_search.yield_after_sec', type: 'number', unit: 's', min: 0, label: 'Keep a turn at least',
                desc: 'Then the card goes back as soon as the queue waits for it.'})
            + _sRow({key: 'media_search.min_free_gb', type: 'number', unit: 'GB', min: 0, step: 1, label: 'Leave free on the disk',
                desc: 'An index run waits while the disk of the index or its stills has less free than this — the queue needs room for long recordings (Diagnostics asks 20 GB of the work folder). 0 = never wait.'})
            + _sRow({key: 'media_search.batch', type: 'number', min: 1, max: 64, label: 'Pictures per pass',
                desc: 'How many stills or clips go through the model at once. More is faster and needs more VRAM.'})
            + _sRow({key: 'media_search.chunk_moments', type: 'number', min: 1, max: 512, label: 'Moments per request',
                desc: 'A stop or a yield waits for the running request.'})
            + _sRow({key: 'media_search.grab_workers', type: 'number', min: 1, max: 32, label: 'Decoding threads',
                desc: 'ffmpeg processes that seek the frames and sounds, one thread each.'})
            + _sRow({key: 'media_search.parallel_files', type: 'number', min: 1, max: 8, label: 'Recordings at once',
                desc: 'While one recording\'s frames are decoded, another\'s go through the model — with one the GPU mostly waits. No more VRAM; decoding threads × this many ffmpeg processes.'})
            + _sRow({key: 'media_search.cpu_threads', type: 'number', min: 1, max: 128, label: 'CPU threads for queries',
                desc: 'Queries run on the CPU: a text in ~0.1 s, an image in ~3.5 s, a video in ~10 s. Takes effect when the service starts next.'})
            + _sRow({key: 'media_search.idle_stop_sec', type: 'number', unit: 's', min: 0, label: 'Stop the service after',
                desc: 'Without a request — it holds 1–3 GB of RAM. 0 = never.'})
            + _sRow({key: 'media_search.start_timeout_sec', type: 'number', unit: 's', min: 10, label: 'Start timeout',
                desc: 'How long the service may take to answer after its start (it loads in ~4 s).'})
            + _sRow({key: 'media_search.port', type: 'number', min: 1, max: 65535, label: 'Port',
                desc: 'Of the service on 127.0.0.1. Takes effect when it starts next.'}),
    })
    + _sSection({id: 'media-llm', title: 'Language model', sub: 'What a model that sees pictures adds (AI models; Qwen3.6 and the like with a vision projector).'})
    + _sCard({
        title: 'Descriptions',
        sub: 'The model describes each still in one sentence; the sentences are searched as text — a kind of its own in the tab (“Descriptions”).',
        body: _sBlock('<div id="msSetDescribe"><div class="set-hint">Loading…</div></div>', {skip: true})
            + _sBlock(`<div class="set-actions">
                    <button type="button" class="set-btn" id="msSetDescribeBtn" onclick="_mediaDescribe(this)">📝 Describe the stills</button>
                    <button type="button" class="set-link" onclick="openLlmQueue()">LLM queue →</button></div>
                <div class="set-hint">Runs in the LLM queue (it takes turns with summaries and deep searches) — measured: the top 10 by description held 79 % matches, by the picture 62 %.</div>`),
    })
    + _sCard({
        title: 'Language model',
        body: _sRow({key: 'media_search.llm_translate_sounds', type: 'toggle', label: 'Ask sounds in English',
                desc: 'The model turns the query into a short English sound label — “lots of people chatting in a crowded pub” → “busy pub crowd”.',
                more: 'Measured on 16 labeled sound kinds: queries in the library\'s own language mean AP 0.115, the model\'s labels 0.325 (~0.5 s per query). Pictures keep the query as typed.'})
            + _sRow({key: 'media_search.llm_check_count', type: 'number', min: 1, max: 200, label: 'Pictures to check',
                desc: '“Check with the language model” judges this many of the best stills or clips, one picture per request (~4 s each).',
                more: 'Measured on 12 queries: precision of the top 10 0.62 → 0.82, nDCG@10 0.67 → 0.96; what it called a match was right 93 % of the time.'})
            + _sRow({key: 'media_search.llm_describe_limit', type: 'number', min: 1, label: 'Describe per run',
                desc: 'A describing run (button above) writes at most this many descriptions — ~5.4 s each, on the language model\'s machine.'})
            + _sRow({key: 'media_search.llm_describe_after_index', type: 'toggle', label: 'Describe after every index run',
                desc: 'Queues a describing run of the new stills in the LLM queue when an index run ends.'})
            + _sRow({key: 'media_search.llm_language', type: 'text', label: 'Descriptions in', placeholder: 'English',
                desc: 'The language the model describes in.'})
            + _sRow({key: 'media_search.llm_model', type: 'text', label: 'Model', placeholder: 'the default of AI models',
                desc: 'Empty = the default model. It must see pictures for checks and descriptions.'}),
    })
    + _sSection({id: 'media-files', title: 'Files', sub: 'Where the index and its pictures are kept.'})
    + _sCard({
        title: 'Files', fold: true,
        body: _sRow({key: 'media_search.path', type: 'text', label: 'Index file', placeholder: 'media.db next to the database', wide: true,
                desc: 'Empty = ' + _sCode('media.db') + ' next to the main database. A disk with room for the whole library: the estimate above says how much.'})
            + _sRow({key: 'media_search.cache_dir', type: 'dir', label: 'Stills and clips', placeholder: 'media in the cache folder',
                desc: 'Thumbnails of the stills (~22 KB each) and the preview clips.'})
            + _sRow({key: 'media_search.thumbs_max_mb', type: 'number', unit: 'MB', min: 0, step: 500, label: 'Keep stills up to',
                desc: 'An index run writes thumbnails until they take this much; beyond it a still is cut from the recording when it is first shown (~0.5 s). 0 = no limit.'})
            + _sRow({key: 'media_search.cache_max_mb', type: 'number', unit: 'MB', min: 50, step: 50, label: 'Keep preview clips up to',
                desc: 'They are cut when played; the oldest go first.'})
            + _sRow({key: 'media_search.thumb_width', type: 'number', unit: 'px', min: 160, max: 1280, step: 20, label: 'Thumbnail width',
                desc: 'Also what the language model sees when it checks or describes.'})
            + _sRow({key: 'media_search.results_per_kind', type: 'number', min: 1, max: 200, label: 'Hits per kind',
                desc: 'How many hits each list of the tab gets (it shows 12 and unfolds the rest).'}),
    }),
    open: () => _mediaLoad(),
},

{
    id: 'audio', group: 'Search & AI', icon: 'music', title: 'Audio analysis', hint: 'Same audio, duplicates, waveform and loudness',
    intro: 'Every recording\'s sound, analyzed once: its fingerprints — so a snippet of a few seconds finds every recording that holds it, at the exact second (the Find window\'s tab “Same audio”) —, the recordings that are the same or contain one another, its waveform for the timeline of the Clips tab and its loudness. It runs on the CPU; new recordings are analyzed while they are transcribed, from the sound the transcription decodes anyway.',
    render: (cfg) => _sSection({id: 'audio-index', title: 'Index', sub: 'What it holds, and the run over the recordings transcribed before.'})
    + _sCard({
        title: 'Index',
        sub: _sCode('audio.db') + ' next to the database, its fingerprints in ' + _sCode('audio.db.fp') + ' — made again from the recordings at any time. About 0.3 MB an hour of sound.',
        body: _sRow({key: 'audio_analysis.enabled', type: 'toggle', label: 'Audio analysis',
                desc: 'Off = nothing is analyzed, and the tab says so.'})
            + _sRow({key: 'audio_analysis.during_transcription', type: 'toggle', label: 'While transcribing',
                desc: 'Analyze every recording as it is transcribed — from the 16 kHz sound the transcription has just decoded, on a thread of its own (~4 s an hour of sound): it is never read twice for this.'})
            + _sBlock('<div id="amSetState"><div class="set-hint">Loading…</div></div>', {skip: true})
            + _sBlock(`<div class="set-actions">
                    <button type="button" class="set-btn primary" id="amSetStart" onclick="_audioStart()">≋ Analyze the library</button>
                    <button type="button" class="set-btn" id="amSetStop" onclick="_audioStop()" hidden>⏹ Stop</button>
                    <button type="button" class="set-btn" onclick="_audioLoad(true)">↻ Refresh</button>
                    <button type="button" class="set-btn" onclick="_audioRetry()">↺ Try unreadable ones again</button>
                    <button type="button" class="set-btn danger" onclick="_audioClear(this)">🗑 Empty</button></div>
                <div class="set-hint">“Analyze the library” reads every recording the library scan found that has no analysis yet, newest first — it can be stopped, and goes on where it stopped, also after a restart. Saves the values of this page first.</div>
                <div id="amSetRun" class="set-result" data-search-skip style="margin-top:8px"></div>
                <div id="amSetEstimate" data-search-skip style="margin-top:12px"></div>`),
    }) + _sCard({
        title: 'The background run',
        sub: 'A guest on the machine: the lowest CPU priority, and as much reading as you allow it.',
        body: _sRow({key: 'audio_analysis.folders', type: 'lines', label: 'Only these folders',
                placeholder: '/mnt/media/recordings', desc: 'One folder per line. Empty = the whole library.'})
            + _sRow({key: 'audio_analysis.auto_run', type: 'toggle', label: 'Run by itself',
                desc: 'Starts after a server start and every ten minutes when the library scan found recordings without an analysis — those not transcribed, too. Off: new recordings are analyzed while they are transcribed.'})
            + _sRow({key: 'audio_analysis.parallel_files', type: 'number', min: 1, max: 16, label: 'Recordings at once',
                desc: 'Read and analyzed at the same time. On a network share two keep the link busy; more only help on fast local disks.'})
            + _sRow({key: 'audio_analysis.max_read_mb_s', type: 'number', unit: 'MB/s', min: 0, step: 5, label: 'Read at most',
                desc: 'In all, across the recordings at once. On a network share the queue reads from it too — 0 = no limit.'})
            + _sRow({key: 'audio_analysis.pause_for_queue', type: 'toggle', label: 'Wait while transcribing',
                desc: 'Read nothing while a transcription runs.'})
            + _sRow({key: 'audio_analysis.min_free_gb', type: 'number', unit: 'GB', min: 0, step: 1, label: 'Leave free on the disk',
                desc: 'The run waits while the disk of the index has less free than this.'}),
    }),
    open: () => _audioLoad(),
},

{
    id: 'models', group: 'Search & AI', icon: 'layers', title: 'AI models', hint: 'Language models and the embedding model',
    intro: 'The models behind the AI features, and where they run: the language models that summarize, correct and translate, and the embedding model behind the semantic search. The speech model (Whisper) is chosen under Transcription.',
    owns: ['llm.providers', 'llm.tasks', 'llm.models'],
    render: (cfg) => {
        const local = _sTruthy(cfg.embedding?.local);
        return _sSection({id: 'llm', title: 'Language models',
                sub: 'The servers that write the summaries, corrections and translations — llama.cpp, Ollama, OpenAI or anything OpenAI-compatible.'})
        + _sCard({
            title: 'Providers',
            sub: 'The first one is the default. “Test” fetches its models and context size. A saved key stays as long as its field is left empty.',
            actions: '<button type="button" class="set-btn sm" onclick="_provAdd()">+ Add provider</button>',
            body: _sBlock('<div id="provList"></div>'),
        }) + _sCard({
            title: 'Which model does what',
            sub: 'Each task can run on its own provider and model — say, a small fast model for correction and a large one for summaries.',
            body: _sBlock('<div id="provTasks"></div>')
                + _sBlock(`<div class="set-actions"><button type="button" class="set-btn" onclick="_testLlmConnection()">🔌 Try a summary</button>
                    <span id="llmTestResult" class="set-result"></span></div>
                    <div class="set-hint">Sends a one-line test through the summary task with the saved settings — the provider and model that answer are named.</div>`),
        }) + _sCard({
            title: 'Model shortcuts', fold: true,
            sub: 'Short names for models, to switch quickly in the summarize dialog and for the automatic summary (AI tasks).',
            body: _sBlock('<div id="llmModelsList" class="set-items"></div>'
                + `<div class="set-add"><input class="set-input mono" id="llmNewModelName" placeholder="Name, e.g. qwen3">
                    <input class="set-input mono" id="llmNewModelId" placeholder="Model id, e.g. qwen3:32b"
                        onkeydown="if(event.key==='Enter'){event.preventDefault();_addLlmModel();}">
                    <button type="button" class="set-btn" onclick="_addLlmModel()">+ Add</button></div>`),
        }) + _sCard({
            title: 'Limits and timeouts', fold: true,
            sub: 'For every task and every provider.',
            body: _sRow({key: 'llm.context_size', type: 'number', unit: 'tokens', min: 0, step: 1024, label: 'Context window',
                    desc: '0 = ask the server (recommended). A value caps every provider and is sent to Ollama as num_ctx.'})
                + _sRow({label: 'Detected context', type: 'static', value: '<span id="llmCtxDetected">–</span>',
                    desc: 'What the default provider reports (refreshed every 5 minutes).'})
                + _sRow({key: 'llm.chunking.overlap_segments', type: 'number', min: 0, max: 20, label: 'Overlap between parts',
                    desc: 'A transcript too long for the context window is split into parts; this many segments of the previous part come along as context.'})
                + _sRow({key: 'llm.timeout', type: 'number', unit: 's', min: 10, label: 'Timeout per request',
                    desc: 'Local models can need minutes for a long prompt. Correction and translation parts stop at 900 s.'})
                + _sRow({key: 'llm.retries', type: 'number', min: 0, max: 10, label: 'Retries',
                    desc: 'Extra attempts after network errors, timeouts or 5xx, with a growing pause; 4xx never. Streamed summaries are not retried.'}),
        })
        + _sSection({id: 'embedding', title: 'Embedding model',
                sub: 'Turns transcript chunks and search questions into vectors for the semantic search (Search). It runs on a server you set up, or the app starts a llama.cpp server itself when needed.'})
        + _sCard({
            title: 'Where the model runs',
            actions: '<button type="button" class="set-btn sm" onclick="_embGuideToggle()">ℹ Setup guide</button>',
            body: _sBlock('<div id="embGuide" hidden></div>')
                + _sRow({key: 'embedding.local', type: 'seg', label: 'Mode', valueType: 'bool',
                    options: [[false, 'A server you run'], [true, 'Started by the app']],
                    onchange: '_embToggleMode()',
                    desc: 'Switching means a full rebuild of the index: vectors of different models don\'t compare.'}),
        }) + _sCard({
            title: 'Your server', id: 'embExternal', hidden: local,
            body: _sRow({key: 'embedding.endpoint', type: 'text', label: 'Server address', placeholder: 'http://localhost:8080/v1', wide: true,
                    desc: 'The OpenAI-compatible address with /v1 — llama.cpp: ' + _sCode('http://localhost:8080/v1') + ' · Ollama: ' + _sCode('http://localhost:11434/v1') + ' · OpenAI: ' + _sCode('https://api.openai.com/v1') + '.',
                    more: '<p>Ollama\'s native path sends one request per text, /v1 a whole batch: measured 4.0 against 73.0 chunks per second — the same computation, 18 times faster.</p>'})
                + _sRow({key: 'embedding.api_key', type: 'secret', label: 'API key', tags: ['secret'], placeholder: 'empty for local servers',
                    desc: 'A saved key shows as dots — leave it as it is to keep it.'})
                + _sRow({key: 'embedding.model', type: 'text', label: 'Model', placeholder: 'qwen3-embedding',
                    desc: 'The name the server knows it by — e.g. qwen3-embedding, bge-m3, text-embedding-3-small. A different model re-embeds everything on the next index run.'}),
        }) + _sCard({
            title: 'Server started by the app', id: 'embLocal', hidden: !local,
            sub: 'A llama-server with a GGUF file, started on the first embedding and stopped again when idle — it sits on the same card as Whisper.',
            body: _sBlock('<div id="embServerCard"><div class="set-hint">Loading…</div></div>'
                    + `<div class="set-actions" style="margin-top:10px">
                        <button type="button" class="set-btn" id="embBtnStart" onclick="_embServerStart(this)">▶ Start now</button>
                        <button type="button" class="set-btn" id="embBtnStop" onclick="_embServerStop(this)">⏹ Stop and free the VRAM</button>
                        <button type="button" class="set-btn" onclick="_embLoadStatus()">↻ Refresh</button></div>`, {skip: true})
                + _sRow({key: 'embedding.local_model', type: 'text', label: 'Model file (GGUF)', wide: true,
                    placeholder: 'models/gguf/Qwen3-Embedding-4B-Q8_0.gguf',
                    desc: 'A relative path starts at the app folder. A different model needs a full rebuild of the index.'})
                + _sRow({key: 'embedding.local_search_on_cpu', type: 'toggle', label: 'Searches on the CPU',
                    desc: 'A search starts the server on the CPU (about 0.2 s per question), so searching never takes VRAM from a transcription. Index runs still use the GPU; a server already running on the GPU is used as it is.',
                    more: '<p>A question is ~8 new tokens (the instruction prefix is cached), a chunk ~415: on the CPU a search costs 214 ms, an index run would be 80 times too slow. A search on the GPU can grab ~6 GB of VRAM while a job runs and push it out of memory.</p>'})
                + _sRow({key: 'embedding.local_idle_stop_sec', type: 'number', unit: 's', min: 0, label: 'Stop when idle for',
                    desc: 'Frees the VRAM after this long without an embedding. 0 = never.'}),
        }) + _sCard({
            title: 'Server tuning', id: 'embLocalTune', hidden: !local, fold: true,
            sub: 'VRAM, port and batch of the server the app starts — the defaults fit a 12 GB card.',
            body: _sRow({key: 'embedding.local_vram_reserve_mb', type: 'number', unit: 'MB', min: 0, step: 500, label: 'VRAM reserve',
                    desc: 'If less than this is free at start — or a transcription is running — the server starts on the CPU instead. Only with GPU layers −1. 0 = no reserve.'})
                + _sRow({key: 'embedding.local_gpu_layers', type: 'number', min: -1, label: 'GPU layers',
                    desc: '−1 = as many as fit into the free VRAM (recommended), 0 = CPU only, more = fixed — and the start fails whenever the card is busy.'})
                + _sRow({key: 'embedding.local_port', type: 'number', min: 1024, max: 65535, label: 'Port',
                    desc: 'Stop the server before changing it. 8080 is usually taken by the chat model.'})
                + _sRow({key: 'embedding.local_batch', type: 'number', unit: 'tokens', min: 128, step: 128, label: 'Batch',
                    desc: 'Must hold the longest chunk in one pass; 2048 covers the default chunks. More only costs VRAM (measured: 4096 and 8192 equally fast).'})
                + _sRow({key: 'embedding.local_ctx', type: 'number', unit: 'tokens', min: 128, step: 128, label: 'Context',
                    desc: 'The server\'s context window; at least as large as the batch.'})
                + _sRow({key: 'embedding.local_start_timeout_sec', type: 'number', unit: 's', min: 10, label: 'Start timeout',
                    desc: 'How long loading the model may take before the start is given up.'}),
        }) + _sCard({
            title: 'Questions and requests',
            body: _sRow({key: 'embedding.question_template', type: 'textarea', rows: 3, label: 'Question template',
                    desc: 'Prefix for search questions — ' + _sCode('{question}') + ' is replaced by the question. It depends on the model: Qwen3 ' + _sCode('Instruct: …\\nQuery:{question}') + ' · nomic ' + _sCode('search_query: {question}') + ' · bge-m3 empty. Changing it needs no re-embedding.',
                    more: '<p>Modern embedding models are trained for asymmetric retrieval: the question gets an instruction, the text stays as it is. Measured on several thousand chunks: with the prefix the best hit changed for 6 of 8 questions, and the gap between best and fifth grew 1.28×.</p>'})
                + _sRow({key: 'rag.batch_size', type: 'number', min: 1, max: 512, label: 'Chunks per request',
                    desc: 'How many chunks an index run sends at once. Larger = fewer requests, which helps a remote server; no gain on a local one.'})
                + _sRow({key: 'embedding.timeout', type: 'number', unit: 's', min: 5, label: 'Timeout per request',
                    desc: 'How long one embedding request may take.'}),
        }) + _sCard({
            title: 'Test and measure',
            sub: 'Both save the values of this page first.',
            body: _sBlock(`<div class="set-actions">
                    <button type="button" class="set-btn" onclick="_testEmbeddingConnection()">🧬 Test</button>
                    <button type="button" class="set-btn" onclick="_embBenchmark(this)">⏱ Measure throughput</button>
                    <span id="embTestResult" class="set-result"></span></div>
                <div class="set-hint">The test asks for a real vector — many servers answer /v1/models but refuse /v1/embeddings without ${_sCode('--embedding')}. The throughput is measured on real chunks of your library.</div>
                <div id="embBenchResult" data-search-skip style="margin-top:10px"></div>`),
        });
    },
    after: (cfg) => {
        _provLoad(); _llmStatusLoad(); _renderLlmModels(cfg.llm || {});
        if (_sTruthy(cfg.embedding?.local)) _embLoadStatus();
    },
},

{
    id: 'tasks', group: 'Search & AI', icon: 'tasks', title: 'AI tasks', hint: 'Summaries, correction, translation',
    intro: 'What the language models do with a transcript: summaries, the 🪄 correction and translations. Which model runs each task is set under AI models.',
    owns: ['llm.presets'],
    render: (cfg) => {
        const l = cfg.llm || {};
        const presets = Object.keys(l.presets || {});
        const models = Object.entries(l.models || {}).filter(([, id]) => typeof id === 'string');
        return _sSection({id: 'summaries', title: 'Summaries',
                sub: 'The 🤖 summarize dialog, its templates, and the summary that runs by itself after a transcription.'})
        + _sCard({
            title: 'Automatic summary',
            body: _sRow({key: 'llm.auto_summarize', type: 'toggle', label: 'Summarize after transcription', tags: ['jobs'],
                    desc: 'Pre-selects the 🤖 switch in the transcription dialogs; the summary then runs on the server when the transcript is done. MCP and API jobs are not affected.'})
                + _sRow({key: 'llm.auto_summarize_preset', type: 'select', label: 'Template',
                    desc: 'The template the automatic summary uses.',
                    options: _sWith([['', 'Default template (below)'], ...presets.map(n => [n, n])], l.auto_summarize_preset)})
                + _sRow({key: 'llm.auto_summarize_model', type: 'select', label: 'Model',
                    desc: 'The model it runs on — a model shortcut, or the one the summary task uses (AI models).',
                    options: _sWith([['', 'The summary task\'s model'], ...models.map(([n, id]) => [id, n !== id ? `${n} (${id})` : n])], l.auto_summarize_model)}),
        }) + _sCard({
            title: 'Default template',
            body: _sRow({key: 'llm.system_prompt', type: 'textarea', rows: 9, mono: false, label: 'Prompt of a summary without a template',
                    desc: 'Used by automatic summaries without a template; the summarize dialog starts with the template that has this text. Empty = no system prompt at all.'}),
        }) + _sCard({
            title: 'Templates',
            sub: 'Each is a complete prompt, chosen in the summarize dialog — it can be edited and saved there too.',
            actions: '<button type="button" class="set-btn sm" onclick="_checkDefaultPresets()" title="Add missing built-in templates or reset changed ones — your own stay untouched">⬇ Built-in templates</button>',
            body: _sBlock('<div id="llmStdPresets"></div><div id="llmPresetsList" class="set-items"></div>'
                + `<div class="set-add"><input class="set-input" id="llmNewPresetName" placeholder="Name of a new template"
                        onkeydown="if(event.key==='Enter'){event.preventDefault();_addLlmPreset();}">
                    <button type="button" class="set-btn" onclick="_addLlmPreset()">+ Add template</button></div>`),
        }) + _sCard({
            title: 'Answer', fold: true,
            sub: 'How long a summary may get, and how the model writes it.',
            body: _sRow({key: 'llm.max_tokens', type: 'number', unit: 'tokens', min: 256, step: 256, label: 'Longest summary',
                    desc: 'Longest answer of a summary; capped at half the context window.'})
                + _sRow({key: 'llm.temperature', type: 'number', min: 0, max: 2, step: 0.1, label: 'Temperature',
                    desc: '0 = deterministic, higher = freer wording. Not used while extended thinking is on.'})
                + _sRow({key: 'llm.thinking', type: 'toggle', label: 'Extended thinking',
                    desc: 'Let the model reason before the final summary. Ollama: the think flag; /v1 servers: a “thinking” field, which some reject.'})
                + _sRow({key: 'llm.thinking_budget', type: 'number', unit: 'tokens', min: 0, label: 'Thinking budget',
                    desc: 'Reasoning tokens while thinking is on (/v1 servers only). 0 = no limit sent.'}),
        })
        + _sSection({id: 'correction', title: 'Correction',
                sub: 'The 🪄 correction of a transcript: uncertain words go to the language model, its suggestions come back for you to review.'})
        + _sCard({
            title: 'What gets checked',
            body: _sRow({key: 'llm.correction.score_threshold', type: 'number', min: 0, max: 1, step: 0.05, label: 'Uncertain words below',
                    desc: 'Words with an alignment score below this are marked uncertain; their segments are reviewed. The correction dialog starts with this value.'})
                + _sRow({key: 'llm.correction.logprob_threshold', type: 'number', max: 0, step: 0.1, label: 'Uncertain segments below',
                    desc: 'Segments whose average log-probability is below this (e.g. −0.8) are reviewed too.'}),
        }) + _sCard({
            title: 'Which suggestions stay',
            body: _sRow({key: 'llm.correction.min_confidence', type: 'number', min: 0, max: 1, step: 0.05, label: 'Keep suggestions from',
                    desc: 'Suggestions the model itself rates below this are dropped.'})
                + _sRow({key: 'llm.correction.max_edit_share', type: 'number', min: 0, max: 1, step: 0.05, label: 'Discard a part that changes more than',
                    desc: 'Share of its segments. A part that wants to change more is discarded whole — a sign of over-correction.'})
                + _sRow({key: 'llm.correction.temperature', type: 'number', min: 0, max: 2, step: 0.1, label: 'Temperature',
                    desc: 'Low keeps corrections reproducible.'}),
        }) + _sCard({
            title: 'Own rules', fold: true,
            body: _sRow({key: 'llm.correction.system_prompt', type: 'textarea', rows: 7, label: 'Correction rules', ctlId: 'corrRulesTa',
                    desc: 'Replaces the rules part of the prompt — what to correct, what to leave alone. Empty = the built-in rules (shown greyed). The answer format is always added.'})
                + _sBlock('<div class="set-actions"><button type="button" class="set-btn sm" onclick="_corrRulesTemplate()">📋 Start from the built-in rules</button></div>'),
        })
        + _sSection({id: 'translation', title: 'Translation',
                sub: 'Translations of a transcript into another language, part by part.'})
        + _sCard({
            title: 'Consistency',
            body: _sRow({key: 'llm.translation.context_pairs', type: 'number', min: 0, max: 50, label: 'Context pairs',
                    desc: 'Segments already translated, sent along with each part to keep names and forms of address consistent. 0 = none.'})
                + _sRow({key: 'llm.translation.temperature', type: 'number', min: 0, max: 2, step: 0.1, label: 'Temperature',
                    desc: 'Low keeps translations close to the source.'}),
        });
    },
    after: (cfg) => _renderLlmPresets(cfg.llm || {}),
},

{
    id: 'mcp', group: 'Search & AI', icon: 'server', title: 'MCP server', hint: 'Let AI assistants use the library',
    intro: 'Lets AI assistants — Claude, Open WebUI, LM Studio and every other MCP client — use the library: search, read transcripts, look up speakers, find conversations and, if allowed, transcribe, download and summarize.',
    render: (cfg) => {
        const m = cfg.mcp || {};
        const port = cfg.server?.port || 9860;
        const wide = (m.host || '127.0.0.1') !== '127.0.0.1' && m.host !== 'localhost';
        const exposed = wide && (m.autostart_http || m.autostart_openapi);
        return (exposed && m.require_token === false ? `<div class="set-note warn" style="margin-bottom:14px">⚠ The network services listen on ${esc(m.host)} without a token — anyone on the network can use them${m.allow_writes !== false ? ', transcribing and downloading included' : ''}. Turn “Require the access token” back on.</div>` : '')
        + _sCard({
            title: 'Network services',
            sub: 'Started with this app, for clients that connect over the network. The switches, address and ports are read when the app starts — restart it after a change.',
            body: _sBlock('<div id="mcpStatus"><div class="set-hint">Loading…</div></div>', {skip: true})
                + _sRow({key: 'mcp.autostart_http', type: 'toggle', label: 'MCP over HTTP', tags: ['restart'],
                    desc: 'For MCP clients that connect over the network — Claude Code, Cursor, LM Studio and others.'})
                + _sRow({key: 'mcp.autostart_openapi', type: 'toggle', label: 'OpenAPI service', tags: ['restart'],
                    desc: 'For Open WebUI and other clients that don\'t speak MCP but take tools as an OpenAPI server. The spec is at ' + _sCode('/openapi.json') + '.'})
                + _sRow({key: 'mcp.host', type: 'text', label: 'Address', placeholder: '127.0.0.1', tags: ['restart'],
                    desc: '127.0.0.1 = this machine only, 0.0.0.0 = the whole network.'})
                + _sRow({key: 'mcp.port', type: 'number', min: 1024, max: 65535, label: 'MCP port', tags: ['restart'],
                    desc: 'Not the port of this app (' + port + ').'})
                + _sRow({key: 'mcp.openapi_port', type: 'number', min: 1024, max: 65535, label: 'OpenAPI port', tags: ['restart'],
                    desc: 'Its own port, so both can run at once.'}),
        }) + _sCard({
            title: 'Access',
            sub: 'Who may use the network services. A client on this machine that starts the server itself (stdio) needs no token.',
            body: _sRow({key: 'mcp.require_token', type: 'toggle', label: 'Require the access token',
                    desc: 'Clients send the token below as ' + _sCode('Authorization: Bearer …') + ' (or ' + _sCode('X-API-Token') + '). For MCP over HTTP and the OpenAPI service alike; applies as soon as it is saved. Off = anyone who reaches the ports may use the tools.'})
                + _sBlock('<div id="mcpToken"><div class="set-hint">Loading…</div></div>', {skip: true}),
        }) + _sCard({
            title: 'Connect a client',
            sub: 'Ready to paste, with this machine\'s address and the token. “Copy” puts in the real token even while it is hidden.',
            body: _sBlock('<div id="mcpClients"><div class="set-hint">Loading…</div></div>', {skip: true}),
        }) + _sCard({
            title: 'What the AI may do',
            body: _sRow({key: 'mcp.allow_writes', type: 'toggle', label: 'Allow work', tags: ['mcp'],
                    desc: 'Off (as shipped) = read only: search, transcripts, speakers. On = may also upload, download, transcribe, summarize, translate, proofread and control the queue — a transcription takes the GPU from a running batch, and what the AI reads can carry instructions for it.'})
                + _sRow({key: 'mcp.max_files_per_call', type: 'number', min: 1, max: 10000, label: 'Most files per call', tags: ['mcp'],
                    desc: 'For transcribing many files at once — more is refused, not cut off.'})
                + _sRow({key: 'mcp.time_limit_min', type: 'number', unit: 'min', min: 1, label: 'Wait for a transcription', tags: ['mcp'],
                    desc: 'How long the transcribe tool waits for the result before it returns the job id. The work goes on either way (a long file can take many minutes).'}),
        }) + _sCard({
            title: 'Connection to this app', fold: true,
            sub: 'How the MCP server itself reaches this app — only needed in special setups.',
            body: _sRow({key: 'mcp.base_url', type: 'text', label: 'App address', placeholder: `http://127.0.0.1:${port}`, tags: ['mcp'],
                    desc: 'Where the MCP server reaches this app. Set it only when the MCP server runs on another machine.'})
                + _sRow({key: 'mcp.listen_base_url', type: 'text', label: 'Address for audio links', tags: ['mcp'],
                    placeholder: `http://${location.hostname || '127.0.0.1'}:${port}`,
                    desc: 'Where a browser reaches this app, for the players in answers. Empty = the app address with 127.0.0.1 replaced by this machine\'s network address.'})
                + _sRow({key: 'mcp.api_token', type: 'secret', label: 'App token', tags: ['mcp', 'secret'], placeholder: 'empty = auth.api_token',
                    desc: 'What the MCP server sends to this app — only needed when the app\'s login is on (' + _sCode('auth.enabled') + ' in config.yaml). Not the access token above.'}),
        }) + _sCard({
            title: 'Start it by hand', fold: true,
            sub: 'Only when this app should not start the services itself. Run with the app\'s Python, in the app folder.',
            body: _sRow({label: 'As a network service', type: 'static', value: '<input class="set-input mono" readonly onclick="this.select()" value="./.venv/bin/python -m whisper_tm.mcp_server --http" aria-label="Command for MCP over HTTP">',
                    wide: true, desc: 'MCP over HTTP; ' + _sCode('--openapi') + ' for the OpenAPI service. Address, port and token as on this page.'})
                + _sRow({label: 'Check the connection', type: 'static', value: '<input class="set-input mono" readonly onclick="this.select()" value="./.venv/bin/python -m whisper_tm.mcp_server --check" aria-label="Command to check the connection">',
                    wide: true, desc: 'Reports how many files the MCP server sees — the fastest way to find out whether it reaches this app.'}),
        });
    },
    open: () => _mcpAccessLoad(),
},

// ── System ───────────────────────────────────────────────────────────────

{
    id: 'database', group: 'System', icon: 'database', title: 'Database & backups', nav: 'Database', hint: 'Status, backups, maintenance, exports',
    intro: 'The SQLite database with the transcripts, their segments and the search index — its state, the backups and maintenance.',
    render: (cfg) => _sCard({
        title: 'Database',
        body: _sBlock('<div id="dbStatus"><div class="set-hint">Loading…</div></div>', {skip: true}),
    }) + _sCard({
        title: 'Backups',
        sub: 'Copies of the database, made while it runs. Transcript changes count — scans and index runs don\'t.',
        body: _sRow({key: 'database.backup_path', type: 'text', label: 'Backup folder', wide: true,
                placeholder: 'e.g. /mnt/nas/whisper-tm-backups',
                desc: 'Ideally on another disk or a NAS. Receives the backups at server start and after every N changes. Empty = start backups go to backups/ next to the database, change backups are off.'})
            + _sRow({key: 'database.backup_every_changes', type: 'number', min: 0, label: 'Back up after',
                unit: 'changes', desc: 'New, edited or deleted transcripts. Needs a backup folder. 0 = off.'})
            + _sRow({key: 'database.auto_backup', type: 'toggle', label: 'Back up at server start', tags: ['restart'],
                desc: 'At most once per the interval below.'})
            + _sRow({key: 'database.backup_min_hours', type: 'number', unit: 'h', min: 0, label: '…at most every',
                desc: 'A start backs up only when the newest backup is older — a restart loop must not rotate the good backups away. 0 = every start.'})
            + _sRow({key: 'database.max_backups', type: 'number', min: 1, max: 1000, label: 'Keep', unit: 'backups',
                desc: 'Per backup folder, manual ones included; the oldest go first.'})
            + _sBlock(`<div id="backupStatusBox" class="set-hint" data-search-skip>Loading…</div>
                <div class="set-actions" style="margin-top:8px">
                    <button type="button" class="set-btn" id="btnBackupNow" onclick="_dbAction('backup', this)">💾 Back up now</button>
                    <button type="button" class="set-btn" onclick="_loadBackupStatus()">↻ Refresh</button></div>`),
    }) + _sCard({
        title: 'Maintenance',
        body: _sAction({title: 'Import transcripts', button: '🔄 Import', click: "_dbAction('sync', this)",
                desc: 'Import the JSON transcripts the database doesn\'t have yet — with transcripts beside the recordings, from every .transcripts folder of the library (after losing the database, this brings them back).'})
            + _sAction({title: 'Rebuild the search index', button: '🔍 Rebuild', click: "_dbAction('rebuild', this)",
                desc: 'Re-creates the full-text index from all segments — only when the search misses what it should find. Takes minutes with a large library.'})
            + _sAction({title: 'Compact', button: '📦 Compact', click: "_dbAction('vacuum', this)",
                desc: 'Gives the space of deleted data back to the disk (VACUUM). Backs up first, and the database is busy meanwhile — best while the queue is paused.'}),
    }) + _sCard({
        title: 'Exports',
        body: _sAction({title: 'Database', button: '⬇ .db', href: '/api/db/export',
                desc: 'A consistent copy of the whole database, made while it runs.'})
            + _sAction({title: 'Transcripts, one per line', button: '⬇ JSONL', href: '/api/export/transcripts?format=jsonl',
                desc: 'Every finished transcript — to import elsewhere or to feed another tool.'})
            + _sAction({title: 'Transcripts, one file each', button: '⬇ ZIP', href: '/api/export/transcripts?format=zip',
                desc: 'One JSON file per transcript.'})
            + _sAction({title: 'Vector index', button: '⬇ .db', href: '/api/rag/export',
                desc: 'The index of the semantic search (Search), a consistent copy made while it runs.'}),
    }) + _sCard({
        title: 'Danger zone', cls: 'set-danger', fold: true,
        body: _sAction({title: 'Delete the database', button: '🗑 Delete', click: '_deleteDb(this)', cls: 'danger',
                desc: 'Deletes transcripts, segments, search index, queue, summaries, translations, edits and job history in the database, and starts empty. JSON transcripts on disk stay — in “Database only” mode there are none. The voice store and the vector index are not touched.'}),
    }),
    after: () => { _dbStatusLoad(); _loadBackupStatus(); },
},

{
    id: 'ui', group: 'System', icon: 'bell', title: 'Notifications', hint: 'The done sound',
    intro: 'The sound when the queue is done or a watched transcription finishes.',
    render: () => _sCard({
        title: 'Done sound',
        body: _sRow({label: 'Play the done sound', type: 'static',
                value: `<label class="set-switch"><input type="checkbox" role="switch" id="setSoundOn" onchange="_setSoundOn(this.checked)" aria-label="Play the done sound"></label>`,
                desc: 'The same switch as the bell in the top bar — it applies to this browser.'})
            + _sBlock(`<div class="set-actions">
                    <select class="set-select" id="settingsDoneSound" onchange="_selectDoneSound(this.value)" style="width:auto;min-width:220px" aria-label="Done sound"></select>
                    <button type="button" class="set-btn" onclick="_previewDoneSound()">▶ Listen</button>
                    <label class="set-btn">⬆ Add a sound…<input type="file" accept=".mp3,.wav,.ogg,.m4a" hidden onchange="_uploadDoneSound(this)"></label>
                </div>
                <div class="set-hint" style="margin-top:6px">MP3, WAV, OGG or M4A. Added sounds are kept in the app's data folder.</div>`),
    }),
    open: () => { _loadSoundList(); _soundSwitchSync(); },
},

{
    id: 'diagnostics', group: 'System', icon: 'tool', title: 'Diagnostics', hint: 'Job logs, database explorer, debug output',
    intro: 'For looking under the hood: the job logs, the tables of the database and debug output.',
    render: () => _sCard({
        title: 'Job logs',
        sub: 'One file per transcription job, newest first.',
        body: _sBlock(`<div class="set-actions">
                <input class="set-input" id="logExplorerFilter" placeholder="Filter by name…" style="max-width:280px" aria-label="Filter the job logs"
                       oninput="clearTimeout(window._logFilterTimer);window._logFilterTimer=setTimeout(_loadLogs,400)">
                <button type="button" class="set-btn" onclick="_loadLogs()">↻ Load</button></div>
            <div id="logExplorerList" class="set-table-wrap" data-search-skip style="max-height:320px;margin-top:8px">
                <div class="set-browser-msg">Opened on request — the folder can hold many thousands of files.</div></div>`),
    }) + _sCard({
        title: 'Database explorer', fold: true, onopen: '_dbExplorerOpened',
        sub: 'Read-only look into the tables.',
        body: _sBlock(`<div class="set-actions">
                <select class="set-select mono" id="dbExploreTable" onchange="_loadDbExplorer()" style="width:auto" aria-label="Table">
                    ${['transcripts', 'media_files', 'segments', 'skiplist', 'job_history', 'user_settings']
                        .map(t => `<option value="${t}">${t}</option>`).join('')}
                </select>
                <input class="set-input mono" id="dbExploreSearch" placeholder="Search — Enter" style="max-width:220px" aria-label="Search the table"
                       onkeydown="if(event.key==='Enter'){event.preventDefault();_loadDbExplorer();}">
                <span id="dbExploreInfo" class="set-hint"></span>
                <span style="flex:1"></span>
                <button type="button" class="set-btn sm" onclick="_dbExpPage(-1)" aria-label="Previous page">◀</button>
                <button type="button" class="set-btn sm" onclick="_dbExpPage(1)" aria-label="Next page">▶</button></div>
            <div id="dbExploreResult" class="set-table-wrap" data-search-skip style="max-height:420px;margin-top:8px"></div>`),
    }) + _sCard({
        title: 'Debug output',
        body: _sRow({label: 'Browser debug output', type: 'static',
                value: `<label class="set-switch"><input type="checkbox" role="switch" id="debugModeToggle" onchange="_toggleDebugMode()" aria-label="Browser debug output"></label>`,
                desc: 'Detailed messages in this browser\'s console (F12). The server\'s debug mode is separate: ' + _sCode('--debug') + ' or ' + _sCode('logging.debug') + ' in config.yaml.'})
            + _sBlock(`<div class="set-actions">
                <a class="set-btn sm" href="/api/debug/timings/text" target="_blank" rel="noopener">⏱ Where the time went</a>
                <a class="set-btn sm" href="/api/debug/gpu" target="_blank" rel="noopener">🖥 GPU state</a></div>
                <div class="set-hint" style="margin-top:6px">Timings are collected while the server runs in debug mode.</div>`),
    }),
    open: () => { _debugToggleSync(); },
},

{
    id: 'about', group: 'System', icon: 'info', title: 'About', hint: 'Version, configuration file, reset',
    intro: 'Which version runs, where its configuration lives — and the ways back to the start.',
    render: () => _sCard({
        title: 'This installation',
        body: _sBlock('<div id="aboutInfo"></div>', {skip: true}),
    }) + _sCard({
        title: 'Configuration file',
        sub: 'Every setting on these pages lives in one YAML file. A few can only be changed there: the server address, the login, the allowed folders and your own MCP token.',
        body: _sBlock(`<div class="set-actions"><a class="set-btn" href="/api/settings/export" download="config.yaml">⬇ Download</a></div>
            <div class="set-hint" style="margin-top:6px">Tokens and keys are replaced by a placeholder in the download — it documents a setup, it doesn't back up its credentials.</div>`),
    }) + _sCard({
        title: 'Danger zone', cls: 'set-danger',
        body: _sAction({title: 'Reset settings', button: '↺ Reset', click: 'resetAllSettings()', cls: 'danger',
                desc: 'Every setting back to its default. Kept: the folders and the database location, the server address, login and allowed folders, all credentials, the LLM providers, templates and vocabularies. Transcripts are not touched.'})
            + _sAction({title: 'Empty the work folder', button: '🧹 Empty', click: 'cleanWorkdir()', cls: 'danger',
                desc: 'Deletes the temporary files of finished and cancelled jobs. Refused while a job is running.'})
            + _sAction({title: 'Forget this browser\'s choices', button: '🗑 Forget', click: 'resetBrowserData()', cls: 'danger',
                desc: 'Favourite folders, the done-sound switch, folded cards, view settings — stored in this browser only. Queue, transcripts and settings stay.'}),
    }),
    open: () => _aboutRender(),
},
];

// A row that does something instead of holding a value: name and what it
// does on the left, the button on the right
function _sAction({title, desc, button, click, href, cls = ''}) {
    const c = 'set-btn' + (cls ? ' ' + cls : '');
    return `<div class="set-row set-action">
        <div class="set-text"><div class="set-label">${title}</div><div class="set-desc">${desc}</div></div>
        <div class="set-ctl">${href ? `<a class="${c}" href="${href}" download>${button}</a>`
                                    : `<button type="button" class="${c}" onclick="${click}">${button}</button>`}
            <span class="set-reset" aria-hidden="true"></span></div>
    </div>`;
}

// ── Page helpers ─────────────────────────────────────────────────────────

// Transcript mode: the central folder row only matters in central mode
function _sTranscriptMode(mode) {
    const row = document.getElementById('setRowTranscripts');
    if (row) row.hidden = mode !== 'central';
}

// Embedding mode: one card for a server of yours, one for the app's own
function _embToggleMode() {
    const input = document.querySelector('#settingsOverlay [data-key="embedding.local"]');
    const local = input ? _sRead(input) : false;
    const ex = document.getElementById('embExternal');
    if (ex) ex.hidden = local;
    for (const id of ['embLocal', 'embLocalTune']) {
        const el = document.getElementById(id);
        if (el) el.hidden = !local;
    }
    if (local) _embLoadStatus();
}
