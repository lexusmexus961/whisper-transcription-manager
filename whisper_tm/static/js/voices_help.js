/* ══════════════════════════════════════════════════════════════════════════
 *  VOICES — the Help tab
 *
 *  What the workspace is for, what each tab does, the words it uses, and
 *  the keys — on one page, with links into the tabs. Plain content; it is
 *  drawn once when the tab is first shown.
 * ══════════════════════════════════════════════════════════════════════════ */

const VX_HELP_TABS = [
    ['people', 'People', 'Everyone named, with voices, speech and recordings. Rename or merge two names. '
        + '"Listen through" lists every voice of a person — the doubtful ones first — to remove wrong names.'],
    ['assign', 'Name voices', 'The fastest way to name many voices: it asks for the group that decides the most open speech, '
        + 'one question at a time. Listen to the samples (Play all, the first seconds of each), look at who the group '
        + 'sounds like, then type the name (suggestions appear while typing), discard it, or skip it. '
        + 'A name spreads to the voices that sound alike — one click often decides thousands of voices.'],
    ['history', 'History', 'Every decision about names and the trash, newest first. Undo takes back exactly what that click '
        + 'changed; Redo puts it in force again. An undone decision can be decided anew right there.'],
    ['map', 'Voice map', 'Every voice of the collection as tiles (or a graph), grouped by how alike they sound. '
        + 'Click a cell to hear the voice and name it, or open a group on the right to name it as a whole. '
        + 'The threshold says how alike voices must be to share a group: stricter splits groups, looser merges them. '
        + 'The list on the right shows the groups — sortable by size or name; the checkboxes keep only those groups in the picture.'],
    ['conversations', 'Conversations', 'Whole conversations — or the stretch of a long recording — by who took part and how they talked: '
        + 'length, sentence length, speaker changes, kind. The same view as Find\'s Conversations tab; a person\'s "Conversations" '
        + 'opens it with them chosen. ✂ hands what is shown to the Voice dataset.'],
    ['extract', 'Voice dataset', 'One person\'s voice as clips with their sentences: filter by certainty, length and date, trim clips '
        + 'in the editor, and export a TTS dataset or one audio file with subtitles — as a download or into a folder. Use a person\'s voice only with their consent — a voice dataset can train a model that speaks like them.'],
];

const VX_HELP_WORDS = [
    ['Voice (window)', 'About 30 seconds of one speaker\'s clean speech in one recording, turned into a vector — the unit everything here counts. '
        + 'A long speaker in a long recording has many windows.'],
    ['Speaker (label)', 'SPEAKER_00, SPEAKER_01 … — how the diarization told the people of ONE recording apart. '
        + 'The same label in two recordings means nothing; only the name connects them.'],
    ['Group', 'Voices that sound alike enough (above the threshold) to be one person. Groups are computed anew after decisions; '
        + 'their numbers change, names never do.'],
    ['Similarity', 'How alike two voices are, 0 to 1. Around 0.8 and above: very likely the same person; '
        + 'from 0.65 a given name spreads; below that: hardly alike.'],
    ['Spreading', 'After a name is given, open voices that sound like that person get the name too ("by spreading"). '
        + 'They can be told apart from names given by hand everywhere.'],
    ['Trash', 'Voices nobody cares about (music, TV, game sounds). Nothing is deleted — the History or the Trash view brings them back.'],
    ['Voices without speech spans', 'A voice without stored speech spans has as its sample just the window\'s first 30 s, which may hold '
        + 'other speakers. "Renew voices" (Queue → From the library → Has voices without stored speech spans) makes them anew and keeps their names.'],
    ['Recurring speech', 'Speech of voices that occur in more than one recording — what assigning is about. '
        + 'A voice heard only once is never asked about.'],
];

const VX_HELP_KEYS = [
    ['Name voices', [['Enter', 'apply the name'], ['Shift+Enter', 'skip'], ['Del (empty field)', 'trash'],
                ['Ctrl+Z (empty field)', 'undo the last decision'], ['Ctrl+P', 'play all samples / pause']]],
    ['Name fields', [['↓ ↑', 'choose a suggested name'], ['Tab', 'take it'], ['Enter', 'take the chosen one and go on'],
                     ['Esc', 'close the suggestions']]],
    ['People → Listen through', [['Space', 'play / pause'], ['← →', 'previous / next voice'], ['N', 'not this person'],
                                 ['T', 'trash'], ['A', 'name as someone else'], ['Esc', 'stop']]],
    ['Everywhere', [['Esc', 'close a menu, then the workspace']]],
];

let VX_HELP_DRAWN = false;

function helpShow() {
    const el = document.getElementById('vxPageHelp');
    if (!el) return;
    vxState('help', 'How the voices workspace works');
    if (VX_HELP_DRAWN) return;
    VX_HELP_DRAWN = true;
    el.innerHTML = `<div class="vx-help">
        <div class="vx-card vx-help-intro">
            <h3>What this is for</h3>
            <div class="vx-sub">Every diarized recording leaves voices behind. Here they are compared across the whole
                collection, so that a person named once is recognized in every recording — for searching who said what,
                for "only this person's sentences", and for extracting one person's voice.</div>
            <ol class="vx-help-steps">
                <li><b>Name voices</b> — answer the questions; each one names a whole group, the name spreads.</li>
                <li><b>People</b> — listen through a person now and then; remove what doesn't belong.</li>
                <li><b>Voice map</b> — for the rest: look around, name single voices, discard noise.</li>
                <li><b>History</b> — anything wrong can be undone, click by click.</li>
                <li><b>Voice dataset</b> — once a person is named well, cut their voice into clips.</li>
            </ol>
        </div>
        <h3 class="vx-help-h">The tabs</h3>
        <div class="vx-help-grid">${VX_HELP_TABS.map(([t, name, text]) => `
            <div class="vx-card"><h3>${esc(name)}</h3><div class="vx-sub">${esc(text)}</div>
                <button class="vx-btn" style="margin-top:10px" onclick="vxTab('${t}')">Open ${esc(name)}</button></div>`).join('')}
        </div>
        <h3 class="vx-help-h">Words</h3>
        <div class="vx-card"><dl class="vx-help-words">${VX_HELP_WORDS.map(([w, t]) =>
            `<dt>${esc(w)}</dt><dd>${esc(t)}</dd>`).join('')}</dl></div>
        <h3 class="vx-help-h">Keys</h3>
        <div class="vx-help-grid">${VX_HELP_KEYS.map(([where, keys]) => `
            <div class="vx-card"><h3>${esc(where)}</h3>${keys.map(([k, t]) =>
                `<div class="vx-help-key"><kbd>${esc(k)}</kbd><span>${esc(t)}</span></div>`).join('')}</div>`).join('')}
        </div>
    </div>`;
}
