/* ══════════════════════════════════════════════════════════════════════════
 *  THE VOICE MAP WORKSPACE — Selection, listening sample, sequence player
 *
 *  A complete redesign of the map's playback panel.
 *
 *  Review first, because almost everything buggy about the old version came
 *  from ONE design flaw: on every selection, `mapPickVoice` rewrote
 *  the whole panel via `innerHTML` — header, audio, name field, transport,
 *  lists. And the sequence player selects on EVERY item. From this followed,
 *  in this order discovered:
 *
 *  1. TWO VOICES AT ONCE. The comment above `mapAttachAudio`
 *     promised "ONE audio element for the whole session, re-attached instead
 *     of rebuilt". That wasn't true: `innerHTML =` detaches the old element
 *     from the tree, after which `getElementById('mapAudio')` finds nothing
 *     and a NEW one was built. A detached `<audio>` keeps playing —
 *     so the old clip kept running underneath the new one. `audioEl.pause()` each
 *     time only hit the fresh, still-silent element.
 *  2. THE TRANSPORT DISAPPEARED. ⏮⏯⏭⏹ sat in `#mapRowState` INSIDE
 *     the panel. Clicking a point while a sequence was running rebuilt
 *     the panel — the sequence kept running, the buttons were gone.
 *  3. TYPED TEXT WAS LOST. You listen, recognize the voice, type "alice" —
 *     after three seconds the sequence advances, the panel gets rebuilt,
 *     what was typed is gone. Exactly the action the map exists for.
 *  4. THE LIST JUMPED. Forty recordings below it were rebuilt on every item;
 *     the scrollbar jumped back to the top each time.
 *  5. THE HANDLERS BELONGED TO NO ONE. `audioEl.onended` pointed to the "next"
 *     of the sequence and stayed stuck there. Anyone who pressed ▶ on a
 *     neighbor while a sequence was running heard that neighbor — and then
 *     got the next item of the sequence afterward, without having requested
 *     it. Likewise `ontimeupdate` cut the individual listening sample off
 *     after three seconds.
 *  6. THE AUDIO KEPT PLAYING AFTER CLOSING. Esc only hid the stage.
 *
 *  THE REDESIGN therefore rests on three rules:
 *
 *  * **One skeleton, built once.** The panel is created exactly once
 *    (`panelFrame`). After that, only text nodes and values are set.
 *    No `innerHTML` over anything the user is currently using.
 *  * **One audio element that never leaves the tree.** It lives in the
 *    skeleton; only `src` changes. Two voices at once are thereby
 *    structurally impossible, not just unlikely.
 *  * **One owner.** `PANEL.owner` says WHO currently owns
 *    playback — the sequence or a single click. The event handlers stay
 *    fixed on the element and ask the owner, instead of being re-attached.
 *
 *  Beyond that, what actually helps while listening: The transport bar
 *  sticks to the top and stays visible at all times, progress in the
 *  sequence AND in the item is visible, the name field pauses the sequence
 *  as soon as you click into it (whoever is typing doesn't want to be
 *  rushed along), Enter saves and moves on, and the
 *  spacebar as well as ← → operate the player without a mouse.
 * ══════════════════════════════════════════════════════════════════════════ */

const PANEL = {
    nodes: -1,          // which node is currently in the panel
    group_nr: null,        // which group is in the lists (rebuild only on change)
    row: null,         // {listing, pos, running, nr, perFile, openOnly}
    owner: null,      // 'row' | 'single' — who owns the audio element
    autoplay: true,      // clicking a voice plays it immediately
};

const PANEL_MEMO = 'whisperx.map.autoplay';

const PANEL_PREVIEW = [0, 1, 2, 3, 5, 10];

/* ── Autoplay ─────────────────────────────────────────────────────────────
 *
 * An autoplay toggle: clicking a node in a group plays it automatically.
 *
 * When assigning names, you click through a group, and doing the same
 * second reach for the play button every time is exactly the action that
 * makes the work tedious. As a TOGGLE, not fixed: someone just viewing a
 * map doesn't want audio on every click.
 *
 * What matters is WHAT triggers it. Not every selection is a click: the
 * sequence player also selects a voice on every item, and that already
 * plays itself. That's why playback hangs off `panelVoiceClicked` — the
 * path only a human click takes — and not off
 * `mapPickVoice`.
 */
function panelToggleAutoplay(on) {
    PANEL.autoplay = !!on;
    try { localStorage.setItem(PANEL_MEMO, on ? '1' : '0'); } catch (e) { /* private mode */ }
    const h = document.getElementById('mapAutoplay');
    if (h) h.checked = PANEL.autoplay;
}

function panelLoadAutoplay() {
    let w = null;
    try { w = localStorage.getItem(PANEL_MEMO); } catch (e) { /* private mode */ }
    PANEL.autoplay = w === null ? true : w === '1';
}

/* The path a click takes — from the picture as well as from the lists.
 *
 * If a sequence is currently running, it stops it: whoever clicks away has
 * left the sequence. Just letting it keep running would mean that three
 * seconds later, without any action, something other than what was clicked
 * would play. */
function panelVoiceClicked(i) {
    panelStopRow();
    if (PANEL.autoplay) graphListen(i);
    else mapPickVoice(i);
}

/* Leave the running sequence without discarding it.
 *
 * It stays intact — ⏭ and ⏮ should keep working — but it doesn't keep
 * playing. This MUST hang off every path that loads a single clip, not just
 * the click into the picture: the ▶ button in the neighbor and
 * recording lists calls `graphListen` directly. Without this stop, the
 * sequence formally kept running, the `ended` listener saw `running === true`
 * and, after the neighbor was listened to, pushed the next sequence item
 * unrequested.
 */
function panelStopRow() {
    if (!PANEL.row || !PANEL.row.running) return;
    PANEL.row.running = false;
    const audioEl = document.getElementById('mapAudio');
    if (audioEl) audioEl.pause();
    panelTransportState();
}

/* ── The skeleton ─────────────────────────────────────────────────────────
 *
 * Built once, never touched again after that. Every spot that can change
 * has its own id — that's the whole trick compared to before.
 */
function panelFrame() {
    const el = document.getElementById('mapSelection');
    if (!el || el.dataset.built) return el;
    el.dataset.built = '1';
    el.innerHTML = `
        <div class="map-p-head">
            <div class="map-p-names">
                <div class="map-p-title" id="mapTitle"></div>
                <div class="map-small map-p-folder" id="mapFolder"></div>
            </div>
            <button class="map-p-closed" id="mapSelectionClose"
                    title="Close this voice and show the largest groups (Esc · Alt+←)" aria-label="Close this voice">✕</button>
        </div>
        <div class="map-small" id="mapMeta"></div>
        <!-- The path INTO the recording. The listening sample plays 30
             seconds from the window; anyone who wants to know what it's
             about needs the context before and after — and that only
             exists in the transcript. -->
        <div class="map-p-spot">
            <button class="map-mini" id="mapShowSpot"
                    title="Open the recording and jump to this spot"
                    onclick="mapShowSpot()">⤴ Location in recording</button>
        </div>

        <!-- Sticks to the top: whoever scrolls down the list would
             otherwise lose exactly the buttons that control playback. -->
        <div class="map-p-transport" id="mapTransport">
            <!-- THE BROWSER'S DEFAULT PLAYER, visible, so the clips can
                 be saved with a right-click — and that's exactly
                 the point: right-click → "Save audio as". A self-built
                 bar can't do that, and the listening sample is a
                 ready-made MP3 at a normal address.
                 Previously this did the opposite (audioEl.controls = false),
                 because two play buttons side by side is confusing. That
                 objection is still valid — it's just answered differently
                 now: the player belongs to the ITEM, the buttons below it to
                 the SEQUENCE, and the row says so too. -->
            <div id="mapAudioSlot" class="map-audio-slot"></div>
            <div class="map-p-keys">
                <span class="map-small map-p-queue-label">Sequence</span>
                <button class="map-mini" id="mapKeyPrev"
                        title="Previous item in the sequence (←)">⏮</button>
                <button class="map-mini" id="mapKeyPlay"
                        title="Pause / resume sequence (spacebar)">▶</button>
                <button class="map-mini" id="mapKeyNext"
                        title="Next item in the sequence (→)">⏭</button>
                <button class="map-mini" id="mapKeyStop"
                        title="End sequence">⏹</button>
                <span class="map-p-state" id="mapRowState"></span>
                <label class="map-small map-p-auto"
                       title="Clicking a voice plays it immediately">
                    <input type="checkbox" id="mapAutoplay">
                    Autoplay
                </label>
                <label class="map-small map-p-preview-field"
                       title="How much of each item is played?">
                    per item
                    <select class="map-preview" id="mapPlayback">
                        ${PANEL_PREVIEW.map(v =>
                            `<option value="${v}">${v ? v + ' s' : 'full'}</option>`
                        ).join('')}
                    </select>
                </label>
            </div>
            <!-- ONLY ONE bar now. "How far am I into this item?"
                 is answered by the browser's own player;
                 a second slider next to it would be the duplication the old
                 comment here rightly warned about. The question
                 "how far am I into the SEQUENCE?" it can't answer,
                 that one stays. -->
            <div class="map-p-bar map-p-bar-queue" title="Position in the sequence">
                <div class="map-p-bar-fill" id="mapBarRow"></div>
            </div>
            <div class="map-small map-p-running" id="mapRunningFile"></div>
        </div>

        <div class="map-rename">
            <input type="text" id="mapName" data-names placeholder="Name the person … (Enter: save and continue)">
            <button id="mapKeyName">Save</button>
            <button id="mapKeyDetach" title="Remove assignment">✕</button>
        </div>
        <div class="map-small" id="mapNamingState"></div>

        <div class="map-p-body" id="mapBody">
            <div id="mapGroupPart"></div>
            <div id="mapContext"></div>
        </div>`;

    panelAudio();                  // the one audio element, now and forever
    panelWire();
    panelLoadAutoplay();
    const auto = document.getElementById('mapAutoplay');
    if (auto) auto.checked = PANEL.autoplay;
    return el;
}

/* Everything via `addEventListener` instead of `onclick=` in the generated
 * HTML: a skeleton that's only built once only needs its handlers attached
 * once — and the old `onclick` strings were the reason every change
 * to the panel risked a new dead wiring (test_voicemap_wiring.py). */
function panelWire() {
    const bind = (id, event, f) => {
        const e = document.getElementById(id);
        if (e) e.addEventListener(event, f);
    };
    bind('mapSelectionClose', 'click', () => (typeof Shell !== 'undefined' ? Shell.leave(panelClose) : panelClose()));
    bind('mapKeyPrev', 'click', () => graphRowNext(-1));
    bind('mapKeyNext', 'click', () => graphRowNext(1));
    bind('mapKeyPlay', 'click', graphRowPause);
    bind('mapKeyStop', 'click', graphRowStop);
    bind('mapPlayback', 'change', e => graphPlaybackSet(e.target.value));
    bind('mapAutoplay', 'change', e => panelToggleAutoplay(e.target.checked));
    /* Seeking within an item is handled by the browser's player
     * — complete with time display, volume, and right-click to save.
     * The custom bar for that is gone for good. */
    bind('mapKeyName', 'click', () => mapRename());
    bind('mapKeyDetach', 'click', () => mapRename(true));

    /* Whoever clicks into the name field has recognized a voice and wants
     * to name it — not hear the next item in three seconds. The sequence
     * pauses. Without this, every bit of typing was a race against the
     * player. */
    bind('mapName', 'focus', () => {
        if (PANEL.row && PANEL.row.running) graphRowPause();
    });
    bind('mapName', 'keydown', e => {
        if (e.key === 'Enter') { e.preventDefault(); mapRename(); }
    });
}

/* The one audio element. It's built here, lives in the skeleton, and is
 * never replaced again — the event handlers may therefore hang off it once
 * and for all and ask the OWNER, instead of being re-attached. */
function panelAudio() {
    let audioEl = document.getElementById('mapAudio');
    if (audioEl) return audioEl;
    /* Otherwise the skeleton would only be created when a point is clicked —
     * and guided assignment has no selection at all. So no audio element was
     * found, `panelSetAudio` returned `null`, and the ▶ button did
     * NOTHING SILENTLY: the audio never played. The server
     * was flawless the whole time, the listening samples arrived as
     * HTTP 200 with a real MP3 — the path ended in the browser.
     *
     * `panelFrame` sets its build flag BEFORE calling here; the
     * recursion therefore ends after exactly one round. */
    const field_el = document.getElementById('mapSelection');
    if (field_el && !field_el.dataset.built) {
        panelFrame();
        audioEl = document.getElementById('mapAudio');
        if (audioEl) return audioEl;
    }
    const slot = document.getElementById('mapAudioSlot');
    if (!slot) return null;
    audioEl = document.createElement('audio');
    audioEl.id = 'mapAudio';
    audioEl.className = 'map-audio';
    /* WITH `controls` — the default browser player, so the clips can be
     * saved with a right-click. The custom bar could do everything except the one thing
     * that matters here: right-click → "Save audio as". */
    audioEl.controls = true;
    audioEl.preload = 'none';
    slot.appendChild(audioEl);

    audioEl.addEventListener('error', () => {
        const f = audioEl.error;
        // An aborted load is not an error, it's the user having clicked
        // onward. Anything else may well be worth surfacing.
        if (!f || f.code === 1 /* MEDIA_ERR_ABORTED */) return;
        Toast.error(`Listening sample not playable (code ${f.code})`);
    });
    audioEl.addEventListener('ended', () => {
        if (PANEL.owner === 'row' && PANEL.row && PANEL.row.running) {
            graphRowNext(1);
        } else {
            panelTransportState();
        }
    });
    audioEl.addEventListener('play', panelTransportState);
    audioEl.addEventListener('pause', panelTransportState);
    /* Only play the first N seconds.
     *
     * Listening to just the first few seconds is enough to recognize a
     * voice — 36 windows of 30 s are 18 minutes, at three seconds two.
     *
     * Via `timeupdate`, NOT via a timer: a `setTimeout` would keep running
     * while paused and fire too early while buffering.
     *
     * ONLY for the sequence. Previously the limit also cut off the
     * individual listening sample — anyone wanting to listen to a neighbor
     * got three seconds and then the next item of a sequence they hadn't
     * even started.
     */
    audioEl.addEventListener('timeupdate', () => {
        if (PANEL.owner !== 'row') return;
        const r = PANEL.row;
        if (!r || !r.running || !GRAPH.playbackSec) return;
        if (audioEl.currentTime >= GRAPH.playbackSec) { audioEl.pause(); graphRowNext(1); }
    });
    /* Whoever drags the seek control wants to hear THIS item. The sequence
     * pauses — otherwise the play-limit would snap shut again immediately
     * and pull them onward, which would feel like a bug and would be
     * one. */
    audioEl.addEventListener('seeking', () => {
        if (PANEL.owner === 'row' && PANEL.row && PANEL.row.running) {
            PANEL.row.running = false;
            panelTransportState();
        }
    });
    return audioEl;
}

/* The only place that sets `src`.
 *
 * `owner === null` means STAGE ONLY: the source follows the selection,
 * but no one claims playback. This used to be a real bug — the deleted
 * `mapAttachAudio` set `src` on every selection, the replacement only
 * on playback. With autoplay off, the panel would then show voice B while
 * ▶ and the spacebar played voice A: you'd type one voice's name into
 * another's field, and because `controls` was off, there was no second
 * player to give it away.
 *
 * `back` rewinds to zero. Without that, `currentTime` stayed put after
 * a play-limit: playing the same voice again (⏮ on the same item, ▶ on
 * the same neighbor, a sequence with the same voice twice) would start at
 * the end of the play window, `timeupdate` would immediately advance
 * again — the item was effectively silent.
 */
function panelSetAudio(vectorId, owner, back = false) {
    const audioEl = panelAudio();
    if (!audioEl) return null;
    if (owner !== null) PANEL.owner = owner;
    if (audioEl.dataset.vid !== String(vectorId)) {
        audioEl.pause();
        audioEl.dataset.vid = String(vectorId);
        audioEl.src = `/api/speaker/sample/${vectorId}`;
    } else if (back && audioEl.currentTime > 0.05) {
        try { audioEl.currentTime = 0; } catch (e) { /* not loadable yet */ }
    }
    return audioEl;
}

/* The details show the chosen voice — or, with none chosen, the largest
 * groups. Both live in the docked side (voices.css), one at a time. */
function panelSideContent() {
    const sel = document.getElementById('mapSelection');
    const legend = document.getElementById('mapLegend');
    const chosen = PANEL.nodes >= 0 && sel && sel.dataset.built;
    if (sel) sel.style.display = chosen ? 'flex' : 'none';
    if (legend) legend.style.display = chosen ? 'none' : 'block';
}

function panelClose() {
    /* ALWAYS pause, not just the sequence. The `<audio>` sits IN the panel:
     * with `display:none` the transport bar disappears, the element stays
     * in the tree and keeps playing — you hear a voice and no longer have a
     * visible button to stop it. `graphRowStop` deliberately only pauses
     * what belongs to the sequence; a single listening sample fell through
     * because of that. */
    const audioEl = document.getElementById('mapAudio');
    if (audioEl) audioEl.pause();
    graphRowStop();
    PANEL.nodes = -1;
    MAP.chosen = null;
    panelSideContent();
}

/* When the whole map closes, the audio MUST stop. Previously Esc only
 * hid the stage, and the listening sample kept playing invisibly — you'd
 * then go looking for the source in another tab. */
function panelAudioOff() {
    const audioEl = document.getElementById('mapAudio');
    if (audioEl) audioEl.pause();
    graphRowStop();
}

// ── One voice into the panel ─────────────────────────────────────────────

/* The replacement for the old `mapPickVoice`, which rewrote everything.
 *
 * `listenAgain` is the core of the savings: when advancing within
 * the same group, the neighbor list and the recording list stay put. They
 * used to hang freshly in the tree on every item — that not only cost
 * time, it also threw the scrollbar back to the top.
 */
function mapPickVoice(i) {
    const m = MAP.members && MAP.members[i];
    if (!m) return;
    const el = panelFrame();
    if (!el) return;
    const fresh = PANEL.nodes < 0;
    MAP.chosen = m;
    PANEL.nodes = i;
    panelSideContent();
    // Choosing a voice is a step: Back (Alt+Left) closes it again; moving
    // on to the next voice is not
    if (fresh && typeof Shell !== 'undefined') Shell.step(panelClose);
    // Choosing a voice opens the details if they were hidden.
    const body = document.getElementById('mapBody');
    if (body && body.classList.contains('side-closed') && typeof vxSideToggle === 'function') {
        vxSideToggle(true);
    }

    const setThem = (id, text) => {
        const e = document.getElementById(id);
        if (e) e.textContent = text;
    };
    /* The filename large, the folder small next to it. Previously the whole
     * path stood bold in the header and took up three lines for deeply nested
     * folders — the name that actually matters stood at the very end. */
    const path = m.display || m.abs_path || '';
    const cut = path.lastIndexOf('/');
    setThem('mapTitle', cut >= 0 ? path.slice(cut + 1) : path);
    setThem('mapFolder', cut >= 0 ? vxLtr(path.slice(0, cut)) : '');
    const header = document.getElementById('mapTitle');
    if (header) header.title = path;
    setThem('mapMeta',
        `ID ${m.label} · window ${m.windows}`
        + ` · at ${Math.round(m.start_sec / 60)} min`
        + ` · ${Math.round(m.speech_sec)} s of speech`);

    const field_el = document.getElementById('mapName');
    // Only set when no one is working in it — see point 3 at the top.
    if (field_el && document.activeElement !== field_el) field_el.value = m.person || '';
    setThem('mapNamingState', '');

    // The listening sample belongs to the SELECTION, not just to playback —
    // otherwise ▶ would play the previous voice (see panelSetAudio).
    panelSetAudio(m.id, null);

    const nr = GRAPH.groupOf ? GRAPH.groupOf[i] : 0;
    if (nr !== PANEL.group_nr) { PANEL.group_nr = nr; graphShowContext(i); }
    else panelShowNeighbours(i);

    panelTransportState();
}

/* Listen without a sequence: a single item, in full, without a play limit
 * and without anything continuing afterward. */
function graphListen(j) {
    panelStopRow();
    mapPickVoice(j);
    const m = MAP.members[j];
    if (!m) return;
    const audioEl = panelSetAudio(m.id, 'single', true);
    if (audioEl) audioEl.play().catch(() => { /* abort is not an error */ });
}

/* Jump to a point without losing the selection. */
function graphJumpTo(j) {
    const cv = document.getElementById('mapCanvas');
    // Nothing to center in the tiles — the map stays fixed there.
    if (cv && (typeof TILE === 'undefined' || !TILE.on)) {
        MAP.ox = cv.clientWidth / 2 - MAP.sx * MAP.x[j];
        MAP.oy = cv.clientHeight / 2 - MAP.sx * MAP.y[j];
    }
    MAP.underMouse = j;
    mapDraw();
    // Clicking a row in the list IS a click on the voice —
    // so with autoplay, if it's turned on.
    panelVoiceClicked(j);
}

// ── Transport ────────────────────────────────────────────────────────────

/* The bar is ALWAYS there — even without a running sequence. Then the
 * buttons are disabled and the status says what's missing. A bar that
 * appears and disappears shifts everything while
 * reading; a disabled one says "this exists,
 * just not right now". */
function panelTransportState() {
    const r = PANEL.row;
    const audioEl = document.getElementById('mapAudio');
    const running = !!(audioEl && !audioEl.paused);

    const setEnabled = (id, on) => {
        const e = document.getElementById(id);
        if (e) e.disabled = !on;
    };
    setEnabled('mapKeyPrev', !!r && r.pos > 0);
    setEnabled('mapKeyNext', !!r && r.pos < r.listing.length - 1);
    setEnabled('mapKeyStop', !!r);
    setEnabled('mapKeyPlay', !!r || !!(audioEl && audioEl.dataset.vid));

    const play = document.getElementById('mapKeyPlay');
    if (play) play.textContent = running ? '⏸' : '▶';

    const state_info = document.getElementById('mapRowState');
    if (state_info) {
        state_info.textContent = r
            ? `${r.pos + 1} / ${r.listing.length}` + (r.running ? '' : ' · paused')
            : 'no queue';
        state_info.classList.toggle('map-p-state-off', !r);
    }
    const kind = document.getElementById('mapRunningFile');
    if (kind) {
        if (r) {
            const m = MAP.members[r.listing[r.pos]] || {};
            kind.textContent = `${r.description} · `
                + String(m.display || '').split('/').pop();
        } else {
            kind.textContent = '';
        }
    }
    const bar = document.getElementById('mapBarRow');
    if (bar) {
        bar.style.width = r && r.listing.length > 1
            ? `${(r.pos / (r.listing.length - 1)) * 100}%` : '0%';
    }
    const choice = document.getElementById('mapPlayback');
    if (choice) choice.value = String(GRAPH.playbackSec);
}

// ── The sequence player ──────────────────────────────────────────────────

/* What the sequence can do and why:
 *
 *  * **Pause and resume**, not just start. Whoever recognizes a voice
 *    wants to stop and name it, not chase after the next one.
 *  * **Forward and back.** The key move while listening is "once more,
 *    the one before" — without it you'd have to restart the whole sequence.
 *  * **Visible status.** Playback where you don't know how much
 *    longer it will last isn't listened to all the way through.
 *  * **All windows, or one per recording.** It's precisely the MULTIPLE
 *    windows of a recording that you want to compare — but for a quick
 *    overview across 40 recordings, one per file is the right thing.
 *  * **The map doesn't jump around.** The running point gets marked;
 *    it's only shifted if it runs off screen.
 */
function graphBuildRow(nr, perFile, openOnly = false) {
    const seen = new Set(), row = [];
    for (let j = 0; j < MAP.n; j++) {
        if (GRAPH.groupOf[j] !== nr) continue;
        if (!graphActive(j)) continue;
        // Only the ones not yet assigned — that's the work list. Whoever
        // listens through a group in order to name it doesn't want to hear
        // twenty times a voice that already has a name.
        if (openOnly && MAP.members[j].person) continue;
        const file = MAP.members[j].display;
        if (perFile) {
            if (seen.has(file)) continue;
            seen.add(file);
        }
        row.push(j);
    }
    return row;
}

function graphPlayGroup(nr, perFile = true, openOnly = false) {
    const listing = graphBuildRow(nr, perFile, openOnly);
    if (!listing.length) return;
    PANEL.row = {
        listing, pos: 0, running: true, nr, perFile, openOnly,
        description: openOnly ? 'open only'
            : (perFile ? 'one per recording' : 'all windows'),
    };
    graphRowPlay();
}

/* Plays the item at the current position. `pos` ALWAYS points to the
 * item currently playing — not to the next one. Otherwise "back" would
 * have to subtract two to go back one. */
function graphRowPlay() {
    const r = PANEL.row;
    if (!r) return;
    if (r.pos < 0) r.pos = 0;
    if (r.pos >= r.listing.length) { graphRowStop(); return; }

    const j = r.listing[r.pos];
    mapPickVoice(j);
    MAP.underMouse = j;
    graphRowIntoView(j);
    mapDraw();

    const m = MAP.members[j];
    const audioEl = panelSetAudio(m.id, 'row', true);
    if (!audioEl) return;
    if (r.running) audioEl.play().catch(() => { /* abort is not an error */ });
    panelTransportState();
}

function graphPlaybackSet(value) {
    GRAPH.playbackSec = Math.max(0, Math.min(30, Number(value) || 0));
    graphSaveSettings();
    panelTransportState();
    // Takes effect immediately: if the sequence is running and the item is
    // already past the new limit, it advances on the next `timeupdate` —
    // exactly what you'd expect when changing the value mid-listen.
}

/* Only shift the view if the point is actually about to run off screen. */
function graphRowIntoView(j) {
    const cv = document.getElementById('mapCanvas');
    if (!cv) return;
    const px = MAP.ox + MAP.sx * MAP.x[j];
    const py = MAP.oy + MAP.sx * MAP.y[j];
    const edge = 60;
    if (px > edge && px < cv.clientWidth - edge
        && py > edge && py < cv.clientHeight - edge) return;
    MAP.ox = cv.clientWidth / 2 - MAP.sx * MAP.x[j];
    MAP.oy = cv.clientHeight / 2 - MAP.sx * MAP.y[j];
}

function graphRowNext(step = 1) {
    const r = PANEL.row;
    if (!r) return;
    r.pos += step;
    if (r.pos < 0) r.pos = 0;
    graphRowPlay();
}

function graphRowPause() {
    const audioEl = document.getElementById('mapAudio');
    const r = PANEL.row;
    if (!r) {
        // Without a sequence, the button simply controls the running item.
        if (audioEl && audioEl.dataset.vid) {
            if (audioEl.paused) audioEl.play().catch(() => {}); else audioEl.pause();
        }
        return;
    }
    r.running = !r.running;
    if (!r.running) {
        if (audioEl) audioEl.pause();
        panelTransportState();
        return;
    }
    /* Resuming means: the SEQUENCE's item. Previously `play()` was called
     * blindly on the element — if you'd listened to a neighbor in the
     * meantime, that clip was loaded there, and ▶ would repeat it endlessly
     * while the display unwaveringly claimed "7 / 22". */
    const m = MAP.members[r.listing[r.pos]];
    if (m && audioEl && audioEl.dataset.vid !== String(m.id)) { graphRowPlay(); return; }
    PANEL.owner = 'row';
    if (audioEl) audioEl.play().catch(() => {});
    panelTransportState();
}

function graphRowStop() {
    const audioEl = document.getElementById('mapAudio');
    if (audioEl && PANEL.owner === 'row') audioEl.pause();
    PANEL.row = null;
    PANEL.owner = null;
    panelTransportState();
}

/* A running sequence needs to end as soon as the graph is reloaded:
 * the indices then point to different voices. */
function graphCancelRowIfNeeded() {
    if (PANEL.row) graphRowStop();
    PANEL.group_nr = null;      // the lists need to be rebuilt after this
}

// ── Surroundings: neighbors and the group ────────────────────────────────

/* What the selection actually answers lives in two lists:
 *
 *   1. **Most similar voices** — this point's neighbors, descending by
 *      similarity. That's the justification for why it sits where it sits.
 *   2. **The group** — all the recordings it consists of, for
 *      listening through.
 *
 * Both without a server request: nodes, edges and groups already live in
 * the browser. A round trip to the server on every click would be
 * noticeable and wouldn't bring anything new.
 *
 * Split into two functions because the two lists change at different
 * rates: the neighbors belong to the NODE and change with every item,
 * the group list belongs to the GROUP and stays the same for a whole
 * sequence.
 */
function graphShowContext(i) {
    panelShowNeighbours(i);
    panelShowGroup(GRAPH.groupOf ? GRAPH.groupOf[i] : 0);
}

function panelRow(j, extra) {
    const m = MAP.members[j];
    return `<div class="map-neighbour" onclick="graphJumpTo(${j})">
        <button class="map-mini" title="Listen"
                onclick="event.stopPropagation();graphListen(${j})">▶</button>
        <span class="map-neighbour-text">${esc(vxLtr(m.display || ''))}</span>
        <span class="map-small">${extra}</span>
    </div>`;
}

function panelShowNeighbours(i) {
    const target = document.getElementById('mapContext');
    if (!target || !GRAPH.kWeight) return;
    const K = GRAPH.edges, KW = GRAPH.kWeight;
    const neighbours = [];
    for (let e = 0; e < KW.length; e++) {
        const a = K[e * 2], b = K[e * 2 + 1];
        if (a === i) neighbours.push([b, KW[e]]);
        else if (b === i) neighbours.push([a, KW[e]]);
    }
    neighbours.sort((x, y) => y[1] - x[1]);
    target.innerHTML = `
        <div class="map-section">Most similar voices</div>
        ${neighbours.length
            ? neighbours.slice(0, 8).map(([j, w]) =>
                panelRow(j, `${w.toFixed(3)}${MAP.members[j].person
                    ? ' · ' + esc(MAP.members[j].person) : ''}`)).join('')
            : '<div class="map-small">No connection above the threshold — '
              + 'this voice stands alone.</div>'}`;
}

function panelShowGroup(nr) {
    const target = document.getElementById('mapGroupPart');
    if (!target) return;
    if (!nr) { target.innerHTML = ''; return; }
    /* Preserve typed text in the group name field.
     *
     * This function is also called after every INDIVIDUAL save (the count
     * of open voices changes, after all). It rewrites the whole
     * group section — including the half-filled-in "Name the whole group
     * …". That's the same design flaw the top of this
     * file fixes for the upper panel, just one level deeper. */
    /* ONLY what was TYPED, and only for the SAME group. The field used to
     * carry any value over — also the name pre-filled for the group shown
     * before: open "Alice"'s group, then one of Bob's, and the
     * field said "Alice" — one click on "All … open" would have named it
     * so. */
    const old_value = document.getElementById('mapGroupName');
    const remembered = old_value && old_value.dataset.typed === '1'
        && Number(old_value.dataset.nr) === Number(nr) ? old_value.value : null;
    const hasFocus = old_value && document.activeElement === old_value;

    // The group bundled by recording: a recording with six windows
    // should be ONE row, not six.
    const files = new Map();
    let totalWindows = 0, openWindows = 0;
    for (let j = 0; j < MAP.n; j++) {
        if (GRAPH.groupOf[j] !== nr) continue;
        const m = MAP.members[j];
        const e = files.get(m.display) || {n: 0, first: j, person: ''};
        e.n++;
        if (m.person) e.person = m.person;
        files.set(m.display, e);
        if (!graphActive(j)) continue;
        totalWindows++;
        if (!m.person) openWindows++;
    }
    const listing = [...files.entries()].sort((a, b) => b[1].n - a[1].n);
    const SHOWN = 40;

    target.innerHTML = `
        <div class="map-section">Group ${nr}: ${files.size}
             ${files.size === 1 ? 'recording' : 'recordings'}</div>
        <div class="map-p-starter">
            <button class="map-mini" title="One window per recording, in sequence"
                    onclick="graphPlayGroup(${nr}, true)">▶ ${plural(files.size, 'recording')}</button>
            <button class="map-mini" title="ALL windows of the group in sequence"
                    onclick="graphPlayGroup(${nr}, false)">▶ all ${totalWindows}</button>
            ${openWindows ? `<button class="map-mini map-mini-emphasis"
                    title="Only the voices without an assignment — the work list"
                    onclick="graphPlayGroup(${nr}, false, true)"
                    >▶ ${openWindows} open</button>` : ''}
            <button class="map-mini" title="Hide everything else"
                    onclick="graphOnlyThisGroup(${nr})">only this</button>
            <!-- Discarding is NOT the same as "only this": one is
                 a view for this session, the other goes into the
                 database and drops out of the grouping. That's why the
                 button stands apart and is marked as a real action. -->
            ${GRAPH.trash
                ? `<button class="map-mini" title="Return this group to the pool"
                        onclick="graphRestoreGroup(${nr})">↩ restore</button>`
                : `<button class="map-mini map-mini-danger"
                        title="Send this group to the trash — it drops out of the grouping but is kept"
                        onclick="graphDiscardGroup(${nr})">🗑 discard</button>`}
        </div>
        ${graphNameGroupRow(nr)}
        ${listing.slice(0, SHOWN).map(([, e]) =>
            panelRow(e.first, `${e.n}×${e.person ? ' · ' + esc(e.person) : ''}`)).join('')}
        ${listing.length > SHOWN
            ? `<div class="map-small">… and ${listing.length - SHOWN} more
               recordings (not shown)</div>` : ''}`;

    // Only write back what the human typed — the pre-filled
    // group name comes fresh from `graphNameGroupRow` anyway.
    const new_value = document.getElementById('mapGroupName');
    if (new_value && remembered !== null && remembered !== new_value.value && remembered.trim()) {
        new_value.value = remembered;
        new_value.dataset.typed = '1';
        if (hasFocus) new_value.focus();
    }
}

/* ── Naming the whole group at once ───────────────────────────────────────
 *
 * Previously a whole group could not be named at once: "Save" always only
 * named the currently selected voice. For a group of 22 voices that would
 * have been 22 individual clicks, and the map depends on naming being cheap.
 *
 * The name is pre-filled with what the group already carries (the most
 * common one in it). Whoever clicks a point that's already named "carol"
 * usually wants to name the remainder the same way — that should be a click,
 * not typing.
 *
 * The row states, BEFORE the click, how many voices are open: "18 of 23
 * unassigned". A button whose effect you only learn afterward is the
 * wrong button for a bulk change.
 */
function graphNameGroupRow(nr) {
    let open_count = 0, total = 0, foreign = 0;
    const name = graphGroupName(nr);
    for (let j = 0; j < MAP.n; j++) {
        if (GRAPH.groupOf[j] !== nr) continue;
        total++;
        const p = MAP.members[j].person;
        if (!p) open_count++;
        else if (name && p.toLowerCase() !== name.toLowerCase()) foreign++;
    }
    return `<div class="map-group-rename">
        <input type="text" id="mapGroupName" data-names placeholder="Name the whole group …"
               value="${esc(name)}" data-nr="${nr}"
               oninput="this.dataset.typed='1'">
        <button onclick="graphNameGroup(${nr})">All ${open_count} open</button>
    </div>
    <div class="map-small" id="mapGroupState">${plural(total, 'voice')} · ${open_count}
         unassigned${foreign
            ? ` · ${foreign} ${foreign === 1 ? 'carries a different name and is' : 'carry a different name and are'} left untouched` : ''}</div>`;
}

async function graphNameGroup(nr) {
    const field_el = document.getElementById('mapGroupName');
    const name = (field_el ? field_el.value : '').trim();
    if (!name) { field_el && field_el.focus(); return; }

    const ids = [];
    for (let j = 0; j < MAP.n; j++) {
        if (GRAPH.groupOf[j] === nr) ids.push(MAP.members[j].id);
    }
    if (!ids.length) return;
    // Group numbers are handed out anew by every graph load (threshold
    // slider, trash, a reload) — the answer is applied by the voice ids
    // sent, and the group's name only while the grouping is still this one.
    const seqAtSend = GRAPH.seq;

    try {
        const r = await fetch('/api/speaker/assign-many', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({vector_ids: ids, person: name}),
        });
        // Check status BEFORE reading: an error page is HTML, and
        // `r.json()` would fail on it with a message about angle
        // brackets instead of about the actual problem (voicemap.js).
        const d = await mapResponse(r);

        // Update the in-browser dataset instead of reloading everything: a
        // reload of the graph would discard the layout that took so long to
        // compute.
        // `MAP.name` runs parallel to `members` and is read while
        // drawing — anyone who only sets one of the two would only see the
        // new name after the next load.
        for (const id of ids) {
            const j = GRAPH.rowOf ? GRAPH.rowOf.get(id) : undefined;
            if (j === undefined) continue;
            const m = MAP.members[j];
            if (!m.person) { m.person = d.person; MAP.name[j] = d.person; }
        }
        if (GRAPH.seq === seqAtSend) {
            // Via the index — once per naming, not per voice, but there's
            // no reason to leave the linear search in place.
            const idx = (typeof graphGroupIndex === 'function')
                ? graphGroupIndex().get(nr) : null;
            const gr = idx ? idx.g : GRAPH.groups.find(g => g.nr === nr);
            if (gr) gr.name = d.person;
        }
        // Names color the tiles — the cached image is stale, and
        // the index holds the color assignment (a new name wants a slot).
        if (typeof graphDropGroupIndex === 'function') graphDropGroupIndex();
        if (typeof tileDiscardImage === 'function') tileDiscardImage();

        if (typeof vxUndoToast === 'function' && d.set_flag) {
            vxUndoToast(d.decision, `${plural(d.set_flag, 'voice')} named "${d.person}"`);
        }
        if (typeof VX !== 'undefined') VX.people = null;
        if (typeof historyRefresh === 'function') historyRefresh();
        let text = `${plural(d.set_flag, 'voice')} named "${esc(d.person)}"`;
        if (d.already_correct) text += ` · ${d.already_correct} already ${d.already_correct === 1 ? 'was' : 'were'}`;
        if (d.skipped) {
            text += ` · ${d.skipped} left untouched `
                  + `(${d.foreign_names.map(esc).join(', ')})`;
        }
        const state_info = document.getElementById('mapGroupState');
        if (state_info) state_info.innerHTML = text;

        mapDraw();
        graphLegend();
        graphApplyFilter();
    } catch (e) {
        const state_info = document.getElementById('mapGroupState');
        const text = 'Naming failed: ' + e.message;
        if (state_info) state_info.textContent = text; else alert(text);
    }
}

// ── Naming a single voice ────────────────────────────────────────────────

/* No longer takes an id, but works on whatever's in the field. Previously
 * the id was baked into the `onclick` text — and that text came from the
 * same `innerHTML` rebuild that broke everything else. */
async function mapRename(detach = false) {
    const m = MAP.chosen;
    if (!m) return;
    const field_el = document.getElementById('mapName');
    const name = detach ? '' : (field_el ? field_el.value.trim() : '');
    const state_info = document.getElementById('mapNamingState');
    const report = (t) => { if (state_info) state_info.textContent = t; };
    // The row of the voice being named — taken NOW: a click on another
    // point (or closing the details) while the request runs changes
    // PANEL.nodes, and the name would land on that voice in the drawing.
    const sent = PANEL.nodes;

    try {
        const r = await fetch('/api/speaker/assign', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({vector_id: m.id, person: name || null}),
        });
        await mapResponse(r);
        if (typeof VX !== 'undefined') VX.people = null;
        if (typeof historyRefresh === 'function') historyRefresh();

        // Still the same row? A graph reload in between renumbers the rows.
        let i = (sent >= 0 && MAP.members[sent] === m) ? sent
            : (GRAPH.rowOf && GRAPH.rowOf.has(m.id) ? GRAPH.rowOf.get(m.id) : -1);
        if (i >= 0 && MAP.members[i] !== m && MAP.members[i]) MAP.members[i].person = name;
        m.person = name;
        if (i >= 0) MAP.name[i] = name;
        if (typeof graphDropGroupIndex === 'function') graphDropGroupIndex();
        if (typeof tileDiscardImage === 'function') tileDiscardImage();
        mapDraw();
        graphLegend();
        graphApplyFilter();
        // The group list shows names and the count of open voices —
        // both of which just changed.
        // ... for the voice the details show now (which may be another one).
        const shown = PANEL.nodes;
        panelShowGroup(GRAPH.groupOf && shown >= 0 ? GRAPH.groupOf[shown] : 0);
        // No toast here: naming voice after voice in a sequence would pile
        // them up. The line says it, and the History undoes it.
        report(name ? `saved as "${name}"` : 'assignment removed');

        /* Automatically continue after naming — THAT's the loop the
         * sequence exists for: listen, name, next. Only when naming,
         * not when removing: whoever removes an assignment usually wants
         * to listen to the voice again right away.
         *
         * And only if the named voice is ALSO the current item of the
         * sequence. Otherwise the panel would jump back after naming any
         * arbitrarily clicked voice into a sequence the user had long
         * since left — complete with playback of an item from an
         * entirely different group. */
        if (name && PANEL.row && PANEL.row.listing[PANEL.row.pos] === i) {
            PANEL.row.running = true;      // the name field had paused it
            graphRowNext(1);
        }
    } catch (e) {
        report('Naming failed: ' + (e.message || e));
    }
}

// ── Keyboard ─────────────────────────────────────────────────────────────

/* While listening through, one hand is on the keyboard and one on the
 * mouse. Without keyboard control, every "once more" means aiming for
 * the other end of the screen — with 200 voices that's the difference
 * between "works" and "not done in practice". */
document.addEventListener('keydown', e => {
    const ov = document.getElementById('mapOverlay');
    if (!ov || !ov.classList.contains('visible')) return;
    // The map's keys only on the map tab, and not when another handler
    // (People → Review, Assign) already took the key: Space there also
    // toggled the hidden map sample.
    if (e.defaultPrevented || (typeof VX !== 'undefined' && VX.tab !== 'map')) return;
    const target = e.target;
    // In an input field, spacebar and arrows belong to the text.
    if (target && (target.tagName === 'INPUT' || target.tagName === 'SELECT'
                 || target.tagName === 'TEXTAREA')) return;
    if (e.key === ' ') { e.preventDefault(); graphRowPause(); }
    else if (e.key === 'ArrowRight' && PANEL.row) {
        e.preventDefault(); graphRowNext(1);
    } else if (e.key === 'ArrowLeft' && PANEL.row) {
        e.preventDefault(); graphRowNext(-1);
    }
});


/* ⤴ Location in recording — the path out of the map back into the transcript.
 *
 * The map can play a voice, but not explain it: 30 seconds from deep
 * inside a long recording don't say what it was about. The context lives in the
 * transcript, and until now there was no way there — you had to find the
 * file by hand and count out the time.
 *
 * `goToSearchResult` already does that completely (find the file, switch
 * directory if needed, select it, jump). A separate path next to it would
 * be a second version of the same thing — and the first one is the
 * tested one.
 *
 * The file name (`file_names[nodes.file_idx]`) is already relative to the
 * media root (`display()` in the endpoint strips the root), so exactly the form
 * expected there.
 */
function mapShowSpot() {
    const i = PANEL.nodes;
    const m = (i != null && i >= 0) ? MAP.members[i] : null;
    if (!m) return;
    const path = m.display || m.abs_path || '';
    if (!path) return;
    // Audio off before the map closes: otherwise the listening sample keeps
    // playing invisibly while the video starts in the transcript — two
    // voices at once, and you go looking for the second one in another tab.
    if (typeof panelAudioOff === 'function') panelAudioOff();
    if (typeof mapClose === 'function') mapClose();
    if (typeof goToSearchResult === 'function') {
        /* Pass the map's ROOT, not the empty string.
         *
         * `goToSearchResult` falls back to `lastSearchMediaDir` when the
         * third parameter is empty — the root of the last SEARCH. That's a
         * different state: the map's paths are shortened against the root
         * that was in effect when the map loaded, and that root shifts when
         * the folder changes. This used to produce "recording not found",
         * even though the server found the file immediately on request. */
        goToSearchResult(path, m.start_sec || 0,
                         (typeof GRAPH !== 'undefined' && GRAPH.root) || '');
    } else if (typeof Toast !== 'undefined') {
        Toast.error('The file view is not loaded.');
    }
}
