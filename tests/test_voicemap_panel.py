"""The voice map's workspace — six errors and their safeguards.

The playback panel in the map was buggy and got a complete redesign and review.

The review showed: almost all buggy stuff came from ONE design error. On each
selection, ``mapPickVoice`` rewrote the entire field via ``innerHTML`` —
header, audio, name field, transport, lists. And the sequence player selects
on EVERY piece. This led to:

1. **Two voices simultaneously.** The comment about ``mapAttachAudio``
   promised a single ``<audio>`` for the entire session. It was wrong:
   ``innerHTML =`` removes the old element from the tree, then ``getElementById``
   finds nothing and a new one was built. A detached ``<audio>`` keeps playing
   — the old piece ran under the new one, and ``audioEl.pause()`` hit only the
   fresh, still muted element.
2. **Transport disappeared.** ⏮⏯⏭⏹ were INSIDE the field. A click on a point
   during a running sequence rebuilt the field — the sequence kept running,
   buttons were gone.
3. **Typed text was lost.** Listen, recognize, type a name — after three seconds
   the sequence advances, the field is rebuilt, the typed text is gone. Exactly
   the gesture the map is for.
4. **The list jumped.** Forty recordings were rebuilt with each piece; the
   scroll bar jumped back to the start.
5. **Handlers belonged to nobody.** ``audioEl.onended`` pointed to the sequence's
   "next" and stayed stuck there: whoever pressed ▶ on a neighbor during a
   running sequence got the next piece of a sequence they didn't start — and
   ``ontimeupdate`` cut the single sample after three seconds.
6. **Audio kept playing after closing.** Esc only hid the stage.

All six were measured in a real browser against a real inventory of
several thousand voices — here are the structural conditions to keep them from coming
back.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
STATIC = REPO / "whisper_tm" / "static"
PANEL = (STATIC / "js" / "voicemap_panel.js").read_text(encoding="utf-8")
# Without comments — several checks below ask "is this still in the CODE?",
# and the file's header TELLS THE STORY of exactly these old lines.
PANEL_CODE = "\n".join(
    z for z in PANEL.splitlines() if not z.lstrip().startswith(("*", "//", "/*"))
)
MAP = (STATIC / "js" / "voicemap.js").read_text(encoding="utf-8")
GRAPH = (STATIC / "js" / "voicemap_graph.js").read_text(encoding="utf-8")
VOICES = (STATIC / "js" / "voices.js").read_text(encoding="utf-8")
# The map's styles live in the voices workspace's own sheet.
CSS = (STATIC / "css" / "voices.css").read_text(encoding="utf-8")
HTML = (STATIC / "index.html").read_text(encoding="utf-8")


def fn_body(source: str, header: str, to_value: str = "\n}") -> str:
    """A function from its head to the closing brace in column 0."""
    i = source.index(header)
    return source[i : source.index(to_value, i)]


class TestTheScaffoldIsBuiltOnlyOnce:
    """The core of the rebuild — everything else hung off this."""

    def test_there_is_a_scaffold_function(self):
        assert "function panelFrame(" in PANEL

    def test_it_builds_only_the_first_time(self):
        """Without the lock, every call would be a full rebuild again."""
        r = fn_body(PANEL, "function panelFrame(")
        assert "el.dataset.built" in r
        assert "if (!el || el.dataset.built) return el;" in r

    def test_the_selection_no_longer_rewrites_everything(self):
        """`mapPickVoice` must no longer discard anything that is in use.

        This is the test that catches errors 1 through 4 at once: as soon as
        an `innerHTML` over the whole field shows up here again, all of them
        are back.
        """
        r = fn_body(PANEL, "function mapPickVoice(")
        assert "innerHTML" not in r, (
            "the selection rebuilds the field again — that's exactly where "
            "the two simultaneous voices, the vanished transport, and "
            "the lost typed text came from"
        )

    def test_the_lists_are_rebuilt_only_on_a_group_change(self):
        """Error 4: the scroll bar jumped to the start on every piece."""
        r = fn_body(PANEL, "function mapPickVoice(")
        assert "if (nr !== PANEL.group_nr)" in r
        assert "else panelShowNeighbours(i);" in r


class TestASingleAudioElement:
    """Error 1 — two voices at once."""

    def test_the_element_is_built_only_once(self):
        r = fn_body(PANEL, "function panelAudio(")
        assert "if (audioEl) return audioEl;" in r
        assert "createElement('audio')" in r

    def test_there_is_only_one_place_that_builds_it(self):
        for file, name in ((MAP, "voicemap.js"), (GRAPH, "voicemap_graph.js")):
            assert "createElement('audio')" not in file, name
        assert PANEL.count("createElement('audio')") == 1

    def test_only_the_source_changes(self):
        """The only way to set `src` — and it's tied to the owner."""
        assert PANEL.count("audioEl.src =") == 1
        r = fn_body(PANEL, "function panelSetAudio(")
        assert "PANEL.owner = owner;" in r
        assert "audioEl.dataset.vid !== String(vectorId)" in r

    def test_the_old_re_parenting_is_gone(self):
        """`mapAttachAudio` used to hang it in a spot that got rebuilt."""
        for file in (MAP, GRAPH, PANEL):
            assert "function mapAttachAudio" not in file


class TestTheOwner:
    """Error 5 — the handlers belonged to nobody."""

    def test_the_listeners_are_attached_to_the_element(self):
        """`addEventListener` instead of `onended =`: whoever assigns,
        overwrites — and whoever forgets to overwrite leaves a ghost behind."""
        assert "audioEl.onended" not in PANEL_CODE
        assert "audioEl.ontimeupdate" not in PANEL_CODE
        assert "audioEl.addEventListener('ended'" in PANEL
        assert "audioEl.addEventListener('timeupdate'" in PANEL

    def test_the_end_asks_the_owner(self):
        i = PANEL.index("audioEl.addEventListener('ended'")
        r = PANEL[i : i + 400]
        assert "PANEL.owner === 'row'" in r

    def test_listening_individually_does_not_belong_to_the_queue(self):
        r = fn_body(PANEL, "function graphListen(")
        assert "'single'" in r

    def test_the_queue_registers_its_claim(self):
        r = fn_body(PANEL, "function graphRowPlay(")
        assert "'row'" in r


class TestTypedTextSurvives:
    """Error 3 — the very handgrip the map exists for."""

    def test_the_name_field_is_not_overwritten_while_typing(self):
        r = fn_body(PANEL, "function mapPickVoice(")
        assert "document.activeElement !== field_el" in r

    def test_clicking_into_the_name_field_halts_the_queue(self):
        """Otherwise every keystroke is a race against the player."""
        i = PANEL.index("bind('mapName', 'focus'")
        assert "graphRowPause()" in PANEL[i : i + 250]

    def test_enter_saves_and_moves_on(self):
        i = PANEL.index("bind('mapName', 'keydown'")
        r = PANEL[i : i + 250]
        assert "'Enter'" in r and "mapRename()" in r
        # … and it continues inside the naming function itself — but only if
        # the named voice is ALSO the sequence's current piece.
        b = fn_body(PANEL, "async function mapRename(")
        assert "PANEL.row.listing[PANEL.row.pos] === i" in b
        assert "graphRowNext(1)" in b

    def test_after_naming_the_queue_runs_again(self):
        """The name field had paused it — otherwise it would stay stopped and
        you'd have to press ▶ again after every name."""
        b = fn_body(PANEL, "async function mapRename(")
        assert "PANEL.row.running = true;" in b


class TestTheTransportStays:
    """Error 2 — the buttons vanished mid-playback."""

    def test_the_bar_is_part_of_the_scaffold(self):
        r = fn_body(PANEL, "function panelFrame(")
        for label in (
            "mapKeyPrev",
            "mapKeyPlay",
            "mapKeyNext",
            "mapKeyStop",
            "mapRowState",
            "mapPlayback",
        ):
            assert f'id="{label}"' in r, label

    def test_without_a_queue_it_is_disabled_not_hidden(self):
        """A bar that appears and disappears shifts everything below it while
        reading."""
        r = fn_body(PANEL, "function panelTransportState(")
        assert "e.disabled = !on;" in r
        assert "'no queue'" in r

    def test_it_sticks_to_the_top_and_only_the_list_scrolls(self):
        block = CSS[CSS.index("\n.map-selection {") :]
        block = block[: block.index("}")]
        assert "flex-direction:column" in block
        assert "overflow:hidden" in block
        body = CSS[CSS.index("\n.map-p-body {") :]
        assert "overflow-y:auto" in body[: body.index("}")]

    def test_the_field_stays_inside_the_stage(self):
        """`100vh` was wrong once: the field stuck out about 200 px past the
        bottom edge, the last lines were unreachable. Docked in the side
        now, it fills the side and shrinks with it — no height of
        its own at all."""
        block = CSS[CSS.index("\n.map-selection {") :]
        block = block[: block.index("}")]
        assert "flex:1" in block and "min-height:0" in block
        side = CSS[CSS.index("\n.vx-side {") :]
        side = side[: side.index("}")]
        assert "min-height:0" in side and "overflow:hidden" in side
        rules = [
            z.strip()
            for z in block.splitlines()
            if ":" in z and not z.strip().startswith(("*", "/*"))
        ]
        assert not any("100vh" in z for z in rules), rules


class TestClosingStops:
    """Error 6 — the audio kept playing invisibly."""

    def test_the_map_turns_the_audio_off(self):
        # Closing goes through the workspace, which stops every player —
        # the map's own included.
        assert "vxClose()" in fn_body(MAP, "function mapClose(")
        assert "vxAudioOff()" in fn_body(VOICES, "function vxClose(")
        assert "panelAudioOff()" in fn_body(VOICES, "function vxAudioOff(")

    def test_and_ends_the_queue_in_doing_so(self):
        r = fn_body(PANEL, "function panelAudioOff(")
        assert "audioEl.pause()" in r and "graphRowStop()" in r


class TestTheBrowsersPlayer:
    """**Reversed deliberately.**

    This class used to record the opposite: `controls` off, with a custom
    bar instead. The objection back then was valid — two playback buttons
    side by side doing different things is confusing. It just weighs less
    than this: the default browser player lets you save the clips with
    right-click.

    Right-click → "Save audio as". A self-built bar can't do that, and the
    listening sample is a finished MP3 under a normal address. The old
    objection is now answered differently: the player belongs to the PIECE,
    the buttons below it belong to the SEQUENCE, and the row says so.
    """

    def test_the_audio_element_carries_the_browser_controls(self):
        assert "audioEl.controls = true;" in PANEL
        assert ".map-audio { display:block;" in CSS, "the player is hidden"

    def test_there_is_no_second_slider_for_the_same_piece(self):
        """The duplication the old comment warned about — just reversed: now
        a custom bar of its OWN would be the second one."""
        assert "mapBarPiece" not in PANEL
        assert "panelBarPiece" not in PANEL
        assert 'id="mapClock"' not in PANEL, "two clocks for the same piece"

    def test_the_queue_keeps_its_own_bar(self):
        """What the browser's player can NOT answer: "how far am I into the
        sequence?" — 36 windows are not a single piece."""
        assert 'id="mapBarRow"' in PANEL
        assert "map-p-queue-label" in PANEL, "the buttons don't say what for"

    def test_the_preview_limit_still_applies_only_to_the_queue(self):
        """It cuts off each piece after N seconds and advances. For a single
        listening sample that would be wrong — whoever clicks it wants to
        hear it in full."""
        r = fn_body(PANEL, "    audioEl.addEventListener('timeupdate'")
        assert "if (PANEL.owner !== 'row') return;" in r
        assert "GRAPH.playbackSec" in r


class TestAutoplay:
    """An autoplay toggle: clicking a node in a group plays it
    automatically.

    While assigning, you click through a group; doing the same second reach
    for the play button every time is the handgrip that makes the work
    tedious. As a toggle, not fixed: someone just looking at a map doesn't
    want audio on every click.
    """

    def test_there_is_a_switch(self):
        r = fn_body(PANEL, "function panelFrame(")
        assert 'id="mapAutoplay"' in r
        assert "function panelToggleAutoplay(" in PANEL
        i = PANEL.index("bind('mapAutoplay', 'change'")
        assert "panelToggleAutoplay" in PANEL[i : i + 200]

    def test_it_defaults_to_on(self):
        r = fn_body(PANEL, "function panelLoadAutoplay(")
        assert "w === null ? true : w === '1'" in r

    def test_it_survives_a_reload(self):
        r = fn_body(PANEL, "function panelToggleAutoplay(")
        assert "localStorage.setItem(PANEL_MEMO" in r

    def test_only_a_click_starts_playback(self):
        """The decisive difference. The sequence player also selects a voice
        on EVERY piece — if playback were tied to `mapPickVoice`, every
        piece would play twice."""
        r = fn_body(PANEL, "function mapPickVoice(")
        assert "PANEL.autoplay" not in r, (
            "the selection itself starts playback — then the sequence also plays twice"
        )
        k = fn_body(PANEL, "function panelVoiceClicked(")
        assert "if (PANEL.autoplay) graphListen(i);" in k
        assert "else mapPickVoice(i);" in k

    def test_the_canvas_takes_the_same_path(self):
        """Graph as well as tiles — both click paths must call the same
        function, otherwise autoplay only applies in one of the two views."""
        i = MAP.index("cv.addEventListener('click'")
        r = MAP[i : i + 600]
        assert r.count("panelVoiceClicked(") == 2
        assert "mapPickVoice(" not in r

    def test_the_list_rows_too(self):
        r = fn_body(PANEL, "function graphJumpTo(")
        assert "panelVoiceClicked(j)" in r

    def test_a_click_halts_the_running_queue(self):
        """Whoever clicks away has left the sequence. Letting it keep running
        would mean that, three seconds later, something other than what was
        clicked would play without any action taken."""
        # The stop has lived in `panelStopRow` since the review — it's
        # hooked into EVERY path that loads a single clip, not just the click.
        assert "panelStopRow()" in fn_body(PANEL, "function panelVoiceClicked(")
        h = fn_body(PANEL, "function panelStopRow(")
        assert "PANEL.row.running = false;" in h
        assert "audioEl.pause()" in h

    def test_afterwards_naming_does_not_jump_back_into_the_queue(self):
        """The paused sequence persists (⏭ should still work), but must not
        jump back in when a completely different voice is named — otherwise
        you'd end up back in a group you left."""
        b = fn_body(PANEL, "async function mapRename(")
        assert "PANEL.row.listing[PANEL.row.pos] === i" in b


class TestWhatTheReviewFound:
    """Ten findings from a code review, six of them here.

    They all hang off the same seam: the audio element now belongs to the
    scaffold, not to the selection anymore — and during the rebuild, the
    link between "what's shown in the field" and "what's loaded in the
    player" got severed.
    """

    def test_the_selection_prepares_the_audio_sample(self):
        """Finding 1. Without this, ▶ with autoplay off would play the
        PREVIOUS voice while the field showed the newly named one — and
        because `controls` is off, there was no second player to reveal
        that. You'd then type the name of one voice into the field of
        another.
        """
        r = fn_body(PANEL, "function mapPickVoice(")
        assert "panelSetAudio(m.id, null)" in r, "the selection doesn't set the audio source"

    def test_preparing_does_not_claim_playback(self):
        """`owner === null` must NOT overwrite the owner — otherwise a
        running sequence loses its claim as soon as it selects something."""
        r = fn_body(PANEL, "function panelSetAudio(")
        assert "if (owner !== null) PANEL.owner = owner;" in r

    def test_playing_again_starts_from_the_beginning(self):
        """Finding 8. `currentTime` stayed put after a playback limit: playing
        the same voice again started at the end of the window, `timeupdate`
        immediately advanced — the piece was silent."""
        r = fn_body(PANEL, "function panelSetAudio(")
        assert "back" in r and "audioEl.currentTime = 0" in r
        # … and both playback paths request it.
        assert "panelSetAudio(m.id, 'single', true)" in PANEL
        assert "panelSetAudio(m.id, 'row', true)" in PANEL

    def test_closing_the_field_always_halts(self):
        """Finding 4. `graphRowStop` deliberately only pauses what belongs
        to the sequence — a single listening sample kept playing invisibly
        after ✕, with the transport bar hidden."""
        r = fn_body(PANEL, "function panelClose(")
        assert "audioEl.pause()" in r
        assert r.index("audioEl.pause()") < r.index("graphRowStop()")

    def test_next_plays_the_queues_piece(self):
        """Finding 7. After a single click in the middle of a sequence, a
        foreign clip sat in the element; ▶ repeated it endlessly while the
        display kept stubbornly claiming "7 / 22"."""
        r = fn_body(PANEL, "function graphRowPause(")
        assert "audioEl.dataset.vid !== String(m.id)" in r
        assert "graphRowPlay(); return;" in r

    def test_every_single_clip_leaves_the_queue(self):
        """Found while re-checking in the browser, not named by the review.

        `panelVoiceClicked` stopped the sequence — but the ▶ button in
        the neighbor and recording lists calls `graphListen` DIRECTLY.
        The sequence formally kept running there, the `ended` listener saw
        `running === true` and, uninvited, pushed the next sequence piece
        after the listened-to neighbor. That's why the stop is now hooked
        into `graphListen` itself, i.e. into every path that loads a
        single clip.
        """
        assert "function panelStopRow(" in PANEL
        for f in ("function panelVoiceClicked(", "function graphListen("):
            assert "panelStopRow()" in fn_body(PANEL, f), f

    def test_the_queue_is_not_discarded_in_the_process(self):
        """⏭ and ⏮ should keep working afterward — paused is not the same as
        finished."""
        r = fn_body(PANEL, "function panelStopRow(")
        assert "PANEL.row = null" not in r
        assert "PANEL.row.running = false;" in r

    def test_the_group_field_survives_a_single_save(self):
        """Finding 9. `panelShowGroup` is called after EVERY individual
        save and used to rewrite the entire group section — including the
        half-filled-in "Name the whole group …" field."""
        r = fn_body(PANEL, "function panelShowGroup(")
        assert "new_value.value = remembered;" in r

    def test_but_only_what_was_typed_for_the_same_group(self):
        """The field used to carry ANY value over — also the name
        pre-filled for the group shown before. Open Bob's group, then one
        of Carol's: the field said "Bob", and "All … open" would have
        named Carol's voices so."""
        r = fn_body(PANEL, "function panelShowGroup(")
        assert "old_value.dataset.typed === '1'" in r
        assert "Number(old_value.dataset.nr) === Number(nr)" in r
        row = fn_body(PANEL, "function graphNameGroupRow(")
        assert 'data-nr="${nr}"' in row
        assert "oninput=\"this.dataset.typed='1'\"" in row


class TestKeyboard:
    """One hand stays on the keyboard while listening through."""

    def test_space_bar_and_arrows(self):
        i = PANEL.index("document.addEventListener('keydown'")
        r = PANEL[i:]
        assert "e.key === ' '" in r and "graphRowPause()" in r
        assert "'ArrowRight'" in r and "graphRowNext(1)" in r
        assert "'ArrowLeft'" in r and "graphRowNext(-1)" in r

    def test_inside_an_input_they_belong_to_the_text(self):
        """Otherwise you couldn't type a space in the name field."""
        i = PANEL.index("document.addEventListener('keydown'")
        r = PANEL[i:]
        assert "'INPUT'" in r and "return;" in r

    def test_only_while_the_map_is_open(self):
        i = PANEL.index("document.addEventListener('keydown'")
        r = PANEL[i : i + 400]
        assert "classList.contains('visible')" in r


class TestWiringUp:
    """The handlers are wired once, on the scaffold, not as text in the HTML."""

    def test_the_bar_is_wired_through_addeventlistener(self):
        r = fn_body(PANEL, "function panelWire(")
        for label in (
            "mapKeyPrev",
            "mapKeyNext",
            "mapKeyPlay",
            "mapKeyStop",
            "mapPlayback",
            "mapKeyName",
            "mapKeyDetach",
            "mapSelectionClose",
        ):
            assert f"'{label}'" in r, label

    def test_every_scaffold_handler_has_a_function(self):
        """Same check as for the static markup, just for the scaffold: the
        list rows still carry `onclick`, because each row reads differently."""
        calls = {
            m.group(2)
            for m in re.finditer(
                r'\bon(click)="(?:event\.stopPropagation\(\);)?'
                r"((?:map|graph)[A-Za-z_$][\w$]*)\s*\(",
                PANEL,
            )
        }
        assert calls, "no handlers found — the regex isn't matching"
        all_items = PANEL + MAP + GRAPH
        for f in sorted(calls):
            assert re.search(rf"function {re.escape(f)}\s*\(", all_items), f

    def test_the_file_is_included(self):
        assert "voicemap_panel.js" in HTML

    def test_a_new_graph_ends_the_queue(self):
        """The indices then point to different voices afterward — and the
        lists need to be rebuilt too."""
        r = fn_body(PANEL, "function graphCancelRowIfNeeded(")
        assert "graphRowStop()" in r
        assert "PANEL.group_nr = null" in r
        i = GRAPH.index("async function mapLoadGraph")
        assert "graphCancelRowIfNeeded()" in GRAPH[i : i + 400]


class TestMeasuredOnTheRealInventory:
    """What the browser said against a real inventory.

    This class doesn't check code — it records that the six points weren't
    just fixed, but re-measured, and WITH WHAT. Without this note, the
    collection above would read like a guess.

        Audio elements after 5 s sequence   1  (before: one per piece)
        Transport after a point click       operable
        Name field after 3 s of typing      'sample name' kept
        Single listening sample after 4 s   3.9 s (before: cut off after 1 s)
        Scroll position across a switch     200 → 200
        Audio after closing                 stopped
    """

    def test_the_measurement_is_in_the_file_header(self):
        header = PANEL[: PANEL.index("const PANEL = {")]
        for keyword in (
            "TWO VOICES AT ONCE",
            "THE TRANSPORT DISAPPEARED",
            "TYPED TEXT WAS LOST",
            "THE LIST JUMPED",
            "THE HANDLERS BELONGED TO NO ONE",
        ):
            assert keyword in header, keyword
