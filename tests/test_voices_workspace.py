"""The voices workspace — tabs instead of boxes over the map.

The floating panels over the map could not be moved and got in the way, so
the workspace is built from tabs; and locating a recording in the map has to
work again.

What these tests pin, as text (like the other map tests — no browser):

* five tabs, one page each, and every handler the pages call exists;
* nothing floats over the map: the details are docked beside it, the
  guided assignment is a page of its own, the hover label never catches a
  click and follows the pointer;
* a decision elsewhere doesn't reload the map behind the page — it is
  marked stale and fetched when shown;
* locating finds the recording by id, through a path parameter an
  absolute path survives, and says why when there is nothing to light;
* the date filter reads the German day-first dates too, the same in the
  map and in the extractor.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
STATIC = REPO / "whisper_tm" / "static"
HTML = (STATIC / "index.html").read_text(encoding="utf-8")
CSS = (STATIC / "css" / "voices.css").read_text(encoding="utf-8")
JS_FILES = [
    "voices.js",
    "voices_history.js",
    "voices_people.js",
    "voices_review.js",
    "voices_help.js",
    "voices_extract.js",
    "voices_clipedit.js",
    "voicemap.js",
    "voicemap_graph.js",
    "voicemap_panel.js",
    "voicemap_assign.js",
    "voicemap_tiles.js",
    "core.js",
    "state.js",
    "shell.js",
]
JS = {n: (STATIC / "js" / n).read_text(encoding="utf-8") for n in JS_FILES}
ALL_JS = "\n".join(JS.values())
HARNESS = Path(__file__).resolve().parent / "js" / "voicemap_harness.js"


def body(src: str, header: str) -> str:
    i = src.index(header)
    return src[i : src.index("\n}\n", i)]


def css_block(selector: str) -> str:
    i = CSS.index("\n" + selector + " {")
    return CSS[i : CSS.index("}", i)]


# ── The shell ────────────────────────────────────────────────────────────


class TestTabs:
    TABS = ("map", "assign", "people", "history", "extract", "help")

    def test_five_tabs_five_pages(self):
        for t in self.TABS:
            assert f'data-tab="{t}"' in HTML and f"vxTab('{t}')" in HTML, t
            assert f'data-page="{t}"' in HTML, t

    def test_a_tab_shows_its_page_and_hides_the_others(self):
        b = body(JS["voices.js"], "function vxTab(")
        assert "p.hidden = p.dataset.page !== name" in b
        # and asks its page to load what it shows
        for f in ("vxShowMap()", "assignOpen()", "peopleShow()", "historyShow()", "extractShow()"):
            assert f in b, f

    def test_the_last_tab_is_remembered(self):
        assert "localStorage.setItem(VX_MEMO, name)" in body(JS["voices.js"], "function vxTab(")
        # The rail's Voices place opens it where it was left
        assert "Shell.go('voices')" in HTML and "vxOpen()" in JS["shell.js"]

    def test_leaving_a_tab_stops_its_players(self):
        assert "vxAudioOff()" in body(JS["voices.js"], "function vxTab(")

    def test_the_close_button_closes_on_every_tab(self):
        """The ✕ must close the workspace on every tab: it used to be
        wired by the map's own set-up — opened on another tab (the one
        left last time), it did nothing."""
        assert re.search(r'id="mapClose"[^>]*onclick="vxClose\(\)"', HTML)
        assert "mapClose" not in body(JS["voicemap.js"], "function mapWireUp(")


def _handlers():
    """Every handler the voices pages call — in the markup and in the HTML
    the scripts write."""
    found = set()
    for src in [HTML, *JS.values()]:
        for m in re.finditer(r'on(?:click|input|change|keydown)="([^"]+)"', src):
            for f in re.findall(
                r"\b((?:vx|map|graph|assign|history|people|extract|panel)[A-Za-z]*)\(", m.group(1)
            ):
                found.add(f)
    return sorted(found)


@pytest.mark.parametrize("fn", _handlers())
def test_every_handler_has_a_function(fn):
    assert re.search(rf"function {fn}\s*\(", ALL_JS), f"{fn}() is called but never defined"


def test_there_are_handlers_to_check():
    assert len(_handlers()) >= 40


# ── Nothing over the map ─────────────────────────────────────────────────


class TestNothingFloatsOverTheMap:
    def test_the_details_and_the_groups_are_not_positioned_over_it(self):
        for sel in (".vx-side", ".map-selection", ".map-legend"):
            assert "position:absolute" not in css_block(sel), sel

    def test_assigning_is_a_page_not_a_box(self):
        assert 'class="vx-page vx-page-assign" data-page="assign" id="mapAssign"' in HTML
        assert ".map-assign {" not in CSS

    def test_the_hover_label_never_catches_a_click(self):
        assert "pointer-events:none" in css_block(".map-hint")

    def test_and_follows_the_pointer(self):
        b = body(JS["voicemap.js"], "function mapHintMove(")
        assert "MAP.mouse.x + 16" in b and "translate(" in b
        # flips near the edges instead of running off them
        assert "cv.clientWidth" in b and "cv.clientHeight" in b

    def test_the_graph_knobs_sit_in_a_menu(self):
        pop = HTML[HTML.index('id="mapLayoutPop"') :]
        pop = pop[: pop.index("</div>\n                </div>")]
        for k in ("mapK", "mapDrag", "mapSteps", "mapDensity", "mapNames", "mapOutlines"):
            assert f'id="{k}"' in pop, k


# ── The map is not reloaded behind the page ──────────────────────────────


class TestTheMapLoadsWhenShown:
    def test_a_decision_marks_it_stale(self):
        b = body(JS["voices.js"], "function vxMapChanged(")
        assert "VX.mapStale = true" in b

    def test_it_reloads_when_it_comes_on_screen(self):
        b = body(JS["voices.js"], "function vxShowMap(")
        assert "if (VX.mapStale)" in b and "mapLoadGraph()" in b

    @pytest.mark.parametrize("fn", ["assignDecide", "assignUndo", "historyAfterChange"])
    def test_no_page_reloads_it_directly(self, fn):
        src = ALL_JS
        head = f"async function {fn}(" if f"async function {fn}(" in src else f"function {fn}("
        b = body(src, head)
        assert "vxMapChanged()" in b and "mapLoadGraph()" not in b


# ── Locating ─────────────────────────────────────────────────────────────


class TestLocating:
    def test_the_path_travels_as_a_parameter(self):
        """An absolute path in the URL path lost its leading slash to the
        router's redirect and resolved to nothing."""
        assert "'/api/speaker/windows?file=' + encodeURIComponent(file)" in body(
            JS["voicemap_graph.js"], "async function mapLocate("
        )
        assert "/api/speaker/recording?file=" in (STATIC / "js" / "speaker.js").read_text("utf-8")

    def test_the_located_voices_are_pinned_through_the_minimum_size(self):
        a = body(JS["voicemap_graph.js"], "function graphBuildActive(")
        assert "GRAPH.pinned && GRAPH.pinned.has(i)" in a
        b = body(JS["voicemap_graph.js"], "function mapLocateApply(")
        assert "GRAPH.pinned = rows.length ? new Set(rows) : null" in b
        assert "shown although their group is below the minimum size" in b

    def test_it_says_why_when_nothing_is_lit(self):
        b = body(JS["voicemap_graph.js"], "function mapLocateApply(")
        assert "no voices are on file for this recording at all" in b
        assert "in the trash" in b

    def test_it_survives_a_new_threshold(self):
        b = body(JS["voicemap_graph.js"], "async function mapLoadGraph(")
        assert "mapLocateApply(false)" in b
        assert "GRAPH.rowOf.set(MAP.members[i].id, i)" in b

    def test_the_chosen_voice_is_kept_by_id_not_by_index(self):
        """After switching to the trash map the details showed one voice's
        file with another's group — they held an index of the old map."""
        b = body(JS["voicemap_graph.js"], "async function mapLoadGraph(")
        assert "const chosenId = MAP.chosen ? MAP.chosen.id : null;" in b
        assert "GRAPH.rowOf.get(chosenId)" in b

    def test_the_file_view_offers_no_map_where_there_is_nothing(self):
        spk = (STATIC / "js" / "speaker.js").read_text("utf-8")
        b = body(spk, "function speakerMapButtons(")
        assert "b.disabled = none" in b
        fv = (STATIC / "js" / "fileview.js").read_text("utf-8")
        assert "s && s !== '—'" in fv, "a Map button for the '—' of an undiarized transcript"

    def test_the_scope_select_reaches_the_filter(self):
        """It wrote to `GRAPH.filter.wo`, which nothing read — "name only"
        and "path only" never did anything."""
        assert "graphFilterSet('scope', this.value)" in HTML
        assert "graphFilterSet('wo'" not in HTML


class TestTheApi:
    @pytest.fixture
    def client(self, tmp_path, monkeypatch):
        from flask import Flask

        from whisper_tm.config import config
        from whisper_tm.core import speaker as speaker_core
        from whisper_tm.database.speaker import SpeakerDB

        media = tmp_path / "media"
        media.mkdir()
        monkeypatch.setattr(type(config), "media_dir", property(lambda self: media))
        db = SpeakerDB(tmp_path / "speaker.db")
        db.initialize()
        monkeypatch.setattr("whisper_tm.api.speaker.speaker_db", lambda: db)
        monkeypatch.setattr("whisper_tm.api.speaker._identity", lambda f: f)
        import numpy as np

        rec = "/mnt/elsewhere/2023-05-14 stream.mkv"  # outside the media folder
        db.write(
            rec,
            speaker_core.MODEL_BRAND,
            [
                {
                    "label": "SPEAKER_00",
                    "windows": i,
                    "start_sec": i * 30.0,
                    "speech_sec": 30.0,
                    "segments": 1,
                    "v_separated": np.ones(192, np.float32),
                }
                for i in range(3)
            ],
        )
        ids = [w["id"] for w in db.windows(rec)]
        db.move_to_trash(ids[:1])
        from whisper_tm.api.speaker import speaker_bp

        a = Flask(__name__)
        a.register_blueprint(speaker_bp)
        return a.test_client(), rec, ids

    def test_an_absolute_path_as_a_parameter(self, client):
        c, rec, ids = client
        d = c.get("/api/speaker/windows", query_string={"file": rec}).get_json()
        assert d["abs_path"] == rec and len(d["windows"]) == 3

    def test_the_windows_say_which_are_in_the_trash(self, client):
        c, rec, ids = client
        d = c.get("/api/speaker/windows", query_string={"file": rec}).get_json()
        assert {w["id"]: w["trashed"] for w in d["windows"]} == {
            ids[0]: True,
            ids[1]: False,
            ids[2]: False,
        }


# ── Dates in file names ──────────────────────────────────────────────────

NAMES = {
    "AUD-20190312-WA0007.mp3": "2019-03-12",
    "2023-05-14 10-20-30_converted.mp4": "2023-05-14",
    "Game Capture 2022.03.01 - 10.20.30.00.DVR.mp4": "2022-03-01",
    "recording - 05.02.22, 10.30.m4a": "2022-02-05",
    "Channel Name - Café Talk - (04.06.19).mp4": "2019-06-04",
    "Voice memo - 20.03.2016, 10_00.wav": "2016-03-20",
    "Channel 12.05.2020 - Vlog.mp4": "2020-05-12",
    "v1.2.3.mp4": "",
    "hash-7f3a.mp4": "",
    "PTT-20210917-WA0042.aac": "2021-09-17",
}


@pytest.mark.parametrize("name,want", sorted(NAMES.items()))
def test_the_extractor_reads_the_date(name, want):
    from whisper_tm.core.voice_extract import date_from_name

    assert date_from_name("/x/" + name) == want


@pytest.mark.skipif(shutil.which("node") is None, reason="node not available")
def test_the_map_reads_it_the_same_way():
    src = (
        f"const {{API}} = require({str(HARNESS)!r});\n"
        f"const names = {json.dumps(list(NAMES))};\n"
        "console.log('§§' + JSON.stringify(names.map(n => API.graphDateFromName('/x/' + n))));"
    )
    done = subprocess.run(
        ["node", "-e", src],
        capture_output=True,
        text=True,
        timeout=60,
        env={
            "MAP_JS": str(STATIC / "js" / "voicemap_graph.js"),
            "PANEL_JS": str(STATIC / "js" / "voicemap_panel.js"),
            "PATH": os.environ.get("PATH", ""),
        },
    )
    assert done.returncode == 0, done.stderr
    got = json.loads([z for z in done.stdout.splitlines() if z.startswith("§§")][-1][2:])
    assert got == list(NAMES.values())


# ── One id, one element ──────────────────────────────────────────────────


def test_no_id_is_given_twice():
    """The extractor's clip list and its "clips at most" field were both
    `xClips`: `getElementById` found the field, and the list was
    written into an input — no clip ever showed. Every fixed id in the
    voices markup and in what the voices scripts write must be unique."""
    seen: dict[str, str] = {}
    page = HTML[HTML.index('id="mapOverlay"') : HTML.index('id="statsOverlay"')]
    for where, src in [("index.html", page)] + [
        (n, JS[n]) for n in JS_FILES if n.startswith("voices")
    ]:
        for m in re.finditer(r'\bid="([A-Za-z][\w-]*)"', src):
            i = m.group(1)
            assert i not in seen, f"id {i!r} in {where} and in {seen[i]}"
            seen[i] = where
    for f in ("num('xClips'", 'id="xClipList"'):
        assert f in JS["voices_extract.js"], f


# ── The clip editor ──────────────────────────────────────────────────────


class TestTheClipEditor:
    ED = JS["voices_clipedit.js"]

    def test_the_audio_is_held_in_memory(self):
        b = body(self.ED, "async function clipEditOpen(")
        assert "/api/voices/extract/context/" in b and "decodeAudioData" in b

    def test_the_pause_rule_is_the_exports(self):
        from whisper_tm.core import voice_extract as vx

        assert f"const CLIPEDIT_FRAME = {vx.SNAP_FRAME};" in self.ED
        assert f"const CLIPEDIT_DIP = {vx.SNAP_DIP};" in self.ED
        assert f"const CLIPEDIT_FLOOR_MAX = {vx.SNAP_FLOOR_MAX};" in self.ED

    def test_esc_closes_the_editor_first_even_while_loading(self):
        i = self.ED.index("document.addEventListener('keydown'")
        handler = self.ED[i : self.ED.index("}, true);", i) + 9]
        assert handler.endswith("}, true);"), "not in the capture phase"
        esc_at = handler.index("e.key === 'Escape'")
        assert esc_at < handler.index("getElementById('xWave')"), (
            "Esc waits for the waveform — and falls through to the workspace while loading"
        )
        assert "e.stopImmediatePropagation()" in handler[esc_at : esc_at + 200]

    def test_the_preview_is_not_computed_twice_for_the_same_filters(self):
        b = body(JS["voices_extract.js"], "function extractWireForm(")
        assert "if (extractKey() === EXTRACT.key) return;" in b


# ── Listening through a person, words, voices without spans ──────────────────────


class TestListeningThrough:
    """A list to listen through everything, with autoplay and filters."""

    def test_a_person_opens_the_whole_list_not_six_samples(self):
        row = body(JS["voices_people.js"], "function peopleRowHtml(")
        assert "reviewOpen(" in row and "vxSamplesHtml" not in row
        assert '<script src="/static/js/voices_review.js' in HTML
        assert HTML.index("voices_people.js") < HTML.index("voices_review.js")

    def test_the_sequence_plays_the_first_seconds_and_goes_on(self):
        w = body(JS["voices_review.js"], "function reviewWire(")
        assert "'timeupdate'" in w and "REVIEW.per" in w and "reviewStep(1)" in w
        assert "'ended'" in w

    def test_esc_stops_the_list_before_it_closes_the_workspace(self):
        js = JS["voices_review.js"]
        i = js.index("document.addEventListener('keydown'")
        handler = js[i : js.index("}, true);", i)]
        assert "stopImmediatePropagation" in handler and "reviewStop()" in handler

    def test_the_spans_column_only_where_older_voices_exist(self):
        draw = body(JS["voices_people.js"], "function peopleDraw(")
        assert "PEOPLE.legacy" in draw
        assert "PEOPLE.legacy ?" in body(JS["voices_people.js"], "function peopleRowHtml(")

    def test_every_review_row_has_the_same_cells(self):
        # Every row is its own grid: only fixed columns line up
        assert (
            "auto" not in css_block(".vx-rv-row").split("grid-template-columns:")[1].split(";")[0]
        )


class TestFromConversations:
    """The search view's Conversations tab hands its filters over (✂
    Extract there): the form says them in words, the preview counts the
    conversations, and the People page's Extract takes all of a voice."""

    def test_the_filters_go_with_every_preview(self):
        assert "conversations: EXTRACT.conv || {}" in body(
            JS["voices_extract.js"], "function extractOptions("
        )
        shell = body(JS["voices_extract.js"], "function extractShellHtml(")
        assert 'id="xConv"' in shell and "extractConvHtml()" in shell

    def test_the_people_page_takes_all_of_a_voice(self):
        b = body(JS["voices_extract.js"], "async function extractFor(")
        assert "EXTRACT.conv = null" in b

    def test_changing_them_closes_this_workspace_first(self):
        b = body(JS["voices_extract.js"], "function extractFindConversations(")
        assert b.index("vxClose()") < b.index("cvShow(")

    def test_the_preview_counts_the_conversations(self):
        d = body(JS["voices_extract.js"], "function extractDrawPreview(")
        assert "conv.found" in d and "conv.people_unknown" in d


class TestOlderVoicesAndWords:
    def test_no_spans_option_in_the_form(self):
        shell = body(JS["voices_extract.js"], "function extractShellHtml(")
        assert "xClean" not in shell and "speech spans" not in shell
        assert "EXTRACT.strict ? 'strict' : 'best'" in body(
            JS["voices_extract.js"], "function extractOptions("
        )

    def test_the_note_only_where_the_preview_has_older_voices(self):
        d = body(JS["voices_extract.js"], "function extractDrawPreview(")
        assert "const legacy = s.by_transcript > 0 || leftOut > 0" in d
        assert "${legacy ?" in d

    def test_words_in_the_list_and_in_the_editor(self):
        assert "extractTextHtml(c)" in body(JS["voices_extract.js"], "function extractClipHead(")
        assert "extractTextHtml(c)" in body(JS["voices_clipedit.js"], "function clipEditHtml(")
        play = body(JS["voices_clipedit.js"], "function clipEditPlay(")
        assert "extractWordsAt(" in play
        assert "c.start + a.currentTime" in JS["voices_extract.js"]
