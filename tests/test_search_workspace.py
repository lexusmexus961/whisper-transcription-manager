"""The search workspace and Ask — the markup and its wiring.

The search page follows the existing design with a bigger panel and a chat:
the search became a workspace in the voices workspace's system (head bar
with tabs, toolbars), and Ask its fourth tab. These tests pin the shell and that every control calls a
function that exists.
"""

from __future__ import annotations

import re
from pathlib import Path

STATIC = Path(__file__).resolve().parent.parent / "whisper_tm" / "static"
HTML = (STATIC / "index.html").read_text(encoding="utf-8")
JS = {
    n: (STATIC / "js" / n).read_text(encoding="utf-8")
    for n in (
        "searchview.js",
        "chat.js",
        "analysis.js",
        "deepsearch.js",
        "pathpicker.js",
        "autoscan.js",
        "state.js",
        "core.js",
        "find.js",
    )
}
ALL_JS = "\n".join(JS.values())
OVERLAY = HTML[HTML.index('id="svOverlay"') : HTML.index("<!-- Hit map (treemap")]


def _defined(name: str) -> bool:
    return bool(re.search(rf"\b(async\s+)?function\s+{re.escape(name)}\s*\(", ALL_JS))


def test_the_workspace_shell():
    assert 'class="map-panel vx-panel sv-panel"' in OVERLAY and 'class="vx-head"' in OVERLAY
    for tab in ("text", "files", "map", "chat"):
        assert f"svTab('{tab}')" in OVERLAY, tab
    assert ">Ask<" in OVERLAY.replace("</svg>", ">")
    for i in (
        "svQueryBar",
        "svModeSeg",
        "svFilterBar",
        "svAdvBar",
        "svViewBtn",
        "svResults",
        "svRead",
        "chatThread",
        "chatInput",
        "chatSources",
        "chatList",
    ):
        assert f'id="{i}"' in OVERLAY, i


def test_the_map_is_a_view_of_the_text_and_files_search_names():
    """The files tab had words/meaning modes and the map had a tab of its
    own, which did not fit. The map draws the word hits of the text
    search, so it is a view of the text (Hits · By recording · Map), not a
    tab of its own; the files tab searches names — no words/meaning, no Ask,
    no deep search, no second name field."""
    nav = OVERLAY[OVERLAY.index('class="vx-tabs"') : OVERLAY.index("</nav>")]
    assert "svTabBtnMap" not in nav and "svTab('map')" not in nav
    views = OVERLAY[OVERLAY.index('id="svViewBtn"') : OVERLAY.index('id="svAdvBar"')]
    assert 'data-view="map"' in views and "svTab('map')" in views
    chrome = JS["searchview.js"][JS["searchview.js"].index("function svTabChrome(") :]
    chrome = chrome[: chrome.index("\n}\n")]
    for hidden in (
        "['#svModeSeg', media || files || map]",
        "['#svQueryBar .sv-ask-btn', media || files]",
        "['#svDeepBtn', media || files]",
        "['#svFileFilter', files]",
        "['#svCountHits', files]",
    ):
        assert hidden in chrome, hidden
    assert "const lit = name === 'map' ? 'text' : name;" in JS["searchview.js"]
    assert 'id="svCountHits"' in OVERLAY


def test_the_analyses_tab():
    """One task over many recordings: its own tab, the form and the
    view in the middle, the passages on the right, the folder field the file
    list's path field (place 'an'), and 🧠 in the file list's selection bar."""
    assert "svTab('analysis')" in OVERLAY and ">Analyses <" in OVERLAY.replace("</svg>", ">")
    for i in (
        "svPaneAnalysis",
        "anList",
        "anForm",
        "anView",
        "anSources",
        "anSourceSeg",
        "anFolder",
        "anPathPicker",
        "anPlan",
        "anPresets",
        "anTask",
        "anOrder",
        "anSpeakers",
        "anMerge",
        "anModel",
        "anStartBtn",
    ):
        assert f'id="{i}"' in OVERLAY, i
    assert "'analysis'" in JS["searchview.js"] and "anOpen()" in JS["searchview.js"]
    assert "an:   {input: 'anFolder'" in JS["pathpicker.js"]
    bar = HTML[HTML.index('id="selectionBar"') : HTML.index('id="batchBar"')]
    assert 'onclick="analyzeSelection()"' in bar
    assert '<script src="/static/js/analysis.js' in HTML


def test_the_deep_search_tab():
    """Find everything, labelled "Deep search" since the Find tab
    exists: the button next to the question starts it, its own tab keeps
    the deep searches (list · the one chosen · its plan), and the LLM
    queue's row leads there."""
    bar = OVERLAY[OVERLAY.index('id="svQueryBar"') : OVERLAY.index('id="svFilterBar"')]
    assert 'onclick="dsStartFromQuery()"' in bar and "Deep search" in bar
    assert "svTab('deep')" in OVERLAY and ">Deep search <" in OVERLAY.replace("</svg>", ">")
    for i in (
        "svTabNDeep",
        "svPaneDeep",
        "dsList",
        "dsView",
        "dsForm",
        "dsQuery",
        "dsStartBtn",
        "dsPlan",
    ):
        assert f'id="{i}"' in OVERLAY, i
    assert '<script src="/static/js/deepsearch.js' in HTML
    assert "'deep'" in JS["searchview.js"] and "dsOpen()" in JS["searchview.js"]
    assert "dsLeave()" in JS["searchview.js"]
    lq = (STATIC / "js" / "llmtools.js").read_text(encoding="utf-8")
    assert "deep_search: 'Deep search'" in lq and "openDeepSearch(" in lq
    assert _defined("openDeepSearch")


def test_every_control_calls_a_function_that_exists():
    calls = set(
        re.findall(
            r'on(?:click|change|keydown)="(?:event\.[a-zA-Z]+\(\);)?\s*([A-Za-z_]\w*)\(', OVERLAY
        )
    )
    missing = sorted(c for c in calls if not _defined(c) and c not in ("if",))
    assert not missing, missing


def test_the_old_german_names_are_gone():
    css = (STATIC / "css" / "searchview.css").read_text(encoding="utf-8")
    for old in ("sv-tabinhalt", "sv-angeheftet"):
        assert old not in OVERLAY and old not in css and old not in JS["searchview.js"], old


def test_the_rail_opens_ask_and_the_folder_bar_switches_scanning():
    shell = (STATIC / "js" / "shell.js").read_text(encoding="utf-8")
    assert "Shell.go('ask')" in HTML and "openAsk()" in shell and _defined("openAsk")
    assert 'id="btnAutoscan"' in HTML and _defined("autoscanMenu")
    assert (
        '<script src="/static/js/chat.js' in HTML and '<script src="/static/js/autoscan.js' in HTML
    )


def test_references_become_chips_that_play():
    chat = JS["chat.js"]
    assert "chatRefPlay" in chat and "svPlay(" in chat and "svGoto(" in chat
    assert r"\[((?:S\d+)" in chat


def test_the_settings_its_tabs_link_to_open_above_it():
    """Ask's footer, Same audio and Images & sounds link to the settings —
    they opened behind the workspace and nothing seemed to happen. Now
    they lie above it (and above the statistics and load views), and
    Escape closes the settings, not the search under them."""
    css = STATIC / "css"

    def layer(f, sel):
        return int(
            re.search(
                re.escape(sel) + r"\s*\{[^}]*?z-index:\s*(\d+)",
                (css / f).read_text(encoding="utf-8"),
            ).group(1)
        )

    settings = layer("settings.css", ".settings-overlay")
    assert settings > layer("searchview.css", ".sv-overlay")
    assert settings > layer("statsview.css", ".stv-overlay")
    for f, link in (
        ("chat.js", "openSettings('search')"),
        ("audiomatch.js", "openSettings('audio')"),
        ("mediasearch.js", "openSettings('media')"),
    ):
        assert link in (STATIC / "js" / f).read_text(encoding="utf-8"), f
    sv = JS["searchview.js"]
    # Noted before the settings' own handler closes them
    assert "if (e.key === 'Escape') onTop = !above();\n    }, true);" in sv
    assert "if (e.key === 'Escape') { if (onTop) closeSearchView(); return; }" in sv
    assert "if (above() || svState.tab !== 'text') return;" in sv


def test_the_reading_pane_keeps_the_searchs_marks():
    """The hit showed "ich" marked as a word, and once its context came,
    re-marked by the plain terms, "ich" lit up inside "mich" too."""
    sv = JS["searchview.js"]
    assert ": s.text === r.text ? _hlHit(r, svState.terms)" in sv


def test_the_direction_shows_only_where_it_turns_the_order():
    """ "File & time" ignores the direction: it said "descending" over a
    list in ascending order."""
    sv = JS["searchview.js"]
    body = sv[sv.index("function svSortSync()") :]
    body = body[: body.index("\n}\n")]
    assert "d.style.display = document.getElementById('svSort')?.value ? '' : 'none';" in body
    assert "document.getElementById('svSort')?.addEventListener('change', svSortSync);" in sv
    for caller in ("function svAdvReset()", "function svToggleFileMode()"):
        fn = sv[sv.index(caller) :]
        assert "svSortSync();" in fn[: fn.index("\n}\n")], caller
