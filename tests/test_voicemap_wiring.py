"""No button without a function — the wiring of the voice map.

This project once had **dead wiring** in exactly this view (Commit `c0e14c0`,
"Voice map: fixed dead wiring of group view"): An `onclick` was in the markup,
but the function had a different name. The button looked like a button and did
nothing, and no test caught it.

Therefore we check both:

1. Every handler in the map markup has a function that actually exists.
2. Every element that JavaScript searches for via ``getElementById`` is also
   in the markup.

Both are pure text reading — no browser needed. This doesn't replace visual
inspection (three map errors were only visible in screenshots), but it catches
the kind of error that arises from renaming.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
HTML = (REPO / "whisper_tm" / "static" / "index.html").read_text(encoding="utf-8")
# `voicemap_assign.js` was once missing here — and that's where a
# `getElementById('mapAudio')` was reaching for nothing.
JS_FILES = [
    "voicemap.js",
    "voicemap_graph.js",
    "voicemap_panel.js",
    "voicemap_tiles.js",
    "voicemap_assign.js",
    "core.js",
    "app.js",
    "speaker.js",
]
JS = "\n".join(
    (REPO / "whisper_tm" / "static" / "js" / n).read_text(encoding="utf-8") for n in JS_FILES
)


# Limited to the map by NAME, not by a section of the page: An excerpt
# ("from mapOverlay to the next <script") on the first attempt spanned
# across settings and search view and tested things this test isn't responsible for.
CALLS = sorted(
    {
        m.group(2)
        for m in re.finditer(
            r'\bon(click|input|change)="((?:map|graph)[A-Za-z_$][\w$]*)\s*\(', HTML
        )
    }
)

# document.getElementById('mapDrag') in JavaScript
WANTED_IDS = sorted(
    {m.group(1) for m in re.finditer(r"getElementById\(['\"](map[\w]*)['\"]\)", JS)}
)


def fn_body(js: str, header: str) -> str:
    """The body of ONE function — up to the closing brace at line start.

    Previously the excerpt ended at what happened to be next. When a function
    disappeared, two tests broke with "substring not found" — they
    weren't checking anything wrong, they just couldn't find their boundary
    anymore. A boundary that depends on a neighbor isn't one.
    """
    i = js.index(header)
    return js[i : js.index("\n}\n", i)]


def test_there_is_something_to_check_at_all():
    """An empty regex match would make the tests below pointless."""
    assert len(CALLS) >= 8, CALLS
    assert len(WANTED_IDS) >= 8, WANTED_IDS


@pytest.mark.parametrize("the_function", CALLS)
def test_every_handler_has_a_function(the_function):
    assert re.search(rf"function {re.escape(the_function)}\s*\(", JS), (
        f"{the_function}() is called in the map markup but not defined in "
        f"{', '.join(JS_FILES)} — dead button"
    )


@pytest.mark.parametrize("label", WANTED_IDS)
def test_every_id_looked_up_is_actually_created(label):
    """Either it's in the markup — or JavaScript builds it itself.

    The audio, name field, and context only appear on clicking a point;
    requiring them in static markup would be wrong. We only require that they
    exist SOMEWHERE.
    """
    created = (
        f'id="{label}"' in HTML
        or f'id="{label}"' in JS
        or f"id = '{label}'" in JS
        or f'"{label}"' in JS
        and "createElement" in JS
        and f"ton.id = '{label}'" in JS
    )
    assert created, (
        f"getElementById('{label}') finds nothing — neither in markup nor created by JavaScript"
    )


def test_the_two_new_sliders_are_complete():
    """Drag and duration: markup, function, effect, and memory.

    Both the pull strength and the simulation duration are adjustable in
    the GUI. A slider that only changes the display would be a dummy.
    """
    assert 'id="mapDrag"' in HTML and 'id="mapSteps"' in HTML
    # … affects the forces …
    assert "GRAPH.drag" in JS and "GRAPH.steps" in JS
    assert re.search(r"0\.018 \* GRAPH\.drag", JS), "Drag doesn't affect the springs"
    assert re.search(r"0\.012 \* GRAPH\.drag", JS), "Drag doesn't affect group attraction"
    assert "GRAPH.ticks / GRAPH.steps" in JS, "duration doesn't affect the cooling schedule"
    # … and survives a reload.
    assert "localStorage.setItem(GRAPH_MEMO" in JS


def test_the_threshold_slider_shows_the_cores_default():
    """One preset, two places — that's bound to go wrong.

    The slider carries its `value` in the markup, but calculation uses
    ``GROUPING_THRESHOLD`` from the core. When it was changed
    (0.65 → 0.71), the slider would still show 0.65 while
    the map below grouped by 0.71.

    We check both: that the numbers in markup match the core — and that the
    view actually adopts the value from the server response. The second is the
    real safeguard; the first just makes the first render before the response
    honest.
    """
    from whisper_tm.core import speaker as speaker_core

    expected = f"{speaker_core.THRESHOLD_GROUPING:.2f}"
    assert re.search(rf'id="mapThreshold"[^>]*value="{expected}"', HTML, re.S), (
        f"the slider is not set to {expected}"
    )
    assert re.search(rf'id="mapThresholdValue">{expected}<', HTML), (
        f"the number next to the slider is not set to {expected}"
    )
    assert "graphShowThreshold(d.threshold)" in JS, (
        "the view doesn't adopt the calculated threshold from the server"
    )


def test_the_default_is_the_same_everywhere():
    """Several places, one number — and two of them once disagreed.

    ``config.default.yaml`` still had 0.65 while the core was already at 0.71:
    Anyone deriving their config from the included file silently got a
    DIFFERENT map than those without one. No one notices because both values
    seem plausible on their own.

    So we check them all: core, included config and the slider in markup.
    The settings page has no number of its own any more: it shows the saved
    value, and its way back to the default comes from config.default.yaml.
    """
    from whisper_tm.core import speaker as speaker_core

    expected = speaker_core.THRESHOLD_GROUPING
    yaml = (REPO / "config.default.yaml").read_text(encoding="utf-8")
    assert re.search(rf"^\s+threshold_grouping:\s*{expected}\s*$", yaml, re.M), (
        f"config.default.yaml is not set to {expected}"
    )
    pages = (REPO / "whisper_tm" / "static" / "js" / "settings_pages.js").read_text(
        encoding="utf-8"
    )
    assert "key: 'speaker.threshold_grouping'" in pages, "missing on the settings page"
    assert "threshold_grouping') ??" not in pages, (
        "the settings page carries a fallback number of its own again"
    )
    assert re.search(rf'id="mapThreshold"[^>]*value="{expected:.2f}"', HTML, re.S)


def test_the_slider_reaches_beyond_the_default():
    """A slider whose preset is stuck at the stop can only move one way. At
    0.94 with an end at 0.95, exactly one step up would be possible — and the
    interesting question ("will it be cleaner at 0.97?") would remain
    unanswered."""
    from whisper_tm.core import speaker as speaker_core

    m = re.search(r'id="mapThreshold"[^>]*max="([\d.]+)"', HTML, re.S)
    assert m, "the slider has no end"
    assert float(m.group(1)) >= speaker_core.THRESHOLD_GROUPING + 0.04, (
        f"the slider ends at {m.group(1)}, preset is at "
        f"{speaker_core.THRESHOLD_GROUPING} — not enough room above"
    )


def test_position_in_recording_is_fully_wired():
    """A button that shows the voice's location in the recording.

    The map can play a voice, but not explain it: 30 seconds from minute 500
    don't say what it was about. The context is in the transcript — but there
    was no way there until now.

    We check the whole chain because each link on its own looks right: button
    in markup, function, and that it uses the TESTED path instead of building
    a second one alongside.
    """
    panel = (REPO / "whisper_tm" / "static" / "js" / "voicemap_panel.js").read_text(
        encoding="utf-8"
    )
    assert 'id="mapShowSpot"' in panel
    assert "function mapShowSpot" in panel
    # The jump uses the existing search navigation.
    assert "goToSearchResult(" in panel, "own path instead of the tested one"
    # And the audio must be off first: otherwise sample and video run at the same time.
    body = panel[panel.index("function mapShowSpot") :]
    assert "panelAudioOff()" in body
    assert body.index("panelAudioOff()") < body.index("goToSearchResult(")


def test_the_jump_passes_the_map_root_on():
    """The empty third parameter inherits `lastSearchMediaDir` — the root of
    the last SEARCH, a different state than the map's. That's exactly why the
    button once failed."""
    panel = (REPO / "whisper_tm" / "static" / "js" / "voicemap_panel.js").read_text(
        encoding="utf-8"
    )
    body = panel[panel.index("function mapShowSpot") :]
    assert "GRAPH.root" in body, "the root is not passed on"
    assert not re.search(r"goToSearchResult\([^)]*,\s*''\s*\)", body)


def test_the_assignment_path_is_fully_wired():
    """A flow with four buttons — and each one must work.

    We check the whole chain because it spans three files (markup, own script,
    endpoints) and each piece looks right on its own.
    """
    h = HTML
    js = (REPO / "whisper_tm" / "static" / "js" / "voicemap_assign.js").read_text(encoding="utf-8")
    api = (REPO / "whisper_tm" / "api" / "speaker.py").read_text(encoding="utf-8")

    assert "voicemap_assign.js" in h, "the script is not included"
    assert 'id="mapAssign"' in h and 'id="mapAssignBody"' in h
    # Its own tab; opening the tab opens the flow.
    assert "vxTab('assign')" in h
    voices = (REPO / "whisper_tm" / "static" / "js" / "voices.js").read_text(encoding="utf-8")
    assert "assignOpen()" in voices

    for f in (
        "assignOpen",
        "assignClose",
        "assignFetch",
        "assignDecide",
        "assignOnlyOnePlays",
        "assignDraw",
    ):
        assert f"function {f}" in js, f"{f} is missing"

    # The three decisions and their counterparts on the server.
    for kind in ("'name'", "'trash'", "'skip'"):
        assert kind in js
    assert "/api/speaker/assignment/suggestion" in js
    assert "/api/speaker/assignment/apply" in js
    assert "assignment/suggestion" in api and "assignment/apply" in api

    # After a decision the map is out of date — the data changed, not just
    # the picture. It is marked stale and fetched again when it is shown,
    # not behind the Assign page after every click (tens of MB on a large
    # map).
    body = fn_body(js, "async function assignDecide")
    assert "vxMapChanged()" in body
    assert "mapLoadGraph()" not in body
