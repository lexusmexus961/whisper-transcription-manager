"""The label above the highlighted voice — what it says and when.

Jumping from a transcript into the voice map ("🗣 Map" next to `SPEAKER_06`)
used to show:

    "SPEAKER_06 · no stored voice windows here"

It was not wrong, but it answered the wrong question. It sounded like *this one
voice is missing*, when in reality there was not a single window for the entire
recording (e.g. `interview.mp4`): it had been transcribed before speaker
recognition existed. In a library that predates the feature, most transcripts
may have no voice data at all — so the case is the rule, not the exception, and
deserves its own sentence.

The line below ("0 of N voices · 0 recordings") said it already, but it
stands elsewhere and reads like a filter result.

So the text itself is checked — three layers, three statements — and the wiring:
that the map counts the second number at all. Without it the function cannot
tell the layers apart, no matter how good its text is.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
HARNESS = Path(__file__).resolve().parent / "js" / "voicemap_harness.js"
MAP_JS = REPO / "whisper_tm" / "static" / "js" / "voicemap_graph.js"
PANEL_JS = MAP_JS.parent / "voicemap_panel.js"
SOURCE = MAP_JS.read_text(encoding="utf-8")

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node not available")


def js(body: str) -> dict:
    source = (
        f"const {{API, paper}} = require({str(HARNESS)!r});\n"
        "const output = (o) => console.log('§§' + JSON.stringify(o));\n"
        f"{body}\n"
    )
    done = subprocess.run(
        ["node", "-e", source],
        capture_output=True,
        text=True,
        timeout=60,
        env={"MAP_JS": str(MAP_JS), "PANEL_JS": str(PANEL_JS), "PATH": os.environ.get("PATH", "")},
    )
    if done.returncode != 0:
        raise AssertionError(f"node failed:\n{done.stderr}")
    lines = [z for z in done.stdout.splitlines() if z.startswith("§§")]
    assert lines, f"no result:\n{done.stdout}\n{done.stderr}"
    return json.loads(lines[-1][2:])


def texts() -> dict:
    """All three layers in one node call — the startup costs more than they do."""
    return js("""
        output({
            found: API.mapHighlightText('SPEAKER_06', 12, 40),
            empty_but_present: API.mapHighlightText('SPEAKER_06', 0, 40),
            nothing_at_all: API.mapHighlightText('SPEAKER_06', 0, 0),
        });
    """)


def test_a_found_voice_names_the_number():
    assert "12 windows" in texts()["found"]


def test_the_three_situations_say_three_different_sentences():
    """The actual bug: two layers, one sentence.

    As long as "this voice has nothing here" and "this recording has nothing
    at all" get the same text, the reader looks for the bug in the wrong
    place — at the speaker instead of at the missing follow-up run.
    """
    t = texts()
    assert len({t["found"], t["empty_but_present"], t["nothing_at_all"]}) == 3, t


def test_a_voice_without_a_window_points_at_the_others():
    """The recording IS evaluated — then the neighbor count is the answer."""
    t = texts()["empty_but_present"]
    assert "40" in t, t
    assert "at all" not in t, t


def test_an_unanalyzed_recording_says_so_about_the_RECORDING():
    """The reported case. The sentence must be about the recording, not
    about the identifier — otherwise it misleads again."""
    t = texts()["nothing_at_all"]
    assert "recording" in t, t
    assert "windows highlighted" not in t, t


def test_the_label_is_included_in_every_situation():
    """The label stays in place while zooming; without the identifier you
    no longer know what it's talking about later."""
    for position, t in texts().items():
        assert "SPEAKER_06" in t, (position, t)


# ── Wiring: the second number must also be collected ───────────────────────


def _body(src: str, header: str) -> str:
    i = src.index(header)
    return src[i : src.index("\n}\n", i)]


def test_locating_counts_the_recordings_voices():
    """A text that CAN distinguish three layers is useless if the caller
    only supplies two numbers. That's exactly how it was before:
    `mapHighlightChip(label, hits.length)` — the recording never appeared.
    Now the numbers are the recording's windows, found by id."""
    body = _body(SOURCE, "function mapLocateApply(")
    assert "mapHighlightText(L.label, marks.length, rows.length)" in body


# ── The jump from the transcript lands in the tiles ───────────────────────


def test_locating_switches_to_tiles():
    """Tiles, not the simulation — for every path from the file view.

    Previously the opposite was here (`mapView('graph')`), because
    the tiles could not draw the highlight. They can now (`tilesDraw`).
    """
    body = _body(SOURCE, "async function mapLocate(")
    assert "mapView('tiles')" in body, body[-500:]
    assert "mapView('graph')" not in body, "still switches to the simulation"


def test_in_the_tiles_it_does_not_jump_to_the_point():
    """`graphJumpTo` moves the canvas to a location in the force
    simulation. That location does not exist in the tiles — the located
    voice is chosen in the details instead."""
    body = _body(SOURCE, "function mapLocateApply(")
    assert "graphJumpTo" not in body
    assert "mapPickVoice(first)" in body


def test_the_label_explains_the_light_boxes():
    """Whoever doesn't know what "bright" means will look for the bug in the
    jump — exactly what happened before this label existed."""
    from pathlib import Path

    graph = (
        Path(__file__).resolve().parent.parent
        / "whisper_tm"
        / "static"
        / "js"
        / "voicemap_graph.js"
    ).read_text(encoding="utf-8")
    body = graph[graph.index("function mapSpotText") :]
    body = body[: body.index("\n}")]
    assert "bright" in body and "recording" in body, body


# ── The highlight must not disappear on its own ─────────────────────────

GRAPH_JS = (
    Path(__file__).resolve().parent.parent / "whisper_tm" / "static" / "js" / "voicemap_graph.js"
).read_text(encoding="utf-8")


def test_the_bridges_wait_for_the_finished_load_run():
    """Symptom: the crosshair disappeared after a short time if something
    had already been selected before.

    Reproduced in the browser and traced back with a trap on ``MAP.highlight``:
    the marker stood for two seconds and was then deleted by
    ``mapLoadGraph`` ("highlight belongs to the old graph" — correct
    reasoning, only here it hit the new one instead).

    The cause is the guard condition. ``MAP.n`` and ``GRAPH.on`` do NOT
    answer whether a load is currently in progress: both are still true from
    the previous run. All bridges now go through ONE wait
    (`mapWhenLoaded`), and it asks the only thing that answers it.
    """
    wait = _body(GRAPH_JS, "function mapWhenLoaded(")
    assert "GRAPH.doneSeq === GRAPH.seq" in wait
    assert "MAP.n" not in wait and "GRAPH.on" not in wait
    assert "await mapWhenLoaded();" in _body(GRAPH_JS, "async function mapLocate(")
    for f in ("mapForFile", "mapForSpeaker", "mapForSpot"):
        assert "mapLocate(" in _body(GRAPH_JS, f"function {f}(")


def test_every_exit_of_the_load_run_reports_done():
    """Otherwise the bridge waits forever — precisely when something has
    already gone wrong anyway (network down, no voices)."""
    body = GRAPH_JS[GRAPH_JS.index("async function mapLoadGraph") :]
    body = body[: body.index("\n/* Initial layout")]
    assert body.count("done();") >= 3, "one exit does not mark the run as finished"
    # And an outdated run must NOT be allowed to count as finished.
    assert "if (mine === GRAPH.seq) GRAPH.doneSeq = mine;" in body
