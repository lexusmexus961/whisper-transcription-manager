"""Mouse buttons on the clip editor's timeline.

A middle-button click on the clip editor's timeline moves the playback
position there, and a hint says so.

The timeline was tied to a single gesture: left-click sets the closer of the
two marks. But when trimming a 90-minute file, you want both — listen to where
you are, then cut. The middle button is free for this; the right button belongs
to the context menu, where clips are saved.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
HARNESS = Path(__file__).resolve().parent / "js" / "clip_harness.js"
STATIC = REPO / "whisper_tm" / "static"
CLIP_JS = STATIC / "js" / "clip.js"
INDEX = STATIC / "index.html"
CSS = STATIC / "css" / "app.css"

node = pytest.mark.skipif(shutil.which("node") is None, reason="node not available")

STRUCTURE = """
const {buildIt} = require(HARNESS);
const h = buildIt();
// 1000 px bar, 1000 s file — 1 px is 1 s
h.run('clipTotalDur = 1000; clipIn = 100; clipOut = 200;');
"""


def js(body: str) -> dict:
    source = (
        f"const HARNESS = {str(HARNESS)!r};\n"
        "const output = (o) => console.log('§§' + JSON.stringify(o));\n"
        f"{STRUCTURE}{body}\n"
    )
    done = subprocess.run(
        ["node", "-e", source],
        capture_output=True,
        text=True,
        timeout=60,
        env={"CLIP_JS": str(CLIP_JS), "PATH": os.environ.get("PATH", "")},
    )
    if done.returncode != 0:
        raise AssertionError(f"node failed:\n{done.stderr}")
    lines = [z for z in done.stdout.splitlines() if z.startswith("§§")]
    assert lines, f"no result:\n{done.stdout}\n{done.stderr}"
    return json.loads(lines[-1][2:])


@node
class TestMiddleButton:
    def test_jumps_to_clicked_position(self):
        d = js("""
        const prevented = h.mouse(1, 640);
        output({timestamp: h.fetch('videoPlayer').currentTime,
                 marker: h.fetch('clipPlayhead').style.left,
                 prevented});
        """)
        assert d["timestamp"] == pytest.approx(640, abs=0.5)
        assert d["marker"] == "64%", "mark must follow immediately"

    def test_leaves_marks_alone(self):
        """The real point: jump without adjusting the cut."""
        d = js("""
        h.mouse(1, 640);
        output({in: h.value('clipIn'), out: h.value('clipOut')});
        """)
        assert (d["in"], d["out"]) == (100, 200)

    def test_suppresses_browser_autoscroll(self):
        """Without preventDefault, Firefox starts its autoscroll mode and
        the bar slips under the cursor."""
        d = js("output({prevented: h.mouse(1, 640)});")
        assert d["prevented"] is True

    def test_without_duration_nothing_happens(self):
        """Before the file loads, every pixel position is meaningless."""
        d = js("""
        h.run('clipTotalDur = 0;');
        h.mouse(1, 640);
        output({timestamp: h.fetch('videoPlayer').currentTime});
        """)
        assert d["timestamp"] == 0


@node
class TestOtherButtons:
    def test_left_still_sets_closer_mark(self):
        """The old gesture must not be sacrificed for the new one."""
        d = js("""
        h.mouse(0, 190);          // closer to OUT (200) than to IN (100)
        output({in: h.value('clipIn'), out: h.value('clipOut'),
                 timestamp: h.fetch('videoPlayer').currentTime});
        """)
        assert d["out"] == pytest.approx(190, abs=0.5)
        assert d["in"] == 100
        assert d["timestamp"] == 0, "left button does not jump"

    def test_right_remains_for_context_menu(self):
        """Through the context menu clips are saved — the right
        button must neither set marks nor jump here."""
        d = js("""
        const prevented = h.mouse(2, 640);
        output({in: h.value('clipIn'), out: h.value('clipOut'),
                 prevented, timestamp: h.fetch('videoPlayer').currentTime});
        """)
        assert (d["in"], d["out"]) == (100, 200)
        assert d["timestamp"] == 0
        assert d["prevented"] is False, "else the page swallows the menu"


class TestHint:
    """The gesture is shown as a hint — a gesture nobody knows about
    is not a feature."""

    def test_hint_is_in_zoom_bar(self):
        html = INDEX.read_text(encoding="utf-8")
        assert "clip-controlhint" in html
        place = html[html.index("clip-controlhint") :][:400]
        assert "Middle button" in place
        assert "playhead position" in place

    def test_hint_is_visually_styled(self):
        css = CSS.read_text(encoding="utf-8")
        assert ".clip-controlhint" in css


class TestAFormula:
    def test_playback_mark_has_only_one_place(self):
        """Interval and jump must use the same formula — two copies are the
        usual source of divergence."""
        source = CLIP_JS.read_text(encoding="utf-8")
        assert source.count("clipPlayhead") == 1, "mark is calculated in more than one place"
        assert source.count("function updateClipPlayhead") == 1


class TestBarStructure:
    """The controls must not float in the middle of the bar.

    The cause was no accident, but two claims on the same gap: the
    selection indicator AND the operation hint both had ``margin-left:auto``,
    and the row had ``justify-content:flex-end``. On a wide screen, the two
    pushed each other apart and left the zoom sliders floating in the middle.
    """

    def _rule(self, name: str) -> str:
        css = CSS.read_text(encoding="utf-8")
        start = css.index(f"\n{name} {{")
        return css[start : css.index("}", start)]

    def test_only_zoom_group_pushes_right(self):
        """Two elements with `margin-left:auto` share the remaining width
        — that's exactly where the gap in the middle comes from."""
        jumper = [
            n
            for n in (".clip-zoom-hint", ".clip-controlhint", ".clip-zoom-group")
            if "margin-left:auto" in self._rule(n)
        ]
        assert jumper == [".clip-zoom-group"], jumper

    def test_bar_has_no_right_aligned_flow(self):
        """`flex-end` on the bar makes any left-align within it ineffective."""
        assert "flex-end" not in self._rule(".clip-zoom-bar")

    def test_controls_sit_together_in_a_group(self):
        html = INDEX.read_text(encoding="utf-8")
        bar = html[html.index('class="clip-zoom-bar"') :]
        bar = bar[: bar.index('class="clip-timeline"')]
        group_nr = bar[bar.index("clip-zoom-group") :]
        for part in ('id="clipZoom"', 'id="clipZoomVal"', "clipZoomToRegion", "clipZoomToPlayhead"):
            assert part in group_nr, f"{part} outside zoom group"

    def test_hint_comes_before_controls(self):
        """Left: state and operation, right: controls — in this
        order, otherwise `margin-left:auto` does nothing."""
        html = INDEX.read_text(encoding="utf-8")
        assert (
            html.index("clip-zoom-hint")
            < html.index("clip-controlhint")
            < html.index("clip-zoom-group")
        )

    def test_zoom_buttons_are_labeled(self):
        """`▶⌖` nobody reads. A button you have to guess is not a button."""
        html = INDEX.read_text(encoding="utf-8")
        group_nr = html[html.index("clip-zoom-group") :][:900]
        assert "Selection</button>" in group_nr
        assert "Position</button>" in group_nr
