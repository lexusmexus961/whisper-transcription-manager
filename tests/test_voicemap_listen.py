"""The ▶ button in guided assignment — and why it did nothing silently.

Symptom: the ▶ button played no audio when guided assignment was first opened.

The server was impeccable: all eight samples of the first suggestion came as
``HTTP 200``, ``audio/mpeg``, a few hundred KB each, within seconds. The path
ended in the browser, at two places simultaneously:

1. The old listen handler fetched ``mapAudio`` via ``getElementById`` — an
   element that only exists when the panel framework is built, and that only
   happens on clicking a map point. There is none in assignment. The early
   ``if (!audioEl) return`` silenced the case.
2. ``voicemap_assign.js`` hung as the only script WITHOUT ``?v={{ v }}`` in the
   page. Even a correct fix would not have reached the browser.

Both errors are invisible — no error in the console, no red test, just a button
that does nothing. These checks are exactly against that.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
STATIC = REPO / "whisper_tm" / "static"
HTML = (STATIC / "index.html").read_text(encoding="utf-8")
ASSIGN = (STATIC / "js" / "voicemap_assign.js").read_text(encoding="utf-8")
PANEL = (STATIC / "js" / "voicemap_panel.js").read_text(encoding="utf-8")
# The sample players of the Assign page, the History and the People list are
# built in ONE place.
VOICES = (STATIC / "js" / "voices.js").read_text(encoding="utf-8")
CSS = (STATIC / "css" / "app.css").read_text(encoding="utf-8")


# ── Cache buster ─────────────────────────────────────────────────────

MEAN = sorted(set(re.findall(r'(?:src|href)="(/static/[^"]+\.(?:js|css)[^"]*)"', HTML)))


def fn_body(js: str, header: str) -> str:
    """The body of ONE function — up to the closing bracket at line start.

    Previously the excerpt ended at what happened to be next. When a function
    disappeared, two tests broke with "substring not found" — they
    checked nothing wrong, they just couldn't find their boundary anymore.
    A boundary that depends on the neighbor is not one.
    """
    i = js.index(header)
    return js[i : js.index("\n}\n", i)]


def test_there_are_scripts_to_check_at_all():
    assert len(MEAN) >= 10, MEAN


@pytest.mark.parametrize("path", MEAN)
def test_every_own_file_carries_the_cache_buster(path):
    """Without ``?v=`` the browser serves the old version for days.

    This is the nasty kind of bug: the fix is in the repo, it's
    tested, it's deployed — and the browser doesn't show it.
    """
    assert "?v={{ v }}" in path, f"{path} is loaded without a cache buster"


# ── The path to the audio ────────────────────────────────────────────────


def test_the_player_appears_even_without_a_selection():
    """The core of the bug: no framework, no audio, no error.

    ``panelAudio`` must build the framework itself when needed. Without that, the
    entire guided-assignment path is silent as long as nobody clicked a point
    beforehand — and in guided assignment, nobody clicks a point.
    """
    body = fn_body(PANEL, "function panelAudio(")
    assert "panelFrame()" in body, (
        "panelAudio does not build the framework — without a prior selection there is no <audio>"
    )


def test_the_recursion_terminates():
    """``panelFrame`` calls ``panelAudio``, ``panelAudio`` calls ``panelFrame``.

    This only works because the build flag is set BEFORE the call.
    If someone reverses that, the page hangs on the first click.
    """
    g = PANEL[PANEL.index("function panelFrame") :]
    g = g[: g.index("panelWire();")]
    assert g.index("dataset.built = '1'") < g.index("panelAudio();"), (
        "the build flag is only set after panelAudio() — infinite loop"
    )


# ── The listening clips in guided assignment ─────────────────────────────
#
# Rebuilt with the browser's default player. Before, each sample was a ▶ button that switched the map's
# ONE player — nothing could be saved from it.


def test_every_sample_has_its_own_player():
    """With `controls`, because that was exactly the point: right-click → save."""
    body = fn_body(VOICES, "function vxSamplesHtml(")
    assert "<audio controls" in body
    assert 'src="/api/speaker/sample/${Number(p.vector_id)}"' in body
    assert "vxSamplesHtml(d.samples)" in ASSIGN


def test_the_samples_are_not_preloaded():
    """Eight clips at ~250 kB per suggestion would be 2 MB of audio that's
    mostly never listened to."""
    assert 'preload="none"' in fn_body(VOICES, "function vxSamplesHtml(")


def test_only_one_plays_at_a_time():
    """The new problem the single player didn't have: eight elements
    can play simultaneously. When comparing two voices, that's
    especially useless — you're listening precisely to tell whether it's the same one."""
    body = fn_body(ASSIGN, "function assignOnlyOnePlays")
    assert "a.pause()" in body and "querySelectorAll" in body
    # Also the row in the graph view, otherwise it advances unasked.
    assert "graphRowStop" in body
    # `play` doesn't bubble — the listener MUST hang in the capture phase.
    listener = ASSIGN[ASSIGN.index("document.addEventListener('play'") :]
    listener = listener[: listener.index("}, true);") + 9]
    assert listener.endswith("}, true);"), listener[-40:]


def test_a_decision_ends_the_sample():
    """Otherwise the old voice talks into the next suggestion — and
    it might get named a second time by mistake."""
    body = fn_body(ASSIGN, "async function assignDecide")
    assert "assignOnlyOnePlays(null)" in body


# ── The buttons in the segment row ───────────────────────────────────────

FILEVIEW = (STATIC / "js" / "fileview.js").read_text(encoding="utf-8")


def test_the_segment_buttons_have_their_own_column():
    """Symptom: the edit and speaker-map buttons in a segment row could not
    be clicked.

    It wasn't the buttons themselves. They were stuck in the time column, which is 110 px
    wide and `nowrap`; four buttons don't fit in there. They ran
    over its edge, and the text field next to it comes LATER in the tree — so it
    sat on top and caught the clicks, which then arrived as the row's
    `seekTo`. The buttons were still visible though: the most annoying case.

    A dedicated grid column can no longer overlap, no matter how many
    buttons get added.
    """
    line = FILEVIEW[FILEVIEW.index('<div class="seg ${BL[si]}"') :]
    line = line[: line.index("`;")]
    ts = line.index('<div class="ts">')
    btns = line.index('<span class="seg-clip-btns">')
    txt = line.index('class="txt"')
    assert btns > txt, "the buttons are stuck in front of the text again"
    assert line[ts:btns].count("<button") == 0, "a button is stuck in the time column again"
    assert "grid-template-columns:110px 1fr auto;" in CSS, (
        "the third column is missing — the buttons overflow again"
    )


def test_the_buttons_are_large_enough_to_hit():
    """4 px of inner padding gives an area you have to hunt for with the
    mouse. What's required is one you hit without aiming."""
    block = CSS[CSS.index(".seg-clip-btns button {") :]
    block = block[: block.index("}")]
    assert "min-width:26px" in block and "height:24px" in block, block


def test_the_buttons_are_there_without_a_mouse_too():
    """Previously `display:none` until hovered. Anyone who doesn't know
    they exist never finds them that way — and a button that only appears
    on aiming moves the target as it appears."""
    block = CSS[CSS.index(".seg-clip-btns {") :]
    block = block[: block.index("}")]
    assert "display:flex" in block, block
    assert "opacity:0.32" in block, "the buttons are invisible instead of subtle"
    assert ".seg:hover .seg-clip-btns { opacity:1; }" in CSS
