"""Workspace wiring found in the UI review: keys and players of a
tab must not act while another tab — or nothing — is on screen, a late
answer must not override a newer one, and a failure must say so."""

from pathlib import Path

JS = Path(__file__).resolve().parent.parent / "whisper_tm" / "static" / "js"


def read(name: str) -> str:
    return (JS / name).read_text("utf-8")


def body(src: str, head: str) -> str:
    i = src.index(head)
    return src[i : src.index("\n}\n", i)]


def test_clip_editor_keys_only_on_the_extract_tab():
    s = read("voices_clipedit.js")
    i = s.index("document.addEventListener('keydown'")
    handler = s[i : s.index("}, true);", i)]
    guard = handler.index("VX.tab !== 'extract'")
    assert "classList.contains('visible')" in handler
    # the guard comes before Esc and Delete are taken
    assert guard < handler.index("'Escape'") < handler.index("'Delete'")


def test_leaving_a_tab_stops_what_it_runs():
    s = read("voices.js")
    leave = body(s, "function vxLeaveTab")
    assert "assignClose()" in leave
    # the clip editor only stops playing — closing it would lose unsaved
    # bounds and text on a mere glance at another tab
    assert "clipEditStop()" in leave and "clipEditClose()" not in leave
    assert "vxLeaveTab(prev)" in body(s, "function vxTab")
    assert "vxLeaveTab(VX.tab)" in body(s, "function vxClose")


def test_a_kept_clip_editor_draws_again_when_its_tab_returns():
    """Found in the browser: leaving the tab stopped playback, the
    stop redrew the waveform while the page was hidden, sized the canvas to
    0x0 — and the kept editor came back with a blank waveform."""
    draw = body(read("voices_clipedit.js"), "function clipEditDraw")
    assert draw.index("if (!w || !h) return;") < draw.index("cv.width =")
    tab = body(read("voices.js"), "function vxTab")
    extract = tab[tab.index("name === 'extract'") :]
    assert "clipEditDraw()" in extract and "CLIPEDIT.i >= 0" in extract


def test_map_keys_only_on_the_map_tab():
    s = read("voicemap_panel.js")
    i = s.index("// ── Keyboard")
    handler = s[i : s.index("});", i)]
    assert "e.defaultPrevented" in handler and "VX.tab !== 'map'" in handler


def test_assign_newest_answer_wins_and_no_hidden_autoplay():
    s = read("voicemap_assign.js")
    f = body(s, "async function assignFetch")
    assert "const mine = ++ASSIGN.fetchSeq;" in f
    assert f.count("mine !== ASSIGN.fetchSeq") == 2
    d = body(s, "function assignDraw")
    assert "assignVisible()" in d.split("assignSeqStart(0)")[0].rsplit("if (", 1)[1]
    # reopening keeps an asked-again question: the memo flag, not `an`
    assert "const first = !ASSIGN.loaded;" in body(s, "function assignOpen")


def test_splitter_relayouts_once_per_frame():
    s = read("voices.js")
    i = s.index("window.addEventListener('mousemove'")
    move = s[i : s.index("window.addEventListener('mouseup'", i)]
    assert "requestAnimationFrame" in move
    # no direct relayout per mousemove: only inside the frame callback
    lines = [z.strip() for z in move.splitlines()]
    assert "vxMapResized();" not in lines


def test_graph_errors_and_failed_restore():
    s = read("voicemap_graph.js")
    load = body(s, "async function mapLoadGraph")
    assert "d.error" in load and "esc(String(e))" not in load
    back = body(s, "async function graphTrashBack")
    assert back.index("if (!d) return;") < back.index("GRAPH.trash = false")


def test_a_decided_question_is_never_shown_again():
    """Apply on a large group, leave the tab before the answer: assignFetch
    asks nothing while off screen, so the decided question must already be
    gone — coming back then asks the server afresh."""
    s = read("voicemap_assign.js")
    f = body(s, "async function assignFetch")
    off = f[: f.index("const mine")]
    assert "if (!ASSIGN.on)" in off and "ASSIGN.suggestion = null" in off
    d = body(s, "async function assignDecide")
    # both the skip and the apply path drop the decided question first
    assert d.count("if (ASSIGN.suggestion === v) ASSIGN.suggestion = null;") == 2
    for part in d.split("if (ASSIGN.suggestion === v) ASSIGN.suggestion = null;")[1:]:
        assert part.lstrip().startswith(("assignFetch();", "// decided"))
