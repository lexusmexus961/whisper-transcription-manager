"""The clip editor's waveform, in the browser's half (static/js/clip.js).

The waveform in the clip editor makes cutting easier — but it must not
block on a 24 h recording. The server never waits (tests/test_waveform.py); this side pins:

* it asks for the visible window only — the bar's own calculation — in as
  many buckets as the timeline has pixels;
* the canvas sits in the bar at the stretch it shows;
* moving a marker asks for nothing, zooming asks for the new window;
* a ``pending`` answer is drawn as it is and asked again a second later;
* an answer for a window that is no longer shown is dropped;
* a recording without sound leaves the bar as it was, and isn't asked again
  for every marker moved.
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
CLIP_JS = REPO / "whisper_tm" / "static" / "js" / "clip.js"

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node not available")

# A 1000 px timeline over a file of 1000 s; every answer as the test sets it
STRUCTURE = """
const {buildIt} = require(HARNESS);
const h = buildIt();
const S = h.sandbox;
const timers = [];
S.setTimeout = (fn, ms) => { timers.push({fn, ms}); return timers.length; };
S.clearTimeout = (id) => { if (timers[id - 1]) timers[id - 1].fn = null; };
const due = () => timers.filter(t => t.fn).map(t => t.ms);
const tick = async () => {                    // every timer due now, and what it awaits
    const now = timers.filter(t => t.fn);
    now.forEach(t => { const f = t.fn; t.fn = null; f(); });
    for (let i = 0; i < 10; i++) await null;
};
const asked = [];
let answer = (q) => ({start: q.start, end: q.end, n: q.n, peaks: Array(q.n).fill(128),
                      pending: false, progress: 1});
S.fetch = async (url) => {
    const q = Object.fromEntries(new URL('http://x' + url).searchParams);
    for (const k of ['start', 'end', 'n']) q[k] = Number(q[k]);
    asked.push(q);
    const d = answer(q);
    return {ok: !d.error, status: d.error ? 422 : 200, json: async () => d};
};
const drawn = [];
const cv = h.fetch('clipWave');
cv.hidden = true;
cv.getContext = () => ({clearRect() {}, fillRect(x) { drawn.push(x); }, set fillStyle(v) {}});
h.fetch('clipWaveState').hidden = true;
h.fetch('clipStatus');
h.fetch('clipTimeline').clientWidth = 1000;
h.fetch('clipTimeline').clientHeight = 48;
S.window.devicePixelRatio = 1;
S.getComputedStyle = () => ({color: 'rgb(1, 2, 3)'});
S.activeFile = 'talk.mp4';
S.activeTranscript = {total_duration_sec: 1000, segments: []};
h.fetch('videoPlayer').currentTime = 500;
"""


def js(body: str) -> dict:
    source = (
        f"const HARNESS = {str(HARNESS)!r};\n"
        "const output = (o) => console.log('§§' + JSON.stringify(o));\n"
        "(async () => {\n"
        + STRUCTURE
        + body
        + "\n})().catch(e => { console.error(e); process.exit(1); });\n"
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


def test_it_asks_for_the_visible_window_in_pixels():
    d = js("""
        h.run('openClipEditor()'); await tick();
        output({asked, left: cv.style.left, width: cv.style.width, hidden: cv.hidden,
                bars: drawn.length});
    """)
    assert d["asked"] == [{"file": "talk.mp4", "start": 0, "end": 1000, "n": 1000}]
    assert (d["left"], d["width"], d["hidden"], d["bars"]) == ("0%", "100%", False, 1000)


def test_zooming_asks_for_the_new_window_and_places_the_canvas_there():
    d = js("""
        h.run('openClipEditor()'); await tick();
        h.run('clipZoom = 10; clipViewCenter = 0.5; updateClipInner()'); await tick();
        output({last: asked[asked.length - 1], left: cv.style.left, width: cv.style.width,
                inner: h.fetch('clipTimelineInner').style.left});
    """)
    assert (d["last"]["start"], d["last"]["end"]) == (450, 550)
    assert (d["left"], d["width"], d["inner"]) == ("45%", "10%", "-450%")


def test_moving_a_marker_asks_for_nothing():
    d = js("""
        h.run('openClipEditor()'); await tick();
        h.run("nudgeClip('in', 1); setClipPoint('out', 600)"); await tick();
        output({asked: asked.length});
    """)
    assert d["asked"] == 1


def test_a_long_recording_fills_in_as_it_comes():
    d = js("""
        let progress = 0.25;
        answer = (q) => ({start: q.start, end: q.end, n: q.n, pending: progress < 1, progress,
                          peaks: Array(q.n).fill(null).map((_, i) => i < q.n * progress ? 200 : null)});
        h.run('openClipEditor()'); await tick();
        const first = {state: h.fetch('clipWaveState').textContent, bars: drawn.length, again: due()};
        progress = 1; drawn.length = 0;
        await tick();                                   // a second later
        output({first, asked: asked.length, bars: drawn.length,
                state: h.fetch('clipWaveState').hidden, again: due()});
    """)
    assert d["first"] == {"state": "〰 25 %", "bars": 250, "again": [1000]}
    assert d["asked"] == 2 and d["bars"] == 1000 and d["state"] is True and d["again"] == []


def test_an_answer_for_a_window_no_longer_shown_is_dropped():
    d = js("""
        let hold;
        S.fetch = async (url) => { asked.push(url);
            return {ok: true, status: 200, json: () => new Promise(r => { hold = r; })}; };
        h.run('openClipEditor()'); await tick();
        const slow = hold;
        h.run('closeClipEditor()');
        slow({start: 0, end: 1000, n: 1000, peaks: Array(1000).fill(99), pending: false});
        for (let i = 0; i < 10; i++) await null;
        output({hidden: cv.hidden, bars: drawn.length});
    """)
    assert d == {"hidden": True, "bars": 0}


def test_a_recording_without_sound_leaves_the_bar_alone():
    d = js("""
        answer = () => ({error: 'This recording has no sound'});
        h.run('openClipEditor()'); await tick();
        h.run('clipZoom = 4; updateClipInner()'); await tick();
        const once = asked.length;
        h.run('closeClipEditor(); openClipEditor()'); await tick();
        output({hidden: cv.hidden, once, reopened: asked.length});
    """)
    assert d == {"hidden": True, "once": 1, "reopened": 2}


def test_a_file_switch_parks_the_points_and_the_way_back_restores_them():
    """Switching files closed the clip editor and lost its
    in/out points. Now they are parked for their file — never carried over
    to another — and the editor opens again with them on the way back."""
    d = js("""
        h.run('openClipEditor()'); await tick();
        h.run("setClipPoint('in', 120); setClipPoint('out', 180)");
        h.run('clipParkForSwitch()');                  // selectFile: talk.mp4 → other.mp4
        const closed = !h.fetch('clipEditor').classList.contains('visible');
        S.activeFile = 'other.mp4';
        h.run("clipRestoreFor('other.mp4')");          // nothing parked for it
        const otherOpen = h.fetch('clipEditor').classList.contains('visible');
        S.activeFile = 'talk.mp4';
        h.run("clipRestoreFor('talk.mp4')");           // its transcript is back
        output({closed, otherOpen,
                open: h.fetch('clipEditor').classList.contains('visible'),
                points: h.run('[clipIn, clipOut, clipFile]')});
    """)
    assert d["closed"] is True and d["otherOpen"] is False
    assert d["open"] is True
    assert d["points"] == [120, 180, "talk.mp4"]
