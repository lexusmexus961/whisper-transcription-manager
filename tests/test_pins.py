"""Pins (static/js/pins.js): which tabs can be pinned, the limit of six,
reordering, and that the list is a UI preference."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

STATIC = Path(__file__).resolve().parent.parent / "whisper_tm" / "static"
PINS = (STATIC / "js" / "pins.js").read_text(encoding="utf-8")
HOME = (STATIC / "js" / "home.js").read_text(encoding="utf-8")

_RUN = r"""
const vm = require('vm');
const prefs = {};
const saved = [], toasts = [];
const box = {innerHTML: '', addEventListener() {}, querySelectorAll: () => []};
const sb = {
    console, setTimeout, clearTimeout,
    document: {getElementById: id => id === 'railPins' ? box : null, querySelectorAll: () => [],
               addEventListener() {}},
    Shell: {pref: k => prefs[k], setPref: (k, v) => { prefs[k] = v; saved.push(v.slice()); }, onPrefs() {}, current: 'home'},
    Toast: {info: m => toasts.push(m), success: m => toasts.push(m)},
    esc: s => String(s), escJs: s => String(s),
};
sb.window = sb; sb.globalThis = sb;
vm.createContext(sb);
vm.runInContext(HOME_SRC + '\nglobalThis.Home = Home;', sb);
vm.runInContext(PINS_SRC + '\nglobalThis.Pins = Pins;', sb);
const P = sb.Pins, out = {};
out.defaults = P.list();
out.core = ['library', 'voices', 'find', 'ask', 'settings', 'add', 'find:chat', 'nope:x'].map(P.pinnable);
out.tabs = ['voices:map', 'find:audio', 'machine:load', 'help:setup'].map(P.pinnable);
P.toggle('voices:history'); P.toggle('find:text');
out.six = P.list().length;
P.toggle('find:files');
out.seventh = P.list().includes('find:files');
out.toast = toasts[toasts.length - 1];
P.toggle('library');                      // a core place: ignored
P.move('find:text', 'voices:map');
out.moved = P.list();
P.toggle('voices:map');                   // unpin
out.after = P.list();
prefs.pins = ['voices:map', 'ask', 'bogus'];   // a stored list with junk
out.cleaned = P.list();
out.drawn = box.innerHTML;
out.saved = saved.length;
console.log('§§' + JSON.stringify(out));
"""


@pytest.fixture(scope="module")
def run() -> dict:
    if shutil.which("node") is None:
        pytest.skip("node not available")
    script = _RUN.replace("HOME_SRC", json.dumps(HOME)).replace("PINS_SRC", json.dumps(PINS))
    done = subprocess.run(
        ["node", "-e", script],
        capture_output=True,
        text=True,
        timeout=60,
        env={"PATH": os.environ.get("PATH", "")},
    )
    assert done.returncode == 0, done.stderr
    line = next(z for z in done.stdout.splitlines() if z.startswith("§§"))
    return json.loads(line[2:])


def test_the_default_pins(run):
    assert run["defaults"] == ["voices:map", "voices:extract", "find:audio", "find:conversations"]


def test_only_window_tabs_can_be_pinned(run):
    assert run["core"] == [False] * 8, "core places, actions and unknown names stay out"
    assert run["tabs"] == [True] * 4


def test_at_most_six_with_a_clear_message(run):
    assert run["six"] == 6
    assert run["seventh"] is False
    assert "At most 6" in run["toast"]


def test_reorder_and_unpin(run):
    assert run["moved"][:2] == ["find:text", "voices:map"]
    assert "voices:map" not in run["after"] and len(run["after"]) == 5


def test_a_stored_list_with_junk_is_cleaned(run):
    assert run["cleaned"] == ["voices:map"]


def test_every_change_is_saved_as_a_preference(run):
    # 2 pins + reorder + unpin; the refused 7th and the core place save nothing
    assert run["saved"] == 4
