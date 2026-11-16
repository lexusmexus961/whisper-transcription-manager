"""Home (static/js/home.js): when it shows, its live cards and its groups.

Drawn in node against a stand-in page: the queue's globals as batch.js
keeps them, Shell's preferences, and fetch answers for what finished last.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

STATIC = Path(__file__).resolve().parent.parent / "whisper_tm" / "static"
HOME = (STATIC / "js" / "home.js").read_text(encoding="utf-8")
HTML = (STATIC / "index.html").read_text(encoding="utf-8")

_RUN = r"""
const fs = require('fs'), vm = require('vm');
const els = {};
const el = id => els[id] || (els[id] = {id, hidden: true, innerHTML: '', textContent: '', title: '',
                                         scrollTop: 0, value: '', classList: {contains: () => false}});
const docL = {};
const prefs = {pins: ['voices:map', 'nonsense'], recent: [{path: '/lib/a/x.mp3', at: Date.now() / 1000 - 120}],
               home_groups: {find: false}};
const setPrefs = [];
const sb = {
    console, setTimeout, clearTimeout, Date, Math, JSON, Promise,
    document: {getElementById: el, addEventListener: (t, f) => (docL[t] = docL[t] || []).push(f)},
    fetch: url => Promise.resolve({ok: true, json: () => Promise.resolve(
        url.startsWith('/api/files') ? {directory: '/lib', total: 1, files: [{name: 'a/done.mp3', tx_date: Date.now() / 1000 - 30}]}
        : url.startsWith('/api/voices/people') ? {people: [{}, {}]}
        : url.startsWith('/api/analyses') ? {analyses: [{}]}
        : {conversations: []})}),
    Shell: {pref: k => prefs[k], setPref: (k, v) => { prefs[k] = v; setPrefs.push([k, v]); }, onPrefs() {}, go() {}},
    esc: s => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/"/g, '&quot;'),
    escJs: s => String(s).replace(/\\/g, '\\\\').replace(/'/g, "\\'"),
    plural: (n, one, many) => `${n} ${n === 1 ? one : (many || one + 's')}`,
};
sb.window = sb; sb.globalThis = sb;
vm.createContext(sb);
vm.runInContext(`
    var activeFile = null;
    var queueRunningItems = [{filename: 'a/run.mp3', dir: '', progress: 42.4}];
    var queuePending = 3, queuePaused = false, queueRunningCount = 1, queueRetryable = 1;
    var queueFailedItems = [{filename: 'a/bad.mp4', dir: '', status: 'error', error: 'No audio track'},
                            {filename: 'a/gone.mp4', dir: '', status: 'cancelled', error: ''}];
    var srvStats = {done: 5, total: 9};
    var files = [];
`, sb);
vm.runInContext(HOME_SRC + '\nglobalThis.Home = Home;', sb);
(docL.DOMContentLoaded || []).forEach(f => f());
(async () => {
    await new Promise(r => setTimeout(r, 200));
    const out = {shownWithoutFile: !el('homeView').hidden, html: el('homeView').innerHTML};
    vm.runInContext("activeFile = 'a/x.mp3'", sb);
    sb.Home.sync();
    out.hiddenWithFile = el('homeView').hidden;
    sb.Home.show();
    out.shownWhenChosen = !el('homeView').hidden;
    sb.Home.toggle('voices');
    out.setPrefs = setPrefs;
    console.log('§§' + JSON.stringify(out));
})();
"""


@pytest.fixture(scope="module")
def run() -> dict:
    if shutil.which("node") is None:
        pytest.skip("node not available")
    script = _RUN.replace("HOME_SRC", json.dumps(HOME))
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


def test_home_shows_without_a_recording_and_when_chosen(run):
    assert run["shownWithoutFile"] is True
    assert run["hiddenWithFile"] is True
    assert run["shownWhenChosen"] is True


def test_the_four_live_cards(run):
    html = run["html"]
    for title in ("Running now", "Just finished", "Failed", "Recently opened"):
        assert title in html, title
    assert "run.mp3" in html and "42 %" in html and "3 files waiting" in html
    assert "done.mp3" in html and "just now" in html
    # only real failures, with the reason, and the existing Try again
    assert "bad.mp4" in html and "No audio track" in html and "gone.mp4" not in html
    assert "retryFailedQueue()" in html
    assert "x.mp3" in html and "2 min ago" in html


def test_pins_skip_unknown_names(run):
    html = run["html"]
    pins = html[html.index('class="home-pins"') : html.index('class="home-cards"')]
    assert "Voice map" in pins and "nonsense" not in pins


def test_groups_fold_and_the_choice_is_a_preference(run):
    html = run["html"]
    assert 'aria-expanded="false" onclick="Home.toggle(\'find\')"' in html
    assert "Same audio" not in html.split("Home.toggle('find')")[1].split("home-group")[0]
    assert run["setPrefs"] == [["home_groups", {"find": False, "voices": False}]]


def test_live_summaries(run):
    html = run["html"]
    assert "5 of 9 files transcribed" in html
    assert "1 running · 3 waiting" in html
    assert "2 people" in html
    # Analyses sits in the folded Find group: not drawn
    assert "1 analysis" not in html


def test_every_place_of_the_rail_is_on_home():
    import re

    rail = set(re.findall(r'data-place="([a-z]+)"', HTML)) - {"add", "home"}
    for place in rail:
        assert f"'{place}'" in HOME, place


def test_home_is_loaded_after_the_shell():
    assert HTML.index('src="/static/js/home.js') > HTML.index('src="/static/js/shell.js')
    assert 'id="homeView"' in HTML
