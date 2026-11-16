"""The file list's filter panel and sort menu (static/js/filterpanel.js):
what it sends to the server and the chips it shows."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

STATIC = Path(__file__).resolve().parent.parent / "whisper_tm" / "static"
SRC = (STATIC / "js" / "filterpanel.js").read_text(encoding="utf-8")
HTML = (STATIC / "index.html").read_text(encoding="utf-8")

_RUN = r"""
const vm = require('vm');
const els = {};
const el = id => els[id] || (els[id] = {id, hidden: true, innerHTML: '', textContent: '', classList: {toggle() {}}, setAttribute() {}});
let reloads = 0;
const sb = {
    console, document: {getElementById: el, querySelector: () => null, querySelectorAll: () => [], addEventListener() {}},
    esc: s => String(s), escJs: s => String(s), reloadFileList: () => reloads++,
    activeTab: 'all', setFileTab: t => { sb.activeTab = t; },
};
sb.window = sb; sb.globalThis = sb;
vm.createContext(sb);
vm.runInContext(SRC + '\nglobalThis.FilterPanel = FilterPanel;', sb);
const F = sb.FilterPanel, out = {};
out.empty = F.query();
F.set('type', 'video'); F.set('len', 'mid'); F.set('added', 7); F.set('language', 'en');
F.toggleHas('summary'); F.toggleHas('warnings'); F.set('sub', 'talks/2024');
F.addPerson('Alice'); F.addPerson('Bob'); F.addPerson('Alice');
out.query = F.query();
out.chips = F.active().map(c => c[0]);
F.toggleHas('summary'); F.removePerson('Alice');
out.after = F.query();
vm.runInContext("activeTab = 'error'", sb);
sb.activeTab = 'error';
out.statusChip = F.active()[0][0];
F.clearAll();
out.cleared = [F.query(), sb.activeTab];
out.reloads = reloads;
console.log('§§' + JSON.stringify(out));
"""


@pytest.fixture(scope="module")
def run() -> dict:
    if shutil.which("node") is None:
        pytest.skip("node not available")
    done = subprocess.run(
        ["node", "-e", _RUN.replace("SRC", json.dumps(SRC))],
        capture_output=True,
        text=True,
        timeout=60,
        env={"PATH": os.environ.get("PATH", "")},
    )
    assert done.returncode == 0, done.stderr
    return json.loads(next(z for z in done.stdout.splitlines() if z.startswith("§§"))[2:])


def test_nothing_chosen_sends_nothing(run):
    assert run["empty"] == ""


def test_every_filter_goes_to_the_server(run):
    q = run["query"]
    for part in (
        "f_type=video",
        "f_len_min=60",
        "f_len_max=600",
        "f_added=7",
        "f_lang=en",
        "f_has=summary,warnings",
        "f_sub=talks%2F2024",
        "f_people=Alice%2CBob",
    ):
        assert part in q, part


def test_chips_name_each_filter_once(run):
    assert run["chips"] == [
        "Video",
        "1–10 min",
        "Added: Last 7 days",
        "With Alice",
        "With Bob",
        "Language: en",
        "Has summary",
        "Has warnings",
        "In talks/2024",
    ]


def test_removing_and_clearing(run):
    assert "f_has=warnings" in run["after"] and "f_people=Bob" in run["after"]
    assert run["statusChip"] == "Status: Failed"
    assert run["cleared"] == ["", "all"], "Clear all also resets a status chosen in the panel"
    assert run["reloads"] >= 10, "each change reloads the list from the server"


def test_the_list_sends_the_panel_with_every_page_and_select_all():
    fl = (STATIC / "js" / "filelist.js").read_text(encoding="utf-8")
    assert fl.count("FilterPanel.query()") == 2
    assert 'id="flFilterBtn"' in HTML and 'id="flSortBtn"' in HTML and 'id="flChips"' in HTML


def test_the_footers_offer_what_the_list_can_do():
    batch = (STATIC / "js" / "batch.js").read_text(encoding="utf-8")
    # ⋯ in the footer: select all open, remove finished AI jobs — shown when they apply
    assert 'id="btnSelectOpen"' in HTML and 'onclick="closeBatchMenu();selectAllOpen()"' in HTML
    assert 'id="btnClearFinishedAi"' in HTML and "_lqClear('done')" in HTML
    assert "async function selectAllOpen()" in batch and "fetchFilteredNames('pending')" in batch
    # the selection footer: transcribe, summarize, analyze, delete, clear
    for b in ("btnTxSelected", "btnSummarizeSelected", "btnAnalyzeSelected", "btnDelSelected"):
        assert f'id="{b}"' in HTML, b
    body = batch[batch.index("async function summarizeSelected()") :]
    assert "kind: 'summary'" in body and "statusOf(n) === 'done'" in body
