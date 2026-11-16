"""Regressions from the UI review of the search workspace.

Each test pins one defect that was found by reading the code:

* the Files tab never paged past its first 200 rows (no scroll listener on
  ``#svFileList``);
* the "Recordings" view appended every card again on each load-more page;
* late facets wiped the filename block in the "Recordings" view;
* a server error in the Files tab was written into the hidden text list;
* one failed RAG call switched "Text + RAG" off for the whole session;
* a hit-map folder was found by (depth, name), so ``B / 2024`` zoomed into
  ``A/2024``;
* the small search's dropdown was refilled by an answer to an older query;
* a hit's start time went into an ``onclick`` unchecked.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
HARNESS = REPO / "tests" / "js" / "search_harness.js"
JS = REPO / "whisper_tm" / "static" / "js"

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node not available")


def js(body: str) -> dict:
    source = (
        f"const A = require({str(HARNESS)!r});\n"
        "const output = (o) => console.log('§§' + JSON.stringify(o));\n"
        f"(async () => {{\n{body}\n}})().catch(e => {{ console.error(e); process.exit(1); }});\n"
    )
    done = subprocess.run(
        ["node", "-e", source],
        capture_output=True,
        text=True,
        timeout=60,
        env={
            "SEARCH_JS": str(JS / "searchview.js"),
            "SEARCH_SMALL": str(JS / "search.js"),
            "SEARCH_MAP": str(JS / "searchmap.js"),
            "PATH": os.environ.get("PATH", ""),
        },
    )
    if done.returncode != 0:
        raise AssertionError(f"node failed:\n{done.stderr}")
    lines = [z for z in done.stdout.splitlines() if z.startswith("§§")]
    assert lines, f"no result:\n{done.stdout}\n{done.stderr}"
    return json.loads(lines[-1][2:])


def test_the_files_tab_pages_on_scroll():
    d = js("""
    A.setupSearchView();
    const box = A.el('svFileList');
    const scroll = (box._ls || {}).scroll || [];
    A.svState.tab = 'files';
    A.svState.query = 'abc';
    A.svState.done = false;
    A.svState.loading = false;
    A.clearFetched();
    Object.assign(box, {scrollTop: 1000, clientHeight: 500, scrollHeight: 1200});
    scroll.forEach(fn => fn());
    await new Promise(r => setTimeout(r, 10));
    output({listeners: scroll.length,
            asked: A.fetched().filter(u => u.startsWith('/api/search?')).length});
    """)
    assert d["listeners"] == 1
    assert d["asked"] == 1


def test_a_scroll_in_the_hidden_list_loads_nothing():
    d = js("""
    A.setupSearchView();
    const box = A.el('svFileList');
    A.svState.tab = 'text';
    A.svState.query = 'abc';
    A.svState.done = false;
    A.clearFetched();
    Object.assign(box, {scrollTop: 1000, clientHeight: 500, scrollHeight: 1200});
    ((box._ls || {}).scroll || []).forEach(fn => fn());
    await new Promise(r => setTimeout(r, 10));
    output({asked: A.fetched().length});
    """)
    assert d["asked"] == 0


RECORDINGS = """
    A.svState.tab = 'text';
    A.svState.view = 'recordings';
    A.svState.files = [];
    A.svState.facets = {files: [{filename: 'a.mp4', hits: 3}, {filename: 'b.mp4', hits: 2}]};
    const hit = (f, t) => ({filename: f, start: t, start_fmt: '0:0' + t, text: 'x'});
"""


def test_load_more_in_the_recordings_view_does_not_repeat_the_cards():
    d = js(
        RECORDINGS
        + """
    A.svRenderHits([hit('a.mp4', 1)], true);
    A.svRenderHits([hit('a.mp4', 2), hit('b.mp4', 3)], false);
    A.svRenderHits([hit('b.mp4', 4)], false);
    output({cards: (A.el('svResults').innerHTML.match(/class="sv-rec"/g) || []).length});
    """
    )
    assert d["cards"] == 2


def test_late_facets_keep_the_filename_block():
    d = js(
        RECORDINGS
        + """
    A.svRenderHits([hit('a.mp4', 1)], true);
    A.svState.files = [{filename: 'zz-named.mp4', display_name: 'zz-named.mp4'}];
    A.svRenderFiles(A.svState.files);
    A.svRenderFacets({files: A.svState.facets.files, file_count: 2});
    output({html: A.el('svResults').innerHTML});
    """
    )
    assert "zz-named.mp4" in d["html"]
    assert d["html"].count('class="sv-rec"') == 2


def test_an_error_in_the_files_tab_lands_in_its_own_list():
    d = js("""
    A.svState.tab = 'files';
    A.svState.query = 'abc';
    A.svState.done = false;
    A.svState.loading = false;
    A.el('svFileList').innerHTML = 'Searching…';
    A.setAnswer({error: 'database locked'});
    await A.svLoadMore(true, A.svState.seq);
    output({files: A.el('svFileList').innerHTML, text: A.el('svResults').innerHTML});
    """)
    assert "database locked" in d["files"]
    assert "database locked" not in d["text"]


def test_rag_is_tried_again_after_a_pause():
    d = js("""
    let now = 1_000_000;
    Date.now = () => now;
    A.svState.mode = 'both';
    A.svState.query = 'someone is upset';
    A.answerFor('/api/rag/search', {ok: false, error: 'llama-server not reachable'});
    const rag = () => A.fetched().filter(u => u.startsWith('/api/rag/search')).length;
    await A.svMeaningSearch(A.svState.seq, 0);
    const first = rag();
    await A.svMeaningSearch(A.svState.seq, 0);      // right after: quiet
    const soon = rag();
    now += 61_000;
    await A.svMeaningSearch(A.svState.seq, 0);      // a minute later: again
    output({first, soon, later: rag(), down: A.ragDown()});
    """)
    assert d["first"] == 1 and d["soon"] == 1 and d["later"] == 2


def test_a_folder_tile_zooms_into_its_own_branch():
    """Two folders named 2024 in different branches: the tile under B must
    zoom into B/2024, not into the first branch that has a 2024."""
    d = js("""
    const tree = A.smBuildTree([
        {filename: 'A/2024/a1.mp4', hits: 10}, {filename: 'A/2023/a2.mp4', hits: 5},
        {filename: 'B/2024/b1.mp4', hits: 3}, {filename: 'B/2022/b2.mp4', hits: 2},
    ]);
    A.setMapPath([tree]);
    const out = [];
    const kids = [...tree.kids.values()].sort((a, b) => b.hits - a.hits);
    A.svMapLevel(kids, 0, 0, 1600, 1000, [], 10, true, out);
    const tiles = A.map().tiles;
    const i = tiles.findIndex(p => p[0].name === 'B' && p[p.length - 1].name === '2024');
    const html = out.join('');
    A.svMapInto(i);
    const path = A.map().path;
    output({i, clickable: html.includes('svMapInto(' + i + ')'),
            names: path.map(n => n.name),
            right: path[path.length - 1] === tree.kids.get('B').kids.get('2024')});
    """)
    assert d["i"] >= 0 and d["clickable"]
    assert d["names"] == ["All hits", "B", "2024"]
    assert d["right"]


def test_the_small_search_drops_an_answer_to_an_older_query():
    d = js("""
    const field = A.el('globalSearch');
    A.setAnswer({results: [{filename: 'x.mp4', start: 1, start_fmt: '0:01', end_fmt: '0:02',
                            text: 'abc'}], files: [], total: 1});
    field.value = 'ab';
    const older = A.runGlobalSearch();
    field.value = 'a';
    await A.runGlobalSearch();
    await older;
    output({html: A.el('spResults').innerHTML});
    """)
    # "a" still offers places to go to, but no hit of the older query
    assert "sp-said" not in d["html"] and "abc" not in d["html"]


def test_a_start_time_goes_into_onclick_only_as_a_number():
    d = js("""
    A.el('globalSearch').value = 'abc';
    A.setAnswer({results: [{filename: 'x.mp4', start: '0);alert(1)//', start_fmt: '0:01',
                            end_fmt: '0:02', text: 'abc'}], files: [], total: 1});
    await A.runGlobalSearch();
    output({html: A.el('spResults').innerHTML});
    """)
    assert "alert(1)" not in d["html"]
    assert "goToSearchResult('x.mp4', 0," in d["html"]


def test_no_german_identifiers_left():
    src = (JS / "searchview.js").read_text(encoding="utf-8")
    css = (JS.parent / "css" / "searchview.css").read_text(encoding="utf-8")
    for word in ("const lies", "const ab ", "roles:", "'kern'"):
        assert word not in src, word
    assert ".kern" not in css


def test_the_hit_map_builds_its_folders_from_one_root():
    """A hit's rel_path is relative to the folder open when the file was
    transcribed: the map mixed those bases and put a file of "edge cases"
    at the top beside the folder "samples"."""
    src = (JS / "searchmap.js").read_text(encoding="utf-8")
    fns = "\n".join(
        src[src.index(f"function {n}(") :].split("\n}\n")[0] + "\n}"
        for n in ("_smAbs", "_smCommonDir", "smBuildTree")
    )
    script = (
        fns
        + """
    const tree = smBuildTree([
        {filename: 'samples/a.opus', file_dir: '/lib/media', hits: 3},
        {filename: 'b.mp3', file_dir: '/lib/media/edge cases', hits: 2},
        {filename: 'c.mp4', file_dir: 'relative', hits: 1}]);
    const one = smBuildTree([{filename: 'samples/a.opus', file_dir: '/lib/media', hits: 3},
                             {filename: 'b.mp3', file_dir: '/lib/media/edge cases', hits: 2}]);
    console.log('§§' + JSON.stringify({
        mixed: [...tree.kids.keys()],
        top: [...one.kids.keys()],
        edge: [...one.kids.get('edge cases').kids.keys()]}));
    """
    )
    done = subprocess.run(["node", "-e", script], capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    r = json.loads([z for z in done.stdout.splitlines() if z.startswith("§§")][-1][2:])
    # A hit without a root keeps the old way: its own name
    assert r["mixed"] == ["samples", "b.mp3", "c.mp4"]
    assert r["top"] == ["samples", "edge cases"] and r["edge"] == ["b.mp3"]
