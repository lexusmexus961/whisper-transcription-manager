"""A naming answer that arrives late must land on the voices that were named.

Both requests take a moment; meanwhile the user clicks another point, or
the threshold slider reloads the graph and hands out new group numbers.
Before, `mapRename` read `PANEL.nodes` AFTER the request (the name landed on
the newly clicked voice), and `graphNameGroup` re-applied the answer by the
group NUMBER (it landed on whichever new group got that number).
"""

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

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node not available")


def js(body: str) -> dict:
    source = (
        f"const {{API}} = require({str(HARNESS)!r});\n"
        "const output = (o) => console.log('§§' + JSON.stringify(o));\n"
        f"(async () => {{ {body} }})().catch(e => {{ console.error(e); process.exit(1); }});\n"
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


SETUP = """
const members = [{id: 11, display: 'a.mp4', person: ''},
                 {id: 12, display: 'b.mp4', person: ''},
                 {id: 13, display: 'c.mp4', person: ''}];
const names = ['', '', ''];
API.mapSet({n: 3, members, name: names, x: new Float64Array(3),
            y: new Float64Array(3), sx: 1, ox: 0, oy: 0});
API.GRAPH.groupOf = Int32Array.from([1, 1, 2]);
API.GRAPH.groups = [{nr: 1, n: 2, name: ''}, {nr: 2, n: 1, name: ''}];
API.GRAPH.rowOf = new Map([[11, 0], [12, 1], [13, 2]]);
let release;
const answer = (body) => new Promise(r => { release = () => r({ok: true, json: async () => body}); });
"""


def test_rename_lands_on_the_voice_sent_not_the_one_clicked_since():
    r = js(
        SETUP
        + """
        API.mapSet({chosen: members[0]});
        API.PANEL.nodes = 0;
        API.setDocument(Object.assign({}, globalThis.document || {}, {
            getElementById: (id) => id === 'mapName' ? {value: 'Alice'} : null,
            querySelectorAll: () => [], addEventListener: () => {}}));
        API.setFetch(() => answer({}));
        const p = API.mapRename();
        API.PANEL.nodes = 2;                 // another point clicked meanwhile
        release(); await p;
        output({person: members.map(m => m.person), drawn: names});
    """
    )
    assert r["person"] == ["Alice", "", ""]
    # the label drawn on the map: voice 0, not the point clicked since
    assert r["drawn"] == ["Alice", "", ""]


def test_group_naming_applies_by_ids_after_a_renumbering_reload():
    r = js(
        SETUP
        + """
        API.setDocument(Object.assign({}, {
            getElementById: (id) => id === 'mapGroupName' ? {value: 'Bob', focus() {}} : null,
            querySelectorAll: () => [], addEventListener: () => {}}));
        API.setFetch(() => answer({person: 'Bob', set_flag: 2, foreign_names: []}));
        const p = API.graphNameGroup(1);
        // A reload renumbers: the old group 1 (ids 11, 12) is now group 2,
        // and number 1 belongs to voice 13.
        API.GRAPH.seq++;
        API.GRAPH.groupOf = Int32Array.from([2, 2, 1]);
        API.GRAPH.groups = [{nr: 1, n: 1, name: ''}, {nr: 2, n: 2, name: ''}];
        API.graphDropGroupIndex();
        release(); await p;
        output({persons: members.map(m => m.person),
                groups: API.GRAPH.groups.map(g => g.name)});
    """
    )
    assert r["persons"] == ["Bob", "Bob", ""]
    # the renumbered group 1 (voice 13) did not get the name
    assert r["groups"][0] == ""
