"""The list must not jump when something finishes in the background.

On the 'all' tab, scrolling through the list jumped back to the top every
time a transcript finished.

The path to this was long and inconspicuous:

    queue stream  →  several files finish in one tick
                  →  _scheduleScan()  →  scanFiles()
                  →  reloadFileList()  →  list empty  →  scrollTop 0

`reloadFileList()` even carried the sentence in its header "Does NOT keep
the scroll position — a new filter is a new list." That is true for
a filter change. For a background update it is wrong:
there the user did nothing, so nothing should move under their eyes
either.

The jump-back happens in `_renderWindow`, because the path goes through two
stations: first `renderFileList` empties the list (the browser clamps
scrollTop to 0 in the process), and only once the first page has arrived
from the server does it have a height again to jump into — this second
call comes from `_srvFetchPage` with no target at all.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
HARNESS = Path(__file__).resolve().parent / "js" / "filelist_harness.js"
FILELIST = REPO / "whisper_tm" / "static" / "js" / "filelist.js"
BATCH = FILELIST.parent / "batch.js"
DIRECTORY = FILELIST.parent / "directory.js"

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node not available")


def js(body: str) -> dict:
    source = (
        f"const {{buildIt}} = require({str(HARNESS)!r});\n"
        "const h = buildIt();\n"
        "const output = (o) => console.log('§§' + JSON.stringify(o));\n"
        "const idle = async () => { for (let i = 0; i < 6; i++) "
        "await new Promise(r => setImmediate(r)); };\n"
        f"(async () => {{\n{body}\n}})();\n"
    )
    done = subprocess.run(
        ["node", "-e", source],
        capture_output=True,
        text=True,
        timeout=60,
        env={"FILELIST_JS": str(FILELIST), "PATH": os.environ.get("PATH", "")},
    )
    if done.returncode != 0:
        raise AssertionError(f"node failed:\n{done.stderr}")
    lines = [z for z in done.stdout.splitlines() if z.startswith("§§")]
    assert lines, f"no result:\n{done.stdout}\n{done.stderr}"
    return json.loads(lines[-1][2:])


class TestThePositionStays:
    def test_in_the_background_the_user_stays_put(self):
        d = js("""
        h.run('reloadFileList()'); await idle();
        h.listing.scrollTop = 4000;
        h.run('reloadFileList({preserveScroll: true})');
        // One tick later: the list is empty, the browser has clamped.
        await new Promise(r => queueMicrotask(r));
        const midway = h.listing.scrollTop;
        await idle();
        output({midway, after: h.listing.scrollTop});
        """)
        assert d["midway"] == 0, (
            "the list was not empty in between — then the test isn't checking what it's supposed to"
        )
        assert d["after"] == 4000, "the user was thrown back to the top"

    def test_a_filter_change_starts_at_the_top(self):
        """The other half of the rule — a new list is a new
        list."""
        d = js("""
        h.run('reloadFileList()'); await idle();
        h.listing.scrollTop = 3000;
        h.run('reloadFileList()');
        await idle();
        output({after: h.listing.scrollTop});
        """)
        assert d["after"] == 0

    def test_the_remembered_position_is_forgotten_afterwards(self):
        """Otherwise it would pull a later scroll back."""
        d = js("""
        h.run('reloadFileList()'); await idle();
        h.listing.scrollTop = 4000;
        h.run('reloadFileList({preserveScroll: true})');
        await idle();
        h.listing.scrollTop = 1200;
        h.run('_renderWindow()');
        output({wish: h.value('srvList.desiredTop'),
                 top: h.listing.scrollTop});
        """)
        assert d["wish"] is None
        assert d["top"] == 1200, "an old wish pulled the scroll"

    def test_past_the_end_of_the_list_it_clamps(self):
        """If the list gets shorter on reload, the old position would give
        an empty area."""
        d = js("""
        h.run('reloadFileList()'); await idle();
        h.listing.scrollTop = 130000;
        h.run('reloadFileList({preserveScroll: true})');
        await idle();
        output({top: h.listing.scrollTop, height: h.value('_vTotalH')});
        """)
        assert d["top"] <= d["height"], "it jumped into the empty space"


class TestTheWayThere:
    """The chain that triggered this — each station recorded individually
    so that a "simplification" of any one of them will stand out."""

    def test_the_background_scan_says_so(self):
        source = BATCH.read_text(encoding="utf-8")
        place = source[source.index("function _scheduleScan") :][:600]
        assert "background: true" in place, (
            "the scheduled scan no longer reports itself as a background run"
        )

    def test_scanfiles_passes_it_on(self):
        source = DIRECTORY.read_text(encoding="utf-8")
        assert "reloadFileList({preserveScroll: !!opts.background})" in source

    def test_a_user_change_passes_nothing_on(self):
        """Tabs, sorting, and the name filter are user actions — there
        the position must NOT be kept."""
        source = FILELIST.read_text(encoding="utf-8")
        for occasion in ("function setFileTab", "function setSort"):
            block = source[source.index(occasion) :][:1400]
            assert "reloadFileList()" in block, occasion
            assert "preserveScroll" not in block, f"{occasion} holds on to the old position"
