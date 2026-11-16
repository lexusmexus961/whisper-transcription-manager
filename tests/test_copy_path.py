"""Being able to copy a file's path — even without a transcript.

A "copy path" button in the file view, shown even when there is no
transcript, placed in the details next to the modified date.

For a file without a transcript the header section shows duration, size
and modification date, but without the button the complete path would
only be accessible via the address bar of a download — or not at all.
Precisely with such files you need it most, because you want to view
them in the file manager or on the command line.

The path itself is composite: `f.name` is RELATIVE to the root that
is currently in effect (docs/PATHS.md), and that is in the directory field. Both
together only make something that you can insert somewhere.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
HARNESS = Path(__file__).resolve().parent / "js" / "fileview_harness.js"
STATIC = REPO / "whisper_tm" / "static"
FILEVIEW = STATIC / "js" / "fileview.js"
CSS = STATIC / "css" / "app.css"

node = pytest.mark.skipif(shutil.which("node") is None, reason="node not available")


def js(body: str, directory: str = "/mnt/media/videos") -> dict:
    source = (
        f"const {{buildIt}} = require({str(HARNESS)!r});\n"
        f"const h = buildIt({directory!r});\n"
        "const output = (o) => console.log('§§' + JSON.stringify(o));\n"
        f"{body}\n"
    )
    done = subprocess.run(
        ["node", "-e", source],
        capture_output=True,
        text=True,
        timeout=60,
        env={"FILEVIEW_JS": str(FILEVIEW), "PATH": os.environ.get("PATH", "")},
    )
    if done.returncode != 0:
        raise AssertionError(f"node failed:\n{done.stderr}")
    lines = [z for z in done.stdout.splitlines() if z.startswith("§§")]
    assert lines, f"no result:\n{done.stdout}\n{done.stderr}"
    return json.loads(lines[-1][2:])


@node
class TestThePath:
    def test_root_and_relative_path_make_the_whole(self):
        d = js("""
        output({path: h.value("filePath({name: 'subfolders/film.mkv'})")});
        """)
        assert d["path"] == "/mnt/media/videos/subfolders/film.mkv"

    def test_one_slash_too_many_does_no_harm(self):
        d = js(
            """output({path: h.value("filePath({name: 'a.mkv'})")});""",
            directory="/mnt/media/videos/",
        )
        assert d["path"] == "/mnt/media/videos/a.mkv"

    def test_without_a_root_the_relative_path_stays(self):
        """A partial path is more useful than none."""
        d = js("""output({path: h.value("filePath({name: 'a.mkv'})")});""", directory="")
        assert d["path"] == "a.mkv"


@node
class TestCopying:
    def test_without_https_the_old_path_is_taken(self):
        """The app runs on a home network via http — navigator.clipboard
        doesn't exist there. The old way is not a fallback here,
        but the normal case."""
        d = js("""
        h.run("var btn = {dataset: {path: '/mnt/media/videos/a.mkv'}, "
               + "innerHTML: 'x'}; fiCopyPath(btn);");
        output({copied: h.copied});
        """)
        assert d["copied"] == ["/mnt/media/videos/a.mkv"]

    def test_the_button_reports_success(self):
        d = js("""
        h.run("var btn = {dataset: {path: '/x/y.mkv'}, innerHTML: 'whatever'};"
               + " fiCopyPath(btn);");
        output({text: h.value("btn.innerHTML")});
        """)
        assert "copied" in d["text"]


class TestItIsAlwaysThere:
    """The actual requirement: even without a transcript."""

    def test_the_button_does_not_depend_on_the_transcript(self):
        # The Details list (recording.js): "Where it is" with the path and
        # Copy is drawn whatever the transcript — outside its own section
        source = (FILEVIEW.parent / "recording.js").read_text(encoding="utf-8")
        body = source[source.index("function detailsHtml(") : source.index("function detailsFill(")]
        where = body[body.index("<h4>Where it is</h4>") :]
        assert "fi-path" in where and "fiCopyPath" in where
        assert body.index("${tx ?") < body.index("<h4>Where it is</h4>"), (
            "not inside the transcript's section"
        )

    def test_it_sits_next_to_the_modified_date(self):
        """The path sits under the file's own facts (its dates among them),
        where one looks for it."""
        source = (FILEVIEW.parent / "recording.js").read_text(encoding="utf-8")
        body = source[source.index("function detailsHtml(") : source.index("function detailsFill(")]
        assert body.index("row('Modified'") < body.index("fi-path")

    def test_it_looks_clickable(self):
        css = CSS.read_text(encoding="utf-8")
        block = css[css.index(".fi-chip.fi-path") :][:300]
        assert "cursor:pointer" in block, "looks like a badge"
        assert ":hover" in css[css.index(".fi-chip.fi-path") :][:400]
