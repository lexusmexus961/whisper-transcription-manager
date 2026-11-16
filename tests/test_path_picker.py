"""Path field: one open button, and ⌂ back to the library root.

"Open this folder" in the picker and the → in the field did the same
thing. The picker's button stays — Enter in the
field still opens what is typed — and the → became ⌂: open the library
root, as the server knows it. Not the folder the page was loaded with:
``config.media_dir`` moves along with every folder change.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
HARNESS = REPO / "tests" / "js" / "pathpicker_harness.js"
JS = REPO / "whisper_tm" / "static" / "js" / "pathpicker.js"
INDEX = REPO / "whisper_tm" / "static" / "index.html"

node = pytest.mark.skipif(shutil.which("node") is None, reason="node not available")


def js(body: str, answers: dict) -> dict:
    source = (
        f"const {{buildIt}} = require({str(HARNESS)!r});\n"
        f"const h = buildIt({{answers: {json.dumps(answers)}}});\n"
        "const output = (o) => console.log('§§' + JSON.stringify(o));\n"
        "(async () => {\n" + body + "\n})().catch(e => { console.error(e); process.exit(1); });\n"
    )
    done = subprocess.run(
        ["node", "-e", source],
        capture_output=True,
        text=True,
        timeout=60,
        env={"PATHPICKER_JS": str(JS), "PATH": os.environ.get("PATH", "")},
    )
    if done.returncode != 0:
        raise AssertionError(f"node failed:\n{done.stderr}\n{done.stdout}")
    lines = [z for z in done.stdout.splitlines() if z.startswith("§§")]
    assert lines, f"no result:\n{done.stdout}\n{done.stderr}"
    return json.loads(lines[-1][2:])


@node
class TestHome:
    def test_home_opens_the_root_the_server_names(self):
        r = js(
            """
            h.elements.dirInput.value = '/mnt/media/Archive/deep/folder';
            await h.run('ppHome()');
            output({requests: h.requests, opened: h.opened});
        """,
            answers={"/api/browse": [200, {"path": "/mnt/media", "dirs": []}]},
        )
        # Asked without a path: the server answers with the first allowed root
        assert r == {"requests": ["/api/browse"], "opened": ["/mnt/media"]}

    def test_without_a_root_nothing_opens_and_it_says_why(self):
        r = js(
            """
            h.elements.dirInput.value = '/mnt/media/Archive';
            await h.run('ppHome()');
            output({opened: h.opened, status: h.elements.dirStatus.textContent,
                    field: h.elements.dirInput.value});
        """,
            answers={"/api/browse": [403, {"error": "No allowed directories configured"}]},
        )
        assert r["opened"] == []
        assert "No allowed directories configured" in r["status"]
        assert r["field"] == "/mnt/media/Archive"


BROWSE = {
    "/api/browse?path=%2Fmnt%2Fmedia": [
        200,
        {"path": "/mnt/media", "dirs": [{"name": "recordings", "path": "/mnt/media/recordings"}]},
    ],
    "/api/browse?path=%2Fmnt%2Fmedia%2Frecordings": [
        200,
        {
            "path": "/mnt/media/recordings",
            "dirs": [{"name": "interviews", "path": "/mnt/media/recordings/interviews"}],
        },
    ],
    "/api/browse?path=%2Fmnt%2Fmedia%2FBackups": [200, {"path": "/mnt/media/Archive", "dirs": []}],
}


@node
class TestTwoPlaces:
    """The queue editor's "Add folder" field got the file list's picker
    (the same dropdown and favorites as the file list).
    Choosing there counts a folder for the queue — it must never open it
    in the file list, and the file list's field must stay as it was."""

    def test_choosing_in_add_folder_counts_and_leaves_the_file_list_alone(self):
        r = js(
            """
            h.elements.dirInput.value = '/mnt/media/Archive';
            h.elements.qaFolder.value = '/mnt/media';
            await h.run("openPathPicker('qa')");
            await new Promise(r => setTimeout(r, 0));
            const shown = h.elements.qaPathPicker.style.display;
            await h.run("ppNavigate('/mnt/media/recordings')");
            const list = h.elements.qaPpList.innerHTML;
            h.run("ppOpen('/mnt/media/recordings/interviews')");
            output({shown, list: list.includes('interviews') && list.includes('>Choose</button>'),
                    hidden: h.elements.qaPathPicker.style.display, chosen: h.chosen,
                    opened: h.opened, fileList: h.elements.dirInput.value,
                    mainList: h.elements.ppList.innerHTML});
        """,
            answers=BROWSE,
        )
        assert r == {
            "shown": "flex",
            "list": True,
            "hidden": "none",
            "chosen": ["/mnt/media/recordings/interviews"],
            "opened": [],
            "fileList": "/mnt/media/Archive",
            "mainList": "",
        }

    def test_the_file_list_still_opens_what_it_chooses(self):
        r = js(
            """
            h.elements.dirInput.value = '/mnt/media';
            await h.run('openPathPicker()');
            await new Promise(r => setTimeout(r, 0));
            h.run("ppOpen('/mnt/media/recordings')");
            output({opened: h.opened, chosen: h.chosen, list: h.elements.ppList.innerHTML.includes('>Open</button>')});
        """,
            answers=BROWSE,
        )
        assert r == {"opened": ["/mnt/media/recordings"], "chosen": [], "list": True}

    def test_enter_counts_a_typed_path_and_cancel_restores_the_field(self):
        r = js(
            """
            h.elements.qaFolder.value = '/mnt/media';
            await h.run("openPathPicker('qa')");
            await h.run("ppNavigate('/mnt/media/recordings')");
            const browsed = h.elements.qaFolder.value;
            h.run("closePathPicker(true)");
            const restored = h.elements.qaFolder.value;
            h.elements.qaFolder.value = '/mnt/media/Archive';
            h.run("onPathKey({key: 'Enter', preventDefault() {}, stopPropagation() {}}, 'qa')");
            output({browsed, restored, chosen: h.chosen, opened: h.opened});
        """,
            answers=BROWSE,
        )
        assert r == {
            "browsed": "/mnt/media/recordings",
            "restored": "/mnt/media",
            "chosen": ["/mnt/media/Archive"],
            "opened": [],
        }

    def test_esc_in_add_folder_closes_only_the_picker(self):
        # The queue editor closes on Esc — the picker's Esc must stop there,
        # in the file list it bubbles on as before.
        r = js(
            """
            const esc = () => { const e = {key: 'Escape', stopped: false, stopPropagation() { this.stopped = true; }}; return e; };
            h.elements.qaFolder.value = '/mnt/media';
            await h.run("openPathPicker('qa')");
            const qa = esc(); h.run('onPathKey')(qa, 'qa');
            const qaClosed = h.elements.qaPathPicker.style.display;
            const again = esc(); h.run('onPathKey')(again, 'qa');
            h.elements.dirInput.value = '/mnt/media';
            await h.run('openPathPicker()');
            const main = esc(); h.run('onPathKey')(main);
            output({qa: qa.stopped, qaClosed, closedAlready: again.stopped, main: main.stopped,
                    mainClosed: h.elements.pathPicker.style.display});
        """,
            answers=BROWSE,
        )
        assert r == {
            "qa": True,
            "qaClosed": "none",
            "closedAlready": False,
            "main": False,
            "mainClosed": "none",
        }

    def test_one_picker_at_a_time(self):
        r = js(
            """
            h.elements.dirInput.value = '/mnt/media';
            await h.run('openPathPicker()');
            await h.run("ppNavigate('/mnt/media/recordings')");
            h.elements.qaFolder.value = '/mnt/media/Archive';
            await h.run("openPathPicker('qa')");
            output({main: h.elements.pathPicker.style.display, qa: h.elements.qaPathPicker.style.display,
                    fileList: h.elements.dirInput.value});
        """,
            answers=BROWSE,
        )
        # The file list's picker closed as cancelled: its field is back
        assert r == {"main": "none", "qa": "flex", "fileList": "/mnt/media"}

    def test_favorites_are_shared_and_the_star_follows(self):
        r = js(
            """
            h.elements.qaFolder.value = '/mnt/media/recordings/interviews';
            h.run("toggleCurrentFav(null, 'qa')");
            const on = [h.elements.qaFavStar.textContent, h.elements.qaFavStar.classList.contains('active')];
            h.elements.dirInput.value = '/mnt/media';
            await h.run('openPathPicker()');
            const mainFavs = h.elements.ppFavs.innerHTML;
            h.run("ppRemoveFav('/mnt/media/recordings/interviews')");
            output({on, favsThen: mainFavs.includes('interviews'), favs: h.favs(),
                    off: h.elements.qaFavStar.textContent});
        """,
            answers=BROWSE,
        )
        assert r == {"on": ["★", True], "favsThen": True, "favs": [], "off": "☆"}

    def test_home_opens_in_the_file_list_even_after_add_folder(self):
        r = js(
            """
            h.elements.qaFolder.value = '/mnt/media';
            await h.run("openPathPicker('qa')");
            h.run("closePathPicker(true)");
            await h.run('ppHome()');
            output({opened: h.opened, chosen: h.chosen});
        """,
            answers={**BROWSE, "/api/browse": [200, {"path": "/mnt/media", "dirs": []}]},
        )
        assert r == {"opened": ["/mnt/media"], "chosen": []}


class TestMarkup:
    def test_the_field_has_home_instead_of_a_second_open_button(self):
        html = INDEX.read_text(encoding="utf-8")
        field = re.search(r'<div class="path-field" id="pathField">(.*?)</div>', html, re.S)
        assert field, "path field missing"
        buttons = re.findall(r"<button[^>]*>[^<]*</button>", field.group(1))
        assert not any("changeDir()" in b for b in buttons), buttons
        assert any('id="btnPathHome"' in b and 'onclick="ppHome(event)"' in b for b in buttons)

    def test_add_folder_has_the_same_field_and_picker(self):
        html = INDEX.read_text(encoding="utf-8")
        field = re.search(r'<div class="path-field" id="qaPathField">(.*?)</div>', html, re.S)
        assert field, "Add folder path field missing"
        inp = re.search(r'<input[^>]*id="qaFolder"[^>]*>', field.group(1), re.S)
        assert inp, "qaFolder input missing"
        for handler in (
            "onfocus=\"openPathPicker('qa')\"",
            "oninput=\"onPathInput('qa')\"",
            "onkeydown=\"onPathKey(event, 'qa')\"",
        ):
            assert handler in inp.group(0), handler
        assert "onclick=\"toggleCurrentFav(event, 'qa')\"" in field.group(1)
        assert "onclick=\"togglePathPicker(event, 'qa')\"" in field.group(1)
        for part in ("qaPathPicker", "qaPpBreadcrumb", "qaPpFavs", "qaPpList", "qaPpHint"):
            assert f'id="{part}"' in html, part

    def test_the_picker_keeps_its_open_button(self):
        html = INDEX.read_text(encoding="utf-8")
        assert re.search(
            r'<button[^>]*id="ppOpenBtn"[^>]*onclick="ppChooseCurrent\(\)"'
            r"[^>]*>Open this folder</button>",
            html,
        )
