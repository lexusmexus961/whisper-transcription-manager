"""The top bar and the rail: one design.

The places (Add, Home, Library, Queue, Find, Ask, Analyses, Voices,
Machine, Settings, Help & setup) live in the rail on the left. The top bar
keeps what is global and always at hand: the search, the live load (click
for its history), the VRAM eject and the done sound. Every control is one
monochrome icon from the sprite.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from html.parser import HTMLParser
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
STATIC = REPO / "whisper_tm" / "static"
INDEX = STATIC / "index.html"

_VOID = {"input", "br", "img", "meta", "link", "hr", "source", "use", "path", "circle"}


class Node:
    def __init__(self, tag, attrs, parent=None):
        self.tag, self.attrs, self.parent = tag, dict(attrs), parent
        self.children: list[Node] = []
        self.text = ""

    @property
    def classes(self) -> set[str]:
        return set((self.attrs.get("class") or "").split())

    def walk(self):
        yield self
        for c in self.children:
            yield from c.walk()

    def all_text(self) -> str:
        return self.text + "".join(c.all_text() for c in self.children)


class _Tree(HTMLParser):
    def __init__(self):
        super().__init__()
        self.root = Node("#root", {})
        self.cur = self.root

    def handle_starttag(self, tag, attrs):
        node = Node(tag, attrs, self.cur)
        self.cur.children.append(node)
        if tag not in _VOID:
            self.cur = node

    def handle_startendtag(self, tag, attrs):
        self.cur.children.append(Node(tag, attrs, self.cur))

    def handle_endtag(self, tag):
        node = self.cur
        while node is not self.root and node.tag != tag:
            node = node.parent
        if node is not self.root:
            self.cur = node.parent

    def handle_data(self, data):
        self.cur.text += data


@pytest.fixture(scope="module")
def page() -> Node:
    t = _Tree()
    t.feed(INDEX.read_text(encoding="utf-8"))
    return t.root


def _find(page: Node, cls: str) -> Node:
    return next(n for n in page.walk() if cls in n.classes)


def _right_side(page: Node) -> list[Node]:
    bar = _find(page, "topbar")
    kids = bar.children
    search = next(i for i, n in enumerate(kids) if "global-search" in n.classes)
    return kids[search + 1 :]


def _handler(node: Node) -> str | None:
    m = re.match(r"\s*([A-Za-z_$][\w$]*)\s*\(", node.attrs.get("onclick") or "")
    return m.group(1) if m else None


def _controls(group: Node) -> list[Node]:
    return [n for n in group.walk() if n is not group and _handler(n)]


def test_the_right_side_is_the_machine_strip(page):
    right = _right_side(page)
    assert len(right) == 1 and {"tb-group", "tb-system"} <= right[0].classes
    assert [_handler(c) for c in _controls(right[0])] == ["unloadVram", "toggleDoneSound"]
    # the load meters open the Machine place on its load history
    meters = next(n for n in right[0].walk() if n.attrs.get("id") == "sysLoad")
    assert meters.attrs["onclick"] == "Machine.open('load')"


def _rail(page: Node) -> Node:
    return next(n for n in page.walk() if n.attrs.get("id") == "rail")


def test_the_rail_lists_every_place_in_order(page):
    places = [n.attrs.get("data-place") for n in _rail(page).walk() if n.attrs.get("data-place")]
    assert places == [
        "add",
        "home",
        "library",
        "queue",
        "find",
        "ask",
        "analyses",
        "voices",
        "machine",
        "settings",
        "help",
    ]
    for n in _rail(page).walk():
        if n.attrs.get("data-place"):
            assert n.attrs["onclick"] == f"Shell.go('{n.attrs['data-place']}')"
    heads = [n.all_text().strip() for n in _rail(page).walk() if "rail-head" in n.classes]
    assert heads == ["Work", "Find", "System"]


def test_every_place_has_an_opener_in_the_shell():
    shell = (STATIC / "js" / "shell.js").read_text(encoding="utf-8")
    for place, opener in [
        ("add", "openUpload"),
        ("library", "openStatsView"),
        ("queue", "openQueueEditorFromBar"),
        ("ask", "searchTab('chat')"),
        ("voices", "vxOpen"),
        ("machine", "openHealthView"),
        ("settings", "openSettings"),
        ("help", "openHelp"),
    ]:
        line = next(z for z in shell.splitlines() if z.strip().startswith(place + ":"))
        assert (opener if opener.endswith(")") else opener + "()") in line, place


def test_the_setup_guide_is_a_tab_of_help(page):
    btn = next(n for n in page.walk() if n.attrs.get("id") == "btnHelpSetup")
    assert btn.attrs["onclick"] == "Help.tab('setup')"
    sg = (STATIC / "js" / "setupguide.js").read_text(encoding="utf-8")
    assert "openHelp('setup')" in sg[sg.index("function openSetupGuide") :]
    help_js = (STATIC / "js" / "help.js").read_text(encoding="utf-8")
    # the guide's pages and navigation move into the tab
    assert (
        "document.querySelector('#setupOverlay .set-main')" in help_js
        and "page.appendChild(main)" in help_js
    )


def test_help_has_four_tabs_and_where_is_what_is_read_off_the_layout(page):
    import re

    html = INDEX.read_text(encoding="utf-8")
    nav = html[html.index('id="helpTabs"') :]
    nav = nav[: nav.index("</nav>")]
    assert re.findall(r'data-tab="([a-z]+)"', nav) == ["where", "howto", "keys", "setup"]
    help_js = (STATIC / "js" / "help.js").read_text(encoding="utf-8")
    body = help_js[help_js.index("function renderWhere") : help_js.index("function init")]
    for sel in (
        "#rail > .rail-item[data-place]",
        "#svTabs .vx-tab",
        "#vxTabs .vx-tab",
        ".qe-tabs .qe-tab",
        "#mcTabs .vx-tab",
        ".dir-bar .dir-actions > button",
        "#folderMenu .bm-item",
    ):
        assert sel in body, sel
    # the how-tos speak of the rail, not of the old top bar buttons
    howto = help_js[help_js.index("function _renderHelp") :]
    assert "in the top bar" not in howto


def test_the_ids_the_scripts_use_are_still_there(page):
    ids = {n.attrs.get("id") for n in page.walk()}
    for i in (
        "sysLoad",
        "slCpu",
        "slRam",
        "slGpu",
        "slVram",
        "connIndicator",
        "connLlm",
        "connLlmDot",
        "connEmb",
        "connEmbDot",
        "railBadgeQueue",
        "railMachineDot",
        "btnSoundToggle",
        "btnAutoscan",
    ):
        assert i in ids, i


def test_every_button_has_an_icon_from_the_sprite_a_name_and_a_tooltip(page):
    sprite = _find(page, "icon-sprite")
    symbols = {n.attrs["id"] for n in sprite.walk() if n.tag == "symbol"}
    buttons = [n for g in _right_side(page) for n in g.walk() if n.tag == "button"]
    buttons += [n for n in _rail(page).walk() if n.attrs.get("data-place")]
    assert len(buttons) == 2 + 11
    for b in buttons:
        name = b.attrs.get("aria-label") or b.all_text().strip()
        assert name and b.attrs.get("title"), b.attrs
        icons = [n.attrs.get("href") for n in b.walk() if n.tag == "use"]
        assert icons, f"{name}: no icon"
        for href in icons:
            assert href.startswith("#") and href[1:] in symbols, (name, href)
        assert all("ui-ico" in n.classes for n in b.walk() if n.tag == "svg"), name


def test_no_colour_emoji_left_on_the_buttons(page):
    # The point of the one icon set: nothing that renders in its own colours
    for g in _right_side(page):
        for b in (n for n in g.walk() if n.tag == "button"):
            text = b.all_text().strip()
            assert re.fullmatch(r"[A-Za-z ]*", text), repr(text)


def test_the_sprite_is_the_first_thing_in_the_body(page):
    # <use href="#…"> needs the symbols in the same document, and the
    # hidden sprite must not take up room or be read out
    body = next(n for n in page.walk() if n.tag == "body")
    first = next(c for c in body.children if c.tag != "#comment")
    assert "icon-sprite" in first.classes
    assert first.attrs.get("aria-hidden") == "true"
    css = (STATIC / "css" / "app.css").read_text(encoding="utf-8")
    rule = re.search(r"\.icon-sprite\s*\{([^}]*)\}", css).group(1)
    assert "width:0" in rule.replace(" ", "") and "height:0" in rule.replace(" ", "")


def test_every_handler_is_defined_in_a_loaded_script(page):
    loaded = [
        n.attrs["src"].split("?")[0]
        for n in page.walk()
        if n.tag == "script" and n.attrs.get("src", "").startswith("/static/js/")
    ]
    source = "\n".join((STATIC / s[len("/static/") :]).read_text(encoding="utf-8") for s in loaded)
    for g in _right_side(page):
        for c in _controls(g):
            name = _handler(c)
            assert re.search(rf"^(?:async\s+)?function {name}\(", source, re.M), name


# ── Behaviour: state.js in node ─────────────────────────────────────────

_STATE_RUN = r"""
const fs = require('fs'), vm = require('vm');
const btn = {title: '', cls: new Set()};
// The icons live inside the button: overwriting its content would wipe them
for (const k of ['textContent', 'innerHTML', 'innerText'])
    Object.defineProperty(btn, k, {set(v) { throw new Error(k + ' = ' + v); }});
btn.classList = {toggle: (c, on) => on ? btn.cls.add(c) : btn.cls.delete(c),
                 add: c => btn.cls.add(c)};
const stored = [];
const sb = {console, setTimeout, clearTimeout,
            document: {getElementById: () => btn, addEventListener() {}},
            localStorage: {setItem: (k, v) => stored.push([k, v]), getItem() {}}};
sb.globalThis = sb;
vm.createContext(sb);
vm.runInContext(fs.readFileSync(process.env.STATE_JS, 'utf8'), sb);
const steps = [];
for (let i = 0; i < 2; i++) {
    vm.runInContext('toggleDoneSound()', sb);
    steps.push({muted: btn.cls.has('muted'), title: btn.title});
}
console.log('§§' + JSON.stringify({steps, stored}));
"""


@pytest.fixture(scope="module")
def state_run() -> dict:
    if shutil.which("node") is None:
        pytest.skip("node not available")
    done = subprocess.run(
        ["node", "-e", _STATE_RUN],
        capture_output=True,
        text=True,
        timeout=60,
        env={"STATE_JS": str(STATIC / "js" / "state.js"), "PATH": os.environ.get("PATH", "")},
    )
    assert done.returncode == 0, done.stderr
    line = next(z for z in done.stdout.splitlines() if z.startswith("§§"))
    return json.loads(line[2:])


def test_the_sound_toggle_switches_the_icon_by_class(state_run):
    assert state_run["steps"] == [
        {"muted": True, "title": "Enable done sound"},
        {"muted": False, "title": "Disable done sound"},
    ]
    assert state_run["stored"] == [["wtm_soundMuted", "1"], ["wtm_soundMuted", ""]]
    css = (STATIC / "css" / "app.css").read_text(encoding="utf-8")
    assert "#btnSoundToggle.muted .ico-on" in css
    assert "#btnSoundToggle.muted .ico-off" in css


def test_a_muted_sound_restored_at_load_keeps_its_icons():
    app = (STATIC / "js" / "app.js").read_text(encoding="utf-8")
    block = app[app.index("Restore sound mute state") :]
    block = block[: block.index("} catch")]
    assert "classList.add('muted')" in block
    assert "textContent" not in block and "innerHTML" not in block


def test_the_sprite_icons_do_not_reuse_the_toolbar_class():
    # .tb-ico is older: the emoji spans of the transcript toolbar. A sized
    # SVG rule under that name squeezed them into 16-px boxes.
    css = (STATIC / "css" / "app.css").read_text(encoding="utf-8")
    rules = re.findall(r"^\.tb-ico\s*\{[^}]*\}", css, re.M)
    assert not any("width" in r for r in rules), rules
    assert re.search(r"^\.ui-ico\s*\{", css, re.M)


def test_help_texts_show_the_icons_the_bar_has(page):
    # Help and setup guide name top-bar buttons by their icon (tbIcon in
    # state.js); a name without a symbol would render as an empty gap
    sprite = _find(page, "icon-sprite")
    symbols = {n.attrs["id"] for n in sprite.walk() if n.tag == "symbol"}
    used = set()
    for js in (STATIC / "js").glob("*.js"):
        used |= set(re.findall(r"tbIcon\('([\w-]+)'\)", js.read_text(encoding="utf-8")))
    assert {"guide", "help", "settings", "queue"} <= used
    assert {f"i-{name}" for name in used} <= symbols
    # And no colour emoji is left standing in for a top-bar button
    for name in ("help.js", "setupguide.js"):
        text = (STATIC / "js" / name).read_text(encoding="utf-8")
        for old in ("💓", "🧭", "❓", "✎ Queue", "⏏", "⚙"):
            assert old not in text, (name, old)


def test_the_search_field_opens_find_in_one_click(page):
    """A small button next to the search field opens the Find window
    directly (before, only Enter did it, unseen)."""
    box = _find(page, "global-search")
    btn = next(n for n in box.walk() if n.attrs.get("id") == "globalSearchOpen")
    assert btn.tag == "button" and _handler(btn) == "openSearchView"
    assert btn.attrs.get("aria-label") == "Open Find" and "Open Find" in btn.attrs.get("title", "")
    sprite = _find(page, "icon-sprite")
    symbols = {n.attrs["id"] for n in sprite.walk() if n.tag == "symbol"}
    assert [n.attrs.get("href") for n in btn.walk() if n.tag == "use"] == [
        "#i-expand"
    ] and "i-expand" in symbols
    css = (STATIC / "css" / "app.css").read_text(encoding="utf-8")
    # the field leaves room for it at its right end
    assert re.search(r"\.global-search input\.gs-trigger \{[^}]*padding:7px 38px 7px 34px", css)


def test_the_embedding_count_is_not_a_file_state(page):
    """ "To embed" is not one of the sidebar's file states — how much is
    embedded is shown under Settings → Search."""
    ids = {n.attrs.get("id") for n in page.walk()}
    assert "tabUnembedded" not in ids and "tabUnembeddedBtn" not in ids
    for name in ("stats.js", "filelist.js"):
        assert "tabUnembedded" not in (STATIC / "js" / name).read_text(encoding="utf-8")
