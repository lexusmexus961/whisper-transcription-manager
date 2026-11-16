"""
Delivered frontend files must be well-formed.

Reason: in ``static/index.html`` a ``<script>`` tag was missing the
closing ``>``:

    <script src="/static/js/search.js"</script>

The parser thereby consumes the FOLLOWING LINE as text content — ``upload.js``
was never created into an element and thus never loaded. The entire upload was
dead (file upload, drag-and-drop, YouTube download), visible only as
``openUpload is not defined`` in the browser console.

One character, one dead feature, no test that would have noticed it.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser
from pathlib import Path

import pytest

_STATIC = Path(__file__).parent.parent / "whisper_tm" / "static"
_INDEX = _STATIC / "index.html"


class _ScriptCollector(HTMLParser):
    def __init__(self):
        super().__init__()
        self.loaded: list[str] = []
        self.garbage: list[tuple] = []
        self.swallowed: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag != "script":
            return
        d = dict(attrs)
        if d.get("src"):
            self.loaded.append(d["src"])
        # A clean tag has at most src/type/defer/async — anything
        # beyond that with empty value indicates a broken tag.
        for name, value in attrs:
            if value is None and name not in ("defer", "async"):
                self.garbage.append((d.get("src"), name))

    def handle_data(self, data):
        if "<script" in data:
            self.swallowed.append(data.strip()[:80])


@pytest.fixture(scope="module")
def parsed():
    p = _ScriptCollector()
    p.feed(_INDEX.read_text(encoding="utf-8"))
    return p


def test_no_script_tag_swallows_next(parsed):
    assert not parsed.swallowed, (
        "A <script> tag is not closed — the parser reads the "
        f"following tag as text: {parsed.swallowed}"
    )


def test_no_garbage_attributes_on_script_tags(parsed):
    assert not parsed.garbage, f"Broken <script> tag (attributes without value): {parsed.garbage}"


def test_each_js_file_is_included(parsed):
    """What's under static/js/ should also be loaded.

    Catches the same error from the other side: a file that exists,
    but nobody includes it.
    """
    available = {p.name for p in (_STATIC / "js").glob("*.js")}
    # Strip query strings for cache busting (dashboard.js?v=6)
    loaded = {Path(s.split("?")[0]).name for s in parsed.loaded}
    missing = available - loaded
    assert not missing, f"JS files not included: {sorted(missing)}"


def test_all_included_files_exist(parsed):
    missing = [
        s
        for s in parsed.loaded
        if s.startswith("/static/") and not (_STATIC / s.split("?")[0][len("/static/") :]).exists()
    ]
    assert not missing, f"Included, but don't exist: {missing}"


@pytest.mark.parametrize("file", sorted((_STATIC / "js").glob("*.js")), ids=lambda p: p.name)
def test_no_literal_line_separator_in_javascript(file):
    """U+2028/U+2029 count as line terminators in JavaScript.

    If they appear raw in a file — even in a comment — the comment ends there
    and the next line is read as code. Exactly this happened when fixing the
    escJs error in state.js.
    """
    text = file.read_text(encoding="utf-8")
    for char, name in ((" ", "U+2028"), (" ", "U+2029")):
        # Written as an escape inside a regex literal it is harmless —
        # what is searched for is the RAW character.
        assert char not in text, (
            f"{file.name} contains a raw {name}; write as escape sequence (\\u2028) write"
        )


@pytest.mark.parametrize("file", sorted((_STATIC / "js").glob("*.js")), ids=lambda p: p.name)
def test_balanced_template_literals(file):
    """Rough plausibility check: equal number of backticks per file.

    Not a parser, but it catches the most common way to break a template literal
    when editing.
    """
    text = file.read_text(encoding="utf-8")
    unescaped = re.sub(r"\\`", "", text)
    assert unescaped.count("`") % 2 == 0, (
        f"{file.name}: odd number of backticks — unclosed template literal?"
    )


# ── Live preview vs real transcript ────────────────────────────────────
#
# Both render into a container with ID `segList`:
#   fileview.js  — the normal transcript
#   batch.js     — the live preview of the running job
#
# `_resetProgressChrome()` cleared this container at the start of every new
# queue job. Anyone reading a different transcript saw it disappear when
# the running file finished, and had to click the file again.


def _js(name: str) -> str:
    return (_STATIC / "js" / name).read_text(encoding="utf-8")


def test_live_preview_is_marked_as_such():
    """Only this way can it be distinguished from the real transcript."""
    assert 'id="segList" data-live="1"' in _js("batch.js"), (
        "The live preview must mark its segList container"
    )


def test_normal_transcript_does_not_have_mark():
    text = _js("fileview.js")
    assert 'id="segList"' in text, "fileview.js still renders to segList"
    assert "data-live" not in text, "The normal transcript must not be treated as live preview"


def test_reset_clears_only_live_preview():
    """The actual regression protection."""
    text = _js("batch.js")
    i = text.index("function _resetProgressChrome")
    body = text[i : text.index("\n}", i)]
    assert "segList" in body, "Test no longer hits the function"
    assert "dataset.live" in body, (
        "_resetProgressChrome() clears segList again unchecked — so "
        "the transcript the user is reading disappears"
    )
