"""The watch buttons carry the folder's path in their click handler.

They built it with JSON.stringify and replaced only the double quotes: the
ampersand stayed, so a folder whose name holds the text ``&quot;`` gave the
browser a real quote, and what followed ran as script on one click. A
harmless ``R&amp;D`` was sent to the server as ``R&D``.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

JS = Path(__file__).resolve().parent.parent / "whisper_tm" / "static" / "js"

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node not available")

NAMES = [
    "/lib/plain",
    "/lib/R&amp;D",
    "/lib/a&quot;+alert(document.domain)+&quot;",
    "/lib/it's",
    'lib/say "hi"',
    "/lib/a&#39;);alert(1);//",
    "/lib/x&#10;y",
    "/lib/back\\slash",
    "/lib/line\nbreak",
    "/lib/</button><img src=x onerror=alert(1)>",
    "/lib/한국어 회의",
    "/lib/${alert(1)}",
    "/lib/`tick`",
]

HARNESS = r"""
const fs = require('fs');
const state = fs.readFileSync(process.env.JS + '/state.js', 'utf8');
const auto = fs.readFileSync(process.env.JS + '/autoscan.js', 'utf8');
function fn(src, name) {
    const at = src.indexOf('function ' + name + '(');
    return src.slice(at, src.indexOf('\n}\n', at) + 3);
}
eval(fn(state, 'esc') + fn(state, 'escJs'));
const at = auto.indexOf('const Autoscan = {');
const Autoscan = eval('(' + auto.slice(at + 'const Autoscan = '.length, auto.indexOf('\n};', at) + 2) + ')');
// What a browser does with an attribute value before it runs the handler
const decode = s => s.replace(/&(#x[0-9a-f]+|#\d+|amp|lt|gt|quot|apos);/gi, (m, e) => {
    e = e.toLowerCase();
    if (e[0] === '#') return String.fromCodePoint(e[1] === 'x' ? parseInt(e.slice(2), 16) : parseInt(e.slice(1), 10));
    return {amp: '&', lt: '<', gt: '>', quot: '"', apos: "'"}[e];
});
const out = [];
for (const name of JSON.parse(process.env.NAMES)) {
    for (const watched of ['self', '']) {
        const html = Autoscan.watchButton(name, watched);
        const m = /onclick="([^"]*)"/.exec(html);
        const calls = [], alerts = [];
        let error = '';
        try {
            new Function('autoscanSet', 'alert', 'document', decode(m[1]))(a => calls.push(a), a => alerts.push(a), {domain: 'd'});
        } catch (e) { error = String(e); }
        // The tag must not end early either
        out.push({name, calls, alerts, error, tags: (html.match(/<button/g) || []).length, img: /<img/.test(html)});
    }
}
console.log(JSON.stringify(out));
"""


def test_a_folder_name_reaches_the_handler_as_itself_and_runs_nothing():
    import os

    done = subprocess.run(
        ["node", "-e", HARNESS],
        capture_output=True,
        text=True,
        timeout=60,
        env={**os.environ, "JS": str(JS), "NAMES": json.dumps(NAMES)},
    )
    assert done.returncode == 0, done.stderr[-800:]
    rows = json.loads(done.stdout)
    assert len(rows) == 2 * len(NAMES)
    for row in rows:
        assert not row["error"] and not row["alerts"], row
        assert row["tags"] == 1 and not row["img"], row
        assert len(row["calls"]) == 1, row
        (call,) = row["calls"]
        assert list(call.values()) == [row["name"]], row


def test_no_handler_of_the_file_builds_its_argument_with_json_stringify():
    src = (JS / "autoscan.js").read_text(encoding="utf-8")
    assert "JSON.stringify(here)" not in src and "JSON.stringify(abs)" not in src
