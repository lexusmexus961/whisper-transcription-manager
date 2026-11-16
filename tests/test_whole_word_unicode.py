"""Find & replace's 'Whole word' agrees between the page's preview and the
server, in any script: JavaScript's \\b knows ASCII letters only."""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent / "whisper_tm"
CASES = [
    ("über", "Ist das über alles? Über!", 2),
    ("Москва", "Москва — столица", 1),
    ("Stra", "Straße", 0),
    ("C++", "we use C++ here", 1),
]


def test_the_server_rule():
    for term, text, n in CASES:
        assert len(re.findall(r"(?<!\w)" + re.escape(term) + r"(?!\w)", text, re.I)) == n


@pytest.mark.skipif(shutil.which("node") is None, reason="no node")
def test_the_preview_counts_the_same():
    src = (ROOT / "static/js/editor.js").read_text(encoding="utf-8")
    assert "(?<![\\\\p{L}\\\\p{N}_])${escaped}(?![\\\\p{L}\\\\p{N}_])" in src and "'giu'" in src
    js = (
        'const mk=(t)=>{const e=t.replace(/[.*+?^${}()|[\\]\\\\]/g,"\\\\$&");'
        'return new RegExp(`(?<![\\\\p{L}\\\\p{N}_])${e}(?![\\\\p{L}\\\\p{N}_])`,"giu")};'
        "const cases=JSON.parse(process.argv[1]);"
        "console.log(JSON.stringify(cases.map(([t,s])=>(s.match(mk(t))||[]).length)));"
    )
    import json

    out = subprocess.run(
        ["node", "-e", js, json.dumps([[t, s] for t, s, _ in CASES])],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert json.loads(out) == [n for _, _, n in CASES]
