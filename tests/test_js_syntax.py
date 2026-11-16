"""Every JavaScript file must be syntactically intact.

Twice the same pitfall: In `voicemap_assign.js` and then in
`fileview.js` were comments with backticks — in the middle of a template
string. A backtick ends the string, and the remainder of the file is nonsense.
The second time, the **entire file view** failed: no transcript, no segments,
only an empty area and ``SyntaxError: Unexpected identifier 'nowrap'`` in the
console.

The whole test suite ran through green. They check the CONTENT of files as text
— whether a button is wired, whether a function exists — but nobody ever asked
if the browser could read the file at all.

`node --check` answers exactly this question, in milliseconds per file. This
does not replace visual inspection or browser testing; it catches the kind of
error that makes everything else worthless.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
JS_DIR = REPO / "whisper_tm" / "static" / "js"
# `vendor/` is left out: foreign, often minified code we do not maintain —
# its state says nothing about our work.
FILES = sorted(JS_DIR.glob("*.js"))
NODE = shutil.which("node")


def test_there_are_files_to_check_at_all():
    """An empty glob would make the check below worthless."""
    assert len(FILES) >= 15, [d.name for d in FILES]


@pytest.mark.skipif(not NODE, reason="node not installed")
@pytest.mark.parametrize("file", FILES, ids=lambda d: d.name)
def test_the_file_is_syntactically_intact(file: Path):
    done = subprocess.run([NODE, "--check", str(file)], capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, f"{file.name} is unreadable to the browser:\n{done.stderr}"


@pytest.mark.skipif(not NODE, reason="node not installed")
def test_the_check_really_fails_on_broken_code(tmp_path):
    """A test that never triggers is no test. This reproduces exactly the
    error described above: a backtick inside a comment of a template string."""
    broken = tmp_path / "broken.js"
    broken.write_text(
        "function f() {\n  return `<div>\n     <!-- the column is `nowrap` -->\n  </div>`;\n}\n",
        encoding="utf-8",
    )
    done = subprocess.run(
        [NODE, "--check", str(broken)], capture_output=True, text=True, timeout=60
    )
    assert done.returncode != 0, "node lets broken code through"
