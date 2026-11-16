"""
Every "Change settings" group of the queue editor hangs its mode bar on an
element of the shared transcription options (txoptions.js). A group whose
anchor id does not exist gets no bar and can never be changed — silently:
the language group pointed at "qeLanguage" while the select is "qeLang".
The UI tests fake the DOM, so this checks the real markup source.
"""

import re
from pathlib import Path

JS = Path(__file__).resolve().parents[1] / "whisper_tm" / "static" / "js"


def test_every_group_anchor_exists_in_the_options_markup():
    editor = (JS / "queue_editor.js").read_text(encoding="utf-8")
    options = (JS / "txoptions.js").read_text(encoding="utf-8")
    anchors = re.findall(r"anchor:\s*'qe([A-Za-z]+)'", editor)
    assert len(anchors) >= 10
    # ids are written as id="${p}X" or passed to a helper as `${p}X`
    ids = set(re.findall(r"\$\{p\}([A-Za-z]+)[\"'`]", options))
    missing = [a for a in anchors if a not in ids]
    assert not missing, f"no element for queue editor anchors: {missing}"
