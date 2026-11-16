"""The export page is the one way out: every way in opens it, and the old
folder-export dialog and the transcript's export menu are gone."""

from __future__ import annotations

from pathlib import Path

STATIC = Path(__file__).resolve().parent.parent / "whisper_tm" / "static"


def _read(rel):
    return (STATIC / rel).read_text(encoding="utf-8")


def test_the_page_is_there_and_the_old_ways_are_gone():
    html = _read("index.html")
    assert 'id="exportOverlay"' in html and "export.css" in html
    assert "folderExportModal" not in html and "openFolderExport" not in html
    js = "".join(_read(f"js/{n}") for n in ("recording.js", "library.js", "export.js"))
    for old in ("exportTXT", "exportSRT", "exportJSON", "openFolderExport", 'rvExportMenu" role'):
        assert old not in js, old


def test_every_way_in_opens_the_same_page():
    html = _read("index.html")
    assert "Export.show({folder:" in html, "the folder bar"
    assert "Export.show({files: [...selectedFiles]})" in html, "the selection footer"
    assert "Export.show({file:" in _read("js/recording.js"), "the recording header"
    assert "Export.show({library: rel})" in _read("js/library.js"), "the Library"


def test_the_shell_knows_the_export_place():
    shell = _read("js/shell.js")
    assert "'exportOverlay'" in shell and "place = 'export'" in shell and "closeExport" in shell


def test_the_download_name_is_read_in_any_script():
    """The server sends the name in the header's UTF-8 form; the page read
    only filename="…" and called everything else "export"."""
    import json
    import shutil
    import subprocess

    import pytest

    if shutil.which("node") is None:
        pytest.skip("node not available")
    src = _read("js/state.js")
    at = src.index("function dispositionName(")
    fn = src[at : src.index("\n}\n", at) + 3]
    cases = {
        "attachment; filename=a.srt": "a.srt",
        'attachment; filename="two words.txt"': "two words.txt",
        'attachment; filename="say \\"hi\\".txt"': 'say "hi".txt',
        "attachment; filename=\"_____.txt\"; filename*=UTF-8''%ED%95%9C%EA%B5%AD%EC%96%B4%20%ED%9A%8C%EC%9D%98.txt": "한국어 회의.txt",
        "attachment; filename*=UTF-8''K%C3%B6ln%20%E2%80%93%20Teil%201.txt": "Köln – Teil 1.txt",
        "attachment": "",
        "": "",
    }
    done = subprocess.run(
        [
            "node",
            "-e",
            fn
            + "\nconsole.log(JSON.stringify("
            + json.dumps(list(cases))
            + ".map(dispositionName)))",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout) == list(cases.values())
    assert "dispositionName(" in _read("js/export.js")


def test_stored_options_are_fitted_before_they_are_used():
    """The server keeps export options and presets as free-form blocks; the
    page wrote three of their values into its markup and called methods on
    the others. A damaged value ran as markup or made the page throw."""
    import json
    import shutil
    import subprocess

    import pytest

    if shutil.which("node") is None:
        pytest.skip("node not available")
    src = _read("js/export.js")
    at = src.index("function fitted(")
    fn = src[at : src.index("\n    }\n", at) + 7]
    body = (
        fn
        + """
const D = {format: 'mp4', size: 'original', subtitles: false, speakers: [], sub_cpl: 42};
const out = [
  fitted(D, {format: 5, size: 'x'.repeat(500), subtitles: 'yes', speakers: 'Alice', sub_cpl: '<img src=x onerror=alert(1)>'}),
  fitted(D, {format: 'webm', subtitles: true, speakers: ['Alice', 7, 'Bob'], sub_cpl: 1/0, extra: 1}),
  fitted(D, null), fitted(D, 'nope'), fitted(D, [1, 2]),
];
console.log(JSON.stringify(out));
"""
    )
    done = subprocess.run(["node", "-e", body], capture_output=True, text=True, timeout=30)
    assert done.returncode == 0, done.stderr
    default = {
        "format": "mp4",
        "size": "original",
        "subtitles": False,
        "speakers": [],
        "sub_cpl": 42,
    }
    got = json.loads(done.stdout)
    assert got[0] == default, "nothing of the wrong kind is taken"
    assert got[1] == {**default, "format": "webm", "subtitles": True, "speakers": ["Alice", "Bob"]}
    assert got[2] == got[3] == got[4] == default
    assert src.count("fitted(OPT_DEFAULTS") == 2 and src.count("fitted(VIDEO_DEFAULTS") == 2
    assert 'value="${opts.sub_cpl}"' not in src and 'value="${Number(opts.sub_cpl) || 42}"' in src
