"""Settings panel: what Save collects, what the pages draw, and how the
settings search matches.

Save: the directory fields were collected with a document-wide
``[data-dir]`` query. The queue editor's folder rows carry that attribute
too — ``<div>``s without a value — so once the editor had listed folders,
every save threw "Cannot read properties of undefined (reading 'trim')"
and nothing was saved (Low-VRAM mode could not be switched off). Since
the settings redesign every setting — folders too —
is one ``data-key`` control inside the dialog, Save sends only the rows
whose value changed, and a value keeps its type: a <select> hands out
strings, and the string "false" is true in Python (the embedding mode could
never be switched back).

Pages: every row edits a key that exists in config.default.yaml, and each
key has exactly one place.

Search: pure match logic, run in node without a DOM — word starts, words
written together, config keys, and the three tiers of ``_ssPick``.
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
HARNESS = REPO / "tests" / "js" / "settings_harness.js"
JS = REPO / "whisper_tm" / "static" / "js" / "settings.js"
PAGES = REPO / "whisper_tm" / "static" / "js" / "settings_pages.js"
INDEX = REPO / "whisper_tm" / "static" / "index.html"

node = pytest.mark.skipif(shutil.which("node") is None, reason="node not available")


def js(body: str, setup: str = "buildIt()", pages: bool = False) -> dict:
    source = (
        f"const {{buildIt, element}} = require({str(HARNESS)!r});\n"
        f"const h = {setup};\n"
        "const output = (o) => console.log('§§' + JSON.stringify(o));\n"
        "(async () => {\n" + body + "\n})().catch(e => { console.error(e); process.exit(1); });\n"
    )
    env = {"SETTINGS_JS": str(JS), "PATH": os.environ.get("PATH", "")}
    if pages:
        env["SETTINGS_EXTRA"] = str(PAGES)
    done = subprocess.run(
        ["node", "-e", source], capture_output=True, text=True, timeout=60, env=env
    )
    if done.returncode != 0:
        raise AssertionError(f"node failed:\n{done.stderr}\n{done.stdout}")
    lines = [ln for ln in done.stdout.splitlines() if ln.startswith("§§")]
    assert lines, f"no result:\n{done.stdout}\n{done.stderr}"
    return json.loads(lines[-1][2:])


# The dialog: one switch, one folder field. Outside it: a folder row of
# the queue editor, rendered exactly like queue_editor.js does it.
DIALOG_AND_EDITOR = """buildIt({
    inside: [
        element('input', {'data-key': 'hardware.low_vram', 'data-type': 'bool'}, {type: 'checkbox', checked: false}),
        element('input', {'data-key': 'directories.work', 'data-type': 'str'}, {type: 'text', value: ' /mnt/work '}),
    ],
    outside: [
        element('div', {'class': 'qe-folder-row', 'data-prefix': 'Archive/', 'data-dir': '/mnt/media'}),
    ],
})"""


@node
class TestSaveCollectsOnlyTheDialog:
    def test_a_queue_editor_folder_row_does_not_break_the_save(self):
        r = js("output(h.value('_collectFormValues()'));", setup=DIALOG_AND_EDITOR)
        assert r == {"hardware": {"low_vram": False}, "directories": {"work": "/mnt/work"}}

    def test_the_folder_row_really_is_what_broke_it(self):
        # The fixture reproduces the bug: the old document-wide loop
        # throws exactly the reported error
        r = js(
            """
            let error = null;
            try {
                h.run(`document.querySelectorAll('[data-dir]').forEach(el => el.value.trim())`);
            } catch (e) { error = e.message; }
            output({error});
        """,
            setup=DIALOG_AND_EDITOR,
        )
        assert r["error"] == "Cannot read properties of undefined (reading 'trim')"

    def test_the_old_folder_attribute_is_not_collected_any_more(self):
        # Nothing reads data-dir now — the editor's rows can't reach Save
        source = JS.read_text(encoding="utf-8")
        assert "data-dir" not in source.split("function _collectFormValues")[1].split("\n}\n")[0]


def typed(tag: str, dtype: str, props: str) -> dict:
    """What _sRead makes of one control."""
    return js(f"""h.sandbox.__el = element('{tag}', {{'data-type': '{dtype}'}}, {props});
                 output({{v: h.value('_sRead(__el)')}});""")["v"]


@node
class TestValuesKeepTheirType:
    def test_a_select_of_booleans_sends_a_boolean(self):
        assert typed("select", "bool", "{value: 'false'}") is False
        assert typed("select", "bool", "{value: 'true'}") is True

    def test_numbers(self):
        assert typed("input", "num", "{type: 'number', value: '8000'}") == 8000
        assert typed("input", "num", "{type: 'number', value: '0.55'}") == 0.55

    def test_an_empty_or_broken_number_is_not_saved(self):
        # undefined → dropped from the save: a 0 would silently cripple a limit
        r = js("""h.sandbox.__a = element('input', {'data-type': 'num'}, {type: 'number', value: ''});
                  h.sandbox.__b = element('input', {'data-type': 'num'}, {type: 'number', value: 'x'});
                  output({empty: h.value('_sRead(__a) === undefined'),
                          broken: h.value('_sRead(__b) === undefined')});""")
        assert r == {"empty": True, "broken": True}

    def test_an_empty_number_is_automatic_where_that_is_the_default(self):
        """An empty default means "automatic" (the memory warning): emptied,
        such a field saves as empty — else it could never go back to it."""
        r = js("""h.sandbox.__a = element('input', {'data-type': 'num'},
                      {type: 'number', value: '', closest: () => ({dataset: {def: '""'}})});
                  h.sandbox.__b = element('input', {'data-type': 'num'},
                      {type: 'number', value: '', closest: () => ({dataset: {def: '8000'}})});
                  output({auto: h.value('_sRead(__a)'), limit: h.value('_sRead(__b) === undefined')});""")
        assert r == {"auto": "", "limit": True}

    def test_an_empty_automatic_number_is_its_default(self):
        r = js("""h.sandbox.__a = element('input', {'data-type': 'num', 'data-initial': '""'}, {type: 'number'});
                  h.sandbox.__b = element('input', {'data-type': 'num', 'data-initial': '20'}, {type: 'number'});
                  output({same: h.value("_sSameAsDefault(__a, '')"),
                          set: h.value("_sSameAsDefault(__b, '')")});""")
        assert r == {"same": True, "set": False}

    def test_a_number_stays_inside_its_bounds(self):
        assert typed("input", "num", "{type: 'number', value: '99', max: '8', min: '0'}") == 8

    def test_lists(self):
        assert typed("input", "list", "{value: ' .rmvb, .vob ,, '}") == [".rmvb", ".vob"]
        assert typed("textarea", "lines", "{value: '.*\\n@eaDir\\n\\n'}") == [".*", "@eaDir"]

    def test_text_fields_are_trimmed_text_areas_are_not(self):
        assert typed("input", "str", "{value: '  /mnt/media  '}") == "/mnt/media"
        assert typed("textarea", "str", "{value: '  a prompt\\n'}") == "  a prompt\n"


@node
class TestSaveSendsOnlyWhatChanged:
    def test_a_row_as_drawn_is_left_out(self):
        r = js("""
            const same = element('input', {'data-key': 'search.max_results', 'data-type': 'num', 'data-initial': '1000'}, {type: 'number', value: '1000'});
            const changed = element('input', {'data-key': 'search.context_segments', 'data-type': 'num', 'data-initial': '2'}, {type: 'number', value: '3'});
            const h2 = buildIt({inside: [same, changed]});
            output(h2.value('_collectFormValues()'));
        """)
        assert r == {"search": {"context_segments": 3}}


def _default_keys() -> set[str]:
    import yaml

    data = yaml.safe_load((REPO / "config.default.yaml").read_text(encoding="utf-8"))
    keys = set()

    def walk(node, path):
        if isinstance(node, dict) and node:
            for k, v in node.items():
                walk(v, path + [str(k)])
        else:
            keys.add(".".join(path))

    walk(data, [])
    return keys


@node
class TestPages:
    """Every page drawn with the shipped defaults, as the browser would."""

    @pytest.fixture(scope="class")
    @classmethod
    def drawn(cls):
        import yaml

        defaults = yaml.safe_load((REPO / "config.default.yaml").read_text(encoding="utf-8"))
        return js(
            f"""
            h.sandbox.__defaults = {json.dumps(defaults)};
            h.run(`_settingsData = {{config: __defaults, defaults: __defaults,
                                    dirs: {{media: '/m', work: '/w'}}, db_path: '/d.db'}}`);
            h.run(`_hwData = {{device: 'cuda', gpus: [], recommendations: {{compute_type: 'float16'}}}}`);
            output(h.value(`SETTINGS_PAGES.map(p => ({{id: p.id, group: p.group, title: p.title,
                intro: p.intro || '', html: p.render(_settingsData.config)}}))`));
        """,
            pages=True,
        )

    def test_every_page_says_what_it_is_for(self, drawn):
        assert len(drawn) >= 15
        for page in drawn:
            assert page["id"] and page["group"] and page["title"], page["id"]
            assert len(page["intro"]) > 30, f"{page['id']}: no intro"

    def test_every_row_edits_a_real_setting(self, drawn):
        known = _default_keys()
        controls = [
            (page["id"], tag, key)
            for page in drawn
            for tag, key in re.findall(r'<(\w+)[^<>]*?\sdata-key="([^"]+)"', page["html"])
        ]
        assert len(controls) > 100
        assert {tag for _, tag, _ in controls} <= {"input", "select", "textarea"}
        unknown = sorted({(pid, key) for pid, _, key in controls if key not in known})
        assert not unknown, f"not in config.default.yaml: {unknown}"

    def test_every_setting_has_one_place(self, drawn):
        seen = {}
        for page in drawn:
            for key in re.findall(r'\sdata-key="([^"]+)"', page["html"]):
                seen.setdefault(key, []).append(page["id"])
        twice = {k: v for k, v in seen.items() if len(v) > 1}
        assert not twice, twice

    def test_search_and_ai_pages(self, drawn):
        # Search with the semantic search in it, the language
        # models and the embedding model together as AI models, what they
        # do as AI tasks, and the MCP server under its own name — instead
        # of six pages that each held a piece of the same topics.
        # Images & sounds sits next to Search — its own index, its
        # own environment, its own run — and the audio analysis next to it
        ids = [p["id"] for p in drawn if p["group"] == "Search & AI"]
        assert ids == ["search", "media", "audio", "models", "tasks", "mcp"]
        mcp = next(p for p in drawn if p["id"] == "mcp")
        assert mcp["title"] == "MCP server"

    def test_sections_can_be_jumped_to(self, drawn):
        ids = [i for p in drawn for i in re.findall(r'class="set-section" id="([^"]+)"', p["html"])]
        assert len(ids) == len(set(ids)), ids
        for want in (
            "set-sec-keyword",
            "set-sec-semantic",
            "set-sec-llm",
            "set-sec-embedding",
            "set-sec-summaries",
            "set-sec-correction",
            "set-sec-translation",
        ):
            assert want in ids, want

    def test_every_page_icon_is_in_the_sprite(self):
        icons = set(re.findall(r"icon: '([\w-]+)'", PAGES.read_text(encoding="utf-8")))
        symbols = set(re.findall(r'<symbol id="i-([\w-]+)"', INDEX.read_text(encoding="utf-8")))
        assert len(icons) >= 16 and icons <= symbols, icons - symbols

    def test_every_row_says_what_it_does(self, drawn):
        # A named row without a description — every setting that is
        # shown needs a good description
        bare = []
        for page in drawn:
            for row in re.findall(
                r'<div class="set-row[^"]*"[^>]*data-row="([^"]+)"[^>]*>(.*?)<div class="set-ctl">',
                page["html"],
                re.S,
            ):
                key, text = row
                if 'class="set-desc"' not in text:
                    bare.append(key)
        assert not bare, bare


def match(body: str) -> dict:
    return js(body)


@node
class TestSearchMatching:
    def test_query_words(self):
        r = match("""output({
            words: h.value("_ssTerms('  Low_VRAM   mode ')"),
            hyphen: h.value("_ssTerms('low-vram')"),
            dupes: h.value("_ssTerms('vram VRAM')"),
            blank: h.value("_ssTerms('   ')"),
        });""")
        assert r == {
            "words": ["low", "vram", "mode"],
            "hyphen": ["low", "vram"],
            "dupes": ["vram"],
            "blank": [],
        }

    @pytest.mark.parametrize(
        "text, term, expected",
        [
            ("Port for the OpenAPI server", "port", True),
            ("Export as CSV, support bundle", "port", False),  # not inside a word
            ("164 Logs", "log", True),
            ("Downloads from the Add dialog", "log", False),
            ("Low-VRAM mode", "vram", True),
            ("Low-VRAM mode", "lowvram", True),  # written together
            ("Word-Level Timestamps", "wordtimestamps", False),  # "level" in between
            ("Word-Level Timestamps", "wordlevel", True),
            ("keep a full model loaded", "llm", False),  # not across a gap
            ("Retries on network errors (5xx/timeout)", "timeout", True),
            ("At least 900 s; 0.5 allows up to 37x", "0.5", True),  # punctuation: as is
            ("hardware.low_vram", "hardware.low", True),
        ],
    )
    def test_word_starts(self, text, term, expected):
        r = match(
            f"output({{hit: h.value('_ssHas(_ssHay({json.dumps(text)}), {json.dumps(term)})')}});"
        )
        assert r["hit"] is expected

    def test_all_words_of_the_row_win_over_a_heading_word(self):
        # Every LLM row has "llm" in its config key; the group title says
        # "timeouts". Only the rows that name the timeout themselves are hits.
        r = match("""output(h.value(`_ssPick('Language Model (LLM) Limits and timeouts', [
            'Timeout (s) Read timeout per LLM call llm.timeout',
            'Retries on network/server errors (5xx/timeout) llm.max_retries',
            'Temperature 0.0 = deterministic llm.temperature',
        ], _ssTerms('llm timeout'))`));""")
        assert r == [True, True, False]

    def test_the_heading_supplies_the_missing_word(self):
        r = match("""output(h.value(`_ssPick('Transcription HuggingFace', [
            'HF Token For the pyannote diarization model transcription.hf_token',
            'Something unrelated',
        ], _ssTerms('huggingface token'))`));""")
        assert r == [True, False]

    def test_a_heading_match_alone_shows_the_whole_group(self):
        r = match("""output(h.value(`_ssPick('Transcription HuggingFace', [
            'HF Token For the pyannote diarization model',
            'Mirror endpoint',
        ], _ssTerms('huggingface'))`));""")
        assert r == [True, True]

    def test_no_match_hides_the_group(self):
        r = match("""output(h.value(`_ssPick('Hardware Compute', [
            'Device', 'GPU index',
        ], _ssTerms('xyzzy'))`));""")
        assert r == [False, False]


@node
class TestOldPageNames:
    """Other parts of the app — and old bookmarks in code — open pages by
    name. Pages that became a section of another page land on it."""

    @pytest.mark.parametrize(
        "name, page, section",
        [
            ("embedding", "models", "embedding"),
            ("llm", "models", "llm"),
            ("rag", "search", "semantic"),
            ("summaries", "tasks", "summaries"),
            ("correction", "tasks", "correction"),
            ("translation", "tasks", "translation"),
            ("ai_clients", "mcp", ""),
            ("mcp", "mcp", ""),
            ("folders", "directories", ""),
            ("models#embedding", "models", "embedding"),
            ("no such page", "directories", ""),
        ],
    )
    def test_where_a_name_lands(self, name, page, section):
        r = js(f"output(h.value(`_settingsTarget({json.dumps(name)})`));", pages=True)
        assert r == {"page": page, "section": section}


class TestMarkup:
    def test_the_search_box_is_wired(self):
        html = INDEX.read_text(encoding="utf-8")
        box = re.search(r'<input[^>]*id="settingsSearch"[^>]*>', html, re.S)
        assert box, "search box missing in the settings nav"
        assert 'oninput="settingsSearch(this.value)"' in box.group(0)
        assert 'onkeydown="_settingsSearchKey(event)"' in box.group(0)
        assert 'id="settingsSearchClear"' in html

    def test_help_and_guide_share_the_settings_shell(self):
        # The same panel, head bar and navigation as the settings — the
        # help and the setup guide were the last ones in the old look
        html = INDEX.read_text(encoding="utf-8")
        for overlay, parts in (
            ("helpOverlay", ("set-panel", "set-head", 'id="helpContent"')),
            ("setupOverlay", ("set-panel", "set-head", "set-nav", 'id="sgNav"')),
        ):
            i = html.index(f'id="{overlay}"')
            block = html[i : html.index("<!--", i)]
            for part in parts:
                assert part in block, (overlay, part)
        assert "settings-panel" not in html and "settings-title-bar" not in html
        for name in ("help.js", "setupguide.js"):
            text = (REPO / "whisper_tm" / "static" / "js" / name).read_text(encoding="utf-8")
            assert "s-group" not in text and "s-btn" not in text, name

    @node
    def test_the_handlers_exist(self):
        r = js("""output({
            search: h.value('typeof settingsSearch'),
            key: h.value('typeof _settingsSearchKey'),
            reset: h.value('typeof _settingsSearchReset'),
            shortcut: (h.listeners.keydown || []).length,
        });""")
        # Two keydown listeners: Escape closes the panel, "/" and Ctrl+F
        # jump into the search
        assert r == {"search": "function", "key": "function", "reset": "function", "shortcut": 2}
