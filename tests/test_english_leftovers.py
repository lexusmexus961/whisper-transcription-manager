"""No German left in the speaker / LLM / MCP code (English-only rule), and
no German assumed in a default.

A review found a German CLI flag (``llm_server.py --zeigen``), German
identifiers in docstrings and a German log message. The flag is ``--show``
now; this test keeps the words out.

Images & sounds used to describe its stills in German by default, and its
examples were German queries. The shipped defaults are English now — the
default is always English; a library in another language sets its own.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "whisper_tm"
FILES = [
    "llm_server.py",
    "mcp_server.py",
    "core/summarize.py",
    "core/llm/runner.py",
    "core/llm/chunking.py",
    "core/speaker_linkage.py",
    "core/chat.py",
    "conversations/__init__.py",
    "conversations/metrics.py",
    "conversations/store.py",
    "conversations/runner.py",
    "conversations/query.py",
    "api/conversations.py",
]
WORDS = re.compile(
    r"\b(zeigen|startbefehl|Segmente|verarbeiter|LlmFehler|"
    r"Hereinbringen|ende|zieht|nicht|jeder)\b"
)


@pytest.mark.parametrize("name", FILES)
def test_no_german_words(name):
    text = (ROOT / name).read_text(encoding="utf-8")
    assert not WORDS.findall(text), WORDS.findall(text)


def test_show_prints_the_command(monkeypatch, capsys):
    from whisper_tm import llm_server
    from whisper_tm.config import config
    from whisper_tm.core import llama_server

    monkeypatch.setattr(config, "load", lambda *a, **k: None)
    monkeypatch.setattr(
        llama_server, "start_command", lambda: ["/bin/llama-server", "--model", "m.gguf"]
    )
    assert llm_server.main(["--show"]) == 0
    out = capsys.readouterr().out
    assert "/bin/llama-server" in out and "--show" not in out


MEDIA = [
    "media/llm.py",
    "media/search.py",
    "media/indexer.py",
    "media/store.py",
    "media/service.py",
    "media/describe.py",
    "api/media_search.py",
    "static/js/mediasearch.js",
    "static/js/settings_media.js",
]
GERMAN = re.compile(
    r"[äöüÄÖÜß]|\b(Polizei|Nacht|draußen|Hund|bellt|Feuer|Kneipe|Stimmengewirr|Menschenmenge|"
    r"Videospiel|Streit|gesagt|Bild|automat)\b"
)


@pytest.mark.parametrize("name", MEDIA)
def test_no_german_in_images_and_sounds(name):
    text = (ROOT / name).read_text(encoding="utf-8")
    assert not GERMAN.findall(text), GERMAN.findall(text)


def test_the_shipped_defaults_are_english():
    import yaml

    d = yaml.safe_load((ROOT.parent / "config.default.yaml").read_text(encoding="utf-8"))
    assert d["media_search"]["llm_language"] == "English"
    # Detected per recording — not a language of its own, and not German
    assert d["transcription"]["default_language"] == "auto"
    assert d["search"]["stopword_languages"] == ["en"]
    assert d["rag"]["fts_language"] == "english"
    from whisper_tm.media import llm

    assert 'llm_language", "English"' in (ROOT / "media" / "llm.py").read_text(encoding="utf-8")
    assert llm.DESCRIBE_SYSTEM.format(lang="English").count("English") == 1


# Review: German examples in the comments of Find and the situation
# search. German words that are data — matched against German input or model
# output (find/plan.py's kind-of-recording words, retrieve._SUFFIXES,
# analysis.NOTHING_SAID) — stay, each with a comment saying so.
SEARCH = [
    "find/plan.py",
    "find/agent.py",
    "rag/speakers.py",
    "rag/retrieve.py",
    "rag/deep_search.py",
]
EXAMPLES = re.compile(
    r"\b(Faxen|macht|beleidigt|soll das|Kacke|treibt|gespräch|satz|wahn|"
    r"ich habe|Teil 1|wird)\b",
    re.I,
)


@pytest.mark.parametrize("name", SEARCH)
def test_no_german_examples_in_find_and_the_situation_search(name):
    text = (ROOT / name).read_text(encoding="utf-8")
    assert not EXAMPLES.findall(text), EXAMPLES.findall(text)


def test_german_data_says_it_is_data():
    plan = (ROOT / "find" / "plan.py").read_text(encoding="utf-8")
    assert "The\n#: German words are data" in plan
    assert "German endings are data" in (ROOT / "rag" / "retrieve.py").read_text(encoding="utf-8")
    assert "German ones are data" in (ROOT / "core" / "analysis.py").read_text(encoding="utf-8")


#: Files whose German is data: words matched in German transcripts and
#: queries (CONTRIBUTING, "English only")
GERMAN_DATA = {
    "whisper_tm/core/stopwords.py",
    "whisper_tm/static/js/stats.js",
    "whisper_tm/core/postprocessing.py",
    "whisper_tm/static/js/audiodups.js",
    "whisper_tm/find/plan.py",
    "whisper_tm/rag/retrieve.py",
    "whisper_tm/core/analysis.py",
}
FUNCTION_WORDS = re.compile(
    r"\b(und|nicht|wird|werden|oder|auch|noch|wenn|dann|schon|sind|aber|weil|"
    r"keine|bitte|für|über|mit dem|ist das|das ist)\b"
)


def test_no_german_sentences_anywhere_in_the_tree():
    """Not only the listed modules: every tracked text file outside the
    tests and the data files above — UI, docs, scripts, config."""
    import subprocess

    root = ROOT.parent
    tracked = subprocess.run(["git", "ls-files"], cwd=root, capture_output=True, text=True)
    if tracked.returncode:
        pytest.skip("not a git checkout")
    found = []
    for name in tracked.stdout.split():
        if name.startswith("tests/") or name in GERMAN_DATA:
            continue
        try:
            text = (root / name).read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        found += [
            f"{name}:{i}: {line.strip()[:80]}"
            for i, line in enumerate(text.splitlines(), 1)
            if FUNCTION_WORDS.search(line)
        ]
    assert not found, "\n".join(found[:20])
