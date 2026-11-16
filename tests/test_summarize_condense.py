"""The reduce step's pre-condensing always merges parts.

Previously, with an unknown context (8192 fallback — OpenAI reports
none) the group target was 2048 tokens, and map parts of more than about
1024 tokens never fit two to a group: every "pair" was a single part sent
alone — one paid LLM call per part, per round, for up to four rounds,
without the list getting any shorter.
"""

from __future__ import annotations

from whisper_tm.config import config
from whisper_tm.core import llm_client
from whisper_tm.core import summarize as s


def _part(i: int) -> str:
    return f"Part {i} (0:00:00–0:10:00):\n" + ("word " * 1100)


def test_every_condense_call_merges_at_least_two_parts(monkeypatch):
    sent: list[str] = []

    def chat(messages, **kw):
        sent.append(messages[-1]["content"])
        return "Part (merged) " + "word " * 900, {}  # still long: rounds go on

    monkeypatch.setattr(llm_client, "chat", chat)
    monkeypatch.setattr(llm_client, "effective_context", lambda task="": 0)
    monkeypatch.setitem(config._data, "llm", {"max_tokens": 4096})

    parts = [_part(i) for i in range(1, 7)]
    out = s._condense_parts(parts, model="", map_max=1200)
    assert sent, "the parts exceed the budget — something must be condensed"
    for text in sent:
        assert text.count("Part ") >= 2, "a single part was condensed alone"
    assert len(out) < len(parts)
