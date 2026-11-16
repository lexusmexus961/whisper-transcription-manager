"""The correction prompt speaks one language.

The format contract (prompts.CORRECTION_FORMAT) tells the model that lines
under "CONTEXT" are for understanding only — but the user prompt called
that heading "KONTEXT (nicht editierbar)", and the segment heading was
German too, in an earlier version. A model that is told to leave "CONTEXT"
alone and then finds no such heading has to guess.
"""

from whisper_tm.core import llm_client
from whisper_tm.core.llm import correction, prompts


def _run(monkeypatch, segs):
    monkeypatch.setattr(llm_client, "effective_context", lambda *a: 8192)
    calls = []

    def fake_chat_json(messages, **kw):
        calls.append(messages)
        return {"edits": [], "checked_ok": []}

    monkeypatch.setattr(llm_client, "chat_json", fake_chat_json)
    correction.build_suggestions(segs)
    return [m[-1]["content"] for m in calls]


def _segment(text, score=0.95):
    return {
        "start": 0.0,
        "end": 3.0,
        "text": text,
        "avg_logprob": -0.1,
        "words": [{"word": w, "score": score} for w in text.split()],
    }


def test_the_headings_are_the_ones_the_contract_names(monkeypatch):
    segs = [
        _segment("The meeting starts at ten."),
        _segment("We meet on Wendsday.", score=0.2),
        _segment("Bring the slides."),
    ]
    users = _run(monkeypatch, segs)
    assert users, "no request was made"
    text = "\n".join(users)
    assert "Segments ([id] = editable, (id) = context only):" in text
    assert "KONTEXT" not in text and "Segmente" not in text
    assert '"CONTEXT"' in prompts.CORRECTION_FORMAT


def test_the_context_heading_when_there_is_context(monkeypatch):
    # Enough uncertain segments to split into parts: a later part gets the
    # previous part's last lines as context
    segs = [_segment(f"Line number {i} is uncertain here.", score=0.2) for i in range(400)]
    users = _run(monkeypatch, segs)
    with_context = [u for u in users if "CONTEXT (not editable):" in u]
    assert len(users) > 1 and with_context, len(users)


def test_the_translation_example_is_not_german():
    assert "German →" not in prompts.TRANSLATION_SYSTEM_TEMPLATE
