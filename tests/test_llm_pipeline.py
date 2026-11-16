"""
Tests of the LLM pipeline (docs/LLM-PIPELINE.md): budget, chunker, JSON extractor,
correction validation, translation loop, map-reduce switch.

The LLM is never actually called — chat/chat_json are monkeypatched
(convention like test_summary_auto.py: plain monkeypatch, no mock framework).
"""

import pytest

from whisper_tm.core import llm_client
from whisper_tm.core.llm import budget, chunking, correction, translation
from whisper_tm.core.llm.runner import run_chunks

# ══════════════════════════════════════════════════════════════════════════
#  budget
# ══════════════════════════════════════════════════════════════════════════


def test_estimating_tokens_for_latin():
    # ~4 characters per token
    text = "a" * 400
    assert 90 <= budget.estimate_tokens(text) <= 110


def test_estimating_tokens_for_cjk():
    # ~1.5 characters per token
    text = "今" * 150
    assert 90 <= budget.estimate_tokens(text) <= 110


def test_estimate_tokens_on_empty_input():
    assert budget.estimate_tokens("") == 0


def test_the_chunk_budget_arithmetic():
    b = budget.chunk_budget(
        system_tokens=500, instruction_tokens=100, answer_reserve=1000, context=10000, margin=0.10
    )
    assert b == 9000 - 500 - 100 - 1000


def test_chunk_budget_fallback_and_lower_bound():
    # Unknown context → fallback; pathologically small budget → 512
    b = budget.chunk_budget(system_tokens=budget.CONTEXT_FALLBACK, context=0)
    assert b == 512


# ══════════════════════════════════════════════════════════════════════════
#  chunking
# ══════════════════════════════════════════════════════════════════════════


def _segs(n, text="Hello World, this is a test segment with some content.", speaker="SPEAKER_00"):
    return [
        {"start": float(i * 5), "end": float(i * 5 + 4), "text": text, "speaker": speaker}
        for i in range(n)
    ]


def test_chunker_keeps_budget():
    segs = _segs(50)
    lines = [f"[{i}] {s['text']}" for i, s in enumerate(segs)]
    per_line = budget.estimate_tokens(lines[0]) + 1
    chunks = chunking.split_segments(segs, lines, budget_tokens=per_line * 10)
    assert len(chunks) >= 5
    for c in chunks:
        assert c.tokens <= per_line * 10
    # Continuous and ordered
    assert chunks[0].from_seg == 0
    assert chunks[-1].to_seg == 49
    for a, b in zip(chunks, chunks[1:]):
        assert b.from_seg == a.to_seg + 1


def test_the_chunker_puts_a_giant_segment_alone():
    segs = _segs(3)
    lines = ["short", "x" * 8000, "short"]
    chunks = chunking.split_segments(segs, lines, budget_tokens=100)
    # The giant segment exceeds the budget, but is allowed through as its own chunk
    # instead of creating an endless loop.
    assert any(c.from_seg == c.to_seg == 1 for c in chunks)


def test_chunker_overlap_is_read_context():
    segs = _segs(20)
    lines = [f"[{i}] {s['text']}" for i, s in enumerate(segs)]
    per_line = budget.estimate_tokens(lines[0]) + 1
    chunks = chunking.split_segments(segs, lines, budget_tokens=per_line * 5, overlap_segments=2)
    assert chunks[0].context_before == []
    second = chunks[1]
    assert len(second.context_before) == 2
    # Context is the last segments BEFORE the chunk
    assert second.context_before[-1][0] == second.from_seg - 1


def test_the_chunker_prefers_a_speaker_change():
    # 10 segments speaker A, then 10 speaker B — budget such that the
    # cut would be just after the switch → backtracking pulls it to
    # the speaker boundary.
    segs = _segs(10, speaker="SPEAKER_00") + _segs(10, speaker="SPEAKER_01")
    for i, s in enumerate(segs):
        s["start"], s["end"] = float(i * 5), float(i * 5 + 4)
    lines = [f"[{i}] {s['text']}" for i, s in enumerate(segs)]
    per_line = budget.estimate_tokens(lines[0]) + 1
    chunks = chunking.split_segments(segs, lines, budget_tokens=per_line * 11)
    assert chunks[0].to_seg == 9  # Cut exactly at speaker switch


def test_chunker_length_mismatch():
    with pytest.raises(ValueError):
        chunking.split_segments(_segs(3), ["a"], 100)


# ══════════════════════════════════════════════════════════════════════════
#  runner
# ══════════════════════════════════════════════════════════════════════════


def test_runner_partial_failure_and_final_retry(monkeypatch):
    monkeypatch.setattr("whisper_tm.core.llm.runner.time.sleep", lambda s: None)
    segs = _segs(6)
    lines = [s["text"] for s in segs]
    chunks = chunking.split_segments(segs, lines, budget_tokens=20)
    assert len(chunks) >= 3
    attempts = {}

    def processor(chunk):
        attempts[chunk.index] = attempts.get(chunk.index, 0) + 1
        if chunk.index == 1 and attempts[chunk.index] < 3:
            raise llm_client.LlmConnectionError("wobble")
        return chunk.index

    results, failed = run_chunks(chunks, processor, attempts_per_chunk=2)
    # Chunk 1 fails 2× in main run, succeeds in end-retry (3rd attempt)
    assert not failed
    assert set(results) == {c.index for c in chunks}


def test_runner_cancel():
    class FakeJob:
        cancel = True

        def update(self, **kw):
            pass

    segs = _segs(4)
    chunks = chunking.split_segments(segs, [s["text"] for s in segs], 20)
    with pytest.raises(InterruptedError):
        run_chunks(chunks, lambda c: c, job=FakeJob())


# ══════════════════════════════════════════════════════════════════════════
#  llm_client: JSON extractor
# ══════════════════════════════════════════════════════════════════════════


def test_extract_json_direct():
    assert llm_client.extract_json('{"a": 1}') == {"a": 1}


def test_extract_json_fence():
    text = 'Here is the result:\n```json\n{"a": [1, 2]}\n```\nDone.'
    assert llm_client.extract_json(text) == {"a": [1, 2]}


def test_extract_json_bracket_balance():
    text = 'Blah blah {"edits": [{"id": 1, "text": "a}b"}]} and more text'
    assert llm_client.extract_json(text) == {"edits": [{"id": 1, "text": "a}b"}]}


def test_extract_json_from_broken_output():
    assert llm_client.extract_json("no json here") is None
    assert llm_client.extract_json("") is None


# ══════════════════════════════════════════════════════════════════════════
#  correction
# ══════════════════════════════════════════════════════════════════════════


def test_mark_uncertain_words():
    seg = {
        "text": "The bugdet is approved.",
        "words": [
            {"word": "The", "score": 0.9},
            {"word": "bugdet", "score": 0.21},
            {"word": "is", "score": 0.95},
            {"word": "approved.", "score": 0.88},
        ],
    }
    out = correction.mark_uncertain_words(seg, 0.4)
    assert out == "The ⟨bugdet|0.21⟩ is approved."


def test_mark_uncertain_words_cjk_without_spaces():
    seg = {
        "text": "今度の事件",
        "words": [
            {"word": "今度", "score": 0.3},
            {"word": "の", "score": 0.9},
            {"word": "事件", "score": 0.9},
        ],
    }
    out = correction.mark_uncertain_words(seg, 0.4)
    assert out == "⟨今度|0.30⟩の事件"


def test_choosing_segments_at_the_thresholds():
    segs = [
        {"text": "ok", "avg_logprob": -0.2, "words": [{"word": "ok", "score": 0.9}]},
        {"text": "uncertain", "avg_logprob": -0.2, "words": [{"word": "uncertain", "score": 0.1}]},
        {"text": "logprob", "avg_logprob": -1.5, "words": []},
    ]
    assert correction.choose_segments(segs, score_threshold=0.4, logprob_threshold=-0.8) == {1, 2}


def _validate(edits, **kw):
    segs = [{"text": "The bugdet is approved."}, {"text": "We meet on Wednesday."}]
    kw.setdefault("min_confidence", 0.5)
    kw.setdefault("max_edit_share", 1.0)
    ok, bad, _best = correction.validate_edits(edits, segs, {0, 1}, **kw)
    return ok, bad


def test_validate_edits_accept():
    ok, bad = _validate(
        [{"id": 0, "text": "The budget is approved.", "reason": "Typo", "confidence": 0.9}]
    )
    assert len(ok) == 1 and not bad
    assert ok[0]["old_value"] == "The bugdet is approved."
    assert ok[0]["new_value"] == "The budget is approved."


def test_validate_edits_rejects():
    ok, bad = _validate(
        [
            {"id": 7, "text": "x", "reason": "", "confidence": 0.9},  # foreign id
            {"id": 0, "text": "", "reason": "", "confidence": 0.9},  # empty
            # identical → now counts as confirmation, not reject
            # (own test below)
            {"id": 1, "text": "Short.", "reason": "", "confidence": 0.3},  # too uncertain
            {
                "id": 1,
                "text": "A completely newly invented paragraph that is ten times "
                "as long as the original and thus certainly no longer "
                "a correction but rather a complete "
                "reformulation of the entire segment.",
                "reason": "",
                "confidence": 0.9,
            },  # ratio
        ]
    )
    assert not ok
    assert len(bad) == 4


def test_validation_discards_an_over_correcting_chunk():
    # Guard only takes effect at 5 editable segments (small chunks: 'all
    # edited' is plausible). Here: 6 segments, 4 edits > 50% → gone.
    segs = [{"text": f"Sentence number {i} here."} for i in range(6)]
    edits = [
        {"id": i, "text": f"Sentence number {i} there.", "reason": "", "confidence": 0.9}
        for i in range(4)
    ]
    ok, bad, _b = correction.validate_edits(
        edits, segs, set(range(6)), min_confidence=0.5, max_edit_share=0.5
    )
    assert not ok
    assert len(bad) == 4
    # Small chunk: 1/1 edited remains allowed
    ok2, _ = _validate(
        [{"id": 0, "text": "The budget is approved.", "reason": "", "confidence": 0.9}],
        max_edit_share=0.5,
    )
    assert len(ok2) == 1


def test_validate_edits_strips_the_markers():
    ok, _ = _validate(
        [{"id": 0, "text": "The ⟨budget|0.21⟩ is approved.", "reason": "", "confidence": 0.9}]
    )
    assert ok[0]["new_value"] == "The budget is approved."


def test_apply_edits():
    tdata = {
        "segments": [
            {"text": "The bugdet is approved.", "words": [{"word": "x"}]},
            {"text": "Stays."},
        ]
    }
    applied = correction.apply_edits_to(
        tdata,
        [
            {"id": 0, "new_value": "The budget is approved."},
            {"id": 5, "new_value": "does not exist"},
        ],
    )
    assert len(applied) == 1
    assert tdata["segments"][0]["text"] == "The budget is approved."
    assert tdata["segments"][0]["edited"] == 1
    assert applied[0]["old_value"] == "The bugdet is approved."
    assert tdata["segments"][1]["text"] == "Stays."


def test_generating_suggestions_end_to_end(monkeypatch):
    """Complete correction run with faked chat_json."""
    monkeypatch.setattr(llm_client, "effective_context", lambda *a: 8192)
    calls = []

    def fake_chat_json(messages, *, schema, **kw):
        calls.append(messages)
        # Only segment 1 'correct'
        return {
            "edits": [
                {"id": 1, "text": "The budget is approved.", "reason": "Typo", "confidence": 0.9}
            ]
        }

    monkeypatch.setattr(llm_client, "chat_json", fake_chat_json)
    segs = [
        {
            "start": 0.0,
            "end": 4.0,
            "text": "All good here.",
            "avg_logprob": -0.1,
            "words": [{"word": "All", "score": 0.9}],
        },
        {
            "start": 5.0,
            "end": 9.0,
            "text": "The bugdet is approved.",
            "avg_logprob": -0.3,
            "words": [
                {"word": "The", "score": 0.95},
                {"word": "bugdet", "score": 0.2},
                {"word": "is", "score": 0.9},
                {"word": "approved.", "score": 0.9},
            ],
        },
    ]
    result = correction.build_suggestions(segs)
    assert len(calls) == 1
    assert len(result["suggestions"]) == 1
    v = result["suggestions"][0]
    assert v["id"] == 1 and v["new_value"] == "The budget is approved."
    assert result["marked_segments"] == 1
    # Transparency payload (UI rebuild): checked segments + run metadata
    assert result["checked"] == [
        {
            "id": 1,
            "text": "The bugdet is approved.",
            "confirmed": False,
            "avg_logprob": -0.3,
            "uncertain_words": 1,
        }
    ]
    assert isinstance(result["discarded"], list)
    assert result["thresholds"]["score"] == 0.4
    assert "duration" in result
    # Prompt contains confidence marking and editability identifier
    user = calls[0][-1]["content"]
    assert "⟨bugdet|0.20⟩" in user
    assert "[1]" in user and "(0)" in user


def test_generating_suggestions_structures_the_rejected_ones(monkeypatch):
    """Rejected edits come back structured (id, old, new, reason) —
    the review UI shows WHAT the LLM wanted and WHY it didn't go through."""
    monkeypatch.setattr(llm_client, "effective_context", lambda *a: 8192)

    def fake_chat_json(messages, *, schema, **kw):
        # Too low self-assessment → validation rejects
        return {
            "edits": [
                {"id": 1, "text": "The budget is approved.", "reason": "guessed", "confidence": 0.2}
            ]
        }

    monkeypatch.setattr(llm_client, "chat_json", fake_chat_json)
    segs = [
        {
            "start": 0.0,
            "end": 4.0,
            "text": "All good.",
            "avg_logprob": -0.1,
            "words": [{"word": "All", "score": 0.9}],
        },
        {
            "start": 5.0,
            "end": 9.0,
            "text": "The bugdet is approved.",
            "avg_logprob": -0.3,
            "words": [
                {"word": "The", "score": 0.95},
                {"word": "bugdet", "score": 0.2},
                {"word": "is", "score": 0.9},
                {"word": "approved.", "score": 0.9},
            ],
        },
    ]
    result = correction.build_suggestions(segs)
    assert result["suggestions"] == []
    assert len(result["discarded"]) == 1
    v = result["discarded"][0]
    assert v["id"] == 1
    assert "confidence" in v["rejected_because"]
    assert v["old_value"] == "The bugdet is approved."


def test_a_confirmation_is_not_a_reject(monkeypatch):
    """action='ok' (and unchanged text) is a CONFIRMATION:
    the LLM has checked and found nothing. Most returns are such
    confirmations — they don't belong in the
    rejected tab, but in 'checked'."""
    monkeypatch.setattr(llm_client, "effective_context", lambda *a: 8192)

    def fake_chat_json(messages, *, schema, **kw):
        return {
            "edits": [
                {"id": 1, "text": "The budget is approved.", "reason": "Typo", "confidence": 0.9},
            ],
            "checked_ok": [0],
        }

    monkeypatch.setattr(llm_client, "chat_json", fake_chat_json)
    segs = [
        {"start": 0.0, "end": 4.0, "text": "All good.", "avg_logprob": -1.2, "words": []},
        {
            "start": 5.0,
            "end": 9.0,
            "text": "The bugdet is approved.",
            "avg_logprob": -1.2,
            "words": [],
        },
    ]
    result = correction.build_suggestions(segs)
    assert len(result["suggestions"]) == 1
    assert result["discarded"] == []  # no reject message
    assert result["confirmed"] == 1
    checked = {g["id"]: g["confirmed"] for g in result["checked"]}
    assert checked == {0: True, 1: False}


def test_generating_suggestions_with_nothing_selected(monkeypatch):
    monkeypatch.setattr(
        llm_client, "chat_json", lambda *a, **k: pytest.fail("no LLM call expected")
    )
    segs = [
        {
            "start": 0,
            "end": 1,
            "text": "Clean.",
            "avg_logprob": -0.1,
            "words": [{"word": "Clean.", "score": 0.99}],
        }
    ]
    result = correction.build_suggestions(segs)
    assert result["suggestions"] == [] and result["chunks"] == 0
    assert result["confidence_data"] is True


def test_build_suggestions_without_scores_checks_everything(monkeypatch):
    """Words WITHOUT score fields (alignment failed)
    counted as confidence data → 0 marked → run after 1 ms 'Done' with
    all zeros. Now: without real scores, the entire transcript is checked from
    the context."""
    monkeypatch.setattr(llm_client, "effective_context", lambda *a: 8192)
    seen = {}

    def fake_chat_json(messages, **kw):
        seen["user"] = messages[-1]["content"]
        return {"edits": [], "checked_ok": [0, 1]}

    monkeypatch.setattr(llm_client, "chat_json", fake_chat_json)
    # avg_logprob can be ONE global value for all segments of a transcript
    # (e.g. exactly -0.032 for every segment) — it must not prevent the fallback,
    # only real word scores count.
    segs = [
        {
            "start": 0,
            "end": 1,
            "text": "Hello folks.",
            "avg_logprob": -0.032,
            "words": [{"word": "Hello"}, {"word": "folks."}],
        },
        {
            "start": 1,
            "end": 2,
            "text": "Let's get started.",
            "avg_logprob": -0.032,
            "words": [{"word": "Let's"}],
        },
    ]
    result = correction.build_suggestions(segs, only_marked=True)
    assert result["confidence_data"] is False
    assert result["chunks"] == 1  # it WAS checked
    assert len(result["checked"]) == 2  # all segments editable
    assert result["confirmed"] == 2
    # Both segments were editable ([id]) in the prompt
    assert "[0]" in seen["user"] and "[1]" in seen["user"]


# ══════════════════════════════════════════════════════════════════════════
#  translation
# ══════════════════════════════════════════════════════════════════════════


def test_translate_completely_with_a_follow_up_request(monkeypatch):
    monkeypatch.setattr(llm_client, "effective_context", lambda *a: 8192)
    calls = []

    def fake_chat_json(messages, *, schema, **kw):
        user = messages[-1]["content"]
        calls.append(user)
        import re

        ids = [int(m) for m in re.findall(r"^\[(\d+)\]", user, re.M)]
        if "still missing" not in user and len(calls) == 1:
            ids = ids[:-1]  # First chunk: last ID 'forgotten'
        return {"translations": [{"id": i, "text": f"TR{i}"} for i in ids]}

    monkeypatch.setattr(llm_client, "chat_json", fake_chat_json)
    segs = _segs(8)
    result = translation.translate(segs, "English")
    assert result["missing"] == []
    assert [u["id"] for u in result["translations"]] == list(range(8))
    # Re-request has taken place
    assert any("still missing" in a for a in calls)


def test_translation_with_a_rolling_context(monkeypatch):
    monkeypatch.setattr(llm_client, "effective_context", lambda *a: 8192)
    # Force tiny budget → multiple chunks
    monkeypatch.setattr(
        translation, "chunk_budget", lambda **kw: 40 * 26
    )  # ~ space for few segments
    calls = []

    def fake_chat_json(messages, *, schema, **kw):
        user = messages[-1]["content"]
        calls.append(user)
        import re

        ids = [int(m) for m in re.findall(r"^\[(\d+)\]", user, re.M)]
        return {"translations": [{"id": i, "text": f"TR{i}"} for i in ids]}

    monkeypatch.setattr(llm_client, "chat_json", fake_chat_json)
    result = translation.translate(_segs(40), "English")
    assert len(calls) >= 2
    assert result["missing"] == []
    # From second chunk onwards, translated context runs
    assert "Already translated" in calls[1]
    assert "→ TR" in calls[1]


def test_translate_without_a_target_language():
    with pytest.raises(ValueError):
        translation.translate(_segs(1), "  ")


# ══════════════════════════════════════════════════════════════════════════
#  summarize: map-reduce switch + stream
# ══════════════════════════════════════════════════════════════════════════


def test_map_reduce_switch(monkeypatch):
    from whisper_tm.core import summarize as s

    monkeypatch.setattr(llm_client, "effective_context", lambda *a: 1000)
    small, _ = s._needs_map_reduce("short", "sys", "")
    large, _ = s._needs_map_reduce("x" * 40000, "sys", "")
    assert small is False
    assert large is True


def test_the_summary_stream_in_one_pass(monkeypatch):
    from whisper_tm.core import summarize as s

    monkeypatch.setattr(llm_client, "effective_context", lambda *a: 100000)
    monkeypatch.setattr(llm_client, "chat_stream", lambda messages, **kw: iter(["Hello ", "World"]))
    events = list(
        s.summary_stream(
            _segs(3),
            system_prompt="sys",
            user_prompt="",
            metadata=None,
            model="",
            include_speakers=True,
            include_timestamps=False,
        )
    )
    assert events == [{"text": "Hello "}, {"text": "World"}]


def test_the_summary_stream_with_map_reduce(monkeypatch):
    from whisper_tm.core import summarize as s

    # Tiny context → force map-reduce
    monkeypatch.setattr(llm_client, "effective_context", lambda *a: 900)
    map_calls = []

    def fake_chat(messages, **kw):
        map_calls.append(messages)
        return "Partial summary.", {}

    def fake_stream(messages, **kw):
        # Reduce sees the parts, not the raw transcript
        assert "Partial summary." in messages[-1]["content"]
        yield "FINAL"

    monkeypatch.setattr(llm_client, "chat", fake_chat)
    monkeypatch.setattr(llm_client, "chat_stream", fake_stream)
    segs = _segs(120)
    events = list(
        s.summary_stream(
            segs,
            system_prompt="sys",
            user_prompt="",
            metadata=None,
            model="",
            include_speakers=True,
            include_timestamps=False,
        )
    )
    statuses = [e["status"] for e in events if "status" in e]
    texts = [e["text"] for e in events if "text" in e]
    assert len(map_calls) >= 2  # several map chunks
    assert any("section" in st for st in statuses)
    assert texts == ["FINAL"]


# ══════════════════════════════════════════════════════════════════════════
#  chat_json: repair retry
# ══════════════════════════════════════════════════════════════════════════


def test_chat_json_truncated_without_repair(monkeypatch):
    """finish_reason=length → LlmResponseTruncated, NO repair retry
    (mirroring the truncated giant response back caused a timeout
    cascade)."""
    calls = []

    def fake_chat(messages, **kw):
        calls.append(1)
        return '{"edits": [', {"finish_reason": "length", "completion_tokens": 9999}

    monkeypatch.setattr(llm_client, "chat", fake_chat)
    with pytest.raises(llm_client.LlmAnswerTruncated):
        llm_client.chat_json([{"role": "user", "content": "x"}], schema={"type": "object"})
    assert len(calls) == 1


def test_translation_halves_on_a_truncated_answer(monkeypatch):
    monkeypatch.setattr(llm_client, "effective_context", lambda *a: 8192)
    calls = []

    def fake_chat_json(messages, *, schema, **kw):
        user = messages[-1]["content"]
        import re

        ids = [int(m) for m in re.findall(r"^\[(\d+)\]", user, re.M)]
        calls.append(len(ids))
        if len(ids) > 4:
            raise llm_client.LlmAnswerTruncated("cut off")
        return {"translations": [{"id": i, "text": f"TR{i}"} for i in ids]}

    monkeypatch.setattr(llm_client, "chat_json", fake_chat_json)
    result = translation.translate(_segs(8), "English")
    assert result["missing"] == []
    assert len(result["translations"]) == 8
    assert max(calls) > 4 and min(calls) <= 4  # it really was halved


def test_correction_halves_on_a_truncated_answer(monkeypatch):
    monkeypatch.setattr(llm_client, "effective_context", lambda *a: 8192)
    calls = []

    def fake_chat_json(messages, *, schema, **kw):
        user = messages[-1]["content"]
        import re

        ids = [int(m) for m in re.findall(r"^\[(\d+)\]", user, re.M)]
        calls.append(len(ids))
        if len(ids) > 3:
            raise llm_client.LlmAnswerTruncated("cut off")
        return {"edits": []}

    monkeypatch.setattr(llm_client, "chat_json", fake_chat_json)
    segs = []
    for i in range(6):
        segs.append(
            {
                "start": i * 5.0,
                "end": i * 5.0 + 4,
                "text": f"Sentence number {i} with content.",
                "avg_logprob": -1.5,
                "words": [],
            }
        )
    result = correction.build_suggestions(segs)
    assert result["chunks_failed"] == 0
    assert max(calls) > 3  # first the large range …
    assert min(calls) <= 3  # … then halved ranges


def test_chat_json_repair_retry(monkeypatch):
    replies = ["no json!", '{"edits": []}']
    calls = []

    def fake_chat(messages, **kw):
        calls.append(messages)
        return replies[len(calls) - 1], {}

    monkeypatch.setattr(llm_client, "chat", fake_chat)
    data = llm_client.chat_json([{"role": "user", "content": "do it"}], schema={"type": "object"})
    assert data == {"edits": []}
    assert len(calls) == 2
    # The retry mirrors back the broken response
    assert any(m["role"] == "assistant" for m in calls[1])
