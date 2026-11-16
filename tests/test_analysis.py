"""Analyses — one task over many recordings (core/analysis.py).

For example: list every question about pricing across a set of support
calls, or summarize three videos on the same topic together.

What these tests pin:

* the collection: a short recording is one line, a long one a heading with
  its sentences, times and speakers; every line a reference that knows its
  recording's place; three orders — and in dialogue order the lines of one
  game dialogue, spread over a folder per voice, together and in line order;
* the parts: blocks stay whole where they fit, a larger one goes on under
  its heading; nothing over the budget;
* the set: a folder with or without subfolders (files not transcribed yet
  count as missing), a selection, paths — nothing outside the allowed folders;
* a run: one call when it fits, else notes per part and one answer from
  them, or the notes themselves; parts with nothing in them say so; the
  references come out as S1, S2 … in the answer's order; cancel; a restart
  keeps the parts already read;
* the routes, the LLM queue entry, the Markdown file, and a conversation in
  "Ask" with the analysis attached.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from whisper_tm.core import analysis, chat
from whisper_tm.core import llm_client as llm

# ── A library ──────────────────────────────────────────────────────────────


@pytest.fixture
def lib(tmp_path, monkeypatch):
    from whisper_tm import extensions as ext
    from whisper_tm.config import config
    from whisper_tm.database import Database

    db = Database(tmp_path / "lib.db")
    db.initialize()
    monkeypatch.setattr(ext, "db", db)
    monkeypatch.setattr(type(config), "data_dir", property(lambda self: tmp_path / "data"))
    root = tmp_path / "media"
    monkeypatch.setattr(analysis, "_roots", lambda: [str(root)])
    monkeypatch.setattr(analysis, "_context", lambda: 100_000)
    # An LLM queue worker another test's app started may still run — it reads
    # whatever database is current and would take up this test's entries
    from whisper_tm.core.llm_queue import llm_queue

    monkeypatch.setattr(llm_queue, "_paused", True)
    # … and the watcher for waiting analyses: the tests drive it themselves
    monkeypatch.setattr(analysis, "_watch_enabled", False)
    analysis._deleted.clear()
    chat._runs.clear()

    def add(rel, *texts, duration=0.0, names=None, speakers=None):
        segs, t = [], 0.0
        for i, text in enumerate(texts):
            segs.append(
                {
                    "start": t,
                    "end": t + 2.0,
                    "text": text,
                    "speaker": (speakers or ["SPEAKER_00"] * len(texts))[i],
                }
            )
            t += (duration / max(1, len(texts))) if duration else 2.0
        abs_path = str(root / rel)
        db.upsert_transcript(
            rel,
            {
                "complete": True,
                "segments": segs,
                "total_duration_sec": duration,
                "speaker_names": names or {},
            },
            media_dir=str(root),
            abs_path=abs_path,
        )
        return abs_path

    def media_only(rel):
        with db.connection() as c:
            c.execute(
                "INSERT INTO media_files (rel_path, abs_path) VALUES (?, ?)", (rel, str(root / rel))
            )
        return str(root / rel)

    g = "dialog"
    files = {
        "hero00": add(f"{g}/hero/DIA_Smith_Spy_15_00.wav", "What should I do?"),
        "smith01": add(
            f"{g}/smith/DIA_Smith_Spy_10_01.wav", "Go to the market.", "Keep your eyes peeled."
        ),
        "smith02": add(f"{g}/smith/DIA_Smith_Spy_10_02.wav", "And tell nobody of this."),
        "hero03": add(f"{g}/hero/DIA_Smith_Spy_15_03.wav", "Right on."),
        "hello": add(f"{g}/smith/DIA_Smith_Hello_10_00.wav", "Hail, stranger."),
        "talk": add(
            "talks/ai_talk.mp4",
            "Today we look at models.",
            "They got faster.",
            "Much faster.",
            duration=300.0,
            names={"SPEAKER_00": "Alice"},
            speakers=["SPEAKER_00", "SPEAKER_01", "SPEAKER_00"],
        ),
        "empty": add(f"{g}/smith/DIA_Smith_Empty_10_00.wav"),
        "missing": media_only(f"{g}/smith/DIA_Smith_Later_10_00.wav"),
    }
    yield {"db": db, "root": str(root), "files": files, "add": add, "tmp": tmp_path}
    db.close_all()


def _lines(corpus):
    out = []
    for b in corpus.blocks:
        if b.heading:
            out.append(b.heading)
        out.extend(ln.text for ln in b.lines)
    return out


# ── The collection ─────────────────────────────────────────────────────────


class TestTheCollection:
    def test_the_prompt_names_the_languages_spoken(self, lib):
        """Told "answer in the language of the recordings", the model answered
        an English template over German recordings in English: the prompt
        names the languages now, most spoken first."""
        f = lib["files"]
        with lib["db"].connection() as c:
            for key, code in (("hero00", "de"), ("smith01", "de"), ("talk", "en")):
                c.execute("UPDATE transcripts SET language=? WHERE abs_path=?", (code, f[key]))
        corpus = analysis.build_corpus([f["hero00"], f["smith01"], f["talk"]])
        assert corpus.languages == ["German", "English"]
        assert analysis.build_corpus([f["hello"]]).spoken == ""
        # Right after the task — in the heading the model went on answering
        # in the task's language
        filled = analysis.SINGLE_USER.format(files=3, spoken=corpus.spoken, corpus="…", task="T")
        assert filled.endswith("Task: T\n\nThe recordings are in German and English.")
        for prompt in (analysis.REDUCE_USER, analysis.OVERVIEW_USER):
            assert "Task: {task}{spoken}\n\nWrite it in the language" in prompt

    def test_the_language_of_the_recordings_is_named_in_the_task(self, lib):
        """With thinking on, the model answered "Answer in the language of
        the recordings" over German recordings in English — also with the
        languages said after the task. Named inside the task, it holds."""
        f = lib["files"]
        with lib["db"].connection() as c:
            for key, code in (("hero00", "de"), ("smith01", "de"), ("talk", "en")):
                c.execute("UPDATE transcripts SET language=? WHERE abs_path=?", (code, f[key]))
        task = "Compare them. Answer in the language of the recordings."
        one = analysis.build_corpus([f["hero00"], f["smith01"]])
        assert analysis._named_language(task, one) == (
            "Compare them. Answer in German (the language of the recordings)."
        )
        mixed = analysis.build_corpus([f["hero00"], f["smith01"], f["talk"]])
        assert (
            "Answer in German (the language most of the recordings are in)."
            in analysis._named_language(task, mixed)
        )
        # Nothing known, or a task without the phrase: as written
        assert analysis._named_language(task, analysis.build_corpus([f["hello"]])) == task
        assert analysis._named_language("Summarize.", one) == "Summarize."

    def test_a_short_recording_is_one_line_a_long_one_its_sentences(self, lib):
        f = lib["files"]
        c = analysis.build_corpus([f["smith01"], f["talk"]])
        lines = _lines(c)
        assert c.root == lib["root"]
        assert (
            "[S1] dialog/smith/DIA_Smith_Spy_10_01: Go to the market. Keep your eyes peeled."
            in lines
        )
        # A short one is the whole recording: its passage shows no time
        assert c.refs["S1"]["short"] is True and "short" not in c.refs["S2"]
        head = lines.index("### talks/ai_talk (5:00)")
        assert lines[head + 1 :] == [
            "[S2] 0:00 Alice: Today we look at models.",
            "[S3] 1:40 SPEAKER_01: They got faster.",
            "[S4] 3:20 Alice: Much faster.",
        ]

    def test_every_reference_knows_its_place(self, lib):
        f = lib["files"]
        c = analysis.build_corpus([f["talk"]])
        r = c.refs["S2"]
        assert r["abs_path"] == f["talk"] and r["filename"] == "talks/ai_talk.mp4"
        assert r["file_dir"] == lib["root"] and r["start"] == 100.0
        assert r["speaker"] == "SPEAKER_01" and r["text"] == "They got faster."

    def test_without_speakers(self, lib):
        c = analysis.build_corpus([lib["files"]["talk"]], speakers=False)
        assert "[S1] 0:00 Today we look at models." in _lines(c)

    def test_dialogue_order_puts_a_dialogue_together(self, lib):
        """A game may keep each voice in a folder: by folder, the hero's lines
        and Smith's lie apart; in dialogue order they are one dialogue, in
        the order of their line numbers."""
        f = lib["files"]
        every = [f[k] for k in ("hero00", "smith01", "smith02", "hero03", "hello")]
        by_folder = [ln.split(":")[0] for ln in _lines(analysis.build_corpus(every))]
        # Shown below the set's own folder
        assert by_folder == [
            "[S1] hero/DIA_Smith_Spy_15_00",
            "[S2] hero/DIA_Smith_Spy_15_03",
            "[S3] smith/DIA_Smith_Hello_10_00",
            "[S4] smith/DIA_Smith_Spy_10_01",
            "[S5] smith/DIA_Smith_Spy_10_02",
        ]
        lines = _lines(analysis.build_corpus(every, order="dialogue"))
        assert [ln.split(":")[0] for ln in lines] == [
            "[S1] smith/DIA_Smith_Hello_10_00",  # a dialogue of one line: no heading
            "### Dialogue DIA_Smith_Spy",
            "[S2] hero/DIA_Smith_Spy_15_00",
            "[S3] smith/DIA_Smith_Spy_10_01",
            "[S4] smith/DIA_Smith_Spy_10_02",
            "[S5] hero/DIA_Smith_Spy_15_03",
        ]

    def test_name_order_counts_numbers_as_numbers(self, lib):
        a10 = lib["add"]("x/clip_10.wav", "ten")
        a2 = lib["add"]("y/clip_2.wav", "two")
        lines = _lines(analysis.build_corpus([a10, a2], order="name"))
        assert [ln.split(" ")[1] for ln in lines] == ["y/clip_2:", "x/clip_10:"]

    def test_missing_and_empty_are_counted_not_read(self, lib):
        f = lib["files"]
        c = analysis.build_corpus([f["hero00"], f["empty"], f["missing"]])
        assert c.files == 1 and c.empty == 1 and c.missing == [f["missing"]]
        assert len(c.refs) == 1

    def test_the_fingerprint_follows_the_transcripts(self, lib):
        f = lib["files"]
        before = analysis.build_corpus([f["hero00"], f["smith01"]]).key
        assert analysis.build_corpus([f["smith01"], f["hero00"]]).key == before
        lib["add"]("dialog/hero/DIA_Smith_Spy_15_00.wav", "What should I do there?")
        assert analysis.build_corpus([f["hero00"], f["smith01"]]).key != before


class TestTheParts:
    def test_blocks_stay_whole_where_they_fit(self, lib):
        f = lib["files"]
        c = analysis.build_corpus(
            [f[k] for k in ("hero00", "smith01", "smith02", "hero03")], order="dialogue"
        )
        whole = c.tokens
        parts = analysis.split(c, whole + 5)
        assert len(parts) == 1 and parts[0].lines[0] == "### Dialogue DIA_Smith_Spy"
        assert parts[0].first == "hero/DIA_Smith_Spy_15_00"
        assert parts[0].last == "hero/DIA_Smith_Spy_15_03"

    def test_a_larger_block_goes_on_under_its_heading(self, lib):
        texts = [f"Sentence number {i} of a long talk about models." for i in range(40)]
        long = lib["add"]("talks/long.mp4", *texts, duration=3600.0)
        c = analysis.build_corpus([long])
        parts = analysis.split(c, 120)
        assert len(parts) > 3
        assert all(p.tokens <= 120 for p in parts)
        assert parts[0].lines[0] == "### long (1:00:00)"
        assert all(p.lines[0] == "### long (1:00:00) (continued)" for p in parts[1:])
        body = [ln for p in parts for ln in p.lines if not ln.startswith("###")]
        assert len(body) == 40 and body[0].startswith("[S1] ") and body[-1].startswith("[S40] ")

    def test_short_lines_fill_the_parts(self, lib):
        f = lib["files"]
        c = analysis.build_corpus(
            [f[k] for k in ("hero00", "smith01", "smith02", "hero03", "hello")]
        )
        parts = analysis.split(c, 30)
        assert sum(len(p.lines) for p in parts) == 5
        assert all(p.tokens <= 30 for p in parts)


class TestTheSet:
    def test_a_folder_with_and_without_subfolders(self, lib):
        f = lib["files"]
        top, label = analysis.resolve_source({"kind": "folder", "folder": lib["root"] + "/dialog"})
        assert f["missing"] in top and f["hero00"] in top and f["talk"] not in top
        assert label == "dialog with its subfolders"
        flat, label = analysis.resolve_source(
            {"kind": "folder", "recursive": False, "folder": lib["root"] + "/dialog/hero"}
        )
        assert sorted(flat) == sorted([f["hero00"], f["hero03"]]) and label == "hero"
        assert (
            analysis.resolve_source(
                {"kind": "folder", "recursive": False, "folder": lib["root"] + "/dialog"}
            )[0]
            == []
        )

    def test_a_selection_and_paths(self, lib):
        f = lib["files"]
        got, label = analysis.resolve_source(
            {
                "kind": "files",
                "base_dir": lib["root"] + "/dialog",
                "filenames": ["hero/DIA_Smith_Spy_15_00.wav", f["talk"]],
            }
        )
        assert sorted(got) == sorted([f["hero00"], f["talk"]]) and label == "2 selected in dialog"
        got, _ = analysis.resolve_source({"kind": "paths", "abs_paths": [f["talk"], f["talk"]]})
        assert got == [f["talk"]]

    def test_nothing_outside_the_allowed_folders(self, lib):
        with pytest.raises(analysis.AnalysisError, match="outside"):
            analysis.resolve_source({"kind": "folder", "folder": "/etc"})
        got, _ = analysis.resolve_source({"kind": "paths", "abs_paths": ["/etc/passwd"]})
        assert got == []
        got, _ = analysis.resolve_source(
            {"kind": "files", "base_dir": lib["root"], "filenames": ["../../etc/passwd"]}
        )
        assert got == []
        with pytest.raises(analysis.AnalysisError):
            analysis.resolve_source({"kind": "nope"})

    def test_the_plan_counts_before_anything_runs(self, lib):
        p = analysis.plan({"kind": "folder", "folder": lib["root"] + "/dialog"})
        assert p["files"] == 7 and p["transcribed"] == 5 and p["missing"] == 1 and p["empty"] == 1
        assert p["single"] is True and p["parts"] == 0 and p["tokens"] > 0
        assert p["estimate_sec"] > 0 and p["sample"][0].startswith("hero/")
        # Nothing transcribed takes no time
        f = lib["files"]
        p = analysis.plan({"kind": "paths", "abs_paths": [f["missing"], f["empty"]]})
        assert p["tokens"] == 0 and p["estimate_sec"] == 0


# ── A run ──────────────────────────────────────────────────────────────────


class FakeLlm:
    """The model: the notes it is given in turn, and a streamed answer."""

    def __init__(self, notes=(), final=("The answer ", "[S2] and [S1]."), ended="stop"):
        self.notes = list(notes)
        self.final = final
        self.ended = ended
        self.calls: list = []
        self.streams: list = []

    def chat(self, messages, **kw):
        self.calls.append((messages, kw))
        note = self.notes.pop(0) if self.notes else "Nothing new."
        if isinstance(note, Exception):
            raise note
        return note, {
            "prompt_tokens": 1000,
            "completion_tokens": 50,
            "prompt_tps": 800.0,
            "answer_tps": 30.0,
        }

    def chat_stream(self, messages, info=None, **kw):
        self.streams.append((messages, kw))
        yield from self.final
        if info is not None:
            info["finish_reason"] = self.ended
            info["timings"] = {
                "prompt_per_second": 900.0,
                "predicted_per_second": 31.0,
                "prompt_n": 2000,
                "predicted_n": 40,
            }


class FakeQueue:
    def __init__(self, abort_after=None):
        self.progresses, self.parts, self.text = [], [], ""
        self.abort_after = abort_after

    def progress(self, entry_id, pct, msg):
        self.progresses.append((pct, msg))

    def partial_result(self, entry_id, part):
        self.parts.append(part)

    def partial_text(self, entry_id, text):
        self.text = text

    def abort_requested(self, entry_id):
        return self.abort_after is not None and len(self.parts) >= self.abort_after


def _made(lib, keys, **kw):
    """An analysis saved as create() saves it — without queueing it."""
    a = {
        "id": "a1b2c3d4e5f6",
        "title": "Small talk",
        "task": kw.get("task", "Find small talk."),
        "source": {"kind": "paths"},
        "label": "test",
        "status": "queued",
        "created": time.time(),
        "options": {
            "order": kw.get("order", "dialogue"),
            "speakers": True,
            "merge": kw.get("merge", True),
        },
        "model": "",
        "parts": [],
        "result": "",
        "refs": {},
        "entry_id": 0,
    }
    analysis._save_files(a["id"], [lib["files"][k] for k in keys])
    analysis.save(a)
    return a["id"]


DIALOG = ("hero00", "smith01", "smith02", "hero03", "hello")


class TestARun:
    def test_one_call_when_it_fits(self, lib, monkeypatch):
        fake = FakeLlm()
        monkeypatch.setattr(llm, "chat", fake.chat)
        monkeypatch.setattr(llm, "chat_stream", fake.chat_stream)
        aid = _made(lib, DIALOG)
        q = FakeQueue()
        out = analysis.run(aid, q, 7)
        assert fake.calls == [] and len(fake.streams) == 1
        system, user = (m["content"] for m in fake.streams[0][0])
        assert system == analysis.SYSTEM
        assert "### Dialogue DIA_Smith_Spy" in user and user.rstrip().endswith(
            "Task: Find small talk."
        )
        a = analysis.load(aid)
        # In the answer's order: what was S2 is now S1
        assert a["status"] == "done" and a["result"] == "The answer [S1] and [S2]."
        assert a["refs"]["S1"]["filename"] == "dialog/hero/DIA_Smith_Spy_15_00.wav"
        assert a["refs"]["S2"]["filename"] == "dialog/smith/DIA_Smith_Hello_10_00.wav"
        assert set(a["refs"]) == {"S1", "S2"}
        assert a["counts"] == {
            "files": 5,
            "transcribed": 5,
            "missing": 0,
            "empty": 0,
            "tokens": a["counts"]["tokens"],
        }
        assert q.text == "The answer [S2] and [S1]."  # streamed as written
        assert out["refs"] == 2 and a["parts"] == []

    def test_parts_then_one_answer_from_their_notes(self, lib, monkeypatch):
        fake = FakeLlm(
            notes=["- Greeting [S1]", analysis.NOTHING, "- Small talk [S5][S4]"],
            final=("All of it: [S5], [S1].",),
        )
        monkeypatch.setattr(llm, "chat", fake.chat)
        monkeypatch.setattr(llm, "chat_stream", fake.chat_stream)
        monkeypatch.setattr(
            analysis,
            "budgets",
            lambda: {"context": 100_000, "part": 35, "single": 0, "final": 1000},
        )
        aid = _made(lib, DIALOG, order="folder")
        q = FakeQueue()
        analysis.run(aid, q, 7)
        assert len(fake.calls) == 3
        first = fake.calls[0][0][1]["content"]
        assert first.startswith("This is part 1 of 3 of a collection of 5 recordings")
        assert analysis.NOTHING in first and fake.calls[0][1]["max_tokens"] == analysis.MAP_MAX
        # Extraction at a low temperature; the configured one varied the finds
        assert fake.calls[0][1]["temperature"] == analysis.MAP_TEMPERATURE
        assert fake.streams[0][1]["temperature"] == analysis.FINAL_TEMPERATURE
        reduce = fake.streams[0][0][1]["content"]
        assert "read in 3 parts" in reduce and "- Greeting [S1]" in reduce
        assert "Notes on part 2" not in reduce  # nothing in it
        a = analysis.load(aid)
        assert [p["empty"] for p in a["parts"]] == [False, True, False]
        assert a["parts"][1]["text"] == ""
        # One numbering for the answer and the notes: the answer's order first
        assert a["result"] == "All of it: [S1], [S2]."
        assert a["parts"][0]["text"] == "- Greeting [S2]"
        assert a["parts"][2]["text"] == "- Small talk [S1][S3]"
        assert len(q.parts) == 3 and q.parts[0]["from_value"] == "hero/DIA_Smith_Spy_15_00"
        assert a["final_mode"] == "answer"
        # The server's own speeds: three notes at 800/30, the answer at 900/31
        assert a["stats"]["prompt_tps"] == 825.0 and a["stats"]["answer_tps"] == 30.2
        assert a["stats"]["calls"] == 4

    def test_the_notes_themselves_when_not_merged(self, lib, monkeypatch):
        fake = FakeLlm(notes=["- one [S1]", "- two [S3]", "- three [S5]"])
        monkeypatch.setattr(llm, "chat", fake.chat)
        monkeypatch.setattr(llm, "chat_stream", fake.chat_stream)
        monkeypatch.setattr(
            analysis,
            "budgets",
            lambda: {"context": 100_000, "part": 35, "single": 0, "final": 1000},
        )
        aid = _made(lib, DIALOG, order="folder", merge=False)
        analysis.run(aid, FakeQueue(), 7)
        assert fake.streams == []
        a = analysis.load(aid)
        assert a["result"] == "- one [S1]\n\n- two [S2]\n\n- three [S3]"
        assert a["final_mode"] == "notes"

    def test_more_notes_than_an_answer_holds_get_an_overview(self, lib, monkeypatch):
        """A broad task can produce tens of thousands of tokens of notes. Written
        out again the answer hits its limit and its references turn into
        folder names — the notes stay as they are, with an overview above."""
        fake = FakeLlm(
            notes=[
                "- Greeting [S1] and a long line of findings",
                "- More small talk [S3] with more words in it",
                "- And the rest of it [S5] once more",
            ],
            final=("Three kinds of small talk: greetings [S1] …",),
        )
        monkeypatch.setattr(llm, "chat", fake.chat)
        monkeypatch.setattr(llm, "chat_stream", fake.chat_stream)
        monkeypatch.setattr(
            analysis, "budgets", lambda: {"context": 100_000, "part": 35, "single": 0, "final": 40}
        )
        aid = _made(lib, DIALOG, order="folder")
        analysis.run(aid, FakeQueue(), 7)
        messages, kw = fake.streams[0]
        assert "Write the overview that goes above them" in messages[1]["content"]
        assert "- More small talk [S3]" in messages[1]["content"]
        assert kw["max_tokens"] == 40
        a = analysis.load(aid)
        assert a["final_mode"] == "overview"
        assert a["result"] == "Three kinds of small talk: greetings [S1] …"
        assert [p["text"] for p in a["parts"]][1] == "- More small talk [S2] with more words in it"
        md = analysis.markdown(a)
        assert "## Every finding, part by part" in md and "## Notes of each part" not in md

    def test_an_answer_cut_at_its_limit_says_so(self, lib, monkeypatch):
        fake = FakeLlm(final=("A long answer [S1]",), ended="length")
        monkeypatch.setattr(llm, "chat_stream", fake.chat_stream)
        aid = _made(lib, DIALOG)
        analysis.run(aid, FakeQueue(), 7)
        assert analysis.load(aid)["result"].endswith(
            "*(The answer stops here: it reached its length limit. Every finding is in the "
            "notes of each part.)*"
        )

    def test_nothing_anywhere(self, lib, monkeypatch):
        fake = FakeLlm(notes=[analysis.NOTHING] * 3)
        monkeypatch.setattr(llm, "chat", fake.chat)
        monkeypatch.setattr(
            analysis,
            "budgets",
            lambda: {"context": 100_000, "part": 35, "single": 0, "final": 1000},
        )
        aid = _made(lib, DIALOG, order="folder")
        analysis.run(aid, FakeQueue(), 7)
        a = analysis.load(aid)
        assert a["result"] == "Nothing in these recordings matches the task." and a["refs"] == {}

    def test_cancel_between_parts(self, lib, monkeypatch):
        fake = FakeLlm(notes=["- a [S1]", "- b [S2]", "- c [S3]"])
        monkeypatch.setattr(llm, "chat", fake.chat)
        monkeypatch.setattr(
            analysis,
            "budgets",
            lambda: {"context": 100_000, "part": 35, "single": 0, "final": 1000},
        )
        aid = _made(lib, DIALOG, order="folder")
        with pytest.raises(InterruptedError):
            analysis.run(aid, FakeQueue(abort_after=1), 7)
        a = analysis.load(aid)
        assert a["status"] == "cancelled" and len(a["parts"]) == 1 and len(fake.calls) == 1

    def test_a_restart_keeps_the_parts_already_read(self, lib, monkeypatch):
        monkeypatch.setattr(
            analysis,
            "budgets",
            lambda: {"context": 100_000, "part": 35, "single": 0, "final": 1000},
        )
        broken = FakeLlm(notes=["- a [S1]", llm.LlmConnectionError("server gone")])
        monkeypatch.setattr(llm, "chat", broken.chat)
        aid = _made(lib, DIALOG, order="folder")
        with pytest.raises(llm.LlmConnectionError):
            analysis.run(aid, FakeQueue(), 7)
        assert analysis.load(aid)["status"] == "error"
        again = FakeLlm(notes=["- b [S3]", "- c [S5]"], final=("Done [S1].",))
        monkeypatch.setattr(llm, "chat", again.chat)
        monkeypatch.setattr(llm, "chat_stream", again.chat_stream)
        analysis.run(aid, FakeQueue(), 8)
        assert len(again.calls) == 2  # parts 2 and 3 only
        a = analysis.load(aid)
        assert a["status"] == "done" and [p["number"] for p in a["parts"]] == [1, 2, 3]
        assert a["parts"][0]["text"] == "- a [S1]"

    def test_a_changed_transcript_reads_everything_again(self, lib, monkeypatch):
        monkeypatch.setattr(
            analysis,
            "budgets",
            lambda: {"context": 100_000, "part": 35, "single": 0, "final": 1000},
        )
        monkeypatch.setattr(llm, "chat", FakeLlm(notes=["- a [S1]", RuntimeError("x")]).chat)
        aid = _made(lib, DIALOG, order="folder")
        with pytest.raises(RuntimeError):
            analysis.run(aid, FakeQueue(), 7)
        lib["add"]("dialog/hero/DIA_Smith_Spy_15_00.wav", "What should I do there?")
        again = FakeLlm(final=("Done.",))
        monkeypatch.setattr(llm, "chat", again.chat)
        monkeypatch.setattr(llm, "chat_stream", again.chat_stream)
        analysis.run(aid, FakeQueue(), 8)
        assert len(again.calls) == 3

    def test_no_transcript_no_analysis(self, lib, monkeypatch):
        aid = _made(lib, ["missing", "empty"])
        with pytest.raises(analysis.AnalysisError, match="None of these recordings"):
            analysis.run(aid, FakeQueue(), 7)
        assert analysis.load(aid)["status"] == "error"
        # Ask on it says it failed, not "not finished yet"
        with pytest.raises(analysis.AnalysisError, match="failed"):
            analysis.ask(aid)

    def test_a_refused_analysis_is_logged_without_a_traceback(self, lib, caplog, monkeypatch):
        # A set without transcripts is a warning, not a full ERROR traceback
        import logging

        from whisper_tm.core import llm_tasks
        from whisper_tm.core.llm_queue import llm_queue

        monkeypatch.setattr(llm_tasks.config, "get", _with_endpoint(llm_tasks.config.get))
        aid = _made(lib, ["missing", "empty"])
        entry = {
            "id": 7,
            "kind": "analysis",
            "filename": "Analysis · x",
            "params": {"analysis_id": aid},
        }
        with caplog.at_level(logging.WARNING, logger="whisper_tm.core.llm_queue"):
            llm_queue._run_one(entry)
        rec = [r for r in caplog.records if "LLM queue entry 7 failed" in r.getMessage()]
        assert rec and rec[0].levelno == logging.WARNING and rec[0].exc_info is None
        assert "None of these recordings" in rec[0].getMessage()


# ── Routes, queue, Markdown, Ask ───────────────────────────────────────────


@pytest.fixture
def client(lib, monkeypatch):
    from flask import Flask

    from whisper_tm.api.analysis import analysis_bp
    from whisper_tm.api.llm_queue import llm_queue_bp

    app = Flask(__name__)
    app.register_blueprint(analysis_bp)
    app.register_blueprint(llm_queue_bp)
    return app.test_client()


def _entry(lib, entry_id):
    with lib["db"].connection() as c:
        return dict(c.execute("SELECT * FROM llm_queue WHERE id=?", (entry_id,)).fetchone())


class TestTheRoutes:
    def test_plan_start_list_get(self, client, lib):
        src = {"kind": "folder", "folder": lib["root"] + "/dialog"}
        p = client.post("/api/analyses/plan", json={"source": src, "order": "dialogue"}).get_json()
        assert p["transcribed"] == 5 and p["missing"] == 1
        r = client.post(
            "/api/analyses",
            json={
                "source": src,
                "task": "Find small talk.\nAll of it.",
                "order": "dialogue",
                "merge": False,
            },
        ).get_json()
        a = r["analysis"]
        assert r["ok"] and a["status"] == "queued" and a["title"] == "Find small talk."
        assert a["options"] == {"order": "dialogue", "speakers": True, "merge": False}
        e = _entry(lib, a["entry_id"])
        assert e["kind"] == "analysis" and e["status"] == "pending"
        assert json.loads(e["params_json"])["analysis_id"] == a["id"]
        assert e["filename"] == "Analysis · Find small talk."
        d = client.get("/api/analyses").get_json()
        assert d["analyses"][0]["id"] == a["id"] and d["presets"][0]["key"] == "together"
        assert "source" not in d["analyses"][0]
        got = client.get(f"/api/analyses/{a['id']}").get_json()
        assert got["task"] == "Find small talk.\nAll of it." and got["live_text"] == ""
        files = client.get(f"/api/analyses/{a['id']}/files").get_json()
        assert files["total"] == 7

    def test_a_selection_keeps_its_list_out_of_the_record(self, client, lib):
        r = client.post(
            "/api/analyses",
            json={
                "source": {
                    "kind": "files",
                    "base_dir": lib["root"],
                    "filenames": ["talks/ai_talk.mp4"],
                },
                "task": "Sum up.",
            },
        ).get_json()
        a = analysis.load(r["analysis"]["id"])
        assert a["source"] == {"kind": "files", "base_dir": lib["root"]}
        assert analysis.files_of(a["id"]) == [lib["files"]["talk"]]

    def test_refused(self, client, lib):
        src = {"kind": "folder", "folder": lib["root"]}
        assert client.post("/api/analyses", json={"source": src, "task": " "}).status_code == 400
        assert (
            client.post(
                "/api/analyses", json={"source": {"kind": "folder", "folder": "/etc"}, "task": "x"}
            ).status_code
            == 400
        )
        assert (
            client.post(
                "/api/analyses", json={"source": {"kind": "paths", "abs_paths": []}, "task": "x"}
            ).status_code
            == 400
        )
        assert client.get("/api/analyses/zzz").status_code == 404
        assert client.get("/api/analyses/aaaaaaaaaaaa").status_code == 404
        # The per-file queue routes take no analysis
        assert (
            client.post(
                "/api/llm/queue/add", json={"kind": "analysis", "filename": "x"}
            ).status_code
            == 400
        )

    def test_cancel_while_waiting_and_delete(self, client, lib):
        src = {"kind": "paths", "abs_paths": [lib["files"]["talk"]]}
        a = client.post("/api/analyses", json={"source": src, "task": "x"}).get_json()["analysis"]
        assert client.post(f"/api/analyses/{a['id']}/cancel").get_json()["ok"]
        assert client.get(f"/api/analyses/{a['id']}").get_json()["status"] == "cancelled"
        assert _entry(lib, a["entry_id"])["status"] == "cancelled"
        b = client.post("/api/analyses", json={"source": src, "task": "y"}).get_json()["analysis"]
        assert client.delete(f"/api/analyses/{b['id']}").get_json()["ok"]
        assert client.get(f"/api/analyses/{b['id']}").status_code == 404
        assert _entry(lib, b["entry_id"])["status"] == "cancelled"
        assert not list((lib["tmp"] / "data" / "analyses").glob(f"{b['id']}*"))
        assert client.delete(f"/api/analyses/{b['id']}").status_code == 404  # not a 200

    def test_an_entry_removed_in_the_llm_queue(self, client, lib):
        src = {"kind": "paths", "abs_paths": [lib["files"]["talk"]]}
        a = client.post("/api/analyses", json={"source": src, "task": "x"}).get_json()["analysis"]
        with lib["db"].connection() as c:
            c.execute("DELETE FROM llm_queue WHERE id=?", (a["entry_id"],))
        got = client.get(f"/api/analyses/{a['id']}").get_json()
        assert got["status"] == "cancelled" and got["message"] == "Removed from the LLM queue"

    def test_the_queue_runs_it_and_it_can_run_again(self, client, lib, monkeypatch):
        from whisper_tm.core import llm_tasks
        from whisper_tm.core.llm_queue import llm_queue

        monkeypatch.setattr(llm, "chat_stream", FakeLlm(final=("Done [S1].",)).chat_stream)
        monkeypatch.setattr(llm_tasks.config, "get", _with_endpoint(llm_tasks.config.get))
        src = {"kind": "folder", "folder": lib["root"] + "/dialog"}
        a = client.post("/api/analyses", json={"source": src, "task": "x"}).get_json()["analysis"]
        entry = llm_queue._next_one()
        assert entry["kind"] == "analysis"
        out = llm_tasks.execute(entry, llm_queue)
        assert out["analysis"] == a["id"] and out["refs"] == 1
        done = client.get(f"/api/analyses/{a['id']}").get_json()
        assert done["status"] == "done" and done["result"] == "Done [S1]."
        # Again — the folder looked at anew, the last run's notes gone
        lib["add"]("dialog/smith/DIA_Smith_New_10_00.wav", "New here?")
        r = client.post(f"/api/analyses/{a['id']}/rerun").get_json()
        assert r["analysis"]["status"] == "queued" and r["analysis"]["result"] == ""
        assert r["analysis"]["counts"]["files"] == 8
        assert client.post(f"/api/analyses/{a['id']}/rerun").status_code == 400  # still waiting

    def test_markdown(self, client, lib, monkeypatch):
        monkeypatch.setattr(llm, "chat_stream", FakeLlm(final=("Greets [S1].",)).chat_stream)
        aid = _made(lib, DIALOG)
        analysis.run(aid, FakeQueue(), 7)
        r = client.get(f"/api/analyses/{aid}/export")
        assert r.mimetype == "text/markdown"
        assert r.headers["Content-Type"] == "text/markdown; charset=utf-8"  # once
        assert 'filename="Small talk.md"' in r.headers["Content-Disposition"]
        md = r.get_data(as_text=True)
        assert md.startswith("# Small talk") and "Greets [S1]." in md
        assert "- **S1** dialog/smith/DIA_Smith_Hello_10_00.wav @ 0:00" in md
        # A title outside Latin-1 must not raise UnicodeEncodeError in the header
        a = analysis.load(aid)
        a["title"] = "東京 talk"
        analysis.save(a)
        r = client.get(f"/api/analyses/{aid}/export")
        assert r.status_code == 200
        assert "filename*=UTF-8''%E6%9D%B1%E4%BA%AC%20talk.md" in r.headers["Content-Disposition"]


def _with_endpoint(get):
    def patched(*keys, default=None):
        if keys == ("llm", "endpoint"):
            return "http://llm.test/v1"
        return get(*keys, default=default)

    return patched


class _OneTool:
    def tools(self):
        class T:
            def __init__(self):
                self.name, self.description = "search", "search tool"
                self.inputSchema = {"type": "object", "properties": {}}

        return [T()]

    def call(self, name, args, timeout=600):
        return {"hits": []}, False


class TestAskAboutIt:
    def test_a_conversation_with_the_analysis_attached(self, lib, client, monkeypatch):
        monkeypatch.setattr(llm, "chat_stream", FakeLlm(final=("Small talk: [S1].",)).chat_stream)
        aid = _made(lib, DIALOG)
        analysis.run(aid, FakeQueue(), 7)
        # Passages are named as the tools take them back: relative to the
        # open folder when they lie there
        from whisper_tm.config import config

        monkeypatch.setattr(type(config), "media_dir", property(lambda self: Path(lib["root"])))
        cid = client.post(f"/api/analyses/{aid}/ask").get_json()["conversation"]
        conv = chat.load(cid)
        assert conv["title"] == "About: Small talk"
        assert conv["refs"]["S1"]["filename"] == "dialog/smith/DIA_Smith_Hello_10_00.wav"
        assert "Small talk: [S1]." in conv["attached"]["text"]
        assert "S1: dialog/smith/DIA_Smith_Hello_10_00.wav @ 0:00" in conv["attached"]["text"]
        shown = chat.public(conv)
        assert shown["attached"] == {"kind": "analysis", "id": aid, "title": "Small talk"}
        assert shown["turns"][0]["role"] == "assistant"
        # A question: the analysis goes along in the system message, and no
        # lookup is forced — the answer may come from it
        seen = []

        def model(messages, tools, on_text, on_thinking, stop):
            seen.append(messages)
            on_text("From the analysis: [S1].")
            return {"content": "From the analysis: [S1].", "calls": [], "finish": "stop"}

        monkeypatch.setattr(chat, "MCP", _OneTool())
        monkeypatch.setattr(chat, "_stream_round", model)
        run = chat.start(cid, "Who greets?")
        for _ in range(200):
            if run.done:
                break
            time.sleep(0.02)
        system = seen[0][0]["content"]
        assert "An analysis is attached" in system and "Small talk: [S1]." in system
        assert seen[0][-1]["content"] == "Who greets?"  # no LOOK_FIRST
        assert chat.load(cid)["turns"][-1]["text"] == "From the analysis: [S1]."

    def test_only_when_done(self, client, lib):
        aid = _made(lib, DIALOG)
        assert client.post(f"/api/analyses/{aid}/ask").status_code == 400


class TestWhatTheModelSaid:
    def test_nothing_after_an_explanation_is_still_nothing(self):
        assert analysis._nothing("NOTHING RELEVANT")
        assert analysis._nothing("**Nothing relevant.**")
        assert analysis._nothing("These lines are all quest dialogue [S3].\n\nNOTHING RELEVANT")
        # The marker in the language of the task
        assert analysis._nothing("Keine Treffer in diesem Teil.\n\nNichts Relevantes")
        assert not analysis._nothing("- A guard jokes about the weather [S1]")
        assert not analysis._nothing(
            "- A joke [S1]\n\n**Everything else is not relevant** for this search in this part."
        )

    def test_a_long_explanation_does_not_reach_the_answer(self, lib, monkeypatch):
        fake = FakeLlm(
            notes=[
                "All of it is quest dialogue [S2].\n\nNOTHING RELEVANT",
                analysis.NOTHING,
                "- Small talk [S5]",
            ],
            final=("Found: [S5].",),
        )
        monkeypatch.setattr(llm, "chat", fake.chat)
        monkeypatch.setattr(llm, "chat_stream", fake.chat_stream)
        monkeypatch.setattr(
            analysis,
            "budgets",
            lambda: {"context": 100_000, "part": 35, "single": 0, "final": 1000},
        )
        aid = _made(lib, DIALOG, order="folder")
        analysis.run(aid, FakeQueue(), 7)
        reduce = fake.streams[0][0][1]["content"]
        assert "quest dialogue" not in reduce and "Notes on part 3" in reduce
        a = analysis.load(aid)
        assert [p["empty"] for p in a["parts"]] == [True, True, False]

    def test_the_model_that_answered_is_named(self, lib, monkeypatch):
        """A llama.cpp server answers with the model it loaded, whatever
        name was asked for — the analysis names that one."""
        fake = FakeLlm()

        def stream(messages, info=None, **kw):
            yield from fake.chat_stream(messages, info=info, **kw)
            info["model"] = "qwen3.6-35b-a3b"

        monkeypatch.setattr(llm, "chat", fake.chat)
        monkeypatch.setattr(llm, "chat_stream", stream)
        aid = _made(lib, DIALOG)
        a = analysis.load(aid)
        a["model"] = "gpt-4o-mini"
        analysis.save(a)
        analysis.run(aid, FakeQueue(), 7)
        assert analysis.load(aid)["model_used"] == "qwen3.6-35b-a3b"

    def test_the_task_names_several_things(self):
        assert "several things" in analysis.MAP_USER and "alone" in analysis.MAP_USER
        # After the task, last: notes only — some models otherwise write their
        # plan for the task into the notes ("The user wants to find …")
        tail = analysis.MAP_USER.split("Task: {task}")[1]
        assert "no reasoning" in tail and "{nothing}" in tail
        assert "several things" in analysis.REDUCE_USER
        # The answer must not switch to the language of one part's notes
        for prompt in (analysis.REDUCE_USER, analysis.OVERVIEW_USER):
            tail = prompt.split("Task: {task}")[1]
            assert "language of the task" in tail and "every part" in tail


def test_a_language_the_task_names_wins():
    """The templates end on "Answer in the language of the recordings", the
    rules said "Answer in the language of the task": an English template over
    German recordings came back in English."""
    assert "language of the recordings" in " ".join(p["task"] for p in analysis.PRESETS)
    assert "Answer in the language the task asks for" in analysis.SYSTEM
    for prompt in (analysis.REDUCE_USER, analysis.OVERVIEW_USER):
        assert "language the task asks for" in prompt.split("Task: {task}")[1]


# ── Which analyses read a recording (the recording's Analyses tab) ──────


def test_the_analyses_of_one_recording(tmp_path, monkeypatch):
    from whisper_tm.config import config
    from whisper_tm.core import analysis as an

    folder = tmp_path / "data" / "analyses"
    folder.mkdir(parents=True)
    monkeypatch.setattr(type(config), "data_dir", property(lambda self: tmp_path / "data"))
    an._file_index.clear()
    import json as _json
    import time as _time

    for aid, created, files in (
        ("aaaaaaaaaaa1", 1, ["/lib/x.mp3", "/lib/y.mp3"]),
        ("aaaaaaaaaaa2", 2, ["/lib/x.mp3"]),
        ("aaaaaaaaaaa3", 3, ["/lib/z.mp3"]),
    ):
        (folder / f"{aid}.json").write_text(
            _json.dumps({"id": aid, "title": aid.upper(), "created": created, "status": "done"})
        )
        (folder / f"{aid}.files.json").write_text(_json.dumps(files))
    assert [a["id"] for a in an.analyses_of_file("/lib/x.mp3")] == ["aaaaaaaaaaa2", "aaaaaaaaaaa1"]
    assert [a["id"] for a in an.analyses_of_file("/lib/z.mp3")] == ["aaaaaaaaaaa3"]
    assert an.analyses_of_file("/lib/none.mp3") == []
    # a changed file list is read again; a deleted analysis drops out
    _time.sleep(0.01)
    (folder / "aaaaaaaaaaa3.files.json").write_text(_json.dumps(["/lib/x.mp3"]))
    import os

    os.utime(folder / "aaaaaaaaaaa3.files.json", (_time.time() + 5, _time.time() + 5))
    (folder / "aaaaaaaaaaa1.json").unlink()
    (folder / "aaaaaaaaaaa1.files.json").unlink()
    assert [a["id"] for a in an.analyses_of_file("/lib/x.mp3")] == ["aaaaaaaaaaa3", "aaaaaaaaaaa2"]
