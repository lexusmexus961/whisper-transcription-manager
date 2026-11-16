"""Speaker names in the meaning search (rag/speakers.py).

The index holds diarization labels (``[SPEAKER_04] Where is the map?``); the
voice library and the transcript editor know who a label is. A person's name
appears in only a fraction of the passages they speak in. For a query like
"alice asks for the map" the others were hard to find, and where found, the
reranker and the language model could not tell who asked. Now names
are put in at search time, and a person the query names brings the
passages they speak in as a candidate list of its own.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from whisper_tm.core import speaker as sp
from whisper_tm.core import speaker_inventory as inventory
from whisper_tm.database import Database
from whisper_tm.database.speaker import SpeakerDB
from whisper_tm.rag import retrieve, speakers
from whisper_tm.rag.vector_db import VectorDB

DIM = 16


@pytest.fixture
def voices(tmp_path, monkeypatch):
    d = SpeakerDB(tmp_path / "speaker.db")
    d.initialize()
    monkeypatch.setattr("whisper_tm.database.speaker.speaker_db", lambda: d)
    inventory.discard()
    speakers.forget()
    yield d
    speakers.forget()
    inventory.discard()


@pytest.fixture
def main_db(tmp_path, monkeypatch):
    db = Database(tmp_path / "wx.db")
    db.initialize()
    monkeypatch.setattr("whisper_tm.extensions.db", db)
    yield db
    db.close_all()


def _windows(db, path, labels):
    """``labels``: {label: number of windows} → {label: [vector ids]}."""
    rng = np.random.default_rng(len(path))
    rows = []
    for label, n in labels.items():
        for i in range(n):
            v = rng.normal(size=192).astype(np.float32)
            rows.append(
                {
                    "label": label,
                    "windows": i + 1,
                    "start_sec": i * 40.0,
                    "speech_sec": 30.0,
                    "segments": 2,
                    "parts": [(i * 40.0, i * 40.0 + 30)],
                    "v_separated": v / np.linalg.norm(v),
                }
            )
    db.write(path, sp.MODEL_BRAND, rows)
    out: dict[str, list[int]] = {}
    for w in db.windows(path):
        out.setdefault(w["label"], []).append(w["id"])
    return out


# ── The voice map's names ──────────────────────────────────────────────────


def test_a_label_carries_the_name_most_of_its_named_windows_carry(voices):
    w = _windows(voices, "/m/a.mkv", {"SPEAKER_00": 5, "SPEAKER_01": 4, "SPEAKER_02": 2})
    voices.assign_many(w["SPEAKER_00"][:3], "Alice")  # 3 of 3 named → Alice
    voices.assign_many(w["SPEAKER_01"][:2], "Bob")  # 2 Bob, 2 Alice → no majority
    voices.assign_many(w["SPEAKER_01"][2:], "Alice")
    voices.assign_many(w["SPEAKER_02"][:1], "Carol")  # 1 of 1 named → Carol
    assert voices.map_names() == {"/m/a.mkv": {"SPEAKER_00": "Alice", "SPEAKER_02": "Carol"}}
    # Of some recordings only (the duplicates view asks for its own)
    w2 = _windows(voices, "/m/b.mkv", {"SPEAKER_00": 2})
    voices.assign_many(w2["SPEAKER_00"], "Dave")
    assert voices.map_names(["/m/b.mkv", "/m/none.mkv"]) == {"/m/b.mkv": {"SPEAKER_00": "Dave"}}
    assert set(voices.map_names()) == {"/m/a.mkv", "/m/b.mkv"}


def test_names_typed_in_the_editor_come_first(voices, main_db, tmp_path):
    path = str(tmp_path / "talk.mp4")
    w = _windows(voices, path, {"SPEAKER_00": 2, "SPEAKER_01": 2})
    voices.assign_many(w["SPEAKER_00"], "Alice")
    voices.assign_many(w["SPEAKER_01"], "Bob")
    main_db.upsert_transcript(
        "talk.mp4",
        {
            "complete": True,
            "language": "de",
            "speaker_names": {"SPEAKER_01": "Erin", "SPEAKER_02": "Frank [host]"},
            "segments": [{"speaker": "SPEAKER_00", "start": 0, "end": 3, "text": "hello"}],
        },
        media_dir=str(tmp_path),
        abs_path=path,
    )
    got = speakers.names_for([path, "/elsewhere.mp4"])
    assert got == {path: {"SPEAKER_00": "Alice", "SPEAKER_01": "Erin", "SPEAKER_02": "Frank host"}}
    # The index's older keys (rel_path) find the typed names too
    assert speakers.names_for(["talk.mp4"])["talk.mp4"]["SPEAKER_01"] == "Erin"


def test_labels_become_names_in_the_text(voices):
    names = {"SPEAKER_04": "Alice"}
    text = "[SPEAKER_04] Where is the map? \n[SPEAKER_02] On the desk. [Music]"
    assert speakers.named(text, names) == (
        "[Alice] Where is the map? \n[SPEAKER_02] On the desk. [Music]"
    )
    assert speakers.named(text, None) == text
    assert speakers.named("", names) == ""


def test_rows_get_names_and_the_speaker_its_name(voices):
    w = _windows(voices, "/m/b.mkv", {"SPEAKER_03": 2})
    voices.assign_many(w["SPEAKER_03"], "Alice")
    rows = [
        {
            "transcript_key": "/m/b.mkv",
            "speaker": "SPEAKER_03",
            "content": "[SPEAKER_03] Ready? \n[SPEAKER_01] Almost.",
        },
        {
            "transcript_key": "/m/none.mkv",
            "speaker": "SPEAKER_00",
            "content": "[SPEAKER_00] nothing",
        },
        None,
    ]
    speakers.name_rows(rows)
    assert rows[0]["content"] == "[Alice] Ready? \n[SPEAKER_01] Almost."
    assert rows[0]["speaker_name"] == "Alice"
    assert rows[1]["content"] == "[SPEAKER_00] nothing" and "speaker_name" not in rows[1]


def test_the_query_names_people_by_any_word_of_their_name(voices):
    w = _windows(voices, "/m/c.mkv", {"SPEAKER_00": 1, "SPEAKER_01": 1, "SPEAKER_02": 1})
    voices.assign_many(w["SPEAKER_00"], "Alice")
    voices.assign_many(w["SPEAKER_01"], "Oscar Moss")
    voices.assign_many(w["SPEAKER_02"], "Alicewood")
    assert speakers.people_named(["alice"]) == ["Alice"]
    assert speakers.people_named(["moss", "nobody"]) == ["Oscar Moss"]
    assert speakers.people_named([]) == []
    assert speakers.labels_of("ALICE") == {"/m/c.mkv": {"SPEAKER_00"}}


def test_a_new_name_on_the_map_shows_at_once(voices):
    w = _windows(voices, "/m/d.mkv", {"SPEAKER_00": 2})
    assert speakers.names_for(["/m/d.mkv"]) == {}
    voices.assign_many(w["SPEAKER_00"], "Alice")  # moves the library's generation
    assert speakers.names_for(["/m/d.mkv"]) == {"/m/d.mkv": {"SPEAKER_00": "Alice"}}


def test_a_name_typed_in_a_transcript_shows_at_once(voices, main_db, tmp_path, monkeypatch):
    """/api/save_speaker_names forgets the held names."""
    from whisper_tm.api import transcription as tr

    path = str(tmp_path / "e.mp4")
    main_db.upsert_transcript(
        "e.mp4",
        {
            "complete": True,
            "language": "de",
            "segments": [{"speaker": "SPEAKER_00", "start": 0, "end": 3, "text": "hello"}],
        },
        media_dir=str(tmp_path),
        abs_path=path,
    )
    assert speakers.names_for([path]) == {}
    store = {}
    monkeypatch.setattr(tr, "_checked", lambda f: None)
    monkeypatch.setattr(
        tr, "_load_transcript_data", lambda f: store.setdefault("t", {"segments": []})
    )

    def save(filename, tdata):
        main_db.upsert_transcript(
            "e.mp4",
            {
                **tdata,
                "complete": True,
                "language": "de",
                "segments": [{"speaker": "SPEAKER_00", "start": 0, "end": 3, "text": "hello"}],
            },
            media_dir=str(tmp_path),
            abs_path=path,
        )

    monkeypatch.setattr(tr, "_save_transcript_data", save)
    from flask import Flask

    app = Flask(__name__)
    app.register_blueprint(tr.transcription_bp)
    r = app.test_client().post(
        "/api/save_speaker_names",
        json={"filename": "e.mp4", "speaker_names": {"SPEAKER_00": "Grace"}},
    )
    assert r.status_code == 200, r.get_data(as_text=True)
    assert speakers.names_for([path]) == {path: {"SPEAKER_00": "Grace"}}


# ── The person's passages as candidates ────────────────────────────────────


def _vec(seed):
    v = np.random.default_rng(seed).normal(size=DIM).astype(np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture
def index(tmp_path):
    rag = VectorDB(str(tmp_path / "rag.db"))
    passages = [
        ("/m/rec1.mkv", 0, "[SPEAKER_04] where is the map \n[SPEAKER_02] it is on the desk", 1),
        ("/m/rec1.mkv", 1, "[SPEAKER_02] the train leaves at noon", 2),  # Alice silent
        ("/m/rec1.mkv", 2, "[SPEAKER_04] i will call later", 3),
        ("/m/other.mkv", 0, "[SPEAKER_04] a whole other person", 4),  # same label, not Alice
        ("/m/rec2.mkv", 0, "[SPEAKER_00] good morning all \n[SPEAKER_01] morning", 5),
    ]
    with rag.connection() as c:
        rag.ensure_schema(c, dimensions=DIM)
        rag.upsert_chunks(
            c,
            [
                {
                    "transcript_key": k,
                    "chunk_index": i,
                    "content": t,
                    "speaker": t[1:11],
                    "start_sec": i * 60.0,
                    "end_sec": i * 60.0 + 60,
                    "embedding": _vec(seed),
                }
                for k, i, t, seed in passages
            ],
        )
    return rag


def _alice(voices):
    a = _windows(voices, "/m/rec1.mkv", {"SPEAKER_04": 2, "SPEAKER_02": 1})
    b = _windows(voices, "/m/rec2.mkv", {"SPEAKER_00": 1})
    voices.assign_many(a["SPEAKER_04"] + b["SPEAKER_00"], "Alice")


def test_a_persons_passages_are_where_their_label_speaks(voices, index):
    _alice(voices)
    got = retrieve.voice_ids(index, ["Alice"])
    rows = index.rows_by_id(got["Alice"])
    assert sorted(r["content"][:22] for r in rows.values()) == [
        "[SPEAKER_00] good morn",
        "[SPEAKER_04] i will ca",
        "[SPEAKER_04] where is ",
    ]


def test_only_the_chosen_ids_compete(index):
    X, ids = index.matrix()
    q = X[3].astype(np.float32)
    got = index.search_vector(q, k=10, min_value=-1, among=[int(ids[0]), int(ids[1]), 999])
    assert sorted(r["id"] for r in got) == sorted([int(ids[0]), int(ids[1])])
    assert index.search_vector(q, k=10, among=[]) == []
    assert index.search_vector(q, k=10, among=[12345]) == []


def test_a_named_person_brings_a_candidate_list(voices, index, monkeypatch):
    _alice(voices)
    monkeypatch.setattr(retrieve, "known_names", lambda: frozenset({"alice"}))
    X, ids = index.matrix()
    qvec = X[list(ids).index(3)].astype(np.float32)  # "i will call later"
    rows, info = retrieve.candidates(
        "alice asks for the map", k_dense=0, k_fts=50, db=index, qvec=qvec, k_voice=10
    )
    assert info["voices"] == {"Alice": 3}
    assert info["lists"]["voice:Alice"] == 3
    by_text = {r["content"][:22]: r for r in rows}
    voice_only = by_text["[SPEAKER_04] i will ca"]
    assert voice_only["lists"] == ["voice:Alice"]
    assert voice_only["match"] == "vector" and voice_only["vscore"] == pytest.approx(1.0, abs=1e-3)
    assert "[SPEAKER_02] the train" not in by_text  # Alice doesn't speak there
    assert "[SPEAKER_04] a whole o" not in by_text  # another file's SPEAKER_04


def test_no_voice_list_without_a_question_vector(voices, index, monkeypatch):
    _alice(voices)
    monkeypatch.setattr(retrieve, "known_names", lambda: frozenset({"alice"}))
    rows, info = retrieve.candidates(
        "alice", k_dense=0, k_fts=50, db=index, k_voice=10, use_dense=False
    )
    assert "voices" not in info and not any(n.startswith("voice") for n in info["lists"])


def test_the_quick_search_reads_and_shows_names(voices, index, monkeypatch):
    _alice(voices)
    monkeypatch.setattr(retrieve, "known_names", lambda: frozenset({"alice"}))
    X, ids = index.matrix()
    seen = []

    def rerank(query, docs, **kw):
        seen.extend(docs)
        return [1.0 if "desk" in d else 0.1 for d in docs]

    out = retrieve.quick_search(
        "alice asks for the map",
        db=index,
        embed_fn=lambda texts: [X[0].astype(np.float32).tolist()],
        rerank_fn=rerank,
    )
    assert any(d.startswith("[Alice] where is the map \n[SPEAKER_02] it is on") for d in seen)
    top = out["results"][0]
    assert top["content"].startswith("[Alice] where is the map")
    assert top["speaker_name"] == "Alice" and top["speaker"] == "SPEAKER_04"
    assert out["voices"] == {"Alice": 3}


# ── The deep search reads names ────────────────────────────────────────────


def test_the_deep_search_ranks_checks_and_shows_with_names(voices, index, monkeypatch):
    from whisper_tm.rag import deep_search

    _alice(voices)
    rows = index.rows_by_id(range(1, 10))
    alice_row = next(r for r in rows.values() if r["content"].startswith("[SPEAKER_04] where"))
    speakers.name_rows([alice_row])
    assert alice_row["content"].startswith("[Alice] where is the map")
    prompt = deep_search.verify_prompt(
        alice_row["content"], {"situation": "Alice asks for the map"}
    )
    assert "[Alice] where is the map" in prompt
    item = deep_search._result(0, "hit", {"id": alice_row["id"]}, alice_row, 0.9, [])
    assert item["snippet"].startswith("[Alice]") and item["speaker_name"] == "Alice"


def test_the_deep_search_ranker_gets_named_texts(voices, index, monkeypatch):
    from whisper_tm.rag import deep_search

    _alice(voices)
    X, ids = index.matrix()
    groups = [{"id": int(i), "copies": []} for i in ids]
    work = {"groups": groups, "scores": [None] * len(groups)}
    seen = []

    def rerank(query, docs, **kw):
        seen.extend(docs)
        return [0.5] * len(docs)

    monkeypatch.setattr(deep_search.rerank_server, "rerank", rerank)

    class Q:
        def progress(self, *a, **k):
            pass

        def abort_requested(self, _e):
            return False

    s = {"id": "abcdefabcdef", "query": "alice", "progress": {}, "message": ""}
    r = deep_search._Run(s, Q(), 1)
    monkeypatch.setattr(deep_search, "save", lambda s: None)
    monkeypatch.setattr(deep_search, "_save_work", lambda *a: None)
    deep_search._rank_all(r, work, "instr", index)
    assert sum(d.startswith("[Alice]") for d in seen) == 3
    assert any(d.startswith("[SPEAKER_04] a whole other") for d in seen)


def test_json_names_in_the_transcript_are_read_safely():
    assert speakers._typed(json.dumps({"SPEAKER_00": "Alice", "SPEAKER_01": ""})) == {
        "SPEAKER_00": "Alice"
    }
    assert speakers._typed("not json") == {}
    assert speakers._typed(json.dumps(["a"])) == {}
    assert speakers._typed(json.dumps({"SPEAKER_00": "SPEAKER_00"})) == {}


def test_an_index_without_labels_in_the_text_finds_the_main_speaker(voices, tmp_path):
    rag = VectorDB(str(tmp_path / "plain.db"))
    with rag.connection() as c:
        rag.ensure_schema(c, dimensions=DIM)
        rag.upsert_chunks(
            c,
            [
                {
                    "transcript_key": "/m/rec1.mkv",
                    "chunk_index": 0,
                    "content": "where is the map",
                    "speaker": "SPEAKER_04",
                    "embedding": _vec(1),
                },
                {
                    "transcript_key": "/m/rec1.mkv",
                    "chunk_index": 1,
                    "content": "the train leaves",
                    "speaker": "SPEAKER_02",
                    "embedding": _vec(2),
                },
            ],
        )
    _alice(voices)
    got = retrieve.voice_ids(rag, ["Alice"])["Alice"]
    assert [r["content"] for r in rag.rows_by_id(got).values()] == ["where is the map"]


def test_the_reading_columns_context_names_the_voices_too(voices, main_db, tmp_path, monkeypatch):
    """A meaning hit reads "[Alice] …" — its context in the reading column says
    Alice as well, and a name typed in the transcript still comes first."""
    from flask import Flask

    from whisper_tm.api import search as search_api

    path = str(tmp_path / "talk.mp4")
    main_db.upsert_transcript(
        "talk.mp4",
        {
            "complete": True,
            "language": "de",
            "speaker_names": {"SPEAKER_01": "Erin"},
            "segments": [
                {"speaker": "SPEAKER_00", "start": 0, "end": 3, "text": "where is the map"},
                {"speaker": "SPEAKER_01", "start": 3, "end": 6, "text": "on the desk"},
                {"speaker": "SPEAKER_02", "start": 6, "end": 9, "text": "quiet"},
            ],
        },
        media_dir=str(tmp_path),
        abs_path=path,
    )
    w = _windows(voices, path, {"SPEAKER_00": 2, "SPEAKER_01": 2})
    voices.assign_many(w["SPEAKER_00"], "Alice")
    voices.assign_many(w["SPEAKER_01"], "Bob")  # the typed Erin wins
    monkeypatch.setattr(search_api, "db", main_db)
    app = Flask(__name__)
    app.register_blueprint(search_api.search_bp)
    d = (
        app.test_client()
        .get(
            "/api/search/context",
            query_string={"file": "talk.mp4", "dir": str(tmp_path), "start": 3, "n": 3},
        )
        .get_json()
    )
    assert [s["speaker"] for s in d["segments"]] == ["Alice", "Erin", "SPEAKER_02"]
