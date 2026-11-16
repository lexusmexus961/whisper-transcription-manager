"""Conversation statistics: the counting (conversations/metrics.py), the
rows every transcript write keeps current (conversations/store.py) and the
background run for what is missing (conversations/runner.py).

The numbers decide which conversations a search finds ("long sentences",
"many speaker changes"), so each rule is pinned with numbers one can check
by hand: sentences span segments, unlabeled segments neither take a turn
nor end one, phantoms count for nobody, a pause ends a turn without being a
speaker change.
"""

from __future__ import annotations

import json
import sqlite3

import pytest

from whisper_tm.conversations import metrics, runner, store
from whisper_tm.database import Database


def _seg(speaker, start, text, end=None, **kw):
    return {
        "speaker": speaker,
        "start": start,
        "end": start + 2 if end is None else end,
        "text": text,
        **kw,
    }


def _tx(segments, duration=None, language="en"):
    return {
        "language": language,
        "segments": segments,
        "complete": True,
        "total_duration_sec": duration or (max((s["end"] for s in segments), default=0)),
        "speakers": sorted({s["speaker"] for s in segments if s["speaker"] != "—"}),
    }


@pytest.fixture
def db(tmp_path):
    d = Database(tmp_path / "x.db")
    d.initialize()
    return d


def _rows(db, table):
    with db.connection() as conn:
        return [dict(r) for r in conn.execute(f"SELECT * FROM {table}")]


# ── Words and sentences ────────────────────────────────────────────────────


class TestWords:
    def test_words_of_every_script(self):
        assert metrics.count_words("Hello there, Alice!") == 3
        assert metrics.count_words("नमस्ते दोस्त") == 2  # vowel signs stay in the word
        assert metrics.count_words("Привет, как дела?") == 3

    def test_a_language_without_spaces_counts_letters(self):
        assert metrics.count_words("今日は雨です。", unspaced=True) == 6
        # Spaced counting would see one word in the whole sentence
        assert metrics.count_words("今日は雨です。") == 1

    def test_digits_and_punctuation_are_no_words(self):
        assert metrics.count_words("3.5 — 42 % !") == 0


class TestSentences:
    def test_cut_at_the_end_marks(self):
        assert metrics.split_sentences("One two. Three four? Five!") == [
            "One two.",
            "Three four?",
            "Five!",
        ]

    def test_a_number_is_no_sentence_end(self):
        assert metrics.split_sentences("It costs 3.5 units. Fine.") == [
            "It costs 3.5 units.",
            "Fine.",
        ]

    def test_a_single_letter_before_the_period_is_an_abbreviation(self):
        assert metrics.split_sentences("We need e. g. bread and milk. Then go.") == [
            "We need e. g. bread and milk.",
            "Then go.",
        ]

    def test_closing_quotes_stay_with_the_sentence(self):
        assert metrics.split_sentences('She said "go." Then left.') == [
            'She said "go."',
            "Then left.",
        ]

    def test_cjk_full_stops_need_no_space(self):
        assert metrics.split_sentences("今日は雨です。明日は晴れ？") == [
            "今日は雨です。",
            "明日は晴れ？",
        ]

    def test_an_unfinished_tail_is_a_sentence(self):
        assert metrics.split_sentences("Done. And then") == ["Done.", "And then"]

    def test_questions(self):
        assert metrics.is_question("Really?")
        assert metrics.is_question('He asked "why?"')
        assert metrics.is_question("本当？")
        assert not metrics.is_question("Sure.")


# ── One transcript ─────────────────────────────────────────────────────────


class TestAnalyze:
    def test_a_sentence_spans_the_segments_of_one_turn(self):
        segs = [
            _seg("SPEAKER_00", 0, "This sentence was cut"),
            _seg("SPEAKER_00", 2, "into two cues. And here"),
            _seg("SPEAKER_00", 4, "ends another one."),
        ]
        s = metrics.analyze(segs)["summary"]
        assert s["sentences"] == 2
        assert s["words"] == 12
        assert s["wps"] == 6.0

    def test_speaker_changes_and_backchannels(self):
        segs = [
            _seg("SPEAKER_00", 0, "I think we should leave early tomorrow."),
            _seg("SPEAKER_01", 2, "Yes."),
            _seg("SPEAKER_00", 4, "Because the road is long."),
            _seg("SPEAKER_01", 6, "That is true, it takes four hours."),
        ]
        out = metrics.analyze(segs)
        assert out["summary"]["turns"] == 4
        assert out["summary"]["short_turns"] == 1  # "Yes."
        assert out["speakers"]["SPEAKER_01"]["short_turns"] == 1
        assert out["speakers"]["SPEAKER_00"]["turns"] == 2

    def test_unlabeled_segments_take_no_turn_and_end_none(self):
        segs = [
            _seg("SPEAKER_00", 0, "First part of a thought."),
            _seg("—", 2, "Something nobody was assigned."),
            _seg("SPEAKER_00", 4, "Second part of it."),
        ]
        out = metrics.analyze(segs)
        assert out["summary"]["turns"] == 1
        assert out["speakers"]["SPEAKER_00"]["turns"] == 1
        assert out["summary"]["unattributed_sec"] == 2.0
        assert out["summary"]["attributed_sec"] == 4.0
        # The unlabeled words still count for the transcript
        assert out["summary"]["words"] == 13

    def test_a_long_pause_ends_a_turn_but_is_no_change(self):
        segs = [
            _seg("SPEAKER_00", 0, "Before the pause."),
            _seg("SPEAKER_00", 60, "After a minute of silence."),
        ]
        out = metrics.analyze(segs)
        sp = out["speakers"]["SPEAKER_00"]
        assert out["summary"]["turns"] == 1
        assert sp["longest_turn_sec"] == 2.0  # not 62 s
        assert out["summary"]["speech_sec"] == 4.0

    def test_phantoms_count_for_nobody(self):
        segs = [
            _seg("SPEAKER_00", 0, "A real sentence here."),
            _seg("SPEAKER_01", 2, "Thanks for watching.", phantom_score=0.8),
        ]
        out = metrics.analyze(segs)
        assert set(out["speakers"]) == {"SPEAKER_00"}
        assert out["summary"]["words"] == 4

    def test_without_punctuation_each_segment_is_a_sentence(self):
        segs = [_seg("SPEAKER_00", i * 2, "word " * 30) for i in range(5)]
        s = metrics.analyze(segs)["summary"]
        assert s["punctuated"] is False
        assert s["sentences"] == 5
        assert s["wps"] is None  # no comparable sentence length

    def test_a_short_exchange_has_two_speakers(self):
        s = metrics.analyze(
            [_seg("SPEAKER_00", 0, "Hello Bob."), _seg("SPEAKER_01", 2, "Hi Alice.")]
        )["summary"]
        assert s["speakers"] == 2

    def test_rare_punctuation_still_counts_as_punctuated(self):
        # One sentence end in 40 words is enough
        segs = [_seg("SPEAKER_00", 0, "word " * 39 + "end.")]
        assert metrics.analyze(segs)["summary"]["punctuated"] is True

    def test_active_speakers_leave_out_noise(self):
        segs = [
            _seg("SPEAKER_00", 0, "Long talk. " * 5, end=100),
            _seg("SPEAKER_01", 100, "Long answer. " * 5, end=200),
            _seg("SPEAKER_02", 200, "Hm.", end=201),
        ]
        s = metrics.analyze(segs)["summary"]
        assert s["labels"] == 3
        assert s["speakers"] == 2

    def test_an_interview_has_one_asker(self):
        segs = []
        t = 0
        for i in range(8):
            segs.append(_seg("SPEAKER_00", t, "What happened next?", end=t + 3))
            segs.append(
                _seg(
                    "SPEAKER_01",
                    t + 3,
                    "Well, it was a long story and it went on and on. " * 3,
                    end=t + 30,
                )
            )
            t += 30
        s = metrics.analyze(segs)["summary"]
        assert s["interview"] is True
        assert metrics.analyze(segs[1:2] * 3)["summary"]["interview"] is False

    def test_rates_need_a_minute_of_speech(self):
        short = metrics.analyze([_seg("SPEAKER_00", 0, "Hi."), _seg("SPEAKER_01", 2, "Hello.")])
        assert short["summary"]["turn_rate"] is None
        long_ = metrics.analyze(
            [
                _seg("SPEAKER_00", 0, "Hi there.", end=30),
                _seg("SPEAKER_01", 30, "Hello you.", end=60),
            ]
        )
        assert long_["summary"]["turn_rate"] == 2.0  # 2 changes in 1 min
        assert long_["summary"]["mean_turn_sec"] == 30.0

    def test_windows_share_a_long_turn_out_by_words(self):
        # One turn of four sentences over two windows: 3 words in the
        # first, 9 in the second
        segs = [
            _seg("SPEAKER_00", 290, "One two three", end=299),
            _seg(
                "SPEAKER_00", 301, "four. Five six. Seven eight. Nine ten eleven twelve.", end=320
            ),
        ]
        w = metrics.analyze(segs)["windows"]["SPEAKER_00"]
        assert [r[0] for r in w] == [0, 1]
        # [window, talk, words, sentences, questions, changes]
        assert w[0][2] == 3 and w[1][2] == 9
        assert w[0][3] == 1.0 and w[1][3] == 3.0  # 4 sentences × 3/12, × 9/12
        assert w[0][5] == 1 and w[1][5] == 0  # the change where it started

    def test_nothing_in_nothing_out(self):
        out = metrics.analyze([])
        assert out["summary"]["words"] == 0
        assert out["summary"]["wps"] is None
        assert out["speakers"] == {} and out["windows"] == {}

    def test_broken_segments_are_skipped(self):
        segs = [
            {"speaker": "SPEAKER_00", "start": "x", "end": 2, "text": "bad"},
            "not a segment",
            None,
            _seg("SPEAKER_00", 0, "Good one."),
        ]
        assert metrics.analyze(segs)["summary"]["words"] == 2


# ── Rows kept by every transcript write ─────────────────────────────────────


class TestStore:
    def test_a_transcript_write_counts_it(self, db):
        tid = db.upsert_transcript(
            "a.mkv",
            _tx([_seg("SPEAKER_00", 0, "Hello Bob."), _seg("SPEAKER_01", 2, "Hi Alice.")]),
            media_dir="/m",
            abs_path="/m/a.mkv",
        )
        [row] = _rows(db, "conv_stats")
        assert row["transcript_id"] == tid
        assert row["version"] == store.STATS_VERSION
        assert row["seg_count"] == 2 and row["speakers"] == 2
        assert {r["label"] for r in _rows(db, "conv_speakers")} == {"SPEAKER_00", "SPEAKER_01"}

    def test_an_edit_replaces_the_rows(self, db):
        db.upsert_transcript(
            "a.mkv",
            _tx([_seg("SPEAKER_00", 0, "Hello."), _seg("SPEAKER_01", 2, "Hi.")]),
            abs_path="/m/a.mkv",
        )
        db.upsert_transcript(
            "a.mkv",
            _tx([_seg("SPEAKER_00", 0, "Hello."), _seg("SPEAKER_00", 2, "Hi.")]),
            abs_path="/m/a.mkv",
        )
        assert [r["label"] for r in _rows(db, "conv_speakers")] == ["SPEAKER_00"]
        assert len(_rows(db, "conv_stats")) == 1

    def test_deleting_the_transcript_takes_the_rows_along(self, db):
        segs = [
            _seg("SPEAKER_00" if i % 2 else "SPEAKER_01", i * 10, "Talk talk.", end=i * 10 + 9)
            for i in range(80)
        ]
        db.upsert_transcript("a.mkv", _tx(segs), media_dir="/m", abs_path="/m/a.mkv")
        assert _rows(db, "conv_windows")
        db.delete_transcript("a.mkv", "/m")
        assert not _rows(db, "conv_stats")
        assert not _rows(db, "conv_speakers")
        assert not _rows(db, "conv_windows")

    def test_windows_only_for_long_diarized_recordings(self, db):
        db.upsert_transcript(
            "short.mkv",
            _tx([_seg("SPEAKER_00", 0, "Hi."), _seg("SPEAKER_01", 2, "Hello.")]),
            abs_path="/m/short.mkv",
        )
        db.upsert_transcript(
            "plain.mkv",
            _tx([_seg("—", 0, "No labels."), _seg("—", 900, "Still none.")]),
            abs_path="/m/plain.mkv",
        )
        assert not _rows(db, "conv_windows")
        db.upsert_transcript(
            "long.mkv",
            _tx([_seg("SPEAKER_00", 0, "Hi."), _seg("SPEAKER_01", 900, "Hello.")]),
            abs_path="/m/long.mkv",
        )
        [w] = _rows(db, "conv_windows")
        data = json.loads(w["data"])
        assert data["w"] == metrics.WINDOW_SEC
        assert set(data["labels"]) == {"SPEAKER_00", "SPEAKER_01"}

    def test_the_date_in_the_file_name_is_kept(self, db):
        db.upsert_transcript(
            "x", _tx([_seg("SPEAKER_00", 0, "Hi.")]), abs_path="/m/call 2024-03-01 18-00.mkv"
        )
        [row] = _rows(db, "conv_stats")
        assert row["name_date"] == store.name_date("/m/call 2024-03-01 18-00.mkv")
        assert row["name_date"] is not None
        assert store.name_date("/m/no date here.mkv") is None

    def test_a_failure_to_count_never_costs_the_transcript(self, db, monkeypatch):
        def broken(data):
            raise RuntimeError("counting broke")

        monkeypatch.setattr(store, "compute", broken)
        tid = db.upsert_transcript(
            "a.mkv", _tx([_seg("SPEAKER_00", 0, "Hi.")]), abs_path="/m/a.mkv"
        )
        assert tid
        assert len(_rows(db, "segments")) == 1
        assert not _rows(db, "conv_stats")

    def test_a_failure_to_write_leaves_no_half_rows(self, db, monkeypatch):
        db.upsert_transcript(
            "a.mkv",
            _tx([_seg("SPEAKER_00", 0, "Hi."), _seg("SPEAKER_01", 2, "Hello.")]),
            abs_path="/m/a.mkv",
        )
        # A column that is not there: the speakers' rows fail after the
        # summary row was already replaced
        monkeypatch.setattr(store, "_SPEAKER_COLS", (*store._SPEAKER_COLS, "nope"))
        db.upsert_transcript(
            "a.mkv",
            _tx([_seg("SPEAKER_00", 0, "Hi again."), _seg("SPEAKER_01", 2, "Hello.")]),
            abs_path="/m/a.mkv",
        )
        # The transcript is the new one; the old numbers are gone instead of
        # passing for its own, and the background run counts it again
        [seg0] = [r for r in _rows(db, "segments") if r["seg_index"] == 0]
        assert seg0["text"] == "Hi again."
        assert not _rows(db, "conv_stats")
        assert not _rows(db, "conv_speakers")
        with db.connection() as conn:
            assert store.stale_count(conn) == 1

    def test_a_lost_transaction_is_reported_not_hidden(self, db):
        """A full disk or an I/O error makes SQLite roll back the whole
        transaction, savepoint included. The write must re-raise THAT
        error — not "no such savepoint" from undoing a savepoint that is
        already gone."""
        tid = db.upsert_transcript(
            "a.mkv",
            _tx([_seg("SPEAKER_00", 0, "Hi."), _seg("SPEAKER_01", 2, "Hello.")]),
            abs_path="/m/a.mkv",
        )
        stats = store.compute(_tx([_seg("SPEAKER_00", 0, "Hi."), _seg("SPEAKER_01", 2, "Hello.")]))

        class DiskFull:
            """Passes everything on, but the summary row hits a full disk:
            SQLite then rolls the transaction back by itself."""

            def __init__(self, conn):
                self.conn = conn

            @property
            def in_transaction(self):
                return self.conn.in_transaction

            def execute(self, sql, *args):
                if sql.startswith("INSERT OR REPLACE INTO conv_stats"):
                    self.conn.rollback()
                    raise sqlite3.OperationalError("database or disk is full")
                return self.conn.execute(sql, *args)

            def executemany(self, sql, *args):
                return self.conn.executemany(sql, *args)

        with db.connection() as conn:
            conn.execute("UPDATE transcripts SET language = 'de' WHERE id = ?", (tid,))
            with pytest.raises(sqlite3.OperationalError, match="disk is full"):
                store.write(DiskFull(conn), tid, stats, seg_count=2, char_count=9)
            assert not conn.in_transaction

    def test_a_rolled_back_transcript_write_fails_loudly(self, db, monkeypatch):
        """When the statistics' write loses the whole transaction, the
        transcript is lost with it: upsert_transcript must raise instead of
        returning an id for a row that was never saved."""

        def disk_full(conn, *a, **kw):
            conn.rollback()  # what SQLite does on SQLITE_FULL
            raise sqlite3.OperationalError("database or disk is full")

        monkeypatch.setattr(store, "write", disk_full)
        with pytest.raises(sqlite3.OperationalError, match="disk is full"):
            db.upsert_transcript(
                "a.mkv",
                _tx([_seg("SPEAKER_00", 0, "Hi."), _seg("SPEAKER_01", 2, "Hello.")]),
                abs_path="/m/a.mkv",
            )
        # Nothing half-written was committed on top of the lost transaction
        assert not _rows(db, "transcripts")
        assert not _rows(db, "segments")

    def test_switched_off_counts_nothing(self, db, monkeypatch):
        monkeypatch.setattr(store, "enabled", lambda: False)
        db.upsert_transcript("a.mkv", _tx([_seg("SPEAKER_00", 0, "Hi.")]), abs_path="/m/a.mkv")
        assert not _rows(db, "conv_stats")

    def test_a_write_while_switched_off_drops_the_old_numbers(self, db, monkeypatch):
        segs = [
            _seg("SPEAKER_00", 0, "Hello."),
            _seg("SPEAKER_01", 2, "Hi."),
            _seg("SPEAKER_00", 4, "How are you?"),
            _seg("SPEAKER_01", 6, "Fine."),
        ]
        db.upsert_transcript("a.mkv", _tx(segs), abs_path="/m/a.mkv")
        assert _rows(db, "conv_stats")[0]["speakers"] == 2
        # Two labels merged into one: the same segments and characters, so
        # kept rows would look current with two speakers and three changes
        monkeypatch.setattr(store, "enabled", lambda: False)
        before = store.generation()
        db.upsert_transcript(
            "a.mkv", _tx([dict(s, speaker="SPEAKER_00") for s in segs]), abs_path="/m/a.mkv"
        )
        assert not _rows(db, "conv_stats")
        assert not _rows(db, "conv_speakers")
        assert store.generation() > before
        with db.connection() as conn:
            assert store.stale_count(conn) == 1

    def test_every_write_moves_the_generation(self, db):
        before = store.generation()
        db.upsert_transcript("a.mkv", _tx([_seg("SPEAKER_00", 0, "Hi.")]), abs_path="/m/a.mkv")
        assert store.generation() > before

    def test_deleting_moves_the_generation(self, db):
        db.upsert_transcript(
            "a.mkv", _tx([_seg("SPEAKER_00", 0, "Hi.")]), media_dir="/m", abs_path="/m/a.mkv"
        )
        before = store.generation()
        assert db.delete_transcript("a.mkv", "/m")
        assert store.generation() > before


class TestStale:
    def test_missing_older_and_changed_rows_are_stale(self, db, monkeypatch):
        for name in ("a", "b", "c"):
            db.upsert_transcript(
                name, _tx([_seg("SPEAKER_00", 0, f"Hi {name}.")]), abs_path=f"/m/{name}.mkv"
            )
        with db.connection() as conn:
            assert store.stale_count(conn) == 0
            ids = [r[0] for r in conn.execute("SELECT id FROM transcripts ORDER BY id")]
            conn.execute("DELETE FROM conv_stats WHERE transcript_id = ?", (ids[0],))
            conn.execute("UPDATE conv_stats SET version = 0 WHERE transcript_id = ?", (ids[1],))
            conn.execute("UPDATE conv_stats SET char_count = -1 WHERE transcript_id = ?", (ids[2],))
        with db.connection() as conn:
            assert store.stale_count(conn) == 3
            assert store.stale_ids(conn, 0, 10) == ids
            assert store.stale_ids(conn, ids[0], 10) == ids[1:]
            assert store.counts(conn) == {"transcripts": 3, "counted": 0, "stale": 3}

    def test_unfinished_transcripts_are_not_asked_for(self, db):
        data = _tx([_seg("SPEAKER_00", 0, "Hi.")])
        data["complete"] = False
        db.upsert_transcript("a", data, abs_path="/m/a.mkv")
        with db.connection() as conn:
            conn.execute("DELETE FROM conv_stats")
            assert store.stale_count(conn) == 0


# ── The background run ─────────────────────────────────────────────────────


class TestRunner:
    @pytest.fixture
    def filled(self, db, monkeypatch):
        """Three transcripts written while counting was off."""
        monkeypatch.setattr("whisper_tm.extensions.db", db)
        monkeypatch.setattr(store, "enabled", lambda: False)
        for name in ("a", "b", "c"):
            db.upsert_transcript(
                name,
                _tx(
                    [
                        _seg("SPEAKER_00", 0, f"Hello {name}. How are you?"),
                        _seg("SPEAKER_01", 2, "Fine."),
                    ]
                ),
                abs_path=f"/m/{name}.mkv",
            )
        monkeypatch.setattr(store, "enabled", lambda: True)
        monkeypatch.setattr(runner, "PAUSE_SEC", 0)
        return db

    def test_counts_what_is_missing(self, filled):
        r = runner.Runner()
        assert r.start(reason="test")
        r.join(10)
        st = r.status()
        assert st["running"] is False
        assert st["done"] == 3 and st["errors"] == 0
        rows = _rows(filled, "conv_stats")
        assert len(rows) == 3
        assert all(row["questions"] == 1 for row in rows)
        with filled.connection() as conn:
            assert store.stale_count(conn) == 0

    def test_counts_the_same_as_the_write(self, filled, monkeypatch):
        r = runner.Runner()
        r.start()
        r.join(10)
        by_run = {row["transcript_id"]: row for row in _rows(filled, "conv_stats")}
        # Writing the same transcripts again counts them inline
        for name in ("a", "b", "c"):
            filled.upsert_transcript(
                name,
                _tx(
                    [
                        _seg("SPEAKER_00", 0, f"Hello {name}. How are you?"),
                        _seg("SPEAKER_01", 2, "Fine."),
                    ]
                ),
                abs_path=f"/m/{name}.mkv",
            )
        for row in _rows(filled, "conv_stats"):
            old = by_run[row["transcript_id"]]
            for col in ("words", "sentences", "questions", "turns", "speakers", "wps", "name_date"):
                assert row[col] == old[col], col

    def test_a_transcript_written_meanwhile_keeps_its_own_numbers(self, filled, monkeypatch):
        real = store.load_segments

        def edit_after_read(conn, ids):
            got = real(conn, ids)
            # An edit lands between the run's read and its write
            filled.upsert_transcript(
                "a", _tx([_seg("SPEAKER_00", 0, "Edited text, longer now.")]), abs_path="/m/a.mkv"
            )
            return got

        monkeypatch.setattr(store, "load_segments", edit_after_read)
        r = runner.Runner()
        r.start()
        r.join(10)
        with filled.connection() as conn:
            tid = conn.execute("SELECT id FROM transcripts WHERE abs_path = '/m/a.mkv'").fetchone()[
                0
            ]
            row = conn.execute(
                "SELECT * FROM conv_stats WHERE transcript_id = ?", (tid,)
            ).fetchone()
        assert row["speakers"] == 1  # the edit's numbers, not the run's

    def test_count_everything_again(self, filled):
        r = runner.Runner()
        r.start()
        r.join(10)
        with filled.connection() as conn:
            conn.execute("UPDATE conv_stats SET words = -5")
        r.start(recount=True)
        r.join(10)
        assert r.status()["done"] == 3
        assert all(row["words"] > 0 for row in _rows(filled, "conv_stats"))

    def test_one_broken_transcript_does_not_stop_the_run(self, filled, monkeypatch):
        real = store.compute
        calls = {"n": 0}

        def sometimes(data):
            calls["n"] += 1
            if calls["n"] == 2:
                raise ValueError("bad transcript")
            return real(data)

        monkeypatch.setattr(store, "compute", sometimes)
        r = runner.Runner()
        r.start()
        r.join(10)
        st = r.status()
        assert st["errors"] == 1 and "bad transcript" in st["last_error"]
        assert len(_rows(filled, "conv_stats")) == 2

    def test_only_one_run_at_a_time_and_stop(self, filled, monkeypatch):
        import threading

        gate = threading.Event()
        real = store.load_segments

        def slow(conn, ids):
            gate.wait(5)
            return real(conn, ids)

        monkeypatch.setattr(store, "load_segments", slow)
        r = runner.Runner()
        assert r.start()
        assert not r.start()
        assert r.stop()
        gate.set()
        r.join(10)
        assert r.status()["running"] is False

    def test_settings_are_clamped(self, monkeypatch):
        values = {"enabled": "false", "auto_run": "0", "min_share": "9"}
        monkeypatch.setattr(runner, "_cfg", lambda name, default=None: values.get(name, default))
        assert runner.settings() == {"enabled": False, "auto_run": False, "min_share": 0.5}
