"""
Fuzzy search with multiple words.

With fuzzy search enabled (``search.fuzzy_enabled``), ``_fuzzy_fts_query``
builds a parenthesized group with similarity alternatives for each search word.
The parts were connected with a simple space — and FTS5 reads a space as an
implicit AND between two simple terms, but not next to a parenthesized group:

    have you*                 ->  hits
    (have OR hive) you*       ->  fts5: syntax error near "you"
    (have OR hive) AND you*   ->  hits

This caused EVERY multi-word search to fail with "Invalid search syntax (FTS5)".
Single words worked because no second group followed — it looked like the search
simply could not handle multiple words.
"""

from __future__ import annotations

import pytest

from whisper_tm.database import Database

# The vocabulary MUST contain similar words, otherwise the fuzzy extension
# finds no alternatives, builds no parenthesized group — and the syntax tests
# below run empty (skipped instead of checked). "hive/hare/hate/haze" are one
# edit away from "have", "mourning" one from "morning".
_TEXTS = [
    "Have you seen the plan?",
    "You have to leave before noon.",
    "The hive was full of bees.",
    "The hare ran fast and we hate the haze.",
    "A hash and a halt.",
    "Good morning, everyone.",
    "They were mourning and warning.",
    "In the morning the mood was good.",
    "This is a test.",
    "Completely different content without context.",
]


@pytest.fixture
def db(tmp_path):
    d = Database(tmp_path / "search.db")
    d.initialize()
    d.upsert_transcript(
        "sample.mp4",
        {
            "complete": True,
            "segments": [{"start": i, "end": i + 1, "text": t} for i, t in enumerate(_TEXTS)],
        },
        media_dir="/data",
        abs_path="/data/sample.mp4",
    )
    return d


def _fts(db, query, threshold=0.8):
    with db.connection() as conn:
        return db._fuzzy_fts_query(conn, query, threshold)


class TestSyntax:
    @pytest.mark.parametrize(
        "query",
        [
            "have you",
            "you have",
            "good morning",
            "this is a test",
        ],
    )
    def test_the_generated_query_is_valid_fts5(self, db, query):
        """The core: what is built must be digested by FTS5.

        If the fuzzy extension finds no alternatives (words too short or too
        rare), it returns None and the normal query takes over — that is also
        tested here.
        """
        built = _fts(db, query)
        fts = built[0] if built else db._build_fts_query(query)
        with db.connection() as conn:
            # Raises OperationalError if syntax is broken
            conn.execute(
                "SELECT COUNT(*) FROM segments_fts WHERE segments_fts MATCH ?", (fts,)
            ).fetchone()

    @pytest.mark.parametrize(
        "query",
        [
            "have you",
            "you have",
            "good morning",
            "this is a test",
        ],
    )
    def test_the_search_does_not_throw(self, db, query):
        """The path the application really takes."""
        assert db.search(query, fuzzy=0.8)["total"] >= 0

    def test_parts_are_joined_with_an_explicit_and(self, db):
        built = _fts(db, "have you")
        assert built is not None, "no fuzzy alternatives in test vocabulary"
        fts = built[0]
        assert "(" in fts, "expected a parenthesized group"
        assert " AND " in fts, (
            f"parenthesized group without explicit operator: {fts!r} — FTS5 rejects that"
        )


class TestSearchResult:
    def test_multiword_search_finds_something(self, db):
        r = db.search("have you", fuzzy=0.8)
        assert r["total"] > 0, "multi-word search returns nothing"

    def test_the_order_does_not_matter(self, db):
        """AND connection, not a phrase — both directions find results."""
        assert db.search("have you", fuzzy=0.8)["total"] > 0
        assert db.search("you have", fuzzy=0.8)["total"] > 0

    def test_still_fine_without_fuzzy(self, db):
        assert db.search("have you", fuzzy=0.0)["total"] > 0

    def test_a_single_word_is_unchanged(self, db):
        assert db.search("have", fuzzy=0.8)["total"] > 0

    def test_more_words_narrow_it_down(self, db):
        """AND means: all words must appear."""
        both = db.search("have you", fuzzy=0.0)["total"]
        impossible = db.search("have zebracrossing", fuzzy=0.0)["total"]
        assert both > 0
        assert impossible == 0
