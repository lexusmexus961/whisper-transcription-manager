"""\"alice\" and \"Alice\" must not be two different people.

A person saved as **\"alice\"** via the voice card is lowercase, while the
other names start with capitals (\"Dave\", \"Carol\", \"Bob\"). The next
time typing \"Alice\" would have created a SECOND person:

    SELECT id FROM speaker_people WHERE name = 'Alice'   → None
    INSERT INTO speaker_people (name) VALUES ('Alice')   → succeeds

This is not a cosmetic issue. This table *is* the identity; splitting it
hits exactly the person whose re-identification the whole effort
is about — and nobody would notice, because both names would look
right in the list. The voices silently distributed across two entries, the
inventory question \"who was there?\" would get two half answers.

Therefore ``COLLATE NOCASE`` on the name column — including migration, because
``CREATE TABLE IF NOT EXISTS`` doesn't touch an existing table, and
precisely the old data matters here.
"""

from __future__ import annotations

import pytest

from whisper_tm.database.speaker import SpeakerDB


@pytest.fixture()
def db(tmp_path) -> SpeakerDB:
    d = SpeakerDB(str(tmp_path / "speaker.db"))
    d.initialize()
    return d


class TestFolding:
    def test_upper_case_finds_lower_case(self, db):
        first = db.person_id("alice")
        again_val = db.person_id("Alice")
        assert again_val == first, "Alice created a second person"

    def test_lower_case_finds_upper_case(self, db):
        first = db.person_id("Bob")
        assert db.person_id("bob") == first

    def test_in_the_middle_of_a_word(self, db):
        first = db.person_id("Frank")
        assert db.person_id("Frank") == first

    def test_different_names_stay_different(self, db):
        """The folding must not go too far."""
        assert db.person_id("Alice") != db.person_id("Erin")

    def test_the_first_spelling_is_kept(self, db):
        """Whoever saved "alice" should not suddenly find "ALICE" in the list
        just because someone else typed it that way."""
        db.person_id("alice")
        db.person_id("ALICE")
        with db.connection() as c:
            names = [r["name"] for r in c.execute("SELECT name FROM speaker_people")]
        assert names == ["alice"]
