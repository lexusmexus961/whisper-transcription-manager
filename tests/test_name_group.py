"""Name an entire group at once — without overwriting existing names.

Before, the GUI had no way to do it: "Save" only named the one selected voice.
For a group of twenty voices that meant twenty individual clicks.

The tricky part is not the assignment, but what happens to voices that already
have a name. A group is a **claim** by the machine ("sounds the same"), an
assignment the **determination** of a human. The claim must not overturn the
determination — otherwise one click on the wrong group would silently turn
twenty careful assignments into twenty wrong ones, and nobody would notice.

Therefore: existing names stay in place, and the response names both numbers.
"""

from __future__ import annotations

import pytest


@pytest.fixture()
def sdb(tmp_path, monkeypatch):
    from whisper_tm.database import speaker as mod

    db = mod.SpeakerDB(str(tmp_path / "speaker.db"))
    db.initialize()
    monkeypatch.setattr(mod, "_db", db, raising=False)
    from whisper_tm.api import speaker as api

    monkeypatch.setattr(api, "speaker_db", lambda: db)
    return db


@pytest.fixture()
def client(sdb):
    from flask import Flask

    from whisper_tm.api.speaker import speaker_bp

    app = Flask(__name__)
    app.register_blueprint(speaker_bp)
    return app.test_client()


def _vector(db, label: str) -> int:
    """Create a voice and return its id."""
    with db.connection() as c:
        cur = c.execute(
            "INSERT INTO speaker_vectors "
            "(abs_path, label, windows, start_sec, speech_sec, model) "
            "VALUES (?,?,?,?,?,?)",
            (f"/mnt/x/{label}.mp4", label, 0, 0.0, 30.0, "redimnet_m"),
        )
        return int(cur.lastrowid)


class TestAssigning:
    def test_open_voices_get_the_name(self, client, sdb):
        ids = [_vector(sdb, f"S{i}") for i in range(5)]
        r = client.post("/api/speaker/assign-many", json={"vector_ids": ids, "person": "Alice"})
        d = r.get_json()
        assert r.status_code == 200 and d["set_flag"] == 5
        assert sdb.names_per_vector() == {i: "Alice" for i in ids}

    def test_foreign_names_stay_untouched(self, client, sdb):
        """The core. Bob stays Bob, even when in Alice's group."""
        ids = [_vector(sdb, f"S{i}") for i in range(4)]
        sdb.assign(ids[1], "Bob")
        r = client.post("/api/speaker/assign-many", json={"vector_ids": ids, "person": "Alice"})
        d = r.get_json()
        assert d["set_flag"] == 3
        assert d["skipped"] == 1
        assert d["foreign_names"] == ["Bob"]
        assert sdb.names_per_vector()[ids[1]] == "Bob"

    def test_already_correct_ones_count_separately(self, client, sdb):
        """'3 set, 2 were already' is different information than '5 set' —
        and more honest."""
        ids = [_vector(sdb, f"S{i}") for i in range(5)]
        sdb.assign(ids[0], "Alice")
        sdb.assign(ids[1], "Alice")
        d = client.post(
            "/api/speaker/assign-many", json={"vector_ids": ids, "person": "Alice"}
        ).get_json()
        assert d["set_flag"] == 3
        assert d["already_correct"] == 2
        assert d["skipped"] == 0

    def test_upper_and_lower_case_count_as_the_same_name(self, client, sdb):
        """Otherwise "alice" would count against "Alice" as a foreign name and the
        group could never be cleanly renamed through."""
        ids = [_vector(sdb, f"S{i}") for i in range(3)]
        sdb.assign(ids[0], "alice")
        d = client.post(
            "/api/speaker/assign-many", json={"vector_ids": ids, "person": "Alice"}
        ).get_json()
        assert d["already_correct"] == 1
        assert d["skipped"] == 0

    def test_overwriting_only_happens_explicitly(self, client, sdb):
        ids = [_vector(sdb, f"S{i}") for i in range(3)]
        sdb.assign(ids[0], "Bob")
        d = client.post(
            "/api/speaker/assign-many",
            json={"vector_ids": ids, "person": "Alice", "overwrite": True},
        ).get_json()
        assert d["set_flag"] == 3
        assert sdb.names_per_vector()[ids[0]] == "Alice"

    def test_the_source_is_traceable(self, client, sdb):
        """Later one must be able to see that this was a mass action and not
        an individual decision."""
        ids = [_vector(sdb, "S0")]
        client.post("/api/speaker/assign-many", json={"vector_ids": ids, "person": "Alice"})
        with sdb.connection() as c:
            q = c.execute("SELECT source FROM speaker_assignment").fetchone()
        assert q["source"] == "map-group"


class TestDefense:
    @pytest.mark.parametrize(
        "body",
        [
            {},
            {"vector_ids": [], "person": "Alice"},
            {"vector_ids": [1, 2]},
            {"vector_ids": [1, 2], "person": "   "},
            {"vector_ids": "1,2", "person": "Alice"},
        ],
    )
    def test_nonsense_is_rejected(self, client, body):
        r = client.post("/api/speaker/assign-many", json=body)
        assert r.status_code == 400

    def test_the_name_is_trimmed(self, client, sdb):
        ids = [_vector(sdb, "S0")]
        d = client.post(
            "/api/speaker/assign-many", json={"vector_ids": ids, "person": "  Alice  "}
        ).get_json()
        assert d["person"] == "Alice"
        assert sdb.names_per_vector()[ids[0]] == "Alice"


class TestErrorMessage:
    """The error a stale server produces.

    Clicking "name entire group" while the running server did not yet know
    the endpoint — the JavaScript comes fresh from disk on each call, the
    Python part only after a restart. In the log:

        HTTP POST /api/speaker/assign-many → 404

    The response was an HTML error page, `r.json()` failed on the first `<`,
    and "this function does not exist here" became "Unexpected token '<'".
    Whoever reads that searches for the error in their data.
    """

    def _js(self, name: str) -> str:
        from pathlib import Path

        return (
            Path(__file__).resolve().parent.parent / "whisper_tm" / "static" / "js" / name
        ).read_text(encoding="utf-8")

    def test_the_status_is_checked_before_reading(self):
        js = self._js("voicemap.js")
        place = js.index("async function mapResponse")
        body = js[place : place + 1200]
        assert "if (r.ok)" in body
        # Naked `await r.json()` before status check is exactly the error —
        # in the helper it must only be WITHIN the ok branches.
        assert body.index("if (r.ok)") < body.index("await r.json()")

    def test_404_is_called_by_its_name(self):
        js = self._js("voicemap.js")
        assert "404" in js
        assert "server needs to be restarted" in js, "the remedy is missing from the message"

    def test_both_naming_paths_use_the_helper(self):
        """Individually and as a group — otherwise one path would have the
        unreadable message again."""
        # Both are in voicemap_panel.js.
        assert self._js("voicemap_panel.js").count("await mapResponse(r)") == 2

    def test_no_raw_json_reading_left_in_the_naming_paths(self):
        for file, the_function in (
            ("voicemap_panel.js", "async function mapRename"),
            ("voicemap_panel.js", "async function graphNameGroup"),
        ):
            js = self._js(file)
            place = js.index(the_function)
            body = js[place : place + 2500]
            assert "await r.json()" not in body, f"{file}: {the_function}"


class TestWiringUp:
    def test_the_button_calls_the_function(self):
        from pathlib import Path

        js = (
            Path(__file__).resolve().parent.parent
            / "whisper_tm"
            / "static"
            / "js"
            / "voicemap_panel.js"
        ).read_text(encoding="utf-8")
        assert "graphNameGroup(" in js
        assert "function graphNameGroup" in js
        assert "assign-many" in js

    def test_the_row_appears_in_the_group_section(self):
        from pathlib import Path

        js = (
            Path(__file__).resolve().parent.parent
            / "whisper_tm"
            / "static"
            / "js"
            / "voicemap_panel.js"
        ).read_text(encoding="utf-8")
        assert "graphNameGroupRow(nr)" in js

    def test_both_name_fields_are_updated(self):
        """`MAP.name` runs parallel to `members`; whoever sets only one
        sees the new name only after the next load."""
        from pathlib import Path

        js = (
            Path(__file__).resolve().parent.parent
            / "whisper_tm"
            / "static"
            / "js"
            / "voicemap_panel.js"
        ).read_text(encoding="utf-8")
        place = js.index("async function graphNameGroup")
        body = js[place : place + 2500]
        assert "m.person = d.person" in body
        assert "MAP.name[j] = d.person" in body
