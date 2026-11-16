"""The voice map's trash — thrown away means OUT OF THE CALCULATION.

Speakers that are not relevant can be moved to a trash in the speaker map
and viewed on a second map.

The design deliberately goes beyond just hiding. That is the decisive feature
and the reason for this file: a noise voice that is only invisible still glues
two real speakers together into a group — you just don't see the lump anymore.
Only when it falls out of the inventory does it fall apart.

There is already hiding in the map (`GRAPH.selection`, `minGroup`), and it is
something different: a view for this session, gone after reload. The trash is
in the database.

Nothing is deleted — the vectors account for computation time. That is why the
tests also check the way back.
"""

from __future__ import annotations

import numpy as np
import pytest
from flask import Flask

from whisper_tm.core import speaker as speaker_core
from whisper_tm.core import speaker_inventory as inventory
from whisper_tm.database.speaker import SpeakerDB

DIM = 192


def _vec(seed):
    return np.random.default_rng(seed).normal(size=DIM).astype(np.float32)


@pytest.fixture
def app(tmp_path, monkeypatch):
    from whisper_tm.config import config

    media = tmp_path / "media"
    media.mkdir()
    monkeypatch.setattr(type(config), "media_dir", property(lambda self: media))

    db = SpeakerDB(tmp_path / "speaker.db")
    db.initialize()
    monkeypatch.setattr("whisper_tm.database.speaker.speaker_db", lambda: db)
    monkeypatch.setattr("whisper_tm.api.speaker.speaker_db", lambda: db)
    inventory.discard()

    from whisper_tm.api.speaker import speaker_bp

    a = Flask(__name__)
    a.register_blueprint(speaker_bp)
    a.db = db
    a.media = media
    yield a
    inventory.discard()


def _populate(app, files=3, per_file=4):
    ids = {}
    for d in range(files):
        basis = _vec(100 + d)
        path = str(app.media / f"recording{d}.mkv")
        app.db.write(
            path,
            speaker_core.MODEL_BRAND,
            [
                {
                    "label": "SPEAKER_00",
                    "windows": i,
                    "start_sec": i * 300.0,
                    "speech_sec": 30.0,
                    "segments": 2,
                    "v_separated": basis + 0.02 * _vec(1000 + d * 10 + i),
                }
                for i in range(per_file)
            ],
        )
        ids[path] = [z["id"] for z in app.db.windows(path)]
    return ids


# ── The trash ────────────────────────────────────────────────────────────


class TestStorage:
    def test_discarded_items_are_not_deleted(self, app):
        ids = _populate(app, files=1, per_file=3)
        all_items = next(iter(ids.values()))
        before = len(app.db.all_vectors(model=speaker_core.MODEL_BRAND)[0])
        app.db.move_to_trash(all_items[:2])
        assert app.db.trash_ids() == set(all_items[:2])
        # The inventory is untouched — otherwise the way back would be a lie.
        assert len(app.db.all_vectors(model=speaker_core.MODEL_BRAND)[0]) == before

    def test_discarding_twice_is_no_error(self, app):
        ids = next(iter(_populate(app, files=1, per_file=2).values()))
        app.db.move_to_trash(ids)
        app.db.move_to_trash(ids)
        assert len(app.db.trash_ids()) == len(ids)

    def test_restoring_individually_and_wholesale(self, app):
        ids = next(iter(_populate(app, files=1, per_file=4).values()))
        app.db.move_to_trash(ids)
        app.db.restore_from_trash(ids[:1])
        assert app.db.trash_ids() == set(ids[1:])
        app.db.restore_from_trash()
        assert app.db.trash_ids() == set()

    def test_discarding_bumps_the_generation(self, app):
        """Without this the grouping's result cache keeps serving the
        old picture — with the discarded item still in it."""
        ids = next(iter(_populate(app, files=1, per_file=2).values()))
        before = app.db.generation
        app.db.move_to_trash(ids[:1])
        assert app.db.generation > before
        between_val = app.db.generation
        app.db.restore_from_trash()
        assert app.db.generation > between_val


# ── The map ──────────────────────────────────────────────────────────────


class TestTheMap:
    def test_discarded_voices_are_missing_from_the_map(self, app):
        ids = _populate(app, files=3, per_file=4)
        c = app.test_client()
        full = c.get("/api/voicemap/graph").get_json()
        assert full["voices"] == 12

        first_one = list(ids.values())[0]
        c.post("/api/voicemap/trash", json={"ids": first_one})
        inventory.discard()
        d = c.get("/api/voicemap/graph").get_json()
        assert d["voices"] == 8, "the discarded ones are still included"
        assert set(d["nodes"]["id"]).isdisjoint(first_one)

    def test_the_second_map_shows_exactly_what_was_discarded(self, app):
        ids = _populate(app, files=3, per_file=4)
        first_one = list(ids.values())[0]
        c = app.test_client()
        c.post("/api/voicemap/trash", json={"ids": first_one})
        inventory.discard()
        d = c.get("/api/voicemap/graph?trash=1").get_json()
        assert d["voices"] == len(first_one)
        assert set(d["nodes"]["id"]) == set(first_one)

    def test_an_empty_trash_says_so_instead_of_staying_silent(self, app):
        _populate(app, files=1, per_file=2)
        d = app.test_client().get("/api/voicemap/graph?trash=1").get_json()
        assert d["voices"] == 0
        assert "trash" in d["hint"].lower()

    def test_the_grouping_computes_without_the_discarded_ones(self, app):
        """The core of the design.

        Three recordings, each with its own voice, produce three groups. If
        one of them falls away, only two may remain — not three, with one
        drawn empty.
        """
        ids = _populate(app, files=3, per_file=4)
        c = app.test_client()
        before = c.get("/api/voicemap/graph").get_json()
        first_one = list(ids.values())[0]
        c.post("/api/voicemap/trash", json={"ids": first_one})
        inventory.discard()
        after = c.get("/api/voicemap/graph").get_json()
        assert len(after["groups"]) < len(before["groups"]), (
            f"{len(before['groups'])} → {len(after['groups'])}"
        )

    def test_the_counter_is_right(self, app):
        ids = next(iter(_populate(app, files=1, per_file=3).values()))
        c = app.test_client()
        assert c.get("/api/voicemap/trash").get_json()["count"] == 0
        d = c.post("/api/voicemap/trash", json={"ids": ids}).get_json()
        assert d["count"] == 3
        d = c.post("/api/voicemap/trash/restore", json={}).get_json()
        assert d["count"] == 0 and d["back"] == 3


class TestInputs:
    def test_without_voices_there_is_an_error_instead_of_silence(self, app):
        c = app.test_client()
        assert c.post("/api/voicemap/trash", json={}).status_code == 400
        assert c.post("/api/voicemap/trash", json={"ids": []}).status_code == 400

    def test_garbage_is_rejected(self, app):
        r = app.test_client().post("/api/voicemap/trash", json={"ids": ["not a number"]})
        assert r.status_code == 400


# ── Operation ────────────────────────────────────────────────────────────
#
# Text checks, no browser: the map already has a scaffold for drawing
# and computing; here it's about the wiring, and that exact wiring already
# broke silently once during a rename (commit c0e14c0).

import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
GRAPH_JS = (REPO / "whisper_tm" / "static" / "js" / "voicemap_graph.js").read_text(encoding="utf-8")
PANEL_JS = (REPO / "whisper_tm" / "static" / "js" / "voicemap_panel.js").read_text(encoding="utf-8")
INDEX = (REPO / "whisper_tm" / "static" / "index.html").read_text(encoding="utf-8")


class TestControls:
    def test_the_tile_has_a_discard_button(self):
        assert "graphDiscardGroup(" in PANEL_JS
        assert "function graphDiscardGroup" in GRAPH_JS

    def test_in_the_trash_it_says_restore_instead_of_discard(self):
        """A "discard" inside the trash would be pointless and confusing."""
        assert "graphRestoreGroup(" in PANEL_JS
        assert "GRAPH.trash" in PANEL_JS, "the button doesn't switch with the view"

    def test_voices_are_sent_not_a_group_number(self):
        """The point where a naive build would be wrong.

        Group numbers are freshly generated on every run. Between the click
        and the request arriving, a different threshold may have been
        computed — a number would then hit the wrong voices.
        """
        body = GRAPH_JS[GRAPH_JS.index("async function graphDiscardGroup") :]
        body = body[: body.index("async function graphTrashBack")]
        assert "graphGroupVoices(nr)" in body
        assert "{ids}" in body
        assert re.search(r"group_nr:\s*nr", body) is None

    def test_the_viewport_is_part_of_the_request(self):
        assert "p.set('trash', '1')" in GRAPH_JS

    def test_there_is_a_visible_bar(self):
        """Two maps that look alike and show different things are a
        trap without a label."""
        assert 'id="mapTrashBar"' in INDEX
        assert "function graphTrashBar" in GRAPH_JS
        assert "graphTrashBar()" in GRAPH_JS, "never called"

    def test_after_discarding_it_reloads(self):
        """The inventory changed, not just the picture — a mere redraw
        kept showing the discarded group."""
        body = GRAPH_JS[GRAPH_JS.index("async function graphTrashRequest") :]
        body = body[: body.index("function graphTrashView")]
        assert "mapLoadGraph()" in body


class TestRoot:
    def test_the_map_sends_its_root_along(self, app):
        """Without it the response's paths can't be resolved.

        `display()` shortens against `config.media_dir` AT LOAD TIME,
        and that shifts when the folder changes. The "Locate in recording"
        button therefore once passed on an empty hint — `goToSearchResult`
        then inserted the root of the last SEARCH, i.e. a stale
        state, and reported "recording not found".
        """
        _populate(app, files=1, per_file=2)
        d = app.test_client().get("/api/voicemap/graph").get_json()
        assert d.get("root"), "the root is missing from the response"
        assert str(app.media) == d["root"]
        # And the paths are really shortened against it.
        assert not d["file_names"][d["nodes"]["file_idx"][0]].startswith("/")
