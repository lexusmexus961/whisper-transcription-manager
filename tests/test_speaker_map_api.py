"""Map endpoints, inventory query and process cache.

Tested is the **effect**: after a write operation, the next
query must see the new state, and a cap must identify itself as a cap
instead of pretending everything is fine.
"""

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
    a.config["TESTING"] = False
    a.db = db
    a.media = media
    yield a
    inventory.discard()


def _stock(app, files=3, each=4):
    """Each file has its own speaker, all windows close together."""
    ids = {}
    for d in range(files):
        base = _vec(100 + d)
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
                    "v_separated": base + 0.02 * _vec(1000 + d * 10 + i),
                }
                for i in range(each)
            ],
        )
        ids[path] = [z["id"] for z in app.db.windows(path)]
    return ids


# ── Inventory query: where does a person appear ──────────────────────────


def test_person_is_found_in_their_recordings(app):
    ids = _stock(app)
    paths = sorted(ids)
    # Name Alice — but only ONE voice from the first recording
    app.db.assign(ids[paths[0]][0], "Alice")

    d = app.test_client().get("/api/speaker/person/Alice").get_json()
    assert d["person"] == "Alice"
    assert d["found"] == 1, d
    a = d["recordings"][0]
    assert a["abs_path"] == paths[0]
    assert a["display"] == "recording0.mkv"  # without the media root
    assert a["windows"] == 4
    assert a["timestamps"] == [0, 5, 10, 15]  # Minutes
    assert a["labels"] == ["SPEAKER_00"]


def test_unknown_person_gives_empty_answer(app):
    _stock(app)
    d = app.test_client().get("/api/speaker/person/Nobody").get_json()
    assert d["found"] == 0 and d["recordings"] == []


def test_threshold_affects_inventory_query(app):
    """The gallery is in its OWN file — otherwise the voice
    finds itself with cosine 1.0 and no threshold applies."""
    _stock(app)
    gallery = str(app.media / "gallery.opus")
    app.db.write(
        gallery,
        speaker_core.MODEL_BRAND,
        [
            {
                "label": "SPEAKER_00",
                "windows": 1,
                "start_sec": 0.0,
                "speech_sec": 30.0,
                "segments": 1,
                "v_separated": _vec(100) + 0.25 * _vec(555),
            }
        ],
    )
    app.db.assign(app.db.windows(gallery)[0]["id"], "Alice")

    c = app.test_client()
    strict = c.get("/api/speaker/person/Alice?threshold=0.99").get_json()
    loose = c.get("/api/speaker/person/Alice?threshold=0.5").get_json()
    assert strict["found"] < loose["found"]
    assert loose["found"] >= 2  # gallery file and recording0


# ── Process cache ───────────────────────────────────────────────────────


def test_cache_sees_new_vectors(app):
    """A concurrent transcription pushes vectors in — the
    next request must see them."""
    _stock(app, files=1)
    assert len(inventory.inventory()["X"]) == 4
    app.db.write(
        str(app.media / "new_value.mkv"),
        speaker_core.MODEL_BRAND,
        [
            {
                "label": "SPEAKER_00",
                "windows": 1,
                "start_sec": 0.0,
                "speech_sec": 30.0,
                "segments": 1,
                "v_separated": _vec(9),
            }
        ],
    )
    assert len(inventory.inventory()["X"]) == 5


def test_cache_sees_new_names(app):
    ids = _stock(app, files=1)
    assert inventory.inventory()["gallery"] == {}
    app.db.assign(list(ids.values())[0][0], "Alice")
    assert "Alice" in inventory.inventory()["gallery"]


def test_cache_holds_with_unchanged_state(app):
    _stock(app, files=1)
    a = inventory.inventory()
    b = inventory.inventory()
    assert a is b, "unchanged should NOT trigger reload"


def test_matrix_is_normalized(app):
    _stock(app, files=2)
    X = inventory.inventory()["X"]
    assert np.allclose(np.linalg.norm(X, axis=1), 1.0, atol=1e-4)


# ── Map: Overview and group ──────────────────────────────────────────────


def test_statistics_names_all_three_thresholds(app):
    d = app.test_client().get("/api/speaker/statistics").get_json()
    b = d["operating_point"]
    assert b["threshold_across_files"] < b["threshold_within_file"] < b["threshold_gallery"]


# ── Map as GRAPH ────────────────────────────────────────────────────────
#
# The bubble view draws one bubble per group — with 200,000 voices the
# only viable way, with about a thousand it throws away structure and
# leaves nothing to see.
# So `/api/voicemap/graph`: each voice a node, edges = similarity.


class TestMapGraph:
    def test_each_voice_becomes_a_node(self, app):
        ids = _stock(app, files=3, each=4)
        d = app.test_client().get("/api/voicemap/graph").get_json()

        assert d["voices"] == 12
        assert d["files"] == 3
        assert len(d["nodes"]["id"]) == 12
        # Columns instead of objects — all same length, otherwise color doesn't
        # match the point.
        lengths = {len(v) for v in d["nodes"].values()}
        assert lengths == {12}, d["nodes"].keys()
        all_ids = {i for id_lists in ids.values() for i in id_lists}
        assert set(d["nodes"]["id"]) == all_ids

    def test_edges_appear_only_once(self, app):
        """Drawing each line twice costs half the frame rate.

        The kNN run provides neighborhoods in BOTH directions; the
        endpoint must merge them.
        """
        _stock(app, files=3, each=4)
        d = app.test_client().get("/api/voicemap/graph").get_json()

        pairs = [(a, b) for a, b, _ in d["edges"]]
        assert len(pairs) == len(set(pairs)), "duplicate edges"
        assert all(a < b for a, b in pairs), "edges not normalized (a<b)"

    def test_no_edge_below_threshold(self, app):
        _stock(app, files=3, each=4)
        d = app.test_client().get("/api/voicemap/graph?threshold=0.9").get_json()
        assert all(w >= 0.9 for _, _, w in d["edges"]), "edge below threshold"

    def test_k_caps_edge_count(self, app):
        """**The reason for kNN.** With about a thousand voices, all edges above
        threshold run into the hundreds of thousands; with k=5 a few
        thousand remain.

        A group with hundreds of nearly identical-sounding voices is almost
        completely connected — drawn a ball of yarn, in which there's nothing
        to see anyway.
        """
        _stock(app, files=4, each=8)  # 32 very similar voices
        c = app.test_client()
        few = c.get("/api/voicemap/graph?k=2&threshold=0.1").get_json()
        many = c.get("/api/voicemap/graph?k=8&threshold=0.1").get_json()

        assert len(few["edges"]) <= 32 * 2
        assert len(few["edges"]) < len(many["edges"]), "k has no effect on edge count"

    def test_group_inherits_most_frequent_name(self, app):
        """Not the group number names the group, but its people.

        Numbers move when sliding the threshold, names don't — and the
        color hangs on the name, so "Bob" doesn't change color at every
        slider step.
        """
        ids = _stock(app, files=2, each=4)
        first = ids[sorted(ids)[0]]
        app.db.assign(first[0], "Bob")
        app.db.assign(first[1], "Bob")
        app.db.assign(first[2], "Carol")

        d = app.test_client().get("/api/voicemap/graph?threshold=0.5").get_json()
        named = [g for g in d["groups"] if g["name"]]
        assert named, "no group got a name"
        bob = next(g for g in named if g["name"] == "Bob")
        # Both names are reported — multiple names in a group are
        # the interesting case, not the one to hide.
        assert dict(bob["people"])["Bob"] == 2
        assert dict(bob["people"]).get("Carol") == 1

    def test_groups_come_by_size(self, app):
        _stock(app, files=3, each=5)
        d = app.test_client().get("/api/voicemap/graph?threshold=0.5").get_json()
        sizes = [g["n"] for g in d["groups"]]
        assert sizes == sorted(sizes, reverse=True)

    def test_empty_storage_says_so_instead_of_crashing(self, app):
        d = app.test_client().get("/api/voicemap/graph").get_json()
        assert d["voices"] == 0
        assert "hint" in d

    def test_nonsensical_values_are_clamped(self, app):
        """A slider can send anything; nothing should crash."""
        _stock(app, files=2, each=3)
        c = app.test_client()
        assert c.get("/api/voicemap/graph?threshold=99").get_json()["threshold"] <= 1.0
        assert c.get("/api/voicemap/graph?threshold=-5").get_json()["threshold"] >= 0.0
        assert c.get("/api/voicemap/graph?k=9999").get_json()["k"] <= 20
        assert (
            c.get("/api/voicemap/graph?link_method=nonsense").get_json()["link_method"] == "average"
        )

    def test_display_path_omits_media_root(self, app):
        _stock(app, files=1, each=2)
        d = app.test_client().get("/api/voicemap/graph").get_json()
        assert d["file_names"][d["nodes"]["file_idx"][0]] == "recording0.mkv"

    def test_file_paths_are_sent_once_per_recording(self, app):
        """Hundreds of thousands of voices come from tens of thousands of
        recordings — repeating the path per voice made up almost half the
        response. Paths go in a list, nodes index it."""
        _stock(app, files=3, each=4)
        d = app.test_client().get("/api/voicemap/graph").get_json()
        assert len(d["file_names"]) == 3 == d["files"]
        assert len(set(d["nodes"]["file_idx"])) == 3
        assert "speech_sec" in d["nodes"] and "language" not in d["nodes"]
