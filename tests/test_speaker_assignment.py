"""Assign with few clicks — the method and the path to it.

The reasoning and all measurements are in the head of
`core/speaker_assignment.py`. These tests hold the three properties fixed that
everything stands on, such that a "simplifying" overhaul would overturn them:

1. **Centroids instead of windows** — otherwise you group tens of thousands
   of noisy windows instead of far fewer speakers.
2. **Complete linkage instead of connectivity** — the only reason why
   the giant blob is gone. One test rebuilds exactly the chain that the
   old method failed on.
3. **By open speaking time** — the difference between a handful of clicks
   and thousands.
"""

from __future__ import annotations

import numpy as np
import pytest
from flask import Flask

from whisper_tm.core import speaker as speaker_core
from whisper_tm.core import speaker_assignment as assignment
from whisper_tm.core import speaker_inventory as inventory
from whisper_tm.database.speaker import SpeakerDB

DIM = 192


def _voice(seed, spread=0.0, basis=None):
    r = np.random.default_rng(seed)
    v = basis if basis is not None else r.normal(size=DIM)
    if spread:
        v = v + spread * r.normal(size=DIM)
    v = v.astype(np.float32)
    return v / np.linalg.norm(v)


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
    assignment.INVENTORY.discard()

    from whisper_tm.api.speaker import speaker_bp

    a = Flask(__name__)
    a.register_blueprint(speaker_bp)
    a.db = db
    a.media = media
    yield a
    inventory.discard()
    assignment.INVENTORY.discard()


def _write(app, file, label, basis, n=4, sec=60.0, seed=0):
    app.db.write(
        str(app.media / file),
        speaker_core.MODEL_BRAND,
        [
            {
                "label": label,
                "windows": i,
                "start_sec": i * 100.0,
                "speech_sec": sec,
                "segments": 3,
                "v_separated": _voice(seed * 100 + i, 0.05, basis),
            }
            for i in range(n)
        ],
    )


class TestCentroids:
    def test_windows_become_speakers(self, app):
        """Four windows of one identifier are ONE speaker, not quadruple.

        Both identifiers in ONE `write()`: The method replaces everything for
        a recording, two calls in a row left only the second —  a trap
        this test first walked into itself.
        """
        basis, second = _voice(1), _voice(2)
        app.db.write(
            str(app.media / "a.mkv"),
            speaker_core.MODEL_BRAND,
            [
                {
                    "label": "SPEAKER_00",
                    "windows": i,
                    "start_sec": i * 100.0,
                    "speech_sec": 60.0,
                    "segments": 3,
                    "v_separated": _voice(100 + i, 0.05, basis),
                }
                for i in range(4)
            ]
            + [
                {
                    "label": "SPEAKER_01",
                    "windows": i,
                    "start_sec": i * 100.0,
                    "speech_sec": 60.0,
                    "segments": 3,
                    "v_separated": _voice(200 + i, 0.05, second),
                }
                for i in range(3)
            ],
        )
        state = assignment.INVENTORY.fetch(force=True)
        assert len(state["C"]) == 2, "Centroids not formed"
        assert set(state["windows"].tolist()) == {4, 3}

    def test_short_speakers_fall_out(self, app):
        """Below 10 seconds of speech, a centroid is not reliable."""
        _write(app, "a.mkv", "SPEAKER_00", _voice(1), n=1, sec=2.0)
        _write(app, "a.mkv", "SPEAKER_01", _voice(2), n=4, sec=60.0)
        state = assignment.INVENTORY.fetch(force=True)
        assert len(state["C"]) == 1


class TestChain:
    def test_chain_is_not_merged(self, app):
        """**The core of the whole refactoring.**

        Three voices A – B – C: A and B resemble each other, B and C resemble each other, A
        and C do not. Connected components throw all three in one pot
        — exactly this once created a single group with tens of thousands
        of voices from hundreds of recordings.

        Complete linkage must not do that: It requires that
        ALL pairs match, and A–C doesn't.
        """
        r = np.random.default_rng(5)
        a = r.normal(size=DIM)
        a /= np.linalg.norm(a)
        c = r.normal(size=DIM)
        c -= (c @ a) * a  # perpendicular to A
        c /= np.linalg.norm(c)
        b = a + c  # exactly in between
        b /= np.linalg.norm(b)
        assert abs(float(a @ c)) < 0.1, "A and C must be dissimilar"

        for i, (name, v) in enumerate((("a", a), ("b", b), ("c", c))):
            _write(app, f"{name}.mkv", "SPEAKER_00", v, n=4, seed=i + 1)
        state = assignment.INVENTORY.fetch(force=True)
        groups = {int(g) for g in state["group_nr"]}
        assert len(groups) >= 2, "A, B and C landed in ONE group — the chain lives"
        # And A and C must never meet under any circumstances.
        idx = {
            state["key"][i][0].split("/")[-1]: int(state["group_nr"][i])
            for i in range(len(state["C"]))
        }
        assert idx["a.mkv"] != idx["c.mkv"]


class TestSuggestion:
    def test_suggested_by_open_speaking_time(self, app):
        """Not the largest group, but the one with the most time."""
        large = _voice(11)
        for d in range(4):  # 4 recordings, lots of time
            _write(app, f"much{d}.mkv", "SPEAKER_00", large, n=4, sec=300.0, seed=10 + d)
        small = _voice(22)
        for d in range(9):  # 9 recordings, little time
            _write(app, f"sparse{d}.mkv", "SPEAKER_00", small, n=1, sec=20.0, seed=30 + d)
        v = assignment.suggestion()
        assert v is not None
        assert v["recordings"] == 4, "time-richer group comes first"
        assert v["speech_hours"] > 1.0

    def test_unique_voices_never_suggested(self, app):
        """On a real inventory a large share of the hours. They never return —
        naming them would be wasted time."""
        for d in range(3):
            _write(
                app, f"one_by_one{d}.mkv", "SPEAKER_00", _voice(50 + d), n=4, sec=600.0, seed=50 + d
            )
        assert assignment.suggestion() is None
        f = assignment.progress(assignment.INVENTORY.fetch())
        assert f["one_off_hours"] > 0
        assert f["target_hours"] == 0.0

    def test_one_name_spreads(self, app):
        """The actual lever: one click doesn't decide one group,
        but everything close enough to it."""
        basis = _voice(77)
        for d in range(3):
            _write(app, f"main{d}.mkv", "SPEAKER_00", basis, n=4, sec=120.0, seed=70 + d)
        # A fourth recording, somewhat further away — it should come along
        # through propagation, not through grouping.
        _write(app, "edge.mkv", "SPEAKER_00", basis + 0.35 * _voice(99), n=4, sec=120.0, seed=90)
        v = assignment.suggestion()
        result = assignment.apply(v["vector_ids"], "Alice")
        assert result["set_flag"] == len(v["vector_ids"])
        names = app.db.names_per_vector()
        assert set(names.values()) == {"Alice"}
        assert result["progress"]["share"] > 0.9


class TestEndpoints:
    def test_whole_path_over_api(self, app):
        basis = _voice(3)
        for d in range(3):
            _write(app, f"x{d}.mkv", "SPEAKER_00", basis, n=4, sec=200.0, seed=d + 1)
        c = app.test_client()
        d = c.get("/api/speaker/assignment/suggestion").get_json()
        assert d["done"] is False and d["samples"]
        r = c.post(
            "/api/speaker/assignment/apply", json={"vector_ids": d["vector_ids"], "name": "Carol"}
        )
        assert r.status_code == 200 and r.get_json()["ok"]
        d2 = c.get("/api/speaker/assignment/suggestion").get_json()
        assert d2["done"] is True

    def test_discard_instead_of_name(self, app):
        basis = _voice(4)
        for d in range(2):
            _write(app, f"tv{d}.mkv", "SPEAKER_00", basis, n=4, sec=200.0, seed=d + 20)
        c = app.test_client()
        d = c.get("/api/speaker/assignment/suggestion").get_json()
        r = c.post(
            "/api/speaker/assignment/apply", json={"vector_ids": d["vector_ids"], "trash": True}
        )
        assert r.status_code == 200
        assert len(app.db.trash_ids()) == len(d["vector_ids"])

    def test_without_names_and_without_voices_there_are_errors(self, app):
        c = app.test_client()
        assert c.post("/api/speaker/assignment/apply", json={}).status_code == 400
        assert c.post("/api/speaker/assignment/apply", json={"vector_ids": [1]}).status_code == 400


class TestSettings:
    """Does the threshold matter for the assignment, or only for the view?

    It revealed a gap: The assignment path had two
    hard-wired constants and read the configuration nowhere. A setting
    you can't see and can't change is exactly what the house rule forbids.
    """

    def test_both_values_come_from_config(self, monkeypatch):
        from whisper_tm.config import config

        assert assignment.control_value("cut") == assignment.CUT
        assert assignment.control_value("spread") == assignment.SPREAD
        monkeypatch.setitem(config._data.setdefault("speaker", {}), "assignment_cut", 0.85)
        assert assignment.control_value("cut") == 0.85

    def test_unusable_values_fall_back_to_measurement(self, monkeypatch):
        """A broken number must not cost an assignment path."""
        from whisper_tm.config import config

        for garbage in ("abc", -1, 0, 5, None):
            monkeypatch.setitem(
                config._data.setdefault("speaker", {}), "assignment_spread", garbage
            )
            assert assignment.control_value("spread") == assignment.SPREAD

    def test_changed_cut_discards_result(self, app, monkeypatch):
        """Otherwise the setting would be a sham.

        The cache only hung on `generation` — which counts changes to the INVENTORY.
        A changed cut alters grouping without touching the inventory;
        without it in the key the old image would stay."""
        from whisper_tm.config import config

        basis = _voice(31)
        for d in range(3):
            _write(
                app,
                f"s{d}.mkv",
                "SPEAKER_00",
                basis + 0.25 * _voice(40 + d),
                n=4,
                sec=100.0,
                seed=40 + d,
            )
        far = assignment.INVENTORY.fetch(force=True)["n_groups"]
        monkeypatch.setitem(config._data.setdefault("speaker", {}), "assignment_cut", 0.99)
        narrow = assignment.INVENTORY.fetch()["n_groups"]  # WITHOUT force
        assert narrow > far, f"threshold has no effect ({far} → {narrow} groups)"


class TestSettingsCompleteness:
    def test_both_controls_are_in_the_config_and_the_interface(self):
        from pathlib import Path

        repo = Path(__file__).resolve().parent.parent
        yaml = (repo / "config.default.yaml").read_text(encoding="utf-8")
        js = "".join(
            (repo / "whisper_tm" / "static" / "js" / name).read_text(encoding="utf-8")
            for name in ("settings.js", "settings_pages.js", "settings_tools.js")
        )
        for k in ("assignment_cut", "assignment_spread"):
            assert k in yaml, f"{k} missing in config.default.yaml"
            assert f"speaker.{k}" in js, f"{k} missing on the settings page"


class TestSkipping:
    """The "next" button must also turn pages — and skipping must survive.

    Can an assignment session be resumed later? When checking, it turned
    out that the "next" button didn't move anything: The skipped group was only
    noted in the browser, the server never learned of it — and it kept
    returning the same group, because it's still the largest open one.

    Skipping is done by VECTOR IDs, not group numbers:
    After each decision, regrouping happens, and `fcluster` assigns its
    numbers anew. A remembered number then pointed to a different
    group — one would skip blindly a group never seen.
    """

    @pytest.fixture
    def two_groups(self, app):
        """Two people, each in two recordings — so two recurring
        groups, and the first is the larger."""
        a, b = _voice(11), _voice(22)
        for file in ("one.mkv", "two.mkv"):
            _write(app, file, "SPEAKER_00", a, n=6, sec=90.0, seed=1)
        for file in ("three.mkv", "four.mkv"):
            _write(app, file, "SPEAKER_00", b, n=3, sec=40.0, seed=2)
        assignment.INVENTORY.discard()
        return app

    def test_without_skipped_ones_the_next_one_comes(self, two_groups):
        first = assignment.suggestion()
        assert first is not None
        anchor = first["samples"][0]["vector_id"]
        second_val = assignment.suggestion(without={anchor})
        assert second_val is not None, "the second group fell through the cracks"
        assert second_val["group_nr"] != first["group_nr"], (
            "the skipped group comes back right away"
        )

    def test_the_anchor_may_be_any_voice_of_the_group(self, two_groups):
        """Not just the first sample — otherwise everything would hinge on the sort order."""
        first = assignment.suggestion()
        for v in first["vector_ids"][:5]:
            n = assignment.suggestion(without={v})
            assert n is None or n["group_nr"] != first["group_nr"]

    def test_everything_skipped_means_done(self, two_groups):
        all_items: set[int] = set()
        for _ in range(10):
            v = assignment.suggestion(without=all_items)
            if v is None:
                break
            before = len(all_items)
            all_items.update(v["vector_ids"])
            assert len(all_items) > before, "infinite loop: nothing was added"
        else:
            pytest.fail("never reaches the end")
        assert all_items, "there was nothing from the start"

    def test_the_normal_case_stays_untouched(self, two_groups):
        assert (
            assignment.suggestion(without=None)["group_nr"]
            == assignment.suggestion(without=set())["group_nr"]
        )

    def test_the_endpoint_passes_the_list_through(self, two_groups):
        c = two_groups.test_client()
        first = c.get("/api/speaker/assignment/suggestion").get_json()
        anchor = first["samples"][0]["vector_id"]
        second_val = c.get(f"/api/speaker/assignment/suggestion?without={anchor}").get_json()
        assert second_val.get("done") or second_val["group_nr"] != first["group_nr"]

    def test_nonsense_in_the_parameter_does_not_throw(self, two_groups):
        """A broken entry in browser storage must not block the way."""
        c = two_groups.test_client()
        r = c.get("/api/speaker/assignment/suggestion?without=abc,,-1,%20,7x")
        assert r.status_code == 200

    def test_the_frontend_remembers_voices_not_numbers(self):
        from pathlib import Path

        js = (
            Path(__file__).resolve().parent.parent
            / "whisper_tm"
            / "static"
            / "js"
            / "voicemap_assign.js"
        ).read_text(encoding="utf-8")
        body = js[js.index("if (kind === 'skip')") :]
        body = body[: body.index("const body =")]
        assert "vector_id" in body, "remembers the group number"
        assert "add(v.group_nr)" not in body

    def test_the_frontend_survives_a_reload(self):
        from pathlib import Path

        js = (
            Path(__file__).resolve().parent.parent
            / "whisper_tm"
            / "static"
            / "js"
            / "voicemap_assign.js"
        ).read_text(encoding="utf-8")
        assert "localStorage.setItem(ASSIGN_MEMO" in js
        assert "assignLoadMemo()" in js
        # … and there is a way back, otherwise the list is a trap.
        assert "function assignSkippedBack" in js
        assert "assignSkippedBack()" in js


class TestOrderings:
    """Several sort methods for the assignment — by hours, by speaker,
    by recordings.

    Three questions on the same inventory, and they lead to different
    answers:

    * **hours** — where is the most open speaking time?
    * **speakers** — where do the most clusters collapse?
    * **recordings** — who appears in the most recordings (regulars)?

    All three count only the OPEN: what already has a name weighs nothing
    more. Otherwise a long-completed group would stay at the top forever.
    """

    @pytest.fixture
    def two_kinds(self, app):
        """Two groups with opposing profiles — otherwise the test checks
        nothing, because any sort order would give the same answer.

        * A: ONE long recording, two speakers, a lot of time
        * B: six short recordings, six speakers, little time
        """
        a, b = _voice(31), _voice(77)
        for file in ("long1.mkv", "long2.mkv"):
            _write(app, file, "SPEAKER_00", a, n=8, sec=300.0, seed=3)
        for i in range(6):
            _write(app, f"short{i}.mkv", "SPEAKER_00", b, n=2, sec=30.0, seed=4)
        assignment.INVENTORY.discard()
        return app

    def test_by_hours_the_long_recording_wins(self, two_kinds):
        v = assignment.suggestion(sort_by="hours")
        assert v["recordings"] == 2 and v["speech_hours"] > 1

    def test_by_recordings_the_regulars_win(self, two_kinds):
        """The real signal: fewer hours, but in six recordings — that
        is the voice running through the whole collection."""
        v = assignment.suggestion(sort_by="recordings")
        assert v["recordings"] == 6
        assert v["speech_hours"] < assignment.suggestion(sort_by="hours")["speech_hours"]

    def test_by_speakers_the_centroids_count(self, two_kinds):
        v = assignment.suggestion(sort_by="speaker")
        assert v["speaker"] == 6

    def test_the_answer_says_what_it_sorted_by(self, two_kinds):
        """Without this, the UI cannot show what it is displaying."""
        for kind in assignment.ORDERINGS:
            assert assignment.suggestion(sort_by=kind)["sort_by"] == kind

    def test_nonsense_falls_back_to_hours(self, two_kinds):
        assert assignment.suggestion(sort_by="nonsense")["sort_by"] == "hours"
        assert assignment.suggestion(sort_by=None)["sort_by"] == "hours"

    def test_on_a_tie_the_speech_time_decides(self, app):
        """ "By recordings" has dozens of groups with the same count. Without
        a second key the order would be arbitrary — and it
        would switch between two calls for no reason."""
        for name, basis, sec in (("x", _voice(41), 200.0), ("y", _voice(88), 40.0)):
            for i in range(3):
                _write(
                    app,
                    f"{name}{i}.mkv",
                    "SPEAKER_00",
                    basis,
                    n=2,
                    sec=sec,
                    seed=5 if name == "x" else 6,
                )
        assignment.INVENTORY.discard()
        v = assignment.suggestion(sort_by="recordings")
        assert v["recordings"] == 3
        # Both have three recordings — the heavier one must win.
        assert v["speech_hours"] > 0.3, v

    def test_the_endpoint_passes_the_choice_through(self, two_kinds):
        c = two_kinds.test_client()
        for kind in assignment.ORDERINGS:
            d = c.get(f"/api/speaker/assignment/suggestion?sort_by={kind}").get_json()
            assert d["sort_by"] == kind
        d = c.get("/api/speaker/assignment/suggestion?sort_by=<script>").get_json()
        assert d["sort_by"] == "hours"

    def test_the_interface_offers_all_three(self):
        from pathlib import Path

        js = (
            Path(__file__).resolve().parent.parent
            / "whisper_tm"
            / "static"
            / "js"
            / "voicemap_assign.js"
        ).read_text(encoding="utf-8")
        for kind in assignment.ORDERINGS:
            assert f"'{kind}'" in js, f"{kind} missing in the UI"
        assert "assignAfterSet" in js
        # … and the choice survives reloading.
        assert "ASSIGN_MEMO + '.sort_by'" in js


class TestWhatHelpsDeciding:
    """More statistics — which named people the voice group is closest to."""

    def test_the_nearest_named_people(self, app):
        alice, bob = _voice(201), _voice(202)
        for d in range(2):
            _write(app, f"alice{d}.mkv", "SPEAKER_00", alice, n=4, sec=200.0, seed=200 + d)
            _write(app, f"bob{d}.mkv", "SPEAKER_00", bob, n=4, sec=200.0, seed=210 + d)
        first = assignment.suggestion()
        assignment.apply(first["vector_ids"], "Alice")
        # The other group is the next question, and it isn't Alice
        for d in range(2):
            _write(
                app,
                f"alice_more{d}.mkv",
                "SPEAKER_00",
                alice + 0.4 * _voice(299),
                n=4,
                sec=100.0,
                seed=230 + d,
            )
        v = assignment.suggestion()
        assert v["nearest"] and v["nearest"][0]["name"] == "Alice"
        assert {"name", "sim", "votes"} <= set(v["nearest"][0])
        assert v["nearest"][0]["votes"] == v["speaker"]

    def test_without_names_nobody_is_near(self, app):
        basis = _voice(5)
        for d in range(2):
            _write(app, f"y{d}.mkv", "SPEAKER_00", basis, n=4, sec=200.0, seed=d + 40)
        v = assignment.suggestion()
        assert v["nearest"] == []

    def test_the_stats(self, app):
        basis = _voice(6)
        for d in range(3):
            _write(
                app,
                f"2024-0{d + 1}-05 talk.mkv",
                "SPEAKER_00",
                basis,
                n=4,
                sec=100.0 + d,
                seed=d + 50,
            )
        s = assignment.suggestion()["stats"]
        assert (s["first_date"], s["last_date"], s["dated"]) == ("2024-01-05", "2024-03-05", 3)
        assert s["folders"][0]["share"] == 1.0
        assert s["windows"] == 12 and s["speech_max_sec"] >= s["speech_median_sec"] > 0

    def test_more_samples_than_eight(self, app):
        basis = _voice(7)
        for d in range(12):
            _write(app, f"z{d}.mkv", "SPEAKER_00", basis, n=4, sec=200.0, seed=d + 60)
        v = assignment.suggestion()
        assert len(v["samples"]) == v["speaker"] == 12
