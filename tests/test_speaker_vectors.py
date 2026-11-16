"""Voice vectors: window formation, storage and the question "who was there".

All numbers tested here are measurement results, documented at the place
they are defined in :mod:`whisper_tm.core.speaker`. The tests hold
the **operating point** fixed — not the implementation. If someone later
touches MIN_SEGMENT, it should be obvious here that a measurement is being
circumvented.

No test needs a GPU or a model: calculations are done on invented
vectors, logic around them is measured.
"""

import numpy as np
import pytest

from whisper_tm.core import speaker as sp
from whisper_tm.database.speaker import SpeakerDB, as_blob, from_blob

# ── Reading diarization ───────────────────────────────────────────────


def test_turns_from_list():
    turns = sp.normalize_turns(
        [
            {"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00"},
        ]
    )
    assert turns == [{"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00"}]


def test_turns_from_dataframe():
    """WhisperX delivers a DataFrame, tools work with lists — both must work."""
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame(
        [{"start": 1.0, "end": 2.0, "speaker": "A"}, {"start": 2.0, "end": 4.0, "speaker": "B"}]
    )
    turns = sp.normalize_turns(df)
    assert [t["speaker"] for t in turns] == ["A", "B"]
    assert turns[1]["end"] == 4.0


def test_turns_without_diarization():
    assert sp.normalize_turns(None) == []


# ── Clean spans ────────────────────────────────────────────────────────


def test_overlap_falls_out():
    """Where two people speak, no vector is created.

    With full overlap, the vector becomes simply the louder speaker
    (0.78 vs 0.22, Cord-Landwehr et al.) — the segment would then have
    the wrong label.
    """
    turns = [
        {"start": 0.0, "end": 10.0, "speaker": "A"},
        {"start": 5.0, "end": 15.0, "speaker": "B"},
    ]
    spans = sp.clean_spans(turns)
    assert spans["A"] == [(0.0, 5.0)]
    assert spans["B"] == [(10.0, 15.0)]


def test_adjacent_is_merged():
    """Otherwise continuous speech would split at every section boundary into
    snippets — and then fail the 4-second threshold."""
    turns = [{"start": 0.0, "end": 5.0, "speaker": "A"}, {"start": 5.0, "end": 9.0, "speaker": "A"}]
    assert sp.clean_spans(turns)["A"] == [(0.0, 9.0)]


def test_tiny_spans_ignored():
    turns = [
        {"start": 0.0, "end": 0.05, "speaker": "A"},
        {"start": 1.0, "end": 6.0, "speaker": "B"},
    ]
    spans = sp.clean_spans(turns)
    assert "A" not in spans


# ── Windows ────────────────────────────────────────────────────────────


def test_windows_are_thirty_seconds():
    """30 s is the measured saturation point, not 15 and not 60."""
    windows = sp.build_windows([(0.0, 95.0)])
    assert len(windows) == 4  # 3 full + one remainder of 5 s
    for _parts, duration, _start in windows[:3]:
        assert duration == pytest.approx(30.0)
    assert windows[3][1] == pytest.approx(5.0)


def test_remainder_window_only_above_minimum_duration():
    """A 2-second remainder produces no vector — at 2 s the
    EER is 8.67% vs 0.00% at 30 s."""
    assert len(sp.build_windows([(0.0, 32.0)])) == 1
    assert len(sp.build_windows([(0.0, 35.0)])) == 2


def test_windows_carry_their_timestamp():
    """The timestamp answers the second question: WHERE approximately."""
    windows = sp.build_windows([(100.0, 200.0)])
    assert windows[0][2] == pytest.approx(100.0)
    assert windows[1][2] == pytest.approx(130.0)


def test_glued_from_many_sections():
    """Gluing is the normal case and an advantage (1.48% vs 4.41% EER)."""
    spans = [(i * 10.0, i * 10.0 + 5.0) for i in range(10)]  # 10 × 5 s
    windows = sp.build_windows(spans)
    parts, duration, _ = windows[0]
    assert duration == pytest.approx(30.0)
    assert len(parts) == 6  # six seams for 30 seconds


def test_minimum_length_filters_short_interjections():
    """Short segments must not contribute.

    Here there are enough long sections for a full window, so the gate applies —
    the 1-second snippets stay out.
    """
    long = [(i * 20.0, i * 20.0 + 8.0) for i in range(5)]  # 5 × 8 s
    short = [(i * 20.0 + 10.0, i * 20.0 + 11.0) for i in range(5)]  # 5 × 1 s
    windows = sp.build_windows(sorted(long + short))
    for parts, _duration, _start in windows:
        for a, b in parts:
            assert b - a >= 1.0
            # each piece comes from a section of at least 4 s
            assert any(qa <= a and b <= qb + 1e-6 for qa, qb in long)


def test_fallback_when_not_enough_long_segments():
    """Fallback is essential: otherwise quiet speakers would be lost
    entirely — and those are exactly the ones to find."""
    snippet = [(i * 3.0, i * 3.0 + 1.5) for i in range(50)]  # only 1.5 s
    windows = sp.build_windows(snippet)
    assert len(windows) == 3  # 75 s speech → 2 full windows + remainder
    assert sum(d for _t, d, _s in windows) == pytest.approx(75.0)


def test_cap_thins_out_evenly():
    """If capped, then distributed — "always the beginning" was the error
    that masked an entire person."""
    windows = sp.build_windows([(0.0, 3000.0)], at_most=5)
    assert len(windows) == 5
    starts = [s for _t, _d, s in windows]
    assert starts[-1] > 2000  # the last one comes from the end, not the beginning


def test_no_cap_is_the_operating_point():
    """Complete rather than sampling — an explicit requirement."""
    assert sp.MAX_WINDOWS == 0
    assert len(sp.build_windows([(0.0, 3000.0)])) == 100


# ── Plan ─────────────────────────────────────────────────────────────────


def test_plan_is_sorted_by_time():
    turns = [
        {"start": 0.0, "end": 40.0, "speaker": "B"},
        {"start": 40.0, "end": 120.0, "speaker": "A"},
    ]
    plan = sp.plan_windows(turns)
    assert [p["start_sec"] for p in plan] == sorted(p["start_sec"] for p in plan)
    assert {p["label"] for p in plan} == {"A", "B"}
    assert all(p["speech_sec"] >= sp.MIN_SPEECH for p in plan)


def test_plan_without_usable_speech():
    turns = [{"start": 0.0, "end": 1.0, "speaker": "A"}]
    assert sp.plan_windows(turns) == []


# ── Storage ────────────────────────────────────────────────────────────────


@pytest.fixture
def db(tmp_path):
    d = SpeakerDB(tmp_path / "speaker.db")
    d.initialize()
    return d


def _vec(seed, dim=192):
    rng = np.random.default_rng(seed)
    return rng.normal(size=dim).astype(np.float32)


def test_blob_there_and_back():
    v = _vec(1)
    assert np.allclose(from_blob(as_blob(v)), v)
    assert as_blob(None) is None


def test_write_and_read(db):
    db.write(
        "/m/a.mkv",
        sp.MODEL_BRAND,
        [
            {
                "label": "SPEAKER_00",
                "windows": 1,
                "start_sec": 0.0,
                "speech_sec": 30.0,
                "segments": 6,
                "v_separated": _vec(1),
            },
            {
                "label": "SPEAKER_01",
                "windows": 1,
                "start_sec": 90.0,
                "speech_sec": 12.0,
                "segments": 2,
                "v_separated": _vec(2),
            },
        ],
    )
    rows = db.windows("/m/a.mkv")
    assert len(rows) == 2
    assert rows[0]["label"] == "SPEAKER_00"  # sorted by time
    assert rows[0]["model"] == sp.MODEL_BRAND
    assert from_blob(rows[0]["v_separated"]).shape == (192,)
    assert db.has_vectors("/m/a.mkv", sp.MODEL_BRAND)
    assert not db.has_vectors("/m/b.mkv")


def test_second_run_replaces_instead_of_mixing(db):
    """A new run has different window numbers — mixing old and
    new state would be silently wrong."""
    db.write(
        "/m/a.mkv",
        sp.MODEL_BRAND,
        [
            {
                "label": "SPEAKER_00",
                "windows": i,
                "start_sec": i * 30.0,
                "speech_sec": 30.0,
                "segments": 1,
                "v_separated": _vec(i),
            }
            for i in range(1, 6)
        ],
    )
    db.write(
        "/m/a.mkv",
        sp.MODEL_BRAND,
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
    assert len(db.windows("/m/a.mkv")) == 1


def test_different_checkpoint_remains(db):
    """A model change must leave old vectors for comparison."""
    row = [
        {
            "label": "SPEAKER_00",
            "windows": 1,
            "start_sec": 0.0,
            "speech_sec": 30.0,
            "segments": 1,
            "v_separated": _vec(1),
        }
    ]
    db.write("/m/a.mkv", "redimnet_m/vb2-ptn", row)
    db.write("/m/a.mkv", "redimnet_m/ft_mix", row)
    assert len(db.windows("/m/a.mkv")) == 2
    assert len(db.windows("/m/a.mkv", model="redimnet_m/vb2-ptn")) == 1


def test_names_stick_to_vector(db):
    """Never to a group number — that changes with merging and
    threshold; an assigned name must not."""
    db.write(
        "/m/a.mkv",
        sp.MODEL_BRAND,
        [
            {
                "label": "SPEAKER_00",
                "windows": 1,
                "start_sec": 0.0,
                "speech_sec": 30.0,
                "segments": 1,
                "v_separated": _vec(1),
            }
        ],
    )
    vid = db.windows("/m/a.mkv")[0]["id"]
    db.assign(vid, "Alice")
    assert db.windows("/m/a.mkv")[0]["person"] == "Alice"
    assert [p["name"] for p in db.people()] == ["Alice"]
    assert db.people()[0]["voices"] == 1
    db.assign(vid, None)
    assert db.windows("/m/a.mkv")[0]["person"] is None


def test_person_vectors_and_statistics(db):
    for i, name in enumerate(["Alice", "Alice", "Bob"], start=1):
        db.write(
            f"/m/{i}.mkv",
            sp.MODEL_BRAND,
            [
                {
                    "label": "SPEAKER_00",
                    "windows": 1,
                    "start_sec": 0.0,
                    "speech_sec": 30.0,
                    "segments": 1,
                    "v_separated": _vec(i),
                }
            ],
        )
        db.assign(db.windows(f"/m/{i}.mkv")[0]["id"], name)
    pv = db.person_vectors()
    assert pv["Alice"].shape == (2, 192)
    assert pv["Bob"].shape == (1, 192)
    s = db.stats_data()
    assert s["windows"] == 3 and s["files"] == 3 and s["people"] == 2


def test_all_vectors_as_matrix(db):
    db.write(
        "/m/a.mkv",
        sp.MODEL_BRAND,
        [
            {
                "label": "SPEAKER_00",
                "windows": i,
                "start_sec": i * 30.0,
                "speech_sec": 30.0,
                "segments": 1,
                "v_separated": _vec(i),
            }
            for i in range(1, 4)
        ],
    )
    X, meta = db.all_vectors()
    assert X.shape == (3, 192) and len(meta) == 3
    assert meta[0]["abs_path"] == "/m/a.mkv"


def test_raw_version_as_fallback(db):
    """A review run without separation fills only v_raw — evaluation must
    not stumble over it."""
    db.write(
        "/m/a.mkv",
        sp.MODEL_BRAND,
        [
            {
                "label": "SPEAKER_00",
                "windows": 1,
                "start_sec": 0.0,
                "speech_sec": 30.0,
                "segments": 1,
                "v_raw": _vec(1),
            }
        ],
    )
    X, _ = db.all_vectors(preferred="v_separated")
    assert X.shape == (1, 192)


# ── Window separation instead of track separation ────────────────────────


def test_separator_called_per_window(monkeypatch):
    """The third way, when the user doesn't want Vocal Isolation.

    Instead of just warning, each 30-second window is separated individually.
    For each window and track, the vector comes out essentially the same
    (0.984/0.994, 22.7) — it's purely a time question, and 30 seconds
    behaves differently than seven hours.
    """
    import torch

    calls = []

    def fake_separator(wave):
        calls.append(wave.shape)
        return wave

    monkeypatch.setattr(sp, "load_model", lambda device="cuda", job=None: lambda w: _vec(1))
    audio = np.zeros(120 * sp.SR, dtype=np.float32)
    plan = sp.plan_windows([{"start": 0.0, "end": 120.0, "speaker": "A"}])
    rows = sp.compute_vectors(audio, plan, device="cpu", separate=fake_separator)
    assert len(calls) == len(rows) == 4  # 120 s → 4 windows
    assert calls[0] == torch.Size([1, 30 * sp.SR])


def test_without_separator_stays_raw(monkeypatch):
    monkeypatch.setattr(sp, "load_model", lambda device="cuda", job=None: lambda w: _vec(1))
    audio = np.zeros(60 * sp.SR, dtype=np.float32)
    plan = sp.plan_windows([{"start": 0.0, "end": 60.0, "speaker": "A"}])
    rows = sp.compute_vectors(audio, plan, device="cpu", column_name="v_raw")
    assert all("v_raw" in z and "v_separated" not in z for z in rows)


def test_window_separation_marked_as_separated(monkeypatch, tmp_path):
    """When window separation kicks in, the result is the separated
    version — otherwise the database would have something different in it."""
    from whisper_tm.database.speaker import SpeakerDB

    db = SpeakerDB(tmp_path / "speaker.db")
    db.initialize()
    monkeypatch.setattr("whisper_tm.database.speaker.speaker_db", lambda: db)
    monkeypatch.setattr(sp, "load_model", lambda device="cuda", job=None: lambda w: _vec(1))
    monkeypatch.setattr(sp, "window_separator", lambda work, job=None: lambda w: w)

    audio = np.zeros(60 * sp.SR, dtype=np.float32)
    turns = [{"start": 0.0, "end": 60.0, "speaker": "A"}]
    b = sp.carry_along(
        "/m/a.mkv",
        audio,
        turns,
        device="cpu",
        separated=False,
        work=tmp_path / "work",
        window_separation=True,
    )
    assert b["variant"] == "separated"
    assert db.windows("/m/a.mkv")[0]["v_separated"] is not None


def test_without_separation_nothing_is_written(monkeypatch, tmp_path):
    """The invariant: better no vector than a raw one.

    The rule: when vectors are created, Vocal Isolation must be on —
    otherwise 5 files have it on, the next 5 off, and nothing matches.
    Exactly what would happen: a raw and a separated vector from the same
    person lie further apart than two separated vectors from different people.
    """
    from whisper_tm.database.speaker import SpeakerDB

    db = SpeakerDB(tmp_path / "speaker.db")
    db.initialize()
    monkeypatch.setattr("whisper_tm.database.speaker.speaker_db", lambda: db)
    monkeypatch.setattr(sp, "load_model", lambda device="cuda", job=None: lambda w: _vec(1))
    audio = np.zeros(60 * sp.SR, dtype=np.float32)
    b = sp.carry_along(
        "/m/a.mkv",
        audio,
        [{"start": 0.0, "end": 60.0, "speaker": "A"}],
        device="cpu",
        separated=False,
        window_separation=False,
    )
    assert b["windows"] == 0
    assert "without separation" in b["reason"]
    assert db.windows("/m/a.mkv") == []


def test_lock_only_lifted_deliberately(monkeypatch, tmp_path):
    """`only_separated=False` is exclusively for measurement runs."""
    from whisper_tm.database.speaker import SpeakerDB

    db = SpeakerDB(tmp_path / "speaker.db")
    db.initialize()
    monkeypatch.setattr("whisper_tm.database.speaker.speaker_db", lambda: db)
    monkeypatch.setattr(sp, "load_model", lambda device="cuda", job=None: lambda w: _vec(1))
    audio = np.zeros(60 * sp.SR, dtype=np.float32)
    b = sp.carry_along(
        "/m/a.mkv",
        audio,
        [{"start": 0.0, "end": 60.0, "speaker": "A"}],
        device="cpu",
        separated=False,
        window_separation=False,
        only_separated=False,
    )
    assert b["variant"] == "raw"
    assert db.windows("/m/a.mkv")[0]["v_raw"] is not None
    assert db.windows("/m/a.mkv")[0]["v_separated"] is None
    # and storage makes the mixture visible
    assert db.stats_data()["revision_raw"] == 2


def test_failed_window_separation_invalidates_run(monkeypatch):
    """A missing window is a hole; a mislabeled one is poison.

    If window separation falls back to raw audio, a raw
    vector lands under the label "separated" — and no one sees it
    afterwards. Previously a hole remained (2 of 4 windows) and
    the run was considered successful. But a run with holes is not a valid
    run — it fails, and no one writes half an inventory.
    """
    monkeypatch.setattr(sp, "load_model", lambda device="cuda", job=None: lambda w: _vec(1))
    audio = np.zeros(120 * sp.SR, dtype=np.float32)
    plan = sp.plan_windows([{"start": 0.0, "end": 120.0, "speaker": "A"}])
    assert len(plan) == 4
    # every second window fails
    counter = {"n": 0}

    def half(wave):
        counter["n"] += 1
        return wave if counter["n"] % 2 else None

    with pytest.raises(RuntimeError, match="2 of 4 windows"):
        sp.compute_vectors(audio, plan, device="cpu", separate=half)


# ── Who was there ──────────────────────────────────────────────────────


@pytest.fixture
def own_db(tmp_path, monkeypatch):
    """Redirect speaker_db() to a disposable file.

    Do NOT use the real database: ``config.db_path`` would point to
    ``~/.local/share/whisper-tm/`` — the user's installation.
    """
    d = SpeakerDB(tmp_path / "speaker.db")
    d.initialize()
    monkeypatch.setattr("whisper_tm.database.speaker.speaker_db", lambda: d)
    return d


def test_who_was_there_finds_the_right_ones(own_db):
    """The core requirement: WHO was involved and WHERE approximately."""
    alice, bob, foreign = _vec(10), _vec(11), _vec(12)

    # Gallery: two named voices each from other recordings
    for i, (name, basis) in enumerate(
        [("Alice", alice), ("Alice", alice), ("Bob", bob), ("Stranger", foreign)]
    ):
        path = f"/m/gallery{i}.opus"
        own_db.write(
            path,
            sp.MODEL_BRAND,
            [
                {
                    "label": "SPEAKER_00",
                    "windows": 1,
                    "start_sec": 0.0,
                    "speech_sec": 30.0,
                    "segments": 1,
                    "v_separated": basis + 0.05 * _vec(100 + i),
                }
            ],
        )
        own_db.assign(own_db.windows(path)[0]["id"], name)

    # The recording: Alice at minute 0 and 10, Bob at minute 20
    own_db.write(
        "/m/gaming.mkv",
        sp.MODEL_BRAND,
        [
            {
                "label": "SPEAKER_00",
                "windows": 1,
                "start_sec": 0.0,
                "speech_sec": 30.0,
                "segments": 4,
                "v_separated": alice + 0.05 * _vec(200),
            },
            {
                "label": "SPEAKER_00",
                "windows": 2,
                "start_sec": 600.0,
                "speech_sec": 30.0,
                "segments": 4,
                "v_separated": alice + 0.05 * _vec(201),
            },
            {
                "label": "SPEAKER_01",
                "windows": 1,
                "start_sec": 1200.0,
                "speech_sec": 30.0,
                "segments": 4,
                "v_separated": bob + 0.05 * _vec(202),
            },
        ],
    )

    hits = sp.who_was_there("/m/gaming.mkv")
    found = {t["person"]: t for t in hits}
    assert set(found) == {"Alice", "Bob"}  # "Stranger" stays excluded
    assert found["Alice"]["windows"] == 2
    assert found["Alice"]["timestamps"] == [0, 10]  # minutes
    assert found["Bob"]["timestamps"] == [20]
    assert found["Alice"]["best_value"] > sp.THRESHOLD_ACROSS_FILES


def test_who_was_there_without_named_people(own_db):
    own_db.write(
        "/m/a.mkv",
        sp.MODEL_BRAND,
        [
            {
                "label": "SPEAKER_00",
                "windows": 1,
                "start_sec": 0.0,
                "speech_sec": 30.0,
                "segments": 1,
                "v_separated": _vec(1),
            }
        ],
    )
    assert sp.who_was_there("/m/a.mkv") == []


def test_who_was_there_without_vectors(own_db):
    assert sp.who_was_there("/m/does_not_exist.mkv") == []


# ── Pinning the operating point ──────────────────────────────────────────


def test_operating_point_unchanged():
    """Each of these numbers is measured. Changing any of them bypasses a measurement
    from — then please measure again and provide the reference."""
    assert sp.MAX_SPEECH == 30.0  # 22.1
    assert sp.MIN_SEGMENT == 4.0  # 23.2
    assert sp.MIN_SPEECH == 4.0
    assert sp.MODEL_BRAND == "redimnet_m/vb2-ptn"  # 22.2
    assert sp.THRESHOLD_ACROSS_FILES == 0.55  # section 20
    assert sp.THRESHOLD_WITHIN_FILE == 0.62  # 23.8
    assert sp.THRESHOLD_GALLERY == 0.75  # 24.10


def test_there_are_three_thresholds_and_they_differ():
    """Confusing the three is the most obvious trap in the speaker code.

    0.55 = two individual voices · 0.62 = within ONE recording (the
    shared channel raises everything) · 0.75 = one recording against the WHOLE
    gallery (with many people and many windows, hundreds of opportunities to cross the
    threshold by chance). Measured: with 0.55, a large share of
    recordings got at least one wrong name.
    """
    assert sp.THRESHOLD_ACROSS_FILES < sp.THRESHOLD_WITHIN_FILE < sp.THRESHOLD_GALLERY


def test_the_gallery_judges_on_the_individual_voices(own_db):
    """Not at the centroid — this is a measured correction.

    Setup: one person has many voices from an OTHER corner of the room and
    a single one that matches the recording. The centroid gets
    pulled away and would be below the threshold; the single best value is not.
    Measured across several hundred files, the single value beats the centroid at every
    operating point — at zero false positives by about 15 points.
    """
    target = _vec(50)
    far = _vec(51)

    gallery = "/m/gallery.opus"
    own_db.write(
        gallery,
        sp.MODEL_BRAND,
        [
            # one voice that matches exactly …
            {
                "label": "SPEAKER_00",
                "windows": 1,
                "start_sec": 0.0,
                "speech_sec": 30.0,
                "segments": 1,
                "v_separated": target,
            },
            # … and four that lie elsewhere and pull the centroid away
            *[
                {
                    "label": "SPEAKER_00",
                    "windows": i,
                    "start_sec": i * 30.0,
                    "speech_sec": 30.0,
                    "segments": 1,
                    "v_separated": far + 0.05 * _vec(60 + i),
                }
                for i in range(2, 6)
            ],
        ],
    )
    for f in own_db.windows(gallery):
        own_db.assign(f["id"], "Alice")

    own_db.write(
        "/m/recording.mkv",
        sp.MODEL_BRAND,
        [
            {
                "label": "SPEAKER_00",
                "windows": 1,
                "start_sec": 0.0,
                "speech_sec": 30.0,
                "segments": 3,
                "v_separated": target,
            }
        ],
    )

    hits = sp.who_was_there("/m/recording.mkv", threshold=0.9)
    assert [t["person"] for t in hits] == ["Alice"]
    t = hits[0]
    assert t["best_value"] > 0.99  # the single value hits
    assert t["centroid_value"] < t["best_value"]
    assert t["centroid_above_threshold"] is False  # the centroid is not
    # Only one of five known voices agrees — that belongs in the result,
    # otherwise a match looks unanimous.
    assert t["voices_total"] == 5
    assert t["voices_share"] == pytest.approx(0.2)


# ── One source instead of many copies ────────────────────────────────────

_REPO = __import__("pathlib").Path(__file__).resolve().parent.parent


def _python_sources():
    for folder in ("whisper_tm", "tools"):
        yield from (_REPO / folder).rglob("*.py")


def test_the_checkpoint_is_defined_in_exactly_one_place():
    """A second copy of the checkpoint makes the vectors of one side silently
    incomparable to the other. The database writes MODEL_BRAND into every
    row and would show the change — a tool does not."""
    assert sp.MODEL_CHECKPOINT == ("vb2", "ptn")
    assert sp.MODEL_BRAND == "redimnet_m/vb2-ptn"
    # the brand is DERIVED, not written alongside it
    assert sp.MODEL_BRAND.endswith("/" + "-".join(sp.MODEL_CHECKPOINT))

    home = _REPO / "whisper_tm" / "core" / "speaker.py"
    copies = [
        str(f.relative_to(_REPO))
        for f in _python_sources()
        if f != home and '("vb2", "ptn")' in f.read_text(encoding="utf-8")
    ]
    assert copies == []


def test_normalization_is_a_single_function():
    """Thirteen copies, two flavors, difference in the third
    decimal place — exactly where 0.55 / 0.62 / 0.75 decide."""
    from whisper_tm.core import speaker_inventory, speaker_knn

    v = np.array([[3.0, 4.0]], dtype=np.float32)
    for f in (sp.nrm, sp._nrm, speaker_knn._nrm, speaker_inventory._nrm):
        assert np.allclose(f(v), [[0.6, 0.8]], atol=1e-6)
        assert f(v).dtype == np.float32

    # and it survives a SINGLE vector — axis=1 did not
    assert np.allclose(sp.nrm(np.array([3.0, 4.0])), [0.6, 0.8], atol=1e-6)


def test_the_tools_use_the_same_normalization():
    """Every speaker tool that normalizes imports the core's function
    instead of writing its own."""
    for name in ("speaker_quality.py",):
        t = (_REPO / "tools" / name).read_text(encoding="utf-8")
        assert "from whisper_tm.core.speaker import nrm" in t, name


# ── Thresholds are configurable, but measured as defaults ────────────────


def test_the_thresholds_come_from_the_config(monkeypatch):
    """They are measurement results, not natural constants — and the MATERIAL
    decides: within a recording 0.62, when grouping 0.94 (raised in steps:
    0.65 for short voice messages → 0.71 → 0.94 once the inventory was
    predominantly long recordings with dense background audio)."""
    from whisper_tm.config import config

    # Against the CONSTANTS, not against the currently configured value.
    #
    # Previously this read `sp.threshold(...) == 0.55` and so on — and that
    # only worked as long as the machine config happened to have the same
    # values. When changing the default to 0.71
    # the test failed even though the core was right: it never checked the
    # default, but the local setting.
    assert sp.THRESHOLD_ACROSS_FILES == 0.55
    assert sp.THRESHOLD_WITHIN_FILE == 0.62
    assert sp.THRESHOLD_GROUPING == 0.94
    assert sp.THRESHOLD_GALLERY == 0.75
    for kind, (_path, standard) in sp._THRESHOLDS.items():
        monkeypatch.delitem(
            config._data.setdefault("speaker", {}), _path.split(".")[1], raising=False
        )
        assert sp.threshold(kind) == standard, kind

    monkeypatch.setitem(config._data.setdefault("speaker", {}), "threshold_gallery", 0.58)
    assert sp.threshold("gallery") == 0.58


def test_an_unusable_threshold_falls_back_to_the_measured_value(monkeypatch):
    """A broken number in the configuration must not cost a run."""
    from whisper_tm.config import config

    for garbage in ("abc", -1, 0, 5, None):
        monkeypatch.setitem(config._data.setdefault("speaker", {}), "threshold_gallery", garbage)
        assert sp.threshold("gallery") == 0.75, garbage


def test_the_three_decisive_thresholds_are_ordered():
    """Three, not four — and that is deliberate.

    ``across_files`` < ``in_file`` < ``gallery`` answer the same question
    at growing risk: two individual voices, two windows sharing a
    channel, one recording against ALL named people. Every step creates
    more chances to hit the threshold randomly, so it must go up.

    ``group_voices`` used to be in this series, back when it was 0.65. At 0.94
    it stands above the gallery, which is correct: it decides
    NOTHING — it's the view of the speaker card. Making it stricter
    gets smaller tiles, not wrong names. Shortening the series to three
    is not a weakening, but the separation of two
    things that were never the same.

    The first two are measured constants, not settings (no feature compares
    with them); the gallery threshold is the configurable one.
    """
    assert sp.THRESHOLD_ACROSS_FILES < sp.THRESHOLD_WITHIN_FILE < sp.threshold("gallery")
