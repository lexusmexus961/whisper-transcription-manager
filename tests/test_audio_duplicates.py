"""The duplicates at a glance (audio/duplicates.py): groups of the same
recording with each copy's place, the space of the extra copies, recordings
that contain others, the folders that hold the same recordings, the stored
waveforms — from the relations table, without decoding anything."""

from __future__ import annotations

import base64

import numpy as np
import pytest

from whisper_tm.audio import duplicates as dup
from whisper_tm.audio import store as store_mod


@pytest.fixture
def store(tmp_path):
    dup.forget()
    s = store_mod.AudioStore(tmp_path / "audio.db")
    yield s
    s.close()
    dup.forget()


def add(store, path, size, duration, *, mtime=1000.0, same=(), holds=(), held_by=(), levels=None):
    """A recording with its relations to ones added before: ``same`` —
    (id, where this one begins in that one); ``holds`` — (id, where that one
    begins in this one); ``held_by`` — (id, where this one begins there)."""
    rel = [
        {
            "other": i,
            "kind": "same",
            "offset": off,
            "matched": duration,
            "score": 60,
            "new_contains": False,
        }
        for i, off in same
    ]
    rel += [
        {
            "other": i,
            "kind": "contains",
            "offset": off,
            "matched": 30.0,
            "score": 60,
            "new_contains": True,
        }
        for i, off in holds
    ]
    rel += [
        {
            "other": i,
            "kind": "contains",
            "offset": off,
            "matched": duration,
            "score": 60,
            "new_contains": False,
        }
        for i, off in held_by
    ]
    result = {"duration": duration}
    if levels is not None:
        result["levels"] = bytes(levels)
    return store.add(path, size, mtime, result=result, relations=rel)


# ── Folders ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("a", "b", "want"),
    [
        # A copied folder keeps its layout: the pair is the folders above it
        (
            "/bak/x/2024/trip/a.mp4",
            "/phone/2024/trip/a (1).mp4",
            ("/bak/x", "/phone", "2024/trip", False),
        ),
        (
            "/phone/2024/trip/a.mp4",
            "/bak/x/2024/trip/a.mp4",
            ("/bak/x", "/phone", "2024/trip", True),
        ),
        # Two copies in one folder
        ("/m/v/a.mp4", "/m/v/a.mp3", ("/m/v", "/m/v", "", False)),
        # Nothing in common below them
        (
            "/m/WhatsApp Video/a.mp4",
            "/m/Camera/b.mp4",
            ("/m/Camera", "/m/WhatsApp Video", "", True),
        ),
        # Each keeps a name of its own: /b/a is a copy of /a
        ("/a/x/f.wav", "/b/a/x/f.wav", ("/a", "/b/a", "x", False)),
        # ... and neither becomes the other's parent ("/m" would be all of it)
        (
            "/m/Photos/2024/v.mp4",
            "/m/Backup/Photos/2024/v.mp4",
            ("/m/Backup/Photos", "/m/Photos", "2024", True),
        ),
        ("/top.mp3", "/m/top.mp3", ("/", "/m", "", False)),
    ],
)
def test_the_folders_a_copy_links(a, b, want):
    assert dup.folder_pair(a, b) == want


def test_the_common_part():
    assert dup.common_part("/g/Game/dlc2", "/g/Game/dlc3") == ("/g/Game", "dlc2", "dlc3")
    assert dup.common_part("/m/v", "/m/v/Sent") == ("/m/v", "", "Sent")
    assert dup.common_part("/a", "/b") == ("/", "a", "b")


# ── Groups ─────────────────────────────────────────────────────────────────


def test_a_group_with_its_places_and_the_space_of_its_extra_copies(store):
    a = add(store, "/lib/v/talk.mp4", 900, 60.0, mtime=50)
    b = add(store, "/lib/a/talk.mp3", 100, 60.0, same=[(a, 0.5)])  # begins 0.5 s into a
    c = add(store, "/bak/a/talk.mp3", 120, 59.6, same=[(b, -0.25)])  # linked through b only
    d = add(store, "/lib/v/talk copy.mp4", 900, 60.0, mtime=40, same=[(a, 0.0)])
    page = dup.group_page(store)
    assert page["total"] == 1
    g = page["groups"][0]
    assert g["count"] == 4 and g["length"] == 60.0
    files = {f["id"]: f for f in g["files"]}
    # Placed on one timeline by the offsets, also through another copy: b
    # begins 0.5 s into a, c 0.25 s before b, d with a
    assert (files[a]["pos"], files[d]["pos"], files[c]["pos"], files[b]["pos"]) == (
        0.0,
        0.0,
        0.25,
        0.5,
    )
    # The largest of each kind is kept — of two the same size, the older
    assert files[d]["largest"] and not files[a]["largest"]
    assert files[c]["largest"] and not files[b]["largest"]
    assert g["extra"] == 2 and g["extra_bytes"] == 900 + 100
    # Videos first, the largest first
    assert [f["video"] for f in g["files"]] == [True, True, False, False]
    assert g["files"][2]["id"] == c


def test_filters_and_orders(store):
    big = add(store, "/m/Movies/film.mkv", 5000, 5400.0)
    add(store, "/m/Backup/Movies/film.mkv", 5000, 5400.0, same=[(big, 0.0)])
    x = add(store, "/m/Games/dlc1/sound/hit.wav", 10, 1.2)
    add(store, "/m/Games/dlc2/sound/hit.wav", 10, 1.2, same=[(x, 0.0)])
    y = add(store, "/m/Games/dlc1/sound/step.wav", 8, 0.8)
    add(store, "/m/Games/dlc2/sound/step.wav", 8, 0.8, same=[(y, 0.0)])
    add(store, "/m/Games/dlc3/sound/step.wav", 8, 0.8, same=[(y, 0.0)])
    assert dup.group_page(store)["groups"][0]["files"][0]["path"].endswith("film.mkv")  # space
    assert dup.group_page(store, sort="copies")["groups"][0]["count"] == 3
    assert dup.group_page(store, sort="length")["groups"][-1]["length"] == 0.8
    assert dup.group_page(store, q="GAMES/DLC1")["total"] == 2  # letters, any case
    assert dup.group_page(store, min_sec=60)["total"] == 1
    assert dup.group_page(store, file_id=y)["groups"][0]["count"] == 3
    page = dup.group_page(store, limit=1, offset=1)
    assert len(page["groups"]) == 1 and page["total"] == 3 and page["offset"] == 1
    # One folder pair: the games' dlc1 and dlc2
    ov = dup.overview(store)
    pair = next(f for f in ov["folders"] if f["a"] == "/m/Games/dlc1")
    assert pair["b"] == "/m/Games/dlc2" and pair["groups"] == 2
    assert dup.group_page(store, pair=pair["id"])["total"] == 2


def test_the_newest_found_first(store):
    a = add(store, "/m/a.mp3", 1, 10.0)
    add(store, "/m/b.mp3", 1, 10.0, same=[(a, 0.0)])
    c = add(store, "/m/c.mp3", 1, 20.0)
    add(store, "/m/d.mp3", 1, 20.0, same=[(c, 0.0)])
    with store.connect() as conn:
        conn.execute("UPDATE relations SET found_at = 5 WHERE a = ?", (c,))
        conn.execute("UPDATE relations SET found_at = 9 WHERE a = ?", (a,))
    dup.forget()
    assert [g["files"][0]["path"] for g in dup.group_page(store, sort="new")["groups"]] == [
        "/m/a.mp3",
        "/m/c.mp3",
    ]


# ── Wholes ─────────────────────────────────────────────────────────────────


def test_a_recording_with_what_it_holds(store):
    clip1 = add(store, "/m/clips/one.mp4", 10, 30.0)
    clip2 = add(store, "/m/clips/two.mp4", 10, 30.0)
    vod = add(store, "/m/vods/stream.mp4", 999, 3600.0, holds=[(clip2, 1200.0), (clip1, 100.0)])
    add(store, "/m/clips/three.mp4", 10, 30.0, held_by=[(vod, 1210.0)])  # overlaps two
    page = dup.whole_page(store)
    assert page["total"] == 1
    w = page["wholes"][0]
    assert w["whole"]["id"] == vod
    assert [p["path"] for p in w["parts"]] == [
        "/m/clips/one.mp4",
        "/m/clips/two.mp4",
        "/m/clips/three.mp4",
    ]
    assert [p["at"] for p in w["parts"]] == [100.0, 1200.0, 1210.0]
    # Covered once where parts overlap: 30 + (1200…1240)
    assert w["covered"] == pytest.approx(70.0)
    assert dup.whole_page(store, q="three")["total"] == 1  # a part's name
    assert dup.whole_page(store, q="nothing")["total"] == 0
    assert dup.whole_page(store, file_id=clip1)["total"] == 1
    assert dup.whole_page(store, min_sec=4000)["total"] == 0
    ov = dup.overview(store)
    assert ov["contains"] == {"wholes": 1, "parts": 3, "pairs": 3}
    assert ov["same"]["groups"] == 0


# ── The overview ───────────────────────────────────────────────────────────


def test_the_overview_counts_and_folders(store):
    # A backup that holds most of the photos folder, laid out alike
    for i in range(4):
        orig = add(store, f"/m/Photos/2024/{i}/v{i}.mp4", 100 + i, 30.0)
        if i < 3:
            add(store, f"/m/Backup/Photos/2024/{i}/v{i}.mp4", 100 + i, 30.0, same=[(orig, 0.0)])
    # Two copies in one folder
    w = add(store, "/m/WA/voice.opus", 5, 12.0)
    add(store, "/m/WA/voice (1).opus", 5, 12.0, same=[(w, 0.0)])
    add(store, "/m/WA/other.opus", 5, 7.0)
    ov = dup.overview(store)
    assert ov["same"] == {"groups": 4, "files": 8, "extra": 4, "extra_bytes": 100 + 101 + 102 + 5}
    assert ov["analyzed"] == 10 and ov["folders_total"] == 2
    top, within = ov["folders"]
    assert (top["a"], top["b"]) == ("/m/Backup/Photos", "/m/Photos") and not top["within"]
    assert top["common"] == "/m" and (top["a_rest"], top["b_rest"]) == ("Backup/Photos", "Photos")
    assert top["groups"] == 3 and top["a_files"] == 3 and top["b_files"] == 3
    assert top["a_total"] == 3 and top["b_total"] == 4  # all of the backup, 3 of 4 photos
    assert top["layout"] == 3  # 2024/0, 2024/1, 2024/2
    assert top["a_bytes"] == top["b_bytes"] == 303
    assert within["within"] and within["a"] == "/m/WA" and within["a_files"] == 2
    assert within["a_total"] == 3 and within["b_total"] == 0
    # Filtered: the tiles and folders follow
    ov = dup.overview(store, q="/wa/")
    assert ov["same"]["groups"] == 1 and [f["a"] for f in ov["folders"]] == ["/m/WA"]
    assert dup.overview(store, top=1)["folders_total"] == 2


def test_a_folder_inside_the_other(store):
    a = add(store, "/m/WhatsApp Video/a.mp4", 10, 9.0)
    add(store, "/m/WhatsApp Video/Sent/a.mp4", 10, 9.0, same=[(a, 0.0)])
    add(store, "/m/WhatsApp Video/b.mp4", 10, 9.0)
    f = dup.overview(store)["folders"][0]
    assert (f["a"], f["b"]) == ("/m/WhatsApp Video", "/m/WhatsApp Video/Sent")
    # The outer folder's own recordings — not those of the inner one
    assert f["a_total"] == 2 and f["b_total"] == 1


def test_worked_out_again_when_the_relations_change(store, monkeypatch):
    a = add(store, "/m/a.mp3", 1, 10.0)
    add(store, "/m/b.mp3", 1, 10.0, same=[(a, 0.0)])
    assert dup.overview(store)["same"]["groups"] == 1
    c = add(store, "/m/c.mp3", 1, 20.0)
    add(store, "/m/d.mp3", 1, 20.0, same=[(c, 0.0)])
    # While the analysis finds pairs: at most every MIN_AGE_SEC
    assert dup.overview(store)["same"]["groups"] == 1
    monkeypatch.setattr(dup, "MIN_AGE_SEC", 0.0)
    assert dup.overview(store)["same"]["groups"] == 2
    monkeypatch.setattr(dup, "MIN_AGE_SEC", 3600.0)
    # Fewer relations (cleared, pruned): at once
    store.remove("/m/d.mp3")
    assert dup.overview(store)["same"]["groups"] == 1
    assert dup.overview(store, fresh=True)["same"]["groups"] == 1


def test_an_empty_index(store):
    ov = dup.overview(store)
    assert ov["same"]["groups"] == 0 and ov["folders"] == [] and ov["analyzed"] == 0
    assert dup.group_page(store)["groups"] == [] and dup.whole_page(store)["wholes"] == []


# ── Waveforms ──────────────────────────────────────────────────────────────


def test_stored_waveforms_in_fewer_values(store):
    lv = np.zeros(1000, np.uint8)
    lv[500] = 200
    lv[999] = 90
    a = add(store, "/m/a.mp3", 1, 50.0, levels=lv)
    b = add(store, "/m/b.mp3", 1, 1.0, levels=np.full(20, 7, np.uint8))
    c = add(store, "/m/c.mp3", 1, 1.0)  # none kept
    got = dup.levels(store, [a, b, c, a], 100)
    assert set(got) == {a, b}
    va = np.frombuffer(base64.b64decode(got[a]), np.uint8)
    assert len(va) == 100 and va.max() == 200 and va[50] == 200 and va[-1] == 90
    # Fewer values than asked: as they are
    assert np.frombuffer(base64.b64decode(got[b]), np.uint8).tolist() == [7] * 20
    assert len(np.frombuffer(base64.b64decode(dup.levels(store, [a], 1)[a]), np.uint8)) == 16


# ── Filters ────────────────────────────────────────────────────────────────


@pytest.fixture
def library(store, monkeypatch):
    """Copies with formats, dates, languages and people; a stream holding
    clips from two folders."""
    monkeypatch.setattr(
        dup,
        "_people_names",
        lambda paths: {"/m/vods/stream 2024-03-01.mp4": {"Alice"}, "/m/clips/a/one.mp4": {"Bob"}},
    )
    monkeypatch.setattr(
        dup,
        "_languages",
        lambda paths: (
            {p: "de" for p in paths if p.startswith("/m/talk")}
            | {"/m/vods/stream 2024-03-01.mp4": "en"}
        ),
    )
    t = add(store, "/m/talk/VID-20230105-WA0001.mp4", 500, 30.0, mtime=1.8e9)
    add(store, "/m/talk/backup/VID-20230105-WA0001.mp3", 50, 30.0, same=[(t, 0.0)])
    s = add(store, "/m/sfx/hit.wav", 10, 1.0, mtime=1.6e9)  # 2020-09-13 by the file
    add(store, "/m/sfx/hit (1).wav", 10, 1.0, mtime=1.6e9, same=[(s, 0.0)])
    add(store, "/m/sfx/hit (2).wav", 10, 1.0, mtime=1.6e9, same=[(s, 0.0)])
    one = add(store, "/m/clips/a/one.mp4", 10, 30.0, mtime=1.7e9)
    two = add(store, "/m/clips/b/two.mp4", 10, 30.0, mtime=1.7e9)
    add(
        store,
        "/m/vods/stream 2024-03-01.mp4",
        999,
        3600.0,
        mtime=1.7e9,
        holds=[(one, 100.0), (two, 900.0)],
    )
    return store


def total(page):
    return page["total"]


def test_filters_of_a_group(library):
    g = lambda **kw: total(dup.group_page(library, **kw))  # noqa: E731
    assert g() == 2
    assert g(media="video") == 1 and g(media="audio") == 2  # a sound copy of the talk counts
    assert g(ext=("WAV",)) == 1 and g(ext=(".mp3", "mp4")) == 1
    # The recording date from the name (2023-01-05), else the file's (2020-09-13)
    assert g(since="2023-01-01") == 1 and g(until="2021-01-01") == 1
    assert g(since="2020-09-13", until="2020-09-13") == 1
    # (both copies of the talk are transcribed: none of it is "not transcribed")
    assert g(lang="de") == 1 and g(transcribed="no") == 1 and g(transcribed="yes") == 1
    assert g(min_copies=3) == 1 and g(spread="within") == 1 and g(spread="across") == 1
    assert g(max_sec=10) == 1 and g(min_sec=10, max_sec=60) == 1
    assert g(media="video", ext=("wav",)) == 0


def test_filters_of_a_recording_holding_others(library):
    w = lambda **kw: dup.whole_page(library, **kw)  # noqa: E731
    assert total(w()) == 1 and w()["wholes"][0]["parts_all"] == 2
    # The clips of one folder inside the stream: the others are left out
    page = w(q_part="clips/a")
    assert [p["path"] for p in page["wholes"][0]["parts"]] == ["/m/clips/a/one.mp4"]
    assert page["wholes"][0]["parts_all"] == 2
    assert total(w(q_part="nowhere")) == 0
    assert total(w(q_whole="vods")) == 1 and total(w(q_whole="clips")) == 0
    # A person speaking in the stream or in one of its parts
    assert (
        total(w(person="alice")) == 1
        and total(w(person="Bob")) == 1
        and total(w(person="Eve")) == 0
    )
    assert total(w(min_parts=3)) == 0 and total(w(min_parts=2)) == 1
    assert total(w(lang="en")) == 1 and total(w(since="2024-03-01", until="2024-03-31")) == 1
    assert total(w(media="audio")) == 0
    row = page["wholes"][0]["whole"]
    assert row["day"] == "2024-03-01" and row["people"] == ["Alice"] and row["lang"] == "en"


def test_the_overview_follows_the_filters_and_offers_choices(library):
    ov = dup.overview(library, media="video")
    assert ov["same"]["groups"] == 1 and ov["contains"]["wholes"] == 1
    f = ov["facets"]
    assert dict(f["people"]) == {"Alice": 1, "Bob": 1}
    assert dict(f["ext"]) == {"mp4": 2, "mp3": 1, "wav": 1}
    assert dict(f["lang"]) == {"de": 1, "en": 1}
    assert f["days"][0] == "2020-09-13" and f["days"][1] == "2024-03-01"
    assert dup.overview(library, person="Bob")["same"]["groups"] == 0
