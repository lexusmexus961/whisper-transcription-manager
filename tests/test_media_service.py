"""The media service's own logic (media/service.py) — with a stand-in model
and stand-in decoding: which moments repeat, what gets a vector, the
thumbnails, file queries. The real model runs in its own environment."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from whisper_tm.media import service


class FakeModel:
    """Vectors from what goes in: a still by its colour, a clip by its first
    frame's colour (with sound: shifted), a sound and a text by their size."""

    def __init__(self):
        self.calls = []

    def encode(
        self,
        inputs,
        batch_size=1,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
        prompt_name=None,
    ):
        self.calls.append((len(inputs), prompt_name, batch_size))
        out = []
        for x in inputs:
            v = np.zeros(8, dtype=np.float32)
            if isinstance(x, Image.Image):
                r, g, b = x.getpixel((0, 0))
                v[:3] = (r + 1, g + 1, b + 1)
            elif isinstance(x, dict) and "video" in x:
                r, g, b = x["video"][0].getpixel((0, 0))
                v[3:6] = (r + 1, g + 1, b + 1)
                if "audio" in x:
                    v[6] = 50.0
            elif isinstance(x, dict) and "array" in x:
                v[6] = 1.0
                v[7] = len(x["array"]) / 16000
            else:
                v[7] = len(str(x)) + 1.0
            out.append(v / np.linalg.norm(v))
        return np.stack(out)


@pytest.fixture
def fake(monkeypatch, tmp_path):
    model = FakeModel()
    monkeypatch.setattr(service, "_load", lambda device, towers: model)

    def color(t):
        # 0–29 s red (a static scene), 30–59 s blue, from 60 s green
        return (200, 0, 0) if t < 30 else (0, 0, 200) if t < 60 else (0, 200, 0)

    grabbed = []

    def grab_frame(path, t, scale):
        grabbed.append(("frame", round(t, 2)))
        if 90 <= t < 100:
            return None  # a black / broken frame there
        return Image.new("RGB", (64, 36), color(t))

    def grab_sound(path, start, dur):
        grabbed.append(("sound", round(start, 2)))
        return np.zeros(int(16000 * dur), dtype=np.float32)

    monkeypatch.setattr(service, "grab_frame", grab_frame)
    monkeypatch.setattr(service, "grab_sound", grab_sound)
    media = tmp_path / "talk.mp4"
    media.write_bytes(b"x")
    return {"model": model, "path": str(media), "grabbed": grabbed, "tmp": tmp_path}


def _moments(ts, thumbs=None):
    return [
        {
            "t": t,
            "start": max(0.0, t - 5),
            "dur": 10.0,
            "text": f"what is said at {t}",
            "thumb": (str(thumbs / f"{int(t)}.jpg") if thumbs else None),
        }
        for t in ts
    ]


def _req(fake, ts, kinds, **kw):
    return {
        "path": fake["path"],
        "video": True,
        "moments": _moments(ts, kw.pop("thumbs", None)),
        "kinds": kinds,
        "clip_sec": 10,
        "clip_frames": 4,
        "dedupe": kw.pop("dedupe", 0.97),
        "prev": kw.pop("prev", None),
        "device": "cpu",
        "batch": 2,
        "workers": 2,
        **kw,
    }


class TestRepeats:
    def test_a_moment_that_looks_like_the_last_kept_one_gets_no_still_and_no_clip(self, fake):
        res = service.index_chunk(_req(fake, [5, 15, 25, 35, 45, 65], ["image", "video", "audio"]))
        rep = [m["repeat"] for m in res["moments"]]
        # red, red, red, blue, blue, green: each first of its colour is new
        assert rep == [False, True, True, False, True, False]
        for m in res["moments"]:
            assert "audio" in m["vectors"]  # the sound counts always
            assert ("image" in m["vectors"]) == (not m["repeat"])
            assert ("video" in m["vectors"]) == (not m["repeat"])

    def test_the_last_kept_still_carries_over_to_the_next_chunk(self, fake):
        first = service.index_chunk(_req(fake, [5, 15], ["image"]))
        assert [m["repeat"] for m in first["moments"]] == [False, True]
        second = service.index_chunk(_req(fake, [25, 35], ["image"], prev=first["prev"]))
        assert [m["repeat"] for m in second["moments"]] == [True, False]

    def test_dedupe_zero_keeps_every_moment(self, fake):
        res = service.index_chunk(_req(fake, [5, 15, 25], ["image"], dedupe=0))
        assert [m["repeat"] for m in res["moments"]] == [False, False, False]

    def test_a_moment_without_a_picture_counts_as_a_repeat_but_keeps_its_sound(self, fake):
        res = service.index_chunk(_req(fake, [95], ["image", "audio"]))
        m = res["moments"][0]
        assert m["repeat"] and "image" not in m["vectors"] and "audio" in m["vectors"]


class TestKinds:
    def test_only_what_is_asked_for(self, fake):
        res = service.index_chunk(_req(fake, [5, 35], ["audio"]))
        assert all(set(m["vectors"]) == {"audio"} for m in res["moments"])
        # no picture was even grabbed
        assert not [g for g in fake["grabbed"] if g[0] == "frame"]

    def test_clip_with_sound_is_its_own_vector(self, fake):
        res = service.index_chunk(_req(fake, [5], ["video", "videoaudio"]))
        v = res["moments"][0]["vectors"]
        a, b = service.unb64(v["video"]), service.unb64(v["videoaudio"])
        assert a.shape == b.shape and float(a @ b) < 0.99

    def test_a_clip_is_spread_over_its_seconds(self, fake):
        service.index_chunk(_req(fake, [50], ["video"], dedupe=0))
        frames = sorted(t for k, t in fake["grabbed"] if k == "frame" and t != 50)
        # 10 s from 30 % before the moment (47 … 57 s), 4 frames at their middles
        assert frames == [48.25, 50.75, 53.25, 55.75]

    def test_speech_needs_three_words(self, fake):
        req = _req(fake, [5, 15], ["speech"])
        req["moments"][1]["text"] = "Yes."
        res = service.index_chunk(req)
        assert (
            "speech" in res["moments"][0]["vectors"]
            and "speech" not in res["moments"][1]["vectors"]
        )
        assert any(p == "Document" for _, p, _ in fake["model"].calls)

    def test_an_audio_file_has_no_pictures(self, fake):
        req = _req(fake, [5, 15], ["image", "video", "audio"])
        req["video"] = False
        res = service.index_chunk(req)
        assert all(set(m["vectors"]) == {"audio"} for m in res["moments"])

    def test_thumbnails_are_written_for_kept_moments_only(self, fake):
        thumbs = fake["tmp"] / "thumbs" / "7"
        res = service.index_chunk(_req(fake, [5, 15, 35], ["image"], thumbs=thumbs, thumb_width=32))
        assert [m["repeat"] for m in res["moments"]] == [False, True, False]
        assert (
            (thumbs / "5.jpg").is_file()
            and not (thumbs / "15.jpg").exists()
            and (thumbs / "35.jpg").is_file()
        )
        assert Image.open(thumbs / "5.jpg").size[0] == 32

    def test_kept_stills_are_marked_and_their_thumbnails_counted(self, fake):
        thumbs = fake["tmp"] / "thumbs" / "8"
        res = service.index_chunk(_req(fake, [5, 15, 35], ["video"], thumbs=thumbs, thumb_width=32))
        assert [m["still"] for m in res["moments"]] == [True, False, True]
        assert (
            res["thumb_bytes"]
            == (thumbs / "5.jpg").stat().st_size + (thumbs / "35.jpg").stat().st_size
        )

    def test_a_full_thumbnail_folder_writes_none(self, fake):
        thumbs = fake["tmp"] / "thumbs" / "9"
        req = _req(fake, [5, 35], ["image"], thumbs=thumbs)
        req["thumbs"] = False
        res = service.index_chunk(req)
        assert [m["still"] for m in res["moments"]] == [True, True]
        assert res["thumb_bytes"] == 0 and not thumbs.exists()

    def test_a_missing_file_says_so(self, fake):
        req = _req(fake, [5], ["image"])
        req["path"] = "/nowhere/x.mp4"
        with pytest.raises(FileNotFoundError):
            service.index_chunk(req)


class TestKindsAddedLater:
    """A file that has moments gets a kind it lacked: the repeats an
    earlier pass found stand, unless the pictures were never looked at."""

    def test_clips_for_the_known_new_moments_without_a_look_at_the_stills(self, fake):
        res = service.index_chunk(
            _req(fake, [5, 15, 35], ["videoaudio"], repeats=[False, True, False])
        )
        assert [m["repeat"] for m in res["moments"]] == [False, True, False]
        assert ["videoaudio" in m["vectors"] for m in res["moments"]] == [True, False, True]
        assert [m["still"] for m in res["moments"]] == [True, False, True]
        # only the clip frames were grabbed, no still to compare
        assert sorted({t for k, t in fake["grabbed"] if k == "frame"}) == sorted(
            {t for k, t in fake["grabbed"] if k == "frame"} - {5.0, 15.0, 35.0}
        )

    def test_stills_added_later_only_for_the_known_new_moments(self, fake):
        res = service.index_chunk(
            _req(fake, [5, 15, 25], ["image"], repeats=[False, False, True], dedupe=0.97)
        )
        # 5 and 15 look alike, but the earlier pass kept both: they stand
        assert [m["repeat"] for m in res["moments"]] == [False, False, True]
        assert ["image" in m["vectors"] for m in res["moments"]] == [True, True, False]
        assert ("frame", 25.0) not in fake["grabbed"]

    def test_existing_thumbnails_are_not_written_again(self, fake):
        thumbs = fake["tmp"] / "thumbs" / "10"
        thumbs.mkdir(parents=True)
        (thumbs / "5.jpg").write_bytes(b"kept")
        res = service.index_chunk(_req(fake, [5], ["image"], thumbs=thumbs, repeats=[False]))
        assert (thumbs / "5.jpg").read_bytes() == b"kept" and res["thumb_bytes"] == 0

    def test_the_repeats_must_fit_the_moments(self, fake):
        with pytest.raises(ValueError):
            service.index_chunk(_req(fake, [5, 15], ["image"], repeats=[False]))


class TestQueries:
    def test_vectors_travel_as_float16(self):
        v = np.asarray([0.6, 0.8, 0.0], dtype=np.float32)
        assert np.allclose(service.unb64(service.b64(v)), v, atol=1e-3)

    def test_a_video_alone_goes_in_as_a_dict(self, fake, monkeypatch):
        monkeypatch.setattr(
            service,
            "frames_of_file",
            lambda path, n, limit=60.0: [Image.new("RGB", (8, 8), (9, 9, 9))] * n,
        )
        seen = []
        orig = fake["model"].encode

        def spy(inputs, **kw):
            seen.append(inputs[0])
            return orig(inputs, **kw)

        fake["model"].encode = spy
        service.embed_file(fake["path"], "video", "", False, 4)
        assert isinstance(seen[-1], dict) and len(seen[-1]["video"]) == 4
        service.embed_file(fake["path"], "video", "at night", True, 2)
        assert seen[-1]["text"] == "at night <|video|> <|audio|>" and "audio" in seen[-1]

    def test_an_unknown_kind_is_refused(self, fake):
        with pytest.raises(ValueError):
            service.embed_file(fake["path"], "pdf")
