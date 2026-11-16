"""What the language model adds to Images & sounds (media/llm.py,
media/describe.py): sound labels, judging pictures, checks, describing."""

from __future__ import annotations

import math
import time

import numpy as np
import pytest

from whisper_tm.core import media_server
from whisper_tm.media import describe, llm
from whisper_tm.media import store as store_mod
from whisper_tm.media.service import b64


@pytest.fixture
def cfg(monkeypatch):
    values = {"llm_translate_sounds": True, "llm_language": "German", "llm_model": ""}
    monkeypatch.setattr(llm, "_cfg", lambda name, default=None: values.get(name, default))
    monkeypatch.setattr(llm, "configured", lambda: True)
    llm._labels.clear()
    return values


def _answer(text, tops=None):
    choice = {"message": {"content": text}}
    if tops is not None:
        choice["logprobs"] = {
            "content": [{"top_logprobs": [{"token": t, "logprob": math.log(p)} for t, p in tops]}]
        }
    return {"choices": [choice]}


@pytest.fixture
def chat(monkeypatch):
    sent = []

    def payload(messages, *, max_tokens):
        return (
            "http://llm/v1/chat/completions",
            {},
            {"messages": messages, "max_tokens": max_tokens},
            False,
            True,
        )

    monkeypatch.setattr(llm, "_payload", payload)
    answers = []

    def post(url, headers, body, timeout=120):
        sent.append(body)
        return answers.pop(0)

    monkeypatch.setattr(llm, "_post", post)
    return {"sent": sent, "answers": answers}


class TestSoundLabels:
    def test_the_query_becomes_a_short_english_label_once(self, cfg, chat):
        chat["answers"].append(_answer('"busy pub crowd"'))
        assert llm.sound_label("lots of people chatting in a crowded pub") == "busy pub crowd"
        assert llm.sound_label("lots of people chatting in a crowded pub") == "busy pub crowd"
        assert len(chat["sent"]) == 1  # kept per query
        assert "SHORT English sound label" in chat["sent"][0]["messages"][0]["content"]

    def test_off_or_without_a_model_it_stays_as_typed(self, cfg, chat, monkeypatch):
        cfg["llm_translate_sounds"] = False
        assert llm.sound_label("dog") is None
        cfg["llm_translate_sounds"] = True
        monkeypatch.setattr(llm, "configured", lambda: False)
        assert llm.sound_label("dog") is None
        assert chat["sent"] == []

    def test_a_failing_model_leaves_the_query_alone(self, cfg, monkeypatch):
        monkeypatch.setattr(
            llm, "_payload", lambda m, max_tokens: (_ for _ in ()).throw(RuntimeError("down"))
        )
        assert llm.sound_label("a dog barks") is None


class TestJudging:
    def test_the_expected_digit_from_the_token_probabilities(self, cfg, chat, tmp_path):
        img = tmp_path / "s.jpg"
        img.write_bytes(b"jpg")
        chat["answers"].append(
            _answer("9", tops=[("9", 0.5), ("7", 0.25), ("x", 0.1), ("3", 0.25)])
        )
        score = llm.judge("fire", str(img))
        assert score == pytest.approx((9 * 0.5 + 7 * 0.25 + 3 * 0.25) / 1.0)
        body = chat["sent"][0]
        assert body["logprobs"] is True and body["top_logprobs"] == 10 and body["max_tokens"] == 2
        parts = body["messages"][1]["content"]
        assert parts[0]["type"] == "image_url" and parts[0]["image_url"]["url"].startswith(
            "data:image/jpeg;base64,"
        )
        assert '"fire"' in parts[1]["text"]

    def test_without_probabilities_the_digit_itself(self, cfg, chat, tmp_path):
        img = tmp_path / "s.jpg"
        img.write_bytes(b"jpg")
        chat["answers"].append(_answer("6"))
        assert llm.judge("x", str(img)) == 6.0
        chat["answers"].append(_answer("no idea"))
        assert llm.judge("x", str(img)) == 0.0

    def test_describe_in_the_set_language(self, cfg, chat, tmp_path):
        img = tmp_path / "s.jpg"
        img.write_bytes(b"jpg")
        cfg["llm_language"] = "English"
        chat["answers"].append(_answer("  A man   at a desk. "))
        assert llm.describe(str(img)) == "A man at a desk."
        assert "ONE sentence of English" in chat["sent"][0]["messages"][0]["content"]

    def test_vision_from_the_servers_props(self, cfg, monkeypatch):
        import requests

        from whisper_tm.core import llm_client

        monkeypatch.setattr(
            llm_client,
            "build_chat_payload",
            lambda messages, model="": ("http://llm:8080/v1/chat/completions", {}, {}, False),
        )

        class R:
            ok = True

            def __init__(self, mods):
                self._m = mods

            def json(self):
                return {"modalities": self._m}

        llm._vision.update(at=0.0, key=None, value=None)
        monkeypatch.setattr(
            requests, "get", lambda url, timeout=4, headers=None: R({"vision": True})
        )
        assert llm.vision_available() is True
        llm._vision.update(at=0.0, key=None, value=None)
        monkeypatch.setattr(
            requests, "get", lambda url, timeout=4, headers=None: R({"vision": False})
        )
        assert llm.vision_available() is False
        llm._vision.update(at=0.0, key=None, value=None)
        monkeypatch.setattr(
            requests,
            "get",
            lambda url, timeout=4, headers=None: (_ for _ in ()).throw(OSError("x")),
        )
        assert llm.vision_available() is None  # unknown, not "no"


class TestChecks:
    def _wait(self, cid):
        for _ in range(200):
            st = llm.check_state(cid)
            if not st["running"]:
                return st
            time.sleep(0.01)
        raise AssertionError("check did not end")

    def test_each_picture_once(self, monkeypatch):
        monkeypatch.setattr(llm, "judge", lambda q, thumb: {"a": 9.0, "b": 2.0}[thumb])
        cid = llm.start_check("fire", [{"key": 1, "thumb": "a"}, {"key": 2, "thumb": "b"}])
        st = self._wait(cid)
        assert st["scores"] == {"1": 9.0, "2": 2.0} and st["done"] == 2 and st["error"] is None

    def test_a_model_that_keeps_failing_ends_the_check(self, monkeypatch):
        def broken(q, thumb):
            raise RuntimeError("no vision")

        monkeypatch.setattr(llm, "judge", broken)
        cid = llm.start_check("x", [{"key": k, "thumb": "t"} for k in range(6)])
        st = self._wait(cid)
        assert st["error"] == "no vision" and st["done"] == 2 and st["errors"] == 3

    def test_cancel(self, monkeypatch):
        monkeypatch.setattr(llm, "judge", lambda q, thumb: time.sleep(0.05) or 5.0)
        cid = llm.start_check("x", [{"key": k, "thumb": "t"} for k in range(20)])
        time.sleep(0.07)
        assert llm.cancel_check(cid)
        st = self._wait(cid)
        assert st["done"] < 20
        assert llm.cancel_check(cid) is False
        assert llm.check_state("nope") is None


class TestDescribingRun:
    @pytest.fixture
    def store(self, tmp_path, monkeypatch):
        monkeypatch.setattr(store_mod, "store_path", lambda: tmp_path / "media.db")
        store_mod._stores.clear()
        s = store_mod.get_store()
        s.ensure_dims(8)
        f = s.start_file("/lib/a.mp4", video=True, duration=60, settings="fp")
        rows = []
        for k in range(5):
            thumb = tmp_path / f"{k}.jpg"
            thumb.write_bytes(b"jpg")
            rows.append(
                {
                    "t": 10 * k,
                    "start": 10 * k,
                    "dur": 10,
                    "repeat": k == 4,
                    "thumb": str(thumb),
                    "vectors": {"image": np.eye(8, dtype=np.float32)[k]},
                }
            )
        s.add_moments(f["id"], rows, next_t=60, prev=None, done=True, seconds=0)
        yield s
        store_mod._stores.clear()

    class Queue:
        def __init__(self, abort_after=None):
            self.progress_calls = []
            self.abort_after = abort_after

        def abort_requested(self, entry_id):
            return self.abort_after is not None and len(self.progress_calls) >= self.abort_after

        def progress(self, entry_id, pct, message):
            self.progress_calls.append((pct, message))

    def test_describes_embeds_and_stores(self, store, monkeypatch):
        monkeypatch.setattr(llm, "describe", lambda thumb: f"Picture {thumb[-5:]}")
        monkeypatch.setattr(describe, "EMBED_BATCH", 2)
        embeds = []

        def call(path, payload=None, *, timeout=120):
            embeds.append(payload)
            return {
                "vectors": [
                    b64(np.ones(8, dtype=np.float32) / np.sqrt(8)) for _ in payload["texts"]
                ]
            }

        monkeypatch.setattr(media_server.SERVER, "call", call)
        q = self.Queue()
        out = describe.run({"id": 1, "params": {"limit": 3}}, q)
        assert out["described"] == 3 and out["failed"] == 0
        assert store.description_counts() == {"described": 3, "stills": 4}
        assert [len(p["texts"]) for p in embeds] == [2, 1] and all(
            p["prompt"] == "Document" for p in embeds
        )
        assert q.progress_calls[-1][0] == 100
        assert describe.run({"id": 2, "params": {"limit": 10}}, q)["described"] == 1  # the rest
        assert describe.run({"id": 3, "params": {}}, q)["described"] == 0

    def test_a_still_never_written_is_cut_or_skipped(self, store, monkeypatch, tmp_path):
        # Past the thumbnail cap the path is stored without the file —
        # the model failed on it run after run
        from whisper_tm.media import cuts

        (tmp_path / "0.jpg").unlink()
        (tmp_path / "1.jpg").unlink()
        cut = []

        def cut_still(src, t, out, *, width=480, timeout=60):
            cut.append((src, t))
            if t == 0:
                out.write_bytes(b"jpg")
                return True
            return False  # t=10: the recording can't be read

        monkeypatch.setattr(cuts, "cut_still", cut_still)
        seen = []
        monkeypatch.setattr(llm, "describe", lambda thumb: seen.append(thumb) or "A picture")
        monkeypatch.setattr(
            media_server.SERVER,
            "call",
            lambda path, payload=None, *, timeout=120: {
                "vectors": [b64(np.ones(8, dtype=np.float32)) for _ in payload["texts"]]
            },
        )
        out = describe.run({"id": 1, "params": {"limit": 10}}, self.Queue())
        assert out["described"] == 3 and out["failed"] == 0
        assert sorted(cut) == [("/lib/a.mp4", 0), ("/lib/a.mp4", 10)]
        assert str(tmp_path / "1.jpg") not in seen and (tmp_path / "0.jpg").is_file()

    def test_cancelling_keeps_what_was_written(self, store, monkeypatch):
        monkeypatch.setattr(llm, "describe", lambda thumb: "A picture")
        monkeypatch.setattr(
            media_server.SERVER,
            "call",
            lambda path, payload=None, *, timeout=120: {
                "vectors": [b64(np.ones(8, dtype=np.float32)) for _ in payload["texts"]]
            },
        )
        with pytest.raises(InterruptedError):
            describe.run({"id": 1, "params": {"limit": 4}}, self.Queue(abort_after=2))
        assert store.description_counts()["described"] == 2

    def test_a_model_that_never_answers_fails_the_run(self, store, monkeypatch):
        def broken(thumb):
            raise RuntimeError("timeout")

        monkeypatch.setattr(llm, "describe", broken)
        with pytest.raises(RuntimeError, match="did not describe"):
            describe.run({"id": 1, "params": {"limit": 4}}, self.Queue())
