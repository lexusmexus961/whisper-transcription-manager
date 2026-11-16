"""The embedding API: state, server control, testing, benchmarking, guide.

Against the real app, but without a real server: the embedding function is
replaced. What is tested is the interface, not llama.cpp — its behavior
is covered in ``test_embed_server.py``.
"""

from __future__ import annotations

import pytest

from whisper_tm.config import config


@pytest.fixture
def client(tmp_path, monkeypatch):
    """A real app on a throwaway HOME.

    Not `create_app()` against the running configuration: that would
    touch the REAL database and start a second worker alongside the
    running app.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    config.load(None)
    config._data.setdefault("directories", {})["media"] = str(tmp_path)
    config._resolve_directories()
    config._ensure_directories()
    from whisper_tm.app import create_app

    return create_app().test_client()


@pytest.fixture
def clean():
    """Save embedding settings and reset them afterward."""
    old_value = dict(config.get("embedding") or {})
    yield
    for k, v in old_value.items():
        config.set("embedding", k, v)


# ── State ──────────────────────────────────────────────────────────────


def test_status_names_mode_and_situation(client, clean):
    config.set("embedding", "local", False)
    config.set("embedding", "endpoint", "http://example/v1")
    config.set("embedding", "model", "some-model")
    d = client.get("/api/embedding/status").get_json()
    assert d["local"] is False
    assert d["endpoint"] == "http://example/v1"
    assert d["model"] == "some-model"
    assert "local_server" in d and "possible" in d["local_server"]


def test_status_does_not_reveal_the_api_key(client, clean):
    """The key must not leave the server — only whether one is present."""
    config.set("embedding", "api_key", "sk-strictly-secret-123")
    raw = client.get("/api/embedding/status").get_data(as_text=True)
    assert "sk-strictly-secret-123" not in raw
    assert client.get("/api/embedding/status").get_json()["api_key_set"] is True


def test_start_without_local_mode_is_refused(client, clean):
    config.set("embedding", "local", False)
    r = client.post("/api/embedding/server/start")
    assert r.status_code == 400
    assert "not enabled" in r.get_json()["error"]


def test_stopping_without_a_server_is_harmless(client):
    d = client.post("/api/embedding/server/stop").get_json()
    assert d["ok"] is True and d["finished"] is False


# ── Test embedding ───────────────────────────────────────────────────────


def test_test_reports_dimension_and_normalization(client, clean, monkeypatch):
    import whisper_tm.rag.service as svc

    # A vector of length 1 — normalized.
    monkeypatch.setattr(svc, "make_embed_fn", lambda purpose="index": lambda t: [[0.6, 0.8]])
    config.set("embedding", "local", False)
    config.set("embedding", "endpoint", "http://x/v1")
    config.set("embedding", "model", "m")
    d = client.post("/api/embedding/test", json={}).get_json()
    assert d["ok"] and d["dimensions"] == 2
    assert d["normalized"] is True


def test_test_reports_unnormalized_output(client, clean, monkeypatch):
    """Not every provider normalizes — that needs to be made visible.

    It's not a bug (the store normalizes on write), but anyone computing
    a similarity by hand needs to know it.
    """
    import whisper_tm.rag.service as svc

    monkeypatch.setattr(svc, "make_embed_fn", lambda purpose="index": lambda t: [[3.0, 4.0]])
    config.set("embedding", "local", False)
    config.set("embedding", "endpoint", "http://x/v1")
    config.set("embedding", "model", "m")
    d = client.post("/api/embedding/test", json={}).get_json()
    assert d["ok"] and d["normalized"] is False and d["length"] == 5.0


def test_a_misconfiguration_is_400_not_success(client, clean):
    config.set("embedding", "local", False)
    config.set("embedding", "endpoint", "")
    config.set("embedding", "model", "")
    r = client.post("/api/embedding/test", json={})
    assert r.status_code == 400
    assert r.get_json()["ok"] is False


def test_a_server_error_is_502(client, clean, monkeypatch):
    """ "You didn't enter anything" and "the server isn't responding" are
    different situations and get different statuses."""
    import whisper_tm.rag.service as svc

    def broken(purpose="index"):
        def _e(t):
            raise RuntimeError("Embedding-API 501: not supported")

        return _e

    monkeypatch.setattr(svc, "make_embed_fn", broken)
    config.set("embedding", "local", False)
    config.set("embedding", "endpoint", "http://x/v1")
    config.set("embedding", "model", "m")
    r = client.post("/api/embedding/test", json={})
    assert r.status_code == 502
    assert "501" in r.get_json()["error"]


# ── Throughput ────────────────────────────────────────────────────────────


def test_the_benchmark_measures_several_batch_sizes(client, clean, monkeypatch):
    import whisper_tm.api.embedding as api
    import whisper_tm.rag.service as svc

    monkeypatch.setattr(api, "_sample_chunks", lambda db, n: ["A chunk with some text."] * n)
    monkeypatch.setattr(api, "_estimate_open_chunks", lambda db: 1000)
    monkeypatch.setattr(
        svc, "make_embed_fn", lambda purpose="index": lambda t: [[0.0, 1.0] for _ in t]
    )
    d = client.post("/api/embedding/benchmark", json={"sizes": [2, 4]}).get_json()
    assert d["ok"]
    assert [r["batch"] for r in d["results"]] == [2, 4]
    assert all(r["chunks"] > 0 for r in d["results"])
    assert d["best_rate"] > 0
    assert d["estimated_chunks"] == 1000


def test_benchmark_without_transcripts_says_so(client, clean, monkeypatch):
    import whisper_tm.api.embedding as api

    monkeypatch.setattr(api, "_sample_chunks", lambda db, n: [])
    r = client.post("/api/embedding/benchmark", json={})
    assert r.status_code == 400
    assert "transcripts" in r.get_json()["error"]


def test_benchmark_caps_absurd_sizes(client, clean, monkeypatch):
    """A batch of 100,000 wouldn't be a measurement, it would be a timeout."""
    import whisper_tm.api.embedding as api
    import whisper_tm.rag.service as svc

    seen = []
    monkeypatch.setattr(api, "_sample_chunks", lambda db, n: ["x"] * n)
    monkeypatch.setattr(api, "_estimate_open_chunks", lambda db: 0)

    def fn(purpose="index"):
        def _e(t):
            seen.append(len(t))
            return [[1.0] for _ in t]

        return _e

    monkeypatch.setattr(svc, "make_embed_fn", fn)
    d = client.post("/api/embedding/benchmark", json={"sizes": [100000]}).get_json()
    assert d["ok"]
    assert max(seen) <= 512


# ── The guide ────────────────────────────────────────────────────────


def test_the_guide_delivers_a_usable_prompt(client):
    d = client.get("/api/embedding/guide").get_json()
    p = d["prompt"]
    # The three known pitfalls MUST be in there —
    # otherwise the prompt leads straight into them.
    assert "--embedding" in p, "the 501 error without --embedding is missing"
    assert "ubatch" in p.lower(), "the ubatch pitfall is missing"
    assert "n-gpu-layers" in p, "the GPU layer self-adjustment is missing"
    # And this installation's port, so the answer fits.
    assert str(config.get("embedding", "local_port", default=8081)) in p
    assert len(p) > 800


def test_the_guide_fills_in_the_local_hardware(client, monkeypatch):
    import whisper_tm.api.embedding as api

    monkeypatch.setattr(api, "_hardware_brief", lambda: "GPU: TestCard (7 GB VRAM)")
    d = client.get("/api/embedding/guide").get_json()
    assert "TestCard" in d["prompt"]
    assert d["hardware"] == "GPU: TestCard (7 GB VRAM)"


def test_the_hardware_summary_never_throws(monkeypatch):
    """Without detected hardware, the guide must still come through."""
    import whisper_tm.api.embedding as api
    import whisper_tm.extensions as ext

    monkeypatch.setattr(ext, "hw", None)
    assert api._hardware_brief() == "unknown"


# ── The instruction prefix for questions ──────────────────────────────────


def test_the_question_gets_the_prefix(clean):
    """Questions and text passages are embedded DIFFERENTLY.

    Qwen3-Embedding is trained for asymmetric retrieval. Measured on a
    real index: with the prefix, the best hit changed for most test
    questions, and where short filler lines used to be on top, a real
    hit appeared.
    """
    from whisper_tm.rag.service import prepare_question

    config.set("embedding", "question_template", "Instruct: something\nQuery:{question}")
    assert prepare_question("Where is the car?") == "Instruct: something\nQuery:Where is the car?"


def test_an_empty_template_leaves_the_question_alone(clean):
    """bge-m3 and relatives want NO prefix — empty must stay empty."""
    from whisper_tm.rag.service import prepare_question

    config.set("embedding", "question_template", "")
    assert prepare_question("unchanged") == "unchanged"


def test_a_template_without_a_placeholder_is_ignored(clean):
    """A template without {question} would swallow the question.

    Better to embed it unchanged than to send a query in which the
    question no longer appears at all — the user would get hits for a
    text they never entered.
    """
    from whisper_tm.rag.service import prepare_question

    config.set("embedding", "question_template", "Instruct: broken, no placeholder")
    assert prepare_question("my question") == "my question"


def test_query_fn_embeds_with_the_prefix(clean, monkeypatch):
    """The whole path: make_query_embed_fn sends the prepared version."""
    import whisper_tm.rag.service as svc

    seen = []
    monkeypatch.setattr(
        svc,
        "make_embed_fn",
        lambda purpose="index": lambda t: (seen.extend(t), [[1.0]] * len(t))[1],
    )
    config.set("embedding", "question_template", "P:{question}")
    svc.make_query_embed_fn()(["abc"])
    assert seen == ["P:abc"]


def test_texts_get_NO_prefix(clean, monkeypatch):
    """The indexer must NOT prefix the passages.

    Otherwise retrieval would be symmetric again — just with a prefix on
    both sides, which the training doesn't account for either.
    """
    import whisper_tm.rag.service as svc

    seen = []
    monkeypatch.setattr(
        svc.llm_client, "embed_texts", lambda t, **k: (seen.extend(t), [[1.0]] * len(t))[1]
    )
    config.set("embedding", "local", False)
    config.set("embedding", "endpoint", "http://x/v1")
    config.set("embedding", "model", "m")
    config.set("embedding", "question_template", "P:{question}")
    svc.make_embed_fn()(["a text passage"])
    assert seen == ["a text passage"]


# ── The stored key only goes to the stored endpoint ──────────────────────
# Security fix: /api/embedding/test used to send the configured API key as a
# Bearer token to ANY endpoint named in the request body — anyone on the
# network could collect it with {"endpoint": "http://their-host/v1"}.


def _capture(monkeypatch):
    from whisper_tm.core import llm_client

    seen = {}

    def embed(texts, endpoint, model, api_key="", timeout=120):
        seen.update(endpoint=endpoint, api_key=api_key)
        return [[1.0, 0.0]]

    monkeypatch.setattr(llm_client, "embed_texts", embed)
    return seen


def test_the_stored_key_does_not_go_to_another_endpoint(client, clean, monkeypatch):
    config.set("embedding", "local", False)
    config.set("embedding", "endpoint", "http://configured/v1")
    config.set("embedding", "api_key", "sk-strictly-secret-123")
    seen = _capture(monkeypatch)
    r = client.post("/api/embedding/test", json={"endpoint": "http://attacker/v1"})
    assert r.get_json()["ok"] is True
    assert seen["endpoint"] == "http://attacker/v1"
    assert seen["api_key"] == ""


def test_a_key_sent_along_is_used_for_another_endpoint(client, clean, monkeypatch):
    config.set("embedding", "local", False)
    config.set("embedding", "endpoint", "http://configured/v1")
    config.set("embedding", "api_key", "sk-strictly-secret-123")
    seen = _capture(monkeypatch)
    client.post("/api/embedding/test", json={"endpoint": "http://new/v1", "api_key": "sk-new"})
    assert seen["api_key"] == "sk-new"


def test_the_stored_key_goes_to_the_stored_endpoint(client, clean, monkeypatch):
    config.set("embedding", "local", False)
    config.set("embedding", "endpoint", "http://configured/v1")
    config.set("embedding", "api_key", "sk-strictly-secret-123")
    seen = _capture(monkeypatch)
    client.post("/api/embedding/test", json={"endpoint": "http://configured/v1/"})
    assert seen["api_key"] == "sk-strictly-secret-123"
