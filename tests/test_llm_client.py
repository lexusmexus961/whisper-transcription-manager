"""
Tests for core/llm_client.py (URL resolution, ping, embeddings) and the
new LLM/embedding/connection endpoints.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from whisper_tm.core import llm_client as c

# ── URL Resolution ─────────────────────────────────────────────────────────


class TestUrlResolution:
    @pytest.mark.parametrize(
        "endpoint,expected_url,expected_ollama",
        [
            (
                "http://localhost:8080/v1",
                "http://localhost:8080/v1/chat/completions",
                False,
            ),  # llama.cpp
            ("https://api.openai.com/v1", "https://api.openai.com/v1/chat/completions", False),
            ("http://localhost:11434", "http://localhost:11434/api/chat", True),  # Ollama native
            ("http://localhost:11434/v1", "http://localhost:11434/v1/chat/completions", False),
            ("http://h/v1/chat/completions", "http://h/v1/chat/completions", False),
            (
                "http://localhost:8080/v1/",
                "http://localhost:8080/v1/chat/completions",
                False,
            ),  # trailing slash
        ],
    )
    def test_chat_url(self, endpoint, expected_url, expected_ollama):
        url, ollama = c.resolve_chat_url(endpoint)
        assert url == expected_url
        assert ollama == expected_ollama

    @pytest.mark.parametrize(
        "endpoint,expected_url,expected_ollama",
        [
            ("http://localhost:8080/v1", "http://localhost:8080/v1/embeddings", False),
            ("http://localhost:11434", "http://localhost:11434/api/embeddings", True),
            ("https://api.openai.com/v1", "https://api.openai.com/v1/embeddings", False),
        ],
    )
    def test_embeddings_url(self, endpoint, expected_url, expected_ollama):
        url, ollama = c.resolve_embeddings_url(endpoint)
        assert url == expected_url
        assert ollama == expected_ollama

    def test_is_openai_style(self):
        assert c.is_openai_style("http://x/v1")
        assert c.is_openai_style("http://x/v1/chat/completions")
        assert not c.is_openai_style("http://x:11434")
        assert not c.is_openai_style("")


# ── Mock Server (llama.cpp / OpenAI-compatible) ────────────────────────────


class _MockHandler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, obj):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(obj).encode())

    def do_GET(self):
        if self.path == "/v1/models":
            self._send(200, {"data": [{"id": "llama-3-8b"}, {"id": "nomic-embed-text"}]})
        elif self.path == "/health":
            self._send(200, {"status": "ok"})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        if self.path == "/v1/embeddings":
            inp = body.get("input", [])
            if isinstance(inp, str):
                inp = [inp]
            self._send(
                200,
                {
                    "data": [
                        {"index": i, "embedding": [0.1, 0.2, 0.3, 0.4]} for i in range(len(inp))
                    ]
                },
            )
        else:
            self._send(404, {"error": "not found"})


@pytest.fixture
def mock_server():
    srv = HTTPServer(("127.0.0.1", 0), _MockHandler)
    port = srv.server_address[1]
    t = threading.Thread(
        target=srv.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True
    )  # shutdown() waits one poll
    t.start()
    yield f"http://127.0.0.1:{port}/v1"
    srv.shutdown()


class TestPing:
    def test_reachable_lists_models(self, mock_server):
        p = c.ping_endpoint(mock_server)
        assert p["reachable"] is True
        assert p["api_type"] == "openai"
        assert "nomic-embed-text" in p["models"]

    def test_unreachable(self):
        p = c.ping_endpoint("http://127.0.0.1:1/v1", timeout=2)
        assert p["reachable"] is False

    def test_empty_endpoint(self):
        p = c.ping_endpoint("")
        assert p["reachable"] is False
        assert p["detail"] == "no endpoint"


class TestEmbed:
    def test_embed_batch(self, mock_server):
        vecs = c.embed_texts(["a", "b"], endpoint=mock_server, model="nomic-embed-text")
        assert len(vecs) == 2
        assert len(vecs[0]) == 4

    def test_embed_empty(self, mock_server):
        assert c.embed_texts([], endpoint=mock_server, model="m") == []


# ── API Endpoints ──────────────────────────────────────────────────────────


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    from whisper_tm.config import config

    config.load(None)
    config._data.setdefault("directories", {})["media"] = str(tmp_path)
    config._resolve_directories()
    config._ensure_directories()
    from whisper_tm.app import create_app

    app = create_app()
    return app.test_client()


class TestEndpoints:
    def test_embedding_status_unconfigured(self, client):
        # Previously own endpoint /api/embedding/status — removed in UI rework,
        # the combined /api/connection/status provides the same information.
        d = client.get("/api/connection/status").get_json()
        assert d["embedding"]["configured"] is False

    def test_connection_status_shape(self, client):
        d = client.get("/api/connection/status").get_json()
        assert "llm" in d and "embedding" in d
        # without sample: reachable remains None
        assert d["llm"]["reachable"] is None

    def test_embedding_test_requires_config(self, client):
        r = client.post("/api/embedding/test", json={})
        assert r.status_code == 400
        assert not r.get_json()["ok"]

    def test_connection_status_probe(self, client, monkeypatch):
        # Configure endpoint + mock ping
        from whisper_tm.config import config

        config.set("embedding", "endpoint", "http://x/v1")
        config.set("embedding", "model", "bge-m3")
        from whisper_tm.core import llm_client

        monkeypatch.setattr(
            llm_client,
            "ping_endpoint",
            lambda ep, key="", timeout=6: {
                "reachable": True,
                "api_type": "openai",
                "models": ["bge-m3"],
                "detail": "ok",
            },
        )
        d = client.get("/api/connection/status?sample=1").get_json()
        assert d["embedding"]["configured"] is True
        assert d["embedding"]["reachable"] is True


class TestTheEmbeddingTrafficLight:
    """**The finding:** the embedding indicator stayed gray even though it
    should have been green.

    The indicator read `embedding.endpoint` and `embedding.model` — the fields of the
    FOREIGN endpoint. Whoever uses their own server on demand has nothing there
    (their settings are called `local_*`), and the indicator
    reported constantly 'Not configured — click to set up', even though
    the server was set up and running.
    """

    def test_local_mode_counts_as_configured(self, client, monkeypatch, tmp_path):
        from whisper_tm.config import config
        from whisper_tm.core import embed_server as es

        bin_ = tmp_path / "llama-server"
        bin_.touch()
        mod = tmp_path / "model.gguf"
        mod.touch()
        monkeypatch.setattr(es, "binary", lambda: bin_)
        monkeypatch.setattr(es, "model_file", lambda: mod)
        config.set("embedding", "local", True)
        config.set("embedding", "endpoint", "")  # belongs to foreign operation
        d = client.get("/api/connection/status").get_json()["embedding"]
        assert d["configured"] is True
        assert d["mode"] == "local"

    def test_a_sleeping_server_is_not_an_outage(self, client, monkeypatch, tmp_path):
        """After five minutes without requests it shuts down by itself and
        frees its VRAM. This is the normal case — red would be a lie."""
        from whisper_tm.config import config
        from whisper_tm.core import embed_server as es

        bin_ = tmp_path / "llama-server"
        bin_.touch()
        mod = tmp_path / "model.gguf"
        mod.touch()
        monkeypatch.setattr(es, "binary", lambda: bin_)
        monkeypatch.setattr(es, "model_file", lambda: mod)
        monkeypatch.setattr(
            es.SERVER,
            "state",
            lambda: {"reachable": False, "running_for_sec": 0.0, "idle_limit_sec": 300},
        )
        config.set("embedding", "local", True)
        d = client.get("/api/connection/status?sample=1").get_json()["embedding"]
        assert d["configured"] is True
        assert d["reachable"] is not False, "idle is not a failure"
        assert d["ready"] is True
        assert "starts on demand" in d["detail"]

    def test_a_running_server_reports_as_connected(self, client, monkeypatch, tmp_path):
        from whisper_tm.config import config
        from whisper_tm.core import embed_server as es

        bin_ = tmp_path / "llama-server"
        bin_.touch()
        mod = tmp_path / "model.gguf"
        mod.touch()
        monkeypatch.setattr(es, "binary", lambda: bin_)
        monkeypatch.setattr(es, "model_file", lambda: mod)
        monkeypatch.setattr(
            es.SERVER,
            "state",
            lambda: {"reachable": True, "running_for_sec": 640.0, "idle_limit_sec": 300},
        )
        config.set("embedding", "local", True)
        d = client.get("/api/connection/status?sample=1").get_json()["embedding"]
        assert d["reachable"] is True and d["running"] is True

    def test_a_missing_model_file_is_red_not_grey(self, client, monkeypatch, tmp_path):
        """The difference matters: Gray means 'set it up', red means
        'you set it up, but something is missing' — and it says what."""
        from whisper_tm.config import config
        from whisper_tm.core import embed_server as es

        bin_ = tmp_path / "llama-server"
        bin_.touch()
        monkeypatch.setattr(es, "binary", lambda: bin_)
        monkeypatch.setattr(es, "model_file", lambda: tmp_path / "gone.gguf")
        config.set("embedding", "local", True)
        d = client.get("/api/connection/status?sample=1").get_json()["embedding"]
        assert d["configured"] is False
        assert d["reachable"] is False
        assert "gone.gguf" in d["detail"]

    def test_the_fast_first_call_does_not_knock(self, client, monkeypatch, tmp_path):
        """Without `sample=1` no network traffic is allowed — the call sits
        in the page's load path."""
        from whisper_tm.config import config
        from whisper_tm.core import embed_server as es

        bin_ = tmp_path / "llama-server"
        bin_.touch()
        mod = tmp_path / "model.gguf"
        mod.touch()
        monkeypatch.setattr(es, "binary", lambda: bin_)
        monkeypatch.setattr(es, "model_file", lambda: mod)

        def do_not_ask():
            raise AssertionError("the fast call knocked")

        monkeypatch.setattr(es.SERVER, "state", do_not_ask)
        config.set("embedding", "local", True)
        d = client.get("/api/connection/status").get_json()["embedding"]
        assert d["configured"] is True and d["ready"] is True

    def test_the_foreign_endpoint_stays_as_it_was(self, client, monkeypatch):
        """Whoever turns off `local` will continue to be evaluated via endpoint/model."""
        from whisper_tm.config import config
        from whisper_tm.core import llm_client

        config.set("embedding", "local", False)
        config.set("embedding", "endpoint", "http://foreign/v1")
        config.set("embedding", "model", "bge-m3")
        monkeypatch.setattr(
            llm_client,
            "ping_endpoint",
            lambda ep, key="", timeout=6: {
                "reachable": True,
                "api_type": "openai",
                "models": [],
                "detail": "ok",
            },
        )
        d = client.get("/api/connection/status?sample=1").get_json()["embedding"]
        assert d["configured"] is True and d["reachable"] is True
        assert d.get("mode") != "local"


# ── chat_template_kwargs: only for a llama.cpp server ───────────────────────


class _ChatHandler(BaseHTTPRequestHandler):
    """A chat server that remembers the bodies it got; ``/props`` like
    llama.cpp when ``llamacpp`` is set."""

    bodies: list = []
    props_calls: list = []
    llamacpp = True

    def log_message(self, *a):
        pass

    def _send(self, code, obj):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/props":
            type(self).props_calls.append(1)
            if type(self).llamacpp:
                self._send(200, {"chat_template_caps": {"supports_tools": True}})
            else:
                self._send(404, {"error": "no"})
        else:
            self._send(404, {})

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        type(self).bodies.append(json.loads(self.rfile.read(n) or b"{}"))
        self._send(
            200,
            {
                "model": "the-loaded-one",
                "choices": [{"message": {"content": "YES"}, "finish_reason": "stop"}],
            },
        )


@pytest.fixture
def chat_server(monkeypatch):
    from whisper_tm.config import config

    _ChatHandler.bodies, _ChatHandler.props_calls = [], []
    srv = HTTPServer(("127.0.0.1", 0), _ChatHandler)
    t = threading.Thread(
        target=srv.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True
    )  # shutdown() waits one poll
    t.start()
    ep = f"http://127.0.0.1:{srv.server_address[1]}/v1"
    monkeypatch.setitem(config._data, "llm", {"endpoint": ep, "model": "m", "api_key": ""})
    monkeypatch.setattr(c, "_task_provider", lambda task: ("", "", ""))
    monkeypatch.setattr(c, "effective_context", lambda task="": 0)
    c._llamacpp.clear()
    yield ep
    srv.shutdown()
    c._llamacpp.clear()


@pytest.mark.parametrize("llamacpp", [True, False])
def test_template_kwargs_go_only_to_a_llamacpp_server(chat_server, llamacpp):
    _ChatHandler.llamacpp = llamacpp
    text, _usage = c.chat(
        [{"role": "user", "content": "x"}],
        max_tokens=5,
        thinking=False,
        template_kwargs={"enable_thinking": False},
        retries=0,
    )
    assert text == "YES"
    body = _ChatHandler.bodies[-1]
    if llamacpp:
        assert body["chat_template_kwargs"] == {"enable_thinking": False}
    else:
        assert "chat_template_kwargs" not in body
    assert body["max_tokens"] == 5


def test_without_template_kwargs_the_server_is_not_asked(chat_server):
    _ChatHandler.llamacpp = True
    c.chat([{"role": "user", "content": "x"}], retries=0)
    assert "chat_template_kwargs" not in _ChatHandler.bodies[-1]
    assert _ChatHandler.props_calls == []


def test_the_llamacpp_question_is_asked_once(chat_server):
    _ChatHandler.llamacpp = True
    url = chat_server + "/chat/completions"
    assert c.is_llamacpp(url) and c.is_llamacpp(url)
    assert len(_ChatHandler.props_calls) == 1
    from whisper_tm.core import chat

    assert chat._is_llamacpp(url) is True
    assert len(_ChatHandler.props_calls) == 1


def test_the_answer_names_the_model_that_wrote_it(chat_server):
    """llama.cpp answers with the model it loaded, not the name asked for."""
    _ChatHandler.llamacpp = False
    _text, usage = c.chat([{"role": "user", "content": "x"}], model="gpt-4o-mini", retries=0)
    assert _ChatHandler.bodies[-1]["model"] == "gpt-4o-mini"
    assert usage["model"] == "the-loaded-one"
