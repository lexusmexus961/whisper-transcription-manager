"""A stored API key only ever goes to the endpoint it was stored for.

The settings never show a key, yet two paths sent the
stored one to any address the request named —

* ``POST /api/llm/providers/test`` with a provider's id and a foreign
  endpoint pinged that endpoint with ``Authorization: Bearer <stored key>``;
* saving a provider with a changed endpoint and no key kept the old key,
  and every later LLM call carried it (and the transcripts) there.

Now a stored key is reused only when the submitted endpoint is the stored
endpoint of that provider id (``llm_provider.stored_key``).
"""

import pytest
from flask import Flask

from whisper_tm.config import config
from whisper_tm.core import llm_provider as lp

OPENAI = "https://api.openai.com/v1"


@pytest.fixture
def providers(monkeypatch):
    monkeypatch.setitem(
        config._data,
        "llm",
        {
            "providers": [
                {
                    "id": "oa",
                    "name": "OpenAI",
                    "kind": "openai",
                    "endpoint": OPENAI,
                    "api_key": "sk-secret",
                    "models": ["gpt-4o-mini"],
                    "default_model": "gpt-4o-mini",
                    "active": True,
                }
            ]
        },
    )
    monkeypatch.setattr(lp, "_local_provider", lambda: None)
    monkeypatch.setattr(config, "save", lambda *a, **k: None)


def _stored():
    return {p.id: p for p in lp.all_items()}


class TestStoredKey:
    def test_same_endpoint_gets_the_key(self, providers):
        assert lp.stored_key("oa", OPENAI) == "sk-secret"
        # Spelling differences the normalization folds are the same address
        assert lp.stored_key("oa", OPENAI + "/models") == "sk-secret"
        assert lp.stored_key("oa", OPENAI + "/") == "sk-secret"

    def test_other_endpoint_gets_nothing(self, providers):
        assert lp.stored_key("oa", "http://attacker.example/v1") == ""
        assert lp.stored_key("nobody", OPENAI) == ""
        assert lp.stored_key("", OPENAI) == ""


class TestSave:
    def test_unchanged_endpoint_keeps_the_key(self, providers):
        sent = [p.as_dict() for p in lp.all_items()]
        lp.save(sent, in_file=False)
        assert _stored()["oa"].api_key == "sk-secret"

    def test_changed_endpoint_drops_the_key(self, providers):
        sent = [p.as_dict() for p in lp.all_items()]
        sent[0]["endpoint"] = "http://attacker.example/v1"
        lp.save(sent, in_file=False)
        assert _stored()["oa"].api_key == ""
        assert config.get("llm", "api_key") == ""

    def test_changed_endpoint_with_a_new_key_takes_it(self, providers):
        sent = [p.as_dict() for p in lp.all_items()]
        sent[0].update(endpoint="http://other.example/v1", api_key="sk-new")
        lp.save(sent, in_file=False)
        assert _stored()["oa"].api_key == "sk-new"


class TestTestRoute:
    @pytest.fixture
    def client(self, providers, monkeypatch):
        from whisper_tm.api.llm import llm_bp

        seen = []
        monkeypatch.setattr(
            lp,
            "detected_models",
            lambda endpoint, api_key="": seen.append((endpoint, api_key)) or {"reachable": True},
        )
        app = Flask(__name__)
        app.register_blueprint(llm_bp)
        return app.test_client(), seen

    def test_foreign_endpoint_never_gets_the_stored_key(self, client):
        c, seen = client
        c.post(
            "/api/llm/providers/test", json={"id": "oa", "endpoint": "http://attacker.example/v1"}
        )
        assert seen == [("http://attacker.example/v1", "")]

    def test_own_endpoint_gets_the_stored_key(self, client):
        c, seen = client
        c.post("/api/llm/providers/test", json={"id": "oa", "endpoint": OPENAI})
        assert seen == [(OPENAI, "sk-secret")]
