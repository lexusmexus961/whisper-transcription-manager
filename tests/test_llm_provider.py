"""
LLM providers from the old single fields (core/llm_provider.py).

The template config describes ``llm.endpoint`` / ``llm.api_key`` /
``llm.model``. Whoever fills in only the key — the endpoint stays at its
empty default — must neither lose it on the next "Save" in the settings
nor end up with an unusable configuration.
"""

import pytest

from whisper_tm.config import config
from whisper_tm.core import llm_provider as lp


@pytest.fixture
def llm_fields(monkeypatch):
    """Old-style fields only; no provider list, no local model."""
    monkeypatch.setitem(
        config._data, "llm", {"endpoint": "", "api_key": "sk-test", "model": "gpt-4o-mini"}
    )
    monkeypatch.setattr(
        lp,
        "_local_provider",
        lambda: lp.Provider(
            id="local",
            name="Local (llama.cpp)",
            kind="llamacpp",
            endpoint="http://127.0.0.1:8080/v1",
        ),
    )
    monkeypatch.setattr(config, "save", lambda *a, **k: None)


def test_a_key_without_an_endpoint_is_openai(llm_fields):
    first = lp.all_items()[0]
    assert (first.kind, first.endpoint) == ("openai", "https://api.openai.com/v1")
    assert first.api_key == "sk-test" and first.default_model == "gpt-4o-mini"


def test_saving_what_the_settings_send_back_keeps_the_key(llm_fields):
    """The settings page sends the provider list back WITHOUT keys."""
    sent = [p.as_dict() for p in lp.all_items()]
    assert all("api_key" not in d for d in sent)
    lp.save(sent, in_file=False)
    assert config.get("llm", "api_key") == "sk-test"
    assert config.get("llm", "endpoint") == "https://api.openai.com/v1"
    assert config.get("llm", "model") == "gpt-4o-mini"


def test_nothing_configured_stays_nothing(llm_fields):
    config._data["llm"]["api_key"] = ""
    assert [p.id for p in lp.all_items()] == ["local"]
