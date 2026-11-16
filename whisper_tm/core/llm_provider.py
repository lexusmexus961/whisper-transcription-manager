"""
LLM providers: multiple providers side by side (llama.cpp, Ollama, OpenAI …).

With only ONE endpoint field, anyone running llama.cpp locally and wanting
to add OpenAI for difficult cases would have to switch manually in
settings. Instead:

* ``llm.providers`` — a list of named providers with type, endpoint,
  key and own model list.
* ``llm.tasks`` — per task (summary/correction/translation/
  embedding) optionally a different provider and different model.
  Not set = default provider.

Backward compatible: a config with old fields ``llm.endpoint``/
``llm.model`` is read on first access as provider "Standard" —
without anyone needing to change anything. Only when a provider
is created in settings does the new structure appear in config.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from whisper_tm.config import config

logger = logging.getLogger(__name__)

# Type → (display name, endpoint template, hint)
TYPES = {
    "llamacpp": (
        "llama.cpp",
        "http://localhost:8080/v1",
        "llama-server with --host 0.0.0.0 --port 8080. Context size is detected automatically.",
    ),
    "ollama": (
        "Ollama",
        "http://localhost:11434",
        "Without /v1 = native API (supports think). Models come from /api/tags.",
    ),
    "openai": ("OpenAI", "https://api.openai.com/v1", "API key required. Costs per token."),
    "vllm": ("vLLM / LM Studio", "http://localhost:8000/v1", "OpenAI-compatible."),
    "custom": ("Other (OpenAI-compatible)", "", "Endpoint with /v1."),
}

# Embedding intentionally missing: it has its own config section
# (embedding.endpoint/model) and doesn't run through chat paths.
# "chat": the Ask tab of the search (core/chat.py) — its own choice of
# model, as a conversation wants a fast one and a summary a thorough one
TASKS = ("summary", "correction", "translation", "chat")


@dataclass
class Provider:
    """A configured provider."""

    id: str
    name: str
    kind: str = "custom"
    endpoint: str = ""
    api_key: str = ""
    models: list[str] = field(default_factory=list)
    default_model: str = ""
    active: bool = True

    def as_dict(self, with_key: bool = False) -> dict:
        d = {
            "id": self.id,
            "name": self.name,
            "kind": self.kind,
            "endpoint": self.endpoint,
            "models": list(self.models),
            "default_model": self.default_model,
            "active": self.active,
            "has_key": bool(self.api_key),
        }
        if with_key:
            d["api_key"] = self.api_key
        return d


def _normalize_endpoint(endpoint: str) -> str:
    """Fix common input errors: if someone enters the doc URL for the model
    list or chat path, they mean the BASE URL — otherwise the client builds
    /v1/models/chat/completions from it and the provider "doesn't work"
    (e.g. https://api.openai.com/v1/models)."""
    e = endpoint.strip().rstrip("/")
    for attachment in ("/chat/completions", "/completions", "/models"):
        if e.endswith(attachment):
            e = e[: -len(attachment)]
            break
    return e


def _from_dict(d: dict) -> Provider:
    models = d.get("models") or []
    if isinstance(models, str):
        models = [m.strip() for m in models.split(",") if m.strip()]
    return Provider(
        id=str(d.get("id") or "").strip() or "standard",
        name=str(d.get("name") or "").strip() or "Unnamed",
        kind=str(d.get("kind") or "custom"),
        endpoint=_normalize_endpoint(str(d.get("endpoint") or "")),
        api_key=str(d.get("api_key") or ""),
        models=[str(m) for m in models if str(m).strip()],
        default_model=str(d.get("default_model") or "").strip(),
        active=bool(d.get("active", True)),
    )


def _legacy_provider() -> Provider | None:
    """Read the old individual fields as provider "Standard".

    A key without an endpoint counts as OpenAI. That is what the template
    suggests (``model: gpt-4o-mini``, the key comment), and it is the one
    reading under which a key alone makes sense. Without it the key
    belonged to no provider: the settings showed only the local server,
    and the first "Save" mirrored that server into the old fields and
    overwrote the key with an empty string.
    """
    endpoint = config.get("llm", "endpoint", default="")
    if not endpoint:
        if not config.get("llm", "api_key", default=""):
            return None
        endpoint = TYPES["openai"][1]
    models_dict = config.get("llm", "models", default={}) or {}
    models = [m for m in models_dict.values() if isinstance(m, str)]
    model = config.get("llm", "model", default="")
    if model and model not in models:
        models.insert(0, model)
    kind = (
        "ollama"
        if "11434" in endpoint and "/v1" not in endpoint
        else (
            "openai"
            if "openai.com" in endpoint
            else ("llamacpp" if ":8080" in endpoint else "custom")
        )
    )
    return Provider(
        id="standard",
        name="Standard",
        kind=kind,
        endpoint=endpoint,
        api_key=config.get("llm", "api_key", default=""),
        models=models,
        default_model=model,
    )


def _local_provider() -> Provider | None:
    """The supplied local llama.cpp server — None if it isn't installed.

    Offered even if nothing is configured — otherwise a fresh system would
    have no provider, even though setup brought one. Whether it's RUNNING
    is shown by health check.

    Only with a llama-server binary, though (found the same way the start
    command finds it, ``llama_server.binary()``). Without one nothing can
    ever answer on that endpoint, and because ``all_items()`` adds this
    provider anew on every read, deleting it in the settings never stuck:
    it was back on the next load.
    """
    from whisper_tm.core import llama_server

    if not llama_server.binary():
        return None
    model = llama_server.model_file()
    return Provider(
        id="local",
        name="Local (llama.cpp)",
        kind="llamacpp",
        endpoint=llama_server.endpoint(),
        models=[model.stem] if model else [],
        default_model=model.stem if model else "",
    )


def all_items() -> list[Provider]:
    """All providers — configured first, then the local server if installed.

    Configured providers are listed as saved, whether or not a binary
    exists; only the automatic addition depends on it.
    """
    raw = config.get("llm", "providers", default=None)
    if isinstance(raw, list) and raw:
        listing = [_from_dict(d) for d in raw if isinstance(d, dict)]
    else:
        legacy = _legacy_provider()
        listing = [legacy] if legacy else []
    # Add local server if installed and not already present (under a
    # different id)
    try:
        local = _local_provider()
        present = local is None or any(
            p.endpoint.rstrip("/") == local.endpoint.rstrip("/") or p.id == local.id
            for p in listing
        )
        if not present:
            listing.append(local)
    except Exception as e:
        logger.debug("Local provider: %s", e)
    return listing


def stored_key(
    provider_id: str, endpoint: str, inventory: dict[str, Provider] | None = None
) -> str:
    """The stored API key of ``provider_id`` — but only for its own endpoint.

    The settings never show a key and never send one back, so "no key
    given" has to mean "keep the stored one". That must not extend to a
    different address: otherwise the "Test" button would ping any endpoint
    it was given with the stored key, and a save with a changed endpoint
    would carry the key — and every transcript after it — to the new host.
    A changed endpoint therefore needs its key typed in again.

    ``endpoint`` is compared after the same normalization the provider
    list applies. ``inventory``: ``{id: Provider}`` if the caller has it.
    Returns "" when there is no match.
    """
    if not provider_id:
        return ""
    if inventory is None:
        inventory = {p.id: p for p in all_items()}
    present = inventory.get(str(provider_id))
    if present is None or not present.api_key:
        return ""
    if _normalize_endpoint(str(endpoint or "")) != _normalize_endpoint(present.endpoint):
        return ""
    return present.api_key


def save(provider_list: list[dict], *, in_file: bool = True) -> None:
    """Write provider list to config (keys are preserved).

    An empty ``api_key`` does NOT overwrite an existing key —
    the UI never sends keys back — as long as the endpoint stays the
    stored one (:func:`stored_key`); a changed endpoint drops it. The same applies to the
    masking placeholder that GET /api/settings inserts: storing it as a real
    key would destroy the existing one.

    ``in_file=False`` changes only the running config: PUT /api/settings
    passes through its ``?save=true``, instead of provider changes ALWAYS
    landing on disk while all other settings respect the flag (otherwise a
    test run would write its test providers to the real config file).
    """
    from whisper_tm.config import SECRET_PLACEHOLDER

    inventory = {p.id: p for p in all_items()}
    clean = []
    for d in provider_list:
        p = _from_dict(d)
        if p.api_key == SECRET_PLACEHOLDER:
            p.api_key = ""
        if not p.api_key:
            # Only to the address it was stored for (see stored_key)
            p.api_key = stored_key(p.id, p.endpoint, inventory)
        clean.append(
            {
                "id": p.id,
                "name": p.name,
                "kind": p.kind,
                "endpoint": p.endpoint,
                "api_key": p.api_key,
                "models": p.models,
                "default_model": p.default_model,
                "active": p.active,
            }
        )
    config.set("llm", "providers", clean)
    # First active element stays mirrored as legacy field, so
    # existing code (and connection indicator) continue working.
    first = next((p for p in clean if p.get("active")), None)
    if first:
        config.set("llm", "endpoint", first["endpoint"])
        config.set("llm", "api_key", first["api_key"])
        # A provider WITHOUT default_model (endpoint entered, never
        # clicked "Test") must not overwrite working llm.model with
        # empty string — same backward compatibility as api_key above.
        if first["default_model"]:
            config.set("llm", "model", first["default_model"])
    else:
        # No active provider left → also clean up legacy fields.
        # Otherwise LLM stays quietly configured and sends transcripts
        # (with stored key) to the just-deleted endpoint — and there's
        # no field for manual clearing since the card refactoring.
        config.set("llm", "endpoint", "")
        config.set("llm", "api_key", "")
        config.set("llm", "model", "")
    if in_file:
        config.save()


def for_model(model: str) -> Provider | None:
    """The active provider that has this model (default model or
    model list). None if none has it.

    This way an explicitly chosen model in the dialog pulls ITS provider
    along: choosing "gpt-4.1-nano" means using OpenAI — otherwise the
    call would go to the task provider (llama.cpp), which silently ignores
    the foreign model name."""
    if not model:
        return None
    for p in all_items():
        if p.active and (model == p.default_model or model in (p.models or [])):
            return p
    return None


def standard() -> Provider | None:
    for p in all_items():
        if p.active:
            return p
    return None


def for_task(task: str) -> tuple[Provider | None, str]:
    """(Provider, model) for a task — with fallback to standard.

    ``llm.tasks.<task>`` = {"provider": id, "model": name}
    """
    assignment = config.get("llm", "tasks", task, default={}) or {}
    provider_id = str(assignment.get("provider") or "")
    model = str(assignment.get("model") or "")
    listing = all_items()
    p = next((x for x in listing if x.id == provider_id and x.active), None) or standard()
    if p and not model:
        model = p.default_model
    return p, model


def detected_models(endpoint: str, api_key: str = "") -> dict:
    """Query models and context size of an endpoint (for test button)."""
    from whisper_tm.core import llm_client

    ping = llm_client.ping_endpoint(endpoint, api_key, timeout=8)
    context = 0
    if ping["reachable"]:
        try:
            context = llm_client.detect_context(endpoint, api_key)
        except Exception as e:
            logger.debug("Context detection: %s", e)
    return {
        "reachable": ping["reachable"],
        "detail": ping["detail"],
        "api_type": ping["api_type"],
        "models": ping.get("models") or [],
        "context": context,
    }
