"""
LLM routes: transcript summarization via OpenAI-compatible API.
"""

import json
import logging

from flask import Blueprint, Response, jsonify, request, stream_with_context

from whisper_tm.config import config
from whisper_tm.core import llm_client
from whisper_tm.core import summarize as _summarize_core
from whisper_tm.core.helpers import parse_bool
from whisper_tm.core.llm import prompts as _prompts

logger = logging.getLogger(__name__)

# The summarization core lives in core/summarize.py — the same logic is used
# by the queue for automatic summarization after transcription (no
# browser, no stream). The routes here keep using it under the previous
# names; that way it continues to exist exactly ONCE.
_build_messages = _summarize_core.build_messages
_format_transcript = _summarize_core.format_transcript
_load_segments = _summarize_core.load_segments
_load_summaries = _summarize_core.load_summaries
_save_summaries = _summarize_core.save_summaries
chat_request = _summarize_core.chat_request

llm_bp = Blueprint("llm", __name__)


def _provider_models() -> list[dict]:
    """Default models of all active providers (for the dialog dropdowns)."""
    try:
        from whisper_tm.core import llm_provider as lp

        return [
            {"model": p.default_model, "providers": p.name}
            for p in lp.all_items()
            if p.active and p.default_model
        ]
    except Exception:
        return []


def _task_models() -> dict:
    """Per task, the model a run uses when the dialog sends no model.

    The dialogs name it in their "default" entry. That entry sends '' so
    the task → provider assignment (``llm.tasks``) decides; the legacy
    ``llm.model`` the entry used to show is not necessarily what the
    assignment picks.
    """
    try:
        from whisper_tm.core import llm_provider as lp

        return {t: llm_client.task_default_model(t) for t in lp.TASKS}
    except Exception:
        return {}


def _correction_score_threshold() -> float:
    """``llm.correction.score_threshold`` — the correction dialog fills its
    field with it; 0.40 is the same fallback the correction itself uses."""
    try:
        return float(config.get("llm", "correction", "score_threshold", default=0.40))
    except (TypeError, ValueError):
        return 0.40


@llm_bp.route("/api/llm/status")
def llm_status():
    """Check the LLM configuration."""
    endpoint = config.get("llm", "endpoint", default="")
    model = config.get("llm", "model", default="")
    context_size = config.get("llm", "context_size", default=0)
    max_tokens = config.get("llm", "max_tokens", default=4096)
    system_prompt = config.get("llm", "system_prompt", default="")
    thinking = config.get("llm", "thinking", default=False)

    # detect endpoint type
    ep = endpoint.rstrip("/")
    if ep.endswith("/v1") or "/v1/" in ep or ep.endswith("/chat/completions"):
        api_type = "openai"
    elif ep:
        api_type = "ollama"
    else:
        api_type = ""

    # context: config override or detected from the server (5-min cache).
    context_effective = llm_client.effective_context() if endpoint else 0

    return jsonify(
        {
            "configured": bool(endpoint),
            "endpoint": endpoint,
            "model": model,
            "api_type": api_type,
            "thinking": thinking,
            "context_size": context_size,
            "context_effective": context_effective,
            "max_tokens": max_tokens,
            # The configured system prompt itself — the frontend shows it
            # for "custom".
            "system_prompt": system_prompt,
            "system_prompt_tokens": _estimate_tokens(system_prompt),
            # Built-in correction rules — the settings show them as a
            # template/placeholder for llm.correction.system_prompt.
            "correction_rules_default": _prompts.CORRECTION_RULES,
            "presets": config.get("llm", "presets", default={}),
            # Names of the shipped templates — the dialog lists them apart
            # from the user's own
            "preset_builtin": list(_prompts.STANDARD_PRESETS),
            "models": config.get("llm", "models", default={}),
            # Default models of the active providers — the dialog dropdowns
            # show them in addition to the quick models, otherwise a
            # freshly set provider model can't be selected anywhere.
            "provider_models": _provider_models(),
            # What the dialogs' "default" entry resolves to, per task
            "task_models": _task_models(),
            # Pre-fills the correction dialog; before, the field had a fixed
            # 0.40 and the setting never reached a run started from there.
            "correction_score_threshold": _correction_score_threshold(),
            # Default values for auto-summarization in the transcription dialogs
            # (txoptions.js reads them together with the preset/model lists —
            # ONE source, no race with a second settings fetch).
            "auto_summarize": bool(config.get("llm", "auto_summarize", default=False)),
            "auto_summarize_preset": config.get("llm", "auto_summarize_preset", default=""),
            "auto_summarize_model": config.get("llm", "auto_summarize_model", default=""),
        }
    )


@llm_bp.route("/api/connection/status")
def connection_status():
    """Combined connection status for the GUI indicator.

    Query: ?sample=1  → actively ping the server (slower, real).
           otherwise  → configuration status only (fast, no network).
    Checks LLM AND embedding.
    """
    sample = request.args.get("sample", "").lower() in ("1", "true", "yes")

    def _one(section: str) -> dict:
        # The own embedding server does NOT have its settings in
        # `endpoint`/`model` — those belong to the external endpoint. Anyone
        # who enabled local operation was therefore permanently stuck at
        # "not configured", even though everything was set up and running.
        if section == "embedding" and parse_bool(config.get("embedding", "local", default=False)):
            return _embedding_local(sample)
        endpoint = config.get(section, "endpoint", default="")
        model = config.get(section, "model", default="")
        api_key = config.get(section, "api_key", default="")
        out = {
            "configured": bool(endpoint) and (section != "embedding" or bool(model)),
            "endpoint": endpoint,
            "model": model,
            "api_type": (
                "openai" if llm_client.is_openai_style(endpoint) else ("ollama" if endpoint else "")
            ),
            "reachable": None,  # None = not checked
            "detail": "",
        }
        if sample and endpoint:
            ping = llm_client.ping_endpoint(endpoint, api_key, timeout=6)
            out["reachable"] = ping["reachable"]
            out["detail"] = ping["detail"]
            if ping["models"]:
                out["available_models"] = ping["models"]
        return out

    return jsonify({"llm": _one("llm"), "embedding": _one("embedding")})


def _embedding_local(sample: bool) -> dict:
    """State of the own embedding server for the topbar indicator.

    The server runs **on demand**: after five minutes without a request it
    shuts itself down and frees its VRAM. "Not currently running" is
    therefore the normal case here, not an error — red would be a lie, gray
    ("not configured") even more so. Hence three states:

        file/program missing   → red, stating what's missing
        set up, resting        → green, note "starts on demand"
        set up, running        → green, with uptime

    Only pinged with ``sample=1``. The quick initial call on page load
    merely checks whether the files are present — that costs no network.
    """
    from whisper_tm.core import embed_server as es

    bin_, model = es.binary(), es.model_file()
    is_missing = []
    if not bin_ or not bin_.exists():
        is_missing.append("llama-server")
    if not model:
        is_missing.append("model file (embedding.local_model)")
    elif not model.exists():
        is_missing.append(f"model file {model.name}")

    out = {
        "configured": not is_missing,
        "mode": "local",
        "endpoint": es.endpoint(),
        "model": (config.get("embedding", "model", default="") or (model.name if model else "")),
        "api_type": "openai",
        "reachable": None,
        "detail": "",
    }
    if is_missing:
        # Enabled but incomplete: this IS an error and belongs
        # in red — otherwise the reason gets sought in search later instead of here.
        out["reachable"] = False
        out["detail"] = " and ".join(is_missing) + " missing"
        return out

    if not sample:
        out["ready"] = True
        return out

    z = es.SERVER.state()
    if z["reachable"]:
        out["reachable"] = True
        out["running"] = True
        out["detail"] = (
            f"running for {int(z['running_for_sec'])} s" if z["running_for_sec"] else "running"
        )
    else:
        out["ready"] = True
        out["running"] = False
        limit = int(z.get("idle_limit_sec") or 0)
        out["detail"] = "resting — starts on demand" + (
            f" (idle shutdown after {limit} s)" if limit else ""
        )
    return out


@llm_bp.route("/api/llm/providers")
def read_providers():
    """Configured providers + type catalog + task assignment."""
    from whisper_tm.core import llm_provider as lp

    return jsonify(
        {
            "providers": [p.as_dict() for p in lp.all_items()],
            "types": [
                {"id": t, "name": n, "template": v, "hint": h} for t, (n, v, h) in lp.TYPES.items()
            ],
            "tasks": {a: (config.get("llm", "tasks", a, default={}) or {}) for a in lp.TASKS},
        }
    )


@llm_bp.route("/api/llm/providers", methods=["POST"])
def save_providers():
    """Replace the provider list. Body: {providers: [...], tasks: {...}}"""
    from whisper_tm.core import llm_provider as lp

    data = request.json if isinstance(request.json, dict) else {}
    listing = data.get("providers")
    if not isinstance(listing, list):
        return jsonify({"error": "field 'providers' must be a list"}), 400
    ids = [str(p.get("id") or "") for p in listing if isinstance(p, dict)]
    if len(set(ids)) != len(ids):
        return jsonify({"error": "provider IDs must be unique"}), 400
    lp.save([p for p in listing if isinstance(p, dict)])

    tasks = data.get("tasks")
    if isinstance(tasks, dict):
        clean = {
            a: {
                "provider": str((tasks.get(a) or {}).get("provider") or ""),
                "model": str((tasks.get(a) or {}).get("model") or ""),
            }
            for a in lp.TASKS
            if a in tasks
        }
        config.set("llm", "tasks", clean)
        config.save()
    return jsonify({"ok": True, "providers": [p.as_dict() for p in lp.all_items()]})


@llm_bp.route("/api/llm/providers/test", methods=["POST"])
def test_provider():
    """Ping the endpoint: reachable? which models? which context?"""
    from whisper_tm.core import llm_provider as lp

    data = request.json if isinstance(request.json, dict) else {}
    endpoint = (data.get("endpoint") or "").strip()
    if not endpoint:
        return jsonify({"error": "no endpoint"}), 400
    api_key = data.get("api_key") or ""
    if not api_key and data.get("id"):
        # key not sent along → the stored one, but only for the endpoint it
        # was stored for: never hand it to an address typed in here
        api_key = lp.stored_key(str(data["id"]), endpoint)
    return jsonify(lp.detected_models(endpoint, api_key))


@llm_bp.route("/api/llm/standard_presets")
def default_presets():
    """The shipped presets along with a comparison to the user config.

    An existing user config has its own preset copy — new or
    improved standard presets otherwise never reach it. This route
    delivers the defaults and says, per preset, whether it's missing, differs, or
    is identical; it's only applied via POST /api/llm/presets/merge.
    """
    from whisper_tm.core.llm.prompts import STANDARD_PRESETS

    own_ones = config.get("llm", "presets", default={}) or {}
    entries = []
    for name, text in STANDARD_PRESETS.items():
        present = own_ones.get(name)
        entries.append(
            {
                "name": name,
                "text": text,
                "status": (
                    "new"
                    if present is None
                    else ("same" if str(present).strip() == text.strip() else "different")
                ),
            }
        )
    # presets the user created themselves — left untouched
    own_names = [n for n in own_ones if n not in STANDARD_PRESETS]
    return jsonify({"presets": entries, "custom": own_names})


@llm_bp.route("/api/llm/presets/merge", methods=["POST"])
def presets_merge():
    """Apply standard presets to the user config.

    Body: {"names": [...], "overwrite": bool}
      names         — which standard presets (empty = all missing ones)
      overwrite— also replace differing ones with the same name
    User's own presets are NEVER touched.
    """
    from whisper_tm.core.llm.prompts import STANDARD_PRESETS

    data = request.json if isinstance(request.json, dict) else {}
    names = data.get("names")
    if names is not None and not isinstance(names, list):
        return jsonify({"error": "field 'names' must be a list"}), 400
    overwrite = bool(data.get("overwrite"))

    own_ones = dict(config.get("llm", "presets", default={}) or {})
    target = [n for n in (names or STANDARD_PRESETS) if n in STANDARD_PRESETS]
    added, replaced = [], []
    for name in target:
        if name not in own_ones:
            own_ones[name] = STANDARD_PRESETS[name]
            added.append(name)
        elif overwrite and str(own_ones[name]).strip() != STANDARD_PRESETS[name].strip():
            own_ones[name] = STANDARD_PRESETS[name]
            replaced.append(name)
    if added or replaced:
        config.set("llm", "presets", own_ones)
        config.save()
    return jsonify({"ok": True, "added": added, "replaced": replaced, "total": len(own_ones)})


@llm_bp.route("/api/llm/presets/save", methods=["POST"])
def presets_save():
    """Keep one template (system prompt) under a name — from the dialog.

    Body: {"name": str, "text": str}. An existing template of that name is
    replaced (the dialog asks first); all others stay as they are.
    """
    data = request.json if isinstance(request.json, dict) else {}
    name = str(data.get("name") or "").strip()
    text = str(data.get("text") or "").strip()
    if not name or len(name) > 80:
        return jsonify({"error": "name required (at most 80 characters)"}), 400
    if not text:
        return jsonify({"error": "the template is empty"}), 400
    presets = dict(config.get("llm", "presets", default={}) or {})
    replaced = name in presets
    presets[name] = text
    config.set("llm", "presets", presets)
    try:
        config.save()
    except Exception as e:
        return jsonify({"error": f"not saved: {e}"}), 500
    return jsonify({"ok": True, "name": name, "replaced": replaced, "presets": presets})


@llm_bp.route("/api/llm/estimate", methods=["POST"])
def estimate_tokens():
    """Token estimation for different modes.

    Body: {"segments": [...]}
    Returns: {
        "text_only": 1234,
        "with_speakers": 1456,
        "with_timestamps": 1678,
        "with_both": 1890,
        "system_prompt": 200,
        "context_size": 128000,
        "max_tokens": 4096
    }
    """
    data = request.json or {}
    segments = data.get("segments")

    if not segments:
        filename = data.get("filename", "")
        if filename:
            segments = _load_segments(filename)
        if not segments:
            return jsonify({"error": "no segments"}), 400

    system_prompt = config.get("llm", "system_prompt", default="")
    context_size = config.get("llm", "context_size", default=0)
    max_tokens = config.get("llm", "max_tokens", default=4096)

    # compute all 4 variants
    text_only = _format_transcript(segments, include_speakers=False, include_timestamps=False)
    with_speakers = _format_transcript(segments, include_speakers=True, include_timestamps=False)
    with_timestamps = _format_transcript(segments, include_speakers=False, include_timestamps=True)
    with_both = _format_transcript(segments, include_speakers=True, include_timestamps=True)

    sp_tokens = _estimate_tokens(system_prompt)

    return jsonify(
        {
            "text_only": _estimate_tokens(text_only),
            "with_speakers": _estimate_tokens(with_speakers),
            "with_timestamps": _estimate_tokens(with_timestamps),
            "with_both": _estimate_tokens(with_both),
            "system_prompt": sp_tokens,
            "context_size": context_size,
            # Detected context — the display uses this, not the
            # config value (which is usually 0 = auto).
            "context_effective": llm_client.effective_context(),
            "max_tokens": max_tokens,
        }
    )


@llm_bp.route("/api/summarize", methods=["POST"])
def summarize_transcript():
    """Summarize a transcript with an LLM.

    Body: {
        "filename": "...",
        "prompt": "Summarize...",                (optional user prompt)
        "include_speakers": true/false,
        "include_timestamps": true/false,
        "include_text_only": false,             (text only, no speakers/timestamps)
        "segments": [...]                       (optional, otherwise loaded from DB/JSON)
    }

    Streams the response as SSE (text/event-stream).
    """
    endpoint = config.get("llm", "endpoint", default="")
    if not endpoint:
        return jsonify(
            {"error": "LLM not configured. Please set the endpoint in Settings → AI models."}
        ), 400

    # Model/prompt: empty means "config default" (set by chat_request or
    # the line below) — overrides come from the request right away.
    model = ""
    system_prompt = config.get("llm", "system_prompt", default="")

    # request body must be a JSON object → otherwise a clean 400 instead of 500
    data = request.json if isinstance(request.json, dict) else None
    if data is None:
        return jsonify({"error": "invalid request body: JSON object expected"}), 400

    # overrides from the request (preset change, model change)
    if data.get("model"):
        model = data["model"]
    if data.get("system_prompt") is not None:
        system_prompt = data["system_prompt"]

    user_prompt = (data.get("prompt") or "").strip()
    include_speakers = data.get("include_speakers", False)
    include_timestamps = data.get("include_timestamps", True)
    metadata = data.get("metadata", {})
    segments = data.get("segments")

    # type guards → clean 400 instead of 500 on wrong type
    if metadata is not None and not isinstance(metadata, dict):
        return jsonify({"error": "field 'metadata' must be an object"}), 400
    if segments is not None and not isinstance(segments, list):
        return jsonify({"error": "field 'segments' must be a list"}), 400

    # load segments if not provided
    if not segments:
        filename = data.get("filename", "")
        if not filename:
            return jsonify({"error": "no filename and no segments"}), 400
        segments = _load_segments(filename, named=True)
        if segments is None:
            return jsonify({"error": "transcript not found"}), 404

    if not any((s.get("text") or "").strip() for s in segments if isinstance(s, dict)):
        return jsonify({"error": "transcript is empty"}), 400

    # streaming via the core (core/summarize.summary_stream):
    # single pass OR map-reduce is decided by the core based on the
    # context budget; map progress arrives as a {"status": ...} event.
    def generate():
        try:
            for ev in _summarize_core.summary_stream(
                segments,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                metadata=metadata,
                model=model,
                include_speakers=include_speakers,
                include_timestamps=include_timestamps,
            ):
                yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
        except llm_client.LlmError as e:
            yield f"data: {json.dumps({'error': str(e)[:300]})}\n\n"
        except Exception as e:
            logger.exception("LLM summarize error")
            yield f"data: {json.dumps({'error': str(e)[:300]})}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@llm_bp.route("/api/llm/debug", methods=["POST"])
def debug_input():
    """Debug: show exactly what is sent to the LLM.

    Returns the full messages, payload, and endpoint info.
    """
    system_prompt = config.get("llm", "system_prompt", default="")
    thinking = config.get("llm", "thinking", default=False)
    temperature = config.get("llm", "temperature", default=0.3)
    max_tokens = config.get("llm", "max_tokens", default=4096)

    data = request.json or {}

    # overrides from the request; no model = the task assignment decides
    if data.get("system_prompt") is not None:
        system_prompt = data["system_prompt"]
    model = data.get("model") or ""

    user_prompt = (data.get("prompt") or "").strip()
    include_speakers = data.get("include_speakers", False)
    include_timestamps = data.get("include_timestamps", True)
    metadata = data.get("metadata", {})
    segments = data.get("segments", [])

    transcript_text = _format_transcript(segments, include_speakers, include_timestamps)
    messages = _build_messages(system_prompt, user_prompt, transcript_text, metadata)

    # Endpoint, type and model resolved by the real path (build_chat_payload
    # with the summary task). The legacy llm.endpoint/llm.model shown here
    # before named a server the run doesn't use as soon as the summary task
    # is assigned to another provider — and the dialog sends no model by
    # default, precisely so that this assignment applies.
    try:
        api_url, _headers, payload, use_ollama_native = llm_client.build_chat_payload(
            messages, model=model, task="summary"
        )
        api_type = "ollama" if use_ollama_native else "openai"
        model = payload.get("model", "")
    except llm_client.LlmNotConfigured:
        api_type = "none"
        api_url = ""
        model = model or config.get("llm", "model", default="")

    # token estimates
    total_input = sum(_estimate_tokens(m["content"]) for m in messages)

    # chunk plan: would this request go through in one pass, or map-reduce?
    chunk_plan = {"map_reduce": False, "chunks": 1}
    try:
        user_prompt_text = user_prompt
        needed, context = _summarize_core._needs_map_reduce(
            transcript_text, system_prompt, user_prompt_text
        )
        chunk_plan["context"] = context or 0
        if needed and segments:
            # The same task the real run hands the map step (summary_stream)
            task = "\n\n".join(t for t in ((system_prompt or "").strip(), user_prompt.strip()) if t)
            chunks, _map_max = _summarize_core._map_chunks(
                segments, include_speakers, include_timestamps, task
            )
            chunk_plan = {
                "map_reduce": True,
                "chunks": len(chunks),
                "context": context or 0,
                "chunk_tokens": [c.tokens for c in chunks],
                # what every section is condensed with first
                "map_system": _prompts.map_system(task),
            }
    except Exception as e:
        chunk_plan["error"] = str(e)[:120]

    return jsonify(
        {
            "api_type": api_type,
            "api_url": api_url,
            "model": model,
            "thinking": thinking,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "context_effective": (llm_client.effective_context("summary") if api_url else 0),
            "chunk_plan": chunk_plan,
            "messages": messages,
            "message_count": len(messages),
            "estimated_input_tokens": total_input,
            "transcript_chars": len(transcript_text),
            "transcript_lines": transcript_text.count("\n") + 1,
        }
    )


def _estimate_tokens(text: str) -> int:
    """Estimate token count — delegates to core/llm/budget.py (the ONE
    source of the heuristic; chunker and routes compute identically)."""
    from whisper_tm.core.llm.budget import estimate_tokens

    return estimate_tokens(text)


# ══════════════════════════════════════════════════════════════════════════
#  Save / load / delete summaries
#  DB is the primary source; the JSON file ({transcript_dir}/{stem}_summaries.json)
#  remains as backup/fallback and heals the DB on a miss.
# ══════════════════════════════════════════════════════════════════════════


@llm_bp.route("/api/summaries")
@llm_bp.route("/api/summaries/<path:filename>")
def get_summaries(filename=None):
    """Load stored summaries for a file — the path also as ``?file=``
    (a full one, within the allowed roots). ``pending``: the queue still
    writes one — the transcript is done, the dialog's summary follows."""
    from whisper_tm.api.transcription import _file_param
    from whisper_tm.core import queue as _q
    from whisper_tm.extensions import db

    name = filename or _file_param()
    pending = False
    try:
        target = db._identity(name, None) if db else ""
        pending = _q.summary_pending(target)
    except Exception as e:  # noqa — the summaries come first
        logger.debug("Summary pending check: %s", e)
    return jsonify({"summaries": _load_summaries(name), "pending": pending})


@llm_bp.route("/api/summaries/save", methods=["POST"])
def save_summary():
    """Save a summary."""
    data = request.json or {}
    filename = data.get("filename", "")
    text = data.get("text", "")
    prompt = data.get("prompt", "")

    if not filename or not text:
        return jsonify({"error": "filename and text required"}), 400

    import time

    entry = {
        "text": text,
        "prompt": prompt,
        "timestamp": time.time(),
        # No model = the dialog's default: the summary assignment chose it
        "model": (str(data.get("model") or "").strip() or llm_client.task_default_model("summary")),
    }
    # One writer at a time, never from a failed read (change_summaries)
    try:
        saved, count = _summarize_core.change_summaries(
            filename, lambda lst: (lst.append(entry), len(lst))[1]
        )
    except Exception as e:
        logger.warning("Summary not saved (%s): %s", filename, e)
        return jsonify({"ok": False, "error": f"save failed: {e}"}), 500
    if not saved:
        return jsonify({"ok": False, "error": "save failed (DB and JSON not writable)"}), 500
    return jsonify({"ok": True, "count": count})


@llm_bp.route("/api/summaries/delete", methods=["POST"])
def delete_summary():
    """Delete a single summary."""
    data = request.json or {}
    filename = data.get("filename", "")
    index = data.get("index")

    if not filename or index is None:
        return jsonify({"error": "filename and index required"}), 400

    try:
        index = int(index)
    except (TypeError, ValueError):
        return jsonify({"error": "invalid index"}), 400

    stamp = data.get("timestamp")

    class _Refused(Exception):
        def __init__(self, code: int, message: str):
            super().__init__(message)
            self.code = code

    def _remove(summaries: list) -> int:
        if not (0 <= index < len(summaries)):
            raise _Refused(404, "index out of valid range")
        # The list may have changed since it was shown (another tab deleted
        # one): an index alone would then delete a different summary
        if stamp is not None:
            try:
                same = abs(float(summaries[index].get("timestamp") or 0) - float(stamp)) < 0.001
            except (TypeError, ValueError):
                same = False
            if not same:
                raise _Refused(409, "the list has changed — reload it and delete again")
        summaries.pop(index)
        return len(summaries)

    # One writer at a time, never from a failed read (change_summaries)
    try:
        saved, count = _summarize_core.change_summaries(filename, _remove)
    except _Refused as e:
        return jsonify({"ok": False, "error": str(e)}), e.code
    except Exception as e:
        logger.warning("Summary not deleted (%s): %s", filename, e)
        return jsonify({"ok": False, "error": f"delete failed: {e}"}), 500
    if not saved:
        return jsonify({"ok": False, "error": "save failed (DB and JSON not writable)"}), 500
    return jsonify({"ok": True, "count": count})
