"""
Unified LLM/embedding client: URL resolution, connection check, embeddings,
chat calls (blocking + stream + JSON-enforced) and context auto-detection.

Supports three server types via ONE endpoint field:
  - OpenAI-compatible (endpoint contains "/v1"):  OpenAI, llama.cpp (llama-server),
    vLLM, LiteLLM, LM Studio, Ollama-in-/v1-mode …
  - Ollama native (endpoint without "/v1"):       http://host:11434
  - llama.cpp: runs as OpenAI-compatible under /v1 → nothing special needed,
    but /health is tried as fallback for connection check.

Since the pipeline refactoring (docs/LLM-PIPELINE.md), this is the
ONE place for chat payloads and calls: core/summarize.py, core/llm/* and
api/llm.py no longer build their own requests.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time

import requests

from whisper_tm.core import local_key

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════
#  Error classes — callers distinguish "misconfigured" (no retry,
#  user must act) from "currently unreachable" (retry makes sense).
# ══════════════════════════════════════════════════════════════════════════


class LlmError(RuntimeError):
    """Base class for all LLM errors."""


class LlmNotConfigured(LlmError):
    """No endpoint set — Settings → AI models."""


class LlmConnectionError(LlmError):
    """Network/Timeout/5xx — retry makes sense."""


class LlmAnswerError(LlmError):
    """4xx or unusable response — retry won't help."""


class LlmAnswerTruncated(LlmAnswerError):
    """Response truncated at max_tokens (finish_reason=length).

    Repeating the exact same call would get truncated identically —
    the caller must reduce the task (halve the chunk)."""


def is_openai_style(endpoint: str) -> bool:
    """True if the endpoint is OpenAI-compatible (contains /v1)."""
    ep = (endpoint or "").rstrip("/")
    return ep.endswith("/v1") or "/v1/" in ep or ep.endswith("/chat/completions")


def resolve_chat_url(endpoint: str) -> tuple[str, bool]:
    """Chat URL + flag for Ollama native.

    Returns:
        (url, use_ollama_native)
    """
    api_url = (endpoint or "").rstrip("/")
    if api_url.endswith("/v1"):
        return api_url + "/chat/completions", False
    if "/v1/" in api_url or api_url.endswith("/chat/completions"):
        if not api_url.endswith("/chat/completions"):
            api_url += "/chat/completions"
        return api_url, False
    # No /v1 → Ollama native
    return api_url + "/api/chat", True


def resolve_embeddings_url(endpoint: str) -> tuple[str, bool]:
    """Embeddings URL + flag for Ollama native.

    OpenAI-compatible:  {base}/v1/embeddings   (llama.cpp, OpenAI, vLLM …)
    Ollama native:      {base}/api/embeddings
    """
    api_url = (endpoint or "").rstrip("/")
    if api_url.endswith("/embeddings"):
        return api_url, False
    if api_url.endswith("/v1"):
        return api_url + "/embeddings", False
    if "/v1/" in api_url:
        return api_url.split("/v1/")[0] + "/v1/embeddings", False
    if api_url.endswith("/chat/completions"):
        return api_url.replace("/chat/completions", "/embeddings"), False
    # No /v1 → Ollama native
    return api_url + "/api/embeddings", True


def ping_endpoint(endpoint: str, api_key: str = "", timeout: int = 6) -> dict:
    """Light connection check WITHOUT token generation.

    Queries model list (OpenAI: GET /v1/models, Ollama: GET /api/tags).
    For llama.cpp, /health is tried as fallback.

    Returns:
        {reachable: bool, api_type: "openai"|"ollama"|"", models: [str],
         detail: str}
    """
    endpoint = (endpoint or "").strip()
    if not endpoint:
        return {"reachable": False, "api_type": "", "models": [], "detail": "no endpoint"}

    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    ep = endpoint.rstrip("/")
    openai_style = is_openai_style(endpoint)

    # Candidate URLs by type
    candidates: list[tuple[str, str]] = []
    if openai_style:
        base = ep
        if base.endswith("/chat/completions"):
            base = base[: -len("/chat/completions")]
        if "/v1/" in base:
            base = base.split("/v1/")[0] + "/v1"
        candidates.append(("openai", base.rstrip("/") + "/models"))
        # llama.cpp health as additional fallback
        root = base.rsplit("/v1", 1)[0]
        candidates.append(("openai", root.rstrip("/") + "/health"))
    else:
        candidates.append(("ollama", ep + "/api/tags"))
        # Some servers speak /v1 anyway
        candidates.append(("openai", ep + "/v1/models"))

    last_err = ""
    for api_type, url in candidates:
        try:
            r = requests.get(url, headers=local_key.headers_for(url, headers), timeout=timeout)
            if r.status_code == 200:
                models = _extract_model_names(r)
                return {"reachable": True, "api_type": api_type, "models": models, "detail": "ok"}
            # 401/403 means: reachable, but auth missing/wrong
            if r.status_code in (401, 403):
                return {
                    "reachable": True,
                    "api_type": api_type,
                    "models": [],
                    "detail": f"reachable, auth error ({r.status_code})",
                }
            last_err = f"HTTP {r.status_code}"
        except requests.exceptions.Timeout:
            last_err = "Timeout"
        except requests.exceptions.ConnectionError:
            last_err = "connection refused / host not found"
        except Exception as e:  # noqa
            last_err = str(e)[:80]

    return {
        "reachable": False,
        "api_type": "openai" if openai_style else "ollama",
        "models": [],
        "detail": last_err or "unreachable",
    }


def _extract_model_names(resp) -> list[str]:
    """Extract model names from /v1/models or /api/tags response."""
    try:
        data = resp.json()
    except Exception:
        return []
    names: list[str] = []
    # OpenAI: {"data": [{"id": "..."}]}
    for m in data.get("data") or []:
        if isinstance(m, dict) and m.get("id"):
            names.append(m["id"])
    # Ollama: {"models": [{"name": "..."}]}
    for m in data.get("models") or []:
        if isinstance(m, dict) and m.get("name"):
            names.append(m["name"])
    return names[:100]


def embed_texts(
    texts: list[str], *, endpoint: str, model: str, api_key: str = "", timeout: int = 120
) -> list[list[float]]:
    """Embed texts (embeddings). Supports OpenAI and Ollama format.

    Returns:
        List of vectors (same order as texts).
    Raises:
        RuntimeError on API errors.
    """
    if not texts:
        return []
    url, ollama = resolve_embeddings_url(endpoint)
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    if ollama:
        # Ollama native: one request per text (/api/embeddings, "prompt")
        out: list[list[float]] = []
        for t in texts:
            r = requests.post(
                url,
                headers=local_key.headers_for(url, headers),
                json={"model": model, "prompt": t},
                timeout=timeout,
            )
            if r.status_code != 200:
                raise RuntimeError(f"Embedding-API {r.status_code}: {r.text[:200]}")
            vec = r.json().get("embedding")
            if not vec:
                raise RuntimeError("Embedding response missing 'embedding'")
            out.append(vec)
        return out

    # OpenAI-compatible (llama.cpp, OpenAI, vLLM …): batch in one request
    r = requests.post(
        url,
        headers=local_key.headers_for(url, headers),
        json={"model": model, "input": texts},
        timeout=timeout,
    )
    if r.status_code != 200:
        raise RuntimeError(f"Embedding-API {r.status_code}: {r.text[:200]}")
    data = r.json().get("data")
    if not data:
        raise RuntimeError("Embedding response missing 'data'")
    # sort by index (OpenAI doesn't guarantee order)
    data = sorted(data, key=lambda d: d.get("index", 0))
    return [d["embedding"] for d in data]


# ══════════════════════════════════════════════════════════════════════════
#  Chat payload — the ONE place for both endpoint dialects
# ══════════════════════════════════════════════════════════════════════════


def _task_provider(task: str) -> tuple[str, str, str]:
    """(endpoint, api_key, model) for a task from the assignment.

    Empty strings if no assignment applies — the caller then falls back
    to legacy fields. The assignment (``llm.tasks``) is maintained in
    settings and read here by every execution path.
    """
    if not task:
        return "", "", ""
    try:
        from whisper_tm.core import llm_provider

        p, model = llm_provider.for_task(task)
        if p and p.endpoint:
            return p.endpoint, p.api_key, model
    except Exception as e:
        logger.debug("Task provider %s: %s", task, e)
    return "", "", ""


def task_default_model(task: str) -> str:
    """The model a chat call for ``task`` runs on when the caller names none.

    Resolved the way :func:`build_chat_payload` does it: the task's
    assignment, else the legacy ``llm.model``. The dialogs show it in their
    "default" entry, whose value stays empty so that exactly this
    resolution applies.
    """
    from whisper_tm.config import config

    _endpoint, _api_key, model = _task_provider(task)
    return model or config.get("llm", "model", default="")


def ready(task: str = "") -> bool:
    """Is a language model set up for ``task``? Whatever else may be wrong
    with it (a server that's down) counts as set up — the call says that."""
    try:
        build_chat_payload([{"role": "user", "content": "-"}], task=task)
        return True
    except LlmNotConfigured:
        return False
    except Exception as e:  # noqa: BLE001
        logger.debug("LLM check (%s): %s", task or "default", e)
        return True


def build_chat_payload(
    messages: list,
    *,
    model: str = "",
    stream: bool = False,
    temperature: float | None = None,
    max_tokens: int | None = None,
    response_schema: dict | None = None,
    schema_name: str = "answer",
    thinking: bool | None = None,
    task: str = "",
):
    """Build endpoint, headers and payload for a chat call.

    All ``None`` parameters fall back to config. ``response_schema``
    enforces structured output: OpenAI-compatible via
    ``response_format: json_schema`` (llama.cpp then samples grammar-
    constrained — verified live), Ollama native via ``format``.

    ``task`` ("summary"/"correction"/"translation") pulls the
    provider assignment from settings; without assignment (or without
    ``task``) the legacy fields ``llm.endpoint``/``llm.model`` apply.

    Returns:
        (api_url, headers, payload, ollama_native)
    Raises:
        LlmNotConfigured if no endpoint is set.
    """
    from whisper_tm.config import config

    endpoint, api_key, task_model = _task_provider(task)
    if not model:
        model = task_model
    elif model != task_model:
        # Explicitly chosen model pulls ITS provider along — otherwise
        # "gpt-4.1-nano" would go to the task provider (llama.cpp),
        # which silently ignores foreign model names. That is also why a
        # dialog's "default" entry must send '' and never a model name:
        # any name here overrides the task assignment.
        try:
            from whisper_tm.core import llm_provider

            p = llm_provider.for_model(model)
            if p and p.endpoint:
                endpoint, api_key = p.endpoint, p.api_key
        except Exception as e:
            logger.debug("Model provider %s: %s", model, e)
    if not endpoint:
        endpoint = config.get("llm", "endpoint", default="")
        api_key = config.get("llm", "api_key", default="")
    if not endpoint:
        raise LlmNotConfigured("LLM not configured (Settings → AI models)")
    if max_tokens is None:
        max_tokens = config.get("llm", "max_tokens", default=4096)
    if temperature is None:
        temperature = config.get("llm", "temperature", default=0.3)
    if thinking is None:
        thinking = config.get("llm", "thinking", default=False)
    thinking_budget = config.get("llm", "thinking_budget", default=0)
    model = model or config.get("llm", "model", default="")

    # max_tokens clamp: response limit above context is always a
    # config error (e.g. 256000 with a 32k context) —
    # llama.cpp would push the prompt out of the window. If context is
    # NOT detectable (OpenAI has no /props), a hard limit applies —
    # otherwise 256000 would go unrestricted and OpenAI rejects every call
    # with 400.
    ctx = effective_context(task)
    limit = (ctx // 2) if ctx else 16384
    if max_tokens and max_tokens > limit:
        logger.warning("llm.max_tokens=%s > limit %s — clamped", max_tokens, limit)
        max_tokens = limit

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    api_url, ollama_native = resolve_chat_url(endpoint)

    if ollama_native:
        payload = {"model": model, "messages": messages, "stream": stream, "think": bool(thinking)}
        opts = {}
        if max_tokens and max_tokens > 0:
            opts["num_predict"] = max_tokens
        if not thinking:
            opts["temperature"] = temperature
        context_size = config.get("llm", "context_size", default=0)
        if context_size and context_size > 0:
            opts["num_ctx"] = context_size
        if opts:
            payload["options"] = opts
        if response_schema:
            payload["format"] = response_schema  # Ollama ≥ 0.5
    else:
        payload = {"model": model, "messages": messages, "stream": stream}
        # OpenAI reasoning family (gpt-5*, o1/o3/o4-*): requires
        # max_completion_tokens instead of max_tokens and accepts only
        # default temperature — otherwise both give hard 400s.
        reasoning_model = bool(re.match(r"(gpt-5|o\d)([.-]|$)", model))
        if reasoning_model:
            payload["max_completion_tokens"] = max_tokens
        else:
            payload["max_tokens"] = max_tokens
        if thinking:
            payload["thinking"] = (
                {"type": "enabled", "budget_tokens": thinking_budget}
                if thinking_budget and thinking_budget > 0
                else {"type": "enabled"}
            )
        elif not reasoning_model:
            payload["temperature"] = temperature
        if response_schema:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": schema_name, "strict": True, "schema": response_schema},
            }

    return api_url, headers, payload, ollama_native


_llamacpp: dict[str, bool] = {}
_llamacpp_lock = threading.Lock()


def is_llamacpp(api_url: str) -> bool:
    """Is the endpoint a llama.cpp server? Only it understands
    ``chat_template_kwargs`` (thinking on/off) — OpenAI would answer 400.
    Asked once per server (``/props``), then remembered."""
    base = api_url.split("/v1/")[0].rstrip("/")
    with _llamacpp_lock:
        if base in _llamacpp:
            return _llamacpp[base]
    try:
        r = requests.get(base + "/props", headers=local_key.headers_for(base + "/props"), timeout=4)
        found = r.ok and "chat_template_caps" in r.text
    except Exception:  # noqa: BLE001 — not answering means: not one
        found = False
    with _llamacpp_lock:
        _llamacpp[base] = found
    return found


def answer_text(data: dict, ollama_native: bool) -> str:
    """Extract text from a NON-streamed chat response."""
    if ollama_native:
        return (data.get("message") or {}).get("content", "") or ""
    try:
        return data["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError):
        return ""


# ══════════════════════════════════════════════════════════════════════════
#  Chat calls: blocking, stream, JSON-enforced — with retry/backoff
# ══════════════════════════════════════════════════════════════════════════


def _retry_loop(func, *, retries: int, description: str):
    """Repeat ``func()`` with backoff — only on connection errors.

    LlmResponseError (4xx, broken JSON) is passed through immediately:
    repeating the same request won't change that.
    """
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return func()
        except LlmConnectionError as e:
            last = e
            if attempt < retries:
                wait_time = min(30, 2**attempt * 2)
                logger.warning(
                    "%s: %s — retry %d/%d in %ds", description, e, attempt + 1, retries, wait_time
                )
                time.sleep(wait_time)
    raise last  # type: ignore[misc]


def _post_chat(api_url, headers, payload, timeout):
    """One POST, error translated to correct class."""
    try:
        r = requests.post(
            api_url, headers=local_key.headers_for(api_url, headers), json=payload, timeout=timeout
        )
    except requests.exceptions.Timeout as e:
        limit = timeout[1] if isinstance(timeout, tuple) else timeout
        raise LlmConnectionError(f"Timeout after {limit}s") from e
    except requests.exceptions.ConnectionError as e:
        raise LlmConnectionError(f"Connection failed: {str(e)[:120]}") from e
    if r.status_code >= 500:
        raise LlmConnectionError(f"LLM server {r.status_code}: {r.text[:200]}")
    if r.status_code != 200:
        raise LlmAnswerError(f"LLM error {r.status_code}: {r.text[:300]}")
    return r


def chat(
    messages: list,
    *,
    model: str = "",
    temperature: float | None = None,
    max_tokens: int | None = None,
    response_schema: dict | None = None,
    schema_name: str = "answer",
    thinking: bool | None = None,
    timeout: int | None = None,
    retries: int | None = None,
    task: str = "",
    template_kwargs: dict | None = None,
    logprobs: int = 0,
):
    """Blocking chat call. Returns ``(text, usage)``.

    ``usage`` is ``{"prompt_tokens": int, "completion_tokens": int}`` if
    the server reports it (llama.cpp and OpenAI do), otherwise empty dict.

    ``template_kwargs``: ``chat_template_kwargs`` for a llama.cpp server —
    ``{"enable_thinking": False}`` keeps a thinking model from spending a
    five-token answer on its thoughts. Left out for any other server.

    ``logprobs``: ask an OpenAI-compatible server (llama.cpp too) for the
    this many most likely tokens at each place; ``usage["first_logprobs"]``
    then holds ``[(token, logprob), …]`` of the answer's first token — a
    digit's probabilities make a verdict continuous (find/judge.py).
    """
    from whisper_tm.config import config

    if timeout is None:
        timeout = config.get("llm", "timeout", default=1800)
    if retries is None:
        retries = config.get("llm", "retries", default=3)

    api_url, headers, payload, ollama_native = build_chat_payload(
        messages,
        model=model,
        stream=False,
        temperature=temperature,
        max_tokens=max_tokens,
        response_schema=response_schema,
        schema_name=schema_name,
        thinking=thinking,
        task=task,
    )
    if template_kwargs and not ollama_native and is_llamacpp(api_url):
        payload["chat_template_kwargs"] = dict(template_kwargs)
    if logprobs and not ollama_native:
        payload["logprobs"] = True
        payload["top_logprobs"] = int(logprobs)

    def _once():
        t0 = time.monotonic()
        r = _post_chat(api_url, headers, payload, (10, timeout))
        duration = time.monotonic() - t0
        try:
            data = r.json()
        except ValueError as e:
            raise LlmAnswerError("Response is not JSON") from e
        text = answer_text(data, ollama_native)
        if not text.strip():
            raise LlmAnswerError("Empty response from LLM")
        usage = {}
        u = data.get("usage") or {}
        if u:
            usage = {
                "prompt_tokens": u.get("prompt_tokens", 0),
                "completion_tokens": u.get("completion_tokens", 0),
            }
        elif ollama_native and data.get("eval_count") is not None:
            usage = {
                "prompt_tokens": data.get("prompt_eval_count", 0),
                "completion_tokens": data.get("eval_count", 0),
            }
        # Include finish_reason: "length" means truncated response — fatal for
        # JSON tasks (truncated JSON would trigger pointless repair retries
        # and timeout cascades).
        if ollama_native:
            usage["finish_reason"] = data.get("done_reason", "")
        else:
            try:
                usage["finish_reason"] = data["choices"][0].get("finish_reason", "") or ""
            except (KeyError, IndexError, TypeError):
                usage["finish_reason"] = ""
        usage["duration"] = round(duration, 2)
        # The model that answered: a llama.cpp server runs the one it
        # loaded whatever name was asked for — an analysis on "gpt-4o-mini"
        # could run on a local model and still report gpt-4o-mini
        if isinstance(data.get("model"), str) and data["model"]:
            usage["model"] = data["model"]
        if logprobs and not ollama_native:
            try:
                first = data["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
                usage["first_logprobs"] = [(str(t["token"]), float(t["logprob"])) for t in first]
            except (KeyError, IndexError, TypeError, ValueError):
                pass
        # llama.cpp says how fast it read the prompt and wrote the answer —
        # an analysis estimates its next run from it (core/analysis.py)
        timings = data.get("timings") if isinstance(data.get("timings"), dict) else {}
        if timings.get("prompt_per_second"):
            usage["prompt_tps"] = round(float(timings["prompt_per_second"]), 1)
        if timings.get("predicted_per_second"):
            usage["answer_tps"] = round(float(timings["predicted_per_second"]), 1)
        # ⏱ Timing per call — same convention as transcription
        # pipeline; lands in debug log via rag category.
        ct = usage.get("completion_tokens") or 0
        logger.info(
            "⏱ LLM-Chat %s: %.1fs · %s→%s tokens · %.1f tok/s%s",
            payload.get("model", "?"),
            duration,
            usage.get("prompt_tokens", "?"),
            ct or "?",
            (ct / duration) if ct and duration > 0 else 0.0,
            " · TRUNCATED" if usage.get("finish_reason") in ("length", "limit") else "",
        )
        return text, usage

    return _retry_loop(_once, retries=retries, description="LLM-Chat")


def chat_stream(
    messages: list,
    *,
    model: str = "",
    temperature: float | None = None,
    max_tokens: int | None = None,
    thinking: bool | None = None,
    timeout: int | None = None,
    task: str = "",
    info: dict | None = None,
):
    """Streaming chat call: generator over text chunks.

    Encapsulates BOTH stream dialects (Ollama: JSON per line, OpenAI: SSE).
    No retry — a broken stream can't be resumed seamlessly;
    the caller decides whether to restart.
    The connection is closed even on GeneratorExit (client gone).

    ``info``, when given, learns how the stream ended: ``finish_reason``
    ("length" = cut at the token limit) and, from llama.cpp, ``timings``.
    """
    from whisper_tm.config import config

    if timeout is None:
        timeout = config.get("llm", "timeout", default=1800)

    api_url, headers, payload, ollama_native = build_chat_payload(
        messages,
        model=model,
        stream=True,
        temperature=temperature,
        max_tokens=max_tokens,
        thinking=thinking,
        task=task,
    )

    resp = None
    t0 = time.monotonic()
    chars = 0
    try:
        try:
            resp = requests.post(
                api_url,
                headers=local_key.headers_for(api_url, headers),
                json=payload,
                stream=True,
                timeout=(10, timeout),
            )
        except requests.exceptions.Timeout as e:
            raise LlmConnectionError("Timeout on connection establishment") from e
        except requests.exceptions.ConnectionError as e:
            raise LlmConnectionError(f"Connection failed: {str(e)[:120]}") from e
        if resp.status_code >= 500:
            raise LlmConnectionError(f"LLM server {resp.status_code}: {resp.text[:200]}")
        if resp.status_code != 200:
            raise LlmAnswerError(f"LLM error {resp.status_code}: {resp.text[:300]}")

        if ollama_native:
            for line in resp.iter_lines():
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue
                content = (chunk.get("message") or {}).get("content", "")
                if content:
                    chars += len(content)
                    yield content
                if chunk.get("done"):
                    if info is not None:
                        info["finish_reason"] = chunk.get("done_reason", "") or ""
                    return
        else:
            for line in resp.iter_lines():
                if not line:
                    continue
                line = line.decode("utf-8", errors="replace")
                if not line.startswith("data: "):
                    continue
                payload_str = line[6:]
                if payload_str.strip() == "[DONE]":
                    return
                try:
                    chunk = json.loads(payload_str)
                    choice = (chunk.get("choices") or [{}])[0]
                    delta = choice.get("delta") or {}
                except (json.JSONDecodeError, IndexError, KeyError, AttributeError):
                    continue
                if info is not None:
                    if choice.get("finish_reason"):
                        info["finish_reason"] = choice["finish_reason"]
                    if isinstance(chunk.get("timings"), dict):
                        info["timings"] = chunk["timings"]
                    if isinstance(chunk.get("model"), str) and chunk["model"]:
                        info["model"] = chunk["model"]
                content = delta.get("content", "")
                if content:
                    chars += len(content)
                    yield content
    finally:
        duration = time.monotonic() - t0
        # ~4 characters/token as approximation — streams don't report usage.
        logger.info(
            "⏱ LLM-Stream %s: %.1fs · %d characters (~%d tokens · %.1f tok/s)",
            payload.get("model", "?"),
            duration,
            chars,
            chars // 4,
            (chars / 4 / duration) if duration > 0 else 0.0,
        )
        if resp is not None:
            try:
                resp.close()
            except Exception:
                pass


# ══════════════════════════════════════════════════════════════════════════
#  JSON-enforced responses (correction/translation)
# ══════════════════════════════════════════════════════════════════════════


def extract_json(text: str):
    """Extract JSON object from LLM response — tolerant.

    Order: direct parse → ```json fence → first balanced
    {...} bracket. Returns None if nothing parsable.
    """
    text = (text or "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"```(?:json)?\s*\n(.*?)```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # First balanced { … } — braces in strings are ignored via a
    # simple quote state machine.
    start = text.find("{")
    if start < 0:
        return None
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def chat_json(
    messages: list,
    *,
    schema: dict,
    schema_name: str = "answer",
    model: str = "",
    temperature: float | None = None,
    max_tokens: int | None = None,
    timeout: int | None = None,
    retries: int | None = None,
    task: str = "",
) -> dict:
    """Chat call with enforced JSON response.

    First with ``response_format``/``format`` (server enforces grammar).
    If server rejects it (400), once WITHOUT schema with strict
    prompt instruction. Parses response tolerantly; on parse error ONE
    repair retry where broken response is mirrored back with error.
    Thinking is always off — deterministic data transformation.
    """

    def _truncated_check(usage):
        if (usage or {}).get("finish_reason") in ("length", "limit"):
            raise LlmAnswerTruncated(
                "Answer cut off at max_tokens — make the request smaller "
                f"(completion_tokens={usage.get('completion_tokens')})"
            )

    try:
        text, _usage = chat(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            response_schema=schema,
            schema_name=schema_name,
            thinking=False,
            timeout=timeout,
            retries=retries,
            task=task,
        )
        _truncated_check(_usage)
    except LlmAnswerTruncated:
        raise
    except LlmAnswerError as e:
        # Fallback ONLY if the 400 really concerns response_format
        # — other 400s (e.g. exceed_context_size) would fail the
        # schema-less retry exactly the same way.
        message = str(e).lower()
        if "400" not in message or not any(
            s in message for s in ("response_format", "json_schema", "grammar", "format")
        ):
            raise
        logger.info("response_format rejected — fallback to prompt JSON")
        hint = (
            "\n\nRespond EXCLUSIVELY with a JSON object per "
            "this schema, no explanation text:\n" + json.dumps(schema, ensure_ascii=False)
        )
        msgs = [dict(m) for m in messages]
        msgs[-1] = dict(msgs[-1])
        msgs[-1]["content"] = msgs[-1]["content"] + hint
        text, _usage = chat(
            msgs,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            thinking=False,
            timeout=timeout,
            retries=retries,
            task=task,
        )
        _truncated_check(_usage)

    data = extract_json(text)
    if data is None:
        # Repair retry: mirror broken response back
        msgs = list(messages) + [
            {"role": "assistant", "content": text[:4000]},
            {
                "role": "user",
                "content": "Your response was not valid JSON. Answer again, "
                "EXCLUSIVELY with the JSON object — no explanations, "
                "no markdown fences.",
            },
        ]
        text, _usage = chat(
            msgs,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            response_schema=schema,
            schema_name=schema_name,
            thinking=False,
            timeout=timeout,
            retries=retries,
            task=task,
        )
        _truncated_check(_usage)
        data = extract_json(text)
    if data is None:
        raise LlmAnswerError(f"LLM did not provide parsable JSON (start: {text[:120]!r})")
    if not isinstance(data, dict):
        raise LlmAnswerError(f"JSON response is not an object: {type(data).__name__}")
    return data


# ══════════════════════════════════════════════════════════════════════════
#  Detect context size from server (llama.cpp /props, /v1/models, Ollama)
# ══════════════════════════════════════════════════════════════════════════

_ctx_cache: dict[str, tuple[float, int]] = {}
_ctx_lock = threading.Lock()
_CTX_TTL = 300  # seconds


def detect_context(endpoint: str, api_key: str = "", model: str = "", timeout: int = 6) -> int:
    """Determine actual loaded context size of server (0 = unknown).

    llama.cpp: GET /props → default_generation_settings.n_ctx (the ACTUALLY
    loaded size, not n_ctx_train). Fallback /v1/models → meta.n_ctx.
    Ollama: POST /api/show → model_info["…context_length"] (training
    context; the real num_ctx used is set by config).
    Result is cached for 5 min (per endpoint+model).
    """
    endpoint = (endpoint or "").rstrip("/")
    if not endpoint:
        return 0
    key = f"{endpoint}|{model}"
    with _ctx_lock:
        entry = _ctx_cache.get(key)
        if entry and time.time() - entry[0] < _CTX_TTL:
            return entry[1]

    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    ctx = 0

    if is_openai_style(endpoint):
        root = endpoint.rsplit("/v1", 1)[0]
        # llama.cpp: /props has the loaded n_ctx
        try:
            r = requests.get(
                root + "/props", headers=local_key.headers_for(root, headers), timeout=timeout
            )
            if r.status_code == 200:
                ctx = int((r.json().get("default_generation_settings") or {}).get("n_ctx") or 0)
        except Exception:
            pass
        if not ctx:
            try:
                r = requests.get(endpoint + "/models", headers=headers, timeout=timeout)
                if r.status_code == 200:
                    for m in r.json().get("data") or []:
                        meta = m.get("meta") or {}
                        if meta.get("n_ctx"):
                            ctx = int(meta["n_ctx"])
                            break
            except Exception:
                pass
    else:
        # Ollama: /api/show provides model_info with *.context_length
        try:
            r = requests.post(
                endpoint + "/api/show", headers=headers, json={"model": model}, timeout=timeout
            )
            if r.status_code == 200:
                info = r.json().get("model_info") or {}
                for k, v in info.items():
                    if k.endswith(".context_length") and v:
                        ctx = int(v)
                        break
        except Exception:
            pass

    with _ctx_lock:
        _ctx_cache[key] = (time.time(), ctx)
    return ctx


def effective_context(task: str = "") -> int:
    """Context size for budget calculations: config override or detected.

    ``llm.context_size > 0`` wins (deliberate user override, on Ollama
    also num_ctx). Otherwise the size detected by server. 0 = unknown —
    callers then take a conservative default (8192).

    With ``task``, the context of the assigned provider counts —
    otherwise e.g. an OpenAI translation calculates with the 32k window
    of the local server (or worse: vice versa).
    """
    from whisper_tm.config import config

    override = config.get("llm", "context_size", default=0)
    if override and override > 0:
        return int(override)
    endpoint, api_key, model = _task_provider(task)
    if not endpoint:
        endpoint = config.get("llm", "endpoint", default="")
        model = config.get("llm", "model", default="")
        api_key = config.get("llm", "api_key", default="")
    return detect_context(endpoint, api_key, model)
