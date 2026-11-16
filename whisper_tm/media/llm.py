"""What the language model adds to "Images & sounds", with measurements.

**Sounds are asked in English.** EmbeddingGemma 2 learned sounds from
English captions: on a labeled test set of 16 sound kinds (windows of
10 s) queries in a non-English language found them with a mean AP of
0.115, the same queries as short English labels 0.31. The language model
writes that label — a query like "lots of people chatting in a crowded pub"
becomes "busy pub crowd" — in ~0.5 s, and reached 0.325 (asked for a long
caption instead: 0.21; both vectors added: 0.24). So sounds are searched
with the label alone; pictures keep the query as typed (other languages
worked as well as English there).

**The check.** A vision model looks at each of the best stills and says how
well it matches, 0–9 (the expected digit over the answer's token
probabilities, where the server reports them). On 12 queries × the best 20
stills, judged by eye: precision of the top 10 0.62 → 0.82, clearly
matching ones 0.40 → 0.55, nDCG@10 0.67 → 0.96; what the model called a
match (≥ 6) was right 93 % of the time. ~3.8 s per picture with a
Qwen3.6-35B-A3B vision model — one picture per request.

**Descriptions.** The same model describes a moment in one sentence, on
request (~5 s), kept in the index.
"""

from __future__ import annotations

import base64
import logging
import math
import threading
import time
import uuid
from collections import OrderedDict
from pathlib import Path

logger = logging.getLogger(__name__)

LABEL_SYSTEM = (
    "Translate a search query for SOUNDS into a SHORT English sound label of 1–5 words, the way "
    "sound libraries name their files (e.g. 'gunshots', 'dog barking', 'busy pub crowd', 'rock "
    "music'). Answer with the label only."
)
JUDGE_SYSTEM = (
    "You judge whether a still from a video matches a search query. Answer with ONE digit from 0 "
    "to 9: 0 = does not match at all, 5 = partly or unclear, 9 = clearly matches."
)
DESCRIBE_SYSTEM = (
    "Describe a still from a video for a search index in ONE sentence of {lang}: who or what is "
    "visible, where it is (inside/outside, day/night, real footage, game, cartoon, screen "
    "recording), what happens, and any large text on screen. No speculation."
)

_labels: OrderedDict[str, str] = OrderedDict()
_labels_lock = threading.Lock()
_vision: dict = {"at": 0.0, "key": None, "value": None}


def _cfg(name: str, default=None):
    from whisper_tm.config import config

    return config.get("media_search", name, default=default)


def _model() -> str:
    return (_cfg("llm_model", "") or "").strip()


def configured() -> bool:
    from whisper_tm.config import config

    return bool(config.get("llm", "endpoint", default="")) or bool(_model())


def _payload(messages: list, *, max_tokens: int):
    from whisper_tm.core import llm_client

    api_url, headers, payload, ollama_native = llm_client.build_chat_payload(
        messages, model=_model(), temperature=0.0, max_tokens=max_tokens, thinking=False
    )
    llama = not ollama_native and llm_client.is_llamacpp(api_url)
    if llama:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    return api_url, headers, payload, ollama_native, llama


def _post(api_url, headers, payload, timeout: float = 120):
    from whisper_tm.core import llm_client

    return llm_client._post_chat(api_url, headers, payload, (10, timeout)).json()


def vision_available() -> bool | None:
    """Can the language model see pictures? llama.cpp says so in /props;
    another server — unknown (None). Asked every five minutes at most."""
    try:
        from whisper_tm.core import llm_client

        api_url, *_ = llm_client.build_chat_payload(
            [{"role": "user", "content": "x"}], model=_model()
        )
    except Exception:  # noqa: BLE001 — not configured
        return False
    base = api_url.split("/v1/")[0].rstrip("/")
    now = time.monotonic()
    if _vision["key"] == base and now - _vision["at"] < 300:
        return _vision["value"]
    value = None
    try:
        import requests

        from whisper_tm.core import local_key

        r = requests.get(base + "/props", headers=local_key.headers_for(base + "/props"), timeout=4)
        if r.ok:
            mods = (r.json() or {}).get("modalities")
            if isinstance(mods, dict):
                value = bool(mods.get("vision"))
    except Exception:  # noqa: BLE001 — unknown then
        value = None
    _vision.update(at=now, key=base, value=value)
    return value


def sound_label(query: str) -> str | None:
    """The query as a short English sound label — None when the language
    model isn't there or the setting is off. Kept per query."""
    from whisper_tm.core.helpers import parse_bool

    q = (query or "").strip()
    if not q or not parse_bool(_cfg("llm_translate_sounds", True)) or not configured():
        return None
    with _labels_lock:
        if q in _labels:
            _labels.move_to_end(q)
            return _labels[q]
    try:
        api_url, headers, payload, ollama, _ = _payload(
            [{"role": "system", "content": LABEL_SYSTEM}, {"role": "user", "content": q}],
            max_tokens=24,
        )
        from whisper_tm.core import llm_client

        text = llm_client.answer_text(_post(api_url, headers, payload, 30), ollama)
    except Exception as e:  # noqa: BLE001 — the query as typed then
        logger.info("Media search: no sound label for %r: %s", q, e)
        return None
    label = " ".join(text.strip().strip("\"'").split())[:80]
    if not label:
        return None
    with _labels_lock:
        _labels[q] = label
        while len(_labels) > 512:
            _labels.popitem(last=False)
    return label


def image_part(path: str) -> dict:
    """A JPEG as a message part for a model that sees pictures."""
    data = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{data}"}}


def judge(query: str, image_path: str) -> float:
    """How well the picture matches ``query``, 0–9."""
    from whisper_tm.core import llm_client

    messages = [
        {"role": "system", "content": JUDGE_SYSTEM},
        {
            "role": "user",
            "content": [
                image_part(image_path),
                {
                    "type": "text",
                    "text": f'Search query: "{query}"\nHow well does this still match? One digit.',
                },
            ],
        },
    ]
    api_url, headers, payload, ollama, llama = _payload(messages, max_tokens=2)
    if llama:
        payload["logprobs"] = True
        payload["top_logprobs"] = 10
    data = _post(api_url, headers, payload, 120)
    text = llm_client.answer_text(data, ollama).strip()
    try:
        tops = data["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
        ps = {
            int(t["token"].strip()): math.exp(t["logprob"])
            for t in tops
            if t["token"].strip().isdigit()
        }
        if ps:
            return sum(d * p for d, p in ps.items()) / sum(ps.values())
    except (KeyError, IndexError, TypeError, ValueError):
        pass
    return float(text[0]) if text[:1].isdigit() else 0.0


def describe(image_path: str) -> str:
    from whisper_tm.core import llm_client

    lang = (_cfg("llm_language", "English") or "English").strip()
    messages = [
        {"role": "system", "content": DESCRIBE_SYSTEM.format(lang=lang)},
        {
            "role": "user",
            "content": [image_part(image_path), {"type": "text", "text": "Describe the still."}],
        },
    ]
    api_url, headers, payload, ollama, _ = _payload(messages, max_tokens=120)
    return " ".join(llm_client.answer_text(_post(api_url, headers, payload, 180), ollama).split())


# ── Checks: one picture per request, in the background ─────────────────────

_checks: OrderedDict[str, dict] = OrderedDict()
_checks_lock = threading.Lock()


def start_check(query: str, items: list[dict]) -> str:
    """Judge ``items`` (``[{"key", "thumb"}]``) for ``query`` — returns the
    check's id; :func:`check_state` follows it."""
    cid = uuid.uuid4().hex[:12]
    st = {
        "id": cid,
        "query": query,
        "total": len(items),
        "done": 0,
        "scores": {},
        "errors": 0,
        "running": True,
        "cancel": False,
        "started": time.time(),
        "ended": None,
        "error": None,
    }
    with _checks_lock:
        _checks[cid] = st
        while len(_checks) > 30:
            _checks.popitem(last=False)
    threading.Thread(
        target=_run_check, args=(st, items), name="media-llm-check", daemon=True
    ).start()
    return cid


def _run_check(st: dict, items: list[dict]) -> None:
    try:
        for it in items:
            if st["cancel"]:
                break
            try:
                score = judge(st["query"], it["thumb"])
                with _checks_lock:
                    st["scores"][str(it["key"])] = round(score, 2)
            except Exception as e:  # noqa: BLE001 — one picture less
                logger.info("Media check: %s", e)
                with _checks_lock:
                    st["errors"] += 1
                    if st["errors"] >= 3 and not st["scores"]:
                        st["error"] = str(e)
                        break
            with _checks_lock:
                st["done"] += 1
    finally:
        with _checks_lock:
            st["running"] = False
            st["ended"] = time.time()


def check_state(cid: str) -> dict | None:
    with _checks_lock:
        st = _checks.get(cid)
        return None if st is None else {**st, "scores": dict(st["scores"])}


def cancel_check(cid: str) -> bool:
    with _checks_lock:
        st = _checks.get(cid)
        if not st or not st["running"]:
            return False
        st["cancel"] = True
        return True
