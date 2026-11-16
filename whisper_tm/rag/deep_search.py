"""Deep search — "find everything" for a situation, checked passage by passage.

The goal: find everything, obscure situations too, without junk in
between. The quick search (rag/retrieve.py) shows the best few dozen; this
one goes through all of it. Measured on a real index against careful gold
labels:

* The right passages lie spread down the dense ranking: its best 100 hold
  24–33 % of them, its best 1,000 76–89 %, its best 3,000 89–97 %. So the
  candidates are wide: dense top ``rerank.deep_dense`` (2,000) and full text
  with the plan's words (up to 5,000; every passage with the must terms, up
  to 10,000).
* The 4B reranker is a very good order and a decent prefilter at a low
  threshold (0.05–0.1 kept 85–97 % of the right passages), but no junk
  filter (precision ≈0.4–0.5 at 0.5).
* A language model asked about ONE passage per call, with criteria, is the
  junk filter: precision 0.92 / 0.95 at recall 0.73 / 0.63. Several passages
  per call were much worse (0.43 / 0.08). ≈2.3 s per passage.

The steps:

1. **Plan** (one LLM call): the situation in one sentence, criteria for yes
   and no, must terms, 10–40 keywords — in the query's language. Without a
   reachable LLM: generic criteria and the query's own words.
2. **Candidates**: the lists above — and every passage a person the query
   names speaks in (the voice library knows their label: up to
   :data:`VOICE_K`, closest to the question first) — fused, copies
   collapsed. From here on a passage is read with its speakers' names
   (``[Alice]`` for ``[SPEAKER_04]``, rag/speakers.py): by the reranker, by the
   check, on screen.
3. **Order**: the reranker over every group, with the plan's criteria as
   its instruction (≈10 passages a second). Without it: the fused order.
4. **Check**: the language model, one passage per call, in that order —
   only groups at ``rerank.deep_prefilter`` or above (all when not
   reranked), at most ``rerank.deep_max_verify``. YES is a hit, UNCLEAR a
   maybe, NO goes. Every verdict is written at once; it stops early after
   150 NO in a row below a score of 0.2.

It runs in the LLM queue (kind ``deep_search``, core/llm_queue.py):
serialised with the other LLM work, visible there, cancellable, and after a
restart it goes on — with its plan, its ranked groups and the passages
already checked. Kept as JSON under the data folder
(``deep_searches/<id>.json``, the ranked groups in ``<id>.work.json``).
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
import uuid
from pathlib import Path

from whisper_tm.config import config
from whisper_tm.core import llm_client, rerank_server
from whisper_tm.rag import retrieve, speakers

logger = logging.getLogger(__name__)

#: The LLM task whose provider and model a deep search uses — the summary's,
#: as for analyses.
TASK = "summary"
#: Full-text list sizes: "must AND keywords" (or the keywords alone), and
#: every passage with the must terms.
FTS_K = 5000
MUST_K = 10000
#: The passages a named person speaks in, closest to the question first.
#: Measured on a query about a named person: the person spoke in thousands of
#: passages (well under half with their name in the words); most of them the
#: search had never seen, and the passages the reranker put first that really
#: matched sat far down the similarity order. So all of them, not a top.
VOICE_K = 5000
#: Early stop: this many NO in a row, and the next score below this.
EARLY_STOP_RUN = 150
EARLY_STOP_SCORE = 0.2
#: Without a language model, this many ranked passages come back unchecked.
UNCHECKED_MAX = 300
#: Rounds without an answer from the language model before the run gives
#: up, and the pause between them.
LLM_FAILURES_MAX = 3
LLM_RETRY_WAIT_SEC = 10.0
MAX_QUERY = 500
VERIFY_MAX_TOKENS = 5
PLAN_MAX_TOKENS = 900
LLM_TIMEOUT_SEC = 300
#: The ranked groups are written at least this often while ranking; the
#: passages are read from the index in blocks of this many.
WORK_SAVE_SEC = 30.0
RANK_BLOCK = 256
#: While the reranker reads, the best passages ranked so far are checked
#: already — at most this many after each ranked block, those scoring at
#: least LEAD_SCORE — so the first hits show within a minute instead of
#: after the whole ranking (several thousand groups take ~15 minutes).
LEAD_SCORE = 0.5
LEAD_PER_BLOCK = 12

PLAN_SYSTEM = """You prepare a search through automatically transcribed spoken \
conversations (transcription errors are possible). The user describes a situation and \
wants to find every excerpt in which it happens. Answer with a JSON object:

- "situation": the situation in one short sentence, in the language of the query.
- "criteria_yes": one or two short, concrete sentences in the language of the query: \
when does an excerpt show the situation actually happening?
- "criteria_no": one or two short sentences in the language of the query: what does \
not count (for example only mentioning a name or the topic, the roles reversed, \
talking about it in general)?
- "rerank_criteria": the same criteria in English, as "Relevant only if ... Not \
relevant if ...".
- "must_terms": names, or words the query puts in quotes, that every matching excerpt \
must contain — lowercase, usually none or one.
- "keywords": 10 to 40 lowercase words that people would actually say in such an \
excerpt, in the language of the query: the query's own content words and their other \
forms, synonyms, typical words for the situation (for an insult: the insults \
themselves). Write a word stem with a trailing * (for example insult*). Every keyword \
once — no repetitions."""

PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "situation": {"type": "string"},
        "criteria_yes": {"type": "string"},
        "criteria_no": {"type": "string"},
        "rerank_criteria": {"type": "string"},
        "must_terms": {"type": "array", "items": {"type": "string"}},
        "keywords": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "situation",
        "criteria_yes",
        "criteria_no",
        "rerank_criteria",
        "must_terms",
        "keywords",
    ],
    "additionalProperties": False,
}

#: The check of one passage. The structure that was measured: the
#: excerpt, the question, the criteria, a one-word answer.
VERIFY_USER = '''Transcript excerpt (automatically transcribed conversation, errors possible):
"""
{passage}
"""

Question: Does the situation "{situation}" happen in this excerpt?
{criteria}
Answer with exactly one English word: YES, UNCLEAR or NO.'''

#: The criteria when the plan has none (no language model for the plan).
GENERIC_CRITERIA = (
    "Only if it actually happens in the excerpt — merely mentioning a "
    "name or the topic is not enough."
)


class DeepSearchError(RuntimeError):
    """Unknown search, or one that can't start."""


# ══════════════════════════════════════════════════════════════════════════
#  Storage
# ══════════════════════════════════════════════════════════════════════════

_ID = re.compile(r"^[a-f0-9]{12}$")
_lock = threading.Lock()
_deleted: set[str] = set()
#: Stops asked for — the run looks between two steps. In memory only: a
#: search retried from the LLM queue after a stop goes on where it stopped.
_stops: set[str] = set()


def _folder() -> Path:
    d = Path(config.data_dir) / "deep_searches"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _path(did: str, suffix: str = ".json") -> Path:
    if not _ID.match(did or ""):
        raise DeepSearchError("unknown deep search")
    return _folder() / f"{did}{suffix}"


def _write(p: Path, data) -> None:
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    tmp.replace(p)


def load(did: str) -> dict:
    p = _path(did)
    if not p.is_file():
        raise DeepSearchError("unknown deep search")
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except ValueError as e:
        raise DeepSearchError(f"deep search {did} is not readable: {e}") from e


def save(s: dict) -> None:
    s["updated"] = time.time()
    with _lock:
        if s["id"] in _deleted:
            return
        _write(_path(s["id"]), s)


def _patch(did: str, **fields) -> None:
    with _lock:
        if did in _deleted:
            return
        p = _path(did)
        if not p.is_file():
            return
        s = json.loads(p.read_text(encoding="utf-8"))
        s.update(fields)
        s["updated"] = time.time()
        _write(p, s)


def _load_work(did: str) -> dict | None:
    p = _path(did, ".work.json")
    if not p.is_file():
        return None
    try:
        work = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        logger.warning(
            "Deep search %s: ranked groups not readable (%s) — collecting the candidates again",
            did,
            e,
        )
        return None
    if not isinstance(work, dict) or not isinstance(work.get("groups"), list):
        return None
    if len(work.get("scores") or []) != len(work["groups"]):
        work["scores"] = [None] * len(work["groups"])
        work["ranked"] = False
    return work


def _save_work(did: str, work: dict) -> None:
    with _lock:
        if did in _deleted:
            return
        _write(_path(did, ".work.json"), work)


def _new_record(query: str) -> dict:
    return {
        "id": uuid.uuid4().hex[:12],
        "query": query,
        "status": "queued",
        "stage": "plan",
        "message": "Waiting in the LLM queue",
        "created": time.time(),
        "started": None,
        "finished": None,
        "entry_id": 0,
        "plan": None,
        "progress": {
            "candidates": 0,
            "groups": 0,
            "ranked": 0,
            "rank_total": 0,
            "verified": 0,
            "verify_total": 0,
        },
        "counts": {"hits": 0, "maybe": 0, "no": 0},
        "results": [],
        "verified": {},
        "streak": 0,
        "reranked": None,
        "note": None,
        "error": None,
        "timings": {},
    }


def create(query: str) -> dict:
    """A new deep search, queued in the LLM queue."""
    query = " ".join((query or "").split())
    if not query:
        raise DeepSearchError("Empty search query")
    if len(query) > MAX_QUERY:
        raise DeepSearchError(f"The query is longer than {MAX_QUERY} characters")
    if _index_empty():
        # It searches the meaning index's passages: without them every list
        # came back empty (and the full text logged a missing table)
        raise DeepSearchError(
            "The meaning index is empty — build it first: Settings → "
            "Search → Semantic search → Embed new transcripts"
        )
    s = _new_record(query)
    _enqueue(s)
    return s


def _enqueue(s: dict) -> None:
    """Into the LLM queue. Saved first: the worker may take it up before
    ``enqueue`` returns — the entry's number goes in afterwards."""
    from whisper_tm.core.llm_queue import TYPE_DEEP_SEARCH, llm_queue

    s["entry_id"] = 0
    save(s)
    # The queue shows the name's last path part — a query with slashes would
    # lose its front
    name = ("Deep search · " + s["query"].replace("/", "∕"))[:160]
    s["entry_id"] = llm_queue.enqueue(TYPE_DEEP_SEARCH, name, {"deep_id": s["id"]})
    if not s["entry_id"]:
        s.update(status="error", error="The LLM queue is not available", finished=time.time())
        save(s)
        raise DeepSearchError("The LLM queue is not available")
    _patch(s["id"], entry_id=s["entry_id"])


def _queue_state(entry_id) -> tuple[str, str]:
    """(status, error) of the LLM queue entry — "" when it is gone."""
    from whisper_tm.extensions import db

    if not db or not entry_id:
        return "", ""
    try:
        with db.connection() as conn:
            r = conn.execute(
                "SELECT status, error FROM llm_queue WHERE id=?", (int(entry_id),)
            ).fetchone()
    except Exception:  # noqa: BLE001 — the stored state stands then
        return "?", ""
    return (r["status"], r["error"] or "") if r else ("", "")


def _sync(s: dict) -> dict:
    """A search as its queue entry stands: cancelled or removed there, put
    back to waiting by a restart — or retried there after a failure or a
    stop (it then goes on where it ended)."""
    if s.get("status") not in ("queued", "running", "error", "stopped") or not s.get("entry_id"):
        return s
    status, error = _queue_state(s.get("entry_id"))
    if s.get("status") in ("error", "stopped"):
        if status in ("pending", "running"):
            s.update(
                status="queued" if status == "pending" else "running", error=None, finished=None
            )
        return s
    if status == "pending":
        s["status"] = "queued"
    elif status == "running":
        s["status"] = "running"
    elif status in ("cancelled", ""):
        s.update(
            status="stopped", message="Removed from the LLM queue" if not status else "Stopped"
        )
    elif status == "error":
        s.update(status="error", error=s.get("error") or error)
    return s


_PLAN_PUBLIC = (
    "situation",
    "criteria_yes",
    "criteria_no",
    "rerank_criteria",
    "must_terms",
    "keywords",
    "source",
)


def public(s: dict, since: int = 0) -> dict:
    """What the page gets: the state, and the results from ``since`` on."""
    s = _sync(dict(s))
    results = s.get("results") or []
    since = max(0, min(int(since or 0), len(results)))
    started, finished = s.get("started"), s.get("finished")
    elapsed = ((finished or time.time()) - started) if started else 0.0
    plan = s.get("plan")
    counts = dict(s.get("counts") or {})
    counts.setdefault("hits", 0)
    counts.setdefault("maybe", 0)
    shown = [dict(r, copies=[dict(c) for c in r.get("copies") or []]) for r in results[since:]]
    _refresh_places(shown, s.get("id"))
    return {
        "id": s["id"],
        "query": s.get("query"),
        "status": s.get("status"),
        "stage": s.get("stage"),
        "message": s.get("message") or "",
        "plan": {k: plan.get(k) for k in _PLAN_PUBLIC} if plan else None,
        "progress": s.get("progress") or {},
        "counts": counts,
        "results": shown,
        "next": len(results),
        "total_results": len(results),
        "created": s.get("created"),
        "started": started,
        "finished": finished,
        "elapsed_sec": round(max(0.0, elapsed), 1),
        "reranked": s.get("reranked"),
        "note": s.get("note"),
        "error": s.get("error") or None,
        "entry_id": s.get("entry_id"),
        "timings": s.get("timings") or {},
    }


def _refresh_places(items: list[dict], did=None) -> None:
    """Look the places of stored results up again: one written by an older
    version carried the stored media_dir as its root — wrong for many
    transcripts (rag/retrieve.display_paths) — and a file may have moved or
    gone since. Without an answer from the database the stored ones stand."""
    keys = [x.get("transcript_key") for it in items for x in [it, *(it.get("copies") or [])]]
    try:
        places = retrieve.display_paths(keys, _main_db())
    except Exception as e:  # noqa: BLE001
        logger.debug("Deep search %s: places not refreshed: %s", did, e)
        return
    if not places:
        return
    for it in items:
        for x in [it, *(it.get("copies") or [])]:
            p = places.get(x.get("transcript_key") or "", {})
            x["rel_path"] = p.get("rel_path")
            x["file_dir"] = p.get("file_dir")


def listing(limit: int = 200) -> list[dict]:
    """The deep searches, newest first — without their results."""
    out = []
    for p in _folder().glob("*.json"):
        if p.name.endswith(".work.json"):
            continue
        try:
            s = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(s, dict) or not _ID.match(str(s.get("id") or "")):
            continue
        _sync(s)
        counts = s.get("counts") or {}
        out.append(
            {
                "id": s["id"],
                "query": s.get("query"),
                "status": s.get("status"),
                "stage": s.get("stage"),
                "created": s.get("created"),
                "started": s.get("started"),
                "finished": s.get("finished"),
                "counts": {"hits": counts.get("hits", 0), "maybe": counts.get("maybe", 0)},
                "entry_id": s.get("entry_id"),
            }
        )
    out.sort(key=lambda x: -(x.get("created") or 0))
    return out[:limit]


def stop(did: str) -> bool:
    """Stop a waiting or running search; what it found stays."""
    s = _sync(load(did))
    if s.get("status") not in ("queued", "running"):
        return False
    _stops.add(did)
    entry = s.get("entry_id")
    if entry:
        from whisper_tm.core.llm_queue import llm_queue

        llm_queue.cancel(int(entry))
    status, _e = _queue_state(entry)
    if status in ("cancelled", "") or s.get("status") == "queued" and status != "running":
        # It never started (or is gone from the queue): nobody else writes it.
        # Nor reads the stop — left in _stops, it would stop the first retry
        # at once
        _stops.discard(did)
        s = load(did)
        if s.get("status") in ("queued", "running"):
            s.update(
                status="stopped",
                message="Stopped before it started" if not s.get("started") else "Stopped",
                finished=time.time(),
            )
            save(s)
    return True


def delete(did: str) -> bool:
    """Delete a search — a running one is stopped first."""
    try:
        s = _sync(load(did))
    except DeepSearchError:
        return False
    if s.get("status") in ("queued", "running"):
        try:
            stop(did)
        except Exception as e:  # noqa: BLE001 — it goes either way
            logger.warning("Deep search %s: stop before deleting failed: %s", did, e)
    with _lock:
        _deleted.add(did)
        for suffix in (".json", ".work.json"):
            p = _path(did, suffix)
            if p.is_file():
                p.unlink()
    return True


# ══════════════════════════════════════════════════════════════════════════
#  1. The plan
# ══════════════════════════════════════════════════════════════════════════

_STRING_FIELD = r'"{}"\s*:\s*"((?:[^"\\]|\\.)*)"'
_ARRAY_FIELD = r'"{}"\s*:\s*\[([^\]]*)'
_ITEM = re.compile(r'"((?:[^"\\]|\\.)*)"')


def _unescape(s: str) -> str:
    try:
        return json.loads(f'"{s}"')
    except ValueError:
        return s


def _salvage(text: str) -> dict:
    """What can be read from a cut-off JSON answer — a model looping over its
    keywords runs into its token limit (e.g. one keyword repeated many times)."""
    out: dict = {}
    for key in ("situation", "criteria_yes", "criteria_no", "rerank_criteria"):
        m = re.search(_STRING_FIELD.format(key), text or "")
        if m:
            out[key] = _unescape(m.group(1))
    for key in ("must_terms", "keywords"):
        m = re.search(_ARRAY_FIELD.format(key), text or "")
        if m:
            out[key] = [_unescape(x) for x in _ITEM.findall(m.group(1))]
    return out


def _keyword(raw: str, stop: set[str]) -> str | None:
    """One keyword as a search term: lowercase tokens, a trailing ``*`` kept
    for a single stem; stopwords, short words and long phrases left out."""
    raw = str(raw or "").strip().lower()
    star = raw.endswith("*")
    toks = retrieve._WORD.findall(raw)
    if not toks or len(toks) > 3:
        return None
    if len(toks) == 1:
        tok = toks[0]
        if len(tok) < retrieve.MIN_WORD or tok in stop:
            return None
        return tok + ("*" if star else "")
    # A phrase of stopwords only ("I have") matches nearly
    # every excerpt and only widens the candidates
    if all(t in stop or len(t) < retrieve.MIN_WORD for t in toks):
        return None
    return " ".join(toks)


def parse_plan(
    text: str, query: str, *, stop: set[str] | None = None, known_names=()
) -> dict | None:
    """The plan from the model's answer — tolerant of a cut-off answer.

    Keywords: deduplicated, stopwords out, at most 40, the query's own words
    first. Must terms only where the query has them (a model must not narrow
    the search on its own), plus what the query quotes and the names of the
    voice library it names. ``None`` when the answer holds nothing usable.
    """
    stop = retrieve._stop() if stop is None else stop
    data = llm_client.extract_json(text or "")
    if not isinstance(data, dict):
        data = _salvage(text or "")
    if not data:
        return None
    own = retrieve.query_terms(query, stop=stop, known_names=known_names)
    query_tokens = set(retrieve._WORD.findall((query or "").lower()))
    must = []
    for m in data.get("must_terms") or []:
        toks = retrieve._WORD.findall(str(m or "").lower())
        if toks and len(toks) <= 3 and all(t in query_tokens for t in toks):
            must.append(" ".join(toks))
    must = retrieve._dedupe(must + own["quoted"] + own["names"])[: retrieve.MAX_MUST]
    words = [_keyword(k, stop) for k in data.get("keywords") or []]
    keywords = [
        k for k in retrieve._dedupe(own["words"] + own["capitalised"] + words) if k not in must
    ][: retrieve.MAX_WORDS]

    def text_of(key, limit):
        return " ".join(str(data.get(key) or "").split())[:limit]

    situation = text_of("situation", 300)
    if not situation and not keywords:
        return None
    return {
        "situation": situation or query,
        "criteria_yes": text_of("criteria_yes", 400),
        "criteria_no": text_of("criteria_no", 400),
        "rerank_criteria": text_of("rerank_criteria", 600),
        "must_terms": must,
        "keywords": keywords,
        "source": "llm",
    }


def fallback_plan(query: str, reason: str = "") -> dict:
    """The plan without a language model: generic criteria, the query's own
    words, must terms from quotes and the voice library's names."""
    t = retrieve.query_terms(query, known_names=retrieve.known_names())
    return {
        "situation": query,
        "criteria_yes": "",
        "criteria_no": "",
        "rerank_criteria": "",
        "must_terms": retrieve._dedupe(t["quoted"] + t["names"]),
        "keywords": retrieve._dedupe(t["words"] + t["capitalised"]),
        "source": "fallback",
        "reason": reason,
    }


def _llm_ready() -> bool:
    """Is a language model set up for the deep search's task?"""
    return llm_client.ready(TASK)


def make_plan(query: str) -> dict:
    """Ask the language model for the plan; the fallback when it can't."""
    if not _llm_ready():
        return fallback_plan(query, "no language model set up")
    messages = [
        {"role": "system", "content": PLAN_SYSTEM},
        {"role": "user", "content": f"Query: {query}"},
    ]
    common = dict(
        max_tokens=PLAN_MAX_TOKENS,
        temperature=0.0,
        thinking=False,
        template_kwargs={"enable_thinking": False},
        task=TASK,
        timeout=LLM_TIMEOUT_SEC,
        retries=1,
    )
    try:
        try:
            text, _usage = llm_client.chat(
                messages, response_schema=PLAN_SCHEMA, schema_name="search_plan", **common
            )
        except llm_client.LlmAnswerError as e:
            msg = str(e).lower()
            if "400" not in msg or not any(
                w in msg for w in ("response_format", "json_schema", "grammar", "format")
            ):
                raise
            logger.info("Deep search plan: response_format rejected — asking for plain JSON")
            hint = "\n\nAnswer ONLY with the JSON object, no other text:\n" + json.dumps(
                PLAN_SCHEMA
            )
            text, _usage = llm_client.chat(
                [messages[0], {"role": "user", "content": messages[1]["content"] + hint}], **common
            )
    except llm_client.LlmError as e:
        logger.warning(
            "Deep search plan for %r: no answer from the language model (%s) — generic criteria",
            query[:80],
            e,
        )
        return fallback_plan(query, f"the language model did not answer: {str(e)[:160]}")
    plan = parse_plan(text, query, known_names=retrieve.known_names())
    if plan is None:
        logger.warning("Deep search plan for %r: answer not usable: %r", query[:80], text[:300])
        return fallback_plan(query, "the language model's plan was not usable")
    return plan


# ══════════════════════════════════════════════════════════════════════════
#  4. The check of one passage
# ══════════════════════════════════════════════════════════════════════════


def criteria_text(plan: dict) -> str:
    text = " ".join(x for x in (plan.get("criteria_yes"), plan.get("criteria_no")) if x)
    return text.strip() or GENERIC_CRITERIA


def verify_prompt(passage: str, plan: dict) -> str:
    return VERIFY_USER.format(
        passage=(passage or "")[: rerank_server.DOC_CHARS],
        situation=plan.get("situation") or "",
        criteria=criteria_text(plan),
    )


def parse_verdict(text: str) -> str:
    """``hit`` (YES), ``no`` (NO) or ``maybe`` (UNCLEAR, and anything else —
    an answer not understood is kept for the user to look at)."""
    words = retrieve._WORD.findall((text or "").upper())
    first = words[0] if words else ""
    if first in ("YES", "Y"):
        return "hit"
    if first in ("NO", "N"):
        return "no"
    return "maybe"


def ask_verdict(passage: str, plan: dict) -> tuple[str, str]:
    """``(verdict, answer)`` of the language model for one passage."""
    text, _usage = llm_client.chat(
        [{"role": "user", "content": verify_prompt(passage, plan)}],
        max_tokens=VERIFY_MAX_TOKENS,
        temperature=0.0,
        thinking=False,
        template_kwargs={"enable_thinking": False},
        task=TASK,
        timeout=LLM_TIMEOUT_SEC,
        retries=2,
    )
    return parse_verdict(text), text.strip()[:40]


def rerank_instruction(plan: dict) -> str:
    crit = (plan.get("rerank_criteria") or "").strip()
    if not crit and plan.get("source") == "llm":
        crit = criteria_text(plan)
    if not crit:
        return rerank_server.default_instruction()
    return rerank_server.CRITERIA_INSTRUCTION + crit


def verify_order(scores: list, prefilter: float, max_verify: int) -> list[int]:
    """Which groups are checked, in which order: the reranked ones at
    ``prefilter`` or above, best first; then the ones without a score in
    their fused order (all of them when nothing was reranked); at most
    ``max_verify``."""
    ranked = sorted(
        (i for i, sc in enumerate(scores) if sc is not None and sc >= prefilter),
        key=lambda i: (-scores[i], i),
    )
    rest = [i for i, sc in enumerate(scores) if sc is None]
    return (ranked + rest)[: max(0, int(max_verify))]


# ══════════════════════════════════════════════════════════════════════════
#  The run (in the LLM queue's worker)
# ══════════════════════════════════════════════════════════════════════════


def _rag_db():
    from whisper_tm.rag.service import make_rag_db

    return make_rag_db()


def _index_empty() -> bool:
    """No passage in the meaning index yet. Unknown (it can't be read) is
    not empty: the run then says what fails."""
    try:
        st = _rag_db().stats()
    except Exception as e:  # noqa: BLE001
        logger.debug("Deep search: index not readable: %s", e)
        return False
    return not st.get("exists") or not st.get("chunks")


def _embed_fn():
    from whisper_tm.rag.service import make_query_embed_fn

    return make_query_embed_fn()


def _main_db():
    from whisper_tm.extensions import db

    return db


class _Run:
    """One run of a deep search: its record, progress and saving."""

    def __init__(self, s: dict, queue, entry_id: int):
        self.s = s
        self.queue = queue
        self.entry_id = entry_id
        self.did = s["id"]
        self._last_save = 0.0

    def abort(self) -> bool:
        return (
            self.queue.abort_requested(self.entry_id) or self.did in _stops or self.did in _deleted
        )

    def check(self) -> None:
        if self.abort():
            raise InterruptedError("Stopped")

    def stage(self, name: str, message: str, pct: int) -> None:
        self.s.update(stage=name, message=message)
        self.save()
        self.queue.progress(self.entry_id, pct, message)

    def report(self, message: str, pct: int, *, every: float = 0.0) -> None:
        """Progress: into the record (at most every ``every`` seconds) and
        into the LLM queue entry."""
        self.s["message"] = message
        if self.save(every=every):
            self.queue.progress(self.entry_id, pct, message)

    def save(self, *, every: float = 0.0) -> bool:
        now = time.monotonic()
        if every and now - self._last_save < every:
            return False
        self._last_save = now
        save(self.s)
        return True


def run(did: str, queue, entry_id: int) -> dict:
    """Plan, candidates, order, check — called by core/llm_tasks."""
    s = load(did)
    if s.get("status") not in ("queued", "running") or s.get("entry_id") != entry_id:
        # A retry or a new entry: an old stop is not meant for it
        _stops.discard(did)
    r = _Run(s, queue, entry_id)
    t0 = time.time()
    s.update(status="running", error=None, entry_id=entry_id, finished=None)
    s["started"] = s.get("started") or t0
    try:
        r.check()
        timings = s.setdefault("timings", {})
        # ── 1. Plan ──────────────────────────────────────────────────────
        if not s.get("plan"):
            r.stage("plan", "Planning the search…", 1)
            t1 = time.monotonic()
            s["plan"] = make_plan(s["query"])
            timings["plan"] = round(time.monotonic() - t1, 1)
            if s["plan"].get("source") == "fallback" and _llm_ready():
                s["note"] = _join_note(
                    s.get("note"),
                    "Without a plan from the language "
                    f"model ({s['plan'].get('reason')}): generic "
                    "criteria and the query's own words",
                )
            r.save()
        plan = s["plan"]
        r.check()
        db = _rag_db()
        # ── 2. Candidates ────────────────────────────────────────────────
        work = _load_work(did)
        if work is None:
            r.stage("candidates", "Collecting candidates…", 3)
            t1 = time.monotonic()
            work = _collect(s, plan, db)
            timings["candidates"] = round(time.monotonic() - t1, 1)
            _save_work(did, work)
        groups = work["groups"]
        s["progress"].update(candidates=work.get("candidates", 0), groups=len(groups))
        r.save()
        if not groups:
            return _finish(r, t0, "Nothing in the index matches")
        # ── 3. Order ─────────────────────────────────────────────────────
        if not work.get("ranked"):
            _rank(r, work, plan, db)
        s["reranked"] = work.get("reranked")
        if work.get("rank_note"):
            s["note"] = _join_note(s.get("note"), work["rank_note"])
        # ── 4. Check ─────────────────────────────────────────────────────
        _check(r, work, plan, db)
        return _finish(r, t0, None)
    except InterruptedError:
        s.update(status="stopped", message="Stopped", finished=time.time())
        save(s)
        _stops.discard(did)
        raise
    except Exception as e:
        logger.exception("Deep search %s (%r) failed", did, s.get("query", "")[:80])
        s.update(status="error", message="Failed", error=str(e)[:500], finished=time.time())
        save(s)
        raise


def _join_note(old: str | None, new: str) -> str:
    """Notes side by side — once each (a resumed run meets them again)."""
    if not old:
        return new
    return old if new in old.split(" · ") else f"{old} · {new}"


def _finish(r: _Run, t0: float, note: str | None) -> dict:
    s = r.s
    if note:
        s["note"] = _join_note(s.get("note"), note)
    s.update(
        status="done",
        stage="done",
        finished=time.time(),
        message=f"Done: {s['counts'].get('hits', 0)} hits, {s['counts'].get('maybe', 0)} unclear",
    )
    save(s)
    _stops.discard(s["id"])
    return {
        "deep_search": s["id"],
        "hits": s["counts"].get("hits", 0),
        "maybe": s["counts"].get("maybe", 0),
        "checked": s["progress"].get("verified", 0),
        "duration": round(time.time() - t0, 1),
    }


def _collect(s: dict, plan: dict, db) -> dict:
    """Step 2: the wide candidate lists, fused, copies collapsed."""
    must = list(plan.get("must_terms") or [])
    embed = None
    try:
        embed = _embed_fn()
    except Exception as e:  # noqa: BLE001 — the full text still answers
        logger.warning("Deep search %s: no meaning search (%s)", s["id"], e)
    rows, info = retrieve.candidates(
        s["query"],
        k_dense=rerank_server.setting_int("deep_dense") if embed else 0,
        k_fts=FTS_K,
        extra_terms=plan.get("keywords") or [],
        must=must,
        k_must=MUST_K if must else 0,
        or_list=not must,
        db=db,
        embed_fn=embed,
        k_voice=VOICE_K,
        rrf_k=int(config.get("rag", "rrf_k", default=retrieve.RRF_K) or retrieve.RRF_K),
    )
    if info.get("dense_error") or embed is None:
        s["note"] = _join_note(
            s.get("note"), "Meaning search not available — candidates from the full text only"
        )
    vectors = None
    if len(rows) > 1:
        try:
            vectors = db.vectors_for([x["id"] for x in rows])
        except Exception as e:  # noqa: BLE001 — same text still collapses
            logger.warning("Deep search %s: stored vectors not readable: %s", s["id"], e)
    groups = retrieve.group_copies(rows, vectors)
    logger.info(
        "Deep search %s: %d candidates (%s) in %d groups",
        s["id"],
        len(rows),
        info.get("lists"),
        len(groups),
    )
    return {
        "version": 1,
        "candidates": len(rows),
        "lists": info.get("lists") or {},
        "voices": info.get("voices") or {},
        "groups": [
            {
                "id": g["id"],
                "copies": [c["id"] for c in g["copies"]],
                "fused": round(g["fused"], 6),
                "match": g.get("match"),
            }
            for g in groups
        ],
        "scores": [None] * len(groups),
        "ranked": False,
        "reranked": None,
        "rank_note": None,
    }


class _NotRanking(Exception):
    """An error of the check between two blocks, carried past the
    reranker's own handling (its ``__cause__``)."""


def _rank(r: _Run, work: dict, plan: dict, db) -> None:
    """Step 3: the reranker over every group, written as it goes. Without
    the reranker (or when it drops out) the rest keeps its fused order."""
    s, did = r.s, r.did
    scores = work["scores"]
    total = len(scores)
    done = sum(1 for sc in scores if sc is not None)
    s["progress"].update(rank_total=total, ranked=done)
    r.stage("ranking", f"Ranking {done:,} / {total:,}", 5)
    t1 = time.monotonic()
    llm = _llm_ready()

    def lead():
        if llm:
            try:
                _check_leading(r, work, plan, db)
            except Exception as e:
                # Not the reranker's failure: a language model outage must
                # not go down as "reranker failed" and mark the ranking done
                raise _NotRanking(e) from e

    try:
        _rank_all(r, work, rerank_instruction(plan), db, after_block=lead)
        work["reranked"] = True
    except rerank_server.RerankUnavailable as e:
        done = sum(1 for sc in scores if sc is not None)
        logger.warning(
            "Deep search %s: reranker not available after %d of %d (%s): %s",
            did,
            done,
            total,
            e.reason,
            e,
        )
        work["reranked"] = bool(done)
        work["rank_note"] = (
            f"Reranker stopped after {done:,} of {total:,} "
            f"({e.detail or e.reason}) — the rest in fused order"
            if done
            else e.note()
        )
    except InterruptedError:
        _save_work(did, work)  # what is ranked stays for a retry
        raise
    except _NotRanking as e:
        _save_work(did, work)  # what is ranked stays for a retry
        raise e.__cause__ from None
    except Exception as e:  # noqa: BLE001 — the check still runs, in fused order
        logger.exception("Deep search %s: the reranker failed", did)
        done = sum(1 for sc in scores if sc is not None)
        work["reranked"] = bool(done)
        work["rank_note"] = rerank_server.RerankUnavailable("failed", str(e)).note()
    finally:
        timings = s.setdefault("timings", {})
        timings["ranking"] = round(timings.get("ranking", 0.0) + time.monotonic() - t1, 1)
    work["ranked"] = True
    _save_work(did, work)
    s["progress"]["ranked"] = sum(1 for sc in scores if sc is not None)
    r.save()


def _rank_all(r: _Run, work: dict, instruction: str, db, *, after_block=None) -> None:
    """Score every group without a score, in blocks; the scores are written
    to the work file at least every :data:`WORK_SAVE_SEC`. ``after_block()``
    runs after each block (the deep search checks its leading passages)."""
    s, groups, scores = r.s, work["groups"], work["scores"]
    total = len(groups)
    todo = [i for i, sc in enumerate(scores) if sc is None]
    last_save = time.monotonic()
    for b in range(0, len(todo), RANK_BLOCK):
        block = todo[b : b + RANK_BLOCK]
        rows = db.rows_by_id([groups[i]["id"] for i in block])
        speakers.name_rows(rows.values())
        texts = [(rows.get(groups[i]["id"]) or {}).get("content") or "" for i in block]
        base = total - len(todo) + b

        def progress(done, _n, base=base):
            ranked = base + done
            s["progress"]["ranked"] = ranked
            r.report(f"Ranking {ranked:,} / {total:,}", 5 + int(45 * ranked / total), every=2.0)

        got = rerank_server.rerank(
            s["query"],
            texts,
            instruction=instruction,
            wait_sec=rerank_server.DEEP_WAIT_SEC,
            cancel=r.abort,
            progress=progress,
            yield_between=True,
        )
        for i, v in zip(block, got):
            scores[i] = float(v)
        if time.monotonic() - last_save >= WORK_SAVE_SEC:
            _save_work(r.did, work)
            last_save = time.monotonic()
        if after_block is not None and b + RANK_BLOCK < len(todo):
            after_block()
            r.stage(
                "ranking",
                f"Ranking {s['progress'].get('ranked', 0):,} / {total:,}",
                5 + int(45 * s["progress"].get("ranked", 0) / max(1, total)),
            )


def _result(
    n: int,
    verdict: str,
    group: dict,
    row: dict,
    score,
    copies: list[dict],
    *,
    unchecked: bool = False,
) -> dict:
    out = {
        "n": n,
        "verdict": verdict,
        "id": group["id"],
        "transcript_key": row.get("transcript_key"),
        "media_path": row.get("media_path"),
        "chunk_index": row.get("chunk_index"),
        "start_sec": row.get("start_sec"),
        "end_sec": row.get("end_sec"),
        "speaker": row.get("speaker"),
        "speaker_name": row.get("speaker_name"),
        "snippet": (row.get("content") or "")[: rerank_server.DOC_CHARS],
        "score": score,
        "relevance": score,
        "copies": copies,
    }
    if unchecked:
        out["unchecked"] = True
    return out


def _with_places(items: list[dict]) -> None:
    retrieve.place_copies(items, _main_db())


def _copies(db, group: dict) -> list[dict]:
    ids = group.get("copies") or []
    if not ids:
        return []
    rows = db.rows_by_id(ids)
    return [retrieve.copy_of(rows[i]) for i in ids if i in rows]


def _check(r: _Run, work: dict, plan: dict, db) -> None:
    """Step 4: one passage per call, every verdict written at once."""
    s = r.s
    order = verify_order(
        work["scores"],
        rerank_server.setting_float("deep_prefilter"),
        rerank_server.setting_int("deep_max_verify"),
    )
    verified = s.setdefault("verified", {})
    s.setdefault("counts", {"hits": 0, "maybe": 0, "no": 0})
    s["progress"]["verify_total"] = len(order)
    if not _llm_ready():
        _unchecked(r, work, order, db)
        return
    r.stage("checking", f"Checking {len(verified):,} of {len(order):,}", 50)
    t1 = time.monotonic()
    try:
        _verify(r, work, plan, db, order, early_stop=True)
    finally:
        s.setdefault("timings", {})["checking"] = round(
            s.get("timings", {}).get("checking", 0.0) + time.monotonic() - t1, 1
        )


def _check_leading(r: _Run, work: dict, plan: dict, db) -> None:
    """While ranking: check the best passages ranked so far (at most
    :data:`LEAD_PER_BLOCK`, scoring at least :data:`LEAD_SCORE`), so that
    hits show long before the ranking is through."""
    s = r.s
    groups, scores = work["groups"], work["scores"]
    verified = s.setdefault("verified", {})
    lead = sorted(
        (
            i
            for i, sc in enumerate(scores)
            if sc is not None and sc >= LEAD_SCORE and str(groups[i]["id"]) not in verified
        ),
        key=lambda i: -scores[i],
    )[:LEAD_PER_BLOCK]
    if not lead:
        return
    s.setdefault("counts", {"hits": 0, "maybe": 0, "no": 0})
    s["progress"]["verify_total"] = max(
        s["progress"].get("verify_total") or 0,
        sum(
            1
            for sc in scores
            if sc is not None and sc >= rerank_server.setting_float("deep_prefilter")
        ),
    )
    total = len(scores)
    ranked = sum(1 for sc in scores if sc is not None)
    pct = 5 + int(45 * ranked / max(1, total))
    r.stage(
        "checking",
        f"Checking the best {len(lead)} ranked so far (ranked {ranked:,} / {total:,})",
        pct,
    )
    t1 = time.monotonic()

    def lead_report(n, hits):
        r.report(f"Ranked {ranked:,} / {total:,} · checked {n:,} · {hits} hits so far", pct)

    try:
        _verify(r, work, plan, db, lead, early_stop=False, report=lead_report)
    finally:
        s.setdefault("timings", {})["checking"] = round(
            s.get("timings", {}).get("checking", 0.0) + time.monotonic() - t1, 1
        )


def _verify(
    r: _Run, work: dict, plan: dict, db, order: list[int], *, early_stop: bool, report=None
) -> None:
    """Ask the language model about ``order`` (group indices), skipping what
    is verified already; hits and unclear ones go to the results at once."""
    s, did = r.s, r.did
    groups, scores = work["groups"], work["scores"]
    verified = s.setdefault("verified", {})
    counts = s.setdefault("counts", {"hits": 0, "maybe": 0, "no": 0})
    total = max(len(order), s["progress"].get("verify_total") or 0)
    for pos in range(0, len(order), 50):
        block = order[pos : pos + 50]
        rows = db.rows_by_id([groups[i]["id"] for i in block])
        speakers.name_rows(rows.values())
        for i in block:
            g = groups[i]
            key = str(g["id"])
            if key in verified:
                continue
            r.check()
            sc = scores[i]
            if (
                early_stop
                and s.get("streak", 0) >= EARLY_STOP_RUN
                and sc is not None
                and sc < EARLY_STOP_SCORE
            ):
                s["note"] = _join_note(
                    s.get("note"),
                    f"Stopped early: the last "
                    f"{EARLY_STOP_RUN} passages checked were all no, "
                    f"and the rest rank below {EARLY_STOP_SCORE}",
                )
                return
            row = rows.get(g["id"])
            if row is None:
                logger.info("Deep search %s: passage %s is gone from the index", did, g["id"])
                verified[key] = "gone"
                continue
            verdict = _verdict_for(r, row.get("content") or "", plan, g["id"])
            verified[key] = verdict
            counted = "hits" if verdict == "hit" else verdict
            counts[counted] = counts.get(counted, 0) + 1
            s["streak"] = s.get("streak", 0) + 1 if verdict == "no" else 0
            if verdict in ("hit", "maybe"):
                item = _result(len(s["results"]), verdict, g, row, sc, _copies(db, g))
                _with_places([item])
                s["results"].append(item)
            n = len(verified)
            s["progress"]["verified"] = n
            if report is not None:
                report(n, counts.get("hits", 0))
            else:
                r.report(
                    f"Checking {n:,} of {total:,} · {counts.get('hits', 0)} hits so far",
                    50 + int(49 * n / max(1, total)),
                )


def _verdict_for(r: _Run, passage: str, plan: dict, chunk_id) -> str:
    """The language model's verdict on one passage. Without an answer it is
    asked again (the client itself retries twice first); after
    :data:`LLM_FAILURES_MAX` rounds the run fails — what it found stays, and
    a retry from the LLM queue goes on from here. A passage the model can't
    take (a 400, an empty answer) is kept as unclear."""
    failures = 0
    while True:
        try:
            verdict, _answer = ask_verdict(passage, plan)
            return verdict
        except llm_client.LlmConnectionError as e:
            failures += 1
            logger.warning(
                "Deep search %s: no answer from the language model for passage %s (%d of %d): %s",
                r.did,
                chunk_id,
                failures,
                LLM_FAILURES_MAX,
                e,
            )
            if failures >= LLM_FAILURES_MAX:
                raise RuntimeError(f"The language model is not reachable: {e}") from e
            end = time.monotonic() + LLM_RETRY_WAIT_SEC
            while time.monotonic() < end:
                r.check()
                time.sleep(min(0.5, LLM_RETRY_WAIT_SEC))
        except llm_client.LlmError as e:
            logger.warning("Deep search %s: passage %s not checked: %s", r.did, chunk_id, e)
            return "maybe"


def _unchecked(r: _Run, work: dict, order: list[int], db) -> None:
    """No language model: the best ranked passages come back as they are."""
    s = r.s
    groups, scores = work["groups"], work["scores"]
    r.stage("checking", "No language model — listing the ranked passages unchecked", 90)
    s["results"] = []
    picked = order[:UNCHECKED_MAX]
    rows = db.rows_by_id([groups[i]["id"] for i in picked])
    speakers.name_rows(rows.values())
    for i in picked:
        row = rows.get(groups[i]["id"])
        if row is None:
            continue
        s["results"].append(
            _result(
                len(s["results"]),
                "maybe",
                groups[i],
                row,
                scores[i],
                _copies(db, groups[i]),
                unchecked=True,
            )
        )
    _with_places(s["results"])
    s["counts"]["maybe"] = len(s["results"])
    s["note"] = _join_note(
        s.get("note"),
        "No language model set up (Settings → AI models): the passages are ranked, not checked",
    )
    r.save()
