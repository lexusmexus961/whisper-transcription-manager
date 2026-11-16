"""Find's plan — the description split into what each search can find.

The language model writes the plan (structured JSON); then it is checked
against what the library knows (``ground``), because a model reads
"Alice dances around" as the person "Alice Dances", hears "talking" as a sound
and infers "a video" where the user never said so:

* people: kept only as the voice library writes them — a name it doesn't
  know falls back to one of its words that is a library name (a plan that
  asks for conversations keeps the names typed in transcripts too, and an
  unknown one: the Conversations tab says it is unknown);
* a sound label that only says "someone talks" is dropped (speech is what
  the transcript searches are for);
* the kind of recording counts as a hard filter only when the description
  says it in so many words (``media_stated``: "a video", "a voice
  message" …) — inferred by the model, it is a slight preference.

A description may ask for no moment at all (``asks_for``): recordings that
exist more than once, or whole conversations with given people, of a length
or of a kind (the ``conversation`` object) — the run ends with the plan, and
the page shows them where they live (agent.py).
"""

from __future__ import annotations

import json
import logging
import math
import re
import threading
import time

logger = logging.getLogger(__name__)

LLM_TIMEOUT_SEC = 120

PLAN_SYSTEM = """You plan a search through a library of recordings — videos, streams, \
voice messages — that has transcripts of everything said and an index of what is seen \
and heard in them. The user describes a moment in their own words. Split the \
description into what each kind of search can find, and answer with a JSON object:

- "summary": the moment in one short sentence, in the language of the description.
- "said_words": words or short phrases that were literally said and are distinctive \
enough to find word by word (names of games, places, products, quotes) — written the \
way they would be spoken and transcribed, lowercase; at most 3; [] when nothing \
specific is said.
- "said_meaning": what is said or happens in the talk, one sentence in the language of \
the description, for a search by meaning — leave out what is only seen; "" if nothing.
- "seen": what is visible, as a short English caption ("a woman climbing onto a table at \
a party"); "" when nothing visual is described.
- "heard": a SHORT English label of a sound that is not speech, 1 to 4 words \
("shouting", "dog barking", "rock music", "gunshots"); "" when no such sound is \
described — people talking is NOT a sound here, the transcripts cover it.
- "people": the people who are in it — only names, each written exactly as in the \
list of known people when it is one of them ("Alice dances around" → "Alice", not \
"Alice Dances"); [].
- "file_hints": short pieces of the file name or folder the user remembers. WhatsApp \
files are named like PTT-20240301-WA0001.opus or VID-20240301-WA0002.mp4, so a \
WhatsApp file gives "-WA"; [].
- "media": "video" when it is a video, "audio" when it is a voice message or sound \
only, otherwise "any".
- "language": the language the description is written in, as an English word ("English").
- "asks_for": "moments" when it describes something to find in the recordings — nearly \
always; "conversations" when the user wants whole conversations or talks — with given \
people, of a length or of a kind ("a long talk with Alice and Bob", "conversations where \
Carol speaks in long sentences", "an hour of discussion between three people") — rather \
than a moment where something is said, seen or heard; "duplicates" when the user asks \
which recordings exist more than once (copies of the same recording, in other formats or \
folders) rather than for a moment; "contained" when the user asks which recordings \
contain another one (compilations, a longer recording a clip was cut from). With \
"conversations", "duplicates" or "contained", file_hints holds the folder or name they \
are to be in.
- "conversation": what the wanted conversations are like when asks_for is \
"conversations" — who takes part goes into "people"; for anything else 0, 0, "any", \
"any". "min_minutes" and "max_minutes": the length in minutes, 0 = any ("about an hour" \
→ 45 and 75, "long" → 30 and 0, "short" → 0 and 5). "kind": "detailed" (long \
sentences and long turns — people explain at length), "lively" (a quick back-and-forth \
of short turns), "interview" (one asks, the other answers), "monologue" (one voice \
speaks nearly all the time), "group" (three or more people talk — only when no people \
are named: three named people together are a group anyway), otherwise "any". \
"sentences": "longer" when detailed, in-depth or elaborate talk or long sentences are \
wanted, "shorter" for short ones, otherwise "any" — with people named, longer or \
shorter than usual for them; otherwise than usual in the library. With people named, \
"detailed talk" is sentences "longer" and kind "any": a kind is measured against the \
whole library, and people who always talk quickly would have no detailed conversation."""

PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "said_words": {"type": "array", "items": {"type": "string"}},
        "said_meaning": {"type": "string"},
        "seen": {"type": "string"},
        "heard": {"type": "string"},
        "people": {"type": "array", "items": {"type": "string"}},
        "file_hints": {"type": "array", "items": {"type": "string"}},
        "media": {"type": "string", "enum": ["video", "audio", "any"]},
        "language": {"type": "string"},
        "asks_for": {
            "type": "string",
            "enum": ["moments", "conversations", "duplicates", "contained"],
        },
        # Strict JSON schema: a nested object needs every property required
        # and no others — "any" and 0 stand for a field the user didn't set
        "conversation": {
            "type": "object",
            "properties": {
                "min_minutes": {"type": "number"},
                "max_minutes": {"type": "number"},
                "kind": {
                    "type": "string",
                    "enum": ["any", "detailed", "lively", "interview", "monologue", "group"],
                },
                "sentences": {"type": "string", "enum": ["any", "longer", "shorter"]},
            },
            "required": ["min_minutes", "max_minutes", "kind", "sentences"],
            "additionalProperties": False,
        },
    },
    "required": [
        "summary",
        "said_words",
        "said_meaning",
        "seen",
        "heard",
        "people",
        "file_hints",
        "media",
        "language",
        "asks_for",
        "conversation",
    ],
    "additionalProperties": False,
}

#: What a plan's ``conversation`` holds when the description asks for no
#: conversations (or the model left it out)
NO_CONVERSATION = {"min_minutes": 0, "max_minutes": 0, "kind": "any", "sentences": "any"}
CONVERSATION_KINDS = ("any", "detailed", "lively", "interview", "monologue", "group")
#: No conversation is longer (about 70 days) — a larger number is a slip
CONVERSATION_MAX_MINUTES = 100_000

#: Sound labels that only say that someone talks — dropped
_SPEECH = {
    "talk",
    "talking",
    "talks",
    "speech",
    "speaking",
    "speak",
    "speaks",
    "voice",
    "voices",
    "conversation",
    "conversations",
    "dialogue",
    "dialog",
    "narration",
    "narrating",
    "chatter",
    "chatting",
    "chat",
    "discussion",
    "discussing",
    "someone",
    "somebody",
    "people",
    "person",
    "man",
    "woman",
    "men",
    "women",
    "a",
    "an",
    "the",
    "and",
}
#: Words that say what kind of recording it is (English and German). The
#: German words are data, matched against German descriptions.
_VIDEO_WORDS = re.compile(
    r"\b(vid[eé]o\w*|clips?|films?|filmchen|movies?|streams?|vods?|"
    r"livestreams?|footage)\b",
    re.I,
)
_AUDIO_WORDS = re.compile(
    r"\b(audio|voice ?message|voice ?note|sprachnachricht\w*|"
    r"sprachmemo\w*|podcast|song|lied|musik|music|mp3|hörspiel\w*)\b",
    re.I,
)

_people_lock = threading.Lock()
_people_cache: dict = {"at": -1e9, "names": []}
PEOPLE_TTL_SEC = 120.0


def library_people() -> list[str]:
    """The names of the people in the voice library — read at most every two
    minutes; [] when it can't be read."""
    now = time.monotonic()
    with _people_lock:
        if now - _people_cache["at"] < PEOPLE_TTL_SEC:
            return _people_cache["names"]
    names: list[str] = []
    try:
        from whisper_tm.database.speaker import speaker_db

        names = sorted({str(p["name"]).strip() for p in speaker_db().people() if p.get("name")})
    except Exception as e:  # noqa: BLE001 — planned without them then
        logger.debug("Find: the voice library's names not readable: %s", e)
    with _people_lock:
        _people_cache.update(at=now, names=names)
    return names


def conversation_names() -> dict[str, str]:
    """Every name a conversation can be found by, lowercase → as written:
    the voice library's and the ones typed in transcripts — the index the
    conversation queries look people up in (rag/speakers.py, held for
    minutes); {} when it can't be read."""
    try:
        from whisper_tm.rag import speakers

        shown = speakers.index().get("shown") or {}
    except Exception as e:  # noqa: BLE001 — the voice library's names alone then
        logger.debug("Find: the names typed in transcripts not readable: %s", e)
        return {}
    return {str(n).lower(): str(n) for n in shown.values() if str(n).strip()}


def _clean_list(v, n: int, *, lower: bool = False) -> list[str]:
    out = []
    for x in v if isinstance(v, list) else []:
        s = " ".join(str(x).split())[:80]
        if lower:
            s = s.lower()
        if len(s) >= 2 and s not in out:
            out.append(s)
    return out[:n]


def parse_plan(text: str, query: str) -> dict | None:
    """The language model's answer as a plan — ``None`` when it isn't one."""
    try:
        m = re.search(r"\{.*\}", text or "", re.S)
        raw = json.loads(m.group(0) if m else text)
    except (ValueError, AttributeError, TypeError):
        return None
    if not isinstance(raw, dict):
        return None
    media = str(raw.get("media") or "any").lower()
    asks = str(raw.get("asks_for") or "moments").lower()
    asked = ("moments", "conversations", "duplicates", "contained")
    plan = {
        "summary": " ".join(str(raw.get("summary") or query).split())[:300],
        "said_words": _clean_list(raw.get("said_words"), 3, lower=True),
        "said_meaning": " ".join(str(raw.get("said_meaning") or "").split())[:400],
        "seen": " ".join(str(raw.get("seen") or "").split())[:200],
        "heard": " ".join(str(raw.get("heard") or "").split())[:60],
        "people": _clean_list(raw.get("people"), 4),
        "file_hints": _clean_list(raw.get("file_hints"), 4),
        "media": media if media in ("video", "audio", "any") else "any",
        "media_stated": False,
        "language": " ".join(str(raw.get("language") or "").split())[:30],
        "asks_for": asks if asks in asked else "moments",
        "conversation": _conversation(raw.get("conversation")),
        "notes": [],
    }
    # The duplicates and the conversations need no search (the run ends
    # with the plan, agent.py)
    if plan["asks_for"] != "moments":
        return plan
    if not any(
        plan[k] for k in ("said_words", "said_meaning", "seen", "heard", "file_hints", "people")
    ):
        return None
    return plan


def fallback_plan(query: str) -> dict:
    """Without a language model: the description asks by meaning and by
    picture, quoted words word by word, known names as people. Nothing
    quoted: the description's own words — on a fresh install without the
    semantic and the Images & sounds index, they are the only search that
    runs at all."""
    q = " ".join(query.split())
    quoted = [m.lower() for m in re.findall(r'"([^"]{2,80})"', q)][:3] or [q[:80]]
    people = []
    try:
        from whisper_tm.rag import retrieve

        names = retrieve.known_names()
        people = [w for w in re.findall(r"\w+", q) if w.lower() in names][:4]
    except Exception:  # noqa: BLE001 — no voice library, no people
        pass
    hints = ["-WA"] if re.search(r"whats\s*app|\bWA\b", q, re.I) else []
    # Always moments: without the language model, nothing tells a talk
    # with Alice from a moment Alice is in
    return {
        "summary": q[:300],
        "said_words": quoted,
        "said_meaning": q[:400],
        "seen": q[:200],
        "heard": "",
        "people": people,
        "file_hints": hints,
        "media": "any",
        "media_stated": False,
        "language": "",
        "asks_for": "moments",
        "conversation": dict(NO_CONVERSATION),
        "notes": [],
    }


def _conversation(v) -> dict:
    """The model's ``conversation`` object checked field by field — a bad
    or missing field is "any", and the plan is not lost over it."""
    v = v if isinstance(v, dict) else {}
    out = dict(NO_CONVERSATION)
    for key in ("min_minutes", "max_minutes"):
        try:
            x = float(v.get(key) or 0)
        except (TypeError, ValueError):
            x = 0.0
        out[key] = round(x, 1) if math.isfinite(x) and 0 < x <= CONVERSATION_MAX_MINUTES else 0
    if out["min_minutes"] and out["max_minutes"] and out["min_minutes"] > out["max_minutes"]:
        out["min_minutes"], out["max_minutes"] = out["max_minutes"], out["min_minutes"]
    kind = str(v.get("kind") or "any").strip().lower()
    out["kind"] = kind if kind in CONVERSATION_KINDS else "any"
    sentences = str(v.get("sentences") or "any").strip().lower()
    out["sentences"] = sentences if sentences in ("longer", "shorter") else "any"
    return out


def conversation_filters(plan: dict) -> dict:
    """A plan that asks for conversations as the Conversations tab's filters
    — the parameters of ``/api/conversations``, only what the plan sets. The
    run keeps them (agent.py), the page hands them to the tab (find.js)."""
    conv = _conversation(plan.get("conversation"))
    out: dict = {}
    people = [p for p in plan.get("people") or [] if p]
    if people:
        out["people"] = people
        out["match"] = "all"
    for key, api in (("min_minutes", "min_sec"), ("max_minutes", "max_sec")):
        if conv[key]:
            out[api] = round(conv[key] * 60)
    kind, sentences = conv["kind"], conv["sentences"]
    if people and kind == "detailed":
        # A kind is measured against the whole library: people who always
        # talk quickly have no detailed conversation there. Their own
        # measure is the sentence filter — longer than usual for them.
        kind, sentences = "any", "longer" if sentences == "any" else sentences
    if kind != "any":
        out["kind"] = kind
    if kind == "monologue":
        # One voice alone is a monologue too — the tab starts from two
        # speakers, 0 is "no minimum"
        out["speakers_min"] = 0
    if sentences != "any":
        out["sentences"] = sentences
    if plan.get("file_hints"):
        out["q"] = plan["file_hints"][0]
    return out


def ground(plan: dict, query: str, people: list[str] | None = None) -> dict:
    """The plan checked against the library and the description (module
    doc). Adds what changed to ``plan["notes"]``."""
    plan = dict(plan)
    notes = list(plan.get("notes") or [])
    known = people if people is not None else library_people()
    by_lower = {n.lower(): n for n in known}
    first_words: dict[str, str] = {}
    for n in known:
        first_words.setdefault(n.split()[0].lower(), n)
    # The conversations look people up by every name they know — also the
    # ones typed in transcripts (conversations/query.py)
    talk = plan.get("asks_for") == "conversations"
    typed: dict[str, str] | None = None
    kept: list[str] = []
    for name in plan.get("people") or []:
        hit = by_lower.get(name.lower())
        if not hit and talk:
            typed = conversation_names() if typed is None else typed
            hit = typed.get(name.lower())
        if not hit:
            # "Alice Dances" → "Alice": a word of it that is a library name
            hit = next(
                (
                    by_lower.get(w.lower()) or first_words.get(w.lower())
                    for w in re.findall(r"\w+", name)
                    if by_lower.get(w.lower()) or first_words.get(w.lower())
                ),
                None,
            )
            if hit:
                notes.append(f"person “{name}” read as “{hit}” (the voice library's name)")
            elif talk:
                # Dropped, the tab would list the others' conversations as
                # if they were the ones asked for; kept, it says the name
                # is unknown
                hit = name
                if known or typed:
                    notes.append(
                        f"“{name}” is neither in the voice library nor typed in a "
                        "transcript — kept: the Conversations tab says it is unknown"
                    )
            elif known:
                notes.append(f"“{name}” is not in the voice library")
        if hit and hit not in kept:
            kept.append(hit)
        elif not hit and not known:
            kept.append(name)  # no library to check against
    plan["people"] = kept[:4]
    heard = plan.get("heard") or ""
    if heard and set(re.findall(r"[a-z]+", heard.lower())) <= _SPEECH:
        notes.append(f"sound “{heard}” left out — speech is searched in the transcripts")
        plan["heard"] = ""
    said = {"video": _VIDEO_WORDS, "audio": _AUDIO_WORDS}.get(plan.get("media"))
    plan["media_stated"] = bool(said and said.search(query))
    # (the kind of recording matters to the moment searches only)
    if (
        plan.get("media") != "any"
        and not plan["media_stated"]
        and plan.get("asks_for", "moments") == "moments"
    ):
        notes.append(f"“{plan['media']}” only guessed — preferred, not required")
    plan["notes"] = notes
    return plan


def make_plan(query: str) -> tuple[dict, str]:
    """``(plan, source)`` — ``source`` "llm" or "fallback: why"; the plan
    grounded either way."""
    from whisper_tm.core import llm_client

    people = library_people()
    system = PLAN_SYSTEM
    if people:
        system += "\n\nKnown people (the voice library): " + ", ".join(people[:300])
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Description: {query}"},
    ]
    try:
        # (800: room for the conversation object on top of a full plan)
        text, _usage = llm_client.chat(
            messages,
            response_schema=PLAN_SCHEMA,
            schema_name="find_plan",
            max_tokens=800,
            temperature=0.0,
            thinking=False,
            template_kwargs={"enable_thinking": False},
            task="chat",
            timeout=LLM_TIMEOUT_SEC,
            retries=1,
        )
    except llm_client.LlmNotConfigured:
        return ground(fallback_plan(query), query, people), "fallback: no language model set up"
    except llm_client.LlmError as e:
        logger.warning("Find: no plan from the language model (%s)", e)
        return (
            ground(fallback_plan(query), query, people),
            f"fallback: the language model did not answer ({str(e)[:120]})",
        )
    plan = parse_plan(text, query)
    if plan is None:
        logger.warning("Find: plan not usable: %r", (text or "")[:300])
        return ground(fallback_plan(query), query, people), "fallback: the plan was not usable"
    return ground(plan, query, people), "llm"
