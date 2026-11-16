"""Find's judge — the language model checks the best moments one by one.

It reads the description, what is said around the moment (an automatic
transcript), which searches found it — and the FACTS the program knows:
whether the file name has the remembered piece, the kind of recording,
whether a named person speaks in it, the recording's length, its copies.
Told as facts, so it doesn't make them up (otherwise a stream can get a 9
for being "a WhatsApp file (WA in the title)"). With a still of the moment
when it sees pictures.

Its verdict is a digit 0–9; the probabilities of the digits (logprobs, from
llama.cpp and OpenAI-compatible servers) make it continuous — 7.8 instead of
another 8, fewer ties. When something the user SAID fails (the file name
doesn't have the piece, it isn't the kind of recording named), the verdict
is capped at :data:`CAP_FAILED` — "possible", never "clearly it".
"""

from __future__ import annotations

import logging
import math
import os
import re

logger = logging.getLogger(__name__)

#: The best fused moments the language model judges
JUDGE_TOP = 12
#: Seconds of transcript the judge reads around a moment
SAID_WINDOW_SEC = 30.0
#: A verdict's ceiling when a stated constraint fails
CAP_FAILED = 6.0
#: Below this a judged moment is a reject: it goes after the unchecked ones
REJECT_BELOW = 4.0
LLM_TIMEOUT_SEC = 120

JUDGE_SYSTEM = """You check whether a moment of a recording is the one the user is \
looking for. You get the user's description, what was said around the moment (an \
automatic transcript — errors are possible), which searches found it, FACTS the \
program checked and, when there is one, a still of the moment. The FACT lines are \
true — never contradict them, and claim nothing about the file name, the people or \
the kind of recording beyond them. Answer in ONE line: a digit from 0 to 9 (0 = \
clearly not it, 5 = possible, 9 = clearly it), then " — " and a short reason in \
{language}."""

FOUND_BY = {
    "words": "said word for word",
    "meaning": "by meaning",
    "seen": "by the picture",
    "heard": "by the sound",
    "files": "inside the files named like that",
    "people": "in what the named people say",
}


def clock(sec: float) -> str:
    sec = int(max(0, sec or 0))
    h, m, s = sec // 3600, sec % 3600 // 60, sec % 60
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def said_around(db, key: str, t: float, window: float = SAID_WINDOW_SEC) -> str:
    """What is said within ``window`` seconds of ``t`` (labels kept)."""
    with db.connection() as conn:
        row = conn.execute(
            "SELECT id FROM transcripts WHERE abs_path=? AND complete=1 ORDER BY id DESC LIMIT 1",
            (key,),
        ).fetchone()
        if not row:
            return ""
        segs = conn.execute(
            "SELECT start_sec, speaker, text FROM segments WHERE transcript_id=? "
            "AND end_sec >= ? AND start_sec <= ? ORDER BY start_sec LIMIT 60",
            (row[0], t - window, t + window),
        ).fetchall()
    out, last = [], None
    for _start, sp, text in segs:
        txt = (text or "").strip()
        if txt:
            out.append((f"[{sp}] " if sp and sp != last else "") + txt)
            last = sp
    return " ".join(out)[:1500]


def hint_in(hint: str, path: str) -> bool:
    """Does a path have a remembered file-name piece — as the search matches
    it: as written when it has capitals, else in any case."""
    path = path or ""
    return hint in path if hint != hint.lower() else hint.lower() in path.lower()


def facts_for(
    m: dict, plan: dict, *, people_paths=None, lengths=None
) -> tuple[list[str], list[str]]:
    """``(FACT lines, what the user said that fails)`` for a moment."""
    files = [m["key"]] + [d["key"] for d in m.get("dups") or [] if d.get("key")]
    lines, failed = [], []
    for h in plan.get("file_hints") or []:
        hit = next((f for f in files if hint_in(h, f)), None)
        if hit:
            lines.append(f"FACT: the file name contains “{h}”: yes ({os.path.basename(hit)})")
        else:
            lines.append(
                f"FACT: the file name contains “{h}”: no — not in any of its "
                f"{len(files)} file{'s' if len(files) > 1 else ''}"
            )
            failed.append(f"file name without “{h}”")
    kind = "a video" if m.get("video") else "audio only (no picture)"
    lines.append(f"FACT: the recording is {kind}")
    want = plan.get("media") or "any"
    if want != "any" and plan.get("media_stated") and (want == "video") != bool(m.get("video")):
        failed.append(f"not a {'video' if want == 'video' else 'sound recording'}")
    for name in plan.get("people") or []:
        paths = (people_paths or {}).get(name) or set()
        if any(f in paths for f in files):
            lines.append(f"FACT: {name} speaks in this recording (recognized by voice)")
        else:
            lines.append(
                f"FACT: {name}'s voice is not recognized in this recording "
                "(the voice library doesn't know every recording)"
            )
    length = (lengths or {}).get(m["key"])
    if length:
        lines.append(f"FACT: the recording is {clock(length)} long")
    if m.get("dups"):
        lines.append(f"FACT: the same moment is in {len(m['dups'])} other file(s) too (copies)")
    return lines, failed


def judge_text(query: str, plan: dict, m: dict, said: str, facts: list[str]) -> str:
    found = []
    for n, e in (m.get("engines") or {}).items():
        scope = e.get("scope")
        found.append(
            f"{FOUND_BY.get(n, n)} (rank {e['rank'] + 1}"
            + (f" {FOUND_BY[scope]}" if scope in FOUND_BY else "")
            + ")"
        )
    return (
        f'The user looks for: "{query}"\n\n'
        f"Recording: {m.get('rel') or m.get('name')}, moment {clock(m['t'])}\n"
        f"Found {'; '.join(found) or 'by the searches'}.\n" + "\n".join(facts) + "\n"
        f"Said around the moment: {said or '(nothing transcribed)'}\n\n"
        "Is this the moment? One line: digit — reason."
    )


def parse_verdict(text: str, first_logprobs=None) -> tuple[float | None, str]:
    """``(verdict 0–9, reason)``. With the first token's logprobs the verdict
    is the digits' expected value; without, the first digit of the line."""
    value = None
    if first_logprobs:
        ps = {}
        for tok, lp in first_logprobs:
            t = str(tok).strip()
            if len(t) == 1 and t.isdigit():
                ps[int(t)] = ps.get(int(t), 0.0) + math.exp(lp)
        if ps and sum(ps.values()) > 0.2:
            value = round(sum(d * p for d, p in ps.items()) / sum(ps.values()), 2)
    m = re.search(r"\b([0-9])\b", text or "")
    if not m:
        return value, (text or "").strip()[:200]
    reason = re.sub(r"^\s*[0-9]\s*[—–:\-.)]*\s*", "", (text or "").strip()[m.start() :]).strip()
    return (value if value is not None else float(m.group(1))), reason[:240]


def capped(verdict: float | None, failed: list[str]) -> float | None:
    if verdict is None or not failed:
        return verdict
    return min(verdict, CAP_FAILED)


def still_of(m: dict) -> str | None:
    """A still of the moment: Images & sounds' when it has one near, else a
    frame cut from the recording."""
    try:
        from whisper_tm.media.store import get_store

        store = get_store()
        if m.get("moment"):
            row = store.moment(int(m["moment"]))
            if row and row.get("thumb") and os.path.isfile(row["thumb"]):
                return row["thumb"]
        f = store.file(m["key"])
        if f:
            near = min(
                (x for x in store.file_moments(f["id"]) if x.get("thumb")),
                key=lambda x: abs(x["t"] - m["t"]),
                default=None,
            )
            if near and abs(near["t"] - m["t"]) <= 15 and os.path.isfile(near["thumb"]):
                return near["thumb"]
    except Exception:  # noqa: BLE001 — a frame then
        pass
    try:
        from whisper_tm.find.agent import cut_frame

        f = cut_frame(m["key"], m["t"])
        return str(f) if f else None
    except Exception:  # noqa: BLE001 — judged without a picture then
        return None


def judge(
    query: str, plan: dict, m: dict, said: str, facts: list[str], vision: bool
) -> tuple[float | None, str]:
    from whisper_tm.core import llm_client

    text = judge_text(query, plan, m, said, facts)
    still = still_of(m) if (vision and m.get("video")) else None
    system = JUDGE_SYSTEM.format(language=plan.get("language") or "the language of the description")

    def ask(content):
        answer, usage = llm_client.chat(
            [{"role": "system", "content": system}, {"role": "user", "content": content}],
            max_tokens=80,
            temperature=0.0,
            thinking=False,
            template_kwargs={"enable_thinking": False},
            task="chat",
            timeout=LLM_TIMEOUT_SEC,
            retries=1,
            logprobs=10,
        )
        return parse_verdict(answer, usage.get("first_logprobs"))

    if still:
        from whisper_tm.media.llm import image_part

        try:
            note = "\n(The picture is a still of the moment.)"
            return ask([image_part(still), {"type": "text", "text": text + note}])
        except llm_client.LlmError as e:
            # a model that doesn't take pictures: the text alone
            logger.info("Find: judging with the still failed (%s) — without it", e)
    return ask(text)


#: Another moment of a recording already judged counts this much of its
#: score when the next one to judge is picked — little when a moment of it
#: may be it (the recording is represented), more when it was rejected (its
#: next moment may well be the one: a long stream's early moment rejected,
#: a much later one — the phrase said word for word — the moment)
AFTER_GOOD = 0.3
AFTER_REJECT = 0.8


def next_to_judge(best: list[dict], done: set) -> dict | None:
    """The next moment the language model checks: the best score, a
    recording's further moments discounted by what its judged ones got
    (otherwise one recording's three moments can take five of twelve
    checks). ``done``: ids of the moments judged (or tried) so far."""
    verdicts: dict[str, list] = {}
    for m in best:
        if id(m) in done:
            verdicts.setdefault(m.get("rec") or m["key"], []).append(m.get("verdict"))
    pick, top = None, -1.0
    for m in best:
        if id(m) in done:
            continue
        before = verdicts.get(m.get("rec") or m["key"])
        factor = 1.0
        if before:
            good = any(v is not None and v >= REJECT_BELOW for v in before)
            factor = AFTER_GOOD if good else AFTER_REJECT
        value = float(m.get("score") or 0.0) * factor
        if value > top:
            pick, top = m, value
    return pick


def ordered(best: list[dict]) -> list[dict]:
    """Judged moments that may be it first (by verdict), then the ones not
    judged (by score), the rejects last — a 0 is no better than unchecked."""

    def key(m):
        v = m.get("verdict")
        if v is None:
            return (1, 0.0, -m["score"])
        if v >= REJECT_BELOW:
            return (0, -v, -m["score"])
        return (2, -v, -m["score"])

    return sorted(best, key=key)
