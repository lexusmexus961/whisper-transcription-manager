"""
Best moments: the language model picks three short, self-contained
moments of a recording for a clip; each becomes a cut in the recording's
Clips tab (marked AI) to keep or drop.

The transcript goes in as numbered lines; the model answers with the
first and last line of each moment and a short title. A long recording's
lines are joined in pairs until they fit the model's context, so the
whole recording is always seen. A moment is kept between 10 and 90
seconds (a shorter one takes the lines after it in); overlapping ones
are dropped.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

WANT = 3
MIN_SEC, MAX_SEC = 10.0, 90.0
SCHEMA = {
    "type": "object",
    "properties": {
        "moments": {
            "type": "array",
            "maxItems": 5,
            "items": {
                "type": "object",
                "properties": {
                    "first": {"type": "integer"},
                    "last": {"type": "integer"},
                    "title": {"type": "string"},
                },
                "required": ["first", "last", "title"],
            },
        }
    },
    "required": ["moments"],
}
PROMPT = (
    "You pick moments of a recording for short clips. Below is its transcript, one numbered "
    "line each "
    "([number] time speaker: text). Choose the {n} best moments: each self-contained (it makes "
    "sense "
    "without the rest), interesting or funny or important, usually several lines and between 10 "
    "and 60 "
    "seconds long. They must "
    'not overlap. Answer with JSON only: {{"moments": [{{"first": <first line number>, "last": '
    '<last line number>, "title": <a few words saying what happens, in the transcript\'s '
    "language>}}]}}."
)


def _clock(sec: float) -> str:
    sec = int(sec)
    return (
        f"{sec // 60}:{sec % 60:02d}"
        if sec < 3600
        else f"{sec // 3600}:{sec // 60 % 60:02d}:{sec % 60:02d}"
    )


def lines_of(segments: list[dict], budget_chars: int) -> list[dict]:
    """[{start, end, text}] — the segments, joined in pairs while there are
    more lines than fit ``budget_chars`` at ~120 characters each; then each
    line is cut to its share of the budget."""
    rows = [
        {
            "start": float(s.get("start") or 0),
            "end": float(s.get("end") or 0),
            "text": f"{s.get('speaker') or ''}: {(s.get('text') or '').strip()}".strip(": "),
        }
        for s in segments
        if (s.get("text") or "").strip()
    ]

    def size(rs):
        return sum(len(r["text"]) + 14 for r in rs)

    while len(rows) > 1 and size(rows) > budget_chars and len(rows) * 120 > budget_chars:
        joined = []
        for i in range(0, len(rows), 2):
            pair = rows[i : i + 2]
            joined.append(
                {
                    "start": pair[0]["start"],
                    "end": pair[-1]["end"],
                    "text": " / ".join(p["text"] for p in pair),
                }
            )
        rows = joined
    if size(rows) > budget_chars:
        share = max(20, budget_chars // max(1, len(rows)) - 14)
        rows = [
            {**r, "text": r["text"] if len(r["text"]) <= share else r["text"][: share - 1] + "…"}
            for r in rows
        ]
    return rows


def moments_from(answer: dict, rows: list[dict], length: float = 0.0) -> list[dict]:
    """The model's answer as cuts: [{start, end, title, ai}], checked."""
    out: list[dict] = []
    for m in (answer or {}).get("moments") or []:
        try:
            a, b = int(m.get("first")), int(m.get("last"))
        except (TypeError, ValueError, AttributeError):
            continue
        a, b = min(a, b), max(a, b)
        if a < 0 or b >= len(rows):
            # Numbered from 1 by some models
            a, b = a - 1, b - 1
            if a < 0 or b >= len(rows):
                continue
        # Too short a moment takes the lines after it in (models like to
        # name a single line)
        while b + 1 < len(rows) and rows[b]["end"] - rows[a]["start"] < MIN_SEC:
            b += 1
        start, end = rows[a]["start"], rows[b]["end"]
        if end - start > MAX_SEC:
            end = start + MAX_SEC
        if length:
            end = min(end, length)
        if end - start < 1:
            continue
        if any(start < o["end"] and end > o["start"] for o in out):
            continue
        out.append(
            {
                "start": round(start, 3),
                "end": round(end, 3),
                "title": str(m.get("title") or "")[:120].strip(),
                "ai": True,
            }
        )
        if len(out) >= WANT:
            break
    return sorted(out, key=lambda c: c["start"])


def run(entry: dict, queue) -> dict:
    from whisper_tm.core import cuts as cut_store
    from whisper_tm.core import llm_client
    from whisper_tm.core.summarize import named_speakers
    from whisper_tm.extensions import db

    p = entry.get("params") or {}
    abs_path = p.get("abs") or ""
    if not abs_path:
        raise RuntimeError("No recording given")
    data = db.get_transcript(abs_path) if db else None
    if not data:
        raise RuntimeError("Transcript not found")
    segments = named_speakers(data.get("segments") or [], data.get("speaker_names"))
    try:
        ctx = int(llm_client.effective_context("summary") or 8192)
    except Exception:  # noqa — a guess is enough
        ctx = 8192
    rows = lines_of(segments, max(4000, int(ctx * 3 * 0.6)))
    if not rows:
        raise RuntimeError("The transcript is empty — nothing to choose from")
    queue.progress(entry["id"], 10, f"Reading {len(rows)} lines…")
    listing = "\n".join(f"[{i}] {_clock(r['start'])} {r['text']}" for i, r in enumerate(rows))
    if queue.abort_requested(entry["id"]):
        raise InterruptedError()
    answer = llm_client.chat_json(
        [
            {"role": "system", "content": PROMPT.format(n=WANT)},
            {"role": "user", "content": listing},
        ],
        schema=SCHEMA,
        schema_name="moments",
        temperature=0.3,
        max_tokens=800,
        task="summary",
    )
    length = float(segments[-1].get("end") or 0) if segments else 0.0
    found = moments_from(answer, rows, length)
    if not found:
        raise RuntimeError(
            "The language model suggested no usable moments — try again, or mark the parts yourself"
        )
    cut_store.add(abs_path, found)
    queue.progress(entry["id"], 99, f"{len(found)} moments")
    return {"moments": found}
