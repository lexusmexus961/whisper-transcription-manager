"""Describing stills with the language model — a job of the LLM queue.

Measured on 12 queries × the best 20 stills, judged by eye:
ranked by the still's own vector, the top 10 held 62 % matches (nDCG@10
0.67); ranked by a one-sentence description the vision model wrote of each
still, embedded as text, 79 % (nDCG 0.91) — nearly what checking every
still for every query gives (82 %, 0.96), but written ONCE per moment
(~5.4 s with Qwen3.6-35B-A3B) and good for every query after. So
the descriptions become a kind of their own (``description``) in the index,
and the tab lists them next to the stills.

The run goes through the LLM queue (core/llm_queue.py): the language model
has one slot, which summaries, analyses and deep searches use too — the
queue takes turns, shows the progress, and stops it.
"""

from __future__ import annotations

import logging
import time

logger = logging.getLogger(__name__)

#: Descriptions embedded per request to the media service
EMBED_BATCH = 16


def enqueue(limit: int, folders: list[str] | None = None) -> int:
    """Put a describing run into the LLM queue — returns its entry id."""
    from whisper_tm.core.llm_queue import TYPE_MEDIA_DESCRIBE, llm_queue

    params = {"limit": int(limit), "folders": list(folders or [])}
    return llm_queue.enqueue(TYPE_MEDIA_DESCRIBE, "Images & sounds", params)


def run(entry: dict, queue) -> dict:
    """The LLM queue's handler: describe up to ``limit`` stills without a
    description, embed each sentence, store both."""
    from whisper_tm.config import config
    from whisper_tm.core.media_server import SERVER
    from whisper_tm.media import llm
    from whisper_tm.media.service import unb64
    from whisper_tm.media.store import get_store

    params = entry.get("params") or {}
    if isinstance(params, str):
        import json

        params = json.loads(params or "{}")
    limit = max(1, int(params.get("limit") or 500))
    from whisper_tm.core import paths

    prefixes = [(paths.canon(f) or f).rstrip("/") + "/" for f in params.get("folders") or []]
    store = get_store()
    todo = store.undescribed(limit, prefixes)
    if not todo:
        return {"described": 0, "message": "Every still is described already"}
    model = (
        config.get("media_search", "llm_model", default="")
        or config.get("llm", "model", default="")
        or ""
    )
    done, failed, pending = 0, 0, []
    t0 = time.time()

    def flush():
        if not pending:
            return
        res = SERVER.call(
            "/embed", {"texts": [t for _, t in pending], "prompt": "Document"}, timeout=300
        )
        store.add_items(
            [(mid, "description", unb64(v)) for (mid, _), v in zip(pending, res["vectors"])]
        )
        pending.clear()

    width = int(config.get("media_search", "thumb_width", default=480) or 480)
    for k, m in enumerate(todo):
        if queue.abort_requested(entry["id"]):
            flush()
            raise InterruptedError("Cancelled")
        if not m.get("text") and not _still(m, width):
            # The recording can't be read: no still to describe — skipped,
            # not counted as a failure of the language model
            logger.info("Describe moment %s: no still of %s", m["id"], m["path"])
            continue
        try:
            # Written before, not embedded then (the service was away): only embed it
            text = m.get("text") or llm.describe(m["thumb"])
        except Exception as e:  # noqa: BLE001 — one still less; three in a row stop it
            failed += 1
            logger.info("Describe moment %s: %s", m["id"], e)
            if failed >= 3 and not done:
                raise RuntimeError(f"The language model did not describe: {e}") from e
            continue
        if not text:
            continue
        if not m.get("text"):
            store.set_description(m["id"], text, model)
        pending.append((m["id"], text))
        done += 1
        if len(pending) >= EMBED_BATCH:
            flush()
        per = (time.time() - t0) / (k + 1)
        queue.progress(
            entry["id"],
            int(100 * (k + 1) / len(todo)),
            f"{k + 1}/{len(todo)} stills described · ~{per:.1f} s each",
        )
    flush()
    return {"described": done, "failed": failed, "seconds": round(time.time() - t0)}


def _still(m: dict, width: int) -> bool:
    """The moment's still is there — cut now when it isn't: past the thumbnail
    cap the indexer stores its path without writing it, and the model then
    failed on it run after run."""
    import os

    from whisper_tm.media import cuts

    if os.path.isfile(m["thumb"]):
        return True
    return (
        cuts.cached(m["thumb"], lambda out: cuts.cut_still(m["path"], m["t"], out, width=width))
        is not None
    )
