"""Embedding API: control server, check connection, measure throughput.

Why separate endpoints and not part of ``api/rag.py``: The embedding model
is usable independently of the vector store and is also configured
independently. Anyone who enters an external provider (OpenAI, a server on
the network) should be able to check it without RAG being enabled at all.

**Responsibility for the server lies with the user.** Whisper Transcription Manager can
start a local llama.cpp server if needed, when binary and model are
there — but it does not *build* one. The UI provides instructions for this
and a ready-made prompt to pass on to an AI.
"""

from __future__ import annotations

import logging
import statistics
import time

from flask import Blueprint, jsonify, request

from whisper_tm.config import config
from whisper_tm.core import embed_server, llm_client
from whisper_tm.rag.vector_db import RagConfigError

logger = logging.getLogger(__name__)

embedding_bp = Blueprint("embedding", __name__)


def _cfg() -> dict:
    from whisper_tm.rag.service import embedding_config

    return embedding_config()


# ── State ───────────────────────────────────────────────────────────────
@embedding_bp.route("/api/embedding/status")
def embedding_status():
    """What is configured, and is it running?"""
    e = _cfg()
    available, hint = embed_server.available()
    off = {
        "local": e["local"],
        "endpoint": embed_server.endpoint() if e["local"] else e["endpoint"],
        "model": e["model"],
        # Only whether one is set — the key itself never leaves the server.
        "api_key_set": bool(e["api_key"]),
        "local_server": {
            "possible": available,
            "hint": hint,
            "binary": str(embed_server.binary() or ""),
            "model": str(embed_server.model_file() or ""),
            **embed_server.SERVER.state(),
        },
    }
    return jsonify(off)


@embedding_bp.route("/api/embedding/server/start", methods=["POST"])
def embedding_server_start():
    """Start the local server now (otherwise it happens on demand)."""
    if not _cfg()["local"]:
        return jsonify({"error": "Local operation is not enabled (Settings → AI models)"}), 400
    try:
        t = time.time()
        ep = embed_server.SERVER.ensure()
        return jsonify(
            {
                "ok": True,
                "endpoint": ep,
                "duration_sec": round(time.time() - t, 1),
                "state": embed_server.SERVER.state(),
            }
        )
    except Exception as e:  # noqa
        return jsonify({"error": str(e)[:400]}), 500


@embedding_bp.route("/api/embedding/server/stop", methods=["POST"])
def embedding_server_stop():
    """Stop the local server and free the VRAM."""
    terminated = embed_server.SERVER.stop_it()
    return jsonify({"ok": True, "finished": terminated, "state": embed_server.SERVER.state()})


# ── Check ────────────────────────────────────────────────────────────────
@embedding_bp.route("/api/embedding/test", methods=["POST"])
def embedding_test():
    """A test embedding — answers the only question that matters.

    Namely: does a vector come back, and how many dimensions does it have? A
    mere connection check doesn't say that; many endpoints happily answer
    ``/v1/models`` and respond to ``/v1/embeddings`` with 501, because the
    server was started without ``--embedding``.
    """
    data = request.json if isinstance(request.json, dict) else {}
    text = (data.get("text") or "A short English sentence for testing.").strip()
    # `endpoint`/`model` in the body allow checking BEFORE saving —
    # the UI thus tests what's in the form. Without both, what's
    # configured applies (and in local operation, the server's own instance).
    ep_raw = (data.get("endpoint") or "").strip()
    md_raw = (data.get("model") or "").strip()
    key_raw = (data.get("api_key") or "").strip()
    try:
        if ep_raw and not _cfg()["local"]:
            # The stored key only travels to the stored endpoint. Any other
            # address gets only a key sent along in this request — otherwise
            # anyone who can reach this page could have the key delivered to
            # a host of their choosing (core/llm_provider.py follows the
            # same rule).
            same = ep_raw.rstrip("/") == (_cfg()["endpoint"] or "").rstrip("/")
            key = key_raw or (_cfg()["api_key"] if same else "")

            def embed_it(texts):
                return llm_client.embed_texts(
                    texts,
                    endpoint=ep_raw,
                    model=md_raw or _cfg()["model"],
                    api_key=key,
                    timeout=min(int(_cfg()["timeout"]), 60),
                )
        else:
            from whisper_tm.rag.service import make_embed_fn

            embed_it = make_embed_fn()
        t = time.time()
        vecs = embed_it([text])
        duration = time.time() - t
    except RagConfigError as e:
        # Misconfiguration is not success — 400, as before. The status
        # separates "you haven't entered anything" from "the server isn't
        # responding", and the UI shows both differently.
        return jsonify({"ok": False, "error": str(e)[:400]}), 400
    except Exception as e:  # noqa
        return jsonify({"ok": False, "error": str(e)[:400]}), 502
    if not vecs or not vecs[0]:
        return jsonify({"ok": False, "error": "Response without vector"}), 502
    v = vecs[0]
    length = sum(x * x for x in v) ** 0.5
    return jsonify(
        {
            "ok": True,
            "dimensions": len(v),
            "duration_ms": round(duration * 1000, 1),
            # Is the output normalized? That decides whether cosine and
            # dot product are the same — good to know, but not an error:
            # the store normalizes on write anyway.
            "normalized": abs(length - 1.0) < 0.01,
            "length": round(length, 4),
            "endpoint": embed_server.endpoint() if _cfg()["local"] else _cfg()["endpoint"],
        }
    )


# ── Measure throughput ──────────────────────────────────────────────────────
@embedding_bp.route("/api/embedding/benchmark", methods=["POST"])
def embedding_benchmark():
    """How many chunks per second does this endpoint manage?

    Measured against REAL chunks from the app's own database, not
    made-up text: the cost depends on the length of the input, and
    Lorem ipsum has a different length distribution than real transcripts.

    Multiple batch sizes, because that is exactly the knob one sees
    from outside — and it behaves quite differently depending on the
    provider: a local server computes sequentially, an API over the
    network gains mostly from fewer round trips.
    """
    from whisper_tm.extensions import db

    data = request.json if isinstance(request.json, dict) else {}
    sizes = data.get("sizes") or [8, 32, 64]
    try:
        sizes = [max(1, min(int(g), 512)) for g in sizes][:5]
    except (TypeError, ValueError):
        return jsonify({"error": "sizes must be numbers"}), 400
    amount = max(sizes) * 2

    texts = _sample_chunks(db, amount)
    if not texts:
        return jsonify({"error": "No transcripts found for the measurement"}), 400

    try:
        from whisper_tm.rag.service import make_embed_fn

        embed_it = make_embed_fn()
    except Exception as e:  # noqa
        return jsonify({"error": str(e)[:300]}), 400

    # One warm-up run: the first call loads the model on a local server,
    # and that would not be a statement about throughput.
    t0 = time.time()
    try:
        embed_it(texts[:1])
    except Exception as e:  # noqa
        return jsonify({"error": f"Test embedding failed: {str(e)[:300]}"}), 502
    warm_up = round(time.time() - t0, 1)

    results = []
    for g in sizes:
        parts = [texts[i : i + g] for i in range(0, len(texts), g)]
        times, n = [], 0
        try:
            for part in parts:
                t = time.perf_counter()
                embed_it(part)
                times.append(time.perf_counter() - t)
                n += len(part)
        except Exception as e:  # noqa
            results.append({"batch": g, "error": str(e)[:200]})
            continue
        duration = sum(times)
        results.append(
            {
                "batch": g,
                "chunks": n,
                "seconds": round(duration, 2),
                "chunks_per_sec": round(n / duration, 1) if duration else 0,
                "ms_per_request": round(statistics.median(times) * 1000, 1),
            }
        )

    best = max((r.get("chunks_per_sec", 0) for r in results), default=0)
    open_chunks = _estimate_open_chunks(db)
    return jsonify(
        {
            "ok": True,
            "warmup_sec": warm_up,
            "chunk_length_avg": round(statistics.mean(len(t) for t in texts)),
            "results": results,
            "best_rate": best,
            "estimated_chunks": open_chunks,
            "estimated_hours": (
                round(open_chunks / best / 3600, 1) if best and open_chunks else None
            ),
        }
    )


def _sample_chunks(db, amount: int) -> list[str]:
    """Real chunks from the database — as many as needed."""
    if not db:
        return []
    from whisper_tm.rag.chunker import chunk_segments
    from whisper_tm.rag.service import get_rag_config

    cfg = get_rag_config()
    texts: list[str] = []
    try:
        candidates = db.get_rag_candidates()
    except Exception as e:  # noqa
        logger.debug("Benchmark candidates: %s", e)
        return []
    # From the back: the most recently added recordings are more
    # representative of what's about to be indexed than the library's oldest.
    for cand in reversed(candidates):
        try:
            segs = db.get_segments_for_rag(cand["id"])
        except Exception:  # noqa
            continue
        if not segs:
            continue
        texts += [
            ch.content
            for ch in chunk_segments(
                segs,
                max_chars=cfg.get("chunk_max_chars", 1200),
                min_chars=cfg.get("chunk_min_chars", 250),
                overlap_segments=cfg.get("chunk_overlap_segments", 1),
                include_speaker_labels=cfg.get("include_speaker_labels", True),
            )
        ]
        if len(texts) >= amount:
            break
    return texts[:amount]


def _estimate_open_chunks(db) -> int:
    """How many chunks are still outstanding? From the sample's ratio.

    An exact number would only come from chunking ALL transcripts — in a
    large library that would be a computation taking minutes for a display. The
    estimate is therefore derived from segments per chunk in the sample and
    is shown as an estimate.
    """
    if not db:
        return 0
    try:
        from whisper_tm.rag.service import make_rag_db

        indexed = make_rag_db().get_indexed_map()
    except Exception:  # noqa
        indexed = {}
    try:
        from whisper_tm.rag.indexer import _transcript_key

        candidates = db.get_rag_candidates()
    except Exception:  # noqa
        return 0
    open_seg = 0
    for c in candidates:
        if _transcript_key(c) in indexed:
            continue
        open_seg += int(c.get("seg_count") or 0)
    # A rough empirical ratio: about 0.18 chunks per segment. The
    # value depends on the chunk config; as an order of magnitude for a
    # progress display it's good enough.
    return int(open_seg * 0.183)


# ── The instructions for the user ──────────────────────────────────────────
@embedding_bp.route("/api/embedding/guide")
def embedding_guide():
    """A ready-made prompt to hand off to an AI.

    The server is deliberately NOT our component: we integrate an
    endpoint, we don't operate one. So this doesn't turn into a dead end,
    here is the text one can present to an AI — with this machine's
    values already inserted, so the answer fits instead of staying generic.
    """
    from whisper_tm.rag.service import get_rag_config

    hw = _hardware_brief()
    rag = get_rag_config()
    port = config.get("embedding", "local_port", default=8081)
    prompt = _PROMPT_TEMPLATE.format(
        hardware=hw,
        port=port,
        chunk_max=rag.get("chunk_max_chars", 1200),
        ubatch=config.get("embedding", "local_batch", default=2048),
    )
    return jsonify({"prompt": prompt, "hardware": hw})


def _hardware_brief() -> str:
    """One line about this machine — so the AI doesn't have to guess."""
    parts = []
    try:
        import whisper_tm.extensions as ext

        if ext.hw:
            gpu = getattr(ext.hw, "gpu_name", "") or ""
            vram = getattr(ext.hw, "vram_gb", 0) or 0
            if gpu:
                parts.append(f"GPU: {gpu}" + (f" ({vram:.0f} GB VRAM)" if vram else ""))
            cores = getattr(ext.hw, "cpu_cores", 0) or 0
            if cores:
                parts.append(f"CPU: {cores} cores")
            ram = getattr(ext.hw, "ram_gb", 0) or 0
            if ram:
                parts.append(f"RAM: {ram:.0f} GB")
    except Exception as e:  # noqa
        logger.debug("Hardware for the instructions: %s", e)
    return " · ".join(parts) or "unknown"


_PROMPT_TEMPLATE = """\
I want to set up an **embedding server** on my machine, which another \
application will talk to via an OpenAI-compatible HTTP interface. \
Please walk me through it step by step, ask me follow-up questions if \
something is unclear, and give me ready-made commands to copy.

**My machine:** {hardware}

**What the server needs to be able to do:**
- It responds to `POST http://127.0.0.1:{port}/v1/embeddings` in \
OpenAI format (the `input` field takes a list of texts, the response \
contains `data[].embedding`).
- It processes an entire batch of texts in ONE request. This matters: \
a server that needs a separate request per text is many times \
slower.
- The texts are transcript excerpts of up to {chunk_max} \
characters, so roughly 600 tokens.

**What to watch out for in particular (I've already learned this the hard way):**

1. **Don't forget `--embedding`.** Without this flag, \
llama.cpp responds to `/v1/embeddings` with HTTP 501, even though the server \
is running and `/health` responds correctly.

2. **Ubatch size ≥ longest input.** Embedding models are non-causal \
and need the whole sequence in ONE pass. If `--ubatch-size` is too \
small, the server rejects the request with "Context size has been exceeded" \
instead of splitting it up. For my texts, {ubatch} is a good value; larger \
gains nothing except memory use.

3. **Try NOT to fix `--n-gpu-layers` to a specific value.** A fixed value \
disables the automatic adjustment to the free graphics memory, and the \
server dies on startup with "failed to fit params to free device memory ... \
abort" as soon as the card is otherwise in use.

4. **The card may be shared.** Other models may also be \
working on the GPU. The embedding server must not take all the free \
memory — otherwise the other work fails. Tell me how to \
limit it, or fall back to the CPU if necessary.

**Please explain to me:**
- Which embedding model fits my hardware and why (quality \
vs. speed vs. memory), and where to download it.
- How to install and start the server, with the full command line.
- How to set it up as a service so it survives a restart — and \
whether that even makes sense if it sits idle most of the time.
- How to verify it actually works (a `curl` call that returns \
a vector, and how I can tell the response is usable).
- How I'd notice something is wrong, and what the most common causes \
are.

First ask me which operating system I have and whether I'd prefer \
llama.cpp, Ollama, vLLM or something else — and recommend \
one of them with reasoning."""
