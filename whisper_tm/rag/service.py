"""
Glue between config/embedding client and the RAG subsystem.

Builds the vector store and the embedding function from running configuration.
"""

from __future__ import annotations

import logging
import threading

from whisper_tm.config import config
from whisper_tm.core import llm_client

from .vector_db import RagConfigError, VectorDB

logger = logging.getLogger(__name__)

# Defaults if the config section is (still) missing
_RAG_DEFAULTS = {
    "enabled": False,
    "path": "",
    "chunk_max_chars": 1200,
    "chunk_min_chars": 250,
    "chunk_overlap_segments": 1,
    "chunk_on_speaker_change": False,
    "include_speaker_labels": True,
    "dimensions": 0,
    "fts_language": "english",
    "rrf_k": 60,
    "top_k": 20,
    "batch_size": 64,
}


def get_rag_config() -> dict:
    """rag config with filled defaults."""
    cfg = dict(_RAG_DEFAULTS)
    user = config.get("rag") or {}
    if isinstance(user, dict):
        cfg.update({k: v for k, v in user.items() if v is not None})
    return cfg


def vector_db_path() -> str:
    """Where the vector store is.

    Own file next to ``speaker.db`` and NOT in the main database.
    Measured it makes no speed difference (load 1.60s vs 1.61s, app's FTS5
    unchanged at 0.05ms) — but the main DB stays its size, and a search index
    rebuild can just delete the file without touching the source of truth.
    """
    from pathlib import Path

    set_flag = config.path_setting("rag", "path")
    if set_flag:
        return str(set_flag)
    return str(Path(config.db_path).parent / "rag.db")


_db_lock = threading.Lock()
_db_instance: VectorDB | None = None
_db_key: tuple | None = None


def make_rag_db() -> VectorDB:
    """Build the vector store from config — and keep it.

    **Why keeping is the real point.** ``VectorDB`` holds the vector matrix
    in memory after first access and discards it via ``generation`` as soon
    as inventory changes. This cache sits on the INSTANCE — but originally
    a new one was built on every single call here. Every route, every search,
    every coverage question started from zero.

    Measured on a large index (hundreds of thousands of chunks × 2,560
    dimensions):

        Load matrix              3.45s   ← on EVERY search
        Similarity + top-12      0.06s
        Embed query (CPU)        0.11s
        Same matrix again        0.0023s ← if instance stays

    Whole RAG search took 4.2s that way, 82% reloading ~2 GB from SQLite.
    The question "GPU or CPU for embedding" decided 3% by comparison.

    **Can be shared** because ``VectorDB`` holds no connection state:
    ``connect()`` opens its own ``sqlite3`` connection per call,
    and the matrix is protected by its own lock.

    Config changes (other path, other dimensions) trigger rebuild —
    the key is exactly this configuration.
    """
    global _db_instance, _db_key
    cfg = get_rag_config()
    key = (vector_db_path(), int(cfg.get("dimensions", 0) or 0), cfg.get("fts_language", "english"))
    with _db_lock:
        if _db_instance is None or _db_key != key:
            _db_instance = VectorDB(key[0], dimensions=key[1], fts_language=key[2])
            _db_key = key
        return _db_instance


def forget_rag_db() -> None:
    """Discard the held instance — for tests and rebuilds.

    Necessary because otherwise the cache would survive over a deleted or
    replaced ``rag.db``: the key is the PATH, and it doesn't change on
    rebuild at the same place.
    """
    global _db_instance, _db_key
    with _db_lock:
        _db_instance = None
        _db_key = None


def embedding_config() -> dict:
    from whisper_tm.core.helpers import parse_bool

    ecfg = config.get("embedding") or {}
    return {
        "endpoint": (ecfg.get("endpoint") or "").strip(),
        "model": (ecfg.get("model") or "").strip(),
        "api_key": ecfg.get("api_key") or "",
        "timeout": ecfg.get("timeout") or 120,
        # parse_bool, not bool(): the settings page once saved the mode as
        # the string "false" — which bool() reads as True
        "local": parse_bool(ecfg.get("local", False)),
    }


def index_model_name() -> str:
    """What stands for the embedding model in the index fingerprint.

    Remote it's ``embedding.model``: the server runs what the request names.
    Locally it's the GGUF file (``embedding.local_model``) — llama.cpp loads
    exactly that one file and ignores the name in the request, so
    ``embedding.model`` says nothing about what produced the vectors. With
    it in the fingerprint, swapping the GGUF for another model of the same
    dimension went unnoticed: new vectors joined an index of old ones, and
    search just got worse.

    The file NAME, not its path: moving the same file changes no vector.
    """
    e = embedding_config()
    if e["local"]:
        from pathlib import Path

        local = str(config.get("embedding", "local_model", default="") or "")
        name = Path(local.strip()).name
        if name:
            return name
    return e["model"]


def embedding_ready() -> tuple[bool, str]:
    e = embedding_config()
    if e["local"]:
        # In local mode the endpoint isn't configured, it emerges from
        # port and model file — checked is whether the server can run at all.
        from whisper_tm.core import embed_server

        return embed_server.available()
    if not e["endpoint"]:
        return False, "Embedding endpoint not set (Settings → AI models)"
    if not e["model"]:
        return False, "Embedding model not set (Settings → AI models)"
    return True, ""


def make_embed_fn(purpose: str = "index"):
    """Embedding function texts→vectors from embedding config.

    In local mode (``embedding.local``) the llama.cpp server starts on
    FIRST call, not here: ``make_embed_fn()`` runs in status and check paths
    too, and they shouldn't load 4 GB onto the GPU. Each further call just
    extends the idle timeout.

    ``purpose`` travels through to server start: a search starts it on CPU,
    index run on the GPU. Default is ``index`` because that's the expensive
    path — whoever doesn't say what they want shouldn't accidentally get the
    slow one.
    """
    ok, msg = embedding_ready()
    if not ok:
        raise RagConfigError(msg)
    e = embedding_config()

    if e["local"]:
        from whisper_tm.core import embed_server

        def _embed_local(texts: list[str]) -> list[list[float]]:
            ep = embed_server.SERVER.ensure(purpose=purpose)
            try:
                # `model` doesn't matter to llama.cpp — it loads exactly one file.
                # The name only fills the request's model field; what makes
                # a model change visible to the index is the GGUF name in
                # its fingerprint (index_model_name).
                return llm_client.embed_texts(
                    texts,
                    endpoint=ep,
                    model=e["model"] or embed_server.model_file().name,
                    api_key="",
                    timeout=e["timeout"],
                )
            finally:
                embed_server.SERVER.used()

        return _embed_local

    def _embed(texts: list[str]) -> list[list[float]]:
        return llm_client.embed_texts(
            texts,
            endpoint=e["endpoint"],
            model=e["model"],
            api_key=e["api_key"],
            timeout=e["timeout"],
        )

    return _embed


# ══════════════════════════════════════════════════════════════════════════
#  Questions are not text — the instruction prefix
# ══════════════════════════════════════════════════════════════════════════

#: Template for the QUESTION. ``{question}`` is replaced, everything else stays.
#: Empty = no prefix.
DEFAULT_QUESTION_TEMPLATE = (
    "Instruct: Given a search query, retrieve relevant passages from spoken "
    "conversation transcripts\nQuery:{question}"
)


def question_template() -> str:
    v = config.get("embedding", "question_template", default=None)
    return DEFAULT_QUESTION_TEMPLATE if v is None else str(v)


def prepare_question(question: str) -> str:
    """Shape the question as the model saw it in training.

    **Why this isn't a detail.** Qwen3 embedding is trained for ASYMMETRIC
    retrieval: the question gets an instruction prefix, the text chunk stays
    raw. Embed both the same and you treat a question like a text chunk —
    both land elsewhere in the space than training intended.

    Measured on about 10,000 chunks, eight test questions:

    =====================================  ==========
    Top-1 hit changed                       6 of 8
    Separation (best vs fifth)              1.28×
    =====================================  ==========

    Clearest where it was awkward before — a question about cooking and
    food returned without prefix a two-word filler passage at 0.493, with
    prefix a passage that actually talks about food at 0.560.

    **The template is configurable because other models have other conventions.**
    nomic-embed-text wants ``search_query: …`` vs ``search_document: …``,
    bge-m3 wants nothing. A hardwired Qwen prefix would worsen quality there,
    not improve it.

    **It does NOT go in the index fingerprint.** The prefix affects only
    the question; stored vectors don't change. Change the template, no
    re-embedding needed.
    """
    template = question_template()
    if not template or "{question}" not in template:
        return question
    return template.replace("{question}", question)


def make_query_embed_fn():
    """Embedding for QUESTIONS — with prefix, otherwise identical to texts.

    And with ``purpose='search'``: that's the path a search uses to start the
    embedding server on CPU instead of on the GPU.
    """
    raw = make_embed_fn(purpose="search")

    def _question(texts: list[str]) -> list[list[float]]:
        return raw([prepare_question(t) for t in texts])

    return _question
