"""
RAG subsystem: semantic + hybrid search over transcripts.

The main SQLite remains the source of truth. This package builds from it
a rebuildable vector store (own SQLite file alongside ``speaker.db``) and
provides hybrid retrieval: vector search as matrix product, fulltext via FTS5,
fused per reciprocal rank fusion.

**Why no more Postgres/pgvector.** Measured on a few hundred thousand chunks
× 2560 dimensions — writes over ten times slower, more than twice the space,
plus minutes of HNSW build on each rebuild for a search whose speed gain
(milliseconds vs tens of milliseconds) is indistinguishable on the search
mask. Details in docstring of :mod:`whisper_tm.rag.vector_db`.

Public API:
    VectorDB             – storage, schema, upsert, search
    RagIndexer           – resumable indexer (segments → chunks → vectors)
    RagConfigError       – misconfiguration
"""

from __future__ import annotations

from .chunker import Chunk, chunk_segments
from .indexer import RagIndexer
from .vector_db import RagConfigError, VectorDB


class RagUnavailable(RuntimeError):  # noqa: N818 - state name, historical
    """The vector store cannot be opened."""


__all__ = [
    "VectorDB",
    "RagIndexer",
    "chunk_segments",
    "Chunk",
    "RagUnavailable",
    "RagConfigError",
]
