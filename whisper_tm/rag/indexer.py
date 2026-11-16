"""
Resumable RAG indexer: SQLite segments → chunks → embeddings → vector DB.

Source is always SQLite DB (source of truth). Indexer is resumable: already
indexed, unchanged transcripts are skipped (comparison via seg_count + updated_at).
Progress is reported via callback (for SSE streaming).
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Callable

from whisper_tm.core import paths

from .chunker import chunk_segments, text_weight
from .vector_db import RagConfigError, VectorDB

#: How much of a chunk too long for the embedding server is embedded, in
#: Latin-text characters (chunker.text_weight): ~1,500 tokens, under the
#: 2,048 a llama.cpp embedding server takes per input.
EMBED_FIT_WEIGHT = 6000


def _context_error(e: BaseException) -> bool:
    """The server refused an input as longer than its context (llama.cpp:
    "Context size has been exceeded", OpenAI-compatible servers: "maximum
    context length")."""
    msg = str(e).lower()
    return "context" in msg and ("exceed" in msg or "maximum" in msg or "too long" in msg)


def _fit(text: str) -> str:
    """``text`` cut to EMBED_FIT_WEIGHT, at a space where there is one."""
    if text_weight(text) <= EMBED_FIT_WEIGHT:
        return text
    lo, hi = 0, len(text)
    while lo < hi:  # the longest prefix within the weight
        mid = (lo + hi + 1) // 2
        if text_weight(text[:mid]) <= EMBED_FIT_WEIGHT:
            lo = mid
        else:
            hi = mid - 1
    cut = text[:lo]
    space = cut.rfind(" ")
    return cut[:space] if space > lo * 0.8 else cut


def _transcript_key(cand: dict) -> str:
    """Key of a transcript in RAG index.

    The identity, not the display value (docs/PATHS.md). With ``rel_path``
    there were two pitfalls: two same-named files from different roots
    collided on a key — ``_index_one`` deleted the chunks of the other via
    ``delete_transcript`` — and as soon as ``transcripts.rel_path`` changed
    (happens on each import, see ``upsert_transcript``), ``get_indexed_map()``
    found nothing: the entire library was re-embedded and old rows stayed as
    orphaned, still-searchable chunks.

    Old rows without identity keep ``rel_path`` as key.

    NOTE: On first run after this change, keys of existing entries change.
    They are re-embedded once; old rows can be removed with /api/rag/clear.
    """
    ab = (cand.get("abs_path") or "").strip()
    return paths.canon(ab) or ab or cand["rel_path"]


logger = logging.getLogger(__name__)

#: A run gives up after this many transcripts in a row whose embedding or
#: writing failed: the embedding server is gone or broken. Going on, a run
#: over a whole library logged a failure per transcript for hours; the next
#: run picks up where this one stopped.
FAILS_IN_A_ROW = 20

ProgressCb = Callable[[dict], None]


class RagIndexer:
    def __init__(
        self,
        rag_db: VectorDB,
        sqlite_db,
        embed_fn: Callable[[list[str]], list[list[float]]],
        *,
        chunk_max_chars: int = 1200,
        chunk_min_chars: int = 250,
        chunk_overlap_segments: int = 1,
        chunk_on_speaker_change: bool = False,
        include_speaker_labels: bool = True,
        batch_size: int = 64,
        embedding_model: str = "",
        default_media_dir: str = "",
    ):
        self.rag = rag_db
        self.db = sqlite_db
        self.embed_fn = embed_fn
        self.chunk_max_chars = chunk_max_chars
        self.chunk_min_chars = chunk_min_chars
        self.chunk_overlap_segments = chunk_overlap_segments
        self.chunk_on_speaker_change = chunk_on_speaker_change
        self.include_speaker_labels = include_speaker_labels
        self.batch_size = max(int(batch_size), 1)
        self.embedding_model = embedding_model
        self.default_media_dir = default_media_dir or ""

    # ── Helpers ────────────────────────────────────────────────────────────
    def probe_dimension(self) -> int:
        """Determine embedding dimension via a test embedding."""
        vecs = self.embed_fn(["Dimension sample."])
        if not vecs or not vecs[0]:
            raise RagConfigError("Test embedding returned no vector")
        return len(vecs[0])

    def _media_path(self, rel_path: str, media_dir: str, abs_path: str = "") -> str:
        """Location of the media file for a hit.

        Prefers the stored identity. Only when that's missing (legacy rows)
        is it built from root + display value — then via ``paths.abs_from``
        and not via ``os.path.join``: the latter is explicitly forbidden in
        docs/PATHS.md, because it produces OS-native separators, is not
        normalized, and silently discards the root on an absolute
        ``rel_path`` (``os.path.join("/nas", "/etc/passwd")`` yields
        ``/etc/passwd``).
        """
        if abs_path:
            return paths.canon(abs_path) or abs_path
        base = (media_dir or "").strip() or self.default_media_dir
        if not base:
            return rel_path
        return paths.abs_from(rel_path, base) or rel_path

    def _embed_alone(self, texts: list[str]) -> list[list[float]]:
        """Each text on its own; one the server finds too long for its
        context is embedded from its beginning (:func:`_fit`) — the chunk
        keeps its whole text, its vector stands for the start. Without this
        a single overlong chunk left its whole recording out of the index,
        on every run."""
        out = []
        for text in texts:
            try:
                out.extend(self._embed_chunks([text]))
            except Exception as e:  # noqa: BLE001 — only a context error is retried
                if not _context_error(e):
                    raise
                logger.info(
                    "Chunk too long for the embedding server (%d characters) "
                    "— embedded from its beginning",
                    len(text),
                )
                out.extend(self._embed_chunks([_fit(text)]))
        return out

    def _embed_chunks(self, texts: list[str]) -> list[list[float]]:
        """Embed chunk texts in batches."""
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            vecs = self.embed_fn(batch)
            if len(vecs) != len(batch):
                raise RagConfigError(
                    f"Embedding response length {len(vecs)} != request {len(batch)}"
                )
            out.extend(vecs)
        return out

    def _config_fingerprint(self) -> str:
        """Stable fingerprint from chunk config + embedding model.

        Sorted keys → stable serialization (no false "stale" on
        every run). If the value changes, existing embeddings are considered
        outdated and are regenerated.
        """
        payload = {
            "embedding_model": self.embedding_model or "",
            "chunk_max_chars": self.chunk_max_chars,
            "chunk_min_chars": self.chunk_min_chars,
            "chunk_overlap_segments": self.chunk_overlap_segments,
            "chunk_on_speaker_change": bool(self.chunk_on_speaker_change),
            "include_speaker_labels": bool(self.include_speaker_labels),
        }
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha1(blob.encode("utf-8")).hexdigest()

    # ── Main run ─────────────────────────────────────────────────────────
    def run(
        self,
        *,
        progress_cb: ProgressCb | None = None,
        force: bool = False,
        stop_flag=None,
        dimensions: int | None = None,
    ) -> dict:
        """Run an indexing pass.

        Args:
            progress_cb: called with a progress dict.
            force: re-index all transcripts (even unchanged ones).
            stop_flag: object with .is_set() → abort (threading.Event).
            dimensions: override embedding dimension (otherwise config/sample).

        Returns:
            Summary: indexed, skipped, chunks, errors, dimension, …
        """

        def emit(**kw):
            if progress_cb:
                try:
                    progress_cb(kw)
                except Exception as e:  # noqa
                    logger.debug("progress_cb error: %s", e)

        def stopped() -> bool:
            return bool(stop_flag is not None and stop_flag.is_set())

        t0 = time.time()

        # 1) Determine dimension (+ schema)
        dim = int(dimensions or self.rag.dimensions or 0)
        emit(phase="prepare", message="Determining embedding dimension …")
        if dim <= 0:
            dim = self.probe_dimension()
        else:
            # Check explicitly set dimension against the actual model dimension
            # — otherwise the upserts fail late on the ::vector(dim) cast.
            probed = self.probe_dimension()
            if probed != dim:
                raise RagConfigError(
                    f"Configured dimension {dim} ≠ model returns {probed}. "
                    f"Please correct 'dimensions' or leave it empty (auto-sample)."
                )
        self.rag.dimensions = dim

        emit(phase="prepare", message="Ensuring schema …", dimension=dim)
        info = self.rag.ensure_schema(dimensions=dim)
        if info.get("exists") and info.get("dimensions") and info["dimensions"] != dim:
            raise RagConfigError(
                f"Schema has dimension {info['dimensions']}, model returns {dim}. "
                f"Please clear the RAG DB (rebuild) to change the dimension."
            )

        # Store embedding_model in meta + read previous config fingerprint.
        # A change to the chunk config or the embedding model (with the same
        # dimension) makes existing embeddings outdated → re-index everything.
        fingerprint = self._config_fingerprint()
        prev_fingerprint = ""
        try:
            with self.rag.connection() as c:
                prev_fingerprint = self.rag.get_meta(c).get("index_fingerprint", "")
                if self.embedding_model:
                    self.rag._set_meta(c, "embedding_model", self.embedding_model)
        except Exception as e:  # noqa
            logger.debug("Meta embedding_model: %s", e)
        config_changed = bool(prev_fingerprint) and prev_fingerprint != fingerprint

        # 2) Load candidates + already indexed ones
        candidates = self.db.get_rag_candidates()
        indexed_map = {} if force else self.rag.get_indexed_map()
        total = len(candidates)
        emit(phase="scan", total=total, message=f"{total} transcripts found")

        indexed = 0
        skipped = 0
        chunk_total = 0
        removed = 0
        errors: list[dict] = []
        streak = 0  # failed embeddings/writes in a row
        gave_up = ""

        # 3) Indexing, batched ACROSS TRANSCRIPTS.
        #
        # Originally, embedding happened per transcript, and `_embed_chunks`
        # only batched within that single transcript. A typical transcript
        # has only a handful of chunks, and nearly all are below the
        # configured `batch_size=64` — so the configured value was
        # practically never reached.
        #
        # **For the local server, batching gains nothing.** This was
        # measured and was contrary to expectation: the same run with
        # qwen3-embedding-4b on a consumer GPU reached practically the same
        # chunks/s per transcript and batched.
        #
        # The bottleneck is GPU compute time per token, not the overhead per
        # request. Anyone looking for speed here has to work on the model or
        # the card, not the batch size.
        #
        # It stays anyway, but for a different reason: the number of
        # requests drops measurably (30 transcripts used to go out as 30
        # requests, now as 2). As soon as the endpoint is NOT on the same
        # machine — OpenAI, a server on the network —, the round-trip time
        # per request matters, and there it makes a real difference. On the
        # same machine it's a precaution, not a gain.
        #
        # We collect until a batch is full, then embed once,
        # then write each transcript individually. The transaction
        # stays per transcript — an error while writing must not take
        # the others down with it.
        #
        # ONE connection for the whole run. With Postgres this still needed
        # TCP keepalives so the connection, idle during the slow embedding
        # calls, wouldn't get cut off — an SQLite file has no
        # connection anyone could cut.
        with self.rag.connection() as conn:
            buffer: list[tuple[dict, list]] = []
            buffer_chunks = 0

            def flush_buffer():
                nonlocal buffer, buffer_chunks, indexed, chunk_total, streak
                if not buffer:
                    return
                for _cand2, n in self._write_batch(conn, buffer, errors):
                    if n >= 0:
                        indexed += 1
                        chunk_total += n
                        streak = 0
                    else:
                        streak += 1
                buffer = []
                buffer_chunks = 0

            for i, cand in enumerate(candidates):
                if stopped():
                    flush_buffer()
                    emit(
                        phase="stopped",
                        done=i,
                        total=total,
                        indexed=indexed,
                        skipped=skipped,
                        chunks=chunk_total,
                    )
                    break

                key = _transcript_key(cand)
                seg_count = cand.get("seg_count") or 0
                updated_at = cand.get("updated_at") or 0.0

                # Resume: unchanged AND same config → skip
                prev = indexed_map.get(key)
                if (
                    not force
                    and not config_changed
                    and prev
                    and prev.get("seg_count") == seg_count
                    and _close(prev.get("updated_at"), updated_at)
                ):
                    skipped += 1
                    if (i + 1) % 25 == 0 or (i + 1) == total:
                        emit(
                            phase="index",
                            done=i + 1,
                            total=total,
                            indexed=indexed,
                            skipped=skipped,
                            chunks=chunk_total,
                            current=key,
                        )
                    continue

                try:
                    chunks = self._chunks_from(cand)
                except Exception as e:  # noqa
                    logger.warning("RAG index error for %s: %s", key, e)
                    errors.append({"key": key, "error": str(e)[:200]})
                    continue

                buffer.append((cand, chunks))
                buffer_chunks += len(chunks)
                if buffer_chunks >= self.batch_size:
                    flush_buffer()
                if streak >= FAILS_IN_A_ROW:
                    last = errors[-1]["error"] if errors else ""
                    gave_up = (
                        f"Stopped: {streak} transcripts in a row failed — the "
                        f"embedding server seems unreachable or broken ({last})"
                    )
                    logger.warning("RAG index: %s", gave_up)
                    break

                emit(
                    phase="index",
                    done=i + 1,
                    total=total,
                    indexed=indexed,
                    skipped=skipped,
                    chunks=chunk_total,
                    current=key,
                    errors=len(errors),
                )

            flush_buffer()
            emit(
                phase="index",
                done=total,
                total=total,
                indexed=indexed,
                skipped=skipped,
                chunks=chunk_total,
                errors=len(errors),
            )

            # Transcripts that left the source (deleted, no longer complete,
            # stored under an older key): their chunks stayed searchable
            # and counted as embedded. The candidate list is complete even
            # for a stopped run, so this holds then too. An EMPTY list is
            # not trusted — that's a wrong or unreadable database, not a
            # library where everything was deleted.
            if candidates:
                try:
                    removed = self.rag.prune_except(conn, {_transcript_key(c) for c in candidates})
                except Exception as e:  # noqa
                    logger.warning("RAG: removing vanished transcripts failed: %s", e)
                    errors.append({"key": "", "error": f"prune: {str(e)[:180]}"})
                    removed = 0
                if removed:
                    logger.info("RAG: %d vanished transcripts removed from the index", removed)

        # 3.5) Only store the config fingerprint after a complete, error-free
        # run — an aborted/failed run still counts as "config changed" next
        # time and gets re-indexed.
        if not stopped() and not errors:
            try:
                with self.rag.connection() as c:
                    self.rag._set_meta(c, "index_fingerprint", fingerprint)
            except Exception as e:  # noqa
                logger.debug("Meta index_fingerprint: %s", e)

        summary = {
            "indexed": indexed,
            "skipped": skipped,
            "removed": removed,
            "chunks": chunk_total,
            "errors": errors,
            "error_count": len(errors),
            "dimension": dim,
            "total_candidates": total,
            "elapsed_sec": round(time.time() - t0, 1),
            "stopped": stopped(),
            "gave_up": gave_up,
        }
        emit(phase="done", **{k: v for k, v in summary.items() if k != "errors"})
        return summary

    def _chunks_from(self, cand: dict) -> list:
        """The chunks of a transcript — without embedding, without writing.

        Separated from writing so that embedding can be batched across
        multiple transcripts (see the reasoning in `run`).
        """
        segments = self.db.get_segments_for_rag(cand["id"])
        if not segments:
            return []
        return chunk_segments(
            segments,
            max_chars=self.chunk_max_chars,
            min_chars=self.chunk_min_chars,
            overlap_segments=self.chunk_overlap_segments,
            on_speaker_change=self.chunk_on_speaker_change,
            include_speaker_labels=self.include_speaker_labels,
        )

    def _mark_empty(self, conn, cand: dict):
        """A transcript without chunks counts as processed, not as pending.

        Otherwise coverage permanently stays under 100% and the indexer
        picks up the same empty transcripts again on every run.
        """
        key = _transcript_key(cand)
        with self.rag.transaction(conn):
            self.rag.delete_transcript(conn, key)
            self.rag.mark_indexed(
                conn,
                key,
                seg_count=cand.get("seg_count") or 0,
                updated_at=cand.get("updated_at") or 0.0,
                chunk_count=0,
            )

    def _write_batch(
        self, conn, buffer: list[tuple[dict, list]], errors: list[dict]
    ) -> list[tuple[dict, int]]:
        """Embed and write a batch of collected transcripts.

        Returns ``[(cand, chunk_count)]``; ``-1`` means "failed".

        **Errors must not turn the batch all-or-nothing.** If embedding
        fails for the whole batch (server down, timeout), it is
        retried per transcript individually. Otherwise a single
        broken chunk would take down 64 healthy transcripts with it —
        and again on the next run.
        """
        result: list[tuple[dict, int]] = []
        with_chunks = [(c, ch) for c, ch in buffer if ch]
        for cand, chunks in buffer:
            if not chunks:
                try:
                    self._mark_empty(conn, cand)
                    result.append((cand, 0))
                except Exception as e:  # noqa
                    errors.append({"key": _transcript_key(cand), "error": str(e)[:200]})
                    result.append((cand, -1))
        if not with_chunks:
            return result

        texts = [ch.content for _c, chs in with_chunks for ch in chs]
        try:
            vecs = self._embed_chunks(texts)
        except Exception as e:  # noqa
            logger.warning(
                "Batch embedding failed (%d chunks from %d "
                "transcripts) — retrying individually: %s",
                len(texts),
                len(with_chunks),
                e,
            )
            for cand, chunks in with_chunks:
                texts_c = [c.content for c in chunks]
                try:
                    try:
                        vecs_c = self._embed_chunks(texts_c)
                    except Exception as e_ctx:  # noqa: BLE001
                        if not _context_error(e_ctx):
                            raise
                        vecs_c = self._embed_alone(texts_c)
                    result.append((cand, self._write(conn, cand, chunks, vecs_c)))
                except Exception as e2:  # noqa
                    logger.warning("RAG index error for %s: %s", _transcript_key(cand), e2)
                    errors.append({"key": _transcript_key(cand), "error": str(e2)[:200]})
                    result.append((cand, -1))
            return result

        pos = 0
        for cand, chunks in with_chunks:
            part = vecs[pos : pos + len(chunks)]
            pos += len(chunks)
            try:
                result.append((cand, self._write(conn, cand, chunks, part)))
            except Exception as e:  # noqa
                logger.warning("RAG index error for %s: %s", _transcript_key(cand), e)
                errors.append({"key": _transcript_key(cand), "error": str(e)[:200]})
                result.append((cand, -1))
        return result

    def _write(self, conn, cand: dict, chunks: list, vecs: list) -> int:
        """Store a transcript's chunks transactionally. Returns count."""
        key = _transcript_key(cand)
        media_path = self._media_path(
            cand["rel_path"], cand.get("media_dir") or "", cand.get("abs_path") or ""
        )
        language = cand.get("language") or ""
        model = cand.get("model") or ""
        rows = []
        for ch, vec in zip(chunks, vecs, strict=True):
            row = ch.as_row()
            row.update(
                {
                    "transcript_key": key,
                    "language": language,
                    "model": model,
                    "media_path": media_path,
                    "embedding": vec,
                }
            )
            rows.append(row)

        # Transactional: old chunks out, new ones in, mark done
        with self.rag.transaction(conn):
            self.rag.delete_transcript(conn, key)
            self.rag.upsert_chunks(conn, rows)
            self.rag.mark_indexed(
                conn,
                key,
                seg_count=cand.get("seg_count") or 0,
                updated_at=cand.get("updated_at") or 0.0,
                chunk_count=len(rows),
            )
        return len(rows)

    def _index_one(self, conn, cand: dict) -> int:
        """Index a single transcript.

        Kept for callers that want exactly one (tests, a
        single re-index). The main run goes through `_write_batch`,
        because embedding one at a time never fills the batch.
        """
        chunks = self._chunks_from(cand)
        if not chunks:
            self._mark_empty(conn, cand)
            return 0
        vecs = self._embed_chunks([ch.content for ch in chunks])
        return self._write(conn, cand, chunks, vecs)


def _close(a, b, *, eps: float = 0.5) -> bool:
    """updated_at comparison with a small tolerance (float rounding)."""
    try:
        return abs(float(a or 0.0) - float(b or 0.0)) <= eps
    except (TypeError, ValueError):
        return a == b
