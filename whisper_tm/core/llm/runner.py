"""
Chunk loop runner: sequential processing with progress, cancellation and
partial-failure tolerance.

No parallelism — the target setup is a llama.cpp server with one slot;
parallel requests would only slow each other down.
"""

from __future__ import annotations

import logging
import time

from whisper_tm.core.llm_client import LlmAnswerError, LlmConnectionError

logger = logging.getLogger(__name__)


def run_chunks(
    chunks: list,
    processor,
    *,
    job=None,
    description: str = "LLM",
    attempts_per_chunk: int = 2,
    part_callback=None,
):
    """Send all chunks sequentially through ``processor(chunk)``.

    * Progress is reported to ``job`` (message + progress 0–100),
      cancellation via ``job.cancel`` is respected (InterruptedError, as
      in the transcription pipeline).
    * A chunk gets ``attempts_per_chunk`` attempts; after that it moves to
      the failure list and the run CONTINUES (partial result > nothing).
    * At the end, failed chunks are retried exactly once — that's often
      enough for transient server hiccups.

    ``part_callback(dict)`` is called after each finished chunk — the UI
    builds its expandable step log from that (each step can be expanded
    to see what it produced).

    Returns:
        (results, failed)
        results:      {chunk.index: processor return value}
        failed:  [(chunk, last_error_str)]
    """
    results: dict[int, object] = {}
    failed: list[tuple[object, str]] = []
    # Display number = position in the PASSED-IN list. chunk.index counts
    # across all chunks of the transcript — for pre-filtered lists
    # (correction "only marked") the log would otherwise show "section 7/3".
    position = {c.index: i + 1 for i, c in enumerate(chunks)}

    def _report(i: int, total: int, extra: str = ""):
        if job is None:
            return
        percent = int(i / max(1, total) * 100)
        job.update(
            progress=min(99, percent), message=f"{description}: Chunk {i + 1}/{total}{extra}"
        )

    def _record_step(chunk, secs: float):
        """⏱ log per chunk + record as a job step (appears in the job's
        step_timings — same mechanism as transcription)."""
        logger.info(
            "⏱ %s chunk %d (%d segments): %.1fs",
            description,
            chunk.index,
            chunk.to_seg - chunk.from_seg + 1,
            secs,
        )
        if job is not None:
            try:
                from whisper_tm.extensions import job_manager

                if job_manager:
                    job_manager.record_step(
                        job, f"{description} Chunk {position[chunk.index]}", secs
                    )
            except Exception:
                pass

    def _one(chunk) -> tuple[bool, str]:
        last = ""
        for attempt in range(attempts_per_chunk):
            if job is not None and job.cancel:
                raise InterruptedError("Cancelled")
            t0 = time.monotonic()
            try:
                result = processor(chunk)
                results[chunk.index] = result
                duration = time.monotonic() - t0
                _record_step(chunk, duration)
                if part_callback:
                    try:
                        part_callback(
                            {
                                "number": position[chunk.index],
                                "total": len(chunks),
                                "from_seg": chunk.from_seg,
                                "to_seg": chunk.to_seg,
                                "tokens": chunk.tokens,
                                "duration": round(duration, 1),
                                "result": result,
                            }
                        )
                    except Exception as e:  # the log must never interfere
                        logger.debug("part_callback: %s", e)
                return True, ""
            except (LlmConnectionError, LlmAnswerError) as e:
                last = str(e)[:300]
                logger.warning(
                    "%s chunk %d attempt %d/%d: %s",
                    description,
                    chunk.index,
                    attempt + 1,
                    attempts_per_chunk,
                    last,
                )
                if attempt + 1 < attempts_per_chunk:
                    time.sleep(2)
        return False, last

    t_run = time.monotonic()
    total = len(chunks)
    for i, chunk in enumerate(chunks):
        if job is not None and job.cancel:
            raise InterruptedError("Cancelled")
        _report(i, total)
        ok, error = _one(chunk)
        if not ok:
            failed.append((chunk, error))

    # End retry: once over all failures
    if failed:
        remainder = failed
        failed = []
        for j, (chunk, _old) in enumerate(remainder):
            if job is not None and job.cancel:
                raise InterruptedError("Cancelled")
            _report(total - len(remainder) + j, total, " (retry)")
            ok, error = _one(chunk)
            if not ok:
                failed.append((chunk, error))
                logger.error("%s chunk %d finally failed: %s", description, chunk.index, error)

    logger.info(
        "⏱ %s total: %d/%d chunks in %.1fs%s",
        description,
        len(results),
        total,
        time.monotonic() - t_run,
        f" ({len(failed)} failed)" if failed else "",
    )
    return results, failed
