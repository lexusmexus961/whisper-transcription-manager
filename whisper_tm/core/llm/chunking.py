"""
Segment boundary chunker.

Splits a transcript (list of segments) into chunks whose formatted lines
together don't exceed a token budget. Never cuts in the middle of a segment.
Preferred cut points are speaker changes and pauses > 2s — the chunker
searches backward from the budget limit up to 20% into the chunk to find
such a point.

IDs are segment indices in the transcript (stable across the whole pipeline).
Overlap is pure READ context: the last segments of the previous chunk run
along as non-editable context.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from whisper_tm.core.llm.budget import estimate_tokens

# Pause between two segments where a cut is "good" (seconds)
PAUSE_THRESHOLD = 2.0
# How far back from the budget end the chunker searches for a good
# cut point (fraction of chunk segments)
BACKSEARCH_SHARE = 0.2


@dataclass
class Chunk:
    """A processing chunk: segment indices [from_seg, to_seg] inclusive."""

    index: int
    from_seg: int
    to_seg: int
    lines: list[str] = field(default_factory=list)  # formatted lines
    tokens: int = 0
    # Read context: (seg_index, line) of the last segments of the previous chunk
    context_before: list[tuple[int, str]] = field(default_factory=list)

    @property
    def seg_ids(self) -> range:
        return range(self.from_seg, self.to_seg + 1)


def _good_cut_point(segments: list[dict], i: int) -> bool:
    """Is there a good cut point between segment i-1 and i?"""
    if i <= 0 or i >= len(segments):
        return False
    prev, cur = segments[i - 1], segments[i]
    if (cur.get("speaker") or "—") != (prev.get("speaker") or "—"):
        return True
    try:
        return float(cur.get("start", 0)) - float(prev.get("end", 0)) >= PAUSE_THRESHOLD
    except (TypeError, ValueError):
        return False


def split_segments(
    segments: list[dict],
    lines: list[str],
    budget_tokens: int,
    *,
    overlap_segments: int = 3,
    max_segments: int = 0,
) -> list[Chunk]:
    """Split segments into chunks.

    ``lines[i]`` is the already formatted prompt line for ``segments[i]``
    (token count must match the real prompt — so the caller formats,
    not the chunker).

    A single segment over budget becomes its own chunk (the LLM never sees
    a truncated context — only this one long segment; practically only happens
    with broken transcripts).

    ``max_segments`` (0 = off) additionally caps the segment COUNT per chunk:
    tasks with 1:1 answer per segment (translation, correction) are otherwise
    crushed by JSON structure costs for answers when there are many short
    segments.
    """
    if len(segments) != len(lines):
        raise ValueError(
            f"segments ({len(segments)}) and lines ({len(lines)}) must be the same length"
        )
    if not segments:
        return []

    tokens_per_line = [estimate_tokens(z) + 1 for z in lines]  # +1: newline

    chunks: list[Chunk] = []
    start = 0
    while start < len(segments):
        total_sum = 0
        end = start
        while end < len(segments) and (
            total_sum + tokens_per_line[end] <= budget_tokens or end == start
        ):  # Giant segment: pass through alone
            if max_segments and end - start >= max_segments:
                break
            total_sum += tokens_per_line[end]
            end += 1
        # end now points AFTER the last fitting segment.

        # Search backward for a good cut point — but only if segments
        # follow and we're not leaving almost nothing.
        if end < len(segments):
            windows = max(1, int((end - start) * BACKSEARCH_SHARE))
            for candidate in range(end, max(start + 1, end - windows), -1):
                if _good_cut_point(segments, candidate):
                    total_sum -= sum(tokens_per_line[candidate:end])
                    end = candidate
                    break

        chunk = Chunk(
            index=len(chunks),
            from_seg=start,
            to_seg=end - 1,
            lines=lines[start:end],
            tokens=total_sum,
        )
        if chunks and overlap_segments > 0:
            first = max(0, start - overlap_segments)
            chunk.context_before = [(i, lines[i]) for i in range(first, start)]
        chunks.append(chunk)
        start = end

    return chunks
