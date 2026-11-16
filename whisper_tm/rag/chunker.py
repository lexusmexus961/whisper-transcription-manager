"""
Segment → chunk logic for RAG indexer.

Transcript segments are often only a few seconds/words — too granular for
meaningful embeddings. Here consecutive segments are merged into larger,
semantically coherent chunks (goal: ~1000-1500 characters) with optional
overlap so context at chunk boundaries isn't lost.

Purely functional, no external dependencies → easy to test.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Chunk:
    chunk_index: int
    content: str
    speaker: str  # dominant speaker (by character count)
    start_sec: float
    end_sec: float
    seg_start: int  # first seg_index
    seg_end: int  # last seg_index
    seg_count: int
    speakers: list[str] = field(default_factory=list)

    def as_row(self) -> dict:
        return {
            "chunk_index": self.chunk_index,
            "content": self.content,
            "speaker": self.speaker or None,
            "start_sec": self.start_sec,
            "end_sec": self.end_sec,
            "seg_start": self.seg_start,
            "seg_end": self.seg_end,
            "metadata": {"speakers": self.speakers, "seg_count": self.seg_count},
        }


def _seg_start(s: dict) -> float:
    v = s.get("start_sec")
    if v is None:
        v = s.get("start", 0.0)
    try:
        return float(v or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _seg_end(s: dict) -> float:
    v = s.get("end_sec")
    if v is None:
        v = s.get("end", 0.0)
    try:
        return float(v or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _seg_index(s: dict, fallback: int) -> int:
    v = s.get("seg_index")
    if v is None:
        return fallback
    try:
        return int(v)
    except (TypeError, ValueError):
        return fallback


def _is_cjk(cp: int) -> bool:
    """Han, kana, Hangul and fullwidth forms — the ranges of
    core/llm/budget.estimate_tokens."""
    return (
        0x4E00 <= cp <= 0x9FFF
        or 0x3400 <= cp <= 0x4DBF
        or 0x3040 <= cp <= 0x30FF
        or 0xAC00 <= cp <= 0xD7AF
        or 0xFF00 <= cp <= 0xFFEF
    )


def text_weight(text: str) -> int:
    """A text's length in Latin-text characters, which is what ``max_chars``
    means: a CJK character costs as much as ~2.7 Latin ones (~1.5 against
    ~4 characters a token, core/llm/budget.py), so the limit bounds the
    tokens a chunk holds whatever its script. Counted as plain characters,
    1,200 of Japanese came to some 2,000 tokens with the speaker labels —
    over what the embedding server takes per input, and the recording was
    left out of the index. Without CJK it is exactly ``len(text)``: the
    chunks of other languages stay as they were."""
    if not text:
        return 0
    cjk = sum(1 for ch in text if _is_cjk(ord(ch)))
    return len(text) + (cjk * 5 + 1) // 3


def _dominant_speaker(counts: dict[str, int]) -> str:
    real = {k: v for k, v in counts.items() if k and k != "—"}
    if not real:
        return ""
    return max(real.items(), key=lambda kv: kv[1])[0]


def _render(seg_texts: list[tuple[str, str]], include_speaker_labels: bool) -> str:
    """Render segments into chunk text.

    seg_texts: list of (speaker, text). When labels are enabled, a
    "[Speaker]" prefix is inserted per speaker change, otherwise simple
    space concatenation.
    """
    if not include_speaker_labels:
        return " ".join(t for _, t in seg_texts if t).strip()

    out: list[str] = []
    last_spk = None
    for spk, text in seg_texts:
        text = (text or "").strip()
        if not text:
            continue
        spk_norm = spk if (spk and spk != "—") else None
        if spk_norm and spk_norm != last_spk:
            out.append(f"\n[{spk_norm}] {text}")
            last_spk = spk_norm
        else:
            out.append(text)
    return " ".join(out).strip()


def chunk_segments(
    segments: list[dict],
    *,
    max_chars: int = 1200,
    min_chars: int = 250,
    overlap_segments: int = 1,
    on_speaker_change: bool = False,
    include_speaker_labels: bool = True,
) -> list[Chunk]:
    """Greedily merge segments into chunks.

    Args:
        segments: segment dicts with start_sec/end_sec (or start/end),
                  speaker, text, optional seg_index.
        max_chars: target upper bound per chunk, in Latin-text characters
                   (:func:`text_weight` — CJK counts heavier). A soft
                   limit: a single very long segment may exceed it.
        min_chars: a chunk is not closed before reaching this length
                   (prevents tiny chunks on frequent speaker changes).
        overlap_segments: number of segments carried over into the next
                   chunk (context preservation at boundaries).
        on_speaker_change: treat speaker changes as a preferred boundary.
        include_speaker_labels: include speaker prefixes in the chunk text.

    Returns:
        List of Chunk objects (chunk_index sequential from 0).
    """
    if not segments:
        return []

    max_chars = max(int(max_chars), 50)
    min_chars = max(min(int(min_chars), max_chars), 0)
    overlap_segments = max(int(overlap_segments), 0)

    chunks: list[Chunk] = []
    cur: list[dict] = []
    cur_len = 0
    carry_count = 0  # leading overlap segments in cur (already emitted)

    def _flush():
        nonlocal cur, cur_len, carry_count
        if not cur:
            return
        seg_texts = [(s.get("speaker") or "", (s.get("text") or "")) for s in cur]
        content = _render(seg_texts, include_speaker_labels)
        if not content.strip():
            cur = []
            cur_len = 0
            carry_count = 0
            return
        counts: dict[str, int] = {}
        for s in cur:
            spk = s.get("speaker") or ""
            counts[spk] = counts.get(spk, 0) + len(s.get("text") or "")
        speakers = sorted(
            {
                (s.get("speaker") or "").strip()
                for s in cur
                if (s.get("speaker") or "").strip() and (s.get("speaker") or "").strip() != "—"
            }
        )
        idxs = [_seg_index(s, i) for i, s in enumerate(cur)]
        chunks.append(
            Chunk(
                chunk_index=len(chunks),
                content=content,
                speaker=_dominant_speaker(counts),
                start_sec=_seg_start(cur[0]),
                end_sec=_seg_end(cur[-1]),
                seg_start=min(idxs),
                seg_end=max(idxs),
                seg_count=len(cur),
                speakers=speakers,
            )
        )
        # Prepare overlap: last N segments into the next chunk
        carry = cur[-overlap_segments:] if overlap_segments and len(cur) > overlap_segments else []
        cur = list(carry)
        cur_len = sum(text_weight(s.get("text") or "") for s in cur)
        carry_count = len(carry)

    prev_speaker = None
    for s in segments:
        text = (s.get("text") or "").strip()
        spk = (s.get("speaker") or "").strip()

        # Speaker-change boundary (only if chunk is already "full enough")
        if (
            on_speaker_change
            and prev_speaker is not None
            and spk
            and spk != prev_speaker
            and cur_len >= min_chars
        ):
            _flush()

        cur.append(s)
        cur_len += text_weight(text)
        prev_speaker = spk or prev_speaker

        # Character limit
        if cur_len >= max_chars and cur_len >= min_chars:
            _flush()

    # Remainder: don't duplicate pure overlap leftovers as their own chunk —
    # only flush if at least one NEW segment (not already covered by
    # overlap) remains.
    if len(cur) > carry_count:
        _flush()

    # Reassign chunk_index after any skipped empty chunks
    for new_idx, ch in enumerate(chunks):
        ch.chunk_index = new_idx
    return chunks
