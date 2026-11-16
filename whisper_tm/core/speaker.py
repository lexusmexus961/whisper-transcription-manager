"""Voice vectors from a diarized recording.

This is the production part of speaker recognition. It answers two
questions, in this order of priority:

1. **Who was involved in this recording?**
2. **Approximately where did they speak?**

The operating point is measured, not guessed. Every number below carries
the measurement it comes from in its own comment.

The flow per file:

.. code-block:: text

    Diarization  →  clean spans    →  30s windows  →  one 192-D vector
    (SPEAKER_xx)    (only ONE active) (concatenated)   per window

What deliberately does NOT happen here:

* **No averaging.** Multiple windows remain multiple vectors. Combining
  is measurably worse (+0.44 to +1.56 EER points) and also shifts
  the threshold by count (Sholokhov et al., 2302.09523).
* **No sampling.** ALL windows are formed, not three from start,
  middle and end. Twelve windows covered 14% of speech in a 7-hour
  recording — a quiet speaker would have gone unnoticed.
* **No music filter.** By speaking time, only 5–8% of windows are
  background music or singing; this is deliberately left as is.
* **No identity.** ``SPEAKER_00`` is a cut marker from diarization and
  applies only within this one file. Who that is is decided by
  vector comparison.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ── Operating point ────────────────────────────────────────────────────────

SR = 16000

#: Net speech per window. Measured: every step up to 30s gains,
#: nothing above. 15s costs 53% more errors and doubles the hardest
#: quartile — and the hardest quartile is exactly the similar voice pairs
#: we care about.
MAX_SPEECH = 30.0

#: No vector is formed below this. A 10s voice has 2.5× the error rate
#: of a 30s voice.
MIN_SPEECH = 4.0

#: Only speech segments of this length or longer may contribute to a window —
#: with fallback, see :func:`build_windows`. Measured: in a known mix of
#: two speakers, similarity to the dominant person rises
#: from 0.675 to 0.757 with EQUAL total duration. The second speaker is in
#: short interjections, which this gate filters out.
MIN_SEGMENT = 4.0

#: Maximum windows per speaker ID; 0 = unlimited. Unlimited is the
#: operating point — complete coverage, not sampling. The cap exists only
#: for tools and emergencies.
MAX_WINDOWS = 0

#: Shortest span considered at all. Below this it's more like
#: diarization boundary noise than speech.
MIN_SPAN = 0.15

#: Checkpoint written to EVERY row. A change makes old vectors
#: incomparable — you won't see it without this column.
#: ``vb2/ptn`` beats ``vb2+vox2+cnc/ft_mix`` by +0.24 points [+0.01, +0.49]:
#: pretrain-only generalizes better.
MODEL_NAME = "redimnet_m"

#: ``(dataset, train_type)`` for ``torch.hub`` — **the only place**
#: where the checkpoint lives. A second copy anywhere would let one side's
#: vectors silently become incomparable to the other's. The database writes
#: :data:`MODEL_BRAND` to every row and would show a change — tools don't
#: have this column. A test guards that no other file names the checkpoint.
MODEL_CHECKPOINT = ("vb2", "ptn")
MODEL_BRAND = f"{MODEL_NAME}/{MODEL_CHECKPOINT[0]}-{MODEL_CHECKPOINT[1]}"

#: The model's weights: the project's release named "latest" — what it holds
#: could change under that name, so the file is checked against this SHA-256
#: before anything of it is unpickled. The model's code is kept in the
#: package (whisper_tm/vendor/redimnet): fetched with torch.hub from the
#: unpinned "master", whatever was on it at first use ran.
REDIMNET_URL = (
    "https://github.com/IDRnD/ReDimNet/releases/download/latest/"
    f"M-{MODEL_CHECKPOINT[0]}-{MODEL_CHECKPOINT[1]}.pt"
)
REDIMNET_SHA256 = "42c13ce2bcf1ce466afdbe461c8dbe9471fc23204a1480fb4f306f48152f9af3"

#: Thresholds. They are NOT the same, and there are THREE.
#:
#: ``ACROSS_FILES`` (0.55) applies to comparing **two individual
#: voices** — measured this way.
#:
#: ``WITHIN_FILE`` (0.62) applies **within a recording**: all windows share
#: microphone and session, and the common channel lifts every
#: similarity.
#:
#: ``GALLERY`` (0.75) applies to "who was there" — one recording
#: against ALL named people. It must be higher, for a reason unrelated to
#: voices: with a dozen people and multiple windows per recording there are
#: hundreds of chances to cross the threshold by chance. Measured
#: (leave-one-out over a few hundred files, ``tools/speaker_quality.py``):
#:
#: ====== ============ ============
#: Threshold Hit Rate     False Alarms
#: ====== ============ ============
#: 0.55       99.8 %       47.4 %
#: 0.65       99.4 %        8.5 %
#: 0.70       98.1 %        3.7 %
#: **0.75**   **96.0 %**    **0.6 %**
#: 0.78       91.1 %        0.0 %
#: ====== ============ ============
#:
#: At 0.55 nearly every other recording would name someone wrong.
#: That's the costliest error type: a wrong name in a recording you
#: never hear again.
THRESHOLD_ACROSS_FILES = 0.55
THRESHOLD_WITHIN_FILE = 0.62
THRESHOLD_GALLERY = 0.75

#: Cutoff for the grouping run — the VOICE MAP threshold. Measured
#: on voice notes: at 0.55 the two hardest pairs merge, at 0.75
#: one frequent speaker shatters into ten groups with no gain.
#:
#: **0.94** — the third step of the same motion (0.65 → 0.71 → 0.94).
#: The reason is in the data: it shifted from voice notes to game audio
#: from very long recordings, tens of thousands of voices from about a
#: thousand files. There the common channel lifts every similarity, and the
#: map clumps: at 0.71 the biggest tile held over 40% of the inventory in
#: ONE group. The same direction shown by ``WITHIN_FILE`` (0.62)
#: versus ``ACROSS_FILES`` (0.55) and the gallery's 0.75.
#:
#: NOT to be confused with ``assignment_cut``: the assignment path computes
#: on speaker centroids with complete linkage and thus stays at 0.71, where
#: the map with average linkage already clusters.
THRESHOLD_GROUPING = 0.94

#: The values above are **measurement results**, not natural constants — and
#: the material matters. Listening put 0.62 as the best value for game
#: audio within a recording, for voice notes the grouping run measures 0.65.
#: So the ones a feature asks for are configurable, with the measured values
#: as defaults. ACROSS_FILES and WITHIN_FILE are not: no code path compares
#: two single voices with a threshold, and their settings changed nothing.
_THRESHOLDS = {
    "gallery": ("speaker.threshold_gallery", THRESHOLD_GALLERY),
    "group_voices": ("speaker.threshold_grouping", THRESHOLD_GROUPING),
}


def threshold(kind: str) -> float:
    """The configured threshold — or the measured default.

    ``kind`` is ``gallery`` or ``group_voices``. Why not one threshold:
    they answer different questions and so sit at different heights (the
    single-voice values ACROSS_FILES 0.55 and WITHIN_FILE 0.62 above show
    the same direction).

    * **group_voices** (0.94) — the cutoff in the grouping run over
      inventory: the voice map's starting threshold.
    * **gallery** (0.75) — one recording against ALL named people. Must
      be high because with many people and many windows there are
      hundreds of chances to cross the threshold by chance.

    An unusable value in config is silently ignored — a bad number can't
    cost a run, and the default is measured.
    """
    path, standard = _THRESHOLDS.get(kind, (None, 0.55))
    if not path:
        return standard
    try:
        from whisper_tm.config import config

        value = float(config.get(*path.split("."), default=standard))
    except (TypeError, ValueError, Exception):
        return standard
    return value if 0.0 < value <= 1.0 else standard


def nrm(A):
    """L2 normalization — **one version**, for core AND tools.

    There were thirteen copies in two flavors: additive with
    ``1e-9`` in ``float32``, and clamping to ``1e-12`` in ``float64``,
    some with ``axis=1`` instead of ``axis=-1``.

    Numerically the difference is tiny — and exactly why it's dangerous.
    It sits in the third decimal place, where thresholds 0.55,
    0.62 and 0.75 decide. And ``axis=1`` silently breaks on a
    single vector because it only has one axis.

    Computed in ``float32``: that's how vectors sit in the database,
    and conversion to ``float64`` pretends an accuracy the numbers don't have.
    """
    A = np.asarray(A, dtype=np.float32)
    return A / (np.linalg.norm(A, axis=-1, keepdims=True) + 1e-9)


# ── Process diarization ─────────────────────────────────────────────────


def normalize_turns(diar_segs: Any) -> list[dict]:
    """Convert whatever diarization produces into dicts.

    WhisperX returns a ``pandas.DataFrame`` with ``start``/``end``/``speaker``,
    tools in the ``tools/`` folder work with lists of dicts,
    and a pyannote ``Annotation`` can be iterated. All three should
    fit here so production code and analysis tools share the same core.
    """
    if diar_segs is None:
        return []
    out: list[dict] = []
    # pandas.DataFrame
    if hasattr(diar_segs, "iterrows"):
        for _, r in diar_segs.iterrows():
            try:
                out.append(
                    {
                        "start": float(r["start"]),
                        "end": float(r["end"]),
                        "speaker": str(r["speaker"]),
                    }
                )
            except (KeyError, TypeError, ValueError):
                continue
        return out
    # pyannote.core.Annotation
    if hasattr(diar_segs, "itertracks"):
        for seg, _, label in diar_segs.itertracks(yield_label=True):
            out.append({"start": float(seg.start), "end": float(seg.end), "speaker": str(label)})
        return out
    for d in diar_segs:
        try:
            out.append(
                {"start": float(d["start"]), "end": float(d["end"]), "speaker": str(d["speaker"])}
            )
        except (KeyError, TypeError, ValueError):
            continue
    return out


#: Width of a bucket of :func:`_near_index` in seconds
_BUCKET_SEC = 30.0


def _near_index(spans: Sequence[tuple[float, float]], pad: float):
    """Which of ``spans`` lie within ``pad`` of a time range — fast.

    Holding every segment of a recording against every turn was quadratic:
    10,000 × 10,000 steps for a recording of many hours, measured 44 s in
    the worker. Each span goes into the 30-s buckets its
    range (widened by ``pad``) touches; a query reads its own buckets.

    Returns ``near(a, b)``: the indices of every span that can overlap
    ``[a, b]`` or lie within ``pad`` of it — a superset, in ascending order,
    so a loop over them adds up and breaks ties exactly as the loop over
    all spans did.
    """
    import math

    buckets: dict[int, list[int]] = {}
    always: list[int] = []  # no finite range: asked about every time
    for i, (a, b) in enumerate(spans):
        lo, hi = min(a, b) - pad, max(a, b) + pad
        if not (math.isfinite(lo) and math.isfinite(hi)):
            always.append(i)
            continue
        for k in range(math.floor(lo / _BUCKET_SEC), math.floor(hi / _BUCKET_SEC) + 1):
            buckets.setdefault(k, []).append(i)

    def near(a: float, b: float) -> list[int]:
        lo, hi = min(a, b) - pad, max(a, b) + pad
        if not (math.isfinite(lo) and math.isfinite(hi)):
            return list(range(len(spans)))
        found: set[int] = set(always)
        for k in range(math.floor(lo / _BUCKET_SEC), math.floor(hi / _BUCKET_SEC) + 1):
            found.update(buckets.get(k, ()))
        return sorted(found)

    return near


def _overlap_by_label(
    turns: Sequence[dict], labelled: Sequence[tuple[str, float, float]]
) -> dict[str, dict[str, float]]:
    """``{turn speaker: {transcript speaker: seconds of overlap}}``."""
    near = _near_index([(a, b) for _sp, a, b in labelled], 0.0)
    overlap: dict[str, dict[str, float]] = {}
    for t in turns:
        acc = overlap.setdefault(t["speaker"], {})
        for j in near(t["start"], t["end"]):
            sp, a, b = labelled[j]
            ov = min(t["end"], b) - max(t["start"], a)
            if ov > 0:
                acc[sp] = acc.get(sp, 0.0) + ov
    return overlap


def align_labels(
    turns: Sequence[dict], transcript: Sequence[tuple[str, float, float]]
) -> tuple[list[dict], dict]:
    """Name a fresh diarization's speakers after an existing transcript's.

    A transcription stores its voices under the same labels its segments
    carry — one diarization feeds both, so ``speaker_vectors.label`` and
    ``segments.speaker`` match by construction, and "only the sentences of
    person X" relies on it. A voices-only run diarizes the audio again, and
    for a transcript diarized earlier (other separation, older version) the
    fresh run numbers the same people differently: measured on four
    library files, three came back with swapped or shifted labels.

    The audio, the turns and therefore the windows stay exactly as the
    diarization delivered them — only the names change. Each new label
    goes to the transcript speaker it overlaps most, one-to-one, largest
    overlap first, and only if that speaker holds the majority of the
    label's overlapped time. A label without such a counterpart gets a
    free label the transcript does not use: better no link to sentences
    than a link to someone else's.

    ``transcript``: ``(speaker, start, end)`` per segment. Without any
    speaker in it (never diarized) the turns come back unchanged.
    Returns ``(turns, info)`` with ``info`` = ``{"aligned": {new: old},
    "unmatched": {new: free label}}`` or ``{}`` when nothing was compared.
    """
    labelled = [
        (str(sp), float(a), float(b))
        for sp, a, b in transcript
        if sp and str(sp) not in ("—", "-") and b > a
    ]
    turns = [dict(t) for t in turns]
    if not labelled or not turns:
        return turns, {}
    overlap = _overlap_by_label(turns, labelled)
    pairs = sorted(
        ((ov, new, old) for new, acc in overlap.items() for old, ov in acc.items()), reverse=True
    )
    mapping: dict[str, str] = {}
    taken: set[str] = set()
    for ov, new, old in pairs:
        if new in mapping or old in taken:
            continue
        if ov < 0.5 * sum(overlap[new].values()):
            continue  # no majority — not this speaker
        mapping[new] = old
        taken.add(old)
    used = {sp for sp, _a, _b in labelled} | taken
    numbers = [int(x.rsplit("_", 1)[1]) for x in used if x.rsplit("_", 1)[-1].isdigit()]
    nxt = max(numbers, default=-1) + 1
    unmatched: dict[str, str] = {}
    for new in sorted(overlap):
        if new in mapping:
            continue
        while f"SPEAKER_{nxt:02d}" in used:
            nxt += 1
        unmatched[new] = f"SPEAKER_{nxt:02d}"
        used.add(unmatched[new])
        nxt += 1
    rename = {**mapping, **unmatched}
    for t in turns:
        t["speaker"] = rename.get(t["speaker"], t["speaker"])
    return turns, {"aligned": mapping, "unmatched": unmatched}


def label_segments(
    segments: Sequence[tuple[float, float]], turns: Sequence[dict], reach: float = 1.0
) -> list[str]:
    """The speaker of each transcript segment, from a diarization's turns.

    For a transcript made without telling speakers apart: a voices-only run
    diarizes it now, and its sentences get the labels its voices carry —
    so the voice map (with its threshold) can name the speakers.
    A segment goes to the speaker it overlaps most; one between turns (a
    pause the diarization cut out) to the nearest within ``reach`` seconds;
    otherwise it stays "—". ``segments``: ``(start, end)`` each.
    """
    out = []
    # Only the turns that can matter — within ``reach`` (_near_index)
    nearby = _near_index([(t["start"], t["end"]) for t in turns], reach)
    for a, b in segments:
        acc: dict[str, float] = {}
        near, gap = "—", reach
        for k in nearby(a, b):
            t = turns[k]
            ov = min(t["end"], b) - max(t["start"], a)
            if ov > 0:
                acc[t["speaker"]] = acc.get(t["speaker"], 0.0) + ov
            else:
                d = -ov  # the distance between them
                if d <= gap:
                    near, gap = t["speaker"], d
        out.append(max(acc, key=acc.get) if acc else near)
    return out


def clean_spans(turns: Sequence[dict]) -> dict[str, list[tuple[float, float]]]:
    """Per speaker the times when ONLY they are active.

    Overlap is the exclusion criterion, not duration. At full
    overlap the cosine to the louder speaker is 0.78 and to the
    quieter is 0.22 (Cord-Landwehr et al., arXiv 2306.00634) — the vector
    simply becomes the louder one. Dropping overlapped frames measurably
    beats including them, EVEN THOUGH less audio remains (Horiguchi, ICASSP 2025).
    It costs 7–11% of speech.

    Adjacent pieces are re-concatenated, else a continuous speech would shatter
    at every segment boundary into snippets — which would then fail the
    minimum length gate.
    """
    limits = sorted({t for d in turns for t in (d["start"], d["end"])})
    out: dict[str, list[tuple[float, float]]] = {}
    for a, b in zip(limits, limits[1:]):
        if b - a < MIN_SPAN:
            continue
        middle = (a + b) / 2
        active = {d["speaker"] for d in turns if d["start"] <= middle < d["end"]}
        if len(active) != 1:  # silence or overlap
            continue
        sp = active.pop()
        if out.get(sp) and abs(out[sp][-1][1] - a) < 0.05:
            out[sp][-1] = (out[sp][-1][0], b)
        else:
            out.setdefault(sp, []).append((a, b))
    return out


def build_windows(
    spans: Sequence[tuple[float, float]], at_most: int = MAX_WINDOWS
) -> list[tuple[list[tuple[float, float]], float, float]]:
    """ONE speaker's speech into windows of :data:`MAX_SPEECH` each.

    Returns ``[(parts, duration, start_sec), …]``; ``parts`` are the
    time spans concatenated into this window.

    Two things about this are measured and easy to get wrong:

    **Concatenation is an advantage, not a necessary evil.** Concatenated 1.48%
    vs unconcatenated 4.41% EER at equal duration. 30 seconds in one
    chunk is a snapshot, 30 seconds from five places is the breadth of a person.

    **The fallback at the minimum-length gate matters.** If the filtered
    amount isn't enough for a full window, everything is taken again.
    Otherwise quiet speakers would disappear entirely — and those are exactly
    who we want to find.
    """
    long_spans = [(a, b) for a, b in spans if b - a >= MIN_SEGMENT]
    if sum(b - a for a, b in long_spans) >= MAX_SPEECH:
        spans = long_spans

    out: list[tuple[list[tuple[float, float]], float, float]] = []
    parts: list[tuple[float, float]] = []
    duration, start = 0.0, None
    for a, b in spans:
        remainder = b - a
        while remainder >= 0.5:
            length = min(remainder, MAX_SPEECH - duration)
            here = a + (b - a - remainder)
            if start is None:
                start = here
            parts.append((here, here + length))
            duration += length
            remainder -= length
            if duration >= MAX_SPEECH:
                out.append((parts, duration, start))
                parts, duration, start = [], 0.0, None
    if duration >= MIN_SPEECH:
        out.append((parts, duration, start))

    if not at_most or len(out) <= at_most:
        return out
    # Thin out evenly instead of cutting from the front — else we're back to
    # "always the start", which was the error that once hid a quiet speaker.
    step = len(out) / at_most
    return [out[int(i * step)] for i in range(at_most)]


# ── Model ──────────────────────────────────────────────────────────────────


def load_model(device: str = "cuda", job=None) -> Callable[[Any], np.ndarray]:
    """ReDimNet M, checkpoint ``vb2``/``ptn`` — waveform ``[1, T]`` → vector.

    Runs through the same resident cache as Whisper and pyannote: once
    loaded, stays loaded until the idle cleaner or an explicit
    ``release_models()`` removes it. The import is deliberately in the
    function — ``transcription`` imports this module, the reverse would create
    a cycle.
    """
    import torch

    from whisper_tm.core.transcription import _resident_model

    def build():
        from whisper_tm.vendor.redimnet.model import ReDimNetWrap

        state = _redimnet_weights()
        m = ReDimNetWrap(**state["model_config"])
        loaded = m.load_state_dict(state["state_dict"])
        if loaded.missing_keys or loaded.unexpected_keys:
            raise RuntimeError(f"The voice model's weights don't fit its code: {loaded}")
        m.eval().to(device)
        return m

    model = _resident_model(
        ("speaker_embed", device), build, job=job, label="Voice vector (ReDimNet M)"
    )

    def compute_it(waveform) -> np.ndarray:
        with torch.no_grad():
            return model(waveform.to(device)).cpu().numpy().ravel()

    return compute_it


class VoiceWeightsUnavailable(RuntimeError):
    """The voice model's weights are not on this machine and can't be
    downloaded (offline, blocked, server gone)."""


def _redimnet_weights() -> dict:
    """The voice model's weights — downloaded once into torch's checkpoint
    folder (where torch.hub kept them before: an installation has them
    already) and checked against :data:`REDIMNET_SHA256` before they are
    loaded, as plain tensors (``weights_only``)."""
    import hashlib

    import torch

    target = Path(torch.hub.get_dir()) / "checkpoints" / REDIMNET_URL.rsplit("/", 1)[1]
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            torch.hub.download_url_to_file(
                REDIMNET_URL, str(target), hash_prefix=REDIMNET_SHA256, progress=False
            )
        except (OSError, ValueError) as e:  # URLError is an OSError
            raise VoiceWeightsUnavailable(
                f"the voice model's weights ({target.name}, 19 MB) are not downloaded and "
                f"can't be fetched now ({str(e)[:160]}) — `--prefetch` gets them while online"
            ) from e
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    if digest != REDIMNET_SHA256:
        raise RuntimeError(
            f"The voice model's weights {target} are not the expected ones "
            f"(SHA-256 {digest[:16]}…, expected {REDIMNET_SHA256[:16]}…) — "
            f"delete the file to download it again"
        )
    return torch.load(str(target), map_location="cpu", weights_only=True)


def window_separator(work, job=None) -> Callable[[Any], Any] | None:
    """A separator that touches ONLY the 30-second windows.

    The case for which this exists: the user doesn't want **vocal isolation**
    for a recording — it costs most of the pipeline's compute time — but the
    voice vectors should still be comparable with all others. Without
    separation they wouldn't be: for vectors separation is proven (100% vs 89%
    hit rate, confirmed on several hundred pairs).

    This works because window-by-window separation and full-track separation
    deliver **practically identical results** — similarity 0.984 vs
    0.994. It's purely a time question, and 30 seconds vs seven hours
    changes the answer.

    The prefilter decides per call whether separation is needed at all; on
    clean material it skips half.

    Returns ``None`` if separation is unavailable — then raw vectors stay,
    and that's recorded in the database.
    """
    try:
        from whisper_tm.core.vocal_isolation import isolate_vocals
    except Exception as e:
        logger.warning("Window separation not available: %s", str(e)[:120])
        return None

    import subprocess
    import tempfile

    import torchaudio

    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)

    def separate(waveform):
        """Return the separated waveform — or ``None``.

        ``None`` means: this window gets NO vector. Falling back to raw
        audio would be the easier way and the wrong one: the vector would
        end up marked "separated" in the database when it's not,
        and no one could see it later. A missing window is a hole; a
        mislabeled one is poison.
        """
        # Temp is in the job's work directory, NOT in /tmp. That may
        # be a tmpfs in RAM.
        folder = Path(tempfile.mkdtemp(dir=str(work)))
        try:
            raw = folder / "windows.wav"
            torchaudio.save(str(raw), waveform, SR)
            vocals, _ = isolate_vocals(raw, folder, job=job, model_key="mel_roformer_kim")
            if not vocals or not Path(vocals).exists():
                logger.warning("Window separation returned no vocals — window skipped")
                return None
            fine = folder / "window16.wav"
            r = subprocess.run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-y",
                    "-i",
                    str(vocals),
                    "-ar",
                    str(SR),
                    "-ac",
                    "1",
                    str(fine),
                ],
                capture_output=True,
                timeout=300,
            )
            if r.returncode != 0 or not fine.exists():
                logger.warning("Window separation: conversion failed — window skipped")
                return None
            new_value, _ = torchaudio.load(str(fine))
            return new_value
        except InterruptedError:
            raise  # cancelled (also while waiting for the GPU)
        except Exception as e:
            if _is_out_of_memory(e):
                # Not a hole in this window: the card is full, which the
                # queue's OOM retry handles — skipped, every window of the
                # file ended as "separation failed" and the entry as error.
                raise
            logger.warning("Window separation failed, window skipped: %s", str(e)[:120])
            return None
        finally:
            import shutil

            shutil.rmtree(folder, ignore_errors=True)

    return separate


def _waveform(audio: np.ndarray, parts: Sequence[tuple[float, float]], sr: int = SR):
    """Cut time spans from audio and concatenate them.

    The audio is already 16kHz mono in memory at this point —
    diarization loaded it. So here **no ffmpeg** is called and **no file**
    is created: no decoding per window, no temp folder, and thus not the trap
    that ``/tmp`` may be a tmpfs in RAM.
    """
    import torch

    pieces = []
    for a, b in parts:
        i, j = int(a * sr), int(b * sr)
        if j > i:
            pieces.append(audio[i:j])
    if not pieces:
        return None
    w = np.concatenate(pieces).astype(np.float32)
    return torch.from_numpy(w)[None]


# ── Running over one recording ─────────────────────────────────────────────


def plan_windows(turns: Sequence[dict], at_most: int = MAX_WINDOWS) -> list[dict]:
    """Diarization → window plan, without touching audio.

    Separated from computation so the plan can be checked (and tested) without
    needing a GPU.
    """
    spans = clean_spans(turns)
    plan: list[dict] = []
    for label in sorted(spans):
        for nr, (parts, duration, start) in enumerate(build_windows(spans[label], at_most), 1):
            if duration < MIN_SPEECH:
                continue
            plan.append(
                {
                    "label": label,
                    "windows": nr,
                    "start_sec": round(float(start or 0.0), 2),
                    "speech_sec": round(float(duration), 2),
                    "segments": len(parts),
                    "parts": parts,
                }
            )
    plan.sort(key=lambda p: (p["start_sec"], p["label"]))
    return plan


def _is_out_of_memory(e: BaseException) -> bool:
    """Is this a GPU out-of-memory error? (Text because torch doesn't need to
    be imported and OOM classes shift per version.)"""
    return "out of memory" in str(e).lower()


def _clear_gpu_cache() -> None:
    try:
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            time.sleep(1.0)
    except Exception:
        pass


def compute_vectors(
    audio: np.ndarray,
    plan: Sequence[dict],
    device: str = "cuda",
    job=None,
    abort: Callable[[], bool] | None = None,
    column_name: str = "v_separated",
    separate: Callable[[Any], Any] | None = None,
) -> list[dict]:
    """Compute one vector for each planned window.

    ``column_name`` says which audio version this is: in production separation runs
    first and replaces ``audio.wav`` — what arrives here is thus
    **separated** audio. A re-processing run without separation writes
    to ``v_raw`` instead.

    **A window missing for technical reasons invalidates the whole run.**
    Earlier one error cost only that window — and exactly so files with a
    few of their vectors passed as
    ✅ while the GPU was full (another program), most windows skipped with
    "CUDA out of memory". A recording with a fraction of its voices is practically
    invisible in the voice map, and no one sees it. Hence: if any substep
    fails, the whole job must fail — else it's invalid.

    But first rescue what can be rescued: on out-of-memory the cache is cleared
    and the window retried; if it sticks, the remainder of the file computes on CPU
    (ReDimNet M is small, costs seconds). Only if that fails does the error fly —
    with original text so the queue recognizes OOM as OOM and retries once with
    VRAM-friendly settings.

    Windows with no speech or below :data:`MIN_SPEECH` are not errors:
    they're already filtered out in :func:`plan_windows` and don't appear here.
    """
    if not plan:
        return []
    from whisper_tm.core.gpu_gate import GATE

    # One file at a time on the model (queue lanes, core/gpu_gate.py);
    # window separation inside takes the separator's stage in addition.
    with GATE.stage("vectors", job=job):
        return _compute_vectors(
            audio,
            plan,
            device=device,
            job=job,
            abort=abort,
            column_name=column_name,
            separate=separate,
        )


def _compute_vectors(
    audio, plan, device="cuda", job=None, abort=None, column_name="v_separated", separate=None
) -> list[dict]:
    compute_it = load_model(device, job=job)
    out: list[dict] = []
    skipped = 0
    evaded = False
    t0 = time.time()
    for i, p in enumerate(plan):
        if abort is not None and abort():
            # Raise, never return what there is so far: the caller would
            # store it in place of the recording's complete set — a
            # cancelled run is no run, like one with holes.
            logger.info("Voice vectors aborted at %d/%d", i, len(plan))
            raise InterruptedError("Cancelled")
        w = _waveform(audio, p["parts"])
        if w is None:
            continue
        if separate is not None:
            w = separate(w)
            # None means: separation of this window failed. Then
            # there's no vector here — not a raw one under false
            # description.
            if w is None:
                skipped += 1
                continue
        try:
            v = compute_it(w)
        except Exception as e:
            if not _is_out_of_memory(e) or evaded:
                raise RuntimeError(
                    f"Voice vector {p['label']}#{p['windows']}: {str(e)[:300]}"
                ) from e
            logger.warning(
                "Voice vector %s#%d: %s — clearing cache, second attempt",
                p["label"],
                p["windows"],
                str(e)[:120],
            )
            _clear_gpu_cache()
            try:
                v = compute_it(w)
            except Exception as e2:
                if not _is_out_of_memory(e2):
                    raise RuntimeError(
                        f"Voice vector {p['label']}#{p['windows']}: {str(e2)[:300]}"
                    ) from e2
                logger.warning(
                    "Voice vectors: GPU stays full — remainder of this file computes on CPU"
                )
                if job is not None:
                    from whisper_tm.extensions import job_manager

                    job_manager.warn(job, "Voice vectors: GPU full, falling back to CPU")
                compute_it = load_model("cpu", job=job)
                evaded = True
                try:
                    v = compute_it(w)
                except Exception as e3:
                    raise RuntimeError(
                        f"Voice vector {p['label']}#{p['windows']} also fails on "
                        f"CPU: {str(e3)[:300]}"
                    ) from e3
        out.append(
            {
                "label": p["label"],
                "windows": p["windows"],
                "start_sec": p["start_sec"],
                "speech_sec": p["speech_sec"],
                "segments": p["segments"],
                # The spans from which this vector arose. They
                # were once discarded here — and without them
                # the sample plays something else than the vector
                # saw (see schema comment on `parts`).
                "parts": p.get("parts"),
                column_name: v,
            }
        )
    if skipped:
        # A hole is better than a mislabeled vector — but a run with holes
        # is not a valid run (see docstring).
        raise RuntimeError(
            f"Voice vectors: window-by-window separation failed on {skipped} of {len(plan)} windows"
        )
    logger.info("Voice vectors: %d of %d windows in %.1f s", len(out), len(plan), time.time() - t0)
    return out


def carry_along(
    abs_path: str,
    audio: np.ndarray,
    diar_segs: Any,
    device: str = "cuda",
    job=None,
    abort: Callable[[], bool] | None = None,
    separated: bool = True,
    work=None,
    window_separation: bool = True,
    only_separated: bool = True,
    keep_decisions: bool = False,
) -> dict:
    """The one call for the transcription worker.

    Takes the audio already in memory and the diarization that just ran — and
    stores the vectors in ``speaker.db``. Returns a small summary for the
    job log.

    **The invariant: prefer no vector over a raw one.**

    When vectors are created, vocal isolation must be on — vectors must not
    be created wrong; with isolation on for 5 files and off for the next 5,
    nothing matches.

    That's exactly what would happen. A raw and a separated vector of the same
    person lie further apart than two separated vectors of different people —
    the inventory would split in half, unable to find each other. And
    the error would be invisible: both columns have the same shape, both
    numbers look plausible, only noticed when someone wonders why nothing
    matches.

    So the production path **never** writes to ``v_raw``. If audio isn't
    separated, window-by-window separation is caught up (:func:`window_separator`)
    — that delivers practically the same vector as full-track separation
    (0.984/0.994). If that fails too, **nothing** is written and
    the reason logged.

    ``only_separated=False`` lifts the bar. This is only for measurement runs
    and never belongs in the production path.

    ``keep_decisions``: the recording's voices are renewed, and the names,
    the trash and the history of the old ones go to the new voices that
    are the same people — or, where that isn't clear, the recording keeps
    its old voices (:meth:`SpeakerDB.replace_keeping_decisions`). The
    outcome is under ``renewal``.
    """
    from whisper_tm.database.speaker import speaker_db

    turns = normalize_turns(diar_segs)
    if not turns:
        return {"windows": 0, "labels": 0, "reason": "no diarization"}
    plan = plan_windows(turns)
    if not plan:
        return {
            "windows": 0,
            "labels": 0,
            "reason": f"no speaker with {MIN_SPEECH:.0f} s of clean speech",
        }

    separate = None
    if not separated and window_separation and work is not None:
        separate = window_separator(work, job=job)
        if separate is not None:
            separated = True

    if not separated and only_separated:
        return {
            "windows": 0,
            "labels": 0,
            "reason": "no vectors are written without separation — "
            "raw and separated vectors are not comparable",
        }

    lines = compute_vectors(
        audio,
        plan,
        device=device,
        job=job,
        abort=abort,
        column_name="v_separated" if separated else "v_raw",
        separate=separate,
    )
    if not lines:
        return {"windows": 0, "labels": 0, "reason": "no vector computed"}

    # The voices as the sentence check holds them (check_sentences) —
    # computed from what was just measured, stored or not
    centroids = label_centroids(lines, "v_separated" if separated else "v_raw")
    if keep_decisions:
        renewal = speaker_db().replace_keeping_decisions(abs_path, MODEL_BRAND, lines)
        if not renewal["replaced"]:
            return {
                "windows": 0,
                "labels": 0,
                "renewal": renewal,
                "model": MODEL_BRAND,
                "centroids": centroids,
                "reason": "renewal dropped — an old voice with a name, the trash "
                "or history has no clear counterpart; the recording "
                "keeps its voices",
            }
    else:
        renewal = None
        speaker_db().write(abs_path, MODEL_BRAND, lines)
    return {
        **({"renewal": renewal} if renewal else {}),
        "windows": len(lines),
        "labels": len({z["label"] for z in lines}),
        "speech_sec": round(sum(z["speech_sec"] for z in lines), 1),
        "model": MODEL_BRAND,
        "variant": "separated" if separated else "raw",
        "centroids": centroids,
    }


# ── Checking each sentence ─────────────────────────────────────────────────

#: Shortest sentence the check judges. Below a second the vector is mostly
#: noise: a one-word answer scores only about 0.3 against every voice of
#: its recording.
CHECK_MIN_SEC = 1.0

#: How similar a sentence must be to the voice it moves to. Below this no
#: voice of the recording is clearly it, and the diarization's word stands.
CHECK_MIN_SIM = 0.60

#: How far that voice must lead the sentence's own label. The diarization
#: heard the sentence in context, the check hears it alone — a close call
#: goes to the diarization. Measured on hand-corrected sentences: wrongly
#: labelled ones led by well over 0.1, while a correct one that came close
#: trailed by a few hundredths.
CHECK_MARGIN = 0.10

_CHECK_SETTINGS = {
    "min_sec": ("speaker", "check_min_sec", CHECK_MIN_SEC),
    "min_sim": ("speaker", "check_min_sim", CHECK_MIN_SIM),
    "margin": ("speaker", "check_margin", CHECK_MARGIN),
}


def check_settings() -> dict[str, float]:
    """The sentence check's three numbers — configured or measured.

    An unusable value falls back to the default, as with :func:`threshold`.
    """
    out = {}
    for key, (section, name, standard) in _CHECK_SETTINGS.items():
        try:
            from whisper_tm.config import config

            value = float(config.get(section, name, default=standard))
        except Exception:
            value = standard
        out[key] = value if 0.0 <= value <= (60.0 if key == "min_sec" else 1.0) else standard
    return out


def label_centroids(lines: Sequence[dict], column: str = "v_separated") -> dict[str, np.ndarray]:
    """One unit vector per label — the mean of its windows' unit vectors.

    Only for the sentence check (:func:`check_sentences`): it asks which of
    the recording's voices a sentence sounds like, and a voice's windows
    together answer that better than any single one. Map and recognition
    keep comparing single windows (see the module docstring).
    """
    by: dict[str, list[np.ndarray]] = {}
    for z in lines:
        v = z.get(column)
        if v is None:
            continue
        v = np.asarray(v, np.float32).ravel()
        if v.size and np.isfinite(v).all() and np.linalg.norm(v) > 0:
            by.setdefault(str(z["label"]), []).append(nrm(v))
    return {lb: nrm(np.mean(vs, axis=0)) for lb, vs in by.items()}


def check_sentences(
    audio: np.ndarray,
    segments: Sequence[dict],
    centroids: dict[str, np.ndarray],
    embed: Callable[[Any], np.ndarray],
    min_sec: float = CHECK_MIN_SEC,
    min_sim: float = CHECK_MIN_SIM,
    margin: float = CHECK_MARGIN,
    abort: Callable[[], bool] | None = None,
) -> list[dict]:
    """Which sentences the diarization gave to the wrong voice.

    The diarization decides per stretch of audio; where two people trade
    short sentences it can keep them under one label, and every sentence
    in that stretch carries the name of whoever that label is. Here each
    sentence is measured on its own against the voices of the SAME
    recording (``centroids``, per label) — names don't matter, so this
    works before anyone was named and after all names are gone.

    A sentence moves only when another voice is clearly it: at least
    ``min_sim`` similar and ``margin`` ahead of its own label. Sentences
    shorter than ``min_sec``, without a label, or with a label that has
    no voice (too little clean speech for a window) are left alone.

    ``audio`` is the 16-kHz audio the centroids were measured on — the
    separated one when they are separated. ``segments`` need ``start``,
    ``end`` and ``speaker``; nothing is changed here.

    Returns one entry per sentence to move: ``index``, ``start``, ``end``,
    ``old``, ``new``, ``sim`` (to the new voice) and ``was`` (to the old).
    """
    import torch

    labels = list(centroids)
    if len(labels) < 2:
        return []
    C = np.stack([centroids[lb] for lb in labels]).astype(np.float32)
    out: list[dict] = []
    for i, seg in enumerate(segments):
        if abort is not None and i % 200 == 0 and abort():
            raise InterruptedError("Cancelled")
        old = seg.get("speaker")
        if old not in centroids:
            continue
        try:
            a, b = float(seg["start"]), float(seg["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if b - a < min_sec:
            continue
        x = audio[max(0, int(a * SR)) : max(0, int(b * SR))]
        if len(x) < int(min_sec * SR):
            continue  # past the end of the audio
        v = np.asarray(
            embed(torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))[None]), np.float32
        ).ravel()
        if not np.isfinite(v).all() or np.linalg.norm(v) == 0:
            continue
        sims = C @ nrm(v)
        k = int(np.argmax(sims))
        was = float(sims[labels.index(old)])
        if labels[k] != old and sims[k] >= min_sim and sims[k] - was >= margin:
            out.append(
                {
                    "index": i,
                    "start": round(a, 2),
                    "end": round(b, 2),
                    "old": old,
                    "new": labels[k],
                    "sim": round(float(sims[k]), 3),
                    "was": round(was, 3),
                }
            )
    return out


def recheck_sentences(
    audio: np.ndarray,
    segments: list[dict],
    centroids: dict[str, np.ndarray],
    device: str = "cuda",
    job=None,
    abort: Callable[[], bool] | None = None,
) -> dict:
    """:func:`check_sentences` on the GPU, and the moves applied.

    Each moved sentence takes the new label, and so do its words — the
    subtitle split later reads the words' speakers. Returns what to keep
    with the transcript: ``checked`` (sentences long enough to judge),
    ``moved`` and every move (start, old, new, sim, was) — all of them,
    so each can be seen and undone.
    """
    from whisper_tm.core.gpu_gate import GATE

    st = check_settings()
    t0 = time.time()
    with GATE.stage("vectors", job=job):
        embed = load_model(device, job=job)
        try:
            moves = check_sentences(
                audio,
                segments,
                centroids,
                embed,
                st["min_sec"],
                st["min_sim"],
                st["margin"],
                abort=abort,
            )
        except Exception as e:
            if not _is_out_of_memory(e):
                raise
            logger.warning("Sentence check: GPU full — computing on CPU")
            _clear_gpu_cache()
            moves = check_sentences(
                audio,
                segments,
                centroids,
                load_model("cpu", job=job),
                st["min_sec"],
                st["min_sim"],
                st["margin"],
                abort=abort,
            )
    for m in moves:
        seg = segments[m["index"]]
        seg["speaker"] = m["new"]
        for w in seg.get("words") or ():
            if isinstance(w, dict) and w.get("speaker") == m["old"]:
                w["speaker"] = m["new"]
    checked = sum(
        1
        for s in segments
        if s.get("speaker") in centroids
        and float(s.get("end") or 0) - float(s.get("start") or 0) >= st["min_sec"]
    )
    return {
        "checked": checked,
        "moved": len(moves),
        "settings": st,
        "secs": round(time.time() - t0, 1),
        "moves": [{k: m[k] for k in ("start", "old", "new", "sim", "was")} for m in moves],
    }


# ── Who was in this recording? ────────────────────────────────────────────


def _nrm(A):
    # Alias to the canonical version above — the short name stays because it's
    # used at many places in this module.
    return nrm(A)


def who_was_there(
    abs_path: str, threshold: float | None = None, min_windows: int = 1
) -> list[dict]:
    """The question that matters most — and its timestamps.

    Each window of the recording is held against named people;
    returns who exceeds the threshold, with the minutes it happened.

    **Decision is on individual voices, not centroids.** This is
    a correction: the old assumption ("centroid per person, measured
    +3 points") comes from pairwise comparison and doesn't apply here. Measured
    with leave-one-out over a few hundred files (``tools/speaker_quality.py``) the best
    individual value beats the centroid at **every** operating point:

    ============  =====================  =====================
    False Alarms  Centroid               Individual Voices
    ============  =====================  =====================
    ~4 %          0.65 → 96.5 % Hit      0.70 → 98.1 % Hit
    ~1 %          0.70 → 92.7 %          0.75 → 96.0 %
    0 %           0.80 → 75.7 %          0.78 → 91.1 %
    ============  =====================  =====================

    At zero false alarms there are 15 points between them. The centroid is
    still computed and returned — it's a second opinion, and
    its bias with count of averaged vectors (Sholokhov et al.,
    2302.09523) is exactly why it can't decide alone.

    **Accuracy** — is the best person also the right one? — is
    100.0% (every file) for individual voices vs 98.5% for centroid.
    It doesn't depend on the threshold.
    """
    from whisper_tm.core.speaker_inventory import inventory
    from whisper_tm.database.speaker import speaker_db

    if threshold is None:
        threshold = globals()["threshold"]("gallery")

    # Via process cache not database: the gallery is the same for every
    # request, and re-reading it from hundreds of thousands of BLOBs costs 1.7s vs 5.5ms
    # for computation.
    b = inventory()
    known = b["gallery"]
    db = speaker_db()
    windows = db.windows(abs_path)
    if not windows or not known:
        return []

    people = sorted(known)
    lines, meta = [], []
    for f in windows:
        i = b["id_to_row"].get(f["id"])
        if i is None:
            continue
        lines.append(i)
        meta.append(f)
    if not lines:
        return []
    Y = b["X"][lines]

    # Centroid per person: normalize each voice first, then average.
    # The other way, long recordings would dominate the centroid alone.
    Z = _nrm(np.stack([_nrm(known[p]).mean(0) for p in people]))
    E = {p: _nrm(known[p]) for p in people}
    S_hard = Y @ Z.T

    out = []
    for j, p in enumerate(people):
        # (Recording windows × person's voices) — the best individual value
        # per window decides.
        A = Y @ E[p].T
        best = A.max(axis=1)
        hits = [i for i in range(len(Y)) if best[i] >= threshold]
        if len(hits) < min_windows:
            continue
        out.append(
            {
                "person": p,
                "windows": len(hits),
                "of_windows": len(Y),
                "best_value": round(float(best[hits].max()), 3),
                "mean_value": round(float(best[hits].mean()), 3),
                # The second opinion, explicitly labeled as such: the
                # centroid judges differently and is often wrong.
                "centroid_value": round(float(S_hard[hits, j].max()), 3),
                "centroid_above_threshold": bool((S_hard[hits, j] >= threshold).any()),
                # How many of the person's voices agree? A person with 300
                # voices has 300 chances, one with three has three — the
                # ratio makes them comparable.
                "voices_share": round(float((A[hits] >= threshold).mean()), 3),
                "voices_total": int(len(E[p])),
                "timestamps": sorted({int(meta[i]["start_sec"] // 60) for i in hits}),
                "labels": sorted({meta[i]["label"] for i in hits}),
            }
        )
    out.sort(key=lambda r: -r["windows"])
    return out


def names_for_labels(abs_path: str, threshold: float | None = None) -> dict[str, dict]:
    """A name for each speaker of a recording — where one is clear.

    Two sources, the map first:

    * **The map**: windows of the label that carry a person — named there,
      in the guided assignment, or carried along when the voices were
      renewed. The name most of them carry, if it is more than half.
    * **The recognition** (as in "Who was there?"): every window of the
      label against every named voice of the library; a window that reaches
      the threshold (``speaker.threshold_gallery``, 0.75) votes for the
      person it is most alike. A name with at least two thirds of the votes.
      At 0.75 one window in 170 is wrong (who_was_there); a speaker with
      several windows that mostly agree is far safer than that.

    Two rules keep the recognition from writing wrong names:

    * **The recording does not vouch for itself.** Its own named windows
      are left out of the voices it is held against: the shared channel
      lifts every similarity inside a recording (0.62 there against 0.75
      across the gallery), and a second speaker cleared the threshold
      against the first one's windows and got their name.
    * **One name, one speaker.** A name the map gave stays with its label;
      the recognition hands each remaining name to at most one label — the
      one with the most votes for it.

    Returns ``{label: {"name", "source": "map" | "recognition", "windows",
    "of"}}`` — labels without a clear name are missing.
    """
    from collections import Counter, defaultdict

    from whisper_tm.core.speaker_inventory import gallery_ids, inventory
    from whisper_tm.database.speaker import fold_name, speaker_db

    if threshold is None:
        threshold = globals()["threshold"]("gallery")
    db = speaker_db()
    windows = db.windows(abs_path)
    by_label: dict[str, list[dict]] = defaultdict(list)
    for w in windows:
        by_label[w["label"]].append(w)
    out: dict[str, dict] = {}
    for label, ws in by_label.items():
        named = Counter(w["person"] for w in ws if w.get("person"))
        if named:
            person, n = named.most_common(1)[0]
            if 2 * n > sum(named.values()):
                out[label] = {"name": person, "source": "map", "windows": n, "of": len(ws)}
    rest = [w for w in windows if w["label"] not in out]
    if not rest:
        return out
    b = inventory()
    # The held gallery without this recording's own voices (see above) —
    # masked by vector id, not read again from the database
    own = np.fromiter((w["id"] for w in windows), dtype=np.int64, count=len(windows))
    ids = gallery_ids(b)
    known = {}
    for p, V in b["gallery"].items():
        if p in ids and len(ids[p]) == len(V):
            V = V[~np.isin(ids[p], own)]
        if len(V):
            known[p] = V
    rows = [(w, b["id_to_row"].get(w["id"])) for w in rest]
    rows = [(w, i) for w, i in rows if i is not None]
    if not known or not rows:
        return out
    people = sorted(known)
    Y = b["X"][[i for _w, i in rows]]
    # windows × people: each window's best value against each person's voices
    best = np.stack([(Y @ known[p].T).max(axis=1) for p in people], axis=1)
    votes: dict[str, Counter] = defaultdict(Counter)
    for (w, _i), j, v in zip(rows, best.argmax(axis=1), best.max(axis=1), strict=True):
        if v >= threshold:
            votes[w["label"]][people[j]] += 1
    candidates = []
    for label, c in votes.items():
        person, n = c.most_common(1)[0]
        if 3 * n >= 2 * sum(c.values()):
            candidates.append((n, label, person))
    # One label per name: the map's names are taken, then the most votes
    # win (ties: the label first in order, deterministically)
    taken = {fold_name(d["name"]) for d in out.values()}
    for n, label, person in sorted(candidates, key=lambda t: (-t[0], t[1])):
        if fold_name(person) in taken:
            continue
        taken.add(fold_name(person))
        out[label] = {
            "name": person,
            "source": "recognition",
            "windows": n,
            "of": len(by_label[label]),
        }
    return out
