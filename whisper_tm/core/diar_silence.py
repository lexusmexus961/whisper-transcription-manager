"""Keep silence out of diarization — and calculate back the timeline.

**Why.** Diarization has become the most expensive step in the pipeline
on very long recordings, ahead of separation.
It runs over the entire file — even over hours of guaranteed silence.

On a long, day-length recording, silencedetect and the pre-filter can
independently find around half of it silent. Whoever cuts out the
silence then roughly halves the step.

**Why in ONE piece and not chunk by chunk.** It would seem natural to
diarize speech sections individually. This is wrong: pyannote assigns
its identifiers anew for each call. "SPEAKER_00" in the third section would not
be the same person as in the seventh — speaker identity across the file
would fall apart, and exactly that is the purpose of the step. So: concatenate all speech parts,
diarize ONCE, then calculate back the times.

**Calculating back is the actual danger.** An offset in the
timeline goes unnoticed — speakers simply end up at the
wrong places in the transcript. Therefore the mapping here is a pure
function with its own tests, and a section that crosses an interface
is correctly divided into two pieces.
"""

from __future__ import annotations

import logging
import re
import subprocess

logger = logging.getLogger("whisper_tm.core.diar_process")

#: From this length a cut is worthwhile. Shorter silences bring nothing and
#: only make the ffmpeg selection long.
MIN_SILENCE_S = 30.0

#: Two speech parts that are less than this far apart are pulled
#: together into one. This keeps the number of parts small without
#: significantly recovering silence.
GLUE_S = 2.0

#: More parts than this will not be cut — the ffmpeg selection would
#: become unwieldy, and the gain is anyway small with so many short pieces.
MAX_PARTS = 400

#: How deep ffmpeg's expression parser may nest.
#:
#: Measured with ffmpeg n9.0.1: At **exactly 100** terms it works
#: well, from 101 on `Error while parsing expression` comes and the process ends
#: with rc=244 (= −12 = ENOMEM). The number is fixed in libavutil, there is
#: no switch for it.
#:
#: This is not an academic limit: where the cut fails, the pipeline
#: diarizes the ENTIRE file — on a day-long recording several times the
#: time and roughly twice the RAM, enough for an OOM kill on a 32 GB machine.
#:
#: Therefore the expression is BRACKETED (see
#: `selection_filter`) — this makes the depth grow logarithmically instead of linearly, and
#: the limit is practically no longer reachable.
FFMPEG_EXPR_DEPTH = 100


def find_silence(
    wav, threshold_db: float = -50.0, min_duration: float = MIN_SILENCE_S, timeout: int = 900
) -> list[tuple[float, float]]:
    """Find silent areas in a file. Return: ``[(start, end), …]``.

    Uses ffmpeg's ``silencedetect``. On a 16-kHz mono WAV it runs
    **thousands of times faster than real time** — a day-long file costs
    seconds vs many minutes of diarization.

    Errors are NEVER fatal here: whoever finds nothing, just diarizes everything.
    """
    try:
        r = subprocess.run(
            [
                "ffmpeg",
                "-v",
                "info",
                "-i",
                str(wav),
                "-vn",
                "-af",
                f"silencedetect=noise={threshold_db}dB:d={min_duration}",
                "-f",
                "null",
                "-",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except Exception as e:
        logger.debug("Silence detection failed: %s", e)
        return []
    text = r.stderr or ""
    starts = [float(x) for x in re.findall(r"silence_start: ([-\d.]+)", text)]
    ends = [float(x) for x in re.findall(r"silence_end: ([\d.]+)", text)]
    # If the file ends silently, the last `silence_end` is missing. Then
    # silence ends at file end — the caller provides this via `total`.
    pairs = [(max(0.0, s), e) for s, e in zip(starts, ends) if e > s]
    return pairs


def speech_parts(
    silence: list[tuple[float, float]], total: float, glue: float = GLUE_S
) -> list[tuple[float, float]]:
    """Form the speech parts to keep from the silent areas.

    The remainder of the file, with short gaps merged. Empty return
    means: all silent — then there is nothing to diarize.

    >>> speech_parts([(0.0, 10.0), (20.0, 30.0)], 40.0)
    [(10.0, 20.0), (30.0, 40.0)]
    """
    if total <= 0:
        return []
    silence = sorted((max(0.0, a), min(total, b)) for a, b in silence if b > a)
    parts: list[tuple[float, float]] = []
    pos = 0.0
    for a, b in silence:
        if a > pos:
            parts.append((pos, a))
        pos = max(pos, b)
    if pos < total:
        parts.append((pos, total))

    # Close short gaps: keeping two parts with 1.5 s of silence between them as
    # separate pieces costs more than saving the 1.5 s.
    glued: list[tuple[float, float]] = []
    for a, b in parts:
        if glued and a - glued[-1][1] <= glue:
            glued[-1] = (glued[-1][0], b)
        else:
            glued.append((a, b))
    return [(a, b) for a, b in glued if b > a]


def times_back(segments, parts: list[tuple[float, float]]) -> list:
    """Map times from cut version back to ORIGINAL timeline.

    ``segments`` are ``(start, end, speaker)`` in the time of the
    concatenated file; ``parts`` are the speech parts in
    original time, in exactly the order they were concatenated.

    **A section that crosses an interface is split.** Everything
    else would be a lie: between the two parts in the original
    lie minutes or hours of silence, and a continuous speaker section
    across them would claim someone spoke there.

    >>> times_back([(0.0, 5.0, "A")], [(100.0, 110.0)])
    [(100.0, 105.0, 'A')]
    >>> times_back([(8.0, 12.0, "A")], [(0.0, 10.0), (100.0, 110.0)])
    [(8.0, 10.0, 'A'), (100.0, 102.0, 'A')]
    """
    if not parts:
        return list(segments)
    # Start of each part in cut time.
    offset = []
    running_flag = 0.0
    for a, b in parts:
        offset.append(running_flag)
        running_flag += b - a

    out = []
    for entry in segments:
        s, e, speaker = float(entry[0]), float(entry[1]), entry[2]
        if e <= s:
            continue
        for i, (a, b) in enumerate(parts):
            t0, t1 = offset[i], offset[i] + (b - a)
            # Overlap of this section with this part?
            from_value, to_value = max(s, t0), min(e, t1)
            if to_value <= from_value:
                continue
            out.append((a + (from_value - t0), a + (to_value - t0), speaker))
    out.sort(key=lambda x: x[0])
    return out


def cut_worth_it(parts: list[tuple[float, float]], total: float, min_saving: float = 0.15) -> bool:
    """Is the cut worth it at all?

    Below ~15% savings the effort (an additional ffmpeg run
    over the entire file) is no longer worthwhile — and every conversion is
    an opportunity for an error that one doesn't take on unnecessarily.
    """
    if not parts or total <= 0:
        return False
    if len(parts) > MAX_PARTS:
        return False
    keep = sum(b - a for a, b in parts)
    return (total - keep) / total >= min_saving


def _weighted(terms: list[str]) -> str:
    """Bracket the terms into a BALANCED sum tree.

    ``a+b+c+…`` ffmpeg's parser reads recursively from the right: With 130 terms
    it goes 130 levels deep and breaks (see
    `FFMPEG_EXPR_DEPTH`). ``((a+b)+(c+d))`` is the same value, but only
    log₂(n) levels deep — 2,000 terms come out to 11.

    Measured against ffmpeg n9.0.1:

        Terms    flat         bracketed
          120    rc=244       rc=0
        2,000    rc=244       rc=0
    """
    if not terms:
        return "0"  # selects nothing; `cut_worth_it` catches this earlier
    if len(terms) == 1:
        return terms[0]
    middle = len(terms) // 2
    return f"({_weighted(terms[:middle])}+{_weighted(terms[middle:])})"


def select_filter(parts: list[tuple[float, float]]) -> str:
    """ffmpeg filter that keeps exactly the speech parts and concatenates them.

    ``asetpts=N/SR/TB`` is not optional: Without resetting the
    timestamps the output keeps the gaps as time jumps, and the
    cut file would be as long as the original again.

    The bracketing is also not optional — it is the difference
    between "cut succeeds" and "entire file goes to diarization".
    """
    from whisper_tm.core.audio_utils import keep_audio_filter

    terms = [f"between(t,{a:.3f},{b:.3f})" for a, b in parts]
    return keep_audio_filter(parts, _weighted(terms))
