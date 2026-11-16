"""Speaker routes: "who was in this recording, and approximately where?"

The most important question is who is part of a recording at all; the
second most important is approximately where they spoke.

It's answered with no clustering: each 30-second window of the recording is
held against named people. Measured on a multi-hour recording against a
gallery of about a dozen named people: all participants found, **zero** false
alarms across the recording's windows.

Vectors live in their own database (:mod:`whisper_tm.database.speaker`),
not in ``library.db``.
"""

import json
import logging
import time
from pathlib import Path

from flask import Blueprint, jsonify, request

from whisper_tm.config import config
from whisper_tm.core import debug, media_formats
from whisper_tm.core import speaker as speaker_core
from whisper_tm.core.audio_utils import run_cancellable
from whisper_tm.core.helpers import transcript_identity
from whisper_tm.database.speaker import speaker_db

logger = logging.getLogger(__name__)

speaker_bp = Blueprint("speaker", __name__)


def _identity(filename: str) -> str:
    """Display path → ``abs_path``. Empty means: not resolvable."""
    return transcript_identity(filename, media_dir=str(config.media_dir))


@speaker_bp.route("/api/speaker/recording")
@speaker_bp.route("/api/speaker/recording/<path:filename>")
def recording(filename=None):
    """Who was there, and when.

    ``?threshold=`` allows tweaking. Without it the setting
    ``speaker.threshold_gallery`` applies, and ``threshold`` in the response
    says which value was used — the file view starts its slider there. The
    setting's default is **0.75**, and that's not either of the thresholds
    you might expect:

    * **0.55** applies to two INDIVIDUAL voices. Measured: 47% of
      recordings would get at least one wrong name.
    * **0.62** applies WITHIN a recording, where the common channel lifts
      all similarities.
    * **0.75** applies to a recording against the WHOLE gallery. Must
      be higher because with a dozen people and multiple windows there are hundreds
      of chances to cross the threshold by chance. Measured:
      96.0% hit rate at 0.6% false alarms.

    Confusing the three is the most obvious trap in this code.

    The path also comes as ``?file=`` — see ``window_list`` for why an
    absolute one can't travel in the URL path.
    """
    filename = filename or request.args.get("file", "")
    abs_path = _identity(filename)
    if not abs_path:
        return jsonify({"error": "Path not resolvable"}), 400

    try:
        threshold = float(request.args.get("threshold", speaker_core.threshold("gallery")))
    except (TypeError, ValueError):
        threshold = speaker_core.threshold("gallery")
    threshold = min(max(threshold, 0.0), 1.0)

    db = speaker_db()
    windows = db.windows(abs_path)
    if not windows:
        return jsonify(
            {
                "abs_path": abs_path,
                "windows": 0,
                "people": [],
                "hint": "No voice vectors exist for this recording. "
                "They are created during a transcription WITH diarization "
                '— for existing files, queue them as "Voices only" '
                "(queue editor → From library).",
            }
        )

    labels: dict[str, dict] = {}
    for f in windows:
        k = labels.setdefault(
            f["label"],
            {
                "label": f["label"],
                "windows": 0,
                "speech_sec": 0.0,
                "first_sec": f["start_sec"],
                "last_sec": f["start_sec"],
            },
        )
        k["windows"] += 1
        k["speech_sec"] = round(k["speech_sec"] + (f["speech_sec"] or 0), 1)
        k["last_sec"] = max(k["last_sec"], f["start_sec"])

    try:
        people = speaker_core.who_was_there(abs_path, threshold=threshold)
    except Exception as e:
        logger.warning("who_was_there(%s): %s", abs_path, e)
        people = []

    return jsonify(
        {
            "abs_path": abs_path,
            "windows": len(windows),
            "threshold": threshold,
            "model": windows[0]["model"],
            "labels": sorted(labels.values(), key=lambda k: -k["speech_sec"]),
            "people": people,
            # For honesty's sake: without named people the question "who was
            # there" can't be answered at all, no matter how good the vectors are.
            "named_people": len(speaker_db().people()),
        }
    )


@speaker_bp.route("/api/speaker/windows")
@speaker_bp.route("/api/speaker/windows/<path:filename>")
def window_list(filename=None):
    """All windows of a recording — for timeline, naming and locating.

    The path also comes as ``?file=``: an ABSOLUTE path in the URL path
    ("/api/speaker/windows//mnt/…") is answered by the router with a
    redirect that merges the double slash — and the path arrives without
    its leading slash, as a relative one that resolves to nothing. The map
    shows recordings outside the media folder with their absolute path, so
    locating them failed that way.
    """
    filename = filename or request.args.get("file", "")
    abs_path = _identity(filename)
    if not abs_path:
        return jsonify({"error": "Path not resolvable"}), 400
    db = speaker_db()
    rows = db.windows(abs_path)
    # Which are in the trash — the map says so instead of lighting nothing
    # (voicemap_graph.js, "locate").
    with db.connection() as c:
        trashed = db._trash_of(c, [f["id"] for f in rows]) if rows else {}
    out = []
    for f in rows:
        w = {
            k: f[k]
            for k in (
                "id",
                "label",
                "windows",
                "start_sec",
                "speech_sec",
                "segments",
                "model",
                "person",
                "source",
            )
        }
        w["trashed"] = f["id"] in trashed
        out.append(w)
    return jsonify({"abs_path": abs_path, "windows": out})


@speaker_bp.route("/api/speaker/people")
def people():
    """All known people (named voices) in the speaker database."""
    return jsonify({"people": speaker_db().people()})


@speaker_bp.route("/api/speaker/assign", methods=["POST"])
def assign():
    """Assign a vector to a person.

    The name hangs on the **vector**, never on a group number: groups
    change with linkage and threshold, a name once assigned shouldn't.
    """
    data = request.json or {}
    vid = data.get("vector_id")
    if not isinstance(vid, int):
        return jsonify({"error": "vector_id missing"}), 400
    name = data.get("person")
    if name is not None and not str(name).strip():
        name = None
    db = speaker_db()
    try:
        db.assign(
            vid,
            name if name is None else str(name).strip(),
            source=str(data.get("source") or "human"),
            origin=str(data.get("origin") or "map"),
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    return jsonify({"ok": True, "decision": db.last_decision_id()})


@speaker_bp.route("/api/speaker/assign-many", methods=["POST"])
def assign_many():
    """Name an entire group at once.

    Without this, a whole group couldn't be named from the GUI at all: the
    button always named only the selected voice, so a group of twenty voices
    took twenty individual clicks.

    **Existing assignments are NOT overwritten.** A group is a machine claim
    ("sounds the same"), an assignment is a human determination. The claim
    can't overturn the determination — else a click on the wrong group would
    silently turn twenty careful assignments into twenty wrong ones. Anyone
    wanting to really move a single voice does it individually; and with
    ``overwrite: true`` it's deliberately possible here too.

    Response names both counts so you see what did NOT happen.
    """
    data = request.json or {}
    ids = data.get("vector_ids")
    if not isinstance(ids, list) or not ids:
        return jsonify({"error": "vector_ids missing"}), 400
    ids = [int(v) for v in ids if isinstance(v, (int, float))]
    if not ids:
        return jsonify({"error": "no valid vector_ids"}), 400
    name = str(data.get("person") or "").strip()
    if not name:
        return jsonify({"error": "person missing"}), 400
    overwrite = bool(data.get("overwrite"))

    db = speaker_db()
    try:
        present = db.names_per_vector()
        chosen, skipped, already_correct = [], [], 0
        for vid in ids:
            old_value = present.get(vid)
            if old_value and old_value.lower() == name.lower():
                already_correct += 1
                continue
            if old_value and not overwrite:
                skipped.append(old_value)
                continue
            chosen.append(vid)
        # ONE write, one entry of the history: "name the whole group" is
        # one click, and its undo takes the whole group back.
        if chosen:
            db.assign_many(chosen, name, source="map-group", origin="map")
        set_flag = len(chosen)
    except Exception as e:
        logger.warning("assign_many(%s): %s", name, e)
        return jsonify({"error": str(e)}), 500

    return jsonify(
        {
            "ok": True,
            "person": name,
            "set_flag": set_flag,
            "already_correct": already_correct,
            "skipped": len(skipped),
            # Which foreign names were in the way — the number alone doesn't say
            # if you made a mistake.
            "foreign_names": sorted(set(skipped)),
            "decision": db.last_decision_id() if set_flag else None,
        }
    )


@speaker_bp.route("/api/speaker/person/<name>")
def person_recordings(name):
    """**In which recordings does this person appear?**

    The inventory question — a level above "who was in this recording" and the
    actual gain of the whole build. A matrix product against the held inventory;
    milliseconds even for hundreds of thousands of vectors.
    """
    from whisper_tm.core.speaker_inventory import where_it_occurs

    try:
        threshold = float(request.args.get("threshold", speaker_core.threshold("gallery")))
    except (TypeError, ValueError):
        threshold = speaker_core.threshold("gallery")
    threshold = min(max(threshold, 0.0), 1.0)
    try:
        limit = min(max(int(request.args.get("limit", 500)), 1), 5000)
    except (TypeError, ValueError):
        limit = 500
    try:
        d = where_it_occurs(name, threshold=threshold, at_most=limit)
    except Exception as e:
        logger.warning("where_it_occurs(%s): %s", name, e)
        return jsonify({"error": str(e)}), 500

    # Add display path — abs_path is the identity, but no one likes reading
    # /mnt/media/…
    from whisper_tm.core import paths

    # The stored form (C:/… on Windows), compared on a segment boundary
    root = paths.canon(config.media_dir)
    for a in d["recordings"]:
        p = a["abs_path"]
        a["display"] = paths.shown(p, root) if root else p
    return jsonify(d)


#: Maximum number of pieces the sample concatenates.
#:
#: Each piece is its own ffmpeg input (fast seek vs full decode — on a
#: 6-hour recording that's the whole difference). The command line grows
#: linearly, and there's a limit somewhere. 24 covers the normal case: a
#: 30-second window consists on average of 18 pieces per inventory, the
#: 95-percentile window of 27.
MAX_PARTS = 24


def _audio_sample_parts(raw) -> list[tuple[float, float]]:
    """Read stored spans — tolerantly but not naively.

    The column was added later; old rows have NULL, and a
    half-written value can't cost a sample. In doubt, empty list, then the
    old way kicks in.
    """
    if not raw:
        return []
    try:
        pairs = json.loads(raw)
    except (TypeError, ValueError):
        return []
    out: list[tuple[float, float]] = []
    for x in pairs if isinstance(pairs, list) else []:
        try:
            a, b = float(x[0]), float(x[1])
        except (TypeError, ValueError, IndexError):
            continue
        if b > a >= 0 and b - a >= 0.05:
            out.append((a, b))
    out.sort()
    if len(out) <= MAX_PARTS:
        return out
    # Too many: keep the LONGEST, but output in time order.
    # The longest carry the vector most strongly, and a cut list in jump order
    # would be hard to follow.
    longest = sorted(out, key=lambda ab: ab[0] - ab[1])[:MAX_PARTS]
    return sorted(longest)


#: How many cuts the montage distributes across a span.
#:
#: Six at five seconds each give the same 30 seconds as the window, just
#: distributed instead of in one chunk. Fewer cuts miss too much of the span,
#: more get too short to recognize a voice — under three seconds you can't
#: reliably hear tone and timbre.
MONTAGE_PIECES = 6


def _speech_spots(abs_path: str, label: str, start: float, end: float) -> list[tuple[float, float]]:
    """Where this speaker talks per TRANSCRIPT within the span.

    The decisive difference from blind montage: 30 seconds of speech
    spread over a span up to 38 minutes — that's 1.3% of the distance. Six
    evenly-spaced cuts thus almost always land in silence or with other people.
    In a checked case the blind cuts hit 5 segments, none of them from the
    speaker in question.

    Transcript segments are NOT the same as diarization turns (only 7% of
    windows could be reconstructed from them) — but they say WHEN this speaker
    talked, and that's enough to place samples where there's something to hear.
    """
    from whisper_tm.extensions import db as main_db

    if not main_db:
        return []
    try:
        with main_db.connection() as c:
            lines = c.execute(
                """SELECT s.start_sec, s.end_sec FROM segments s
                   JOIN transcripts t ON t.id = s.transcript_id
                   WHERE t.abs_path = ? AND s.speaker = ?
                     AND s.start_sec >= ? AND s.start_sec < ?
                   ORDER BY s.start_sec""",
                (abs_path, label, float(start), float(end)),
            ).fetchall()
    except Exception as e:  # noqa — never cost a sample
        logger.debug("Speech spots not readable: %s", e)
        return []
    out = []
    for r in lines:
        a, b = float(r[0] or 0.0), float(r[1] or 0.0)
        if b > a:
            out.append((a, b))
    return out


def _montage_from_spots(
    spots: list[tuple[float, float]], duration: float, pieces: int = MONTAGE_PIECES
) -> list[tuple[float, float]]:
    """Draw N samples from the speech places, spread across the span.

    Evenly across the LIST, not time: so you get start, middle and end of
    speech, even if the places are unevenly distributed. Each sample starts
    at the beginning of its place — a word sits there, not half a vowel.
    """
    if not spots:
        return []
    n = max(1, min(int(pieces), len(spots)))
    length = max(3.0, float(duration) / n)
    step = len(spots) / n
    out = []
    for i in range(n):
        a, b = spots[int(i * step)]
        out.append((round(a, 2), round(min(b if b - a >= length else a + length, a + length), 2)))
    # Remove overlaps: two samples can fall into the same long place
    # if there are few.
    clean = []
    for a, b in out:
        if clean and a < clean[-1][1]:
            continue
        clean.append((a, b))
    return clean


def _montage_parts(
    start: float, end: float, duration: float, pieces: int = MONTAGE_PIECES
) -> list[tuple[float, float]]:
    """Fallback for windows without stored spans.

    Real spans can't be reconstructed for old windows — diarization turns
    are nowhere, and transcript segments could be reconstructed to only
    **7%** measured. Recomputing them means redoing separation and diarization:
    hundreds of GPU hours on a large inventory — days for better samples,
    not better data.

    Montage is the compromise: it hits the same DISTANCE the vector came from,
    just not the same places. A window over 38 minutes was formerly represented
    by its first 30 seconds; now by six samples spread across the 38 minutes.
    That's not truth, but an honest approximation — and it costs nothing.

    Empty list means "not worth it": if the span is barely longer than the
    window, the head cut is already almost everything.
    """
    span = float(end) - float(start)
    duration = max(3.0, float(duration))
    if span <= duration * 1.5:
        return []
    n = max(2, min(int(pieces), int(span // 3)))
    length = max(3.0, duration / n)
    if span <= length:
        return []
    step = (span - length) / (n - 1)
    return [(round(start + i * step, 2), round(start + i * step + length, 2)) for i in range(n)]


def _sample_command(
    source: Path, target: Path, parts: list[tuple[float, float]], start: float, duration: float
) -> list[str]:
    """The ffmpeg call — concatenated if there are pieces, otherwise in one go.

    Each piece has its own input with `-ss` BEFORE `-i`: that seeks in the
    container instead of decoding from the start. `concat` glues them in one
    track, in the same order `build_windows` planned them.
    """
    command = ["ffmpeg", "-v", "error", "-y"]
    if parts:
        for a, b in parts:
            command += [
                "-ss",
                f"{a:.3f}",
                "-t",
                f"{max(0.05, b - a):.3f}",
                *media_formats.source(source),
            ]
        chains = "".join(f"[{i}:a]" for i in range(len(parts)))
        command += [
            "-filter_complex",
            f"{chains}concat=n={len(parts)}:v=0:a=1[out]",
            "-map",
            "[out]",
        ]
    else:
        command += ["-ss", f"{start:.3f}", "-t", f"{duration:.3f}", *media_formats.source(source)]
    command += [
        "-vn",
        "-codec:a",
        "libmp3lame",
        "-q:a",
        "5",
        "-ac",
        "1",
        # `-f mp3` is NOT optional: written to `.mp3.tmp`, and ffmpeg
        # can't guess a format from that extension (rc=234, "Unable to
        # find a suitable output format").
        # Same trap already in api/media.py.
        "-f",
        "mp3",
        str(target),
    ]
    return command


@speaker_bp.route("/api/speaker/sample/<int:vector_id>")
def audio_sample(vector_id):
    """Audio from a window — cut on demand, not in advance.

    Precutting all samples of a large inventory takes **many gigabytes**.
    Here it's cut when someone listens, and
    the result sits after in the playback cache with its LRU limit.

    **It concatenates like the vector.**

    Older versions cut 30 seconds in one chunk from the window start. That's
    almost never the material the vector came from. A checked example:

        recording A, SPEAKER_13   heard: 11 segments, 0× the speaker in question
                                  span:  38 min,       6× the speaker in question
        recording B, SPEAKER_24   span:  381 min (!)

    The listener heard three other people and was asked to decide on a group
    the model formed because of a completely different voice. The whole
    assignment path lives on the sample telling the truth.

    Now the spans sit in column ``parts`` (the same ``build_windows`` planned),
    and ffmpeg cuts exactly them and concatenates in order. For rows from
    before, they don't exist — then the header says ``X-Voice-Sample: start``
    (or a montage kind) instead of ``parts``.
    """
    # Measurement points per step: request timing once reported minutes for
    # this endpoint while ffmpeg logged ~100 ms. Where the remainder went was
    # nowhere. Now it is.
    mp = debug.MeasurePoints(f"Sample v{vector_id}")

    db = speaker_db()
    with db.connection() as c:
        r = c.execute(
            "SELECT abs_path, start_sec, speech_sec, label, "
            "windows, parts FROM speaker_vectors WHERE id = ?",
            (vector_id,),
        ).fetchone()
    mp.point("DB row")
    if not r:
        return jsonify({"error": "Window unknown"}), 404

    from flask import send_file

    from whisper_tm.core.security import root_allowed

    source = Path(r["abs_path"])
    # Same check everywhere: only within allowed roots. The
    # vector database is not a trusted source for paths.
    allowed = root_allowed(str(source))
    mp.point("root check")
    # Measured separately because it's the only step here that touches NAS —
    # a stat over CIFS can take very long on cold connection, and then
    # you want exactly that in the line.
    reachable = allowed and source.exists()
    mp.point("file stat (NAS)")
    if not reachable:
        return jsonify({"error": "File unreachable"}), 404

    duration = min(max(float(r["speech_sec"] or 30.0), 3.0), 60.0)
    start = max(0.0, float(r["start_sec"] or 0.0))
    parts = _audio_sample_parts(r["parts"])
    kind = "parts"
    if not parts:
        # No stored plan: then at least spread across the whole span
        # instead of 30 seconds from start. The end of the span is the
        # start of the next window of the SAME speaker ID — windows of a
        # speaker follow their speech in order.
        with db.connection() as c:
            n = c.execute(
                "SELECT MIN(start_sec) FROM speaker_vectors "
                "WHERE abs_path = ? AND label = ? AND start_sec > ?",
                (r["abs_path"], r["label"], r["start_sec"]),
            ).fetchone()[0]
        if n:
            from_value, duration_f = float(r["start_sec"] or 0.0), float(r["speech_sec"] or 30.0)
            # First aim where this speaker talks per transcript.
            spots = _speech_spots(r["abs_path"], r["label"], from_value, float(n))
            parts = _montage_from_spots(spots, duration_f)
            kind = "montage-speech"
            if not parts:
                parts = _montage_parts(from_value, float(n), duration_f)
                kind = "montage"
        if not parts:
            kind = "start"
    cache = config.dirs.get("cache") or config.dirs["work"]
    folder = Path(cache) / "voice_samples"
    folder.mkdir(parents=True, exist_ok=True)
    # The type belongs in the filename: else the cache keeps serving the
    # old head cut after re-processing, and no one notices.
    target = folder / f"v{vector_id}_{int(start)}_{int(duration)}_{kind}.mp3"
    fresh = not target.exists() or target.stat().st_size < 1000
    mp.point("cache check")

    if fresh:
        # ffmpeg runs via run_cancellable, not subprocess.run: all
        # external tools in this app go through it (timing measurement
        # hangs there too).
        tmp = target.with_suffix(".mp3.tmp")
        try:
            command = _sample_command(source, tmp, parts, start, duration)
            r2 = run_cancellable(command, job=None, timeout=300, label=f"sample ({kind})")
            if tmp.exists() and tmp.stat().st_size > 1000:
                tmp.replace(target)
            else:
                logger.warning(
                    "sample %s empty (ffmpeg rc=%s): %s",
                    vector_id,
                    r2.returncode,
                    (r2.stderr or "")[-200:],
                )
                tmp.unlink(missing_ok=True)
                return jsonify({"error": "Excerpt empty"}), 500
        except Exception as e:
            logger.warning("sample %s: %s", vector_id, e)
            return jsonify({"error": str(e)[:120]}), 500
    mp.point("ffmpeg" if fresh else "cache hit")
    try:
        target.touch()  # LRU
    except OSError:
        pass
    answer = send_file(str(target), mimetype="audio/mpeg", conditional=True)
    # So the UI can say WHAT it's playing there.
    answer.headers["X-Voice-Sample"] = kind
    answer.headers["X-Voice-Sample-Parts"] = str(len(parts) if parts else 0)
    mp.point("send_file")
    mp.done(("from cache" if not fresh else "freshly cut") + f" ({kind})")
    return answer


@speaker_bp.route("/api/voicemap/graph")
def map_graph():
    """The map as a **graph**: every voice a point, edges = similarity.

    The old overview drew one bubble per group. At 200,000 voices that's
    the only viable approach — but at about a thousand it throws away
    exactly what you want to see: **the structure**. In the bubble view
    you couldn't make out anything.

    **Why kNN and not all edges above the threshold.** Measured on an
    inventory of about a thousand voices:

    ====================  ==========
    all edges ≥ 0.55        ~180,000
    kNN k=5, only ≥ 0.55      ~4,000
    ====================  ==========

    A group with about 500 members is almost fully connected and accounts for
    ~140,000 edges on its own. Drawn out, that would produce a ball of
    yarn where you can make out even less — and that's exactly why the
    proof-of-concept map got away with "all edges": it showed under a
    hundred voices.

    Keeping the ``k`` strongest neighbours per node is the usual answer
    to that, and also the one that makes the force layout readable.

    **What is held, and by what** (``core/speaker_graph.py``): the
    neighbourhood, the grouping and every column that doesn't depend on
    names are held by the vectors (``inventory_generation``), the trash
    view and the slider position. A request after a name change — the
    reload after each "Apply" in the guided assignment — only fills in the
    names: on an inventory of several hundred thousand voices it took close
    to a minute before.
    """
    from flask import current_app

    from whisper_tm.core import speaker_graph, speaker_inventory

    # Timing per section.
    #
    # The endpoint once took over half a minute (~50k voices), and the question
    # "what for, exactly?" couldn't be answered without measurement — the
    # log only showed the total request duration. The sections now show up
    # individually in the debug log, so the next slowdown doesn't have to
    # be guessed at again.
    with debug.timed("Map: loading inventory"):
        state_info = speaker_inventory.inventory()

    # `?trash=1` shows exactly what was discarded — see `speaker_graph`.
    trash_only = str(request.args.get("trash", "")).lower() in ("1", "true", "yes")
    root = str(config.media_dir)
    with debug.timed("Map: voices"):
        v = speaker_graph.view(state_info, trash_only, root)

    if not v["n"]:
        return jsonify(
            {
                "nodes": {"id": []},
                "edges": [],
                "groups": [],
                "voices": 0,
                "trash": v["trash"],
                "trash_only": trash_only,
                "hint": (
                    "The trash is empty." if trash_only else "No voice vectors in storage yet."
                ),
            }
        )

    limit = request.args.get("threshold", type=float)
    if limit is None:
        limit = speaker_core.threshold("group_voices")
    limit = min(max(limit, 0.0), 1.0)
    link_method = request.args.get("link_method", "average")
    if link_method not in ("average", "complete", "single"):
        link_method = "average"
    # k caps the number of edges. 5 is the measured compromise: readable,
    # and the groups stay connected.
    k = min(max(request.args.get("k", 5, type=int), 1), 20)

    device = "cpu" if v["n"] <= 20000 else "cuda"
    c = speaker_graph.cut(state_info, v, limit, link_method, k, device)

    with debug.timed("Map: names per vector"):
        names = speaker_db().names_per_vector()
    with debug.timed("Map: answer"):
        body = speaker_graph.answer(
            v, c, names, limit=limit, link_method=link_method, k=k, root=root
        )
    return current_app.response_class(body, mimetype="application/json")


@speaker_bp.route("/api/voicemap/threshold-search")
def map_threshold_search():
    """At WHICH threshold is this person's group largest?

    An automatic search across all thresholds that shows where a person's
    group is largest — some people form a large group only at one specific
    threshold.

    That's exactly it: the threshold is not a global quality knob, but
    a different question for every voice. A distinctive voice holds
    together up to 0.80; one with few, short windows already falls apart
    at 0.66 and merges with its neighbour at 0.58. By hand you only find
    that by trying — and you'd miss a narrow maximum while doing so.

    **What "their group" means.** The same rule as everywhere on the map:
    a group's name is the most frequently assigned name in it. So what's
    searched for is the largest group that CARRIES this name. That's
    self-limiting — if "alice" merges into "bob" at a low threshold, the
    giant group is called "bob", and it no longer counts as the Alice group.
    Without this rule, the answer would always be the lowest threshold at
    which everything is a lump.

    **Two recommendations, not one.** ``largest`` (largest) is literally
    what was asked for. ``best`` (best) additionally weighs how much
    FOREIGN material got swept in (F₁ from purity and coverage) — a group
    of 400 voices, of which only 30 are the person being searched for, is
    not a find but a warning. Both values are shown side by side, the
    decision stays with the human.

    **Why this takes seconds and not minutes.** Two things, both measured
    on about 8,000 voices:

    1. The kNN neighbourhood does not depend on the threshold. Computed
       once (1.0 s), it costs nothing more afterward.
    2. **Only the component the person is actually in gets fine-sliced.**
       Fine-slicing over the whole inventory costs 0.8 s per threshold —
       51 thresholds measured at 41.6 s that way. The connected components
       are disjoint and ``_finer`` runs independently per component
       anyway; everything outside can be left alone.

       That alone was NOT enough: measured at 37.9 s instead of 41.6. At
       low thresholds, even a small person hangs in the giant component —
       a person with a dozen voices sat at 0.40 in a lump of several thousand.

    3. So on top of that, **two-stage**: first a coarse grid (0.05), then
       fine (0.01) only where something moves — around the coarse maximum
       and over every coarse interval where the group size at least
       doubles or appears for the first time. A maximum narrower than the
       coarse grid AND looking the same at both ends can slip through
       here; that's the deliberate price. Measured, this brings the search
       down from 38 s to a few seconds.

    What gets lost along the way is stated honestly in the response:
    ``component`` says how large the decomposed part was, and ``coarse``
    marks the points of the coarse grid. This search does NOT provide a
    total count of all groups in the inventory — it never computed it.
    """
    import numpy as np

    from whisper_tm.core import speaker_graph, speaker_inventory, speaker_knn

    wanted = (request.args.get("person") or "").strip()
    if not wanted:
        return jsonify({"error": "person missing"}), 400

    state_info = speaker_inventory.inventory()
    # The map's voices — the trash left out, like on the map. Earlier
    # the search ran over the whole inventory, the discarded voices
    # included: it recommended thresholds for groups the map, which
    # groups without the trash, never showed.
    # One selection also means one neighbourhood for both.
    v = speaker_graph.view(state_info, False, str(config.media_dir))
    X = speaker_graph.matrix(state_info, v)
    if not len(X):
        return jsonify(
            {"person": wanted, "total": 0, "points": [], "hint": "No voice vectors in storage yet."}
        )

    names = speaker_db().names_per_vector()
    small = wanted.lower()
    # Case doesn't matter — same rule as in the person table
    # (`speaker_people.name COLLATE NOCASE`).
    person_from = [(names.get(i) or "").lower() for i in v["ids"].tolist()]
    is_wanted = np.array([p == small for p in person_from])
    total = int(is_wanted.sum())
    if not total:
        # Typographic quotation marks NOT in the Python string: the
        # closing character would end the literal (a project trap).
        return jsonify(
            {
                "person": wanted,
                "total": 0,
                "points": [],
                "hint": (
                    f"No voice is assigned to '{wanted}' - the "
                    f"search needs at least one named voice "
                    f"as an anchor."
                ),
            }
        )

    from_value = min(max(request.args.get("from_value", 0.40, type=float), 0.05), 0.99)
    to_value = min(max(request.args.get("to_value", 0.95, type=float), 0.05), 0.99)
    if to_value < from_value:
        from_value, to_value = to_value, from_value
    # `step` (step) set by hand means: exactly this grid, without the
    # two-stage approach. The tests need that, as does anyone who wants
    # to see the curve in full.
    step = request.args.get("step", type=float)
    fixed = step is not None
    if fixed:
        step = min(max(step, 0.005), 0.2)
    link_method = request.args.get("link_method", "average")
    if link_method not in ("average", "complete", "single"):
        link_method = "average"

    def raster(a: float, b: float, width: float) -> list[float]:
        out, s = [], a
        # Upper bound against a call that ties up the server for minutes.
        while s <= b + 1e-9 and len(out) < 200:
            out.append(round(s, 4))
            s += width
        return out

    device = "cpu" if len(X) <= 20000 else "cuda"
    # The map's neighbourhood — held, and usually already there: the
    # search recomputed it on every call (many seconds on a large inventory).
    idx, value = speaker_knn.held_neighbours(
        X, v["marker"], k=speaker_knn.K, device=device, ids=v["ids"], family=v["family"]
    )
    # The fine cut sees the vectors as `group_voices` does for the map.
    Xn = speaker_knn._nrm(X.astype(np.float32))

    t0 = time.time()
    measured: dict[float, dict] = {}

    def measure(limit: float) -> dict:
        if limit in measured:
            return measured[limit]
        comp = speaker_knn.components(idx, value, len(X), limit)
        # Only the components in which the person being searched for
        # actually occurs — the remainder of the inventory can be left alone.
        affected = np.unique(comp[is_wanted])
        best = None
        comp_n = 0
        for kid in affected:
            members = np.flatnonzero(comp == kid)
            m = len(members)
            comp_n += m
            if m >= 3 and m <= 50_000:
                below = speaker_knn._finer(Xn[members], limit, link_method)
            else:
                # Under three members, even the full run doesn't split
                # further and assigns individual labels instead (see
                # `group_voices`).
                below = np.arange(m, dtype=np.int32)
            for u in np.unique(below):
                part = members[below == u]
                hits = int(is_wanted[part].sum())
                if not hits:
                    continue
                # Does this part carry THIS name? It does if no other
                # name occurs more often in it — the same rule by which
                # the map names its groups.
                #
                # On a tie, it counts. That's more generous than the
                # `max()` used for map naming, which decides arbitrarily
                # there; for a SEARCH that's the right direction — showing
                # a candidate and noting its purity alongside is better
                # than hiding it.
                foreign: dict[str, int] = {}
                for i in part:
                    p = person_from[i]
                    if p and p != small:
                        foreign[p] = foreign.get(p, 0) + 1
                if foreign and max(foreign.values()) > hits:
                    continue
                n = len(part)
                purity = hits / n
                coverage = hits / total
                f1 = 2 * purity * coverage / (purity + coverage) if purity + coverage else 0.0
                candidate = {
                    "n": n,
                    "hits": hits,
                    "purity": round(purity, 4),
                    "coverage": round(coverage, 4),
                    "f1": round(f1, 4),
                }
                if best is None or (n, f1) > (best["n"], best["f1"]):
                    best = candidate
        entry = {
            "threshold": limit,
            "component": int(comp_n),
            **(best or {"n": 0, "hits": 0, "purity": 0.0, "coverage": 0.0, "f1": 0.0}),
        }
        measured[limit] = entry
        return entry

    COARSE = 0.05
    if fixed:
        for g in raster(from_value, to_value, step):
            measure(g)
        coarse_ones = []
    else:
        coarse_ones = raster(from_value, to_value, COARSE)
        for g in coarse_ones:
            measure(g)
        # Where is a close look worth it? Around the maximum — and
        # anywhere the size changes sharply between two coarse points. A
        # jump from 1 to 12 between 0.65 and 0.70 is exactly
        # the window where the best value lies.
        peak = max(coarse_ones, key=lambda g: measured[g]["n"])
        narrow = {(max(from_value, peak - COARSE), min(to_value, peak + COARSE))}
        for a, b in zip(coarse_ones, coarse_ones[1:]):
            na, nb = measured[a]["n"], measured[b]["n"]
            high, low = max(na, nb), min(na, nb)
            if high > 0 and (low == 0 or high >= 2 * low):
                narrow.add((a, b))
        for a, b in sorted(narrow):
            for g in raster(a, b, 0.01):
                measure(g)

    points = [measured[s] for s in sorted(measured)]
    for p in points:
        p["coarse"] = p["threshold"] in coarse_ones
    with_ = [p for p in points if p["n"] > 0]
    largest = max(with_, key=lambda p: (p["n"], p["f1"])) if with_ else None
    best_f1 = max(with_, key=lambda p: (p["f1"], p["n"])) if with_ else None
    return jsonify(
        {
            "person": wanted,
            "total": total,
            "link_method": link_method,
            "points": points,
            "largest": largest,
            "best": best_f1,
            "two_stage": not fixed,
            "duration_sec": round(time.time() - t0, 1),
        }
    )


@speaker_bp.route("/api/speaker/statistics")
def stats_data():
    """Speaker database statistics plus the operating point: embedding model,
    window lengths and the similarity thresholds in use.
    """
    s = speaker_db().stats_data()
    s["operating_point"] = {
        "model": speaker_core.MODEL_BRAND,
        "window_length_sec": speaker_core.MAX_SPEECH,
        "min_segment_sec": speaker_core.MIN_SEGMENT,
        "threshold_across_files": speaker_core.THRESHOLD_ACROSS_FILES,
        "threshold_within_file": speaker_core.THRESHOLD_WITHIN_FILE,
        "threshold_gallery": speaker_core.threshold("gallery"),
    }
    return jsonify(s)


# ── Trash ────────────────────────────────────────────────────────────────
#
# Speakers that aren't relevant can be hidden from the speaker map —
# moved to a trash that is viewable as a second map.
#
# Discarding happens per tile, storage happens per VOICE. The reason is
# the group number: it's created fresh on every run and means something
# different after the next threshold change. A trash bin that holds
# numbers would eventually throw away the wrong voices.
#
# Nothing is deleted. The vectors represent compute time, and a mistake
# would otherwise be permanent.


@speaker_bp.route("/api/voicemap/trash", methods=["GET"])
def trash_state():
    """What's in the trash?"""
    ids = speaker_db().trash_ids()
    return jsonify({"count": len(ids)})


@speaker_bp.route("/api/voicemap/trash", methods=["POST"])
def move_to_trash():
    """Discard voices — as a list of vector IDs.

    The IDs come from the map (`nodes.id`); the view already knows which
    ones belong to a tile. The server does NOT look them up via a group
    number: the map's groups come from a run at a particular threshold,
    and by the time the request arrives here, a different one may long
    since have been computed.
    """
    d = request.get_json(silent=True) or {}
    ids = d.get("ids") or d.get("vector_ids") or []
    if not isinstance(ids, list) or not ids:
        return jsonify({"error": "No voices specified"}), 400
    try:
        clean = [int(x) for x in ids]
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid ID"}), 400
    db = speaker_db()
    n = db.move_to_trash(clean, reason=str(d.get("reason") or ""), origin="map")
    return jsonify(
        {
            "ok": True,
            "discarded": n,
            "decision": db.last_decision_id(),
            "count": len(db.trash_ids()),
        }
    )


@speaker_bp.route("/api/voicemap/trash/restore", methods=["POST"])
def restore_from_trash():
    """Restore. Without `ids`, the whole trash is emptied.

    Only a request WITHOUT the field means "everything" — an empty list is
    an empty selection, and it used to restore the whole trash
    (``[] or None`` is None).
    """
    d = request.get_json(silent=True) or {}
    key = "ids" if "ids" in d else "vector_ids"
    ids = d.get(key)
    if ids is not None:
        if not isinstance(ids, list):
            return jsonify({"error": "Invalid list"}), 400
        if not ids:
            return jsonify({"error": "No voices specified"}), 400
        try:
            ids = [int(x) for x in ids]
        except (TypeError, ValueError):
            return jsonify({"error": "Invalid ID"}), 400
    db = speaker_db()
    n = db.restore_from_trash(ids, origin="map")
    return jsonify(
        {"ok": True, "back": n, "decision": db.last_decision_id(), "count": len(db.trash_ids())}
    )


# ── Guided assignment ───────────────────────────────────────────────────
#
# A mechanism that makes assigning names to voices faster, with a GUI path
# that covers about 95% of the work.
#
# The procedure and the measurements are in the header of
# `core/speaker_assignment.py`. Briefly: centroids per recording and ID
# instead of individual windows, complete linkage instead of connected
# components, then spreading via the name centroid.


@speaker_bp.route("/api/speaker/assignment/suggestion")
def assignment_suggestion():
    """The next group that brings the most benefit — plus progress."""
    from whisper_tm.core import speaker_assignment as assignment

    threshold = request.args.get("spread", type=float) or assignment.control_value("spread")
    threshold = min(max(threshold, 0.3), 0.99)
    # Skipped groups, as vector IDs of their largest voice. This state
    # belongs to this person's session, not to the inventory — so it's
    # passed along with every request instead of living server-side.
    without = set()
    for piece in str(request.args.get("without", "")).split(","):
        piece = piece.strip()
        if piece.isdigit():
            without.add(int(piece))
        if len(without) >= 500:  # no session has 500 skipped entries
            break
    sort_by = str(request.args.get("sort_by", "hours")).lower()
    if sort_by not in assignment.ORDERINGS:
        sort_by = "hours"
    try:
        v = assignment.suggestion(threshold, without, sort_by)
    except Exception as e:  # noqa — the path must not die on a compute error
        logger.exception("Assignment: suggestion failed")
        return jsonify({"error": str(e)}), 500
    if v is None:
        state_info = assignment.INVENTORY.fetch()
        return jsonify(
            {
                "done": True,
                "skipped": len(without),
                "sort_by": sort_by,
                "progress": assignment.progress(state_info, threshold),
                "hint": (
                    f"Nothing open left except the {len(without)} skipped ones."
                    if without
                    else "Nothing left open that recurs."
                ),
            }
        )
    v["done"] = False
    return jsonify(v)


@speaker_bp.route("/api/speaker/assignment/apply", methods=["POST"])
def apply_assignment():
    """Assign a name (and spread it) or discard the group."""
    from whisper_tm.core import speaker_assignment as assignment

    d = request.get_json(silent=True) or {}
    ids = d.get("vector_ids") or []
    if not isinstance(ids, list) or not ids:
        return jsonify({"error": "No voices specified"}), 400
    try:
        ids = [int(x) for x in ids]
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid ID"}), 400

    if d.get("trash"):
        db = speaker_db()
        db.move_to_trash(ids, reason="assignment")
        # No recomputation in the click: it cost 8–14 s per "Discard".
        # The held grouping closes the discarded voices
        # (`speaker_assignment.discarded`) and follows in the background.
        state_info = assignment.INVENTORY.fetch()
        return jsonify(
            {
                "ok": True,
                "discarded": len(ids),
                "decision": db.last_decision_id(),
                "progress": assignment.progress(state_info),
            }
        )

    name = str(d.get("name") or "").strip()
    if not name:
        return jsonify({"error": "No name"}), 400
    try:
        result = assignment.apply(ids, name)
    except Exception as e:  # noqa
        logger.exception("Assignment: applying failed")
        return jsonify({"error": str(e)}), 500
    result["ok"] = True
    return jsonify(result)


@speaker_bp.route("/api/speaker/assignment/progress")
def assignment_progress():
    """Progress of the speaker assignment across the library (how many
    speakers are assigned, open, uncertain).
    """
    from whisper_tm.core import speaker_assignment as assignment

    try:
        return jsonify(assignment.progress(assignment.INVENTORY.fetch()))
    except Exception as e:  # noqa
        return jsonify({"error": str(e)}), 500
