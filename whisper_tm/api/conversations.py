"""Conversations — found by who takes part and how they talk: length,
sentence length against the library's usual, speaker changes, kind; for
people in a long recording, the stretch where they talk together
(whisper_tm/conversations).

Endpoints::

    GET  /api/conversations            conversations by people, length, kind, sentences …
    GET  /api/conversations/file       ?file= : one recording — how they talk, who talked when
    GET  /api/conversations/person     ?person= : how one person talks, and with whom
    GET  /api/conversations/people     everyone named who takes part in a counted recording
    GET  /api/conversations/status     the counts, the settings, the background run
    POST /api/conversations/run/start  {"recount": bool}: count what is missing (or all again)
    POST /api/conversations/run/stop
"""

from __future__ import annotations

import logging
import sqlite3

from flask import Blueprint, jsonify, request

from whisper_tm.config import config
from whisper_tm.conversations import query, runner
from whisper_tm.core.helpers import parse_bool, transcript_identity
from whisper_tm.extensions import db

logger = logging.getLogger(__name__)

conversations_bp = Blueprint("conversations", __name__)

SWITCHED_OFF = "Conversation statistics are switched off (conversations.enabled)."
#: A path from a request is never longer than this
PATH_CHARS = 4096


def _refused():
    """No database (501), or the statistics switched off (503) — the status
    still answers then, the page says why the tab is empty."""
    if not db:
        return jsonify({"error": "No database configured"}), 501
    if not runner.settings()["enabled"]:
        return jsonify({"error": SWITCHED_OFF}), 503
    return None


def _failed(what: str, e: Exception):
    """A query that broke: JSON, and the log says where. A busy database
    is worth another try, anything else is the server's."""
    if isinstance(e, sqlite3.OperationalError) and "locked" in str(e):
        return jsonify({"error": "The database is busy — try again in a moment."}), 503
    logger.warning("Conversations: %s failed: %s", what, e, exc_info=True)
    return jsonify({"error": f"{what} failed: {str(e)[:200]}"}), 500


@conversations_bp.route("/api/conversations")
def conversations_find():
    """Find conversations by who takes part and how they talk.

    Whole recordings, or — for people together in a long recording — each
    stretch where all of them talk (``whole`` false, ``start_sec`` …
    ``end_sec``). Every parameter is optional; a value that can't be read
    turns off only its own filter and is named in ``ignored``.

    Query parameters: ``people`` (repeated or comma-separated names of the
    voice library or typed in transcripts), ``match`` (``all`` together,
    default, or ``any``), ``min_share`` (0…0.5: the share of a conversation
    that counts as taking part; default the setting conversations.min_share),
    ``min_sec``/``max_sec`` (its length), ``kind`` (comma list of detailed,
    lively, interview, monologue, conversation, unknown, silent; ``group``:
    three voices or more), ``sentences`` (``longer``/``shorter`` than usual
    for the library's conversations in that language), ``wps_min``/
    ``wps_max`` (words per sentence), ``rate_min``/``rate_max`` (speaker
    changes per minute), ``speakers_min``/``speakers_max`` (active voices),
    ``questions_min`` (0…1 share of the sentences), ``language``, ``q``
    (letters of the path), ``folder`` (absolute, or below the open folder),
    ``date_from``/``date_to`` (YYYY-MM-DD; the date in the file name, else
    the file's), ``sort`` (longest, shortest, sentences, rate, questions,
    monologue, newest, oldest, together — the people's share, with people),
    ``copies`` (``fold``: one result per recording kept several times, or
    ``all``), ``offset``, ``limit`` (1…200, 50).

    Answer: ``total`` (after folding copies) and ``total_sec`` (how long
    they last together), ``results`` (file names as the searches hand them
    on: ``filename`` below ``file_dir``), ``baseline``
    (the deciles "usual" refers to), ``people_unknown``, ``ignored``,
    ``status`` (how much of the library is counted), ``media_dir``.
    """
    refused = _refused()
    if refused:
        return refused
    raw = {
        "people": request.args.getlist("people"),
        "match": request.args.get("match"),
        "min_share": request.args.get("min_share"),
        "min_sec": request.args.get("min_sec"),
        "max_sec": request.args.get("max_sec"),
        "kind": request.args.getlist("kind"),
        "sentences": request.args.get("sentences"),
        "wps_min": request.args.get("wps_min"),
        "wps_max": request.args.get("wps_max"),
        "rate_min": request.args.get("rate_min"),
        "rate_max": request.args.get("rate_max"),
        "speakers_min": request.args.get("speakers_min"),
        "speakers_max": request.args.get("speakers_max"),
        "questions_min": request.args.get("questions_min"),
        "language": request.args.get("language"),
        "q": request.args.get("q"),
        "folder": request.args.get("folder"),
        "date_from": request.args.get("date_from"),
        "date_to": request.args.get("date_to"),
        "sort": request.args.get("sort"),
        "copies": request.args.get("copies"),
    }
    filters, ignored = query.parse_filters(raw, media_dir=str(config.media_dir))
    offset, limit, bad = query.parse_page(request.args.get("offset"), request.args.get("limit"))
    try:
        out = query.find(db, filters, offset=offset, limit=limit)
    except Exception as e:  # noqa: BLE001 — answered as JSON, see _failed
        return _failed("Finding conversations", e)
    # (too many people: the parsing says so for many values, the search for
    # one value with many names — once is enough)
    out["ignored"] = list(dict.fromkeys(ignored + bad + out.get("ignored", [])))
    out["media_dir"] = str(config.media_dir)
    return jsonify(out)


@conversations_bp.route("/api/conversations/file")
def conversations_file():
    """One recording's conversation: how they talk, who talked when.

    ``?file=`` — the absolute path (or one below the open folder). Answer:
    ``summary`` (every counted number, with ``wps_pct``/``rate_pct``/
    ``turn_pct``: where it lies among the library's conversations in that
    language), ``kind``, ``group``, ``reasons``, ``speakers`` (per label,
    with its name where one is known and whether it is ``active``),
    ``people`` (per person, their labels added up), ``unnamed`` (active
    voices without a name), ``parts`` (who talked when; empty for
    recordings shorter than ten minutes or without speakers told apart),
    ``baseline``. ``counted`` false: its numbers are not counted yet. 404:
    no transcript.
    """
    refused = _refused()
    if refused:
        return refused
    raw = (request.args.get("file") or "").strip()[:PATH_CHARS]
    if not raw:
        return jsonify({"error": "No file given (?file=)"}), 400
    abs_path = transcript_identity(raw, media_dir=str(config.media_dir))
    if not abs_path:
        return jsonify({"error": "Path not resolvable"}), 400
    try:
        out = query.file_stats(db, abs_path)
    except Exception as e:  # noqa: BLE001
        return _failed("Reading the conversation", e)
    if out is None:
        return jsonify({"error": "No transcript for this file", "abs_path": abs_path}), 404
    out["media_dir"] = str(config.media_dir)
    return jsonify(out)


def _share():
    """``min_share`` of a request (0…0.5) — None: the setting."""
    filters, _ = query.parse_filters({"min_share": request.args.get("min_share")})
    return filters.min_share


@conversations_bp.route("/api/conversations/person")
def conversations_person():
    """How one person talks, and with whom.

    ``?person=`` — a name of the voice library or typed in a transcript (any
    case); ``min_share`` as for the search. Answer: ``recordings`` they take
    part in, ``conversations`` (of those, with two voices or more),
    ``talk_sec``, their own ``wps`` and ``questions_share``, ``partners``
    (who takes part in the same recordings, by how many), ``kinds``,
    ``first_date``/``last_date``, ``longest`` (their five longest
    conversations, as the search answers them). A name nobody has:
    ``known`` false and zeros.
    """
    refused = _refused()
    if refused:
        return refused
    name = query.clean_name(request.args.get("person"))
    if not name:
        return jsonify({"error": "No person given (?person=)"}), 400
    try:
        out = query.person(db, name, min_share=_share())
    except Exception as e:  # noqa: BLE001
        return _failed("Reading the person", e)
    out = dict(out, media_dir=str(config.media_dir))
    return jsonify(out)


@conversations_bp.route("/api/conversations/people")
def conversations_people():
    """Everyone named who takes part in at least one counted recording —
    ``[{name, recordings, conversations}]``, most conversations first (the
    people picker's list). ``min_share`` as for the search."""
    refused = _refused()
    if refused:
        return refused
    try:
        return jsonify(query.people_list(db, min_share=_share()))
    except Exception as e:  # noqa: BLE001
        return _failed("Listing the people", e)


@conversations_bp.route("/api/conversations/status")
def conversations_status():
    """How much of the library is counted, the settings and the background
    run — answers also while the statistics are switched off."""
    if not db:
        return jsonify({"error": "No database configured"}), 501
    try:
        counts = query.counts(db)
    except Exception as e:  # noqa: BLE001 — the status must answer
        logger.warning("Conversations: counts not readable: %s", e)
        counts = {"transcripts": 0, "counted": 0, "stale": 0}
    settings = runner.settings()
    return jsonify(
        {
            "enabled": settings["enabled"],
            "settings": settings,
            "counts": counts,
            "run": runner.RUNNER.status(),
        }
    )


@conversations_bp.route("/api/conversations/run/start", methods=["POST"])
def conversations_run_start():
    """Count the transcripts without numbers in the background —
    ``{"recount": true}``: every transcript again (the old numbers stay
    readable until then). ``started`` false: a run is going."""
    if not db:
        return jsonify({"error": "No database configured"}), 501
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        data = {}
    recount = parse_bool(data.get("recount", False))
    started = runner.RUNNER.start(reason="count again" if recount else "manual", recount=recount)
    return jsonify({"started": started, "run": runner.RUNNER.status()})


@conversations_bp.route("/api/conversations/run/stop", methods=["POST"])
def conversations_run_stop():
    """Stop the background run after its current batch — what it counted
    stays. ``stopped`` false: none was going."""
    if not db:
        return jsonify({"error": "No database configured"}), 501
    return jsonify({"stopped": runner.RUNNER.stop()})
