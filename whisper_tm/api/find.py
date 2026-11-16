"""
Find anything — the search view's start tab (find/agent.py).

One description; the language model plans the searches (words, meaning,
the picture, the sound, inside the remembered files, in what the named
people say), they run at once, copies are folded, and the best moments are
judged by the language model. The page starts a run and polls it; one run
at a time.

Endpoints::

    POST /api/find                 {"query", "owner"} → {"id"}   owner: "tab" (default)
                                   or "chat" — a new run stops only its owner's
    GET  /api/find/<id>?rev=N&erev=M   the run (unchanged since N → {"rev", "same": true};
                                   the searches' lists left out while erev is M)
    POST /api/find/<id>/stop
    POST /api/find/<id>/ask        a conversation in Ask with the run attached
    GET  /api/find/frame?key=<recording>&t=<seconds>
    GET  /api/find/clip?key=<recording>&t=<seconds>[&end=<seconds>]
"""

from __future__ import annotations

from flask import Blueprint, jsonify, request, send_file

find_bp = Blueprint("find", __name__)


@find_bp.route("/api/find", methods=["POST"])
def find_start():
    """Start a run for a description (``query``, at least 3 characters) —
    the one still going stops."""
    from whisper_tm.find import agent

    data = request.get_json(silent=True) or {}
    # A JSON array must not be a 500, nor a list as the query become its repr
    if not isinstance(data, dict) or not isinstance(data.get("query") or "", str):
        return jsonify(error='Send {"query": "a description"}'), 400
    q = " ".join(str(data.get("query") or "").split())
    if len(q) < 3:
        return jsonify(error="Describe what you are looking for (at least 3 characters)"), 400
    owner = str(data.get("owner") or "tab")
    if owner not in agent.OWNERS:
        return jsonify(error=f"owner is one of {list(agent.OWNERS)}"), 400
    run = agent.start(q, owner)
    return jsonify(id=run.id, state=run.snapshot())


def _int_arg(name: str) -> int:
    try:
        return int(request.args.get(name) or -1)
    except ValueError:
        return -1


@find_bp.route("/api/find/<run_id>")
def find_state(run_id: str):
    """The run's plan, every search's state and hits, the best matches and
    the log — or only ``{"same": true}`` when nothing changed since ``rev``;
    the searches' lists only when they changed since ``erev``."""
    from whisper_tm.find import agent

    run = agent.get(run_id)
    if not run:
        return jsonify(error="No such run (they are kept until the server restarts)"), 404
    since = _int_arg("rev")
    if since >= 0 and run.state["rev"] == since:
        return jsonify(rev=since, same=True)
    erev = _int_arg("erev")
    return jsonify(run.snapshot(erev if erev >= 0 else None))


@find_bp.route("/api/find/<run_id>/stop", methods=["POST"])
def find_stop(run_id: str):
    """Stop a run — what was found stays, the judging stops."""
    from whisper_tm.find import agent

    if not agent.get(run_id):
        return jsonify(error="No such run (they are kept until the server restarts)"), 404
    return jsonify(stopping=agent.stop(run_id))


@find_bp.route("/api/find/<run_id>/ask", methods=["POST"])
def find_ask(run_id: str):
    """A conversation in Ask about this run — its best matches attached."""
    from whisper_tm.find import agent

    try:
        return jsonify(ok=True, conversation=agent.ask(run_id))
    except LookupError as e:
        return jsonify(error=str(e)), 404


# JSON like the rest of the API — abort() would answer with an HTML page
def _bad_time():
    return jsonify(error="t and end are seconds (a finite number)"), 400


def _not_found(why: str = "not a recording of the library"):
    return jsonify(error=f"Nothing to show: {why}"), 404


def _past_the_end(t: float, rec: dict) -> bool:
    return rec["duration"] > 0 and t >= rec["duration"]


@find_bp.route("/api/find/frame")
def find_frame():
    """A still of a recording of the library at ``t`` seconds — for the
    moments Images & sounds hasn't indexed (cut once, kept in the cache)."""
    from whisper_tm.extensions import db
    from whisper_tm.find import agent

    key = request.args.get("key") or ""
    t = agent.finite_seconds(request.args.get("t") or 0)
    if t is None:
        return _bad_time()
    rec = agent.recording(db, key) if key and db else None
    if not rec:
        return _not_found()
    if _past_the_end(t, rec):
        return _not_found("past the recording's end")
    out = agent.cut_frame(key, t)
    if not out:
        return _not_found("no still there")
    return send_file(str(out), mimetype="image/jpeg", max_age=86400)


@find_bp.route("/api/find/clip")
def find_clip():
    """The moment, playable in the preview — from a little before ``t`` (a
    passage to its ``end``), H.264 with its sound or AAC for a recording
    without a picture, whatever the recording's own format. Cut once, then
    kept (find/agent.cut_clip)."""
    from whisper_tm.extensions import db
    from whisper_tm.find import agent

    key = request.args.get("key") or ""
    t = agent.finite_seconds(request.args.get("t") or 0)
    end = agent.finite_seconds(request.args["end"]) if request.args.get("end") else None
    if t is None or (request.args.get("end") and end is None):
        return _bad_time()
    rec = agent.recording(db, key) if key and db else None
    if not rec:
        return _not_found()
    if _past_the_end(t, rec):
        # Otherwise an empty clip would be answered with a 200 and kept in the cache
        return _not_found("past the recording's end")
    start, length = agent.clip_window(t, end)
    if rec["duration"] > 0:
        length = round(min(length, rec["duration"] - start), 1)
    out = agent.cut_clip(key, start, length, rec["video"])
    if not out:
        return _not_found("no clip there")
    return send_file(
        str(out),
        mimetype="video/mp4" if rec["video"] else "audio/mp4",
        conditional=True,
        max_age=86400,
    )
