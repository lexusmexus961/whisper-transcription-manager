"""Links in — YouTube videos to transcripts, summaries, analyses (core/ingest.py).

For the browser extension, and anything else with links: ``POST /api/ingest``
takes the links and a mode and answers at once; ``GET /api/ingest/<id>``
says where every video stands — downloading, queued, transcribed,
summarized (with the summary). ``POST /api/ingest/known`` tells which videos
of a page are in the library already.

The extension's side panel: ``GET /api/ingest/options`` (the choices and the
defaults of its forms), ``POST /api/ingest/defaults`` (its ⚙),
``POST /api/ingest/transcript`` (a video's transcript and summaries) and
``POST /api/ingest/ask`` (a chat conversation about a video — asked on
through ``/api/chat/send``).
"""

from __future__ import annotations

from flask import Blueprint, jsonify, request

from whisper_tm.core import ingest

ingest_bp = Blueprint("ingest", __name__)


def _body() -> dict:
    d = request.get_json(silent=True)
    return d if isinstance(d, dict) else {}


def _fail(e: ingest.IngestError):
    return jsonify({"error": str(e)}), (404 if str(e) == "unknown batch" else 400)


@ingest_bp.route("/api/ingest", methods=["POST"])
def ingest_start():
    """Start. Body: ``urls`` (videos, playlists), ``mode`` (transcribe |
    summarize | analyze), for an analysis ``task`` and ``title``;
    ``audio_only`` (default: the setting youtube.audio_only); ``options`` —
    how they are transcribed and summarized (see GET /api/ingest/options;
    left out = the defaults); ``again`` — one more summary of a video
    summarized before."""
    d = _body()
    urls = d.get("urls")
    if isinstance(urls, str):
        urls = [urls]
    if not isinstance(urls, list):
        return jsonify({"error": "urls must be a list"}), 400
    audio_only = d.get("audio_only")
    options = d.get("options")
    if options is not None and not isinstance(options, dict):
        return jsonify({"error": "options must be an object"}), 400
    try:
        b = ingest.start(
            urls,
            str(d.get("mode") or "transcribe"),
            task=str(d.get("task") or ""),
            title=str(d.get("title") or ""),
            audio_only=None if audio_only is None else bool(audio_only),
            options=options,
            again=bool(d.get("again")),
        )
    except ingest.IngestError as e:
        return _fail(e)
    return jsonify({"ok": True, "batch": ingest.public(b)})


@ingest_bp.route("/api/ingest")
def ingest_list():
    """The recent batches, newest first."""
    try:
        limit = max(1, min(int(request.args.get("limit", 30)), 200))
    except ValueError:
        limit = 30
    return jsonify({"batches": ingest.listing(limit)})


@ingest_bp.route("/api/ingest/<bid>")
def ingest_get(bid):
    """One batch, every video's state read now."""
    try:
        return jsonify(ingest.public(ingest.load(bid)))
    except ingest.IngestError as e:
        return _fail(e)


@ingest_bp.route("/api/ingest/<bid>/cancel", methods=["POST"])
def ingest_cancel(bid):
    """Stop the downloads; what is queued stays queued."""
    try:
        return jsonify({"ok": ingest.cancel(bid)})
    except ingest.IngestError as e:
        return _fail(e)


@ingest_bp.route("/api/ingest/<bid>/drop", methods=["POST"])
def ingest_drop(bid):
    """Take videos out of a batch. Body: ``urls`` — left out, all of them.
    One not fetched yet isn't, one downloading stops, one the batch put in
    the queue leaves it while it waits; transcripts stay."""
    urls = _body().get("urls")
    if urls is not None and not (isinstance(urls, list) and all(isinstance(u, str) for u in urls)):
        return jsonify({"error": "urls must be a list of links"}), 400
    try:
        return jsonify({"ok": True, **ingest.drop(bid, urls)})
    except ingest.IngestError as e:
        return _fail(e)


@ingest_bp.route("/api/ingest/known", methods=["POST"])
def ingest_known():
    """Which of these videos are in the library already. Body:
    ``videos: [{url, title}]`` — the title finds a download from before the
    video ids were noted; ``summaries: true`` adds each one's summary."""
    d = _body()
    videos = d.get("videos")
    if not isinstance(videos, list):
        return jsonify({"error": "videos must be a list"}), 400
    return jsonify(
        {
            "videos": ingest.known(
                [v for v in videos if isinstance(v, dict)], summaries=bool(d.get("summaries"))
            )
        }
    )


@ingest_bp.route("/api/ingest/options")
def ingest_options():
    """What the extension's forms offer: Whisper models, languages,
    vocabularies, summary templates and models — and what is set: the
    general settings and the defaults for videos from the browser."""
    return jsonify(ingest.options())


@ingest_bp.route("/api/ingest/defaults", methods=["POST"])
def ingest_defaults():
    """Change the defaults for videos from the browser — the options sent;
    "" (or null) = the general setting."""
    return jsonify({"ok": True, "defaults": ingest.save_defaults(_body())})


@ingest_bp.route("/api/ingest/transcript", methods=["POST"])
def ingest_transcript():
    """A video's transcript and summaries. Body: ``url``, ``title``."""
    d = _body()
    try:
        return jsonify(ingest.transcript(str(d.get("url") or ""), str(d.get("title") or "")))
    except ingest.IngestError as e:
        return jsonify({"error": str(e)}), 404


@ingest_bp.route("/api/ingest/ask", methods=["POST"])
def ingest_ask():
    """A conversation about a video, its transcript attached — the one from
    before unless ``new``. Body: ``url``, ``title``, ``new``. Answers
    ``conversation`` (asked on through POST /api/chat/send)."""
    d = _body()
    try:
        return jsonify(
            ingest.ask(str(d.get("url") or ""), str(d.get("title") or ""), new=bool(d.get("new")))
        )
    except ingest.IngestError as e:
        return jsonify({"error": str(e)}), 404
