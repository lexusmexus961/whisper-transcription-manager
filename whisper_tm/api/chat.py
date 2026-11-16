"""Ask the library — the chat's routes (core/chat.py).

A turn runs in the background: ``POST /api/chat/send`` starts it and
answers with the run's id at once, ``GET /api/chat/runs/<id>`` hands out
its events (text as it is written, each tool call, the end) from a given
number on. Polled, not streamed: a turn can take minutes, and a long-held
request is what a proxy or a sleeping laptop cuts first.
"""

from __future__ import annotations

from flask import Blueprint, jsonify, request

from whisper_tm.core import chat

chat_bp = Blueprint("chat", __name__)


@chat_bp.route("/api/chat/status")
def chat_status():
    """What the chat needs — model, endpoint, the MCP tools it may use."""
    return jsonify(chat.readiness())


@chat_bp.route("/api/chat/conversations")
def chat_conversations():
    """The saved conversations, newest first."""
    return jsonify({"conversations": chat.listing()})


@chat_bp.route("/api/chat/conversations/<cid>")
def chat_conversation(cid):
    """One conversation: its turns and references."""
    try:
        return jsonify(chat.public(chat.load(cid)))
    except chat.ChatError as e:
        return jsonify({"error": str(e)}), 404


@chat_bp.route("/api/chat/conversations/<cid>", methods=["DELETE"])
def chat_delete(cid):
    """Delete a conversation."""
    try:
        if not chat.delete(cid):
            # Gone already — a 404 like every unknown id
            return jsonify({"error": "unknown conversation"}), 404
        return jsonify({"ok": True})
    except chat.ChatError as e:
        return jsonify({"error": str(e)}), 404


@chat_bp.route("/api/chat/send", methods=["POST"])
def chat_send():
    """Ask. Body: ``text``, ``conversation`` (id, empty = a new one),
    ``scope`` (what the page shows — the search, its filters — as text for
    the model)."""
    d = request.get_json(silent=True) or {}
    if not isinstance(d, dict):
        # A JSON array must not end in a 500
        return jsonify({"error": "send an object: text, conversation, scope"}), 400
    try:
        run = chat.start(
            str(d.get("conversation") or "") or None,
            str(d.get("text") or ""),
            str(d.get("scope") or ""),
        )
    except chat.ChatError as e:
        return jsonify({"error": str(e)}), 400
    return jsonify({"ok": True, "run": run.id, "conversation": run.conv["id"]})


@chat_bp.route("/api/chat/runs/<rid>")
def chat_run(rid):
    """A run's events from number ``after`` on (the last one again: its
    text may have grown)."""
    run = chat.get_run(rid)
    if run is None:
        return jsonify({"error": "unknown run"}), 404
    try:
        after = max(0, int(request.args.get("after", 0)))
    except ValueError:
        after = 0
    return jsonify({"events": run.since(after), "done": run.done, "conversation": run.conv["id"]})


@chat_bp.route("/api/chat/runs/<rid>/stop", methods=["POST"])
def chat_stop(rid):
    """Stop a run — the answer so far is kept."""
    if chat.get_run(rid) is None:
        return jsonify({"error": "unknown run"}), 404
    return jsonify({"ok": chat.stop(rid)})
