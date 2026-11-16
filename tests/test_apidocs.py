"""Generated API reference (``api/apidocs.py``).

Two promises: every /api route carries a description — otherwise the
reference an LLM gets has holes — and the generator reads methods, path,
query and body parameters correctly from a view.
"""

from __future__ import annotations

import re
from pathlib import Path

from flask import Blueprint, Flask, Response, jsonify, request

from whisper_tm.api.apidocs import apidocs_bp, collect, to_markdown

REPO = Path(__file__).resolve().parent.parent


# ── Every route is documented ────────────────────────────────────────────


def test_every_blueprint_route_has_a_docstring():
    from whisper_tm.api import register_blueprints

    app = Flask(__name__)
    register_blueprints(app)
    ref = collect(app)
    missing = [
        f"{' '.join(e['methods'])} {e['path']}"
        for g in ref["groups"]
        for e in g["endpoints"]
        if not e["doc"].strip()
    ]
    assert not missing, (
        "routes without a docstring (they'd be blank in /api/docs.md):\n" + "\n".join(missing)
    )


def test_every_app_level_route_has_a_docstring():
    """app.py registers a few routes inside create_app — checked from the source."""
    src = (REPO / "whisper_tm" / "app.py").read_text(encoding="utf-8")
    views = re.findall(r'@app\.route\("(/api[^"]*)"[^\n]*\n\s*def \w+\([^)]*\):\n(\s*)(\S+)', src)
    assert views
    missing = [path for path, _indent, first in views if not first.startswith('"""')]
    assert not missing, f"app.py routes without a docstring: {missing}"


# ── What the generator reads ─────────────────────────────────────────────


def _sample_app() -> Flask:
    bp = Blueprint("things", __name__)

    @bp.route("/api/things/<int:thing_id>")
    def get_thing(thing_id):
        """Fetch one thing.

        Second paragraph with details that the brief mode drops."""
        limit = request.args.get("limit", type=int)
        return jsonify({"id": thing_id, "limit": limit})

    @bp.route("/api/things", methods=["POST"])
    def add_thing():
        """Add a thing."""
        data = request.json if isinstance(request.json, dict) else {}
        return jsonify({"name": data.get("name"), "tags": data["tags"]})

    @bp.route("/api/things/stream")
    def thing_stream():
        """Live things."""
        return Response("data: {}\n\n", mimetype="text/event-stream")

    app = Flask(__name__)
    app.register_blueprint(bp)
    app.register_blueprint(apidocs_bp)
    return app


def _endpoint(ref, path):
    return next(e for g in ref["groups"] for e in g["endpoints"] if e["path"] == path)


class TestCollect:
    def test_path_query_and_body(self):
        ref = collect(_sample_app())
        get = _endpoint(ref, "/api/things/<int:thing_id>")
        assert get["methods"] == ["GET"]
        assert get["path_params"] == [{"name": "thing_id", "type": "int"}]
        assert get["query_params"] == ["limit"]
        post = _endpoint(ref, "/api/things")
        assert post["methods"] == ["POST"]
        assert post["body_fields"] == ["name", "tags"]

    def test_stream_is_marked(self):
        ref = collect(_sample_app())
        assert _endpoint(ref, "/api/things/stream")["stream"] is True
        assert _endpoint(ref, "/api/things")["stream"] is False

    def test_grouped_by_blueprint(self):
        names = [g["name"] for g in collect(_sample_app())["groups"]]
        assert "things" in names and "apidocs" in names

    def test_non_api_routes_stay_out(self):
        app = _sample_app()
        app.add_url_rule("/", "index", lambda: "x")
        paths = [e["path"] for g in collect(app)["groups"] for e in g["endpoints"]]
        assert "/" not in paths


class TestMarkdown:
    def test_contains_every_route(self):
        ref = collect(_sample_app())
        md = to_markdown(ref, "http://host:5077/")
        for g in ref["groups"]:
            for e in g["endpoints"]:
                assert e["path"] in md
        assert "base URL `http://host:5077`" in md
        assert "- Query parameters: `limit`" in md

    def test_brief_keeps_the_first_paragraph(self):
        ref = collect(_sample_app())
        assert "Second paragraph" in to_markdown(ref)
        brief = to_markdown(ref, brief=True)
        assert "Fetch one thing." in brief and "Second paragraph" not in brief


class TestEndpoints:
    def test_json(self):
        d = _sample_app().test_client().get("/api/docs").get_json()
        assert d["count"] >= 5 and d["groups"]

    def test_markdown_download(self):
        r = _sample_app().test_client().get("/api/docs.md?download=1&brief=1")
        assert r.status_code == 200
        assert r.mimetype == "text/markdown"
        assert 'filename="whisper-tm-api-brief.md"' in r.headers["Content-Disposition"]
        assert r.get_data(as_text=True).startswith(
            "# Whisper Transcription Manager — HTTP API reference"
        )

    def test_markdown_inline_by_default(self):
        r = _sample_app().test_client().get("/api/docs.md")
        assert "Content-Disposition" not in r.headers
        # The charset once — a hand-written one came twice
        assert r.headers["Content-Type"] == "text/markdown; charset=utf-8"


def test_the_markdown_names_the_login():
    """It said 'there is no authentication' although a login and the
    X-API-Token header exist."""
    from pathlib import Path

    src = (Path(__file__).resolve().parent.parent / "whisper_tm/api/apidocs.py").read_text(
        encoding="utf-8"
    )
    assert "there is no authentication" not in src
    assert "`X-API-Token` header" in src
