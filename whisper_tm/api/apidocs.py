"""API reference, generated from the running app.

    GET /api/docs            → the reference as JSON (the explorer in the
                               setup guide reads this)
    GET /api/docs.md         → the same as Markdown, e.g. to hand to an LLM
        ?brief=1             → only the first paragraph of each docstring
        ?download=1          → as a file download

Nothing here is maintained by hand: routes, methods and path parameters come
from Flask's URL map, descriptions from the view docstrings, and query/body
fields are read from each view's source (``request.args.get("x")``,
``data.get("x")`` on a JSON body). That last part is a best-effort scan and
labelled as such — fields read inside helper functions are not seen.
``tests/test_apidocs.py`` makes sure every route keeps a docstring.
"""

from __future__ import annotations

import inspect
import re
import time
from datetime import datetime

from flask import Blueprint, Response, current_app, jsonify, request

apidocs_bp = Blueprint("apidocs", __name__)

_QUERY = re.compile(r"request\.args\.get(?:list)?\(\s*['\"]([\w\-]+)['\"]")
_QUERY_IDX = re.compile(r"request\.args\[\s*['\"]([\w\-]+)['\"]\s*\]")
# A view reads a JSON/form body …
_HAS_BODY = re.compile(r"request\.(?:json|get_json|form|files)|_mutation_data\(")
# … and then pulls fields from the variable it stored it in.
_BODY = re.compile(r"\b(?:data|body|payload|params|d|j|js|req)\.get\(\s*['\"]([\w\-]+)['\"]")
_BODY_IDX = re.compile(r"\b(?:data|body|payload|params)\[\s*['\"]([\w\-]+)['\"]\s*\]")
_ARG = re.compile(r"<(?:(\w+):)?(\w+)>")
_HIDDEN_METHODS = {"HEAD", "OPTIONS"}


def _first_paragraph(text: str) -> str:
    return (text or "").strip().split("\n\n", 1)[0].strip()


def _unique(items) -> list[str]:
    seen, out = set(), []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _endpoint_info(rule, fn) -> dict:
    target = inspect.unwrap(fn)
    try:
        src = inspect.getsource(target)
    except (OSError, TypeError):
        src = ""
    body = []
    if _HAS_BODY.search(src):
        body = _unique(_BODY.findall(src) + _BODY_IDX.findall(src))
    module = inspect.getmodule(target)
    return {
        "path": rule.rule,
        "methods": sorted(set(rule.methods or ()) - _HIDDEN_METHODS),
        "group": rule.endpoint.split(".", 1)[0] if "." in rule.endpoint else "app",
        "doc": inspect.getdoc(fn) or "",
        "path_params": [{"name": n, "type": t or "string"} for t, n in _ARG.findall(rule.rule)],
        "query_params": _unique(_QUERY.findall(src) + _QUERY_IDX.findall(src)),
        "body_fields": body,
        "stream": "text/event-stream" in src,
        "module_doc": _first_paragraph(inspect.getdoc(module) or "") if module else "",
    }


def collect(app) -> dict:
    """All /api routes of ``app``, grouped, with their descriptions."""
    endpoints = []
    for rule in app.url_map.iter_rules():
        if not rule.rule.startswith("/api"):
            continue
        fn = app.view_functions.get(rule.endpoint)
        if fn is None:
            continue
        endpoints.append(_endpoint_info(rule, fn))
    endpoints.sort(key=lambda e: (e["group"], e["path"], e["methods"]))
    groups: dict[str, dict] = {}
    for e in endpoints:
        g = groups.setdefault(e["group"], {"name": e["group"], "intro": "", "endpoints": []})
        g["intro"] = g["intro"] or e.pop("module_doc")
        e.pop("module_doc", None)
        g["endpoints"].append(e)
    try:
        from whisper_tm import __version__
    except ImportError:
        __version__ = ""
    return {
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "version": __version__,
        "count": len(endpoints),
        "groups": list(groups.values()),
    }


def to_markdown(ref: dict, base_url: str = "", brief: bool = False) -> str:
    out = [
        "# Whisper Transcription Manager — HTTP API reference",
        "",
        f"Generated from the running server on {ref['generated']}"
        + (f" (version {ref['version']})" if ref["version"] else "")
        + f". {ref['count']} endpoints"
        + (f", base URL `{base_url.rstrip('/')}`" if base_url else "")
        + ".",
        "",
        "- Requests and responses are JSON unless an endpoint says otherwise.",
        "- No login by default. With `auth.enabled`, send the API token as the "
        "`X-API-Token` header (or use a logged-in browser session).",
        "- Endpoints marked **stream** answer with Server-Sent Events "
        "(`text/event-stream`), one JSON object per `data:` line.",
        "- *Query parameters* and *JSON body fields* are read from each view's "
        "source code — a best-effort list; the description is authoritative.",
        "- For AI clients there is also an MCP server (`python -m whisper_tm.mcp_server`) "
        "with typed tools for search, transcripts, speakers, conversations and the queue.",
        "",
        "## Contents",
        "",
    ]
    for g in ref["groups"]:
        out.append(f"- [{g['name']}](#{g['name'].lower()}) — {len(g['endpoints'])} endpoints")
    for g in ref["groups"]:
        out += ["", f"## {g['name']}", ""]
        if g["intro"]:
            out += [g["intro"], ""]
        for e in g["endpoints"]:
            head = f"### `{' | '.join(e['methods'])} {e['path']}`"
            out += [head + ("  · **stream**" if e["stream"] else ""), ""]
            doc = _first_paragraph(e["doc"]) if brief else e["doc"].strip()
            out += [doc or "_No description._", ""]
            if e["path_params"]:
                out.append(
                    "- Path parameters: "
                    + ", ".join(f"`{p['name']}` ({p['type']})" for p in e["path_params"])
                )
            if e["query_params"]:
                out.append("- Query parameters: " + ", ".join(f"`{q}`" for q in e["query_params"]))
            if e["body_fields"]:
                out.append("- JSON body fields: " + ", ".join(f"`{b}`" for b in e["body_fields"]))
            if e["path_params"] or e["query_params"] or e["body_fields"]:
                out.append("")
    return "\n".join(out).rstrip() + "\n"


_cache: dict[str, tuple[float, dict]] = {}


def _reference() -> dict:
    """The routes don't change while the server runs — build once per minute at most."""
    hit = _cache.get("ref")
    if hit and time.time() - hit[0] < 60:
        return hit[1]
    ref = collect(current_app)
    _cache["ref"] = (time.time(), ref)
    return ref


@apidocs_bp.route("/api/docs")
def api_docs_json():
    """The API reference as JSON: every /api route with methods, path/query/body
    parameters and its description, grouped by blueprint."""
    return jsonify(_reference())


@apidocs_bp.route("/api/docs.md")
def api_docs_markdown():
    """The API reference as Markdown — e.g. to hand to an LLM.

    ``?brief=1`` keeps only the first paragraph of each description (a
    fraction of the size); ``?download=1`` sends it as a file.
    """
    brief = request.args.get("brief", "") in ("1", "true", "yes")
    md = to_markdown(_reference(), request.host_url, brief=brief)
    resp = Response(md, mimetype="text/markdown")  # the charset comes along
    if request.args.get("download", "") in ("1", "true", "yes"):
        name = "whisper-tm-api" + ("-brief" if brief else "") + ".md"
        resp.headers["Content-Disposition"] = f'attachment; filename="{name}"'
    return resp
