"""Setup guide: first-run flag, system check, GPU test, the Hugging Face
token for speaker diarization, internal docs.

The guide opens by itself on a fresh installation (no transcripts yet and
not dismissed) and from 🧭 in the top bar at any time. Checks come from
``core/diagnostics.py``; the GPU test runs in a subprocess and is refused
while a transcription runs — it would compete for the same VRAM.
"""

from __future__ import annotations

import logging
import re
import threading
from pathlib import Path

from flask import Blueprint, jsonify, request

from whisper_tm.config import config
from whisper_tm.core import debug
from whisper_tm.extensions import db

logger = logging.getLogger(__name__)

setup_bp = Blueprint("setup", __name__)

DONE_KEY = "setup_guide_done"

#: One GPU test at a time.
_gpu_test_lock = threading.Lock()

# docs/ of the repository. In a Docker image it may be missing — the guide
# then simply lists no documents.
DOCS_DIR = Path(__file__).resolve().parent.parent.parent / "docs"


def _has_transcripts() -> bool:
    try:
        with db.connection() as conn:
            return conn.execute("SELECT 1 FROM transcripts LIMIT 1").fetchone() is not None
    except Exception:
        return False


@setup_bp.route("/api/setup/status")
def setup_status():
    """Should the guide open by itself?

    Only on a fresh installation: an existing library is not "first run",
    even if the flag is missing because it predates the guide.
    """
    if not db:
        return jsonify({"first_run": False, "done": False, "commands": _commands()})
    done = db.get_setting(DONE_KEY, "") == "1"
    return jsonify(
        {"first_run": not done and not _has_transcripts(), "done": done, "commands": _commands()}
    )


def _commands() -> dict:
    """The setup and start commands, for the guide's texts — and whether the
    server runs in the Docker image (no scripts, no llama.cpp there)."""
    from whisper_tm.core import commands

    return {
        "setup_llama": commands.setup("--llama"),
        "start": commands.start(),
        "start_llm": commands.start_llm(),
        "check": commands.start("--check"),
        "docker": commands.in_docker(),
        "host_url": commands.DOCKER_HOST_URL,
    }


@setup_bp.route("/api/setup/done", methods=["POST"])
def setup_done():
    """Mark the setup guide as seen: it no longer opens by itself."""
    if db:
        db.set_setting(DONE_KEY, "1")
    return jsonify({"ok": True})


def _paths() -> dict:
    return {
        "config": str(config.config_path or ""),
        "database": str(config.db_path or ""),
        "media": str(config.media_dir or ""),
        "work": str(config.work_dir or ""),
        "logs_debug": str(debug.logs_root()),
        "logs_jobs": str(config.dirs.get("logs") or ""),
        "docs": str(DOCS_DIR) if DOCS_DIR.is_dir() else "",
    }


def _ports() -> dict:
    return {
        "web": request.host.rsplit(":", 1)[-1]
        if ":" in request.host
        else str(config.get("server", "port", default=9860)),
        "mcp_http": config.get("mcp", "port", default=5399),
        "mcp_openapi": config.get("mcp", "openapi_port", default=5400),
        "llama": config.get("llama", "port", default=8080),
        "embedding": config.get("embedding", "local_port", default=8081),
    }


@setup_bp.route("/api/setup/check")
def setup_check():
    """Quick system check — never touches the GPU."""
    from whisper_tm.core import diagnostics

    result = diagnostics.quick_checks()
    result["paths"] = _paths()
    result["ports"] = _ports()
    return jsonify(result)


@setup_bp.route("/api/setup/gpu_test", methods=["POST"])
def setup_gpu_test():
    """Compute on the GPU in a subprocess (PyTorch, CTranslate2, Whisper tiny)."""
    from whisper_tm.core import diagnostics
    from whisper_tm.core.model_idle import _job_running

    if _job_running():
        return jsonify(
            {
                "ok": False,
                "steps": [],
                "error": "A transcription is running — test after it is done.",
            }
        ), 409
    idx = int(config.get("hardware", "gpu_index", default=0) or 0)
    # One test at a time, and the card to itself while it runs: the queue's
    # GPU gate is held exclusively, so a file starting meanwhile waits at
    # its first GPU stage instead of meeting Whisper tiny on the card. Only
    # taken when free — a running stage means a job uses the GPU right now.
    if not _gpu_test_lock.acquire(blocking=False):
        return jsonify({"ok": False, "steps": [], "error": "A GPU test is already running."}), 409
    try:
        from whisper_tm.core.gpu_gate import GATE

        with GATE.try_exclusive("GPU test (setup guide)") as got:
            if not got:
                return jsonify(
                    {
                        "ok": False,
                        "steps": [],
                        "error": "The GPU is in use by the queue — test after it is done.",
                    }
                ), 409
            result = diagnostics.run_gpu_test_subprocess(idx)
        return jsonify(result)
    finally:
        _gpu_test_lock.release()


@setup_bp.route("/api/setup/hf_token")
def setup_hf_token_status():
    """Speaker diarization's Hugging Face side, without the network: where
    the token comes from (``config``, ``env`` = HF_TOKEN, "" = none), whether
    the gated model is downloaded, and the links of the steps."""
    from whisper_tm.core import hf_access

    return jsonify(
        {
            "token": hf_access.token_source(),
            "model_cached": hf_access.model_cached(),
            "repo": hf_access.REPO,
            "repo_url": hf_access.REPO_URL,
            "tokens_url": hf_access.TOKENS_URL,
            "join_url": hf_access.JOIN_URL,
            "model_mb": hf_access.MODEL_MB,
        }
    )


#: What a Hugging Face token looks like ("hf_" and letters/digits); anything
#: else is refused before it goes into a request header or the config
_TOKEN_RE = re.compile(r"[A-Za-z0-9_-]{8,200}")


@setup_bp.route("/api/setup/hf_token", methods=["POST"])
def setup_hf_token_check():
    """Check a Hugging Face token, save it, download the diarization model.

    Body ``{"token": "hf_…"}``; without one the configured token is checked.
    A given token is saved when Hugging Face accepts it (status ``ok`` or
    ``gated``) or can't be reached (``offline``, saved unchecked); with
    access the model (~32 MB) is downloaded right away. The answer never
    contains the token: ``{status, user, message, saved, model_cached}``,
    statuses as in core/hf_access.check.
    """
    from whisper_tm.core import hf_access

    body = request.get_json(silent=True)
    if body is not None and not isinstance(body, dict):
        return jsonify({"error": "expected a JSON object"}), 400
    given = str((body or {}).get("token") or "").strip()
    if given and not _TOKEN_RE.fullmatch(given):
        return jsonify({"error": "that is not a Hugging Face token"}), 400
    res = hf_access.check(given or config.hf_token)
    out = {"status": res["status"], "user": res["user"], "message": res["message"], "saved": False}
    if given and res["status"] in ("ok", "gated", "offline"):
        try:
            hf_access.save_token(given)
            out["saved"] = True
        except OSError as e:
            out["save_error"] = f"The token could not be saved: {e.strerror or e}"
    cached = hf_access.model_cached()
    if res["status"] == "ok" and not cached:
        try:
            hf_access.download(given or config.hf_token)
            cached = True
        except Exception as e:  # noqa: BLE001 — the first diarized file tries again
            logger.warning("Diarization model not downloaded: %s", type(e).__name__)
            out["download_error"] = (
                f"The download failed ({type(e).__name__}); the "
                "first file with speakers tries again."
            )
    out["model_cached"] = cached
    return jsonify(out)


def _doc_files() -> dict[str, Path]:
    """Markdown files in docs/ by name. Only .md — any other file in docs/
    (data files, JSON) is never served."""
    if not DOCS_DIR.is_dir():
        return {}
    return {p.name: p for p in sorted(DOCS_DIR.glob("*.md")) if p.is_file()}


@setup_bp.route("/api/setup/docs")
def setup_docs():
    """The design documents in docs/ (Markdown only): name and title."""
    out = []
    for name, p in _doc_files().items():
        title = name
        try:
            first = p.read_text(encoding="utf-8").lstrip().splitlines()[0]
            title = first.lstrip("# ").strip() or name
        except (OSError, IndexError):
            pass
        out.append({"name": name, "title": title})
    return jsonify({"docs": out})


@setup_bp.route("/api/setup/docs/<name>")
def setup_doc(name: str):
    """One design document's text by its name from /api/setup/docs."""
    # Looked up in the listing, never joined to a path — no traversal.
    p = _doc_files().get(name)
    if p is None:
        return jsonify({"error": "unknown document"}), 404
    return jsonify({"name": name, "text": p.read_text(encoding="utf-8")})
