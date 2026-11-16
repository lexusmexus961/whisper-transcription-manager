"""
Authentication: optional password protection.

Supports:
- Session-based (browser login via form)
- Token-based (API access via X-API-Token header)
- Public paths (health check, static files)

Activation: auth.enabled: true in config.yaml
For production environments a reverse proxy with LDAP/SSO is recommended.
"""

from __future__ import annotations

import hmac
import logging
import threading
import time

from flask import jsonify, redirect, render_template_string, request, session

from whisper_tm.config import config

logger = logging.getLogger(__name__)

# ── Login Template ────────────────────────────────────────────────────────

_LOGIN_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Whisper Transcription Manager — Login</title>
<style>
* { margin:0; padding:0; box-sizing:border-box; }
body { font-family:'DM Sans',sans-serif; background:#07070c; color:#e4e4ee;
       display:flex; align-items:center; justify-content:center; min-height:100vh; }
.login-box { background:#0d0d14; border:1px solid #2a2a3a; border-radius:12px;
             padding:40px; width:100%; max-width:380px; text-align:center; }
.login-box h1 { font-family:'IBM Plex Mono',monospace; font-size:1rem;
                color:#3b82f6; margin-bottom:8px; }
.login-box p { color:#8a8aa2; font-size:0.8rem; margin-bottom:24px; }
.login-box input { width:100%; padding:10px 14px; background:#13131c;
                   border:1px solid #2a2a3a; border-radius:8px; color:#e4e4ee;
                   font-family:inherit; font-size:0.9rem; outline:none;
                   margin-bottom:16px; }
.login-box input:focus { border-color:#3b82f6; }
.login-box button { width:100%; padding:10px; background:#3b82f6; color:#fff;
                    border:none; border-radius:8px; font-family:inherit;
                    font-size:0.9rem; font-weight:600; cursor:pointer; }
.login-box button:hover { background:#2563eb; }
.error { color:#ef4444; font-size:0.8rem; margin-bottom:12px; }
</style>
</head>
<body>
<div class="login-box">
    <h1>WHISPERX PRO</h1>
    <p>Please enter password</p>
    {% if error %}<div class="error">{{ error }}</div>{% endif %}
    <form method="POST" action="/auth/login">
        <input type="password" name="password" placeholder="Password" autofocus>
        <button type="submit">Log in</button>
    </form>
</div>
</body>
</html>"""


# ── Auth Check ────────────────────────────────────────────────────────────


def is_auth_enabled() -> bool:
    """Whether authentication is enabled in the config."""
    return bool(config.get("auth", "enabled", default=False))


def _get_password() -> str:
    return config.get("auth", "password", default="") or ""


def _get_api_token() -> str:
    return config.get("auth", "api_token", default="") or ""


def _get_public_paths() -> list[str]:
    return config.get("auth", "public_paths", default=["/api/health", "/static/"])


#: Public paths without a trailing slash that are meant exactly as written.
_EXACT_DEFAULTS = ("/api/health",)


def public_path_warnings() -> list[str]:
    """Warnings for custom public paths that lost their subtree.

    Earlier versions let an entry like "/api/foo" also make everything below it
    public ("/api/foo/bar"). Now only an entry ending in "/" does — a
    custom entry without one is named at startup, so the change doesn't
    pass unnoticed."""
    out = []
    for pp in _get_public_paths() or []:
        pp = str(pp)
        if pp and not pp.endswith("/") and pp not in _EXACT_DEFAULTS:
            out.append(
                f"auth.public_paths: {pp!r} now matches only exactly — "
                f"write {pp + '/'!r} to keep everything below it public"
            )
    return out


def _is_public_path(path: str) -> bool:
    """Check whether the path is reachable without auth."""
    public = _get_public_paths()
    for pp in public:
        # Directory entries (ending in "/") match by prefix; anything else
        # only exactly. "/api/health" used to match below itself too, which
        # left /api/health/detail (current file, paths, NAS errors) open
        # with the login on. Whoever wants a subtree public writes it with
        # its trailing slash.
        if pp.endswith("/"):
            if path.startswith(pp):
                return True
        elif path == pp:
            return True
    # Login paths are always public
    return path.startswith("/auth/")


def _check_password(input_password: str) -> bool:
    """Check password (plain text or bcrypt)."""
    stored = _get_password()
    if not stored:
        return False

    # Detect bcrypt hash
    if stored.startswith("$2b$") or stored.startswith("$2a$"):
        try:
            import bcrypt

            return bcrypt.checkpw(input_password.encode(), stored.encode())
        except ImportError:
            # Without bcrypt a hash CANNOT be checked. The previous
            # "plain-text comparison" compared the password with the hash
            # string and thus always failed — misleading. Fail clearly:
            logger.error(
                "Password is bcrypt-hashed, but bcrypt is not "
                "installed — login impossible. Fix: pip install bcrypt"
            )
            return False

    # Plain text (for simple setups). As bytes: compare_digest refuses str
    # with anything but ASCII — a password with an umlaut crashed the login
    return hmac.compare_digest(input_password.encode("utf-8"), stored.encode("utf-8"))


def _check_token(token: str) -> bool:
    """Check API token."""
    stored = _get_api_token()
    if not stored:
        return False
    return hmac.compare_digest(token.encode("utf-8"), stored.encode("utf-8"))


# ── Guessing brake ────────────────────────────────────────────────────────

#: Failed logins or tokens a client may have in FAIL_WINDOW before it waits
FREE_FAILURES = 5
FAIL_WINDOW = 15 * 60
#: The wait doubles from 1 s with every failure past FREE_FAILURES, up to this
MAX_WAIT = 15 * 60

_failures: dict[str, list[float]] = {}
_failures_lock = threading.Lock()


def _client() -> str:
    return request.remote_addr or "?"


def wait_left(client: str, now: float | None = None) -> float:
    """Seconds this client still has to wait before its next attempt."""
    now = time.time() if now is None else now
    with _failures_lock:
        times = [t for t in _failures.get(client, []) if now - t < FAIL_WINDOW]
        if times:
            _failures[client] = times
        else:
            _failures.pop(client, None)
        over = len(times) - FREE_FAILURES
        if over < 0:
            return 0.0
        return max(0.0, times[-1] + min(2.0**over, MAX_WAIT) - now)


def note_failure(client: str, now: float | None = None) -> None:
    with _failures_lock:
        _failures.setdefault(client, []).append(time.time() if now is None else now)
        if len(_failures) > 10000:  # a flood of addresses: forget the oldest
            for k in sorted(_failures, key=lambda k: _failures[k][-1])[:5000]:
                _failures.pop(k, None)


def note_success(client: str) -> None:
    with _failures_lock:
        _failures.pop(client, None)


def _too_many(client: str):
    left = wait_left(client)
    if left <= 0:
        return None
    secs = int(left) + 1
    logger.warning("Login attempts from %s paused for %d s", client, secs)
    body = {"error": f"Too many failed attempts — try again in {secs} s"}
    return jsonify(body), 429, {"Retry-After": str(secs)}


def _password_mark() -> str:
    """A short keyed hash of the stored password: a login cookie carries it,
    so changing auth.password ends every session made with the old one."""
    from flask import current_app

    key = str(current_app.config.get("SECRET_KEY") or "").encode()
    return hmac.new(key, _get_password().encode("utf-8"), "sha256").hexdigest()[:16]


# ── Flask Integration ─────────────────────────────────────────────────────


def register_auth(app):
    """Register auth middleware and login routes.

    Only called when auth.enabled = true.
    """

    @app.before_request
    def _check_auth():
        # Let public paths through
        if _is_public_path(request.path):
            return None

        # Session check (browser) — made with the password that is set now
        if session.get("authenticated") and session.get("pw") == _password_mark():
            return None

        # Token check (API), behind the guessing brake
        token = request.headers.get("X-API-Token", "")
        if token:
            client = _client()
            paused = _too_many(client)
            if paused:
                return paused
            if _check_token(token):
                note_success(client)
                return None
            note_failure(client)
            logger.warning("Wrong X-API-Token from %s", client)

        # Not authenticated
        if request.path.startswith("/api/"):
            return jsonify({"error": "Not authenticated"}), 401
        return redirect("/auth/login")

    @app.route("/auth/login", methods=["GET", "POST"])
    def _login():
        if request.method == "GET":
            return render_template_string(_LOGIN_HTML, error=None)

        client = _client()
        left = wait_left(client)
        if left > 0:
            secs = int(left) + 1
            logger.warning("Login attempts from %s paused for %d s", client, secs)
            return (
                render_template_string(
                    _LOGIN_HTML, error=f"Too many failed attempts — try again in {secs} s"
                ),
                429,
                {"Retry-After": str(secs)},
            )
        password = request.form.get("password", "")
        if _check_password(password):
            note_success(client)
            session["authenticated"] = True
            session["pw"] = _password_mark()
            session.permanent = True
            logger.info("Login successful from %s", client)
            return redirect("/")

        note_failure(client)
        logger.warning("Login failed from %s", client)
        return render_template_string(_LOGIN_HTML, error="Wrong password"), 401

    @app.route("/auth/logout", methods=["POST"])
    def _logout():
        session.clear()
        return redirect("/auth/login")

    for warning in public_path_warnings():
        logger.warning(warning)
    logger.info("Auth middleware activated")
