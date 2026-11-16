"""
Security utilities: path validation, input sanitization.

Protects against:
- Path traversal (../../etc/passwd)
- Symlink escapes
- Overly long inputs
- Invalid filenames
- Roots outside allowed trees (see :func:`root_allowed`)

On the last point: ``safe_path`` checks a path against a **base**. That's only
as good as the base itself. In this application the base is runtime-switchable
(``config.media_dir``) and is additionally taken from the request at several
endpoints (``base_dir``). Without root checking, the base could be set to ``/``
— and then every ``safe_path`` check in the house was a check against a prison
chosen by the caller themselves.
"""

from __future__ import annotations

import logging
from pathlib import Path

from flask import abort

logger = logging.getLogger(__name__)

#: Trees the owner chose as library in this run (:func:`remember_root`).
#: After a restart the saved ``directories.media`` stands for them.
_owner_roots: list[Path] = []


def owner_request() -> bool:
    """Does this request come from the person the server belongs to — from
    this machine itself, or through the login?

    Such a request may choose the library folder anywhere: it is the first
    thing somebody with a NAS does, and without it the app refused every
    folder outside ``~/Videos`` until config.yaml was edited by hand. The
    same rule guards the settings only this machine may change
    (api/settings.py). A request from another machine without the login
    stays inside :func:`allowed_roots`.
    """
    from flask import has_request_context, request

    if not has_request_context():
        return False
    from whisper_tm.core.auth import is_auth_enabled

    return is_auth_enabled() or request_is_local(request.remote_addr, request.headers)


def remember_root(path: str | Path) -> None:
    """Count a tree the owner chose as allowed for the rest of this run —
    otherwise opening one of its subfolders would shut its other folders out."""
    try:
        target = Path(path).expanduser().resolve()
    except (OSError, ValueError):
        return
    if target not in _owner_roots and len(_owner_roots) < 50:
        _owner_roots.append(target)


def allowed_roots() -> list[Path]:
    """The directory trees the application is allowed to touch.

    Source is ``security.allowed_roots``. If the list is empty, the media
    directory configured at startup applies — not the current one: otherwise a
    once-set ``media_dir`` would legitimize itself and the check would go nowhere.
    """
    from whisper_tm.config import config

    roots: list[Path] = []
    for entry in config.get("security", "allowed_roots", default=[]) or []:
        text = str(entry).strip()
        if not text:
            continue
        try:
            roots.append(Path(text).expanduser().resolve())
        except (OSError, ValueError):
            logger.warning("security.allowed_roots: unusable path: %s", text)
    if not roots:
        # No explicit entry: the media directory as it stood on load. Deliberately
        # the snapshot, NOT the current value — otherwise every directory change
        # would shift the permission list with it and the check would go nowhere.
        start = getattr(config, "media_on_start", None)
        if start:
            roots.append(Path(start).resolve())

    for chosen in _owner_roots:
        if chosen not in roots:
            roots.append(chosen)

    # The CURRENT media directory always counts. In the running application, this
    # is not an expansion: it can only be switched via HTTP, and that checks
    # against this list first — the current value already lies in an allowed tree.
    # For the programmatic path (startup, tools, tests) it's necessary though,
    # else the application would lock itself out without security.allowed_roots.
    try:
        current = Path(config.media_dir).resolve()
        if current not in roots:
            roots.append(current)
    except Exception:  # noqa — check must not prevent startup
        pass
    return roots


def root_allowed(path: str | Path) -> bool:
    """Is ``path`` in an allowed tree (or is it one itself)?"""
    if not path:
        return False
    try:
        target = Path(path).expanduser().resolve()
    except (OSError, ValueError):
        return False
    roots = allowed_roots()
    if not roots:
        # No root determinable (no media_dir configured): nothing to allow then.
        # Better to allow nothing than everything.
        return False
    for w in roots:
        if target == w:
            return True
        try:
            target.relative_to(w)
            return True
        except ValueError:
            continue
    return False


def check_root(path: str | Path, *, field_name: str = "Path") -> Path:
    """Like :func:`root_allowed`, but aborts request with 403.

    For endpoints that receive a root from the request.
    """
    if not root_allowed(path):
        logger.warning("Root rejected (%s): %s", field_name, path)
        abort(403, f"{field_name} is outside allowed directories (security.allowed_roots)")
    return Path(path).expanduser().resolve()


def safe_path(base_dir: Path, user_path: str) -> Path:
    """Safe path join: prevents path traversal.

    Ensures that the resolved path lies within base_dir.
    Returns the resolved path or calls abort(403).

    Args:
        base_dir: allowed base directory
        user_path: relative path supplied by the user
    Returns:
        Resolved path within base_dir
    """
    if not user_path:
        abort(400, "No path")

    # Normalize and remove null bytes
    clean = user_path.replace("\x00", "").strip()
    if not clean:
        abort(400, "Invalid path")

    try:
        resolved = (base_dir / clean).resolve()
        base_resolved = base_dir.resolve()

        # Check whether resolved lies within base.
        # Do NOT use str.startswith() — that would also match
        # "/media/data-evil" against base="/media/data".
        # relative_to() raises ValueError if the path does not
        # lie within base.
        try:
            resolved.relative_to(base_resolved)
        except ValueError:
            abort(403, "Access denied")

        return resolved
    except (OSError, ValueError):
        abort(400, "Invalid path")


def safe_path_or_none(base_dir: Path, user_path: str) -> Path | None:
    """Like safe_path, but returns None instead of aborting."""
    if not user_path:
        return None
    clean = user_path.replace("\x00", "").strip()
    if not clean:
        return None
    try:
        resolved = (base_dir / clean).resolve()
        base_resolved = base_dir.resolve()
        try:
            resolved.relative_to(base_resolved)
        except ValueError:
            return None
        return resolved
    except (OSError, ValueError):
        return None


# ── Cross-site requests ───────────────────────────────────────────────────
#
# Without a login (auth.enabled off, the default) every endpoint answers
# anyone who reaches the port — and a browser on the LAN reaches it from ANY
# web page it shows: a POST without a body ("simple request", no CORS
# preflight) to /api/db/delete from a foreign page deleted the database.
# So state-changing requests from a browser must come from the app itself
# (same host and port) or from the browser extension. Requests without an
# Origin header — curl, scripts, the MCP server — are not browser requests
# from a foreign page and pass.

#: Methods that change nothing; never checked.
SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS", "TRACE"})

#: Browser-extension origins: the side panel, popup and service worker of
#: the user's extension call the API from there.
EXTENSION_SCHEMES = ("chrome-extension://", "moz-extension://")
#: Our extension (browser-extension/manifest.json: its "key" fixes the ID
#: an unpacked install gets). Any other extension is a foreign origin unless
#: listed in security.trusted_extensions — every installed extension used to
#: pass, and could change the server's state.
OUR_EXTENSIONS = ("chrome-extension://mloofppjoccofakkdoahcmjpppamkgcd",)


def extension_ok(origin: str | None, extra: list[str] | tuple[str, ...] = ()) -> bool:
    """Whether ``origin`` is our browser extension, or one listed in
    ``extra`` (security.trusted_extensions: a store build has another ID)."""
    o = (origin or "").strip().rstrip("/").lower()
    if not o.startswith(EXTENSION_SCHEMES):
        return False
    allowed = {x.strip().rstrip("/").lower() for x in (*OUR_EXTENSIONS, *(extra or ()))}
    allowed |= {
        "chrome-extension://" + x.strip().lower() for x in (extra or ()) if x and "://" not in x
    }  # a bare ID
    return o in allowed


def _origin_key(origin: str) -> tuple[str, str, int | None] | None:
    """("http", "host", port) of an origin URL — None if unreadable."""
    from urllib.parse import urlsplit

    try:
        parts = urlsplit(origin.strip().rstrip("/"))
        host = (parts.hostname or "").lower()
        if not parts.scheme or not host:
            return None
        return parts.scheme.lower(), host, parts.port or _default_port(parts.scheme)
    except ValueError:
        return None


def _default_port(scheme: str) -> int | None:
    return {"http": 80, "https": 443}.get(scheme)


#: The machine's own names, looked up once (getfqdn may ask DNS).
_own_names_cache: dict[str, frozenset[str]] = {}


def _own_names() -> frozenset[str]:
    """Names this machine answers to: localhost, its host name, FQDN, .local."""
    hit = _own_names_cache.get("names")
    if hit is not None:
        return hit
    import socket

    names = {"localhost"}
    try:
        short = socket.gethostname().lower()
        fqdn = socket.getfqdn().lower()
        for n in (short, fqdn, short.split(".")[0]):
            if n:
                names.add(n)
                names.add(n.split(".")[0] + ".local")
    except OSError:
        pass
    _own_names_cache["names"] = frozenset(names)
    return _own_names_cache["names"]


def host_is_own(host: str, trusted_hosts: frozenset[str] = frozenset()) -> bool:
    """Is ``host`` (from the Host header, without port) this machine?"""
    h = (host or "").strip().lower().strip("[]")
    return bool(h) and (h in _own_names() or h in _own_addresses() or h in trusted_hosts)


def _is_ip_address(host: str) -> bool:
    import ipaddress

    try:
        ipaddress.ip_address(host)
    except ValueError:
        return False
    return True


def host_is_accepted(host: str, trusted_hosts: frozenset[str] = frozenset()) -> bool:
    """May a request sent to ``host`` (Host header, without port) be answered?

    This machine's names and addresses, a trusted host — and any IP address:
    DNS rebinding needs a name whose answer the attacker controls, and a page
    sends same-origin requests only to the address it was loaded from. So a
    server in a container, reached through the host's LAN address it can't
    see, needs no security.trusted_origins entry; a name does.
    """
    h = (host or "").strip().lower().strip("[]")
    return host_is_own(h, trusted_hosts) or _is_ip_address(h)


def origin_problem(
    method: str,
    origin: str | None,
    host: str | None,
    fetch_site: str | None = None,
    trusted: list[str] | tuple[str, ...] = (),
    extensions: list[str] | tuple[str, ...] = (),
) -> str:
    """Why a request must be refused as cross-site ("" = it may pass).

    ``origin``: the Origin header, ``host``: the Host header the request was
    sent to, ``fetch_site``: Sec-Fetch-Site, ``trusted``: origins accepted
    whatever the Host says (security.trusted_origins — a reverse proxy that
    rewrites Host).

    An Origin equal to the Host is not enough on its own: with DNS
    rebinding a foreign page's name points at this machine, and its
    requests carry Origin == Host == "evil.example:9860". So the Host must
    also be one of this machine's names, a trusted host or an IP address
    (:func:`host_is_accepted`).
    """
    if (method or "").upper() in SAFE_METHODS:
        return ""
    origin = (origin or "").strip()
    site = (fetch_site or "").strip().lower()
    if origin.lower().startswith(EXTENSION_SCHEMES):
        return "" if extension_ok(origin, extensions) else f"the extension {origin} is not ours"
    if not origin:
        if site == "cross-site":
            return "cross-site request without an origin"
        return ""
    if origin == "null":
        return "requests from an opaque origin are not accepted"
    key = _origin_key(origin)
    if key is None:
        return f"unreadable origin {origin!r}"
    trusted_keys = {k for k in (_origin_key(t) for t in trusted or ()) if k}
    if key in trusted_keys:
        return ""
    from urllib.parse import urlsplit

    try:
        req = urlsplit("//" + (host or ""))
        req_host = (req.hostname or "").lower()
        req_port = req.port or _default_port(key[0])
    except ValueError:
        return f"unreadable host {host!r}"
    if not (key[1] == req_host and key[2] == req_port):
        return f"cross-site request from {origin} refused"
    if not host_is_accepted(req_host, frozenset(k[1] for k in trusted_keys)):
        return (
            f"the server was reached as {req_host!r}, a name it doesn't know "
            f"as its own (list it in security.trusted_origins)"
        )
    return ""


#: GET requests that start work: the streams (an EventSource can only GET)
#: and the big exports. A foreign page can fire them blind — an <img> needs
#: no CORS, the answer is never read — and /api/rag/build?rebuild=1 empties
#: the semantic index and starts hours of GPU work. A browser marks such a
#: request with Sec-Fetch-Site; a client that isn't a browser sends none.
GETS_WITH_EFFECTS = frozenset(
    {
        "/api/rag/build",
        "/api/index",
        "/api/scan",
        "/api/scan_stream",
        "/api/db/export",
        "/api/rag/export",
        "/api/export/transcripts",
        "/api/settings/export",
        # the searches start the Images & sounds service and the reranker on the GPU
        "/api/media-search",
        "/api/rag/search",
    }
)

#: GETs that cost ffmpeg (or model) time, for the app's own pages only —
#: a foreign page's <img> or <audio> tags could keep the machine busy.
#: Not /api/clip/audio: its links are made for the chats of other sites
#: (the MCP tools' players).
GETS_WITH_EFFECTS_UNDER = (
    "/api/scan_entry",
    "/api/clip/waveform",
    "/api/find/frame",
    "/api/find/clip",
    "/api/media-search/thumb/",
    "/api/media-search/clip/",
    "/api/speaker/sample/",
    "/api/voices/extract/clip",
)


def get_effect_problem(
    method: str,
    path: str,
    fetch_site: str | None,
    origin: str | None = None,
    extensions: list[str] | tuple[str, ...] = (),
) -> str:
    """Why a GET that starts work must be refused ("" = it may pass): it
    came from another site's page. The app's own page (same-origin), a
    typed URL (none), the browser extension and every client that isn't a
    browser (no Sec-Fetch-Site) pass."""
    if (method or "").upper() not in ("GET", "HEAD"):
        return ""
    path = path or ""
    if path not in GETS_WITH_EFFECTS and not path.startswith(GETS_WITH_EFFECTS_UNDER):
        return ""
    if extension_ok(origin, extensions):
        return ""
    if (fetch_site or "").strip().lower() in ("cross-site", "same-site"):
        return f"{path} starts work and answers only the app's own page"
    return ""


def host_problem(
    path: str,
    origin: str | None,
    host: str | None,
    trusted: list[str] | tuple[str, ...] = (),
    extensions: list[str] | tuple[str, ...] = (),
) -> str:
    """Why an API request must be refused for the name it was sent to
    ("" = it may pass) — whatever its method.

    DNS rebinding: a foreign page whose name points at
    this machine reads every GET of the API as its own — no Origin is sent
    for it, so only the Host tells. An API request must reach one of this
    machine's names, a host of ``trusted`` or an IP address (the same rule
    as :func:`origin_problem`). The browser extension (an origin a page
    can't send) and a trusted origin pass whatever the Host says.
    """
    if not (path or "").startswith("/api/"):
        return ""
    origin = (origin or "").strip()
    if extension_ok(origin, extensions):
        return ""
    trusted_keys = {k for k in (_origin_key(t) for t in trusted or ()) if k}
    key = _origin_key(origin) if origin else None
    if key is not None and key in trusted_keys:
        return ""
    from urllib.parse import urlsplit

    try:
        req_host = (urlsplit("//" + (host or "")).hostname or "").lower()
    except ValueError:
        return f"unreadable host {host!r}"
    if host_is_accepted(req_host, frozenset(k[1] for k in trusted_keys)):
        return ""
    return (
        f"the server was reached as {req_host!r}, a name it doesn't know "
        f"as its own (list it in security.trusted_origins)"
    )


def register_origin_check(app) -> None:
    """Refuse state-changing requests from foreign web pages, and API
    requests sent to a foreign name (403, JSON)."""
    from flask import jsonify, request

    from whisper_tm.config import config

    @app.before_request
    def _check_origin():
        trusted = config.get("security", "trusted_origins", default=[]) or []
        if isinstance(trusted, str):
            trusted = [trusted]
        extensions = config.get("security", "trusted_extensions", default=[]) or []
        if isinstance(extensions, str):
            extensions = [extensions]
        extensions = [str(x) for x in extensions]
        why = origin_problem(
            request.method,
            request.headers.get("Origin"),
            request.host,
            request.headers.get("Sec-Fetch-Site"),
            trusted=[str(t) for t in trusted],
            extensions=extensions,
        )
        why = why or host_problem(
            request.path,
            request.headers.get("Origin"),
            request.host,
            trusted=[str(t) for t in trusted],
            extensions=extensions,
        )
        why = why or get_effect_problem(
            request.method,
            request.path,
            request.headers.get("Sec-Fetch-Site"),
            request.headers.get("Origin"),
            extensions=extensions,
        )
        if why:
            logger.warning(
                "Refused %s %s: %s (from %s)",
                request.method,
                request.path,
                why,
                request.remote_addr,
            )
            return jsonify({"error": f"Refused: {why}. Use the app's own page."}), 403
        return None


_addresses_cache: dict[str, tuple[float, set[str]]] = {}


def _own_addresses() -> set[str]:
    """This machine's IP addresses (loopback included) — cached for a minute,
    the origin check asks on every state-changing request."""
    import time

    hit = _addresses_cache.get("a")
    if hit and time.monotonic() - hit[0] < 60:
        return hit[1]
    out = _read_addresses()
    _addresses_cache["a"] = (time.monotonic(), out)
    return out


def _read_addresses() -> set[str]:
    out = {"127.0.0.1", "::1"}
    try:
        import socket

        import psutil

        for addrs in psutil.net_if_addrs().values():
            for a in addrs:
                # IP addresses only — not the interfaces' MAC addresses
                if a.address and a.family in (socket.AF_INET, socket.AF_INET6):
                    out.add(a.address.split("%", 1)[0].lower())
    except Exception:  # noqa — loopback alone then
        pass
    return out


#: Headers a reverse proxy adds: behind one on this machine every request
#: comes from 127.0.0.1, whoever sent it
_PROXY_HEADERS = ("X-Forwarded-For", "Forwarded", "X-Real-IP", "X-Forwarded-Host")


def request_is_local(remote_addr: str | None, headers=None) -> bool:
    """Does the request come from this machine (loopback, or one of its own
    network addresses — a browser on this machine opening the LAN address)?

    A request a proxy forwarded is never local: the address is the proxy's,
    and anyone who reaches the proxy would count as this machine — read the
    MCP token, switch its protection off. (Forging such a header only makes
    a request less trusted.)"""
    if headers is not None and any(headers.get(h) for h in _PROXY_HEADERS):
        return False
    addr = (remote_addr or "").strip()
    if addr.startswith("::ffff:"):
        addr = addr[7:]
    return bool(addr) and addr in _own_addresses()
