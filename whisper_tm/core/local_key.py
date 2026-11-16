"""The key the bundled llama-servers (chat, embeddings, reranking) require.

They listen on loopback ports any local program can reach; without a key,
every program on the machine — or on the network, with ``host: 0.0.0.0``
— could use the model and read what it answers. The app makes a random
key at the first start (``llama_api_key`` in the config folder, readable
only by its owner), starts each server with ``--api-key-file`` and sends
the key to those three addresses. Another program that should talk to a
server reads the same file; replace the file's line to change the key.
``llama.require_key: false`` starts the servers without one.
"""

from __future__ import annotations

import logging
import secrets
import threading
from pathlib import Path
from urllib.parse import urlsplit

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_LOOPBACK = {"127.0.0.1", "localhost", "::1", "0.0.0.0", "::"}


def enabled() -> bool:
    from whisper_tm.config import config

    return bool(config.get("llama", "require_key", default=True))


def key_file() -> Path:
    """Next to config.yaml, like secret_key and mcp_token."""
    from platformdirs import user_config_dir

    from whisper_tm.config import APP_NAME, config

    base = (
        Path(config.config_path).parent if config.config_path else Path(user_config_dir(APP_NAME))
    )
    return base / "llama_api_key"


def key() -> str:
    """The key, made on first use."""
    f = key_file()
    with _lock:
        try:
            for line in f.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    return line
        except FileNotFoundError:
            pass
        new = secrets.token_urlsafe(32)
        from whisper_tm.config import _write_private

        f.parent.mkdir(parents=True, exist_ok=True)
        _write_private(f, new + "\n")
        logger.info("Key for the local llama-servers written: %s", f)
        return new


def launch_args(bin_: Path | None) -> list[str]:
    """``--api-key-file`` for a server start, if it is on and the build knows it."""
    if not enabled():
        return []
    from whisper_tm.core import llama_server

    if not llama_server._known("--api-key-file", bin_):
        logger.warning(
            "This llama-server is older than --api-key-file: it starts "
            "without a key (./setup.sh --llama fetches a current one)"
        )
        return []
    key()
    return ["--api-key-file", str(key_file())]


def _bundled() -> set[tuple[str, int]]:
    from whisper_tm.core import embed_server, llama_server, rerank_server

    out = set()
    for ep in (llama_server.endpoint(), embed_server.endpoint(), rerank_server.endpoint()):
        try:
            u = urlsplit(ep)
            if u.port:
                out.add((_norm(u.hostname or ""), u.port))
        except Exception:  # noqa: BLE001 — a broken setting names no server
            continue
    return out


def _norm(host: str) -> str:
    return "loopback" if host.lower() in _LOOPBACK else host.lower()


def headers_for(url: str, headers: dict | None = None) -> dict:
    """``headers`` plus the key when ``url`` is one of the bundled servers and
    no Authorization is set already (a configured provider key wins)."""
    out = dict(headers or {})
    if not enabled() or any(k.lower() == "authorization" for k in out):
        return out
    try:
        u = urlsplit(url)
        if u.port and (_norm(u.hostname or ""), u.port) in _bundled():
            out["Authorization"] = f"Bearer {key()}"
    except Exception as e:  # noqa: BLE001 — without the key the server says 401
        logger.debug("No local key for %s: %s", url, e)
    return out
