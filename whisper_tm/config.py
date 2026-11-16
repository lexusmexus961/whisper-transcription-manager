"""
Configuration manager: YAML-based, cross-platform, hot-reloadable.

Search paths for config.yaml (in priority order):
  1. Explicit path via --config CLI argument
  2. WHISPER_TM_CONFIG environment variable
  3. config.yaml in the user config folder (platformdirs) — persistent:
     ~/.config/whisper-tm/ (Linux), ~/Library/Application Support/
     whisper-tm/ (macOS), %LOCALAPPDATA%/whisper-tm/whisper-tm/ (Windows)
  4. ./config.yaml (working directory, fallback only)
  5. config.default.yaml (embedded defaults)

The settings page saves into the file that was loaded (only the values that
differ from config.default.yaml); with none loaded, into the user config
folder's config.yaml, so replacing the app folder loses nothing.
"""

from __future__ import annotations

import contextlib
import copy
import errno
import json
import logging
import os
import platform
import re
import shutil
import sys
import threading
from pathlib import Path
from typing import Any

import yaml
from platformdirs import user_cache_dir, user_config_dir, user_data_dir

logger = logging.getLogger(__name__)


# ── Constants ────────────────────────────────────────────────────────────

APP_NAME = "whisper-tm"
IS_WINDOWS = platform.system() == "Windows"
IS_MACOS = platform.system() == "Darwin"

VIDEO_EXTENSIONS = {
    ".mp4",
    ".mkv",
    ".avi",
    ".mov",
    ".webm",
    ".ts",
    ".mts",
    ".m2ts",
    ".mpg",
    ".mpeg",
    ".wmv",
    ".flv",
    ".m4v",
    ".ogv",
    ".3gp",
}
AUDIO_EXTENSIONS = {
    ".mp3",
    ".wav",
    ".flac",
    ".ogg",
    ".m4a",
    ".wma",
    ".aac",
    ".opus",
    ".3gp",
    ".3gpp",
    ".amr",
    ".awb",
    ".spx",
    ".caf",
    ".au",
    ".mka",
    ".ra",
    ".gsm",
    ".oga",
    ".snd",
    ".voc",
    ".webm",
}
MEDIA_EXTENSIONS = VIDEO_EXTENSIONS | AUDIO_EXTENSIONS

VALID_WHISPER_MODELS = {
    "tiny",
    "tiny.en",
    "base",
    "base.en",
    "small",
    "small.en",
    "medium",
    "medium.en",
    "large-v1",
    "large-v2",
    "large-v3",
    "turbo",
    "large-v3-turbo",
    "distil-large-v2",
    "distil-large-v3",
    "distil-small.en",
    "distil-medium.en",
}

# ── Environment Variable Expansion ────────────────────────────────────────

_ENV_RE = re.compile(r"\$\{(\w+)\}")


def _expand_path(value: str, base_dir: Path | None = None) -> Path:
    """Expand path: ~, ${VAR}, relative paths."""
    if not value:
        return Path("")
    return _expand_lexical(value, base_dir).resolve()


def _expand_lexical(value: str, base_dir: Path | None = None) -> Path:
    """Like :func:`_expand_path`, without asking the file system (no resolve)."""
    if not value:
        return Path("")
    # Replace environment variables. Do NOT pass undefined variables literally
    # — otherwise a literal "${VAR}" ends up in the path and mkdir creates
    # a correspondingly named folder.
    missing: list[str] = []

    def _sub(m):
        val = os.environ.get(m.group(1))
        if val is None:
            missing.append(m.group(1))
            return m.group(0)
        return val

    value = _ENV_RE.sub(_sub, value)
    if missing:
        raise ValueError(
            "Undefined environment variable(s) in path: " + ", ".join(f"${{{n}}}" for n in missing)
        )
    # Home expansion
    value = os.path.expanduser(value)
    p = Path(value)
    # Resolve relative paths relative to config directory
    if not p.is_absolute() and base_dir:
        p = base_dir / p
    return p


# ── Deep Merge ────────────────────────────────────────────────────────────

# ── Secrets ───────────────────────────────────────────────────────────────
#
# GET /api/settings and /api/settings/export deliver the configuration to the
# browser — both without authentication when auth.enabled is off. Without
# masking, API keys and HuggingFace tokens were visible
# in plain text there.
#
# The placeholder is sent back: the UI puts the value in an <input> and sends
# it when saving. A simple "***" would overwrite the real secret. Instead, a
# unique value that the write side recognizes and skips (see api/settings.py).
SECRET_PLACEHOLDER = "__UNCHANGED__"


def is_placeholder(value: Any) -> bool:
    """Is ``value`` the masking placeholder?"""
    return isinstance(value, str) and value == SECRET_PLACEHOLDER


# Key suffixes that hint at a secret. Deliberately by name pattern rather than
# a fixed list: a new option named "…_token" is thus protected from the start.
_SECRET_PATTERNS = ("api_key", "apikey", "token", "password", "passwort", "secret", "credentials")


def is_secret(key: str) -> bool:
    """Does the key name hint at a secret?"""
    k = str(key).lower()
    return any(m in k for m in _SECRET_PATTERNS)


#: Named lists the user writes (templates, model shortcuts, vocabularies):
#: their entry NAMES are free text. A template called "Token counts" is not
#: a secret — masked, the page would save the placeholder over its prompt.
_USER_LISTS = ("presets", "models", "vocab_presets")


#: Credentials inside a value whose key isn't a secret's: a password in a
#: URL (http://user:pass@host) and the value of an Authorization or Cookie
#: header in command-line options
_EMBEDDED = (
    re.compile(r"(\b[a-z][a-z0-9+.-]*://[^/\s:@]+:)([^@\s/]+)(@)", re.I),
    re.compile(r"((?:Authorization|Proxy-Authorization|Cookie)\s*:\s*)([^'\"\n]+)", re.I),
)


def mask_embedded(value: str) -> str:
    """``value`` with credentials inside it replaced by the placeholder."""

    def hide(m: re.Match) -> str:
        return m.group(1) + SECRET_PLACEHOLDER + (m.group(3) if m.lastindex >= 3 else "")

    for rx in _EMBEDDED:
        value = rx.sub(hide, value)
    return value


def unmask_embedded(new: str, old: Any) -> str | None:
    """What to save for ``new`` that carries the placeholder of an embedded
    credential: the stored value when nothing else changed, else None (the
    credentials have to be typed again)."""
    if isinstance(old, str) and mask_embedded(old) == new:
        return old
    return None


def _mask_secrets(nodes: Any) -> None:
    """Replace all secret-looking values in the tree with the placeholder.

    Empty values remain empty — otherwise an unset option would look like a
    set one, and the UI would report "configured" where nothing is set.
    """
    if isinstance(nodes, dict):
        for k, v in nodes.items():
            if k in _USER_LISTS and isinstance(v, dict):
                continue
            if isinstance(v, (dict, list)):
                _mask_secrets(v)
            elif is_secret(k) and isinstance(v, str) and v.strip():
                nodes[k] = SECRET_PLACEHOLDER
            elif isinstance(v, str) and ("@" in v or ":" in v):
                nodes[k] = mask_embedded(v)
    elif isinstance(nodes, list):
        for v in nodes:
            _mask_secrets(v)


#: Named lists the user edits as a whole: the user's version replaces the
#: shipped one instead of being merged into it. Merged, every built-in
#: template or vocabulary the user had deleted came back at the next start.
#: New built-in templates reach an existing config through Settings →
#: Summaries → "⬇ Built-in templates" (core/llm/prompts.py).
_WHOLE_LISTS = {("llm", "presets"), ("llm", "models"), ("transcription", "vocab_presets")}


class ConfigError(Exception):
    """The user's config.yaml can't be used as it is — the message says
    where and how to go on (printed instead of a traceback at start)."""


def _read_user_config(path: Path) -> dict:
    """The settings of the user's file. A stray tab or quote used to end in
    a page of Python traceback; a file that holds a list instead of
    ``key: value`` lines crashed further down."""
    try:
        with open(path, encoding="utf-8") as f:
            found = yaml.safe_load(f)
    except yaml.YAMLError as e:
        mark = getattr(e, "problem_mark", None)
        where = f", line {mark.line + 1}, column {mark.column + 1}" if mark else ""
        what = getattr(e, "problem", None) or str(e).splitlines()[0]
        raise ConfigError(
            f"The settings file {path} can't be read{where}: {what}.\n"
            f'Correct that line (YAML: "key: value", indented with spaces), or move the '
            f"file away to start with the defaults."
        ) from None
    except UnicodeDecodeError:
        raise ConfigError(
            f"The settings file {path} is not UTF-8 text. Save it as UTF-8, or "
            f"move it away to start with the defaults."
        ) from None
    if found is None:
        return {}
    if not isinstance(found, dict):
        raise ConfigError(
            f"The settings file {path} holds a {type(found).__name__}, not settings. It needs "
            f'"section:" lines with "key: value" below them — see config.default.yaml.'
        )
    return found


def _deep_merge(base: dict, override: dict, path: tuple = ()) -> dict:
    """Deep merge of two dicts. Override wins on conflicts."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
            and path + (key,) not in _WHOLE_LISTS
        ):
            result[key] = _deep_merge(result[key], value, path + (key,))
        else:
            result[key] = copy.deepcopy(value)
    return result


#: "No value" — distinct from None, which a YAML file can hold.
_MISSING = object()


def _put_or_drop(data: dict, path: list[str], value: Any) -> None:
    """Write ``value`` at ``path`` — or remove the key for _MISSING."""
    node = data
    for part in path[:-1]:
        if not isinstance(node.get(part), dict):
            if value is _MISSING:
                return
            node[part] = {}
        node = node[part]
    if value is _MISSING:
        node.pop(path[-1], None)
    else:
        node[path[-1]] = copy.deepcopy(value)


_TRUE_WORDS = {"true", "yes", "on", "1"}
_FALSE_WORDS = {"false", "no", "off", "0", ""}


def _coerce_like(user: dict, defaults: dict) -> None:
    """Give the user's values the type of their default, where that is sure.

    Values used to reach config.yaml as strings: a <select> in the settings
    page hands out "true"/"false", and ``bool("false")`` is True — the
    embedding mode could never be switched back. A number in quotes breaks
    comparisons. Only the unambiguous cases are converted; anything else
    stays as it is.
    """
    for key, value in list(user.items()):
        if key not in defaults:
            continue
        default = defaults[key]
        if isinstance(default, dict):
            if isinstance(value, dict):
                _coerce_like(value, default)
            continue
        user[key] = coerce_value(value, default)


def coerce_value(value: Any, default: Any) -> Any:
    """``value`` in the type of ``default`` — when that is unambiguous."""
    if isinstance(default, bool):
        if isinstance(value, str) and value.strip().lower() in _TRUE_WORDS | _FALSE_WORDS:
            return value.strip().lower() in _TRUE_WORDS
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value in (0, 1):
            return bool(value)
        return value
    if isinstance(default, (int, float)) and not isinstance(value, bool):
        if isinstance(value, str):
            try:
                number = float(value.strip())
            except ValueError:
                return value
            return int(number) if isinstance(default, int) and number.is_integer() else number
        return value
    if isinstance(default, list) and isinstance(value, str):
        # One per line where there are lines, else a comma list
        parts = value.splitlines() if "\n" in value else value.split(",")
        return [part.strip() for part in parts if part.strip()]
    return value


def _extensions(values: Any) -> set[str]:
    """Extra extensions as the scanner compares them: lowercase, with the
    dot — "RMVB" and ".rmvb" both mean ".rmvb" (file names are lowercased
    before the lookup, so ".RMVB" never matched anything)."""
    if isinstance(values, str):
        values = values.split(",")
    out = set()
    for v in values or []:
        v = str(v).strip().lower()
        if v:
            out.add(v if v.startswith(".") else "." + v)
    return out


#: Folders no app directory may be — they hold the operating system. A
#: directory key set to one of them (or, for the trees, inside one) is
#: refused: the logs folder is read and deleted from over HTTP, the work
#: folder is emptied recursively.
_SYSTEM_TREES = (
    "/bin",
    "/boot",
    "/dev",
    "/etc",
    "/lib",
    "/lib32",
    "/lib64",
    "/libx32",
    "/proc",
    "/sbin",
    "/sys",
    "/usr",
)
_SYSTEM_DIRS = (
    "/",
    "/home",
    "/mnt",
    "/media",
    "/opt",
    "/root",
    "/run",
    "/srv",
    "/tmp",
    "/var",
    "/Users",
    "/Volumes",
)

#: How long a directory check waits for a path to resolve (a hung NAS
#: mount never answers). Then the path is taken as written, normalized.
RESOLVE_TIMEOUT_S = 5.0
_resolver: threading.Thread | None = None
_resolver_lock = threading.Lock()


def _resolve_all(paths: list[Path]) -> list[Path]:
    """``Path.resolve()`` for each path — bounded in time and never under a lock.

    resolve() asks the file system, and on a hung CIFS mount it waits for
    good. It runs in a worker thread; after RESOLVE_TIMEOUT_S the paths are
    taken lexically (absolute, normalized, links not followed). While an
    earlier worker still hangs, no new one starts — lexical right away.
    """
    global _resolver
    lexical = [Path(os.path.normpath(os.path.abspath(str(p)))) for p in paths]
    box: dict = {}
    with _resolver_lock:
        if _resolver is not None and _resolver.is_alive():
            return lexical

        def run():
            try:
                box["out"] = [Path(p).resolve() for p in paths]
            except (OSError, RuntimeError, ValueError):
                box["out"] = None

        _resolver = threading.Thread(target=run, name="path-resolve", daemon=True)
        worker = _resolver
        worker.start()
    worker.join(RESOLVE_TIMEOUT_S)
    out = box.get("out")
    if out is None:
        if worker.is_alive():
            logger.warning(
                "Path check: no answer from the file system within "
                "%.0f s — checking the paths as written",
                RESOLVE_TIMEOUT_S,
            )
        return lexical
    return out


#: Directory keys the settings page may set, besides ``directories.media``
#: (which goes through the allowed-roots check).
CHECKED_DIRECTORIES = ("work", "uploads", "clips", "logs", "transcripts")


def _write_private(target: Path, text: str) -> None:
    """Replace ``target`` with ``text`` — atomically, readable only by its owner.

    The configuration holds API keys, tokens and the login password. It used
    to be truncated and rewritten in place with the default mode 0644: a
    crash or a full disk in the middle left half a file (whose remainder the
    next start loaded without complaint, losing every setting after the cut),
    and every local user could read the secrets. Now the text goes into a
    0600 temporary file next to it, reaches the disk, and replaces the old
    file in one rename. A symlinked config is written where the link points.

    Where a rename can't replace the file — a config bind-mounted on its own
    into a container (EBUSY), a folder that isn't writable while the file is
    (EACCES/EPERM), another file system (EXDEV) — the text is written into
    the file itself instead, as before, but synced and with mode 0600.

    The mode is always 0600: owner read and write, nobody else. No other
    mode is stricter without locking the owner out of their own config.
    """
    import tempfile

    target = Path(target)
    if target.is_symlink():
        target = target.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd, tmp = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent))
    except OSError as e:
        if e.errno not in _IN_PLACE_ERRNOS:
            raise
        _write_in_place(target, text)
        return
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, 0o600)
        try:
            os.replace(tmp, target)
        except OSError as e:
            if e.errno not in _IN_PLACE_ERRNOS:
                raise
            os.unlink(tmp)
            logger.info("Config not replaceable by rename (%s) — written in place", e)
            _write_in_place(target, text)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


#: Rename failures that writing into the file itself gets around.
_IN_PLACE_ERRNOS = {errno.EBUSY, errno.EXDEV, errno.EACCES, errno.EPERM}


def _write_in_place(target: Path, text: str) -> None:
    """Overwrite ``target``'s content (fallback of :func:`_write_private`)."""
    data = text.encode("utf-8")
    fd = os.open(str(target), os.O_WRONLY | os.O_CREAT, 0o600)
    try:
        # a mount may not allow it; Windows' Python has it from 3.13 on
        with contextlib.suppress(OSError, AttributeError):
            os.fchmod(fd, 0o600)
        os.ftruncate(fd, 0)
        view = memoryview(data)
        while view:
            view = view[os.write(fd, view) :]
        os.fsync(fd)
    finally:
        os.close(fd)


def _without_defaults(data: Any, defaults: Any, path: tuple = ()) -> Any:
    """What of ``data`` differs from ``defaults``: sections recursed, every
    other value compared whole, keys the defaults lack kept. What a saved
    file holds — a value left at its default follows the defaults of the
    next version instead of freezing this one's.

    A named list (_WHOLE_LISTS) is one value: written whole when it differs.
    Recursed, a saved template list held only the user's own template, and
    loaded back it replaced the shipped list — every built-in template was
    gone after the next start."""
    out = {}
    for key, value in data.items():
        default = defaults.get(key, _MISSING) if isinstance(defaults, dict) else _MISSING
        if (
            isinstance(value, dict)
            and isinstance(default, dict)
            and path + (key,) not in _WHOLE_LISTS
        ):
            rest = _without_defaults(value, default, path + (key,))
            if rest:
                out[key] = rest
        elif default is _MISSING or value != default:
            out[key] = value
    return out


#: The head of a file the settings page writes
SAVED_HEADER = """# Whisper Transcription Manager — your settings: only what differs from
# config.default.yaml, which documents every setting. The settings page saves into this file.

"""


def set_in_yaml_text(text: str, section: str, key: str, value: Any) -> str | None:
    """``text`` (a config file) with ``section.key`` set to ``value``: that one
    line replaced, or added at the top of the section, or the section
    appended. Everything else — the comments of a file copied from
    config.default.yaml above all — stays as it is.

    None when the result would not read back as exactly that change (a
    flow-style section, a key nested deeper, YAML this can't follow): the
    caller then writes the file the plain way."""
    rendered = json.dumps(value, ensure_ascii=False)  # JSON scalars are YAML
    lines = text.splitlines(keepends=True)
    head = next(
        (i for i, ln in enumerate(lines) if re.match(rf"{re.escape(section)}:\s*(#.*)?$", ln)), None
    )
    if head is None:
        sep = "" if not text or text.endswith("\n") else "\n"
        new = f"{text}{sep}{section}:\n  {key}: {rendered}\n"
    else:
        end = next(
            (i for i in range(head + 1, len(lines)) if re.match(r"[^\s#]", lines[i])), len(lines)
        )
        first = next(
            (lines[i] for i in range(head + 1, end) if re.match(r"\s+[^\s#]", lines[i])), "  "
        )
        indent = re.match(r"\s*", first).group(0) or "  "
        hit = next(
            (i for i in range(head + 1, end) if lines[i].startswith(f"{indent}{key}:")), None
        )
        if hit is not None:
            nl = "\n" if lines[hit].endswith("\n") else ""
            lines[hit] = f"{indent}{key}: {rendered}{nl}"
        else:
            lines.insert(head + 1, f"{indent}{key}: {rendered}\n")
        new = "".join(lines)
    try:
        before = yaml.safe_load(text) or {}
        after = yaml.safe_load(new) or {}
    except yaml.YAMLError:
        return None
    if not isinstance(before, dict) or not isinstance(after, dict):
        return None
    part = before.get(section)
    if part is not None and not isinstance(part, dict):
        return None
    expect = dict(before)
    expect[section] = {**(part or {}), key: value}
    return new if after == expect else None


def _default_clips_dir() -> Path:
    """Default clips directory: ~/Videos/Whisper-TM-Clips — made by the first
    clip, not by the start."""
    try:
        home = Path.home()
        for name in ["Videos", "Video", "Movies"]:
            vdir = home / name
            if vdir.is_dir():
                return vdir / "Whisper-TM-Clips"
    except Exception as e:
        logger.debug("Unhandled exception: %s", e)
    return Path(user_data_dir(APP_NAME)) / "clips"


# ── Config Manager ────────────────────────────────────────────────────────


class ConfigManager:
    """Thread-safe configuration manager with hot-reload.

    Supports:
    - Cross-platform path resolution
    - Environment variables in paths: ${VIDEO_DIR}
    - Nested access: cfg["server.port"] or cfg.get("server", "port")
    - Hot-reload on file changes
    - Validation of all settings
    - Export of current configuration
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._save_lock = threading.Lock()
        self._data: dict = {}
        self._config_path: Path | None = None
        self._base_dir: Path | None = None
        self._shipped_defaults = self._load_defaults()
        self._defaults = copy.deepcopy(self._shipped_defaults)
        self._machine_defaults: dict[str, Any] = {}
        self._user_cfg: dict = {}
        self._callbacks: list = []
        # Values set only for this run (command-line flags): config key →
        # what the file had, restored by save(). See override().
        self._file_values: dict[str, Any] = {}
        self._loaded: bool = False
        # True if the loaded file was forced (--config or
        # WHISPER_TM_CONFIG) — save() writes back there instead of
        # to the home config (see save()).
        self._config_forced: bool = False
        # Media directory as it was in the file when loading. Basis for
        # root check, see load() and core/security.allowed_roots.
        self._media_on_start: Path | None = None

    @property
    def media_on_start(self) -> Path | None:
        """The media directory configured at load time (immutable).

        Deliberately separate from :attr:`media_dir`: this is the
        trust anchor for root validation, while that is the runtime
        switchable working value.
        """
        with self._lock:
            return self._media_on_start

    # ── Load ─────────────────────────────────────────────────────────

    def _load_defaults(self) -> dict:
        """Load embedded default configuration."""
        default_file = Path(__file__).parent.parent / "config.default.yaml"
        if default_file.exists():
            with open(default_file, encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {}

    def load(self, config_path: str | Path | None = None) -> None:
        """Load configuration. Searches automatically if no path given."""
        with self._lock:
            # The shipped defaults again: a machine's defaults
            # (set_machine_defaults) belong to one start, not to the next load
            self._defaults = copy.deepcopy(self._shipped_defaults)
            self._machine_defaults = {}
            self._user_cfg = {}
            self._data = copy.deepcopy(self._defaults)

            path = self._find_config(config_path)
            if path and path.exists():
                self._config_path = path
                self._base_dir = path.parent
                # Remember if the file was FORCED (--config or
                # WHISPER_TM_CONFIG): then save() must write back there —
                # changes saved to Home would not be loaded on next start
                # (the Env-Var has priority), and a test run with own config
                # would still overwrite the home config of the real installation.
                env_path = os.environ.get("WHISPER_TM_CONFIG")
                self._config_forced = bool(config_path) or bool(
                    env_path and Path(env_path).resolve() == path
                )
                user_cfg = _read_user_config(path)
                _coerce_like(user_cfg, self._defaults)
                self._user_cfg = copy.deepcopy(user_cfg)
                self._data = _deep_merge(self._data, user_cfg)
            else:
                self._base_dir = Path.cwd()
                self._config_forced = False

            # Initialize directories
            self._resolve_directories()
            self._ensure_directories()
            # Snapshot root for security check
            # (core/security.allowed_roots). It comes from the FILE and
            # is deliberately NOT tracked by the media_dir setter: the setter
            # writes _data["directories"]["media"] too, a derived value would
            # trail the media directory and the allowed list would legitimize
            # itself anew with every change.
            self._media_on_start = self._resolved_media
            self._loaded = True

    @property
    def is_loaded(self) -> bool:
        """Whether configuration has been loaded."""
        return self._loaded

    @property
    def config_path(self) -> Path | None:
        """Path of the loaded configuration file."""
        return self._config_path

    @property
    def config_forced(self) -> bool:
        """Was the loaded file forced (--config/WHISPER_TM_CONFIG)? Then
        :meth:`save` writes back to it, not to the home config."""
        return self._config_forced

    def _find_config(self, explicit: str | Path | None = None) -> Path | None:
        """Find config file in search paths.

        Priority:
          1. Explicit path (--config)
          2. WHISPER_TM_CONFIG env variable
          3. ~/.config/whisper-tm/config.yaml (persistent, survives app updates)
          4. ./config.yaml (working directory, fallback only)
        """
        candidates = []

        if explicit:
            candidates.append(Path(explicit))

        env_path = os.environ.get("WHISPER_TM_CONFIG")
        if env_path:
            candidates.append(Path(env_path))

        # Home config has priority over CWD (survives folder replacement)
        candidates.append(Path(user_config_dir(APP_NAME)) / "config.yaml")
        candidates.append(Path.cwd() / "config.yaml")

        for p in candidates:
            if p.exists():
                return p.resolve()
        return None

    # ── Directory Resolution ─────────────────────────────────────────

    def _resolve_directories(self) -> None:
        """Resolve all directory paths and set platform-appropriately."""
        # Mutation of _resolved_* under _lock so concurrent readers
        # (dirs/media_dir, also in runtime settings PUT) don't see torn reads.
        # _lock is an RLock → reentrant, load() may already hold it.
        with self._lock:
            # `directories:` can be an empty section (None) in YAML — then
            # .get() returns None instead of a dict. `or {}` catches that.
            dirs = self._data.get("directories") or {}
            data_dir = Path(user_data_dir(APP_NAME))
            cache_dir = Path(user_cache_dir(APP_NAME))

            # Media directory
            media = dirs.get("media", "")
            if media:
                self._resolved_media = _expand_path(media, self._base_dir)
            else:
                # Platform default
                if IS_WINDOWS:
                    self._resolved_media = Path.home() / "Videos"
                elif IS_MACOS:
                    self._resolved_media = Path.home() / "Movies"
                else:
                    self._resolved_media = Path(
                        os.environ.get("VIDEO_DIR", str(Path.home() / "Videos"))
                    )

            # Work directory (for temporary files → SSD!)
            work = dirs.get("work", "")
            if work:
                self._resolved_work = _expand_path(work, self._base_dir)
            else:
                self._resolved_work = cache_dir / "work"

            # Transcript mode
            mode = dirs.get("transcript_mode", "alongside")
            if mode == "alongside":
                self._resolved_transcripts = self._resolved_media / ".transcripts"
            elif mode == "central":
                tp = dirs.get("transcripts", "")
                self._resolved_transcripts = (
                    _expand_path(tp, self._base_dir) if tp else data_dir / "transcripts"
                )
            else:
                # "database" mode — still set path for fallback
                self._resolved_transcripts = data_dir / "transcripts"

            # Additional directories (only expand if configured)
            # Uploads: empty = uploads/ IN the media directory. The old
            # default (user data directory) lay outside the library —
            # uploaded/downloaded files appeared in no list and couldn't be
            # queued. With explicit value _uploads_explicit does NOT follow
            # the media_dir setter (user has decided).
            _up = dirs.get("uploads", "")
            self._uploads_explicit = bool(_up)
            self._resolved_uploads = (
                _expand_path(_up, self._base_dir) if _up else self._resolved_media / "uploads"
            )

            _cl = dirs.get("clips", "")
            self._resolved_clips = (
                _expand_path(_cl, self._base_dir) if _cl else _default_clips_dir()
            )

            _lg = dirs.get("logs", "")
            self._resolved_logs = _expand_path(_lg, self._base_dir) if _lg else data_dir / "logs"

            self._resolved_cache = cache_dir
            self._resolved_chunks = cache_dir / "chunks"

            # DB path (`database:` can be empty section → None)
            db_path = (self._data.get("database") or {}).get("path", "")
            self._resolved_db = (
                _expand_path(db_path, self._base_dir) if db_path else data_dir / "library.db"
            )

    def _ensure_directories(self) -> None:
        """Create all directories.

        Individual errors (e.g. read-only NAS mount for .transcripts) must NOT
        prevent app startup — only warn.
        """
        # Consistently collect _resolved_* under lock (no torn read vs.
        # _resolve_directories), but execute the actual mkdir WITHOUT lock
        # — mkdir on slow NAS mounts must not block config.get().
        #
        # The folders inside the library (uploads/ and .transcripts/ by
        # default) only once a library is chosen, the default clips folder
        # only with the first clip: a first start, before a library was
        # chosen, left all three in ~/Videos. Whatever writes into them
        # makes them when needed.
        with self._lock:
            dirs = self._data.get("directories") or {}
            chosen = bool(str(dirs.get("media") or "").strip())
            targets = [
                self._resolved_work,
                self._resolved_logs,
                self._resolved_chunks,
                self._resolved_db.parent,
            ]
            if chosen or getattr(self, "_uploads_explicit", False):
                targets.append(self._resolved_uploads)
            if str(dirs.get("clips") or "").strip():
                targets.append(self._resolved_clips)
            if chosen or dirs.get("transcript_mode", "alongside") != "alongside":
                targets.append(self._resolved_transcripts)
        for d in targets:
            if d and str(d):
                try:
                    d.mkdir(parents=True, exist_ok=True)
                except OSError as e:
                    logger.warning("Directory not creatable: %s (%s)", d, e)
        self._mark_workdir(warn=True)
        self._keep_private(Path(user_data_dir(APP_NAME)), self._resolved_db.parent)

    @staticmethod
    def _keep_private(*folders: Path) -> None:
        """The data folder and the database's folder readable by their owner
        only (0700): transcripts, voice vectors (biometric data), chats and
        backups were 0755 under the usual umask, readable by every account
        where the home folder can be entered. A folder of another owner, or
        one that refuses the change (a network share), is left as it is."""
        if os.name != "posix":
            return
        for folder in dict.fromkeys(folders):
            try:
                st = folder.stat()
                if st.st_uid == os.getuid() and st.st_mode & 0o077:
                    folder.chmod(st.st_mode & 0o700)
            except OSError as e:
                logger.debug("Folder not made private: %s (%s)", folder, e)

    # ── Access ───────────────────────────────────────────────────────

    def get(self, *keys: str, default: Any = None) -> Any:
        """Nested access: cfg.get("server", "port") or cfg.get("server.port")"""
        with self._lock:
            # Flatten dotted keys
            flat = []
            for k in keys:
                flat.extend(k.split("."))

            node = self._data
            for key in flat:
                if isinstance(node, dict) and key in node:
                    node = node[key]
                else:
                    return default
            return node

    def set(self, *args) -> None:
        """Set value: cfg.set("server", "port", 8080) or cfg.set("server.port", 8080)"""
        with self._lock:
            value = args[-1]
            keys = []
            for k in args[:-1]:
                keys.extend(k.split("."))
            # Set on purpose: from now on the file gets this value, not the
            # one a command-line flag covered up
            self._file_values.pop(".".join(keys), None)

            node = self._data
            for key in keys[:-1]:
                if key not in node or not isinstance(node[key], dict):
                    node[key] = {}
                node = node[key]
            node[keys[-1]] = value

    def override(self, key: str, value: Any) -> None:
        """Set a value for this run only — a command-line flag.

        ``--port 5299`` for a test instance must not end up in config.yaml
        the next time someone saves the settings page: that page saves the
        whole configuration, and the next normal start would come up on the
        test port. save() writes what the file had instead.
        """
        with self._lock:
            original = self._file_values.get(key, _MISSING)
            if key not in self._file_values:
                original = self.get(key, default=_MISSING)
            self.set(key, value)  # forgets the key …
            self._file_values[key] = original  # … this one stays run-only

    def set_machine_defaults(self, values: dict[str, Any]) -> dict[str, Any]:
        """Make ``values`` ({"section.key": value}) this machine's defaults.

        A few settings have no value that suits every machine: the model, the
        vocal isolation and the number of files at once fit a 12 GB card,
        take hours on a CPU and don't run at all on a 4 GB card. The app
        chooses them from the hardware it finds (hardware.machine_defaults),
        once, after the config is loaded.

        A value becomes the *default* here: what applies unless the user set
        the key (in config.yaml or on the settings page), what "reset" goes
        back to, and what save() does not write — so a config.yaml carried
        to another machine gets that machine's defaults. Returns the values
        that took effect.
        """
        applied: dict[str, Any] = {}
        with self._lock:
            for dotted, value in values.items():
                keys = dotted.split(".")
                node: Any = self._user_cfg
                for k in keys:
                    node = node.get(k, _MISSING) if isinstance(node, dict) else _MISSING
                    if node is _MISSING:
                        break
                user_set = node is not _MISSING
                current = self._at(self._data, keys)
                shipped = self._at(self._defaults, keys)
                _put_or_drop(self._defaults, keys, value)
                # Not set in the file, and not changed since it was loaded
                if not user_set and current == shipped:
                    _put_or_drop(self._data, keys, value)
                    applied[dotted] = copy.deepcopy(value)
            self._machine_defaults = copy.deepcopy(values)
        return applied

    @property
    def machine_defaults(self) -> dict[str, Any]:
        """The defaults chosen for this machine ({} before the app start)."""
        with self._lock:
            return copy.deepcopy(self._machine_defaults)

    @staticmethod
    def _at(data: Any, keys: list[str]) -> Any:
        for k in keys:
            if not isinstance(data, dict) or k not in data:
                return _MISSING
            data = data[k]
        return data

    def default(self, *keys: str) -> Any:
        """The default value: config.default.yaml's, or this machine's
        (:meth:`set_machine_defaults`) — None if there is none."""
        node: Any = self._defaults
        for k in ".".join(keys).split("."):
            if not isinstance(node, dict) or k not in node:
                return None
            node = node[k]
        return copy.deepcopy(node)

    def defaults_masked(self) -> dict:
        """config.default.yaml as the settings page compares against it."""
        data = copy.deepcopy(self._defaults)
        _mask_secrets(data)
        return data

    def __getitem__(self, key: str) -> Any:
        result = self.get(key)
        if result is None:
            raise KeyError(key)
        return result

    def all(self) -> dict:
        """Complete configuration as dict."""
        with self._lock:
            return copy.deepcopy(self._data)

    # ── Resolved Directories ──────────────────────────────────────

    @property
    def dirs(self) -> dict[str, Path]:
        """All resolved directories."""
        with self._lock:
            return {
                "media": self._resolved_media,
                "work": self._resolved_work,
                "transcripts": self._resolved_transcripts,
                "uploads": self._resolved_uploads,
                "clips": self._resolved_clips,
                "logs": self._resolved_logs,
                "chunks": self._resolved_chunks,
            }

    @property
    def db_path(self) -> Path:
        with self._lock:
            return self._resolved_db

    @property
    def media_dir(self) -> Path:
        with self._lock:
            return self._resolved_media

    @media_dir.setter
    def media_dir(self, path: str | Path) -> None:
        """Switch media directory (runtime).

        Deliberately WITHOUT root check: this is the programmatic way
        (startup, tests, tools). The check against
        ``security.allowed_roots`` belongs at the HTTP boundary — that's where
        input comes from outside. It would be wrong here too because the setter
        also writes ``_data["directories"]["media"]``: the allowed list would
        trail the media directory and thus legitimize itself.

        When switching the directory from a request, first call
        ``core.security.check_root()``. Affected are
        ``/api/set_directory``, ``/api/settings/directory`` and
        ``/api/navigate_to_file``.
        """
        p = Path(path).resolve()
        if not p.exists() or not p.is_dir():
            raise ValueError(f"Directory does not exist: {p}")
        with self._lock:
            self._resolved_media = p
            # IMPORTANT: also update _data — otherwise the next
            # config.save() writes the OLD path to config.yaml and after
            # a restart the app points to the old directory again.
            self._data.setdefault("directories", {})["media"] = str(p)
            mode = self.get("directories", "transcript_mode", default="alongside")
            if mode == "alongside":
                self._resolved_transcripts = p / ".transcripts"
                try:
                    self._resolved_transcripts.mkdir(parents=True, exist_ok=True)
                except OSError as e:
                    logger.warning("Cannot create .transcripts (read-only mount?): %s", e)
            # Uploads default follows media directory (see
            # _resolve_directories) — unless the user has configured a fixed
            # path.
            if not getattr(self, "_uploads_explicit", False):
                self._resolved_uploads = p / "uploads"
        # Fire callbacks ONLY after releasing the lock — a callback may
        # trigger a watcher full rescan, which would block all config.get()
        # while running under the lock.
        self._notify("directories.media", str(p))

    @property
    def transcript_dir(self) -> Path:
        with self._lock:
            return self._resolved_transcripts

    @property
    def data_dir(self) -> Path:
        """Where the app keeps what it made itself and must not lose:
        the database (by default), custom done sounds. Unlike the work
        folder it is never emptied. (The secret key and the MCP token lie
        next to the config file.)"""
        return Path(user_data_dir(APP_NAME))

    @property
    def cache_dir(self) -> Path:
        """Root of cache (work files, chunks, models).

        Public because not only config needs it: third-party libraries
        otherwise place their models where THEIR default points —
        ``audio-separator`` e.g. to ``/tmp/audio-separator-models``, and
        on many systems ``/tmp`` is a tmpfs in RAM
        (`core/vocal_isolation.py::separation_model_dir`).
        """
        with self._lock:
            return self._resolved_cache

    @property
    def work_dir(self) -> Path:
        with self._lock:
            return self._resolved_work

    @property
    def upload_dir(self) -> Path:
        """Target directory for uploads and link downloads.

        THE one place for them — callers never build the path themselves
        (directories.uploads from settings; empty = uploads/ in
        media directory, follows directory changes automatically).
        """
        with self._lock:
            return self._resolved_uploads

    # ── Persistence ─────────────────────────────────────────────────

    def save(self, path: str | Path | None = None) -> Path:
        """Save the current configuration — what differs from the defaults
        only, into the file it was loaded from (the one the next start reads
        again); without one, into ~/.config/whisper-tm/config.yaml.

        It used to write the whole merged configuration into the home
        config whenever the file wasn't forced: an installation loaded from
        ./config.yaml got a home file on its first save, which from then on
        outranked ./config.yaml (edits there stopped working, the MCP token
        next to it was made anew), and every default was frozen at this
        version's value.
        """
        if path:
            target = Path(path)
        elif self._config_path:
            target = Path(self._config_path)
        else:
            target = Path(user_config_dir(APP_NAME)) / "config.yaml"
            target.parent.mkdir(parents=True, exist_ok=True)

        # Serialized under the config lock (a consistent snapshot), written
        # after releasing it: disk I/O — a slow or hung home share — must
        # not block config.get(). The save lock keeps two saves in order.
        with self._save_lock:
            with self._lock:
                data = self._data
                if self._file_values:
                    data = copy.deepcopy(self._data)
                    for key, original in self._file_values.items():
                        _put_or_drop(data, key.split("."), original)
                data = _without_defaults(data, self._defaults)
                text = SAVED_HEADER + yaml.safe_dump(
                    data, default_flow_style=False, allow_unicode=True, sort_keys=False
                )
            _write_private(target, text)
            with self._lock:
                self._config_path = target
        return target

    def save_value(self, section: str, key: str, value: Any) -> Path:
        """Set ``section.key`` and write just that into the config file this
        instance loads next start — the one it was loaded from.

        Unlike :meth:`save` the file keeps its comments: only that line
        changes (:func:`set_in_yaml_text`). For the setup's own steps — the
        token, the LLM endpoint — written into the file it makes. Without a
        loaded file, or one that can't be edited line by line, it is
        :meth:`save`.
        """
        self.set(section, key, value)
        path = self._config_path
        if path and Path(path).is_file():
            with self._save_lock:
                new = set_in_yaml_text(Path(path).read_text(encoding="utf-8"), section, key, value)
                if new is not None:
                    _write_private(Path(path), new)
                    return Path(path)
            return self.save(path)
        return self.save()

    def export_config(self, masked: bool = True) -> str:
        """Configuration as YAML string.

        By default without secrets — the export is a download via a
        route without authentication.
        """
        with self._lock:
            data = copy.deepcopy(self._data)
        if masked:
            _mask_secrets(data)
        return yaml.dump(data, default_flow_style=False, allow_unicode=True, sort_keys=False)

    def all_masked(self) -> dict:
        """Like :meth:`all`, but with starred secrets."""
        data = copy.deepcopy(self.all())
        _mask_secrets(data)
        return data

    # ── Callbacks for changes ─────────────────────────────────────────

    def on_change(self, callback):
        """Register callback: fn(key, value)"""
        self._callbacks.append(callback)

    def _notify(self, key: str, value: Any):
        # Snapshot the callback list under the lock, but call the callbacks
        # themselves OUTSIDE the lock — a slow callback (e.g. a watcher
        # full rescan) must not block all config.get()/set().
        with self._lock:
            callbacks = list(self._callbacks)
        for cb in callbacks:
            try:
                cb(key, value)
            except Exception as e:
                logger.debug("Unhandled exception: %s", e)

    # ── Directory checks ─────────────────────────────────────────────

    def _data_trees(self) -> list[tuple[str, Path]]:
        """Folders that hold what must never be emptied or served: media,
        the allowed roots, the app's data and config folders, the home.

        Resolved through :func:`_resolve_all` — bounded, and with no lock
        held (the media folder and the roots are NAS paths)."""
        from whisper_tm.core.security import allowed_roots

        out = [
            ("the home folder", Path.home()),
            ("the media folder", Path(self.media_dir)),
            ("the data folder", self.data_dir),
            ("the config folder", Path(user_config_dir(APP_NAME))),
        ]
        config_path = self.config_path
        if config_path:
            out.append(("the config folder", Path(config_path).parent))
        resolved = _resolve_all([p for _label, p in out])
        trees = [(label, p) for (label, _raw), p in zip(out, resolved, strict=True)]
        return trees + [("an allowed root", r) for r in allowed_roots()]

    def _default_work_dir(self) -> Path:
        """Where the work folder is when directories.work is empty."""
        return self.cache_dir / "work"

    @staticmethod
    def _home_or_above(tree: Path, home: Path) -> bool:
        """Is ``tree`` the home folder or one of its ancestors ("/", "/home")?

        Such a tree (an allowed root "~" or "/", a media folder "~") holds
        the default work folder below it — being inside it says nothing."""
        return tree == home or tree in home.parents

    def check_directory(self, name: str, value: Any) -> str:
        """Why ``value`` can't be ``directories.<name>`` ("" = it can).

        The page sets these folders, and the app does a lot with them: the
        logs folder is listed, read and deleted from over HTTP, the work
        folder is emptied recursively. So none may be a system folder, the
        home folder, or a folder that contains the media, an allowed root
        or the app's own data. The work folder additionally must not lie
        INSIDE any of those (except below the home folder, where its default
        is): emptying it would delete recordings and transcripts.
        An empty value is always fine — it means the default.
        """
        text = str(value or "").strip()
        if not text:
            return ""
        try:
            p = _expand_lexical(text, self._base_dir)
        except ValueError as e:
            return str(e)
        p = _resolve_all([p])[0]
        if not p.is_absolute():
            return f"{text} is not a full path"
        s = p.as_posix()
        if (
            s in _SYSTEM_DIRS
            or p.parent == p
            or any(s == t or s.startswith(t + "/") for t in _SYSTEM_TREES)
        ):
            return f"{p} is a system folder"
        trees = self._data_trees()
        for label, other in trees:
            if (
                name == "uploads"
                and label in ("the media folder", "an allowed root")
                and p == other
            ):
                continue  # uploads straight into the library is fine
            if p == other or p in other.parents:
                return f"{p} is or contains {label} ({other})"
        if name == "work" and p != _resolve_all([self._default_work_dir()])[0]:
            # Inside a folder of recordings, transcripts or kept output, the
            # folder's content is someone's data; inside the home, data or
            # config folder it is just a subfolder of its own. The default
            # work folder is always fine.
            for label, other in self._inside_trees(trees):
                if p == other or other in p.parents:
                    return f"{p} lies inside {label} ({other}) — the work folder is emptied"
        return ""

    def _inside_trees(self, trees: list[tuple[str, Path]]) -> list[tuple[str, Path]]:
        """Trees a work folder must not lie inside: media, allowed roots,
        transcripts, uploads, clips, logs, backups — except a tree that is
        the home folder or above it."""
        with self._lock:  # copy the Paths only; resolving happens after
            raw = [
                ("the transcript folder", self._resolved_transcripts),
                ("the uploads folder", self._resolved_uploads),
                ("the clips folder", self._resolved_clips),
                ("the logs folder", self._resolved_logs),
            ]
        backup = self.backup_dir
        if backup:
            raw.append(("the backup folder", backup))
        resolved = _resolve_all([p for _label, p in raw])
        out = [t for t in trees if t[0] in ("the media folder", "an allowed root")]
        out += [(label, p) for (label, _raw), p in zip(raw, resolved, strict=True)]
        home = _resolve_all([Path.home()])[0]
        return [(label, p) for label, p in out if not self._home_or_above(p, home)]

    # ── Validation ────────────────────────────────────────────────────

    def validate(self) -> list[str]:
        """Validate configuration. Returns list of warnings."""
        warnings = []

        model = self.get("transcription", "default_model", default="turbo")
        if model not in VALID_WHISPER_MODELS and "/" not in model:
            warnings.append(f"Unknown Whisper model: {model}")

        device = self.get("hardware", "device", default="auto")
        if device not in {"auto", "cuda", "rocm", "directml", "mps", "cpu"}:
            warnings.append(f"Unknown device: {device}")

        mode = self.get("directories", "transcript_mode", default="alongside")
        if mode not in {"alongside", "central", "database"}:
            warnings.append(f"Unknown transcript_mode: {mode}")

        port = self.get("server", "port", default=9860)
        # Port can come as YAML string ("9860") → cast before comparison,
        # otherwise `1 <= port` raises TypeError instead of clean warning.
        try:
            port_num = int(port)
        except (TypeError, ValueError):
            warnings.append(f"Invalid port: {port}")
        else:
            if not (1 <= port_num <= 65535):
                warnings.append(f"Invalid port: {port}")

        compute = self.get("hardware", "compute_type", default="auto")
        if compute not in {
            "auto",
            "float16",
            "float32",
            "int8",
            "int8_float16",
            "int8_float32",
            "bfloat16",
        }:
            warnings.append(f"Unknown compute type: {compute}")

        # Sizes a job divides by: 0 or a word would fail every long file
        for section, key in (
            ("hardware", "asr_block_max_sec"),
            ("hardware", "sep_chunk_duration_sec"),
        ):
            value = self.get(section, key)
            try:
                ok = value is None or float(value) > 0
            except (TypeError, ValueError):
                ok = False
            if not ok:
                warnings.append(f"{section}.{key} must be a number above 0 (is {value!r})")

        # Images & sounds: values the indexer can't use fall back to their
        # defaults there — said here, so a hand edit doesn't vanish silently
        media = self.get("media_search") or {}
        if isinstance(media, dict):
            kinds = media.get("kinds") or []
            if isinstance(kinds, str):
                kinds = [k.strip() for k in kinds.split(",")]
            known = {"image", "video", "videoaudio", "audio", "speech"}
            unknown = [str(k) for k in kinds if k not in known]
            if unknown:
                warnings.append(f"media_search.kinds: unknown kind(s) {', '.join(unknown)}")
            dims = media.get("dims", 768)
            if str(dims) not in {"768", "512", "256", "128"}:
                warnings.append(f"media_search.dims must be 768, 512, 256 or 128 (is {dims!r})")
            device = media.get("index_device", "cuda")
            if str(device).lower() not in {"cuda", "cpu"}:
                warnings.append(f"media_search.index_device must be cuda or cpu (is {device!r})")

        # A value of the wrong type (a hand edit) must produce a warning here,
        # not a crash of the settings page
        try:
            mdx_ovl = float(self.get("vocal_isolation", "mdx_overlap", default=0.25))
        except (TypeError, ValueError):
            mdx_ovl = 0.25
        if mdx_ovl > 0.5:
            warnings.append(
                f"vocal_isolation.mdx_overlap={mdx_ovl} is very high! "
                f"UVR5 default is 0.25. Higher = MUCH slower. "
                f"Recommendation: 0.25"
            )
        return warnings

    # ── Workdir Management ────────────────────────────────────────────

    #: Who is currently working in this workdir? Contains PID and command line.
    USAGE_FILE = ".server-busy.json"

    #: Says "this folder is a Whisper Transcription Manager work folder — its content may be
    #: deleted". Written into a work folder that is new or empty (or the
    #: default one); clean_workdir() empties no folder without it.
    WORK_MARKER = ".whisper-tm-work"

    #: What the app itself leaves in a work folder: job folders
    #: ("<name>_<ext>_<8 hex>"), the separator's and voice tools' folders,
    #: audio blocks and chunks, the busy markers. A work folder from before
    #: the marker that holds only such things is marked once (migration).
    _APP_WORK_NAMES = frozenset(
        {
            "_sep_api_out",
            "voice_samples",
            "ytdl_tmp",
            "chunks",
            "sep_chunks_in",
            "sep_chunks_out",
            "sep_trimmed",
            "sep_direct",
            "vocal_iso_work",
        }
    )
    _APP_WORK_RE = re.compile(
        r"^(?:[^.]+_[0-9a-f]{8}|chunk_.*|.*\.(?:wav|flac|m4a|mp3|opus|tmp|part))$"
    )

    def _is_app_leftover(self, name: str) -> bool:
        return (
            name == self.USAGE_FILE
            or name in self._APP_WORK_NAMES
            or bool(self._APP_WORK_RE.match(name))
        )

    def _mark_workdir(self, warn: bool = False) -> bool:
        """Is the work folder marked as ours — marking it if it is new, the
        default location, or holds only what the app itself leaves there?

        ``warn``: say at startup, in the log, why an unmarked folder stays
        uncleaned and how to mark it."""
        wd = self.work_dir
        try:
            if (wd / self.WORK_MARKER).is_file():
                return True
            if not wd.is_dir():
                return False
            names = [e.name for e in wd.iterdir()]
            default = _resolve_all([self._default_work_dir(), wd])
            foreign = [n for n in names if not self._is_app_leftover(n)]
            if default[0] == default[1] or not foreign:
                (wd / self.WORK_MARKER).write_text(
                    "Whisper Transcription Manager work folder: everything in here is deleted "
                    "on every server start.\n",
                    encoding="utf-8",
                )
                return True
            if warn:
                logger.warning(
                    "Work folder %s is not marked as a Whisper Transcription Manager work folder "
                    "and holds other files (e.g. %s) — it is NOT cleaned. If "
                    "everything in it may be deleted, create an empty file %s "
                    "there; otherwise choose an empty folder (directories.work).",
                    wd,
                    ", ".join(sorted(foreign)[:3]),
                    self.WORK_MARKER,
                )
        except OSError as e:
            logger.debug("Work folder marker: %s", e)
        return False

    def claim_workdir(self) -> None:
        """Mark the workdir as occupied.

        To be called AFTER :meth:`clean_workdir` — the marker lies in
        the workdir and would otherwise be deleted immediately.

        Never over the marker of a different LIVE server. A test app next
        to a production server once did exactly that: its
        ``clean_workdir()`` rightly stood back, then ``claim_workdir()``
        wrote its own PID over the production server's. The next
        ``create_app()`` in the same test process took the workdir for its
        own, emptied it, and the running jobs failed ("No such file or
        directory").
        """
        foreign = self._workdir_taken_by_other()
        if foreign:
            logger.error(
                "Workdir %s stays marked for %s — not claimed. A second "
                "server needs its own work directory (directories.work).",
                self.work_dir,
                foreign,
            )
            return
        try:
            wd = self.work_dir
            wd.mkdir(parents=True, exist_ok=True)
            (wd / self.USAGE_FILE).write_text(
                json.dumps(
                    {
                        "pid": os.getpid(),
                        "cmdline": " ".join(sys.argv),
                    }
                ),
                encoding="utf-8",
            )
        except Exception as e:  # never prevent startup
            logger.debug("Workdir marker not written: %s", e)

    def _workdir_taken_by_other(self) -> str:
        """Is a DIFFERENT server running in this workdir? Return: reason or "".

        An expired marker (process dead, PID reused) must not permanently
        block cleanup — so we check not just the PID but also whether
        something behind it looks like this program.
        """
        marker_time = 0.0
        try:
            data = json.loads((self.work_dir / self.USAGE_FILE).read_text(encoding="utf-8"))
            marker_time = (self.work_dir / self.USAGE_FILE).stat().st_mtime
        except Exception:
            data = None
        if data is None:
            return ""  # no marker or broken marker
        pid = int(data.get("pid") or 0)
        if not pid or pid == os.getpid():
            return ""
        try:
            import psutil

            if not psutil.pid_exists(pid):
                return ""
            proc = psutil.Process(pid)
            # Started well after the marker was written: the PID was reused
            # (Windows hands them out again quickly). Five minutes of slack —
            # a clock set forward after the start moves create_time with it,
            # and taking a running server for gone would empty its folder.
            if marker_time and proc.create_time() > marker_time + 300:
                return ""
            line = " ".join(proc.cmdline())
        except Exception:
            # Without psutil, or its command line not readable (another
            # user's process): the marker's own line, if the PID is alive.
            # Signal 0 asks that on POSIX; on Windows os.kill(pid, 0) sends
            # Ctrl+C to a process group and succeeds for any number.
            if os.name != "nt":
                try:
                    os.kill(pid, 0)
                except OSError:
                    return ""
            line = str(data.get("cmdline") or "")
        low = line.lower()
        if "whisper_tm" not in low and "whisper-tm" not in low:
            return ""  # PID reused
        # The command line stays in the log: this reason goes out over HTTP
        # (/api/clean_workdir), and a process's command line never leaves
        # the server.
        logger.info("Work folder %s is used by PID %d: %s", self.work_dir, pid, line[:200])
        return f"PID {pid}"

    def clean_workdir(self, busy=None) -> dict:
        """Delete the work folder's contents — all but this server's marker.

        ``busy`` (optional) is a function answering "is a job using the
        folder right now?" with a reason, or "" when it is free. It is asked
        before EVERY entry: a job that starts while the cleanup runs keeps
        the folder it has just created. /api/clean_workdir passes the
        queue's state.

        Returns what happened, for the caller to report: ``ok``, ``removed``
        and ``failed`` (entries deleted / not deletable), ``refused`` (it
        stood back on purpose) and, unless ``ok``, an ``error`` — why it
        stood back or what it could not delete. It used to return nothing,
        and the settings page said "Workdir cleaned" even when not a single
        file had been touched.
        """

        def refused(reason: str, removed: int = 0, failed: int = 0) -> dict:
            return {
                "ok": False,
                "refused": True,
                "removed": removed,
                "failed": failed,
                "error": reason,
            }

        wd = self.work_dir
        # OPERATIONAL SAFETY: A second server with the same configuration
        # would delete the FIRST server's work files at startup.
        # Example: a test server on a second port gets a config copy —
        # including `work: ~/.cache/whisper-tm/work` — and the
        # production server's running job dies seconds later with
        # "asr_block_001.wav: No such file or directory".
        #
        # The checks below protect USER DATA from misconfiguration.
        # This one protects RUNNING WORK from a second process —
        # a different case that path validation can't see because both
        # servers legitimately use the same directory.
        foreign = self._workdir_taken_by_other()
        if foreign:
            logger.error(
                "clean_workdir aborted: %s already working in %s — "
                "NOTHING is deleted. A second server needs its own "
                "work directory (directories.work).",
                foreign,
                wd,
            )
            return refused(
                f"Another server ({foreign}) works in {wd} — "
                f"nothing deleted. A second server needs its own "
                f"work directory (directories.work)."
            )
        # DATA PROTECTION: clean_workdir recursively deletes EVERYTHING under workdir.
        # With misconfiguration (e.g. work_dir == media_dir) it would
        # irreversibly destroy real user data. So refuse to delete if work_dir
        # is, or contains, any folder that holds data: media, transcripts,
        # the databases, uploads, clips, logs, the backup target, an allowed
        # root, the config folder — or the home folder itself (a work folder
        # typed as "~" or "/"). And refuse a work folder INSIDE a folder of
        # recordings or kept output (media, allowed roots, transcripts,
        # uploads, clips, logs, backups): "<media>/Series" as work folder
        # used to be emptied, recordings, .transcripts/ and all.
        try:
            from whisper_tm.core.security import allowed_roots

            with self._lock:  # copy the Paths; resolve without the lock
                raw = [
                    ("work", self._resolved_work),
                    ("default_work", self._resolved_cache / "work"),
                    ("home", Path.home()),
                    ("media_dir", self._resolved_media),
                    ("transcript_dir", self._resolved_transcripts),
                    ("db_dir", self._resolved_db.parent),
                    ("uploads", self._resolved_uploads),
                    ("clips", self._resolved_clips),
                    ("logs", self._resolved_logs),
                    ("data_dir", self.data_dir),
                    ("config_dir", Path(user_config_dir(APP_NAME))),
                ]
            backup = self.backup_dir
            if backup:
                raw.append(("backup_path", backup))
            resolved = _resolve_all([p for _n, p in raw])
            wd_r, default_r, home_r = resolved[:3]
            protected = [(n, p) for (n, _raw), p in zip(raw[2:], resolved[2:], strict=True)]
            protected += [("allowed_root", r) for r in allowed_roots()]
        except (OSError, ValueError) as e:
            logger.error("clean_workdir aborted (path not resolvable): %s", e)
            return refused(f"Work folder path not resolvable: {e}")
        # Inside these it is just a subfolder of its own — and inside a tree
        # that is the home folder or above it (root "~" or "/", media "~"),
        # where the default work folder lives. The default is always fine.
        not_inside = ("db_dir", "data_dir", "config_dir", "home")
        for name, other in protected:
            inside = (
                wd_r != default_r
                and name not in not_inside
                and not self._home_or_above(other, home_r)
                and other in wd_r.parents
            )
            if wd_r == other or wd_r in other.parents or inside:
                logger.error(
                    "clean_workdir aborted: work_dir %s overlaps with %s (%s) "
                    "— NOTHING deleted to protect user data.",
                    wd_r,
                    other,
                    name,
                )
                return refused(
                    f"The work folder {wd_r} overlaps with {other} "
                    f"({name}) — nothing deleted to protect user data."
                )
        # Only a folder this app made its work folder is emptied: one that
        # carries the work marker. A folder that is already full of foreign
        # files (directories.work pointed at ~/Documents) stays untouched.
        if not self._mark_workdir():
            logger.error(
                "clean_workdir aborted: %s has no %s marker and is not empty — NOTHING deleted.",
                wd_r,
                self.WORK_MARKER,
            )
            return refused(
                f"The work folder {wd_r} is not marked as a Whisper Transcription Manager work "
                f"folder ({self.WORK_MARKER}) and holds other files — nothing "
                f"deleted. Choose an empty folder, or create the marker file "
                f"there if everything in it may go."
            )
        # A list, not the live directory iterator: whatever a job creates
        # after this look is never touched.
        try:
            items = list(wd.iterdir()) if wd.exists() else []
        except OSError as e:
            return {
                "ok": False,
                "refused": False,
                "removed": 0,
                "failed": 0,
                "error": f"Work folder not readable: {e}",
            }
        removed, failed, first_error = 0, 0, ""
        for item in items:
            # The marker says this server works here. Deleting it left the
            # folder unmarked until the next restart — a second server would
            # have taken it as free and cleared it under a running job.
            if item.name in (self.USAGE_FILE, self.WORK_MARKER):
                continue
            reason = busy() if busy else ""
            if reason:
                logger.warning("clean_workdir stopped after %d entries: %s", removed, reason)
                return refused(reason, removed, failed)
            try:
                if item.is_dir() and not item.is_symlink():
                    shutil.rmtree(item)
                else:
                    item.unlink()  # a link goes, never what it points to
                removed += 1
            except Exception as e:
                failed += 1
                first_error = first_error or f"{item.name}: {e}"
                logger.debug("Work folder entry not removed (%s): %s", item, e)
        result = {"ok": not failed, "refused": False, "removed": removed, "failed": failed}
        if failed:
            result["error"] = f"{failed} entries could not be deleted (first: {first_error})"
        return result

    # ── Media Extensions ──────────────────────────────────────────────

    @property
    def video_extensions(self) -> set[str]:
        return VIDEO_EXTENSIONS | _extensions(
            self.get("scanner", "extra_video_extensions", default=[])
        )

    @property
    def audio_extensions(self) -> set[str]:
        return AUDIO_EXTENSIONS | _extensions(
            self.get("scanner", "extra_audio_extensions", default=[])
        )

    @property
    def media_extensions(self) -> set[str]:
        return self.video_extensions | self.audio_extensions

    # ── Convenience ───────────────────────────────────────────────────

    @property
    def hf_token(self) -> str:
        return self.get("transcription", "hf_token", default="") or os.environ.get("HF_TOKEN", "")

    @property
    def backup_dir(self) -> Path | None:
        """``database.backup_path`` as a folder, or None when it is not set
        (backups then go next to the database)."""
        return self.path_setting("database", "backup_path")

    def path_setting(self, section: str, key: str) -> Path | None:
        """A file or folder setting outside ``directories`` as a path, or
        None when it is empty (see :meth:`expand_setting`)."""
        return self.expand_setting(self.get(section, key, default=""), f"{section}.{key}")

    def expand_setting(self, raw, name: str) -> Path | None:
        """``raw``, the value of the setting ``name``, with ``~``, ``${VAR}``
        and a path relative to the config file resolved like the folders —
        or None when it is empty. An undefined variable leaves it unset,
        with a warning: taken literally, mkdir would create a folder named
        ``${VAR}``."""
        raw = str(raw or "").strip()
        if not raw:
            return None
        try:
            return _expand_lexical(raw, self._base_dir)
        except ValueError as e:
            logger.warning("%s ignored: %s", name, e)
            return None

    @property
    def transcript_mode(self) -> str:
        return self.get("directories", "transcript_mode", default="alongside")


# ── Singleton ─────────────────────────────────────────────────────────────

config = ConfigManager()
