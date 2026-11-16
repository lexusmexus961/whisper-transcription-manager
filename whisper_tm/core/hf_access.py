"""The Hugging Face token and the gated diarization model.

Speaker diarization runs pyannote's ``speaker-diarization-community-1``
(segmentation, embedding and PLDA in one repository, ~32 MB). The model is
free but gated: Hugging Face hands it out only to an account that accepted
its conditions (a short form, approved at once), and only with a token of
that account. Once it is in the local cache it loads without one
(core/transcription.diarization_possible).

The setup scripts (``whisper-tm --hf-token``), the setup guide
(``/api/setup/hf_token``), the system check and the start banner ask the
same questions, answered here:

* :func:`token_source` — is a token configured (config file or ``HF_TOKEN``)?
* :func:`model_cached` — is the model downloaded already?
* :func:`check` — does a token work, and may it fetch the model? Two plain
  HTTP requests with a timeout: huggingface_hub's own ``whoami`` and
  ``auth_check`` set none and can hang on a dead network.
* :func:`save_token` — into the config file the app loads, its comments kept.
* :func:`download` — the model into the Hugging Face cache.

A token is never logged, printed or returned.
"""

from __future__ import annotations

import getpass
import os
import sys
from pathlib import Path

REPO = "pyannote/speaker-diarization-community-1"
SITE = "https://huggingface.co"
REPO_URL = f"{SITE}/{REPO}"
TOKENS_URL = f"{SITE}/settings/tokens"
JOIN_URL = f"{SITE}/join"
#: Download size of the repository, for the messages
MODEL_MB = 32

#: What a person does once, in this order
STEPS = (
    f"Sign in at {SITE} (or create a free account: {JOIN_URL}).",
    f"Open {REPO_URL} and accept the conditions — the form asks for a "
    "company or university and a use case; access is granted at once.",
    f'Create an access token of type "Read" at {TOKENS_URL}.',
)

#: A fine-grained token reads a gated model only with this permission ticked
FINE_GRAINED_HINT = (
    'A fine-grained token also needs "Read access to contents of all '
    'public gated repos you can access" — or use a token of type "Read".'
)


def _endpoint() -> str:
    """huggingface_hub's own setting: a mirror set in HF_ENDPOINT is used for
    the downloads, so it is asked here too."""
    return (os.environ.get("HF_ENDPOINT") or SITE).rstrip("/")


def token_source() -> str:
    """Where the app's token comes from: "config", "env" (``HF_TOKEN``) or ""."""
    from whisper_tm.config import config

    if config.get("transcription", "hf_token", default=""):
        return "config"
    return "env" if os.environ.get("HF_TOKEN", "") else ""


def repo_cached(repo_id: str) -> bool:
    """Is the HF repo fully cached locally? (then offline loading is allowed)

    Two places need the same answer and must not answer it differently:
    the context manager in core/transcription (own process, flips
    ``constants.HF_HUB_OFFLINE``) and the start of the diarization worker
    (child process, gets ``HF_HUB_OFFLINE`` via the environment).
    """
    try:
        from huggingface_hub import snapshot_download

        # Raises if the repo is not fully cached.
        snapshot_download(repo_id, local_files_only=True)
        return True
    except Exception:
        return False


def model_cached() -> bool:
    """Is the diarization model downloaded? Then no token is needed any more."""
    return repo_cached(REPO)


def check(token: str | None, timeout: float = 15.0) -> dict:
    """Does ``token`` work, and may it download the model?

    ``{"status", "user", "message"}`` — ``user`` is the account's name. Status:

    * ``ok`` — the token works and may download the model
    * ``gated`` — the token works, but its account has not accepted the
      model's conditions (or a fine-grained token lacks the permission)
    * ``invalid`` — Hugging Face rejects the token
    * ``no_token`` — there is none
    * ``offline`` — Hugging Face did not answer (network, proxy, timeout);
      nothing is known about the token
    * ``error`` — any other answer, named in the message
    """
    token = (token or "").strip()
    if not token:
        return {
            "status": "no_token",
            "user": "",
            "message": "No token — the model can't be downloaded.",
        }
    import requests

    headers = {"Authorization": f"Bearer {token}"}
    try:
        who = requests.get(f"{_endpoint()}/api/whoami-v2", headers=headers, timeout=timeout)
    except requests.RequestException as e:
        return {
            "status": "offline",
            "user": "",
            "message": f"Hugging Face is not reachable ({type(e).__name__}).",
        }
    if who.status_code == 401:
        return {
            "status": "invalid",
            "user": "",
            "message": "Hugging Face rejects this token — copy it again, or create "
            f"a new one at {TOKENS_URL}.",
        }
    if who.status_code != 200:
        return {
            "status": "error",
            "user": "",
            "message": f"Hugging Face answered {who.status_code} to the token check.",
        }
    try:
        me = who.json()
    except ValueError:
        me = {}
    user = str(me.get("name") or "")
    role = ((me.get("auth") or {}).get("accessToken") or {}).get("role", "")
    try:
        r = requests.get(
            f"{_endpoint()}/api/models/{REPO}/auth-check", headers=headers, timeout=timeout
        )
    except requests.RequestException as e:
        return {
            "status": "offline",
            "user": user,
            "message": f"Hugging Face is not reachable ({type(e).__name__}).",
        }
    if r.status_code == 200:
        return {
            "status": "ok",
            "user": user,
            "message": f"The token works and may download {REPO}.",
        }
    if r.status_code in (401, 403):
        hint = f" {FINE_GRAINED_HINT}" if role == "fineGrained" else ""
        return {
            "status": "gated",
            "user": user,
            "message": f"The token works, but its account has not accepted the "
            f"conditions of {REPO} yet: open {REPO_URL} and accept "
            f"them, then check again.{hint}",
        }
    return {
        "status": "error",
        "user": user,
        "message": f"Hugging Face answered {r.status_code} for {REPO}.",
    }


def save_token(token: str) -> Path:
    """Write ``token`` into the config file the app loads next start (the one
    it was loaded from), and into the running configuration.

    The file keeps its comments: only the ``hf_token`` line changes. Without
    a file, or one that can't be edited line by line, the configuration is
    saved the usual way (config.save).
    """
    from whisper_tm.config import config

    if not config._loaded:
        config.load(None)
    return config.save_value("transcription", "hf_token", token)


def download(token: str | None = None) -> str:
    """The model into the Hugging Face cache; its folder. Raises on failure."""
    from huggingface_hub import snapshot_download

    return snapshot_download(REPO, token=(token or "").strip() or None)


# ── whisper-tm --hf-token ──────────────────────────────────────────────


def _say(mark: str, text: str) -> None:
    print(f"  {mark} {text}", flush=True)


def _print_steps() -> None:
    print("  Telling speakers apart (diarization) uses pyannote's free model")
    print(f"  {REPO}. Hugging Face hands it out only to")
    print("  signed-in accounts that accepted its conditions — once; after the")
    print(f"  first download (~{MODEL_MB} MB) it loads from the cache without a token.")
    print()
    for n, s in enumerate(STEPS, 1):
        print(f"    {n}. {s}")
    print()
    print("  Without it everything else works — transcripts just have no speakers.")
    print()


def _ask(prompt: str, hidden: bool = False) -> str:
    try:
        return (getpass.getpass(prompt) if hidden else input(prompt)).strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return ""


def cli(interactive: bool | None = None) -> int:
    """``whisper-tm --hf-token``: ask for the token, check it, save it,
    download the model.

    The token comes from the ``HF_TOKEN`` environment variable when it is set
    — that is how the setup scripts hand it over, never on a command line,
    where other users' ``ps`` would show it — else from the configuration or
    a hidden prompt. ``interactive=False`` (``--no-input``) never asks.

    Exit code 0 when diarization can run (the model downloaded, or the token
    may download it); 1 when not.
    """
    from whisper_tm.config import config

    if not config._loaded:
        config.load(None)
    if interactive is None:
        interactive = sys.stdin.isatty()
    print()
    print("Speaker diarization — Hugging Face token")
    print("=" * 50)
    cached = model_cached()
    configured = str(config.get("transcription", "hf_token", default="") or "").strip()
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token and interactive:
        _print_steps()
        if cached:
            _say("✓", "The model is downloaded already — a token is only needed to fetch it again.")
        prompt = (
            "  Paste a new token (hidden), or press Enter to keep the configured one: "
            if configured
            else "  Paste the token (hidden), or press Enter to skip: "
        )
        token = _ask(prompt, hidden=True)
    if not token:
        token = configured
    if not token:
        if cached:
            _say("✓", "No token, but the model is downloaded — diarization works.")
            return 0
        if not interactive:
            _print_steps()
        from whisper_tm.core import commands

        _say("·", f"Skipped. Later: {commands.start('--hf-token')}, or Settings → Speakers.")
        return 1

    while True:
        res = check(token)
        st = res["status"]
        who = f" (account {res['user']})" if res.get("user") else ""
        if st == "invalid":
            _say("✗", res["message"])
            if not interactive:
                return 0 if cached else 1
            token = _ask("  Paste the token again (hidden), or press Enter to skip: ", hidden=True)
            if not token:
                return 0 if cached else 1
            continue
        if st in ("ok", "gated", "offline") and token != configured:
            path = save_token(token)
            configured = token
            _say("✓", f"Token saved in {path}")
        if st == "ok":
            _say("✓", f"Access to {REPO}{who}.")
            if not cached:
                _say("·", f"Downloading the model (~{MODEL_MB} MB) …")
                try:
                    download(token)
                    cached = True
                    _say("✓", "Model downloaded — speakers can be told apart.")
                except Exception as e:  # noqa: BLE001 — the first diarized file fetches it again
                    _say(
                        "⚠",
                        f"Download failed ({type(e).__name__}); the first file with "
                        "speakers tries again.",
                    )
            return 0
        if st == "gated":
            _say("⚠", res["message"] + who)
            if not interactive:
                return 0 if cached else 1
            again = _ask("  Accept them, then press Enter to check again (s = go on without): ")
            if again.lower().startswith("s"):
                return 0 if cached else 1
            continue
        if st == "offline":
            from whisper_tm.core import commands

            _say(
                "⚠",
                res["message"] + " The token is saved unchecked; check it later "
                f"with {commands.start('--hf-token')}.",
            )
            return 0 if cached else 1
        _say("✗", res["message"])
        return 0 if cached else 1
