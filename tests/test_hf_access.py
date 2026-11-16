"""
The Hugging Face token for speaker diarization (core/hf_access).

The pyannote model is gated: a token works only for an account that accepted
the model's conditions. The setup scripts, ``whisper-tm --hf-token``, the
setup guide and the system check must tell those cases apart — no token, a
rejected token, a token whose account has not accepted the conditions yet,
no network — and say what to do, without the token ever leaving the process
in an answer, a log line or a command line.
"""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

import pytest
import requests
import yaml
from flask import Flask

from whisper_tm.config import ConfigManager, set_in_yaml_text
from whisper_tm.core import hf_access

ROOT = Path(__file__).resolve().parents[1]
TOKEN = "hf_test0123456789abcdef"


class _Resp:
    def __init__(self, status, body=None):
        self.status_code = status
        self._body = body or {}

    def json(self):
        return self._body


@pytest.fixture
def hub(monkeypatch):
    """Hugging Face's two answers: ``who`` for the token, ``repo`` for the model."""
    state = {
        "who": _Resp(200, {"name": "alice", "auth": {"accessToken": {"role": "read"}}}),
        "repo": _Resp(200),
        "calls": [],
    }

    def get(url, headers=None, timeout=None):
        state["calls"].append((url, headers, timeout))
        key = "who" if url.endswith("/api/whoami-v2") else "repo"
        answer = state[key]
        if isinstance(answer, Exception):
            raise answer
        return answer

    monkeypatch.setattr(requests, "get", get)
    monkeypatch.delenv("HF_ENDPOINT", raising=False)
    return state


class TestCheck:
    def test_no_token(self, hub):
        assert hf_access.check("")["status"] == "no_token"
        assert hf_access.check(None)["status"] == "no_token"
        assert hub["calls"] == []

    def test_a_token_with_access(self, hub):
        res = hf_access.check(TOKEN)
        assert res["status"] == "ok" and res["user"] == "alice"
        (who, h1, t1), (repo, h2, t2) = hub["calls"]
        assert who == "https://huggingface.co/api/whoami-v2"
        assert repo == f"https://huggingface.co/api/models/{hf_access.REPO}/auth-check"
        assert h1 == h2 == {"Authorization": f"Bearer {TOKEN}"}
        assert t1 and t2, "every request has a timeout"
        assert TOKEN not in str(res)

    def test_conditions_not_accepted(self, hub):
        hub["repo"] = _Resp(403)
        res = hf_access.check(TOKEN)
        assert res["status"] == "gated" and res["user"] == "alice"
        assert hf_access.REPO_URL in res["message"]
        assert "fine-grained" not in res["message"]

    def test_a_fine_grained_token_hears_about_its_permission(self, hub):
        hub["who"] = _Resp(200, {"name": "alice", "auth": {"accessToken": {"role": "fineGrained"}}})
        hub["repo"] = _Resp(403)
        assert "public gated repos" in hf_access.check(TOKEN)["message"]

    def test_a_rejected_token(self, hub):
        hub["who"] = _Resp(401)
        res = hf_access.check(TOKEN)
        assert res["status"] == "invalid" and hf_access.TOKENS_URL in res["message"]
        assert len(hub["calls"]) == 1

    def test_no_network_says_nothing_about_the_token(self, hub):
        hub["who"] = requests.ConnectionError("down")
        assert hf_access.check(TOKEN)["status"] == "offline"
        hub["who"] = _Resp(200, {"name": "alice"})
        hub["repo"] = requests.Timeout("slow")
        assert hf_access.check(TOKEN)["status"] == "offline"

    def test_any_other_answer_is_named(self, hub):
        hub["repo"] = _Resp(500)
        res = hf_access.check(TOKEN)
        assert res["status"] == "error" and "500" in res["message"]

    def test_a_mirror_in_hf_endpoint_is_asked(self, hub, monkeypatch):
        monkeypatch.setenv("HF_ENDPOINT", "https://hf-mirror.example/")
        hf_access.check(TOKEN)
        assert all(u.startswith("https://hf-mirror.example/api/") for u, _h, _t in hub["calls"])


class TestOneLineInTheConfig:
    def test_the_default_file_changes_in_that_line_only(self):
        text = (ROOT / "config.default.yaml").read_text(encoding="utf-8")
        new = set_in_yaml_text(text, "transcription", "hf_token", TOKEN)
        changed = [(a, b) for a, b in zip(text.splitlines(), new.splitlines()) if a != b]
        assert changed == [('  hf_token: ""', f'  hf_token: "{TOKEN}"')]
        assert len(text.splitlines()) == len(new.splitlines())

    def test_a_missing_key_goes_to_the_top_of_its_section(self):
        text = "server:\n  port: 1\ntranscription:\n  # a comment\n  model: x\n"
        new = set_in_yaml_text(text, "transcription", "hf_token", "t")
        assert (
            new
            == 'server:\n  port: 1\ntranscription:\n  hf_token: "t"\n  # a comment\n  model: x\n'
        )

    def test_a_missing_section_is_appended(self):
        assert (
            set_in_yaml_text("", "transcription", "hf_token", "t")
            == 'transcription:\n  hf_token: "t"\n'
        )
        assert (
            set_in_yaml_text("server:\n  port: 1", "llm", "endpoint", "u")
            == 'server:\n  port: 1\nllm:\n  endpoint: "u"\n'
        )

    def test_a_deeper_key_of_the_same_name_is_left_alone(self):
        text = "transcription:\n  model: x\n  sub:\n    hf_token: keep\n"
        data = yaml.safe_load(set_in_yaml_text(text, "transcription", "hf_token", "t"))
        assert data["transcription"]["hf_token"] == "t"
        assert data["transcription"]["sub"]["hf_token"] == "keep"

    def test_what_it_cant_follow_is_left_to_a_plain_save(self):
        assert (
            set_in_yaml_text("transcription: {model: x}\n", "transcription", "hf_token", "t")
            is None
        )
        assert set_in_yaml_text("transcription: 5\n", "transcription", "hf_token", "t") is None
        assert set_in_yaml_text("- a list\n", "transcription", "hf_token", "t") is None

    @pytest.mark.parametrize("value", ['say "hi": #x', "ümlaut ✓", 8080, True])
    def test_values_read_back_exactly(self, value):
        new = set_in_yaml_text('llm:\n  endpoint: ""\n', "llm", "endpoint", value)
        assert yaml.safe_load(new)["llm"]["endpoint"] == value

    def test_save_value_keeps_the_comments_and_the_mode(self, tmp_path):
        file = tmp_path / "config.yaml"
        file.write_text('# my notes\ntranscription:\n  # why\n  hf_token: ""\n', encoding="utf-8")
        os.chmod(file, 0o644)
        c = ConfigManager()
        c.load(file)
        assert c.save_value("transcription", "hf_token", TOKEN) == file
        assert file.read_text() == f'# my notes\ntranscription:\n  # why\n  hf_token: "{TOKEN}"\n'
        assert stat.S_IMODE(file.stat().st_mode) == 0o600
        assert c.get("transcription", "hf_token") == TOKEN

    def test_save_value_falls_back_to_a_full_save(self, tmp_path):
        file = tmp_path / "config.yaml"
        file.write_text("transcription: {model: x}\n", encoding="utf-8")
        c = ConfigManager()
        c.load(file)
        c.save_value("transcription", "hf_token", TOKEN)
        data = yaml.safe_load(file.read_text())
        assert data["transcription"]["hf_token"] == TOKEN
        assert data["transcription"]["model"] == "x"


@pytest.fixture
def cli_env(monkeypatch, capsys):
    """cli() with Hugging Face, the config file and the cache replaced."""
    from whisper_tm.config import config

    state = {"answers": [], "saved": [], "downloads": 0, "cached": False, "configured": ""}

    def check(token, timeout=15.0):
        state.setdefault("checked", []).append(token)
        status = state["answers"].pop(0) if state["answers"] else "ok"
        return {"status": status, "user": "alice", "message": f"[{status}]"}

    def download(token=None):
        state["downloads"] += 1
        state["cached"] = True
        return "/cache"

    real_get = config.get

    def get(*keys, default=None):
        if keys == ("transcription", "hf_token"):
            return state["configured"]
        return real_get(*keys, default=default)

    monkeypatch.setattr(hf_access, "check", check)
    monkeypatch.setattr(
        hf_access, "save_token", lambda t: state["saved"].append(t) or Path("/cfg.yaml")
    )
    monkeypatch.setattr(hf_access, "download", download)
    monkeypatch.setattr(hf_access, "model_cached", lambda: state["cached"])
    monkeypatch.setattr(config, "get", get)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    state["out"] = lambda: capsys.readouterr().out
    return state


class TestCommandLine:
    def test_a_token_from_the_environment_is_checked_saved_and_used(self, cli_env, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", TOKEN)
        assert hf_access.cli(interactive=False) == 0
        assert cli_env["saved"] == [TOKEN] and cli_env["downloads"] == 1
        assert TOKEN not in cli_env["out"]()

    def test_not_accepted_yet_is_saved_but_not_ready(self, cli_env, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", TOKEN)
        cli_env["answers"] = ["gated"]
        assert hf_access.cli(interactive=False) == 1
        assert cli_env["saved"] == [TOKEN] and cli_env["downloads"] == 0
        assert "[gated]" in cli_env["out"]()

    def test_a_rejected_token_is_not_saved(self, cli_env, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", TOKEN)
        cli_env["answers"] = ["invalid"]
        assert hf_access.cli(interactive=False) == 1
        assert cli_env["saved"] == []

    def test_the_configured_token_is_not_saved_again(self, cli_env):
        cli_env["configured"] = TOKEN
        assert hf_access.cli(interactive=False) == 0
        assert cli_env["checked"] == [TOKEN] and cli_env["saved"] == []

    def test_nothing_to_do_without_a_token(self, cli_env):
        assert hf_access.cli(interactive=False) == 1
        assert hf_access.REPO_URL in cli_env["out"](), "the steps are shown"
        cli_env["cached"] = True
        assert hf_access.cli(interactive=False) == 0

    def test_asked_with_hidden_input_and_checked_again_after_accepting(self, cli_env, monkeypatch):
        asked = []
        monkeypatch.setattr(hf_access.getpass, "getpass", lambda p: asked.append(p) or TOKEN)
        monkeypatch.setattr("builtins.input", lambda p: "")  # Enter: check again
        cli_env["answers"] = ["gated", "ok"]
        assert hf_access.cli(interactive=True) == 0
        assert len(asked) == 1 and cli_env["checked"] == [TOKEN, TOKEN]
        assert cli_env["saved"] == [TOKEN] and cli_env["downloads"] == 1

    def test_s_goes_on_without(self, cli_env, monkeypatch):
        monkeypatch.setattr(hf_access.getpass, "getpass", lambda p: TOKEN)
        monkeypatch.setattr("builtins.input", lambda p: "s")
        cli_env["answers"] = ["gated"]
        assert hf_access.cli(interactive=True) == 1

    def test_the_command_line_flag(self):
        main = (ROOT / "whisper_tm" / "__main__.py").read_text(encoding="utf-8")
        assert '"--hf-token"' in main and "hf_access.cli(" in main


@pytest.fixture
def client(monkeypatch):
    from whisper_tm.api.setup import setup_bp

    calls = {"check": [], "saved": [], "downloads": 0, "cached": False, "status": "ok"}

    def check(token, timeout=15.0):
        calls["check"].append(token)
        return {"status": calls["status"], "user": "alice", "message": "m"}

    def download(token=None):
        calls["downloads"] += 1
        calls["cached"] = True

    monkeypatch.setattr(hf_access, "check", check)
    monkeypatch.setattr(hf_access, "save_token", lambda t: calls["saved"].append(t))
    monkeypatch.setattr(hf_access, "download", download)
    monkeypatch.setattr(hf_access, "model_cached", lambda: calls["cached"])
    app = Flask(__name__)
    app.register_blueprint(setup_bp)
    c = app.test_client()
    c.calls = calls
    return c


class TestSetupGuideEndpoints:
    def test_status_needs_no_network(self, client):
        d = client.get("/api/setup/hf_token").get_json()
        assert d["model_cached"] is False and d["repo_url"] == hf_access.REPO_URL
        assert d["token"] in ("", "config", "env")
        assert client.calls["check"] == []

    def test_a_working_token_is_saved_and_the_model_fetched(self, client):
        r = client.post("/api/setup/hf_token", json={"token": f" {TOKEN} "})
        d = r.get_json()
        assert d["status"] == "ok" and d["saved"] is True and d["model_cached"] is True
        assert client.calls["saved"] == [TOKEN] and client.calls["downloads"] == 1
        assert TOKEN not in r.get_data(as_text=True)

    def test_not_accepted_yet_is_saved_without_a_download(self, client):
        client.calls["status"] = "gated"
        d = client.post("/api/setup/hf_token", json={"token": TOKEN}).get_json()
        assert d["status"] == "gated" and d["saved"] is True and client.calls["downloads"] == 0

    def test_a_rejected_token_is_not_saved(self, client):
        client.calls["status"] = "invalid"
        d = client.post("/api/setup/hf_token", json={"token": TOKEN}).get_json()
        assert d["saved"] is False and client.calls["saved"] == []

    def test_without_a_token_the_configured_one_is_checked(self, client, monkeypatch):
        from whisper_tm.config import config

        monkeypatch.setattr(type(config), "hf_token", property(lambda self: "hf_configured0"))
        d = client.post("/api/setup/hf_token", json={}).get_json()
        assert client.calls["check"] == ["hf_configured0"] and d["saved"] is False

    @pytest.mark.parametrize("body", [{"token": "hf_x\nX-Evil: 1"}, {"token": "short"}, ["hf_x"]])
    def test_what_is_no_token_is_refused(self, client, body):
        assert client.post("/api/setup/hf_token", json=body).status_code == 400
        assert client.calls["check"] == []


class TestSystemCheck:
    @pytest.mark.parametrize(
        "cached,token,status,word",
        [
            (True, "", "ok", "downloaded"),
            (False, TOKEN, "ok", "token set"),
            (False, "", "info", "no Hugging Face token"),
        ],
    )
    def test_the_diarization_row(self, monkeypatch, cached, token, status, word):
        from whisper_tm.config import config
        from whisper_tm.core import diagnostics as dg

        monkeypatch.setattr(hf_access, "model_cached", lambda: cached)
        monkeypatch.setattr(type(config), "hf_token", property(lambda self: token))
        row = next(c for c in dg._feature_checks() if c["title"] == "Speaker diarization")
        assert row["status"] == status and word in row["detail"]
        if status != "ok":
            assert hf_access.REPO_URL in row["hint"] and "--hf-token" in row["hint"]


class TestTheSetupScripts:
    SETUP = (ROOT / "setup.sh").read_text(encoding="utf-8")

    def _fn(self, name):
        i = self.SETUP.index(f"{name}() {{")
        return self.SETUP[i : self.SETUP.index("\n}\n", i) + 3]

    def test_the_token_is_never_an_argument(self):
        r = subprocess.run(
            ["bash", str(ROOT / "setup.sh"), "--hf-token", TOKEN], capture_output=True, text=True
        )
        # the way out is the hidden question — not HF_TOKEN=… on the command
        # line, which the shell history keeps as well
        out = r.stdout + r.stderr
        assert r.returncode == 1 and "./start.sh --hf-token" in out and "HF_TOKEN=" not in out
        assert (
            'HF_TOKEN="$HF_TOKEN_VALUE" "$VENV_DIR/bin/python" -m whisper_tm --hf-token --no-input'
        ) in self.SETUP
        assert "read -rs -p" in self.SETUP

    def test_help_names_the_options(self):
        r = subprocess.run(
            ["bash", str(ROOT / "setup.sh"), "--help"], capture_output=True, text=True
        )
        assert r.returncode == 0 and "--no-hf-token" in r.stdout and "HF_TOKEN=" not in r.stdout
        assert "./start.sh --hf-token" in r.stdout and "shell history" in r.stdout

    @pytest.mark.parametrize(
        "code,status",
        [
            ("200", "ok"),
            ("403", "gated"),
            ("401", "invalid"),
            ("000", "offline"),
            ("502", "error 502"),
        ],
    )
    def test_the_early_check_hands_the_token_to_curl_on_stdin(self, tmp_path, code, status):
        fake = tmp_path / "curl"
        fake.write_text(
            "#!/bin/bash\n"
            f'for a in "$@"; do [[ "$a" == *{TOKEN}* ]] && {{ echo LEAK; exit 0; }}; done\n'
            f'grep -q "Authorization: Bearer {TOKEN}" || {{ echo NOHEADER; exit 0; }}\n'
            f"printf {code}\n"
        )
        fake.chmod(0o755)
        script = f'set -euo pipefail\nHF_REPO=r/m\n{self._fn("hf_status")}\nhf_status "$1"'
        r = subprocess.run(
            ["bash", "-c", script, "x", TOKEN],
            capture_output=True,
            text=True,
            env={**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}"},
        )
        assert r.stdout.strip() == status

    def test_the_config_the_app_loads_is_found(self, tmp_path):
        home = tmp_path / "home"
        (home / ".config/whisper-tm").mkdir(parents=True)
        here = tmp_path / "app"
        here.mkdir()
        (here / "config.yaml").write_text('transcription:\n  hf_token: "hf_x"\n')
        script = (
            f'OS=Linux\nSCRIPT_DIR="{here}"\n{self._fn("app_config_file")}\n'
            f"{self._fn('hf_token_configured')}\napp_config_file; hf_token_configured && echo yes"
        )
        env = {
            k: v for k, v in os.environ.items() if k not in ("XDG_CONFIG_HOME", "WHISPER_TM_CONFIG")
        }
        env["HOME"] = str(home)
        run = lambda: subprocess.run(  # noqa: E731
            ["bash", "-c", script],
            capture_output=True,
            text=True,
            env=env,
        ).stdout.split()
        assert run() == [str(here / "config.yaml"), "yes"]
        (home / ".config/whisper-tm/config.yaml").write_text("server:\n  port: 1\n")
        assert run() == [str(home / ".config/whisper-tm/config.yaml")], "the home one wins"

    def test_python_versions(self, tmp_path):
        old = tmp_path / "python3.14"
        old.write_text("#!/bin/bash\nexec python3 -c 'print(\"\")'\n")  # what 3.14 answers
        old.chmod(0o755)
        script = (
            f"{self._fn('python_fits')}\n"
            f'python_fits "$1" && echo fits; python_fits "{old}" || echo no'
        )
        import sys

        r = subprocess.run(
            ["bash", "-c", script, "x", sys.executable], capture_output=True, text=True
        )
        assert r.stdout.split() == ["fits", "no"]

    def test_torch_keeps_the_version_whisperx_pinned(self):
        assert "cu121" not in self.SETUP and "rocm6.0" not in self.SETUP
        pins = self.SETUP[self.SETUP.index("<< 'PYPINS'") + 12 : self.SETUP.index("\nPYPINS")]
        import sys

        out = subprocess.run(
            [sys.executable, "-c", pins, "cu128"], capture_output=True, text=True
        ).stdout
        torch_version = __import__("importlib.metadata").metadata.version("torch").split("+")[0]
        assert f"torch=={torch_version}+cu128" in out.split()

    def test_lavasr_is_gone(self):
        makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
        assert "lavasr" not in (self.SETUP + makefile).lower()


def test_the_setup_guide_page():
    js = (ROOT / "whisper_tm/static/js/setupguide.js").read_text(encoding="utf-8")
    assert "id: 'speakers'" in js and "speakers: sgSpeakers" in js
    for value in (hf_access.REPO, hf_access.REPO_URL, hf_access.TOKENS_URL, hf_access.JOIN_URL):
        assert value in js, f"the guide's fallback {value} matches core/hf_access"
    assert "/api/setup/hf_token" in js
