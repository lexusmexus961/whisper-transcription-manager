"""The login and the API token behind a guessing brake, and login cookies
that end when the password changes."""

import pytest
from flask import Flask

from whisper_tm.config import config
from whisper_tm.core import auth


@pytest.fixture
def client(monkeypatch):
    if not config.is_loaded:
        config.load()
    values = {
        "enabled": True,
        "password": "right",
        "api_token": "tok",
        "public_paths": ["/api/health"],
    }
    real_get = config.get
    monkeypatch.setattr(
        config,
        "get",
        lambda sec, key=None, default=None: (
            values.get(key, default) if sec == "auth" else real_get(sec, key, default=default)
        ),
    )
    app = Flask(__name__)
    app.config["SECRET_KEY"] = "k"
    auth.register_auth(app)

    @app.route("/api/thing")
    def thing():
        return {"ok": True}

    c = app.test_client()
    c.values = values
    return c


def test_five_wrong_passwords_are_free_then_it_waits(client):
    for _ in range(auth.FREE_FAILURES):
        assert client.post("/auth/login", data={"password": "no"}).status_code == 401
    r = client.post("/auth/login", data={"password": "right"})
    assert r.status_code == 429 and int(r.headers["Retry-After"]) >= 1


def test_the_wait_doubles_and_ends(monkeypatch):
    now = 1000.0
    for i in range(auth.FREE_FAILURES + 3):
        auth.note_failure("x", now=now + i)
    last = now + auth.FREE_FAILURES + 2
    assert auth.wait_left("x", now=last) == pytest.approx(8.0)
    assert auth.wait_left("x", now=last + 9) == 0.0
    assert auth.wait_left("x", now=last + auth.FAIL_WINDOW + 1) == 0.0  # forgotten


def test_a_success_clears_the_count(client):
    for _ in range(auth.FREE_FAILURES - 1):
        client.post("/auth/login", data={"password": "no"})
    assert client.post("/auth/login", data={"password": "right"}).status_code == 302
    assert auth.wait_left("127.0.0.1") == 0.0


def test_wrong_tokens_are_braked_too(client):
    for _ in range(auth.FREE_FAILURES):
        assert client.get("/api/thing", headers={"X-API-Token": "bad"}).status_code == 401
    assert client.get("/api/thing", headers={"X-API-Token": "tok"}).status_code == 429


def test_a_password_change_ends_old_sessions(client):
    assert client.post("/auth/login", data={"password": "right"}).status_code == 302
    assert client.get("/api/thing").status_code == 200
    client.values["password"] = "new"
    assert client.get("/api/thing").status_code == 401


def test_a_wrong_password_gets_401_and_no_session(client):
    r = client.post("/auth/login", data={"password": "no"})
    assert r.status_code == 401
    assert client.get("/api/thing").status_code == 401


def test_logout_ends_the_session(client):
    assert client.post("/auth/login", data={"password": "right"}).status_code == 302
    assert client.get("/api/thing").status_code == 200
    assert client.post("/auth/logout").status_code == 302
    assert client.get("/api/thing").status_code == 401


def test_the_real_apps_cookie(tmp_path, monkeypatch):
    """Through create_app: the install's own cookie name and SameSite=Lax."""
    from whisper_tm.app import create_app

    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        f"directories:\n  media: {tmp_path}\nserver:\n  secret_key: k1\n"
        f"auth:\n  enabled: true\n  password: pw\n",
        encoding="utf-8",
    )
    config.load(str(cfg))
    app = create_app(str(cfg))
    c = app.test_client()
    r = c.post("/auth/login", data={"password": "pw"})
    assert r.status_code == 302
    cookie = r.headers.get("Set-Cookie", "")
    assert cookie.startswith(app.config["SESSION_COOKIE_NAME"] + "=")
    assert "HttpOnly" in cookie and "SameSite=Lax" in cookie
    config.load(None)


def test_the_cookie_can_be_kept_to_https(tmp_path):
    from whisper_tm.app import create_app

    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        f"directories:\n  media: {tmp_path}\nserver:\n  secure_cookie: true\n", encoding="utf-8"
    )
    config.load(str(cfg))
    assert create_app(str(cfg)).config["SESSION_COOKIE_SECURE"] is True
    config.load(None)


def test_a_new_secret_key_is_private_from_the_start(tmp_path):
    import os
    import stat

    from whisper_tm.app import _load_or_create_secret

    cfg = tmp_path / "config.yaml"
    cfg.write_text(f"directories:\n  media: {tmp_path}\n", encoding="utf-8")
    config.load(str(cfg))
    key = _load_or_create_secret()
    f = tmp_path / "secret_key"
    assert f.read_text(encoding="utf-8").strip() == key
    if os.name == "posix":
        assert stat.S_IMODE(f.stat().st_mode) == 0o600
    config.load(None)


def test_a_login_lasts_its_days_and_is_not_renewed(tmp_path):
    from datetime import timedelta

    from whisper_tm.app import create_app

    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        f"directories:\n  media: {tmp_path}\nauth:\n  session_days: 3\n", encoding="utf-8"
    )
    config.load(str(cfg))
    app = create_app(str(cfg))
    assert app.config["PERMANENT_SESSION_LIFETIME"] == timedelta(days=3)
    assert app.config["SESSION_REFRESH_EACH_REQUEST"] is False
    config.load(None)
