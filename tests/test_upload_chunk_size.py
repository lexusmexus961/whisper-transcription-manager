"""The web upload's piece size comes from the server: 32 MB passes
Cloudflare's 100 MB request limit, and a stricter proxy needs one setting."""

import pytest


@pytest.fixture
def client(tmp_path, monkeypatch):
    from whisper_tm.app import create_app
    from whisper_tm.config import config

    cfg = tmp_path / "config.yaml"
    cfg.write_text(f"directories:\n  media: {tmp_path}\n", encoding="utf-8")
    config.load(str(cfg))
    app = create_app(str(cfg))
    return app.test_client()


def test_the_default_is_32_mb(client):
    d = client.get("/api/upload_limits").get_json()
    assert d["chunk_bytes"] == 32 * 1024 * 1024 and d["max_file_gb"] == 100


@pytest.mark.parametrize(
    "chunk,body,want_mb", [(8, 200, 8), (0, 200, 1), (500, 200, 199), ("x", 200, 32)]
)
def test_the_setting_stays_within_a_request(chunk, body, want_mb, monkeypatch):
    from whisper_tm.api import media
    from whisper_tm.config import config

    values = {"upload_chunk_mb": chunk, "max_upload_mb": body}
    monkeypatch.setattr(config, "get", lambda s, k, default=None: values.get(k, default))
    assert media.upload_chunk_bytes() == want_mb * 1024 * 1024


def test_the_page_asks_the_server():
    from pathlib import Path

    js = (Path(__file__).resolve().parent.parent / "whisper_tm/static/js/upload.js").read_text(
        encoding="utf-8"
    )
    assert "/api/upload_limits" in js and "100 * 1024 * 1024" not in js
