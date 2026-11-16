"""The bundled llama-servers start with a key and the app sends it to them."""

import os
import stat

import pytest

from whisper_tm.core import local_key


@pytest.fixture()
def cfgdir(tmp_path, monkeypatch):
    monkeypatch.setattr(local_key, "key_file", lambda: tmp_path / "llama_api_key")
    monkeypatch.setattr(
        local_key, "_bundled", lambda: {("loopback", 8080), ("loopback", 8081), ("loopback", 8082)}
    )
    return tmp_path


def test_the_key_is_made_once_and_private(cfgdir):
    k = local_key.key()
    assert len(k) >= 40 and local_key.key() == k
    f = cfgdir / "llama_api_key"
    assert stat.S_IMODE(os.stat(f).st_mode) == 0o600


def test_an_edited_key_file_is_used(cfgdir):
    (cfgdir / "llama_api_key").write_text("# mine\nchosen-by-me\n")
    assert local_key.key() == "chosen-by-me"


def test_sent_to_the_bundled_servers_only(cfgdir):
    k = local_key.key()
    for url in (
        "http://127.0.0.1:8080/v1/chat/completions",
        "http://localhost:8081/v1/embeddings",
        "http://127.0.0.1:8082/v1/rerank",
    ):
        assert local_key.headers_for(url) == {"Authorization": f"Bearer {k}"}
    for url in (
        "http://127.0.0.1:11434/api/chat",
        "https://api.example.com/v1/chat/completions",
        "http://10.0.0.5:8080/v1/chat/completions",
    ):
        assert local_key.headers_for(url, {"X": "1"}) == {"X": "1"}


def test_a_configured_key_wins(cfgdir):
    h = local_key.headers_for("http://127.0.0.1:8080/v1", {"Authorization": "Bearer theirs"})
    assert h == {"Authorization": "Bearer theirs"}


def test_off_means_no_key_anywhere(cfgdir, monkeypatch):
    monkeypatch.setattr(local_key, "enabled", lambda: False)
    assert local_key.headers_for("http://127.0.0.1:8080/v1") == {}
    assert local_key.launch_args(None) == []


def test_every_launcher_passes_the_key_file(cfgdir, monkeypatch, tmp_path):
    from whisper_tm.core import llama_server

    monkeypatch.setattr(llama_server, "_known", lambda flag, bin_=None: True)
    args = local_key.launch_args(tmp_path / "llama-server")
    assert args == ["--api-key-file", str(cfgdir / "llama_api_key")]
    monkeypatch.setattr(llama_server, "_known", lambda flag, bin_=None: flag != "--api-key-file")
    assert local_key.launch_args(tmp_path / "llama-server") == []


def test_the_real_bundled_addresses_are_found():
    found = local_key.__dict__["_bundled"]()
    ports = {p for _h, p in found}
    assert {8080, 8081} <= ports
