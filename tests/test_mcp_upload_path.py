"""``upload_file(path=…)`` reads a file of THIS machine — an audio or video
file below the allowed roots only, over every transport.

Over MCP-HTTP or OpenAPI a client (or text in a transcript steering an
outside assistant) could name any file the server user can read —
``~/.ssh/id_rsa``, ``config.yaml`` with its API keys — and the tool copied
it into the library, where the app serves it. Over stdio any file was taken,
on the reasoning that the client already is this user; but the text that
steers the client isn't. Now a path must lie below the allowed roots
(``security.allowed_roots``) and be an audio or video file, and the upload
keeps its extension: a key can't come in as ``key.mp4`` either. Anything else
goes as ``content_base64`` — bytes the client already holds.
"""

from __future__ import annotations

import pytest

from tests.test_mcp_server import Peer  # noqa: F401
from whisper_tm import mcp_server as mcp

pytest.importorskip("fastmcp", reason="fastmcp not installed")


@pytest.fixture
def peer(monkeypatch, tmp_path):
    from whisper_tm.config import config

    media = tmp_path / "media"
    media.mkdir()
    monkeypatch.setattr(type(config), "media_dir", property(lambda self: media))
    monkeypatch.setitem(config._data, "security", {"allowed_roots": [str(media)]})
    g = Peer().start()
    g.set_value(
        "/api/upload_chunk",
        {"status": "complete", "filename": "uploads/x.mp4", "display_name": "x.mp4"},
    )
    old = mcp.CONN, mcp.WRITING_ALLOWED
    mcp.CONN = mcp.Connection(g.base_url, time_limit=5)
    mcp.WRITING_ALLOWED = True
    yield g, media
    mcp.CONN, mcp.WRITING_ALLOWED = old
    g.stop()


def _upload(**kw):
    import asyncio

    server = mcp.build_server()
    t = asyncio.run(server.get_tool("upload_file"))
    return asyncio.run(t.run(kw)).structured_content


def _uploads(g):
    return [c for c in g.calls if c["path"] == "/api/upload_chunk"]


def test_a_path_outside_the_roots_is_refused(peer, tmp_path):
    g, _media = peer
    secret = tmp_path / "id_rsa"
    secret.write_text("PRIVATE KEY")
    with pytest.raises(Exception, match="outside the allowed folders"):
        _upload(path=str(secret))
    assert not _uploads(g)


def test_a_path_inside_the_roots_is_taken(peer):
    g, media = peer
    f = media / "clip.mp4"
    f.write_bytes(b"x" * 10)
    assert _upload(path=str(f))["file"] == "uploads/x.mp4"


def test_only_audio_and_video_by_path(peer):
    """Inside the roots too: the config, a key, a script stay where they are."""
    g, media = peer
    f = media / "config.yaml"
    f.write_text("api_key: secret")
    with pytest.raises(Exception, match="not an audio or video file"):
        _upload(path=str(f))
    assert not _uploads(g)


def test_the_upload_keeps_the_files_extension(peer, monkeypatch):
    _g, media = peer
    f = media / "clip.mp4"
    f.write_bytes(b"x" * 10)
    sent = []

    def send_file(path, fields, file):
        sent.append((fields["fileName"], file[0]))
        return {"status": "complete", "filename": "uploads/renamed.mp4"}

    monkeypatch.setattr(mcp.CONN, "send_file", send_file)
    _upload(path=str(f), name="renamed.sh")
    assert sent == [("renamed.mp4", "renamed.mp4")]
