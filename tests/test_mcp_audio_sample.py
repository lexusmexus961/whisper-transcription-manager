"""Audio samples in the MCP server responses.

A model in a chat client such as Open WebUI can compile findings from the
transcripts — but only as quotes, with nothing to listen to. The responses
can carry the clips as well.

Two ways, because clients are at different stages:

* **The audio itself.** MCP has known the ``audio`` content type since
  2025-03-26 (base64 + mimeType), and Open WebUI treats it in
  ``process_tool_result()`` like an image: base64 → file → attachment on
  tool result.
* **A line of text.** ``<audio>…</audio>`` in the middle of the response, right
  at the quote. Open WebUI makes a player from it — but it reads the
  address from the TAG CONTENT, not from ``src``
  (``HTMLToken.svelte``). With ``src=…`` the element would remain empty.
"""

from __future__ import annotations

import asyncio

import pytest

import whisper_tm.mcp_server as mcp


@pytest.fixture
def app_base(monkeypatch):
    monkeypatch.setattr(mcp, "CONN", mcp.Connection("http://127.0.0.1:5077"))
    monkeypatch.setattr(mcp, "SAMPLE_BASE", "http://192.0.2.51:5077")
    return mcp


class TestTheLine:
    def test_the_address_is_in_the_content_not_in_src(self, app_base):
        """The point where it otherwise fails silently: Open WebUI takes
        the text BETWEEN the tags as the source."""
        z = mcp.audio_sample_line("videos/a.mp4", 100)
        assert z.startswith("<audio>\nhttp://"), z
        assert z.endswith("</audio>"), z
        assert "src=" not in z

    def test_it_spans_three_lines(self, app_base):
        """Two layouts fail — a bullet list, and the tag on one line as
        its own paragraph. Both show the raw text in the chat.

        `audio` is not one of the block elements CommonMark knows. An HTML
        block then only arises via rule 7: a complete tag that stands
        ALONE on its line. Measured with `marked` — the same parser Open
        WebUI uses:

            * <audio>URL</audio>       → list,      0 html tokens
            <audio>URL</audio>         → paragraph, 0 html tokens
            <audio> URL </audio>       → html,      1 token → player
                (the third with real line breaks instead of spaces)
        """
        lines = mcp.audio_sample_line("videos/a.mp4", 100).splitlines()
        assert len(lines) == 3, lines
        assert lines[0] == "<audio>"
        assert lines[1].startswith("http://")
        assert lines[2] == "</audio>"

    def test_it_points_at_the_clip_on_demand(self, app_base):
        z = mcp.audio_sample_line("videos/a.mp4", 100)
        assert "/api/clip/audio?file=" in z
        assert "start=98.0" in z, z  # two seconds lead-in
        assert "end=128.0" in z, z  # thirty seconds length

    def test_a_lead_in_never_goes_below_zero(self, app_base):
        assert "start=0.0" in mcp.audio_sample_line("videos/a.mp4", 1)

    def test_paths_with_spaces_are_encoded(self, app_base):
        z = mcp.audio_sample_line("videos/my film.mp4", 10)
        assert "my%20film.mp4" in z, z
        assert " " not in z.split("file=")[1]

    def test_without_a_file_no_line(self, app_base):
        assert mcp.audio_sample_line("", 10) == ""


class TestTheBareAddress:
    """The bullet point where it failed silently.

    The model wrote the line as ``* <audio>…</audio>``, and afterward the
    chat showed raw text with ``&amp;`` instead of a player. At block
    level, ``<audio>…</audio>`` is ONE html token that Open WebUI finds;
    inside a bullet list, Markdown splits the same character sequence into
    ``<audio>``, text and ``</audio>`` — and the rule doesn't apply to any
    of them.

    That's why the address is now also available bare: an ordinary link
    made from it works everywhere, including inside a bullet list.
    """

    def test_the_address_comes_without_tags(self, app_base):
        a = mcp.sample_address("videos/a.mp4", 100)
        assert a.startswith("http://")
        assert "<" not in a and ">" not in a

    def test_line_and_address_point_at_the_same_thing(self, app_base):
        a = mcp.sample_address("videos/a.mp4", 100)
        assert mcp.audio_sample_line("videos/a.mp4", 100) == f"<audio>\n{a}\n</audio>"

    def test_without_a_file_no_address_either(self, app_base):
        assert mcp.sample_address("", 10) == ""


class TestWhereThePlayerPoints:
    """The bug that would look like a broken clip: the MCP server reaches
    the app via 127.0.0.1 because it runs alongside it. But the player
    sits in a chat on an ENTIRELY DIFFERENT machine — there, 127.0.0.1
    leads home, and the app isn't running there."""

    def test_the_configured_base_wins(self, app_base):
        assert mcp._sample_base() == "http://192.0.2.51:5077"

    def test_without_a_setting_loopback_is_replaced(self, monkeypatch):
        monkeypatch.setattr(mcp, "SAMPLE_BASE", "")
        monkeypatch.setattr(mcp, "CONN", mcp.Connection("http://127.0.0.1:5077"))
        monkeypatch.setattr(mcp, "_lan_address", lambda: "192.0.2.51")
        assert mcp._sample_base() == "http://192.0.2.51:5077"

    def test_localhost_and_zero_too(self, monkeypatch):
        monkeypatch.setattr(mcp, "SAMPLE_BASE", "")
        monkeypatch.setattr(mcp, "_lan_address", lambda: "192.0.2.51")
        for address in ("http://localhost:5077", "http://0.0.0.0:5077"):
            monkeypatch.setattr(mcp, "CONN", mcp.Connection(address))
            assert "192.0.2.51" in mcp._sample_base()

    def test_a_foreign_address_is_left_alone(self, monkeypatch):
        monkeypatch.setattr(mcp, "SAMPLE_BASE", "")
        monkeypatch.setattr(mcp, "CONN", mcp.Connection("http://nas.local:5077"))
        assert mcp._sample_base() == "http://nas.local:5077"

    def test_the_lan_address_sends_nothing(self):
        """The socket only connects in the kernel's own bookkeeping —
        this only checks that something usable comes out."""
        a = mcp._lan_address()
        assert a.count(".") == 3, a


class TestAbsolutePaths:
    """Semantic search responds with absolute paths; the app rejects
    absolute foreign paths with 403 (docs/PATHS.md)."""

    def test_the_root_is_stripped(self, monkeypatch, app_base):
        monkeypatch.setattr(mcp, "_media_root", lambda: "/mnt/media")
        assert mcp._rel_path("/mnt/media/videos/a.mp4") == "videos/a.mp4"

    def test_relative_ones_stay_as_they_are(self, monkeypatch, app_base):
        monkeypatch.setattr(mcp, "_media_root", lambda: "/mnt/media")
        assert mcp._rel_path("videos/a.mp4") == "videos/a.mp4"

    def test_a_foreign_path_is_not_mangled(self, monkeypatch, app_base):
        monkeypatch.setattr(mcp, "_media_root", lambda: "/mnt/media")
        assert mcp._rel_path("/elsewhere/a.mp4") == "/elsewhere/a.mp4"


def _tool(name: str, **kw):
    server = mcp.build_server()
    t = asyncio.run(server.get_tool(name))
    return asyncio.run(t.run(kw))


class TestTheClipTool:
    @pytest.fixture
    def cut(self, monkeypatch, app_base):
        asked = {}

        def _raw(path, time_limit=None, **params):
            asked.update({"path": path, **params})
            return b"ID3\x04" + b"\x00" * 500, "audio/mpeg"

        monkeypatch.setattr(mcp.CONN, "raw", _raw)
        monkeypatch.setattr(mcp, "_media_root", lambda: "/mnt/media")
        return asked

    def test_audio_AND_a_line_come_back(self, cut):
        r = _tool("clip", file="videos/a.mp4", start_sec=10, end_sec=25)
        kinds = [c.type for c in r.content]
        assert "audio" in kinds, kinds
        assert "text" in kinds, kinds

    def test_the_audio_is_declared_as_mp3(self, cut):
        r = _tool("clip", file="videos/a.mp4", start_sec=10, end_sec=25)
        audio = [c for c in r.content if c.type == "audio"][0]
        assert audio.mimeType == "audio/mp3", audio.mimeType
        assert audio.data, "no data"

    def test_the_text_carries_the_player_line(self, cut):
        r = _tool("clip", file="videos/a.mp4", start_sec=10, end_sec=25)
        text = [c for c in r.content if c.type == "text"][0].text
        assert "<audio>" in text and "</audio>" in text

    def test_the_clip_is_cut_exactly_like_that(self, cut):
        _tool("clip", file="videos/a.mp4", start_sec=10, end_sec=25)
        assert cut["path"] == "/api/clip/audio"
        assert cut["file"] == "videos/a.mp4"
        assert cut["start"] == "10.000"
        assert cut["end"] == "25.000"

    def test_absolute_paths_are_trimmed(self, cut):
        _tool("clip", file="/mnt/media/videos/a.mp4", start_sec=0, end_sec=5)
        assert cut["file"] == "videos/a.mp4"

    def test_backwards_is_an_error(self, cut):
        with pytest.raises(Exception, match="start_sec"):
            _tool("clip", file="videos/a.mp4", start_sec=30, end_sec=10)

    def test_longer_than_ten_minutes_is_refused(self, cut):
        """Not just on principle: a model that accidentally requests four
        hours would hold up the response for minutes."""
        with pytest.raises(Exception, match="10 minutes"):
            _tool("clip", file="videos/a.mp4", start_sec=0, end_sec=4000)


class TestTheSearchesPassAudioThrough:
    @pytest.fixture
    def search(self, monkeypatch, app_base):
        monkeypatch.setattr(mcp, "_media_root", lambda: "/mnt/media")
        monkeypatch.setattr(
            mcp.CONN,
            "fetch",
            lambda path, **kw: (
                {
                    "total": 1,
                    "files": 1,
                    "truncated": False,
                    "results": [
                        {
                            "filename": "videos/a.mp4",
                            "start": 100.0,
                            "start_fmt": "0:01:40",
                            "text": "hello",
                        }
                    ],
                    "mode": "hybrid",
                }
                if "rag" not in path
                else {
                    "mode": "hybrid",
                    "results": [
                        {
                            "media_path": "/mnt/media/videos/b.mp4",
                            "start_sec": 50.0,
                            "content": "hello",
                            "score": 0.8,
                        }
                    ],
                }
            ),
        )

    def test_without_the_switch_everything_stays_as_before(self, search):
        d = _tool("search", text="hello").structured_content
        assert "player" not in d["hits"][0]

    def test_with_the_switch_a_player_is_attached(self, search):
        d = _tool("search", text="hello", with_audio=True).structured_content
        assert d["hits"][0]["player"].startswith("<audio>")

    def test_and_the_address_just_in_case(self, search):
        """For responses where the line would end up in a bullet list —
        there it never becomes a player."""
        d = _tool("search", text="hello", with_audio=True).structured_content
        assert d["hits"][0]["play_url"].startswith("http://")
        assert "<audio>" not in d["hits"][0]["play_url"]

    def test_the_semantic_search_too(self, search):
        d = _tool("semantic_search", question="hello", with_audio=True).structured_content
        assert "<audio>" in d["hits"][0]["player"]

    def test_and_it_points_at_the_relative_path(self, search):
        """Otherwise the app answers with 403 — absolute foreign paths are
        blocked."""
        d = _tool("semantic_search", question="hello", with_audio=True).structured_content
        assert "file=videos/b.mp4" in d["hits"][0]["player"]
        assert d["hits"][0]["file"] == "videos/b.mp4"


class TestThroughTheOpenApiPath:
    """The OpenAPI path (as set up under Open WebUI's External Tool
    Servers). There are no
    content blocks there — and a `null` would be the worst possible answer.

    Open WebUI has its own rule for this: in `process_tool_result()`,
    every list entry that starts with `data:` gets pulled out of the
    result and attached to the response as a file.
    """

    @pytest.fixture
    def client(self, monkeypatch, app_base):
        pytest.importorskip("starlette")
        from starlette.testclient import TestClient

        def _raw(path, time_limit=None, **params):
            return b"ID3\x04" + b"\x00" * 200, "audio/mpeg"

        monkeypatch.setattr(mcp.CONN, "raw", _raw)
        monkeypatch.setattr(mcp, "_media_root", lambda: "/mnt/media")
        return TestClient(mcp.build_openapi_app(mcp.build_server()), base_url="http://127.0.0.1")

    def test_the_audio_comes_as_a_data_uri(self, client):
        r = client.post("/clip", json={"file": "videos/a.mp4", "start_sec": 10, "end_sec": 20})
        assert r.status_code == 200, r.text[:300]
        d = r.json()
        assert isinstance(d, list), d
        assert any(isinstance(x, str) and x.startswith("data:audio/") for x in d), d

    def test_the_player_line_is_included_too(self, client):
        d = client.post(
            "/clip", json={"file": "videos/a.mp4", "start_sec": 10, "end_sec": 20}
        ).json()
        assert any("<audio>" in x for x in d if isinstance(x, str)), d

    def test_tools_with_data_still_answer_as_an_object(self, client, monkeypatch):
        """The change must not touch the 25 other tools."""
        monkeypatch.setattr(mcp.CONN, "fetch", lambda path, **kw: {"people": []})
        r = client.post("/list_people", json={})
        assert isinstance(r.json(), dict), r.text[:200]


class TestTheGuideTravelsAlong:
    """The usage rule travels with the tool.

    Otherwise it would have to live in every prompt — repeated for every
    question. Now it sits in three places, because a model needs it at
    three different times: in the server instructions (before it picks a
    tool), in the tool description (when calling it) and in the response
    itself (when writing — that's where the decision happens).
    """

    def test_the_server_guide_shows_the_three_line_example(self):
        guide = mcp.build_server().instructions
        assert "<audio>\n" in guide, "no example with line break"
        assert "</audio>" in guide
        assert "with_audio" in guide
        for trap in ("bullet list", "backticks", "one line"):
            assert trap in guide, trap

    def test_the_tool_description_shows_it_too(self):
        import asyncio

        server = mcp.build_server()
        t = asyncio.run(server.get_tool("search"))
        assert "<audio>\n" in t.description
        assert "play_url" in t.description

    def test_the_answer_carries_the_rule(self, monkeypatch, app_base):
        """The most important of the three places: a model reads
        descriptions before selecting, and the response while writing."""
        monkeypatch.setattr(mcp, "_media_root", lambda: "/mnt/media")
        monkeypatch.setattr(
            mcp.CONN,
            "fetch",
            lambda path, **kw: {
                "total": 1,
                "files": 1,
                "results": [{"filename": "videos/a.mp4", "start": 10.0, "text": "x"}],
            },
        )
        d = _tool("search", text="hello", with_audio=True).structured_content
        assert "three lines" in d["audio_usage"]
        assert "play_url" in d["audio_usage"]

    def test_every_hit_not_just_the_pretty_ones(self, monkeypatch, app_base):
        """A model tends to give only some of its findings a player — it
        considers the short quotes too insignificant. Exactly the opposite:
        a short one-liner is the claim a reader is most likely to want to
        hear for themselves."""
        monkeypatch.setattr(mcp, "_media_root", lambda: "/mnt/media")
        monkeypatch.setattr(
            mcp.CONN,
            "fetch",
            lambda path, **kw: {
                "total": 1,
                "files": 1,
                "results": [{"filename": "videos/a.mp4", "start": 10.0, "text": "x"}],
            },
        )
        d = _tool("search", text="hello", with_audio=True).structured_content
        assert "EVERY" in d["audio_usage"], d["audio_usage"]

        guide = mcp.build_server().instructions
        assert "EVERY hit" in guide
        assert "not a selection" in guide

    def test_without_audio_no_lecture(self, monkeypatch, app_base):
        """Whoever doesn't want audio samples shouldn't get instructions
        for them either — every field costs context space."""
        monkeypatch.setattr(
            mcp.CONN, "fetch", lambda path, **kw: {"total": 0, "files": 0, "results": []}
        )
        d = _tool("search", text="hello").structured_content
        assert "audio_usage" not in d
