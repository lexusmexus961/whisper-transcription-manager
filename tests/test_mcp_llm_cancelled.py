"""An LLM task cancelled in the app ends the MCP wait at once.

Previously the wait knew "done" and "error" only, while the LLM queue
also ends a task as "cancelled". A summary cancelled in the UI kept the
MCP client (and the in-app chat) polling until its time limit — and then
said the task "CONTINUES".
"""

from __future__ import annotations

import time

import pytest

from tests.test_mcp_server import Peer  # noqa: F401
from whisper_tm import mcp_server as mcp

pytest.importorskip("fastmcp", reason="fastmcp not installed")


@pytest.fixture
def peer():
    g = Peer().start()
    old = mcp.CONN, mcp.WRITING_ALLOWED, mcp.POLL_SEC, mcp.SEC_PER_MIN
    mcp.CONN = mcp.Connection(g.base_url, time_limit=5)
    mcp.WRITING_ALLOWED = True
    mcp.POLL_SEC, mcp.SEC_PER_MIN = 0.01, 2.0
    yield g
    mcp.CONN, mcp.WRITING_ALLOWED, mcp.POLL_SEC, mcp.SEC_PER_MIN = old
    g.stop()


def _call(name, **kw):
    import asyncio

    server = mcp.build_server()
    t = asyncio.run(server.get_tool(name))
    return asyncio.run(t.run(kw)).structured_content


def test_a_cancelled_task_ends_the_wait(peer):
    peer.set_value("/api/llm/queue/add", {"id": 7})
    peer.set_value("/api/llm/queue/item/7", {"id": 7, "status": "cancelled", "kind": "summary"})
    t0 = time.time()
    with pytest.raises(Exception, match="cancelled"):
        _call("summarize", file="a.mp4", wait=True, timeout_min=1)
    assert time.time() - t0 < 1.5  # not the 2-second "minute"
