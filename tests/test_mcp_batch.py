"""Splitting large result sets into batches that are processed in parallel.

A tool that just returns the count lets the LLM hand out the work:
subagent one evaluates the list from one to fifty, subagent two
fifty to a hundred … so huge amounts of data can be processed and
the best items picked out.

**Splitting happens by RECORDING, not by running number.** Two reasons:

* The offset shifts. Sorting is by ``rel_path, start_sec``, and the
  queue keeps writing new transcripts continuously. If a file is added
  during the run that alphabetically precedes the current page, every
  later page shifts by its hits —
  subagent 7 would partly see the same as subagent 6.
* Context beats even distribution. "This recording, these 47
  matches" can be judged; "hits 300 to 350 scattered across eight
  files" is confetti.

The offset still exists — but only WITHIN a file, whose hits
can no longer shift.
"""

from __future__ import annotations

import asyncio

import pytest

import whisper_tm.mcp_server as mcp


def _plan(**kw):
    server = mcp.build_server()
    t = asyncio.run(server.get_tool("search_plan"))
    return asyncio.run(t.run(kw)).structured_content


@pytest.fixture
def facets(monkeypatch):
    """The counterpart responds with a distribution like the real one."""
    asked = {}
    state = {"files": [], "total": 0, "file_count": 0, "files_truncated": False}

    def _fetch(path, **params):
        asked.update({"path": path, **params})
        return state

    monkeypatch.setattr(mcp, "CONN", mcp.Connection("http://127.0.0.1:5077"))
    monkeypatch.setattr(mcp.CONN, "fetch", _fetch)
    state["asked"] = asked
    return state


def _files(*pairs):
    return [{"filename": n, "hits": h} for n, h in pairs]


class TestTheSplit:
    def test_small_files_are_bundled(self, facets):
        facets.update(
            total=60,
            file_count=6,
            files=_files(
                ("a.mp4", 10),
                ("b.mp4", 10),
                ("c.mp4", 10),
                ("d.mp4", 10),
                ("e.mp4", 10),
                ("f.mp4", 10),
            ),
        )
        d = _plan(text="x", per_batch=30)
        assert d["batches_total"] == 2, d["batches"]
        assert [b["hits"] for b in d["batches"]] == [30, 30]
        assert len(d["batches"][0]["calls"]) == 3

    def test_no_file_sits_in_two_batches(self, facets):
        facets.update(
            total=100, file_count=10, files=_files(*[(f"{i}.mp4", 10) for i in range(10)])
        )
        d = _plan(text="x", per_batch=25)
        seen = [c["args"]["file"] for b in d["batches"] for c in b["calls"]]
        assert len(seen) == len(set(seen)), seen

    def test_a_large_file_is_split_within_itself(self, facets):
        """The only case where the offset is needed — and the
        harmless one: it refers to a single file."""
        facets.update(total=120, file_count=1, files=_files(("large.mp4", 120)))
        d = _plan(text="x", per_batch=50)
        assert d["batches_total"] == 3
        offsets = [b["calls"][0]["args"]["offset"] for b in d["batches"]]
        assert offsets == [0, 50, 100]
        assert [b["hits"] for b in d["batches"]] == [50, 50, 20]
        assert all(b["calls"][0]["args"]["file"] == "large.mp4" for b in d["batches"])

    def test_the_batches_cover_every_hit(self, facets):
        facets.update(
            total=137,
            file_count=5,
            files=_files(("a.mp4", 70), ("b.mp4", 30), ("c.mp4", 20), ("d.mp4", 10), ("e.mp4", 7)),
        )
        d = _plan(text="x", per_batch=40)
        assert d["hits_covered"] == 137 == d["total_hits"]

    def test_the_calls_are_ready_to_send(self, facets):
        """The model shouldn't have to calculate anything — that's where
        it miscounts."""
        facets.update(total=10, file_count=1, files=_files(("a.mp4", 10)))
        d = _plan(text="budget", per_batch=50)
        call_it = d["batches"][0]["calls"][0]
        assert call_it["tool"] == "search"
        assert call_it["args"]["text"] == "budget"
        assert call_it["args"]["exact_file"] is True, "otherwise batches overlap"
        assert call_it["args"]["with_audio"] is True


class TestHonesty:
    def test_an_incomplete_plan_says_so(self, facets):
        """The facets cover at most 5,000 files. A plan that pretends
        to be complete is worse than none at all."""
        facets.update(
            total=2500,
            file_count=1200,
            files_truncated=True,
            files=_files(("a.mp4", 20), ("b.mp4", 10)),
        )
        d = _plan(text="x", per_batch=50)
        assert "incomplete" in d
        assert "30 of 2500" in d["incomplete"], d["incomplete"]

    def test_a_complete_plan_stays_quiet(self, facets):
        facets.update(total=30, file_count=2, files=_files(("a.mp4", 20), ("b.mp4", 10)))
        assert "incomplete" not in _plan(text="x")

    def test_omitted_batches_are_counted(self, facets):
        facets.update(
            total=500, file_count=50, files=_files(*[(f"{i}.mp4", 10) for i in range(50)])
        )
        d = _plan(text="x", per_batch=10, max_batches=5)
        assert len(d["batches"]) == 5
        assert d["batches_total"] == 50
        assert d["batches_omitted"] == 45

    def test_it_fetches_every_file_not_just_the_first_500(self, facets):
        """The facet cap is otherwise 500 — for a common word
        that could be 500 of well over a thousand recordings."""
        facets.update(total=0, file_count=0, files=[])
        _plan(text="x")
        assert facets["asked"]["max_files"] == 5000
        assert facets["asked"]["buckets"] == 1, "histogram is dead weight here"


class TestTheOffsetOnTheSearch:
    @pytest.fixture
    def search(self, monkeypatch):
        asked = {}
        monkeypatch.setattr(mcp, "CONN", mcp.Connection("http://127.0.0.1:5077"))
        monkeypatch.setattr(
            mcp.CONN,
            "fetch",
            lambda path, **kw: (
                asked.update(kw)
                or {"total": 500, "files": 12, "offset": kw.get("offset") or 0, "results": []}
            ),
        )
        return asked

    def _search(self, **kw):
        server = mcp.build_server()
        t = asyncio.run(server.get_tool("search"))
        return asyncio.run(t.run(kw)).structured_content

    def test_the_offset_is_passed_through(self, search):
        self._search(text="xy", offset=120)
        assert search["offset"] == 120

    def test_without_an_offset_it_is_not_sent(self, search):
        self._search(text="xy")
        assert search.get("offset") is None

    def test_exact_file_only_with_a_file(self, search):
        self._search(text="xy", exact_file=True)
        assert search.get("file_exact") is None, "pointless without a file"
        self._search(text="xy", file="a.mp4", exact_file=True)
        assert search["file_exact"] == 1

    def test_the_answer_names_its_offset(self, search):
        """A batch must be able to identify itself."""
        d = self._search(text="xy", offset=60)
        assert d["offset"] == 60
