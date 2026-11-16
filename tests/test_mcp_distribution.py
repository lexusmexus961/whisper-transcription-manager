"""`search_distribution` must give an answer, not a data dump.

Without a limit, the distribution search tool produced far too much output.

Measured against a large library, before:

    q='hello'        ~170 kB JSON  (500 files × 40 histogram values)
    q='the'          ~180 kB JSON

That's tens of thousands of characters — going unfiltered into a language
model's context, with about 60 kB of that being the histograms alone. Those are meant for
the web UI's heatmap; a model can't do anything with forty raw numbers per
file.

After, same queries:

    q='hello'          ~4 kB
    q='the'            ~4 kB

What comes back instead answers the same question better: the densest
recordings with their hit count AND the spot where the hits are thickest —
i.e. the second you'd listen in at.
"""

from __future__ import annotations

import asyncio
import json

import pytest

import whisper_tm.mcp_server as mcp


def _facets(n_files: int, hits_total: int = 1000, file_count: int | None = None) -> dict:
    """A response like the app's — with a histogram per file."""
    return {
        "total": hits_total,
        "file_count": n_files if file_count is None else file_count,
        "buckets": 12,
        "files": [
            {
                "filename": f"videos/a{i}.mp4",
                "hits": 100 - i,
                "duration": 1200.0,
                "file_dir": "/mnt/media",
                "media_dir": "/mnt/media",
                # peak shifts per file — this lets us verify that the
                # densest point is actually computed, not guessed
                "histogram": [0] * (i % 12) + [9] + [0] * (11 - i % 12),
            }
            for i in range(n_files)
        ],
    }


@pytest.fixture
def peer(monkeypatch):
    """The app responds; what's recorded is WHAT was asked for."""
    asked: dict = {}
    state: dict = {"answer": _facets(50)}

    def _fetch(path, **params):
        asked.clear()
        asked.update({"path": path, **params})
        return state["answer"]

    monkeypatch.setattr(mcp, "CONN", mcp.Connection("http://127.0.0.1:5077"))
    monkeypatch.setattr(mcp.CONN, "fetch", _fetch)
    monkeypatch.setattr(mcp, "SAMPLE_BASE", "http://192.0.2.10:5077")
    monkeypatch.setattr(mcp, "_media_root", lambda: "/mnt/media")
    state["asked"] = asked
    return state


def _call(**kw) -> dict:
    server = mcp.build_server()
    t = asyncio.run(server.get_tool("search_distribution"))
    return asyncio.run(t.run(kw)).structured_content


class TestTheCap:
    def test_twenty_come_by_default(self, peer):
        peer["answer"] = _facets(50)
        d = _call(text="hello")
        assert d["shown"] == 20
        assert len(d["recordings"]) == 20

    def test_the_cap_can_be_raised(self, peer):
        peer["answer"] = _facets(50)
        assert _call(text="hello", limit=50)["shown"] == 50

    def test_but_not_without_bound(self, peer):
        """A hundred recordings is already a lot for one answer."""
        peer["answer"] = _facets(400)
        assert _call(text="hello", limit=9999)["shown"] == 100

    def test_nonsense_values_do_not_topple_it(self, peer):
        peer["answer"] = _facets(50)
        assert _call(text="hello", limit=0)["shown"] == 1
        assert _call(text="hello", limit=-5)["shown"] == 1

    def test_nothing_more_is_fetched_in_the_first_place(self, peer):
        """The cap also saves the server work — it sorts the files by hit
        count anyway."""
        _call(text="hello", limit=5)
        assert peer["asked"]["max_files"] == 5

    def test_fewer_buckets_than_the_heatmap(self, peer):
        """Forty buckets per file were the lion's share of the ~170 kB."""
        _call(text="hello")
        assert peer["asked"]["buckets"] == 12


class TestTheNumbersStayComplete:
    """A cap on the LIST must not falsify the TOTALS — otherwise "too much
    output" would have turned into a wrong result."""

    def test_the_totals_count_everything(self, peer):
        peer["answer"] = _facets(30, hits_total=5000, file_count=2000)
        d = _call(text="hello", limit=5)
        assert d["total_hits"] == 5000
        assert d["files_with_hits"] == 2000
        assert d["shown"] == 5

    def test_what_is_missing_is_stated(self, peer):
        """Otherwise a list of twenty reads like "that's all of it"."""
        peer["answer"] = _facets(30, file_count=2000)
        d = _call(text="hello", limit=20)
        assert d["omitted"] == 2000 - 20
        assert "limit" in d["note"]

    def test_with_a_complete_list_no_hint(self, peer):
        peer["answer"] = _facets(3, file_count=3)
        d = _call(text="hello")
        assert "omitted" not in d
        assert "note" not in d


class TestTheDensestPoint:
    """What replaces the histogram — and is worth more than it."""

    def test_the_position_is_computed(self, peer):
        # file 0: peak in the first of twelve buckets of a 1200-s file
        peer["answer"] = _facets(1)
        d = _call(text="hello")
        assert d["recordings"][0]["busiest_at_sec"] == pytest.approx(50.0)

    def test_it_moves_with_the_peak(self, peer):
        peer["answer"] = _facets(4)
        spots = [r["busiest_at_sec"] for r in _call(text="hello")["recordings"]]
        assert spots == sorted(spots), spots
        assert len(set(spots)) == 4, "all files report the same spot"

    def test_without_hits_in_the_histogram_no_position(self, peer):
        """A made-up second would be worse than none."""
        peer["answer"] = {
            "total": 1,
            "file_count": 1,
            "buckets": 12,
            "files": [{"filename": "a.mp4", "hits": 1, "duration": 0, "histogram": []}],
        }
        assert _call(text="hello")["recordings"][0]["busiest_at_sec"] is None

    def test_the_player_points_at_the_position(self, peer):
        peer["answer"] = _facets(1)
        r = _call(text="hello", with_audio=True)["recordings"][0]
        assert "start=48.0" in r["player"], r["player"]
        assert r["player"].startswith("<audio>\n")


class TestTheHeapIsGone:
    def test_the_histogram_is_no_longer_sent_out(self, peer):
        peer["answer"] = _facets(20)
        raw = json.dumps(_call(text="hello"))
        assert "histogram" not in raw

    def test_nor_the_repeated_roots(self, peer):
        """`media_dir` and `file_dir` used to appear identically 500 times."""
        peer["answer"] = _facets(20)
        raw = json.dumps(_call(text="hello"))
        assert "media_dir" not in raw and "file_dir" not in raw

    def test_the_answer_stays_small(self, peer):
        """The number this was about. Twenty recordings with long paths
        come to about 4 kB against a large library — before it was ~170 kB."""
        peer["answer"] = _facets(200, file_count=2000)
        raw = json.dumps(_call(text="hello"), ensure_ascii=False)
        assert len(raw) < 8000, f"{len(raw)} characters"
