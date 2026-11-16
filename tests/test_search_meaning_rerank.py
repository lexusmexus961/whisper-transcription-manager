"""The meaning list, ranked by relevance (situation search).

The goal: find everything — also obscure situations ("someone is insulted") —
without junk in between. The vector order alone puts only a few relevant
passages among the first ten; a reranker — a cross-encoder that reads the
question and a passage together — puts far more there. So the "Meaning" mode
asks the backend for the ranked, grouped list (`rerank=1`, `group=1`), and
the view has to show what that means:

* the hits in the reranker's order, each with a relevance bar;
* copies of one passage (backups in other folders — often a large share
  of the candidates) folded into their hit — "+3 copies", each
  one shown, played and opened like a hit;
* what scored below the backend's threshold behind "show weaker matches";
* when the reranker could not run: a quiet note, and the similarity order.

Runs the real searchview.js in tests/js/search_harness.js — with the app's
real escaping (state.js), since these lines carry text from the transcripts.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
HARNESS = REPO / "tests" / "js" / "search_harness.js"
JS = REPO / "whisper_tm" / "static" / "js"

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node not available")

PRELUDE = r"""
const A_COPIES = [
    {transcript_key: '/nas/backup/a.mp4', rel_path: 'backup/a.mp4', file_dir: '/nas',
     media_path: '/nas/backup/a.mp4', start_sec: 120.5},
    {transcript_key: '/nas/old/a.mp4', rel_path: 'old/a.mp4', file_dir: '/nas',
     media_path: '/nas/old/a.mp4', start_sec: 121},
    {transcript_key: '/gone/a.mp4', rel_path: null, file_dir: null,
     media_path: '/gone/a.mp4', start_sec: 120.5}];
const R = (name, rel, over) => Object.assign({
    transcript_key: '/nas/v/' + name + '.mp4', rel_path: 'v/' + name + '.mp4', file_dir: '/nas',
    media_path: '/nas/v/' + name + '.mp4', content: 'passage ' + name, speaker: 'Alice',
    start_sec: 30, score: rel, relevance: rel, fused: 0.03, vscore: 0.5, fscore: 3,
    match: 'hybrid', copies: []}, over);
const QUICK = {ok: true, mode: 'vector', elapsed_ms: 4210, reranked: true, grouped: true,
    note: null, min_score: 0.05, candidates: 400,
    results: [R('b', 0.4), R('a', 0.91, {start_sec: 120.5, copies: A_COPIES}),
              R('weak', 0.01), R('unranked', null)]};
const later = (ms) => new Promise(r => setTimeout(r, ms || 30));
async function meaning(answer) {
    A.answerFor('/api/rag/search', answer || QUICK);
    A.el('svQuery').value = 'alice is insulted';
    A.svSetMode('meaning');
    await later();
}
"""


def js(body: str) -> dict:
    source = (
        f"const A = require({str(HARNESS)!r});\n"
        "const output = (o) => console.log('§§' + JSON.stringify(o));\n"
        + PRELUDE
        + f"(async () => {{\n{body}\n}})().catch(e => {{ console.error(e); process.exit(1); }});\n"
    )
    done = subprocess.run(
        ["node", "-e", source],
        capture_output=True,
        text=True,
        timeout=60,
        env={
            "SEARCH_JS": str(JS / "searchview.js"),
            "SEARCH_SMALL": str(JS / "search.js"),
            "SEARCH_STATE": str(JS / "state.js"),
            "PATH": os.environ.get("PATH", ""),
        },
    )
    if done.returncode != 0:
        raise AssertionError(f"node failed:\n{done.stderr}")
    lines = [z for z in done.stdout.splitlines() if z.startswith("§§")]
    assert lines, f"no result:\n{done.stdout}\n{done.stderr}"
    return json.loads(lines[-1][2:])


def test_it_asks_for_the_ranked_grouped_list():
    d = js("""
    await meaning();
    output({u: A.fetched().find(x => x.startsWith('/api/rag/search')) || '',
            words: A.fetched().some(x => x.startsWith('/api/search?'))});
    """)
    for part in ("mode=vector", "k=50", "rerank=1", "group=1", "q=alice%20is%20insulted"):
        assert part in d["u"], (part, d["u"])
    assert not d["words"], "the meaning mode must not run the full-text search"


def test_the_hits_in_the_rerankers_order_with_a_bar():
    d = js("""
    await meaning();
    output({h: A.el('svResults').innerHTML, order: A.svState.hits.map(x => x.text),
            sel: A.svState.selection, total: A.el('svTotal').textContent});
    """)
    h = d["h"]
    assert d["order"] == ["passage a", "passage b"], "not sorted by relevance"
    assert h.index("passage a") < h.index("passage b")
    assert '<b style="width:91.0%"></b></i>91%' in h and "40%" in h
    assert "2 passages by relevance" in h and "from 400 candidates" in h
    assert "3 copies folded in" in h
    assert d["sel"] == 0, "the first hit opens in the reading column"
    assert d["total"] == "4"


def test_weaker_matches_wait_behind_a_button():
    d = js("""
    await meaning();
    const before = A.el('svResults').innerHTML, n0 = A.svState.hits.length;
    A.svQuickWeak();
    const open = A.el('svResults').innerHTML, n1 = A.svState.hits.length;
    A.svQuickWeak();
    output({before, n0, open, n1, closed: A.el('svResults').innerHTML, n2: A.svState.hits.length});
    """)
    assert "passage weak" not in d["before"] and "passage unranked" not in d["before"]
    assert "show weaker matches (2)" in d["before"] and d["n0"] == 2
    assert "passage weak" in d["open"] and "passage unranked" in d["open"]
    assert "hide the weaker matches" in d["open"] and d["n1"] == 4, "they join the arrow keys too"
    assert 'class="sv-row with-score with-rel weak' in d["open"]
    assert "Not ranked" in d["open"], "an unscored one must not look like a 0 % one"
    assert "passage weak" not in d["closed"] and d["n2"] == 2


def test_the_threshold_is_the_backends():
    d = js("""
    const q1 = A.svQuickPrepare(Object.assign({}, QUICK, {min_score: 0.5}));
    const q2 = A.svQuickPrepare(Object.assign({}, QUICK, {min_score: null}));
    output({s1: q1.strong.map(h => h.text), w1: q1.weak.map(h => h.text),
            s2: q2.strong.map(h => h.text), min2: q2.minScore});
    """)
    assert d["s1"] == ["passage a"]
    assert d["w1"] == ["passage b", "passage weak", "passage unranked"]
    assert d["s2"] == ["passage a", "passage b"] and d["min2"] == 0.05


def test_nothing_clear_points_to_find_everything():
    d = js("""
    await meaning(Object.assign({}, QUICK, {results: [R('x', 0.02), R('y', 0.01)]}));
    output({h: A.el('svResults').innerHTML, hits: A.svState.hits.length});
    """)
    assert "No passage clearly matches" in d["h"] and "2 weaker matches below" in d["h"]
    assert "dsStartFromQuery()" in d["h"]
    assert d["hits"] == 0 and "passage x" not in d["h"]


def test_copies_fold_into_their_hit():
    d = js("""
    await meaning();
    const folded = A.el('svResults').innerHTML;
    A.svQuickCopies(0);
    output({folded, open: A.el('svResults').innerHTML});
    """)
    assert "+3 copies" in d["folded"] and "backup/a.mp4" not in d["folded"]
    o = d["open"]
    assert "backup/a.mp4" in o and "old/a.mp4" in o
    assert "svPlay('backup/a.mp4', 120.5, '/nas')" in o
    assert "svGoto('old/a.mp4', 121, '/nas')" in o
    assert "/gone/a.mp4" in o and "not in the library" in o
    assert "svSelectCopy(0, 2)" not in o, "a copy that is gone must not be clickable"


def test_a_copy_reads_like_a_hit():
    d = js("""
    await meaning();
    A.svQuickCopies(0);
    A.clearFetched();
    await A.svSelectCopy(0, 1);
    const read = A.el('svRead').innerHTML;
    const shown = {copy: A.el('svC0_1').classList.contains('active'),
                   row: A.el('svZ0').classList.contains('active'),
                   sel: A.svState.selection, copySel: A.svState.copySel};
    await A.svMoveSelection(1);            // the arrow keys go on from its hit
    output({read, shown, ctx: A.fetched().find(u => u.startsWith('/api/search/context')) || '',
            after: {copy: A.el('svC0_1').classList.contains('active'),
                    sel: A.svState.selection, copySel: A.svState.copySel}});
    """)
    assert "svPlay('old/a.mp4', 121, '/nas')" in d["read"] and "play from 2:01" in d["read"]
    assert "file=old%2Fa.mp4" in d["ctx"] and "start=121" in d["ctx"]
    assert d["shown"] == {"copy": True, "row": False, "sel": 0, "copySel": "svC0_1"}
    assert d["after"] == {"copy": False, "sel": 1, "copySel": ""}


def test_folding_a_shown_copy_hands_the_reading_column_back_to_its_hit():
    d = js("""
    await meaning();
    A.svQuickCopies(0);
    await A.svSelectCopy(0, 0);
    A.svQuickCopies(0);
    await later();
    output({read: A.el('svRead').innerHTML, copySel: A.svState.copySel, sel: A.svState.selection});
    """)
    assert d["copySel"] == "" and d["sel"] == 0
    assert "svPlay('v/a.mp4', 120.5, '/nas')" in d["read"]


def test_without_the_reranker_a_quiet_note_and_the_fused_order():
    d = js("""
    await meaning({ok: true, reranked: false, grouped: true, min_score: 0.05, elapsed_ms: 300,
        note: 'Reranker busy (the GPU is in use) — sorted by meaning and wording',
        results: [R('a', null, {score: 0.0325, vscore: 0.61}),
                  R('b', null, {score: 0.031, vscore: null, match: 'fts'}),
                  R('c', null, {score: 0.03, vscore: 0.12})]});
    output({h: A.el('svResults').innerHTML, n: A.svState.hits.length});
    """)
    h = d["h"]
    assert "Reranker busy (the GPU is in use) — sorted by meaning and wording" in h
    assert "the 3 closest passages" in h
    assert "show weaker matches" not in h and d["n"] == 3, "nothing to hide without scores"
    assert "sv-z-rel" not in h, "no relevance bars without the reranker"
    assert ">0.61<" in h and ">0.12<" in h
    assert "0.03" not in h, "the fusion's value is no similarity"
    assert ">words<" in h, "a passage found by its words says so"


def test_a_ranked_list_from_fewer_candidates_says_so():
    """Ranked — but the meaning half failed and only full-text candidates
    were there: the server's note shows next to the bars."""
    d = js("""
    await meaning(Object.assign({}, QUICK, {
        note: 'Meaning search failed (embedding server down) — full-text candidates only'}));
    output({h: A.el('svResults').innerHTML});
    """)
    assert "full-text candidates only" in d["h"] and "passages by relevance" in d["h"]


def test_an_older_server_keeps_its_similarity_column():
    """Without `relevance` on its hits the server is the old one: its
    `score` is the cosine, even next to an empty `vscore`."""
    d = js("""
    const q = A.svQuickPrepare({ok: true, results: [
        {content: 'x', start_sec: 3, score: 0.5231, vscore: null, rel_path: 'b.mp4', file_dir: '/m'}]});
    output({sim: q.strong[0].similarity, known: q.known, reranked: q.reranked});
    """)
    assert d["sim"] == 0.5231 and d["known"] is False and d["reranked"] is False


def test_everything_from_the_transcripts_is_escaped():
    d = js("""
    const evil = '<img src=x onerror=alert(1)>';
    await meaning(Object.assign({}, QUICK, {note: evil, results: [
        R('x', 0.9, {content: evil, speaker: evil, rel_path: "v/it's" + evil + ".mp4",
                     copies: [{rel_path: 'b/' + evil, file_dir: "/n'as", start_sec: 1}]})]}));
    A.svQuickCopies(0);
    const h = A.el('svResults').innerHTML + A.el('svRead').innerHTML;
    output({h, raw: h.includes('<img')});
    """)
    assert d["raw"] is False, "markup from a transcript got into the page"
    assert "&lt;img src=x onerror=alert(1)&gt;" in d["h"]
    assert "svGoto('v/it\\x27s&lt;img" in d["h"], "the path is escaped for its onclick"
    assert "'/n\\x27as'" in d["h"]


def test_a_detour_to_the_files_tab_keeps_the_list_working():
    d = js("""
    await meaning();
    A.setAnswer({total: 5, files: 0, mode: 'files', results: [{filename: 'x.mp4', file_dir: '/m'}],
                 files: []});
    A.svTab('files');
    await later();
    A.svTab('text');
    await later();
    const q = !!A.quick();
    A.svQuickCopies(0);
    output({q, h: A.el('svResults').innerHTML, hits: A.svState.hits.length,
            asked: A.fetched().filter(u => u.startsWith('/api/rag/search') && u.includes('rerank=1')).length});
    """)
    assert d["q"], "coming back, the list's copies and weaker matches were dead"
    assert "backup/a.mp4" in d["h"] and d["hits"] == 2
    assert d["asked"] == 1, "coming back must not search again"


def test_an_answer_that_lands_while_another_tab_is_shown_is_kept():
    """The ranked list takes seconds. Whoever moves on to Ask or the map
    meanwhile must find the list when coming back — not the "running …"
    line the tab switch remembered, and not a second run of the search."""
    d = js("""
    A.answerFor('/api/rag/search', QUICK);
    A.el('svQuery').value = 'alice is insulted';
    A.svSetMode('meaning');                // the request goes out …
    A.svTab('chat');                       // … and the user moves on before it answers
    await later();
    A.svTab('text');
    await later();
    output({h: A.el('svResults').innerHTML, hits: A.svState.hits.length,
            asked: A.fetched().filter(u => u.startsWith('/api/rag/search') && u.includes('rerank=1')).length});
    """)
    assert "passage a" in d["h"] and "running" not in d["h"]
    assert d["hits"] == 2
    assert d["asked"] == 1


def test_an_answer_dropped_by_the_files_tab_is_asked_for_again():
    """The files tab runs a search of its own — the meaning list's answer
    arriving after it is dropped. Coming back, the remembered "running …"
    line must not stand forever: the search runs again."""
    d = js("""
    A.setAnswer({total: 5, files: 0, mode: 'files', results: [{filename: 'x.mp4', file_dir: '/m'}],
                 files: []});
    A.answerFor('/api/rag/search', QUICK);
    A.el('svQuery').value = 'alice is insulted';
    A.svSetMode('meaning');                // the request goes out …
    A.svTab('files');                      // … and the files tab's search drops its answer
    await later();
    A.svTab('text');
    await later();
    output({h: A.el('svResults').innerHTML,
            asked: A.fetched().filter(u => u.startsWith('/api/rag/search') && u.includes('rerank=1')).length});
    """)
    assert "running" not in d["h"] and "passage a" in d["h"]
    assert d["asked"] == 2


def test_the_fused_list_first_then_the_ranked_one():
    """Ranking the candidate passages takes the reranker several seconds:
    the fused list shows at once with "ranking by relevance …", and the
    ranked answer replaces it — without taking the reader away from the
    passage they opened meanwhile."""
    d = js("""
    const FUSED = Object.assign({}, QUICK, {reranked: false, note: null,
        results: [R('b', null), R('a', null, {copies: A_COPIES})]});
    A.answerFor('rerank=1', QUICK);
    A.answerFor('/api/rag/search', FUSED);
    A.el('svQuery').value = 'alice is insulted';
    A.svSetMode('meaning');
    await later();
    output({urls: A.fetched().filter(u => u.startsWith('/api/rag/search')),
            h: A.el('svResults').innerHTML, first: A.svState.hits[0] && A.svState.hits[0].text});
    """)
    assert len(d["urls"]) == 2 and sum("rerank=1" in u for u in d["urls"]) == 1
    assert all("group=1" in u for u in d["urls"])
    assert "passages by relevance" in d["h"] and "ranking by relevance" not in d["h"]
    assert d["first"] == "passage a", "the ranked order replaced the fused one"


def test_the_automatic_pick_follows_the_ranking():
    """Previously, the reading column kept the fused list's first
    passage while the ranked list marked another one as selected."""
    d = js("""
    let release;
    const late = new Promise(r => { release = r; });
    const FUSED = Object.assign({}, QUICK, {reranked: false, note: null,
        results: [R('b', null), R('a', null)]});
    A.answerFor('rerank=1', late);
    A.answerFor('/api/rag/search', FUSED);
    A.el('svQuery').value = 'alice is insulted';
    A.svSetMode('meaning');
    await later();
    const firstRead = A.el('svRead').innerHTML;
    release(QUICK);                    // ranked: "a" (0.91) before "b" (0.4)
    await later(); await later();
    output({firstRead, read: A.el('svRead').innerHTML, sel: A.svState.selection,
            marked: A.el('svZ0').classList.contains('active')});
    """)
    assert "v/b.mp4" in d["firstRead"]
    assert "v/a.mp4" in d["read"] and d["sel"] == 0 and d["marked"]


def test_a_passage_the_reader_picked_stays_picked():
    d = js("""
    let release;
    const late = new Promise(r => { release = r; });
    const FUSED = Object.assign({}, QUICK, {reranked: false, note: null,
        results: [R('a', null), R('b', null)]});
    A.answerFor('rerank=1', late);
    A.answerFor('/api/rag/search', FUSED);
    A.el('svQuery').value = 'alice is insulted';
    A.svSetMode('meaning');
    await later();
    await A.svSelect(1);              // the reader opens "b"
    release(QUICK);                   // ranked: "a" first, "b" second
    await later(); await later();
    output({read: A.el('svRead').innerHTML, sel: A.svState.selection,
            marked: A.el('svZ1').classList.contains('active'),
            top: A.el('svZ0').classList.contains('active')});
    """)
    assert "v/b.mp4" in d["read"], "the reading column keeps the reader's passage"
    assert d["sel"] == 1 and d["marked"] and not d["top"]


def test_while_ranking_the_fused_list_says_so():
    d = js("""
    const FUSED = Object.assign({}, QUICK, {reranked: false, note: null,
        results: [R('b', null), R('a', null)]});
    A.answerFor('rerank=1', new Promise(() => {}));    // the reranker still reads
    A.answerFor('/api/rag/search', FUSED);
    A.el('svQuery').value = 'alice is insulted';
    A.svSetMode('meaning');
    await later();
    output({h: A.el('svResults').innerHTML, hits: A.svState.hits.length});
    """)
    assert "ranking by relevance" in d["h"] and "Reranker busy" not in d["h"]
    assert d["hits"] == 2


def test_a_failed_ranking_keeps_the_fused_list():
    d = js("""
    const FUSED = Object.assign({}, QUICK, {reranked: false, note: null,
        results: [R('b', null), R('a', null)]});
    A.answerFor('rerank=1', {ok: false, error: 'Reranker failed (boom)'});
    A.answerFor('/api/rag/search', FUSED);
    A.el('svQuery').value = 'alice is insulted';
    A.svSetMode('meaning');
    await later();
    output({h: A.el('svResults').innerHTML, hits: A.svState.hits.length});
    """)
    assert "Reranker failed (boom)" in d["h"] and d["hits"] == 2
    assert "not available" not in d["h"]


def test_the_view_switch_keeps_the_meaning_list():
    d = js("""
    await meaning();
    A.svToggleView();
    const h = A.el('svResults').innerHTML;
    A.svToggleView();
    output({h, back: A.el('svResults').innerHTML});
    """)
    assert "sv-z-rel" in d["h"] and "No recordings" not in d["h"]
    assert "sv-z-rel" in d["back"]


def test_the_helpers():
    d = js("""
    output({pct: [0.004, 0.91, 0, 1.7, null, 'x'].map(A.svPct),
            copies: A.svCopyList({rel_path: 'v/a.mp4', file_dir: '/nas', media_path: '/nas/v/a.mp4',
                                  start_sec: 10, copies: [
        {rel_path: 'v/a.mp4', file_dir: '/nas', media_path: '/nas/v/a.mp4', start_sec: 10},
        {rel_path: 'v/a.mp4', file_dir: '/backup', media_path: '/backup/v/a.mp4', start_sec: 10},
        {rel_path: 'b/a.mp4', file_dir: '/nas', media_path: '/nas/b/a.mp4', start_sec: 10},
        {rel_path: 'b/a.mp4', file_dir: '/nas', media_path: '/nas/b/a.mp4', start_sec: 10},
        {rel_path: null, media_path: '/gone/a.mp4', start_sec: 10},
        {rel_path: 'c/a.mp4', start_sec: 12}]})});
    """)
    assert d["pct"] == ["0.4%", "91%", "0%", "100%", "–", "–"]
    got = [(c["filename"], c["file_dir"], c["start"], c["dead"]) for c in d["copies"]]
    assert got == [
        ("v/a.mp4", "/backup", 10, False),  # same path, another root: a copy
        ("b/a.mp4", "/nas", 10, False),  # listed twice: once
        ("", "/nas", 10, True),  # no longer in the library
        ("c/a.mp4", "/nas", 12, False),
    ]  # no root of its own: the hit's
    assert d["copies"][2]["label"] == "/gone/a.mp4"


def test_the_deep_search_tab_is_wired_into_the_workspace():
    """Its tab hides the query bar (it has a field of its own) and asks the
    server only while it is in view: shown → dsOpen, left or closed →
    dsLeave, reopened on it → dsOpen again. A NEW query from the top bar
    goes where it belongs (svRouteQuery) — a few words to the text."""
    d = js("""
    const calls = [];
    globalThis.dsOpen = () => calls.push('open');
    globalThis.dsLeave = () => calls.push('leave');
    A.svTab('deep');
    const bar = A.el('svQueryBar').style.display, filters = A.el('svFilterBar').style.display;
    A.svTab('text');
    A.el('svOverlay').classList.add('visible');
    A.svTab('deep');
    A.closeSearchView();
    A.openSearchView('');
    await later();
    const back = A.svState.tab;
    A.closeSearchView();
    A.openSearchView('x');
    await later();
    output({calls, bar, filters, tab: back, routed: A.svState.tab});
    """)
    assert d["calls"] == ["open", "leave", "open", "leave", "open", "leave"]
    assert d["bar"] == "none" and d["filters"] == "none"
    assert d["tab"] == "deep" and d["routed"] == "text"


def test_a_hit_carries_the_name_the_voice_library_knows():
    """rag/speakers.py: the server sends the label's name as speaker_name and
    names the speakers in the text — the hit shows the name."""
    d = js("""
    const named = Object.assign({}, QUICK, {results: [
        R('a', 0.9, {speaker: 'SPEAKER_04', speaker_name: 'Alice',
                     content: '[Alice] what is this about \\n[SPEAKER_02] no idea'}),
        R('b', 0.5, {speaker: 'SPEAKER_01', speaker_name: null})]});
    await meaning(named);
    output({sp: A.svState.hits.map(x => x.speaker), h: A.el('svResults').innerHTML});
    """)
    assert d["sp"] == ["Alice", "SPEAKER_01"]
    assert "[Alice] what is this" in d["h"]


def test_the_head_says_whose_voice_brought_passages():
    d = js("""
    await meaning(Object.assign({}, QUICK, {voices: {'Alice': 2000,'Nobody': 0, '<b>x</b>': 3}}));
    output({h: A.el('svResults').innerHTML});
    """)
    assert "with the 2,000 passages Alice speaks in" in d["h"]
    assert "Nobody" not in d["h"]
    assert "&lt;b&gt;x&lt;/b&gt; speaks in" in d["h"] and "<b>x</b>" not in d["h"]
