"""The large search view — and why it looked "broken" when it wasn't.

**The finding.** A search for "garden": the small search finds 400 results
in 250 files (fuzzy matches included), the large one exactly one — and shows:

    No results in spoken text — but 1 file(s) whose name matches
    "videos/Archive/meeting_2024-03-01.mp4"

So the large search looked broken: it found one thing while the small one
found more.

It was not the API. Replicated at the endpoint:

    q=garden                            → 400 hits
    q=garden & file=…meeting_2024-03-01… →   0 hits

There were TWO errors in the view, and only together do they paint the picture:

1. **The pinning survived closing.** `svState.pinned` lives in the module.
   `openSearchView()` took file filter and speaker fresh from the small
   panel — except the pinned file. Whoever closed the search, later
   reopened it and typed something new, searched unnoticed further in a
   file they had clicked on at some point.

2. **The fallback path misused the filter as the search term.** If the
   search finds nothing in spoken text, it interprets the term as a filename.
   The line for it was

       const name = svState.pinned || svState.file || svState.query;

   so: filter first, search term last. With a pinned file, the
   fallback searched for files whose name matched THIS file — and found
   exactly it. The message looked like a result, but was only
   the filter finding itself.

**Why no endpoint test could find this.** The API answered every
question correctly; the view posed the wrong one. That's why since this
finding there is a JS harness for the search view (`tests/js/search_harness.js`) —
it had not had a single JS test until then.
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
SEARCH_JS = REPO / "whisper_tm" / "static" / "js" / "searchview.js"
SEARCH_SMALL = SEARCH_JS.parent / "search.js"

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node not available")


def js(body: str) -> dict:
    source = (
        f"const A = require({str(HARNESS)!r});\n"
        "const output = (o) => console.log('§§' + JSON.stringify(o));\n"
        f"{body}\n"
    )
    done = subprocess.run(
        ["node", "-e", source],
        capture_output=True,
        text=True,
        timeout=60,
        env={
            "SEARCH_JS": str(SEARCH_JS),
            "SEARCH_SMALL": str(SEARCH_SMALL),
            "PATH": os.environ.get("PATH", ""),
        },
    )
    if done.returncode != 0:
        raise AssertionError(f"node failed:\n{done.stderr}")
    lines = [z for z in done.stdout.splitlines() if z.startswith("§§")]
    assert lines, f"no result:\n{done.stdout}\n{done.stderr}"
    return json.loads(lines[-1][2:])


class TestPinningDoesNotSurvive:
    def test_opening_clears_old_pin(self):
        """The silent survivor.

        All other filters are taken fresh on opening; the
        pinning was the exception — and the only one you don't see.
        """
        d = js("""
        A.svState.pinned = 'videos/Archive/old_value.mp4';
        A.openSearchView('garden');
        output({pinned: A.svState.pinned, query: A.el('svQuery').value});
        """)
        assert d["pinned"] == "", "a file from before is still filtering the new search"
        assert d["query"] == "garden"

    def test_it_persists_within_a_session(self):
        """It should do something: whoever clicks a file wants to
        search in it — as long as the view is open."""
        d = js("""
        A.svPinFile('videos/a.mp4');
        const after1 = A.svState.pinned;
        A.svRun();
        output({after1, after2: A.svState.pinned});
        """)
        assert d["after1"] == "videos/a.mp4"
        assert d["after2"] == "videos/a.mp4"

    def test_second_click_clears_it(self):
        d = js("""
        A.svPinFile('videos/a.mp4');
        A.svPinFile('videos/a.mp4');
        output({pinned: A.svState.pinned});
        """)
        assert d["pinned"] == ""


class TestFallbackPath:
    """If the search finds nothing in spoken text, it interprets the term
    as a filename. This is useful — as long as it takes the TERM."""

    def test_without_filter_it_searches_the_term(self):
        d = js("""
        A.svState.query = 'vacation';
        A.svState.pinned = ''; A.svState.file = '';
        A.clearFetched();
        A.svFileFallback(A.svState.seq).then(() => {
            output({urls: A.fetched()});
        });
        """)
        assert d["urls"], "no search was performed"
        assert "file=vacation" in d["urls"][0], d["urls"]

    def test_with_filter_it_does_NOT_search_the_filter(self):
        """The core of the finding.

        Previously `file=<the pinned path>` stood here — the query searched
        for the file that was already filtering, and reported it as
        a hit.
        """
        d = js("""
        A.svState.query = 'garden';
        A.svState.pinned = 'videos/Archive/meeting_2024-03-01.mp4';
        A.clearFetched();
        A.svFileFallback(A.svState.seq).then(() => {
            output({urls: A.fetched(),
                     text: A.el('svResults').innerHTML});
        });
        """)
        assert not d["urls"], f"despite filter being set, a search was performed: {d['urls']}"
        # And the old misleading phrasing must not come back:
        # It read like a HIT, but was the filter finding itself.
        assert "whose" not in d["text"] and "matches" not in d["text"], d["text"]

    def test_and_says_why(self):
        """An empty result without explanation sends the user to
        search for an error that doesn't exist."""
        d = js("""
        A.svState.query = 'garden';
        A.svState.pinned = 'videos/Archive/one_of.mp4';
        A.svFileFallback(A.svState.seq).then(() => {
            output({text: A.el('svResults').innerHTML});
        });
        """)
        t = d["text"]
        assert "garden" in t, "the search term is not named"
        assert "one_of.mp4" in t, "the filter is not named"
        assert "restricted" in t
        assert "svClearFilters()" in t, "no way out offered"


class TestTheQuery:
    def test_the_filter_goes_along_as_filter(self):
        """So there's no misunderstanding: The file filter BELONGS in the
        normal search — just as a filter, not as a search term."""
        d = js("""
        A.svState.query = 'hello';
        A.svState.pinned = 'videos/x.mp4';
        output({p: A.svParams()});
        """)
        assert "q=hello" in d["p"]
        assert "file=videos%2Fx.mp4" in d["p"]

    def test_a_pinned_file_beats_the_input_field(self):
        """Otherwise sidebar and text field would contradict each other."""
        d = js("""
        A.svState.query = 'hello';
        A.svState.file = 'from-the-field';
        A.svState.pinned = 'from-the-list';
        output({p: A.svParams()});
        """)
        assert "file=from-the-list" in d["p"]
        assert "from-the-field" not in d["p"]


class TestTheValueFilters:
    """The advanced filters go along as query parameters to the server.

    Here an error is particularly insidious: The search then simply provides
    *different* results, and no one notices they're the wrong ones. There is
    no error message to hang it on.
    """

    def test_the_duration_goes_out_in_SECONDS(self):
        """The UI asks for MINUTES, the server calculates in seconds.

        If the conversion was missing, "from 5 minutes" would filter "from 5
        seconds" — so practically not at all. The hit list would look
        completely plausible.
        """
        d = js("""
        A.el('svDurationMin').value = '5';
        A.el('svDurationMax').value = '90';
        output({p: A.svAdvParams()});
        """)
        assert "duration_min=300" in d["p"], d["p"]
        assert "duration_max=5400" in d["p"], d["p"]

    def test_empty_fields_are_not_sent_at_all(self):
        """Sending an empty field as `duration_min=` would mean giving the
        server a filter the user never set."""
        d = js("output({p: A.svAdvParams()});")
        assert d["p"] == ""

    def test_date_and_word_count_pass_through_unchanged(self):
        d = js("""
        A.el('svDateFrom').value = '2024-01-01';
        A.el('svWordMin').value = '100';
        output({p: A.svAdvParams()});
        """)
        assert "date_from=2024-01-01" in d["p"]
        assert "word_count_min=100" in d["p"]

    def test_sorting_always_carries_a_direction(self):
        """Without a direction the server sorts by its own default — the
        view showed a different order than the one in the field
        implied."""
        d = js("""
        A.el('svSort').value = 'duration';
        output({p: A.svAdvParams()});
        """)
        assert "sort=duration" in d["p"] and "direction=desc" in d["p"], d["p"]

    def test_a_value_filter_counts_as_a_filter(self):
        """`svHasFilter` decides whether to search at all without a search term.
        If a value filter didn't count, the view would stay at
        "duration from 5 min" without a term simply empty."""
        d = js("""
        const without = A.svHasFilter();
        A.el('svDurationMin').value = '5';
        output({without, with_filter: A.svHasFilter()});
        """)
        assert d["without"] is False
        assert d["with_filter"] is True


class TestReloading:
    """Paging via `offset`. An error shows up as duplicate or
    missing hits — not as a crash."""

    def test_offset_grows_by_loaded(self):
        d = js("""
        A.setAnswer({total: 500, files: 3,
                         results: Array.from({length: 200}, (_, i) => ({
                             filename: 'a.mp4', start: i, text: 'x',
                             file_dir: '/m'}))});
        A.svState.query = 'hello'; A.svState.offset = 0;
        A.svState.done = false; A.svState.loading = false;
        A.svLoadMore(true, A.svState.seq).then(() => {
            output({offset: A.svState.offset, done: A.svState.done,
                     total: A.svState.total});
        });
        """)
        assert d["offset"] == 200
        assert d["total"] == 500
        assert d["done"] is False, "there are still 300 hits left"

    def test_at_end_does_not_page_further(self):
        d = js("""
        A.setAnswer({total: 5, files: 1,
                         results: Array.from({length: 5}, (_, i) => ({
                             filename: 'a.mp4', start: i, text: 'x',
                             file_dir: '/m'}))});
        A.svState.query = 'hello'; A.svState.offset = 0;
        A.svState.done = false; A.svState.loading = false;
        A.svLoadMore(true, A.svState.seq).then(() => {
            output({offset: A.svState.offset, done: A.svState.done});
        });
        """)
        assert d["offset"] == 5
        assert d["done"] is True

    def test_lock_is_ALWAYS_released(self):
        """The comment in the code names the reason: If `loading` stayed set,
        the results list would never reload — while the sidebar
        kept updating and hid it well.

        Tested is the unpleasant case: the response comes, but is
        outdated (a new search is already running).
        """
        d = js("""
        A.setAnswer({total: 1, files: 1,
                         results: [{filename: 'a.mp4', start: 0, text: 'x',
                                    file_dir: '/m'}]});
        A.svState.query = 'hello'; A.svState.offset = 0;
        A.svState.done = false; A.svState.loading = false;
        const oldSeq = A.svState.seq;
        const p = A.svLoadMore(true, oldSeq);
        A.svState.seq++;                       // "meanwhile searched anew"
        p.then(() => output({loading: A.svState.loading}));
        """)
        assert d["loading"] is False, "the lock stayed set — the list never reloads"

    def test_without_term_and_without_filter_does_not_search(self):
        """Otherwise an empty search field pulls the entire library."""
        d = js("""
        A.svState.query = ''; A.svState.pinned = ''; A.svState.file = '';
        A.svState.done = false; A.svState.loading = false;
        A.clearFetched();
        A.svLoadMore(true, A.svState.seq).then(() => {
            output({urls: A.fetched()});
        });
        """)
        assert d["urls"] == []


class TestHeader:
    def test_file_count_comes_from_facets(self):
        """The reason is in the code: `data.files` counts only the files of the
        loaded PAGE. With 4,500 hits it would otherwise show "2 files"
        there, because by chance two were on page 1."""
        d = js("""
        A.svState.fileCount = 250;
        A.svState.offset = 200;
        A.svSetStatus(400, 2);          // 2 = only this page
        output({total: A.el('svTotal').textContent,
                 files: A.el('svFiles').textContent,
                 shown: A.el('svShown').textContent});
        """)
        assert d["files"] == "250", "the page count instead of the real one"
        assert d["total"] == "400"
        assert d["shown"] == "200"

    def test_without_facets_page_count_remains(self):
        """Better a cautious number than none — as long as the facets
        are still loading."""
        d = js("""
        A.svState.fileCount = null;
        A.svSetStatus(10, 3);
        output({files: A.el('svFiles').textContent});
        """)
        assert d["files"] == "3"


class TestFileBlockNoLongerDisplaces:
    """**The finding:** the header for the files was so big that no
    transcripts were visible anymore. A query matched 40 filenames and
    thousands of text passages. Measured in the browser: the file block was
    taller than the visible window — and it also sat as a
    THIRD sibling in `.sv-body`, a flex container with row direction. It
    took not only height from the hit list, but also width: of a wide
    window, only a narrow strip remained on the right edge, in which the
    spoken text broke letter by letter.

    With the collapse, the result looked like "the transcripts are gone".
    """

    BODY = """
    A.setAnswer({total: 12000, files: 400,
        results: [{filename: 'a.mp4', start: 0, text: 'x', file_dir: '/m'}],
        files: Array.from({length: 40}, (_, i) => ({
            filename: 'videos/v' + i + '.mp4', display_name: 'v' + i,
            title: 'Title ' + i, duration_sec: 60, is_video: true,
            has_transcript: true, file_dir: '/m'}))});
    A.answerFor('/api/rag/search', {ok: true, results: []});
    A.svState.query = '2024'; A.svState.offset = 0;
    A.svState.done = false; A.svState.loading = false;
    """

    def test_the_block_lands_in_the_hit_list(self):
        """Not beside it. That's the width part of the bug.

        It can be checked at all with this harness because the old version
        called `res.parentNode.insertBefore(...)` — and there is no
        `parentNode` here, so the call would have stood out with an
        exception.
        """
        d = js(
            self.BODY
            + """
        A.svLoadMore(true, A.svState.seq).then(() => {
            const h = A.el('svResults').innerHTML;
            output({in_result: h.includes('sv-files'),
                     lines: (h.match(/sv-d-item/g) || []).length});
        });
        """
        )
        assert d["in_result"], "the file block is not in the hit list"

    def test_only_three_lines_are_shown_open(self):
        d = js(
            self.BODY
            + """
        A.svLoadMore(true, A.svState.seq).then(() => {
            const h = A.el('svResults').innerHTML;
            output({lines: (h.match(/sv-d-item/g) || []).length,
                     button: h.includes('show all 40')});
        });
        """
        )
        assert d["lines"] == 3, "the block shows all hits at once again"
        assert d["button"], "without a button you couldn't reach the remaining 37"

    def test_the_button_shows_all_and_then_fewer_again(self):
        d = js(
            self.BODY
            + """
        A.svLoadMore(true, A.svState.seq).then(() => {
            A.svToggleFiles();
            const open_ = (A.el('svResults').innerHTML.match(/sv-d-item/g) || []).length;
            A.svToggleFiles();
            const closed = (A.el('svResults').innerHTML.match(/sv-d-item/g) || []).length;
            output({open_: open_, closed});
        });
        """
        )
        assert d["open_"] == 40
        assert d["closed"] == 3

    def test_a_new_search_collapses_it_again(self):
        """Otherwise a once-expanded list would drag its height through all
        following searches — exactly the state that hid the
        transcripts."""
        d = js(
            self.BODY
            + """
        A.svLoadMore(true, A.svState.seq).then(() => {
            A.svToggleFiles();
            // svRun reads the query from the INPUT FIELD, not from svState —
            // a directly set svState.query is overwritten by this.
            A.el('svQuery').value = 'something else';
            A.svRun();
            setTimeout(() => {
                output({lines: (A.el('svResults').innerHTML
                                  .match(/sv-d-item/g) || []).length});
            }, 60);
        });
        """
        )
        assert d["lines"] == 3


class TestTheSemanticSearch:
    """The vector search as a second track in the large search.

    It answers what full-text search structurally cannot: whoever searches
    for "it's about cooking" finds the passage "that's called simmering" only
    via the meaning — no searched word appears there at all.
    """

    MEANING = """
    A.answerFor('/api/rag/search', {ok: true, elapsed_ms: 66, results: [
        {content: 'that is called simmering', speaker: 'SPEAKER_00',
         start_sec: 87.95, score: 0.55, rel_path: 'videos/a.mp4',
         file_dir: '/mnt/media'}]});
    """

    def test_it_runs_alongside_the_literal_search(self):
        d = js(
            self.MEANING
            + """
        A.setAnswer({total: 3, files: 1, results: [
            {filename: 'a.mp4', start: 0, text: 'x', file_dir: '/m'}]});
        """
            + self.MEANING
            + """
        A.svState.query = 'cooking'; A.svState.offset = 0;
        A.svState.done = false; A.svState.loading = false;
        A.svLoadMore(true, A.svState.seq).then(() => setTimeout(() => {
            output({asked: A.fetched().some(u => u.includes('/api/rag/search')),
                     mode_vector: (A.fetched().find(u => u.includes('/api/rag/search')) || '')
                             .includes('mode=vector')});
        }, 30));
        """
        )
        assert d["asked"], "the semantic search was never queried at all"
        assert d["mode_vector"], (
            "hybrid would double the full-text part — what's wanted is the extra gain"
        )

    def test_without_a_literal_hit_it_is_STILL_queried(self):
        """The error of the first attempt, and the most costly one imaginable.

        If the full-text search finds nothing, `svLoadMore` jumps into the
        filename fallback and returns with `return`. The semantic search only
        sat behind the regular path — so it dropped out exactly in the case
        it exists for: "someone is annoyed about a technical problem" has,
        with high probability, zero word hits.
        """
        d = js(
            self.MEANING
            + """
        A.setAnswer({total: 0, files: 0, results: [], files: []});
        """
            + self.MEANING
            + """
        A.svState.query = 'someone is annoyed about a technical problem';
        A.svState.offset = 0; A.svState.done = false; A.svState.loading = false;
        A.svLoadMore(true, A.svState.seq).then(() => setTimeout(() => {
            output({asked: A.fetched().some(u => u.includes('/api/rag/search'))});
        }, 40));
        """
        )
        assert d["asked"], (
            "without a word hit, the semantic search is the only remaining "
            "answer — and it was skipped"
        )

    def test_without_a_literal_hit_it_stays_open(self):
        d = js(
            self.MEANING
            + """
        A.setAnswer({total: 0, files: 0, results: [], files: []});
        """
            + self.MEANING
            + """
        A.svState.query = 'cooking'; A.svState.offset = 0;
        A.svState.done = false; A.svState.loading = false;
        A.svLoadMore(true, A.svState.seq).then(() => setTimeout(() => {
            const h = A.el('svResults').innerHTML;
            output({open_count: h.includes('sv-semantic-item'),
                     text: h.includes('simmering')});
        }, 40));
        """
        )
        assert d["open_count"], "the only answer is hiding behind a button"
        assert d["text"]

    def test_an_error_in_the_semantic_search_stays_a_footnote(self):
        """Whoever hasn't set up an embedding server still has a
        complete full-text search. Its outage must not drag that down too."""
        d = js("""
        A.setAnswer({total: 2, files: 1, results: [
            {filename: 'a.mp4', start: 0, text: 'hits', file_dir: '/m'}]});
        A.answerFor('/api/rag/search', {error: 'embedding server unreachable'});
        A.svState.query = 'cooking'; A.svState.offset = 0;
        A.svState.done = false; A.svState.loading = false;
        A.svLoadMore(true, A.svState.seq).then(() => setTimeout(() => {
            const h = A.el('svResults').innerHTML;
            output({literal_there: h.includes('hits'),
                     hint: h.includes('not available'),
                     total: A.svState.total});
        }, 40));
        """)
        assert d["literal_there"], "the outage of the bonus feature ate the hits"
        assert d["hint"], "the outage is being hidden"
        assert d["total"] == 2

    def test_an_outdated_answer_is_discarded(self):
        """The semantic search needs ~2 s. If the user types something
        new in the meantime, the old answer must not fall into the new hit list."""
        d = js(
            self.MEANING
            + """
        A.svState.query = 'cooking';
        const oldSeq = A.svState.seq;
        A.svState.seq = oldSeq + 1;          // meanwhile searched anew
        A.svMeaningSearch(oldSeq, 0);
        setTimeout(() => {
            output({inserted: A.el('svResults').innerHTML.includes('simmering')});
        }, 40);
        """
        )
        assert not d["inserted"]

    def test_clearing_the_search_field_sweeps_away_the_extra_blocks(self):
        """Noticed while writing the tests: whoever clears their search field
        got "Enter at least 2 characters" — and beneath that, unchanged, the
        semantically related passages of the previous query."""
        d = js("""
        A.setAnswer({total: 0, files: 0, results: [], files: []});
        A.answerFor('/api/rag/search', {ok: true, results: [
            {content: 'old answer', start_sec: 1, score: 0.5,
             rel_path: 'a.mp4', file_dir: '/m'}]});
        A.el('svQuery').value = 'a rather long question';
        A.svRun();
        setTimeout(() => {
            const before = A.el('svResults').innerHTML.includes('old answer');
            A.el('svQuery').value = 'a';
            A.svRun();
            setTimeout(() => {
                output({before, after:
                    A.el('svResults').innerHTML.includes('old answer')});
            }, 60);
        }, 60);
        """)
        assert d["before"], "the semantic search never ran at all"
        assert not d["after"], "the block from the previous query is still there"


class TestTheTabs:
    """**The rework:** tabs instead of a
    split area, a reading column instead of paragraphs in the list,
    sidebar removed, map recursive.
    """

    def test_the_file_tab_does_not_search_the_spoken_text(self):
        """The most costly bug of the rework, found in the browser.

        The file tab sent the text query along too. That kept the server
        in full-text search, `mode=files` never took hold, and the tab was
        empty — even though the whole library would have been there. The query
        stays in the field, it just isn't sent along here.
        """
        d = js("""
        A.el('svQuery').value = 'cooking';
        A.svState.query = 'cooking';
        A.svTab('text');
        const inText = A.svParams();
        A.svTab('files');
        const inFile = A.svParams();
        output({inText, inFile});
        """)
        assert "q=cooking" in d["inText"]
        assert (
            "q=&" in d["inFile"]
            or d["inFile"].startswith("q=&")
            or "q=" in d["inFile"]
            and "cooking" not in d["inFile"]
        )
        assert "mode=files" in d["inFile"]
        assert "mode=files" not in d["inText"]

    def test_each_tab_writes_to_its_own_list(self):
        """If both shared `#svResults`, every tab switch would overwrite
        the other one's list."""
        d = js("""
        A.svTab('text');    const a = A.svListBox().id;
        A.svTab('files'); const b = A.svListBox().id;
        output({a, b});
        """)
        assert d["a"] == "svResults"
        assert d["b"] == "svFileList"
        assert d["a"] != d["b"]

    def test_the_hit_list_is_one_line_per_hit(self):
        """Previously every hit came with its whole paragraph."""
        d = js("""
        A.setAnswer({total: 4, files: 2, files: [], results: [
            {filename: 'a.mp4', start: 1, start_fmt: '0:01', text: 'short', file_dir: '/m'},
            {filename: 'a.mp4', start: 2, start_fmt: '0:02', text: 'still oneOf', file_dir: '/m'},
            {filename: 'b.mp4', start: 3, start_fmt: '0:03', text: 'third', file_dir: '/m'}]});
        A.answerFor('/api/rag/search', {ok: true, results: []});
        A.svState.query = 'cooking'; A.svState.offset = 0;
        A.svState.done = false; A.svState.loading = false;
        A.svLoadMore(true, A.svState.seq).then(() => setTimeout(() => {
            const h = A.el('svResults').innerHTML;
            output({lines: (h.match(/sv-row/g) || []).length,
                     groups: (h.match(/sv-group/g) || []).length,
                     hits: A.svState.hits.length});
        }, 40));
        """)
        assert d["lines"] >= 3
        assert d["groups"] == 0, (
            "group headers cost half the area: 200 hits spread across 125 files"
        )
        assert d["hits"] == 3

    def test_the_first_hit_is_opened_right_away(self):
        """Otherwise half the view stays empty and you have to guess whether
        it does anything at all."""
        d = js("""
        A.setAnswer({total: 2, files: 1, files: [], results: [
            {filename: 'a.mp4', start: 1, start_fmt: '0:01', text: 'one', file_dir: '/m'}]});
        A.answerFor('/api/rag/search', {ok: true, results: []});
        A.svState.query = 'cooking'; A.svState.offset = 0;
        A.svState.done = false; A.svState.loading = false;
        A.svLoadMore(true, A.svState.seq).then(() => setTimeout(() => {
            output({selection: A.svState.selection,
                     read: A.el('svRead').innerHTML.length});
        }, 40));
        """)
        assert d["selection"] == 0
        assert d["read"] > 50

    def test_arrow_keys_do_not_run_past_the_edges(self):
        d = js("""
        A.svState.hits = [{filename:'a', start:0, text:'x'},
                             {filename:'a', start:1, text:'y'}];
        A.svState.selection = 0;
        A.svMoveSelection(-1);  const bottom = A.svState.selection;
        A.svMoveSelection(1); A.svMoveSelection(1); A.svMoveSelection(1);
        output({bottom, top: A.svState.selection});
        """)
        assert d["bottom"] == 0, "there's nothing before the first hit"
        assert d["top"] == 1, "nor behind the last one"

    def test_a_broken_facets_response_does_not_drag_down_the_hits(self):
        """Introduced during the rework and found by the test.

        `svRenderFacets` lost both of its guards (`unavailable`, empty
        list). A response without `file_count` set `fileCount = undefined`,
        the next `svSetStatus` threw, and the catch in `svLoadMore` wrote
        "Error: …" over the hit list that was already complete. An outage
        of the SIDE-DISH query thus dragged down the actual answer too.
        """
        d = js("""
        A.svState.total = 42; A.svState.offset = 7;
        // Exactly the shape that triggered it: `files` is a NUMBER, not
        // a list — that's how /api/search answers (there `files` counts the
        // files of the page). `files.length` was then undefined, and
        // `data.file_count || files.length` resulted in undefined instead of 0.
        // With `{unavailable: true}` alone the bug would NOT have shown up:
        // there `files` is empty and `.length` is cleanly 0.
        A.svRenderFacets({files: 4});
        const number = typeof A.svState.fileCount;
        let error = '';
        try { A.svSetStatus(42, 0); } catch (e) { error = e.message; }
        output({number, value: A.svState.fileCount, error,
                 shown_total: A.el('svTotal').textContent,
                 files: A.el('svFiles').textContent});
        """)
        # First lock: `fileCount` stays a number, whatever comes in.
        assert d["number"] == "number", f"fileCount is {d['number']} ({d['value']})"
        # Second lock: the status line must not throw either, even if
        # something else comes through some day.
        assert d["error"] == "", f"status line throws: {d['error']}"
        assert d["shown_total"] == "42"
        assert d["files"] == "0"

    def test_the_short_name_drops_the_path(self):
        d = js("""
        output({a: A.svShortName('videos/Archive/meeting_2024-03-01.mp4'),
                 b: A.svShortName('withoutPath.mp3'), c: A.svShortName('')});
        """)
        assert d["a"] == "meeting_2024-03-01.mp4"
        assert d["b"] == "withoutPath.mp3"
        assert d["c"] == ""

    def test_a_tab_switch_does_not_corrupt_the_numbers(self):
        """**The finding:** the UI looked different after switching tabs
        and going back.

        Both tabs query the same endpoint with different parameters, but
        share `svState`. The text tab therefore showed "N hits in N files"
        — the numbers of the file tab — over a list with far fewer hits,
        and the block with the filename hits was gone.
        """
        d = js("""
        A.setAnswer({total: 5000, files: 2, results: [
            {filename: 'a.mp4', start: 1, start_fmt: '0:01', text: 'one', file_dir: '/m'}],
            files: [{filename:'v.mp4', display_name:'v', duration_sec:60,
                       is_video:true, has_transcript:true, file_dir:'/m'}]});
        A.answerFor('/api/rag/search', {ok: true, results: []});
        A.el('svQuery').value = 'test';
        A.svRun();
        setTimeout(() => {
            const before = {total: A.svState.total,
                            files: A.svState.files.length,
                            listing: A.el('svResults').innerHTML.length};
            // The file tab returns entirely different numbers …
            A.setAnswer({total: 200000, files: 0, mode: 'files',
                             results: [{filename: 'x.mp4', file_dir: '/m'}], files: []});
            A.svTab('files');
            setTimeout(() => {
                const inFile = A.svState.total;
                A.svTab('text');
                setTimeout(() => {
                    output({before, inFile, after: {
                        total: A.svState.total,
                        files: A.svState.files.length,
                        listing: A.el('svResults').innerHTML.length}});
                }, 40);
            }, 40);
        }, 40);
        """)
        assert d["inFile"] == 200000, "the file tab shows its own number"
        assert d["after"]["total"] == d["before"]["total"], (
            "the file tab's number stuck around in the text tab"
        )
        assert d["after"]["files"] == d["before"]["files"], (
            "the block with the filename hits has disappeared"
        )
        assert d["after"]["listing"] == d["before"]["listing"], (
            "the hit list came back differently than it was"
        )

    def test_on_a_changed_query_it_searches_again(self):
        """The remembered state must not become a can of preserves: if the
        query changes, it's worthless and a search must run."""
        d = js("""
        A.setAnswer({total: 3, files: 1, files: [], results: [
            {filename: 'a.mp4', start: 1, start_fmt: '0:01', text: 'x', file_dir: '/m'}]});
        A.answerFor('/api/rag/search', {ok: true, results: []});
        A.el('svQuery').value = 'test';
        A.svRun();
        setTimeout(() => {
            A.svSaveState();
            A.el('svQuery').value = 'something else';
            A.svState.query = 'something else';
            output({reused: A.svLoadState('text')});
        }, 40);
        """)
        assert d["reused"] is False

    def test_the_mode_decides_which_endpoints_get_queried(self):
        """At the top of the search one can choose only meaning (RAG), only
        text, or hybrid.

        The three modes are not settings of the same search —
        "literal" searches all transcripts in the full-text index,
        "semantic" only the embedded ones in the vector index.
        """

        def asked(mode):
            return js(
                """
            A.setAnswer({total: 2, files: 1, files: [], results: [
                {filename: 'a.mp4', start: 1, start_fmt: '0:01', text: 'x', file_dir: '/m'}]});
            A.answerFor('/api/rag/search', {ok: true, results: [
                {content: 'y', start_sec: 3, score: 0.5, rel_path: 'b.mp4', file_dir: '/m'}]});
            A.el('svQuery').value = 'cooking';
            A.clearFetched();
            A.svSetMode('%s');
            setTimeout(() => {
                const u = A.fetched();
                output({fulltext: u.some(x => x.includes('/api/search?')),
                         vector: u.some(x => x.includes('/api/rag/search')),
                         lines: (A.el('svResults').innerHTML.match(/sv-row/g) || []).length});
            }, 60);
            """
                % mode
            )

        both_of_them = asked("both")
        assert both_of_them["fulltext"] and both_of_them["vector"], both_of_them

        text_only = asked("text")
        assert text_only["fulltext"], text_only
        assert not text_only["vector"], "in literal mode nothing is embedded"

        meaning_only = asked("meaning")
        assert meaning_only["vector"], meaning_only
        assert not meaning_only["fulltext"], (
            "the pure semantic mode must not query the full-text search at all"
        )
        assert meaning_only["lines"] >= 1, "the vector hits ARE the list here"

    def test_in_semantic_mode_the_lines_carry_their_score(self):
        """Because the number of hits says nothing about their quality (the
        vector search always returns k neighbours), the score has to
        stand on the line."""
        d = js("""
        A.answerFor('/api/rag/search', {ok: true, elapsed_ms: 42, results: [
            {content: 'Cooking', start_sec: 3, score: 0.5231, rel_path: 'b.mp4', file_dir: '/m'}]});
        A.el('svQuery').value = 'cooking';
        A.svSetMode('meaning');
        setTimeout(() => {
            const h = A.el('svResults').innerHTML;
            output({value: h.includes('0.52'), column_name: h.includes('sv-z-score'),
                     header: h.includes('most similar')});
        }, 60);
        """)
        assert d["value"] and d["column_name"], "the similarity score is missing on the line"
        assert d["header"], 'the header must say "most similar", not "matching"'

    def test_playing_leaves_the_search_open(self):
        """Pressing play keeps the search window open; playback runs in
        the background.

        Previously every play click called `svGoto`, and that closed the
        view. Whoever wanted to listen through 200 hits had to reopen the
        search every single time.
        """
        d = js("""
        A.el('svOverlay').classList.add('visible');
        A.svPlay('videos/a.mp4', 12.5, '/mnt');
        setTimeout(() => {
            output({open_count: A.el('svOverlay').classList.contains('visible'),
                     bar: A.el('svPlayer').style.display,
                     jumped: A.jumped(),
                     plays: A.svState.plays});
        }, 40);
        """)
        assert d["open_count"] is True, "the search was closed"
        assert d["bar"] == "flex", "the playback bar stays hidden"
        assert d["jumped"][0]["f"] == "videos/a.mp4"
        assert d["jumped"][0]["t"] == 12.5
        assert d["plays"]["filename"] == "videos/a.mp4"

    def test_ending_playback_removes_the_bar(self):
        d = js("""
        A.el('svOverlay').classList.add('visible');
        A.svPlay('videos/a.mp4', 1, '/mnt');
        setTimeout(() => {
            A.svWgStop();
            output({bar: A.el('svPlayer').style.display, plays: A.svState.plays});
        }, 40);
        """)
        assert d["bar"] == "none"
        assert d["plays"] is None

    def test_reopening_continues_instead_of_searching_anew(self):
        """Reopening the search continues where it was last left off."""
        d = js("""
        A.setAnswer({total: 7, files: 1, files: [], results: [
            {filename: 'a.mp4', start: 1, start_fmt: '0:01', text: 'one', file_dir: '/m'}]});
        A.answerFor('/api/rag/search', {ok: true, results: []});
        A.openSearchView('cooking');
        setTimeout(() => {
            const before = {total: A.svState.total, selection: A.svState.selection,
                            listing: A.el('svResults').innerHTML.length};
            A.closeSearchView();
            A.clearFetched();
            A.openSearchView('cooking');           // the same query
            setTimeout(() => {
                output({before, after: {total: A.svState.total,
                         selection: A.svState.selection,
                         listing: A.el('svResults').innerHTML.length},
                         refetched: A.fetched().filter(u => u.includes('/api/search?')).length});
            }, 60);
        }, 60);
        """)
        assert d["after"] == d["before"], "the view came back differently"
        assert d["refetched"] == 0, "continuing must not trigger a new search"

    def test_a_new_query_still_searches_anew(self):
        """The remembered state must not get in the way when you actually
        want to know something different."""
        d = js("""
        A.setAnswer({total: 7, files: 1, files: [], results: [
            {filename: 'a.mp4', start: 1, start_fmt: '0:01', text: 'one', file_dir: '/m'}]});
        A.answerFor('/api/rag/search', {ok: true, results: []});
        A.openSearchView('cooking');
        setTimeout(() => {
            A.closeSearchView();
            A.clearFetched();
            A.openSearchView('something else');
            setTimeout(() => {
                output({refetched: A.fetched().filter(u => u.includes('/api/search?')).length,
                         question: A.svState.query});
            }, 60);
        }, 60);
        """)
        assert d["refetched"] >= 1
        assert d["question"] == "something else"

    def test_a_pinned_file_becomes_a_badge(self):
        """The pinning survives reopening now on purpose — but then it must
        be visible. Its invisibility was exactly the bug
        described at the top."""
        d = js("""
        A.svState.pinned = 'videos/Archive/long/meeting_2024-03-01.mp4';
        A.svShowPin();
        const on = {display: A.el('svPinned').style.display,
                    text: A.el('svPinned').innerHTML};
        A.svState.pinned = '';
        A.svShowPin();
        output({on, off: A.el('svPinned').style.display});
        """)
        assert d["on"]["display"] != "none"
        assert "meeting_2024-03-01.mp4" in d["on"]["text"]
        assert "Archive" not in d["on"]["text"], "in the badge the filename is enough"
        assert d["off"] == "none"


class TestFileTabUsesTheQuery:
    """The file search seemed not to work.

    ``voice_note_`` was in the search field at the top, and below it
    "N hits in N files" — i.e. the whole library.

    The reason was a decision that is correct on its own: the file tab
    throws away the text query, otherwise the server runs a full-text
    search and the file list never comes. It just left it STANDING in the
    field. You see your search term and a result that has nothing to do
    with it; that there is a second, dedicated field for filenames has to
    be found first.

    At the endpoint: ``file=voice_note`` returns the matching files,
    ``q=voice_note`` in file mode returns none.
    """

    def test_the_query_filters_the_filename(self):
        d = js("""
            A.svState.tab = 'files';
            A.svState.query = 'voice_note_';
            A.svState.file = '';
            A.svState.pinned = null;
            output({p: A.svParams()});
        """)
        assert "file=voice_note_" in d["p"], d["p"]
        # … and NOT as full-text, otherwise the file list stays empty.
        assert "q=&" in d["p"] or d["p"].startswith("q=&"), d["p"]

    def test_a_filename_filter_of_the_text_tab_stays_there(self):
        """The files tab hides the "File name contains" field —
        its query IS the file name, and two fields for one thing were
        confusing. A value left in that field by the text tab must then not
        narrow the file search unseen."""
        d = js("""
            A.svState.tab = 'files';
            A.svState.query = 'voice_note_';
            A.svState.file = 'vacation';
            A.svState.pinned = null;
            output({p: A.svParams()});
        """)
        assert "file=voice_note_" in d["p"] and "vacation" not in d["p"], d["p"]

    def test_the_files_tab_counts_its_files(self):
        """Its "hits in" part is hidden — the number left is the
        file count, also when the tab is restored (no count passed)."""
        d = js("""
            A.svState.tab = 'files';
            A.svState.fileCount = null;
            A.svState.offset = 200;
            A.svSetStatus(1300, 0);
            output({files: A.el('svFiles').textContent, shown: A.el('svShown').textContent});
        """)
        assert d["files"].replace(",", "").replace(".", "") == "1300", d
        assert d["shown"] == "200"

    def test_a_pinned_file_beats_everything(self):
        d = js("""
            A.svState.tab = 'files';
            A.svState.query = 'voice_note_';
            A.svState.file = 'vacation';
            A.svState.pinned = 'videos/a.mp4';
            output({p: A.svParams()});
        """)
        assert "file=videos%2Fa.mp4" in d["p"], d["p"]

    def test_in_the_text_tab_the_query_stays_the_query(self):
        """The fix must not touch the normal case: there, the search term is
        the spoken word, not the filename."""
        d = js("""
            A.svState.tab = 'text';
            A.svState.query = 'vinegar';
            A.svState.file = '';
            A.svState.pinned = null;
            output({p: A.svParams()});
        """)
        assert "q=vinegar" in d["p"] and "file=" not in d["p"], d["p"]


class TestThePersonFilter:
    """Search for "x" only where a chosen person (e.g. Alice) speaks.

    Two things have to be right in the view, otherwise the sidebar
    shows different numbers than the list: the filter has to go into the
    SAME parameter string as everything else, and the checkbox "only her
    sentences" must not exist at all without a chosen person — a switch
    with no effect is worse than none.
    """

    def test_without_a_person_the_query_stays_as_before(self):
        d = js("""
        A.svState.person = ''; A.svState.personSegments = false;
        output({p: A.svPersonParams()});
        """)
        assert d["p"] == ""

    def test_with_a_person_it_is_appended(self):
        d = js("""
        A.svState.person = 'Alice'; A.svState.personSegments = false;
        output({p: A.svPersonParams()});
        """)
        assert d["p"] == "&person=Alice"

    def test_names_with_special_characters_are_encoded(self):
        d = js("""
        A.svState.person = 'Alice & Bob';
        output({p: A.svPersonParams()});
        """)
        assert "%26" in d["p"], d["p"]
        assert " " not in d["p"]

    def test_only_her_sentences_is_included(self):
        d = js("""
        A.svState.person = 'Alice'; A.svState.personSegments = true;
        output({p: A.svPersonParams()});
        """)
        assert d["p"] == "&person=Alice&person_segments=1"

    def test_the_filter_sits_in_the_shared_string(self):
        """The hit list and facets both use `svParams()` — that's the only
        way their numbers stay consistent."""
        d = js("""
        A.svState.query = 'cake'; A.svState.person = 'Alice';
        A.svState.tab = 'text'; A.svState.pinned = ''; A.svState.file = '';
        output({p: A.svParams()});
        """)
        assert "person=Alice" in d["p"]
        assert d["p"].startswith("q=cake")

    def test_the_checkbox_appears_only_with_a_person(self):
        d = js("""
        A.el('svPerson').value = '';
        A.svShowPersonBox();
        const off_ = A.el('svPersonSegmentsBox').style.display;
        A.el('svPerson').value = 'Alice';
        A.svShowPersonBox();
        output({without: off_, with_: A.el('svPersonSegmentsBox').style.display});
        """)
        assert d["without"] == "none"
        assert d["with_"] == ""

    def test_deselecting_drops_the_checkmark(self):
        """Otherwise an invisible checkmark would keep filtering the next search."""
        d = js("""
        A.el('svPerson').value = 'Alice';
        A.el('svPersonSegments').checked = true;
        A.el('svPerson').value = '';
        A.svShowPersonBox();
        output({tick_mark: A.el('svPersonSegments').checked});
        """)
        assert d["tick_mark"] is False


class TestTheFilesTab:
    """The file page was hard to read with the whole path on every line:
    now the folder once as a head line, the line itself
    only the file's name, its values in columns."""

    ITEMS = """[
        {filename: 'Archive/old/projects/lib/io/tests/data/a.wav', file_dir: '/nas',
         file_size: 3586, file_date: 1704067200, has_transcript: false},
        {filename: 'Archive/old/projects/lib/io/tests/data/b.wav', file_dir: '/nas',
         file_size: 3586, file_duration: 0.4, has_transcript: true, file_word_count: 12},
        {filename: 'Archive/c.mp4', file_dir: '/nas', file_size: 51000000}]"""

    def test_one_head_per_folder_and_names_in_the_lines(self):
        d = js(f"""
        A.svState.tab = 'files';
        A.svRenderFileList({self.ITEMS}, true);
        output({{html: A.el('svFileList').innerHTML}});
        """)
        html = d["html"]
        assert html.count('class="sv-fold"') == 2
        # The long folder folds its middle; the full path is in the tooltip
        assert "<span>Archive</span><i>/</i><span>…</span>" in html
        assert 'title="/nas/Archive/old/projects/lib/io/tests/data"' in html
        assert '<span class="sv-fl-name">a.wav</span>' in html
        assert "12 words" in html and "no transcript" in html

    def test_load_more_continues_the_last_folder(self):
        d = js(f"""
        A.svState.tab = 'files';
        A.svRenderFileList({self.ITEMS}, true);
        A.svRenderFileList([{{filename: 'Archive/d.mp4', file_dir: '/nas'}}], false);
        output({{html: A.el('svFileList').innerHTML}});
        """)
        assert d["html"].count('class="sv-fold"') == 2


class TestTheMapsColors:
    def test_one_hue_brighter_with_more_hits(self):
        d = js("output({h: [0, 1, 25, 100].map(n => A.svMapHeat(n, 100))});")
        assert d["h"][0] == 14 and d["h"][-1] == 80 and d["h"] == sorted(d["h"])

    def test_an_open_folder_stays_behind_its_tiles(self):
        """Hovering a folder's name band lifted the frame over its own
        tiles — they vanished, and the clicks meant for them zoomed in."""
        css = (SEARCH_JS.parent.parent / "css" / "searchview.css").read_text(encoding="utf-8")

        def rule(sel):
            return css[css.index(sel + " {") :].split("}")[0]

        assert "transparent" in rule(".sv-tile.folder.open")
        assert "z-index" not in rule(".sv-tile.folder:hover")
        assert "z-index" in rule(".sv-tile.folder:not(.open):hover")

    def test_no_rainbow_left_in_the_tab(self):
        src = SEARCH_JS.read_text(encoding="utf-8")
        level = src[src.index("function svMapLevel(") : src.index("/* Click on a folder: ZOOM IN.")]
        assert "smHue" not in level and "--heat" in level


class TestTheTopBarRoutes:
    """Enter in the top bar used to start a
    language-model run of a minute for every word. A description goes to
    Find; a few words or quoted words to the text, a file name to the files."""

    def test_where_a_query_goes(self):
        d = js("""
            const qs = ['Erin Baker', 'quarterly budget review', '"good morning" he said',
                        'VID-20240301-WA0001.mp4', 'vacation/2024', 'clip*', 'PTT-2021',
                        'Alice walks home at night and finds a lost dog', 'where does alice find the dog?', '  '];
            output({r: qs.map(q => A.svRouteQuery(q))});
        """)
        assert d["r"] == [
            "text",
            "text",
            "text",
            "files",
            "files",
            "files",
            "files",
            "find",
            "find",
            None,
        ]
