"""The shell (static/js/shell.js): the rail's places, which one is lit, the
badges, and the history behind Alt+Left / Alt+Right and Esc.

The behaviour runs in node against a small stand-in for the page: windows
are elements whose "visible" class the openers set, the browser history is
a list with pushState / back / forward that fires popstate.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

STATIC = Path(__file__).resolve().parent.parent / "whisper_tm" / "static"
SHELL = (STATIC / "js" / "shell.js").read_text(encoding="utf-8")
HTML = (STATIC / "index.html").read_text(encoding="utf-8")

_RUN = r"""
const fs = require('fs'), vm = require('vm');
const els = {};
function el(id) {
    if (els[id]) return els[id];
    const cls = new Set();
    return els[id] = {
        id, cls, dataset: {}, hidden: false, textContent: '', title: '',
        firstChild: {textContent: ''},
        classList: {contains: c => cls.has(c), add: c => cls.add(c), remove: c => cls.delete(c),
                    toggle: (c, on) => { const v = on === undefined ? !cls.has(c) : on; v ? cls.add(c) : cls.delete(c); return v; }},
        setAttribute() {}, closest() { return null; },
    };
}
const winL = {}, docL = {};
const states = []; let idx = -1;
const fire = () => (winL.popstate || []).forEach(f => f({state: states[idx]}));
const history = {
    pushState(s) { states.length = idx + 1; states.push(s); idx++; },
    replaceState(s) { if (idx < 0) { states.push(s); idx = 0; } else states[idx] = s; },
    back() { if (idx > 0) { idx--; fire(); } },
    forward() { if (idx < states.length - 1) { idx++; fire(); } },
};
const calls = [];
const sb = {
    console, setTimeout, clearTimeout, history, location: {href: 'http://host/'},
    fetch: (url, o = {}) => { calls.push([o.method || 'GET', url, o.body || '']);
                              return Promise.resolve({ok: true, json: () => Promise.resolve({})}); },
    localStorage: {getItem: () => null, setItem() {}},
    MutationObserver: class { observe() {} },
    addEventListener: (t, f) => (winL[t] = winL[t] || []).push(f),
    document: {
        getElementById: el, querySelectorAll: () => [], body: el('#body'),
        addEventListener: (t, f) => (docL[t] = docL[t] || []).push(f),
    },
};
sb.window = sb; sb.globalThis = sb;
vm.createContext(sb);
const show = id => `function(){ document.getElementById('${id}').classList.add('visible'); }`;
const hide = id => `function(){ document.getElementById('${id}').classList.remove('visible'); }`;
vm.runInContext(`
    var files = [{name: 'a.mp3'}, {name: 'b.mp3'}];
    var activeFile = null;
    async function selectFile(f) { activeFile = f.name; }
    var VX = {tab: 'map'};
    function vxTab(n) { VX.tab = n; }
    var openStatsView = ${show('statsOverlay')}, closeStatsView = ${hide('statsOverlay')};
    var vxOpen = ${show('mapOverlay')}, vxClose = ${hide('mapOverlay')};
    var openSettings = ${show('settingsOverlay')}, closeSettings = ${hide('settingsOverlay')};
    var openHealthView = ${show('healthOverlay')}, closeHealthView = ${hide('healthOverlay')};
`, sb);
vm.runInContext(fs.readFileSync(process.env.SHELL_JS, 'utf8') + '\nglobalThis.Shell = Shell;', sb);
el('dirInput').value = '/lib/media';
(docL.DOMContentLoaded || []).forEach(f => f());
const tick = () => new Promise(r => setTimeout(r, 0));
const S = sb.Shell;
const where = () => `${S.current}|${sb.VX.tab === 'map' || S.current !== 'voices' ? '' : sb.VX.tab}|${vm.runInContext('activeFile', sb) || ''}`;
const key = (k, extra = {}) => {
    let stopped = false;
    const ev = {key: k, altKey: true, ctrlKey: false, metaKey: false, shiftKey: false,
                preventDefault() { stopped = true; }, ...extra};
    (docL.keydown || []).forEach(f => f(ev));
    return stopped;
};
(async () => {
    const out = {};
    S.go('library');
    await sb.selectFile(sb.files[0]);
    S.go('voices');
    sb.vxTab('people'); sb.svShowTab = undefined; S.detect();
    let undone = 0;
    S.step(() => { undone++; });
    out.size = S.history.size;
    out.walk = [where()];
    out.altLeftStopsBrowser = key('ArrowLeft'); await tick();
    out.undone = undone;
    out.walk.push(where());
    for (let i = 0; i < 4; i++) { history.back(); await tick(); out.walk.push(where()); }
    // forward again, then a new move drops what lay ahead
    key('ArrowRight'); await tick(); out.fwd = where();
    S.go('settings');
    out.afterNewMove = S.history;
    // Esc inside a window: a step of this place first
    S.go('voices');
    let closed = 0;
    S.step(() => { closed++; });
    out.undoStep = S.undoStep(); await tick();
    out.closedByEsc = closed;
    out.undoStepAgain = S.undoStep();
    // leave(): the view's own close control goes through the history
    S.step(() => { closed++; });
    const before = S.history.at;
    S.leave(() => { closed++; }); // not the newest step: just runs it
    out.leaveOther = [closed, S.history.at === before];
    out.plainAlt = key('ArrowLeft', {ctrlKey: true});
    // preferences: loaded once, a fold is saved a moment later
    S.fold(true);
    await new Promise(r => setTimeout(r, 450));
    out.calls = calls;
    console.log('§§' + JSON.stringify(out));
})();
"""


@pytest.fixture(scope="module")
def run() -> dict:
    if shutil.which("node") is None:
        pytest.skip("node not available")
    done = subprocess.run(
        ["node", "-e", _RUN],
        capture_output=True,
        text=True,
        timeout=60,
        env={"SHELL_JS": str(STATIC / "js" / "shell.js"), "PATH": os.environ.get("PATH", "")},
    )
    assert done.returncode == 0, done.stderr
    line = next(z for z in done.stdout.splitlines() if z.startswith("§§"))
    return json.loads(line[2:])


def test_every_move_is_an_entry(run):
    # home, library, library + a recording, voices, voices/people, the step
    assert run["size"] == 6


def test_alt_left_walks_back_through_steps_tabs_places_and_recordings(run):
    assert run["altLeftStopsBrowser"] is True, "one press must be one step"
    assert run["undone"] == 1, "leaving a step backwards undoes it"
    assert run["walk"] == [
        "voices|people|a.mp3",  # the step
        "voices|people|a.mp3",  # step undone
        "voices||a.mp3",  # the tab before
        "library||a.mp3",  # the place before, with its recording
        "library||a.mp3",  # going back never closes a recording
        "home||a.mp3",
    ]


def test_forward_and_a_new_move(run):
    assert run["fwd"] == "library||a.mp3"
    assert run["afterNewMove"]["at"] == run["afterNewMove"]["size"] - 1


def test_esc_undoes_a_step_of_this_place_first(run):
    assert run["undoStep"] is True and run["closedByEsc"] == 1
    assert run["undoStepAgain"] is False, "no step left: Esc closes the window"


def test_leave_runs_a_step_that_is_not_the_newest(run):
    assert run["leaveOther"] == [2, True]


def test_other_modifier_combinations_are_left_alone(run):
    assert run["plainAlt"] is False


def test_preferences_load_once_and_a_fold_is_saved(run):
    gets = [c for c in run["calls"] if c[0] == "GET" and c[1] == "/api/ui/prefs"]
    puts = [c for c in run["calls"] if c[0] == "PUT" and c[1] == "/api/ui/prefs"]
    assert len(gets) == 1
    assert len(puts) == 1 and '"rail_folded":true' in puts[0][2]


def test_opening_a_recording_notes_it_as_recent(run):
    posts = [c for c in run["calls"] if c[0] == "POST" and c[1] == "/api/ui/recent"]
    # a.mp3 opened once (going back to it later does not count again)
    assert posts and posts[0][2] == '{"path":"/lib/media/a.mp3"}'


# ── Wiring ──────────────────────────────────────────────────────────────


def test_the_page_loads_the_shell_after_the_app():
    app = HTML.index('src="/static/js/app.js')
    assert HTML.index('src="/static/js/shell.js') > app


def test_steps_are_set_by_the_views_that_have_them():
    review = (STATIC / "js" / "voices_review.js").read_text(encoding="utf-8")
    panel = (STATIC / "js" / "voicemap_panel.js").read_text(encoding="utf-8")
    voices = (STATIC / "js" / "voices.js").read_text(encoding="utf-8")
    assert "Shell.step(reviewClose)" in review and "Shell.leave(reviewClose)" in review
    assert "Shell.step(panelClose)" in panel and "Shell.leave(panelClose)" in panel
    assert "Shell.undoStep()" in voices


def test_no_back_arrow_buttons_left_in_the_voices_views():
    for name in ("voices_review.js", "voicemap_panel.js"):
        src = (STATIC / "js" / name).read_text(encoding="utf-8")
        assert "← People" not in src and "← Groups" not in src, name


def test_names_changing_reloads_people_next_time():
    voices = (STATIC / "js" / "voices.js").read_text(encoding="utf-8")
    body = voices[voices.index("function vxMapChanged()") :]
    body = body[: body.index("\n}\n")]
    assert "PEOPLE.stale = true" in body


def test_the_machine_dot_uses_the_existing_health_poll():
    app = (STATIC / "js" / "app.js").read_text(encoding="utf-8")
    assert "fetch('/api/health/detail'" in app and "Shell.health(" in app
    assert "setInterval" not in SHELL, "badges read existing counters, no timers of their own"


# ── The search field's dropdown (search.js) ─────────────────────────────

SEARCH = (STATIC / "js" / "search.js").read_text(encoding="utf-8")


def test_the_dropdown_has_places_hits_and_a_footer():
    assert 'id="spFoot"' in HTML and 'id="spResults"' in HTML
    for f in ("function spPlaceHits(", "function spSaidHtml(", "function spFooter("):
        assert f in SEARCH, f
    # the old filters moved to Find's Text tab
    assert 'id="spFileFilter"' not in HTML and 'id="spSpeakerFilter"' not in HTML
    assert 'id="svFileFilter"' in HTML and 'id="svSpeaker"' in HTML


def test_every_place_in_the_dropdown_exists_in_the_rail_or_a_window():
    labels = __import__("re").findall(r"\{label: '([^']+)', where: '([^']*)'", SEARCH)
    assert len(labels) >= 20
    names = {a for a, _ in labels}
    for must in (
        "Voices",
        "People",
        "Voice map",
        "Same audio",
        "Conversations",
        "Settings",
        "Setup guide",
    ):
        assert must in names, must


def test_slash_focuses_the_field_but_not_while_typing():
    body = SEARCH[SEARCH.index("e.key !== '/'") :]
    body = body[: body.index("input.focus()")]
    assert "INPUT" in body and "TEXTAREA" in body and "settingsOverlay" in body


# ── Find, Ask and Analyses: one window, three places (P2.01-P2.03) ───────

SV = (STATIC / "js" / "searchview.js").read_text(encoding="utf-8")


def test_ask_and_analyses_are_not_tabs_of_find_any_more():
    import re

    for tab in ("svTabBtnChat", "svTabBtnAnalysis"):
        tag = re.search(rf'<button[^>]*id="{tab}"[^>]*>', HTML).group(0)
        assert " hidden" in tag, tab
    assert 'id="svTitleText">Find<' in HTML


def test_the_window_shows_its_place_by_mode():
    body = SV[SV.index("function svMode(") :]
    body = body[: body.index("\n}\n")]
    assert "ov.dataset.mode = mode" in body and "tabs.hidden = mode !== 'find'" in body
    assert "chat: ['ask', 'Ask'" in SV and "analysis: ['analyses', 'Analyses'" in SV
    assert "svMode(name);" in SV[SV.index("function svShowTab(") :]


def test_no_search_view_wording_left_in_texts():
    import re

    for name in ("help.js", "stats.js", "settings_pages.js"):
        src = (STATIC / "js" / name).read_text(encoding="utf-8")
        code = "\n".join(z for z in src.splitlines() if not z.strip().startswith(("//", "*", "/*")))
        assert not re.search(r"[Ss]earch view", code), name


# ── Voices: tab order, and Conversations shared with Find (P2.04/P2.05) ──

VOICES = (STATIC / "js" / "voices.js").read_text(encoding="utf-8")


def test_the_voices_tabs_in_their_order_opening_on_people():
    import re

    nav = HTML[HTML.index('id="vxTabs"') :]
    nav = nav[: nav.index("</nav>")]
    assert re.findall(r'data-tab="([a-z]+)"', nav) == [
        "people",
        "assign",
        "history",
        "map",
        "conversations",
        "extract",
        "help",
    ]
    labels = re.findall(r"</svg>([^<]+)</button>", nav)
    assert labels == [
        "People",
        "Name voices",
        "History",
        "Voice map",
        "Conversations",
        "Voice dataset",
        "How it works",
    ]
    assert (
        "const VX_TABS = ['people', 'assign', 'history', 'map', 'conversations', 'extract', 'help'];"
        in VOICES
    )
    assert "VX_TABS.includes(t) ? t : 'people'" in VOICES


_MOVE = r"""
const vm = require('vm');
function node(id, parent) {
    const n = {id, kids: [], parentNode: null, classList: {set: new Set(),
        add(c) { this.set.add(c); }, toggle(c, on) { on ? this.set.add(c) : this.set.delete(c); }, contains(c) { return this.set.has(c); }}};
    n.appendChild = k => { if (k.parentNode) k.parentNode.kids.splice(k.parentNode.kids.indexOf(k), 1); n.kids.push(k); k.parentNode = n; return k; };
    n.insertBefore = (k, ref) => { if (k.parentNode) k.parentNode.kids.splice(k.parentNode.kids.indexOf(k), 1); n.kids.splice(n.kids.indexOf(ref), 0, k); k.parentNode = n; return k; };
    n.remove = () => { n.parentNode.kids.splice(n.parentNode.kids.indexOf(n), 1); n.parentNode = null; };
    if (parent) parent.appendChild(n);
    return n;
}
const sv = node('svBody'), page = node('vxPageConversations');
const pane = node('svPaneConversations', sv), player = node('svPlayer', sv);
const byId = {svPaneConversations: pane, svPlayer: player, vxPageConversations: page};
let opened = 0, left = 0, stopped = 0;
const sb = {document: {getElementById: id => byId[id] || null,
                       createComment: t => node('#comment ' + t)},
            svState: {tab: 'text', plays: {f: 1}}, cvOpen: () => opened++, cvLeave: () => left++,
            svWgStop: () => stopped++};
vm.createContext(sb);
vm.runInContext(SRC, sb);
const where = () => [pane.parentNode.id, player.parentNode.id, sv.kids.map(k => k.id).join(',')];
vm.runInContext('vxBorrowConversations()', sb);
const inVoices = where().concat([pane.classList.contains('active'), opened, vm.runInContext('vxConversationsHere()', sb)]);
vm.runInContext('vxReturnConversations()', sb);
const back = where().concat([pane.classList.contains('active'), left, stopped, vm.runInContext('vxConversationsHere()', sb)]);
console.log('§§' + JSON.stringify({inVoices, back}));
"""


def test_conversations_move_into_voices_and_back_to_their_place():
    import json
    import os
    import shutil
    import subprocess

    if shutil.which("node") is None:
        import pytest

        pytest.skip("node not available")
    start = VOICES.index("const VX_CV = ")
    end = VOICES.index("\n}\n", VOICES.index("function vxReturnConversations(")) + 3
    src = VOICES[start:end]
    done = subprocess.run(
        ["node", "-e", _MOVE.replace("SRC", json.dumps(src))],
        capture_output=True,
        text=True,
        timeout=60,
        env={"PATH": os.environ.get("PATH", "")},
    )
    assert done.returncode == 0, done.stderr
    d = json.loads(next(z for z in done.stdout.splitlines() if z.startswith("§§"))[2:])
    assert d["inVoices"][:2] == ["vxPageConversations", "vxPageConversations"]
    assert d["inVoices"][3:] == [True, 1, True]
    # back exactly where they were, in their order; inactive while Find shows Text
    assert d["back"][:3] == ["svBody", "svBody", "svPaneConversations,svPlayer"]
    assert d["back"][3:] == [False, 1, 1, False], "a playing conversation stops with the tab"


# ── Machine (P2.07) ──────────────────────────────────────────────────────


def test_machine_is_one_window_with_three_tabs():
    import re

    nav = HTML[HTML.index('id="mcTabs"') :]
    nav = nav[: nav.index("</nav>")]
    assert re.findall(r'data-tab="([a-z]+)"', nav) == ["health", "load", "models"]
    for page in ('id="healthBody"', 'id="mcLoad"', 'id="mcModels"'):
        assert page in HTML, page
    machine = (STATIC / "js" / "machine.js").read_text(encoding="utf-8")
    # the load history's body lives in its tab; it runs only while the tab shows
    assert (
        '<div class="stv-body mc-page" data-page="load" id="mcLoad" hidden><div class="pv-body" id="perfBody">'
        in HTML
    )
    assert "pvStart()" in machine and "pvStop()" in machine


def test_the_load_history_follows_its_tab_not_an_overlay():
    pv = (STATIC / "js" / "perfview.js").read_text(encoding="utf-8")
    assert "function pvShown() { return _pvOn; }" in pv
    assert pv.count("pvShown()") >= 4
    assert "Machine.open('load')" in pv[pv.index("function openPerfView") :]


def test_problems_are_named_with_a_way_to_fix_them():
    machine = (STATIC / "js" / "machine.js").read_text(encoding="utf-8")
    for words in (
        "does not answer",
        "not set up",
        "not installed",
        "Check the address in Settings",
        "Set it up in Settings",
    ):
        assert words in machine, words


# ── Settings deep links (P2.08) ──────────────────────────────────────────


def test_every_settings_link_in_the_code_names_a_page_or_an_alias():
    import re

    settings = (STATIC / "js" / "settings.js").read_text(encoding="utf-8")
    pages_src = (STATIC / "js" / "settings_pages.js").read_text(encoding="utf-8")
    pages = set(re.findall(r"^\s*id: '([a-z_]+)', group:", pages_src, re.M))
    sections = set(re.findall(r"_sSection\(\{id: '([a-z_-]+)'", pages_src))
    aliases = dict(
        re.findall(
            r"(\w+): '([a-z_]+(?:#[a-z_-]+)?)'",
            settings[
                settings.index("const _SETTINGS_ALIASES") : settings.index(
                    "};", settings.index("const _SETTINGS_ALIASES")
                )
            ],
        )
    )
    for target in aliases.values():
        page, _, section = target.partition("#")
        assert page in pages, target
        assert not section or section in sections, target
    calls = set()
    for f in (STATIC / "js").glob("*.js"):
        calls |= set(re.findall(r"openSettings\('([a-z_#-]+)'\)", f.read_text(encoding="utf-8")))
    calls |= set(re.findall(r"openSettings\('([a-z_#-]+)'\)", HTML))
    for c in calls:
        name, _, section = c.partition("#")
        assert name in pages or name in aliases, c
        assert not section or section in sections, c
    # a section of the Search page by its own name lands there
    for s in ("keyword", "semantic", "ask", "conversations", "wordstats"):
        assert aliases.get(s) == f"search#{s}", s
