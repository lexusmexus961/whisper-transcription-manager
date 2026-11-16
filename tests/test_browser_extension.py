"""The browser extension (browser-extension/) — what can be tried without a browser.

Wherever there is a YouTube video it can be selected — not just one but
several. The page side is exercised in a real browser; these tests pin the
parts that decide what happens:

* which links are a video, which a playlist, and which nothing — and that a
  video inside a playlist link is that video, not the whole list;
* the videos of a page: each once, from thumbnails, links, embedded players
  and a watch page's own player, with the title the page gives them; a
  relative link counts on YouTube only;
* the server side of the extension: what it sends, the batches it keeps,
  the address it accepts, the badge and the word when a batch is done;
* the manifest: Manifest V3, every file it names exists;
* the side panel: a summary's Markdown for reading, every time in
  it a link that plays the video and every chat reference a chip; the form
  speaks the server's options — "as in Whisper Transcription Manager" is empty, the speakers
  are one choice — and a video summarized already is summarized again; a
  send tells the page, whose button or bar then shows it; the page and the
  popup open the panel during their click, with what it should show.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

EXT = Path(__file__).resolve().parent.parent / "browser-extension"

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node not available")


def node(script: str) -> dict:
    done = subprocess.run(
        ["node", "-e", script],
        capture_output=True,
        text=True,
        timeout=60,
        env={"EXT": str(EXT), "PATH": os.environ.get("PATH", "")},
    )
    if done.returncode != 0:
        raise AssertionError(f"node failed:\n{done.stderr}")
    lines = [z for z in done.stdout.splitlines() if z.startswith("§§")]
    assert lines, done.stdout + done.stderr
    return json.loads(lines[-1][2:])


VIDEOS = r"""
const fs = require('fs'), vm = require('vm');
const ctx = {URL, console};
ctx.globalThis = ctx;
vm.createContext(ctx);
vm.runInContext(fs.readFileSync(process.env.EXT + '/videos.js', 'utf8'), ctx);
const W = ctx.WTM;
const out = o => console.log('§§' + JSON.stringify(o));
"""

ID = "dQw4w9WgXcQ"


@pytest.mark.parametrize(
    "href, key, kind",
    [
        (f"https://www.youtube.com/watch?v={ID}", ID, "video"),
        (f"https://www.youtube.com/watch?v={ID}&list=PLx1234567890&index=3", ID, "video"),
        (f"https://youtu.be/{ID}?si=abc", ID, "video"),
        (f"https://m.youtube.com/watch?v={ID}&t=40s", ID, "video"),
        (f"https://music.youtube.com/watch?v={ID}", ID, "video"),
        (f"https://www.youtube.com/shorts/{ID}", ID, "video"),
        (f"https://www.youtube.com/live/{ID}?feature=share", ID, "video"),
        (f"https://www.youtube-nocookie.com/embed/{ID}?rel=0", ID, "video"),
        (f"/watch?v={ID}", ID, "video"),  # relative, on YouTube
        ("https://www.youtube.com/playlist?list=PLx1234567890", "list:PLx1234567890", "playlist"),
        ("https://www.youtube.com/watch?v=short", None, None),
        ("https://www.youtube.com/@channel", None, None),
        ("https://www.youtube.com/", None, None),
        ("https://vimeo.com/123456", None, None),
        ("javascript:alert(1)", None, None),
    ],
)
def test_which_links_are_videos(href, key, kind):
    d = node(VIDEOS + f"out(W.parse({json.dumps(href)}) || {{}});")
    assert d.get("key") == key and d.get("kind") == kind
    if kind == "video":
        assert d["url"] == f"https://www.youtube.com/watch?v={ID}"


def test_videos_js_runs_twice_in_one_page():
    """background.js injects videos.js into tabs that may have it already (every open tab after an install or a
    reload of the extension): a second run in the same page must not stop at "Identifier has already been
    declared" — and must still work."""
    d = node(
        VIDEOS
        + r"""
vm.runInContext(fs.readFileSync(process.env.EXT + '/videos.js', 'utf8'), ctx);
const again = ctx.WTM;
out({key: (again.parse('https://youtu.be/%s') || {}).key,
     title: again.title({getAttribute: n => n === 'title' ? 'A title' : null, textContent: ''})});
"""
        % ID
    )
    assert d == {"key": ID, "title": "A title"}


# A page as the scan sees it: elements with what it asks of them
PAGE = r"""
function el(tag, attrs, opts) {
    opts = opts || {};
    const e = {tagName: tag, attrs: attrs || {}, textContent: opts.text || '', card: opts.card || null,
               img: opts.img || null, isConnected: true,
               getAttribute(n) { return this.attrs[n] ?? null; },
               closest(sel) { return this.card; },
               querySelector(sel) { return sel.startsWith('img') ? this.img : null; },
               getBoundingClientRect() { return opts.rect || {width: 0, height: 0}; }};
    return e;
}
function card(title) {
    const t = {getAttribute: () => null, textContent: title};
    return {querySelector: () => t};
}
function doc(anchors, iframes, extra) {
    return {title: (extra && extra.title) || '',
            querySelectorAll(sel) { return sel.startsWith('a') ? anchors : sel.startsWith('iframe') ? iframes : []; },
            querySelector(sel) { return (extra && extra.player) || null; }};
}
const show = m => [...m.values()].map(v => ({key: v.key, kind: v.kind, title: v.title, els: v.els.length}));
"""


def test_the_videos_of_a_youtube_page():
    d = node(
        VIDEOS
        + PAGE
        + f"""
        const c = card('  An Example   Talk About Models  ');
        const anchors = [
            el('a', {{href: '/watch?v={ID}'}}, {{card: c, img: {{getAttribute: () => 'x'}}}}),      // thumbnail
            el('a', {{href: '/watch?v={ID}&pp=xyz'}}, {{card: c, text: 'An Example …'}}),          // its title link
            el('a', {{href: '/watch?v=abcdefghijk'}}, {{text: 'YouTube'}}),                       // nothing better
            el('a', {{href: '/@somechannel'}}, {{text: 'Channel'}}),
            el('a', {{href: '/playlist?list=PLx1234567890'}}, {{text: 'My list'}}),
        ];
        const player = el('div', {{}}, {{}});
        const m = W.scan(doc(anchors, [], {{title: '(3) Watching this - YouTube', player}}),
                         {{href: 'https://www.youtube.com/watch?v=zzzzzzzzzzz', hostname: 'www.youtube.com'}});
        out(show(m));"""
    )
    assert d == [
        {"key": ID, "kind": "video", "title": "An Example Talk About Models", "els": 2},
        {"key": "abcdefghijk", "kind": "video", "title": "", "els": 1},
        {"key": "list:PLx1234567890", "kind": "playlist", "title": "My list", "els": 1},
        {"key": "zzzzzzzzzzz", "kind": "video", "title": "Watching this", "els": 1},
    ]


def test_a_cards_title_beats_the_end_screens():
    # The end screen inside YouTube's player comes first in the page, and its
    # links only have an aria-label with the length in it
    d = node(
        VIDEOS
        + PAGE
        + f"""
        const anchors = [
            el('a', {{href: '/watch?v={ID}', 'aria-label': 'Example Ambient Video 3 hours, 15 minutes'}}),
            el('a', {{href: '/watch?v={ID}'}}, {{card: card('Example Ambient Video')}}),
        ];
        const m = W.scan(doc(anchors, []), {{href: 'https://www.youtube.com/', hostname: 'www.youtube.com'}});
        out(show(m));"""
    )
    assert d == [{"key": ID, "kind": "video", "title": "Example Ambient Video", "els": 2}]


def test_the_watch_pages_own_video_is_titled_by_the_page():
    # Its chapter links in the player come first and carry the chapter's name
    d = node(
        VIDEOS
        + PAGE
        + f"""
        const anchors = [el('a', {{href: '/watch?v={ID}&t=0s'}}, {{text: 'Introduction'}})];
        const m = W.scan(doc(anchors, [], {{title: '(49) An example watch page - YouTube',
                                            player: el('div', {{}}, {{}})}}),
                         {{href: 'https://www.youtube.com/watch?v={ID}', hostname: 'www.youtube.com'}});
        out(show(m));"""
    )
    assert d == [{"key": ID, "kind": "video", "title": "An example watch page", "els": 2}]


def test_the_videos_of_any_other_page():
    d = node(
        VIDEOS
        + PAGE
        + f"""
        const anchors = [
            el('a', {{href: 'https://youtu.be/{ID}', title: 'A talk about models'}}),
            el('a', {{href: '/watch?v=abcdefghijk'}}, {{text: 'not YouTube here'}}),   // this site's own path
            el('a', {{href: 'https://example.com/watch?v=abcdefghijk'}}),
        ];
        const iframes = [
            el('iframe', {{src: 'https://www.youtube-nocookie.com/embed/abcdefghijk', title: 'YouTube video player'}}),
            el('iframe', {{src: 'https://maps.example.com/embed'}}),
        ];
        const m = W.scan(doc(anchors, iframes), {{href: 'https://blog.example.com/post', hostname: 'blog.example.com'}});
        out(show(m));"""
    )
    assert d == [
        {"key": ID, "kind": "video", "title": "A talk about models", "els": 1},
        {"key": "abcdefghijk", "kind": "video", "title": "", "els": 1},
    ]


def test_the_badge_goes_on_the_largest_visible_element():
    d = node(
        VIDEOS
        + PAGE
        + """
        const small = el('a', {}, {rect: {width: 120, height: 18}});
        const big = el('a', {}, {rect: {width: 360, height: 200}});
        const hidden = el('a', {}, {rect: {width: 0, height: 0}});
        out({pick: W.anchor({els: [small, hidden, big]}) === big,
             none: W.anchor({els: [hidden]}) === null});"""
    )
    assert d == {"pick": True, "none": True}


BACKGROUND = r"""
const fs = require('fs'), vm = require('vm');
const store = {sync: {}, local: {}, session: {}}, calls = [], made = [], opened = [];
let badge = '';
const area = name => ({
    get: async k => ({[k]: store[name][k]}),
    set: async o => { Object.assign(store[name], o); },
});
const chrome = {
    storage: {sync: area('sync'), local: area('local'), session: area('session'),
              onChanged: {addListener() {}}},
    sidePanel: {open: o => { opened.push(o); return Promise.resolve(); }},
    windows: {WINDOW_ID_CURRENT: -2},
    runtime: {onMessage: {addListener: f => { ctx.onMessage = f; }}, onInstalled: {addListener() {}},
              lastError: null},
    alarms: {create() {}, onAlarm: {addListener() {}}},
    notifications: {create: (id, o) => made.push(o || id), onClicked: {addListener() {}}},
    contextMenus: {removeAll() {}, create() {}, onClicked: {addListener() {}}},
    action: {setBadgeText: o => { badge = o.text; }, setBadgeBackgroundColor() {}},
    tabs: {create: o => calls.push(['tab', o.url]), sendMessage() {}},
};
let answers = {};
const fetch = async (url, init) => {
    calls.push([init && init.method || 'GET', url, init && init.body ? JSON.parse(init.body) : null]);
    const path = url.replace(/^https?:\/\/[^/]+/, '');
    const a = typeof answers[path] === 'function' ? answers[path]() : answers[path];
    return {ok: !(a && a.error), status: a && a.error ? 400 : 200, json: async () => a || {}};
};
const ctx = {chrome, fetch, console, Date, JSON, Promise, encodeURIComponent, String, Array, Object};
vm.createContext(ctx);
vm.runInContext(fs.readFileSync(process.env.EXT + '/background.js', 'utf8') + '\nthis.handle = handle; this.refresh = refresh;', ctx);
const out = o => console.log('§§' + JSON.stringify(o));
"""


def test_what_the_extension_sends_and_keeps():
    d = node(
        BACKGROUND
        + r"""
        (async () => {
            answers['/api/ingest'] = {ok: true, batch: {id: 'b1', mode: 'summarize', status: 'working',
                                                       items: [{state: 'waiting', title: ''}]}};
            const r = await ctx.handle({type: 'ingest', mode: 'summarize', urls: ['https://youtu.be/x']});
            const kept = store.local.batches;
            const bad = await ctx.handle({type: 'set-server', server: 'not an address'}).catch(e => e.message);
            const good = await ctx.handle({type: 'set-server', server: 'http://192.0.2.5:5077/'});
            await ctx.handle({type: 'open', path: '/?open=/a/b.m4a'});
            out({first: calls[0], id: r.batch.id, kept: kept.map(b => b.id), bad, good: good.server,
                 tab: calls.filter(c => c[0] === 'tab')});
        })();"""
    )
    assert d["first"] == [
        "POST",
        "http://localhost:9860/api/ingest",
        {
            "urls": ["https://youtu.be/x"],
            "mode": "summarize",
            "task": "",
            "title": "",
            "again": False,
        },
    ]
    assert d["id"] == "b1" and d["kept"] == ["b1"]
    assert d["bad"] == "An address like http://192.0.2.10:9860"
    assert d["good"] == "http://192.0.2.5:5077"
    assert d["tab"] == [["tab", "http://192.0.2.5:5077/?open=/a/b.m4a"]]


def test_the_badge_counts_what_works_and_a_finished_batch_says_so():
    d = node(
        BACKGROUND
        + r"""
        (async () => {
            store.local.batches = [{id: 'b1', last: {id: 'b1', status: 'working'}},
                                   {id: 'b2', last: {id: 'b2', status: 'working'}}];
            answers['/api/ingest/b1'] = {id: 'b1', status: 'working', mode: 'transcribe',
                                         items: [{state: 'queued'}, {state: 'transcribed'}, {state: 'downloading'}]};
            answers['/api/ingest/b2'] = {id: 'b2', status: 'done', mode: 'summarize',
                                         items: [{state: 'summarized', title: 'Talk'}]};
            await ctx.refresh();
            const first = {badge, made: made.slice()};
            await ctx.refresh();                       // b2 was said once — not again
            out({first, again: made.length, notified: store.local.batches.find(b => b.id === 'b2').notified});
        })();"""
    )
    assert d["first"]["badge"] == "2"
    assert d["first"]["made"] == [
        {
            "type": "basic",
            "iconUrl": "icons/icon128.png",
            "title": "Whisper Transcription Manager",
            "message": "1 video summarized: Talk",
        }
    ]
    assert d["again"] == 1 and d["notified"] is True


def test_the_manifest():
    m = json.loads((EXT / "manifest.json").read_text(encoding="utf-8"))
    assert m["manifest_version"] == 3
    named = [
        m["background"]["service_worker"],
        m["action"]["default_popup"],
        m["side_panel"]["default_path"],
        *m["content_scripts"][0]["js"],
        *m["icons"].values(),
        *m["action"]["default_icon"].values(),
    ]
    for f in named:
        assert (EXT / f).is_file(), f
    assert m["content_scripts"][0]["js"] == ["videos.js", "content.js"]  # the finder first
    assert set(m["permissions"]) == {
        "storage",
        "contextMenus",
        "notifications",
        "alarms",
        "scripting",
        "sidePanel",
    }
    popup = (EXT / "popup.html").read_text(encoding="utf-8")
    assert (
        '<script src="ui.js"></script>\n<script src="popup.js">' in popup
        and 'href="popup.css"' in popup
    )
    panel = (EXT / "sidepanel.html").read_text(encoding="utf-8")
    assert '<script src="ui.js"></script>\n<script src="sidepanel.js">' in panel
    assert 'href="sidepanel.css"' in panel
    # The Chrome Web Store takes 132 characters; sidePanel.open() needs 116
    assert len(m["description"]) <= 132
    assert m["minimum_chrome_version"] == "116"


def test_the_release_zips(tmp_path):
    """One to load unpacked with the fixed ID the server trusts, one for the
    Chrome Web Store, which refuses a manifest with a key."""
    import importlib.util
    import zipfile

    spec = importlib.util.spec_from_file_location(
        "extension_zip", EXT.parent / "tools" / "extension_zip.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    plain, store = mod.build(tmp_path)
    version = json.loads((EXT / "manifest.json").read_text(encoding="utf-8"))["version"]
    assert plain.name == f"whisper-tm-extension-{version}.zip"
    assert store.name == f"whisper-tm-extension-{version}-store.zip"
    for z, keyed in ((plain, True), (store, False)):
        with zipfile.ZipFile(z) as zf:
            names = zf.namelist()
            assert "manifest.json" in names and "background.js" in names
            assert "README.md" not in names
            assert ("key" in json.loads(zf.read("manifest.json"))) is keyed


POPUP = r"""
const fs = require('fs'), vm = require('vm');
const els = {};
const el = id => els[id] || (els[id] = {id, innerHTML: '', textContent: '', hidden: false, value: '',
                                        dataset: {}, open: false, src: '',
                                        addEventListener() {}, querySelectorAll: () => [], focus() {},
                                        querySelector: () => ({addEventListener() {}})});
const ctx = {
    console, Date, JSON, Promise, Set, Map, String, Number, Array, Object,
    document: {getElementById: el, addEventListener() {}},
    chrome: {runtime: {sendMessage: (m, cb) => cb({}), lastError: null},
             tabs: {query: async () => [{id: 1}], sendMessage: (id, m, cb) => cb({})},
             storage: {sync: {get: async () => ({})}, session: {set: async () => {}}}},
    setTimeout: () => 0, clearTimeout() {},
};
vm.createContext(ctx);
vm.runInContext(fs.readFileSync(process.env.EXT + '/ui.js', 'utf8') + '\n'
    + fs.readFileSync(process.env.EXT + '/popup.js', 'utf8')
    + '\nthis.renderRecent = renderRecent; this.md = md; this.P = P;', ctx);
const out = o => console.log('§§' + JSON.stringify(o));
"""


def test_the_popup_lists_what_was_sent():
    d = node(
        POPUP
        + r"""
        const rows = [{id: 'b1', at: Date.now(), last: {id: 'b1', mode: 'summarize', status: 'working', items: [
            {title: 'Talk one', state: 'summarized', summary: '## Gist\n- **Fast:** yes', open: '/?open=/u/a.m4a'},
            {title: '', url: 'https://youtu.be/x', state: 'downloading', progress: 40},
            {title: 'Broken', state: 'error', error: 'HTTP Error 403'}]}},
            {id: 'b2', at: Date.now(), last: {id: 'b2', mode: 'analyze', status: 'done', items: [],
             analysis: {id: 'a1', title: 'Compare', status: 'done', open: '/?analysis=a1'}}}];
        ctx.P.open.add('b1:0');
        ctx.renderRecent(rows);
        out({html: els.recent.innerHTML, md: ctx.md('# T\n* a **b** <i>')});"""
    )
    html = d["html"]
    assert "<b>Summarize</b> · 3 videos" in html and '<span class="chip run">working</span>' in html
    assert '<span class="chip ok" title="">summarized</span>' in html
    assert (
        '<div class="sum" data-url=""><h4>Gist</h4><ul><li><b>Fast:</b> yes</li></ul></div>' in html
    )  # unfolded
    assert 'data-open="/?open=/u/a.m4a"' in html
    assert "https://youtu.be/x" in html and "downloading 40 %" in html
    assert 'title="HTTP Error 403">failed</span>' in html
    assert "<b>Analyze</b> · 0 videos" in html and "Analysis: Compare" in html
    assert 'data-open="/?analysis=a1"' in html
    assert d["md"] == "<h4>T</h4><ul><li>a <b>b</b> &lt;i&gt;</li></ul>"  # escaped first


def test_hidden_means_hidden():
    """`.acts { display: flex }` beat the hidden attribute: the popup offered
    Summarize on a page without the helper, and the click got "Could not
    establish connection. Receiving end does not exist." — and an
    empty badge sat in the top left corner of every page."""
    assert "[hidden] { display: none !important; }" in (EXT / "popup.css").read_text(
        encoding="utf-8"
    )
    assert "[hidden] { display: none !important; }" in (EXT / "content.js").read_text(
        encoding="utf-8"
    )


def test_our_elements_take_the_pages_visibility():
    """A ⊕ behind a YouTube link in a menu hidden with visibility stood
    alone in the page — ``all: initial`` sets visibility to visible. Every element the extension
    puts INTO the page takes the page's visibility along; the two overlays
    hang on the document itself."""
    js = (EXT / "content.js").read_text(encoding="utf-8")
    hosts = re.findall(r":host \{ all: initial;[^}]*\}", js)
    in_page = [h for h in hosts if "position: fixed" not in h and h != ":host { all: initial; }"]
    assert len(in_page) == 3, in_page  # the ⊕, the action button, the panel
    assert all("visibility: inherit" in h for h in in_page), in_page
    assert js.count("(document.documentElement || document.body).appendChild(") == 2


def test_open_tabs_get_the_helper():
    """A page open from before the install (or a reload of the extension) has
    no content script: the extension brings it in — into every open page on
    install, into the popup's page when it doesn't answer."""
    d = node(
        BACKGROUND
        + r"""
        (async () => {
            const injected = [];
            chrome.tabs.query = async q => [{id: 3}, {id: 4}];
            chrome.scripting = {executeScript: async o => {
                if (o.target.tabId === 4) throw new Error('Cannot access a chrome:// URL');
                injected.push([o.target.tabId, o.files]);
            }};
            const one = await ctx.handle({type: 'inject', tabId: 3});
            const all = await ctx.handle({type: 'inject'});
            out({one, all, injected});
        })();"""
    )
    assert d["one"] == {"ok": True} and d["all"] == {"ok": True}
    assert d["injected"] == [[3, ["videos.js", "content.js"]], [3, ["videos.js", "content.js"]]]


def test_the_popup_brings_the_helper_and_hides_what_cant_go():
    d = node(
        POPUP.replace(
            "tabs: {query: async () => [{id: 1}], sendMessage: (id, m, cb) => cb({})}",
            "tabs: {query: async () => [{id: 1}], "
            "sendMessage: (id, m, cb) => (ctx.answer || ((m2, c) => c({})))(m, cb)}",
        ).replace(
            "chrome: {runtime: {sendMessage: (m, cb) => cb({}), lastError: null},",
            "chrome: {runtime: {sendMessage: (m, cb) => cb(m.type === 'inject' ? {ok: ctx.canInject} : {}), lastError: null},",
        )
        + r"""
        (async () => {
            // The popup asks on its own when it opens — let that pass first
            for (let i = 0; i < 5; i++) await new Promise(r => setImmediate(r));
            const asked = [];
            ctx.answer = (m, cb) => {
                asked.push(m.type);
                if (!ctx.injected) { ctx.chrome.runtime.lastError = {message: 'Could not establish connection. Receiving end does not exist.'}; cb(); ctx.chrome.runtime.lastError = null; return; }
                cb({videos: [{key: 'a', url: 'u', title: 'T', selected: false}]});
            };
            // Injection works: asked again, the videos are there
            ctx.canInject = true;
            const origAsk = ctx.chrome.runtime.sendMessage;
            ctx.chrome.runtime.sendMessage = (m, cb) => { if (m.type === 'inject') ctx.injected = true; origAsk(m, cb); };
            await vm.runInContext('loadVideos()', ctx);
            const first = {asked: asked.slice(), videos: ctx.P.videos.length};
            // A page that takes no extension: a word, no buttons
            ctx.injected = false; ctx.canInject = false; asked.length = 0;
            ctx.chrome.runtime.sendMessage = origAsk;
            await vm.runInContext('loadVideos()', ctx);
            out({first, second: {asked, acts: els.acts.hidden, note: els.videos.innerHTML}});
        })();"""
    )
    assert d["first"] == {"asked": ["list", "list"], "videos": 1}
    assert d["second"]["asked"] == ["list"] and d["second"]["acts"] is True
    assert "takes no extension" in d["second"]["note"]


def test_the_popup_puts_the_playing_video_first():
    """The main video should not have to be searched for among the others:
    the video a watch page plays on top, with its picture and its summary;
    the others folded away below it."""
    d = node(
        POPUP.replace(
            "tabs: {query: async () => [{id: 1}], sendMessage: (id, m, cb) => cb({})}",
            "tabs: {query: async () => [{id: 1}], sendMessage: (id, m, cb) => cb(ctx.list)}",
        )
        + r"""
        (async () => {
            for (let i = 0; i < 5; i++) await new Promise(r => setImmediate(r));
            ctx.list = {main: {key: 'mainVideo01', url: 'https://www.youtube.com/watch?v=mainVideo01',
                               title: 'An example watch page', state: 'summarized',
                               summary: '- **Fast**', open: '/?open=/u/a.m4a'},
                        videos: [{key: 'mainVideo01', main: true, url: 'x', title: 'An example'},
                                 {key: 'abcdefghijk', id: 'abcdefghijk', url: 'y', title: 'Other one', selected: false},
                                 {key: 'bbbbbbbbbbb', id: 'bbbbbbbbbbb', url: 'z', title: 'Third', selected: true,
                                  known: {state: 'transcribed'}}]};
            await vm.runInContext('loadVideos()', ctx);
            out({main: !els.mainBox.hidden, thumb: els.mainThumb.src, title: els.mainTitle.textContent,
                 state: els.mainState.innerHTML, sumLink: !els.mainSum.hidden, open: !els.mainOpen.hidden,
                 folded: els.others.open, count: els.pageCount.textContent, list: els.videos.innerHTML,
                 acts: !els.acts.hidden, sel: els.pickHint.textContent});
        })();"""
    )
    assert d["main"] and d["title"] == "An example watch page"
    assert d["thumb"] == "https://i.ytimg.com/vi/mainVideo01/mqdefault.jpg"
    assert '<span class="chip ok">summarized</span>' in d["state"]
    assert d["sumLink"] and d["open"]
    assert d["folded"] is False and d["count"] == "2"  # the playing one is not among them
    assert "Other one" in d["list"] and "An example" not in d["list"]
    assert 'src="https://i.ytimg.com/vi/abcdefghijk/default.jpg"' in d["list"]
    assert '<span class="chip ok">transcribed</span>' in d["list"]
    assert d["acts"] and d["sel"] == "1 selected"


# ══════════════════════════════════════════════════════════════════════════
#  The side panel
# ══════════════════════════════════════════════════════════════════════════

UI = r"""
const fs = require('fs'), vm = require('vm');
const ctx = {console, JSON, String, Number, Math, Array, Object, RegExp, Date};
vm.createContext(ctx);
vm.runInContext(fs.readFileSync(process.env.EXT + '/ui.js', 'utf8') + '\nthis.md = md; this.clock = clock;', ctx);
const out = o => console.log('§§' + JSON.stringify(o));
"""


def test_markdown_for_reading_with_times_that_play():
    d = node(
        UI
        + r"""
        out({html: ctx.md('## Gist\n- **Fast** at [0:04:12]\n- *quick* `x`\n1. one\n\nplain <b>x</b> [S1] [S9]\n'
                          + '**[1:02:03]** — later', {S1: {text: 'said "so"'}}),
             clock: [ctx.clock(5), ctx.clock(252), ctx.clock(3723)]});"""
    )
    html = d["html"]
    assert html.startswith(
        "<h4>Gist</h4><ul><li><b>Fast</b> at "
        '<a class="time" href="#" data-t="252" title="Play from 0:04:12">4:12</a></li>'
    )
    assert "<li><i>quick</i> <code>x</code></li></ul><ol><li>one</li></ol>" in html
    # Escaped first; a reference the conversation knows becomes a chip, another stays text
    assert (
        '<p>plain &lt;b&gt;x&lt;/b&gt; <a class="ref" href="#" data-ref="S1" '
        'title="said &quot;so&quot;">S1</a> [S9]<br>'
    ) in html
    assert (
        '<b><a class="time" href="#" data-t="3723" title="Play from 1:02:03">1:02:03</a></b> — later'
        in html
    )
    assert d["clock"] == ["0:05", "4:12", "1:02:03"]


SIDEPANEL = r"""
const fs = require('fs'), vm = require('vm');
const els = {};
const el = id => els[id] || (els[id] = {id, innerHTML: '', textContent: '', hidden: false, value: '',
    checked: false, className: '', title: '', src: '', disabled: false, dataset: {}, style: {},
    firstElementChild: {style: {}}, scrollTop: 0, scrollHeight: 0, clientHeight: 0,
    addEventListener() {}, querySelectorAll: () => [], querySelector: () => null, scrollIntoView() {},
    focus() {}});
const sent = [], paged = [], tabs = [];
const store = {sync: {}, session: {}};
const area = name => ({get: async k => ({[k]: store[name][k]}), set: async o => { Object.assign(store[name], o); },
                       remove: async k => { delete store[name][k]; }});
const chrome = {
    runtime: {sendMessage: (m, cb) => { sent.push(m); Promise.resolve((ctx.answer || (() => ({})))(m)).then(cb); },
              lastError: null, onMessage: {addListener: f => { ctx.heard = f; }}},
    tabs: {query: async () => [], sendMessage: (id, m, cb) => { paged.push(m); cb({}); },
           create: o => tabs.push(o.url), onActivated: {addListener() {}}, onUpdated: {addListener() {}}},
    storage: {sync: area('sync'), session: area('session'), onChanged: {addListener() {}}},
};
const document = {getElementById: el, querySelector: () => null, querySelectorAll: () => [],
                  addEventListener() {}};
const ctx = {chrome, document, console, Date, JSON, Promise, Math, Number, String, Array, Object, RegExp, Set,
             setTimeout: () => 0, clearTimeout() {}, setInterval: () => 0, clearInterval() {},
             encodeURIComponent, encodeURI};
vm.createContext(ctx);
vm.runInContext(fs.readFileSync(process.env.EXT + '/ui.js', 'utf8') + '\n'
    + fs.readFileSync(process.env.EXT + '/sidepanel.js', 'utf8')
    + '\nthis.P = P; this.fieldsHtml = fieldsHtml; this.readFields = readFields; this.send = send;'
    + ' this.openForm = openForm; this.setV = v => { V = v; }; this.getV = () => V;'
    + ' this.renderSummary = renderSummary; this.highlight = highlight; this.take = take;'
    + ' this.pollVideo = pollVideo; this.loadConv = loadConv; this.loadTab = loadTab; this.C = C;'
    + ' this.renderRecent = renderRecent; this.renderPage = renderPage; this.pickAllLabel = pickAllLabel;'
    + ' this.takeList = takeList;', ctx);
const OPTS = {
    whisper_models: [{id: 'turbo', label: 'turbo — fast'}, {id: 'large-v3', label: 'large-v3'}],
    languages: [{id: 'de', label: 'German'}, {id: 'en', label: 'English'}],
    vocab_presets: ['Gaming'], summary_presets: ['Summary', 'Bullet Points'], summary_models: [],
    general: {model: 'large-v3', language: 'de', diarize: false, numSpeakers: 0, vocal_isolation: true,
              vocab_preset: '', summary_preset: '', summary_model: '', summary_language: '', audio_only: false},
    defaults: {model: '', language: 'auto', diarize: null, numSpeakers: null, vocal_isolation: null,
               vocab_preset: '', summary_preset: '', summary_model: '', summary_language: ''},
};
const URL1 = 'https://www.youtube.com/watch?v=mainVideo01';
const out = o => console.log('§§' + JSON.stringify(o));
"""


def test_the_form_speaks_the_servers_options():
    d = node(
        SIDEPANEL
        + r"""
        ctx.P.opts = OPTS;
        const html = ctx.fieldsHtml({summary_preset: 'Bullet Points', summary_language: 'de', model: '',
            language: 'auto', diarize: true, numSpeakers: 2, vocal_isolation: false, vocab_preset: ''},
            {summary: true, transcription: true});
        const box = {querySelectorAll: () => [
            {dataset: {k: 'speakers'}, value: 'auto'}, {dataset: {k: 'vocal_isolation'}, value: ''},
            {dataset: {k: 'summary_language'}, value: 'de'}, {dataset: {k: 'model'}, value: 'turbo'}],
            querySelector: () => null};
        out({html, read: ctx.readFields(box)});"""
    )
    html = d["html"]
    assert '<option value="Bullet Points" selected>Bullet Points</option>' in html
    assert '<option value="">As in Whisper Transcription Manager (its default)</option>' in html
    assert '<option value="de" selected>German</option>' in html  # written in German
    assert '<option value="auto" selected>Detect automatically</option>' in html
    assert '<option value="">As in Whisper Transcription Manager (large-v3)</option>' in html
    assert '<option value="2" selected>2 speakers</option>' in html
    assert '<option value="off" selected>Off</option>' in html  # no vocal isolation
    assert '<option value="">As in Whisper Transcription Manager (German)</option>' in html
    assert d["read"] == {
        "diarize": True,
        "numSpeakers": 0,
        "vocal_isolation": None,
        "summary_language": "de",
        "model": "turbo",
    }


def test_custom_writes_its_own_instruction():
    """The template's list ends on "Custom…", which opens a field for an own
    instruction — sent instead of a template, kept as a default like any other choice."""
    d = node(
        SIDEPANEL
        + r"""
        ctx.P.opts = OPTS;
        const plain = ctx.fieldsHtml({summary_preset: 'Bullet Points'}, {summary: true});
        const mine = ctx.fieldsHtml({summary_prompt: 'Only the jokes <b>'}, {summary: true});
        const box = (preset, text) => ({
            querySelectorAll: () => [{dataset: {k: 'summary_preset'}, value: preset}],
            querySelector: () => ({value: text})});
        out({plain, mine, custom: ctx.readFields(box('__custom__', '  Only the jokes.  ')),
             template: ctx.readFields(box('Bullet Points', 'left over'))});"""
    )
    assert '<option value="__custom__">Custom…</option>' in d["plain"]
    assert (
        'class="own" data-k="summary_prompt"' in d["plain"]
        and "hidden" in d["plain"].split('data-k="summary_prompt"')[1][:120]
    )
    assert '<option value="__custom__" selected>Custom…</option>' in d["mine"]
    assert "Only the jokes &lt;b&gt;</textarea>" in d["mine"]  # escaped
    assert d["custom"] == {"summary_preset": "", "summary_prompt": "Only the jokes."}
    assert d["template"] == {"summary_preset": "Bullet Points", "summary_prompt": ""}


def test_a_send_from_the_panel_tells_the_page():
    d = node(
        SIDEPANEL
        + r"""
        (async () => {
            ctx.P.tab = {id: 3};
            ctx.P.playing = 'mainVideo01';
            ctx.setV({url: URL1, id: 'mainVideo01', title: 'T', state: '', open: '', abs: '', data: null,
                      sum: 0, batch: '', progress: 0, error: ''});
            ctx.answer = m => m.type === 'ingest' ? {batch: {id: 'b1', status: 'working',
                items: [{url: URL1, state: 'downloading', progress: 10}]}} : {};
            sent.length = 0;
            await ctx.send('summarize', [URL1], {summary_language: 'de'}, false);
            const one = {ingest: sent.find(m => m.type === 'ingest'), page: paged.slice(-1)[0].type,
                         state: ctx.getV().state};
            await ctx.send('transcribe', [URL1, 'https://www.youtube.com/watch?v=abcdefghijk'], undefined, false, true);
            const two = paged.slice(-1)[0], before = paged.length;
            // One video picked in the panel (Sent, Page) — not the page's selection: the page is left alone
            ctx.answer = m => m.type === 'ingest' ? {batch: {id: 'b3', status: 'working',
                items: [{url: 'https://www.youtube.com/watch?v=abcdefghijk', state: 'waiting'}]}} : {};
            await ctx.send('summarize', ['https://www.youtube.com/watch?v=abcdefghijk'], undefined, false);
            out({one, two, three: paged.length - before});
        })();"""
    )
    assert d["one"]["ingest"] == {
        "type": "ingest",
        "mode": "summarize",
        "urls": ["https://www.youtube.com/watch?v=mainVideo01"],
        "options": {"summary_language": "de"},
        "again": False,
    }
    assert d["one"]["page"] == "follow-main" and d["one"]["state"] == "downloading"
    assert d["two"]["type"] == "follow" and d["two"]["clear"] is True  # the page's bar shows them
    assert d["three"] == 0  # its selection stays


def test_one_summarized_is_summarized_again():
    d = node(
        SIDEPANEL
        + r"""
        (async () => {
            ctx.P.opts = OPTS;
            ctx.setV({url: URL1, id: 'mainVideo01', title: 'T', state: 'summarized', open: '', abs: '',
                      data: null, sum: 0, batch: '', progress: 0, error: ''});
            await ctx.openForm({mode: 'transcribe', urls: [URL1], titles: ['T']});
            out({form: ctx.P.form, html: els.form.innerHTML});
        })();"""
    )
    assert d["form"]["again"] is True and d["form"]["mode"] == "summarize"
    assert "Summarize again" in d["html"] and "Transcribed already" in d["html"]
    assert 'data-k="model"' not in d["html"]  # nothing to transcribe


def test_the_page_and_the_popup_open_the_panel_during_their_click():
    """Chrome lets an extension open its side panel only during the click
    that asked — so it is opened before anything is awaited, and what to show
    goes along in the session storage."""
    d = node(
        BACKGROUND
        + r"""
        (async () => {
            let answered = null;
            ctx.onMessage({type: 'panel', intent: {form: 'summarize', url: 'u'}}, {tab: {id: 7}},
                          r => { answered = r; });
            const now = opened.slice();                 // before anything was awaited
            await new Promise(r => setTimeout(r, 0));
            out({now, intent: store.session.panelIntent, answered});
        })();"""
    )
    assert d["now"] == [{"tabId": 7}]
    assert (
        d["intent"]["form"] == "summarize"
        and d["intent"]["url"] == "u"
        and d["intent"]["tabId"] == 7
    )
    assert d["answered"] == {"ok": True}


def test_the_side_panels_questions_reach_the_server():
    d = node(
        BACKGROUND
        + r"""
        (async () => {
            for (const m of [{type: 'options'}, {type: 'set-defaults', defaults: {summary_language: 'de'}},
                             {type: 'transcript', url: 'u', title: 't'}, {type: 'ask', url: 'u', new: true},
                             {type: 'chat-conv', id: 'c1'}, {type: 'chat-send', text: 'Why?', conversation: 'c1'},
                             {type: 'chat-run', id: 'r1', after: 4}, {type: 'chat-stop', id: 'r1'},
                             {type: 'ingest', mode: 'summarize', urls: ['u'], options: {model: 'turbo'}, again: true}]) {
                answers[{}] = null;
                await ctx.handle(m).catch(() => {});
            }
            out(calls.map(c => [c[0], c[1].replace('http://localhost:9860', ''), c[2]]));
        })();"""
    )
    assert d == [
        ["GET", "/api/ingest/options", None],
        ["POST", "/api/ingest/defaults", {"summary_language": "de"}],
        ["POST", "/api/ingest/transcript", {"url": "u", "title": "t"}],
        ["POST", "/api/ingest/ask", {"url": "u", "title": "", "new": True}],
        ["GET", "/api/chat/conversations/c1", None],
        ["POST", "/api/chat/send", {"text": "Why?", "conversation": "c1"}],
        ["GET", "/api/chat/runs/r1?after=4", None],
        ["POST", "/api/chat/runs/r1/stop", None],
        [
            "POST",
            "/api/ingest",
            {
                "urls": ["u"],
                "mode": "summarize",
                "task": "",
                "title": "",
                "options": {"model": "turbo"},
                "again": True,
            },
        ],
    ]


def test_a_video_not_in_the_library_can_be_summarized_or_transcribed():
    """Two buttons: summarize and transcribe — and a form to choose how;
    one transcribed already only gets summarized."""
    d = node(
        SIDEPANEL
        + r"""
        const base = {url: URL1, id: 'mainVideo01', title: 'T', open: '', abs: '', data: null, sum: 0,
                      batch: '', progress: 0, error: ''};
        ctx.setV({...base, state: ''});
        ctx.renderSummary();
        const fresh = els['pane-summary'].innerHTML;
        ctx.setV({...base, state: 'transcribed'});
        ctx.renderSummary();
        out({fresh, transcribed: els['pane-summary'].innerHTML});"""
    )
    assert 'data-send="summarize"' in d["fresh"] and 'data-send="transcribe"' in d["fresh"]
    assert 'data-form="summarize"' in d["fresh"]
    assert 'data-send="summarize"' in d["transcribed"]
    assert 'data-send="transcribe"' not in d["transcribed"]


def test_round_things_stay_round():
    """The server's dot and the bar's logo sit in flex rows — squeezed by a
    narrow window, they turned oval."""
    for f in ("popup.css", "sidepanel.css"):
        assert re.search(r"\.dot \{[^}]*flex: none", (EXT / f).read_text(encoding="utf-8")), f
    assert re.search(r"\.logo \{[^}]*flex: none", (EXT / "content.js").read_text(encoding="utf-8"))


# ══════════════════════════════════════════════════════════════════════════
#  1.3.0 — what a web page may not do, and answers that come too late
# ══════════════════════════════════════════════════════════════════════════


def test_a_finished_batch_says_so_even_when_a_page_saw_it_first():
    """The page's bar asks every few seconds: it saw the batch finish before
    the half-minute check, which then found nothing working to tell about —
    the word never came while the tab was open."""
    d = node(
        BACKGROUND
        + r"""
        (async () => {
            store.local.batches = [{id: 'b1', last: {id: 'b1', status: 'working'}}];
            answers['/api/ingest/b1'] = {id: 'b1', status: 'done', mode: 'transcribe',
                                         items: [{state: 'transcribed', title: 'Talk'}]};
            await ctx.handle({type: 'batch', id: 'b1'});         // the page's bar
            await ctx.handle({type: 'batch', id: 'b1'});         // again — said once
            await ctx.refresh();
            out({made: made.map(m => m.message), notified: store.local.batches[0].notified});
        })();"""
    )
    assert d == {"made": ["1 video transcribed: Talk"], "notified": True}


def test_batches_kept_at_the_same_moment_are_all_kept():
    """Two read-change-write rounds at once lost the batch one of them added."""
    d = node(
        BACKGROUND
        + r"""
        (async () => {
            answers['/api/ingest/b1'] = {id: 'b1', status: 'working', items: []};
            answers['/api/ingest/b2'] = {id: 'b2', status: 'working', items: []};
            answers['/api/ingest/b3'] = {id: 'b3', status: 'working', items: []};
            await Promise.all(['b1', 'b2', 'b3'].map(id => ctx.handle({type: 'batch', id})));
            out(store.local.batches.map(b => b.id).sort());
        })();"""
    )
    assert d == ["b1", "b2", "b3"]


@pytest.mark.parametrize(
    "path, opened",
    [
        ("/?open=/a/b.m4a", "/?open=/a/b.m4a"),
        ("@evil.example/", "/"),
        ("//evil.example/", "/"),
        ("/\\evil.example/", "/"),
        ("javascript:alert(1)", "/"),
        ("/x\nSet-Cookie: a", "/"),
    ],
)
def test_open_stays_in_the_app(path, opened):
    """Behind the server's address, "@evil.example" opened another site
    (http://localhost:9860@evil.example)."""
    d = node(
        BACKGROUND
        + r"""
        (async () => {
            await ctx.handle({type: 'open', path: %s});
            out(calls.filter(c => c[0] === 'tab').map(c => c[1]));
        })();"""
        % json.dumps(path)
    )
    assert d == ["http://localhost:9860" + opened]


def test_a_page_asks_only_for_what_its_buttons_need():
    """The helper in a web page may send, follow, cancel and look up — not
    change the server's address, the defaults, or chat; the popup and the
    side panel (extension pages) may."""
    d = node(
        BACKGROUND
        + r"""
        chrome.runtime.id = 'ext';
        chrome.runtime.getURL = p => 'chrome-extension://ext/' + p;
        const askAs = (msg, sender) => new Promise(r => ctx.onMessage(msg, sender, r));
        (async () => {
            const page = {id: 'ext', tab: {id: 5}, url: 'https://evil.example/'};
            const panel = {id: 'ext', url: 'chrome-extension://ext/sidepanel.html'};
            answers['/api/ingest/known'] = {videos: []};
            out({
                setServer: await askAs({type: 'set-server', server: 'http://evil.example'}, page),
                forgetToken: await askAs({type: 'forget-token'}, page),
                server: await askAs({type: 'server'}, page),
                chat: await askAs({type: 'chat-send', text: 'x', conversation: 'c'}, page),
                known: await askAs({type: 'known', videos: []}, page),
                fromPanel: await askAs({type: 'set-server', server: 'http://192.0.2.5:5077'}, panel),
                other: await askAs({type: 'recent'}, {id: 'another-extension'}),
            });
        })();"""
    )
    assert d["setServer"]["error"].startswith("Not asked from a page")
    assert d["forgetToken"]["error"].startswith("Not asked from a page")
    assert d["server"]["error"].startswith("Not asked from a page")
    assert d["chat"]["error"].startswith("Not asked from a page")
    assert d["known"] == {"videos": []}
    assert d["fromPanel"] == {"server": "http://192.0.2.5:5077", "token": False}
    assert d["other"] == {"error": "Not from this extension"}


def test_with_the_login_on_the_token_goes_along():
    """With the server's login on, every call got 401: the extension had no
    way to send its token. It sends X-API-Token now — kept in this browser
    only (not synced), never handed out: the panel learns that one is set."""
    d = node(
        BACKGROUND
        + r"""
        const seen = [];
        ctx.fetch = async (url, init) => { seen.push(init.headers['X-API-Token'] || null);
            return {ok: true, status: 200, json: async () => ({status: 'ok'})}; };
        vm.runInContext('fetch = this.fetch;', ctx);
        (async () => {
            await ctx.handle({type: 'health'});
            const before = await ctx.handle({type: 'server'});
            const saved = await ctx.handle({type: 'set-server', server: 'http://192.0.2.5:9860', token: ' s3cret '});
            await ctx.handle({type: 'health'});
            const kept = await ctx.handle({type: 'set-server', server: 'http://192.0.2.5:9860', token: ''});
            await ctx.handle({type: 'health'});
            const forgot = await ctx.handle({type: 'forget-token'});
            await ctx.handle({type: 'health'});
            out({seen, before, saved, kept, forgot, synced: store.sync.token || null,
                 local: store.local.token});
        })();"""
    )
    assert d["seen"] == [None, "s3cret", "s3cret", None]
    assert d["before"]["token"] is False
    assert d["saved"] == {"server": "http://192.0.2.5:9860", "token": True}
    assert d["kept"]["token"] is True and d["forgot"] == {"token": False}
    assert d["synced"] is None and d["local"] == ""
    assert "s3cret" not in json.dumps([d["before"], d["saved"], d["kept"]])


def test_a_server_with_its_login_on_says_what_to_do():
    d = node(
        BACKGROUND
        + r"""
        ctx.fetch = async () => ({ok: false, status: 401, json: async () => ({error: 'Not authenticated'})});
        vm.runInContext('fetch = this.fetch;', ctx);
        (async () => {
            const without = await ctx.handle({type: 'health'}).catch(e => e.message);
            await ctx.handle({type: 'set-server', server: '', token: 'wrong'});
            const wrong = await ctx.handle({type: 'health'}).catch(e => e.message);
            out({without, wrong});
        })();"""
    )
    assert d["without"] == (
        "The server has its login on — enter its access token (auth.api_token) under ⚙"
    )
    assert d["wrong"] == "The server refused the access token — check it under ⚙"


def test_a_server_that_hangs_ends_the_wait():
    d = node(
        BACKGROUND
        + r"""
        ctx.AbortSignal = {timeout: ms => ({ms})};
        let signal = null;
        ctx.fetch = async (url, init) => { signal = init.signal; const e = new Error('x'); e.name = 'TimeoutError'; throw e; };
        vm.runInContext('fetch = this.fetch; AbortSignal = this.AbortSignal;', ctx);
        (async () => {
            const e = await ctx.handle({type: 'health'}).catch(e => e.message);
            out({e, signal});
        })();"""
    )
    assert d["e"] == "The server at http://localhost:9860 didn't answer in time"
    assert d["signal"] == {"ms": 20000}


def test_a_search_in_the_transcript_marks_the_text_not_its_escaping():
    """Searching "amp" marked the inside of "&amp;"."""
    d = node(
        SIDEPANEL
        + r"""
        out([ctx.highlight('Q&A about amp <b>', 'amp'), ctx.highlight('a<b', 'a<'), ctx.highlight('x', '')]);"""
    )
    assert d == ["Q&amp;A about <mark>amp</mark> &lt;b&gt;", "<mark>a&lt;</mark>b", "x"]


def test_late_answers_stay_with_their_video():
    """The panel switched to another video while a question was out: the
    old batch's state, the old conversation landed on the new one."""
    d = node(
        SIDEPANEL
        + r"""
        (async () => {
            const base = {title: 'T', open: '', abs: '', data: null, sum: 0, progress: 0, error: ''};
            // A batch with another video only: nothing of it is this one's
            ctx.setV({...base, url: URL1, id: 'mainVideo01', state: '', batch: ''});
            ctx.take({status: 'working', items: [{url: 'https://www.youtube.com/watch?v=abcdefghijk', state: 'summarized'}]});
            const other = ctx.getV().state;
            // The batch asked, another video shown before the answer
            let answerBatch, answerConv;
            ctx.answer = m => m.type === 'batch' ? new Promise(r => { answerBatch = r; })
                            : m.type === 'chat-conv' ? new Promise(r => { answerConv = r; }) : {};
            ctx.setV({...base, url: URL1, id: 'mainVideo01', state: 'queued', batch: 'b1'});
            const polling = ctx.pollVideo(), talking = ctx.loadConv('c1');
            await new Promise(r => setImmediate(r));
            ctx.setV({...base, url: 'https://www.youtube.com/watch?v=abcdefghijk', id: 'abcdefghijk', state: '', batch: ''});
            answerBatch({batch: {id: 'b1', status: 'done', items: [{url: URL1, state: 'summarized'}]}});
            answerConv({id: 'c1', turns: []});
            await polling; await talking;
            out({other, state: ctx.getV().state, conv: ctx.C.conv});
        })();"""
    )
    assert d == {"other": "", "state": "", "conv": None}


def test_only_the_latest_tab_counts():
    """Switching tabs fast: the slower answer of the tab left showed its
    video in the panel."""
    d = node(
        SIDEPANEL
        + r"""
        (async () => {
            for (let i = 0; i < 10; i++) await new Promise(r => setImmediate(r));    // the panel's own start first
            let tab = {id: 1}, slow;
            chrome.tabs.query = async () => [tab];
            chrome.tabs.sendMessage = (id, m, cb) => {
                const main = id === 1 ? 'https://www.youtube.com/watch?v=aaaaaaaaaaa' : 'https://www.youtube.com/watch?v=bbbbbbbbbbb';
                const answer = {main: {url: main, title: 'tab ' + id, state: 'summarized'}, videos: []};
                if (id === 1) slow = () => cb(answer); else cb(answer);
            };
            const first = ctx.loadTab();
            await new Promise(r => setImmediate(r));
            tab = {id: 2};
            await ctx.loadTab();
            slow();
            await first;
            out({tab: ctx.P.tab.id, video: ctx.getV().id, playing: ctx.P.playing});
        })();"""
    )
    assert d == {"tab": 2, "video": "bbbbbbbbbbb", "playing": "bbbbbbbbbbb"}


def test_a_reference_opens_any_file_name():
    js = (EXT / "sidepanel.js").read_text(encoding="utf-8")
    assert "'/?open=' + encodeURIComponent(ref.abs_path)" in js and "encodeURI(" not in js


# content.js in a page made of stand-ins: what it builds, what it asks, what it answers
CONTENT = r"""
const fs = require('fs'), vm = require('vm');
const sent = [], made = [], removed = [];
let hold = null;
function E(tag, o) {
    o = o || {};
    const e = {
        tagName: tag.toUpperCase(), nodeType: 1, attrs: {...(o.attrs || {})}, style: {}, dataset: {},
        listeners: {}, textContent: o.text || '', hidden: false, className: '', title: '', isConnected: true,
        isContentEditable: !!o.editable, rects: o.rects ?? 1,
        rect: o.rect || {left: 0, top: 0, right: 100, bottom: 20, width: 100, height: 20},
        nextElementSibling: null, parentElement: null, shadowRoot: null, _q: {}, _html: '',
        getAttribute(n) { return this.attrs[n] ?? null; },
        setAttribute(n, v) { this.attrs[n] = String(v); },
        toggleAttribute(n, on) { if (on) this.attrs[n] = ''; else delete this.attrs[n]; },
        addEventListener(t, f) { (this.listeners[t] = this.listeners[t] || []).push(f); },
        closest(sel) { return o.closest ? o.closest(sel) : null; },
        matches() { return false; },
        getClientRects() { return {length: this.rects}; },
        getBoundingClientRect() { return this.rect; },
        attachShadow(opt) {
            const r = E('#shadow'); r.host = this; this.mode = opt.mode; this._shadow = r;
            if (opt.mode === 'open') this.shadowRoot = r;
            return r;
        },
        querySelector(sel) { return this._q[sel] || (this._q[sel] = E('q')); },
        querySelectorAll() { return []; },
        after(n) { this.nextElementSibling = n; n.isConnected = true; },
        remove() { removed.push(this); this.isConnected = false; },
        appendChild(n) {},
        getRootNode() { return {}; },
        focus() {},
        contains(n) { return n === this || (o.kids || []).includes(n); },
        set innerHTML(v) { this._html = String(v); this._q = {}; },
        get innerHTML() { return this._html; },
    };
    return e;
}
const links = [];
const document = {
    documentElement: E('html'), body: E('body'), title: '',
    createElement: t => { const e = E(t); e.isConnected = false; made.push(e); return e; },
    querySelectorAll: sel => sel === 'a[href]' ? links
        : sel === 'wtm-pick' ? made.filter(e => e.tagName === 'WTM-PICK' && e.isConnected) : [],
    querySelector: () => null,
    addEventListener() {},
};
const known = {};
let onMessage = null;
const chrome = {
    runtime: {id: 'ext', lastError: null, onMessage: {addListener: f => { onMessage = f; }},
              sendMessage: (m, cb) => {
                  sent.push(m);
                  if (hold && hold(m)) { hold.cb = () => cb(hold.answer); return; }
                  cb(m.type === 'known' ? {videos: m.videos.filter(v => known[v.url]).map(v => ({url: v.url, state: known[v.url]}))} : {});
              }},
    storage: {sync: {get: async () => ({})}, onChanged: {addListener() {}}},
};
const ctx = {
    console, JSON, Promise, Map, Set, WeakMap, URL, Math, Number, String, Array, Object, RegExp, Date,
    document, chrome, location: {hostname: 'blog.example.com', href: 'https://blog.example.com/post', pathname: '/post'},
    MutationObserver: class { constructor(f) { ctx.observed = f; } observe(t, o) { ctx.observeOptions = o; } disconnect() {} },
    getComputedStyle: () => ({backgroundColor: ctx.bg || 'rgb(255, 255, 255)'}),
    matchMedia: () => ({matches: false, addEventListener: (t, f) => { ctx.schemeChange = f; }}),
    setTimeout: () => 0, clearTimeout() {}, requestAnimationFrame: () => 0, innerHeight: 800, innerWidth: 1200,
    addEventListener() {},
};
ctx.window = ctx; ctx.top = ctx; ctx.globalThis = ctx;
vm.createContext(ctx);
const load = () => {
    vm.runInContext(fs.readFileSync(process.env.EXT + '/videos.js', 'utf8'), ctx);
    vm.runInContext(fs.readFileSync(process.env.EXT + '/content.js', 'utf8'), ctx);
};
const ask = msg => new Promise(r => onMessage(msg, {}, r));
const button = pick => pick._shadow.querySelector('button');
const click = (el, trusted, target) => {
    const done = {prevented: false, stopped: false};
    (el.listeners.click || []).forEach(f => f({isTrusted: trusted, target: target || el,
        preventDefault() { done.prevented = true; }, stopPropagation() { done.stopped = true; }}));
    return done;
};
const tick = () => new Promise(r => setImmediate(r));
const out = o => console.log('§§' + JSON.stringify(o));
const YT = id => 'https://www.youtube.com/watch?v=' + id;
"""


def test_the_page_cant_read_or_press_what_the_extension_puts_in_it():
    """Any site could put YouTube links in, wait, and read off our ⊕ which
    videos — and, through their titles, which file names — the library has;
    and press our buttons from its script to start downloads on the server."""
    d = node(
        CONTENT
        + r"""
        links.push(E('a', {attrs: {href: YT('aaaaaaaaaaa')}, text: 'Q3 board meeting'}));
        known[YT('aaaaaaaaaaa')] = 'transcribed';
        // Library marks on other sites: the user's choice (the side panel's ⚙)
        chrome.storage.sync.get = async () => ({marksEverywhere: true});
        load();
        (async () => {
            await new Promise(r => setImmediate(r));
            await tick();
            const pick = links[0].nextElementSibling;
            const knownAsked = sent.find(m => m.type === 'known');
            const cls = button(pick).className;
            click(button(pick), false);                       // a script's click (Vimium): selecting is fine
            const selected = (await ask({type: 'list'})).videos[0].selected;
            // Sending is the user's own click only — a script's passes through untouched
            const bar = made.find(e => e.tagName === 'WTM-ROOT')._shadow.querySelector('.bar');
            const summarize = {closest: () => ({dataset: {a: 'summarize'}})};
            const script = click(bar, false, summarize);
            await tick();
            const sentByScript = sent.filter(m => m.type === 'ingest').length;
            click(bar, true, summarize);
            await tick();
            out({modes: made.filter(e => e.mode).map(e => e.mode), attrs: pick.attrs, dataset: pick.dataset,
                 shadowRoot: pick.shadowRoot, cls, titles: knownAsked.videos.map(v => v.title),
                 selected, script, sentByScript, sentByUser: sent.filter(m => m.type === 'ingest').length,
                 hook: typeof ctx.__wxpTest, observe: ctx.observeOptions});
        })();"""
    )
    assert d["modes"] and set(d["modes"]) == {"closed"}
    assert (
        d["attrs"] == {} and d["dataset"] == {} and d["shadowRoot"] is None
    )  # nothing to read in the page
    assert d["cls"] == "light known"  # shown inside
    assert d["titles"] == [""]  # a page's title is not looked up
    assert d["selected"] is True
    assert d["script"] == {"prevented": False, "stopped": False} and d["sentByScript"] == 0
    assert d["sentByUser"] == 1
    assert d["hook"] == "undefined"  # no flag, no hook
    assert d["observe"]["attributeFilter"] == ["href", "src"]


def test_off_youtube_nothing_is_asked_unless_chosen():
    """Every YouTube link on every page went to the server to find its ✓ —
    over the network, often as plain http. Off YouTube the ⊕ still selects,
    but the library is asked only when the user chose so."""
    d = node(
        CONTENT
        + r"""
        links.push(E('a', {attrs: {href: YT('aaaaaaaaaaa')}, text: 'a talk'}));
        known[YT('aaaaaaaaaaa')] = 'transcribed';
        load();
        (async () => {
            await tick(); await tick();
            const pick = links[0].nextElementSibling;
            out({asked: sent.filter(m => m.type === 'known').length, cls: button(pick).className,
                 text: button(pick).textContent});
        })();"""
    )
    assert d["asked"] == 0
    assert d["cls"] == "light" and d["text"] == "+"


def test_no_pick_in_edited_text_nor_behind_hidden_links():
    """A ⊕ went into a mail being written (and editors took it out, and it
    came back); a link hidden with display: none or only for screen readers
    got one standing alone in the page. A ⊕ whose link hid goes."""
    d = node(
        CONTENT
        + r"""
        ctx.WTM_TEST = true;
        links.push(E('a', {attrs: {href: YT('aaaaaaaaaaa')}, text: 'shown'}),
                   E('a', {attrs: {href: YT('bbbbbbbbbbb')}, text: 'in a mail', editable: true}),
                   E('a', {attrs: {href: YT('ccccccccccc')}, text: 'in a textbox', closest: s => s.includes('textbox') ? {} : null}),
                   E('a', {attrs: {href: YT('ddddddddddd')}, text: 'display none', rects: 0}),
                   E('a', {attrs: {href: YT('eeeeeeeeeee')}, text: 'screen readers', rect: {width: 1, height: 1}}));
        load();
        const first = links.map(a => !!a.nextElementSibling);
        const pick = links[0].nextElementSibling;
        links[0].rects = 0;                                   // it hides
        ctx.__wxpTest.scan();
        out({first, gone: removed.includes(pick), shadow: ctx.__wxpTest.shadow(pick) === pick._shadow});"""
    )
    assert d["first"] == [True, False, False, False, False]
    assert d["gone"] is True and d["shadow"] is True


def test_only_changes_that_matter_start_a_round():
    """Every change of the page — a player's clock each second — scanned the
    whole page again and rewrote the summary under the video, which lost the
    text being selected in it."""
    d = node(
        CONTENT
        + r"""
        ctx.WTM_TEST = true;
        load();
        const m = ctx.__wxpTest.matters;
        const node = (tag, inner) => ({nodeType: 1, tagName: tag, matches: () => false, querySelector: () => inner ? {} : null});
        out({
            text: m([{type: 'childList', addedNodes: [{nodeType: 3}], removedNodes: []}]),
            plain: m([{type: 'childList', addedNodes: [node('DIV', false)], removedNodes: []}]),
            ours: m([{type: 'childList', addedNodes: [node('WTM-PICK', true)], removedNodes: []}]),
            withLink: m([{type: 'childList', addedNodes: [node('DIV', true)], removedNodes: []}]),
            linkGone: m([{type: 'childList', addedNodes: [], removedNodes: [node('DIV', true)]}]),
            pointed: m([{type: 'attributes', attributeName: 'href', oldValue: '/about',
                         target: {getAttribute: () => 'https://youtu.be/aaaaaaaaaaa'}}]),
            pointedAway: m([{type: 'attributes', attributeName: 'href', oldValue: 'https://youtu.be/aaaaaaaaaaa',
                             target: {getAttribute: () => '/about'}}]),
            picture: m([{type: 'attributes', attributeName: 'src', oldValue: 'https://i.ytimg.com/a.jpg',
                         target: {getAttribute: () => 'https://i.ytimg.com/b.jpg'}}]),
        });"""
    )
    assert d == {
        "text": False,
        "plain": False,
        "ours": False,
        "withLink": True,
        "linkGone": True,
        "pointed": True,
        "pointedAway": True,
        "picture": False,
    }


def test_the_watch_pages_button_keeps_to_its_own_video():
    """An answer for the video before (YouTube changes the page, not the tab)
    showed that one's progress and summary on the next video's button."""
    d = node(
        CONTENT
        + r"""
        ctx.WTM_TEST = true;
        Object.assign(ctx.location, {hostname: 'www.youtube.com', href: YT('aaaaaaaaaaa'), pathname: '/watch'});
        load();
        (async () => {
            // A batch without this video: nothing of it shows here
            await ask({type: 'follow-main', batch: {id: 'x', status: 'working',
                       items: [{url: YT('zzzzzzzzzzz'), state: 'summarized', summary: 'not mine'}]}});
            const foreign = {state: ctx.__wxpTest.main.state, summary: ctx.__wxpTest.main.summary};
            // Sent, and on to the next video before the answer
            hold = m => m.type === 'ingest';
            hold.answer = {batch: {id: 'b1', status: 'working', items: [{url: YT('aaaaaaaaaaa'), state: 'downloading'}]}};
            const sending = ask({type: 'main-send', mode: 'summarize'});
            await new Promise(r => setImmediate(r));
            Object.assign(ctx.location, {href: YT('bbbbbbbbbbb')});
            ctx.__wxpTest.scan();
            hold.cb();
            const answer = await sending;
            const M = ctx.__wxpTest.main;
            out({foreign, answer: answer.batch.id, key: M.key, state: M.state, batch: M.batchId});
        })();"""
    )
    assert d["foreign"] == {"state": "", "summary": ""}
    assert d["answer"] == "b1"  # the popup still hears it was sent
    assert d == {**d, "key": "bbbbbbbbbbb", "state": "", "batch": ""}


def test_a_hidden_link_gets_its_pick_once_shown():
    """A link inside a closed menu, a tab or a <details> had no box when the
    page was looked at — opened later (a class, a style, hidden or open
    changed), it never got its ⊕. Watched only while such a link waits."""
    d = node(
        CONTENT
        + r"""
        ctx.WTM_TEST = true;
        const link = E('a', {attrs: {href: YT('aaaaaaaaaaa')}, text: 'in a menu', rects: 0});
        const menu = E('div', {kids: [link]}), elsewhere = E('div');
        links.push(link);
        load();
        const m = ctx.__wxpTest.matters;
        const changed = (target, name) => [{type: 'attributes', attributeName: name, oldValue: '', target}];
        const waiting = {filter: ctx.observeOptions.attributeFilter, menu: m(changed(menu, 'class')),
                         open: m(changed(menu, 'open')), self: m(changed(link, 'style')),
                         elsewhere: m(changed(elsewhere, 'class')), pick: !!link.nextElementSibling};
        link.rects = 1;                                       // the menu opened
        ctx.__wxpTest.scan();
        out({waiting, pick: !!link.nextElementSibling, filter: ctx.observeOptions.attributeFilter,
             after: m(changed(menu, 'class'))});"""
    )
    assert d["waiting"] == {
        "filter": ["href", "src", "class", "style", "hidden", "open"],
        "menu": True,
        "open": True,
        "self": True,
        "elsewhere": False,
        "pick": False,
    }
    assert d["pick"] is True and d["filter"] == ["href", "src"] and d["after"] is False


def test_youtubes_player_starts_no_round():
    d = node(
        CONTENT
        + r"""
        ctx.WTM_TEST = true;
        Object.assign(ctx.location, {hostname: 'www.youtube.com', href: YT('aaaaaaaaaaa'), pathname: '/watch'});
        links.push(E('a', {attrs: {href: '/watch?v=bbbbbbbbbbb'}, text: 'a hidden one', rects: 0}));
        load();
        const player = E('div');
        out({filter: ctx.observeOptions.attributeFilter,
             style: ctx.__wxpTest.matters([{type: 'attributes', attributeName: 'style', oldValue: 'width: 1px', target: player}]),
             text: ctx.__wxpTest.matters([{type: 'childList', addedNodes: [{nodeType: 3}], removedNodes: []}])});"""
    )
    assert d == {"filter": ["href", "src"], "style": False, "text": False}


def test_the_picks_colours_follow_a_dark_mode_switch():
    d = node(
        CONTENT
        + r"""
        ctx.WTM_TEST = true;
        links.push(E('a', {attrs: {href: YT('aaaaaaaaaaa')}, text: 'x'}));
        load();
        const b = button(links[0].nextElementSibling);
        const first = b.className;
        ctx.bg = 'rgb(15, 15, 15)';
        ctx.schemeChange();                                   // the system went dark
        const switched = b.className;
        ctx.bg = 'rgb(250, 250, 250)';
        ctx.__wxpTest.scan();                                 // the page's own switch, seen the next round
        out([first, switched, b.className]);"""
    )
    assert d == ["light", "dark", "light"]


def test_a_refresh_asked_during_a_round_runs_once_more():
    """A batch sent while the half-minute round was running waited for the
    next one to count on the badge."""
    d = node(
        BACKGROUND
        + r"""
        (async () => {
            let release;
            store.local.batches = [{id: 'b1', last: {id: 'b1', status: 'working'}}];
            answers['/api/ingest/b1'] = () => new Promise(r => { release = () => r({id: 'b1', status: 'working',
                                                                   items: [{state: 'queued'}]}); });
            answers['/api/ingest/b2'] = {id: 'b2', status: 'working', items: [{state: 'downloading'}, {state: 'queued'}]};
            const round = ctx.refresh();
            await new Promise(r => setImmediate(r));
            // sent just now (a new list, as chrome.storage hands out copies)
            store.local.batches = [{id: 'b2', last: {id: 'b2', status: 'working'}}, ...store.local.batches];
            const a = ctx.refresh(), b = ctx.refresh();          // twice: one more round, not two
            answers['/api/ingest/b1'] = {id: 'b1', status: 'working', items: [{state: 'queued'}]};
            release();
            await round; await a; await b;
            out({badge, asked: calls.filter(c => c[1].endsWith('/b2')).length});
        })();"""
    )
    assert d == {"badge": "3", "asked": 1}


def test_audio_only_is_chosen_in_the_form():
    """The download: as in Whisper Transcription Manager, audio only, or video and audio — a
    switch like vocal isolation."""
    d = node(
        SIDEPANEL
        + r"""
        ctx.P.opts = OPTS;
        const html = ctx.fieldsHtml({audio_only: true}, {transcription: true});
        const read = v => ctx.readFields({querySelectorAll: () => [{dataset: {k: 'audio_only'}, value: v}],
                                          querySelector: () => null});
        out({html, on: read('on'), off: read('off'), general: read('')});"""
    )
    assert '<option value="">As in Whisper Transcription Manager (video)</option>' in d["html"]
    assert '<option value="on" selected>Audio only — smaller and faster</option>' in d["html"]
    assert d["on"] == {"audio_only": True} and d["off"] == {"audio_only": False}
    assert d["general"] == {"audio_only": None}


def test_select_all_says_it_ticks_the_pages_videos():
    """ "select all shown" read as every batch in the queue — it says what it ticks."""
    d = node(
        SIDEPANEL
        + r"""
        const vs = n => Array.from({length: n}, (_, i) => ({key: 'k' + i, url: 'u' + i, title: 'T' + i,
                                                             selected: false}));
        ctx.P.tab = {id: 1};
        const list = vs(3);
        ctx.takeList({videos: list});
        const none = {all: els.btnAll.textContent, clear: els.selBox.hidden};
        list[0].selected = true;
        ctx.takeList({videos: list});
        const one = {all: els.btnAll.textContent, clear: els.selBox.hidden, count: els.selCount.textContent};
        list.forEach(v => { v.selected = true; });
        ctx.takeList({videos: list});
        const every = {all: els.btnAll.textContent, clear: els.selBox.hidden};
        out({none, one, every, filtered: ctx.pickAllLabel(vs(2), 9, true)});"""
    )
    assert d["none"] == {"all": "select all 3 videos on this page", "clear": True}
    assert d["one"] == {
        "all": "select all 3 videos on this page",
        "clear": False,
        "count": "1 selected",
    }
    assert d["every"] == {"all": "unselect all", "clear": False}
    assert d["filtered"] == "select the 2 matching"


def test_sent_videos_can_be_taken_out_and_the_list_cleared():
    d = node(
        SIDEPANEL
        + r"""
        const rows = [
            {id: 'b1', at: 1, last: {id: 'b1', mode: 'transcribe', status: 'working', items: [
                {url: 'u1', title: 'One', state: 'downloading'}, {url: 'u2', title: 'Two', state: 'transcribed'}]}},
            {id: 'b2', at: 1, last: {id: 'b2', mode: 'summarize', status: 'done', items: [
                {url: 'u3', title: 'Three', state: 'summarized'}]}}];
        ctx.renderRecent(rows);
        const mixed = {html: els.recent.innerHTML, done: els.btnClearDone.hidden, all: els.btnClearAll.hidden,
                       title: els.btnClearAll.title};
        ctx.renderRecent([rows[1]]);
        const finished = {done: els.btnClearDone.hidden, title: els.btnClearAll.title};
        ctx.renderRecent([]);
        out({mixed, finished, empty: els.btnClearAll.hidden});"""
    )
    html = d["mixed"]["html"]
    assert html.count('data-drop="b1"') == 2 and html.count('data-drop="b2"') == 1
    assert "Take it out — it won’t be fetched or transcribed" in html  # the downloading one
    assert "Remove from this list — the transcript stays" in html  # the transcribed one
    assert "Take them all out" in html  # the working batch's ✕
    assert d["mixed"]["done"] is False and d["mixed"]["all"] is False
    assert "the batch still working is taken out" in d["mixed"]["title"]
    assert d["finished"] == {"done": True, "title": "Empty this list — the transcripts stay"}
    assert d["empty"] is True


def test_taking_out_reaches_the_server_and_the_list():
    d = node(
        BACKGROUND
        + r"""
        (async () => {
            store.local.batches = [
                {id: 'b1', last: {id: 'b1', status: 'working', items: [{url: 'u1'}, {url: 'u2'}]}},
                {id: 'b2', last: {id: 'b2', status: 'done', items: [{url: 'u3'}]}},
                {id: 'b3', last: {id: 'b3', status: 'working', items: [{url: 'u4'}]}}];
            answers['/api/ingest/b1/drop'] = {ok: true, dropped: 1, unqueued: 0};
            answers['/api/ingest/b1'] = {id: 'b1', status: 'working', items: [{url: 'u1'}, {url: 'u2'}]};
            const r = await ctx.handle({type: 'drop', id: 'b1', urls: ['u1']});
            const recent = (await ctx.handle({type: 'recent'})).batches;
            // The server lost b2: off the list all the same
            answers['/api/ingest/b2/drop'] = {error: 'unknown batch'};
            await ctx.handle({type: 'drop', id: 'b2', urls: ['u3']});
            const afterLost = store.local.batches.map(b => b.id);
            await ctx.handle({type: 'clear-sent', all: false});
            const finished = store.local.batches.map(b => b.id);
            calls.length = 0;
            await ctx.handle({type: 'clear-sent', all: true});
            out({r, items: recent[0].last.items.map(i => i.url), afterLost, finished,
                 cleared: store.local.batches.length,
                 drops: calls.filter(c => c[1].endsWith('/drop')).map(c => [c[1].replace('http://localhost:9860', ''), c[2]])});
        })();"""
    )
    assert d["r"] == {"ok": True, "dropped": 1, "unqueued": 0}
    assert d["items"] == ["u2"]  # hidden here even if the server still lists it
    assert d["afterLost"] == ["b1", "b3"]
    assert d["finished"] == ["b1", "b3"]  # both still working
    assert d["cleared"] == 0
    assert sorted(x[0] for x in d["drops"]) == ["/api/ingest/b1/drop", "/api/ingest/b3/drop"]
    assert all(x[1] == {} for x in d["drops"])  # all of their videos


def test_what_is_selected_on_the_page_shows_in_the_panel_at_once():
    """Selected with the page's ⊕: the panel's "Selected" box lists it above
    the tabs — the watch page's own video and one scrolled away too — and a
    change on the page has the panel ask the page again."""
    d = node(
        SIDEPANEL
        + r"""
        (async () => {
            ctx.P.tab = {id: 5};
            const W = 'https://www.youtube.com/watch?v=';
            ctx.takeList({videos: [{key: 'mainVideo01', url: W + 'mainVideo01', title: 'Main', main: true,
                                    kind: 'video', selected: true},
                                   {key: 'otherVideo1', url: W + 'otherVideo1', title: 'Other <b>', kind: 'video',
                                    selected: false}],
                          selection: [{key: 'mainVideo01', url: W + 'mainVideo01', title: 'Main', kind: 'video'},
                                      {key: 'goneVideo01', url: W + 'goneVideo01', title: 'Scrolled away', kind: 'video'}]});
            const shown = {hidden: els.selBox.hidden, title: els.selTitle.textContent, html: els.selList.innerHTML};
            paged.length = 0;
            ctx.heard({type: 'selection', count: 0}, {tab: {id: 9}});       // another tab: nothing
            const other = paged.length;
            ctx.heard({type: 'selection', count: 0}, {tab: {id: 5}});
            await new Promise(r => setTimeout(r, 0));
            out({shown, other, asked: paged.map(m => m.type), after: els.selBox.hidden});
        })();"""
    )
    s = d["shown"]
    assert s["hidden"] is False and s["title"] == "2 videos selected on this page"
    assert 'data-unpick="mainVideo01"' in s["html"] and "Scrolled away" in s["html"]
    assert "i.ytimg.com/vi/goneVideo01/" in s["html"]
    assert d["other"] == 0 and d["asked"] == ["list"]
    assert d["after"] is True  # the page answered with nothing selected


def test_the_page_tells_of_a_changed_selection():
    assert "send({type: 'selection'" in (EXT / "content.js").read_text(encoding="utf-8")
    d = node(
        BACKGROUND
        + r"""
        (async () => {
            let r = null;
            ctx.onMessage({type: 'selection', count: 2}, {tab: {id: 3}, url: 'https://www.youtube.com/'}, x => { r = x; });
            await new Promise(res => setTimeout(res, 0));
            out(r);
        })();"""
    )
    assert d == {"ok": True}
