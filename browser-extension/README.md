# Whisper Transcription Manager — browser extension

Select YouTube videos on any page and send them to your Whisper Transcription Manager server:
**transcribe** them, **summarize** each one, or **analyze them together** with one
task ("compare what they say", "summarize them together", …).

- **The video you watch.** On a YouTube watch page, **✦ Whisper TM ▾** sits in the
  row under the video, next to *Share*: *Summarize*, *Transcribe*, *With options…*,
  *Read & ask in the side panel*, *Add to selection*. The button shows how far it is —
  downloading, queued, transcribing, summarizing — and the summary unfolds under the
  title. Once summarized, the button folds and unfolds the summary; its ▾ opens the
  menu.
- **The side panel.** Stays open beside the page and follows the tab: the video's
  summaries for reading (every time in them — *[0:04:12]* — plays the video from
  there), **Ask** — questions to the language model, answered from the video's
  transcript, with the times — and the **Transcript**, to find in and play from. The
  page's other videos and everything sent are tabs of their own; a title there shows
  that video in the panel. Open it from the popup (▤), the menu on the video, or
  *Ask about it ▤* on its summary.
- **What was sent, tidied.** In the side panel's *Sent* tab a ✕ beside a video takes
  it out: one that still waits — not fetched yet, downloading, or waiting in the
  queue — is not fetched or transcribed at all; a finished one only leaves the list
  (its transcript stays). The ✕ of a batch does that for all its videos; *clear
  finished* and *clear all* empty the list (*clear all* asks once more when batches
  are still working).
- **The selection in the side panel.** Whatever is selected on the page shows at once
  above the panel's tabs — each video with a ✕ to unselect it, *Summarize*,
  *Transcribe*, *Options…*, *Analyze…* and *clear selection*. In the *Page* tab,
  *select all videos on this page* ticks the page's videos (with a filter: the ones
  it shows).
- **Every other video.** Search results, the home page, the suggestions beside the
  player, a playlist: a small ⊕ behind each title selects that video with one click
  (blue ✓). A green ✓ means it is in the library already — it is never downloaded
  twice.
- **Any other page.** A ⊕ behind every link to a YouTube video, and a badge on an
  embedded player when the mouse is over it.
- **The selection.** A bar at the bottom left shows the selected videos as small
  pictures (*show the list* for their titles) and offers *Summarize*, *Transcribe*,
  *Options…* and *Analyze…* (one task over all of them together); then it shows where
  each video stands, with the summary to unfold and a link that opens the recording
  in Whisper Transcription Manager.
- **The toolbar popup.** On top the video the page plays, with its state, its
  buttons and its summary; below it the page's other videos to tick, folded away,
  with a filter; then every batch sent, and the server address (⚙).
- **How, chosen once or each time.** The side panel's ⚙ holds the defaults for videos
  from the browser — the summary's template (*Summary*, *Bullet Points*, *Chapter
  Marks* …), the language it is written in (*German*, whatever the video speaks), the
  Whisper model, language (detected by default), speakers, vocal isolation,
  vocabulary, and the download — *Audio only* (smaller and faster; nothing to watch
  in the app afterwards) or *Video and audio*; *As in Whisper Transcription Manager* is the app's own setting. *Options…* chooses for
  one send; with **Ask before sending** on, Summarize and Transcribe always open that
  form first. A video summarized already can be summarized once more (*another
  one…*) — with another template, in another language.
- **Right-click** a YouTube link, or a watch page: *Summarize this video* /
  *Transcribe this video* / *Choose how… (side panel)*.
- When a batch is done, a notification says so; the toolbar icon counts the videos
  still working.

Everything happens on the server (`core/ingest.py`): the video is downloaded into
the upload folder, queued at the front, transcribed as its options say (the side
panel's defaults — `ingest.defaults` in the configuration — and what a send chose)
and — for *Summarize* — summarized right after. *Analyze…* starts an analysis
(*Analyses* in the app's rail) once every transcript is there.
Closing the tab or the browser stops nothing.

## Install (Chrome, Edge, Brave)

1. Open `chrome://extensions` and switch on **Developer mode** (top right).
2. **Load unpacked** → choose this folder (`browser-extension/` in the Whisper Transcription Manager
   folder).
3. Click the extension's icon → ⚙ → enter the address Whisper Transcription Manager is reached at from
   this browser (default `http://localhost:9860`; from another machine e.g.
   `http://192.0.2.10:9860`) → *Save*. The dot turns green when the server answers.
4. Pages that were open before get the extension's helper on install; one that
   still shows nothing needs a reload.

After an update of these files: `chrome://extensions` → the extension's ↻.
It needs Chrome 116 or newer (or an Edge or Brave of that base).

`make extension-zip` puts two zip files into `dist/`: one to unpack and load as
above (it keeps the fixed ID below), and one for the Chrome Web Store, which
refuses the manifest's `key` — the store gives the extension an ID of its own.

The extension has a fixed ID, `mloofppjoccofakkdoahcmjpppamkgcd` (the manifest's
`key` sets it): the server takes requests that change something only from this
extension, not from any other one installed in the browser. A build of it with
another ID — from a store, or with the `key` removed — must be listed in
`security.trusted_extensions` of the server's config file.

## What it may do, and why

| Permission | For |
|---|---|
| Content script on all pages | finding the YouTube videos of any page |
| `http://*/*`, `https://*/*` | speaking to your server, wherever it runs |
| `storage` | the server address, the batches sent |
| `contextMenus` | the right-click entries |
| `scripting` | bringing the helper into pages that were open before the install |
| `sidePanel` | the side panel: reading, asking, choosing how |
| `notifications`, `alarms` | the word when a batch is done, checked every half minute |

It sends nothing anywhere but to the server address you give it: the links of the
videos you select, and — to mark videos already in the library — the links of the
videos on a YouTube page you are looking at (their titles too: a download from
before the video ids were noted is found by its title). On other sites it marks them
only with **Library marks on other sites** switched on (the side panel's ⚙) — then
the links of the YouTube videos linked there go to the server as well. Every request
goes through the extension's service worker (`background.js`), never from the page.
The address is usually plain `http` in your own network: what goes there can be read
on the way.

The selection's small pictures come from YouTube's image server (`i.ytimg.com`),
loaded by the browser as on YouTube itself.

**With the server's login switched on** (`auth.enabled`), enter its access token
(`auth.api_token` in the server's config) under ⚙, below the address: the extension
sends it with every request (`X-API-Token`). It is kept in this browser only — not
synced with your browser profile — and never given to the pages; *Forget* removes it.
Over plain `http` it travels readable, like everything else.

The page can't see or use what the extension puts into it: its elements have closed
shadow roots (which videos are in the library, the summary under a YouTube video stay
inside), and the helper in a page may only ask for what its buttons need — not change
the server address, the defaults, or chat. Nothing goes into a field being edited (a
mail, an editor), nor behind a hidden link (shown later — a menu or a tab opened —
the link gets its ⊕ then).

Clicks made by a script (`isTrusted` false — keyboard helpers like Vimium or
Surfingkeys click that way) work for what stays in the page: selecting with ⊕, the
badge and the ticks, the ✦ menu itself, unfolding the summary or the list. Where a click
sends to the server or opens something — Summarize, Transcribe, Options…, Start, Stop
downloading, open ↗, Ask about it, Read & ask — only the user's own click counts; a
script's click there is let through untouched (not stopped, not acted on).

### For tests

With `globalThis.WTM_TEST = true` set in the content script's own world *before*
`content.js` runs, `globalThis.__wxpTest` exists: `shadow(el)` (the closed shadow root
of one of our elements — `wtm-root`, `wtm-pick`, `wtm-action`, `wtm-menu`,
`wtm-panel`), `root()` (the overlay's), `scan()`, `state`, `main`, and clicks made by a
script (`.click()`) count. A web page lives in another world and can't set the flag;
without it neither the hook nor the script clicks exist.

## Downloads and rights

Downloading from YouTube may be restricted by its terms of service and by
copyright: you are responsible for having the right to download and transcribe
what you send.

## The server's side

The extension carries the server's version number: it is made for the server of the
same version. After updating the server, load the extension from the same checkout
again (*Reload* on `chrome://extensions`).

`POST /api/ingest` (`urls`, `mode`: transcribe · summarize · analyze, `task`),
`GET /api/ingest/<id>` (every video's state, the summaries), `POST /api/ingest/known`
(which videos are in the library; with `summaries: true` their summaries too). The side
panel: `GET /api/ingest/options` (the forms' choices, the general settings, the
defaults), `POST /api/ingest/defaults`, `POST /api/ingest/transcript` (a video's
transcript and summaries), `POST /api/ingest/ask` (a chat conversation with the video's
transcript attached — asked on through `/api/chat/send`); `POST /api/ingest` takes
`options` and `again` — `whisper_tm/api/ingest.py`. Links that open the
app: `/?open=<full path>&t=<seconds>`, `/?analysis=<id>`, `/?chat=<id>`.
