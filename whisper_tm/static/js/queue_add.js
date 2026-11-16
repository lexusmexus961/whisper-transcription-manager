// ── Queue editor, tab "Add" ──────────────────────────────────────────────
//
// A folder's files without a transcript into the queue — with the options
// of the transcription dialogs, and at the front (next up) or at the end.
// The sidebar queues single files at the front (⤒), and its "＋ Queue N
// open files" only at the end, behind whatever long run is waiting.
//
// The folder counts with its subfolders, as the file list shows it. The
// count comes from the library (/api/queue/library/select) — files not
// indexed yet don't count until the folder was opened once. Queuing sends
// the count the user saw along: if the library changed in between, the
// server refuses instead of queuing something else (as "From library").
//
// The folder field is the file list's path field (pathpicker.js, place
// 'qa'): browse, filter, favorites; choosing a folder counts it. The list
// shows the first files and more as it is scrolled.

const QA_FIRST = 100;       // files listed at first
const QA_PAGE = 200;        // files per "more"

let qaState = {last: null, queued: 0, known: null, seq: 0, busy: false, ready: false,
               loadingMore: false, end: false};

function _qaEl(id) { return document.getElementById(id); }
function _qaFront() {
    const r = document.querySelector('input[name="qaWhere"]:checked');
    return !r || r.value === 'front';
}
function _qaFolder() { return (_qaEl('qaFolder').value || '').trim().replace(/\/+$/, ''); }
function _qaCriteria() { return {folders: [_qaFolder()], status: ['open']}; }

function qaInit() {
    if (!qaState.ready) {
        qaState.ready = true;
        // An open confirmation describes the options of the moment it was
        // asked — one changed after it would be sent, not the one it shows.
        const opts = _qaEl('qaTxOptions');
        if (opts) {
            opts.addEventListener('change', qaCancelConfirm);
            opts.addEventListener('input', qaCancelConfirm);
        }
        // Near the end of the list, the next files load by themselves.
        const list = _qaEl('qaList');
        if (list) {
            list.addEventListener('scroll', () => {
                if (list.scrollTop + list.clientHeight >= list.scrollHeight - 300) qaLoadMore();
            });
        }
        qaUseOpenFolder();
        return;
    }
    qaCount();
}

// The folder open in the file list — what the user looks at when they
// think "this folder".
function qaUseOpenFolder() {
    const dir = document.getElementById('dirInput');
    _qaEl('qaFolder').value = dir ? dir.value : '';
    if (typeof showPathEnd === 'function') showPathEnd(_qaEl('qaFolder'));
    qaCount();
}

// The picker chose a folder — a row, a favorite, "Choose this folder", or
// Enter on a typed path.
function qaFolderChosen() {
    qaCount();
}

async function _qaSelect(criteria, limit, estimate, offset = 0) {
    const r = await fetch('/api/queue/library/select', {method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({criteria, limit, estimate, ...(offset ? {offset} : {})})});
    return r.json();
}

async function qaCount() {
    const seq = ++qaState.seq;
    qaCancelConfirm();
    qaState.loadingMore = false;
    qaState.end = false;
    if (typeof ppUpdateStars === 'function') ppUpdateStars();
    if (!_qaFolder()) {
        qaState.last = null;
        _qaEl('qaSummary').innerHTML = 'Choose a folder.';
        _qaEl('qaList').innerHTML = '';
        qaRenderButton();
        return;
    }
    _qaEl('qaSummary').classList.add('ls-stale');
    let d, inQueue;
    try {
        [d, inQueue] = await Promise.all([
            _qaSelect(_qaCriteria(), QA_FIRST, true),
            _qaSelect({..._qaCriteria(), in_queue: true}, 0, false)]);
    } catch (e) {
        d = {error: 'Server not reachable'};
    }
    if (seq !== qaState.seq) return;          // a newer count is on its way
    _qaEl('qaSummary').classList.remove('ls-stale');
    if (d.error) {
        qaState.last = null;
        _qaEl('qaSummary').innerHTML = `<span class="ls-error">⚠ ${esc(d.error)}</span>`;
        _qaEl('qaList').innerHTML = '';
        qaRenderButton();
        return;
    }
    qaState.last = d;
    qaState.queued = (inQueue && !inQueue.error) ? inQueue.count : 0;
    qaState.known = null;
    if (!d.count) {
        // Nothing open: a folder that is done, or one the library hasn't
        // seen yet — two different things to say.
        try {
            const all = await _qaSelect({folders: [_qaFolder()]}, 0, false);
            if (seq !== qaState.seq) return;
            qaState.known = all.error ? null : all.count;
        } catch (e) { /* then the hint stays the general one */ }
    }
    qaRenderResult();
}

function qaRenderResult() {
    const d = qaState.last;
    if (!d) return;
    const name = _qaFolder().split('/').pop() || _qaFolder();
    const est = d.estimate_sec ? ` · about ${_lsHours(d.estimate_sec)} to transcribe` : '';
    // A file's length is known once it was probed — a folder indexed just
    // now has none yet, and "0 min" would read as empty files.
    const audio = d.audio_sec ? `${_lsHours(d.audio_sec)} audio` : 'length not known yet';
    const one = qaState.queued === 1;
    const queued = qaState.queued
        ? `<div class="qa-note">${qaState.queued.toLocaleString()} of them ${one ? 'is' : 'are'}
           queued already — ${_qaFront()
               ? (one ? 'it moves to the front too, with the settings it was queued with'
                      : 'they move to the front too, with the settings they were queued with')
               : (one ? 'it stays where it is' : 'they stay where they are')}.</div>` : '';
    _qaEl('qaSummary').innerHTML = d.count
        ? `<b>${d.count.toLocaleString()}</b> file${d.count === 1 ? '' : 's'} without a transcript in
           <b>${esc(name)}</b> and its subfolders · ${audio}${est}${queued}`
        : qaState.known
        ? `No file without a transcript in <b>${esc(name)}</b> — ${qaState.known === 1
              ? 'its one file is' : `all ${qaState.known.toLocaleString()} are`} transcribed or skipped.`
        : `No file without a transcript in <b>${esc(name)}</b>.
           <div class="qa-note">Only files the library knows count — open the folder in the file list
           once to index it.</div>`;
    _qaRenderList();
    qaRenderButton();
}

function _qaRenderList() {
    const d = qaState.last;
    const more = qaState.end ? 0 : d.count - d.sample.length;
    const button = qaState.loadingMore ? '<button class="tx-act-btn" disabled>Loading…</button>'
        : `<button class="tx-act-btn" onclick="qaLoadMore()">Show ${
              Math.min(more, QA_PAGE).toLocaleString()} more</button>`;
    _qaEl('qaList').innerHTML = d.sample.length ? `<table class="ls-table"><tbody>${d.sample.map(s =>
        `<tr title="${esc(s.abs_path)}"><td>${esc(s.name)}<div class="ls-dir">${esc(
            s.abs_path.slice(_qaFolder().length + 1, -s.name.length))}</div></td>
         <td>${s.duration_sec ? _lsHours(s.duration_sec) : '—'}</td></tr>`).join('')}</tbody></table>
        ${more > 0 ? `<div class="ls-empty qa-more">… and ${more.toLocaleString()} more ${button}</div>` : ''}`
        : '';
}

// The next files of the list — by the button, or by scrolling near its end.
async function qaLoadMore() {
    const d = qaState.last;
    if (!d || qaState.loadingMore || qaState.end || d.sample.length >= d.count) return;
    const seq = qaState.seq;
    qaState.loadingMore = true;
    _qaRenderList();
    let page;
    try {
        page = await _qaSelect(_qaCriteria(), QA_PAGE, false, d.sample.length);
    } catch (e) {
        page = {error: 'Server not reachable'};
    }
    if (seq !== qaState.seq) return;          // counted anew meanwhile — that list is gone
    qaState.loadingMore = false;
    if (page.error) {
        Toast.error(page.error);
    } else {
        const rows = page.sample || [];
        d.sample.push(...rows);
        // Fewer than asked: the folder shrank since it was counted — the
        // count stays what was counted (it guards the queuing), the list ends.
        if (rows.length < QA_PAGE && d.sample.length < d.count) qaState.end = true;
    }
    _qaRenderList();
}

// What a click queues: at the front every file (the waiting ones move up
// with the new ones), at the end only the ones not in the queue yet.
function _qaToQueue() {
    const d = qaState.last;
    if (!d) return 0;
    return _qaFront() ? d.count : Math.max(d.count - qaState.queued, 0);
}

function qaRenderButton() {
    const d = qaState.last;
    const n = _qaToQueue();
    const b = _qaEl('qaBtnQueue');
    b.textContent = d && d.count && !n
        ? 'All of them are in the queue already'
        : `Queue ${n.toLocaleString()} file${n === 1 ? '' : 's'} `
          + (_qaFront() ? 'at the front' : 'at the end');
    b.disabled = !n || qaState.busy;
}

function qaWhereChanged() {
    qaCancelConfirm();
    if (qaState.last) qaRenderResult();
    else qaRenderButton();
}

// What the options say, in a few words — so the confirmation shows what
// the files will be transcribed with.
function _qaDescribe(s) {
    const lang = document.getElementById('qaLang');
    const langName = lang && lang.selectedOptions[0] ? lang.selectedOptions[0].textContent : s.language;
    return [s.model, langName,
            s.vocal_isolation ? 'vocal isolation' : 'no vocal isolation',
            s.diarize ? 'speakers' : 'no speakers',
            s.word_timestamps ? 'word timestamps' : '']
        .filter(Boolean).join(', ');
}

function qaAskConfirm() {
    const d = qaState.last;
    const n = _qaToQueue();
    if (!d || !n) return;
    const s = txCollect('qa');
    const running = (typeof queueRunningTotal === 'function') ? queueRunningTotal() : 0;
    const pending = (typeof queuePending !== 'undefined') ? queuePending : 0;
    const entries = pending === 1 ? 'entry' : 'entries';
    // One file is "it", not "they"
    const one = n === 1;
    const when = _qaFront()
        ? (running ? `The ${running} running file${running === 1 ? ' finishes' : 's finish'} first;
                      then ${one ? 'this one is' : 'these are'} next` : (one ? 'This one is next' : 'These are next'))
          + (pending ? `, before the ${pending.toLocaleString()} waiting ${entries}.` : '.')
        : pending ? `${one ? 'It waits' : 'They wait'} behind the ${pending.toLocaleString()} ${entries} already waiting.`
        : running ? `${one ? 'It follows' : 'They follow'} the ${running} running file${running === 1 ? '' : 's'}.`
        : `The queue is empty — ${one ? 'it is' : 'they are'} next.`;
    // The audio is that of every match — only right when every match goes.
    const audio = d.audio_sec && n === d.count ? ` (${_lsHours(d.audio_sec)} audio)` : '';
    _qaEl('qaConfirmText').innerHTML = `Queue <b>${n.toLocaleString()}</b>
        file${n === 1 ? '' : 's'}${audio}
        <b>${_qaFront() ? 'at the front' : 'at the end'}</b> to transcribe — ${esc(_qaDescribe(s))}?
        <div class="qa-note">${when}</div>`;
    _qaEl('qaBtnConfirm').textContent = one ? 'Yes, queue it' : 'Yes, queue them';
    _qaEl('qaConfirm').style.display = '';
}

function qaCancelConfirm() {
    const c = _qaEl('qaConfirm');
    if (c) c.style.display = 'none';
}

async function qaEnqueue() {
    const d = qaState.last;
    if (!d || qaState.busy) return;
    qaState.busy = true;
    _qaEl('qaBtnConfirm').disabled = true;
    qaRenderButton();
    let out;
    try {
        const r = await fetch('/api/queue/library/enqueue', {method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({criteria: _qaCriteria(), job: 'transcribe',
                                  settings: txCollect('qa'), front: _qaFront(),
                                  expected_count: d.count})});
        out = await r.json();
    } catch (e) {
        out = {error: 'Server not reachable'};
    }
    qaState.busy = false;
    _qaEl('qaBtnConfirm').disabled = false;
    qaCancelConfirm();
    if (out.error) {
        Toast.error(out.error);
    } else {
        const parts = [`${out.added.toLocaleString()} file${out.added === 1 ? '' : 's'} queued `
                       + (out.front ? 'at the front' : 'at the end')];
        if (out.moved_up) parts.push(`${out.moved_up.toLocaleString()} already waiting moved to the front`);
        if (out.already_queued) parts.push(`${out.already_queued.toLocaleString()} already in the queue`);
        if (out.outside_allowed_roots) parts.push(`${out.outside_allowed_roots} outside the allowed folders`);
        Toast.success(parts.join(' · ') + (out.paused ? ' — the queue is paused; click "Resume".' : '.'));
        if (typeof refreshQueue === 'function') refreshQueue();
    }
    qaCount();
}
