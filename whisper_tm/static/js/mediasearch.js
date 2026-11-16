// ── Images & sounds (api/media_search.py) ──────────────────────────────────
//
// Finds moments by what is SEEN or HEARD. Stills, clips, clips with their
// sound, sound windows, spoken text and the language model's descriptions of
// the stills lie in ONE vector space (EmbeddingGemma 2), so any of it asks
// all of it:
//
//   a text      the search field (sounds are asked with the language
//               model's English label of it — "busy pub crowd")
//   a hit       "Similar" — the hit's own vector asks
//   a file      📎 an image, a sound or a video of your own, optionally
//               together with the text in the field (one vector)
//
// Each kind is ranked on its own: a query meets a still and a sound on
// different scales. "Check with the language model" lets a model that sees
// pictures judge the best stills of a list one by one and sorts them by its
// verdict (measured: precision of the top 10 0.62 → 0.82).
//
// Spoken text is the words of the 10 s around a moment, in the same space:
// it shows what is said where something is seen or heard. What was said —
// whole passages with their context — is the semantic search's (the Text
// tab); the list says so and takes the query there in one click.

const MS_KINDS = [
    { k: 'image', label: 'Stills', one: 'still', icon: '🖼', tiles: true, check: true },
    { k: 'description', label: 'Descriptions', one: 'description', icon: '📝', tiles: true, check: true },
    { k: 'video', label: 'Clips', one: 'clip', icon: '🎞', tiles: true, check: true },
    { k: 'videoaudio', label: 'Clips with sound', one: 'clip with sound', icon: '🎞🔊', tiles: true, check: true },
    { k: 'audio', label: 'Sounds', one: 'sound', icon: '🔊', tiles: false },
    { k: 'speech', label: 'Spoken text', one: 'spoken text', icon: '💬', tiles: false },
];
const MS_SHOW = { tiles: 12, rows: 8 };
const MS_EXAMPLES = ['police', 'outdoors at night', 'a video game', 'inside a car', 'a crowd', 'fire',
    'gunshots', 'rock music', 'laughter', 'dog barking', 'applause', 'an argument about money'];

const _ms = {
    info: null, seq: 0, last: '',
    source: { type: 'text' },      // {type:'text'} | {type:'like', item} | {type:'file', qid, name, kind, text, url}
    dims: 0,                       // 0 = what the index holds
    on: new Set(MS_KINDS.map(x => x.k)),
    open: {},                      // kind → all hits shown
    results: {},
    soundLabel: null,
    checks: {},                    // kind → {id, total, done, scores:{moment: score}, running}
    checking: new Set(),           // kinds whose check is being asked for
    active: null,                  // {kind, n} — the hit in the preview
    took: '',
};

function msClock(sec) {
    sec = Math.max(0, Math.floor(Number(sec) || 0));
    const h = Math.floor(sec / 3600), m = Math.floor((sec % 3600) / 60), s = sec % 60;
    const mm = String(m).padStart(h ? 2 : 1, '0'), ss = String(s).padStart(2, '0');
    return h ? `${h}:${mm}:${ss}` : `${mm}:${ss}`;
}

function msKind(k) { return MS_KINDS.find(x => x.k === k) || { k, label: k, icon: '' }; }

// ── Opening the tab ─────────────────────────────────────────────────────────

async function msOpen() {
    await msLoadInfo(true);
    msRenderControls();
    const q = document.getElementById('svQuery').value.trim();
    if (msReady() && (q !== _ms.last || !document.getElementById('msResults')?.innerHTML.trim())) msRun();
}

function msReady() { return !!(_ms.info && _ms.info.ready); }

async function msLoadInfo(force = false) {
    if (!force && _ms.info && !_ms.info.indexer?.running) return;
    try { _ms.info = await api('/api/media-search/info'); }
    catch (e) { _ms.info = { ready: false, error: e.message }; }
    msRenderInfo();
}

function msRenderInfo() {
    const el = document.getElementById('msInfo');
    const box = document.getElementById('msResults');
    if (!el) return;
    const i = _ms.info || {};
    const counts = (i.counts || {}).kinds || {};
    const parts = MS_KINDS.filter(x => counts[x.k]).map(x => plural(counts[x.k], x.one, x.label.toLowerCase()));
    const files = (i.counts || {}).files || {};
    const run = i.indexer || {};
    let line;
    if (i.error) line = `<span class="ms-err">Not reachable: ${esc(i.error)}</span>`;
    else if (i.enabled === false) line = '<span class="ms-err">Images &amp; sounds is switched off.</span>';
    else if (i.environment_ok === false) line = `<span class="ms-err">Not set up yet: ${esc((i.environment_problems || []).join(', '))}.</span>`;
    else if (!parts.length) line = 'The index is empty.';
    else line = `${parts.join(' · ')} — from ${plural(files.done, 'recording')}`;
    if (run.running) line += ` · <span class="ms-building">⟳ indexing: ${esc(run.message || '')}</span>`;
    el.innerHTML = `${line} <button class="ms-link" onclick="openSettings('media')">Settings →</button>`;
    if (box && !msReady() && i.enabled !== undefined) {
        box.innerHTML = `<div class="ms-empty">
            <div class="ms-empty-title">🖼🔊 Find moments by what is seen or heard</div>
            <p>Stills, short clips, sounds and spoken text of your recordings in one vector space — searched by a
               description, by a hit, or by an image, sound or video of your own. What was said is the Text tab's:
               its meaning search reads whole passages.</p>
            <p>${i.enabled === false ? 'Switch it on' : i.environment_ok === false ? 'Install its environment' : 'Build the index'}
               under Settings → Images &amp; sounds.</p>
            <button class="vx-btn primary" onclick="openSettings('media')">Open the settings</button></div>`;
    }
}

function msRenderControls() {
    const box = document.getElementById('msControls');
    if (!box) return;
    const counts = ((_ms.info || {}).counts || {}).kinds || {};
    const stored = Number((_ms.info || {}).dims) || 768;
    box.innerHTML = `
        <span class="ms-lbl">Show</span>
        ${MS_KINDS.map(x => {
            const n = counts[x.k] || 0;
            return `<button class="ms-chip ms-kind${_ms.on.has(x.k) ? ' on' : ''}${n ? '' : ' empty'}" data-kind="${x.k}"
                onclick="msToggleKind('${x.k}')" title="${n ? `${n.toLocaleString()} in the index` : 'none in the index'}">${x.icon} ${x.label}</button>`;
        }).join('')}
        <span class="ms-sep"></span>
        <label class="ms-lbl" title="Matryoshka: the first N numbers of each vector, unit length again — what does a smaller index keep?">Vector size
            <select id="msDims" onchange="msSetDims(this.value)">
                ${[768, 512, 256, 128].filter(d => d <= stored).map(d =>
                    `<option value="${d}"${(_ms.dims || stored) === d ? ' selected' : ''}>${d}${d === stored ? ' (all)' : ''}</option>`).join('')}
            </select></label>
        <span class="ms-sep"></span>
        <button class="vx-btn" onclick="document.getElementById('msFile').click()"
            title="Use your own image, sound or video as the query">📎 Query with a file …</button>
        <input type="file" id="msFile" accept="image/*,audio/*,video/*" style="display:none" onchange="msPickFile(this)">
        <label class="ms-lbl" title="The text in the search field goes into the same vector as the file — e.g. a photo + &quot;at night&quot;">
            <input type="checkbox" id="msWithText"> + the text</label>
        <label class="ms-lbl" title="For a video: its sound too, in the same vector">
            <input type="checkbox" id="msWithSound" checked> video with its sound</label>`;
    const ex = document.getElementById('msExamples');
    if (ex) ex.innerHTML = '<span class="ms-lbl">Try</span>' + MS_EXAMPLES.map(q =>
        `<button class="ms-chip" onclick="msTry('${escJs(q)}')">${esc(q)}</button>`).join('');
}

function msToggleKind(k) {
    if (_ms.on.has(k)) _ms.on.delete(k); else _ms.on.add(k);
    document.querySelectorAll('#msControls .ms-kind').forEach(b => b.classList.toggle('on', _ms.on.has(b.dataset.kind)));
    msRender();
}

function msSetDims(d) {
    _ms.dims = Number(d) || 0;
    if (_ms.source.type !== 'text' || _ms.last.length >= 2) msSearch();
}

function msTry(q) {
    document.getElementById('svQuery').value = q;
    msRun();
}

// ── Asking ──────────────────────────────────────────────────────────────────

// Enter in the search field: a text search (whatever asked before)
function msRun() {
    const q = document.getElementById('svQuery').value.trim();
    _ms.last = q;
    if (_ms.source.url) URL.revokeObjectURL(_ms.source.url);
    _ms.source = { type: 'text' };
    if (!msReady()) { msRenderInfo(); return; }
    if (q.length < 2) {
        msRenderSource();
        const box = document.getElementById('msResults');
        if (box) box.innerHTML = '<div class="sv-empty">Describe what is seen or heard — '
            + '"police at night", "a crowd", "gunshots", "rock music" — or drop in a file of your own (📎). '
            + 'Pictures understand other languages as well as English; sounds are asked in English for you.</div>';
        return;
    }
    msSearch();
}

// "Similar": this hit's own vector asks
function msLike(kind, n) {
    const h = (_ms.results[kind] || [])[n];
    if (!h) return;
    if (_ms.source.url) URL.revokeObjectURL(_ms.source.url);
    _ms.source = { type: 'like', item: h };
    msSearch();
}

async function msPickFile(input) {
    const f = input.files && input.files[0];
    input.value = '';
    if (!f) return;
    const box = document.getElementById('msResults');
    const fd = new FormData();
    fd.append('file', f);
    const text = document.getElementById('msWithText')?.checked ? document.getElementById('svQuery').value.trim() : '';
    if (text) fd.append('q', text);
    if (document.getElementById('msWithSound')?.checked) fd.append('with_sound', '1');
    const seq = ++_ms.seq;
    box.innerHTML = `<div class="sv-empty">Reading ${esc(f.name)} … (on the CPU: an image takes ~3 s, a video ~10 s)</div>`;
    let r;
    try { r = await api('/api/media-search/query-file', { method: 'POST', body: fd }); }
    catch (e) {
        if (seq === _ms.seq) box.innerHTML = `<div class="sv-empty ms-err">${esc(e.message)}</div>`;
        return;
    }
    if (seq !== _ms.seq) return;
    if (_ms.source.url) URL.revokeObjectURL(_ms.source.url);
    _ms.source = { type: 'file', qid: r.qid, name: r.name, kind: r.kind, text: r.text, url: URL.createObjectURL(f) };
    msSearch();
}

async function msSearch() {
    const box = document.getElementById('msResults');
    if (!box) return;
    const s = _ms.source;
    let url = `/api/media-search?limit=${(_ms.info || {}).results_per_kind || 48}`;
    if (_ms.dims) url += `&dims=${_ms.dims}`;
    if (s.type === 'file') url += '&qid=' + encodeURIComponent(s.qid);
    else if (s.type === 'like') url += '&like=' + s.item.id;
    else url += '&q=' + encodeURIComponent(_ms.last);
    const seq = ++_ms.seq;
    _ms.active = null;
    msCancelChecks();
    msRenderSource();
    box.innerHTML = '<div class="sv-empty">Searching …</div>';
    let data;
    try { data = await api(url); }
    catch (e) {
        if (seq === _ms.seq) box.innerHTML = `<div class="sv-empty ms-err">${esc(e.message)}</div>`;
        return;
    }
    if (seq !== _ms.seq) return;
    _ms.results = data.results || {};
    _ms.soundLabel = data.sound_label || null;
    _ms.took = `query ${data.embed_ms} ms · ranked in ${data.rank_ms} ms · ${data.dims} numbers per vector`;
    _ms.open = {};
    _ms.checks = {};
    msRender();
    box.scrollTop = 0;
}

// What asks: shown above the results, with the way back to the text
function msRenderSource() {
    const el = document.getElementById('msSource');
    if (!el) return;
    const s = _ms.source;
    if (s.type === 'text') { el.style.display = 'none'; el.innerHTML = ''; return; }
    let what;
    if (s.type === 'like') {
        const h = s.item;
        what = `${h.thumb ? `<img src="${esc(h.thumb)}" alt="">` : `<span class="ms-src-ico">${msKind(h.kind).icon}</span>`}
            <span><b>Similar to</b> ${msKind(h.kind).icon} ${esc(msKind(h.kind).label)} · ${esc(h.name)} · ${msClock(h.t)}</span>`;
    } else {
        const media = s.kind === 'image' ? `<img src="${esc(s.url)}" alt="">`
            : s.kind === 'video' ? `<video src="${esc(s.url)}" muted playsinline></video>`
            : '<span class="ms-src-ico">🔊</span>';
        what = `${media}<span><b>Your file</b> ${esc(s.name)} (${esc(s.kind)})${s.text ? ` + text “${esc(s.text)}”` : ''}</span>`;
    }
    el.style.display = '';
    el.innerHTML = `${what}<button class="vx-icon" onclick="msRun()" title="Back to the text in the search field">✕</button>`;
}

// ── The lists ───────────────────────────────────────────────────────────────

// The query text a check asks with — a "Similar" search has none
function msCheckQuery() {
    const s = _ms.source;
    if (s.type === 'text') return _ms.last;
    if (s.type === 'file') return s.text || '';
    return '';
}

function msRender() {
    const box = document.getElementById('msResults');
    if (!box) return;
    const llm = (_ms.info || {}).llm || {};
    const canCheck = llm.configured && llm.vision !== false && !!msCheckQuery();
    let html = `<div class="ms-took">${esc(_ms.took || '')}</div>`;
    if (_ms.soundLabel && _ms.on.has('audio')) {
        html += `<div class="ms-note">🔊 Sounds asked as “${esc(_ms.soundLabel)}” — the language model's English label of your query</div>`;
    }
    for (const kind of MS_KINDS) {
        if (!_ms.on.has(kind.k)) continue;
        const hits = msOrdered(kind.k);
        const all = !!_ms.open[kind.k];
        const shown = all ? hits : hits.slice(0, kind.tiles ? MS_SHOW.tiles : MS_SHOW.rows);
        const ck = _ms.checks[kind.k];
        let tools = '';
        if (kind.check && hits.length && canCheck) {
            tools = ck && ck.running
                ? `<span class="ms-check-state">🔎 the language model checks ${ck.done}/${ck.total} …</span>
                   <button class="ms-more ms-stop" onclick="msCancelCheck('${kind.k}')">stop</button>`
                : `<button class="ms-check" onclick="msCheck('${kind.k}')" title="A model that sees pictures judges the best ${llm.check_count || 24} one by one (~4 s each) and sorts them by its verdict">
                    🔎 ${ck ? 'Checked — again' : 'Check with the language model'}</button>`;
        }
        html += `<div class="ms-head">${kind.icon} ${esc(kind.label)} <span class="vx-count">${hits.length}</span>${tools}`
            + (hits.length > shown.length || all
                ? `<button class="ms-more" onclick="msMore('${kind.k}')">${all ? 'fewer' : `all ${hits.length}`}</button>` : '')
            + '</div>';
        if (kind.k === 'speech' && (hits.length || _ms.source.type === 'text')) {
            html += `<div class="ms-note ms-said">💬 The words of the 10 s around these moments, found in the same space as the
                pictures — where something is seen or heard. What was said, in whole passages with their context, the semantic
                search finds better: <button class="ms-link" onclick="msSearchSaid()">Search what was said →</button></div>`;
        }
        if (!hits.length) {
            const n = (((_ms.info || {}).counts || {}).kinds || {})[kind.k] || 0;
            html += `<div class="ms-none">${n ? 'nothing found' : 'none in the index'}</div>`;
            continue;
        }
        html += kind.tiles
            ? `<div class="ms-grid">${shown.map(h => msTile(kind.k, h)).join('')}</div>`
            : `<div class="ms-rows">${shown.map(h => msRow(kind.k, h)).join('')}</div>`;
    }
    box.innerHTML = html;
}

// A checked list: by the model's verdict, the unchecked after in their order
function msOrdered(kind) {
    const hits = _ms.results[kind] || [];
    const ck = _ms.checks[kind];
    if (!ck || !Object.keys(ck.scores || {}).length) return hits;
    const sc = (h) => ck.scores[String(h.moment)];
    return [...hits].sort((a, b) => {
        const x = sc(a), y = sc(b);
        if (x === undefined && y === undefined) return 0;
        if (x === undefined) return 1;
        if (y === undefined) return -1;
        return y - x;
    });
}

function msMore(k) { _ms.open[k] = !_ms.open[k]; msRender(); }

// The same words in the Text tab — keyword and meaning search
function msSearchSaid() {
    if (typeof svTab !== 'function') return;
    svTab('text');
    if (typeof svRun === 'function') svRun();
}

function msTagText(h) { return (h.tags || []).map(t => t[0]).join(', '); }

function msVerdict(kind, h) {
    const s = _ms.checks[kind]?.scores?.[String(h.moment)];
    if (s === undefined) return '';
    const cls = s >= 6 ? 'yes' : s >= 4 ? 'maybe' : 'no';
    return `<span class="ms-verdict ${cls}" title="The language model's verdict, 0–9">${s >= 6 ? '✓' : s >= 4 ? '~' : '✗'} ${s.toFixed(1)}</span>`;
}

function msTile(kind, h) {
    const n = (_ms.results[kind] || []).indexOf(h);
    const s = _ms.checks[kind]?.scores?.[String(h.moment)];
    const dim = s !== undefined && s < 4 ? ' dim' : '';
    return `
        <button class="ms-tile${dim}${msIsActive(kind, n)}" data-kind="${kind}" data-n="${n}" onclick="msPlay('${kind}', ${n})"
                title="${esc(h.path)} · ${msClock(h.t)}${h.tags?.length ? ' · closest tags: ' + esc(msTagText(h)) : ''}">
            ${h.thumb ? `<img loading="lazy" src="${esc(h.thumb)}" alt="">` : '<span class="ms-noimg">no picture</span>'}
            <span class="ms-rank">${n + 1}</span>
            <span class="ms-score">${h.score.toFixed(3)}</span>
            ${msVerdict(kind, h)}
            ${kind !== 'image' ? `<span class="ms-badge">${msKind(kind).icon}</span>` : ''}
            <span class="ms-cap">${esc(h.name)} · ${msClock(h.t)}</span>
        </button>`;
}

function msRow(kind, h) {
    const n = (_ms.results[kind] || []).indexOf(h);
    const span = `${msClock(h.start)}–${msClock(h.start + h.dur)}`;
    const said = h.text ? esc(h.text) : '<i>nothing said</i>';
    return `
        <button class="ms-row ${kind === 'speech' ? 'ms-row-speech' : ''}${msIsActive(kind, n)}" data-kind="${kind}" data-n="${n}"
                onclick="msPlay('${kind}', ${n})" title="${esc(h.path)}">
            <span class="ms-rank">${n + 1}</span>
            ${h.thumb ? `<img loading="lazy" src="${esc(h.thumb)}" alt="">` : `<span class="ms-row-ico">${msKind(kind).icon}</span>`}
            <span class="ms-row-main">
                <span class="ms-row-name">${esc(h.name)} · ${span}${h.tags?.length ? ` · <span class="ms-tagline">${esc(msTagText(h))}</span>` : ''}</span>
                <span class="ms-row-text">${said}</span>
            </span>
            <span class="ms-score">${h.score.toFixed(3)}</span>
        </button>`;
}

// ── The check by the language model ────────────────────────────────────────

async function msCheck(kind) {
    const q = msCheckQuery();
    const count = ((_ms.info || {}).llm || {}).check_count || 24;
    const hits = (_ms.results[kind] || []).filter(h => h.thumb).slice(0, count);
    if (!q || !hits.length || _ms.checking.has(kind)) return;
    msCancelCheck(kind);
    // The results the check is for: a search that starts while it is asked
    // for gets none of it (its scores would land on other moments)
    const seq = _ms.seq;
    _ms.checking.add(kind);
    let r;
    try {
        r = await api('/api/media-search/check', { method: 'POST',
            body: { query: q, moments: [...new Set(hits.map(h => h.moment))] } });
    } catch (e) { if (typeof Toast !== 'undefined') Toast.error(e.message); return; }
    finally { _ms.checking.delete(kind); }
    if (seq !== _ms.seq) {
        fetch(`/api/media-search/check/${r.id}/cancel`, { method: 'POST' }).catch(() => {});
        return;
    }
    const ck = { id: r.id, total: r.total, done: 0, scores: {}, running: true, seq };
    _ms.checks[kind] = ck;
    msRender();
    const tick = async () => {
        if (_ms.checks[kind] !== ck || ck.seq !== _ms.seq) return;
        let st;
        try { st = await api('/api/media-search/check/' + ck.id); } catch (e) { ck.running = false; msRender(); return; }
        Object.assign(ck, { done: st.done, total: st.total, scores: st.scores || {}, running: st.running });
        msRender();
        if (st.error && typeof Toast !== 'undefined') Toast.error('Check: ' + st.error);
        if (st.running) ck.timer = setTimeout(tick, 1500);
    };
    ck.timer = setTimeout(tick, 1200);
}

function msCancelCheck(kind) {
    const ck = _ms.checks[kind];
    if (!ck) return;
    clearTimeout(ck.timer);
    if (ck.running) {
        fetch(`/api/media-search/check/${ck.id}/cancel`, { method: 'POST' }).catch(() => {});
        ck.running = false;
    }
    msRender();
}

function msCancelChecks() {
    for (const k of Object.keys(_ms.checks)) {
        const ck = _ms.checks[k];
        clearTimeout(ck.timer);
        if (ck.running) fetch(`/api/media-search/check/${ck.id}/cancel`, { method: 'POST' }).catch(() => {});
    }
    _ms.checks = {};
}

// ── The preview ─────────────────────────────────────────────────────────────

// Play a hit in the preview column: everything with a picture as the 10 s
// clip of its moment (with its sound), a sound of an audio file as itself.
// A preview is cut on its first play (~1–3 s).
function msPlay(kind, n) {
    const h = (_ms.results[kind] || [])[n];
    const pv = document.getElementById('msPreview');
    if (!h || !pv) return;
    _ms.active = { kind, n };
    document.querySelectorAll('#msResults .active').forEach(el => el.classList.remove('active'));
    document.querySelector(`#msResults [data-kind="${kind}"][data-n="${n}"]`)?.classList.add('active');
    const media = h.video
        ? `<video class="ms-video" src="${esc(h.clip)}" ${h.thumb ? `poster="${esc(h.thumb)}"` : ''} controls autoplay playsinline></video>`
        : `<audio class="ms-audio" src="${esc(h.clip)}" controls autoplay></audio>`;
    const at = h.video
        ? `moment ${msClock(h.t)}`
        : `${msClock(h.start)}–${msClock(h.start + h.dur)}`;
    const tags = (h.tags || []).map(t => `<span class="ms-tag">${esc(t[0])} <small>${t[1].toFixed(2)}</small></span>`).join('');
    const verdict = _ms.checks[kind]?.scores?.[String(h.moment)];
    pv.innerHTML = `
        ${media}
        <div class="ms-pv-kind">${msKind(kind).icon} ${esc(msKind(kind).label)} · #${n + 1} · score ${h.score.toFixed(3)}${
            verdict !== undefined ? ` · language model ${verdict.toFixed(1)}/9` : ''}</div>
        <div class="ms-pv-name">${esc(h.name)}</div>
        <div class="ms-pv-meta">${at}${h.repeat ? ' · looks like the moment before (its picture)' : ''}</div>
        <div class="ms-pv-folder" title="${esc(h.path)}">${esc(h.file_dir || '')}</div>
        <div class="ms-pv-actions">
            <button class="vx-btn" onclick="msOpenAt('${kind}', ${n}, false)" title="Play it in the player of the main window">▶ In the player</button>
            <button class="vx-btn" onclick="msOpenAt('${kind}', ${n}, true)" title="Open the transcript at this moment">📄 Transcript</button>
            <button class="vx-btn" onclick="msLike('${kind}', ${n})" title="Search with this ${esc(msKind(kind).label.toLowerCase())} itself — in every kind">🔁 Similar</button>
            ${h.thumb ? `<button class="vx-btn" onclick="msDescribe(${h.moment}, this)" title="The language model describes the still in one sentence">📝 Describe</button>` : ''}
        </div>
        <div class="ms-pv-desc" id="msPvDesc"></div>
        ${tags ? `<div class="ms-pv-tags"><b>Closest tags</b>${tags}</div>` : ''}
        <div class="ms-pv-said"><b>Said around this moment</b>${h.text ? esc(h.text) : '<i>nothing</i>'}</div>`;
    // …and the other way round: the preview stops the main player
    pv.querySelector('video, audio')?.addEventListener('play', () => {
        try { svPlayer()?.pause(); } catch (e) { /* none */ }
    });
    if (kind === 'description') msDescribe(h.moment, null);
}

// The hit in the preview stays marked when the results are drawn again
// (each check tick redraws them)
function msIsActive(kind, n) {
    return _ms.active && _ms.active.kind === kind && _ms.active.n === n ? ' active' : '';
}

function msPausePreview() {
    document.querySelectorAll('#msPreview video, #msPreview audio').forEach(m => {
        try { m.pause(); } catch (e) { /* gone */ }
    });
}

// Leaving the tab or the search: the preview stops, as Find's does
function msLeave() { msPausePreview(); }

function msOpenAt(kind, n, transcript) {
    const h = (_ms.results[kind] || [])[n];
    if (!h) return;
    msPausePreview();               // one sound at a time
    const t = h.video ? Math.max(0, h.t - 2) : h.start;
    if (transcript) svGoto(h.filename, t, h.file_dir);
    else svPlay(h.filename, t, h.file_dir);
}

async function msDescribe(moment, btn) {
    const box = document.getElementById('msPvDesc');
    if (!box) return;
    if (btn) btn.disabled = true;
    box.innerHTML = '<span class="ms-hint">📝 …</span>';
    try {
        const r = await api('/api/media-search/describe/' + moment, { method: 'POST', body: {} });
        box.innerHTML = `<b>Described${r.model ? ' by ' + esc(r.model) : ''}</b>${esc(r.text)}`;
    } catch (e) {
        box.innerHTML = `<span class="ms-err">${esc(e.message)}</span>`;
    } finally {
        if (btn) btn.disabled = false;
    }
}
