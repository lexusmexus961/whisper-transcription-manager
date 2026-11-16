// ── Transcript Editor ──

// ══════════════════════════════════════════════════════════════
//  Inline Segment Edit
// ══════════════════════════════════════════════════════════════

function editSegment(idx) {
    const d = activeTranscript;
    if (!d || !d.segments || idx < 0 || idx >= d.segments.length) return;

    const seg = d.segments[idx];
    const el = document.getElementById(`seg-txt-${idx}`);
    if (!el) return;

    // Already editing?
    if (el.querySelector('.seg-edit-input')) return;

    const oldText = seg.text || '';

    // Replace text with input
    el.innerHTML = `
        <div style="display:flex;gap:4px;align-items:flex-start" onclick="event.stopPropagation()">
            <textarea class="seg-edit-input" style="flex:1;padding:6px 8px;border:1px solid var(--accent);border-radius:4px;background:var(--bg-0);color:var(--tx-1);font-size:0.88rem;font-family:inherit;resize:vertical;min-height:2.2em;line-height:1.5">${esc(oldText)}</textarea>
            <button onclick="_saveSegEdit(${idx})" style="padding:4px 8px;background:var(--accent);color:#fff;border:none;border-radius:4px;cursor:pointer;font-size:0.78rem;white-space:nowrap">💾</button>
            <button onclick="_cancelSegEdit(${idx})" style="padding:4px 8px;background:var(--bg-2);color:var(--tx-2);border:1px solid var(--border);border-radius:4px;cursor:pointer;font-size:0.78rem">✕</button>
        </div>
    `;

    // Focus and select all
    const ta = el.querySelector('textarea');
    ta.focus();
    ta.select();

    // Enter = save, Escape = cancel
    ta.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            _saveSegEdit(idx);
        }
        if (e.key === 'Escape') {
            _cancelSegEdit(idx);
        }
    });
}

async function _saveSegEdit(idx) {
    const el = document.getElementById(`seg-txt-${idx}`);
    const ta = el?.querySelector('textarea');
    if (!ta) return;

    const newText = ta.value.trim();
    // activeFile first: the endpoints resolve the name against the current
    // root, and only activeFile refers to it (see renderTranscript).
    const filename = activeFile || activeTranscript?.filename;
    // The transcript this edit belongs to — the user may switch files
    // while the request runs.
    const tx = activeTranscript;

    try {
        const r = await fetch('/api/edit_segment', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({filename, index: idx, text: newText})
        });
        const d = await r.json();
        if (d.error) throw new Error(d.error);

        // Saved on the server. Another file opened in the meantime: its
        // in-memory segment `idx` is NOT the edited one — leave it alone
        // (before, it was overwritten and ended up in that file's export).
        if (activeTranscript !== tx || !tx?.segments?.[idx]) {
            if (typeof Toast !== 'undefined') Toast.success('Segment saved');
            return;
        }

        // Update local data
        tx.segments[idx].text = newText;

        // Update DOM
        el.textContent = newText;

        // Update data-text for search
        const segEl = el.closest('.seg');
        if (segEl) segEl.dataset.text = newText.toLowerCase();
        _segEditWords(tx);

        if (typeof Toast !== 'undefined') Toast.success('Segment saved');
    } catch(e) {
        if (typeof Toast !== 'undefined') Toast.error('Error: ' + e.message);
    }
}

// The word count in the header follows an edit — it showed the old one
// until the file was opened again
function _segEditWords(tx) {
    const n = tx.segments.map(s => s.text || '').join(' ').split(/\s+/).filter(Boolean).length;
    for (const c of document.querySelectorAll('.fi-chip')) {
        if (c.querySelector('small')?.textContent === 'Words' && c.lastChild?.nodeType === 3)
            c.lastChild.textContent = n.toLocaleString();
    }
    const f = (typeof fileByName === 'function') ? fileByName(activeFile) : null;
    if (f) f.tx_word_count = n;
}

function _cancelSegEdit(idx) {
    const seg = activeTranscript?.segments?.[idx];
    if (!seg) return;
    const el = document.getElementById(`seg-txt-${idx}`);
    if (el) el.textContent = seg.text || '';
}


// ══════════════════════════════════════════════════════════════
//  Find & Replace
// ══════════════════════════════════════════════════════════════

function openFindReplace() {
    if (!activeTranscript?.segments?.length) {
        if (typeof Toast !== 'undefined') Toast.warn('No transcript loaded');
        return;
    }
    document.getElementById('frFind').value = '';
    document.getElementById('frReplace').value = '';
    document.getElementById('frCaseSensitive').checked = false;
    document.getElementById('frWholeWord').checked = false;
    document.getElementById('frPreview').style.display = 'none';
    document.getElementById('frPreview').innerHTML = '';
    document.getElementById('frStatus').textContent = '';
    document.getElementById('btnFrGo').disabled = true;

    // If there's a search query, pre-fill
    const searchVal = document.getElementById('txSearch')?.value || '';
    if (searchVal) {
        document.getElementById('frFind').value = searchVal;
        setTimeout(_frPreview, 50);
    }

    document.getElementById('findReplaceModal').classList.add('visible');
    document.getElementById('frFind').focus();
}

function closeFindReplace() {
    document.getElementById('findReplaceModal').classList.remove('visible');
}

function _frPreview() {
    const findStr = document.getElementById('frFind').value;
    const replaceStr = document.getElementById('frReplace').value;
    const caseSensitive = document.getElementById('frCaseSensitive').checked;
    const wholeWord = document.getElementById('frWholeWord').checked;
    const previewEl = document.getElementById('frPreview');
    const statusEl = document.getElementById('frStatus');
    const btn = document.getElementById('btnFrGo');

    if (!findStr) {
        previewEl.style.display = 'none';
        statusEl.textContent = '';
        btn.disabled = true;
        return;
    }

    const segments = activeTranscript?.segments || [];

    // Build regex
    let pattern;
    try {
        const escaped = findStr.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
        // Whole word in any script: JavaScript's \b knows ASCII letters only
        // ("café", "Москва" found nothing, "caf" matched inside "cafés").
        // The same rule as the server's (?<!\w)…(?!\w): letters, digits, _
        const src = wholeWord ? `(?<![\\p{L}\\p{N}_])${escaped}(?![\\p{L}\\p{N}_])` : escaped;
        pattern = new RegExp(src, caseSensitive ? 'gu' : 'giu');
    } catch(e) {
        statusEl.textContent = 'Invalid search term';
        btn.disabled = true;
        return;
    }

    let totalMatches = 0;
    let matchSegments = 0;
    let previewHtml = '';
    const MAX_PREVIEW = 20;

    segments.forEach((seg, idx) => {
        const text = seg.text || '';
        const matches = text.match(pattern);
        if (!matches) return;
        totalMatches += matches.length;
        matchSegments++;
        if (matchSegments <= MAX_PREVIEW) {
            // Highlight matches - escape all text parts properly
            const parts = [];
            let lastIdx = 0;
            let m;
            const p2 = new RegExp(pattern.source, pattern.flags);
            while ((m = p2.exec(text)) !== null) {
                if (m.index > lastIdx) parts.push(esc(text.slice(lastIdx, m.index)));
                parts.push(`<span style="background:#f59e0b40;color:var(--orange);font-weight:600;text-decoration:line-through">${esc(m[0])}</span>`);
                parts.push(`<span style="background:#22c55e30;color:var(--green);font-weight:600">${esc(replaceStr)}</span>`);
                lastIdx = m.index + m[0].length;
                if (!pattern.global) break;
            }
            if (lastIdx < text.length) parts.push(esc(text.slice(lastIdx)));
            previewHtml += `<div style="padding:4px 0;border-bottom:1px solid var(--border)">`;
            previewHtml += `<span style="font-family:var(--font-mono);font-size:0.72rem;color:var(--tx-3);margin-right:6px">${esc(seg.start_fmt || '')}</span>`;
            previewHtml += parts.join('');
            previewHtml += `</div>`;
        }
    });

    if (totalMatches === 0) {
        previewEl.style.display = 'none';
        statusEl.textContent = `No matches for "${findStr}"`;
        statusEl.style.color = 'var(--tx-3)';
        btn.disabled = true;
    } else {
        previewEl.style.display = '';
        previewEl.innerHTML = previewHtml;
        if (matchSegments > MAX_PREVIEW) {
            previewEl.innerHTML += `<div style="color:var(--tx-3);font-size:0.76rem;padding:4px 0">… and ${plural(matchSegments - MAX_PREVIEW, 'more segment')}</div>`;
        }
        statusEl.textContent = `${plural(totalMatches, 'match', 'matches')} in ${plural(matchSegments, 'segment')}`;
        statusEl.style.color = 'var(--accent)';
        btn.textContent = `✏️ Replace all (${totalMatches}×)`;
        // Enable even when replacing with empty (= delete)
        btn.disabled = false;
    }
}

async function doFindReplace() {
    const findStr = document.getElementById('frFind').value;
    const replaceStr = document.getElementById('frReplace').value;
    const caseSensitive = document.getElementById('frCaseSensitive').checked;
    const wholeWord = document.getElementById('frWholeWord').checked;
    // activeFile first (see _saveSegEdit) — with the stored name first, the
    // reload below asked loadTranscript for a name that isn't activeFile,
    // which it ignores: the view kept the text from before the replace.
    const filename = activeFile || activeTranscript?.filename;

    if (!findStr || !filename) return;

    const btn = document.getElementById('btnFrGo');
    btn.disabled = true;
    btn.textContent = '⏳ Replacing…';

    try {
        const r = await fetch('/api/find_replace', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                filename, find: findStr, replace: replaceStr,
                case_sensitive: caseSensitive, whole_word: wholeWord,
            })
        });
        const d = await r.json();
        if (d.error) throw new Error(d.error);

        closeFindReplace();

        if (typeof Toast !== 'undefined') {
            Toast.success(`${d.replaced}× replaced in ${plural(d.segments_changed, 'segment')}`);
        }

        // Reload transcript — only if it is still the open file
        if (activeFile === filename) await loadTranscript(filename);

    } catch(e) {
        if (typeof Toast !== 'undefined') Toast.error('Error: ' + e.message);
        btn.disabled = false;
        btn.textContent = '✏️ Replace all';
    }
}
