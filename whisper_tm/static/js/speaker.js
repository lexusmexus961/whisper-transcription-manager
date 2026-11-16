/* Speaker recognition in the file view.
 *
 * Answers the two questions that matter:
 *   1. WHO was involved in this recording?
 *   2. WHERE roughly did they speak?
 *
 * The second question is why there are minutes here instead of a
 * percentage: a number without a timestamp can't be listened back to. Every
 * timestamp jumps into playback.
 *
 * The display is deliberately talkative where it is uncertain. "18 of 94
 * windows" says more than a checkmark, and the slider below it makes visible
 * that the threshold is a decision, not a law of nature.
 */

// The slider starts at the setting speaker.threshold_gallery, default
// 0.75 — and that is deliberately neither of the two numbers you'd expect.
// 0.55 applies to two INDIVIDUAL voices, 0.62 WITHIN one recording. For
// a recording against the WHOLE gallery it has to be higher: with a dozen
// people and several windows there are hundreds of chances to trip the
// threshold by accident. In a leave-one-out measurement, 0.55 gave about
// half of the recordings at least one wrong name, 0.75 well under 1% —
// while still finding the large majority of the people.
//
// null stands for the setting: the request then carries no threshold, the
// server applies the configured one and reports it back, and the slider
// starts there. A number here would override the setting on every
// request. Only a value picked with the slider is sent — and only for the
// file it was picked on; the next file view starts at the setting again.
let speakerThreshold = null;
let speakerFile = null;
// Every request gets a number; only the newest may draw. Switching files
// (or dragging the slider) quickly otherwise let an older, slower answer
// land last — and "Apply names" then wrote file A's people into file B.
let _speakerSeq = 0;

function speakerMinute(min) {
    return min >= 60 ? `${Math.floor(min / 60)}:${String(min % 60).padStart(2, '0')} h`
                     : `${min} min`;
}

/* The "🗣 Map" buttons only where the map has something to show.
 *
 * A recording without voice vectors (transcribed without telling speakers
 * apart, or with an older version) opened a map with nothing lit, which
 * looked broken. Now the button says why it can't, and a voice without windows (too little speech alone)
 * says so on its own row. */
function speakerMapButtons(d) {
    const labels = new Set((d.labels || []).map(l => l.label));
    document.querySelectorAll('.sp-map-btn').forEach(b => {
        const label = b.dataset.label;
        const none = !d.windows || (label && !labels.has(label));
        b.disabled = none;
        b.title = !d.windows
            ? 'No voices on the map for this recording — it has no voice vectors. '
              + 'A "Voices only" job adds them (queue editor → From library).'
            : (none ? `${label} has no voice windows — too little speech on its own `
                      + '(under 4 seconds without anyone else talking)'
                    : (label ? 'Highlight this voice on the voice map'
                             : 'Show this recording on the voice map'));
    });
}

async function loadSpeakerRecognition(fn) {
    const box = document.getElementById('speakerRecognition');
    if (!box) return;
    if (fn !== speakerFile) speakerThreshold = null;   // new file: the setting again
    speakerFile = fn;
    const seq = ++_speakerSeq;
    // The hits belong to the file they were computed for — none until the
    // answer for THIS file is in.
    window._speakerHits = [];
    box.innerHTML = '<div class="sp-rec-loading">Comparing voices …</div>';
    const stale = () => seq !== _speakerSeq || fn !== speakerFile || !box.isConnected;
    let d;
    try {
        // The path as a parameter: an absolute one loses its leading slash
        // to the router's redirect in the URL path (api/speaker.py).
        const r = await fetch(`/api/speaker/recording?file=${encodeURIComponent(fn)}`
                              + (speakerThreshold === null ? ''
                                 : `&threshold=${speakerThreshold}`));
        d = await r.json();
    } catch (e) {
        if (stale()) return;
        box.innerHTML = `<div class="sp-rec-empty">Voice matching unavailable: ${esc(String(e))}</div>`;
        return;
    }
    if (stale()) return;
    if (d.error) {
        box.innerHTML = `<div class="sp-rec-empty">${esc(d.error)}</div>`;
        return;
    }
    speakerMapButtons(d);
    if (!d.windows) {
        // Say honestly why there's nothing here — and what to do about it.
        box.innerHTML = `<div class="sp-rec-empty">${esc(d.hint || 'No voice vectors.')}</div>`;
        return;
    }

    // The value the server applied: the setting, or the slider value
    // clamped to 0..1. The range stretches to take it in — the setting is
    // a free number, and a thumb pinned at the end would show a different
    // threshold than the one in effect.
    const threshold = d.threshold;
    let html = `
      <div class="sp-rec-head">
        <span class="sp-rec-title">Who was there?</span>
        <span class="sp-rec-meta">${plural(d.windows, 'window')} · ${plural(d.labels.length, 'ID')} · ${esc(d.model)}</span>
        <span class="sp-rec-slider">
          Threshold <b id="spkThresholdValue">${threshold.toFixed(2)}</b>
          <input type="range" min="${Math.min(0.40, threshold)}" max="${Math.max(0.95, threshold)}"
                 step="0.01" value="${threshold}"
                 oninput="document.getElementById('spkThresholdValue').textContent=(+this.value).toFixed(2)"
                 onchange="setSpeakerThreshold(this.value)">
        </span>
      </div>`;

    if (!d.named_people) {
        html += `<div class="sp-rec-empty">No person named yet — without a gallery
                 the question "who was there" can't be answered.
                 Name voices in the voice map.</div>`;
    } else if (!d.people.length) {
        html += `<div class="sp-rec-empty">None of the ${d.named_people} named
                 people reach the threshold ${threshold.toFixed(2)}.
                 This can mean: no one known was there — or the
                 recording is too loud.</div>`;
    } else {
        html += '<div class="sp-rec-list">';
        for (const p of d.people) {
            const share = Math.round(100 * p.windows / p.of_windows);
            const times = p.timestamps.slice(0, 24).map(m =>
                `<button class="sp-rec-time" onclick="seekTo(${m * 60})"
                         title="Jump to minute ${m}">${speakerMinute(m)}</button>`
            ).join('');
            const more = p.timestamps.length > 24
                ? `<span class="sp-rec-more">+${p.timestamps.length - 24} more</span>` : '';
            html += `
              <div class="sp-rec-person">
                <div class="sp-rec-row">
                  <span class="sp-rec-name">${esc(p.person)}</span>
                  <span class="sp-rec-bar"><i style="width:${share}%"></i></span>
                  <span class="sp-rec-numbers">${p.windows} of ${plural(p.of_windows, 'window')}
                    · best score ${p.best_value.toFixed(3)}
                    · ${Math.round(100 * p.voices_share)}% of ${plural(p.voices_total, 'known voice')}
                    · ${esc(p.labels.join(', '))}</span>
                </div>
                <div class="sp-rec-times">${times}${more}</div>
              </div>`;
        }
        html += '</div>';
        html += `<div class="sp-rec-foot">
                   <button class="btn-save-names" onclick="applySpeakerNames()"
                           title="Enter the recognized names into the speaker fields above">
                     ↑ Apply names to the speaker list</button>
                   <span class="sp-rec-warn">A window with several people hits the
                     loudest one — the assignment says WHO, not seamlessly WHEN.</span>
                 </div>`;
        window._speakerHits = d.people;
    }
    box.innerHTML = html;
}

function setSpeakerThreshold(value) {
    const v = parseFloat(value);
    // Something unusable falls back to the setting, not to a number from here
    speakerThreshold = Number.isNaN(v) ? null : Math.min(Math.max(v, 0), 1);
    if (speakerFile) loadSpeakerRecognition(speakerFile);
}

/* Enter the recognized names into the existing speaker fields.
 *
 * This is NOT saved automatically: applying is a suggestion, saving
 * remains a deliberate click on "Save names". An ID that matches several
 * people gets both names with a slash — it's then probably a mix,
 * and that should be visible rather than smoothed over.
 */
function applySpeakerNames() {
    const hits = window._speakerHits || [];
    const perLabel = {};
    for (const p of hits) {
        for (const k of p.labels) {
            (perLabel[k] = perLabel[k] || []).push(p);
        }
    }
    let set_flag = 0;
    document.querySelectorAll('.sp-stat-name[data-speaker-id]').forEach(inp => {
        const people = perLabel[inp.dataset.speakerId];
        if (!people || !people.length) return;
        people.sort((a, b) => b.best_value - a.best_value);
        inp.value = people.map(p => p.person).join(' / ');
        inp.classList.add('sp-stat-name-suggestion');
        // Setting .value fires no change event: without this the names
        // never reached speakerNames, "Save names" saved the old ones and
        // the suggestions vanished on the next render. State per field
        // here; the labels are redrawn once below (fileview.js).
        if (typeof speakerNameFromInput === 'function') speakerNameFromInput(inp);
        else if (typeof onSpeakerRename === 'function') onSpeakerRename(inp);
        set_flag++;
    });
    if (set_flag && typeof speakerNamesChanged === 'function') speakerNamesChanged();
    const st = document.getElementById('saveNameStatus');
    if (st) {
        st.textContent = set_flag
            ? `${plural(set_flag, 'suggestion')} entered — not yet saved`
            : 'No matching ID found';
    }
}
