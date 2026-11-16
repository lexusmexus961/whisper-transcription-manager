// ── Pins: window tabs of one's own in the rail ──────────────────────────
// Up to six window tabs (Voice map, Same audio, …) sit in the rail's
// "Pinned" group and in Home's pinned row — the same list, a UI preference
// on the server, so it is the same in every browser. Places that already
// have a rail item, and actions, can't be pinned.
//
// Pin: the pin on a tab (shown on hover) or on a Home tile, or right-click
// the tab or tile. Unpin: the same, or × on the pin in the rail. Drag a
// pin in the rail to reorder. While a pinned tab is open its pin is lit
// and its window's rail item carries a dot.

const Pins = (() => {
    const MAX = 6;
    const DEFAULT = ['voices:map', 'voices:extract', 'find:audio', 'find:conversations'];
    // The tabs of the search window that are their own place in the rail
    const CORE_SV_TABS = {chat: 'ask', analysis: 'analyses'};

    const catalog = () => (typeof Home !== 'undefined' ? Home.TABS : {});
    /* Only window tabs ("window:tab") can be pinned. */
    const pinnable = id => id.includes(':') && !!catalog()[id];

    function list() {
        const p = typeof Shell !== 'undefined' ? Shell.pref('pins') : null;
        return (Array.isArray(p) ? p : DEFAULT).filter(pinnable);
    }

    function save(pins) {
        Shell.setPref('pins', pins);
        render();
        if (typeof Home !== 'undefined') Home.redraw();
    }

    function has(id) { return list().includes(id); }

    function toggle(id) {
        if (!pinnable(id)) return;
        const pins = list();
        const i = pins.indexOf(id);
        if (i >= 0) {
            pins.splice(i, 1);
            save(pins);
            return;
        }
        if (pins.length >= MAX) {
            if (typeof Toast !== 'undefined') Toast.info(`At most ${MAX} pins — unpin one first (× on a pin in the rail)`);
            return;
        }
        pins.push(id);
        save(pins);
        const name = catalog()[id][0];
        if (typeof Toast !== 'undefined') Toast.success(`${name} pinned to the rail`);
    }

    function move(id, before) {
        const pins = list().filter(p => p !== id);
        const at = before ? pins.indexOf(before) : -1;
        pins.splice(at < 0 ? pins.length : at, 0, id);
        save(pins);
    }

    /* The pin of what is on screen now (a window and its tab), or null. */
    function current() {
        if (typeof Shell === 'undefined') return null;
        const place = Shell.current;
        if (['find', 'ask', 'analyses'].includes(place) && typeof svState !== 'undefined') return 'find:' + svState.tab;
        if (place === 'voices' && typeof VX !== 'undefined') return 'voices:' + VX.tab;
        return null;
    }

    /* ── The rail's group ────────────────────────────────────────────── */

    function render() {
        const box = document.getElementById('railPins');
        if (!box) return;
        const pins = list();
        if (!pins.length) { box.innerHTML = ''; return; }
        const cat = catalog();
        box.innerHTML = '<div class="rail-head">Pinned</div>' + pins.map(id => {
            const [label, where, icon] = cat[id];
            return `<button type="button" class="rail-item rail-pin" data-pin="${esc(id)}" draggable="true"`
                + ` onclick="Pins.open('${escJs(id)}')" title="${esc(label)} (${esc(where)}) — drag to reorder, right-click to unpin">`
                + `<svg class="ui-ico"><use href="#${icon}"/></svg><span class="rail-label">${esc(label)}</span>`
                + `<span class="rail-unpin" role="button" aria-label="Unpin ${esc(label)}" title="Unpin"`
                + ` onclick="event.stopPropagation();Pins.toggle('${escJs(id)}')">×</span></button>`;
        }).join('');
        sync();
    }

    function open(id) {
        if (typeof Home !== 'undefined') Home.open(id);
    }

    /* Light the pin of what is open, and give its window's item a dot. */
    function sync() {
        const now = current();
        const lit = now && has(now) ? now : null;
        document.querySelectorAll('#railPins .rail-pin').forEach(b => {
            const on = b.dataset.pin === lit;
            b.classList.toggle('on', on);
            b.setAttribute('aria-current', on ? 'page' : 'false');
        });
        const parent = lit ? (lit.startsWith('voices:') ? 'voices' : Shell.current) : null;
        document.querySelectorAll('#rail .rail-item[data-place]').forEach(b => {
            const par = b.dataset.place === parent;
            b.classList.toggle('par', par);
            if (par) { b.classList.remove('on'); b.setAttribute('aria-current', 'false'); }
        });
        decorate();
    }

    /* ── Pin buttons on the windows' tabs ────────────────────────────── */

    function tabId(btn) {
        if (btn.closest('#vxTabs')) return btn.dataset.tab ? 'voices:' + btn.dataset.tab : null;
        const m = /^svTabBtn(\w+)$/.exec(btn.id || '');
        if (!m) return null;
        const tab = m[1][0].toLowerCase() + m[1].slice(1);
        return CORE_SV_TABS[tab] ? null : 'find:' + tab;
    }

    function decorate() {
        const pins = list();
        document.querySelectorAll('#vxTabs .vx-tab, #svOverlay .vx-tabs .vx-tab').forEach(btn => {
            const id = tabId(btn);
            if (!id || !pinnable(id)) return;
            let pin = btn.querySelector('.tab-pin');
            if (!pin) {
                pin = document.createElement('span');
                pin.className = 'tab-pin';
                pin.setAttribute('role', 'button');
                pin.innerHTML = '<svg class="ui-ico"><use href="#i-pin"/></svg>';
                pin.addEventListener('click', e => { e.stopPropagation(); e.preventDefault(); toggle(id); });
                btn.appendChild(pin);
                btn.addEventListener('contextmenu', e => { e.preventDefault(); toggle(id); });
            }
            const on = pins.includes(id);
            pin.classList.toggle('on', on);
            pin.title = on ? 'Unpin from the rail' : 'Pin to the rail';
            pin.setAttribute('aria-label', pin.title);
        });
    }

    /* ── Dragging in the rail ────────────────────────────────────────── */

    function wireDrag() {
        const box = document.getElementById('railPins');
        if (!box) return;
        let dragged = null;
        box.addEventListener('dragstart', e => {
            const b = e.target.closest('.rail-pin');
            if (!b) return;
            dragged = b.dataset.pin;
            e.dataTransfer.effectAllowed = 'move';
            try { e.dataTransfer.setData('text/plain', dragged); } catch (err) { /* some browsers */ }
            b.classList.add('dragging');
        });
        box.addEventListener('dragover', e => {
            if (!dragged) return;
            e.preventDefault();
            box.querySelectorAll('.rail-pin').forEach(b => b.classList.toggle('drop-before', b === e.target.closest('.rail-pin') && b.dataset.pin !== dragged));
        });
        box.addEventListener('drop', e => {
            if (!dragged) return;
            e.preventDefault();
            const over = e.target.closest('.rail-pin');
            move(dragged, over && over.dataset.pin !== dragged ? over.dataset.pin : null);
            dragged = null;
        });
        box.addEventListener('dragend', () => {
            dragged = null;
            box.querySelectorAll('.rail-pin').forEach(b => b.classList.remove('dragging', 'drop-before'));
        });
        box.addEventListener('contextmenu', e => {
            const b = e.target.closest('.rail-pin');
            if (!b) return;
            e.preventDefault();
            toggle(b.dataset.pin);
        });
    }

    function init() {
        wireDrag();
        if (typeof Shell !== 'undefined') Shell.onPrefs(render);
        render();
    }

    document.addEventListener('DOMContentLoaded', init);
    return {list, has, toggle, move, open, render, sync, pinnable, MAX, DEFAULT};
})();
