// ── Video player (Plyr, hosted locally under /static/vendor/plyr/) ──
//
// Replaces the native browser controls with Plyr: speed selection
// (0.5–3x), 5s jumps, PiP, fullscreen, keyboard shortcuts
// (space/arrows/M/F when the player is focused). Important:
// Plyr keeps the original <video id="videoPlayer"> with the standard
// Media API — seekTo(), the clip editor and the playback interval in
// fileview.js still access the element directly and need no
// adjustment. If the vendor script fails to load, the native
// controls (controls attribute) remain as a fallback.
let plyrPlayer = null;

document.addEventListener('DOMContentLoaded', () => {
    const video = document.getElementById('videoPlayer');
    if (!video || typeof Plyr === 'undefined') return;

    plyrPlayer = new Plyr(video, {
        iconUrl: '/static/vendor/plyr/plyr.svg',
        // Default points to cdn.plyr.io — the app also runs offline/on the LAN
        blankVideo: '',
        controls: ['play-large', 'rewind', 'play', 'fast-forward',
                   'progress', 'current-time', 'duration', 'mute',
                   'volume', 'settings', 'pip', 'fullscreen'],
        settings: ['speed'],
        speed: { selected: 1,
                 options: [0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.5, 3] },
        seekTime: 5,
        keyboard: { focused: true, global: false },
        tooltips: { controls: true, seek: true },
        // remembers volume + selected speed across sessions
        storage: { enabled: true, key: 'wtm_player' },
        invertTime: false,
        i18n: {
            play: 'Play', pause: 'Pause',
            rewind: 'Rewind 5s', fastForward: 'Forward 5s',
            seek: 'Seek', seekLabel: '{currentTime} of {duration}',
            played: 'Played', buffered: 'Buffered',
            currentTime: 'Current time', duration: 'Duration',
            volume: 'Volume', mute: 'Mute', unmute: 'Unmute',
            enterFullscreen: 'Enter fullscreen', exitFullscreen: 'Exit fullscreen',
            settings: 'Settings', speed: 'Speed',
            normal: 'Normal', pip: 'Picture-in-picture',
            menuBack: 'Back',
        },
    });

    // Re-parent the subtitle overlay into the Plyr container: only that way
    // does it stay above the video in fullscreen too (the fullscreen
    // element is the Plyr wrapper).
    plyrPlayer.on('ready', () => {
        const ov = document.getElementById('subtitleOverlay');
        if (ov && plyrPlayer.elements.container) {
            plyrPlayer.elements.container.appendChild(ov);
        }
        // The menus open upward inside the player, which clips them: they
        // take its height as their limit (app.css, --wx-player-h)
        const box = plyrPlayer.elements.container;
        if (box && typeof ResizeObserver === 'function') {
            new ResizeObserver(() => box.style.setProperty('--wx-player-h', box.clientHeight + 'px'))
                .observe(box);
        }
        try {
            _installSubtitleControls(plyrPlayer);
        } catch (e) {
            // The controls are a convenience — a DOM error here must never
            // take down the player itself.
            console.error('Subtitle controls:', e);
        }
    });
});

// ── Subtitle controls IN the player ───────────────────────────────────────
// 💬 toggles the subtitles, Aa opens a small menu with size and
// confidence coloring. Both sit in the Plyr bar and the menu in the
// Plyr container — that way it also works in fullscreen. State comes
// from fileview.js (localStorage: wtmSubOn/wtmSubSize/wtmSubConfidence).
function _installSubtitleControls(p) {
    const controls = p.elements?.controls;
    const container = p.elements?.container;
    if (!controls || !container || document.getElementById('wtmSubBtn')) return;

    const mk = (id, text, title) => {
        const b = document.createElement('button');
        b.type = 'button';
        b.id = id;
        b.className = 'plyr__controls__item plyr__control wtm-sub-ctrl';
        b.title = title;
        b.textContent = text;
        return b;
    };
    const btnSub = mk('wtmSubBtn', '💬', 'Toggle subtitles');
    const btnMenu = mk('wtmSubMenuBtn', 'Aa', 'Subtitle settings');

    // Insert before the gear icon. WARNING: the settings button is NOT
    // a direct child of the controls bar (Plyr wraps it in a
    // .plyr__menu wrapper) — but insertBefore needs a direct child,
    // otherwise it throws an exception and the buttons never appear. So
    // walk up to the direct child; if
    // there's no match, fall back to the end of the bar.
    let anchor = controls.querySelector('.plyr__menu')
        || controls.querySelector('[data-plyr="settings"]');
    while (anchor && anchor.parentElement !== controls) {
        anchor = anchor.parentElement;
    }
    controls.insertBefore(btnSub, anchor || null);
    controls.insertBefore(btnMenu, anchor || null);

    // ── Menu ──
    const menu = document.createElement('div');
    menu.id = 'wtmSubMenu';
    menu.className = 'wtm-sub-menu';
    menu.innerHTML = `
        <label class="wtm-sub-row">
            <input type="checkbox" id="wtmSubMenuOn"> Show subtitles
        </label>
        <div class="wtm-sub-row" style="flex-direction:column;align-items:stretch;gap:4px">
            <span>Size: <b id="wtmSubSizeVal">100%</b></span>
            <input type="range" id="wtmSubSize" min="60" max="400" step="10">
        </div>
        <label class="wtm-sub-row" title="Color uncertain words: orange < 0.85, red < 0.6 — requires a transcript with word timestamps">
            <input type="checkbox" id="wtmSubConf"> 🎯 Color word confidence
        </label>`;
    container.appendChild(menu);

    const syncMenu = () => {
        const g = (typeof _subSize === 'function') ? _subSize() : 100;
        menu.querySelector('#wtmSubMenuOn').checked = _subOn();
        menu.querySelector('#wtmSubConf').checked = _subConfidence();
        menu.querySelector('#wtmSubSize').value = g;
        menu.querySelector('#wtmSubSizeVal').textContent = g + '%';
    };

    btnSub.addEventListener('click', () => toggleSubtitles());
    btnMenu.addEventListener('click', (e) => {
        e.stopPropagation();
        const isOpen = menu.classList.toggle('open');
        if (isOpen) syncMenu();
    });
    // Clicking outside closes the menu (also in fullscreen — capture phase)
    document.addEventListener('click', (e) => {
        if (!menu.contains(e.target) && e.target !== btnMenu) {
            menu.classList.remove('open');
        }
    }, true);

    menu.querySelector('#wtmSubMenuOn').addEventListener('change', () => toggleSubtitles());
    menu.querySelector('#wtmSubConf').addEventListener('change', (e) => {
        try { localStorage.setItem('wtmSubConfidence', e.target.checked ? '1' : '0'); } catch (err) {}
        // Coloring takes effect on the next overlay update (500ms interval)
    });
    menu.querySelector('#wtmSubSize').addEventListener('input', (e) => {
        try { localStorage.setItem('wtmSubSize', e.target.value); } catch (err) {}
        menu.querySelector('#wtmSubSizeVal').textContent = e.target.value + '%';
        applySubtitleState();
    });

    applySubtitleState();   // establish button state + size on startup
}
