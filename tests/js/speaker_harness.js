/* Loads `speaker.js` in node — the voice match in the file view.
 *
 * Only as much DOM as it touches: the result box. The fake server answers
 * like /api/speaker/recording does: it applies `?threshold=` when there is
 * one, otherwise the configured gallery threshold, clamps it to 0..1 and
 * reports the value it applied in `threshold`.
 */
const fs = require('fs');
const vm = require('vm');

function buildIt({configured = 0.75, answer = {}} = {}) {
    const box = {innerHTML: '', dataset: {}, isConnected: true};
    const requests = [];
    const sandbox = {
        console, JSON, Math, Number, String, Array, Object, parseFloat,
        encodeURIComponent,
        document: {
            getElementById: id => (id === 'speakerRecognition' ? box : null),
            querySelectorAll: () => [],
        },
        window: {},
        esc: s => String(s ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;')
                                 .replace(/>/g, '&gt;').replace(/"/g, '&quot;'),
        // core.js's counted noun: "1 window", "3 windows"
        plural: (n, one, many) => {
            const k = Number(n) || 0;
            return `${k.toLocaleString()} ${k === 1 ? one : (many || one + 's')}`;
        },
        fetch: async url => {
            requests.push(url);
            const q = new URL(url, 'http://app').searchParams;
            const threshold = q.has('threshold')
                ? Math.min(Math.max(parseFloat(q.get('threshold')), 0), 1)
                : configured;
            const body = {abs_path: '/m/a.mkv', windows: 3, model: 'm', labels: [],
                          people: [], named_people: 2, threshold, ...answer};
            return {ok: true, json: async () => body};
        },
    };
    sandbox.globalThis = sandbox;
    vm.createContext(sandbox);
    vm.runInContext(fs.readFileSync(process.env.SPEAKER_JS, 'utf8'), sandbox,
                    {filename: 'speaker.js'});

    const attr = (name) => {
        const m = box.innerHTML.match(new RegExp(`<input type="range"[^>]*\\b${name}="([^"]*)"`));
        return m ? m[1] : null;
    };
    return {
        box, requests,
        run: code => vm.runInContext(code, sandbox),
        // Lets the requests a slider change starts without awaiting them finish
        settle: () => new Promise(resolve => setTimeout(resolve, 0)),
        // The slider as the file view shows it
        slider: () => {
            const label = box.innerHTML.match(/id="spkThresholdValue">([^<]*)</);
            return {value: attr('value'), min: attr('min'), max: attr('max'),
                    label: label ? label[1] : null};
        },
    };
}

module.exports = {buildIt};
