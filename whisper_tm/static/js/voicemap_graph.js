/* Voice map, graph view — every voice a point, edges = similarity.
 *
 * WHY THIS VIEW EXISTS. The former bubble view drew one
 * bubble per group. At hundreds of thousands of voices that's the only workable
 * approach — but at about a thousand it throws away exactly what you want to
 * see: a field of same-colored dots with numbers next to them shows no structure.
 *
 * This view takes over what the standalone prototype map did right:
 * edges, color per group, legend, live threshold slider. The force layout
 * comes from there too, along with the lessons that cost time — they're
 * noted as comments at the respective force.
 *
 * WHAT IT DOES DIFFERENTLY FROM THE PROTOTYPE — and why it has to:
 *
 *   The prototype drew ALL edges above the threshold. That works for its
 *   ~100 voices. Measured on a library of about a thousand voices
 *   (threshold 0.55):
 *
 *       all edges             ~180,000
 *       kNN k=5, only ≥0.55     ~4,100
 *
 *   A group with ~500 members is nearly fully connected and alone accounts
 *   for ~140,000 edges — drawn, a tangle of wool in which nothing at all
 *   can be recognized. The server therefore only delivers the k strongest
 *   neighbors per node (`/api/voicemap/graph`).
 */

/* ══════════════════════════════════════════════════════════════════════════
 *  HIDING — and why that's something different from filtering
 *
 *  The filters further below are a VIEW: non-matches are dimmed,
 *  not removed, because otherwise "who is this?" would give a different
 *  answer depending on the viewing angle. That still applies and is not
 *  touched here.
 *
 *  Hiding serves a different need: leaving groups below a minimum size
 *  out of the view entirely, or showing only some chosen groups with the
 *  remainder hidden completely. That's not a statement about
 *  membership, but about the SECTION you want to look at — and
 *  a section is allowed to be truly empty.
 *
 *  The distinction has consequences you can see: hidden nodes
 *  no longer take part in the force layout. Hundreds of tiny groups filling
 *  the edge otherwise push the big ones together into the center; without
 *  them, the remaining ones have room and pull together into recognizable
 *  clusters.
 *
 *  `GRAPH.active` is the one criterion everything hangs on: forces, edges,
 *  points, outlines, names, hit testing and fitting. If it were forgotten
 *  at any one of these spots, you'd get invisible nodes that still
 *  push things around or catch clicks — the worst state of all.
 * ══════════════════════════════════════════════════════════════════════════ */

const GRAPH = {
    on: false,
    minGroup: 1,          // omit groups below this size entirely
    selection: null,         // Set of group numbers, or null = all
    active: null,           // Uint8Array per node, 1 = participates
    activeN: 0,             // how many that is
    edges: null,          // Int32Array [a0,b0,a1,b1,…]
    kWeight: null,        // Float32Array
    groupOf: null,       // Int32Array per node
    groups: [],           // metrics per group, descending by size
    threshold: null, link_method: 'average', k: 5,
    /* Which map is currently being shown: the inventory or the trash.
     * It is the SAME map with a different section — a second
     * drawing routine would be a second truth. */
    trash: false, trashCount: 0,
    root: '',              // root against which the file names are shortened
    showNames: true, showOutlines: true, groupsOnly: false,
    ticks: 0, running: false, seq: 0,
    doneSeq: 0,         // a loading run is complete up to here

    arranged: false,     // has the force layout already arranged this data?
    // Adjustable, see below.
    drag: 1.0, steps: 420, density: 1.0,
    clusterNames: true,     // arrange identical names as ONE cluster
    playbackSec: 0,         // 0 = play all the way through, otherwise only the first N seconds
};

/* ══════════════════════════════════════════════════════════════════════════
 *  THE LIGHTWEIGHT RUN
 *
 *  On larger maps the view lagged, and the measurement shows where it's
 *  stuck. At several thousand voices, ~24,000 edges and ~1,000 groups:
 *
 *      one simulation step     53.4 ms
 *      drawing the whole frame 11.7 ms
 *        of that: points 2.6 · names 1.6 · edges 1.1 · the remainder outlines
 *
 *  So it's NOT the drawing. At 420 steps, 53 ms per step is
 *  **22 seconds** during which the browser can't make progress on anything else.
 *
 *  Repulsion is the expensive part, and there was an inconsistency in it:
 *  the grid has 90-unit cells, the search went over ±2 cells (so 180 wide),
 *  but the force's limit reached to 270. So it was being cut off anyway —
 *  just in the MOST EXPENSIVE variant, with 25 cells per
 *  node.
 *
 *  The lightweight run searches over ±1 cell (9 instead of 25) and scales the
 *  range accordingly. Plus two economy measures that only apply DURING the
 *  layout: not drawing every frame, and leaving out outlines and names. Both
 *  are decoration on a picture that's moving anyway — at the end it's
 *  displayed completely.
 * ══════════════════════════════════════════════════════════════════════════ */

/* ── Identical names belong in one cluster ────────────────────────────────
 *
 * One name is often split across several groups. Bundling them into one
 * cluster is possible, and it's even the more honest option. A typical
 * picture: **"me" spread across eleven groups** (most voices in one, the
 * rest across ten splinters), "Alice" across five, "Bob" across four,
 * "Carol" across three. At threshold 0.65 these parts are not
 * mathematically connected.
 *
 * But: the group is a CLAIM by the machine ("sounds the same"), the
 * assigned name is a HUMAN'S DETERMINATION ("that's me"). Whoever named
 * eleven groups "me" has said they are one person. Keeping them apart
 * in the picture because a threshold doesn't connect them puts the
 * calculation above the statement.
 *
 * The key below is the unit over which the layout is arranged: the NAME,
 * where there is one, otherwise the group number. It affects attraction to
 * the centroid, repulsion between units, outlines,
 * and labeling — everywhere "the group" previously stood.
 *
 * Can be turned off, because the split is also information: it shows where
 * recognition doesn't yet reach.
 */
function graphKey(nr) {
    if (GRAPH.clusterNames) {
        const name = graphGroupName(nr);
        if (name) return 'n:' + name.toLowerCase();
    }
    return 'g:' + nr;
}

/* Recompute the one criterion. Call after every change to `minGroup`,
 * `selection`, or after loading a new graph. */
function graphBuildActive() {
    if (!MAP.n || !GRAPH.groupOf) { GRAPH.active = null; GRAPH.activeN = 0; return; }
    // Measured is the UNIT the layout is also arranged over. Otherwise
    // at "15 or more" the ten small "me" parts would vanish, even though
    // "me" with hundreds of voices is the largest person of all — the cluster would
    // lose its satellites and nobody would know why.
    const size = new Map();
    for (const g of GRAPH.groups) {
        const s = graphKey(g.nr);
        size.set(s, (size.get(s) || 0) + g.n);
    }
    const a = new Uint8Array(MAP.n);
    let n = 0;
    for (let i = 0; i < MAP.n; i++) {
        // A located recording's voices are shown whatever their group
        // size (`mapLocateApply`) — hiding them would make locating a lie.
        if (GRAPH.pinned && GRAPH.pinned.has(i)) { a[i] = 1; n++; continue; }
        const nr = GRAPH.groupOf[i];
        // Group 0 is the unaffiliated. By definition they have
        // size 1 and thus fall out under any minimum size > 1 —
        // exactly right, they are the bulk of the edge noise.
        const gr = nr ? (size.get(graphKey(nr)) || 0) : 1;
        if (gr < GRAPH.minGroup) continue;
        if (GRAPH.selection && !GRAPH.selection.has(nr)) continue;
        a[i] = 1; n++;
    }
    GRAPH.active = a;
    GRAPH.activeN = n;
}

/* Does this node participate at all? With no criterion set: yes. */
function graphActive(i) {
    return !GRAPH.active || GRAPH.active[i] === 1;
}

/* ── Two adjustment knobs for the layout ──────────────────────────────────
 *
 * Both are adjustable in the UI: which arrangement looks "correct" depends
 * on the material, and that's decided by looking at the picture, not in
 * the source code.
 *
 *   DRAG (PULL) scales both attracting forces TOGETHER: the springs along the
 *   edges and the attraction to the group centroid. Regulating them
 *   separately would be two sliders for one question ("how tightly should
 *   related ones move together?"), and their ratio to each other is
 *   measured, not guessed. Measured on ~1,000 voices: pull 1.0 → spread 899,
 *   pull 3.5 → 767; the clusters visibly draw together and thereby
 *   separate more clearly from each other.
 *
 *   STEPS (STEPS) is the duration. It has a double effect: more steps
 *   mean more movement AND a more slowly tapering cooling schedule (it
 *   computes in fractions of the total duration). A longer simulation is
 *   therefore not "the same arrangement, just later" — it goes further.
 *
 * The speed cap from the PoC remains and is especially important here:
 * it's the reason a high pull doesn't make the layout
 * oscillate out of control.
 *
 * Both values survive a reload — anyone who has fine-tuned an
 * arrangement doesn't want to redo it after F5.
 */
/* Upper limit for the pull. Originally 5.0, since raised. Safe, because the
 * speed cap from the PoC stands behind it: it exists precisely so that
 * strong forces don't make the layout oscillate out of control (at about a thousand nodes,
 * points otherwise jumped more than a hundred pixels per step). Re-measured at
 * 10.0: the layout stays calm and settles within the same number of steps. */
const DRAG_MAX = 25.0;

const GRAPH_MEMO = 'voicemap:arrangement';

function graphLoadSettings() {
    try {
        const m = JSON.parse(localStorage.getItem(GRAPH_MEMO) || '{}');
        if (m.drag > 0) GRAPH.drag = Math.min(DRAG_MAX, Math.max(0.1, m.drag));
        if (m.steps > 0) GRAPH.steps = Math.min(5000, Math.max(50, m.steps));
        // Explicitly checked against `undefined`: a remembered `false` is
        // a decision by the user and must not fall back to `true`.
        if (m.outlines !== undefined) GRAPH.showOutlines = !!m.outlines;
        if (m.density > 0) GRAPH.density = Math.min(2, Math.max(0.05, m.density));
        if (m.minGroup > 0) GRAPH.minGroup = Math.min(500, Math.max(1, m.minGroup));
        if (m.cluster !== undefined) GRAPH.clusterNames = !!m.cluster;
        if (m.playback !== undefined) GRAPH.playbackSec =
            Math.max(0, Math.min(30, Number(m.playback) || 0));
    } catch (e) { /* a corrupt entry is no reason to refuse the map */ }
}

function graphSaveSettings() {
    try {
        localStorage.setItem(GRAPH_MEMO, JSON.stringify(
            {drag: GRAPH.drag, steps: GRAPH.steps,
             outlines: GRAPH.showOutlines, density: GRAPH.density,
             minGroup: GRAPH.minGroup, cluster: GRAPH.clusterNames,
             playback: GRAPH.playbackSec}));
    } catch (e) { /* private mode or similar — then just for this session */ }
}

/* Colors per group. Group 0 = loners, deliberately gray: they have no
 * bond and shouldn't clutter up the map. */
/* 28 colors, in two rows.
 *
 * More colors make groups easier to tell apart. With the earlier 14 colors,
 * two names could easily share one: Alice and Bob both blue, Carol and Dave
 * both red. At 14 colors and ten names,
 * a collision is more likely than not according to the birthday paradox (~97%).
 *
 * The second row is LIGHT tones, not dark ones: the background is dark,
 * a darkened variant of the same color disappears into it. Light against
 * bold is distinguishable at a glance even for the same hue.
 */
const GRAPH_COLORS = [
    // bold
    '#3b82f6', '#22c55e', '#eab308', '#ef4444', '#a78bfa',
    '#06b6d4', '#f97316', '#ec4899', '#84cc16', '#14b8a6', '#f43f5e',
    '#8b5cf6', '#0ea5e9', '#65a30d',
    // light
    '#93c5fd', '#86efac', '#fde047', '#fca5a5', '#d8b4fe',
    '#67e8f9', '#fdba74', '#f9a8d4', '#bef264', '#5eead4', '#fda4af',
    '#c4b5fd', '#7dd3fc', '#a3e635'];

/* Named groups get their color from the NAME, not from the
 * rank number. Otherwise "Alice" would change color when the threshold is
 * moved, because her group slides from rank 3 to rank 7 — and you'd have
 * to search for her again each time. */
function graphColourIndex(s) {
    let h = 0;
    for (let i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) & 0xffff;
    return h % GRAPH_COLORS.length;
}

/* ── The group index ──────────────────────────────────────────────────────
 *
 * On a large map the browser reported "this page is slowing down".
 *
 * There used to be a `find` over `GRAPH.groups` here — a linear search. Harmless
 * on its own; it was only called ONCE PER VOICE, from the
 * tile layout (`graphKey`) and from drawing (`graphColour`).
 * At ~100,000 voices and ~50,000 groups that's an average of billions of
 * comparisons per frame. Measured: layout 4.2 s, base image 5.9 s, loading the map
 * 14.9 s in total — the main thread stalls, and the browser says so.
 *
 * It only showed at a high threshold (0.94): at 0.71 there were about six
 * times fewer groups, so the same bug with a sixfold smaller factor.
 *
 * The index depends on the IDENTITY of the list, not its length: if
 * `GRAPH.groups` is reassigned, it's rebuilt; a `gr.name = …` on
 * an entry takes effect immediately, because the index holds the same object.
 */
let _groupIndex = null, _groupIndexFor = null;

/* To be called when renaming from outside: the index holds the COLORS, and
 * those depend on the inventory of names. Without this, the freshly named group
 * would get a color from the hash — possibly another group's. */
function graphDropGroupIndex() {
    _groupIndexFor = null;
}

function graphGroupIndex() {
    if (_groupIndex && _groupIndexFor === GRAPH.groups) return _groupIndex;
    const groups = GRAPH.groups || [];
    const m = new Map();

    /* ── Colors, without collisions ────────────────────────────────────────
     *
     * The hash alone isn't enough: it can throw two names onto the same
     * color, and that's exactly what was seen. So hash as the DESIRED SLOT,
     * and if it's taken, the next free one.
     *
     * Over the alphabetically sorted names, not over the order of the
     * groups: otherwise "Alice" would change color as soon as another group
     * passes it — and you'd have to search for them on the map again each
     * time. That was the reason the color depends on the name at all.
     */
    const names = [...new Set(groups.map(g => g.name).filter(Boolean))].sort();
    const taken = new Set();
    const byName = new Map();
    for (const n of names) {
        let k = graphColourIndex(n);
        for (let i = 0; i < GRAPH_COLORS.length && taken.has(k); i++) {
            k = (k + 1) % GRAPH_COLORS.length;
        }
        if (taken.size < GRAPH_COLORS.length) taken.add(k);
        byName.set(n, GRAPH_COLORS[k]);
    }

    /* The unnamed groups get whatever's left — by size order.
     * Without this separation, the largest unnamed group would wear
     * "Alice"'s blue, and the map would claim a relationship that
     * doesn't exist. If there aren't enough for all, the remainder stay gray: repeating
     * colors would be worse than not assigning any. */
    const free = GRAPH_COLORS.filter((_, i) => !taken.has(i));
    let n_without = 0;
    groups.forEach((g, rankNr) => {
        const color = g.name
            ? (byName.get(g.name) || GRAPH_COLORS[graphColourIndex(g.name)])
            : (n_without < free.length ? free[n_without++] : '#5b6478');
        m.set(g.nr, {g, rankNr, color});
    });

    _groupIndex = m;
    _groupIndexFor = GRAPH.groups;
    return m;
}

function graphGroupName(nr) {
    const e = graphGroupIndex().get(nr);
    return (e && e.g.name) || '';
}

/* At ~60 groups and 14 colors, every color repeats four times. Two
 * same-colored clusters at the other end of the map would then look
 * related without being so. So only the largest groups get their own
 * color, the remainder stay muted — membership is shown in the tooltip, and
 * position in the graph shows it anyway. */
function graphColour(nr) {
    if (!nr) return '#39404f';
    // One lookup for everything — the color is already sitting in the index.
    // The second linear pass (`findIndex` for the rank) used to sit right here.
    const e = graphGroupIndex().get(nr);
    return e ? e.color : '#5b6478';
}

function graphColourOf(i) {
    return graphColour(GRAPH.groupOf ? GRAPH.groupOf[i] : 0);
}

// ── Loading ──────────────────────────────────────────────────────────────

/* The map takes long on a big library (the graph is computed on the
 * server, half a minute and more): a box in the middle of the stage says
 * so, with the seconds counting, until the newest load is in. */
let _mapLoadingTimer = null;

function mapLoading(on) {
    const box = document.getElementById('mapLoading');
    if (!box) return;
    clearInterval(_mapLoadingTimer);
    box.hidden = !on;
    if (!on) return;
    const t0 = Date.now(), sec = box.querySelector('.map-loading-sec');
    if (sec) sec.textContent = '';
    _mapLoadingTimer = setInterval(() => {
        const s = Math.round((Date.now() - t0) / 1000);
        if (sec) sec.textContent = s >= 3 ? `${s} s — a big library takes a while` : '';
    }, 1000);
}

async function mapLoadGraph(opts = {}) {
    graphCancelRowIfNeeded();
    const mine = ++GRAPH.seq;
    mapLoading(true);
    /* When is a loading run DONE?
     *
     * `GRAPH.on` and `MAP.n` don't answer that: both are still true from
     * the PREVIOUS run while the new one is in flight. That is exactly what
     * broke the jump from the transcript: it set its
     * marker after 400 ms, the running load arrived two seconds
     * later and cleared it with `MAP.highlight = null`. It looked
     * as if the crosshair disappeared on its own.
     *
     * `doneSeq === seq` means: nothing is in flight anymore. */
    const done = () => {
        if (mine === GRAPH.seq) GRAPH.doneSeq = mine;
        if (GRAPH.doneSeq === GRAPH.seq) mapLoading(false);     // nothing newer in flight
    };
    if (opts.threshold !== undefined) GRAPH.threshold = opts.threshold;
    if (opts.link_method) GRAPH.link_method = opts.link_method;
    if (opts.k) GRAPH.k = opts.k;

    graphSetSlider();      // remembered values, before computing
    mapReport('Computing graph …');
    const p = new URLSearchParams({link_method: GRAPH.link_method, k: GRAPH.k});
    if (GRAPH.threshold !== null) p.set('threshold', GRAPH.threshold);
    if (GRAPH.trash) p.set('trash', '1');
    let d;
    try {
        d = await (await fetch('/api/voicemap/graph?' + p)).json();
    } catch (e) {
        // mapReport writes text (textContent) — esc() here showed &amp; etc.
        mapReport(`Graph unavailable: ${String(e)}`);
        done();
        return;
    }
    if (mine !== GRAPH.seq) return;      // a newer request has superseded this one
    if (d && d.error) {
        // An error answer (500 → {error}) is no empty storage: say what failed.
        mapReport(`Graph unavailable: ${d.error}`);
        done();
        return;
    }
    if (!d.voices) {
        mapReport(d.hint || 'No voices in storage.');
        done();
        return;
    }

    GRAPH.on = true;
    // The root belongs to THESE paths — it changes when switching folders.
    GRAPH.root = d.root || '';
    if (typeof d.trash === 'number') GRAPH.trashCount = d.trash;
    if (typeof graphTrashBar === 'function') graphTrashBar();
    GRAPH.threshold = d.threshold;
    /* The slider adopts whatever the server ACTUALLY computed with.
     *
     * Previously, the default lived in two places: as `value`
     * on the slider in the markup, and as `GROUPING_THRESHOLD` in the core.
     * When changing the default (0.65 → 0.71), the slider would otherwise
     * keep showing 0.65 while the map is grouped by 0.71 —
     * a slider claiming something different than the picture below it. */
    graphShowThreshold(d.threshold);
    GRAPH.k = d.k;
    GRAPH.groups = d.groups || [];
    GRAPH.groupOf = Int32Array.from(d.nodes.group_nr);

    // The chosen voice, by id — the indices are about to change.
    const chosenId = MAP.chosen ? MAP.chosen.id : null;
    const n = d.nodes.id.length;
    MAP.n = n;
    if (typeof mapGraphButtonState === 'function') mapGraphButtonState();
    MAP.x = new Float32Array(n);
    MAP.y = new Float32Array(n);
    MAP.size = new Int32Array(n).fill(1);
    MAP.number = Int32Array.from(d.nodes.group_nr);
    MAP.name = d.nodes.name.map(x => x || '');
    // Same shape as in the member view — so tooltip, listening sample,
    // and naming keep working unchanged.
    MAP.members = d.nodes.id.map((id, i) => ({
        id, label: d.nodes.label[i],
        windows: d.nodes.windows[i], speech_sec: d.nodes.speech_sec[i] || 0,
        start_sec: d.nodes.start[i] || 0,
        display: d.file_names[d.nodes.file_idx[i]], abs_path: d.file_names[d.nodes.file_idx[i]],
        person: d.nodes.name[i] || '', group_nr: d.nodes.group_nr[i],
    }));

    const m = (d.edges || []).length;
    GRAPH.edges = new Int32Array(m * 2);
    GRAPH.kWeight = new Float32Array(m);
    for (let e = 0; e < m; e++) {
        GRAPH.edges[e * 2] = d.edges[e][0];
        GRAPH.edges[e * 2 + 1] = d.edges[e][1];
        GRAPH.kWeight[e] = d.edges[e][2];
    }

    // A selection from the previous run is worthless: group numbers shift
    // when the threshold changes — "only group 29" would then show
    // a different group. The minimum size stays, it doesn't depend on any number.
    GRAPH.selection = null;
    MAP.highlight = null;   // highlight belongs to the old graph
    GRAPH.pinned = null;    // indices of the old graph, too
    GRAPH.filter.only = null;
    // Vector id → node, for everything that arrives with ids (locating,
    // the History's "show on map").
    GRAPH.rowOf = new Map();
    for (let i = 0; i < n; i++) GRAPH.rowOf.set(MAP.members[i].id, i);
    graphBuildActive();
    graphHideState();
    graphBuildFilter();     // folder list from what's actually there
    graphApplyFilter();     // a set filter survives the reload
    mapHead(`${plural(d.voices, 'voice')} from `
        + `${plural(d.files, 'recording')} · `
        + `${plural(d.group_count, 'group')} · ${plural(d.loners, 'loner')}`);
    graphLegend();
    /* The initial layout always — it costs nothing and gives every node
     * a location. The FORCE LAYOUT only when the graph is actually visible.
     *
     * Computing the graph while the tiles are shown would be wasted work:
     * it was 420 steps at 53 ms each, so about 22
     * seconds of computation for a picture nobody's looking at — and that's
     * exactly what blocked the interface you were trying to click tiles in.
     */
    GRAPH.arranged = false;
    graphInitialLayout();
    if (typeof TILE === 'undefined' || !TILE.on) {
        /* The size cap holds for EVERY load, not only when the view is
         * picked: Graph on the small trash map, then "Library" reloaded
         * hundreds of thousands of voices and ran the force layout on all of them — the tab
         * froze. Too big → the tiles, as mapView() does on a click. */
        if (typeof mapGraphTooBig === 'function' && mapGraphTooBig()
            && typeof mapView === 'function') mapView('tiles');
        else graphRun();
    }
    // A located recording stays lit through a new threshold.
    if (GRAPH.locate && GRAPH.locate.windows) mapLocateApply(false);
    /* The details keep their voice where it is on the new map, found by
     * id. Before, they kept an INDEX of the old map: after switching to the
     * trash map they showed one voice's file with another's group. */
    if (chosenId !== null && typeof mapPickVoice === 'function') {
        const j = GRAPH.rowOf.get(chosenId);
        if (j !== undefined) mapPickVoice(j);
        else panelClose();
    }
    done();
}

/* Initial layout: groups get their own spot from the outset.
 *
 * Starting from a purely random scatter, the force layout would have to
 * laboriously untangle itself — groups that start inside one another often
 * never separate at all. The spots lie on a Fermat spiral (like
 * sunflower seeds): evenly distributed, without the rings and spokes
 * of a circular grid. */
function graphInitialLayout() {
    const cv = document.getElementById('mapCanvas');
    const W = cv?.clientWidth || 1200, H = cv?.clientHeight || 800;
    const gold = Math.PI * (3 - Math.sqrt(5));
    const numbers = GRAPH.groups.map(g => g.nr);
    const slot = new Map();
    const R = Math.min(W, H) * 0.42;
    numbers.forEach((nr, k) => {
        const r = R * Math.sqrt((k + 0.5) / Math.max(1, numbers.length));
        slot.set(nr, {x: W / 2 + r * Math.cos(k * gold),
                      y: H / 2 + r * Math.sin(k * gold)});
    });
    const size = new Map(GRAPH.groups.map(g => [g.nr, g.n]));
    GRAPH.vx = new Float32Array(MAP.n);
    GRAPH.vy = new Float32Array(MAP.n);
    let one_by_one = 0;
    for (let i = 0; i < MAP.n; i++) {
        const g = GRAPH.groupOf[i];
        if (g > 0 && slot.has(g)) {
            const p = slot.get(g);
            // Spread by group size: a group of 500 needs more
            // room than a pair, otherwise it sits piled up as a clump.
            const s = 8 + Math.sqrt(size.get(g) || 4) * 7;
            /* Scatter into a DISK, not a square.
             *
             * This used to be `(Math.random()-0.5) * s * 2` for x
             * and y — i.e. a square. That didn't matter for small groups: for
             * them, the repulsion (225 px) reaches clear across the whole
             * group and rounds it off within a few steps.
             *
             * For a very large group it does matter. ~10,000
             * voices give s≈722, i.e. a square with a ~1,444 px edge —
             * much wider than the range. The INSIDE of such a large,
             * evenly filled area is force-free: every node has
             * the same number of neighbors all around, everything cancels out. So
             * there's nothing that would ever round off the square — the shape
             * from the first frame survives to the last.
             *
             * That's exactly what it looked like: a rectangle with straight
             * edges and rounded corners (forces only act there), while
             * all the other groups next to it were round. Two overhauls
             * of the force layout didn't change anything about it, because the cause
             * was never there.
             *
             * `Math.sqrt(u)` is not decorative: without the square root, half
             * of all points would pile up in the inner quarter of the area.
             */
            const w = Math.random() * 2 * Math.PI;
            const r = s * Math.sqrt(Math.random());
            MAP.x[i] = p.x + r * Math.cos(w);
            MAP.y[i] = p.y + r * Math.sin(w);
        } else {
            // Loners to the outside: without a bond they'd otherwise lie
            // between the groups and push them apart.
            const a = (one_by_one++) * gold, r = R * 1.15 + Math.random() * R * 0.25;
            MAP.x[i] = W / 2 + r * Math.cos(a);
            MAP.y[i] = H / 2 + r * Math.sin(a);
        }
    }
    GRAPH.ticks = 0;
}

/* ══════════════════════════════════════════════════════════════════════════
 *  THE TREE — why repulsion is computed with a quadtree
 *
 *  Previously, a grid of 90-unit cells lay over the area, and every node
 *  only repelled what was in its ±2 cells. That was built for about a
 *  thousand nodes. At ~30,000 the measurement in the browser looked like
 *  this:
 *
 *      total cells            ~270
 *      densest cell        ~2,600 nodes
 *      average               ~110 nodes
 *      one step         3.2–4.0 s      (drawing: 80 ms)
 *
 *  So the grid was doing nothing anymore: at ~110 nodes per cell, everyone
 *  sees roughly 2,700 neighbors, making 80 million pairs per frame — about
 *  one frame every 2 seconds.
 *
 *  **And the same grid distorted the map.** The repulsion reached
 *  exactly 270 px and was searched over a SQUARE of ±2 cells. Inside
 *  a cluster of thousands of nodes, all forces cancel out; only the
 *  edge feels anything — and that's shaped by the search window.
 *  That's why the biggest cluster became a rectangle with straight edges
 *  instead of a ball. The old comment looked for the cause in the STRENGTH
 *  (`460 * density`); it was in the SHAPE.
 *
 *  Barnes–Hut fixes both with the same trick: a quadtree combines distant
 *  nodes into their center of mass. That makes repulsion
 *  unlimited in range AND direction-independent — there's no window anymore
 *  whose edges anything could align to.
 *
 *  The tree lives on typed arrays that are reused between frames. A tree
 *  of objects per frame would cost more in
 *  cleanup work than the approximation saves — that's the reason for the
 *  somewhat unwieldy construction here.
 * ══════════════════════════════════════════════════════════════════════════ */

const TREE = {
    cap: 0,            // how many tree nodes have room
    kind: null,        // Int32Array(4*cap) — children, −1 = none
    mass: null,       // Int32Array(cap)  — how many points lie beneath
    sx: null, sy: null,// Float64Array(cap) — sum of positions → centroid
    first: null,      // Int32Array(cap)  — first point of a leaf, −1
    x0: null, y0: null,// Float64Array(cap) — top-left corner of the cell
    page: null,       // Float64Array(cap) — edge length of the cell
    next_one: null,   // Int32Array(points) — next point IN THE SAME leaf
    cnt: 0,            // used tree nodes
    batch: null,      // Int32Array — work stack for the traversal
};

/* Below this depth, splitting stops.
 *
 * Two voices can lie exactly on top of each other (same recording, same
 * starting point). Without a limit, the tree would keep splitting such
 * pairs down to the limits of floating point and would fill up. From here
 * on, a leaf just collects multiple points — it then computes them
 * directly against each other, which is exactly right. */
const TREE_MAX_DEPTH = 24;

function treeSlot(nNodes, nPoints) {
    if (TREE.cap >= nNodes && TREE.next_one
        && TREE.next_one.length >= nPoints) return;
    // Grow generously: recopying is more expensive than memory.
    const k = Math.max(nNodes, 64, TREE.cap * 2);
    TREE.cap = k;
    TREE.kind = new Int32Array(4 * k);
    TREE.mass = new Int32Array(k);
    TREE.sx = new Float64Array(k);
    TREE.sy = new Float64Array(k);
    TREE.first = new Int32Array(k);
    TREE.x0 = new Float64Array(k);
    TREE.y0 = new Float64Array(k);
    TREE.page = new Float64Array(k);
    TREE.batch = new Int32Array(4 * TREE_MAX_DEPTH + 64);
    if (!TREE.next_one || TREE.next_one.length < nPoints) {
        TREE.next_one = new Int32Array(Math.max(nPoints, 64));
    }
}

function treeNewNode(x0, y0, page) {
    const k = TREE.cnt++;
    if (k >= TREE.cap) {          // grows mid-build: rescue everything
        const old_value = {
            kind: TREE.kind, mass: TREE.mass, sx: TREE.sx, sy: TREE.sy,
            first: TREE.first, x0: TREE.x0, y0: TREE.y0, page: TREE.page,
            cap: TREE.cap,
        };
        TREE.cap = old_value.cap * 2;
        TREE.kind = new Int32Array(4 * TREE.cap); TREE.kind.set(old_value.kind);
        TREE.mass = new Int32Array(TREE.cap); TREE.mass.set(old_value.mass);
        TREE.sx = new Float64Array(TREE.cap); TREE.sx.set(old_value.sx);
        TREE.sy = new Float64Array(TREE.cap); TREE.sy.set(old_value.sy);
        TREE.first = new Int32Array(TREE.cap); TREE.first.set(old_value.first);
        TREE.x0 = new Float64Array(TREE.cap); TREE.x0.set(old_value.x0);
        TREE.y0 = new Float64Array(TREE.cap); TREE.y0.set(old_value.y0);
        TREE.page = new Float64Array(TREE.cap); TREE.page.set(old_value.page);
    }
    TREE.kind[4 * k] = TREE.kind[4 * k + 1] = -1;
    TREE.kind[4 * k + 2] = TREE.kind[4 * k + 3] = -1;
    TREE.mass[k] = 0; TREE.sx[k] = 0; TREE.sy[k] = 0;
    TREE.first[k] = -1;
    TREE.x0[k] = x0; TREE.y0[k] = y0; TREE.page[k] = page;
    return k;
}

/* Which quadrant of `k` does (x,y) fall into? 0=tl 1=tr 2=bl 3=br */
function treeQuarter(k, x, y) {
    const h = TREE.page[k] / 2;
    return (x >= TREE.x0[k] + h ? 1 : 0) + (y >= TREE.y0[k] + h ? 2 : 0);
}

function treeSplit(k) {
    const h = TREE.page[k] / 2, x0 = TREE.x0[k], y0 = TREE.y0[k];
    /* Create all four FIRST, THEN assign — not in one go.
     *
     * `TREE.kind[4*k] = treeNewNode(…)` looks more harmless, but is
     * wrong: JavaScript evaluates the left side FIRST. If the call on
     * the right grows the tree, `TREE.kind` would then still point to the old,
     * discarded array — the assignment would land in nothing, and the subtree
     * would be unreachable forever. */
    const tl = treeNewNode(x0, y0, h);
    const tr = treeNewNode(x0 + h, y0, h);
    const bl = treeNewNode(x0, y0 + h, h);
    const br = treeNewNode(x0 + h, y0 + h, h);
    const kind = TREE.kind;
    kind[4 * k] = tl; kind[4 * k + 1] = tr;
    kind[4 * k + 2] = bl; kind[4 * k + 3] = br;
}

function treeInsert(p, X, Y) {
    let k = 0, depth = 0;
    for (;;) {
        if (TREE.kind[4 * k] === -1) {                       // leaf
            if (TREE.first[k] === -1) {                     // still empty
                TREE.first[k] = p; TREE.next_one[p] = -1; return;
            }
            if (depth >= TREE_MAX_DEPTH) {                    // deep enough
                TREE.next_one[p] = TREE.first[k]; TREE.first[k] = p; return;
            }
            // Split and push the existing points one level deeper.
            const old_value = TREE.first[k];
            TREE.first[k] = -1;
            treeSplit(k);
            let q = old_value;
            while (q !== -1) {
                const nx = TREE.next_one[q];
                const kk = TREE.kind[4 * k + treeQuarter(k, X[q], Y[q])];
                TREE.next_one[q] = TREE.first[kk];
                TREE.first[kk] = q;
                q = nx;
            }
            // `p` also falls deeper now — the loop continues.
        }
        k = TREE.kind[4 * k + treeQuarter(k, X[p], Y[p])];
        depth++;
    }
}

/* Build the tree over all PARTICIPATING nodes. Return value: did it work? */
function treeBuild(X, Y, n) {
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    let number = 0;
    for (let i = 0; i < n; i++) {
        if (!graphActive(i)) continue;
        number++;
        if (X[i] < minX) minX = X[i];
        if (Y[i] < minY) minY = Y[i];
        if (X[i] > maxX) maxX = X[i];
        if (Y[i] > maxY) maxY = Y[i];
    }
    if (number < 2) return false;
    // Square root cell: only this way do the quadrants stay square, and
    // only then is `side/distance` an honest measure of "far enough away".
    const page = Math.max(maxX - minX, maxY - minY) * 1.01 + 1;
    // Capacity: in practice ~2·n nodes; with margin, so growing
    // mid-build stays the exception.
    treeSlot(Math.max(64, 3 * number), n);
    TREE.cnt = 0;
    treeNewNode(minX, minY, page);
    for (let i = 0; i < n; i++) {
        if (graphActive(i)) treeInsert(i, X, Y);
    }
    // Masses and centroids from bottom to top. A child always gets
    // a HIGHER number than its parent — so running backwards is enough,
    // a second tree traversal isn't necessary.
    for (let k = TREE.cnt - 1; k >= 0; k--) {
        if (TREE.kind[4 * k] === -1) {
            let q = TREE.first[k], m = 0, sx = 0, sy = 0;
            while (q !== -1) { m++; sx += X[q]; sy += Y[q]; q = TREE.next_one[q]; }
            TREE.mass[k] = m; TREE.sx[k] = sx; TREE.sy[k] = sy;
        } else {
            let m = 0, sx = 0, sy = 0;
            for (let c = 0; c < 4; c++) {
                const kk = TREE.kind[4 * k + c];
                m += TREE.mass[kk]; sx += TREE.sx[kk]; sy += TREE.sy[kk];
            }
            TREE.mass[k] = m; TREE.sx[k] = sx; TREE.sy[k] = sy;
        }
    }
    return true;
}

/* How coarse is the approximation allowed to be?
 *
 * A tree node counts as "far enough away" if its edge length divided
 * by the distance is below this value. The usual choice is 0.9 (d3
 * uses that). Here it's 0.5, and that's measured — on a ring of 720
 * points whose forces should mathematically cancel to zero, and on the
 * mean directional error against the exact computation (900 nodes):
 *
 *     θ      residual in ring   directional error   one step (20,000)
 *     0.4        10.9               1.20°               167 ms
 *     0.5        10.9               1.30°               121 ms
 *     0.6        83.9               2.72°                95 ms
 *     0.7        83.9               3.19°                78 ms
 *     0.9        83.9               4.07°                71 ms
 *     1.1        83.9              10.11°                53 ms
 *
 * There's a cliff between 0.5 and 0.6: the tree climbs one level
 * less deep there, and the residual error jumps eightfold.
 *
 * Even so, this is set to 0.8 and not 0.5 — because the table above doesn't
 * answer the question that actually matters. Measured on the finished
 * arrangement:
 *
 *     θ=0.9   aspect ratio of the largest group 1.01 · step 165 ms
 *     θ=0.5   aspect ratio of the largest group 1.03 · step 288 ms
 *
 * So roundness does NOT come from θ. It comes from the boundary now
 * being circular instead of square — and that's circular for every θ.
 * What θ changes is only the noise in individual forces,
 * and that averages out over 420 steps.
 *
 * 0.8 instead of the usual 0.9: a third less directional error, without it
 * being measurable in time cost. Anyone who needs more accuracy — say for
 * an arrangement that should settle after just a few steps — sets it
 * lower and pays for it roughly quadratically. */
const TREE_THETA = 0.8;

/* How far an edge BETWEEN two groups pulls at most.
 *
 * 300 px gives a maximum force of 300 · 0.018 · 0.2 = 1.08 per weight —
 * about a sixth of what holds the group itself together at the edge (≈ 6.6).
 * That way the hint stays effective while the groups are close
 * together, and loses influence once they're far apart —
 * which matches the point: two widely separated clusters are simply NOT
 * the same, no matter how many individual similarities exist between them. */
const FOREIGN_EDGE_MAX = 300;

// ── Force layout ────────────────────────────────────────────────────────

function graphStep() {
    const n = MAP.n, X = MAP.x, Y = MAP.y, VX = GRAPH.vx, VY = GRAPH.vy;
    const cv = document.getElementById('mapCanvas');
    const W = cv?.clientWidth || 1200, H = cv?.clientHeight || 800;

    /* 1. Repulsion via a quadtree (Barnes–Hut, see the top of the
     *    section further up) — WITH the old range.
     *
     * The range is intentional and was never the bug. An attempt
     * without it (pure Barnes–Hut, unlimited range) looked round
     * but was unusable: at ~30,000 voices, the system then behaves
     * like an evenly charged disk. All ~3,000 groups
     * merged into ONE circle in which nothing could be distinguished anymore —
     * faster and prettier, but meaningless.
     *
     * The bug was the SHAPE of the boundary. The search was over a square of
     * ±2 cells, but the cutoff was circular at 225 px — the two didn't
     * match. Whoever was diagonally 225 px away fell into or out of
     * the window depending on their position within their cell. That
     * direction-dependent gap is what pulled the largest cluster into a
     * rectangle with straight edges.
     *
     * The tree doesn't know a window. It discards a subtree exactly
     * when its NEAREST corner is farther away than the range — that's
     * circular and the same in all directions. And it groups together what's
     * far enough away instead of going through it one by one: instead of about
     * 2,700 neighbors, a node sees a few hundred cells.
     *
     * `density` still controls the strength. It's the reason clusters
     * do NOT draw further together: the pull via springs and
     * centroid pulls, the repulsion holds against it. Anyone who wants
     * "really pull together" needs to lower this, not raise the pull.
     */
    const strength = 460 * GRAPH.density;
    const rich = 225;
    const limit = rich * rich;
    if (treeBuild(X, Y, n)) {
        const theta2 = TREE_THETA * TREE_THETA;
        const batch = TREE.batch;
        const kind = TREE.kind, mass = TREE.mass;
        const bsx = TREE.sx, bsy = TREE.sy, bx0 = TREE.x0, by0 = TREE.y0;
        const bPage = TREE.page, first = TREE.first, next_one = TREE.next_one;
        for (let i = 0; i < n; i++) {
            if (!graphActive(i)) continue;
            const xi = X[i], yi = Y[i];
            let fx = 0, fy = 0, sp = 0;
            batch[sp++] = 0;
            while (sp > 0) {
                const k = batch[--sp];
                const m = mass[k];
                if (m === 0) continue;
                /* Is the whole cell outside the range?
                 * Measured to the nearest corner — if that's too far,
                 * every point inside it is too far. That's the cut that
                 * saves the work, and it's circular. */
                const s = bPage[k];
                const ax = bx0[k], ay = by0[k];
                const nx = xi < ax ? ax - xi : (xi > ax + s ? xi - ax - s : 0);
                const ny = yi < ay ? ay - yi : (yi > ay + s ? yi - ay - s : 0);
                if (nx * nx + ny * ny > limit) continue;
                /* DISCARDED: additionally requiring that the
                 * cell lies ENTIRELY within the range (distance to the
                 * FARTHEST corner), and otherwise splitting it further.
                 *
                 * The idea was correct: a cell that lies half inside
                 * and half outside is counted with its full mass
                 * and the falloff for that is read off at the centroid —
                 * the round boundary thereby gets quantized into an
                 * axis-aligned grid, just finer than the old grid.
                 *
                 * But the cost of that isn't small, it's ruinous:
                 * 1,310 ms instead of 165 ms per step, measured. For a
                 * node DEEP inside a cluster, almost every cell
                 * fails this condition — its far corner almost
                 * always sticks out past 225 px. So it splits all the way
                 * to leaf level, and the tree saves exactly nothing anymore.
                 *
                 * And the benefit wasn't small either, it was zero. The
                 * angular group didn't come from this, but from the
                 * initial layout, which scattered into a SQUARE (see
                 * `graphInitialLayout`). After that was fixed,
                 * the shape was even slightly better without the condition —
                 * radius max/min over 24 sectors of the largest group:
                 *
                 *     with    1.13   ·  1,310 ms per step
                 *     without 1.10   ·    165 ms per step
                 */
                if (kind[4 * k] === -1) {
                    // Leaf: the few points in it, one by one — this is exactly
                    // where the near-field effect that keeps clusters apart happens.
                    let q = first[k];
                    while (q !== -1) {
                        if (q !== i) {
                            let ex = xi - X[q], ey = yi - Y[q];
                            let d2 = ex * ex + ey * ey;
                            if (d2 < 1) {
                                ex = Math.random() - 0.5;
                                ey = Math.random() - 0.5;
                                d2 = 1;
                            }
                            if (d2 <= limit) {
                                // Taper smoothly to zero. With a hard edge,
                                // the force would jump to zero at the boundary
                                // and points would lock in place there.
                                const waste = 1 - d2 / limit;
                                const f = strength / d2 * waste * waste;
                                fx += ex * f; fy += ey * f;
                            }
                        }
                        q = next_one[q];
                    }
                    continue;
                }
                const ex = xi - bsx[k] / m, ey = yi - bsy[k] / m;
                const d2 = ex * ex + ey * ey;
                if (s * s < theta2 * d2 && d2 >= 1) {
                    /* Far enough away: the whole subtree as ONE point.
                     * The falloff is evaluated at the centroid — for a
                     * cell that's small relative to its distance, that's
                     * exactly the approximation Barnes–Hut makes anyway. */
                    if (d2 <= limit) {
                        const waste = 1 - d2 / limit;
                        const f = strength * m / d2 * waste * waste;
                        fx += ex * f; fy += ey * f;
                    }
                } else {
                    // Too close for the approximation — the four quadrants individually.
                    // The stack is enough: a path downward puts at most
                    // four entries per level on it, and it never goes deeper
                    // than TREE_MAX_DEPTH.
                    for (let c = 0; c < 4; c++) {
                        const kk = kind[4 * k + c];
                        if (mass[kk] !== 0) batch[sp++] = kk;
                    }
                }
            }
            VX[i] += fx; VY[i] += fy;
        }
    }

    // 2. Springs along the edges.
    const K = GRAPH.edges, KW = GRAPH.kWeight;
    for (let e = 0; e < KW.length; e++) {
        const a = K[e * 2], b = K[e * 2 + 1], w = KW[e];
        // A spring with a hidden endpoint would pull a visible
        // point toward an invisible one — the movement would have no
        // reason left in the picture.
        if (!graphActive(a) || !graphActive(b)) continue;
        let dx = X[b] - X[a], dy = Y[b] - Y[a];
        const d = Math.hypot(dx, dy) || 1;
        const target = 70 + (1 - w) * 220;   // more similar = closer
        // Edges WITHIN a group hold the cluster together and pull
        // fully. Edges BETWEEN groups are just a hint ("these sound
        // similar") — if they pulled with the same strength, they'd push
        // the clusters into each other, and you'd see four colors in
        // one clump instead of four groups side by side.
        const ga = GRAPH.groupOf[a], gb = GRAPH.groupOf[b];
        const inside = (ga !== 0 && ga === gb);
        const pullFactor = inside ? 1 : 0.2;
        /* An edge BETWEEN groups may only pull LIMITED amount.
         *
         * Spring force grows linearly with distance, and groups end up
         * far apart. For example, a cross-group edge over 2,000 px gives
         *
         *     (2000 − 70) · 0.018 · 0.2 · w  ≈  6.95 · w
         *
         * The cohesion of the group itself at its edge, by comparison, is only
         *
         *     0.012 · 550  ≈  6.6
         *
         * So a SINGLE foreign edge pulls as strongly as the entire
         * group bond — and two pull the member right out. Measured on
         * the group with the most outside edges (about 30%): aspect ratio
         * 3.42, while groups without outside edges were at 1.06 to 1.29.
         * In the picture that's a cluster being stretched into a cigar
         * shape, with the bundle of edges as a tail.
         *
         * The limit says exactly what the comment above already says:
         * an edge between groups is a HINT, not a bond. A
         * hint must not tear a member out of its group, no matter
         * how far away the other group is. Compression (negative) remains
         * unlimited — otherwise clusters could overlap. */
        const stretch = inside ? (d - target)
                             : Math.min(d - target, FOREIGN_EDGE_MAX);
        const f = stretch * 0.018 * GRAPH.drag * w * pullFactor;
        dx /= d; dy /= d;
        VX[a] += dx * f; VY[a] += dy * f;
        VX[b] -= dx * f; VY[b] -= dy * f;
    }

    // 3. Group attraction toward its own centroid. Without it the graph
    //    stays a tangle where you can only recognize groups by color — and
    //    showing exactly that sense of belonging together is what the map
    //    is for.
    // The unit is the key, not the group number: all voices
    // of the same name pull toward ONE centroid (see the top of the file).
    /* Every unit gets a running NUMBER instead of a key.
     *
     * Previously this spot called `graphKey(g)` for every node and
     * then looked the key up twice in a Map — at ~30,000
     * nodes that's 60,000 freshly built strings per frame, just to tell
     * a few thousand units apart. The key depends solely on the
     * group number; once per group is enough. */
    const keyPerGroup = new Map();
    const unitOf = (GRAPH._unitOf && GRAPH._unitOf.length === n)
        ? GRAPH._unitOf : (GRAPH._unitOf = new Int32Array(n));
    const numberPerKey = new Map();
    let eCnt = 0;
    for (let i = 0; i < n; i++) {
        unitOf[i] = -1;
        const g = GRAPH.groupOf[i];
        if (!g || !graphActive(i)) continue;
        let s = keyPerGroup.get(g);
        if (s === undefined) { s = graphKey(g); keyPerGroup.set(g, s); }
        let e = numberPerKey.get(s);
        if (e === undefined) { e = eCnt++; numberPerKey.set(s, e); }
        unitOf[i] = e;
    }
    const eSx = new Float64Array(eCnt), eSy = new Float64Array(eCnt);
    const eNum = new Int32Array(eCnt);
    for (let i = 0; i < n; i++) {
        const e = unitOf[i];
        if (e < 0) continue;
        eSx[e] += X[i]; eSy[e] += Y[i]; eNum[e]++;
    }
    for (let i = 0; i < n; i++) {
        const e = unitOf[i];
        if (e < 0) continue;
        // Weaker than in the PoC (0.02): for a group of 500, a
        // strong center attraction presses all members into an evenly
        // filled disk — the edges then have nothing left to say.
        VX[i] += (eSx[e] / eNum[e] - X[i]) * 0.012 * GRAPH.drag;
        VY[i] += (eSy[e] / eNum[e] - Y[i]) * 0.012 * GRAPH.drag;
    }

    /* 4. Whole groups repel each other. Without this, two clusters
     *    push into each other as soon as individual members are connected —
     *    you'd see two colors in one clump and take it for one group.
     * Also via the unit: two "me" parts must not
     * push each other away, otherwise one force would pull together what the
     * other separates.
     *
     * PRE-SORTED. Before that, this loop compared every unit
     * with every other one — at ~3,000 units that's ~4 million pairs per frame
     * and a measured 121 ms, so the most expensive part of the whole step
     * once repulsion had switched to Barnes–Hut.
     *
     * Unlike with the nodes, an approximation is NOT needed here: the force
     * has a hard range (`expected`), beyond which it's zero. So it's
     * enough to never even look at pairs that can't possibly
     * touch each other. For that, each unit gets a radius of
     *
     *     radius = 13 + √n · 11        (since expected = radius_A + radius_B)
     *
     * and the units are sorted by their left edge. Walking then
     * from left to right, you can stop as soon as the left edge
     * of the next unit lies to the right of the right edge of the current one —
     * everything further out is guaranteed to be even further right. */
    const zx = new Float64Array(eCnt), zy = new Float64Array(eCnt);
    const zr = new Float64Array(eCnt);
    for (let e = 0; e < eCnt; e++) {
        zx[e] = eSx[e] / eNum[e];
        zy[e] = eSy[e] / eNum[e];
        zr[e] = 13 + Math.sqrt(eNum[e]) * 11;
    }
    const zsx = new Float64Array(eCnt), zsy = new Float64Array(eCnt);
    const sequence = new Int32Array(eCnt);
    for (let e = 0; e < eCnt; e++) sequence[e] = e;
    sequence.sort((a, b) => (zx[a] - zr[a]) - (zx[b] - zr[b]));
    for (let ai = 0; ai < eCnt; ai++) {
        const a = sequence[ai];
        const ax = zx[a], ay = zy[a], ar = zr[a], right = ax + ar;
        for (let bi = ai + 1; bi < eCnt; bi++) {
            const b = sequence[bi];
            if (zx[b] - zr[b] > right) break;      // impossible from here on
            const expected = ar + zr[b];
            const dy = ay - zy[b];
            if (dy > expected || dy < -expected) continue;  // cheaper than hypot
            const dx = ax - zx[b];
            const d = Math.hypot(dx, dy) || 1;
            if (d >= expected) continue;
            const f = (expected - d) / d * 0.28;
            zsx[a] += dx * f; zsy[a] += dy * f;
            zsx[b] -= dx * f; zsy[b] -= dy * f;
        }
    }
    for (let i = 0; i < n; i++) {
        const e = unitOf[i];
        if (e >= 0) { VX[i] += zsx[e]; VY[i] += zsy[e]; }
    }

    // 5. Cool down AND cap velocity. Just damping wasn't enough in the PoC:
    //    at about a thousand nodes, repulsion and group attraction fed off each
    //    other so much that points jumped more than a hundred pixels per
    //    step — the graph ran into the step limit instead of coming
    //    to remainder. The cap takes that energy out of the system.
    const p = Math.min(1, GRAPH.ticks / GRAPH.steps);
    const cool = 0.85 - 0.30 * p;
    const vmax = 26 * (1 - 0.92 * p) + 0.5;
    let motion = 0;
    let moved = 0;
    for (let i = 0; i < n; i++) {
        // Hidden ones stay where they are. Moving them would cost time
        // for a result nobody sees — and their movement would feed into
        // the remainder measure that decides when the layout ends.
        if (!graphActive(i)) continue;
        moved++;
        VX[i] += (W / 2 - X[i]) * 0.0006;
        VY[i] += (H / 2 - Y[i]) * 0.0006;
        VX[i] *= cool; VY[i] *= cool;
        const v = Math.hypot(VX[i], VY[i]);
        if (v > vmax) { VX[i] *= vmax / v; VY[i] *= vmax / v; }
        X[i] += VX[i]; Y[i] += VY[i];
        motion += Math.abs(VX[i]) + Math.abs(VY[i]);
    }
    GRAPH.ticks++;
    return motion / Math.max(1, moved);
}

function graphRun() {
    /* Already running: a reload's own call bounces off here, and the old
     * run stops at its next step (seq) — the new graph stayed unarranged
     * and undrawn, "0 of 21 visible" after moving the threshold during a
     * layout. The stopping run starts it (below). */
    if (GRAPH.running) { GRAPH.runPending = true; return; }
    GRAPH.runPending = false;
    // Safety net: there's nothing to arrange in the tile view.
    if (typeof TILE !== 'undefined' && TILE.on) return;
    /* Without data there's nothing to arrange — and an attempt would be
     * expensive.
     *
     * `mapView('graph')` starts the layout IMMEDIATELY, but the graph
     * comes over the network. Measured: 11.3 s for 4.9 MB. Anyone
     * who clicks "Graph" in that window — and at 11 s of empty screen
     * everyone does — would run into `GRAPH.kWeight === null` inside
     * `graphStep`.
     *
     * Worse than the bug was its consequence: `GRAPH.running` was already
     * `true` and never got reset. Every later call bounced
     * off at the top, even the one after loading finished. The view stayed
     * black until the page was reloaded.
     */
    if (!GRAPH.kWeight) return;
    GRAPH.running = true;
    /* A new load cancels the running step.
     *
     * This used to be a layer check (`… !== 'graph'`) — the cancel on switching
     * to the bubble view. That view no longer exists, but the danger does:
     * if a reload happens during the layout, the old tick keeps computing on
     * arrays that meanwhile belong to someone else. `seq` counts exactly that. */
    const mine = GRAPH.seq;
    const tick = () => {
        if (GRAPH.seq !== mine) {
            GRAPH.running = false;
            // The run a reload asked for meanwhile; one still loading
            // starts itself once its data is in
            if (GRAPH.runPending && GRAPH.doneSeq === GRAPH.seq) graphRun();
            return;
        }
        /* If someone switches to the tiles mid-run, the simulation stops —
         * but `ticks` stays where it was. Anyone switching back continues
         * computing from where it left off, instead of starting over. */
        if (typeof TILE !== 'undefined' && TILE.on) {
            GRAPH.running = false;
            return;
        }
        /* Whatever goes wrong here must not lock the view forever.
         *
         * The known trigger is caught above (`GRAPH.kWeight`), but the
         * shape of the bug remains: `GRAPH.running` stays `true`,
         * `requestAnimationFrame` is only set again at the END — so
         * ANY exception would permanently lock the layout, and the user
         * would just see a black screen with no explanation.
         *
         * Hence: clean up, report, stop. A visible failure is
         * better than a silent standstill.
         */
        let idle;
        try {
            idle = graphStep();
        } catch (e) {
            GRAPH.running = false;
            console.error('Layout aborted:', e);
            mapHead('The layout failed — please reload.');
            return;
        }
        // Fit after computing, otherwise the map wanders out of frame.
        mapFit();
        /* In the lightweight run, draw only every third frame.
         *
         * Drawing costs 11.7 ms, the computation step 53.4 — so saving two
         * out of three frames brings roughly 8 ms per step. None of this is
         * visible: the layout moves in small steps, and the eye sees the
         * same motion at 15 frames per second as at 45. The LAST frame is
         * always drawn (below).
         */
        const done = GRAPH.ticks >= GRAPH.steps || idle < 0.05;
        if (!done && GRAPH.ticks % 3 === 0) mapDraw();
        if (done) {
            // Clear the flag FIRST, THEN draw — otherwise the final frame
            // would be missing exactly the outlines and names that were
            // skipped during the run.
            GRAPH.running = false;
            GRAPH.arranged = true;
            mapDraw();
            // Hit testing needs the grid over the FINAL positions.
            // During the layout it would be stale on every frame — and
            // rebuilding a grid every frame would be the most expensive
            // part of the loop.
            mapBuildGrid();
            const f = document.getElementById('mapFoot');
            if (f) f.textContent += ` · arranged in ${plural(GRAPH.ticks, 'step')}`;
            return;
        }
        requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
}

// ── Drawing edges ────────────────────────────────────────────────────────

/* Called from mapDraw(), BEFORE the nodes — lines belong under
 * the points, not above them. */
function graphDrawEdges(g, s, ox, oy, w, h) {
    if (!GRAPH.kWeight || !GRAPH.kWeight.length) return;
    const K = GRAPH.edges, KW = GRAPH.kWeight, X = MAP.x, Y = MAP.y;
    /* The more edges, the fainter. At 4,000 lines on one screen, so many
     * overlap that a fixed opacity would cover up the points —
     * and the points are the actual message. The edges are meant to
     * SUGGEST structure, not dominate the picture. */
    const dense = Math.min(1, KW.length / 4000);
    g.lineWidth = 0.8;
    // One color for all edges and ONE path: stroking 4,000 lines
    // individually costs 4,000 state changes per frame.
    g.strokeStyle = `rgba(150,165,190,${(0.30 - 0.12 * dense).toFixed(3)})`;
    g.beginPath();
    const sight = GRAPH.filtered ? GRAPH.visible : null;
    for (let e = 0; e < KW.length; e++) {
        const a = K[e * 2], b = K[e * 2 + 1];
        if (!graphActive(a) || !graphActive(b)) continue;
        // An edge counts as a match if BOTH ends are matches.
        // Half-edges into the dimmed area would be lines into nowhere.
        if (sight && !(sight[a] && sight[b])) continue;
        const ax = ox + s * X[a], ay = oy + s * Y[a];
        const bx = ox + s * X[b], by = oy + s * Y[b];
        // Both ends outside → don't even draw it.
        if ((ax < 0 && bx < 0) || (ax > w && bx > w)
            || (ay < 0 && by < 0) || (ay > h && by > h)) continue;
        g.moveTo(ax, ay); g.lineTo(bx, by);
    }
    g.stroke();
}

// ── Outlines around the groups ───────────────────────────────────────────
//
//  Many unnamed groups would otherwise be just gray and blur
//  into each other. The cause is in `graphColour()`: only
//  the fourteen largest groups get their own color, all the remainder
//  share `#5b6478`. Two adjacent clusters then look like one.
//
//  More colors are NOT the solution — at 47 groups and 14 colors,
//  every color would appear three times, and two same-colored clusters at
//  opposite ends of the map would look related without being so.
//  A closed line doesn't claim any relationship; it just says
//  "this here is ONE group". That's exactly what was missing.
//
//  What's drawn is the convex hull, padded outward (the points
//  should lie INSIDE, not on the edge) and rounded over the edge
//  midpoints — an angular hull reads as a geometric figure, a
//  rounded one as a surrounding area.
//
//  Group 0 is left out: those are the loners without a bond, scattered
//  across the whole map. Their hull would be the whole picture.

/* Convex hull via Andrew's monotone chain over SCREEN coordinates.
 * Returns: indices into px/py, in traversal order. */
function graphHull(px, py) {
    const idx = px.map((_, i) => i)
        .sort((a, b) => (px[a] - px[b]) || (py[a] - py[b]));
    const cross = (o, a, b) =>
        (px[a] - px[o]) * (py[b] - py[o]) - (py[a] - py[o]) * (px[b] - px[o]);
    const build = (row) => {
        const s = [];
        for (const i of row) {
            while (s.length >= 2
                   && cross(s[s.length - 2], s[s.length - 1], i) <= 0) s.pop();
            s.push(i);
        }
        s.pop();                    // the endpoint belongs to the other half
        return s;
    };
    return build(idx).concat(build(idx.slice().reverse()));
}

/* Called from mapDraw(), right at the BOTTOM — even before the edges.
 * An outline over the lines would cut the graph into regions. */
function graphDrawHulls(g, s, ox, oy, w, h) {
    if (!GRAPH.showOutlines || !GRAPH.groupOf || !GRAPH.groups.length) return;
    // While the layout is running, computing a hull per group is the most
    // expensive part of the frame — and with about a thousand groups all moving, also
    // the most useless. At the end it's just there.
    if (GRAPH.running) return;
    // Computing a hull with very many nodes per frame is wasteful;
    // same threshold as the point-saving mode in voicemap.js.
    if (MAP.n > 20000) return;

    const sight = GRAPH.filtered ? GRAPH.visible : null;
    // Collect points per group — in screen coordinates, because the
    // padding applies in pixels and shouldn't zoom along.
    const sort_by = new Map();
    for (let i = 0; i < MAP.n; i++) {
        const nr = GRAPH.groupOf[i];
        if (!nr || !graphActive(i)) continue;     // loners: no hull
        // One outline per UNIT: if the eleven "me" parts are drawn
        // together into one cluster, one line per part would be scribbles inside it.
        //
        // Do NOT name it `s` — that's the zoom factor of this call. Exactly
        // that shadowing once made `ox + s * x` compute with a string, and
        // no outline appeared at all anymore.
        const key_ = graphKey(nr);
        let e = sort_by.get(key_);
        if (!e) { e = {px: [], py: [], hits: 0, nr}; sort_by.set(key_, e); }
        e.px.push(ox + s * MAP.x[i]);
        e.py.push(oy + s * MAP.y[i]);
        if (!sight || sight[i]) e.hits++;
    }

    g.lineJoin = 'round';
    for (const [, e] of sort_by) {
        const nr = e.nr;
        const n = e.px.length;
        let minX = 1e9, maxX = -1e9, minY = 1e9, maxY = -1e9, cx = 0, cy = 0;
        for (let j = 0; j < n; j++) {
            const x = e.px[j], y = e.py[j];
            cx += x; cy += y;
            if (x < minX) minX = x;  if (x > maxX) maxX = x;
            if (y < minY) minY = y;  if (y > maxY) maxY = y;
        }
        cx /= n; cy /= n;
        // Entirely outside the viewport → don't even compute it.
        if (maxX < -60 || minX > w + 60 || maxY < -60 || minY > h + 60) continue;

        // Padding: a bit more than the point radius (4 px), so the line
        // doesn't touch the outermost points. Grows slightly with zoom,
        // otherwise it sticks to the points when zooming in.
        const pad = 9 + 3 * Math.min(2, s);
        const color = graphColour(nr);
        // Filtered groups without a single match are dimmed —
        // just like their points. Nothing is removed here either.
        const muted = sight && !e.hits;
        g.globalAlpha = muted ? 0.18 : 1;

        g.beginPath();
        if (n < 3) {
            // Two points don't span an area; an oval around their
            // extent says the same thing without special cases in the path.
            g.ellipse((minX + maxX) / 2, (minY + maxY) / 2,
                      (maxX - minX) / 2 + pad, (maxY - minY) / 2 + pad,
                      0, 0, 6.283);
        } else {
            const H = graphHull(e.px, e.py);
            // Push every corner point outward, away from the centroid.
            const hx = [], hy = [];
            for (const j of H) {
                const dx = e.px[j] - cx, dy = e.py[j] - cy;
                const d = Math.hypot(dx, dy) || 1;
                hx.push(e.px[j] + dx / d * pad);
                hy.push(e.py[j] + dy / d * pad);
            }
            const m = hx.length;
            if (m < 3) { g.globalAlpha = 1; continue; }
            // Rounding: run through the edge midpoints, using the corners as
            // control points. Produces a closed, soft curve.
            g.moveTo((hx[m - 1] + hx[0]) / 2, (hy[m - 1] + hy[0]) / 2);
            for (let j = 0; j < m; j++) {
                const k = (j + 1) % m;
                g.quadraticCurveTo(hx[j], hy[j],
                                   (hx[j] + hx[k]) / 2, (hy[j] + hy[k]) / 2);
            }
            g.closePath();
        }
        // Very faint fill, so overlapping groups show as
        // an overlap instead of covering each other up.
        g.fillStyle = color + '14';          // ~8% opacity
        g.fill();
        g.strokeStyle = color + 'aa';
        g.lineWidth = 1.6;
        g.stroke();
        g.globalAlpha = 1;
    }
}

function graphToggleNameClusters(on) {
    GRAPH.clusterNames = !!on;
    graphSaveSettings();
    graphBuildActive();
    graphHideState();
    graphRearrange();
}

function graphToggleOutlines(on) {
    GRAPH.showOutlines = !!on;
    graphSaveSettings();
    mapDraw();
}

// ══════════════════════════════════════════════════════════════════════════
//  FILTER — a VIEW onto the graph, not a second truth
//
//  The principle everything here hangs on: **grouping is a property
//  of the whole inventory, not of the current section.** If a filter were to
//  regroup, "who is this?" would get a different answer
//  depending on the viewing angle — the same voice would be in group 3 in
//  one case and alone in another, depending on which folder you happened
//  to be looking at. That would no longer be filtering, but a second
//  inventory.
//
//  So: non-matches are DIMMED, not removed. You then see
//  that a group spans folders and years — and that's exactly the
//  insight you open the map for.
//
//  Everything runs in the browser: checking ~1,000 nodes takes under a
//  millisecond, a server round trip cost 1.2 s per keystroke.
// ══════════════════════════════════════════════════════════════════════════

GRAPH.filter = {text: '', scope: 'all_items', folder: '', from_value: '', to_value: '',
                naming: 'all_items', only: null};
GRAPH.textSource = null;     // per node: 1 = name, 2 = path, 3 = both
GRAPH.visible = null;      // Uint8Array per node, 1 = match
GRAPH.filtered = false;    // is a filter set at all?
/* Counts every change to the view. The tiles discard their painted frame
 * and need a marker that REALLY changes on every change —
 * `GRAPH.visible` is an array whose identity says nothing about its
 * content, and a mere match count would be the same for two different
 * filters with the same number of matches. */
GRAPH.viewState = 0;

/* Recording date from the FILENAME, not from the file timestamp.
 *
 * `modified_at` in the database is often the BACKUP DATE: a voice message
 * recorded years earlier carries the day its folder was copied.
 * A filter on that would silently filter the wrong thing — and nobody
 * would notice.
 *
 * The name, on the other hand, carries it almost always: `AUD-20240301-WA0001.mp3`,
 * `PTT-20240301-…`, `2024-03-01 10-00-00_converted.mp4`. In practice the large
 * majority of files is recognized; the exceptions are
 * hash names without any indication.
 *
 * Return value: "YYYY-MM-DD" or "" — empty means "unknown", not "old".
 */
function graphDateFromName(path) {
    const name = String(path || '').split('/').pop();
    // Year first: AUD-20240301, 2024-03-01 10-00-00, 2024.03.01
    const m = name.match(/(20\d{2})[-_. ]?(\d{2})[-_. ]?(\d{2})/);
    if (m) {
        const [, y, mo, d] = m;
        if (+mo >= 1 && +mo <= 12 && +d >= 1 && +d <= 31) return `${y}-${mo}-${d}`;
    }
    /* Day first, the European way: "Recording 01.03.2024", "(01.03.24)"
     * — the same rule as the extractor's (core/voice_extract.date_from_name). */
    for (const g of name.matchAll(/(?<!\d)(\d{1,2})\.(\d{1,2})\.((?:19|20)?\d{2})(?!\d)/g)) {
        const d = +g[1], mo = +g[2];
        const y = g[3].length === 2 ? '20' + g[3] : g[3];
        if (mo >= 1 && mo <= 12 && d >= 1 && d <= 31 && y >= '2000' && y <= '2099') {
            return `${y}-${String(mo).padStart(2, '0')}-${String(d).padStart(2, '0')}`;
        }
    }
    return '';
}

/* The folder selection from what's REALLY there — not from a list
 * someone would have to maintain. Four path levels are the level at which
 * a typical library differs (e.g. phone/voice-notes vs. videos/meetings). */
function graphFolderList() {
    const count = new Map();
    for (const m of (MAP.members || [])) {
        const parts = String(m.display || '').split('/');
        if (parts.length < 2) continue;
        const dir = parts.slice(0, Math.min(4, parts.length - 1)).join('/');
        count.set(dir, (count.get(dir) || 0) + 1);
    }
    return [...count.entries()].sort((a, b) => b[1] - a[1]).slice(0, 25);
}

function graphApplyFilter() {
    const f = GRAPH.filter;
    const text = (f.text || '').toLowerCase().trim();
    GRAPH.filtered = !!(text || f.folder || f.from_value || f.to_value
                         || f.naming !== 'all_items' || f.only);
    if (!GRAPH.filtered) {
        GRAPH.visible = null;
        GRAPH.viewState++;
        graphFilterState();
        mapDraw();
        return;
    }
    const s = new Uint8Array(MAP.n);
    const reason = new Uint8Array(MAP.n);   // 1 = name, 2 = path, 3 = both
    // Look up group names once instead of searching per node.
    const gname = new Map(GRAPH.groups.map(gr => [gr.nr, gr.name || '']));
    for (let i = 0; i < MAP.n; i++) {
        // A located recording: exactly its voices, by id (`mapLocate`).
        if (f.only && !f.only.has(i)) continue;
        const m = MAP.members[i] || {};
        const path = String(m.display || '');
        if (f.folder && !path.startsWith(f.folder)) continue;
        /* TWO kinds of "named", and they are not the same.
         *
         * With "only named", half of a group called "me" can be grayed
         * out even though those voices are in that group.
         *
         * They are — but the group is a CLAIM by the machine
         * ("these voices sound alike"), the assignment is a HUMAN
         * DETERMINATION ("this voice is me"). On top of that, the group
         * name is only the most frequently assigned name WITHIN it
         * (`api/speaker.py`) — so a group is already called "me" if
         * part of its voices are assigned that way. Typically only a
         * minority of voices carry an assignment.
         *
         * Both are meaningful questions, so both are offered —
         * and the most interesting is the fourth: voices in a named
         * group that don't themselves have an assignment yet. That's
         * exactly the grayed-out half, and at the same time the work
         * list: "this is probably me — confirm it".
         */
        const groupNamed = !!gname.get(GRAPH.groupOf ? GRAPH.groupOf[i] : 0);
        if (f.naming === 'named' && !m.person) continue;
        if (f.naming === 'unnamed' && m.person) continue;
        if (f.naming === 'group_nr' && !groupNamed) continue;
        if (f.naming === 'open' && (m.person || !groupNamed)) continue;
        if (f.from_value || f.to_value) {
            const d = graphDateFromName(path);
            // Without a recognizable date, the voice stays OUT as soon as
            // time is being filtered on — otherwise it could distort any period.
            if (!d) continue;
            if (f.from_value && d < f.from_value) continue;
            if (f.to_value && d > f.to_value) continue;
        }
        if (text) {
            /* WHERE the search looks is a question you need to be able to ask.
             *
             * A search for "alice" can highlight Bob. Not a
             * bug, but a folder: `…/alice/chats/…`. In such a case
             * most matches can come only through the path — Bob,
             * Carol, "me", and unnamed ones that all
             * happen to sit in this folder; only a few through the name.
             *
             * Both are legitimate: sometimes you're searching for the PERSON,
             * sometimes for everything from a FOLDER. It's just that
             * "everywhere" is the wrong default for the more common
             * question. The scope now sits next to it, and the status
             * line says which part came from where — the question "why is
             * Bob lighting up?" should be answered by the view itself.
             */
            const scope = f.scope || 'all_items';
            const inName = ((m.person || '') + ' ' + (m.label || ''))
                .toLowerCase().includes(text);
            const inPath = path.toLowerCase().includes(text);
            if (scope === 'person' && !inName) continue;
            if (scope === 'file' && !inPath) continue;
            if (scope === 'all_items' && !inName && !inPath) continue;
            // For the status line, remember WHY this point is included.
            reason[i] = inName ? (inPath ? 3 : 1) : 2;
        }
        s[i] = 1;
    }
    GRAPH.textSource = text ? reason : null;
    GRAPH.visible = s;
    GRAPH.viewState++;
    graphFilterState();
    mapDraw();
}

function graphFilterState() {
    const el = document.getElementById('mapFilterState');
    if (!el) return;
    if (!GRAPH.filtered) { el.textContent = ''; return; }
    let n = 0;
    const groups = new Set();
    for (let i = 0; i < MAP.n; i++) {
        if (GRAPH.visible[i]) { n++; groups.add(GRAPH.groupOf[i]); }
    }
    const files = new Set();
    for (let i = 0; i < MAP.n; i++) {
        if (GRAPH.visible[i]) files.add(MAP.members[i].display);
    }
    let line = `${n.toLocaleString()} of `
        + `${plural(MAP.n, 'voice')} · ${plural(files.size, 'recording')} `
        + `· ${plural(groups.size, 'group')}`;
    /* Where the matches come from — the answer to "why is Bob lighting up?".
     * Only when there really are both kinds; otherwise it'd be clutter. */
    if (GRAPH.textSource) {
        let name = 0, path = 0;
        for (let i = 0; i < MAP.n; i++) {
            if (!GRAPH.visible[i]) continue;
            if (GRAPH.textSource[i] & 1) name++;
            if (GRAPH.textSource[i] === 2) path++;
        }
        if (name && path) {
            line += ` — ${name} via name, ${path} via path only`;
        }
    }
    el.textContent = line;
}

// The text field filters after a short pause, not on every key: one pass
// over a map of several hundred thousand voices takes ~1.4 s (mostly
// redrawing), and typing a four-letter name used to run it four times back
// to back.
let _graphFilterTimer = null;
function graphFilterSet(field_el, value) {
    GRAPH.filter[field_el] = value;
    clearTimeout(_graphFilterTimer);
    if (field_el === 'text') {
        _graphFilterTimer = setTimeout(graphApplyFilter, MAP.n > 20000 ? 400 : 120);
    } else {
        graphApplyFilter();
    }
}

function graphClearFilter() {
    // "Reset" puts everything back — a located recording too.
    if (GRAPH.locate || GRAPH.filter.only) {
        GRAPH.locate = null;
        mapHighlightChip(null, 0);
        mapLocateReset();
    }
    GRAPH.filter = {text: '', scope: 'all_items', folder: '', from_value: '', to_value: '',
                    naming: 'all_items', only: null};
    GRAPH.textSource = null;
    for (const [id, value] of [['mapFText', ''], ['mapFWhere', 'all_items'],
                              ['mapFFolder', ''],
                              ['mapFFrom', ''], ['mapFTo', ''],
                              ['mapFNaming', 'all_items']]) {
        const el = document.getElementById(id);
        if (el) el.value = value;
    }
    graphApplyFilter();
}

/* Fill the folder selection — rebuilt after every load, because the
 * inventory changes. */
function graphBuildFilter() {
    const sel = document.getElementById('mapFFolder');
    if (!sel) return;
    const old_value = sel.value;
    sel.innerHTML = '<option value="">— all folders —</option>'
        + graphFolderList().map(([dir, n]) =>
            `<option value="${esc(dir)}">${esc(dir.length > 46 ? '…' + dir.slice(-45) : dir)} (${n})</option>`
          ).join('');
    if (old_value) sel.value = old_value;
}

/* ══════════════════════════════════════════════════════════════════════════
 *  FROM A RECORDING TO THE MAP — "locate"
 *
 *  Three bridges lead here: the recording (🗣 Map in the file view and the
 *  search), one voice of it (🗣 Map next to SPEAKER_06) and one spot (🗺
 *  next to a sentence). Locating used to be broken three ways:
 *
 *  1. **By NAME, not by the recording.** The file name went into the
 *     map's text filter. In a real library many file names belong to more
 *     than one recording (a WhatsApp PTT-20240301-WA0001.aac sits in
 *     every backup): the map lit the voices of all of them and jumped to
 *     whichever came first. Now the server resolves the recording and
 *     hands its windows (`/api/speaker/windows?file=…`); the map lights exactly
 *     those, by vector id.
 *  2. **Hidden voices stayed hidden.** With a minimum group size — the map
 *     remembers it; at 6, around 40% of the voices can be out of the
 *     picture — a recording whose voices sit in small groups lit NOTHING.
 *     Now the located voices are pinned: shown whatever their group size.
 *  3. **Silence where there is nothing.** A recording without voice
 *     vectors (transcribed without telling speakers apart, or before voice
 *     vectors existed), or with its voices in the trash, opened a map with nothing
 *     lit and no word why. Now the chip says which case it is — and the
 *     file view doesn't offer the buttons where there is nothing to find.
 *
 *  The located recording survives a new threshold: it is held by vector
 *  id and lit again after every load (`mapLocateApply` in `mapLoadGraph`).
 * ══════════════════════════════════════════════════════════════════════════ */

GRAPH.locate = null;     // {file, name, label, t, windows: [{id, label, start_sec, trashed}]}
GRAPH.pinned = null;     // node indices shown whatever their group size

/* Resolves when the map has finished loading. NOT "there are nodes":
 * those are still from the previous run while a new one is in flight,
 * and that run clears the marker. */
function mapWhenLoaded() {
    return new Promise(resolve => {
        const check = () => {
            if (GRAPH.doneSeq === GRAPH.seq) resolve();
            else setTimeout(check, 250);
        };
        check();
    });
}

/* The one bridge. `opts.label` narrows to one voice of the recording,
 * `opts.t` (seconds) to the window a spot lies in. */
async function mapLocate(file, opts = {}) {
    const name = String(file || '').split('/').pop();
    const L = {file, name, label: opts.label || '', t: opts.t, windows: null};
    GRAPH.locate = L;
    mapOpen();
    mapHighlightChip(null, 0, 0, `Locating <b>${esc(name)}</b> …`);
    let d;
    try {
        // As a parameter, not in the URL path: an absolute path there loses
        // its leading slash to the router's redirect (api/speaker.py).
        d = await mapResponse(await fetch('/api/speaker/windows?file=' + encodeURIComponent(file)));
    } catch (e) {
        if (GRAPH.locate === L) {
            mapHighlightChip(null, 0, 0, `<b>${esc(name)}</b> · could not be located: `
                                         + esc(e.message || String(e)));
        }
        return;
    }
    if (GRAPH.locate !== L) return;          // a newer locate took over
    L.windows = d.windows || [];
    await mapWhenLoaded();
    if (GRAPH.locate !== L) return;
    /* The TILES are the entry point, even for a single voice — not the
     * simulation; they draw the
     * highlight themselves (border around the tile, mark on the cell). */
    if (typeof TILE !== 'undefined' && !TILE.on && typeof mapView === 'function') {
        mapView('tiles');
    }
    mapLocateApply(true);
}

function mapForFile(displayPath) {
    return mapLocate(displayPath);
}

function mapForSpeaker(displayPath, label) {
    return mapLocate(displayPath, {label});
}

/* One spot: jump from a chunk in the transcript to the map. A window is
 * 30 s of NET speech
 * glued from several places, so the right one is the last window of THIS
 * label that doesn't start after the spot (`mapSpotWindow`). */
function mapForSpot(displayPath, seconds, label) {
    return mapLocate(displayPath, {label, t: Number(seconds) || 0});
}

/* Light the located recording on the map as it is now — after locating,
 * and again after every load (the indices change with the threshold). */
function mapLocateApply(select = false) {
    const L = GRAPH.locate;
    if (!L || !L.windows) return;
    const head = `<b>${esc(L.name)}</b> · `;
    if (!L.windows.length) {
        mapLocateReset();
        mapHighlightChip(null, 0, 0, head + 'no voices are on file for this recording at all — '
            + 'it was transcribed without telling speakers apart, or before voice vectors '
            + 'existed. A <i>Voices only</i> job adds them (queue editor → From library).');
        return;
    }
    const rows = [];
    for (const w of L.windows) {
        const i = GRAPH.rowOf ? GRAPH.rowOf.get(w.id) : undefined;
        if (i !== undefined) rows.push(i);
    }
    const trashed = L.windows.filter(w => w.trashed).length;
    // What the minimum size or a group selection would hide — pinned below.
    GRAPH.pinned = null;
    graphBuildActive();
    const hidden = rows.filter(i => !graphActive(i)).length;
    GRAPH.pinned = rows.length ? new Set(rows) : null;
    graphBuildActive();
    graphHideState();
    GRAPH.filter.only = rows.length ? new Set(rows) : null;

    let marks = rows, spot = -1, text;
    if (L.label) marks = rows.filter(i => (MAP.members[i] || {}).label === L.label);
    if (L.t !== undefined && L.t !== null) {
        const w = mapSpotWindow(L.label, L.t, rows);
        spot = w.hits;
        marks = spot >= 0 ? [spot] : [];
        text = mapSpotText(L.label, L.t, spot >= 0, w.ofLabel, w.inFile,
                           w.from_value, w.to_value);
    } else if (L.label) {
        text = mapHighlightText(L.label, marks.length, rows.length);
    } else if (!L.file) {
        // Voices given by id (the History's "show on the map")
        text = `${vxN(rows.length)} of ${vxN(L.windows.length)} voice${L.windows.length === 1 ? '' : 's'} lit`
            + (rows.length < L.windows.length ? ` — ${vxN(L.windows.length - rows.length)} ${L.windows.length - rows.length === 1 ? 'is' : 'are'} not on `
               + `this map (${GRAPH.trash ? 'not in the trash' : 'in the trash or gone'})` : '');
    } else {
        text = `${rows.length} ${rows.length === 1 ? 'voice' : 'voices'} of this recording lit`;
    }
    MAP.highlight = marks.length ? new Set(marks) : null;
    graphApplyFilter();
    graphRearrange();

    const notes = [];
    if (!GRAPH.trash && trashed) {
        notes.push(`${trashed} in the trash <button class="map-mini" onclick="graphTrashView(true)">show there</button>`);
    }
    if (GRAPH.trash && L.windows.length > rows.length) {
        notes.push(`${L.windows.length - rows.length} not in the trash `
                   + `<button class="map-mini" onclick="graphTrashView(false)">show in the library</button>`);
    }
    if (hidden) notes.push(`${hidden} shown although their group is below the minimum size`);
    if (!rows.length && L.file) {
        text = GRAPH.trash ? 'none of its voices is in the trash'
                           : 'all its voices are in the trash';
    }
    mapHighlightChip(null, 0, 0, head + text + (notes.length ? ' · ' + notes.join(' · ') : ''));
    if (select && (spot >= 0 || marks.length || rows.length)) {
        const first = spot >= 0 ? spot : (marks.length ? marks[0] : rows[0]);
        if (typeof mapPickVoice === 'function') mapPickVoice(first);
    } else {
        mapDraw();
    }
}

/* Forget the located recording: its pins, its light, its marks. */
function mapLocateReset() {
    const had = !!(GRAPH.pinned || GRAPH.filter.only);
    GRAPH.pinned = null;
    GRAPH.filter.only = null;
    MAP.highlight = null;
    if (had) {
        graphBuildActive();
        graphHideState();
        graphApplyFilter();
        graphRearrange();
    }
}

/* The SELECTION, separate from the path getting there.
 *
 * As its own function, because this is where the rule lives, not the
 * click handling: within the ID, the last window that doesn't start
 * after `t`. A window is 30s of net speech from multiple spots, so window
 * k ends where k+1 begins — pauses and other speakers in between still
 * belong to k.
 *
 * If `t` lies BEFORE the ID's first window (a short interjection that
 * didn't make it in), the first one is the most honest answer. It never
 * falls back to a DIFFERENT ID: that would be another person,
 * highlighted as if they were the one being searched for.
 */
function mapSpotWindow(label, t, rows = null) {
    // The candidates: the located recording's voices — or, without them,
    // what the filter leaves bright.
    const ids = rows || [...Array(MAP.n).keys()].filter(
        i => !(GRAPH.visible && !GRAPH.visible[i]));
    let hits = -1, bestTime = -Infinity;
    let firstOne = -1, firstTime = Infinity, ofLabel = 0, inFile = 0;
    /* The start of the NEXT window of the same ID — it's the end
     * of the matched one.
     *
     * Locating a section can land slightly before it. The
     * assignment is correct (e.g. line 1:02:40, window 1:02:00–1:02:48), it's
     * just that the listening sample begins at the START of that span —
     * 40 seconds earlier. That's not an inaccuracy, it's the construction:
     * 30 seconds of net speech out of 48 seconds of recording. It just
     * needs to be said. */
    let to_value = Infinity;
    for (const i of ids) {
        inFile++;
        const m = (MAP.members || [])[i] || {};
        if (label && m.label !== label) continue;
        ofLabel++;
        const s0 = Number(m.start_sec) || 0;
        if (s0 <= t + 0.001 && s0 > bestTime) { bestTime = s0; hits = i; }
        if (s0 < firstTime) { firstTime = s0; firstOne = i; }
    }
    if (hits < 0) hits = firstOne;
    if (hits >= 0) {
        const from_value = Number((MAP.members[hits] || {}).start_sec) || 0;
        for (const i of ids) {
            const m = (MAP.members || [])[i] || {};
            if (label && m.label !== label) continue;
            const s0 = Number(m.start_sec) || 0;
            if (s0 > from_value && s0 < to_value) to_value = s0;
        }
        return {hits, ofLabel, inFile, from_value, to_value};
    }
    return {hits, ofLabel, inFile, from_value: null, to_value: null};
}

/* The sentence on the label. Four cases that need to be told apart —
 * otherwise the same message sends the user in different directions. */
/* Clock time within the RECORDING's timeline, with hours.
 *
 * This used to show `125:30` — 125 minutes. That's computed correctly and
 * unreadable: nobody converts that to 2:05 in their head, and the
 * transcript right next to it has long since shown time with hours. */
function mapClockTime(sec) {
    sec = Math.max(0, Number(sec) || 0);
    const h = Math.floor(sec / 3600);
    const m = String(Math.floor((sec % 3600) / 60)).padStart(2, '0');
    const s = String(Math.floor(sec % 60)).padStart(2, '0');
    return h ? `${h}:${m}:${s}` : `${Number(m)}:${s}`;
}

function mapSpotText(label, sec, found, ofLabel, inFile,
                         from_value = null, to_value = null) {
    const timestamp = mapClockTime(sec);
    const k = label ? `<b>${esc(label)}</b> · ` : '';
    /* This addition isn't decoration. The jump restricts the map to the
     * RECORDING, and its voices light up scattered across the tiles of many
     * people — that looked like "lots of matches" even though exactly
     * one is marked. Anyone who doesn't know what "bright" means
     * would go looking for a bug in the jump. */
    if (found) {
        /* The SPAN, not just the start. A window is 30 seconds of net
         * speech from a longer stretch; the listening sample starts at the
         * beginning. Anyone who only reads "window at 1:02" would
         * take it for a wrong hit — with the span next to it, you see right
         * away that your own spot lies within it. */
        const span = (from_value !== null && Number.isFinite(to_value))
            ? `${mapClockTime(from_value)}–${mapClockTime(to_value)}`
            : (from_value !== null ? `from ${mapClockTime(from_value)}` : timestamp);
        return k + `Window ${span} <span class="map-small">`
                 + `· your spot ${timestamp} (crosshair) · bright = remaining voices `
                 + `in this recording</span>`;
    }
    if (ofLabel) return k + `no window before ${timestamp}`;
    if (inFile) return k + `no windows — the recording only has `
                          + plural(inFile, 'other');
    return k + 'no voices are on file for this recording at all';
}

/* What's on the label — as its own function, because this is where the
 * information logic lives, not the markup.
 *
 * Searching for `SPEAKER_06` from a recording once showed
 * *"no voice windows on file here"*. That sounded like "this one voice
 * is missing" — in fact there was not a single window for the whole
 * recording: it was transcribed before voice recognition existed, and
 * older transcripts often have no voice data at all (see
 * `speaker.db`). Answering both situations with the same sentence sends
 * the search in the wrong direction: in one case the voice was too quiet,
 * in the other the analysis was never done.
 */
function mapHighlightText(label, count, inFile) {
    const k = `<b>${esc(label)}</b> · `;
    if (count) return k + `${plural(count, 'window')} highlighted`;
    if (inFile) return k + `no windows — the recording only has `
                          + plural(inFile, 'other');
    return k + 'no voices are on file for this recording at all';
}

// The highlight label. It stays until you click it away —
// the highlight survives zooming and filtering so you don't lose the
// voice while exploring, and therefore has to stay visible.
function mapHighlightChip(label, count, inFile, text) {
    const el = document.getElementById('mapHighlightChip');
    if (!el) return;
    // `text` wins: the jump to ONE spot needs to say something different
    // than the jump to a whole ID (mapSpotText).
    if (!label && !text) { el.style.display = 'none'; el.innerHTML = ''; return; }
    el.style.display = '';
    // The long sentence belongs on the mouse hover, not in the line: the
    // label sits above the map and must not cover it up.
    el.title = (!count && !inFile)
        ? 'Voice vectors are created for every new transcription. '
          + 'Older recordings get them as a "Voices only" job '
          + '(queue editor → From library).'
        : 'Clear highlight';
    const x = ' <span class="map-highlight-x" onclick="mapClearHighlight()"'
            + ' title="Clear highlight">✕</span>';
    el.innerHTML = (text || mapHighlightText(label, count, inFile)) + x;
}

function mapClearHighlight() {
    GRAPH.locate = null;
    mapLocateReset();
    MAP.highlight = null;
    mapHighlightChip(null, 0);
    mapDraw();
}

/* Selection, environment, the sequential player, and naming live in
 * `voicemap_panel.js`.
 *
 * They're one coherent piece of UI logic and never belonged in the
 * drawing file: the sequential player selected a voice with every clip,
 * and the selection rebuilt the panel that held the player's own
 * buttons while doing so. The detailed reasoning is documented there. */

// ── Names on the map ──────────────────────────────────────────────────────

/* ONE label per group, at its centroid, in its color.
 *
 * Names used to be too small to read and didn't scale along. Two bugs
 * were behind that:
 *
 *   1. The labeling from the old bubble view depended on ``s >= 0.9``
 *      (zoom level 0.9). The graph sits at about 0.26 after fitting —
 *      so no name appeared at all, only the legend knew them.
 *   2. Even then it would have been 11 pixels, regardless of anything else.
 *
 * Here the font grows with GROUP SIZE (a group of 500 is the
 * map's main message and is allowed to show it) and with ZOOM. Both are
 * clamped: below 12 px nobody can read it, above 34 px it covers the points.
 *
 * Zooming in brings smaller groups into view — anyone examining a corner
 * closely wants the small names there too. On the overview, on the other
 * hand, 61 labels would be a carpet.
 */
function graphDrawNames(g, s, ox, oy, w, h) {
    if (!GRAPH.showNames || !GRAPH.groupOf) return;
    if (GRAPH.running) return;   // decoration, see outlines

    // Centroid per group — over the nodes, not over the legend: the
    // points move while the force layout is running.
    const middle = new Map();
    for (let i = 0; i < MAP.n; i++) {
        const nr = GRAPH.groupOf[i];
        if (!nr || !graphActive(i)) continue;
        const s = graphKey(nr);
        let m = middle.get(s);
        if (!m) { m = {x: 0, y: 0, n: 0}; middle.set(s, m); }
        m.x += MAP.x[i]; m.y += MAP.y[i]; m.n++;
    }

    /* NAMED groups are ALWAYS shown — regardless of size, rank, and
     * zoom.
     *
     * Take a small named group, say "Alice". The
     * thinning-out below discarded it first when zooming out (no label
     * below five voices, and only the 22 largest at all) — exactly
     * when you're looking at the overview and actually want to find them.
     *
     * That was the wrong order. An assigned name is the only piece of
     * information on this map that a HUMAN made; everything else was
     * computed by the machine. What gets thinned out should be "Group 37",
     * not "Alice".
     *
     * THE SAME NAME MULTIPLE TIMES is the normal case here, not the
     * exception: a library easily has several times more named groups than
     * people — "me" nine times, "Bob" six times, "Alice" three
     * times. The machine hasn't merged them, and that's a statement that
     * belongs on the map. Nine "me"s next to each other on the overview,
     * though, is no longer a statement, just color.
     *
     * So: on the overview, each NAME appears once, on its largest
     * group. Anyone zooming in sees all occurrences — that's where you
     * look for the splinters, on the overview you look for the person.
     *
     * The outlines show the remaining groups anyway; only the
     * label is hidden, never the group. */
    const fromSize = s >= 2 ? 2 : (s >= 1 ? 3 : 5);
    const howMany = s >= 1 ? 60 : 22;
    // In cluster mode, all parts of a name share ONE centroid.
    // Without bundling, eleven "me"s would sit exactly on top of each
    // other — the same ink, eleven times, and the outline would grow
    // thick and unreadable from it.
    const bundleNames = GRAPH.clusterNames || s < 1;

    g.textAlign = 'center';
    g.textBaseline = 'middle';
    g.lineJoin = 'round';
    let painted = 0;
    const alreadyThere = new Set();
    // Named ones first, so they end up ON TOP when overlapping. `groups`
    // is sorted descending by size — so the first match of a name is
    // its largest group.
    const row = GRAPH.groups.filter(gr => gr.name)
        .concat(GRAPH.groups.filter(gr => !gr.name));
    for (const gr of row) {
        if (gr.name) {
            // Only marked as done AFTER drawing, further below. Otherwise
            // a group that's currently outside the viewport would use up
            // its name's one slot — and the name would be missing
            // entirely, even though a smaller group would be visible.
            if (bundleNames && alreadyThere.has(gr.name)) continue;
        } else {
            if (painted >= howMany) continue;
            if (gr.n < fromSize) continue;
        }
        const m = middle.get(graphKey(gr.nr));
        if (!m) continue;
        // `m.n` instead of `gr.n`: what's counted is what's REALLY in the
        // picture. If groups are hidden or several parts are bundled into
        // one cluster, the number from the group list would differ from
        // what's visible — and a number you can count that's wrong is
        // worse than none.
        const count = m.n;
        const px = ox + s * (m.x / m.n), py = oy + s * (m.y / m.n);
        if (px < -80 || px > w + 80 || py < -20 || py > h + 20) continue;

        const basis = 11 + Math.sqrt(gr.n) * 1.05;
        const size = Math.max(12, Math.min(34,
            basis * (0.9 + 0.25 * Math.min(2, s))));
        g.font = `600 ${size.toFixed(1)}px system-ui, sans-serif`;
        const text = gr.name || `Group ${gr.nr}`;

        // Dark outline around the text: over a cluster of same-colored
        // points, colored text would otherwise be unreadable. Cheaper and
        // more reliable than a shadow — that gets rasterized fresh
        // every frame and costs noticeably at 60 labels.
        g.lineWidth = Math.max(3, size / 7);
        g.strokeStyle = 'rgba(10,12,18,0.92)';
        g.strokeText(text, px, py);
        g.fillStyle = graphColour(gr.nr);
        g.fillText(text, px, py);

        /* The count directly below, smaller and dimmer.
         *
         * It used to appear only in the legend — and that shows ten
         * groups out of hundreds. For everything else you had to click. But the
         * number is exactly what puts the name in context: "me" with 500
         * is something different from "Erin" with 3.
         *
         * Half the font size and muted, so it accompanies the name
         * instead of competing with it. Only from 13 px readable base font
         * — below that it would become a gray smudge. */
        if (size >= 13) {
            const small = Math.max(9, size * 0.52);
            g.font = `500 ${small.toFixed(1)}px system-ui, sans-serif`;
            const number = count.toLocaleString();
            const ny = py + size * 0.72;
            g.lineWidth = Math.max(2.5, small / 7);
            g.strokeText(number, px, ny);
            g.globalAlpha = 0.72;
            g.fillText(number, px, ny);
            g.globalAlpha = 1;
        }
        // Only the anonymous ones count against the quota. If the named
        // ones were counted too, dozens of names would take everything away
        // from the remainder — thinning out would then discard the wrong thing again.
        if (gr.name) alreadyThere.add(gr.name); else painted++;
    }
    g.textBaseline = 'alphabetic';
}

function graphToggleNames(on) {
    GRAPH.showNames = !!on;
    mapDraw();
}

// ── Legend ──────────────────────────────────────────────────────────────

/* ── The largest groups — the details with nothing chosen ───────────────
 *
 * The legend used to float at the bottom left over the map, collapsed by
 * default because it covered voices. Docked in the details it covers
 * nothing, so it can show more than ten: the forty largest groups, with
 * the checkboxes that keep only them in the picture. A click opens a
 * group — its largest voice, the recordings, naming it as a whole.
 */
const LEGEND_SHOWN = 40;

/* The list's order: by number of voices or by name, either way round.
 * By name, the named groups come first (A–Z or Z–A) and the unnamed ones
 * after them by number — "Group 12" sorted as text between the names would
 * bury the people. Remembered per browser. */
const LEGEND_SORTS = {
    size_desc: ['Voices ↓', (a, b) => b.n - a.n || a.nr - b.nr],
    size_asc: ['Voices ↑', (a, b) => a.n - b.n || a.nr - b.nr],
    name_asc: ['Name A–Z', (a, b) => _legendByName(a, b, 1)],
    name_desc: ['Name Z–A', (a, b) => _legendByName(a, b, -1)],
};

function _legendByName(a, b, dir) {
    if (!!a.name !== !!b.name) return a.name ? -1 : 1;
    if (!a.name) return a.nr - b.nr;
    return dir * a.name.localeCompare(b.name, undefined, {sensitivity: 'base'}) || b.n - a.n;
}

function _legendSort() {
    if (!GRAPH.legendSort) {
        let kept = null;
        try { kept = localStorage.getItem('whisperx.voices.legendSort'); } catch (e) { /* none */ }
        GRAPH.legendSort = LEGEND_SORTS[kept] ? kept : 'size_desc';
    }
    return GRAPH.legendSort;
}

function graphLegendSort(key) {
    if (!LEGEND_SORTS[key]) return;
    GRAPH.legendSort = key;
    try { localStorage.setItem('whisperx.voices.legendSort', key); } catch (e) { /* none */ }
    graphLegend();
}

function graphLegend() {
    const el = document.getElementById('mapLegend');
    if (!el) return;
    const groups = GRAPH.groups || [];
    const sort = _legendSort();
    const top = [...groups].sort(LEGEND_SORTS[sort][1]).slice(0, LEGEND_SHOWN);
    const tiles = typeof TILE !== 'undefined' && TILE.on;
    el.innerHTML = `
        <div class="map-legend-head">${sort === 'size_desc' ? 'Largest groups' : sort === 'size_asc' ? 'Smallest groups' : 'Groups by name'}
            <span class="map-small">· ${vxN(groups.length)} at threshold
            ${(GRAPH.threshold ?? 0).toFixed(2)}</span></div>
        <div class="map-legend-sort">${Object.entries(LEGEND_SORTS).map(([k, [label]]) =>
            `<button class="${k === sort ? 'on' : ''}" onclick="graphLegendSort('${k}')">${label}</button>`).join('')}</div>
        ${GRAPH.selection ? `<div style="margin:0 0 8px"><button class="map-mini"
             onclick="graphClearSelection()">Show all groups again</button>
             <span class="map-small">${GRAPH.selection.size} chosen</span></div>` : ''}
        ${top.length ? top.map(gr => `
            <div class="map-legend-row" onclick="graphOpenGroup(${gr.nr})"
                 title="Open this group — its voices, recordings and name">
                <input type="checkbox" class="map-legend-check"
                       title="Keep only the checked groups in the picture"
                       ${GRAPH.selection && GRAPH.selection.has(gr.nr) ? 'checked' : ''}
                       onclick="event.stopPropagation();graphToggleSelection(${gr.nr})">
                <span class="map-point" style="background:${graphColour(gr.nr)}"></span>
                <span class="map-legend-name">${gr.name ? esc(gr.name) : 'Group ' + gr.nr}</span>
                <span class="map-legend-n">${vxN(gr.n)}</span>
            </div>`).join('')
          : '<div class="map-small">No groups at this threshold.</div>'}
        <div class="map-small map-legend-help">`
        // Controls differ between the two views: "wheel = zoom" would be
        // wrong in the tiles — there is neither zoom nor dragging there.
        + (tiles ? 'One tile = one group, one cell = one voice. Click a cell to listen and name it.'
                 : 'One point = one voice. Click a point to listen and name it; wheel = zoom, drag = pan.')
        + ` Grey = no group. The checkboxes keep only the checked groups in the picture.</div>`;
    if (typeof panelSideContent === 'function') panelSideContent();
}

/* Open a group from the list: in the graph, zoom there; in both views,
 * choose its voice with the most speech — the details then show the
 * group's recordings and its "name the whole group". */
function graphOpenGroup(nr) {
    let best = -1, bestSpeech = -1;
    for (let i = 0; i < MAP.n; i++) {
        if (GRAPH.groupOf[i] !== nr || !graphActive(i)) continue;
        const sp = Number(MAP.members[i].speech_sec) || 0;
        if (sp > bestSpeech) { bestSpeech = sp; best = i; }
    }
    if (best < 0) return;
    if (typeof TILE === 'undefined' || !TILE.on) graphOnGroup(nr);
    MAP.highlight = null;
    mapPickVoice(best);
    mapDraw();
}

/* Zoom to a group — the replacement for the earlier "click into the
 * group". The group stays visible IN its surroundings while doing so; that's
 * exactly what got lost in the old layered view. */
function graphOnGroup(nr) {
    let sx = 0, sy = 0, k = 0, minX = 1e9, maxX = -1e9, minY = 1e9, maxY = -1e9;
    for (let i = 0; i < MAP.n; i++) {
        if (GRAPH.groupOf[i] !== nr) continue;
        sx += MAP.x[i]; sy += MAP.y[i]; k++;
        minX = Math.min(minX, MAP.x[i]); maxX = Math.max(maxX, MAP.x[i]);
        minY = Math.min(minY, MAP.y[i]); maxY = Math.max(maxY, MAP.y[i]);
    }
    if (!k) return;
    const cv = document.getElementById('mapCanvas');
    const w = cv.clientWidth, h = cv.clientHeight;
    const bx = Math.max(60, maxX - minX), by = Math.max(60, maxY - minY);
    MAP.sx = Math.min(6, 0.7 * Math.min(w / bx, h / by));
    MAP.ox = w / 2 - MAP.sx * (sx / k);
    MAP.oy = h / 2 - MAP.sx * (sy / k);
    mapDraw();
}

// ── Controls ─────────────────────────────────────────────────────────────

let _graphWaiting = null;

/* The slider triggers a recomputation on the server (measured ~1.2 s for
 * grouping at ~1,000 voices). Hence debounced: it computes when
 * the user releases the slider, not on every intermediate value. */
/* Only DISPLAY, don't compute: set slider and number to a value.
 * Separate from `graphThresholdSet`, because applying the server response
 * would otherwise immediately trigger a new request with the same value. */
function graphShowThreshold(value) {
    if (value == null) return;
    const s = document.getElementById('mapThreshold');
    if (s) s.value = Number(value).toFixed(2);
    const cnt = document.getElementById('mapThresholdValue');
    if (cnt) cnt.textContent = Number(value).toFixed(2);
}

function graphThresholdSet(value) {
    const cnt = document.getElementById('mapThresholdValue');
    if (cnt) cnt.textContent = Number(value).toFixed(2);
    clearTimeout(_graphWaiting);
    _graphWaiting = setTimeout(() => mapLoadGraph({threshold: Number(value)}), 350);
}

/* A single step without dragging.
 *
 * Up and down arrows control the threshold without dragging — the
 * slider spans 0.30 to 0.95
 * over about 200 pixels, so about three pixels per hundredth. But exactly
 * one hundredth makes a visible difference: 0.62 → 0.61 changes the inventory.
 *
 * The step comes from the slider's own `step`, not from a second
 * number here — otherwise the arrow and dragging would eventually drift apart.
 */
function graphThresholdStep(direction) {
    const s = document.getElementById('mapThreshold');
    if (!s) return;
    const level = Number(s.step) || 0.01;
    const min = Number(s.min), max = Number(s.max);
    // Round to a multiple of the step: a value dragged by hand like
    // 0.6449 would otherwise jump to 0.6549 while the display kept showing 0.65.
    const now = Math.round(Number(s.value) / level) * level;
    const new_value = Math.min(max, Math.max(min, now + direction * level));
    s.value = new_value.toFixed(2);
    graphThresholdSet(s.value);
}

/* ══════════════════════════════════════════════════════════════════════════
 *  AUTO-SEARCH — at which threshold is THIS person the largest?
 *
 *  The auto-search runs one person's group over all thresholds and shows
 *  where it is biggest — some people only form a big group at one
 *  specific threshold.
 *
 *  That's the point: the threshold is not a
 *  global quality knob. An example:
 *
 *      Alice    best threshold 0.58 → 23 of 24 voices, purity 1.00
 *      Bob      best threshold 0.67 → 12 voices (only one at 0.65)
 *      Carol    best threshold 0.70 → 150 voices, purity 0.99
 *
 *  For Bob, between 0.65 and 0.66 lies the difference between "doesn't
 *  exist" and "twelve voices". By hand, you'd only find that through
 *  trial and error.
 *
 *  TWO RESULTS, not one. "Biggest" alone is
 *  misleading: Carol grows to 400 voices at 0.46 — with
 *  purity 0.40, because the group is then half made of other people.
 *  That's why the best BALANCED threshold (F₁ of purity and coverage)
 *  sits next to it on equal footing. Which one to use is a decision for
 *  the human.
 * ══════════════════════════════════════════════════════════════════════════ */

let _graphSearchRun = 0;

function graphThresholdSearch(name) {
    const wanted = (name || graphSearchName() || '').trim();
    const bar = document.getElementById('mapSearchBar');
    if (!bar) return;
    if (!wanted) {
        graphSearchReport('Name a person first — type a name into the filter field '
            + 'or click a named voice.');
        return;
    }
    const mine = ++_graphSearchRun;
    graphSearchReport(`Searching for the best threshold for "${esc(wanted)}" … `
        + 'this takes about 20 seconds (every threshold is computed).');
    const p = new URLSearchParams({person: wanted,
                                   link_method: GRAPH.link_method});
    fetch('/api/voicemap/threshold-search?' + p)
        .then(mapResponse)
        .then(d => {
            if (mine !== _graphSearchRun) return;   // a newer one has superseded this
            graphSearchResult(d);
        })
        .catch(e => {
            if (mine !== _graphSearchRun) return;
            graphSearchReport('Search failed: ' + (e.message || e));
        });
}

/* Where the name comes from when none is given: first the filter field,
 * then the currently selected voice. Both are what you're currently
 * working on — having to type the name a third time would be the kind of
 * friction that makes you stop using a tool. */
function graphSearchName() {
    const field_el = document.getElementById('mapFText');
    if (field_el && field_el.value.trim()) return field_el.value.trim();
    if (typeof MAP !== 'undefined' && MAP.chosen && MAP.chosen.person) {
        return MAP.chosen.person;
    }
    return '';
}

function graphSearchReport(html) {
    const bar = document.getElementById('mapSearchBar');
    if (!bar) return;
    bar.style.display = '';
    bar.innerHTML = `<span class="map-small">${html}</span>`;
}

function graphSearchResult(d) {
    const bar = document.getElementById('mapSearchBar');
    if (!bar) return;
    if (d.hint || !d.largest) {
        graphSearchReport(esc(d.hint
            || `Could not find a group at any threshold for "${d.person}" `
               + 'that carries this name.'));
        return;
    }
    const g = d.largest, b = d.best;
    const high = Math.max(...d.points.map(p => p.n)) || 1;
    /* The whole curve as clickable bars. It's the actual finding:
     * you SEE how narrow the window is in which this person exists —
     * and a click sets the threshold there. */
    const bars = d.points.map(p => {
        const h = Math.max(2, Math.round(20 * p.n / high));
        const kind = p.threshold === g.threshold ? ' map-search-max'
                  : (b && p.threshold === b.threshold ? ' map-search-best' : '');
        return `<span class="map-search-bar${kind}" style="height:${h}px"
                      title="Threshold ${p.threshold.toFixed(2)}: ${plural(p.n, 'voice')},
                             ${p.hits} of which named ${esc(d.person)}
                             (purity ${(p.purity * 100).toFixed(0)}%)"
                      onclick="graphSetThreshold(${p.threshold})"></span>`;
    }).join('');

    const button = (p, was) => `<button class="map-mini"
        onclick="graphSetThreshold(${p.threshold})">${was}: ${p.threshold.toFixed(2)}
        → ${plural(p.n, 'voice')}</button>`;
    bar.style.display = '';
    bar.innerHTML = `
        <span class="map-small"><b>${esc(d.person)}</b> ·
            ${plural(d.total, 'named voice')} · ${d.points.length} thresholds
            in ${d.duration_sec} s</span>
        ${button(g, 'biggest')}
        ${b && b.threshold !== g.threshold ? button(b, 'most balanced') : ''}
        <span class="map-search-curve">${bars}</span>
        <span class="map-small">Purity ${(g.purity * 100).toFixed(0)}%
            · covers ${(g.coverage * 100).toFixed(0)}% of the named ones</span>
        <button class="map-mini" onclick="graphSearchClose()"
                title="Hide bar">✕</button>`;
}

function graphSearchClose() {
    _graphSearchRun++;
    const bar = document.getElementById('mapSearchBar');
    if (bar) { bar.style.display = 'none'; bar.innerHTML = ''; }
}

/* Set the threshold AND compute — the same path as the slider, so
 * display, slider, and inventory don't drift apart. */
function graphSetThreshold(value) {
    const s = document.getElementById('mapThreshold');
    if (s) s.value = Number(value).toFixed(2);
    graphThresholdSet(Number(value).toFixed(2));
}

/* Re-arrange — and re-tile in the tile view.
 *
 * This is the path through which EVERY hide and selection change makes it
 * back into the picture: "only this", the legend checkboxes, "clear
 * selection", the minimum size, the name bundling. It used to be
 * `graphInitialLayout(); graphRun();` — and `graphRun` returns
 * immediately in the tile view. So NOTHING got redrawn there: the
 * supposedly hidden groups sat unchanged in the picture while the line
 * below claimed "N of M hidden" — exactly the contradiction that
 * the surrounding comments are meant to rule out.
 */
function graphRearrange() {
    if (typeof TILE !== 'undefined' && TILE.on) {
        tileLayout(true);
        mapDraw();
        return;
    }
    graphInitialLayout();
    graphRun();
}

/* Pull and duration take effect WITHOUT a server request: nodes, edges,
 * and groups are unchanged, only their arrangement is recomputed. A reload
 * from the server would have cost 1.2 s here and delivered the same data. */
function graphDragSet(value) {
    GRAPH.drag = Math.min(DRAG_MAX, Math.max(0.1, Number(value) || 1));
    const cnt = document.getElementById('mapDragValue');
    if (cnt) cnt.textContent = GRAPH.drag.toFixed(1) + '×';
    // Move the slider along. When operated by hand it's the source and
    // is already correct anyway — but not when the value comes from
    // somewhere else (clamped, restored from memory, set programmatically).
    // Measured, the slider then showed "1" while "10.0×" was displayed
    // next to it and was also in effect. A slider showing something
    // different from what it does is worse than none — the same rule
    // already applies to the duration.
    const slider = document.getElementById('mapDrag');
    if (slider && Number(slider.value) !== GRAPH.drag) {
        slider.value = GRAPH.drag;
    }
    graphSaveSettings();
    graphRearrange();
}

function graphStepsSet(value) {
    GRAPH.steps = Math.min(5000, Math.max(50, Math.round(Number(value) || 420)));
    const field_el = document.getElementById('mapSteps');
    if (field_el && Number(field_el.value) !== GRAPH.steps) field_el.value = GRAPH.steps;
    graphSaveSettings();
    graphRearrange();
}

/* Bring the sliders to the remembered state. Must run BEFORE the first
 * layout starts — otherwise the first run computes with the defaults
 * while the fields show the remembered values. A slider showing something
 * different from what it does is worse than none. */
/* ── Hiding: minimum size and selection ───────────────────────────────────
 *
 * Both change WHO participates at all — so re-arranging is required
 * afterward. Just redrawing would show the remaining groups at the old
 * locations, with the old gaps between them; the actual benefit (they now
 * have room and draw together) would stay invisible.
 */
function graphMinGroupSet(value) {
    GRAPH.minGroup = Math.min(500, Math.max(1, Math.round(Number(value) || 1)));
    const field_el = document.getElementById('mapMinGroup');
    if (field_el && Number(field_el.value) !== GRAPH.minGroup) field_el.value = GRAPH.minGroup;
    graphSaveSettings();
    graphBuildActive();
    graphHideState();
    graphRearrange();
}

/* Density: scales repulsion.
 *
 * To make nodes actually pull together into tight clusters, the pull
 * slider alone isn't enough, because repulsion holds against it. IT is the reason
 * a cluster doesn't get tighter no matter how hard you pull. Smaller
 * value = denser.
 *
 * Open-ended downward to 0.05; capped upward at 2.0. The reason for the
 * cap lies in the force layout: at 900 (i.e. ~2.0), points lock into a
 * regular grid that looks like structure and isn't.
 */
function graphDensitySet(value) {
    GRAPH.density = Math.min(2, Math.max(0.05, Number(value) || 1));
    const field_el = document.getElementById('mapDensity');
    if (field_el && Number(field_el.value) !== GRAPH.density) field_el.value = GRAPH.density;
    const cnt = document.getElementById('mapDensityValue');
    if (cnt) cnt.textContent = GRAPH.density.toFixed(2) + '×';
    graphSaveSettings();
    graphRearrange();
}

/* Add or remove a group from the selection. Empty selection
 * means "all" — a picture without a single point isn't a state you'd
 * want to end up in without meaning to. */
function graphToggleSelection(nr) {
    if (!GRAPH.selection) GRAPH.selection = new Set();
    if (GRAPH.selection.has(nr)) GRAPH.selection.delete(nr);
    else GRAPH.selection.add(nr);
    if (!GRAPH.selection.size) GRAPH.selection = null;
    graphBuildActive();
    graphHideState();
    graphLegend();
    graphRearrange();
}

/* "Only this group" — the path to a group that isn't among the top
 * ten and therefore has no row in the legend. */
function graphOnlyThisGroup(nr) {
    GRAPH.selection = new Set([nr]);
    graphBuildActive();
    graphHideState();
    graphLegend();
    graphRearrange();
}

function graphClearSelection() {
    GRAPH.selection = null;
    graphBuildActive();
    graphHideState();
    graphLegend();
    graphRearrange();
}

/* What's currently hidden belongs visibly written down. A picture where
 * half is missing without that being stated is a trap. */
function graphHideState() {
    const el = document.getElementById('mapHideState');
    if (!el) return;
    if (!GRAPH.active || GRAPH.activeN === MAP.n) {
        el.textContent = '';
        el.style.display = 'none';
        return;
    }
    el.style.display = '';
    const gone = MAP.n - GRAPH.activeN;
    const parts = [`${gone.toLocaleString()} of `
                   + `${plural(MAP.n, 'voice')} hidden`];
    if (GRAPH.minGroup > 1) parts.push(`groups below ${GRAPH.minGroup}`);
    if (GRAPH.selection) parts.push(`only ${GRAPH.selection.size} `
        + (GRAPH.selection.size === 1 ? 'group' : 'groups') + ' chosen');
    el.innerHTML = parts.join(' · ')
        + ` <button class="map-mini" onclick="graphUnhide()">show all</button>`;
}

function graphUnhide() {
    GRAPH.selection = null;
    GRAPH.minGroup = 1;
    const field_el = document.getElementById('mapMinGroup');
    if (field_el) field_el.value = 1;
    graphSaveSettings();
    graphBuildActive();
    graphHideState();
    graphLegend();
    graphRearrange();
}

function graphSetSlider() {
    graphLoadSettings();
    const dr = document.getElementById('mapDrag');
    if (dr) dr.value = GRAPH.drag;
    const dv = document.getElementById('mapDragValue');
    if (dv) dv.textContent = GRAPH.drag.toFixed(1) + '×';
    const sc = document.getElementById('mapSteps');
    if (sc) sc.value = GRAPH.steps;
    const ol = document.getElementById('mapOutlines');
    if (ol) ol.checked = GRAPH.showOutlines;
    const mg = document.getElementById('mapMinGroup');
    if (mg) mg.value = GRAPH.minGroup;
    const cl = document.getElementById('mapCluster');
    if (cl) cl.checked = GRAPH.clusterNames;
    const di = document.getElementById('mapDensity');
    if (di) di.value = GRAPH.density;
    const dnv = document.getElementById('mapDensityValue');
    if (dnv) dnv.textContent = GRAPH.density.toFixed(2) + '×';
}

/* ══════════════════════════════════════════════════════════════════════════
 *  TRASH
 *
 *  Speakers that aren't relevant can be hidden from the speaker map —
 *  trashed — and viewed on a second map.
 *
 *  Discarding happens tile by tile, but what's sent is the VOICES. The
 *  reason is the group number: it gets recreated on every run. Between
 *  the click and the request arriving, computation could already have
 *  happened at a different threshold — then a number would hit the wrong ones.
 *
 *  This is different from the existing hiding (`GRAPH.selection`,
 *  `minGroup`): that's a view for this session. The trash lives
 *  in the database and takes effect BEFORE grouping.
 * ══════════════════════════════════════════════════════════════════════════ */

/* The voices of a group — the same computation as in the workspace panel. */
function graphGroupVoices(nr) {
    const out = [];
    for (let i = 0; i < MAP.n; i++) {
        if (GRAPH.groupOf && GRAPH.groupOf[i] === nr) {
            const m = MAP.members[i];
            if (m && m.id != null) out.push(m.id);
        }
    }
    return out;
}

async function graphDiscardGroup(nr) {
    const ids = graphGroupVoices(nr);
    if (!ids.length) return;
    const name = graphGroupName ? graphGroupName(nr) : '';
    const was = name ? `"${name}" (group ${nr})` : `Group ${nr}`;
    /* No "are you sure?" any more: the History undoes it, and the
     * toast offers the undo right away. */
    const d = await graphTrashRequest('/api/voicemap/trash', {ids});
    if (d) vxUndoToast(d.decision, `${was}: ${vxN(ids.length)} voice${ids.length === 1 ? '' : 's'} discarded`);
}

async function graphTrashBack() {
    if (!confirm(`Restore ${GRAPH.trashCount === 1 ? 'the voice' : `all ${vxN(GRAPH.trashCount)} voices`} from the trash?\n\n`
                 + 'The History can undo this, too.')) return;
    const d = await graphTrashRequest('/api/voicemap/trash/restore', {});
    // Failed: stay in the trash and keep the failure line on screen
    // (graphTrashRequest reported it) instead of "Computing graph …".
    if (!d) return;
    GRAPH.trash = false;      // the trash is empty, nothing left here
    graphTrashBar();
    vxUndoToast(d.decision, 'The trash was emptied into the library');
    mapLoadGraph();
}

async function graphTrashRequest(path, body) {
    mapReport('Writing to trash …');
    let d;
    try {
        const r = await fetch(path, {
            method: 'POST', headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(body)});
        d = await r.json();
        if (!r.ok) throw new Error(d.error || r.status);
        GRAPH.trashCount = d.count ?? GRAPH.trashCount;
    } catch (e) {
        mapReport(`Trash: ${String(e)}`);
        return null;
    }
    // Reload, because the INVENTORY changed — not just the picture.
    if (typeof VX !== 'undefined') VX.people = null;
    if (typeof historyRefresh === 'function') historyRefresh();
    mapLoadGraph();
    graphTrashBar();
    return d;
}

/* Switch between the library and the trash. The SAME map with a different
 * section — two drawing routines would be two truths. */
function graphTrashView(on) {
    GRAPH.trash = !!on;
    GRAPH.selection = null;          // a selection from the other map doesn't apply here
    graphTrashBar();
    mapLoadGraph();
}

/* Where the map is — library or trash — must be visible at all times:
 * two maps that look the same and show different things are a trap (you
 * name a voice in the trash and wonder why it doesn't show up). The
 * switch in the toolbar says it, and in the trash a chip says it again. */
function graphTrashBar() {
    const n = GRAPH.trashCount || 0;
    const cnt = document.getElementById('mapTrashCount');
    if (cnt) cnt.textContent = n ? vxN(n) : '';
    document.getElementById('mapSrcLibrary')?.classList.toggle('active', !GRAPH.trash);
    document.getElementById('mapSrcTrash')?.classList.toggle('active', !!GRAPH.trash);
    const el = document.getElementById('mapTrashBar');
    if (!el) return;
    if (GRAPH.trash) {
        el.style.display = '';
        el.innerHTML = `<b>Trash</b> · ${vxN(n)} discarded ${n === 1 ? 'voice' : 'voices'}`
            + ` — restore a group from its details, a click from the History, or`
            + ` <button class="map-mini" onclick="graphTrashBack()">everything</button>`;
    } else {
        el.style.display = 'none';
        el.innerHTML = '';
    }
}

/* Restore a group from the trash back into the library. Only visible there
 * — in the library there'd be nothing to restore. */
async function graphRestoreGroup(nr) {
    const ids = graphGroupVoices(nr);
    if (!ids.length) return;
    const d = await graphTrashRequest('/api/voicemap/trash/restore', {ids});
    if (d) vxUndoToast(d.decision, `${vxN(ids.length)} voice${ids.length === 1 ? '' : 's'} restored`);
}
