/**
 * MiniMax References Manager — native widgets (openrouter_api_key, model) stay on top,
 * untouched; everything custom lives in one DOM block BELOW them. Per the visual
 * spec the DOM/canvas split is VISIBLE: no wrapping panel, buttons and prompt sit
 * directly on ComfyUI's grey node body, and the canvas is a BLACK slab that
 * announces "this is the reference area":
 *   [⬆ Image] [⬆ Video] [⬆ Audio] [Save config] [Load config] · · · · · · · [⚙]
 *   <canvas> — black; Images (N/9) / Videos (N/3) / Audio (N/3), three fixed
 *              sections separated by drawn divider lines, a "+" add slot per row
 *   Prompt — one large plain textarea (bound to the hidden `direction` widget)
 *   Original prompt — a second textarea (bound to hidden `original_prompt`) that
 *                     keeps the idea Auto Prompt rewrote away from `direction`
 * The ⚙ opens a modal for the `system_prompt` widget — never drawn on the node body,
 * only editable behind that modal (see openSystemPromptModal). `rewrite_pin` is a
 * native combo appended after `original_prompt` so a saved graph picks the rewriter
 * without pasting into ⚙. Named pins replace packaged six-section; a non-blank ⚙
 * `system_prompt` still wins.
 *
 * RESIZING: the node keeps the user's width and height, including workflow restores.
 * The prompt grows with the node; the toolbar wraps and the reference canvas scrolls
 * horizontally on narrow nodes. The three reference rows keep their tile geometry
 * whether they hold 0 or 9 items, so painting and hit testing share one coordinate
 * system. lib/layout.js supplies the default size and minimum content dimensions.
 *
 * RENDERING: the reference rows are a single <canvas>, not per-cell DOM — the LTX
 * Director hybrid (ltx_director.js draws its timeline onto one canvas at :2703 but
 * keeps buttons and the prompt as real DOM). Per-cell DOM kept producing
 * overlap/leakage bugs on this frontend; one drawn surface has nothing to overlap.
 * Everything inside the rows — headers, thumbnails, tag badges, delete, the
 * soundtrack toggle, the add slots, play glyphs, selection — is painted by draw(),
 * the single entry point. No requestAnimationFrame loop: scheduleDraw() coalesces
 * change events (refs, selection, thumb load, probe answer, preview state, the
 * canvas's one real mount via a ResizeObserver) into at most one paint per frame,
 * and draw() never calls setSize/computeSize, so paint can't recurse into layout.
 *
 * PLAYABLE PREVIEWS: click the drawn ▶ on a video or audio tile to play it, click
 * again (or click the playing video) to stop. ONE shared <video> overlay and ONE
 * shared <audio> element per node — never per-cell DOM. The overlay is a child of
 * the gallery, so ComfyUI's zoom transform applies to it automatically; positioning
 * is plain CSS-px math off the tile rect. Clicking a tile anywhere OUTSIDE the ▶
 * selects it, exactly as before.
 *
 * TILE REORDER: the tiles are painted, not DOM, so swapping them is a pointer
 * drag on the canvas rather than HTML5 drag (that path stays file drops onto the
 * node). Mousedown on the tile well — not a chip — and drag past a neighbour's
 * midpoint; the two swap and the MiniMax tags follow slot order, so Picture 1
 * dragged onto Picture 2 becomes Picture 2. Tiles stay in their own row. The
 * chips (delete, edit, play, soundtrack, audio-only) keep firing on mousedown
 * and never start a reorder. The array math is lib/references.js's moveRefInKind and
 * slotIndexAtX.
 *
 * HIDING `references_json`, `direction` and `system_prompt`: mirrors WhatDreamsCost
 * multi_image_loader.js's hide-widget pattern (:122-146) — a plain `w.hidden = true`
 * gets silently overwritten on V3 (vueNodesMode), which re-derives widget visibility
 * from `hidden`/`type` on every reactive redraw, and a widget left with its native
 * computeSize reserves real layout height: three stacked, that read as a thin bar
 * overlapping our controls. The fix: Object.defineProperty-lock `hidden`/`type` as
 * always-on getters V3's own writes can't override, set computeSize = [0,0]
 * unconditionally, and poll every 50ms for 1s for `w.element` — V3 creates the DOM
 * element backing a multiline STRING widget ASYNCHRONOUSLY, so a one-shot
 * display:none misses it. Every property hideWidget() touches is try/catch-wrapped:
 * a prior version of this file assigned `domWidget.node = node` directly, which is a
 * getter-only accessor on the V3 frontend's BaseWidget and threw a TypeError that
 * aborted loading ANY workflow containing this node — fatal, not cosmetic. That
 * assignment is gone (the node is captured via closure instead); hideWidget()'s own
 * property writes are defensive against the same class of bug.
 *
 * Each video tile carries a soundtrack MUTE (bottom-left: ♪ on a light chip when on,
 * a struck-through speaker when muted, plus a "muted" badge line) — it's the only
 * thing that routes that video's audio to its index-matched video_audio_N output, so
 * it's real functionality, not decoration. The edit modal repeats it as a worded
 * "Soundtrack: on/muted" button. Gated on GET /minimax_refpack/probe?file=
 * (media.py's has_audio — a DECODABLE track): dim/pending until the probe answers,
 * permanently disabled for a clip with no audio track. See probeHasAudio()/
 * syncProbes() and drawTile().
 *
 * USE AUDIO ONLY: the mirror chip (bottom-right, speaker + out-arrow) and the edit
 * modal's "Audio only" button replace a video reference with an AUDIO reference to
 * the same file (lib/references.js's videoToAudioOnly): <Audio n>, no <Video k>,
 * trim and audio roles kept. The Audio row's upload/picker accept video files too;
 * such a tile says "video soundtrack" (or "video · no audio track", with the danger
 * border) on its badge. Canvas hover titles name every chip's action (hitTitle).
 *
 * NOT SURFACED ON THE NODE BODY: the system prompt itself — lives only behind the ⚙
 * modal (openSystemPromptModal), never drawn on the body.
 *
 * CROP + TRIM (openEditModal): one modal edits both. Entry points: the scissors chip
 * in a tile's TOP-LEFT corner (the only free corner — delete owns top-right, ♪ sits
 * bottom-left above the badge, ▶ is centred) and a double-click anywhere on the tile.
 * The modal shows the REAL media via /view (fileUrl) — an <img> for stills, a <video>
 * for clips — with a draggable fraction-space crop rect + corner handles and aspect
 * presets; video/audio get in/out trim handles on a bar plus 2dp second fields. Each
 * row ends in a right-aligned Clear button ("Clear crop" / "Clear trim") that resets
 * that edit — dimmed when there is nothing to clear, so the modal states at a glance
 * whether the reference carries an edit; clearing the crop also releases the aspect
 * lock and unhighlights every preset. No preset is highlighted on open — the highlight
 * means the user chose a lock, not the free default. Save
 * writes crop ([x,y,w,h] fractions) / trim ([start,end] seconds) onto the reference —
 * refs.py validates and media.py applies them — and drops the file's thumbCache entry
 * so the tile redraws from the thumb route with the edit baked in. The scissors chip
 * inverts to a light chip while an edit is set (same signalling as ♪), the badge grows
 * a third line ("2.00-6.50s · cropped"), and the click-to-play preview plays only the
 * trimmed span (seek to the in-point, timeupdate stops it at the out-point). All of it
 * keeps the module invariants: no per-cell DOM, one shared <video>/<audio> per node,
 * scheduleDraw coalescing, no requestAnimationFrame loop.
 *
 * MODULES: this file is the ComfyUI extension — it imports app/api, owns every piece
 * of module state (thumb and probe caches, liveNodes), draws, builds the DOM and
 * registers the node. Logic that tests run under node lives in lib/*.js. Those modules
 * import nothing from ComfyUI, touch no DOM at load time and keep no module state:
 * ComfyUI also loads each of them as an extension in its own right, and node must be
 * able to import them bare. What they need from here arrives as parameters
 * (createControlUi's host, applyConfig's, installInputControls').
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { applyConfig, buildConfig, configFilename, parseConfigFile } from "./lib/config.js";
import { ensureControlOutput, invalidateControlOperation, syncControlsButton } from "./lib/control.js";
import { createControlUi } from "./lib/control_ui.js";
import { DROP_KINDS, bucketDroppedFiles, isVideoFile } from "./lib/drop.js";
import { createEditUi } from "./lib/edit_ui.js";
import { editSummary, hasEdit } from "./lib/edits.js";
import { installInputControls } from "./lib/inputs.js";
import {
    CANVAS_ROWS, CAPS, CL, CONTENT, KINDS, SECTION_LABEL,
    addBlockWidget, defaultSize, installSizeGuards, nodeSize,
} from "./lib/layout.js";
import { mdebug, mlog, mwarn } from "./lib/log.js";
import { appendFileOptions, createPlannerModal } from "./lib/modal.js";
import {
    ORDER_CURRENT, detectLayout, migrateMatchValue, migrateProviderValue, migrateProviderWidget,
    remapWidgetValues,
} from "./lib/migrate.js";
import {
    assignTags, cloneRefs, deriveTaskPlanState, emptyRefs, fromReferencesEnvelope, moveRefInKind,
    newReference, slotIndexAtX, toReferencesEnvelope, toReferencesList, videoToAudioOnly,
} from "./lib/references.js";
import { createTaskPlanUi } from "./lib/task_plan_ui.js";
import { PROVIDER_FIELDS, setWidget, setWidgetVisible, widgetByName } from "./lib/widgets.js";

const NODE_NAME = "MiniMaxH3ReferencePack";

// LTX Director's flat neutral palette (WhatDreamsCost js/ltx_director.js), counted
// out of that file, not invented here. The canvas reads these directly; refpack.css
// carries the DOM half (buttons, prompt box, modal). The wells are nudged up from
// the reference's #111/#121212: those values assume a #1e1e1e panel behind them,
// and this canvas is BLACK — at #111 a tile well is 17/255 off its background,
// structurally present but invisible (same class of contrast bug as the prompt box).
const C = {
    well: "#1a1a1a",
    wellDeep: "#141414",
    surface: "#2a2a2a",
    raised: "#333",
    border: "#444",
    text: "#e0e0e0",
    textMuted: "#aaa",
    textFaint: "#888",
    textDim: "#666",
    danger: "#ff4444",
};

// ---------------------------------------------------------------------------
// API helpers
// ---------------------------------------------------------------------------

async function apiUpload(file, subfolder = "") {
    const form = new FormData();
    form.append("image", file);
    form.append("type", "input");
    form.append("overwrite", "true");
    if (subfolder) form.append("subfolder", subfolder);
    const res = await fetch("/upload/image", { method: "POST", body: form });
    if (!res.ok) throw new Error(`upload failed: ${res.status}`);
    const info = await res.json();
    return info.subfolder ? `${info.subfolder}/${info.name}` : info.name;
}

async function apiListFiles(kind) {
    const res = await fetch(`/minimax_refpack/files?kind=${encodeURIComponent(kind)}`);
    if (!res.ok) throw new Error(`input listing failed: ${res.status}`);
    const data = await res.json();
    return Array.isArray(data.files) ? data.files.slice().sort((a, b) => a.localeCompare(b)) : [];
}

// The tile thumb: cropped by the route (media.thumbnail_png), and for a trimmed
// video taken at the in-point (`t=`) so the tile previews the span that will
// actually be emitted, not frame 0 of the untrimmed file.
function thumbUrl(file, ref) {
    let url = `/minimax_refpack/thumb?file=${encodeURIComponent(file)}`;
    if (ref && Array.isArray(ref.crop)) url += `&crop=${ref.crop.join(",")}`;
    if (ref && Array.isArray(ref.trim)) url += `&t=${ref.trim[0]}`;
    return url;
}

// Raw file, for the click-to-play previews (thumbUrl is a server-generated still).
// Stock ComfyUI route — confirmed at server.py:511 (`@routes.get("/view")`, no /api prefix).
function fileUrl(file) {
    return `/view?filename=${encodeURIComponent(file)}&type=input`;
}

// NOTE: there is no apiSavePack/apiLoadPack/apiListPacks any more. Configs are files
// on the user's own machine (download + file picker, see "Config save/load" below),
// so the /minimax_refpack/packs routes are no longer called from the UI at all.

async function apiSystemPromptDefault() {
    const res = await fetch("/minimax_refpack/system_prompt");
    if (!res.ok) throw new Error(`fetch failed: ${res.status}`);
    const data = await res.json();
    return data.default || "";
}

// Deliberately asks the SERVER to probe rather than probing from here. Two reasons, and
// the first is the one that matters: `localhost` has to mean whatever the ComfyUI process
// can reach, so a Dockerised or remote ComfyUI gets told the truth about its own network
// instead of about the machine this browser happens to be sitting on. The second is that
// a local LLM server has no reason to send CORS headers, so the browser could open the
// socket and still not be allowed to read the answer.
async function apiDetectServers(base) {
    const qs = base ? `?base=${encodeURIComponent(base)}` : "";
    const res = await fetch(`/minimax_refpack/detect${qs}`);
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.error || `detect failed: ${res.status}`);
    return data.servers || [];
}

// One probe per filename, cached module-wide and reused across every redraw (a
// naive per-draw fetch would hit this route constantly). probeResults holds the
// RESOLVED value so draw() — which is synchronous — can read it without awaiting:
// absent = still pending, true/false = probe's answer, null = probe itself failed
// (unknown, not "no audio").
const probeCache = new Map();
const probeResults = new Map();

function probeHasAudio(file) {
    if (!probeCache.has(file)) {
        probeCache.set(
            file,
            fetch(`/minimax_refpack/probe?file=${encodeURIComponent(file)}`)
                .then((res) => (res.ok ? res.json() : null))
                .then((data) => (data ? !!data.has_audio : null))
                .catch(() => null)
        );
    }
    return probeCache.get(file);
}

// Kick (or re-kick) probes for every current video and repaint when they answer.
// A saved use_soundtrack=true against a clip the probe says is SILENT (e.g. a config
// restored against a since-replaced file) is forced off — otherwise a stale <Audio N>
// tag points at nothing. Probe FAILURE (null) does not force off: "couldn't check"
// is not "no audio", so the user's saved intent survives a flaky probe.
function syncProbes(node) {
    for (const ref of node._mmrpRefs.videos) {
        const file = ref.file;
        probeHasAudio(file).then((hasAudio) => {
            probeResults.set(file, hasAudio);
            if (hasAudio === false) {
                const cur = node._mmrpRefs.videos.find((v) => v.file === file);
                if (cur && (cur.use_soundtrack || (cur.roles || []).some(
                    (role) => role === "audio_reuse" || role === "audio_reference"
                ))) {
                    const next = cloneRefs(node._mmrpRefs);
                    const target = next.videos.find((v) => v.file === file);
                    target.use_soundtrack = false;
                    target.roles = (target.roles || []).filter(
                        (role) => role !== "audio_reuse" && role !== "audio_reference"
                    );
                    applyRefs(node, next);
                    return; // applyRefs already repaints
                }
            }
            scheduleDraw(node);
        });
    }
    // Audio references that point at a video file: record the answer for the tile's
    // badge and warning border only. Never mutate the ref - build() raises a named
    // error for a clip with no decodable track, which is the louder signal.
    for (const ref of node._mmrpRefs.audios) {
        if (!isVideoFile(ref.file)) continue;
        const file = ref.file;
        probeHasAudio(file).then((hasAudio) => {
            probeResults.set(file, hasAudio);
            scheduleDraw(node);
        });
    }
}

// ---------------------------------------------------------------------------
// Styles
// ---------------------------------------------------------------------------

function injectStyles() {
    if (document.getElementById("mmrp-styles")) return;
    const link = document.createElement("link");
    link.id = "mmrp-styles";
    link.rel = "stylesheet";
    link.href = new URL("./refpack.css", import.meta.url).href;
    document.head.appendChild(link);
}

// ---------------------------------------------------------------------------
// Hidden-widget helper — adapted from WhatDreamsCost multi_image_loader.js's
// hide-widget pattern: defineProperty-lock hidden/type against V3's reactive
// redraws, force computeSize unconditionally, poll for the async-created DOM
// element. See the header comment for why every half of this is load-bearing.
// ---------------------------------------------------------------------------

// Returns the hide-poll interval id (or undefined if there was no widget) so the
// caller can clear it early on node removal.
function hideWidget(w) {
    if (!w) return undefined;
    // Every write here is try/catch-wrapped: some frontend versions define these as
    // getter-only reactive accessors (that's exactly what broke on `domWidget.node =`,
    // see the header comment) — degrade instead of throwing and aborting workflow load.
    try {
        Object.defineProperty(w, "hidden", { get: () => true, set: () => {} });
    } catch (_) {
        try {
            w.hidden = true;
        } catch (_) {}
    }
    try {
        Object.defineProperty(w, "type", { get: () => "hidden", set: () => {} });
    } catch (_) {}
    try {
        if (!w.options) w.options = {};
        w.options.hidden = true;
    } catch (_) {}
    // computeSize = [0,0] UNCONDITIONALLY. Gating this behind a "skip on vueNodesMode"
    // check (a previous bug) left these multiline STRING widgets reserving real layout
    // height on V3 — stacked, that's the thin bar that overlapped the upload row.
    try {
        w.computeSize = () => [0, 0];
    } catch (_) {}
    if (!window.LiteGraph || !window.LiteGraph.vueNodesMode) {
        try {
            w.draw = () => {};
        } catch (_) {}
    }
    try {
        if (w.element) w.element.style.display = "none";
    } catch (_) {}
    // V3 creates the DOM element backing a multiline STRING widget ASYNCHRONOUSLY — it
    // does not exist yet at this point in onNodeCreated, so the display:none above is a
    // no-op the first time through. Poll briefly to catch it once it appears ("Catch
    // for V3 delayed DOM rendering to ensure no stubborn inputs appear" — same as the
    // reference). Self-clears after 1s; the onRemoved wrapper also clears it early.
    const hideInterval = setInterval(() => {
        try {
            if (w.element) w.element.style.display = "none";
        } catch (_) {}
    }, 50);
    setTimeout(() => clearInterval(hideInterval), 1000);
    return hideInterval;
}

// ---------------------------------------------------------------------------
// Thumbnail cache — module-wide, keyed by filename, shared across node instances.
// The thumb route serves a PNG for images AND videos (media.py thumbnail_png), so
// both kinds draw the same way; audio tiles get a drawn speaker glyph instead.
// liveNodes tracks which nodes to repaint when a thumb lands.
// ---------------------------------------------------------------------------

const thumbCache = new Map(); // file -> { img, state: "loading"|"ok"|"error", tries, reqs, timer }
const liveNodes = new Set();

// The ControlNet guide (lib/control_ui.js), the task plan and input picker
// (lib/task_plan_ui.js) and the crop/trim editor (lib/edit_ui.js). The collaborators
// below are function declarations, so they are hoisted; liveNodes and probeResults
// are the only ones that must already exist, which is why this sits after them.
const { watchControlPreparation, openControlModal } = createControlUi({
    api, liveNodes, syncReferencesWidget, renderNodeBody,
    createPlannerModal, appendFileOptions, apiListFiles, apiUpload,
});
const { openAddReferenceModal, openTaskPlanModal } = createTaskPlanUi({
    createPlannerModal, appendFileOptions, apiListFiles, thumbUrl, dropThumb, probeResults,
    applyRefs,
});
const { openEditModal } = createEditUi({
    fileUrl, probeResults, dropThumb, toggleSoundtrack, convertToAudioOnly, applyRefs,
});

// A thumb request fails for reasons that have nothing to do with the file: thumb_route
// decodes SYNCHRONOUSLY on the aiohttp loop, so while ComfyUI is still starting up (model
// loads, Manager's registry fetch) the request stalls and whatever proxy sits in front of
// the pod kills it. The first version cached that <img> error forever — one unlucky
// request and the tile read "no preview" until the tab was reloaded, with the file sitting
// happily in input/ the whole time. So: back off and retry, and only claim "no preview"
// once the ladder is exhausted.
const THUMB_RETRY_MS = [1000, 2000, 4000, 8000, 15000];

function repaintLive() {
    for (const n of liveNodes) scheduleDraw(n);
}

// `reqs` is monotonic and only feeds the cache-buster — a retry must not be answered from
// whatever cached the failure. The first request stays clean so it can be cached normally.
// `entry.base` carries the crop/trim query the entry was created with; an edit drops the
// whole entry (dropThumb) rather than mutating it.
function requestThumb(entry) {
    entry.timer = null;
    entry.reqs += 1;
    entry.img.src = entry.reqs === 1 ? entry.base : `${entry.base}&retry=${entry.reqs - 1}`;
}

function getThumb(file, ref) {
    let entry = thumbCache.get(file);
    if (!entry) {
        const img = new Image();
        entry = { img, file, base: thumbUrl(file, ref), state: "loading", tries: 0, reqs: 0, timer: null };
        img.onload = () => {
            entry.state = "ok";
            entry.tries = 0;
            repaintLive();
        };
        img.onerror = () => {
            const delay = THUMB_RETRY_MS[entry.tries];
            entry.tries += 1;
            if (delay === undefined) {
                // Ladder spent (~30s). Say so on the tile — but the entry stays retryable,
                // see retryFailedThumbs().
                entry.state = "error";
                mwarn("thumb_failed", { file: entry.file, tries: entry.tries - 1 });
                repaintLive();
                return;
            }
            // Still trying: the tile shows the plain well, not a verdict it may have to
            // take back a second later.
            entry.state = "loading";
            mdebug("thumb_retry", { file: entry.file, attempt: entry.tries, in_ms: delay });
            entry.timer = setTimeout(() => requestThumb(entry), delay);
        };
        thumbCache.set(file, entry);
        requestThumb(entry);
    }
    return entry;
}

// Saving an edit invalidates the file's cached thumb so the next draw re-fetches
// through the route with the new crop/t baked into the URL.
function dropThumb(file) {
    const entry = thumbCache.get(file);
    if (!entry) return;
    if (entry.timer) clearTimeout(entry.timer);
    entry.img.onload = null;
    entry.img.onerror = null;
    thumbCache.delete(file);
}

// Giving up is not giving up for good. Anything that says the world may have changed — the
// tab coming back to the front, the browser going back online — puts every failed thumb
// back in flight, so a pod that took minutes to finish booting recovers on its own instead
// of demanding a page reload. Cheap: entries that loaded fine are skipped, and nothing
// here polls.
function retryFailedThumbs() {
    let any = false;
    for (const [file, entry] of thumbCache) {
        if (entry.state !== "error") continue;
        entry.state = "loading";
        entry.tries = 0;
        requestThumb(entry);
        any = true;
    }
    if (any) repaintLive();
}

window.addEventListener("focus", retryFailedThumbs);
window.addEventListener("online", retryFailedThumbs);

// The single repaint funnel: every change (refs, selection, thumb load, probe
// answer, preview toggle, the canvas's one real mount) lands here, coalesced to one
// draw() per frame. NOT a permanent rAF loop — nothing re-queues unless something
// changes again.
function scheduleDraw(node) {
    if (node._mmrpDrawQueued) return;
    node._mmrpDrawQueued = true;
    requestAnimationFrame(() => {
        node._mmrpDrawQueued = false;
        draw(node);
    });
}

// ---------------------------------------------------------------------------
// Drawing. All geometry in CSS px; hit regions are recorded during draw() in the
// same space getMousePos() reports, so hit-testing is a point-in-rect scan over
// what was actually painted last.
// ---------------------------------------------------------------------------

function pathRoundRect(ctx, x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
}

// object-fit: cover, in canvas terms — crop the source centrally to the tile's
// aspect instead of squashing it.
function drawCover(ctx, img, x, y, w, h) {
    const iw = img.naturalWidth;
    const ih = img.naturalHeight;
    if (!iw || !ih) return;
    const s = Math.max(w / iw, h / ih);
    const sw = w / s;
    const sh = h / s;
    ctx.drawImage(img, (iw - sw) / 2, (ih - sh) / 2, sw, sh, x, y, w, h);
}

// Monochrome vector speaker for audio tiles — a fillText emoji would render in
// color and break the greyscale palette.
function drawSpeaker(ctx, cx, cy, s, color) {
    ctx.save();
    ctx.fillStyle = color;
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(cx - s * 0.7, cy - s * 0.28);
    ctx.lineTo(cx - s * 0.3, cy - s * 0.28);
    ctx.lineTo(cx + s * 0.1, cy - s * 0.65);
    ctx.lineTo(cx + s * 0.1, cy + s * 0.65);
    ctx.lineTo(cx - s * 0.3, cy + s * 0.28);
    ctx.lineTo(cx - s * 0.7, cy + s * 0.28);
    ctx.closePath();
    ctx.fill();
    ctx.beginPath();
    ctx.arc(cx + s * 0.25, cy, s * 0.45, -Math.PI / 3, Math.PI / 3);
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(cx + s * 0.25, cy, s * 0.78, -Math.PI / 3, Math.PI / 3);
    ctx.stroke();
    ctx.restore();
}

// The click-to-play affordance: dark circle, drawn ▶ or ❚❚ (vector, no emoji).
function drawPlayGlyph(ctx, cx, cy, playing) {
    ctx.fillStyle = "rgba(0, 0, 0, 0.55)";
    ctx.beginPath();
    ctx.arc(cx, cy, CL.playR, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = C.text;
    if (playing) {
        ctx.fillRect(cx - 6, cy - 6.5, 4.5, 13);
        ctx.fillRect(cx + 1.5, cy - 6.5, 4.5, 13);
    } else {
        ctx.beginPath();
        ctx.moveTo(cx - 4.5, cy - 7);
        ctx.lineTo(cx + 8, cy);
        ctx.lineTo(cx - 4.5, cy + 7);
        ctx.closePath();
        ctx.fill();
    }
}

// The crop/trim entry point: a scissors chip in the tile's top-left — the same quiet
// dark-circle family as the delete chip (vector, no emoji). Inverts to a light chip
// while the reference carries an edit, mirroring how ♪ signals on/off.
function drawEditChip(ctx, x, y, edited) {
    const dR = CL.del / 2;
    const cx = x + dR + 3;
    const cy = y + dR + 3;
    ctx.fillStyle = edited ? C.text : "rgba(0, 0, 0, 0.65)";
    ctx.beginPath();
    ctx.arc(cx, cy, dR, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = edited ? "#000" : "#555";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.arc(cx, cy, dR - 0.5, 0, Math.PI * 2);
    ctx.stroke();
    // Scissors: two crossing blades up, two finger loops down.
    const g = edited ? "#000" : "#fff";
    ctx.strokeStyle = g;
    ctx.lineWidth = 1.3;
    ctx.beginPath();
    ctx.moveTo(cx - 2.2, cy + 1.6);
    ctx.lineTo(cx + 4, cy - 4.2);
    ctx.moveTo(cx + 2.2, cy + 1.6);
    ctx.lineTo(cx - 4, cy - 4.2);
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(cx - 3.1, cy + 3.1, 1.7, 0, Math.PI * 2);
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(cx + 3.1, cy + 3.1, 1.7, 0, Math.PI * 2);
    ctx.stroke();
}

// One tile: well, thumbnail/glyph, tag badge, delete, edit chip, soundtrack state,
// play glyph, missing treatment, selection stroke. `lines` is the pre-computed badge
// text (up to three: tag, a video's <Audio N> or "muted" / an audio tile's source note,
// the crop/trim summary); `playState` is null | "play" | "pause" (null = no preview
// affordance: images, missing refs). `extra.audioOnly` is null | "on" | "dim" (the
// video tile's "use audio only" chip); `extra.danger` paints the warning border.
function drawTile(ctx, kind, ref, lines, x, y, selected, soundState, badgeH, playState, extra = {}) {
    const T = CL.tile;

    ctx.save();
    pathRoundRect(ctx, x, y, T, T, 4);
    ctx.clip();

    ctx.fillStyle = kind === "audio" ? C.surface : C.well;
    ctx.fillRect(x, y, T, T);

    if (kind === "audio") {
        // Speaker in the upper half; the play glyph takes the lower half.
        drawSpeaker(ctx, x + T / 2, y + 34, 18, ref.missing ? C.textDim : C.textFaint);
    } else {
        const entry = getThumb(ref.file, ref);
        if (entry.state === "ok") {
            if (ref.missing) ctx.globalAlpha = 0.6;
            drawCover(ctx, entry.img, x, y, T, T);
            ctx.globalAlpha = 1;
        } else if (entry.state === "error" && !ref.missing) {
            ctx.fillStyle = C.textDim;
            ctx.font = "11px sans-serif";
            ctx.textAlign = "center";
            ctx.fillText("no preview", x + T / 2, y + T / 2);
        }
    }

    if (ref.missing) {
        ctx.fillStyle = C.danger;
        ctx.font = "bold 11px sans-serif";
        ctx.textAlign = "center";
        ctx.fillText("missing", x + T / 2, y + T - badgeH - 8);
    }

    if (playState) {
        // Video: slightly above center to clear the badge; audio: lower half.
        // Audio at y+72, not lower: a three-line badge (badge3 = 42) would cover it.
        drawPlayGlyph(ctx, x + T / 2, kind === "audio" ? y + 72 : y + 58, playState === "pause");
    }

    // Tag badge — drawn text on a translucent strip pinned to the tile's bottom.
    // Deliberately dark: it always overlays a thumbnail or a lightened well, never
    // the black ground, and #e0e0e0 text carries it.
    ctx.fillStyle = "rgba(0, 0, 0, 0.7)";
    ctx.fillRect(x, y + T - badgeH, T, badgeH);
    ctx.fillStyle = C.text;
    ctx.font = "11px sans-serif";
    ctx.textAlign = "center";
    lines.forEach((line, i) => {
        ctx.fillText(line, x + T / 2, y + T - badgeH + 13 + i * 13);
    });

    // Delete affordance — always visible (canvas has no cheap hover), but QUIET:
    // a dark chip, not a red square. Red is the node's one accent and it belongs
    // to selection, not to a destructive secondary action (UI review #5).
    const dR = CL.del / 2;
    const dcx = x + T - dR - 3;
    const dcy = y + dR + 3;
    ctx.fillStyle = "rgba(0, 0, 0, 0.65)";
    ctx.beginPath();
    ctx.arc(dcx, dcy, dR, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = "#555";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.arc(dcx, dcy, dR - 0.5, 0, Math.PI * 2);
    ctx.stroke();
    ctx.strokeStyle = "#fff";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(dcx - 4, dcy - 4);
    ctx.lineTo(dcx + 4, dcy + 4);
    ctx.moveTo(dcx + 4, dcy - 4);
    ctx.lineTo(dcx - 4, dcy + 4);
    ctx.stroke();

    // Crop/trim entry point — top-left, the one corner nothing else owns. A missing
    // file has no media to show in the editor, so it gets no chip.
    if (!ref.missing) drawEditChip(ctx, x, y, hasEdit(ref));

    if (soundState) {
        const S = CL.sound;
        const sy = y + T - badgeH - S - 3;
        const on = soundState === "on";
        // FULLY OPAQUE chip (LEDGER "see-through music icon"): the old fills were
        // rgba(255,255,255,0.15) when on / rgba(0,0,0,0.65) otherwise, compositing
        // the thumbnail through — on a light or busy clip the toggle all but
        // vanished. Solid fills only, with a border to lift the chip's edge off a
        // dark thumbnail; "on" inverts to a light chip with a black glyph so the
        // two states can't be confused. Same rect, same radius — geometry untouched.
        ctx.fillStyle = on ? C.text : "#111";
        pathRoundRect(ctx, x + 3, sy, S, S, 4);
        ctx.fill();
        ctx.strokeStyle = on ? "#000" : "#555";
        ctx.lineWidth = 1;
        pathRoundRect(ctx, x + 3.5, sy + 0.5, S - 1, S - 1, 4);
        ctx.stroke();
        if (soundState === "off") {
            // MUTED: a struck-through speaker, so "off" reads as a mute rather than as
            // a dim note. The strike is brighter than the no-track state's below.
            drawSpeaker(ctx, x + 3 + S / 2 - 1, sy + S / 2, 9, C.textMuted);
            ctx.strokeStyle = C.text;
            ctx.lineWidth = 1.8;
            ctx.beginPath();
            ctx.moveTo(x + 8, sy + S - 6);
            ctx.lineTo(x + S - 2, sy + 6);
            ctx.stroke();
        } else {
            ctx.fillStyle =
                soundState === "pending" ? "#555" : soundState === "none" ? "#444" : on ? "#000" : C.textMuted;
            ctx.font = "16px sans-serif";
            ctx.textAlign = "center";
            ctx.fillText("♪", x + 3 + S / 2, sy + 20);
        }
        if (soundState === "none") {
            // Struck through: this clip has no audio track, permanently disabled.
            ctx.strokeStyle = "#555";
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            ctx.moveTo(x + 7, sy + S - 5);
            ctx.lineTo(x + S - 1, sy + 5);
            ctx.stroke();
        }
    }

    if (extra.audioOnly) {
        // "Use audio only" - bottom-right, the ♪ chip's mirror. A speaker with an
        // out-arrow: take the sound, leave the picture.
        const S = CL.sound;
        const ax = x + T - S - 3;
        const sy = y + T - badgeH - S - 3;
        const live = extra.audioOnly === "on";
        ctx.fillStyle = "#111";
        pathRoundRect(ctx, ax, sy, S, S, 4);
        ctx.fill();
        ctx.strokeStyle = "#555";
        ctx.lineWidth = 1;
        pathRoundRect(ctx, ax + 0.5, sy + 0.5, S - 1, S - 1, 4);
        ctx.stroke();
        const g = live ? C.text : "#555";
        drawSpeaker(ctx, ax + S / 2 - 3, sy + S / 2 + 2, 7, g);
        ctx.strokeStyle = g;
        ctx.lineWidth = 1.4;
        ctx.beginPath();
        ctx.moveTo(ax + S - 12, sy + 11);
        ctx.lineTo(ax + S - 6, sy + 5);
        ctx.moveTo(ax + S - 10, sy + 5);
        ctx.lineTo(ax + S - 6, sy + 5);
        ctx.lineTo(ax + S - 6, sy + 9);
        ctx.stroke();
    }

    ctx.restore();

    ctx.strokeStyle = ref.missing || extra.danger ? C.danger : C.border;
    ctx.lineWidth = 1;
    pathRoundRect(ctx, x + 0.5, y + 0.5, T - 1, T - 1, 4);
    ctx.stroke();

    if (selected) {
        // The one accent in the whole node: LTX-red rectangle around the tile.
        ctx.strokeStyle = C.danger;
        ctx.lineWidth = 3;
        pathRoundRect(ctx, x - 2.5, y - 2.5, T + 5, T + 5, 7);
        ctx.stroke();
    }
}

// Vector upload arrow — same family as the toolbar's SVG icon (shaft, chevron
// head, tray) so the two read as one set. Drawn, not text; no emoji.
function drawUploadGlyph(ctx, cx, cy, s, color) {
    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.beginPath();
    ctx.moveTo(cx, cy + s * 0.15);
    ctx.lineTo(cx, cy - s * 0.5);
    ctx.moveTo(cx - s * 0.3, cy - s * 0.2);
    ctx.lineTo(cx, cy - s * 0.5);
    ctx.lineTo(cx + s * 0.3, cy - s * 0.2);
    ctx.moveTo(cx - s * 0.45, cy + s * 0.3);
    ctx.lineTo(cx - s * 0.45, cy + s * 0.5);
    ctx.lineTo(cx + s * 0.45, cy + s * 0.5);
    ctx.lineTo(cx + s * 0.45, cy + s * 0.3);
    ctx.stroke();
    ctx.restore();
}

// The per-row add affordance (UI review #1): an icon-only square that opens that
// row's file picker. Deliberately NOT the toolbar-button treatment — a toolbar
// clone sitting on the media surface read wrong; this is a quiet well with an
// upload glyph. Drawn dimmed (not removed) at cap: a control that vanishes
// teaches nothing, a dimmed one shows the row is full (UI review #4).
function drawAddSquare(ctx, x, y, dimmed) {
    const S = CL.addBtn;
    ctx.save();
    if (dimmed) ctx.globalAlpha = 0.4;
    ctx.fillStyle = C.wellDeep;
    pathRoundRect(ctx, x + 0.5, y + 0.5, S - 1, S - 1, 6);
    ctx.fill();
    ctx.strokeStyle = C.raised;
    ctx.lineWidth = 1;
    pathRoundRect(ctx, x + 0.5, y + 0.5, S - 1, S - 1, 6);
    ctx.stroke();
    drawUploadGlyph(ctx, x + S / 2, y + S / 2, 18, C.textFaint);
    ctx.restore();
}

function draw(node) {
    const body = node._mmrpBody;
    if (!body) return;
    const canvas = body.canvas;
    const ctx = body.ctx;
    const cssW = canvas.clientWidth;
    const cssH = canvas.clientHeight;
    // Not attached / zero-sized yet (V3 mounts the DOM widget asynchronously) —
    // the ResizeObserver fires once real dimensions arrive and repaints then.
    if (!cssW || !cssH) return;

    // HiDPI: back the canvas at devicePixelRatio and scale the context so all
    // layout math (and hit regions) stay in CSS px while text/thumbs stay sharp.
    const dpr = window.devicePixelRatio || 1;
    const bw = Math.round(cssW * dpr);
    const bh = Math.round(cssH * dpr);
    if (canvas.width !== bw || canvas.height !== bh) {
        canvas.width = bw;
        canvas.height = bh;
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cssW, cssH);

    // The slab itself: the CSS keeps the element black; a subtle drawn edge marks
    // it off against the grey node body without the border-vs-content-box coordinate
    // mismatch a CSS border would introduce into hit-testing.
    ctx.strokeStyle = C.raised;
    ctx.lineWidth = 1;
    pathRoundRect(ctx, 0.5, 0.5, cssW - 1, cssH - 1, 6);
    ctx.stroke();

    const refs = node._mmrpRefs;
    const tagged = assignTags(refs);
    const viewW = cssW - CL.x0 * 2;
    const regions = [];
    node._mmrpHit = { regions };
    const playing = node._mmrpPlaying;

    for (const row of CANVAS_ROWS.rows) {
        const kind = row.kind;
        const arr = refs[`${kind}s`];
        const taggedArr = tagged[`${kind}s`];
        const atCap = arr.length >= CAPS[kind];
        const tileY = row.stripY + CL.stripPad;

        // Divider at the midpoint of the inter-section gap — per spec: a viewer
        // should never be unsure which section a tile belongs to.
        if (row.y > CL.padTop) {
            const dy = Math.round(row.y - CL.rowGap / 2) + 0.5;
            ctx.strokeStyle = "#2a2a2a";
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(CL.x0, dy);
            ctx.lineTo(CL.x0 + viewW, dy);
            ctx.stroke();
        }

        // The labels are the only wayfinding on the slab — #aaa/12px, not the
        // quietest text in the node (UI review #7). At cap they brighten further.
        ctx.fillStyle = atCap ? C.text : C.textMuted;
        ctx.font = "12px sans-serif";
        ctx.textAlign = "left";
        ctx.fillText(`${SECTION_LABEL[kind]} (${arr.length}/${CAPS[kind]})`, CL.x0, row.y + 12);

        arr.forEach((ref, i) => {
            const tx = CL.x0 + i * (CL.tile + CL.gap);
            const t = taggedArr[i];
            // "vid_audio <Audio 1>" not bare "<Audio 1>": the tag is what MiniMax binds
            // and has to stay visible, but unlabelled it reads as a standalone audio ref.
            const upstream = node.inputs?.some((input) => input.name === `${kind}_${i + 1}` && input.link != null);
            const lines = kind === "video" && t.audioTag
                ? [t.tag, `vid_audio ${t.audioTag}`]
                : [t.tag];
            if (upstream) lines.push("upstream input");
            const probed = probeResults.has(ref.file) ? probeResults.get(ref.file) : undefined;
            // A muted clip that HAS sound says so: the ♪ chip alone did not read as mute.
            if (kind === "video" && !ref.use_soundtrack && probed === true) lines.push("muted");
            // An audio reference to a video file names its source - exactly one line.
            const videoAudio = kind === "audio" && isVideoFile(ref.file);
            if (videoAudio) lines.push(probed === false ? "video · no audio track" : "video soundtrack");
            // An edited reference states its edit on the badge: "2.00-6.50s · cropped".
            const summary = editSummary(ref);
            if (summary) lines.push(summary);
            lines.length = Math.min(lines.length, 3); // three badge heights exist, no fourth
            const badgeH = [CL.badge1, CL.badge2, CL.badge3][lines.length - 1];
            const selected =
                node._mmrpSelected && node._mmrpSelected.kind === kind && node._mmrpSelected.index === i;

            let soundState = null;
            if (kind === "video") {
                const res = probeResults.has(ref.file) ? probeResults.get(ref.file) : undefined;
                soundState =
                    res === undefined
                        ? "pending"
                        : res === true
                          ? ref.use_soundtrack
                              ? "on"
                              : "off"
                          : res === false
                            ? "none"
                            : "unknown";
            }

            let playState = null;
            if (kind !== "image" && !ref.missing) {
                playState =
                    playing && playing.kind === kind && playing.file === ref.file ? "pause" : "play";
            }

            // Audio-only chip: live only once the probe says there is sound to take and
            // the audio row has room; dimmed (and not clickable) otherwise.
            let audioOnly = null;
            if (kind === "video") {
                const room = node._mmrpRefs.audios.length < CAPS.audio;
                audioOnly = !ref.missing && probed === true && room ? "on" : "dim";
            }
            const danger = videoAudio && probed === false;

            drawTile(ctx, kind, ref, lines, tx, tileY, selected, soundState, badgeH, playState,
                     { audioOnly, danger });

            // Hit regions, most specific last — hitTest scans in reverse so the
            // play/delete/soundtrack/edit affordances win over the tile containing them.
            regions.push({ type: "tile", kind, index: i, file: ref.file, x: tx, y: tileY, w: CL.tile, h: CL.tile });
            if (!ref.missing) {
                regions.push({
                    type: "edit",
                    kind,
                    index: i,
                    file: ref.file,
                    x: tx + 3,
                    y: tileY + 3,
                    w: CL.del,
                    h: CL.del,
                });
            }
            if (soundState === "on" || soundState === "off") {
                regions.push({
                    type: "sound",
                    kind,
                    index: i,
                    file: ref.file,
                    x: tx + 3,
                    y: tileY + CL.tile - badgeH - CL.sound - 3,
                    w: CL.sound,
                    h: CL.sound,
                });
            }
            if (audioOnly === "on") {
                regions.push({
                    type: "audio_only",
                    kind,
                    index: i,
                    file: ref.file,
                    x: tx + CL.tile - CL.sound - 3,
                    y: tileY + CL.tile - badgeH - CL.sound - 3,
                    w: CL.sound,
                    h: CL.sound,
                });
            }
            regions.push({
                type: "del",
                kind,
                index: i,
                file: ref.file,
                x: tx + CL.tile - CL.del - 3,
                y: tileY + 3,
                w: CL.del,
                h: CL.del,
            });
            if (playState) {
                const cy = kind === "audio" ? tileY + 72 : tileY + 58;
                regions.push({
                    type: "play",
                    kind,
                    index: i,
                    file: ref.file,
                    x: tx + CL.tile / 2 - CL.playR,
                    y: cy - CL.playR,
                    w: CL.playR * 2,
                    h: CL.playR * 2,
                    tileX: tx,
                    tileY,
                });
            }
        });

        // FIXED placement (UI review #2): always where the next tile would go —
        // the row's left edge when empty, after the last tile (plus the 24px
        // separating gap) otherwise. It never centres itself, so it never jumps;
        // it only ever moves rightwards as tiles are added. Vertically centred
        // on the strip. Drawn dimmed at cap, and only clickable below cap.
        const ax = arr.length
            ? CL.x0 + arr.length * CL.tile + (arr.length - 1) * CL.gap + CL.addGap
            : CL.x0;
        const ay = tileY + Math.round((CL.tile - CL.addBtn) / 2);
        drawAddSquare(ctx, ax, ay, atCap);
        if (!atCap) {
            regions.push({ type: "add", kind, x: ax, y: ay, w: CL.addBtn, h: CL.addBtn });
        }
    }
}

// ---------------------------------------------------------------------------
// Canvas hit-testing. ComfyUI zooms the whole DOM widget with a CSS transform,
// so a client-space pixel is NOT a canvas-space pixel — map through the bounding
// rect scaled back by the element's own layout size (ltx_director.js getMousePos,
// :3900). Everything comes out in the same CSS-px space draw() painted in.
// ---------------------------------------------------------------------------

function getMousePos(canvas, e) {
    const rect = canvas.getBoundingClientRect();
    if (!rect.width || !rect.height) return { x: -1, y: -1 };
    return {
        x: ((e.clientX - rect.left) * canvas.offsetWidth) / rect.width,
        y: ((e.clientY - rect.top) * canvas.offsetHeight) / rect.height,
    };
}

export function hitTest(node, x, y) {
    const hit = node._mmrpHit;
    if (!hit) return null;
    for (let i = hit.regions.length - 1; i >= 0; i--) {
        const r = hit.regions[i];
        if (x >= r.x && x <= r.x + r.w && y >= r.y && y <= r.y + r.h) return r;
    }
    return null;
}

function onCanvasMouseDown(node, e) {
    if (e.button !== 0) return;
    const pos = getMousePos(node._mmrpBody.canvas, e);
    const hit = hitTest(node, pos.x, pos.y);
    if (!hit) {
        if (node._mmrpSelected) {
            node._mmrpSelected = null;
            scheduleDraw(node);
        }
        return;
    }
    if (hit.type === "del") {
        removeRef(node, hit.kind, hit.index);
    } else if (hit.type === "sound") {
        toggleSoundtrack(node, hit.file);
    } else if (hit.type === "audio_only") {
        convertToAudioOnly(node, hit.index);
    } else if (hit.type === "play") {
        togglePreview(node, hit);
    } else if (hit.type === "edit") {
        openEditModal(node, hit.kind, hit.index);
    } else if (hit.type === "add") {
        openAddReferenceModal(node, hit.kind);
    } else {
        node._mmrpSelected = { kind: hit.kind, index: hit.index };
        scheduleDraw(node);
        // Reorder starts on the tile well, never a chip: those already fired above.
        if (hit.type === "tile") beginReorder(node, e, hit);
    }
}

// Double-clicking a tile opens the crop/trim editor too — same modal as the chip.
// The affordance chips keep their own single-click meanings.
function onCanvasDblClick(node, e) {
    const pos = getMousePos(node._mmrpBody.canvas, e);
    const hit = hitTest(node, pos.x, pos.y);
    if (!hit || (hit.type !== "tile" && hit.type !== "edit")) return;
    const ref = node._mmrpRefs[`${hit.kind}s`][hit.index];
    if (!ref || ref.missing) return;
    openEditModal(node, hit.kind, hit.index);
}

function hitTitle(node, hit) {
    if (!hit) return "";
    if (hit.type === "sound") {
        const ref = node._mmrpRefs.videos[hit.index];
        return ref && ref.use_soundtrack ? "Mute soundtrack" : "Unmute soundtrack";
    }
    if (hit.type === "audio_only") return "Use audio only";
    if (hit.type === "edit") {
        return hit.kind === "audio" ? "Edit trim" : hit.kind === "image" ? "Edit crop" : "Edit crop & trim";
    }
    if (hit.type === "del") return "Remove";
    if (hit.type === "play") {
        const p = node._mmrpPlaying;
        return p && p.kind === hit.kind && p.file === hit.file ? "Stop" : "Play";
    }
    if (hit.type === "tile") {
        const n = (node._mmrpRefs[`${hit.kind}s`] || []).length;
        return n > 1 ? "Drag to swap with another in this row" : "";
    }
    return "";
}

const REORDER_ARM_PX = 8;

function beginReorder(node, e, hit) {
    const count = (node._mmrpRefs[`${hit.kind}s`] || []).length;
    if (count < 2) return;
    e.preventDefault();
    e.stopPropagation();
    const canvas = node._mmrpBody.canvas;
    const pos = getMousePos(canvas, e);
    const drag = {
        kind: hit.kind,
        index: hit.index,
        fromIndex: hit.index,
        startX: pos.x,
        startY: pos.y,
        armed: false,
        snapshot: null,
        canvas,
    };
    const onMove = (ev) => onReorderMove(node, ev);
    const onUp = () => finishReorder(node, false);
    const onKey = (ev) => {
        if (ev.key === "Escape") {
            ev.preventDefault();
            finishReorder(node, true);
        }
    };
    drag.onMove = onMove;
    drag.onUp = onUp;
    drag.onKey = onKey;
    node._mmrpReorder = drag;
    window.addEventListener("mousemove", onMove, true);
    window.addEventListener("mouseup", onUp, true);
    window.addEventListener("blur", onUp, true);
    window.addEventListener("keydown", onKey, true);
}

function onReorderMove(node, e) {
    const drag = node._mmrpReorder;
    if (!drag) return;
    const pos = getMousePos(drag.canvas, e);
    if (!drag.armed) {
        const dx = pos.x - drag.startX;
        const dy = pos.y - drag.startY;
        if (dx * dx + dy * dy < REORDER_ARM_PX * REORDER_ARM_PX) return;
        drag.armed = true;
        drag.snapshot = cloneRefs(node._mmrpRefs);
        drag.canvas.style.cursor = "grabbing";
        stopPreview(node);
    }
    e.preventDefault();
    const count = (node._mmrpRefs[`${drag.kind}s`] || []).length;
    const to = slotIndexAtX(pos.x, count, CL.x0, CL.tile, CL.gap);
    if (to === drag.index) return;
    const { refs, error } = moveRefInKind(cloneRefs(node._mmrpRefs), drag.kind, drag.index, to);
    if (error) return;
    drag.index = to;
    node._mmrpSelected = { kind: drag.kind, index: to };
    applyRefs(node, refs);
}

function finishReorder(node, restore) {
    const drag = node._mmrpReorder;
    if (!drag) return;
    window.removeEventListener("mousemove", drag.onMove, true);
    window.removeEventListener("mouseup", drag.onUp, true);
    window.removeEventListener("blur", drag.onUp, true);
    window.removeEventListener("keydown", drag.onKey, true);
    drag.canvas.style.cursor = "";
    node._mmrpReorder = null;
    if (!drag.armed) return;
    if (restore && drag.snapshot) {
        node._mmrpSelected = { kind: drag.kind, index: drag.fromIndex };
        applyRefs(node, drag.snapshot);
        return;
    }
    if (drag.index !== drag.fromIndex) {
        mlog("reference_reordered", { kind: drag.kind, from: drag.fromIndex, to: drag.index });
    }
}

function toggleSoundtrack(node, file) {
    if (probeResults.get(file) !== true) return; // pending/silent/unknown — not toggleable
    const next = cloneRefs(node._mmrpRefs);
    const target = next.videos.find((v) => v.file === file);
    if (target) target.use_soundtrack = !target.use_soundtrack;
    if (target && !target.use_soundtrack) {
        target.roles = (target.roles || []).filter(
            (role) => role !== "audio_reuse" && role !== "audio_reference"
        );
    }
    if (target) mlog("soundtrack", { file, on: target.use_soundtrack });
    applyRefs(node, next);
}

// ---------------------------------------------------------------------------
// Click-to-play previews — ONE shared <video> overlay + ONE shared <audio> per
// node, never per-cell DOM. The rule: the drawn ▶ circle is the play hit region;
// clicking anywhere else on the tile selects. Any refs change stops the preview
// (tile positions shift, the overlay would sit on the wrong tile), as do 'ended',
// a second click, clicking the playing overlay itself, and onRemoved.
// ---------------------------------------------------------------------------

function stopPreview(node) {
    const body = node._mmrpBody;
    if (!body || !node._mmrpPlaying) return;
    body.previewVideo.pause();
    body.previewVideo.removeAttribute("src");
    body.previewVideo.style.display = "none";
    body.previewAudio.pause();
    body.previewAudio.removeAttribute("src");
    node._mmrpPlaying = null;
    scheduleDraw(node);
}

function togglePreview(node, hit) {
    const body = node._mmrpBody;
    const playing = node._mmrpPlaying;
    if (playing && playing.kind === hit.kind && playing.file === hit.file) {
        stopPreview(node);
        return;
    }
    stopPreview(node);
    // A trimmed reference previews ONLY its span: seek to the in-point here, stop at
    // the out-point in the shared elements' timeupdate handlers (buildCustomBlock).
    // Setting currentTime before metadata sets the default playback start position
    // (per the HTML spec), and the timeupdate handler re-seeks if a browser drops it.
    const ref = node._mmrpRefs[`${hit.kind}s`][hit.index];
    const trim = ref && Array.isArray(ref.trim) ? ref.trim : null;
    if (hit.kind === "video") {
        const v = body.previewVideo;
        v.src = fileUrl(hit.file);
        if (trim) v.currentTime = trim[0];
        // Same coordinate space as the canvas — the gallery is the offsetParent for
        // both, and ComfyUI's zoom transform applies to the overlay automatically.
        v.style.left = `${body.canvas.offsetLeft + hit.tileX}px`;
        v.style.top = `${body.canvas.offsetTop + hit.tileY}px`;
        v.style.display = "block";
        v.play().catch(() => stopPreview(node));
    } else {
        const a = body.previewAudio;
        a.src = fileUrl(hit.file);
        if (trim) a.currentTime = trim[0];
        a.play().catch(() => stopPreview(node));
    }
    node._mmrpPlaying = { kind: hit.kind, file: hit.file, trim };
    scheduleDraw(node);
}

// ---------------------------------------------------------------------------
// State <-> widgets
// ---------------------------------------------------------------------------

// Never let a malformed references_json abort workflow loading.
//
// ComfyUI restores widget values POSITIONALLY from widgets_values. Any change to the
// widget list — ours moved the DOM widget from first to last, and the Python side later
// added `system_prompt` — shifts that mapping for workflows saved by an older build, so
// this widget can come back holding a neighbour's value (a model id, an API key, a
// prompt). Parsing that threw a SyntaxError out of onNodeCreated/onConfigure and killed
// the entire workflow load, taking every other node with it.
//
// A reference set we cannot read is recoverable — the user re-adds the files. A workflow
// that will not open is not. So: parse defensively, fall back to empty, warn once.
function parseRefsValue(widget) {
    if (!widget) return emptyRefs();
    const raw = widget.value;
    if (typeof raw !== "string" || !raw.trim()) return emptyRefs();
    try {
        const parsed = JSON.parse(raw);
        return fromReferencesEnvelope(parsed);
    } catch (e) {
        console.warn(
            "[MiniMaxRefPack] references_json held a value this build cannot read " +
                "(likely a widget-order shift from an older saved workflow). Starting with " +
                "no references; re-add them and re-save. Raw value:",
            raw
        );
        return emptyRefs();
    }
}

function syncReferencesWidget(node) {
    const w = widgetByName(node, "references_json");
    if (!w) return;
    const flat = [...node._mmrpRefs.images, ...node._mmrpRefs.videos, ...node._mmrpRefs.audios];
    const list = toReferencesList(node._mmrpRefs).map((r, i) => {
        // carry the "missing" flag through without teaching refs.py about it: it's
        // stripped by fromReferencesList's kind switch on the Python side.
        const src = flat[i];
        return src && src.missing ? { ...r, missing: true } : r;
    });
    const envelope = toReferencesEnvelope(node._mmrpRefs);
    envelope.references = list;
    w.value = JSON.stringify(envelope);
}

function applyRefs(node, refs) {
    stopPreview(node); // tile positions shift with any change — never leave the overlay stale
    node._mmrpRefs = refs;
    syncReferencesWidget(node);
    renderNodeBody(node);
}

async function addFiles(node, kind, files) {
    const room = CAPS[kind] - node._mmrpRefs[`${kind}s`].length;
    if (room <= 0) return;
    const all = Array.from(files);
    const take = all.slice(0, room);
    // Caps are hard, but never SILENTLY hard — say what didn't fit.
    if (take.length < all.length) {
        alert(`Only ${room} ${kind} slot${room === 1 ? "" : "s"} left — skipped ${all.length - take.length} file(s).`);
    }
    mlog("upload", { kind, files: take.length });
    const uploadStarted = performance.now();
    const uploaded = [];
    for (const file of take) {
        try {
            const name = await apiUpload(file);
            mlog("uploaded", { kind, file: name, bytes: file.size });
            // soundtrack ON by default; the probe turns it back off for a silent clip
            uploaded.push(newReference(kind, name));
        } catch (e) {
            mwarn("upload_failed", { kind, file: file.name, error: e.message });
            alert(`Upload failed for ${file.name}: ${e.message}`);
        }
    }
    mlog("upload_done", { kind, files: take.length, ms: performance.now() - uploadStarted });
    // The clone is taken HERE, after the awaits — never at entry. Every `await` above is
    // a window in which another gesture can finish its own applyRefs: a second drop, an
    // upload-button pick, or a tile deletion. A clone taken at entry would be written
    // back over all of it, silently discarding refs whose files are already on the
    // server (and resurrecting ones the user just deleted). Drag-and-drop is what makes
    // this easy to hit — two drops a second apart need no file dialog in between.
    const refs = cloneRefs(node._mmrpRefs);
    const arr = refs[`${kind}s`];
    const stillFree = Math.max(0, CAPS[kind] - arr.length);
    if (uploaded.length > stillFree) {
        // Someone else took the slots while these were uploading. The files are on the
        // server either way; they just do not get a reference.
        mwarn("upload_raced", { kind, dropped: uploaded.length - stillFree });
    }
    arr.push(...uploaded.slice(0, stillFree));
    applyRefs(node, refs);
}

function removeRef(node, kind, index) {
    const refs = cloneRefs(node._mmrpRefs);
    const [gone] = refs[`${kind}s`].splice(index, 1);
    mlog("reference_removed", { kind, file: gone && gone.file });
    if (node._mmrpSelected && node._mmrpSelected.kind === kind) {
        if (node._mmrpSelected.index === index) node._mmrpSelected = null;
        else if (node._mmrpSelected.index > index) node._mmrpSelected.index -= 1;
    }
    applyRefs(node, refs);
}

// The DOM-side reaction to a state change: button disabled states, probes, repaint.
// Reference changes repaint the canvas without resetting the user's node size.
function renderNodeBody(node) {
    const body = node._mmrpBody;
    if (!body) return;
    const refs = node._mmrpRefs;
    for (const kind of KINDS) {
        body.uploadButtons[kind].disabled = refs[`${kind}s`].length >= CAPS[kind];
    }
    if (body.planButton) {
        const state = deriveTaskPlanState(refs);
        body.planButton.textContent = state.error
            ? "Plan: needs attention"
            : state.mode === "explicit"
              ? `Plan: ${state.prefix}`
              : state.mode === "auto"
                ? "Plan: infer roles"
                : "Plan: legacy";
        body.planButton.classList.toggle("mmrp-plan-error", !!state.error);
        body.planButton.title = state.error || "Choose what each reference contributes";
    }
    syncControlsButton(node);
    if (refs.control?.preparation && !refs.control.preparation.error) watchControlPreparation(node);
    node._mmrpControlUpdate?.();
    syncProbes(node);
    scheduleDraw(node);
}

// ---------------------------------------------------------------------------
// Custom block: upload row (DOM buttons), the canvas, prompt textarea
// ---------------------------------------------------------------------------

// Monochrome upload-arrow icon (stroke: currentColor, so it inherits the button's
// white) — one family across all three upload buttons, per the sketch:
// [⬆ Image] [⬆ Video] [⬆ Audio]. The icon carries "upload", the label carries the
// kind, which is what lets the buttons shrink horizontally.
const UPLOAD_ICON =
    '<svg width="12" height="12" viewBox="0 0 16 16" fill="none" stroke="currentColor" ' +
    'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">' +
    '<path d="M8 10.5V3M4.5 6 8 2.5 11.5 6M2.5 12.5v1a1 1 0 0 0 1 1h9a1 1 0 0 0 1-1v-1"/></svg>';

const KIND_UPLOAD_META = [
    ["image", "Image", "image/*"],
    ["video", "Video", "video/*"],
    // video/* too: a clip picked here becomes an audio reference to its soundtrack.
    ["audio", "Audio", "audio/*,video/*"],
];

function convertToAudioOnly(node, index) {
    const before = node._mmrpRefs.videos[index];
    const { refs, error } = videoToAudioOnly(cloneRefs(node._mmrpRefs), index, CAPS.audio);
    if (error) {
        mwarn("audio_only_refused", { index, error });
        return;
    }
    mlog("audio_only", { file: before.file, trim: before.trim || null });
    if (node._mmrpSelected && node._mmrpSelected.kind === "video") {
        if (node._mmrpSelected.index === index) node._mmrpSelected = null;
        else if (node._mmrpSelected.index > index) node._mmrpSelected.index -= 1;
    }
    // Removing a video renumbers the later ones, which can break an explicit plan.
    // renderNodeBody (via applyRefs) puts the Plan button into its error state; this
    // only makes the cause findable in the console.
    const plan = deriveTaskPlanState(refs);
    if (plan.error) mwarn("audio_only_plan_invalid", { file: before.file, error: plan.error });
    applyRefs(node, refs); // also stops any preview: tile positions shift
}

function buildCustomBlock(node) {
    const container = document.createElement("div");
    container.className = "mmrp-block";

    const uploadRow = document.createElement("div");
    uploadRow.className = "mmrp-uploads";
    const uploadButtons = {};
    const fileInputs = {};
    for (const [kind, label, accept] of KIND_UPLOAD_META) {
        const btn = document.createElement("button");
        btn.className = "mmrp-btn mmrp-upload-btn";
        btn.innerHTML = `${UPLOAD_ICON}<span>${label}</span>`;
        const input = document.createElement("input");
        input.type = "file";
        input.accept = accept;
        input.multiple = true;
        input.style.display = "none";
        input.onchange = async (e) => {
            await addFiles(node, kind, e.target.files);
            input.value = "";
        };
        btn.onclick = () => input.click();
        uploadRow.appendChild(btn);
        uploadRow.appendChild(input);
        uploadButtons[kind] = btn;
        fileInputs[kind] = input; // the per-row "+" add slots click these too
    }

    const planButton = document.createElement("button");
    planButton.className = "mmrp-btn mmrp-upload-btn mmrp-plan-btn mmrp-group-start";
    planButton.textContent = "Plan: legacy";
    planButton.title = "Choose what each reference contributes";
    planButton.onclick = () => openTaskPlanModal(node);
    uploadRow.appendChild(planButton);

    const controlsButton = document.createElement("button");
    controlsButton.className = "mmrp-btn mmrp-upload-btn mmrp-plan-btn";
    controlsButton.textContent = "ControlNet";
    controlsButton.onclick = () => openControlModal(node);
    uploadRow.appendChild(controlsButton);

    const saveConfigBtn = document.createElement("button");
    saveConfigBtn.className = "mmrp-btn mmrp-upload-btn mmrp-group-start";
    saveConfigBtn.textContent = "Save config";
    saveConfigBtn.onclick = () => saveConfig(node, saveConfigBtn);
    uploadRow.appendChild(saveConfigBtn);

    const loadConfigBtn = document.createElement("button");
    loadConfigBtn.className = "mmrp-btn mmrp-upload-btn";
    loadConfigBtn.textContent = "Load config";
    loadConfigBtn.onclick = () => openLoadConfigPanel(node, loadConfigBtn);
    uploadRow.appendChild(loadConfigBtn);

    const localBtn = document.createElement("button");
    localBtn.className = "mmrp-btn mmrp-upload-btn mmrp-local-btn";
    localBtn.textContent = "Local LLM";
    localBtn.title = "Find an OpenAI-compatible server on this machine and use it to write prompts";
    localBtn.onclick = () => openLocalServerModal(node, localBtn);
    uploadRow.appendChild(localBtn);
    // Only meaningful while prompt_provider is `local`, so it is hidden otherwise rather
    // than sitting there inert. Removing it from layout shortens the toolbar.
    node._mmrpLocalBtn = localBtn;
    syncLocalBtn(node);

    const gearBtn = document.createElement("button");
    gearBtn.className = "mmrp-btn mmrp-gear-btn";
    gearBtn.textContent = "⚙";
    gearBtn.title = "System prompt settings";
    gearBtn.onclick = () => openSystemPromptModal(node);
    uploadRow.appendChild(gearBtn);

    container.appendChild(uploadRow);

    const gallery = document.createElement("div");
    gallery.className = "mmrp-gallery";
    gallery.style.height = `${CANVAS_ROWS.height + CONTENT.scrollbarH}px`;
    container.appendChild(gallery);

    const canvas = document.createElement("canvas");
    canvas.className = "mmrp-canvas";
    canvas.style.height = `${CANVAS_ROWS.height}px`; // fixed, set once
    canvas.style.minWidth = `${CONTENT.width - CONTENT.pad * 2}px`;
    canvas.addEventListener("mousedown", (e) => onCanvasMouseDown(node, e));
    canvas.addEventListener("dblclick", (e) => onCanvasDblClick(node, e));
    // Painted tiles are not DOM, so a native drag would pick up the whole canvas
    // as an image. File drops still work: those start outside this element.
    canvas.addEventListener("dragstart", (e) => e.preventDefault());
    // The chips carry no text, so a hover names what a click will do.
    canvas.addEventListener("mousemove", (e) => {
        if (node._mmrpReorder && node._mmrpReorder.armed) return;
        const pos = getMousePos(canvas, e);
        const hit = hitTest(node, pos.x, pos.y);
        const title = hitTitle(node, hit);
        if (canvas.title !== title) canvas.title = title;
        const grab = hit && hit.type === "tile"
            && (node._mmrpRefs[`${hit.kind}s`] || []).length > 1;
        canvas.style.cursor = grab ? "grab" : "";
    });
    // Keep litegraph's node context menu off the tiles (matches the reference's
    // per-item contextmenu swallow).
    canvas.addEventListener("contextmenu", (e) => e.stopPropagation());
    gallery.appendChild(canvas);

    // Paint after V3 mounts the block and after resizing. A wrapping toolbar can
    // increase the minimum height, but never resets a larger user-chosen size.
    const resizeObserver = new ResizeObserver(() => {
        const size = nodeSize(node);
        if (node.size[0] !== size[0] || node.size[1] !== size[1]) node.setSize(size);
        scheduleDraw(node);
    });
    resizeObserver.observe(canvas);
    resizeObserver.observe(uploadRow);

    // The preview shares the canvas's scrolling offsetParent and zoom transform.
    const previewVideo = document.createElement("video");
    previewVideo.className = "mmrp-preview";
    previewVideo.playsInline = true;
    previewVideo.addEventListener("ended", () => stopPreview(node));
    previewVideo.addEventListener("click", () => stopPreview(node));
    gallery.appendChild(previewVideo);

    const previewAudio = document.createElement("audio");
    previewAudio.addEventListener("ended", () => stopPreview(node));
    container.appendChild(previewAudio);

    // Trim enforcement for the shared preview pair: stop at the out-point, and
    // re-seek to the in-point if the pre-metadata seek in togglePreview was dropped.
    const keepToTrim = (el) => () => {
        const p = node._mmrpPlaying;
        if (!p || !p.trim) return;
        if (el.currentTime >= p.trim[1]) stopPreview(node);
        else if (el.currentTime < p.trim[0] - 0.25) el.currentTime = p.trim[0];
    };
    previewVideo.addEventListener("timeupdate", keepToTrim(previewVideo));
    previewAudio.addEventListener("timeupdate", keepToTrim(previewAudio));

    // Per spec: "just a large text area" — one plain DOM textarea on the grey
    // node body, no wrapper, no overlay label, no unfocused dimming (the dimming
    // plus a low-contrast border is exactly what once made this box invisible).
    const directionInput = document.createElement("textarea");
    directionInput.className = "mmrp-direction";
    directionInput.placeholder = "Prompt — describe the shot…";
    directionInput.spellcheck = false;
    container.appendChild(directionInput);

    const originalLabel = document.createElement("div");
    originalLabel.className = "mmrp-original-label";
    originalLabel.textContent = "Original prompt";
    container.appendChild(originalLabel);
    const originalInput = document.createElement("textarea");
    originalInput.className = "mmrp-original";
    originalInput.placeholder = "Idea before Auto Prompt rewrote the box above";
    originalInput.spellcheck = false;
    container.appendChild(originalInput);

    // --- drag-and-drop ---------------------------------------------------------
    //
    // Attached to `container` rather than to the canvas: it is the real DOM element
    // layered over the litegraph canvas by addDOMWidget, so native drag events fire
    // on it, and drops over the button row, the canvas and the prompt box all bubble
    // here. These listeners die with the element, so onRemoved has nothing to undo.
    //
    // The hijack guard is on `drop`, not on `dragover`. ComfyUI ingests dropped files
    // from a DOCUMENT-level drop listener that bails when event.defaultPrevented, so a
    // swallow on a sibling overlay's dragover never reaches it. Measured on frontend
    // 1.48.6: with neither guard below, dropping an image here also spawns a LoadImage
    // node; either one alone suppresses that. Both are kept - they fail differently
    // (defaultPrevented vs never reaching document), so a frontend that drops one bail
    // is still covered.
    const dragHasFiles = (e) => {
        // Every handler is gated on this. The prompt textarea is a CHILD of this
        // container, and calling preventDefault on a text drag would take away its
        // perfectly good default: dropping selected text into the box.
        const types = e.dataTransfer && e.dataTransfer.types;
        return !!types && Array.prototype.indexOf.call(types, "Files") !== -1;
    };
    // A counter, not a boolean: dragenter/dragleave fire again for every child element
    // the pointer crosses, so a boolean flickers the highlight off mid-drag.
    let dragDepth = 0;
    const clearDragState = () => {
        dragDepth = 0;
        container.classList.remove("mmrp-dragover");
    };
    container.addEventListener("dragenter", (e) => {
        if (!dragHasFiles(e)) return;
        e.preventDefault();
        dragDepth += 1;
        container.classList.add("mmrp-dragover");
    });
    container.addEventListener("dragover", (e) => {
        if (!dragHasFiles(e)) return;
        e.preventDefault();                    // without this the drop never fires at all
        e.dataTransfer.dropEffect = "copy";
    });
    container.addEventListener("dragleave", (e) => {
        if (!dragHasFiles(e)) return;
        dragDepth -= 1;
        // relatedTarget is what the pointer moved INTO - null when it left the window.
        // Checking it as well as the counter makes the count self-correcting: a drag
        // cancelled earlier can leave the depth stuck above zero, and without this the
        // highlight would survive the next drag that merely flies across the node.
        if (dragDepth <= 0 || !container.contains(e.relatedTarget)) clearDragState();
    });
    // Esc mid-drag, or letting go outside the browser window, cancels the drag with NO
    // dragleave anywhere in the page (measured in Chromium 2026-08-18), so the highlight
    // would sit on the node until the next drag. A live HTML5 drag suppresses mousemove,
    // so a mousemove over the block can only mean the drag is already over.
    container.addEventListener("mousemove", () => {
        if (dragDepth !== 0) clearDragState();
    });
    container.addEventListener("drop", async (e) => {
        if (!dragHasFiles(e)) return;
        clearDragState();
        const { buckets, rejected } = bucketDroppedFiles(e.dataTransfer.files);
        // Look BEFORE claiming. Suppressing the event is how ComfyUI is told to keep its
        // hands off, which is what turns a dropped image into a reference instead of a
        // LoadImage node - but claiming a drop this node cannot use costs the user
        // ComfyUI's own handling of it, and dropping a workflow .json onto a node this
        // large is an easy thing to do. Nothing usable in the drop means it was never
        // ours: let it through and the workflow loads exactly as it does off-node.
        const accepted = DROP_KINDS.reduce((n, kind) => n + buckets[kind].length, 0);
        if (accepted === 0) {
            mlog("drop_passed_through", { rejected: rejected.length });
            return;
        }
        e.preventDefault();     // the documented bail in ComfyUI's document-level ingest
        e.stopPropagation();    // and it never reaches document either way
        if (rejected.length) {
            alert(`Not an image, video or audio file — skipped ${rejected.join(", ")}.`);
        }
        mlog("drop", {
            image: buckets.image.length, video: buckets.video.length,
            audio: buckets.audio.length, rejected: rejected.length,
        });
        // Awaited one kind at a time. addFiles clones the refs, uploads serially and
        // finishes with applyRefs; two of them running concurrently would each apply a
        // clone taken before the other started, and one side's uploads would vanish.
        // DROP_KINDS, not the module-level KINDS: `buckets` is keyed by the former, and
        // a future entry in the latter would index undefined and throw in here.
        for (const kind of DROP_KINDS) {
            if (buckets[kind].length) await addFiles(node, kind, buckets[kind]);
        }
    });

    node._mmrpBody = {
        root: container,
        uploadRow,
        uploadButtons,
        fileInputs,
        planButton,
        controlsButton,
        canvas,
        ctx: canvas.getContext("2d"),
        directionInput,
        originalInput,
        previewVideo,
        previewAudio,
        resizeObserver,
    };
    return container;
}

// ---------------------------------------------------------------------------
// Controls: managed source, explicit preparation, then a reusable map preview.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Config (reference pack) save/load
// ---------------------------------------------------------------------------

// Configs are FILES ON THE USER'S MACHINE, not server state. Save downloads a .json through
// the browser; Load reads one back through a file picker. The point is portability
// across deployments: a pod can be rebuilt or swapped and the config still opens.
// Nothing here touches /minimax_refpack/packs — the only server call is the input-dir
// listing used to work out which referenced files are actually present on THIS pod.
// lib/config.js builds, checks and applies the JSON; the download, the picker and that
// listing are here.

function downloadJson(filename, obj) {
    const blob = new Blob([JSON.stringify(obj, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    // Must be in the document for the click to count as user-initiated in Firefox.
    document.body.appendChild(a);
    a.click();
    a.remove();
    // Revoke on the next tick, not synchronously: Safari aborts a download whose
    // object URL is revoked before it has started reading the blob.
    setTimeout(() => URL.revokeObjectURL(url), 0);
}

function saveConfig(node, anchorBtn) {
    document.querySelectorAll(".mmrp-pack-panel").forEach((p) => p.remove());

    const panel = document.createElement("div");
    panel.className = "mmrp-pack-panel mmrp-save-panel";

    const input = document.createElement("input");
    input.className = "mmrp-save-name";
    input.type = "text";
    input.placeholder = "Config name";
    input.spellcheck = false;
    panel.appendChild(input);

    const saveBtn = document.createElement("button");
    saveBtn.className = "mmrp-btn mmrp-btn-primary";
    saveBtn.textContent = "Download";
    panel.appendChild(saveBtn);

    const status = document.createElement("div");
    status.className = "mmrp-save-status";
    panel.appendChild(status);

    const doSave = () => {
        const name = input.value.trim();
        if (!name) {
            status.textContent = "Name it first.";
            return;
        }
        try {
            const filename = configFilename(name);
            downloadJson(filename, buildConfig(node, name));
            status.textContent = `Downloaded ${filename}`;
            input.disabled = true;
            saveBtn.disabled = true;
            setTimeout(() => panel.remove(), 2000);
        } catch (e) {
            status.textContent = `Download failed: ${e.message}`;
        }
    };
    saveBtn.onclick = doSave;
    input.addEventListener("keydown", (e) => {
        if (e.key === "Enter") doSave();
        else if (e.key === "Escape") panel.remove();
        // Keep graph-level hotkeys (node delete etc.) away from typing.
        e.stopPropagation();
    });

    anchorBtn.parentElement.appendChild(panel);
    input.focus();

    const closeOnOutside = (e) => {
        if (panel.contains(e.target) || e.target === anchorBtn) return;
        panel.remove();
        document.removeEventListener("mousedown", closeOnOutside, true);
    };
    setTimeout(() => document.addEventListener("mousedown", closeOnOutside, true), 0);
}

// Which of these filenames actually exist in THIS pod's input dir. Uses the listing
// route the upload pickers already use, one call per kind that appears in the config.
async function missingFiles(references) {
    const kinds = [...new Set(references.map((r) => r.kind))];
    const present = new Set();
    await Promise.all(
        kinds.map(async (kind) => {
            try {
                const res = await fetch(`/minimax_refpack/files?kind=${encodeURIComponent(kind)}`);
                if (!res.ok) return;
                const data = await res.json();
                for (const f of data.files || []) present.add(f);
            } catch (e) {
                // Unreachable listing: report nothing missing rather than flagging
                // every tile red on a transient network blip.
            }
        }),
    );
    if (!present.size) return new Set();
    return new Set(references.map((r) => r.file).filter((f) => !present.has(f)));
}

function openLoadConfigPanel(node, anchorBtn) {
    document.querySelectorAll(".mmrp-pack-panel").forEach((p) => p.remove());

    const picker = document.createElement("input");
    picker.type = "file";
    picker.accept = ".json,application/json";
    picker.style.display = "none";
    picker.onchange = async () => {
        const file = picker.files && picker.files[0];
        picker.remove();
        if (!file) return;
        try {
            const data = parseConfigFile(await file.text());
            await applyConfig(node, data, file.name, { missingFiles, applyRefs, renderNodeBody });
        } catch (e) {
            alert(`Couldn't load ${file.name}: ${e.message}`);
        }
    };
    document.body.appendChild(picker);
    picker.click();
}

// ---------------------------------------------------------------------------
// System prompt modal — the ONLY place `system_prompt` is ever shown. The widget
// itself is hidden on the node body like direction/references_json (onNodeCreated).
// "Load default" only fills the textarea from the server's packaged default; nothing
// commits to the widget until Save & close, so the user can edit from that starting
// point without losing their in-progress edits by opening/closing the modal.
// ---------------------------------------------------------------------------

// Show the Local LLM button only while prompt_provider is `local`. Called from three
// places because a combo can change three ways: the user clicks it, a saved graph
// restores it, or applyPick() sets it from the picker itself.
function syncLocalBtn(node) {
    const w = widgetByName(node, "prompt_provider");
    const provider = migrateProviderValue(w ? w.value : undefined);

    const btn = node?._mmrpLocalBtn;
    if (btn) btn.style.display = provider === "local" ? "" : "none";

    // A field the run will ignore is worse than absent: it invites you to fill it in and
    // then silently does nothing with it, which is exactly how a local model id ended up
    // being posted to OpenRouter. `none` hides both groups - it calls nobody.
    for (const [name, fields] of Object.entries(PROVIDER_FIELDS)) {
        for (const field of fields) {
            setWidgetVisible(widgetByName(node, field), provider === name);
        }
    }
    // The block's y only settles after litegraph's next layout pass, so re-assert on the frame
    // after rather than reading a stale height now.
    if (node.setSize) {
        setTimeout(() => {
            try {
                node.setSize(nodeSize(node));
                app.graph?.setDirtyCanvas(true, true);
            } catch (_) {}
        }, 0);
    }
}

// litegraph gives a combo widget its own callback; wrapping it is how we hear the user
// changing the value. Guarded against double-wrapping because onNodeCreated can run more
// than once for a node across a paste or an undo, and a chain of wrappers would fire the
// original callback once per wrap.
function watchProviderWidget(node) {
    const w = widgetByName(node, "prompt_provider");
    if (!w || w._mmrpWatched) return;
    const orig = w.callback;
    w.callback = function (...args) {
        const out = orig ? orig.apply(this, args) : undefined;
        syncLocalBtn(node);
        return out;
    };
    w._mmrpWatched = true;
}



// Answers "how do I know the api_base?" by not making the user answer it. Sweeps the
// known local ports server-side, lists what actually replied, and writes all three
// widgets from one click - provider, URL and model id together, because getting two of
// the three right still fails at queue time with a 404 nobody can read.
function openLocalServerModal(node, anchorBtn) {
    const overlay = document.createElement("div");
    overlay.className = "mmrp-overlay";
    overlay.onclick = (e) => {
        if (e.target === overlay) overlay.remove();
    };

    const modal = document.createElement("div");
    modal.className = "mmrp-modal";
    overlay.appendChild(modal);

    const header = document.createElement("div");
    header.className = "mmrp-modal-header";
    header.textContent = "Local LLM";
    modal.appendChild(header);

    const hint = document.createElement("div");
    hint.className = "mmrp-modal-hint";
    hint.textContent = "Looking for a server on this machine...";
    modal.appendChild(hint);

    const list = document.createElement("div");
    list.className = "mmrp-server-list";
    modal.appendChild(list);

    const applyPick = (base, model) => {
        setWidget(node, "prompt_provider", "local");
        setWidget(node, "api_base", base);
        setWidget(node, "local_model_slug", model);
        syncLocalBtn(node);
        overlay.remove();
        if (anchorBtn) {
            const original = anchorBtn.textContent;
            anchorBtn.textContent = "Using local ✓";
            setTimeout(() => { anchorBtn.textContent = original; }, 2000);
        }
        app.graph?.setDirtyCanvas(true, true);
    };

    const render = (servers) => {
        list.replaceChildren();
        if (!servers.length) {
            hint.textContent =
                "No OpenAI-compatible server answered on this machine. Start LM Studio " +
                "(Developer tab), or `ollama serve`, then Rescan. If your server runs " +
                "somewhere else, type its URL into api_base by hand.";
            return;
        }
        hint.textContent =
            "Pick the model that should write your prompts. Videos will be sent as " +
            "still frames and audio will not be sent at all.";
        for (const s of servers) {
            const group = document.createElement("div");
            group.className = "mmrp-server-group";

            const title = document.createElement("div");
            title.className = "mmrp-server-title";
            title.textContent = `${s.label} — ${s.base}`;
            group.appendChild(title);

            if (!s.models.length) {
                const empty = document.createElement("div");
                empty.className = "mmrp-server-empty";
                empty.textContent = "answering, but no model is loaded";
                group.appendChild(empty);
            }
            for (const m of s.models) {
                const row = document.createElement("button");
                row.className = "mmrp-btn mmrp-server-model";
                row.textContent = m;
                row.onclick = () => applyPick(s.base, m);
                group.appendChild(row);
            }
            list.appendChild(group);
        }
    };

    const sweep = () => {
        hint.textContent = "Looking for a server on this machine...";
        list.replaceChildren();
        apiDetectServers()
            .then(render)
            .catch((e) => { hint.textContent = `Could not scan: ${e.message}`; });
    };

    const footer = document.createElement("div");
    footer.className = "mmrp-modal-footer";

    const rescanBtn = document.createElement("button");
    rescanBtn.className = "mmrp-btn";
    rescanBtn.textContent = "Rescan";
    rescanBtn.onclick = sweep;

    const closeBtn = document.createElement("button");
    closeBtn.className = "mmrp-btn mmrp-btn-primary";
    closeBtn.textContent = "Close";
    closeBtn.onclick = () => overlay.remove();

    footer.appendChild(rescanBtn);
    footer.appendChild(closeBtn);
    modal.appendChild(footer);

    document.body.appendChild(overlay);
    sweep();
}


function openSystemPromptModal(node) {
    const systemPromptWidget = widgetByName(node, "system_prompt");

    const overlay = document.createElement("div");
    overlay.className = "mmrp-overlay";
    overlay.onclick = (e) => {
        if (e.target === overlay) overlay.remove();
    };

    const modal = document.createElement("div");
    modal.className = "mmrp-modal";
    overlay.appendChild(modal);

    const header = document.createElement("div");
    header.className = "mmrp-modal-header";
    header.textContent = "System prompt";
    modal.appendChild(header);

    const hint = document.createElement("div");
    hint.className = "mmrp-modal-hint";
    hint.textContent = "Empty = use the built-in default.";
    modal.appendChild(hint);

    const textarea = document.createElement("textarea");
    textarea.className = "mmrp-modal-textarea";
    textarea.spellcheck = false;
    textarea.value = systemPromptWidget ? systemPromptWidget.value || "" : "";
    modal.appendChild(textarea);

    // A blank widget means "use the packaged default", but an empty box hides WHICH
    // prompt is running. Show it, so the modal is the node's prompt rather than a
    // guess about it. Saving an untouched prefill is a no-op: Save writes the
    // textarea back only if it differs from the default (see saveBtn).
    let prefilledDefault = "";
    if (!textarea.value.trim()) {
        hint.textContent = "Showing the built-in default. Edit to override it for this workflow.";
        apiSystemPromptDefault()
            .then((text) => {
                // Don't clobber anything the user typed while the fetch was in flight.
                if (!textarea.value.trim()) {
                    prefilledDefault = text;
                    textarea.value = text;
                }
            })
            .catch(() => {
                hint.textContent = "Empty = use the built-in default.";
            });
    }

    const footer = document.createElement("div");
    footer.className = "mmrp-modal-footer";

    const loadDefaultBtn = document.createElement("button");
    loadDefaultBtn.className = "mmrp-btn";
    loadDefaultBtn.textContent = "Load default";
    loadDefaultBtn.onclick = async () => {
        try {
            textarea.value = await apiSystemPromptDefault();
        } catch (e) {
            alert(`Couldn't load the default: ${e.message}`);
        }
    };

    const resetBtn = document.createElement("button");
    resetBtn.className = "mmrp-btn";
    resetBtn.textContent = "Reset";
    resetBtn.title = "Discard edits, revert to the last-saved value";
    resetBtn.onclick = () => {
        textarea.value = systemPromptWidget ? systemPromptWidget.value || "" : "";
    };

    const cancelBtn = document.createElement("button");
    cancelBtn.className = "mmrp-btn";
    cancelBtn.textContent = "Cancel";
    cancelBtn.onclick = () => overlay.remove();

    const saveCloseBtn = document.createElement("button");
    saveCloseBtn.className = "mmrp-btn mmrp-btn-primary";
    saveCloseBtn.textContent = "Save & close";
    saveCloseBtn.onclick = () => {
        if (systemPromptWidget) {
            // An untouched prefill saves as blank, so the workflow keeps tracking the
            // packaged prompt instead of freezing today's copy into the graph.
            const v = textarea.value;
            systemPromptWidget.value = prefilledDefault && v === prefilledDefault ? "" : v;
        }
        overlay.remove();
    };

    footer.appendChild(loadDefaultBtn);
    footer.appendChild(resetBtn);
    footer.appendChild(cancelBtn);
    footer.appendChild(saveCloseBtn);
    modal.appendChild(footer);

    document.body.appendChild(overlay);
    textarea.focus();
}

// ---------------------------------------------------------------------------
// Selection + delete-key handling. The key handler is CAPTURE-phase and swallows
// the event completely — otherwise ComfyUI's own keydown handler deletes the whole
// NODE on Delete/Backspace. It no-ops while an INPUT/TEXTAREA has focus so
// Backspace in the prompt edits text. Torn down in onRemoved.
// ---------------------------------------------------------------------------

function installSelectionHandlers(node) {
    const body = node._mmrpBody;

    const clearOnOutsideClick = (e) => {
        if (!node._mmrpSelected) return;
        // The canvas's own mousedown decides what a click there means (tile vs
        // empty area); everywhere else, any click drops the selection.
        if (e.target === body.canvas) return;
        node._mmrpSelected = null;
        scheduleDraw(node);
    };

    const keyHandler = (e) => {
        if (!node._mmrpSelected) return;
        const active = document.activeElement;
        if (active && (active.tagName === "TEXTAREA" || active.tagName === "INPUT")) return;
        if (node._mmrpReorder) return; // live drag owns Escape; don't delete mid-swap
        if (e.key === "Delete" || e.key === "Backspace") {
            e.preventDefault();
            e.stopPropagation();
            e.stopImmediatePropagation();
            const { kind, index } = node._mmrpSelected;
            removeRef(node, kind, index);
        } else if (e.key === "Escape") {
            node._mmrpSelected = null;
            scheduleDraw(node);
        }
    };

    document.addEventListener("mousedown", clearOnOutsideClick, true);
    document.addEventListener("keydown", keyHandler, true);

    const origOnRemoved = node.onRemoved;
    node.onRemoved = function () {
        document.removeEventListener("mousedown", clearOnOutsideClick, true);
        document.removeEventListener("keydown", keyHandler, true);
        finishReorder(this, false);
        (node._mmrpHideIntervals || []).forEach((id) => clearInterval(id));
        stopPreview(node);
        if (node._mmrpBody) node._mmrpBody.resizeObserver.disconnect();
        liveNodes.delete(node);
        if (origOnRemoved) origOnRemoved.apply(this, arguments);
    };
}

// ---------------------------------------------------------------------------
// Extension registration
// ---------------------------------------------------------------------------

app.registerExtension({
    name: "MiniMaxRefPack.RefManager",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_NAME) return;

        const origOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            origOnNodeCreated?.apply(this, arguments);
            injectStyles();
            const node = this;
            installInputControls(node, {
                redraw: scheduleDraw,
                connectionsChanged: (changed) => {
                    syncControlsButton(changed);
                    changed._mmrpControlUpdate?.();
                    scheduleDraw(changed);
                },
            });

            const refsWidget = widgetByName(node, "references_json");
            node._mmrpRefs = parseRefsValue(refsWidget);
            const ivRefs = hideWidget(refsWidget);

            const directionWidget = widgetByName(node, "direction");
            const ivDirection = hideWidget(directionWidget);

            const originalWidget = widgetByName(node, "original_prompt");
            const ivOriginal = hideWidget(originalWidget);

            const systemPromptWidget = widgetByName(node, "system_prompt");
            const ivSystemPrompt = hideWidget(systemPromptWidget);

            // Kept in INPUT_TYPES only as a positional compatibility field for old
            // workflows. New routing is owned by the asset-first Task plan button.
            const jobTypeWidget = widgetByName(node, "job_type");
            const ivJobType = hideWidget(jobTypeWidget);

            // Cleared early in installSelectionHandlers' onRemoved wrapper so a deleted
            // node doesn't leave a hide-poll timer running (they also self-clear after
            // 1s regardless, this just avoids the wait on an early delete).
            node._mmrpHideIntervals = [ivRefs, ivDirection, ivOriginal, ivSystemPrompt, ivJobType]
                .filter((id) => id !== undefined);

            const bodyEl = buildCustomBlock(node);
            // hideOnZoom defaults to TRUE in addDOMWidget. Below the frontend's LOD scale
            // threshold the overlay stops taking pointer events, and a drop on the
            // zoomed-out node then lands on the litegraph canvas and gets hijacked into
            // a LoadImage node. Keeping the overlay live is what stops that.
            addBlockWidget(node, bodyEl, { serialize: false, hideOnZoom: false });
            liveNodes.add(node);

            node._mmrpBody.directionInput.value = directionWidget ? directionWidget.value || "" : "";
            node._mmrpBody.directionInput.addEventListener("input", () => {
                if (!directionWidget) return;
                directionWidget.value = node._mmrpBody.directionInput.value;
                if (directionWidget.callback) directionWidget.callback(directionWidget.value);
            });
            node._mmrpBody.originalInput.value = originalWidget ? originalWidget.value || "" : "";
            node._mmrpBody.originalInput.addEventListener("input", () => {
                if (!originalWidget) return;
                originalWidget.value = node._mmrpBody.originalInput.value;
                if (originalWidget.callback) originalWidget.callback(originalWidget.value);
            });

            installSizeGuards(node, scheduleDraw);
            node.setSize(defaultSize(node));
            installSelectionHandlers(node);
            watchProviderWidget(node);
            syncLocalBtn(node);
            renderNodeBody(node);
            syncReferencesWidget(node);

            const origOnConfigure = node.onConfigure;
            node.onConfigure = function (info) {
                const out = origOnConfigure ? origOnConfigure.apply(this, arguments) : undefined;
                invalidateControlOperation(this);
                ensureControlOutput(this);
                // configure() has just applied widgets_values POSITIONALLY, which for any
                // graph saved before 0.3.3 means the values landed in the wrong widgets.
                // The original array is still on `info`, so re-place it by name from the
                // layout it was actually written in.
                const saved = info && info.widgets_values;
                const byName = remapWidgetValues(saved);
                if (byName && detectLayout(saved) !== ORDER_CURRENT) {
                    for (const [name, value] of Object.entries(byName)) {
                        setWidget(this, name, value);
                    }
                    console.log(
                        `[MiniMaxRefPack] event=migrated_layout slots=${saved.length} ` +
                        `provider=${byName.prompt_provider}`
                    );
                } else if (migrateProviderWidget(widgetByName(this, "prompt_provider"))) {
                    console.log(
                        "[MiniMaxRefPack] event=migrated_provider from=use_openrouter " +
                        `to=${widgetByName(this, "prompt_provider").value}`
                    );
                }
                const matchWidget = widgetByName(this, "match_video_aspect");
                if (matchWidget && matchWidget.value !== migrateMatchValue(matchWidget.value)) {
                    matchWidget.value = migrateMatchValue(matchWidget.value);
                    console.log("[MiniMaxRefPack] event=migrated_match_video_aspect to=false");
                }
                // After the migration above, so a 0.3.1 graph's `true` has already become
                // "openrouter" and the button is hidden rather than flickering on.
                watchProviderWidget(this);
                syncLocalBtn(this);
                const rw = widgetByName(this, "references_json");
                stopPreview(this);
                this._mmrpRefs = parseRefsValue(rw);
                const dw = widgetByName(this, "direction");
                const ow = widgetByName(this, "original_prompt");
                const named = info && info.widgets_values_named;
                if (ow && !(String(ow.value || "").trim())) {
                    const stored = named && typeof named.original_prompt === "string"
                        ? named.original_prompt : "";
                    const namedDir = named && typeof named.direction === "string"
                        ? named.direction : "";
                    const direction = dw ? String(dw.value || "") : "";
                    if (stored.trim()) ow.value = stored;
                    else if (namedDir && namedDir !== direction) ow.value = namedDir;
                }
                if (this._mmrpBody) {
                    this._mmrpBody.directionInput.value = dw ? dw.value || "" : "";
                    this._mmrpBody.originalInput.value = ow ? ow.value || "" : "";
                }
                this._mmrpSelected = null;
                if (info && info.size) this._mmrpRestored = true;
                // configure() writes the serialized size straight onto node.size,
                // bypassing our setSize wrapper — preserve it within content minimums.
                this.setSize(nodeSize(this));
                renderNodeBody(this);
                return out;
            };

            // The block's y is known only after the first layout pass: a new node takes
            // its default size again from the real y, a restored one keeps its saved size.
            setTimeout(() => {
                node.setSize(node._mmrpRestored ? nodeSize(node) : defaultSize(node));
                app.graph?.setDirtyCanvas(true, true);
                scheduleDraw(node);
            }, 50);
        };
    },
});
