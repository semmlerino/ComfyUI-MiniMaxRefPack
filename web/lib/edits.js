// ---------------------------------------------------------------------------
// Crop/trim pure helpers — fraction-space math only, no DOM, so node can exercise
// them the way pytest exercises refs.py.
// ---------------------------------------------------------------------------

// The badge's extra line: "2.00-6.50s · cropped" / "2.00-6.50s" / "cropped",
// or null when the reference is untouched.
export function editSummary(ref) {
    const parts = [];
    if (ref && Array.isArray(ref.trim)) parts.push(`${ref.trim[0].toFixed(2)}-${ref.trim[1].toFixed(2)}s`);
    if (ref && Array.isArray(ref.crop)) parts.push("cropped");
    return parts.length ? parts.join(" · ") : null;
}

export function hasEdit(ref) {
    return !!(ref && (Array.isArray(ref.crop) || Array.isArray(ref.trim)));
}

export function clamp01(v, lo, hi) {
    return Math.min(Math.max(v, lo), hi);
}

// What Save actually writes: round to 4 decimals (plenty below one pixel at 8K),
// clamp into the unit square with w capped at 1-x so refs.py's validate_crop can
// never reject a rect this produced, and collapse a (near-)full-frame rect to null
// — no crop at all, so an untouched reference serialises exactly as before.
export function normalizeCrop(rect) {
    if (!rect) return null;
    const r4 = (v) => Math.round(v * 1e4) / 1e4;
    const x = clamp01(r4(rect[0]), 0, 1);
    const y = clamp01(r4(rect[1]), 0, 1);
    const w = Math.min(r4(rect[2]), r4(1 - x));
    const h = Math.min(r4(rect[3]), r4(1 - y));
    if (w <= 0 || h <= 0) return null;
    if (x === 0 && y === 0 && w === 1 && h === 1) return null;
    return [x, y, w, h];
}

// What Save writes for the trim: [start, end] rounded to 2dp, or null when the
// window covers (within the 2dp rounding, 4ms) the whole clip — no trim at all,
// mirroring normalizeCrop's full-frame collapse. Also the predicate behind the
// modal's "Clear trim" disabled state: null here means there is nothing to clear.
export function normalizeTrim(trim, duration) {
    if (!trim || !duration) return null;
    const r2 = (v) => Math.round(v * 100) / 100;
    const s = r2(trim[0]);
    const e = r2(trim[1]);
    if (e <= s) return null;
    if (s <= 0.004 && e >= r2(duration) - 0.004) return null;
    return [s, e];
}

// One pointer-drag step over a fraction rect. mode = "move" | "nw"|"ne"|"sw"|"se";
// dx/dy are pointer deltas as fractions of the media box. `ratio` (a PIXEL w:h,
// e.g. 16/9) locks the rect's pixel aspect, which in fraction space means
// hFrac = wFrac * mediaW / (ratio * mediaH). The corner opposite the dragged one
// is the anchor and never moves.
export function dragCrop(rect, mode, dx, dy, ratio, mediaW, mediaH) {
    const MIN = 0.02;
    const [x, y, w, h] = rect;
    if (mode === "move") {
        return [clamp01(x + dx, 0, 1 - w), clamp01(y + dy, 0, 1 - h), w, h];
    }
    const ax = mode === "nw" || mode === "sw" ? x + w : x;
    const ay = mode === "nw" || mode === "ne" ? y + h : y;
    const cx = clamp01((mode === "nw" || mode === "sw" ? x : x + w) + dx, 0, 1);
    const cy = clamp01((mode === "nw" || mode === "ne" ? y : y + h) + dy, 0, 1);
    let nw = Math.max(Math.abs(cx - ax), MIN);
    let nh = Math.max(Math.abs(cy - ay), MIN);
    if (ratio && mediaW && mediaH) {
        nh = (nw * mediaW) / (ratio * mediaH);
        const maxH = cy >= ay ? 1 - ay : ay; // room on the side being dragged into
        if (nh > maxH) {
            nh = maxH;
            nw = (nh * ratio * mediaH) / mediaW;
        }
    }
    const nx = cx >= ax ? ax : ax - nw;
    const ny = cy >= ay ? ay : ay - nh;
    return [clamp01(nx, 0, 1 - nw), clamp01(ny, 0, 1 - nh), nw, nh];
}

// An aspect-preset click: reshape the current rect around its own centre to the
// given PIXEL ratio, spilling as little as possible past the frame.
export function setRectAspect(rect, ratio, mediaW, mediaH) {
    const [x, y, w, h] = rect;
    const cx = x + w / 2;
    const cy = y + h / 2;
    let nw = w;
    let nh = (w * mediaW) / (ratio * mediaH);
    if (nh > 1) {
        nw = nw / nh;
        nh = 1;
    }
    return [clamp01(cx - nw / 2, 0, 1 - nw), clamp01(cy - nh / 2, 0, 1 - nh), nw, nh];
}

/** Framing for "play the modified video": the crop region blown up to fill the same
 * footprint the untouched media occupies, so the preview shows what the socket emits
 * rather than a rectangle drawn over the original.
 *
 * `dw`/`dh` are the media's CURRENT displayed size in CSS px. The crop region is
 * scaled by min(dw/cw, dh/ch) - the largest scale that still fits the footprint - so
 * the region's own aspect is preserved and the box is never overflowed. The media
 * itself scales by the same factor and shifts by the crop origin, which is what puts
 * the region under the wrapper's visible window.
 */
export function cropPreviewBox(crop, dw, dh) {
    const [x, y, w, h] = crop;
    const cw = dw * w;
    const ch = dh * h;
    const scale = Math.min(dw / cw, dh / ch);
    return {
        wrapW: cw * scale,
        wrapH: ch * scale,
        mediaW: dw * scale,
        mediaH: dh * scale,
        left: -x * dw * scale,
        top: -y * dh * scale,
    };
}

export const ASPECT_PRESETS = [
    ["1:1", 1],
    ["16:9", 16 / 9],
    ["9:16", 9 / 16],
];

// The trim window the editor accepts: 2dp seconds inside the clip, with the in-point
// at least 0.05s before the out-point.
export function clampTrim(start, end, duration) {
    const r2 = (v) => Math.round(v * 100) / 100;
    const s = clamp01(r2(start), 0, Math.max(0, r2(duration) - 0.05));
    return [s, clamp01(r2(end), s + 0.05, r2(duration))];
}

// Writes the editor's crop and trim onto `target` the way Save serialises them: an
// edit that covers the whole frame or clip, or one the kind cannot carry, is removed.
// Returns the normalised pair.
export function applyEdit(target, kind, crop, trim, duration) {
    const nc = kind !== "audio" ? normalizeCrop(crop) : null;
    if (nc) target.crop = nc;
    else delete target.crop;
    const nt = kind !== "image" ? normalizeTrim(trim, duration) : null;
    if (nt) target.trim = nt;
    else delete target.trim;
    return { crop: nc, trim: nt };
}
