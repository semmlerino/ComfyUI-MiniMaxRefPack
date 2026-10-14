// The crop/trim editor: one modal edits both, on the real media from the stock /view
// route, and previews what the socket will carry. Crop lives on image and video, trim
// on video and audio; the modal shows whichever applies. Everything is edited in
// fraction/second space and only committed on Save; Cancel (or clicking the backdrop)
// discards. The math is in lib/edits.js.

import {
    ASPECT_PRESETS, applyEdit, clamp01, clampTrim, cropPreviewBox, dragCrop, normalizeCrop,
    normalizeTrim, setRectAspect,
} from "./edits.js";
import { CAPS } from "./layout.js";
import { mlog, mwarn } from "./log.js";
import { cloneRefs } from "./references.js";
import { widgetByName } from "./widgets.js";

// `host` is what lives in refpack.js: the media URL, the soundtrack-probe and thumbnail
// caches, the soundtrack and audio-only actions, and applyRefs, which commits an edit.
export function createEditUi(host) {
    const { fileUrl, probeResults, dropThumb, toggleSoundtrack, convertToAudioOnly, applyRefs } = host;

    function openEditModal(node, kind, index) {
        const ref = node._mmrpRefs[`${kind}s`][index];
        if (!ref || ref.missing) return;
        mlog("edit_open", { kind, file: ref.file, crop: ref.crop, trim: ref.trim });
        const wantsCrop = kind !== "audio";
        const wantsTrim = kind !== "image";

        let crop = Array.isArray(ref.crop) ? ref.crop.slice() : [0, 0, 1, 1];
        let trim = Array.isArray(ref.trim) ? ref.trim.slice() : null; // null until duration known
        let duration = null;
        let ratio = null; // aspect lock; null = free
        let mediaW = 0;
        let mediaH = 0;
        const r2 = (v) => Math.round(v * 100) / 100;  // the duration label's rounding

        // The two Clear buttons dim when there is nothing to clear, so the modal shows
        // at a glance whether this reference currently carries an edit. Live state, the
        // same predicates Save uses — dragging a rect out enables Clear crop instantly.
        let clearCropBtn = null;
        let clearTrimBtn = null;
        const syncClears = () => {
            if (clearCropBtn) clearCropBtn.disabled = normalizeCrop(crop) === null;
            if (clearTrimBtn) clearTrimBtn.disabled = normalizeTrim(trim, duration) === null;
        };

        const overlay = document.createElement("div");
        overlay.className = "mmrp-overlay";
        overlay.onclick = (e) => {
            if (e.target === overlay) overlay.remove();
        };

        const modal = document.createElement("div");
        modal.className = "mmrp-modal mmrp-edit-modal";
        overlay.appendChild(modal);

        const header = document.createElement("div");
        header.className = "mmrp-modal-header";
        header.textContent = `Edit — ${ref.file}`;
        modal.appendChild(header);

        // ---- the media, real pixels via the stock /view route ----
        const stage = document.createElement("div");
        stage.className = "mmrp-edit-stage";
        let media;
        if (kind === "image") {
            media = document.createElement("img");
        } else if (kind === "video") {
            media = document.createElement("video");
            media.muted = true;
            media.playsInline = true;
            media.preload = "auto";
        } else {
            media = document.createElement("audio");
            media.controls = true;
            media.preload = "metadata";
        }
        media.src = fileUrl(ref.file);
        // The media lives in a wrapper so "Play edit" can reframe it (the wrapper becomes
        // the crop's window; the media is scaled and shifted inside it). Untouched, the
        // wrapper shrink-wraps the media and nothing about the layout changes.
        const mediaWrap = document.createElement("div");
        mediaWrap.className = "mmrp-edit-media-wrap";
        mediaWrap.appendChild(media);
        stage.appendChild(mediaWrap);
        modal.appendChild(stage);

        let cropLayer = null;   // set below when the media can be cropped

        // ---- crop rect + corner handles (image/video) ----
        let syncCropRect = () => {};
        if (wantsCrop) {
            const layer = document.createElement("div");
            layer.className = "mmrp-crop-layer";
            const rectEl = document.createElement("div");
            rectEl.className = "mmrp-crop-rect";
            layer.appendChild(rectEl);
            const handles = {};
            for (const corner of ["nw", "ne", "sw", "se"]) {
                const h = document.createElement("div");
                h.className = `mmrp-crop-handle mmrp-crop-${corner}`;
                rectEl.appendChild(h);
                handles[corner] = h;
            }
            stage.appendChild(layer);
            cropLayer = layer;

            syncCropRect = () => {
                rectEl.style.left = `${crop[0] * 100}%`;
                rectEl.style.top = `${crop[1] * 100}%`;
                rectEl.style.width = `${crop[2] * 100}%`;
                rectEl.style.height = `${crop[3] * 100}%`;
                syncClears();
            };
            syncCropRect();

            // Pointer deltas in fractions of the layer box; dragCrop does the math.
            const startDrag = (e, mode) => {
                stopPlayback();
                e.preventDefault();
                e.stopPropagation();
                const box = layer.getBoundingClientRect();
                if (!box.width || !box.height) return;
                const from = { x: e.clientX, y: e.clientY, crop: crop.slice() };
                const move = (ev) => {
                    const dx = (ev.clientX - from.x) / box.width;
                    const dy = (ev.clientY - from.y) / box.height;
                    crop = dragCrop(from.crop, mode, dx, dy, ratio, mediaW, mediaH);
                    syncCropRect();
                };
                const up = () => {
                    window.removeEventListener("mousemove", move);
                    window.removeEventListener("mouseup", up);
                };
                window.addEventListener("mousemove", move);
                window.addEventListener("mouseup", up);
            };
            rectEl.addEventListener("mousedown", (e) => startDrag(e, "move"));
            for (const corner of ["nw", "ne", "sw", "se"]) {
                handles[corner].addEventListener("mousedown", (e) => startDrag(e, corner));
            }
        }

        // ---- aspect presets (crop only): free / the node's width:height / fixed ratios ----
        if (wantsCrop) {
            const row = document.createElement("div");
            row.className = "mmrp-aspect-row";
            const label = document.createElement("span");
            label.className = "mmrp-edit-label";
            label.textContent = "Crop";
            row.appendChild(label);

            const presets = [["Free", null]];
            const wWidget = widgetByName(node, "width");
            const hWidget = widgetByName(node, "height");
            const nodeW = wWidget ? Number(wWidget.value) : 0;
            const nodeH = hWidget ? Number(hWidget.value) : 0;
            if (nodeW > 0 && nodeH > 0) presets.push([`${nodeW}:${nodeH}`, nodeW / nodeH]);
            presets.push(...ASPECT_PRESETS);

            const buttons = [];
            for (const [text, r] of presets) {
                const btn = document.createElement("button");
                btn.className = "mmrp-btn";
                btn.textContent = text;
                btn.onclick = () => {
                    stopPlayback();
                    ratio = r;
                    for (const b of buttons) b.classList.toggle("mmrp-active", b === btn);
                    if (r) {
                        crop = setRectAspect(crop, r, mediaW, mediaH);
                        syncCropRect();
                    }
                };
                row.appendChild(btn);
                buttons.push(btn);
            }
            // Nothing is highlighted on open. Free IS the starting behaviour (ratio = null),
            // but showing it selected reads as a choice the user made, and then "Clear crop"
            // has nothing left to visibly clear. The highlight means "you picked a lock".

            // Right-aligned, paired with the trim row's Clear trim: back to the whole
            // frame, aspect lock released and every preset unhighlighted. Save then deletes
            // the saved crop, exactly like a hand-dragged full-frame rect would
            // (normalizeCrop -> null).
            clearCropBtn = document.createElement("button");
            clearCropBtn.className = "mmrp-btn mmrp-clear-btn";
            clearCropBtn.textContent = "Clear crop";
            clearCropBtn.onclick = () => {
                stopPlayback();
                mlog("edit_cleared", { file: ref.file, what: "crop" });
                crop = [0, 0, 1, 1];
                ratio = null;
                for (const b of buttons) b.classList.remove("mmrp-active");
                syncCropRect();
            };
            row.appendChild(clearCropBtn);
            modal.appendChild(row);
        }

        // ---- trim bar + 2dp second fields (video/audio) ----
        let syncTrim = () => {};
        if (wantsTrim) {
            const row = document.createElement("div");
            row.className = "mmrp-trim-row";
            const label = document.createElement("span");
            label.className = "mmrp-edit-label";
            label.textContent = "Trim";
            row.appendChild(label);

            const inNum = document.createElement("input");
            const outNum = document.createElement("input");
            for (const el of [inNum, outNum]) {
                el.className = "mmrp-trim-num";
                el.type = "number";
                el.step = "0.01";
                el.min = "0";
                el.disabled = true;
                // Keep graph hotkeys away from typing, same as the save-name box.
                el.addEventListener("keydown", (e) => e.stopPropagation());
            }

            const bar = document.createElement("div");
            bar.className = "mmrp-trim-bar";
            const span = document.createElement("div");
            span.className = "mmrp-trim-span";
            const inHandle = document.createElement("div");
            inHandle.className = "mmrp-trim-handle";
            const outHandle = document.createElement("div");
            outHandle.className = "mmrp-trim-handle";
            bar.appendChild(span);
            bar.appendChild(inHandle);
            bar.appendChild(outHandle);

            const durLabel = document.createElement("span");
            durLabel.className = "mmrp-edit-label";
            durLabel.textContent = "…";

            // "Clear crop"'s pair: back to the whole clip, so Save deletes the saved
            // trim (normalizeTrim collapses a full-span window to null).
            clearTrimBtn = document.createElement("button");
            clearTrimBtn.className = "mmrp-btn mmrp-clear-btn";
            clearTrimBtn.textContent = "Clear trim";

            row.appendChild(inNum);
            row.appendChild(bar);
            row.appendChild(outNum);
            row.appendChild(durLabel);
            row.appendChild(clearTrimBtn);
            modal.appendChild(row);

            syncTrim = () => {
                if (!duration || !trim) return;
                inNum.value = trim[0].toFixed(2);
                outNum.value = trim[1].toFixed(2);
                span.style.left = `${(trim[0] / duration) * 100}%`;
                span.style.width = `${((trim[1] - trim[0]) / duration) * 100}%`;
                inHandle.style.left = `calc(${(trim[0] / duration) * 100}% - 5px)`;
                outHandle.style.left = `calc(${(trim[1] / duration) * 100}% - 5px)`;
                syncClears();
            };

            const setTrim = (start, end, seekTo) => {
                if (!duration) return;
                trim = clampTrim(start, end, duration);
                syncTrim();
                // scrub the modal's own video so the user sees the frame they chose
                if (kind === "video" && seekTo !== undefined) media.currentTime = seekTo;
            };

            const dragHandle = (which) => (e) => {
                stopPlayback();
                e.preventDefault();
                const box = bar.getBoundingClientRect();
                if (!box.width || !duration) return;
                const move = (ev) => {
                    const t = clamp01((ev.clientX - box.left) / box.width, 0, 1) * duration;
                    if (which === "in") setTrim(t, trim[1], t);
                    else setTrim(trim[0], t, t);
                };
                const up = () => {
                    window.removeEventListener("mousemove", move);
                    window.removeEventListener("mouseup", up);
                };
                window.addEventListener("mousemove", move);
                window.addEventListener("mouseup", up);
            };
            inHandle.addEventListener("mousedown", dragHandle("in"));
            outHandle.addEventListener("mousedown", dragHandle("out"));

            inNum.addEventListener("change", () => stopPlayback() || setTrim(parseFloat(inNum.value) || 0, trim ? trim[1] : 0, parseFloat(inNum.value) || 0));
            outNum.addEventListener("change", () => stopPlayback() || setTrim(trim ? trim[0] : 0, parseFloat(outNum.value) || 0));
            clearTrimBtn.onclick = () => {
                mlog("edit_cleared", { file: ref.file, what: "trim" });
                stopPlayback();
                setTrim(0, duration || 0);
            };

            media.addEventListener("loadedmetadata", () => {
                duration = media.duration;
                durLabel.textContent = `/ ${r2(duration).toFixed(2)}s`;
                inNum.disabled = false;
                outNum.disabled = false;
                if (!trim) trim = [0, r2(duration)];
                // a saved trim on a since-replaced, shorter file still clamps sanely
                setTrim(trim[0], trim[1], kind === "video" ? trim[0] : undefined);
            });
        }

        // ---- play the original / play the edit (video, audio) -------------------------
        // The tile's ▶ plays the reference; this plays what the SOCKET will carry. "Play
        // edit" seeks to the in-point, stops at the out-point, and (for video) reframes the
        // media so the crop fills the window - the rect overlay is hidden while it runs,
        // because the media underneath it has moved.
        let stopPlayback = () => {};
        if (kind !== "image") {
            const row = document.createElement("div");
            row.className = "mmrp-play-row";
            const label = document.createElement("span");
            label.className = "mmrp-edit-label";
            label.textContent = "Preview";
            row.appendChild(label);

            const origBtn = document.createElement("button");
            origBtn.className = "mmrp-btn";
            const editBtn = document.createElement("button");
            editBtn.className = "mmrp-btn";
            const ORIG = "Play original";
            const EDIT = wantsCrop ? "Play edit" : "Play trim";
            origBtn.textContent = ORIG;
            editBtn.textContent = EDIT;
            row.appendChild(origBtn);
            row.appendChild(editBtn);
            if (kind === "video") {
                // The one place with words for the tile's ♪ chip: same toggle, applied at
                // once (not staged for Save - Save re-reads the live refs, so it survives).
                const soundBtn = document.createElement("button");
                soundBtn.className = "mmrp-btn";
                const syncSound = () => {
                    const live = node._mmrpRefs.videos.find((v) => v.file === ref.file);
                    const hasAudio = probeResults.get(ref.file);
                    soundBtn.disabled = hasAudio !== true;
                    soundBtn.textContent = hasAudio === false
                        ? "Soundtrack: none"
                        : live && live.use_soundtrack ? "Soundtrack: on" : "Soundtrack: muted";
                    soundBtn.title = hasAudio === false
                        ? "This clip has no decodable audio track"
                        : live && live.use_soundtrack ? "Mute this video's soundtrack" : "Unmute this video's soundtrack";
                };
                soundBtn.onclick = () => {
                    toggleSoundtrack(node, ref.file);
                    syncSound();
                };
                syncSound();
                row.appendChild(soundBtn);
            }
            modal.appendChild(row);

            let mode = null;      // null | "original" | "edit"
            let stopAt = null;
            let raf = null;       // out-point watchdog; timeupdate alone fires every ~250ms,
                                  // which overshoots the out-point by a visible quarter second.
                                  // This is NOT the canvas draw loop the header rules out - it
                                  // exists only while the modal's own media is playing.

            const frame = (on) => {
                if (!wantsCrop) return;
                if (on) {
                    const box = media.getBoundingClientRect();
                    if (!box.width || !box.height) return;
                    const f = cropPreviewBox(crop, box.width, box.height);
                    mediaWrap.style.width = `${f.wrapW}px`;
                    mediaWrap.style.height = `${f.wrapH}px`;
                    Object.assign(media.style, {
                        position: "absolute",
                        maxWidth: "none",
                        maxHeight: "none",
                        width: `${f.mediaW}px`,
                        height: `${f.mediaH}px`,
                        left: `${f.left}px`,
                        top: `${f.top}px`,
                    });
                    if (cropLayer) cropLayer.style.display = "none";
                } else {
                    mediaWrap.style.width = "";
                    mediaWrap.style.height = "";
                    for (const k of ["position", "maxWidth", "maxHeight", "width", "height", "left", "top"]) {
                        media.style[k] = "";
                    }
                    if (cropLayer) cropLayer.style.display = "";
                }
            };

            const sync = () => {
                origBtn.textContent = mode === "original" ? "Stop" : ORIG;
                editBtn.textContent = mode === "edit" ? "Stop" : EDIT;
            };

            stopPlayback = () => {
                if (!mode) return;
                if (raf !== null) {
                    cancelAnimationFrame(raf);
                    raf = null;
                }
                media.pause();
                mode = null;
                stopAt = null;
                if (kind === "video") media.muted = true;
                frame(false);
                sync();
            };

            const start = (next) => {
                if (mode === next) {
                    stopPlayback();
                    return;
                }
                frame(false);
                mode = next;
                if (next === "edit") {
                    frame(true);
                    media.currentTime = trim ? trim[0] : 0;
                    stopAt = trim ? trim[1] : null;
                } else {
                    media.currentTime = 0;
                    stopAt = null;
                }
                media.muted = false;   // the point of pressing play is to hear it too
                mlog("preview", { file: ref.file, mode: next,
                                  from: next === "edit" && trim ? trim[0] : 0,
                                  to: next === "edit" && trim ? trim[1] : null });
                media.play().catch(() => stopPlayback());
                const tick = () => {
                    if (!mode) return;
                    if (stopAt !== null && media.currentTime >= stopAt) {
                        stopPlayback();
                        return;
                    }
                    raf = requestAnimationFrame(tick);
                };
                raf = requestAnimationFrame(tick);
                sync();
            };

            origBtn.onclick = () => start("original");
            editBtn.onclick = () => start("edit");
            media.addEventListener("timeupdate", () => {
                if (stopAt !== null && media.currentTime >= stopAt) stopPlayback();
            });
            media.addEventListener("ended", () => stopPlayback());
        }

        // For an image, size math only needs the natural dimensions.
        if (kind === "image") {
            media.addEventListener("load", () => {
                mediaW = media.naturalWidth;
                mediaH = media.naturalHeight;
            });
        } else if (kind === "video") {
            media.addEventListener("loadedmetadata", () => {
                mediaW = media.videoWidth;
                mediaH = media.videoHeight;
            });
        }

        // Initial disabled state for the Clear pair — the sync calls above ran before
        // the buttons existed. (Clear trim re-syncs again on loadedmetadata.)
        syncClears();

        // ---- footer ----
        const footer = document.createElement("div");
        footer.className = "mmrp-modal-footer";

        const cancelBtn = document.createElement("button");
        cancelBtn.className = "mmrp-btn";
        cancelBtn.textContent = "Cancel";
        cancelBtn.onclick = () => overlay.remove();

        const saveBtn = document.createElement("button");
        saveBtn.className = "mmrp-btn mmrp-btn-primary";
        saveBtn.textContent = "Save";
        saveBtn.onclick = () => {
            const next = cloneRefs(node._mmrpRefs);
            // the full clip is "no trim" — serialise nothing, like refs.py omits it
            const { crop: nc, trim: nt } = applyEdit(next[`${kind}s`][index], kind, crop, trim, duration);

            mlog("edit_saved", { kind, file: ref.file, crop: nt === null && nc === null ? null : nc,
                                 trim: nt, cleared: nc === null && nt === null });
            dropThumb(ref.file); // the tile re-fetches through the route with the edit
            overlay.remove();
            applyRefs(node, next);
        };

        if (kind === "video") {
            const audioOnlyBtn = document.createElement("button");
            audioOnlyBtn.className = "mmrp-btn";
            audioOnlyBtn.textContent = "Audio only";
            audioOnlyBtn.title =
                "Replace this video with an audio reference to its soundtrack. Uses the SAVED " +
                "trim - an unsaved trim edit in this window is discarded.";
            audioOnlyBtn.disabled =
                probeResults.get(ref.file) !== true || node._mmrpRefs.audios.length >= CAPS.audio;
            audioOnlyBtn.onclick = () => {
                stopPlayback();
                overlay.remove();
                // By index, never re-found by filename: the same clip may be referenced
                // twice with different trims, and findIndex would convert the wrong one.
                if (node._mmrpRefs.videos[index] === ref) convertToAudioOnly(node, index);
                else mwarn("audio_only_stale", { file: ref.file, index });
            };
            footer.appendChild(audioOnlyBtn);
        }
        footer.appendChild(cancelBtn);
        footer.appendChild(saveBtn);
        modal.appendChild(footer);

        document.body.appendChild(overlay);
    }

    return { openEditModal };
}
