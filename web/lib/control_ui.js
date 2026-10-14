// The ControlNet guide's browser side: queueing a preparation job, checking an
// existing map, watching the job's history, and the dialog itself.

import {
    CONTROL_MAPS_DIR, CONTROL_MIN_TRIM, CONTROL_MODE_HELP, CONTROL_MODE_LABELS,
    CONTROL_RESOLUTIONS, CONTROL_UPLOAD_OPTION, canApplyControlMap, changeControlSettings,
    controlChangedSettings, controlFileUrl, controlHistoryState, controlInputType,
    controlLength, controlLengthFact, controlMapCurrent, controlOutputTargets,
    controlPickerFiles, controlSettingsFingerprint, invalidateControlOperation,
} from "./control.js";
import { ASPECT_PRESETS, clamp01, dragCrop, normalizeCrop, normalizeTrim, setRectAspect } from "./edits.js";
import { isVideoFile } from "./drop.js";
import { takeControl } from "./references.js";
import { widgetByName } from "./widgets.js";

// `host` is what lives in refpack.js: ComfyUI's api, the set of live nodes, the node
// body's widget sync and repaint, and the shared modal and input-file helpers.
export function createControlUi(host) {
    const {
        api, liveNodes, syncReferencesWidget, renderNodeBody,
        createPlannerModal, appendFileOptions, apiListFiles, apiUpload,
    } = host;

    async function queueControlPreparation(node, plan, expectedFingerprint, operation = null) {
        if (controlLength(node) == null) {
            throw new Error("The linked length_seconds cannot be read before execution. Connect a PrimitiveFloat/PrimitiveNode duration or disconnect it and set length_seconds on the manager.");
        }
        if (controlInputType(node._mmrpRefs.control) === "map") {
            throw new Error("Check map validates an existing map without detector preparation.");
        }
        if ((operation != null && node._mmrpControlOperation !== operation)
            || controlSettingsFingerprint(node._mmrpRefs.control, controlLength(node)) !== expectedFingerprint) {
            throw new Error("Control settings changed while preparing the job. Generate again.");
        }
        if (!plan.prompt || !plan.workflow || plan.output_node_id == null || !plan.recipe) {
            throw new Error("The server returned an incomplete preparation workflow.");
        }
        const result = await api.queuePrompt(0, { output: plan.prompt, workflow: plan.workflow });
        if (!result?.prompt_id) throw new Error("Preparation was not accepted by ComfyUI.");
        // The request may finish after the user removes the control or changes its source.
        // The queue entry remains visible in ComfyUI; its result cannot restore those edits.
        if ((operation != null && node._mmrpControlOperation !== operation)
            || !node._mmrpRefs.control
            || controlSettingsFingerprint(node._mmrpRefs.control, controlLength(node)) !== expectedFingerprint) {
            return null;
        }
        const preparation = {
            prompt_id: result.prompt_id, output_node_id: String(plan.output_node_id),
            recipe: plan.recipe, settings_fingerprint: expectedFingerprint, status: "queued",
        };
        node._mmrpRefs.control.preparation = preparation;
        syncReferencesWidget(node);
        renderNodeBody(node);
        return preparation;
    }

    async function controlRequest(path, options) {
        const response = await api.fetchApi(`/minimax_refpack/control/${path}`, options);
        const data = await response.json().catch(() => ({}));
        if (!response.ok) throw new Error(data.error || `Control request failed: ${response.status}`);
        return data;
    }

    async function useExistingControlMap(node) {
        const lengthSeconds = controlLength(node);
        const control = takeControl(node._mmrpRefs.control);
        if (!control?.file || controlInputType(control) !== "map") {
            throw new Error("Choose an existing control map first.");
        }
        if (lengthSeconds == null) {
            throw new Error("The linked length_seconds cannot be read before execution. Use a PrimitiveFloat/PrimitiveNode duration or disconnect it and set the manager's length_seconds.");
        }
        const expected = controlSettingsFingerprint(control, lengthSeconds);
        const operation = invalidateControlOperation(node);
        const data = await controlRequest("import", {
            method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ control, length_seconds: lengthSeconds }),
        });
        if (node._mmrpControlOperation !== operation || !node._mmrpRefs.control
            || controlLength(node) == null
            || controlSettingsFingerprint(node._mmrpRefs.control, controlLength(node)) !== expected) return null;
        const imported = takeControl(data.control);
        if (!imported || controlSettingsFingerprint(imported, lengthSeconds) !== expected
            || !controlMapCurrent(imported, lengthSeconds)) {
            throw new Error("The server returned a control map for different settings. Check the map again.");
        }
        node._mmrpRefs.control = imported;
        syncReferencesWidget(node);
        renderNodeBody(node);
        return imported;
    }

    function watchControlPreparation(node) {
        const preparation = node._mmrpRefs.control?.preparation;
        if (!preparation || preparation.error || node._mmrpControlWatcher === preparation.prompt_id) return;
        node._mmrpControlWatcher = preparation.prompt_id;
        const active = () => liveNodes.has(node)
            && node._mmrpRefs.control?.preparation?.prompt_id === preparation.prompt_id;
        const fail = (message) => {
            if (!active()) return;
            node._mmrpRefs.control.preparation.error = message;
            syncReferencesWidget(node);
            renderNodeBody(node);
        };
        (async () => {
            let absent = 0;
            for (let poll = 0; poll < 1800 && active(); poll++) {
                try {
                    const response = await api.fetchApi(`/history/${encodeURIComponent(preparation.prompt_id)}`);
                    if (!response.ok) throw new Error(`History unavailable: ${response.status}`);
                    const result = controlHistoryState(await response.json(), preparation.prompt_id, preparation.output_node_id);
                    if (!active()) return;
                    if (result.state === "failed") { fail(result.message); return; }
                    if (result.state === "complete") {
                        if (!canApplyControlMap(node._mmrpRefs.control, preparation, result.map, controlLength(node))) {
                            fail("Settings changed or the returned recipe differs. Generate a new control map.");
                            return;
                        }
                        node._mmrpRefs.control.map = takeControl(result.map);
                        delete node._mmrpRefs.control.preparation;
                        syncReferencesWidget(node);
                        renderNodeBody(node);
                        return;
                    }
                    const queueResponse = await api.fetchApi("/queue");
                    if (queueResponse.ok) {
                        const queue = await queueResponse.json();
                        const running = (queue.queue_running || []).some((item) => item[1] === preparation.prompt_id);
                        const queued = (queue.queue_pending || []).some((item) => item[1] === preparation.prompt_id);
                        absent = running || queued ? 0 : absent + 1;
                        if (absent >= 3) { fail("Preparation left the queue without a result; it may have been cancelled or the server restarted."); return; }
                        if (active()) {
                            node._mmrpRefs.control.preparation.status = running ? "running" : "queued";
                            delete node._mmrpRefs.control.preparation.notice;
                        }
                    }
                } catch (error) {
                    if (active()) node._mmrpRefs.control.preparation.notice = error.message;
                }
                node._mmrpControlUpdate?.();
                await new Promise((resolve) => setTimeout(resolve, 1500));
            }
            if (active()) fail("Preparation is still pending after 45 minutes. Check ComfyUI history before generating another job.");
        })().catch((error) => fail(error.message)).finally(() => {
            if (node._mmrpControlWatcher === preparation.prompt_id) node._mmrpControlWatcher = null;
        });
    }

    // The ControlNet guide: the input clip with its crop and trim on the left, the control
    // type, resolution and resulting map on the right, then whether the map covers the
    // render length and reaches Apply MiniMax H3 Fun ControlNet. Every change is saved
    // immediately; a settings change drops the stored map (changeControlSettings), and the
    // dialog keeps showing the map it last saw, greyed, with what changed since.
    async function openControlModal(node) {
        const { modal, close } = createPlannerModal("ControlNet guide", "mmrp-control-modal");
        const tabStash = {};   // each tab's control as it was left, while the dialog is open
        const emptyControl = () => ({ file: "", mode: "pose", resolution: 512, trim: null, crop: null, map: null });
        let draft = takeControl(node._mmrpRefs.control) || emptyControl();
        const current = () => node._mmrpRefs.control || draft;
        let modes = [];
        let busy = false;
        let message = "Loading available processors…";
        let messageError = false;
        let sourceRevision = 0;
        let previous = null;      // {file, map} of the last map this dialog showed
        let undoRemove = null;
        const make = (tag, className = "", text = "") => {
            const element = document.createElement(tag);
            if (className) element.className = className;
            if (text) element.textContent = text;
            return element;
        };

        const hint = modal.appendChild(make("div", "mmrp-modal-hint"));

        // ---- tabs: two tasks, not a mode dropdown ----
        const tabs = modal.appendChild(make("div", "mmrp-ctl-tabs"));
        tabs.setAttribute("role", "tablist");
        const tab = (id, label) => {
            const button = tabs.appendChild(make("button", "mmrp-ctl-tab", label));
            button.setAttribute("role", "tab");
            button.onclick = () => {
                const from = controlInputType(current());
                if (from === id) return;
                if (current().file) tabStash[from] = takeControl(current());
                const back = tabStash[id];
                if (!back) { setControl({ input_type: id }); return; }
                delete back.preparation;
                invalidateControlOperation(node);
                draft = back;
                node._mmrpRefs.control = back;
                undoRemove = null;
                message = "";
                messageError = false;
                syncReferencesWidget(node);
                renderNodeBody(node);
                update();
            };
            return button;
        };
        const sourceTab = tab("source", "Make from a video");
        const mapTab = tab("map", "Use a map I already have");

        const columns = modal.appendChild(make("div", "mmrp-ctl-body"));
        const left = columns.appendChild(make("div", "mmrp-ctl-col"));
        const right = columns.appendChild(make("div", "mmrp-ctl-col"));

        // ---- left: file picker, framed clip, crop and trim ----
        const leftTitle = left.appendChild(make("div", "mmrp-ctl-title"));
        const source = left.appendChild(make("select", "mmrp-source-select"));
        const emptySourceOption = make("option");
        emptySourceOption.value = "";
        const uploadOption = make("option");
        uploadOption.value = CONTROL_UPLOAD_OPTION;
        source.appendChild(emptySourceOption);
        source.appendChild(uploadOption);
        const showAllLabel = left.appendChild(make("label", "mmrp-ctl-showall"));
        const showAll = showAllLabel.appendChild(make("input"));
        showAll.type = "checkbox";
        showAllLabel.appendChild(make("span", "", `Show all input videos, not just ${CONTROL_MAPS_DIR}/`));
        let videoFiles = null;    // the input listing, once it arrives
        let pickerKey = null;     // which list the picker holds: tab and show-all

        const stageBox = left.appendChild(make("div", "mmrp-ctl-stage-box"));
        const stageEmpty = stageBox.appendChild(make("div", "mmrp-ctl-empty"));
        const stage = stageBox.appendChild(make("div", "mmrp-ctl-stage"));
        const stageVideo = stage.appendChild(make("video"));
        stageVideo.muted = true;
        stageVideo.playsInline = true;
        stageVideo.preload = "auto";
        const cropLayer = stage.appendChild(make("div", "mmrp-crop-layer"));
        const cropRect = cropLayer.appendChild(make("div", "mmrp-crop-rect"));
        const cropSize = cropRect.appendChild(make("span", "mmrp-ctl-crop-size"));
        const cropHandles = {};
        for (const corner of ["nw", "ne", "sw", "se"]) {
            cropHandles[corner] = cropRect.appendChild(make("div", `mmrp-crop-handle mmrp-crop-${corner}`));
        }

        const aspectRow = left.appendChild(make("div", "mmrp-aspect-row"));
        aspectRow.appendChild(make("span", "mmrp-edit-label", "Crop"));
        const renderW = Number(widgetByName(node, "width")?.value) || 0;
        const renderH = Number(widgetByName(node, "height")?.value) || 0;
        const presets = [["Free", null]];
        if (renderW > 0 && renderH > 0) presets.push([`Render ${renderW}:${renderH}`, renderW / renderH]);
        presets.push(...ASPECT_PRESETS);
        const aspectButtons = presets.map(([label, ratio]) => {
            const button = aspectRow.appendChild(make("button", "mmrp-btn", label));
            button._mmrpRatio = ratio;
            return button;
        });
        const clearCrop = aspectRow.appendChild(make("button", "mmrp-btn mmrp-clear-btn", "Clear crop"));

        const trimRow = left.appendChild(make("div", "mmrp-trim-row"));
        const playClip = trimRow.appendChild(make("button", "mmrp-btn", "▶"));
        playClip.setAttribute("aria-label", "Play the trimmed clip");
        trimRow.appendChild(make("span", "mmrp-edit-label", "Trim"));
        const trimIn = trimRow.appendChild(make("input", "mmrp-trim-num"));
        const trimBar = trimRow.appendChild(make("div", "mmrp-trim-bar"));
        const trimSpan = trimBar.appendChild(make("div", "mmrp-trim-span"));
        const inHandle = trimBar.appendChild(make("div", "mmrp-trim-handle"));
        const outHandle = trimBar.appendChild(make("div", "mmrp-trim-handle"));
        const trimOut = trimRow.appendChild(make("input", "mmrp-trim-num"));
        const clearTrim = trimRow.appendChild(make("button", "mmrp-btn mmrp-clear-btn", "Clear trim"));
        for (const [input, label] of [[trimIn, "Trim in (seconds)"], [trimOut, "Trim out (seconds)"]]) {
            input.type = "number";
            input.step = "0.01";
            input.min = "0";
            input.setAttribute("aria-label", label);
        }
        const trimRead = left.appendChild(make("div", "mmrp-ctl-trim-read"));

        // ---- right: type, resolution, result ----
        right.appendChild(make("div", "mmrp-ctl-title", "Guide"));
        const typeLabel = right.appendChild(make("div", "mmrp-ctl-label"));
        const typeGroup = right.appendChild(make("div", "mmrp-ctl-seg"));
        typeGroup.setAttribute("role", "group");
        const typeButtons = Object.entries(CONTROL_MODE_LABELS).map(([id, label]) => {
            const button = typeGroup.appendChild(make("button", "mmrp-ctl-seg-btn", label));
            button.dataset.mode = id;
            button.onclick = () => setControl({ mode: id });
            return button;
        });
        const typeHelp = right.appendChild(make("div", "mmrp-ctl-help"));
        const resolutionLabel = right.appendChild(make("label", "mmrp-ctl-label"));
        const resolutionCaption = resolutionLabel.appendChild(make("span"));
        const resolution = resolutionLabel.appendChild(make("select", "mmrp-source-select"));
        for (const [size, label] of CONTROL_RESOLUTIONS) {
            const option = resolution.appendChild(make("option", "", label));
            option.value = String(size);
        }
        if (!CONTROL_RESOLUTIONS.some(([size]) => size === Number(draft.resolution))) {
            const option = resolution.appendChild(make("option", "", `${draft.resolution} px`));
            option.value = String(draft.resolution);
        }
        const resolutionHelp = right.appendChild(make("div", "mmrp-ctl-help"));

        const result = right.appendChild(make("div", "mmrp-ctl-result"));
        result.setAttribute("role", "status");
        const resultHead = result.appendChild(make("div", "mmrp-ctl-result-head"));
        const resultTitle = resultHead.appendChild(make("span"));
        const badge = resultHead.appendChild(make("span", "mmrp-ctl-badge"));
        const mapView = result.appendChild(make("div", "mmrp-ctl-mapview"));
        const mapVideo = mapView.appendChild(make("video"));
        mapVideo.muted = true;
        mapVideo.playsInline = true;
        mapVideo.preload = "metadata";
        mapVideo.controls = true;
        const mapEmpty = mapView.appendChild(make("div", "mmrp-ctl-empty"));
        const staleNote = mapView.appendChild(make("div", "mmrp-ctl-stale-note"));
        const resultMeta = result.appendChild(make("div", "mmrp-ctl-meta"));
        const resultNote = result.appendChild(make("div", "mmrp-ctl-help"));
        const resultActions = result.appendChild(make("div", "mmrp-ctl-actions"));
        const playBoth = resultActions.appendChild(make("button", "mmrp-btn", "▶ Play with source"));
        // A current receipt can still name a file that changed on disk; rendering rejects
        // it by hash, so rebuilding must not require touching a setting first.
        const redo = resultActions.appendChild(make("button", "mmrp-btn"));

        // ---- facts: render length, wiring ----
        const facts = modal.appendChild(make("div", "mmrp-ctl-facts"));
        const fact = () => {
            const row = facts.appendChild(make("div", "mmrp-ctl-fact"));
            row._icon = row.appendChild(make("span", "mmrp-ctl-fact-icon"));
            row._head = row.appendChild(make("strong"));
            row._detail = row.appendChild(make("span", "mmrp-ctl-fact-detail"));
            return row;
        };
        const lengthFact = fact();
        const wiringFact = fact();
        const setFact = (row, level, head, detail) => {
            row.classList.toggle("mmrp-ctl-fact-warn", level === "warn");
            row._icon.textContent = level === "warn" ? "!" : "✓";
            row._head.textContent = head;
            row._detail.textContent = detail ? ` ${detail}` : "";
        };

        // ---- footer: remove sits apart from the primary action ----
        const footer = modal.appendChild(make("div", "mmrp-modal-footer mmrp-ctl-footer"));
        const remove = footer.appendChild(make("button", "mmrp-btn mmrp-ctl-remove", "Remove guide"));
        const toast = footer.appendChild(make("span", "mmrp-ctl-toast"));
        const toastText = toast.appendChild(make("span"));
        const undo = toast.appendChild(make("button", "mmrp-ctl-link", "Undo"));
        footer.appendChild(make("span", "mmrp-ctl-spacer"));
        footer.appendChild(make("button", "mmrp-btn", "Close")).onclick = close;
        const primary = footer.appendChild(make("button", "mmrp-btn mmrp-btn-primary"));

        const uploadInput = modal.appendChild(make("input"));
        uploadInput.type = "file";
        uploadInput.accept = "video/*,.mkv,.webm,.avi,.mov,.mp4";
        uploadInput.hidden = true;

        // ---- state ----
        let crop = [0, 0, 1, 1];
        let trim = null;            // null = whole clip
        let ratio = null;
        let aspectChoice = null;  // the preset button last clicked; equal ratios stay distinct
        let editing = false;        // a crop or trim drag owns the local values
        let endDrag = null;         // removes a live drag's window listeners
        let playing = false;
        let previewFailed = false;
        stageVideo.addEventListener("error", () => { previewFailed = !!stageVideo.getAttribute?.("src"); update(); });
        const duration = () => (Number.isFinite(stageVideo.duration) && stageVideo.duration > 0 ? stageVideo.duration : null);
        const mediaSize = () => [stageVideo.videoWidth || 0, stageVideo.videoHeight || 0];

        const setControl = (changes) => {
            const before = node._mmrpRefs.control || draft;
            const next = changeControlSettings(before, changes);
            if (controlSettingsFingerprint(next) !== controlSettingsFingerprint(before)) {
                invalidateControlOperation(node);
            }
            draft = next;
            node._mmrpRefs.control = next.file ? next : null;
            message = "";
            messageError = false;
            undoRemove = null;
            syncReferencesWidget(node);
            renderNodeBody(node);
            update();
        };

        const stopBoth = () => {
            playing = false;
            stageVideo.pause();
            mapVideo.pause();
            playBoth.textContent = "▶ Play with source";
            playClip.textContent = "▶";
        };
        playClip.onclick = () => {
            if (!stageVideo.paused) { stopBoth(); return; }
            stopBoth();
            stageVideo.currentTime = trim?.[0] || 0;
            playClip.textContent = "■";
            Promise.resolve(stageVideo.play()).catch(() => stopBoth());
        };
        stageVideo.addEventListener("pause", () => { playClip.textContent = "▶"; });
        playBoth.onclick = () => {
            if (playing) { stopBoth(); return; }
            playing = true;
            playBoth.textContent = "■ Stop";
            stageVideo.currentTime = trim?.[0] || 0;
            mapVideo.currentTime = 0;
            Promise.allSettled([stageVideo.play(), mapVideo.play()]);
        };
        mapVideo.addEventListener("ended", () => { if (playing) stopBoth(); });
        stageVideo.addEventListener("timeupdate", () => {
            if (trim && !stageVideo.paused && stageVideo.currentTime >= trim[1]) {
                stopBoth();
                stageVideo.currentTime = trim[0];
            }
        });
        stageVideo.addEventListener("loadedmetadata", () => {
            stageVideo.currentTime = trim?.[0] || 0;
            update();
        });
        modal.addEventListener("mmrp-modal-close", () => {
            endDrag?.();
            stopBoth();
            showVideo(stageVideo, "");
            showVideo(mapVideo, "");
            if (node._mmrpControlUpdate === update) node._mmrpControlUpdate = null;
        });

        // ---- crop ----
        const syncCrop = () => {
            cropRect.style.left = `${crop[0] * 100}%`;
            cropRect.style.top = `${crop[1] * 100}%`;
            cropRect.style.width = `${crop[2] * 100}%`;
            cropRect.style.height = `${crop[3] * 100}%`;
            const [width, height] = mediaSize();
            cropSize.textContent = width && height
                ? `${Math.round(crop[2] * width)} × ${Math.round(crop[3] * height)}` : "";
            clearCrop.disabled = busy || normalizeCrop(crop) === null;
            for (const button of aspectButtons) button.classList.toggle("mmrp-active", button === aspectChoice && ratio !== null);
        };
        const dragPointer = (event, onMove, onDone) => {
            event.preventDefault();
            event.stopPropagation();
            if (busy || !current().file) return;
            stopBoth();
            editing = true;
            let moved = false;
            const move = (moveEvent) => { moved = true; onMove(moveEvent); };
            const end = () => {
                window.removeEventListener("mousemove", move);
                window.removeEventListener("mouseup", up);
                editing = false;
                endDrag = null;
            };
            // Escape mid-drag closes the dialog; the release must not save an edit nobody sees.
            const up = () => {
                end();
                if (!moved) { update(); return; }
                // The click after a release outside the modal lands on the backdrop and closes it.
                const swallow = (clickEvent) => clickEvent.stopPropagation();
                window.addEventListener("click", swallow, true);
                setTimeout(() => window.removeEventListener("click", swallow, true), 0);
                if (modal.isConnected) onDone();
            };
            endDrag = end;
            window.addEventListener("mousemove", move);
            window.addEventListener("mouseup", up);
        };
        const startCropDrag = (event, mode) => {
            const box = cropLayer.getBoundingClientRect();
            if (!box.width || !box.height) return;
            const from = { x: event.clientX, y: event.clientY, crop: crop.slice() };
            const [width, height] = mediaSize();
            dragPointer(event, (moveEvent) => {
                crop = dragCrop(from.crop, mode, (moveEvent.clientX - from.x) / box.width,
                    (moveEvent.clientY - from.y) / box.height, ratio, width, height);
                syncCrop();
            }, () => setControl({ crop: normalizeCrop(crop) }));
        };
        cropRect.addEventListener("mousedown", (event) => startCropDrag(event, "move"));
        for (const [corner, handle] of Object.entries(cropHandles)) {
            handle.addEventListener("mousedown", (event) => startCropDrag(event, corner));
        }
        for (const button of aspectButtons) {
            button.onclick = () => {
                ratio = button._mmrpRatio;
                aspectChoice = button;
                const [width, height] = mediaSize();
                if (ratio && width && height) setControl({ crop: normalizeCrop(setRectAspect(crop, ratio, width, height)) });
                else syncCrop();
            };
        }
        clearCrop.onclick = () => {
            ratio = null;
            aspectChoice = null;
            setControl({ crop: null });
        };

        // ---- trim ----
        const r2 = (value) => Math.round(value * 100) / 100;
        const syncTrim = () => {
            const total = duration();
            const span = trim || (total ? [0, total] : null);
            trimIn.disabled = trimOut.disabled = busy || !total;
            clearTrim.disabled = busy || !trim;
            if (!total || !span) {
                trimSpan.style.width = "0";
                trimRead.textContent = !current().file ? ""
                    : previewFailed ? "This browser can't play the clip, so crop and trim can't be shown." : "Loading clip…";
                return;
            }
            if (document.activeElement !== trimIn) trimIn.value = span[0].toFixed(2);
            if (document.activeElement !== trimOut) trimOut.value = span[1].toFixed(2);
            trimSpan.style.left = `${(span[0] / total) * 100}%`;
            trimSpan.style.width = `${((span[1] - span[0]) / total) * 100}%`;
            inHandle.style.left = `calc(${(span[0] / total) * 100}% - 5px)`;
            outHandle.style.left = `calc(${(span[1] / total) * 100}% - 5px)`;
            trimRead.textContent = `${(span[1] - span[0]).toFixed(2)} s used of ${total.toFixed(2)} s`;
        };
        const setTrimLocal = (start, end) => {
            const total = duration();
            if (!total) return;
            start = clamp01(r2(start), 0, Math.max(0, r2(total) - CONTROL_MIN_TRIM));
            end = clamp01(r2(end), start + CONTROL_MIN_TRIM, r2(total));
            trim = [start, end];
            syncTrim();
        };
        const commitTrim = () => {
            setControl({ trim: normalizeTrim(trim, duration()) });
            const span = trim || [0, duration() || 0];
            trimIn.value = span[0].toFixed(2);
            trimOut.value = span[1].toFixed(2);
        };
        const dragTrim = (which) => (event) => {
            const box = trimBar.getBoundingClientRect();
            const total = duration();
            if (!box.width || !total) return;
            const span = trim || [0, total];
            trim = span.slice();
            dragPointer(event, (moveEvent) => {
                const time = clamp01((moveEvent.clientX - box.left) / box.width, 0, 1) * total;
                if (which === "in") setTrimLocal(time, trim[1]);
                else setTrimLocal(trim[0], time);
                stageVideo.currentTime = time;
            }, commitTrim);
        };
        inHandle.addEventListener("mousedown", dragTrim("in"));
        outHandle.addEventListener("mousedown", dragTrim("out"));
        trimIn.addEventListener("change", () => {
            const total = duration();
            if (!total) return;
            setTrimLocal(parseFloat(trimIn.value) || 0, (trim || [0, total])[1]);
            commitTrim();
        });
        trimOut.addEventListener("change", () => {
            const total = duration();
            if (!total) return;
            setTrimLocal((trim || [0, total])[0], parseFloat(trimOut.value) || total);
            commitTrim();
        });
        clearTrim.onclick = () => setControl({ trim: null });

        // ---- file selection, upload and drop ----
        source.onchange = () => {
            if (source.value === CONTROL_UPLOAD_OPTION) {
                source.value = current().file;
                uploadInput.click();
                return;
            }
            setControl({ file: source.value, trim: null, crop: null });
        };
        resolution.onchange = () => setControl({ resolution: Number(resolution.value) });
        showAll.onchange = () => update();
        const fillPicker = (imported) => {
            const key = `${imported}:${showAll.checked}`;
            if (videoFiles === null || key === pickerKey) return;
            pickerKey = key;
            appendFileOptions(source, controlPickerFiles(videoFiles, imported, showAll.checked), current().file);
            source.prepend(emptySourceOption);
            source.appendChild(uploadOption);
        };
        const addSourceOption = (file) => {
            if ([...source.options].some((item) => item.value === file)) return;
            const option = make("option", "", file);
            option.value = file;
            source.insertBefore(option, uploadOption);
        };
        const uploadSource = async (files) => {
            const list = Array.from(files || []);
            const imported = controlInputType(current()) === "map";
            if (list.length !== 1 || !isVideoFile(list[0].name)) {
                message = imported ? "Choose or drop one control map video." : "Choose or drop one source video.";
                messageError = true;
                update();
                return;
            }
            busy = true;
            message = imported ? "Uploading control map…" : "Uploading source video…";
            messageError = false;
            const operation = invalidateControlOperation(node);
            const expected = controlSettingsFingerprint(current(), controlLength(node));
            update();
            try {
                const file = await apiUpload(list[0], imported ? CONTROL_MAPS_DIR : "");
                if (videoFiles && !videoFiles.includes(file)) videoFiles.push(file);
                addSourceOption(file);
                if (node._mmrpControlOperation !== operation
                    || controlSettingsFingerprint(current(), controlLength(node)) !== expected) {
                    message = "The file was uploaded, but the settings changed while uploading. Select the uploaded file to use it.";
                    return;
                }
                sourceRevision += 1;
                const next = changeControlSettings(current(), { file, trim: null, crop: null, map: null });
                // Upload overwrites the managed file; the same filename still needs a new map.
                delete next.preparation;
                draft = next;
                node._mmrpRefs.control = next;
                message = "";
                undoRemove = null;
                syncReferencesWidget(node);
                renderNodeBody(node);
            } catch (error) {
                message = error.message;
                messageError = true;
            } finally {
                busy = false;
                uploadInput.value = "";
                update();
            }
        };
        uploadInput.onchange = () => uploadSource(uploadInput.files);
        modal.ondragover = (event) => { event.preventDefault(); event.stopPropagation(); };
        modal.ondrop = (event) => {
            event.preventDefault();
            event.stopPropagation();
            if (!busy) uploadSource(event.dataTransfer.files);
        };

        // ---- actions ----
        remove.onclick = () => {
            const removed = takeControl(node._mmrpRefs.control);
            if (!removed) return;
            const hadPreparation = !!removed.preparation;
            stopBoth();
            invalidateControlOperation(node);
            node._mmrpRefs.control = null;
            draft = emptyControl();
            previous = null;
            message = "";
            messageError = false;
            syncReferencesWidget(node);
            renderNodeBody(node);
            // Undo restores the settings and any finished map; a preparation still queued
            // was orphaned by the invalidation above and must be generated again.
            delete removed.preparation;
            undoRemove = { control: removed, text: hadPreparation
                ? "Guide removed. The queued preparation stays in ComfyUI's queue." : "Guide removed." };
            update();
        };
        undo.onclick = () => {
            if (!undoRemove) return;
            const { control } = undoRemove;
            undoRemove = null;
            draft = control;
            node._mmrpRefs.control = control;
            syncReferencesWidget(node);
            renderNodeBody(node);
            update();
        };
        const prepare = async () => {
            const control = current();
            const imported = controlInputType(control) === "map";
            busy = true;
            stopBoth();
            message = imported ? "Checking the control map…" : "Preparing the control job…";
            messageError = false;
            update();
            try {
                if (controlLength(node) == null) throw new Error(controlLengthFact(null).detail);
                if (imported) {
                    const accepted = await useExistingControlMap(node);
                    message = accepted ? "" : "The check finished, but the settings changed meanwhile. Check the map again.";
                    return;
                }
                const operation = invalidateControlOperation(node);
                const prepared = takeControl(node._mmrpRefs.control);
                const expected = controlSettingsFingerprint(prepared, controlLength(node));
                const plan = await controlRequest("prepare", {
                    method: "POST", headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ control: prepared, length_seconds: controlLength(node) }),
                });
                const pending = await queueControlPreparation(node, plan, expected, operation);
                message = pending ? "" : "The preparation was queued, but its result will be ignored because the settings changed.";
            } catch (error) {
                message = error.message || error.response?.error?.message || (imported
                    ? "The control map check failed. Check it again."
                    : "Preparation could not be queued. Check ComfyUI's queue before retrying.");
                messageError = true;
            } finally {
                busy = false;
                update();
            }
        };
        primary.onclick = () => (controlMapCurrent(current(), controlLength(node)) ? close() : prepare());
        redo.onclick = prepare;

        const showVideo = (video, url) => {
            if (video.dataset.source === url) return;
            video.pause();
            video.dataset.source = url;
            if (url) video.src = url;
            else { video.removeAttribute("src"); video.load(); }
        };

        // ---- one render pass over the whole dialog ----
        const update = () => {
            if (!modal.isConnected) {
                stopBoth();
                if (node._mmrpControlUpdate === update) node._mmrpControlUpdate = null;
                return;
            }
            const control = current();
            const imported = controlInputType(control) === "map";
            const lengthSeconds = controlLength(node);
            const pending = !!control.preparation && !control.preparation.error;
            const ready = controlMapCurrent(control, lengthSeconds);
            if (control.map?.file) previous = { file: control.file, map: takeControl(control.map) };
            // A map from the other tab is not "out of date", just not this tab's map.
            const stale = !ready && !pending && !!previous?.map?.file && !!control.file
                && (previous.map.recipe?.input_type || "source") === controlInputType(control);
            const capability = modes.find((item) => item.id === control.mode);
            const modeLabel = CONTROL_MODE_LABELS[control.mode] || "Control";

            hint.textContent = imported
                ? "Check a saved Pose, Depth or Edges map without a detector or queue job. Rendering normalizes it to 24 fps and applies the crop, trim and resolution set here."
                : "A Pose, Depth or Edges video that steers the render. Generating queues only control preparation; rendering stays a separate Run.";
            sourceTab.setAttribute("aria-selected", String(!imported));
            mapTab.setAttribute("aria-selected", String(imported));
            sourceTab.classList.toggle("mmrp-active", !imported);
            mapTab.classList.toggle("mmrp-active", imported);
            sourceTab.disabled = mapTab.disabled = busy;

            leftTitle.textContent = imported ? "Map file" : "Source video";
            source.setAttribute("aria-label", leftTitle.textContent);
            emptySourceOption.textContent = imported ? "Choose a control map video…" : "Choose a source video…";
            uploadOption.textContent = imported ? `Upload a map to ${CONTROL_MAPS_DIR}/…` : "Upload a video…";
            showAllLabel.hidden = !imported;
            fillPicker(imported);
            if (control.file) addSourceOption(control.file);
            source.value = control.file;
            source.disabled = busy;
            stageEmpty.hidden = !!control.file;
            stageEmpty.textContent = imported ? "Choose or drop a control map video." : "Choose or drop a source video.";
            stage.hidden = !control.file;
            const stageUrl = control.file ? `${controlFileUrl(control.file)}&revision=${sourceRevision}` : "";
            if (stageVideo.dataset.source !== stageUrl) previewFailed = false;
            showVideo(stageVideo, stageUrl);
            if (!editing) {
                crop = Array.isArray(control.crop) ? control.crop.slice() : [0, 0, 1, 1];
                trim = Array.isArray(control.trim) ? control.trim.slice() : null;
            }
            syncCrop();
            syncTrim();
            for (const button of aspectButtons) button.disabled = busy || !control.file;
            playClip.disabled = busy || !duration();

            typeLabel.textContent = imported ? "This map is a" : "Control type";
            typeGroup.setAttribute("aria-label", typeLabel.textContent);
            for (const button of typeButtons) {
                const id = button.dataset.mode;
                const option = modes.find((item) => item.id === id);
                const available = imported || !!option?.available;
                button.disabled = busy || !available;
                button.title = available ? "" : option?.reason || "Processor unavailable";
                button.textContent = `${option?.label || CONTROL_MODE_LABELS[id]}${available || !modes.length ? "" : " (unavailable)"}`;
                button.setAttribute("aria-pressed", String(control.mode === id));
                button.classList.toggle("mmrp-active", control.mode === id);
            }
            typeHelp.textContent = imported
                ? "Say what the file holds; it is checked, not regenerated."
                : `${CONTROL_MODE_HELP[control.mode] || ""}${control.mode === "edges" ? "" : " First use may download the detector weights."}`;
            resolutionCaption.textContent = imported ? "Map resolution" : "Resolution";
            resolution.setAttribute("aria-label", resolutionCaption.textContent);
            resolution.value = String(control.resolution);
            resolution.disabled = busy;
            resolutionHelp.textContent = imported
                ? "Short side the map is resized to for the render."
                : "Short side of the map, and the size the detector works at.";

            // Result panel: no file, working, preparing, failed, ready, out of date, or no map yet.
            const failure = control.preparation?.error || "";
            const unavailable = !imported && modes.length && !capability?.available
                ? capability?.reason || "Processor unavailable" : "";
            let state;
            if (!control.file) state = ["none", "No file", imported ? "Choose a map file on the left." : "Choose a source video on the left."];
            else if (busy) state = ["run", "Working", message];
            else if (pending) state = ["run", control.preparation.status === "running" ? "Running" : "Queued", "The map appears here when the job finishes."];
            else if (failure) state = ["error", "Failed", failure];
            else if (ready) state = ["ok", imported ? "Checked" : "Ready", ""];
            else if (stale) state = ["warn", "Out of date", ""];
            else if (unavailable) state = ["error", "Unavailable", unavailable];
            else state = ["none", "No map yet", imported ? "Check the map before rendering." : "Generate a map for these settings."];
            const [level, badgeText, emptyText] = state;
            resultTitle.textContent = imported ? "Map check" : `${modeLabel} map`;
            badge.textContent = badgeText;
            badge.dataset.level = level;
            const shownMap = !imported && (ready || stale) ? (ready ? control.map : previous.map) : null;
            mapView.hidden = imported && (ready || stale);
            mapView.classList.toggle("mmrp-ctl-stale", stale);
            showVideo(mapVideo, shownMap ? controlFileUrl(shownMap.file) : "");
            mapVideo.hidden = !shownMap;
            mapEmpty.hidden = !!shownMap;
            mapEmpty.textContent = emptyText;
            const changed = stale ? controlChangedSettings(previous, control, lengthSeconds) : [];
            staleNote.hidden = !stale || imported;
            staleNote.textContent = `You changed ${changed.length ? changed.join(" and ") : "a setting"} after this map was made. It won't be used until you regenerate.`;
            const map = ready ? control.map : stale ? previous.map : null;
            resultMeta.hidden = !map && !pending;
            resultMeta.textContent = pending
                ? `Job ${control.preparation.prompt_id}${control.preparation.notice ? ` — ${control.preparation.notice}` : ""}`
                : map ? `${stale ? "Previous map: " : ""}${map.frame_count} frames · ${map.fps} fps · ${map.width} × ${map.height}` : "";
            const notes = [];
            if (pending) notes.push("You can close this dialog; the job keeps running in ComfyUI's queue.");
            if (imported && ready) notes.push("The preview on the left is the original file; rendering normalizes it to these settings.");
            if (imported && stale) notes.push(`You changed ${changed.length ? changed.join(" and ") : "a setting"} after the check. Check the map again before rendering.`);
            if (unavailable && (stale || failure)) notes.push(unavailable);
            if (message && !busy) notes.push(message);
            resultNote.textContent = notes.join(" ");
            resultNote.hidden = !notes.length;
            resultNote.classList.toggle("mmrp-ctl-error", messageError && !busy);
            resultActions.hidden = !ready || pending;
            playBoth.hidden = imported;
            playBoth.disabled = busy;
            redo.textContent = imported ? "Check again" : "Regenerate";
            redo.disabled = busy || lengthSeconds == null || (!imported && !capability?.available);

            // Facts under the columns.
            const total = duration();
            const span = trim ? (total ? Math.min(trim[1], total) : trim[1]) - trim[0] : total;
            const lengthLine = controlLengthFact(lengthSeconds, span);
            setFact(lengthFact, lengthLine.level, lengthLine.head, lengthLine.detail);
            const targets = controlOutputTargets(node);
            const applied = targets.find((target) => target.type === "MiniMaxH3FunControlNetApply");
            if (!targets.length) {
                setFact(wiringFact, "warn", "control_video isn't connected.",
                    "This map won't affect the render until it feeds Apply MiniMax H3 Fun ControlNet.");
            } else if (applied) {
                setFact(wiringFact, "ok", `control_video → ${applied.title}.`,
                    "Strength and start/end are set on that node.");
            } else {
                // A reroute or subgraph may still reach it; say what is seen, not that it's wrong.
                setFact(wiringFact, "warn", `control_video → ${targets.map((target) => target.title).join(", ")}.`,
                    "Not directly Apply MiniMax H3 Fun ControlNet; make sure the map reaches it.");
            }

            // Footer.
            remove.disabled = busy || !node._mmrpRefs.control;
            toast.hidden = !undoRemove;
            toastText.textContent = undoRemove?.text || "";
            primary.textContent = ready ? "Done"
                : pending ? "Preparing…"
                    : imported ? (stale || failure ? "Check again" : "Check map")
                        : stale || failure ? "Regenerate" : `Generate ${modeLabel.toLowerCase()} map`;
            primary.disabled = !ready && (busy || pending || lengthSeconds == null || !control.file
                || (!imported && !capability?.available));
        };
        node._mmrpControlUpdate = update;
        update();
        const listed = await Promise.allSettled([apiListFiles("video"), controlRequest("capabilities")]);
        if (!modal.isConnected) return;
        videoFiles = listed[0].status === "fulfilled" ? listed[0].value : [];
        if (listed[1].status === "fulfilled") modes = listed[1].value.modes || [];
        else { message = listed[1].reason.message; messageError = true; }
        if (listed[0].status === "rejected") {
            message = `Video listing failed: ${listed[0].reason.message}. You can still upload a video.`;
            messageError = true;
        } else if (listed[1].status === "fulfilled" || controlInputType(current()) === "map") {
            message = "";
            messageError = false;
        }
        update();
        watchControlPreparation(node);
    }

    return {
        controlRequest, queueControlPreparation, useExistingControlMap,
        watchControlPreparation, openControlModal,
    };
}
