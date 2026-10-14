// The ControlNet guide's settings, receipts and readiness. Preparation is a separate
// queue entry; control_ui.js queues it and watches it, these only decide.

import { takeControl } from "./references.js";
import { widgetByName } from "./widgets.js";

export function controlInputType(control) {
    return control?.input_type === "map" ? "map" : "source";
}

export function controlSettings(control, lengthSeconds = 0) {
    return {
        input_type: controlInputType(control),
        file: control?.file || "", mode: control?.mode || "pose",
        resolution: Number(control?.resolution ?? 512),
        trim: control?.trim || null, crop: control?.crop || null,
        length_seconds: Number(lengthSeconds) || 0,
    };
}

export function ensureControlOutput(node) {
    // LiteGraph may restore the saved output array after node creation. Append the
    // new socket to older managers without moving any serialized link's slot.
    if (!(node.outputs || []).some((output) => output.name === "control_video")) {
        node.addOutput("control_video", "IMAGE");
    }
}

export function controlSettingsFingerprint(control, lengthSeconds = 0) {
    return JSON.stringify(controlSettings(control, lengthSeconds));
}

export function controlSettingsMatch(control, lengthSeconds, fingerprint) {
    try {
        const settings = JSON.parse(fingerprint);
        // Pending source jobs saved before input_type existed retain their identity.
        settings.input_type ??= "source";
        return controlRecipeFingerprint(settings)
            === controlRecipeFingerprint(controlSettings(control, lengthSeconds));
    } catch (_) { return false; }
}

export function invalidateControlOperation(node) {
    node._mmrpControlOperation = (node._mmrpControlOperation || 0) + 1;
    return node._mmrpControlOperation;
}

export function changeControlSettings(control, changes) {
    const next = { ...(takeControl(control) || {
        file: "", mode: "pose", resolution: 512, trim: null, crop: null, map: null,
    }), ...changes };
    if (controlSettingsFingerprint(next) !== controlSettingsFingerprint(control)) {
        next.map = null;
        delete next.preparation;
    }
    return next;
}

export function controlLength(node) {
    const upstreamAt = (current, index) => {
        const direct = current.getInputNode?.(index);
        if (direct) return direct;
        const graph = current.graph;
        const linkId = current.inputs?.[index]?.link;
        const link = graph?.links?.get?.(linkId) || graph?.links?.[linkId];
        return link ? graph.getNodeById?.(link.origin_id) : null;
    };
    const seen = new Set();
    const constant = (current) => {
        if (!current || seen.has(current)) return null;
        seen.add(current);
        const type = current.comfyClass || current.type;
        if (type === "Reroute") {
            const index = current.inputs?.findIndex((input) => input.link != null) ?? -1;
            return index >= 0 ? constant(upstreamAt(current, index)) : null;
        }
        if (!["PrimitiveFloat", "PrimitiveInt", "PrimitiveNode"].includes(type)) return null;
        const linkedValue = current.inputs?.findIndex((input) => input.name === "value" && input.link != null) ?? -1;
        if (linkedValue >= 0) return constant(upstreamAt(current, linkedValue));
        const value = Number(widgetByName(current, "value")?.value);
        return Number.isFinite(value) && value >= 0 ? value : null;
    };
    const linked = node.inputs?.findIndex((input) => input.name === "length_seconds" && input.link != null) ?? -1;
    if (linked >= 0) return constant(upstreamAt(node, linked));
    const value = Number(widgetByName(node, "length_seconds")?.value ?? 0);
    return Number.isFinite(value) && value >= 0 ? value : null;
}

export function controlMapCurrent(control, lengthSeconds) {
    if (lengthSeconds == null) return false;
    const recipe = control?.map?.recipe;
    if (!control?.map?.file || !recipe) return false;
    const expected = controlSettings(control, lengthSeconds);
    if ((recipe.input_type || "source") !== expected.input_type) return false;
    if (expected.input_type === "map" && control.map.file !== control.file) return false;
    return ["mode", "resolution", "crop", "trim", "length_seconds"].every(
        (key) => JSON.stringify(recipe[key] ?? null) === JSON.stringify(expected[key])
    );
}

export function controlRecipeFingerprint(recipe) {
    const ordered = (value) => Array.isArray(value) ? value.map(ordered)
        : value && typeof value === "object"
            ? Object.fromEntries(Object.keys(value).sort().map((key) => [key, ordered(value[key])]))
            : value;
    return JSON.stringify(ordered(recipe));
}

export function canApplyControlMap(control, preparation, map, lengthSeconds) {
    return lengthSeconds != null && !!control && !!map?.file
        && control.preparation?.prompt_id === preparation.prompt_id
        && controlSettingsMatch(control, lengthSeconds, preparation.settings_fingerprint)
        && controlRecipeFingerprint(map.recipe) === controlRecipeFingerprint(preparation.recipe);
}

export function controlHistoryState(history, promptId, outputNodeId) {
    const entry = history?.[promptId];
    if (!entry) return { state: "pending", map: null, message: "" };
    const status = entry.status || {};
    if (status.status_str === "error") {
        const error = (status.messages || []).find(([type]) =>
            type === "execution_error" || type === "execution_interrupted");
        return { state: "failed", map: null,
            message: error?.[1]?.exception_message || "Control preparation failed or was interrupted." };
    }
    if (!status.completed) return { state: "pending", map: null, message: "" };
    const map = entry.outputs?.[String(outputNodeId)]?.control_map?.[0];
    return map?.file
        ? { state: "complete", map, message: "" }
        : { state: "failed", map: null, message: "Preparation completed without a control map." };
}

export const CONTROL_MODE_LABELS = { pose: "Pose", depth: "Depth", edges: "Edges" };

// controls.target_frames: 24 fps, rounded UP to MiniMax's 17k+5 grid; 0 = clip-driven.
export function controlTargetFrames(lengthSeconds) {
    const seconds = Number(lengthSeconds);
    if (!Number.isFinite(seconds) || seconds <= 0) return null;
    // Python's round() is half-to-even: 0.9375 s is 22.5 frames and rounds to 22, not 23.
    const raw = seconds * 24;
    const floor = Math.floor(raw);
    const rounded = raw - floor !== 0.5 ? Math.round(raw) : floor % 2 === 0 ? floor : floor + 1;
    const count = Math.max(5, rounded);
    return 5 + Math.floor((count - 5 + 16) / 17) * 17;
}

// The dialog's length line. `spanSeconds` is the trimmed clip (null until its
// duration is known). A short clip holds its last frame; a long one is cut.
export function controlLengthFact(lengthSeconds, spanSeconds) {
    if (lengthSeconds == null) {
        return { level: "warn", head: "Render length unknown.",
            detail: "The linked length_seconds cannot be read before the graph runs. Feed it from a PrimitiveFloat or PrimitiveNode, or disconnect it and set length_seconds on the manager." };
    }
    if (lengthSeconds > 0 && lengthSeconds < 5 / 24) {
        return { level: "warn", head: `Render ${Number(lengthSeconds).toFixed(2)} s is too short.`,
            detail: "A control map needs at least 5 frames at 24 fps (0.21 s)." };
    }
    const frames = controlTargetFrames(lengthSeconds);
    if (frames == null) {
        return { level: "ok", head: "length_seconds is 0.",
            detail: "The map keeps the clip's own length, cut down to MiniMax's 17k+5 frame grid." };
    }
    const head = `Render ${Number(lengthSeconds).toFixed(1)} s → ${frames} frames at 24 fps.`;
    if (spanSeconds == null || !(spanSeconds > 0)) return { level: "ok", head, detail: "" };
    const spanFrames = Math.round(spanSeconds * 24);
    if (spanFrames < frames) {
        return { level: "warn", head, detail: `The clip gives ${spanSeconds.toFixed(2)} s; the map holds its last frame for the final ${((frames - spanFrames) / 24).toFixed(2)} s.` };
    }
    const extra = (spanFrames - frames) / 24;
    return { level: "ok", head, detail: extra >= 0.05
        ? `The last ${extra.toFixed(2)} s of the clip are not used.`
        : "The clip covers it." };
}

// Nodes fed by the appended control_video output, as {type, title}.
export function controlOutputTargets(node) {
    const output = (node.outputs || []).find((item) => item.name === "control_video");
    const graph = node.graph;
    return (output?.links || []).map((id) => {
        const link = graph?.links?.get?.(id) || graph?.links?.[id];
        const target = link ? graph?.getNodeById?.(link.target_id) : null;
        return target ? { type: target.comfyClass || target.type, title: target.title || target.type } : null;
    }).filter(Boolean);
}

// What differs between the settings a map was made for and the current ones, worded
// for "You changed … after this map was made".
export function controlChangedSettings(previous, control, lengthSeconds) {
    const recipe = previous?.map?.recipe;
    if (!recipe) return [];
    const now = controlSettings(control, lengthSeconds);
    const same = (key) => JSON.stringify(recipe[key] ?? null) === JSON.stringify(now[key] ?? null);
    const changed = [];
    if ((previous.file || "") !== now.file) changed.push("the file");
    if (!same("mode")) changed.push("the control type");
    if (!same("resolution")) changed.push("the resolution");
    if (!same("crop")) changed.push("the crop");
    if (!same("trim")) changed.push("the trim");
    if (lengthSeconds != null && !same("length_seconds")) changed.push("the render length");
    return changed;
}

// The map tab's picker. Generated maps live in control_maps/, so only those are listed
// until the user asks for every input video; the source tab always lists them all.
export const CONTROL_MAPS_DIR = "control_maps";
export function controlPickerFiles(files, imported, showAll) {
    return imported && !showAll ? files.filter((file) => file.startsWith(`${CONTROL_MAPS_DIR}/`)) : files;
}

export function controlFileUrl(file) {
    const parts = String(file || "").split("/");
    const filename = parts.pop();
    const query = new URLSearchParams({ filename, type: "input" });
    if (parts.length) query.set("subfolder", parts.join("/"));
    return `/view?${query}`;
}

export const CONTROL_MODE_HELP = {
    pose: "Body skeleton from DWPose. Keeps the performance; clothes, face and setting come from the references.",
    depth: "Distance from camera from Depth Anything V2. Keeps layout and camera move; surfaces and light stay free.",
    edges: "Canny outlines, built into ComfyUI. Follows shapes closely.",
};
export const CONTROL_RESOLUTIONS = [
    [256, "256 px · fastest"], [512, "512 px · default"], [768, "768 px · finer"], [1024, "1024 px · finest, slowest"],
];
export const CONTROL_UPLOAD_OPTION = "\u0000upload";
// The control decoder needs at least five frames at 24 fps (0.21 s); 0.25 s keeps a
// frame of margin against how the trimmed window's edges round.
export const CONTROL_MIN_TRIM = 0.25;

// The node's ControlNet button says whether the guide will reach the render: no map,
// preparing, ready, or ready but control_video wired to nothing.
export function syncControlsButton(node) {
    const button = node._mmrpBody?.controlsButton;
    if (!button) return;
    const control = node._mmrpRefs.control;
    const imported = controlInputType(control) === "map";
    const pending = control?.preparation && !control.preparation.error;
    const ready = controlMapCurrent(control, controlLength(node));
    const unwired = !!control && controlOutputTargets(node).length === 0;
    const label = CONTROL_MODE_LABELS[control?.mode] || "ControlNet";
    button.textContent = !control ? "ControlNet" : pending ? "ControlNet: preparing"
        : !ready ? (imported ? "ControlNet: check map" : "ControlNet: needs map")
            : unwired ? `ControlNet: ${label} · not wired` : `ControlNet: ${label}`;
    button.title = !control ? "Add a Pose, Depth or Edges video that steers the render"
        : unwired ? "control_video is not connected to Apply MiniMax H3 Fun ControlNet"
            : "Open the ControlNet guide";
    button.classList.toggle("mmrp-plan-error", !!control && !pending && (!ready || unwired));
}
