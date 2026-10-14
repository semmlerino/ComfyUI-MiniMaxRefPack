// Configs are FILES ON THE USER'S MACHINE, not server state: refpack.js downloads one
// through the browser and reads one back through a file picker. These build, check and
// apply the JSON; refpack.js owns the download, the picker and the input listing.

import { controlLength, invalidateControlOperation } from "./control.js";
import { fromReferencesEnvelope, toReferencesEnvelope } from "./references.js";
import { setWidget, widgetByName } from "./widgets.js";

export function configFilename(name) {
    const safe = (name || "config").replace(/[^A-Za-z0-9 ._-]+/g, "_").trim() || "config";
    return `minimax-refpack-${safe}.json`;
}

export function buildConfig(node, name) {
    const w = (n) => {
        const widget = widgetByName(node, n);
        return widget ? widget.value || "" : "";
    };
    const envelope = toReferencesEnvelope(node._mmrpRefs);
    const config = {
        version: 1,
        name,
        direction: w("direction"),
        model: w("model"),
        reasoning_effort: w("reasoning_effort"),
        references: envelope.references,
    };
    if (envelope.task_plan) config.task_plan = envelope.task_plan;
    if (envelope.control) config.control = envelope.control;
    const lengthSeconds = controlLength(node);
    if (lengthSeconds != null) config.length_seconds = lengthSeconds;
    return config;
}

// One of two guarded JSON.parse calls. The other is refpack.js's parseRefsValue, which
// guards a widget value; this one guards a file the user picked. Both are wrapped
// because both parse bytes we did not write — an unguarded throw here would surface
// as a dead button, which is the exact class of bug this section already had once.
export function parseConfigFile(text) {
    let data;
    try {
        data = JSON.parse(text);
    } catch (e) {
        throw new Error("not valid JSON");
    }
    if (!data || typeof data !== "object" || Array.isArray(data)) {
        throw new Error("not a config object");
    }
    if (!Array.isArray(data.references)) {
        throw new Error("no references list");
    }
    return data;
}

// `host` is refpack.js's: missingFiles(references) lists which files this install lacks,
// applyRefs(node, refs) installs the new reference state, renderNodeBody(node) repaints.
export async function applyConfig(node, data, sourceLabel, { missingFiles, applyRefs, renderNodeBody }) {
    const missing = await missingFiles(data.references);
    const list = data.references.map((r) => ({ ...r, missing: missing.has(r.file) }));
    node._mmrpSelected = null;
    invalidateControlOperation(node);
    applyRefs(node, fromReferencesEnvelope({ references: list, task_plan: data.task_plan, control: data.control }));

    const directionWidget = widgetByName(node, "direction");
    if (directionWidget && typeof data.direction === "string") {
        directionWidget.value = data.direction;
        if (node._mmrpBody) node._mmrpBody.directionInput.value = data.direction;
    }
    // Do not overwrite a linked duration node. Its visible constant remains the
    // authority; the map's recipe will show if a portable config needs regeneration.
    const linkedLength = node.inputs?.some((input) => input.name === "length_seconds" && input.link != null);
    if (!linkedLength && typeof data.length_seconds === "number"
        && Number.isFinite(data.length_seconds) && data.length_seconds >= 0) {
        setWidget(node, "length_seconds", data.length_seconds);
        renderNodeBody(node);
    }

    // A config written before these fields existed carries "" — leave the widget alone
    // rather than blanking a valid setting. A model no longer in the dropdown is
    // skipped too: setting a value litegraph cannot offer would show a combo
    // displaying something the user can never pick again.
    const restore = (widgetName, value) => {
        if (typeof value !== "string" || !value) return null;
        const w = widgetByName(node, widgetName);
        if (!w) return null;
        const options = w.options && w.options.values;
        if (Array.isArray(options) && !options.includes(value)) return value;
        w.value = value;
        return null;
    };
    const unavailable = [
        restore("model", data.model),
        restore("reasoning_effort", data.reasoning_effort),
    ].filter(Boolean);

    const problems = [];
    if (missing.size) problems.push(`not on this pod: ${[...missing].join(", ")}`);
    if (unavailable.length) problems.push(`unavailable setting(s): ${unavailable.join(", ")}`);
    if (problems.length) alert(`${sourceLabel} — ${problems.join("; ")}`);
}
