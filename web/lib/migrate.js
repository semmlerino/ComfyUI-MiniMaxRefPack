// ---------------------------------------------------------------------------
// 0.3.1 -> 0.3.2 widget migration.
//
// `use_openrouter` was a BOOLEAN. `prompt_provider` is a combo in the SAME slot,
// because appending it instead would have re-pointed every widget after it - the
// exact class of bug 0.3.1 shipped and fixed. litegraph restores widgets_values
// positionally and does not type-check, so a workflow saved at 0.3.1 hands the
// combo a raw `true`, which then fails validation at queue time.
//
// configure() applies widgets_values BEFORE calling onConfigure, so by the time we
// run the wrong value is already sitting on the widget and this is a repair rather
// than an interception. That is fine and is why it is written this way: one place,
// after the fact, no hooking of litegraph internals. tests/test_migration.py runs this
// module under node against endpoint.normalize_provider.
export const PROVIDER_VALUES = ["openrouter", "local", "none"];

export function migrateProviderValue(raw) {
    // Mirrors endpoint.normalize_provider in Python. Both exist on purpose: this one
    // fixes the graph the user is looking at, the Python one catches an API client
    // that never loaded a browser. Total by design - an unknown value resolves to the
    // default rather than throwing, because a broken workflow that will not queue is
    // worse than a workflow that quietly writes its prompt the old way.
    if (raw === true) return "openrouter";
    if (raw === false) return "none";
    const text = String(raw ?? "").trim().toLowerCase();
    if (PROVIDER_VALUES.includes(text)) return text;
    if (text === "true") return "openrouter";
    if (text === "false") return "none";
    return "openrouter";
}

export function migrateProviderWidget(widget) {
    // Returns true when it changed something, so the caller can log it once.
    if (!widget) return false;
    const migrated = migrateProviderValue(widget.value);
    if (widget.value === migrated) return false;
    widget.value = migrated;
    return true;
}

// --- 0.3.3 reorder -------------------------------------------------------------
//
// widgets_values is a positional array, so the declaration order in nodes.py is a wire
// format. 0.3.3 regrouped the widgets by decision flow, which means every array saved
// before it now decodes into the wrong widgets - width into prompt_provider, and so on.
//
// Detection is by VALUE SHAPE rather than by properties.ver. A graph can reach us
// without a ver (hand-edited, older frontend, copied node), and being wrong here is
// worse than the bug it fixes: it would silently scramble a workflow that was fine.
// Length plus the type at the provider slot identifies each layout unambiguously.
export const ORDER_0_3_1 = [
    "direction", "openrouter_api_key", "openrouter_model", "references_json",
    "system_prompt", "width", "height", "length_seconds", "prompt_provider",
    "reasoning_effort", "job_type", "max_reference_edge",
];
export const ORDER_0_3_2 = ORDER_0_3_1.concat(["api_base", "local_model_slug"]);
export const ORDER_0_3_3 = [
    "direction", "references_json", "system_prompt", "prompt_provider",
    "openrouter_api_key", "openrouter_model", "reasoning_effort", "api_base",
    "local_model_slug", "job_type", "width", "height", "length_seconds",
    "max_reference_edge",
];
// 0.4.2 appended match_video_aspect. Appending keeps every 0.3.3 slot where it was, so
// one detection covers both: names past the end of an older array simply do not appear.
export const ORDER_0_4_2 = ORDER_0_3_3.concat(["match_video_aspect"]);
// original_prompt is display-only and also appended, so a 0.4.2 array still maps.
// rewrite_pin is a visible combo appended after that: older arrays stay aligned.
export const ORDER_CURRENT = ORDER_0_4_2.concat(["original_prompt", "rewrite_pin"]);

export function detectLayout(values) {
    if (!Array.isArray(values)) return null;
    // 0.3.3 or later: slot 3 holds a provider string.
    if (PROVIDER_VALUES.includes(String(values[3] ?? "").trim().toLowerCase())) {
        return ORDER_CURRENT;
    }
    // 0.3.1: use_openrouter was a boolean at slot 8. 0.3.2: a provider string there.
    const slot8 = values[8];
    if (typeof slot8 === "boolean") return ORDER_0_3_1;
    if (PROVIDER_VALUES.includes(String(slot8 ?? "").trim().toLowerCase())) {
        return values.length > 12 ? ORDER_0_3_2 : ORDER_0_3_1;
    }
    return null;   // unrecognised: leave it alone rather than guess
}

export function migrateMatchValue(raw) {
    // Only a real true survives. A graph saved before the switch existed can restore a
    // stray trailing value into its slot, and a BOOLEAN widget holding "" or "True" would
    // otherwise ride through to the server, where bool() reads any non-empty text as on.
    return raw === true;
}

export function remapWidgetValues(values) {
    // -> {name: value} for whatever layout this array was written in, or null if the
    // array is not one we recognise. Names absent from the old layout simply do not
    // appear, so the caller leaves those widgets at their defaults.
    const order = detectLayout(values);
    if (!order) return null;
    const out = {};
    for (let i = 0; i < order.length && i < values.length; i++) {
        out[order[i]] = values[i];
    }
    // The old `model` slot is this pack's openrouter_model; the old generic override is
    // the local slug. Both renames happened in 0.3.3 alongside the reorder.
    if (out.model !== undefined && out.openrouter_model === undefined) {
        out.openrouter_model = out.model;
    }
    if (out.model_override !== undefined && out.local_model_slug === undefined) {
        out.local_model_slug = out.model_override;
    }
    if (out.prompt_provider !== undefined) {
        out.prompt_provider = migrateProviderValue(out.prompt_provider);
    }
    if (out.match_video_aspect !== undefined) {
        out.match_video_aspect = migrateMatchValue(out.match_video_aspect);
    }
    return out;
}
