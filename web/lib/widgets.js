// Reading, writing and showing the node's native litegraph widgets.

export function widgetByName(node, name) {
    return (node.widgets || []).find((w) => w.name === name);
}

// REVERSIBLE, unlike hideWidget(): that one installs getter-only accessors and is a
// one-way door, which is right for the three permanently-hidden widgets and wrong here.
// These come back when the provider changes, so the real type is stashed and restored.
export function setWidgetVisible(w, visible) {
    if (!w) return;
    if (w._mmrpType === undefined) {
        w._mmrpType = w.type;
        w._mmrpDraw = w.draw;      // usually undefined: litegraph draws by type
    }
    try {
        w.type = visible ? w._mmrpType : "hidden";
        w.hidden = !visible;
        if (!w.options) w.options = {};
        w.options.hidden = !visible;
        // [0,0] removes the row from litegraph's layout, which is what lets fixedSize()
        // shrink the node by exactly the hidden rows on the next pass.
        w.computeSize = visible ? undefined : () => [0, 0];
        // Load-bearing, and its absence was a real bug (2026-08-17): zero height keeps a
        // widget out of the LAYOUT but not out of the PAINT. Litegraph kept drawing the
        // hidden ones at their last_y from when they were visible, so `api_base` painted
        // its URL straight over the `reasoning_effort` row. Suppress the paint, and drop
        // the stale coordinate so nothing can be drawn or hit-tested at it either.
        w.draw = visible ? w._mmrpDraw : () => {};
        if (!visible) w.last_y = undefined;
    } catch (_) {
        // Some frontend builds make these reactive accessors. Degrade to "always shown"
        // rather than throwing mid-configure and aborting the workflow load.
    }
}

// reasoning_effort is OpenRouter-only (endpoint.sends_reasoning), so it hides with the
// rest of that group. A dropdown that silently does nothing on `local` is the same trap
// that let a local model id sit in a field OpenRouter then read.
export const PROVIDER_FIELDS = {
    openrouter: ["openrouter_api_key", "openrouter_model", "reasoning_effort"],
    local: ["api_base", "local_model_slug"],
};

// Writes a native widget the way litegraph expects: value first, then its callback, so
// anything watching that widget (serialization, other extensions) sees the change rather
// than a value that appeared behind its back.
export function setWidget(node, name, value) {
    const w = widgetByName(node, name);
    if (!w) return false;
    w.value = value;
    if (w.callback) w.callback(value);
    return true;
}
