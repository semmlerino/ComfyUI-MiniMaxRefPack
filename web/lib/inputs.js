import { nodeSize, layoutExposedInputs } from "./layout.js";

// Optional sockets are exposed individually; their backend schema is always present.
export const REFERENCE_INPUTS = Object.fromEntries([
    ...Array.from({ length: 9 }, (_, i) => [`image_${i + 1}`, "IMAGE"]),
    ...Array.from({ length: 3 }, (_, i) => [`video_${i + 1}`, "IMAGE"]),
    ...Array.from({ length: 3 }, (_, i) => [`video_audio_${i + 1}`, "AUDIO"]),
    ...Array.from({ length: 3 }, (_, i) => [`audio_${i + 1}`, "AUDIO"]),
]);

// `redraw(node)` repaints after a socket is exposed or hidden; `connectionsChanged(node)`
// runs after layout whenever a link is made or broken.
export function installInputControls(node, { redraw, connectionsChanged }) {
    node._mmrpBaseWidgetsStartY = node.widgets_start_y;
    // New nodes start compact. onConfigure restores any sockets saved in a workflow.
    for (let i = (node.inputs?.length || 0) - 1; i >= 0; i--) {
        if (REFERENCE_INPUTS[node.inputs[i].name]) node.removeInput(i);
    }
    const previousMenu = node.getExtraMenuOptions;
    node.getExtraMenuOptions = function (canvas, options) {
        previousMenu?.call(this, canvas, options);
        const entries = { ...REFERENCE_INPUTS, direction: "STRING", system_prompt: "STRING",
            references_json: "STRING" };
        options.push({ content: "Expose input", submenu: { options: Object.entries(entries).map(([name, type]) => {
            const index = this.inputs?.findIndex((input) => input.name === name) ?? -1;
            const input = index >= 0 ? this.inputs[index] : null;
            return {
                content: `${input ? "Hide" : "Expose"} ${name}${input?.link != null ? " (disconnect first)" : ""}`,
                disabled: input?.link != null,
                callback: () => {
                    if (index >= 0) this.removeInput(index);
                    else this.addInput(name, type);
                    this.setSize(nodeSize(this));
                    this.setDirtyCanvas(true, true);
                    redraw(this);
                },
            };
        }) } });
    };
    const previousConnections = node.onConnectionsChange;
    node.onConnectionsChange = function () {
        previousConnections?.apply(this, arguments);
        layoutExposedInputs(this);
        connectionsChanged(this);
    };
    const previousDraw = node.onDrawBackground;
    node.onDrawBackground = function () {
        // Widget positions can change after provider switches and workflow restore.
        const size = nodeSize(this);
        if (this.size[0] !== size[0] || this.size[1] !== size[1]) this.setSize(size);
        previousDraw?.apply(this, arguments);
    };
    const previousConnectionPos = node.getConnectionPos;
    node.getConnectionPos = function () {
        layoutExposedInputs(this);
        return previousConnectionPos.apply(this, arguments);
    };
}
