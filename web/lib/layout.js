// Reference rows, resizable node geometry and exposed input sockets.

export const KINDS = ["image", "video", "audio"];
export const CAPS = { image: 9, video: 3, audio: 3 };
export const SECTION_LABEL = { image: "Images", video: "Videos", audio: "Audio" };

// Canvas row geometry, all in CSS px (draw() scales the backing store by
// devicePixelRatio, so layout math never sees physical pixels). Tile size started
// as LTX Director's 75px gallery cell (multi_image_loader.js:208), scaled 1.75×
// per spec ("make them about 1.75 times the current size") — the on-tile furniture
// (badge, delete, ♪, ▶) is scaled with it so the proportions hold.
export const CL = {
    x0: 10, // inner side padding of the black slab (also clears the selection stroke)
    padTop: 8,
    headerH: 15,
    headGap: 12, // 8px of real air between a section label and its strip (UI review #7)
    tile: 131, // 75 × 1.75
    gap: 6,
    stripPad: 3, // above/below tiles, room for the selection stroke
    // Generous per spec: three DISTINCT sections, divider drawn at the midpoint
    // of this gap (see draw()).
    rowGap: 18,
    // On-tile furniture, scaled with the tile.
    del: 20, // delete chip circle DIAMETER — a quiet chip, not red (UI review #5)
    sound: 28, // soundtrack toggle box
    playR: 14, // play glyph circle radius — r20 buried a third of the thumbnail
    badge1: 18, // tag badge height, one line (11px type)
    badge2: 30, // tag badge height, two lines
    badge3: 42, // tag badge height, three lines (tag + vid_audio + the crop/trim line)
    // The per-row add affordance: an icon-only square, NOT a toolbar-button clone
    // (UI review #1). Fixed placement — always where the next tile would go, with
    // a 24px gap separating it from the reference group so it doesn't read as
    // another tile; it never centres itself and only ever moves rightwards.
    addBtn: 44,
    addGap: 24,
};
CL.stripH = CL.stripPad * 2 + CL.tile;

// The rows never move: every section is always drawn at full tile height, 0 items or
// 9. Narrow nodes scroll the canvas horizontally instead of changing tile sizes.
export const CANVAS_ROWS = (() => {
    const rows = [];
    let y = CL.padTop;
    for (const kind of KINDS) {
        rows.push({ kind, y, stripY: y + CL.headerH + CL.headGap });
        y += CL.headerH + CL.headGap + CL.stripH + CL.rowGap;
    }
    // The trailing rowGap doubles as the slab's bottom padding.
    return { rows, height: y };
})();

// The default width fits all nine images and their add button. The prompt takes
// the remaining height when resized; the toolbar can wrap on narrower nodes.
export const CONTENT = {
    width: 1340,
    minWidth: 600,
    pad: 10, // side inset of the DOM block within the node
    uploadsH: 30, // minimum toolbar height
    promptH: 880, // default prompt height
    minPromptH: 110,
    scrollbarH: 16,
    originalH: 88, // label + .mmrp-original — keeps the idea after Auto Prompt rewrites direction
    gap: 8, // .mmrp-block flex gap
};
CONTENT.height =
    CONTENT.uploadsH + CANVAS_ROWS.height + CONTENT.scrollbarH + CONTENT.promptH
    + CONTENT.gap * 4 + CONTENT.originalH;

function widgetTop(node) {
    return node._mmrpDomWidget?.last_y || 80;
}

function minimumBlockHeight(node) {
    const toolbarH = node._mmrpBody?.uploadRow?.offsetHeight || CONTENT.uploadsH;
    return CONTENT.height - CONTENT.promptH + CONTENT.minPromptH + toolbarH - CONTENT.uploadsH;
}

export function minimumSize(node) {
    const inputsMin = layoutExposedInputs(node);
    const outputsMin = Math.max(node.outputs?.length || 1, node.inputs?.length || 1) * 20 + 40;
    return [CONTENT.minWidth, Math.max(widgetTop(node) + minimumBlockHeight(node) + CONTENT.pad,
        outputsMin, inputsMin)];
}

export function defaultSize(node) {
    const min = minimumSize(node);
    return [CONTENT.width, Math.max(widgetTop(node) + CONTENT.height + CONTENT.pad, min[1])];
}

// Clamp only the minimum; workflow restores and automatic layout keep user sizes.
export function nodeSize(node, requested = node.size) {
    const min = minimumSize(node);
    const fallback = defaultSize(node);
    return min.map((floor, i) => Math.max(floor, Number.isFinite(requested?.[i]) ? requested[i] : fallback[i]));
}

export function blockSize(node) {
    const size = nodeSize(node);
    return [size[0] - CONTENT.pad * 2, size[1] - widgetTop(node) - CONTENT.pad];
}

export function installSizeGuards(node, resized) {
    node.resizable = true;
    const previousResize = node.onResize;
    node.onResize = function (size) {
        const clamped = nodeSize(this, size);
        size[0] = clamped[0];
        size[1] = clamped[1];
        previousResize?.call(this, size);
        const root = this._mmrpBody?.root;
        if (root) root.style.height = `${blockSize(this)[1]}px`;
        resized?.(this);
    };
    const previousCompute = node.computeSize;
    node.computeSize = function () {
        previousCompute?.apply(this, arguments);
        this.min_size = minimumSize(this);
        return [...this.min_size];
    };
    const previousSet = node.setSize;
    node.setSize = function (size) {
        const clamped = nodeSize(this, size);
        if (previousSet) previousSet.call(this, clamped);
        else {
            this.size = clamped;
            this.onResize(clamped);
        }
    };
}

export function layoutExposedInputs(node) {
    // Plain sockets get their own rows above the widgets. Converted setting sockets
    // stay with their widgets, and the DOM reference area moves down with them.
    const rowHeight = 24;
    let row = 0;
    for (const input of node.inputs || []) {
        if (input.widget) continue;
        input.pos = [10, 14 + row++ * rowHeight];
    }
    node.widgets_start_y = row
        ? Math.max(node._mmrpBaseWidgetsStartY || 0, row * rowHeight + 16)
        : node._mmrpBaseWidgetsStartY;
    return row * rowHeight + 16;
}
