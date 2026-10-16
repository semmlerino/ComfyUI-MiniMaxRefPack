// The node's fixed geometry: reference rows, the DOM block and exposed input sockets.

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
// 9. Static layout, computed once — this is the whole point of the fixed-size node.
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

// Fixed block geometry. The DOM heights (uploadsH, promptH, gap) are pinned in
// refpack.css to the same values — the two must agree, there is no measurement.
// THE WIDTH PRESERVES THE NO-SCROLL INVARIANT: the longest row is 9 image tiles
// (9×131 + 8×6 = 1227px) plus the 24px add-gap and the 44px add square, which is
// drawn dimmed even at cap = 1295px of row viewport; +2×10 slab padding = 1315
// canvas; +2×10 block inset = 1335 node — fixed at 1340 for slack. That invariant
// is why no scroll machinery exists; change tile/gap/x0/pad/addBtn/addGap only
// together with this width.
export const CONTENT = {
    width: 1340,
    pad: 10, // side inset of the DOM block within the node
    uploadsH: 30, // .mmrp-uploads fixed height
    promptH: 880, // .mmrp-direction fixed height — tall enough to show a full shot prompt without scrolling
    originalH: 88, // label + .mmrp-original — keeps the idea after Auto Prompt rewrites direction
    gap: 8, // .mmrp-block flex gap
};
CONTENT.height =
    CONTENT.uploadsH + CONTENT.gap + CANVAS_ROWS.height + CONTENT.gap + CONTENT.promptH
    + CONTENT.gap + CONTENT.originalH;

// The ONE node size. Height floors against the 19 output sockets; widgetY is where
// litegraph placed the DOM block below the native widgets (fallback for the beat
// before the first layout assigns last_y).
export function fixedSize(node) {
    const inputsMin = layoutExposedInputs(node);
    const widgetY = (node._mmrpDomWidget && node._mmrpDomWidget.last_y) || 80;
    const outputsMin = Math.max(node.outputs?.length || 1, node.inputs?.length || 1) * 20 + 40;
    return [CONTENT.width, Math.max(widgetY + CONTENT.height + CONTENT.pad, outputsMin, inputsMin)];
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
