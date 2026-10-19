"""User sizing survives the manager's automatic layout passes."""

from _js import requires_node, run


@requires_node
def test_manager_keeps_user_size_when_inputs_and_widget_positions_change():
    script = '''
    const node = {
        size: [1700, 2100], inputs: [], outputs: Array(22).fill({}),
        _mmrpDomWidget: {last_y: 280},
        setSize(size) { this.size = size; },
        setDirtyCanvas() {},
    };
    installInputControls(node, {redraw() {}, connectionsChanged() {}});
    node.onDrawBackground();
    const afterDraw = [...node.size];
    const options = [];
    node.getExtraMenuOptions(null, options);
    node.addInput = (name, type) => node.inputs.push({name, type});
    options[0].submenu.options.find(item => item.content === "Expose image_1").callback();
    console.log(JSON.stringify({afterDraw, afterExpose: node.size}));
    '''
    got = run(script, "inputs", "layout")
    assert got == {"afterDraw": [1700, 2100], "afterExpose": [1700, 2100]}


@requires_node
def test_resize_can_shrink_and_grow_and_restored_size_is_preserved():
    script = '''
    let previousResizes = 0, previousComputes = 0, redraws = 0;
    const node = {
        size: [1340, 1800],
        _mmrpDomWidget: {last_y: 280},
        _mmrpBody: {root: {style: {}}, uploadRow: {offsetWidth: 740, offsetHeight: 68}},
        onResize() { previousResizes++; },
        computeSize() { previousComputes++; return [1340, 1800]; },
        setSize(size) { this.size = size; this.onResize(size); },
    };
    installSizeGuards(node, () => redraws++);
    const requested = [760, 1300];
    node.setSize(requested);
    const small = [...node.size];
    const rootHeight = node._mmrpBody.root.style.height ?? null;
    const min = node.computeSize();
    node.setSize([1700, 2100]);
    const large = [...node.size];
    // LiteGraph configure restores size directly before the extension's callback.
    node.size = [900, 1450];
    node.setSize(nodeSize(node));
    const restored = [...node.size];
    node.setSize([1, 1]);
    console.log(JSON.stringify({small, rootHeight, large, restored,
        resizable: node.resizable, inputUnchanged: requested[0] === 760 && requested[1] === 1300,
        canShrink: min[0] < small[0] && min[1] < small[1],
        floor: node.size[0] === min[0] && node.size[1] === min[1],
        previousResizes, previousComputes, redraws}));
    '''
    got = run(script, "layout")
    assert got == {
        "small": [760, 1300], "rootHeight": None,
        "large": [1700, 2100], "restored": [900, 1450], "resizable": True,
        "inputUnchanged": True, "canShrink": True, "floor": True,
        "previousResizes": 4, "previousComputes": 1, "redraws": 4,
    }


@requires_node
def test_block_reports_a_minimum_so_layout_passes_never_grow_a_node_that_fits():
    script = '''
    const node = {
        size: [1340, 2326], inputs: [], outputs: [],
        // Before the frontend first positions it, the container is 0px wide and every
        // toolbar button wraps onto its own row.
        _mmrpBody: {uploadRow: {offsetWidth: 0, offsetHeight: 296}},
        addDOMWidget(name, type, element, options) { return {name, type, element, options, y: 0}; },
        setSize(size) { this.size = size; },
    };
    const widget = addBlockWidget(node, {}, {serialize: false, hideOnZoom: false});
    // Frontend 1.53.6 _arrangeWidgets for a node whose last layout widget is the block:
    // a widget with computeSize keeps that height plus 4, a DOM widget without one gets
    // the height left below its y but at least its getMinHeight, and the node grows to fit.
    const arrange = () => {
        const height = widget.computeSize ? widget.computeSize()[1] + 4
            : Math.max(widget.options.getMinHeight(), node.size[1] - widget.y);
        if (widget.y + height > node.size[1]) node.setSize([node.size[0], widget.y + height]);
    };
    widget.y = 706; // arranged but not yet drawn, so last_y is unset
    arrange();
    arrange();
    const beforeDraw = [...node.size];
    widget.last_y = 706;
    arrange();
    const afterDraw = [...node.size];
    node.setSize([1340, 1200]);
    arrange();
    const grownToMinimum = [...node.size];
    node._mmrpBody.uploadRow = {offsetWidth: 880, offsetHeight: 68};
    arrange();
    const wrappedToolbar = [...node.size];
    widget.y = 634; // a widget above was hidden; last_y still says 706 until the next draw
    console.log(JSON.stringify({beforeDraw, afterDraw, grownToMinimum, wrappedToolbar,
        minimumAfterMove: minimumSize(node)[1], options: Object.keys(widget.options).sort(),
        registered: node._mmrpDomWidget === widget}));
    '''
    got = run(script, "layout")
    assert got == {
        "beforeDraw": [1340, 2326], "afterDraw": [1340, 2326],
        "grownToMinimum": [1340, 1556], "wrappedToolbar": [1340, 1594],
        "minimumAfterMove": 1522, "options": ["getMinHeight", "hideOnZoom", "serialize"],
        "registered": True,
    }
