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
        _mmrpBody: {root: {style: {}}, uploadRow: {offsetHeight: 68}},
        onResize() { previousResizes++; },
        computeSize() { previousComputes++; return [1340, 1800]; },
        setSize(size) { this.size = size; this.onResize(size); },
    };
    installSizeGuards(node, () => redraws++);
    const requested = [760, 1300];
    node.setSize(requested);
    const small = [...node.size];
    const smallBlock = blockSize(node);
    const rootHeight = node._mmrpBody.root.style.height;
    const min = node.computeSize();
    node.setSize([1700, 2100]);
    const large = [...node.size];
    // LiteGraph configure restores size directly before the extension's callback.
    node.size = [900, 1450];
    node.setSize(nodeSize(node));
    const restored = [...node.size];
    node.setSize([1, 1]);
    console.log(JSON.stringify({small, smallBlock, rootHeight, large, restored,
        resizable: node.resizable, inputUnchanged: requested[0] === 760 && requested[1] === 1300,
        canShrink: min[0] < small[0] && min[1] < small[1],
        floor: node.size[0] === min[0] && node.size[1] === min[1],
        previousResizes, previousComputes, redraws}));
    '''
    got = run(script, "layout")
    assert got == {
        "small": [760, 1300], "smallBlock": [740, 1010], "rootHeight": "1010px",
        "large": [1700, 2100], "restored": [900, 1450], "resizable": True,
        "inputUnchanged": True, "canShrink": True, "floor": True,
        "previousResizes": 4, "previousComputes": 1, "redraws": 4,
    }
