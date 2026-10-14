"""Exercise the shipped socket menu with a minimal LiteGraph node."""
from _js import requires_node, run


@requires_node
def test_expose_hide_and_link_protection():
    script = '''
    const node = {
        inputs: Object.entries(REFERENCE_INPUTS).map(([name, type]) => ({name, type, link: null})),
        removeInput(index) { this.inputs.splice(index, 1); },
        addInput(name, type) { this.inputs.push({name, type, link: null}); },
        setSize() {}, setDirtyCanvas() {},
    };
    const redrawn = [];
    const changed = [];
    installInputControls(node, {redraw: (n) => redrawn.push(n), connectionsChanged: (n) => changed.push(n)});
    const initiallyCompact = node.inputs.length === 0;
    const menu = () => { const options = []; node.getExtraMenuOptions(null, options); return options[0].submenu.options; };
    menu().find(item => item.content === "Expose image_1").callback();
    const exposed = node.inputs[0].name === "image_1" && node.inputs[0].type === "IMAGE";
    node.inputs[0].link = 12;
    const protectedLink = menu().find(item => item.content.startsWith("Hide image_1")).disabled;
    node.inputs[0].link = null;
    menu().find(item => item.content === "Hide image_1").callback();
    const hidden = node.inputs.length === 0;
    menu().find(item => item.content === "Expose direction").callback();
    node.onConnectionsChange();
    console.log(JSON.stringify({initiallyCompact, exposed, protectedLink, hidden,
        direction: node.inputs[0].name === "direction",
        redrawnPerMenuAction: redrawn.length === 3 && redrawn.every(n => n === node),
        connectionsChangedOnce: changed.length === 1 && changed[0] === node}));
    '''
    assert all(run(script, "inputs", "layout").values())


@requires_node
def test_reference_input_rows_reserve_space_before_widgets_and_reflow():
    script = '''
    const setting = {name: "width", widget: {name: "width"}, pos: [10, 100]};
    const node = {
        _mmrpDomWidget: {last_y: 300},
        inputs: [setting, {name: "image_1"}, {name: "image_2"}, {name: "direction"}],
    };
    const bottom = layoutExposedInputs(node);
    const first = node.inputs[1].pos[1];
    const second = node.inputs[2].pos[1];
    const separated = first === 14 && second - first === 24 && node.widgets_start_y > node.inputs[3].pos[1];
    const settingUnchanged = setting.pos[1] === 100;
    node.inputs.splice(1, 1);
    node._mmrpDomWidget.last_y = 400;
    layoutExposedInputs(node);
    const reflowed = node.inputs[1].pos[1] === 14 && node.inputs[2].pos[1] === 38 && node.widgets_start_y === 64;
    console.log(JSON.stringify({separated, settingUnchanged, reflowed, fits: bottom > second}));
    '''
    assert all(run(script, "inputs", "layout").values())
