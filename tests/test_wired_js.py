"""Exercise the shipped socket menu with a minimal LiteGraph node."""
import json
from pathlib import Path
import subprocess


def test_expose_hide_and_link_protection():
    text = (Path(__file__).resolve().parent.parent / "web/refpack.js").read_text()
    block = text.split("// >>> MMRP-INPUTS", 1)[1].split("// <<< MMRP-INPUTS", 1)[0]
    script = block + '''
    const fixedSize = () => [1340, 1000];
    const scheduleDraw = () => {};
    const node = {
        inputs: Object.entries(REFERENCE_INPUTS).map(([name, type]) => ({name, type, link: null})),
        removeInput(index) { this.inputs.splice(index, 1); },
        addInput(name, type) { this.inputs.push({name, type, link: null}); },
        setSize() {}, setDirtyCanvas() {},
    };
    installInputControls(node);
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
    console.log(JSON.stringify({initiallyCompact, exposed, protectedLink, hidden,
        direction: node.inputs[0].name === "direction"}));
    '''
    result = subprocess.run(["node", "--input-type=module", "-e", script], text=True, capture_output=True, check=True)
    assert all(json.loads(result.stdout).values())


def test_reference_input_rows_reserve_space_before_widgets_and_reflow():
    text = (Path(__file__).resolve().parent.parent / "web/refpack.js").read_text()
    block = text.split("// >>> MMRP-INPUTS", 1)[1].split("// <<< MMRP-INPUTS", 1)[0]
    script = block + '''
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
    result = subprocess.run(["node", "--input-type=module", "-e", script], text=True, capture_output=True, check=True)
    assert all(json.loads(result.stdout).values())
