"""Browser control transport and preparation lifecycle, without a real queue."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest


JS = Path(__file__).resolve().parents[1] / "web" / "refpack.js"
NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="node is not installed")


def _block(name):
    text = JS.read_text()
    return text.split(f"// >>> MMRP-{name}", 1)[1].split(
        f"// <<< MMRP-{name}", 1
    )[0]


def _run(expression, setup="", runtime=False):
    script = _block("TASK-PLAN") + _block("CONTROL") + setup
    if runtime:
        script += _block("CONTROL-RUNTIME")
    script += f"\nconsole.log(JSON.stringify(await ({expression})));\n"
    result = subprocess.run(
        [NODE, "--input-type=module", "-e", script],
        capture_output=True, text=True, check=False, timeout=10,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def _control():
    return {
        "file": "source.mp4", "mode": "edges", "resolution": 512,
        "trim": [1, 3], "crop": [0, 0, 1, 1],
        "map": {
            "file": "controls/map.mp4", "source_sha256": "source",
            "recipe": {"source": "source", "mode": "edges"},
            "sha256": "map", "frame_count": 53, "fps": 24,
            "width": 512, "height": 288,
        },
    }


def test_control_round_trips_outside_reference_roles_and_tags():
    envelope = {
        "references": [{"kind": "image", "file": "face.png"}],
        "task_plan": {"mode": "auto"}, "control": _control(),
    }
    assert _run(
        f"toReferencesEnvelope(fromReferencesEnvelope({json.dumps(envelope)}))"
    ) == envelope
    assert _run(
        f"deriveTaskPlanState(fromReferencesEnvelope({json.dumps(envelope)})).prefix"
    ) == "Auto detect"


def test_legacy_config_does_not_gain_a_control():
    envelope = {"references": []}
    assert _run(
        f"toReferencesEnvelope(fromReferencesEnvelope({json.dumps(envelope)}))"
    ) == envelope


def test_legacy_22_output_restore_appends_control_without_changing_links():
    got = _run("(() => {const before = JSON.stringify(node.outputs); ensureControlOutput(node);"
               "const unchanged = JSON.stringify(node.outputs.slice(0, 22)) === before;"
               "ensureControlOutput(node); return {unchanged, length: node.outputs.length, last: node.outputs[22]};})()", """
    const node = {outputs: Array.from({length: 22}, (_, i) => ({name: 'old_' + i, type: 'STRING', links: [100 + i]})),
        addOutput(name, type) {this.outputs.push({name, type, links: null});}};
    """)
    assert got == {"unchanged": True, "length": 23,
                   "last": {"name": "control_video", "type": "IMAGE", "links": None}}
    assert "ensureControlOutput(this)" in JS.read_text()


def test_linked_primitive_duration_overrides_the_manager_fallback():
    got = _run("controlLength(node)", """
    const widgetByName = (node, name) => node.widgets?.find(widget => widget.name === name);
    const upstream = {type: 'PrimitiveFloat', widgets: [{name: 'value', value: 5}]};
    const node = {inputs: [{name: 'length_seconds', link: 12}],
        widgets: [{name: 'length_seconds', value: 8}], getInputNode: () => upstream};
    """)
    assert got == 5


def test_linked_duration_resolves_through_reroute_and_graph_links():
    got = _run("controlLength(node)", """
    const widgetByName = (node, name) => node.widgets?.find(widget => widget.name === name);
    const primitive = {type: 'PrimitiveNode', widgets: [{name: 'value', value: 5}]};
    const reroute = {type: 'Reroute', inputs: [{link: 11}], getInputNode: () => primitive};
    const graph = {links: {12: {origin_id: 3}}, getNodeById: () => reroute};
    const node = {graph, inputs: [{name: 'length_seconds', link: 12}],
        widgets: [{name: 'length_seconds', value: 8}]};
    """)
    assert got == 5


def test_unresolved_linked_duration_never_uses_fallback_or_queues():
    got = _run("(async () => {let error; try {await queueControlPreparation(node, {}, '');}"
               "catch (caught) {error = caught.message;} return {length: controlLength(node), error};})()", """
    const widgetByName = (node, name) => node.widgets?.find(widget => widget.name === name);
    const node = {inputs: [{name: 'length_seconds', link: 12}],
        widgets: [{name: 'length_seconds', value: 8}], getInputNode: () => ({type: 'RuntimeMath'})};
    const api = {queuePrompt: () => {throw new Error('Should not queue');}};
    """)
    assert got["length"] is None
    assert "linked length_seconds" in got["error"]


def test_source_recipe_change_invalidates_map_and_pending_preparation():
    control = _control()
    control["preparation"] = {"prompt_id": "old"}
    got = _run(
        f"changeControlSettings({json.dumps(control)}, {{mode: 'pose'}})"
    )
    assert got["mode"] == "pose"
    assert got["map"] is None
    assert "preparation" not in got


def test_unchanged_settings_preserve_ready_map():
    assert _run(
        f"changeControlSettings({json.dumps(_control())}, {{resolution: 512}})"
    ) == _control()


def test_control_completion_does_not_make_reference_role_planner_stale():
    got = _run("(() => {const refs = fromReferencesEnvelope({references: [{kind: 'image', file: 'face.png'}]});"
               "const before = referencesFingerprint(refs);"
               f"refs.control = {json.dumps(_control())};"
               "return referencesFingerprint(refs) === before;})()")
    assert got is True
    assert "working.control = takeControl(node._mmrpRefs.control)" in JS.read_text()


def test_history_requires_success_and_surfaces_execution_failure():
    control = _control()
    completed = {"p": {"status": {"completed": True, "status_str": "success"},
                       "outputs": {"4": {"control_map": [control["map"]]}}}}
    assert _run(
        f"controlHistoryState({json.dumps(completed)}, 'p', '4')"
    ) == {"state": "complete", "map": control["map"], "message": ""}
    failed = {"p": {"status": {"completed": False, "status_str": "error",
                              "messages": [["execution_error", {"exception_message": "Detector unavailable"}]]},
                    "outputs": {}}}
    assert _run(
        f"controlHistoryState({json.dumps(failed)}, 'p', '4')"
    ) == {"state": "failed", "map": None, "message": "Detector unavailable"}
    assert _run("controlHistoryState({}, 'p', '4').state") == "pending"


def test_successful_history_without_a_map_is_a_failure():
    history = {"p": {"status": {"completed": True, "status_str": "success"},
                     "outputs": {}}}
    result = _run(f"controlHistoryState({json.dumps(history)}, 'p', '4')")
    assert result["state"] == "failed"
    assert "control map" in result["message"]


def test_late_results_cannot_restore_removed_or_changed_control():
    control = _control()
    setup = f"""
    const original = {json.dumps(control)};
    const preparation = {{prompt_id: 'p', output_node_id: '4', recipe: original.map.recipe,
        settings_fingerprint: controlSettingsFingerprint(original, 4)}};
    original.preparation = preparation;
    """
    got = _run("({same: canApplyControlMap(original, preparation, original.map, 4),"
               "removed: canApplyControlMap(null, preparation, original.map, 4),"
               "changed: canApplyControlMap({...original, mode: 'pose'}, preparation, original.map, 4),"
               "length: canApplyControlMap(original, preparation, original.map, 6),"
               "newJob: canApplyControlMap({...original, preparation: {...preparation, prompt_id: 'new'}}, preparation, original.map, 4)})", setup)
    assert got == {"same": True, "removed": False, "changed": False,
                   "length": False, "newJob": False}


def test_receipt_recipe_matches_with_reordered_nested_keys():
    got = _run("(() => {"
               "const control = {file: 's.mp4', mode: 'pose', resolution: 512};"
               "const preparation = {prompt_id: 'p', recipe: {mode: 'pose', processor: {class_type: 'DWPreprocessor', inputs: {resolution: 512, detect_body: 'enable'}}},"
               "settings_fingerprint: controlSettingsFingerprint(control, 4)};"
               "control.preparation = preparation;"
               "const map = {file: 'map.mp4', recipe: {processor: {inputs: {detect_body: 'enable', resolution: 512}, class_type: 'DWPreprocessor'}, mode: 'pose'}};"
               "return canApplyControlMap(control, preparation, map, 4);})()")
    assert got is True


def test_explicit_queue_uses_preparation_prompt_and_workflow_metadata_only():
    control = _control()
    setup = f"""
    const calls = [];
    const api = {{queuePrompt: async (...args) => {{calls.push(args); return {{prompt_id: 'p'}};}}}};
    const node = {{_mmrpRefs: fromReferencesEnvelope({{references: [], control: {json.dumps(control)}}}),
        widgets: [{{name: 'length_seconds', value: 4}}]}};
    const widgetByName = (node, name) => node.widgets.find(w => w.name === name);
    const syncReferencesWidget = () => {{}};
    const renderNodeBody = () => {{}};
    const plan = {{prompt: {{'1': {{class_type: 'MiniMaxPrepareControlMap', inputs: {{}}}}}},
        workflow: {{version: 0.4, nodes: [], links: []}}, output_node_id: '1', recipe: {{id: 'recipe'}}}};
    """
    got = _run("(async () => {await queueControlPreparation(node, plan, controlSettingsFingerprint(node._mmrpRefs.control, 4));"
               "return {calls, preparation: node._mmrpRefs.control.preparation};})()", setup)
    assert len(got["calls"]) == 1
    assert got["calls"][0][0] == 0
    assert got["calls"][0][1]["output"] == {
        "1": {"class_type": "MiniMaxPrepareControlMap", "inputs": {}}
    }
    assert got["calls"][0][1]["workflow"]["version"] == 0.4
    assert got["preparation"]["prompt_id"] == "p"


def test_pending_preparation_round_trips_for_modal_reopen_and_workflow_restore():
    control = _control()
    control["map"] = None
    control["preparation"] = {
        "prompt_id": "p", "output_node_id": "4", "recipe": {"id": "r"},
        "settings_fingerprint": "persisted",
    }
    assert _run(
        f"toReferencesEnvelope(fromReferencesEnvelope({{references: [], control: {json.dumps(control)}}})).control"
    ) == control


@pytest.mark.parametrize("outcome", ["complete", "failed", "cancelled"])
def test_history_watcher_updates_persisted_state_without_queueing_a_render(outcome):
    control = _control()
    map_data = control["map"]
    control["map"] = None
    if outcome == "complete":
        entry = {"p": {"status": {"completed": True, "status_str": "success"},
                       "outputs": {"4": {"control_map": [map_data]}}}}
    elif outcome == "failed":
        entry = {"p": {"status": {"completed": False, "status_str": "error"}, "outputs": {}}}
    else:
        entry = {}
    setup = f"""
    const node = {{_mmrpRefs: fromReferencesEnvelope({{references: [], control: {json.dumps(control)}}})}};
    node._mmrpRefs.control.preparation = {{prompt_id: 'p', output_node_id: '4',
        recipe: {json.dumps(map_data['recipe'])}, settings_fingerprint: controlSettingsFingerprint(node._mmrpRefs.control, 4)}};
    const liveNodes = new Set([node]);
    const widgetByName = () => ({{value: 4}});
    let saves = 0;
    let renders = 0;
    const syncReferencesWidget = () => {{saves++;}};
    const renderNodeBody = () => {{renders++;}};
    const setTimeout = (callback) => queueMicrotask(callback);
    const requests = [];
    const api = {{fetchApi: async (path) => {{
        requests.push(path);
        return {{ok: true, json: async () => path.startsWith('/history/')
            ? {json.dumps(entry)} : {{queue_running: [], queue_pending: []}}}};
    }}, queuePrompt: () => {{throw new Error('Watcher must never queue');}}}};
    """
    got = _run("(async () => {watchControlPreparation(node); await new Promise(resolve => setImmediate(resolve));"
               "return {control: node._mmrpRefs.control, saves, renders, requests};})()", setup, runtime=True)
    assert got["saves"] == 1
    assert got["renders"] == 1
    assert got["requests"][0] == "/history/p"
    if outcome == "complete":
        assert got["control"]["map"] == map_data
        assert "preparation" not in got["control"]
    else:
        assert got["control"]["map"] is None
        assert got["control"]["preparation"]["error"]


def test_ui_keeps_preparation_separate_from_render_and_restores_config_control():
    text = JS.read_text()
    assert 'import { api } from "../../scripts/api.js";' in text
    assert 'controlsButton.onclick = () => openControlModal(node)' in text
    assert "control: data.control" in text
    assert "config.control = envelope.control" in text
    modal = text.split("async function openControlModal(node)", 1)[1].split(
        "// Config (reference pack) save/load", 1
    )[0]
    assert "Generate control map" in modal
    assert "only control preparation" in modal
    assert "may download" in modal
    assert "app.queuePrompt" not in modal
    assert "graphToPrompt" not in modal
