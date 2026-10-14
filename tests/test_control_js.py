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


def _run(expression, setup="", runtime=False, modal=False):
    script = _block("TASK-PLAN") + _block("CONTROL") + setup
    if runtime:
        script += _block("CONTROL-RUNTIME")
    if modal:
        text = JS.read_text()
        start = text.index("function controlFileUrl(file)")
        end = text.index("// Config (reference pack) save/load", start)
        script += text[start:end]
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


def _map_control():
    return {
        "input_type": "map", "file": "pose.mp4", "mode": "pose",
        "resolution": 512, "trim": None, "crop": None,
        "map": {
            "file": "pose.mp4", "source_sha256": "source", "sha256": "map",
            "recipe": {"input_type": "map", "source_sha256": "source",
                       "mode": "pose", "resolution": 512, "trim": None, "crop": None,
                       "length_seconds": 4, "fps": 24, "processor": None,
                       "frame_policy": "24fps-target-ceil-17k+5-last-frame-hold-v2"},
            "frame_count": 107, "fps": 24, "width": 512, "height": 288,
        },
    }


def _import_setup(deferred=False):
    control = _map_control()
    pending = {**control, "map": None}
    return f"""
    const ready = {json.dumps(control)};
    const node = {{_mmrpRefs: fromReferencesEnvelope({{references: [], control: {json.dumps(pending)}}}),
        widgets: [{{name: 'length_seconds', value: 4}}]}};
    const widgetByName = (node, name) => node.widgets?.find(widget => widget.name === name);
    const requests = [];
    const responses = [];
    let saves = 0;
    const syncReferencesWidget = () => {{saves++;}};
    const renderNodeBody = () => {{}};
    const api = {{queuePrompt: () => {{throw new Error('Imported maps must not queue');}},
        fetchApi: (path, options) => {{requests.push({{path, body: JSON.parse(options.body)}});
            return {'new Promise(resolve => responses.push(resolve))' if deferred else 'Promise.resolve({ok: true, json: async () => ({control: ready})})'};
        }}}};
    """


def test_existing_map_round_trip_and_input_kind_affects_readiness():
    control = _map_control()
    envelope = {"references": [], "control": control}
    assert _run(f"toReferencesEnvelope(fromReferencesEnvelope({json.dumps(envelope)}))") == envelope
    got = _run("(() => {const control = " + json.dumps(control)
               + "; return {current: controlMapCurrent(control, 4),"
               "source: controlMapCurrent({...control, input_type: 'source'}, 4),"
               "duration: controlMapCurrent(control, 5),"
               "differentIdentity: controlSettingsFingerprint(control, 4) !== controlSettingsFingerprint({...control, input_type: 'source'}, 4),"
               "cleared: changeControlSettings(control, {input_type: 'source'}).map};})()")
    assert got == {"current": True, "source": False, "duration": False,
                   "differentIdentity": True, "cleared": None}


def test_source_recipe_without_input_kind_remains_current():
    control = _map_control()
    del control["input_type"]
    del control["map"]["recipe"]["input_type"]
    assert _run(f"controlMapCurrent({json.dumps(control)}, 4)") is True


def test_source_pending_fingerprint_before_input_kind_was_added_still_matches():
    got = _run("(() => {const control = {file: 's.mp4', mode: 'edges', resolution: 512};"
               "const settings = controlSettings(control, 4); delete settings.input_type;"
               "return controlSettingsMatch(control, 4, JSON.stringify(settings));})()")
    assert got is True


def test_use_existing_map_validates_and_saves_without_detector_or_comfy_queue():
    got = _run("(async () => {const imported = await useExistingControlMap(node);"
               "return {imported, current: node._mmrpRefs.control, requests, saves};})()",
               _import_setup(), runtime=True)
    assert got["imported"] == _map_control()
    assert got["current"] == _map_control()
    assert got["saves"] == 1
    assert len(got["requests"]) == 1
    assert got["requests"][0]["path"] == "/minimax_refpack/control/import"
    assert got["requests"][0]["body"]["control"]["input_type"] == "map"
    assert got["requests"][0]["body"]["length_seconds"] == 4


@pytest.mark.parametrize("mutation", [
    "node._mmrpRefs.control.file = 'other.mp4'",
    "node._mmrpRefs.control.input_type = 'source'",
    "node.widgets[0].value = 5",
    "node._mmrpRefs.control = null",
    "invalidateControlOperation(node)",
])
def test_import_late_result_cannot_restore_removed_changed_or_invalidated_settings(mutation):
    got = _run("(async () => {const pending = useExistingControlMap(node);"
               + mutation + "; responses[0]({ok: true, json: async () => ({control: ready})});"
               "const imported = await pending; return {imported, saves};})()",
               _import_setup(deferred=True), runtime=True)
    assert got == {"imported": None, "saves": 0}


def test_newer_import_wins_when_identical_requests_finish_out_of_order():
    got = _run("(async () => {const first = useExistingControlMap(node);"
               "const second = useExistingControlMap(node);"
               "const newer = structuredClone(ready); newer.map.sha256 = 'newer';"
               "responses[1]({ok: true, json: async () => ({control: newer})}); await second;"
               "responses[0]({ok: true, json: async () => ({control: ready})});"
               "return {first: await first, sha: node._mmrpRefs.control.map.sha256, saves};})()",
               _import_setup(deferred=True), runtime=True)
    assert got == {"first": None, "sha": "newer", "saves": 1}


def test_import_rejects_a_receipt_for_different_settings():
    got = _run("(async () => {const pending = useExistingControlMap(node);"
               "ready.map.recipe.length_seconds = 5;"
               "responses[0]({ok: true, json: async () => ({control: ready})});"
               "try {await pending;} catch (error) {return {message: error.message, saves};}})()",
               _import_setup(deferred=True), runtime=True)
    assert "different settings" in got["message"]
    assert got["saves"] == 0


def test_map_ui_imports_when_detectors_are_unavailable_and_source_mode_stays_gated():
    control = _map_control()
    setup = f"""
    class Element {{
        constructor(tag) {{this.tag = tag; this.children = []; this.style = {{}}; this.dataset = {{}};
            this.attributes = {{}}; this.listeners = {{}}; this.isConnected = true;
            this.classList = {{add() {{}}, toggle() {{}}}};}}
        appendChild(child) {{this.children.push(child); child.parentElement = this; return child;}}
        prepend(child) {{this.children.unshift(child); child.parentElement = this;}}
        setAttribute(name, value) {{this.attributes[name] = value;}}
        removeAttribute(name) {{delete this[name];}}
        addEventListener(name, callback) {{this.listeners[name] = callback;}}
        get options() {{return this.children;}}
        pause() {{this.paused = true;}}
        load() {{}}
    }}
    const document = {{createElement: tag => new Element(tag), activeElement: null}};
    let modal;
    const createPlannerModal = () => {{modal = new Element('div'); return {{modal, close() {{modal.isConnected = false;}}}};}};
    const apiListFiles = async () => ['pose.mp4'];
    const appendFileOptions = (select, files, current) => {{select.children = [];
        for (const file of new Set([current, ...files].filter(Boolean))) {{const option = new Element('option');
            option.value = file; option.textContent = file; select.appendChild(option);}}}};
    const isVideoFile = () => true;
    const ready = {json.dumps(control)};
    const node = {{_mmrpRefs: fromReferencesEnvelope({{references: [], control: {{...ready, map: null}}}}),
        widgets: [{{name: 'length_seconds', value: 4}}]}};
    const widgetByName = (node, name) => node.widgets?.find(widget => widget.name === name);
    const requests = [];
    const syncReferencesWidget = () => {{}};
    const renderNodeBody = () => node._mmrpControlUpdate?.();
    const api = {{queuePrompt() {{throw new Error('Unexpected queue');}}, fetchApi: async (path) => {{
        requests.push(path);
        return {{ok: true, json: async () => path.endsWith('/capabilities')
            ? {{modes: ['pose', 'depth', 'edges'].map(id => ({{id, label: id, available: false, reason: 'No detector'}}))}}
            : {{control: ready}}}};
    }}}};
    const elements = root => [root, ...root.children.flatMap(elements)];
    """
    # The author display:flex rule overrides the browser's default [hidden] style.
    got = _run("(async () => {await openControlModal(node);"
               "const all = elements(modal); const button = all.find(el => el.textContent === 'Use control map');"
               "const visiblePreviews = () => all.filter(el => el.tag === 'video' && el.parentElement.style.display !== 'none');"
               "const beforeValidation = visiblePreviews().map(el => el._mmrpLabel.textContent);"
               "const enabled = !button.disabled; const type = all.find(el => el.attributes['aria-label'] === 'Control input');"
               "const semantic = all.find(el => el.attributes['aria-label'] === 'Control type');"
               "const mapOptionsEnabled = semantic.options.every(option => !option.disabled);"
               "await button.onclick();"
               "const preview = all.find(el => el.tag === 'video' && el._mmrpLabel.textContent === 'Map file preview');"
               "const originalPreview = preview.dataset.source; const imported = node._mmrpRefs.control.input_type;"
               "const importedPreviews = visiblePreviews().map(el => el._mmrpLabel.textContent);"
               "type.value = 'source'; type.onchange();"
               "node._mmrpRefs.control = {...ready, input_type: 'source', map: {...ready.map, file: 'generated.mp4',"
               "recipe: {...ready.map.recipe, input_type: 'source'}}}; node._mmrpControlUpdate();"
               "const sourcePreviews = visiblePreviews().filter(el => !el.hidden).map(el => el._mmrpLabel.textContent);"
               "return {enabled, mapOptionsEnabled, imported, originalPreview, sourceDisabled: button.disabled,"
               "beforeValidation, importedPreviews, sourcePreviews, sourceButton: button.textContent, requests};})()", setup, runtime=True, modal=True)
    assert got == {
        "enabled": True, "mapOptionsEnabled": True, "imported": "map",
        "originalPreview": "/view?filename=pose.mp4&type=input",
        "beforeValidation": ["Selected map file"],
        "importedPreviews": ["Map file preview"],
        "sourcePreviews": ["Source video", "Prepared control map"],
        "sourceDisabled": True, "sourceButton": "Generate control map",
        "requests": ["/minimax_refpack/control/capabilities", "/minimax_refpack/control/import"],
    }


def test_existing_map_ui_is_explicit_and_distinct_from_source_generation():
    text = JS.read_text()
    assert "Generate from source video" in text
    assert "Use existing control map" in text
    assert "Use control map" in text
    assert "Map resolution" in text
    assert "24 fps" in text
    assert "hold" in text


def test_portable_config_saves_and_loads_imported_map_and_duration():
    text = JS.read_text()
    build = text[text.index("function buildConfig(node, name)"):text.index("function downloadJson(filename, obj)")]
    apply = text[text.index("async function applyConfig(node, data, sourceLabel)"):text.index("function openLoadConfigPanel(node, anchorBtn)")]
    control = _map_control()
    setup = f"""
    {build}
    {apply}
    const widgetByName = (node, name) => node.widgets?.find(widget => widget.name === name);
    const missingFiles = async () => new Set();
    const applyRefs = (node, refs) => {{node._mmrpRefs = refs;}};
    const renderNodeBody = () => {{}};
    const setWidget = (node, name, value) => {{widgetByName(node, name).value = value;}};
    const node = {{_mmrpRefs: fromReferencesEnvelope({{references: [], control: {json.dumps(control)}}}),
        widgets: [{{name: 'length_seconds', value: 4}}]}};
    const restored = {{_mmrpRefs: fromReferencesEnvelope({{references: []}}),
        widgets: [{{name: 'length_seconds', value: 8}}]}};
    """
    got = _run("(async () => {const config = buildConfig(node, 'imported');"
               "await applyConfig(restored, config, 'test'); return {saved: config.control,"
               "restored: restored._mmrpRefs.control, length: controlLength(restored)};})()", setup)
    assert got == {"saved": control, "restored": control, "length": 4}


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
