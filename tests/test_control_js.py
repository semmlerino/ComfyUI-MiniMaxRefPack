"""Browser control transport and preparation lifecycle, without a real queue."""

import json
from typing import Any

import pytest

from _js import LIB, WEB, requires_node, run

pytestmark = requires_node

MODULES = ("references", "control", "control_ui", "edits", "config")

# What createControlUi takes from refpack.js. A setup that does not declare one gets a
# stub that throws when called, so a test reaching a collaborator it did not fake fails
# loudly instead of reaching the real browser API.
HOST = ("api", "liveNodes", "syncReferencesWidget", "renderNodeBody", "createPlannerModal",
        "appendFileOptions", "apiListFiles", "apiUpload")


def _host() -> str:
    def entry(name: str) -> str:
        stub = "new Set()" if name == "liveNodes" else (
            f"(() => {{ throw new Error('test did not fake {name}'); }})")
        return f"{name}: typeof {name} === 'undefined' ? {stub} : {name}"
    return ("const { controlRequest, queueControlPreparation, useExistingControlMap,"
            " watchControlPreparation, openControlModal } = createControlUi({"
            + ", ".join(entry(name) for name in HOST) + "});\n")


def _run(expression, setup=""):
    script = setup + "\n" + _host() + f"console.log(JSON.stringify(await ({expression})));\n"
    return run(script, *MODULES)


def _control() -> dict[str, Any]:
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


def _map_control() -> dict[str, Any]:
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
               _import_setup())
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
               _import_setup(deferred=True))
    assert got == {"imported": None, "saves": 0}


def test_newer_import_wins_when_identical_requests_finish_out_of_order():
    got = _run("(async () => {const first = useExistingControlMap(node);"
               "const second = useExistingControlMap(node);"
               "const newer = structuredClone(ready); newer.map.sha256 = 'newer';"
               "responses[1]({ok: true, json: async () => ({control: newer})}); await second;"
               "responses[0]({ok: true, json: async () => ({control: ready})});"
               "return {first: await first, sha: node._mmrpRefs.control.map.sha256, saves};})()",
               _import_setup(deferred=True))
    assert got == {"first": None, "sha": "newer", "saves": 1}


def test_import_rejects_a_receipt_for_different_settings():
    got = _run("(async () => {const pending = useExistingControlMap(node);"
               "ready.map.recipe.length_seconds = 5;"
               "responses[0]({ok: true, json: async () => ({control: ready})});"
               "try {await pending;} catch (error) {return {message: error.message, saves};}})()",
               _import_setup(deferred=True))
    assert "different settings" in got["message"]
    assert got["saves"] == 0


def _modal_setup(control, available=False, wired=False, imported=None, target="MiniMaxH3FunControlNetApply",
                 files=None):
    """A fake DOM just deep enough for openControlModal, and a node around `control`.

    `imported` is what the server answers an import check with; `files` is the input
    video listing (default: just the control's own file).
    """
    outputs = ("[{name: 'control_video', links: [7]}]" if wired
               else "[{name: 'control_video', links: []}]")
    return f"""
    class Element {{
        constructor(tag) {{this.tag = tag; this.children = []; this.style = {{}}; this.dataset = {{}};
            this.attributes = {{}}; this.listeners = {{}}; this.isConnected = true; this.textContent = '';
            const classes = new Set(); this.classes = classes;
            this.classList = {{add: name => classes.add(name), contains: name => classes.has(name),
                toggle: (name, force) => (force ?? !classes.has(name)) ? classes.add(name) : classes.delete(name)}};}}
        set className(value) {{this.classes.clear(); String(value).split(' ').filter(Boolean).forEach(c => this.classes.add(c));}}
        get className() {{return [...this.classes].join(' ');}}
        appendChild(child) {{this.children.push(child); child.parentElement = this; return child;}}
        insertBefore(child, before) {{const at = this.children.indexOf(before);
            this.children.splice(at < 0 ? this.children.length : at, 0, child); child.parentElement = this; return child;}}
        prepend(child) {{this.children.unshift(child); child.parentElement = this;}}
        setAttribute(name, value) {{this.attributes[name] = value;}}
        getAttribute(name) {{return this.attributes[name];}}
        removeAttribute(name) {{delete this.attributes[name]; delete this[name];}}
        addEventListener(name, callback) {{this.listeners[name] = callback;}}
        get options() {{return this.children;}}
        pause() {{this.paused = true;}}
        load() {{}}
        getBoundingClientRect() {{return {{left: 0, top: 0, width: 100, height: 100}};}}
    }}
    globalThis.document = {{createElement: tag => new Element(tag), activeElement: null}};
    globalThis.window = {{listeners: {{}}, addEventListener(name, callback) {{this.listeners[name] = callback;}},
        removeEventListener(name, callback) {{if (this.listeners[name] === callback) delete this.listeners[name];}}}};
    let modal;
    const createPlannerModal = () => {{modal = new Element('div'); return {{modal, close() {{modal.isConnected = false;}}}};}};
    const apiListFiles = async () => {json.dumps(files if files is not None else [control["file"]])};
    const uploads = [];
    const apiUpload = async (file, subfolder) => {{uploads.push([file.name, subfolder || null]);
        return subfolder ? `${{subfolder}}/${{file.name}}` : file.name;}};
    const appendFileOptions = (select, files, current) => {{select.children = [];
        for (const file of new Set([current, ...files].filter(Boolean))) {{const option = new Element('option');
            option.value = file; option.textContent = file; select.appendChild(option);}}}};
    const ready = {json.dumps(imported or control)};
    const start = {json.dumps(control)};
    const target = {{type: {json.dumps(target)}, title: {json.dumps("Apply MiniMax H3 Fun ControlNet" if target == "MiniMaxH3FunControlNetApply" else target)}}};
    const node = {{_mmrpRefs: fromReferencesEnvelope({{references: [], control: start}}),
        widgets: [{{name: 'length_seconds', value: 4}}, {{name: 'width', value: 720}}, {{name: 'height', value: 1280}}],
        outputs: {outputs}, graph: {{links: new Map([[7, {{target_id: 3}}]]), getNodeById: id => id === 3 ? target : null}}}};
    node._mmrpBody = {{controlsButton: new Element('button')}};
    // Not a live node: a restored preparation is shown, never polled.
    const liveNodes = new Set();
    const requests = [];
    let saves = 0;
    const syncReferencesWidget = () => {{saves++;}};
    const renderNodeBody = () => {{syncControlsButton(node); node._mmrpControlUpdate?.();}};
    const api = {{queuePrompt() {{throw new Error('Unexpected queue');}}, fetchApi: async (path) => {{
        requests.push(path);
        return {{ok: true, json: async () => path.endsWith('/capabilities')
            ? {{modes: ['pose', 'depth', 'edges'].map(id => ({{id, label: id[0].toUpperCase() + id.slice(1),
                available: {json.dumps(available)}, reason: 'No detector'}}))}}
            : {{control: ready}}}};
    }}}};
    const elements = root => [root, ...root.children.flatMap(elements)];
    const find = text => elements(modal).find(el => el.textContent === text);
    const byClass = name => elements(modal).find(el => el.classList.contains(name));
    const view = () => ({{badge: byClass('mmrp-ctl-badge').textContent,
        primary: byClass('mmrp-btn-primary').textContent, primaryDisabled: !!byClass('mmrp-btn-primary').disabled,
        note: byClass('mmrp-ctl-stale-note').hidden ? null : byClass('mmrp-ctl-stale-note').textContent,
        map: elements(byClass('mmrp-ctl-mapview')).find(el => el.tag === 'video').dataset.source || null,
        mapShown: !byClass('mmrp-ctl-mapview').hidden && !elements(byClass('mmrp-ctl-mapview')).find(el => el.tag === 'video').hidden,
        stageShown: !byClass('mmrp-ctl-stage').hidden,
        redo: byClass('mmrp-ctl-actions').hidden ? null : byClass('mmrp-ctl-actions').children[1].textContent,
        facts: elements(modal).filter(el => el.classList.contains('mmrp-ctl-fact')).map(row =>
            [row.classList.contains('mmrp-ctl-fact-warn') ? 'warn' : 'ok', row._head.textContent]),
        button: node._mmrpBody.controlsButton.textContent}});
    """


def _ready_source_control() -> dict[str, Any]:
    crop, trim = [0.25, 0, 0.5, 1], [1, 5.5]
    return {
        "file": "source.mp4", "mode": "edges", "resolution": 512, "trim": trim, "crop": crop,
        "map": {
            "file": "control_maps/source_edges.mp4", "source_sha256": "source", "sha256": "map",
            "recipe": {"input_type": "source", "source_sha256": "source", "mode": "edges",
                       "resolution": 512, "crop": crop, "trim": trim, "length_seconds": 4, "fps": 24},
            "frame_count": 107, "fps": 24, "width": 512, "height": 910,
        },
    }


def test_map_tab_imports_without_detectors_and_source_tab_stays_gated():
    control = {**_map_control(), "map": None}
    got = _run("(async () => {await openControlModal(node); const before = view();"
               "const typesEnabled = elements(modal).filter(el => el.dataset.mode).every(el => !el.disabled);"
               "const stage = elements(modal).find(el => el.tag === 'video').dataset.source;"
               "await byClass('mmrp-btn-primary').onclick(); const checked = view();"
               "const imported = node._mmrpRefs.control.input_type;"
               "find('Make from a video').onclick(); const source = view();"
               "return {before, typesEnabled, stage, checked, imported, source, requests};})()",
               _modal_setup(control, imported=_map_control()))
    assert got["before"]["badge"] == "No map yet"
    assert (got["before"]["primary"], got["before"]["primaryDisabled"]) == ("Check map", False)
    assert got["typesEnabled"] is True
    assert got["stage"] == "/view?filename=pose.mp4&type=input&revision=0"
    assert got["imported"] == "map"
    assert (got["checked"]["badge"], got["checked"]["primary"]) == ("Checked", "Done")
    # One preview of an imported map: the file on the left, no second copy on the right.
    assert (got["before"]["stageShown"], got["checked"]["stageShown"], got["checked"]["mapShown"]) == (True, True, False)
    assert got["checked"]["redo"] == "Check again"
    assert got["checked"]["button"] == "ControlNet: Pose · not wired"
    # Switching task drops the imported receipt; with no detector, generating stays off
    # and the imported map is not presented as this tab's out-of-date map.
    assert got["source"]["badge"] == "Unavailable"
    assert got["source"]["primaryDisabled"] is True
    assert got["source"]["note"] is None
    assert got["requests"] == ["/minimax_refpack/control/capabilities", "/minimax_refpack/control/import"]


def test_picker_lists_control_maps_on_the_map_tab_unless_all_are_asked_for():
    files = ["clip.mp4", "control_maps/clip-pose-1.mp4", "videos/plate.mp4", "control_maps/old.mp4"]
    got = _run(f"[controlPickerFiles({json.dumps(files)}, false, false),"
               f" controlPickerFiles({json.dumps(files)}, true, false),"
               f" controlPickerFiles({json.dumps(files)}, true, true)]")
    assert got == [files, ["control_maps/clip-pose-1.mp4", "control_maps/old.mp4"], files]


def test_map_tab_picker_starts_in_control_maps_and_uploads_there():
    files = ["clip.mp4", "control_maps/clip-pose-1.mp4", "videos/plate.mp4"]
    got = _run("(async () => {await openControlModal(node);"
               "const select = byClass('mmrp-source-select');"
               "const listed = () => select.options.map(o => o.value).filter(v => v && !v.startsWith('\\u0000'));"
               "const showAll = elements(modal).find(el => el.tag === 'input' && el.type === 'checkbox');"
               "const mapTab = listed(); const toggleShown = !showAll.parentElement.hidden;"
               "showAll.checked = true; showAll.onchange(); const all = listed();"
               "showAll.checked = false; showAll.onchange();"
               "await modal.ondrop({preventDefault() {}, stopPropagation() {}, dataTransfer: {files: [{name: 'new.mp4'}]}});"
               "const uploaded = node._mmrpRefs.control.file;"
               "find('Make from a video').onclick(); const sourceTab = listed();"
               "const toggleHidden = !!showAll.parentElement.hidden;"
               "await modal.ondrop({preventDefault() {}, stopPropagation() {}, dataTransfer: {files: [{name: 'plate2.mp4'}]}});"
               "return {mapTab, toggleShown, all, uploaded, sourceTab, toggleHidden, uploads};})()",
               _modal_setup({**_map_control(), "map": None}, files=files))
    # The current file stays listed even outside control_maps/, so the select can show it.
    assert got["mapTab"] == ["pose.mp4", "control_maps/clip-pose-1.mp4"]
    assert got["toggleShown"] is True
    assert got["all"] == ["pose.mp4", *files]
    assert got["uploaded"] == "control_maps/new.mp4"
    # The source tab lists every video again; the file carried over from the map tab stays first.
    assert got["sourceTab"] == ["control_maps/new.mp4", *files]
    assert got["toggleHidden"] is True
    assert got["uploads"] == [["new.mp4", "control_maps"], ["plate2.mp4", None]]


def test_ready_map_goes_out_of_date_and_names_what_changed():
    got = _run("(async () => {syncControlsButton(node); await openControlModal(node); const before = view();"
               "find('Clear crop').onclick(); const after = view();"
               "return {before, after, stored: node._mmrpRefs.control.map, crop: node._mmrpRefs.control.crop};})()",
               _modal_setup(_ready_source_control(), available=True, wired=True))
    before, after = got["before"], got["after"]
    assert (before["badge"], before["primary"]) == ("Ready", "Done")
    assert before["map"] == "/view?filename=source_edges.mp4&type=input&subfolder=control_maps"
    assert (before["stageShown"], before["mapShown"], before["redo"]) == (True, True, "Regenerate")
    assert before["button"] == "ControlNet: Edges"
    assert before["facts"] == [
        ["ok", "Render 4.0 s → 107 frames at 24 fps."],
        ["ok", "control_video → Apply MiniMax H3 Fun ControlNet."],
    ]
    # The stored map is gone, as before; the dialog still shows it, greyed, with why.
    assert got["stored"] is None and got["crop"] is None
    assert (after["badge"], after["primary"], after["primaryDisabled"]) == ("Out of date", "Regenerate", False)
    assert after["map"] == before["map"] and after["mapShown"] is True
    assert "You changed the crop after this map was made" in after["note"]
    assert after["button"] == "ControlNet: needs map"


def test_remove_guide_can_be_undone_without_its_queued_preparation():
    control = {**_ready_source_control(), "preparation": {"prompt_id": "p1", "status": "queued"}}
    got = _run("(async () => {await openControlModal(node); byClass('mmrp-ctl-remove').onclick();"
               "const removed = node._mmrpRefs.control; const toast = byClass('mmrp-ctl-toast');"
               "const message = toast.hidden ? null : toast.children[0].textContent; const button = view().button;"
               "find('Undo').onclick();"
               "return {removed, message, button, restored: node._mmrpRefs.control, toastHidden: toast.hidden};})()",
               _modal_setup(control, available=True, wired=True))
    assert got["removed"] is None
    assert got["message"] == "Guide removed. The queued preparation stays in ComfyUI's queue."
    assert got["button"] == "ControlNet"
    assert got["restored"]["file"] == "source.mp4"
    assert got["restored"]["map"] == _ready_source_control()["map"]
    assert "preparation" not in got["restored"]
    assert got["toastHidden"] is True


def test_unwired_output_is_reported_in_dialog_and_on_node_button():
    got = _run("(async () => {syncControlsButton(node); await openControlModal(node); return view();})()",
               _modal_setup(_ready_source_control(), available=True))
    assert got["facts"][1] == ["warn", "control_video isn't connected."]
    assert got["button"] == "ControlNet: Edges · not wired"


def test_current_map_can_be_rebuilt_without_changing_a_setting():
    got = _run("(async () => {await openControlModal(node); const actions = byClass('mmrp-ctl-actions');"
               "await actions.children[1].onclick(); return {requests, view: view()};})()",
               _modal_setup(_ready_source_control(), available=True, wired=True))
    # The fake server answers prepare with no workflow, so the queue step refuses it;
    # what matters is that the rebuild asked the server instead of closing.
    assert got["requests"] == ["/minimax_refpack/control/capabilities", "/minimax_refpack/control/prepare"]


def test_a_drag_released_after_the_dialog_closed_saves_nothing():
    got = _run("(async () => {await openControlModal(node); const before = node._mmrpRefs.control.crop;"
               "const rect = byClass('mmrp-crop-rect');"
               "rect.listeners.mousedown({clientX: 10, clientY: 10, preventDefault() {}, stopPropagation() {}});"
               "window.listeners.mousemove({clientX: 40, clientY: 10});"
               "const up = window.listeners.mouseup; modal.isConnected = false; modal.listeners['mmrp-modal-close']();"
               "const left = Object.keys(window.listeners);"
               "up?.(); return {before, after: node._mmrpRefs.control.crop, saves, left};})()",
               _modal_setup(_ready_source_control(), available=True, wired=True))
    assert got["after"] == got["before"]
    assert got["saves"] == 0
    assert got["left"] == []


def test_node_button_reports_a_preparation_in_progress():
    control = {**_ready_source_control(), "map": None, "preparation": {"prompt_id": "p1", "status": "running"}}
    got = _run("(() => {syncControlsButton(node); return node._mmrpBody.controlsButton.textContent;})()",
               _modal_setup(control, available=True, wired=True))
    assert got == "ControlNet: preparing"


def test_a_render_length_change_is_named_as_what_changed():
    control = _ready_source_control()
    got = _run(f"controlChangedSettings({{file: 'source.mp4', map: {json.dumps(control['map'])}}}, "
               f"{json.dumps(control)}, 6)")
    assert got == ["the render length"]


def test_output_feeding_another_node_is_flagged_but_still_counts_as_wired():
    got = _run("(async () => {syncControlsButton(node); await openControlModal(node); return view();})()",
               _modal_setup(_ready_source_control(), available=True, wired=True, target="PreviewImage"))
    assert got["facts"][1] == ["warn", "control_video → PreviewImage."]
    assert got["button"] == "ControlNet: Edges"


def test_a_click_without_a_drag_keeps_an_old_full_frame_crop_and_its_map():
    control = _ready_source_control()
    control["crop"] = [0, 0, 1, 1]
    control["map"]["recipe"]["crop"] = [0, 0, 1, 1]
    got = _run("(async () => {await openControlModal(node); const rect = byClass('mmrp-crop-rect');"
               "rect.listeners.mousedown({clientX: 10, clientY: 10, preventDefault() {}, stopPropagation() {}});"
               "window.listeners.mouseup(); return {crop: node._mmrpRefs.control.crop,"
               "map: !!node._mmrpRefs.control.map, saves, badge: view().badge};})()",
               _modal_setup(control, available=True, wired=True))
    assert got == {"crop": [0, 0, 1, 1], "map": True, "saves": 0, "badge": "Ready"}


def test_switching_tabs_and_back_restores_the_ready_map():
    got = _run("(async () => {await openControlModal(node); find('Use a map I already have').onclick();"
               "const away = {type: node._mmrpRefs.control.input_type, map: node._mmrpRefs.control.map};"
               "find('Make from a video').onclick(); return {away, back: view(),"
               "map: node._mmrpRefs.control.map?.file};})()",
               _modal_setup(_ready_source_control(), available=True, wired=True))
    assert got["away"] == {"type": "map", "map": None}
    assert (got["back"]["badge"], got["back"]["primary"]) == ("Ready", "Done")
    assert got["map"] == "control_maps/source_edges.mp4"


def test_no_second_rebuild_while_one_is_queued():
    control = {**_ready_source_control(), "preparation": {"prompt_id": "p1", "status": "queued"}}
    got = _run("(async () => {await openControlModal(node); return view();})()",
               _modal_setup(control, available=True, wired=True))
    assert (got["badge"], got["primary"], got["primaryDisabled"], got["redo"]) == ("Queued", "Done", False, None)


@pytest.mark.parametrize("seconds", [0.25, 0.9375, 1, 1.0625, 4, 4.2, 5, 8, 10.04, 12.5])
def test_target_frames_match_the_server(seconds):
    from minimax_refpack import controls
    assert _run(f"controlTargetFrames({seconds})") == controls.target_frames(seconds)


def test_length_fact_reports_hold_cut_cover_unknown_and_clip_length():
    got = _run("[controlLengthFact(4, 2.5), controlLengthFact(4, 6), controlLengthFact(8, 8),"
               "controlLengthFact(null, 3), controlLengthFact(0, 3), controlLengthFact(4, null), controlLengthFact(0.2, 3)]"
               ".map(f => [f.level, f.head, f.detail])")
    assert got == [
        ["warn", "Render 4.0 s → 107 frames at 24 fps.",
         "The clip gives 2.50 s; the map holds its last frame for the final 1.96 s."],
        ["ok", "Render 4.0 s → 107 frames at 24 fps.", "The last 1.54 s of the clip are not used."],
        ["ok", "Render 8.0 s → 192 frames at 24 fps.", "The clip covers it."],
        ["warn", "Render length unknown.", got[3][2]],
        ["ok", "length_seconds is 0.", "The map keeps the clip's own length, cut down to MiniMax's 17k+5 frame grid."],
        ["ok", "Render 4.0 s → 107 frames at 24 fps.", ""],
        ["warn", "Render 0.20 s is too short.", "A control map needs at least 5 frames at 24 fps (0.21 s)."],
    ]
    assert "PrimitiveFloat" in got[3][2]


def test_output_targets_read_map_and_object_link_tables():
    got = _run("(() => {const target = {comfyClass: 'MiniMaxH3FunControlNetApply', title: 'Apply'};"
               "const node = (links) => ({outputs: [{name: 'prompt', links: [1]}, {name: 'control_video', links: [7]}],"
               "graph: {links, getNodeById: id => id === 3 ? target : null}});"
               "return [controlOutputTargets(node(new Map([[7, {target_id: 3}]]))),"
               "controlOutputTargets(node({7: {target_id: 3}})), controlOutputTargets({outputs: []}),"
               "controlOutputTargets(node(new Map()))];})()")
    expected = [{"type": "MiniMaxH3FunControlNetApply", "title": "Apply"}]
    assert got == [expected, expected, [], []]


def test_dialog_keeps_preparation_and_rendering_separate_in_its_wording():
    text = (LIB / "control_ui.js").read_text() + (LIB / "control.js").read_text()
    assert "Make from a video" in text
    assert "Use a map I already have" in text
    assert "Check map" in text
    assert "Map resolution" in text
    assert "24 fps" in text
    assert "holds its last frame" in text


def test_portable_config_saves_and_loads_imported_map_and_duration():
    control = _map_control()
    setup = f"""
    const missingFiles = async () => new Set();
    const applyRefs = (node, refs) => {{node._mmrpRefs = refs;}};
    const renderNodeBody = () => {{}};
    const node = {{_mmrpRefs: fromReferencesEnvelope({{references: [], control: {json.dumps(control)}}}),
        widgets: [{{name: 'length_seconds', value: 4}}]}};
    const restored = {{_mmrpRefs: fromReferencesEnvelope({{references: []}}),
        widgets: [{{name: 'length_seconds', value: 8}}]}};
    """
    got = _run("(async () => {const config = buildConfig(node, 'imported');"
               "await applyConfig(restored, config, 'test', {missingFiles, applyRefs, renderNodeBody}); return {saved: config.control,"
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
    assert "ensureControlOutput(this)" in (WEB / "refpack.js").read_text()


def test_linked_primitive_duration_overrides_the_manager_fallback():
    got = _run("controlLength(node)", """
    const upstream = {type: 'PrimitiveFloat', widgets: [{name: 'value', value: 5}]};
    const node = {inputs: [{name: 'length_seconds', link: 12}],
        widgets: [{name: 'length_seconds', value: 8}], getInputNode: () => upstream};
    """)
    assert got == 5


def test_linked_duration_resolves_through_reroute_and_graph_links():
    got = _run("controlLength(node)", """
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
    assert "working.control = takeControl(node._mmrpRefs.control)" in (WEB / "refpack.js").read_text()


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
    const node = {{_mmrpRefs: fromReferencesEnvelope({{references: [], control: {json.dumps(control)}}}),
        widgets: [{{name: 'length_seconds', value: 4}}]}};
    node._mmrpRefs.control.preparation = {{prompt_id: 'p', output_node_id: '4',
        recipe: {json.dumps(map_data['recipe'])}, settings_fingerprint: controlSettingsFingerprint(node._mmrpRefs.control, 4)}};
    const liveNodes = new Set([node]);
    let saves = 0;
    let renders = 0;
    const syncReferencesWidget = () => {{saves++;}};
    const renderNodeBody = () => {{renders++;}};
    globalThis.setTimeout = (callback) => queueMicrotask(callback);
    const requests = [];
    const api = {{fetchApi: async (path) => {{
        requests.push(path);
        return {{ok: true, json: async () => path.startsWith('/history/')
            ? {json.dumps(entry)} : {{queue_running: [], queue_pending: []}}}};
    }}, queuePrompt: () => {{throw new Error('Watcher must never queue');}}}};
    """
    got = _run("(async () => {watchControlPreparation(node); await new Promise(resolve => setImmediate(resolve));"
               "return {control: node._mmrpRefs.control, saves, renders, requests};})()", setup)
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
    text = (WEB / "refpack.js").read_text()
    assert 'import { api } from "../../scripts/api.js";' in text
    assert 'controlsButton.onclick = () => openControlModal(node)' in text
    config = (LIB / "config.js").read_text()
    assert "control: data.control" in config
    assert "config.control = envelope.control" in config
    modal = (LIB / "control_ui.js").read_text()
    assert "Generate ${modeLabel.toLowerCase()} map" in modal
    assert "only control preparation" in modal
    assert "may download" in modal
    assert "app.queuePrompt" not in modal
    assert "graphToPrompt" not in modal
