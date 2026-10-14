"""Exercise the task-plan/reference transport code and the task plan dialog from the shipped browser modules."""

import json

import pytest

from _js import FAKE_DOM, WEB, evaluate, requires_node, run

REFPACK_CSS = WEB / "refpack.css"


def _run(expression: str):
    return evaluate(expression, "references")


def _dialog(envelope: dict, body: str, *, probes: dict | None = None, job_type: str = "auto",
            files: dict | None = None):
    """Open the real task plan dialog on a node holding `envelope` and run `body`.

    `probes` is the soundtrack probe cache (file -> has audio); `files` the input listing
    per kind. `body` sees `modal`, `node`, `applied` (what Apply committed), `dropped`
    (thumbnails invalidated) and the row helpers, and must evaluate to one JSON value.
    """
    setup = FAKE_DOM + f"""
    const node = {{_mmrpRefs: fromReferencesEnvelope({json.dumps(envelope)}),
        widgets: [{{name: 'job_type', value: {json.dumps(job_type)}}}]}};
    const dropped = [];
    let applied = null;
    const ui = createTaskPlanUi({{
        createPlannerModal, appendFileOptions,
        apiListFiles: async (kind) => ({json.dumps(files or {})})[kind] || [],
        thumbUrl: (file) => `/thumb/${{file}}`, dropThumb: (file) => dropped.push(file),
        probeResults: new Map(Object.entries({json.dumps(probes or {})})),
        applyRefs: (target, refs) => {{ applied = refs; target._mmrpRefs = refs; }},
    }});
    await ui.openTaskPlanModal(node);
    const modal = byClass(document.body, 'mmrp-task-modal')[0];
    const rows = byClass(modal, 'mmrp-plan-asset');
    const role = (row, id) => all(rows[row]).find(el => el.dataset.roleControl === 'true' && el.value === id);
    const primary = (row) => all(rows[row]).find(el => el.dataset.primaryControl === 'true');
    const choose = (select, value) => {{ select.value = value; fire(select, 'change'); }};
    const status = () => byClass(modal, 'mmrp-plan-status')[0];
    const apply = () => fire(byText(modal, 'Apply plan'), 'click');
    """
    return evaluate(body, "references", "task_plan_ui", "modal", setup=setup)


@requires_node
def test_task_plan_shows_and_refreshes_picture_and_video_thumbnails():
    envelope = {"references": [
        {"kind": "image", "file": "face.png"},
        {"kind": "video", "file": "plate.mp4", "trim": [1, 2]},
        {"kind": "audio", "file": "voice.wav"},
    ]}
    got = _dialog(envelope, """(() => {
        const thumbs = () => rows.map(row => byClass(row, 'mmrp-plan-thumbnail-image').map(img => [img.src, img.alt]));
        const before = thumbs();
        choose(byClass(rows[1], 'mmrp-source-select')[0], 'other.mp4');
        return {before, after: thumbs(), dropped, video: node._mmrpRefs.videos[0]};
    })()""", files={"video": ["plate.mp4", "other.mp4"]})

    assert got["before"] == [
        [["/thumb/face.png", "Picture preview: face.png"]],
        [["/thumb/plate.mp4", "Video preview: plate.mp4"]],
        [],
    ]
    assert got["after"][1] == [["/thumb/other.mp4", "Video preview: other.mp4"]]
    assert got["dropped"] == ["plate.mp4"]
    assert got["video"]["file"] == "plate.mp4"  # the node keeps its refs until Apply
    assert ".mmrp-plan-thumbnail-image" in REFPACK_CSS.read_text()


@requires_node
def test_task_plan_primary_video_control_explains_and_requires_a_driver_role():
    envelope = {"references": [{"kind": "video", "file": "plate.mp4", "use_soundtrack": True}],
                "task_plan": {"mode": "explicit"}}
    got = _dialog(envelope, """(() => {
        const state = () => ({primary: primary(0).checked, disabled: primary(0).disabled});
        const start = state();
        fire(Object.assign(role(0, 'video_editing'), {checked: true}), 'change');
        const driven = state();
        fire(Object.assign(role(0, 'video_editing'), {checked: false}), 'change');
        return {start, driven, undriven: state(),
                label: primary(0).parentElement.children[1].textContent,
                help: byClass(rows[0], 'mmrp-primary-help')[0].textContent};
    })()""")

    assert got["start"] == {"primary": False, "disabled": True}
    assert got["driven"] == {"primary": True, "disabled": False}  # first driver role marks it
    assert got["undriven"] == {"primary": False, "disabled": True}
    assert got["label"] == "Primary video"
    assert "Replacement plans use the editing source." in got["help"]


@requires_node
def test_apply_commits_the_plan_and_resets_the_legacy_job_type():
    envelope = {"references": [
        {"kind": "video", "file": "guide.mp4", "use_soundtrack": True},
        {"kind": "video", "file": "plate.mp4", "use_soundtrack": True},
    ]}
    got = _dialog(envelope, """(() => {
        const disabledAtOpen = role(1, 'video_editing').disabled;
        fire(Object.assign(role(1, 'video_editing'), {checked: true}), 'change');
        fire(Object.assign(role(0, 'reference_generation'), {checked: true}), 'change');
        const prefix = status().textContent;
        apply();
        return {disabledAtOpen, prefix, envelope: toReferencesEnvelope(applied),
                jobType: node.widgets[0].value};
    })()""", job_type="replacement")

    assert got["disabledAtOpen"] is False  # legacy job_type=replacement opens explicit
    assert got["prefix"].startswith("Derived prefix: [video editing")
    # The primary editing source moves to <Video 1>.
    assert [r["file"] for r in got["envelope"]["references"]] == ["plate.mp4", "guide.mp4"]
    assert got["envelope"]["references"][0]["primary"] is True
    assert got["envelope"]["task_plan"] == {"mode": "explicit"}
    assert got["jobType"] == "auto"


@requires_node
def test_a_silent_new_source_loses_its_soundtrack_roles():
    envelope = {"references": [{"kind": "video", "file": "plate.mp4", "use_soundtrack": True,
                                "roles": ["reference_generation", "audio_reuse"]}],
                "task_plan": {"mode": "explicit"}}
    got = _dialog(envelope, """(() => {
        const before = role(0, 'audio_reuse').disabled;
        choose(byClass(rows[0], 'mmrp-source-select')[0], 'silent.mp4');
        return {before, after: role(0, 'audio_reuse').disabled, checked: role(0, 'audio_reuse').checked};
    })()""", probes={"silent.mp4": False}, files={"video": ["plate.mp4", "silent.mp4"]})

    assert got == {"before": False, "after": True, "checked": False}


@requires_node
def test_references_changed_while_open_block_apply():
    envelope = {"references": [{"kind": "image", "file": "face.png", "roles": ["reference_generation"]}],
                "task_plan": {"mode": "explicit"}}
    got = _dialog(envelope, """(() => {
        node._mmrpRefs = fromReferencesEnvelope({references: []});
        apply();
        return {applied, status: status().textContent,
                save: byText(modal, 'Apply plan').disabled};
    })()""")

    assert got["applied"] is None
    assert got["status"].startswith("References changed while this planner was open.")
    assert got["save"] is True


@requires_node
def test_apply_keeps_a_control_receipt_that_landed_while_open():
    envelope = {"references": [{"kind": "image", "file": "face.png", "roles": ["reference_generation"]}],
                "task_plan": {"mode": "explicit"}}
    receipt = {"source": {"file": "plate.mp4"}, "mode": "pose", "map": {"file": "control_maps/plate_pose.mp4"}}
    got = _dialog(envelope, f"""(() => {{
        node._mmrpRefs.control = {json.dumps(receipt)};
        apply();
        return applied.control;
    }})()""")

    assert got == receipt


@requires_node
@pytest.mark.parametrize("has_audio,soundtrack,roles", [
    (None, True, ["reference_generation", "audio_reuse"]),   # probe pending: sound back on
    (True, True, ["reference_generation", "audio_reuse"]),
    (False, False, ["reference_generation"]),                # known silent: sound roles go
])
def test_a_new_video_source_drops_edits_and_resets_its_soundtrack(has_audio, soundtrack, roles):
    got = _run(f"""(() => {{
        const reference = {{file: 'old.mp4', missing: true, crop: [0, 0, 0.5, 1], trim: [1, 2],
            use_soundtrack: false, roles: ['reference_generation', 'audio_reuse']}};
        changeReferenceSource(reference, 'video', 'new.mp4', {json.dumps(has_audio)});
        return reference;
    }})()""")

    assert got == {"file": "new.mp4", "missing": False, "use_soundtrack": soundtrack, "roles": roles}


@requires_node
def test_a_new_image_source_drops_edits_only():
    got = _run("""(() => {
        const reference = {file: 'a.png', crop: [0, 0, 0.5, 1], roles: ['reference_generation']};
        changeReferenceSource(reference, 'image', 'b.png', false);
        return reference;
    })()""")

    assert got == {"file": "b.png", "missing": False, "roles": ["reference_generation"]}


@requires_node
def test_toggled_roles_keep_the_option_order():
    got = _run("""(() => {
        const refs = fromReferencesEnvelope({references: [{kind: 'video', file: 'v.mp4'}]});
        const video = refs.videos[0];
        toggleTaskRole(refs, 'video', video, 'audio_reuse', true);
        toggleTaskRole(refs, 'video', video, 'video_editing', true);
        return video.roles;
    })()""")

    assert got == ["video_editing", "audio_reuse"]


@requires_node
def test_legacy_envelope_stays_free_of_task_metadata():
    got = _run(
        "toReferencesEnvelope(fromReferencesEnvelope("
        + json.dumps({"references": [{"kind": "image", "file": "a.png"}]})
        + "))"
    )
    assert got == {"references": [{"kind": "image", "file": "a.png"}]}


@requires_node
def test_roles_and_task_plan_round_trip_in_the_browser_transport():
    envelope = {
        "references": [
            {
                "kind": "video",
                "file": "plate.mp4",
                "use_soundtrack": True,
                "primary": True,
                "roles": ["video_editing", "audio_reuse"],
            }
        ],
        "task_plan": {"mode": "explicit", "specialization": "object_replacement"},
    }
    assert _run(
        "toReferencesEnvelope(fromReferencesEnvelope(" + json.dumps(envelope) + "))"
    ) == envelope


@requires_node
def test_browser_derivation_matches_the_backend_prefix_order():
    expression = """
    (() => {
      const refs = fromReferencesEnvelope({
        references: [
          {kind: 'image', file: 'face.png', roles: ['keyframe_completion', 'reference_generation']},
          {kind: 'video', file: 'plate.mp4', use_soundtrack: true,
           primary: true, roles: ['video_editing', 'audio_reuse']}
        ],
        task_plan: {mode: 'explicit', specialization: 'character_replacement'}
      });
      return deriveTaskPlanState(refs);
    })()
    """
    got = _run(expression)
    assert got["prefix"] == "[video editing + keyframe completion + reference generation + audio reuse]"
    assert got["error"] is None


@requires_node
@pytest.mark.parametrize(
    "references,error",
    [
        ([{"kind": "image", "file": "a.png"}], "at least one reference role"),
        (
            [
                {"kind": "video", "file": "guide.mp4", "roles": ["reference_generation"]},
                {"kind": "video", "file": "plate.mp4", "roles": ["video_editing"]},
            ],
            "<Video 1>",
        ),
        (
            [
                {
                    "kind": "video",
                    "file": "silent.mp4",
                    "use_soundtrack": False,
                    "roles": ["audio_reference"],
                }
            ],
            "soundtrack",
        ),
    ],
)
def test_browser_validation_matches_backend_invariants(references, error):
    envelope = {"references": references, "task_plan": {"mode": "explicit"}}
    got = _run(
        "deriveTaskPlanState(fromReferencesEnvelope(" + json.dumps(envelope) + "))"
    )
    assert error in got["error"]


@requires_node
def test_new_reference_has_no_invented_role_and_video_sound_is_on():
    assert _run("newReference('image', 'a.png')") == {"file": "a.png", "roles": []}
    assert _run("newReference('video', 'v.mp4')") == {
        "file": "v.mp4",
        "roles": [],
        "use_soundtrack": True,
        "primary": False,
    }


@requires_node
def test_legacy_video_without_soundtrack_key_keeps_the_backend_default():
    envelope = {"references": [{"kind": "video", "file": "legacy.mp4"}]}
    assert _run(
        "toReferencesEnvelope(fromReferencesEnvelope(" + json.dumps(envelope) + "))"
    ) == {
        "references": [
            {"kind": "video", "file": "legacy.mp4", "use_soundtrack": True}
        ]
    }


@requires_node
def test_primary_video_round_trips_independently_of_roles():
    envelope = {
        "references": [
            {
                "kind": "video",
                "file": "plate.mp4",
                "use_soundtrack": True,
                "primary": True,
                "roles": ["video_editing"],
            }
        ],
        "task_plan": {"mode": "explicit"},
    }
    assert _run(
        "toReferencesEnvelope(fromReferencesEnvelope(" + json.dumps(envelope) + "))"
    ) == envelope


@requires_node
def test_replacement_plan_makes_a_roleless_primary_an_edit_source():
    expression = """
    (() => {
      const refs = fromReferencesEnvelope({
        references: [
          {kind: 'image', file: 'face.png', roles: ['reference_generation']},
          {kind: 'video', file: 'plate.mp4', primary: true}
        ],
        task_plan: {mode: 'explicit', specialization: 'character_replacement'}
      });
      return {
        changed: ensureReplacementPrimaryRole(refs),
        roles: refs.videos[0].roles,
        state: deriveTaskPlanState(refs)
      };
    })()
    """
    got = _run(expression)

    assert got["changed"] is True
    assert got["roles"] == ["video_editing"]
    assert got["state"]["error"] is None


@requires_node
def test_general_plan_does_not_guess_a_primary_video_role():
    expression = """
    (() => {
      const refs = fromReferencesEnvelope({
        references: [{
          kind: 'video', file: 'plate.mp4', primary: true,
          roles: ['reference_generation']
        }],
        task_plan: {mode: 'explicit', specialization: 'none'}
      });
      return {
        changed: ensureReplacementPrimaryRole(refs),
        roles: refs.videos[0].roles,
        state: deriveTaskPlanState(refs)
      };
    })()
    """
    got = _run(expression)

    assert got["changed"] is False
    assert got["roles"] == ["reference_generation"]
    assert "Use as editing source" in got["state"]["error"]


@requires_node
def test_selected_primary_is_promoted_to_video_one_without_reordering_guides():
    expression = """
    (() => {
      const refs = fromReferencesEnvelope({references: [
        {kind: 'video', file: 'guide-a.mp4', roles: ['reference_generation']},
        {kind: 'video', file: 'plate.mp4', primary: true, roles: ['video_editing']},
        {kind: 'video', file: 'guide-b.mp4', roles: ['reference_generation']}
      ], task_plan: {mode: 'explicit'}});
      normalizePrimaryVideo(refs);
      return {files: refs.videos.map((video) => video.file), state: deriveTaskPlanState(refs)};
    })()
    """
    got = _run(expression)

    assert got["files"] == ["plate.mp4", "guide-a.mp4", "guide-b.mp4"]
    assert got["state"]["error"] is None



@requires_node
def test_audio_only_on_the_editing_source_invalidates_a_replacement_plan():
    """Converting a video is not repaired for the plan: turning the primary editing
    source into audio drops its `video_editing` role, and a replacement specialization
    re-derives as invalid. The caller surfaces that on the Plan button.

    (Converting a video in FRONT of the driver cannot invalidate on position - the
    driver moves up into <Video 1>, which is the valid slot.)"""
    refs = {
        "images": [{"file": "face.png", "roles": ["reference_generation"]}],
        "videos": [
            {"file": "plate.mp4", "use_soundtrack": True, "primary": True,
             "roles": ["video_editing", "audio_reuse"]},
        ],
        "audios": [],
        "taskPlan": {"mode": "explicit", "specialization": "character_replacement"},
    }
    got = run(
        "const refs = " + json.dumps(refs) + ";"
        "\nconst converted = videoToAudioOnly(refs, 0, 3);"
        "\nconsole.log(JSON.stringify({before: deriveTaskPlanState(refs).error,"
        " after: deriveTaskPlanState(converted.refs).error,"
        " audios: converted.refs.audios}));\n",
        "references",
    )

    assert got["before"] is None
    assert got["after"] == 'A replacement specialization requires "Use as editing source".'
    assert got["audios"] == [{"file": "plate.mp4", "roles": ["audio_reuse"], "missing": False}]
