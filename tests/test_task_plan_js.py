"""Exercise the task-plan/reference transport code from the shipped browser module."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
REFPACK_JS = REPO_ROOT / "web" / "refpack.js"
REFPACK_CSS = REPO_ROOT / "web" / "refpack.css"
NODE = shutil.which("node")
requires_node = pytest.mark.skipif(NODE is None, reason="node is not installed")


def _extract_js() -> str:
    text = REFPACK_JS.read_text()
    start = text.find("// >>> MMRP-TASK-PLAN")
    end = text.find("// <<< MMRP-TASK-PLAN")
    assert start != -1 and end != -1, "the MMRP-TASK-PLAN test seam is missing"
    return text[start:end]


def _run(expression: str):
    script = _extract_js() + f"\nconsole.log(JSON.stringify({expression}));\n"
    proc = subprocess.run(
        [NODE, "--input-type=module", "-e", script],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"node failed: {proc.stderr}"
    return json.loads(proc.stdout.strip())


def test_task_plan_shows_and_refreshes_picture_and_video_thumbnails():
    text = REFPACK_JS.read_text()
    start = text.index("async function openTaskPlanModal(node)")
    end = text.index("// Config (reference pack) save/load", start)
    modal = text[start:end]

    assert 'if (kind === "image" || kind === "video")' in modal
    assert 'thumbnail.className = "mmrp-plan-thumbnail-image"' in modal
    assert "thumbnail.src = thumbUrl(reference.file, reference)" in modal
    assert modal.count("refreshThumbnail();") == 2  # initial render + Source change
    assert ".mmrp-plan-thumbnail-image" in REFPACK_CSS.read_text()


def test_task_plan_primary_video_control_explains_and_requires_a_driver_role():
    text = REFPACK_JS.read_text()
    start = text.index("async function openTaskPlanModal(node)")
    end = text.index("// Config (reference pack) save/load", start)
    modal = text[start:end]

    assert 'document.createTextNode("Primary video")' in modal
    assert "Primary edit / continuation video" not in modal
    assert 'primaryControl.dataset.requiresDriver = "true"' in modal
    assert "Replacement plans use the editing source." in modal


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
    text = REFPACK_JS.read_text()
    start = text.find("// >>> MMRP-AUDIO-ONLY")
    end = text.find("// <<< MMRP-AUDIO-ONLY")
    assert start != -1 and end != -1, "the MMRP-AUDIO-ONLY test seam is missing"
    refs = {
        "images": [{"file": "face.png", "roles": ["reference_generation"]}],
        "videos": [
            {"file": "plate.mp4", "use_soundtrack": True, "primary": True,
             "roles": ["video_editing", "audio_reuse"]},
        ],
        "audios": [],
        "taskPlan": {"mode": "explicit", "specialization": "character_replacement"},
    }
    script = (
        _extract_js() + text[start:end]
        + "\nconst refs = " + json.dumps(refs) + ";"
        + "\nconst converted = videoToAudioOnly(refs, 0, 3);"
        + "\nconsole.log(JSON.stringify({before: deriveTaskPlanState(refs).error,"
        + " after: deriveTaskPlanState(converted.refs).error,"
        + " audios: converted.refs.audios}));\n"
    )
    proc = subprocess.run(
        [NODE, "--input-type=module", "-e", script], capture_output=True, text=True, check=False
    )
    assert proc.returncode == 0, f"node failed: {proc.stderr}"
    got = json.loads(proc.stdout.strip())

    assert got["before"] is None
    assert got["after"] == 'A replacement specialization requires "Use as editing source".'
    assert got["audios"] == [{"file": "plate.mp4", "roles": ["audio_reuse"], "missing": False}]
