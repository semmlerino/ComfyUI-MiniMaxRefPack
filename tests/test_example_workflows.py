"""The shipped example workflows, checked against the node's own INPUT_TYPES.

Two user-reported breakages came from this file rather than from the code, and both
were invisible to every other test because nothing loaded the JSON we hand out:

  - "The value for MiniMax References Manager's max_reference_edge couldn't be
    converted to INT" - widgets_values carried "" in the slot litegraph restores
    max_reference_edge from. Positional restore means a wrong value in slot N is a
    wrong value for widget N, so a type check per slot is the whole guard.
  - "Unknown pack: MinimaxH3referencepack" - the node's properties carried no
    cnr_id, so ComfyUI could not map the missing node to its registry pack and
    guessed a name from the node type.

These run without ComfyUI: INPUT_TYPES is read straight off the class, and the only
entry that touches the network (`model`, whose list comes from OpenRouter) falls back
to a hardcoded list, so the widget ORDER is stable either way.
"""

import json
import pathlib
import re

import pytest

from minimax_refpack import nodes

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
WORKFLOW_DIR = REPO_ROOT / "example_workflows"

# What `comfy node publish` registered this pack as. ComfyUI reads properties.cnr_id
# to offer "Install Missing Custom Nodes"; without it the user gets an unresolvable
# pack name and a dead end.
REGISTRY_ID = "comfyui-minimaxrefpack"


def _package_version() -> str:
    """The version pyproject declares, which is what `comfy node publish` uploads."""
    text = (REPO_ROOT / "pyproject.toml").read_text()
    match = re.search(r'^version = "([^"]+)"', text, re.M)
    assert match, "no version in pyproject.toml"
    return match.group(1)

WORKFLOWS = sorted(WORKFLOW_DIR.glob("*.json"))


def _widget_spec():
    """(name, type_spec) per widget, in the order litegraph restores them."""
    spec = nodes.MiniMaxH3ReferencePack.INPUT_TYPES()
    ordered = list(spec.get("required", {}).items()) + list(spec.get("optional", {}).items())
    return [(name, entry[0]) for name, entry in ordered]


def _pack_nodes(path):
    graph = json.loads(path.read_text())
    return [n for n in graph.get("nodes", []) if n.get("type") == "MiniMaxH3ReferencePack"]


def test_there_is_at_least_one_example_workflow():
    assert WORKFLOWS, f"no workflows found under {WORKFLOW_DIR}"


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
def test_every_workflow_is_valid_json(path):
    json.loads(path.read_text())


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
def test_saved_widget_values_match_the_declared_types(path):
    """A saved value in slot N must be loadable into widget N.

    ComfyUI validates widgets by declared type before the node ever runs, so a
    string in an INT slot fails the prompt with the error the user pasted, not
    with anything this repo controls.
    """
    checkers = {
        "INT": lambda v: isinstance(v, int) and not isinstance(v, bool),
        "FLOAT": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
        "STRING": lambda v: isinstance(v, str),
        "BOOLEAN": lambda v: isinstance(v, bool),
    }
    widgets = _widget_spec()

    for node in _pack_nodes(path):
        values = node.get("widgets_values") or []
        assert len(values) <= len(widgets), (
            f"{path.name}: {len(values)} saved values for {len(widgets)} widgets - "
            "a stale value would land in the wrong slot"
        )
        for value, (name, type_spec) in zip(values, widgets):
            if isinstance(type_spec, list):        # a combo: the value must be one of them
                assert value in type_spec, (
                    f"{path.name}: {name}={value!r} is not one of {type_spec}"
                )
                continue
            check = checkers.get(type_spec)
            if check is None:
                continue
            assert check(value), (
                f"{path.name}: {name} is declared {type_spec} but the workflow saves "
                f"{value!r} ({type(value).__name__}). ComfyUI will refuse to run this."
            )


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
def test_the_node_declares_its_registry_pack(path):
    """Without cnr_id, "Install Missing Custom Nodes" cannot find this pack."""
    for node in _pack_nodes(path):
        properties = node.get("properties") or {}
        assert properties.get("cnr_id") == REGISTRY_ID, (
            f"{path.name}: properties.cnr_id is {properties.get('cnr_id')!r}, expected "
            f"{REGISTRY_ID!r} - otherwise ComfyUI reports 'Unknown pack' for a pack that "
            "is published and installable."
        )
        assert properties.get("ver"), f"{path.name}: properties.ver is missing"


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
def test_the_workflows_version_stamp_tracks_the_release(path):
    """`properties.ver` must equal the version being published, on every release.

    This is a chore that will otherwise be forgotten, so it is a test rather than a
    habit. ComfyUI stamps this field from whatever the AUTHOR had installed, and a dev
    checkout stamps a git SHA: the file handed over on 2026-08-17 carried
    760cc261e7bc... which is 0.2.1, three releases stale. Shipping that means "Install
    Missing Custom Nodes" resolves against an old git ref instead of the published
    version, which is a quieter version of the "Unknown pack" bug 0.3.1 shipped with.
    """
    expected = _package_version()
    for node in _pack_nodes(path):
        properties = node.get("properties") or {}
        assert properties.get("ver") == expected, (
            f"{path.name}: properties.ver is {properties.get('ver')!r} but pyproject "
            f"declares {expected!r}. Bump the workflow's ver when you bump the release."
        )


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
def test_the_workflows_claim_registry_provenance_not_a_git_checkout(path):
    """`aux_id` is what ComfyUI stamps for a git-installed pack. A shipped workflow is
    installed from the registry, so it carries `cnr_id` alone; having both invites the
    frontend to resolve the wrong one."""
    for node in _pack_nodes(path):
        properties = node.get("properties") or {}
        assert "aux_id" not in properties, (
            f"{path.name}: properties carries aux_id={properties.get('aux_id')!r}, which "
            "means this was saved from a dev checkout rather than a registry install"
        )


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
def test_no_link_points_at_a_node_that_is_not_there(path):
    """Deleting a node by hand can leave a link behind, which breaks the graph on load."""
    graph = json.loads(path.read_text())
    ids = {n["id"] for n in graph.get("nodes", [])}
    dangling = [l for l in graph.get("links", []) if l[1] not in ids or l[3] not in ids]
    assert not dangling, f"{path.name}: {len(dangling)} link(s) reference a missing node"


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
def test_video_save_uses_render_time_prefix_and_embeds_metadata(path):
    """Example renders should be identifiable and reopenable after downloading."""
    graph = json.loads(path.read_text())
    by_type = {node["type"]: node for node in graph.get("nodes", [])}
    timer = by_type.get("RenderTimeFilenamePrefix")
    stamper = by_type.get("PromptToWorkflow")
    save = by_type.get("VHS_VideoCombine")

    assert timer is not None, f"{path.name}: render-time filename node is missing"
    assert stamper is not None, f"{path.name}: generated-prompt metadata node is missing"
    assert save is not None, f"{path.name}: VHS video save node is missing"
    assert save.get("widgets_values", {}).get("save_metadata") is True, (
        f"{path.name}: VHS save_metadata must stay enabled"
    )

    links = {link[0]: link for link in graph.get("links", [])}
    save_inputs = {entry["name"]: entry for entry in save.get("inputs", [])}
    images_link = links[save_inputs["images"]["link"]]
    prefix_link = links[save_inputs["filename_prefix"]["link"]]
    assert images_link[1:3] == [stamper["id"], 0], (
        f"{path.name}: saved frames do not pass through the prompt metadata node"
    )
    assert prefix_link[1:3] == [timer["id"], 1], (
        f"{path.name}: save filename_prefix is not driven by the render timer"
    )

    stamper_inputs = {entry["name"]: entry for entry in stamper.get("inputs", [])}
    passthrough_link = links[stamper_inputs["passthrough"]["link"]]
    prompt_link = links[stamper_inputs["prompt"]["link"]]
    pack = by_type["MiniMaxH3ReferencePack"]
    assert passthrough_link[1:3] == [timer["id"], 0], (
        f"{path.name}: metadata node does not receive the timer's frames"
    )
    assert prompt_link[1:3] == [pack["id"], 18], (
        f"{path.name}: metadata node does not receive the generated Auto Prompt"
    )


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
def test_runtime_models_match_the_supported_pod_pack(path):
    graph = json.loads(path.read_text())
    by_type = {node["type"]: node for node in graph.get("nodes", [])}

    unet = by_type["UNETLoader"]
    expected_unet = "minimax_h3_ref2va_pruned_int8_convrot.safetensors"
    assert unet["widgets_values"][0] == expected_unet
    assert unet["properties"]["models"][0]["name"] == expected_unet

    power_lora = by_type["Power Lora Loader (rgthree)"]
    choices = [
        value
        for value in power_lora["widgets_values"]
        if isinstance(value, dict) and "lora" in value
    ]
    assert [choice["lora"] for choice in choices] == [
        "HMNSFW_AIO_V2.safetensors",
        "HMBreasts_085e0750_e40.safetensors",
    ]
    assert choices[0]["on"] is True
    assert choices[0]["strength"] == 0.4


# Preview nodes cache their last result INTO the saved graph. Whatever the author last
# generated therefore ships with the workflow. On 2026-08-17 that was 2,910 characters of
# explicit generated prose sitting in a Display Any node, which would have gone to a
# public repo and the ComfyUI registry listing. The direction field had already been
# cleaned by hand; this one was invisible because nobody thinks of a display as content.
PREVIEW_NODES = ("Display Any (rgthree)", "PreviewAny", "ShowText|pysssss", "Note")


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
def test_preview_nodes_ship_empty(path):
    graph = json.loads(path.read_text())
    for node in graph.get("nodes", []):
        if node.get("type") not in PREVIEW_NODES:
            continue
        for value in node.get("widgets_values") or []:
            assert not (isinstance(value, str) and value.strip()), (
                f"{path.name}: {node['type']} carries {len(value)} chars of a previous "
                "run. Clear it before shipping: whatever you last generated goes public "
                "with the workflow."
            )


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
def test_the_render_takes_its_frame_from_the_pack(path):
    """The shipped graph renders the frame its prompt was written for.

    With match_video_aspect on, the pack's width/height outputs follow the reference
    video's shape. Feeding MiniMax's node from the ResolutionSelector instead would tell
    the model one frame and render another.
    """
    from minimax_refpack import refs

    graph = json.loads(path.read_text())
    links = {row[0]: row for row in graph.get("links", [])}
    names = [name for name, _ in _widget_spec()]
    for pack in _pack_nodes(path):
        values = dict(zip(names, pack.get("widgets_values") or []))
        assert values.get("match_video_aspect") is True, f"{path.name}: switch is off"
        outputs = [o["name"] for o in pack.get("outputs") or []]
        assert outputs == list(refs.output_names()), f"{path.name}: outputs {outputs}"
        renders = [n for n in graph["nodes"] if n.get("type") == "MiniMaxH3ReferenceToVideo"]
        assert renders, f"{path.name}: no MiniMaxH3ReferenceToVideo"
        for render in renders:
            for entry in render.get("inputs") or []:
                if entry["name"] not in ("width", "height"):
                    continue
                row = links[entry["link"]]
                assert (row[1], row[2]) == (pack["id"], refs.slot_index(entry["name"])), (
                    f"{path.name}: {entry['name']} comes from node {row[1]} slot {row[2]}"
                )
