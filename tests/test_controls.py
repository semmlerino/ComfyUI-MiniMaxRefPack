"""Control metadata, portable preparation recipes and explicit HTTP graph planning."""

import asyncio
import json
import sys
import types

import pytest
import torch

from minimax_refpack import controls, nodes, refs, routes


class Edges:
    RETURN_TYPES = ("IMAGE",)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",),
                             "low_threshold": ("FLOAT", {"default": 0.4}),
                             "high_threshold": ("FLOAT", {"default": 0.8})}}


class Pose:
    RETURN_TYPES = ("IMAGE", "POSE_KEYPOINT")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",)}, "optional": {
            "resolution": ("INT", {"default": 512}),
            "detect_hand": (["enable", "disable"], {"default": "enable"}),
            "pose_estimator": (["a.pt", "b.onnx"], {"default": "a.pt"}),
        }}


@pytest.fixture
def prepared(tmp_path):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source video")
    control = {"file": source.name, "mode": "edges", "resolution": 64}
    plan = controls.preparation_graph(control, tmp_path, 1.0, {"Canny": Edges})
    mapped = tmp_path / "map.mp4"
    mapped.write_bytes(b"prepared map")
    ready = {"file": mapped.name, "source_sha256": controls.file_sha256(source),
             "recipe": plan["recipe"], "sha256": controls.file_sha256(mapped),
             "frame_count": 39, "fps": 24, "width": 64, "height": 96}
    return tmp_path, {**control, "map": ready}, plan


def test_control_round_trip_is_separate_from_references_and_task_plan(prepared):
    _directory, control, _plan = prepared
    envelope = {"references": [{"kind": "image", "file": "look.png"}],
                "task_plan": {"mode": "auto"}, "control": control}
    result = refs.ReferenceSet.from_obj(envelope)
    assert result.counts() == {"image": 1, "video": 0, "audio": 0}
    assert [t.tag for t in result.assign_tags()] == ["<Picture 1>"]
    assert json.loads(result.to_json())["control"] == controls.normalize_control(control)
    assert json.loads(refs.ReferenceSet.from_obj({"references": []}).to_json()) == {"references": []}


def test_preparation_graph_has_only_source_detector_and_save(prepared):
    _directory, _control, plan = prepared
    assert [n["class_type"] for n in plan["prompt"].values()] == [
        "MiniMaxH3ControlSource", "Canny", "MiniMaxH3ControlMapSave"]
    assert plan["prompt"]["2"]["inputs"] == {
        "image": ["1", 0], "low_threshold": 0.4, "high_threshold": 0.8}
    assert plan["prompt"]["3"]["inputs"]["images"] == ["2", 0]
    assert plan["output_node_id"] == "3"
    assert len(plan["workflow"]["id"]) == 36
    assert plan["workflow"]["extra"]["workflow_name"] == "MiniMax Control Map - Edges"
    assert plan["workflow"]["nodes"][1]["widgets_values"] == [0.4, 0.8]


def test_processor_defaults_and_availability_follow_registered_schema():
    settings = controls.processor_settings("pose", 768, {"DWPreprocessor": Pose})
    assert settings["inputs"] == {"resolution": 768, "detect_hand": "enable", "pose_estimator": "a.pt"}
    modes = {mode["id"]: mode for mode in controls.capabilities({"Canny": Edges})["modes"]}
    assert modes["edges"]["available"] is True
    assert not modes["depth"]["available"]
    assert "DepthAnythingV2Preprocessor" in modes["depth"]["reason"]


def test_incompatible_required_detector_input_is_reported():
    class Changed(Edges):
        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {"image": ("IMAGE",), "model": ("MODEL",)}}

    with pytest.raises(controls.ControlError, match="required input model has no default"):
        controls.processor_settings("edges", node_registry={"Canny": Changed})


@pytest.mark.parametrize("field,value", [
    ("file", "../outside.mp4"), ("file", "/tmp/source.mp4"),
    ("mode", "unknown"), ("resolution", True), ("resolution", 65),
    ("crop", [float("nan"), 0, 1, 1]), ("trim", [0, float("inf")]),
])
def test_malformed_control_is_rejected(field, value):
    control = {"file": "source.mp4", "mode": "edges", field: value}
    with pytest.raises(ValueError):
        controls.normalize_control(control)


def test_symlink_escape_is_rejected(tmp_path):
    inputs = tmp_path / "input"
    inputs.mkdir()
    outside = tmp_path / "outside.mp4"
    outside.write_bytes(b"private")
    (inputs / "linked.mp4").symlink_to(outside)
    with pytest.raises(controls.ControlError, match="inside"):
        controls.safe_path(inputs, "linked.mp4")


def test_outbox_renames_and_transient_preparation_do_not_change_recipe(prepared):
    directory, control, _plan = prepared
    (directory / "relabel-source.mp4").write_bytes((directory / control["file"]).read_bytes())
    (directory / "relabel-map.mp4").write_bytes((directory / control["map"]["file"]).read_bytes())
    moved = {**control, "file": "relabel-source.mp4", "map": {**control["map"], "file": "relabel-map.mp4"},
             "preparation": {"prompt_id": "new", "recipe": "ignored"}}
    cache = types.SimpleNamespace(video=lambda *a, **k: (torch.zeros(39, 96, 64, 3), None))
    assert controls.load_ready(moved, directory, 1, cache).shape == (39, 96, 64, 3)
    assert "file" not in moved["map"]["recipe"]


@pytest.mark.parametrize("mutation", ["source", "map", "duration", "resolution", "trim", "crop"])
def test_overwrites_and_recipe_changes_reject_stale_maps(prepared, mutation):
    directory, control, _plan = prepared
    seconds = 1
    if mutation in ("source", "map"):
        filename = control["file"] if mutation == "source" else control["map"]["file"]
        (directory / filename).write_bytes(b"replaced")
    elif mutation == "duration":
        seconds = 2
    else:
        control[mutation] = {"resolution": 128, "trim": [0, 1], "crop": [0, 0, 0.5, 1]}[mutation]
    with pytest.raises(controls.ControlError, match="stale|changed"):
        controls.load_ready(control, directory, seconds, None)


def test_manager_requires_map_before_rendering(prepared, monkeypatch):
    directory, control, _plan = prepared
    control["map"] = None
    monkeypatch.setattr(nodes, "_provider_of", lambda *args: "none")
    monkeypatch.setitem(sys.modules, "folder_paths", types.SimpleNamespace(get_input_directory=lambda: str(directory)))
    with pytest.raises(controls.ControlError, match="Generate control map"):
        nodes.MiniMaxH3ReferencePack().build(references_json=json.dumps({"control": control}), length_seconds=1)


def test_manager_control_output_aspect_cache_and_no_vlm_refs(prepared, monkeypatch):
    directory, control, _plan = prepared
    monkeypatch.setitem(sys.modules, "folder_paths", types.SimpleNamespace(get_input_directory=lambda: str(directory)))
    monkeypatch.setattr(nodes.media, "load_video", lambda *a, **k: (torch.zeros(39, 96, 64, 3), None))
    raw = json.dumps({"references": [], "control": control})
    manager = nodes.MiniMaxH3ReferencePack()
    before = manager.IS_CHANGED(references_json=raw, length_seconds=1)
    outputs = dict(zip(refs.output_names(), manager.build(
        references_json=raw, length_seconds=1, prompt_provider="none", direction="shot",
        match_video_aspect=True, width=1280, height=720)))
    assert refs.slot_index("control_video") == 22
    assert outputs["control_video"].shape == (39, 96, 64, 3)
    assert outputs["video_1"] is None
    assert outputs["prompt"] == "shot"
    assert (outputs["width"], outputs["height"]) == (768, 1184)
    assert "references: 0 (none)" in outputs["debug"]
    (directory / control["map"]["file"]).write_bytes(b"new map")
    assert manager.IS_CHANGED(references_json=raw, length_seconds=1) != before


def test_http_prepare_returns_graph_without_submission(prepared, monkeypatch):
    directory, control, _plan = prepared
    monkeypatch.setattr(routes, "folder_paths", types.SimpleNamespace(get_input_directory=lambda: str(directory)))
    monkeypatch.setattr(controls, "registry", lambda: {"Canny": Edges})

    class Request:
        async def json(self):
            return {"control": control, "length_seconds": 1}

    response = asyncio.run(routes.control_prepare_route(Request()))
    assert response.status == 200
    assert json.loads(response.text)["output_node_id"] == "3"
    control["file"] = "../outside.mp4"
    response = asyncio.run(routes.control_prepare_route(Request()))
    assert response.status == 400
