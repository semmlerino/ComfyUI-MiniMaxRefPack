"""Real CPU media preparation, map encoding and receipt validation."""

import json

import av
import pytest
import torch

from minimax_refpack import control_nodes, controls, media
from .conftest import write_clip
from .test_controls import Edges


def setup_source(tmp_path, *, crop=None, trim=None, seconds=1):
    source = write_clip(tmp_path / "source.mp4", n=48, fps=24, size=(64, 48), container="mp4")
    control = {"file": "source.mp4", "mode": "edges", "resolution": 64, "crop": crop, "trim": trim}
    recipe = controls.preparation_graph(control, tmp_path, seconds, {"Canny": Edges})["recipe"]
    assert source.endswith("source.mp4")
    return control, recipe


def test_source_decode_is_cpu_aligned_trimmed_and_resized(tmp_path):
    control, recipe = setup_source(tmp_path, crop=[0, 0, 0.5, 1], trim=[0.25, 1.5])
    frames = controls.prepare_frames(control, recipe, tmp_path)
    assert frames.device.type == "cpu"
    assert frames.dtype == torch.float32
    assert frames.shape == (39, 96, 64, 3)


def test_source_replacement_before_preparation_is_rejected(tmp_path):
    control, recipe = setup_source(tmp_path)
    (tmp_path / control["file"]).write_bytes(b"replaced")
    with pytest.raises(controls.ControlError, match="stale"):
        controls.prepare_frames(control, recipe, tmp_path)


@pytest.mark.parametrize("seconds,count", [(5, 124), (8, 192)])
def test_target_duration_snaps_up_and_short_source_holds_last_frame(tmp_path, seconds, count):
    control, recipe = setup_source(tmp_path, seconds=seconds)
    frames = controls.prepare_frames(control, recipe, tmp_path)
    assert frames.shape[0] == count
    assert torch.equal(frames[-1], frames[47])
    assert not torch.equal(frames[0], frames[-1])


def test_source_driven_length_snaps_down_without_padding(tmp_path):
    control, recipe = setup_source(tmp_path, seconds=0)
    assert controls.prepare_frames(control, recipe, tmp_path).shape[0] == 39


def test_prepared_output_is_checked_against_existing_media_budget(tmp_path, monkeypatch):
    control, recipe = setup_source(tmp_path, seconds=8)
    monkeypatch.setattr(media, "load_video", lambda *a, **k: (torch.zeros(48, 48, 64, 3), None))
    monkeypatch.setattr(media, "_decode_budget", lambda: 1000)
    with pytest.raises(controls.ControlError, match="decode budget"):
        controls.prepare_frames(control, recipe, tmp_path)


def test_decode_window_covers_complete_snapped_target(tmp_path, monkeypatch):
    control, recipe = setup_source(tmp_path, seconds=5)
    seen = []
    original = media.load_video

    def load(path, **kwargs):
        seen.append(kwargs["trim"])
        return original(path, **kwargs)

    monkeypatch.setattr(media, "load_video", load)
    controls.prepare_frames(control, recipe, tmp_path)
    assert seen == [[0, 124 / 24]]


def test_streamed_h264_map_round_trip_and_history_receipt(tmp_path, monkeypatch):
    control, recipe = setup_source(tmp_path)
    frames = torch.zeros(39, 48, 64, 3)
    frames[..., 0] = 0.8
    monkeypatch.setattr(control_nodes, "_input_dir", lambda: str(tmp_path))
    result = control_nodes.MiniMaxH3ControlMapSave().save(frames, json.dumps(control), json.dumps(recipe))
    ready = result["ui"]["control_map"][0]
    assert result["result"] == ()
    assert ready["frame_count"] == 39
    assert ready["fps"] == 24
    assert ready["file"].startswith("control_maps/source-edges-")
    with av.open(str(tmp_path / ready["file"])) as container:
        stream = container.streams.video[0]
        assert stream.average_rate == 24
        assert stream.codec_context.name == "h264"
        assert stream.codec_context.pix_fmt == "yuv420p"
        assert len(list(container.decode(stream))) == 39
        assert len(container.streams.audio) == 0
    decoded = controls.load_ready({**control, "map": ready}, tmp_path, 1, media.MediaCache())
    assert decoded.shape == frames.shape
    assert torch.max(torch.abs(decoded - frames)).item() < 0.06
    assert not list((tmp_path / "control_maps").glob(".control-*"))


def test_bad_pixel_data_cannot_publish_a_partial_map(tmp_path):
    control, recipe = setup_source(tmp_path)
    frames = torch.full((39, 48, 64, 3), float("nan"))
    with pytest.raises(controls.ControlError, match="finite"):
        control_nodes.write_map(frames, control, recipe, tmp_path)
    assert not list((tmp_path / "control_maps").glob("*.mp4"))


def test_regeneration_preserves_previous_reviewed_artifact(tmp_path):
    control, recipe = setup_source(tmp_path)
    frames = torch.zeros(39, 48, 64, 3)
    first = control_nodes.write_map(frames, control, recipe, tmp_path)
    frames[..., 1] = 0.9
    second = control_nodes.write_map(frames, control, recipe, tmp_path)
    assert first["file"] != second["file"]
    assert controls.file_sha256(tmp_path / first["file"]) == first["sha256"]
    assert controls.file_sha256(tmp_path / second["file"]) == second["sha256"]


@pytest.mark.parametrize("file,stem", [
    ("source.mp4", "source"),
    ("videos/In Wien (final).mp4", "In_Wien_final"),
    ("clips/.hidden.mov", "hidden"),
    ("???.mp4", "source"),
    (f"{'a' * 80}.mp4", "a" * 48),
])
def test_map_name_carries_its_source_video_stem(file, stem):
    assert control_nodes.map_name_stem(file) == stem


def test_control_maps_directory_symlink_cannot_escape_input(tmp_path):
    inputs = tmp_path / "input"
    inputs.mkdir()
    control, recipe = setup_source(inputs)
    outside = tmp_path / "outside"
    outside.mkdir()
    (inputs / "control_maps").symlink_to(outside)
    with pytest.raises(controls.ControlError, match="inside"):
        control_nodes.write_map(torch.zeros(39, 48, 64, 3), control, recipe, inputs)
    assert not list(outside.iterdir())


@pytest.mark.parametrize("shape", [(1, 48, 64, 3), (6, 48, 64, 3), (5, 47, 64, 3)])
def test_invalid_map_geometry_is_rejected(tmp_path, shape):
    control, recipe = setup_source(tmp_path)
    with pytest.raises(controls.ControlError):
        control_nodes.write_map(torch.zeros(shape), control, recipe, tmp_path)


def test_output_compatibility_and_source_node_cache_identity(tmp_path, monkeypatch):
    from minimax_refpack import refs

    assert refs.output_names()[:22] == (
        *(f"image_{i}" for i in range(1, 10)), *(f"video_{i}" for i in range(1, 4)),
        *(f"video_audio_{i}" for i in range(1, 4)), *(f"audio_{i}" for i in range(1, 4)),
        "prompt", "debug", "width", "height")
    assert refs.output_types()[22] == "IMAGE"
    assert refs.empty_outputs()[18:] == ["", "", 0, 0, None]
    control, recipe = setup_source(tmp_path)
    monkeypatch.setattr(control_nodes, "_input_dir", lambda: str(tmp_path))
    args = (json.dumps(control), json.dumps(recipe))
    before = control_nodes.MiniMaxH3ControlSource.IS_CHANGED(*args)
    (tmp_path / "source.mp4").write_bytes(b"replacement")
    assert control_nodes.MiniMaxH3ControlSource.IS_CHANGED(*args) != before
