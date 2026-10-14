"""Standalone saved control maps need no original plate, detector or queued job."""

import asyncio
import json
import sys
import threading
import time
import types

import pytest
import torch
from PIL import Image

from minimax_refpack import controls, media, nodes, refs, routes
from .conftest import write_clip


@pytest.fixture
def saved_map(tmp_path):
    write_clip(tmp_path / "pose.mp4", n=60, fps=30, size=(64, 48), container="mp4", marker="halves")
    return {"file": "pose.mp4", "input_type": "map", "mode": "pose", "resolution": 64,
            "trim": [0.2, 1.0], "crop": [0, 0, 0.5, 1]}


def test_saved_map_imports_and_loads_without_source_detector_or_derivative(saved_map, tmp_path, monkeypatch):
    monkeypatch.setattr(controls, "registry", lambda: pytest.fail("an imported map must not consult detectors"))
    before = set(tmp_path.iterdir())
    ready = controls.import_map(saved_map, tmp_path, 1)
    assert set(tmp_path.iterdir()) == before
    assert ready["input_type"] == "map"
    assert ready["map"]["file"] == ready["file"]
    assert ready["map"]["sha256"] == ready["map"]["source_sha256"] == controls.file_sha256(tmp_path / ready["file"])
    assert ready["map"]["recipe"]["input_type"] == "map"
    assert ready["map"]["recipe"]["processor"] is None
    assert "file" not in ready["map"]["recipe"]
    assert (ready["map"]["frame_count"], ready["map"]["fps"], ready["map"]["width"], ready["map"]["height"]) == (39, 24, 64, 96)
    frames = controls.load_ready(ready, tmp_path, 1, media.MediaCache())
    assert frames.shape == (39, 96, 64, 3)
    assert frames.device.type == "cpu"
    assert frames[..., 0].mean() > 0.9
    # H.264 chroma crosses the red/blue boundary; the crop's interior stays red.
    assert frames[:, :, :-4, 1:].max() < 0.1
    assert torch.equal(frames[-1], frames[-2])
    serialized = refs.ReferenceSet.from_obj({"references": [], "control": ready}).to_json()
    assert json.loads(serialized)["control"] == ready


def test_raw_map_source_driven_frame_length_resamples_to_24fps(tmp_path):
    write_clip(tmp_path / "depth.mp4", n=60, fps=30, size=(64, 48), container="mp4")
    ready = controls.import_map({"input_type": "map", "mode": "depth", "file": "depth.mp4", "resolution": 64}, tmp_path, 0)
    frames = controls.load_ready(ready, tmp_path, 0, media.MediaCache())
    assert frames.shape[0] == 39  # 60 frames at 30fps -> 48 at 24fps -> 39 on the grid.
    assert frames.shape[1:3] == (64, 86)


def test_import_timing_uses_trimmed_source_frames_and_holds_only_the_tail(tmp_path):
    write_clip(tmp_path / "map.nut", n=60, fps=30, size=(64, 48), pix_fmt="rgb24")
    raw = {"input_type": "map", "mode": "edges", "file": "map.nut", "resolution": 64, "trim": [0.2, 1]}
    ready = controls.import_map(raw, tmp_path, 1)
    frames = controls.load_ready(ready, tmp_path, 1, media.MediaCache())
    red = torch.round(frames[:, 0, 0, 0] * 255).tolist()
    assert red[0] == 6
    assert red[1] == 7
    assert red[18] == 28
    assert red[19:] == [28] * 20


def test_manager_keeps_standalone_map_and_editing_video_separate(saved_map, tmp_path, monkeypatch):
    ready = controls.import_map(saved_map, tmp_path, 1)
    write_clip(tmp_path / "video-b.mp4", n=22, fps=24, size=(96, 64), container="mp4")
    monkeypatch.setattr(controls, "registry", lambda: pytest.fail("detectors must not be used when rendering an imported map"))
    monkeypatch.setitem(sys.modules, "folder_paths", types.SimpleNamespace(get_input_directory=lambda: str(tmp_path)))
    raw = json.dumps({"references": [{"kind": "video", "file": "video-b.mp4", "use_soundtrack": False}], "control": ready})
    result = nodes.MiniMaxH3ReferencePack().build(references_json=raw, length_seconds=1, prompt_provider="none")
    outputs = dict(zip(refs.output_names(), result))
    assert outputs["control_video"].shape == (39, 96, 64, 3)
    assert outputs["video_1"].shape == (22, 64, 96, 3)
    assert "references: 1 (<Video 1> video-b.mp4)" in outputs["debug"]
    assert outputs["video_audio_1"] is None


def test_input_picker_lists_nested_saved_maps(saved_map, tmp_path, monkeypatch):
    (tmp_path / "control_maps").mkdir()
    (tmp_path / saved_map["file"]).rename(tmp_path / "control_maps" / saved_map["file"])
    saved_map["file"] = "control_maps/pose.mp4"
    ready = controls.import_map(saved_map, tmp_path, 1)
    monkeypatch.setattr(routes, "folder_paths", types.SimpleNamespace(
        get_input_directory=lambda: str(tmp_path),
        filter_files_content_types=lambda files, kinds: [file for file in files if file.endswith(".mp4")],
    ))
    response = asyncio.run(routes.list_files_route(types.SimpleNamespace(query={"kind": "video"})))
    assert ready["file"] in json.loads(response.text)["files"]


@pytest.mark.parametrize("change", ["file", "mode", "resolution", "trim", "crop", "duration", "input_type"])
def test_imported_map_rejects_replacements_and_stale_settings(saved_map, tmp_path, change):
    ready = controls.import_map(saved_map, tmp_path, 1)
    seconds = 1
    if change == "file":
        (tmp_path / ready["file"]).write_bytes(b"replacement")
    elif change == "duration":
        seconds = 2
    else:
        ready[change] = {"mode": "depth", "resolution": 128, "trim": [0, 1],
                         "crop": None, "input_type": "source"}[change]
    with pytest.raises(controls.ControlError, match="stale|recipe"):
        controls.load_ready(ready, tmp_path, seconds, media.MediaCache())


def test_same_file_outbox_relocation_keeps_imported_receipt_valid(saved_map, tmp_path):
    ready = controls.import_map(saved_map, tmp_path, 1)
    (tmp_path / "pod-map.mp4").write_bytes((tmp_path / ready["file"]).read_bytes())
    (tmp_path / ready["file"]).unlink()
    moved = {**ready, "file": "pod-map.mp4", "map": {**ready["map"], "file": "pod-map.mp4"}}
    assert controls.load_ready(moved, tmp_path, 1, media.MediaCache()).shape == (39, 96, 64, 3)


@pytest.mark.parametrize("bad_input", ["corrupt", "image", "audio", "short", "short_trim"])
def test_invalid_or_short_imported_media_has_clear_error(tmp_path, bad_input):
    filename = "map.mp4"
    trim = None
    if bad_input == "corrupt":
        (tmp_path / filename).write_bytes(b"not a video")
    elif bad_input == "image":
        filename = "map.png"
        Image.new("RGB", (64, 48)).save(tmp_path / filename)
    elif bad_input == "audio":
        write_clip(tmp_path / filename, n=0, fps=24, container="mp4", audio="aac", audio_seconds=1)
    else:
        write_clip(tmp_path / filename, n=3 if bad_input == "short" else 48, fps=24, container="mp4")
        trim = [0, 0.05] if bad_input == "short_trim" else None
    with pytest.raises(controls.ControlError, match="video|5 frame"):
        controls.import_map({"input_type": "map", "file": filename, "mode": "edges", "resolution": 64, "trim": trim}, tmp_path, 1)


@pytest.mark.parametrize("filename", ["../private.mp4", "/tmp/private.mp4", "linked.mp4"])
def test_import_cannot_escape_input_directory(tmp_path, filename):
    inputs = tmp_path / "input"
    inputs.mkdir()
    outside = tmp_path / "private.mp4"
    outside.write_bytes(b"private")
    (inputs / "linked.mp4").symlink_to(outside)
    with pytest.raises(controls.ControlError, match="inside"):
        controls.import_map({"input_type": "map", "file": filename, "mode": "edges"}, inputs, 1)


def test_map_input_cannot_enter_detector_preparation(saved_map, tmp_path):
    with pytest.raises(controls.ControlError, match="import"):
        controls.preparation_graph(saved_map, tmp_path, 1, {})


def test_unknown_input_type_is_rejected(saved_map):
    with pytest.raises(controls.ControlError, match="input_type"):
        controls.normalize_control({**saved_map, "input_type": "other"})


def test_imported_receipt_cannot_reference_a_second_file(saved_map, tmp_path):
    ready = controls.import_map(saved_map, tmp_path, 1)
    ready["map"]["file"] = "other.mp4"
    with pytest.raises(controls.ControlError, match="same file"):
        controls.load_ready(ready, tmp_path, 1, media.MediaCache())


def test_import_rejects_a_file_replaced_while_decoding(saved_map, tmp_path, monkeypatch):
    original = controls.prepare_frames

    def decode(*args, **kwargs):
        frames = original(*args, **kwargs)
        write_clip(tmp_path / saved_map["file"], n=60, fps=30, size=(64, 48), container="mp4", marker="gray")
        return frames

    monkeypatch.setattr(controls, "prepare_frames", decode)
    with pytest.raises(controls.ControlError, match="changed|stale"):
        controls.import_map(saved_map, tmp_path, 1)


def test_load_rejects_an_imported_map_replaced_during_decode(saved_map, tmp_path, monkeypatch):
    ready = controls.import_map(saved_map, tmp_path, 1)
    original = controls.prepare_frames

    def decode(*args, **kwargs):
        frames = original(*args, **kwargs)
        write_clip(tmp_path / saved_map["file"], n=60, fps=30, size=(64, 48), container="mp4", marker="gray")
        return frames

    monkeypatch.setattr(controls, "prepare_frames", decode)
    with pytest.raises(controls.ControlError, match="changed|stale"):
        controls.load_ready(ready, tmp_path, 1, media.MediaCache())


def test_http_import_is_threaded_cpu_validation_with_no_prompt(saved_map, tmp_path, monkeypatch):
    monkeypatch.setattr(routes, "folder_paths", types.SimpleNamespace(get_input_directory=lambda: str(tmp_path)))
    calls = []
    original = routes.asyncio.to_thread

    async def offload(function, *args):
        calls.append(function)
        return await original(function, *args)

    monkeypatch.setattr(routes.asyncio, "to_thread", offload)

    class Request:
        async def json(self):
            return {"control": saved_map, "length_seconds": 1}

    response = asyncio.run(routes.control_import_route(Request()))
    assert response.status == 200
    body = json.loads(response.text)
    assert set(body) == {"control"}
    assert body["control"]["map"]["frame_count"] == 39
    assert calls == [controls.import_map]
    assert ("POST", "/minimax_refpack/control/import") in {(method, path) for method, path, _handler in routes._ROUTES}
    saved_map["file"] = "../private.mp4"
    response = asyncio.run(routes.control_import_route(Request()))
    assert response.status == 400


@pytest.mark.parametrize("body", [None, [], {"control": "invalid"}, {"control": {"input_type": "source"}}])
def test_http_import_rejects_malformed_body(tmp_path, monkeypatch, body):
    monkeypatch.setattr(routes, "folder_paths", types.SimpleNamespace(get_input_directory=lambda: str(tmp_path)))

    class Request:
        async def json(self):
            return body

    response = asyncio.run(routes.control_import_route(Request()))
    assert response.status == 400
    assert "error" in json.loads(response.text)


def test_concurrent_http_imports_share_one_decode_budget(tmp_path, monkeypatch):
    monkeypatch.setattr(routes, "folder_paths", types.SimpleNamespace(get_input_directory=lambda: str(tmp_path)))
    monkeypatch.setattr(routes, "_CONTROL_IMPORT_LOCK", asyncio.Lock())
    active, peak = 0, 0
    state_lock = threading.Lock()

    def imported(*args):
        nonlocal active, peak
        with state_lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.03)
        with state_lock:
            active -= 1
        return {"input_type": "map"}

    monkeypatch.setattr(controls, "import_map", imported)

    class Request:
        async def json(self):
            return {"control": {"input_type": "map"}}

    async def requests():
        return await asyncio.gather(routes.control_import_route(Request()), routes.control_import_route(Request()))

    responses = asyncio.run(requests())
    assert [response.status for response in responses] == [200, 200]
    assert peak == 1


@pytest.mark.parametrize("cancellations", [1, 2])
def test_canceled_import_does_not_release_decoder_budget_early(tmp_path, monkeypatch, cancellations):
    monkeypatch.setattr(routes, "folder_paths", types.SimpleNamespace(get_input_directory=lambda: str(tmp_path)))
    monkeypatch.setattr(routes, "_CONTROL_IMPORT_LOCK", asyncio.Lock())
    started = threading.Event()
    finish = threading.Event()
    calls = []

    def imported(*args):
        calls.append(1)
        started.set()
        assert finish.wait(2)
        return {"input_type": "map"}

    monkeypatch.setattr(controls, "import_map", imported)

    class Request:
        async def json(self):
            return {"control": {"input_type": "map"}}

    async def requests():
        first = asyncio.create_task(routes.control_import_route(Request()))
        assert await asyncio.to_thread(started.wait, 2)
        first.cancel()
        if cancellations == 2:
            await asyncio.sleep(0.01)
            first.cancel()
        second = asyncio.create_task(routes.control_import_route(Request()))
        await asyncio.sleep(0.02)
        assert len(calls) == 1
        finish.set()
        with pytest.raises(asyncio.CancelledError):
            await first
        assert (await second).status == 200

    asyncio.run(requests())
    assert len(calls) == 2
