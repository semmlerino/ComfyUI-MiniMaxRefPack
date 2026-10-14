"""Explicit CPU source preparation and streamed map persistence for queued graphs."""

import json
import hashlib
import os
from pathlib import Path
import tempfile

from . import controls, media


def _config(control_json, recipe_json):
    try:
        control = controls.normalize_control(json.loads(control_json))
        prepared_recipe = json.loads(recipe_json)
    except (ValueError, TypeError) as error:
        raise controls.ControlError(f"invalid control preparation configuration: {error}") from error
    if control is None or not isinstance(prepared_recipe, dict):
        raise controls.ControlError("invalid control preparation configuration")
    return control, prepared_recipe


def _input_dir():
    import folder_paths
    return folder_paths.get_input_directory()


def _prepare_frames(control, prepared_recipe, input_dir, cache=None):
    import torch
    import torch.nn.functional as functional

    seconds = prepared_recipe.get("length_seconds", 0)
    controls.verify_source(control, prepared_recipe, input_dir, seconds)
    requested_count = controls.target_frames(seconds)
    trim = control["trim"]
    if seconds:
        start = trim[0] if trim else 0
        decode_seconds = requested_count / controls.FPS
        end = min(trim[1], start + decode_seconds) if trim else start + decode_seconds
        trim = [start, end]
    path = controls.safe_path(input_dir, control["file"])
    imported = control.get("input_type") == "map"
    try:
        if imported:
            import av
            with av.open(path) as container:
                formats = container.format.name.split(",")
                if (not container.streams.video
                    or any(name.endswith("_pipe") or name in ("image2", "image2pipe") for name in formats)):
                    raise controls.ControlError("imported control map must be a video with at least 5 frames")
        loader = media.load_video if cache is None else cache.video
        frames, _audio = loader(path, target_fps=controls.FPS, crop=control["crop"], trim=trim)
    except Exception as error:
        if not imported or isinstance(error, controls.ControlError):
            raise
        raise controls.ControlError(f"cannot read imported control map video: {error}") from error
    available = int(frames.shape[0])
    count = requested_count if requested_count is not None else 5 + ((available - 5) // 17) * 17
    if count < 5:
        raise controls.ControlError("control source must contain at least 5 frames at 24 fps")
    height, width = map(int, frames.shape[1:3])
    scale = control["resolution"] / min(width, height)
    # Only even dimensions for playback; a 64-pixel grid here would distort the source.
    out_w = max(2, round(width * scale / 2) * 2)
    out_h = max(2, round(height * scale / 2) * 2)
    if count * out_w * out_h * 3 * 4 > media._decode_budget():
        raise controls.ControlError("prepared control frames exceed the reference decode budget; reduce resolution or duration")
    # One destination and one temporary frame, rather than a full resized batch copy.
    result = torch.empty((count, out_h, out_w, 3), dtype=torch.float32, device="cpu")
    for index in range(count):
        frame = frames[min(index, available - 1)].to(device="cpu", dtype=torch.float32).permute(2, 0, 1)[None]
        resized = functional.interpolate(frame, size=(out_h, out_w), mode="bilinear", align_corners=False)
        result[index].copy_(resized[0].permute(1, 2, 0))
    return result


class MiniMaxH3ControlSource:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "control_json": ("STRING", {"multiline": True, "dynamicPrompts": False}),
            "recipe_json": ("STRING", {"multiline": True, "dynamicPrompts": False}),
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "load"
    CATEGORY = "MiniMax H3/Control preparation"

    @classmethod
    def IS_CHANGED(cls, control_json, recipe_json):
        control, _recipe = _config(control_json, recipe_json)
        return controls.file_sha256(controls.safe_path(_input_dir(), control["file"]))

    def load(self, control_json, recipe_json):
        control, prepared_recipe = _config(control_json, recipe_json)
        return (_prepare_frames(control, prepared_recipe, _input_dir()),)


def write_map(images, control, prepared_recipe, input_dir):
    """Write browser-playable H.264 one frame at a time, then publish atomically."""
    import av
    import numpy as np

    shape = tuple(getattr(images, "shape", ()))
    if len(shape) != 4 or shape[-1] != 3 or shape[0] < 5 or (shape[0] - 5) % 17:
        raise controls.ControlError("control map expects IMAGE frames [17k+5, height, width, 3]")
    count, height, width, _channels = shape
    expected_count = controls.target_frames(prepared_recipe.get("length_seconds", 0))
    if expected_count is not None and count != expected_count:
        raise controls.ControlError("control map frame count does not match the preparation duration")
    if height < 2 or width < 2 or height % 2 or width % 2:
        raise controls.ControlError("control map width and height must be even for H.264 playback")
    source_sha = controls.verify_source(control, prepared_recipe, input_dir, prepared_recipe.get("length_seconds", 0))
    digest = hashlib.sha256(json.dumps(prepared_recipe, sort_keys=True).encode()).hexdigest()
    directory = Path(controls.safe_path(input_dir, "control_maps/.map")).parent
    directory.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".control-", suffix=".mp4", dir=directory)
    os.close(descriptor)
    try:
        codec = av.codec.Codec("libx264", "w")
        if "yuv420p" not in {pixel.name for pixel in codec.video_formats}:
            raise controls.ControlError("libx264 does not support the required yuv420p pixel format")
        with av.open(temporary, "w", format="mp4", options={"movflags": "+faststart"}) as container:
            stream = container.add_stream("libx264", rate=controls.FPS, options={"crf": "10", "preset": "fast"})
            stream.width, stream.height = width, height
            stream.pix_fmt = "yuv420p"
            for index in range(count):
                frame = images[index]
                detach = getattr(frame, "detach", None)
                if detach is not None:
                    import torch
                    frame = detach().to(device="cpu", dtype=torch.float32).numpy()
                pixels = np.asarray(frame)
                if not np.isfinite(pixels).all() or pixels.min() < 0 or pixels.max() > 1:
                    raise controls.ControlError("control map pixels must be finite and between 0 and 1")
                rgb = np.rint(pixels * 255).astype(np.uint8)
                video_frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
                video_frame.pts = index
                for packet in stream.encode(video_frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        # A source replaced during detector execution must not receive a ready receipt.
        controls.verify_source(control, prepared_recipe, input_dir, prepared_recipe.get("length_seconds", 0))
        map_sha = controls.file_sha256(temporary)
        # Include encoded bytes in the final name: regenerating a recipe must not
        # overwrite another workflow's reviewed map if detector output changes.
        relative = f"control_maps/{control['mode']}-{digest[:16]}-{map_sha}.mp4"
        destination = controls.safe_path(input_dir, relative)
        os.replace(temporary, destination)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return {"file": relative, "source_sha256": source_sha, "recipe": prepared_recipe,
            "sha256": map_sha, "frame_count": count, "fps": controls.FPS,
            "width": width, "height": height}


class MiniMaxH3ControlMapSave:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "control_json": ("STRING", {"multiline": True, "dynamicPrompts": False}),
            "recipe_json": ("STRING", {"multiline": True, "dynamicPrompts": False}),
        }}

    RETURN_TYPES = ()
    FUNCTION = "save"
    CATEGORY = "MiniMax H3/Control preparation"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Saving is an explicit preparation action; re-queue repairs a removed map too.
        return float("nan")

    def save(self, images, control_json, recipe_json):
        control, prepared_recipe = _config(control_json, recipe_json)
        ready = write_map(images, control, prepared_recipe, _input_dir())
        return {"ui": {"control_map": [ready]}, "result": ()}


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3ControlSource": MiniMaxH3ControlSource,
    "MiniMaxH3ControlMapSave": MiniMaxH3ControlMapSave,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3ControlSource": "MiniMax Control Source",
    "MiniMaxH3ControlMapSave": "MiniMax Save Control Map",
}
