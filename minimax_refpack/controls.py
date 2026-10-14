"""Control-map configuration, preparation graphs and CPU frame decoding; no detector execution here."""

import hashlib
import json
import math
import os
from pathlib import Path
import uuid

from . import media

MODES = {
    "pose": ("Pose", "DWPreprocessor"),
    "depth": ("Depth", "DepthAnythingV2Preprocessor"),
    "edges": ("Edges", "Canny"),
}
FPS = 24
FRAME_POLICY = "24fps-target-ceil-17k+5-last-frame-hold-v2"


class ControlError(ValueError):
    """An invalid, missing or stale control map."""


def _filename(filename):
    if not isinstance(filename, str) or not filename.strip():
        raise ControlError("control file must be a non-empty relative input path")
    if ".." in filename or "\\" in filename or Path(filename).is_absolute():
        raise ControlError("control file must stay inside the ComfyUI input directory")
    return filename


def safe_path(input_dir, filename):
    """Resolve a managed input file, including symlinks, inside the input root."""
    _filename(filename)
    root = Path(input_dir).resolve()
    path = (root / filename).resolve()
    if not path.is_relative_to(root):
        raise ControlError("control file must stay inside the ComfyUI input directory")
    return str(path)


def file_sha256(path):
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise ControlError(f"control file cannot be read: {path}") from error
    return digest.hexdigest()


def _digest(value, field):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ControlError(f"{field} must be a SHA-256 digest")
    return value


def normalize_control(raw):
    """Keep only the persistent contract; transient frontend preparation is ignored."""
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ControlError("control must be an object")
    input_type = raw.get("input_type", "source")
    if input_type not in ("source", "map"):
        raise ControlError("control input_type must be source or map")
    filename = raw.get("file")
    # Validate the name even when no input directory is available yet.
    _filename(filename)
    mode = raw.get("mode")
    if mode not in MODES:
        raise ControlError("control mode must be pose, depth or edges")
    resolution = raw.get("resolution", 512)
    if type(resolution) is not int or not 64 <= resolution <= 4096 or resolution % 64:
        raise ControlError("control resolution must be a multiple of 64 between 64 and 4096")
    from .refs import validate_crop, validate_trim

    crop = validate_crop(raw["crop"]) if raw.get("crop") is not None else None
    trim = validate_trim(raw["trim"]) if raw.get("trim") is not None else None
    if any(not math.isfinite(v) for edit in (crop, trim) if edit for v in edit):
        raise ControlError("control crop and trim must contain finite numbers")
    ready = raw.get("map")
    if ready is not None:
        if not isinstance(ready, dict):
            raise ControlError("control map must be an object")
        _filename(ready.get("file"))
        if not isinstance(ready.get("recipe"), dict):
            raise ControlError("control map recipe must be an object")
        for field in ("source_sha256", "sha256"):
            _digest(ready.get(field), f"control map {field}")
        for field in ("frame_count", "width", "height"):
            if type(ready.get(field)) is not int or ready[field] < 1:
                raise ControlError(f"control map {field} must be a positive integer")
        if ready.get("fps") != FPS or ready["frame_count"] < 5 or (ready["frame_count"] - 5) % 17:
            raise ControlError("control map must have 24 fps frames on the 17k+5 grid")
        ready = {field: ready[field] for field in (
            "file", "source_sha256", "recipe", "sha256", "frame_count", "fps", "width", "height"
        )}
    result = {"file": filename, "mode": mode, "resolution": resolution,
              "crop": crop, "trim": trim, "map": ready}
    if "input_type" in raw:
        result["input_type"] = input_type
    return result


def duration(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ControlError("control length_seconds must be a finite non-negative number")
    if 0 < value < 5 / FPS:
        raise ControlError("control duration must allow at least 5 frames at 24 fps")
    return float(value)


def target_frames(length_seconds):
    seconds = duration(length_seconds)
    if not seconds:
        return None
    count = max(5, round(seconds * FPS))
    return 5 + ((count - 5 + 16) // 17) * 17


def recipe(control, source_sha256, length_seconds, processor):
    """Filename-independent identity, so portable staging may rename both files."""
    control = normalize_control(control)
    result = {
        "source_sha256": _digest(source_sha256, "control source_sha256"),
        "mode": control["mode"], "resolution": control["resolution"],
        "crop": control["crop"], "trim": control["trim"],
        "length_seconds": duration(length_seconds), "fps": FPS,
        "frame_policy": FRAME_POLICY, "processor": processor,
    }
    if control.get("input_type") == "map":
        result["input_type"] = "map"
        result["processor"] = None
    return result


def registry():
    try:
        import nodes
        return nodes.NODE_CLASS_MAPPINGS
    except (ImportError, AttributeError):
        return {}


def processor_settings(mode, resolution=512, node_registry=None):
    """Read defaults from registered schemas without instantiating a detector."""
    registered = registry() if node_registry is None else node_registry
    class_type = MODES[mode][1]
    node = registered.get(class_type)
    if node is None:
        package = "ComfyUI" if mode == "edges" else "comfyui_controlnet_aux"
        raise ControlError(f"{MODES[mode][0]} requires registered {class_type} from {package}")
    try:
        schema = node.INPUT_TYPES()
        if tuple(node.RETURN_TYPES)[0] != "IMAGE":
            raise ControlError(f"{class_type} must emit IMAGE at slot 0")
        required = schema.get("required", {})
        optional = schema.get("optional", {})
        if required.get("image", (None,))[0] != "IMAGE":
            raise ControlError(f"{class_type} must accept the IMAGE input image")
        settings = {}
        for name, spec in {**required, **optional}.items():
            if name == "image":
                continue
            kind, options = spec[0], spec[1] if len(spec) > 1 else {}
            if name == "resolution":
                value = resolution
            elif "default" in options:
                value = options["default"]
            elif isinstance(kind, (list, tuple)) and kind:
                value = kind[0]
            elif name in required:
                raise ControlError(f"{class_type} required input {name} has no default")
            else:
                continue
            if isinstance(kind, (list, tuple)) and value not in kind:
                raise ControlError(f"{class_type} has an invalid default for {name}")
            settings[name] = value
        return {"class_type": class_type, "inputs": settings}
    except ControlError:
        raise
    except Exception as error:
        raise ControlError(f"{class_type} schema is incompatible: {error}") from error


def capabilities(node_registry=None):
    modes = []
    for mode, (label, _class_type) in MODES.items():
        try:
            processor_settings(mode, node_registry=node_registry)
            available, reason = True, ""
        except ControlError as error:
            available, reason = False, str(error)
        modes.append({"id": mode, "label": label, "available": available, "reason": reason})
    return {"modes": modes}


def verify_source(control, prepared_recipe, input_dir, length_seconds):
    control = normalize_control(control)
    action = "open ControlNet and click Check again" if control.get("input_type") == "map" else "open ControlNet and click Regenerate"
    if not isinstance(prepared_recipe, dict):
        raise ControlError(f"control recipe is invalid; {action}")
    processor = prepared_recipe.get("processor")
    if control.get("input_type") == "map":
        if prepared_recipe.get("input_type") != "map" or processor is not None:
            raise ControlError(f"imported control recipe is invalid; {action}")
    elif (not isinstance(processor, dict) or processor.get("class_type") != MODES[control["mode"]][1]
          or not isinstance(processor.get("inputs"), dict)):
        raise ControlError(f"control recipe processor does not match its mode; {action}")
    source_sha = file_sha256(safe_path(input_dir, control["file"]))
    expected = recipe(control, source_sha, length_seconds, processor)
    if prepared_recipe != expected:
        raise ControlError(f"control map is stale: input, mode, crop, trim, resolution or duration changed; {action}")
    return source_sha


def prepare_frames(control, prepared_recipe, input_dir, cache=None):
    """Decode, crop, trim and resize a source or imported map to the recipe's frames, on CPU."""
    import torch
    import torch.nn.functional as functional

    seconds = prepared_recipe.get("length_seconds", 0)
    verify_source(control, prepared_recipe, input_dir, seconds)
    requested_count = target_frames(seconds)
    trim = control["trim"]
    if requested_count is not None:
        start = trim[0] if trim else 0
        decode_seconds = requested_count / FPS
        end = min(trim[1], start + decode_seconds) if trim else start + decode_seconds
        trim = [start, end]
    path = safe_path(input_dir, control["file"])
    imported = control.get("input_type") == "map"
    try:
        if imported:
            import av
            with av.open(path) as container:
                formats = container.format.name.split(",")
                if (not container.streams.video
                    or any(name.endswith("_pipe") or name in ("image2", "image2pipe") for name in formats)):
                    raise ControlError("imported control map must be a video with at least 5 frames")
        loader = media.load_video if cache is None else cache.video
        frames, _audio = loader(path, target_fps=FPS, crop=control["crop"], trim=trim)
    except Exception as error:
        if not imported or isinstance(error, ControlError):
            raise
        raise ControlError(f"cannot read imported control map video: {error}") from error
    available = int(frames.shape[0])
    count = requested_count if requested_count is not None else 5 + ((available - 5) // 17) * 17
    if count < 5:
        raise ControlError("control source must contain at least 5 frames at 24 fps")
    height, width = map(int, frames.shape[1:3])
    scale = control["resolution"] / min(width, height)
    # Only even dimensions for playback; a 64-pixel grid here would distort the source.
    out_w = max(2, round(width * scale / 2) * 2)
    out_h = max(2, round(height * scale / 2) * 2)
    if count * out_w * out_h * 3 * 4 > media._decode_budget():
        raise ControlError("prepared control frames exceed the reference decode budget; reduce resolution or duration")
    # One destination and one temporary frame, rather than a full resized batch copy.
    result = torch.empty((count, out_h, out_w, 3), dtype=torch.float32, device="cpu")
    for index in range(count):
        frame = frames[min(index, available - 1)].to(device="cpu", dtype=torch.float32).permute(2, 0, 1)[None]
        resized = functional.interpolate(frame, size=(out_h, out_w), mode="bilinear", align_corners=False)
        result[index].copy_(resized[0].permute(1, 2, 0))
    return result


def import_map(raw, input_dir, length_seconds=0):
    """Validate a standalone map on CPU and return its receipt without writing media."""
    if not isinstance(raw, dict) or raw.get("input_type") != "map":
        raise ControlError("choose input_type map to import an existing control map")
    control = normalize_control({**raw, "map": None})
    source_sha = file_sha256(safe_path(input_dir, control["file"]))
    prepared_recipe = recipe(control, source_sha, length_seconds, None)
    frames = prepare_frames(control, prepared_recipe, input_dir)
    verify_source(control, prepared_recipe, input_dir, length_seconds)
    count, height, width, _channels = map(int, frames.shape)
    control["map"] = {
        "file": control["file"], "source_sha256": source_sha,
        "recipe": prepared_recipe, "sha256": source_sha,
        "frame_count": count, "fps": FPS, "width": width, "height": height,
    }
    return control


def load_ready(control, input_dir, length_seconds, cache):
    control = normalize_control(control)
    ready = control["map"]
    if ready is None:
        if control.get("input_type") == "map":
            raise ControlError("control map has not been checked; open ControlNet and click Check map before rendering")
        raise ControlError("control map has not been generated; open ControlNet and click Generate before rendering")
    imported = control.get("input_type") == "map"
    action = "open ControlNet and click Check again" if imported else "open ControlNet and click Regenerate"
    if imported and ready["file"] != control["file"]:
        raise ControlError("imported control and its receipt must name the same file")
    source_sha = verify_source(control, ready["recipe"], input_dir, length_seconds)
    if ready["source_sha256"] != source_sha:
        raise ControlError(f"control map input digest is stale; {action}")
    expected_count = target_frames(length_seconds)
    if expected_count is not None and ready["frame_count"] != expected_count:
        raise ControlError(f"control map frame count does not match the requested duration; {action}")
    path = safe_path(input_dir, ready["file"])
    if file_sha256(path) != ready["sha256"]:
        raise ControlError(f"control map file changed; {action}")
    if imported:
        frames = prepare_frames(control, ready["recipe"], input_dir, cache=cache)
        verify_source(control, ready["recipe"], input_dir, length_seconds)
    else:
        frames, _audio = cache.video(path, target_fps=FPS)
    if tuple(frames.shape) != (ready["frame_count"], ready["height"], ready["width"], 3):
        raise ControlError("control map dimensions or frame count do not match its receipt")
    return frames


def file_signature(control, input_dir):
    """Cheap overwrite detection for ComfyUI and prepared-pack cache identities."""
    if control is None:
        return ""
    files = [control["file"]]
    if control.get("map"):
        files.append(control["map"]["file"])
    stamps = []
    for filename in files:
        path = safe_path(input_dir, filename)
        try:
            stat = os.stat(path)
            stamps.append((filename, stat.st_mtime_ns, stat.st_size))
        except OSError:
            stamps.append((filename, "missing"))
    return json.dumps(stamps, separators=(",", ":"))


def preparation_graph(control, input_dir, length_seconds=0, node_registry=None):
    control = normalize_control(control)
    if control is None:
        raise ControlError("choose a control video first")
    if control.get("input_type") == "map":
        raise ControlError("an existing control map is imported, not prepared; click Check map on the ControlNet tab Use a map I already have")
    processor = processor_settings(control["mode"], control["resolution"], node_registry)
    source_sha = file_sha256(safe_path(input_dir, control["file"]))
    prepared_recipe = recipe(control, source_sha, length_seconds, processor)
    source_config = {**control, "map": None}
    control_json = json.dumps(source_config, separators=(",", ":"))
    recipe_json = json.dumps(prepared_recipe, separators=(",", ":"), sort_keys=True)
    prompt = {
        "1": {"class_type": "MiniMaxH3ControlSource", "inputs": {
            "control_json": control_json, "recipe_json": recipe_json,
        }},
        "2": {"class_type": processor["class_type"], "inputs": {
            "image": ["1", 0], **processor["inputs"],
        }},
        "3": {"class_type": "MiniMaxH3ControlMapSave", "inputs": {
            "images": ["2", 0], "control_json": control_json, "recipe_json": recipe_json,
        }},
    }
    title = f"MiniMax Control Map - {MODES[control['mode']][0]}"
    workflow = {
        "id": str(uuid.uuid5(uuid.NAMESPACE_URL, "minimax-control:" + recipe_json)),
        "version": 0.4, "last_node_id": 3, "last_link_id": 2,
        "nodes": [
            _workflow_node(1, prompt["1"], [0, 0], [control_json, recipe_json], [], [1]),
            _workflow_node(2, prompt["2"], [400, 0], list(processor["inputs"].values()), [1], [2]),
            _workflow_node(3, prompt["3"], [800, 0], [control_json, recipe_json], [2], []),
        ],
        "links": [[1, 1, 0, 2, 0, "IMAGE"], [2, 2, 0, 3, 0, "IMAGE"]],
        "groups": [], "config": {}, "extra": {"workflow_name": title}, "workflow_name": title,
    }
    return {"prompt": prompt, "workflow": workflow, "output_node_id": "3", "recipe": prepared_recipe}


def _workflow_node(node_id, spec, position, widgets, input_links, output_links):
    return {
        "id": node_id, "type": spec["class_type"], "pos": position, "size": [320, 200],
        "flags": {}, "order": node_id - 1, "mode": 0,
        "inputs": [{"name": "images" if node_id == 3 else "image", "type": "IMAGE", "link": link}
                   for link in input_links],
        "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": output_links}] if output_links else [],
        "properties": {"Node name for S&R": spec["class_type"]}, "widgets_values": widgets,
    }
