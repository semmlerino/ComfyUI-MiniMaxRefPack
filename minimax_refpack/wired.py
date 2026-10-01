"""Resolve optional upstream media without writing tensors to the input directory."""
from dataclasses import replace
import os

from . import media, refs

INPUT_TYPES = {
    **{f"image_{i}": "IMAGE" for i in range(1, 10)},
    **{f"video_{i}": "IMAGE" for i in range(1, 4)},
    **{f"video_audio_{i}": "AUDIO" for i in range(1, 4)},
    **{f"audio_{i}": "AUDIO" for i in range(1, 4)},
}


def active(inputs):
    return {name: value for name, value in inputs.items()
            if name in INPUT_TYPES and value is not None}


class WiredCache(media.MediaCache):
    """Serve upstream values to both the output sockets and the prompt writer."""

    def __init__(self):
        super().__init__()
        self.overrides = {}

    def image(self, path, *, crop=None, max_edge=0):
        if path in self.overrides:
            return self.overrides[path]
        return super().image(path, crop=crop, max_edge=max_edge)

    def video(self, path, *, target_fps=24, crop=None, trim=None):
        if path in self.overrides:
            return self.overrides[path]
        return super().video(path, target_fps=target_fps, crop=crop, trim=trim)

    def audio(self, path, *, trim=None):
        if path in self.overrides:
            return self.overrides[path]
        return super().audio(path, trim=trim)


def resolve(reference_set, inputs, input_dir):
    """Replace occupied slots or append consecutive slots; never silently renumber gaps.

    Roles and primary-video status belong to the slot. Crop/trim belong to the old
    file, so upstream values are already prepared and bypass those edits and the cap.
    Video IMAGE inputs are frames at 24 fps, matching the pack's existing outputs.
    """
    inputs = active(inputs)
    cache = WiredCache()
    for kind in refs.KINDS:
        group = reference_set.of_kind(kind)
        for slot in range(1, refs.MAX_BY_KIND[kind] + 1):
            name = f"{kind}_{slot}"
            value = inputs.get(name)
            soundtrack = inputs.get(f"video_audio_{slot}") if kind == "video" else None
            if value is None and soundtrack is None:
                continue
            if slot > len(group) + 1:
                raise ValueError(f"{name} needs the preceding {kind} slots filled; MiniMax compacts empty slots")
            old = group[slot - 1] if slot <= len(group) else None
            if kind == "video" and value is None:
                if old is None:
                    raise ValueError(f"video_audio_{slot} needs video_{slot} or an internal video")
                value, _ = cache.video(os.path.join(input_dir, old.file), crop=old.crop, trim=old.trim)
            if kind in ("image", "video"):
                shape = getattr(value, "shape", ())
                if len(shape) != 4 or shape[-1] != 3 or shape[0] < 1:
                    raise ValueError(f"{name} expects an IMAGE tensor [frames, height, width, 3]")
                if kind == "image" and shape[0] != 1:
                    raise ValueError(f"{name} accepts one picture; select one image from the batch upstream")
                if kind == "video" and (shape[0] < 5 or (shape[0] - 5) % 17):
                    raise ValueError(f"{name} expects 24 fps frames on MiniMax's 17k+5 frame grid")
            else:
                _validate_audio(name, value)
            if soundtrack is not None:
                _validate_audio(f"video_audio_{slot}", soundtrack)
            # .wired forces standalone audio through the PCM encoder, never raw-file inline.
            file = f"__upstream__/{name}.wired"
            ref = (replace(old, file=file, crop=None, trim=None) if old else refs.Reference(kind, file))
            if kind == "video":
                ref.use_soundtrack = soundtrack is not None
                value = (value, soundtrack)
            cache.overrides[os.path.join(input_dir, file)] = value
            if old is None:
                reference_set.references.append(ref)
                group.append(ref)
            else:
                index = next(i for i, item in enumerate(reference_set.references) if item is old)
                reference_set.references[index] = ref
                group[slot - 1] = ref
    return reference_set, cache


def _validate_audio(name, value):
    if not isinstance(value, dict) or "waveform" not in value or not value.get("sample_rate"):
        raise ValueError(f"{name} expects AUDIO with waveform and sample_rate")
