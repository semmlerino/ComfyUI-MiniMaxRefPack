"""File -> ComfyUI tensor/audio loading, 24fps resample, the VLM's copy, probe metadata.

FROZEN CONTRACT: nodes.py, routes.py and prompt.py all import these names directly -
`resample_indices`, `load_image`, `load_audio`, `load_video`, `MediaCache`,
`encode_reference_mp4`, `probe`, `thumbnail_png`.

A reference is decoded ONCE per build: `MediaCache` memoises the three loaders and both
the node's sockets and the prompt payload draw from it, and `encode_reference_mp4` builds
the VLM's copy out of those already-decoded frames rather than reading the file again.

Only `resample_indices` is pure/torch-free by requirement (it must
be exhaustively unit-testable); everything else lazy-imports torch/av/PIL/comfy so a
plain `pytest tests/test_media.py` runs without ComfyUI installed (folder_paths,
comfy_api, comfy_extras are not importable outside a ComfyUI process - verified: a bare
`import folder_paths` in this venv raises ModuleNotFoundError).

The two `_video_from_file_cls` / `_audio_load_fn` indirections below exist so tests can
monkeypatch the ComfyUI-side decoder without comfy_api/comfy_extras being installed at
all - they patch the accessor, not the (unimportable) real module.
"""

from __future__ import annotations

import math
import mimetypes
from typing import Any

from . import logs


def resample_indices(n_src: int, src_fps: float, target_fps: int = 24) -> list[int]:
    """Which source frame each output frame reads from, resampling to `target_fps`.

    Preserves the clip's real-world duration: output frame count is
    round(n_src * target_fps / src_fps), not n_src. Output frame i reads source frame
    round(i * src_fps / target_fps), clamped to [0, n_src-1]. Core never touches
    framerate (CU/comfy_extras/nodes_minimax_h3.py:246-252 only resizes/trims by frame
    count) - this resample is ours alone.
    """
    if n_src <= 0:
        return []
    if src_fps <= 0:
        raise ValueError(f"source fps must be positive, got {src_fps!r}")
    n_out = max(1, round(n_src * target_fps / src_fps))
    return [min(max(round(i * src_fps / target_fps), 0), n_src - 1) for i in range(n_out)]


def _crop_box(crop, width: int, height: int) -> tuple[int, int, int, int]:
    """Fraction rect [x, y, w, h] -> integer (left, top, right, bottom) pixel box.

    THE one fraction->pixel rule - the loaders and the thumbnails all come through
    here, so a tile's thumb and the emitted tensor can never disagree. Each edge is
    floor(fraction * dimension + 0.5): half-up, not Python's round(), whose
    round-half-to-even would move an edge by a pixel depending on parity. Then clamp
    to the frame so a rect that rounds past an edge still keeps at least one pixel
    on each axis - a zero-width crop can never come out of here.
    """
    x, y, w, h = crop
    left = min(max(int(math.floor(x * width + 0.5)), 0), width - 1)
    top = min(max(int(math.floor(y * height + 0.5)), 0), height - 1)
    right = min(max(int(math.floor((x + w) * width + 0.5)), left + 1), width)
    bottom = min(max(int(math.floor((y + h) * height + 0.5)), top + 1), height)
    return left, top, right, bottom


def _slice_audio(audio: dict, trim) -> dict:
    """Slice a {"waveform","sample_rate"} dict to [start, end) seconds. Returns a NEW
    dict - callers may still hold the unsliced original."""
    sr = audio["sample_rate"]
    start, end = trim
    return {"waveform": audio["waveform"][..., int(start * sr):int(end * sr)], "sample_rate": sr}


def _guess_kind(path: str) -> str:
    mime, _ = mimetypes.guess_type(path)
    if mime and mime.startswith("video/"):
        return "video"
    if mime and mime.startswith("audio/"):
        return "audio"
    return "image"


def _video_from_file_cls():
    """Indirection point so tests can substitute a fake without comfy_api installed."""
    from comfy_api.latest._input_impl.video_types import VideoFromFile

    return VideoFromFile


def _audio_load_fn():
    """Indirection point so tests can substitute a fake without comfy_extras installed."""
    from comfy_extras.nodes_audio import load as _load

    return _load


def load_image(path: str, crop=None, max_edge: int = 0):
    """[1, H, W, 3] float32 in 0..1. Plain PIL decode - references are single stills
    (not animated), so we skip the ImageSequence handling CU/nodes.py:1734 LoadImage
    needs for animated webp.

    `crop` is a [x, y, w, h] fraction rect (refs.Reference.crop), applied AFTER the
    EXIF transpose so the fractions refer to the image as the editor showed it.

    `max_edge` caps the LONG edge (0 = off), applied AFTER the crop so it measures what
    is actually emitted. Core sizes reference images off the SHORT edge - `scale =
    min(1.0, REF_IMAGE_SHORT_EDGE / min(w, h))` at CU/comfy_extras/nodes_minimax_h3.py:301
    with REF_IMAGE_SHORT_EDGE = 2048 (:29) - so at ref_image_size="max" a wide sheet
    reaches the VAE enormous: 5000x2550 lands at 4000x2048, ~32k reference tokens that
    ride every sampling step. thumbnail() shrinks in place and never enlarges, so a
    reference already under the cap is emitted untouched.
    """
    import os

    import numpy as np
    import torch
    from PIL import Image, ImageOps

    with logs.timed("load_image", file=os.path.basename(path), crop=crop,
                    cap=max_edge or None) as fields:
        img = Image.open(path)
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")
        fields["src"] = f"{img.width}x{img.height}"
        if crop is not None:
            img = img.crop(_crop_box(crop, img.width, img.height))
        if max_edge:
            img.thumbnail((max_edge, max_edge), Image.LANCZOS)
        fields["out"] = f"{img.width}x{img.height}"
        arr = np.array(img).astype(np.float32) / 255.0
        return torch.from_numpy(arr)[None, ...]


def load_audio(path: str, trim=None) -> dict:
    """{"waveform": [1, C, L], "sample_rate": int} - CU/comfy_extras/nodes_audio.py:333
    returns (waveform, sample_rate); the AUDIO socket shape wraps it with a batch dim
    (:380-381). `trim` = [start, end] seconds, sliced exactly the way a video's
    soundtrack is (_slice_audio)."""
    import os

    with logs.timed("load_audio", file=os.path.basename(path), trim=trim) as fields:
        waveform, sample_rate = _audio_load_fn()(path)
        audio = {"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate}
        if trim is not None:
            audio = _slice_audio(audio, trim)
        fields["sample_rate"] = sample_rate
        return audio


def load_video(path: str, target_fps: int = 24, crop=None, trim=None):
    """(frames [N,H,W,3] resampled to target_fps, audio dict or None).

    CU/comfy_api/latest/_input_impl/video_types.py:118 VideoFromFile.get_components().
    `.audio` is already built as {"waveform","sample_rate"} (video_types.py:445-448),
    the same shape core AUDIO sockets use, so it's passed through unchanged.

    `trim` = [start, end] seconds. Cut on SOURCE frames first - a source frame is kept
    when its timestamp i/src_fps lands in [start, end) - then the usual resample runs
    on that span, so the output preserves the span's real duration. The soundtrack is
    sliced to the SAME window or it would drift out of sync with its frames.

    `crop` = [x, y, w, h] fractions, applied to the resampled frames (_crop_box).

    MiniMax needs >=5 frames per reference video (CU/comfy_extras/nodes_minimax_h3.py:250)
    - the check fires on the RESULT: a trim too short to survive it raises, naming the
    file and the requested window, never silently clamping or padding.
    """
    import os

    with logs.timed("load_video", file=os.path.basename(path), crop=crop, trim=trim) as fields:
        return _decode_video(path, target_fps, crop, trim, fields)


def _decode_video(path, target_fps, crop, trim, fields):
    """The body of load_video. Split out only so the timing/logging wrapper above stays
    a plain `with` block instead of wrapping 30 lines."""
    video_cls = _video_from_file_cls()
    if trim is None:
        source = video_cls(path)
    else:
        start, end = trim
        # VideoFromFile seeks to the keyframe before start_time and stops decoding at
        # duration. Passing the window here avoids materialising the whole source clip
        # only to throw most of its frames away below.
        source = video_cls(path, start_time=start, duration=end - start)
    components = source.get_components()
    frames = components.images
    n_src = frames.shape[0]
    src_fps = float(components.frame_rate)
    audio = components.audio
    fields["src_frames"] = n_src
    fields["fps"] = src_fps

    indices = resample_indices(n_src, src_fps, target_fps)
    if len(indices) < 5:
        duration = (n_src / src_fps) if src_fps else 0.0
        window = f" trimmed to {trim[0]:.2f}-{trim[1]:.2f}s" if trim is not None else ""
        raise ValueError(
            f"reference video {path!r}{window} has only {len(indices)} frame(s) at {target_fps}fps "
            f"(source: {n_src} frames, {duration:.2f}s) - MiniMax H3 needs at least 5"
        )
    out = frames[indices]
    if crop is not None:
        left, top, right, bottom = _crop_box(crop, out.shape[2], out.shape[1])
        out = out[:, top:bottom, left:right, :]
    fields["frames"] = len(indices)
    fields["audio"] = audio is not None
    return out, audio


# ---- one preparation per reference, per build ------------------------------------


def _cache_key(value):
    """A crop/trim list -> something hashable. `None` is the common path and stays None;
    `tuple(None)` raises TypeError, which is the whole reason this exists."""
    return None if value is None else tuple(value)


def _abspath(path: str) -> str:
    import os

    return os.path.abspath(path)


class MediaCache:
    """Decode each reference once, then hand the same result to every consumer.

    The node's sockets and the VLM payload need the same pixels, and before this existed
    they each fetched their own - an edited clip with a soundtrack was decoded twice
    through `load_video` (the second call reached only for `audio` and discarded its
    frames) on top of a third, independent decode inside the VLM transcoder.

    PER BUILD, NEVER MODULE-LEVEL. The browser re-uploads an edited reference under the
    same filename with `overwrite=true` - which is exactly why `nodes.IS_CHANGED` hashes
    mtime+size rather than trusting the name - so a cache that outlived one build would
    serve pixels from a file that no longer exists.

    Keyed on field VALUES, not object identity: `assign_tags()` runs three times per
    build and returns fresh `TaggedReference` objects each time, so two lookups for the
    same reference never see the same object.
    """

    def __init__(self):
        self._entries: dict[tuple[Any, ...], Any] = {}

    def image(self, path: str, *, crop=None, max_edge: int = 0):
        key = ("image", _abspath(path), _cache_key(crop), int(max_edge))
        return self._get(
            key, path, lambda: load_image(path, crop=crop, max_edge=max_edge)
        )

    def video(self, path: str, *, target_fps: int = 24, crop=None, trim=None):
        key = (
            "video",
            _abspath(path),
            int(target_fps),
            _cache_key(crop),
            _cache_key(trim),
        )
        return self._get(
            key,
            path,
            lambda: load_video(path, target_fps=target_fps, crop=crop, trim=trim),
        )

    def audio(self, path: str, *, trim=None) -> dict:
        key = ("audio", _abspath(path), _cache_key(trim))
        return self._get(key, path, lambda: load_audio(path, trim=trim))

    def _get(self, key, path: str, load):
        """A line of its own rather than a `cached=` field on the loader's `logs.timed`
        span: a hit never enters `load_image`/`load_video`/`load_audio`, so there is no
        span to carry the field, and emitting one anyway would contradict the
        one-line-per-real-decode reading this change is measured by."""
        import os

        hit = key in self._entries
        logs.log("media_cache", kind=key[0], file=os.path.basename(path), hit=hit)
        if not hit:
            self._entries[key] = load()
        return self._entries[key]


# ---- the VLM's copy of a reference ------------------------------------------------

# The video twins of prompt.VLM_IMAGE_LONG_EDGE, and measured the same way. A live
# 10.12s 1080p clip billed 660 video tokens on google/gemini-3-flash-preview - about
# 1fps at the provider's low media resolution - so every pixel and every frame above
# these numbers is decoded, paid for in upload time, and discarded on the far side.
VLM_VIDEO_LONG_EDGE = 768
VLM_VIDEO_FPS = 8

# Frames are converted a chunk at a time rather than in one batched op. The prepared
# 24fps stack for one 10s 1080p reference is ~6GB of float32 and build() holds it live
# until the node returns, so a batched mul() over an 80-frame selection would put ~2GB
# more beside it. 16 1080p frames is ~35MB.
_ENCODE_CHUNK = 16

# resample_indices(5, 24, 8) is exactly 2, so a clip that only just cleared load_video's
# >=5-frame minimum would otherwise reach the VLM as a 2-frame video. Below this many
# selected frames every frame is kept and the clip goes out at its own rate.
_MIN_VLM_FRAMES = 5

# h264 first - it is what the providers take, and the PyPI PyAV wheel bundles an FFmpeg
# built with it (verified: av 18.1.0 ships ffmpeg 8.1.2 with libx264 usable). A PyAV
# linked against an FFmpeg WITHOUT it is rare but real, and it matters more than it used
# to: an untouched mp4 used to skip encoding altogether, so a bare add_stream failure
# would take away a path that worked before. The old transcoder already needed libx264
# for every cropped, trimmed or odd-container reference, so this widens an existing
# dependency rather than inventing one - but widening it silently is not acceptable.
# mpeg4 is core FFmpeg and present in every build, so the list ends somewhere real.
_VIDEO_ENCODERS = ("libx264", "libopenh264", "mpeg4")

_AUDIO_CHUNK = 4096
# av.AudioLayout takes a name, not a channel count (verified, PyAV 18.1.0: passing an
# int raises TypeError). Anything not in here loses its soundtrack rather than being
# silently downmixed, and the manifest says so.
_LAYOUTS = {1: "mono", 2: "stereo"}


class _AudioMuxFailed(Exception):
    """AAC encoding failed after the mp4 header had already been written.

    A stream cannot be withdrawn once the header is out - adding one after the first
    `mux()` fails with `ValueError: Cannot rebase to zero time` - so the only honest
    recovery is to throw the container away and build a video-only one from scratch.
    """


def encode_reference_mp4(
    frames,
    src_fps: float = 24,
    audio=None,
    *,
    long_edge: int = VLM_VIDEO_LONG_EDGE,
    target_fps: int = VLM_VIDEO_FPS,
) -> tuple[bytes, bool]:
    """(mp4 bytes, whether the soundtrack is really inside it).

    `frames` is the [N,H,W,3] 0..1 stack `load_video` already produced for the node's
    sockets, so the VLM's copy costs an encode and not a second decode. Frame selection
    goes through `resample_indices`, the same duration-preserving rule the loader uses.

    The `muxed` half of the return is not decoration: an `<Audio N>` tag has already
    been minted by `assign_tags()` by the time this runs, so a soundtrack that could not
    be encoded has to be reported, not assumed - `_build_content` turns a False here
    into the manifest's "NOT sent" note, which both system prompts already handle.
    """
    n_src = len(frames)
    if n_src == 0:
        raise ValueError("cannot encode a reference video with no frames")

    indices = resample_indices(n_src, src_fps, target_fps)
    fps: float = target_fps
    if len(indices) < _MIN_VLM_FRAMES:
        indices = list(range(n_src))
        fps = src_fps

    height, width = int(frames[0].shape[0]), int(frames[0].shape[1])
    scale = min(1.0, long_edge / max(width, height))
    # h264 needs even dimensions. `dim & ~1` rather than `dim - dim % 2` because
    # _crop_box guarantees only that a box is never zero-width: a legal 1px crop would
    # build a 0-sized encoder through the subtraction and avcodec_open2 fails EINVAL.
    out_w = max(2, int(width * scale) & ~1)
    out_h = max(2, int(height * scale) & ~1)

    with logs.timed(
        "video_bytes", w=out_w, h=out_h, fps=fps, frames=len(indices)
    ) as fields:
        try:
            data, muxed = _encode_mp4(frames, indices, fps, out_w, out_h, audio)
        except _AudioMuxFailed as e:
            logs.warn("video_audio_dropped", stage="encode", error=str(e))
            data, muxed = _encode_mp4(frames, indices, fps, out_w, out_h, None)
        fields["bytes"] = len(data)
        fields["audio"] = muxed
        return data, muxed


def _encode_mp4(frames, indices, fps, out_w, out_h, audio) -> tuple[bytes, bool]:
    """One pass: build the container, declare every stream, then write."""
    import io
    from fractions import Fraction

    import av
    import numpy as np

    buf = io.BytesIO()
    container = av.open(buf, "w", format="mp4")
    try:
        video = _add_video_stream(
            container, Fraction(fps).limit_denominator(1000), out_w, out_h
        )

        # EVERY output stream must exist before the first mux(). See _AudioMuxFailed.
        sound = None if audio is None else _open_audio_stream(container, audio)

        for start in range(0, len(indices), _ENCODE_CHUNK):
            block = _chunk_to_uint8(frames[indices[start : start + _ENCODE_CHUNK]])
            for i in range(block.shape[0]):
                # Downscale with swscale, AFTER the uint8 conversion: measured 3.97s for
                # 80 1080p frames against 6.35s for a resize-first OpenCV variant on the
                # same real data, and PyAV is already a hard dependency where OpenCV is
                # not a ComfyUI core one.
                frame = av.VideoFrame.from_ndarray(
                    np.ascontiguousarray(block[i]), format="rgb24"
                ).reformat(width=out_w, height=out_h, format="yuv420p")
                for packet in video.encode(frame):
                    container.mux(packet)
        for packet in video.encode():
            container.mux(packet)

        if sound is not None:
            try:
                _encode_audio(container, sound, audio)
            except Exception as e:
                raise _AudioMuxFailed(f"{type(e).__name__}: {e}") from e
    except BaseException:
        # This container is being discarded, so closing it is best effort: a muxer that
        # cannot write its trailer after a half-encoded audio stream must not replace the
        # _AudioMuxFailed the caller is waiting to catch. Swallowing it there would turn a
        # recoverable "encode the video without sound" into a failed build.
        try:
            container.close()
        except Exception:
            pass
        raise
    container.close()

    return buf.getvalue(), sound is not None


def _add_video_stream(container, rate, out_w, out_h):
    """The video stream, on the first encoder this FFmpeg build actually has.

    Raises rather than returning None: unlike the soundtrack, there is no meaningful
    reference left without a video stream, and a named failure beats an add_stream
    traceback that says only "Unknown encoder".
    """
    tried: list[str] = []
    for name in _VIDEO_ENCODERS:
        try:
            stream = container.add_stream(name, rate=rate)
        except Exception as e:
            tried.append(f"{name} ({type(e).__name__})")
            continue
        stream.width = out_w
        stream.height = out_h
        stream.pix_fmt = "yuv420p"
        if tried:
            logs.warn("video_encoder_fallback", encoder=name, unavailable=tried)
        return stream
    raise RuntimeError(
        "this PyAV build has no usable video encoder for the VLM's copy of a reference "
        f"(tried {', '.join(tried)}). PyAV's own wheels bundle an FFmpeg with libx264: "
        "`pip install --force-reinstall av` in the ComfyUI environment."
    )


def _chunk_to_uint8(chunk):
    """A [n,H,W,3] float chunk in 0..1 -> uint8 ndarray.

    Duck-typed rather than branched on an import: production always hands this torch
    tensors, and the test stack has numpy and no torch at all. Verified byte-identical
    between the two branches on in-range and out-of-range input (max-diff 0) - torch
    clamps after scaling, numpy clips before it, and both land on the same 0..255.
    """
    import numpy as np

    if getattr(chunk, "mul", None) is None:
        return (np.clip(np.asarray(chunk, dtype=np.float32), 0.0, 1.0) * 255.0).astype(
            np.uint8
        )

    import torch

    return chunk.mul(255).clamp_(0, 255).to(torch.uint8).cpu().numpy()


def _planar_waveform(waveform):
    """A [1,C,L] / [C,L] / [L] float waveform -> contiguous float32 [C,L] in -1..1.

    The batch dimension is the trap: the cached waveform is batch-first and PyAV's
    AudioFrame is not, so passing it through unchanged produces a stream that is present
    in the container and carries nothing.
    """
    import numpy as np

    arr = waveform
    if getattr(arr, "detach", None) is not None:
        arr = arr.detach().cpu().numpy()
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[0]
    if arr.ndim == 1:
        arr = arr[None, :]
    return np.clip(arr, -1.0, 1.0)


def _open_audio_stream(container, audio):
    """The AAC stream, or None if this soundtrack cannot get one.

    Returns None rather than raising: nothing has been muxed yet, so a video-only file
    is still the right answer, and the caller reports muxed=False - which is what puts
    the "NOT sent" note beside the tag instead of silently promising sound.
    """
    try:
        channels = _planar_waveform(audio["waveform"]).shape[0]
        layout = _LAYOUTS.get(channels)
        if layout is None:
            raise ValueError(f"unsupported channel count {channels}")
        return container.add_stream(
            "aac", rate=int(audio["sample_rate"]), layout=layout
        )
    except Exception as e:
        logs.warn(
            "video_audio_dropped", stage="setup", error=f"{type(e).__name__}: {e}"
        )
        return None


def _encode_audio(container, stream, audio) -> None:
    """Feed the waveform through a resampler into the AAC stream.

    Chunked with an explicit monotonic pts in the source's own time base; the resampler
    rebases onto the encoder's layout, format and rate.
    """
    from fractions import Fraction

    import av
    import numpy as np

    arr = _planar_waveform(audio["waveform"])
    sample_rate = int(audio["sample_rate"])
    layout = _LAYOUTS[arr.shape[0]]
    resampler = av.AudioResampler(
        format=stream.format, layout=stream.layout, rate=stream.rate
    )
    time_base = Fraction(1, sample_rate)

    def _mux(resampled_frames):
        for resampled in resampled_frames:
            for packet in stream.encode(resampled):
                container.mux(packet)

    for start in range(0, arr.shape[1], _AUDIO_CHUNK):
        frame = av.AudioFrame.from_ndarray(
            np.ascontiguousarray(arr[:, start : start + _AUDIO_CHUNK]),
            format="fltp",
            layout=layout,
        )
        frame.sample_rate = sample_rate
        frame.pts = start
        frame.time_base = time_base
        _mux(resampler.resample(frame))
    _mux(resampler.resample(None))
    for packet in stream.encode(None):
        container.mux(packet)


def probe(path: str) -> dict:
    """{"kind","width","height","fps","duration","has_audio"} for the modal's reference rows."""
    kind = _guess_kind(path)

    if kind == "image":
        from PIL import Image

        with Image.open(path) as img:
            w, h = img.size
        return {"kind": "image", "width": w, "height": h, "fps": None, "duration": None, "has_audio": False}

    if kind == "video":
        import av

        v = _video_from_file_cls()(path)
        w, h = v.get_dimensions()
        fps = float(v.get_frame_rate())
        duration = v.get_duration()
        with av.open(path) as container:
            has_audio = len(container.streams.audio) > 0
        return {"kind": "video", "width": w, "height": h, "fps": fps, "duration": duration, "has_audio": has_audio}

    # audio
    import av

    with av.open(path) as container:
        duration = float(container.duration / av.time_base) if container.duration else 0.0
    return {"kind": "audio", "width": None, "height": None, "fps": None, "duration": duration, "has_audio": True}


def thumbnail_png(path: str, max_edge: int = 256, crop=None, at_seconds=None) -> bytes:
    """One frame for a video, downscaled full image for a still. `crop` (fraction
    rect) is applied before the downscale, through the same _crop_box rule the
    loaders use, so the tile always previews exactly what the pack will emit.

    `at_seconds` picks the video frame: an indexed seek in the stream's own time
    base, then decode forward until the target pts - the same seek-then-skip-to-pts
    approach core uses (CU/comfy_api/latest/_input_impl/video_types.py:316-325). That
    decodes at most the GOP between the landed keyframe and the target, never the
    whole clip; without it, only frame 0 is decoded, exactly as before. A time past
    the end keeps the last decodable frame rather than failing the tile.
    """
    import io as _io

    from PIL import Image

    if _guess_kind(path) == "video":
        import av

        with av.open(path) as container:
            stream = container.streams.video[0]
            if at_seconds:
                target_pts = int(at_seconds / stream.time_base)
                container.seek(target_pts, stream=stream)
                frame = None
                for frame in container.decode(stream):
                    if frame.pts is not None and frame.pts >= target_pts:
                        break
                if frame is None:
                    raise ValueError(f"could not decode a frame of {path!r} at {at_seconds}s")
            else:
                frame = next(container.decode(stream))
        img = frame.to_image()
    else:
        img = Image.open(path).convert("RGB")

    if crop is not None:
        img = img.crop(_crop_box(crop, img.width, img.height))
    img.thumbnail((max_edge, max_edge))
    buf = _io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
