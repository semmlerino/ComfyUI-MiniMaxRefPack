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

Video decoding is ours, on raw PyAV, and streams: one demux pass selects, rotates, crops
and fills the output tensor in place. `VideoFromFile` materialised the whole trimmed
window as a list of per-frame tensors, `torch.stack`ed it, and then `frames[indices]`
copied it again - measured at 2.7-2.8x the final tensor in peak host RAM, which
extrapolates to ~19 GB for a 10s 1080p reference on a 30 GB box.

The `_open_container` / `_audio_load_fn` indirections below exist so tests can substitute
or instrument the decoder without comfy_api/comfy_extras being installed at all - they
patch the accessor, not the (unimportable) real module.
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
    return [min(max(_source_index(i, src_fps, target_fps), 0), n_src - 1) for i in range(n_out)]


def _source_index(i: int, src_fps: float, target_fps: int) -> int:
    """Which SOURCE frame output frame `i` reads from, before the end-of-clip clamp.

    Extracted so the streaming decoder's cursor cannot drift from `resample_indices`,
    which stays the authority: the decoder asks, for the source frame it has just
    decoded, how many output slots read it, and that has to be the same rule.

    It does NOT depend on `n_src` - `n_src` only sets `n_out` and clamps the tail - so
    the selection is prefix-determined and computable while decoding frame `j` from `j`
    alone. That is what makes one pass possible.
    """
    return round(i * src_fps / target_fps)


def _capacity_for(n_src: int, src_fps: float, target_fps: int) -> int:
    """How many output slots the streaming cursor can REACH for `n_src` source frames.

    Not `len(resample_indices(...))`, and the difference is the whole reason this
    exists. The cursor advances while `_source_index(next_out) <= j`, so its reachable
    maximum is max{i : _source_index(i) <= n_src-1}, which EXCEEDS n_out for real
    inputs: 25fps - the library's most common rate, 45 of 175 clips - overruns at
    n_src = 13, 63, 113, 163, ... (any n_src == 13 mod 50), and 50/59.94/60fps at 11-13.
    Sized to n_out exactly, that write is out of bounds; with the slack it lands in the
    tail and `buf[:n_out]` discards it.

    Walked rather than solved in closed form, because `round()` is half-to-even and the
    inverted inequality is wrong at exactly the ties this has to be right about.
    """
    capacity = max(1, round(n_src * target_fps / src_fps))
    while _source_index(capacity, src_fps, target_fps) <= n_src - 1:
        capacity += 1
    return capacity


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
    waveform = audio["waveform"]
    sliced = waveform[..., int(start * sr):int(end * sr)]
    if sliced.shape[-1] != waveform.shape[-1]:
        # Detach the window from its source when the slice really truncates: a torch
        # view (like a numpy one) keeps its WHOLE storage alive, so a 2s window on a ten
        # minute soundtrack would pin all ten minutes for as long as the MediaCache
        # entry lives. Same defect the streaming decoder removes, one function along.
        #
        # Duck-typed for the same reason `_chunk_to_uint8` is: production always hands
        # this a torch tensor, and the test stack reaches it with numpy.
        release = getattr(sliced, "clone", None) or getattr(sliced, "copy", None)
        if release is not None:
            sliced = release()
    return {"waveform": sliced, "sample_rate": sr}


def _guess_kind(path: str) -> str:
    mime, _ = mimetypes.guess_type(path)
    if mime and mime.startswith("video/"):
        return "video"
    if mime and mime.startswith("audio/"):
        return "audio"
    return "image"


def _open_container(path: str):
    """`av.open`, reached through a MODULE-LEVEL name so a test can wrap it.

    A function-local `import av; av.open(...)` is not patchable - `media.av` would not
    exist - and the pre-roll/trim assertions need to count container opens, seek offsets
    and decoded frames. Correct output does not prove bounded work: a decoder that
    ignores the trim, reads to EOF and slices afterwards produces a byte-identical
    tensor and the right `src_frames`, and the open/decode counts are the only
    instrument that can tell the two apart.
    """
    import av

    return av.open(path)


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
    soundtrack is (_slice_audio).

    A video file is allowed (an audio reference to a clip's soundtrack): it decodes
    through `_decode_audio_only`, never core's loader, and raises a named ValueError
    when the clip has no decodable audio track."""
    import os

    with logs.timed("load_audio", file=os.path.basename(path), trim=trim) as fields:
        if _guess_kind(path) == "video":
            fields["source"] = "video"
            audio = _decode_audio_only(path, trim)
            fields["sample_rate"] = audio["sample_rate"]
            return audio
        waveform, sample_rate = _audio_load_fn()(path)
        audio = {"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate}
        if trim is not None:
            audio = _slice_audio(audio, trim)
        fields["sample_rate"] = sample_rate
        return audio


def load_video(path: str, target_fps: int = 24, crop=None, trim=None):
    """(frames [N,H,W,3] resampled to target_fps, audio dict or None).

    Decoded by `_decode_pass`, not by ComfyUI. The audio dict is BUILT here, to the
    {"waveform","sample_rate"} shape core AUDIO sockets use (the shape contract is still
    video_types.py:445-448) - so getting it right is ours, not inherited.

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


# ---- the streaming video decoder ---------------------------------------------------


class DecodeBudgetExceeded(ValueError):
    """A reference video whose decoded tensor would exceed the decode budget.

    `ValueError` is the base so callers already catching it keep working, and
    `logs.timed`'s failure branch records `error=DecodeBudgetExceeded`. The message
    carries the stable token `decode budget` so a test has something to match that
    prose cannot drift away from.
    """


# The largest tensor one reference video may decode to. 8 GiB clears the largest
# legitimate case in the library by a wide margin - a 10s 1080p window is 5.6 GiB - and
# sits under what this box could finish anyway (30 GB total, ~20 GB routinely in use).
# Without a ceiling, a stream with no `average_rate` falls to Fraction(1)
# (video_types.py:437), making n_out 24x n_src, and the failure arrives as an opaque
# MemoryError or an OOM kill rather than as a named error naming the file.
_MAX_DECODE_BYTES = 8.0 * 2**30
_DECODE_BUDGET_ENV = "MINIMAX_REFPACK_MAX_DECODE_GIB"

_decode_budget_cache: float | None = None


def _reset_decode_budget() -> None:
    """Drop the parsed budget so the next read re-consults the environment.

    The parse is cached at first use rather than at import (a test has to be able to set
    the variable), and a cache without this seam makes a parametrised matrix false-green:
    seven env values in one session would all read the first one's answer and six cases
    would pass against a stale limit while proving nothing.
    """
    global _decode_budget_cache
    _decode_budget_cache = None


def _decode_budget() -> float:
    """The decode ceiling in bytes, parsed from the environment once per process."""
    global _decode_budget_cache
    if _decode_budget_cache is None:
        import os

        _decode_budget_cache = _parse_decode_budget(os.environ.get(_DECODE_BUDGET_ENV))
    return _decode_budget_cache


def _parse_decode_budget(raw) -> float:
    """`MINIMAX_REFPACK_MAX_DECODE_GIB` (GiB, float) -> a byte ceiling, or the default.

    The parsed float AND the scaled byte value are both checked, and the second check is
    not redundant: `inf` parses and is positive, which is why the predicate is
    `isfinite and > 0` rather than `> 0`; and `1e308` passes THAT check while
    `1e308 * 2**30` is `inf`, which would silently disable every `bytes > limit`
    comparison one layer down. Anything rejected is warned about and the default stands.
    """
    if raw is None or not str(raw).strip():
        return _MAX_DECODE_BYTES
    try:
        gib = float(raw)
    except ValueError:
        logs.warn("decode_budget_ignored", value=str(raw), reason="not a number")
        return _MAX_DECODE_BYTES
    if not (math.isfinite(gib) and gib > 0):
        logs.warn("decode_budget_ignored", value=str(raw), reason="not a positive finite size")
        return _MAX_DECODE_BYTES
    limit = gib * 2**30
    if not math.isfinite(limit):
        logs.warn("decode_budget_ignored", value=str(raw), reason="overflows to infinity in bytes")
        return _MAX_DECODE_BYTES
    return limit


def _rotation_k(frame) -> int:
    """A frame's display rotation as quarter-turns, ComfyUI's own spelling.

    `// 90` is floor division verbatim, and PyAV reports rotation signed in (-180, 180],
    so 270 arrives as -90 and floors to -1; `np.rot90` and PIL's `rotate` both take that
    as three turns, and `% 4` is applied only where an index is wanted.

    ONE definition, because `probe()`, `thumbnail_png()` and the decoder must agree about
    which way is up - their docstrings claim exactly that - and three copies of the rule
    would let them diverge silently.
    """
    return int(round(frame.rotation // 90)) if frame.rotation else 0


def _image_format_for(frame) -> tuple[str, bool, bool]:
    """(to_ndarray format, frame carries alpha, needs /255) - video_types.py:356-376.

    The ladder's branch point. `yuvj420p`, `yuvj422p`, `yuvj444p`, `rgb24`, `rgba` and
    `pal8` decode to 8-bit rgb24/rgba and get divided by 255; EVERYTHING else decodes to
    gbrpf32le/gbrapf32le already in 0..1, and only that second group ever reaches
    `align_graph`. Library clips are plain `yuv420p`, so the float path is production's.

    Its own function because `pal8` - the one format that takes the alpha branch without
    having an alpha COMPONENT - has no buildable video fixture in this FFmpeg build
    (every encoder that could carry it refuses the pix_fmt, and the gif decoder hands
    back bgra), so a unit test on a stub frame is the only thing that can reach it.

    The `or name == "pal8"` sits inside the comprehension rather than beside it so a
    format with no components behaves exactly as ComfyUI's loop does.
    """
    name = frame.format.name
    alpha = any(comp.is_alpha or name == "pal8" for comp in frame.format.components)
    if name in ("yuvj420p", "yuvj422p", "yuvj444p", "rgb24", "rgba", "pal8"):
        return ("rgba" if alpha else "rgb24"), alpha, True
    return ("gbrapf32le" if alpha else "gbrpf32le"), alpha, False


class _FrameConverter:
    """One decoded frame -> the [h, w, 3] array a slot is filled from.

    video_types.py:356-406 a frame at a time instead of appended to a list: format
    decision on the first kept frame, the 32-alignment filter graph, rotation, then the
    alpha drop. The scale is reported rather than applied, because it belongs on the
    tensor. Retained state is one filter graph.
    """

    def __init__(self, time_base):
        self._time_base = time_base
        self._image_format: str | None = None
        self._alpha = False
        self.scale = False
        self._align = None

    def convert(self, frame):
        import numpy as np

        if self._image_format is None:
            self._image_format, self._alpha, self.scale = _image_format_for(frame)

        if self._image_format in ("gbrpf32le", "gbrapf32le") and frame.width % 32 != 0:
            img = np.ascontiguousarray(
                self._aligned(frame).to_ndarray(format=self._image_format)[
                    : frame.height, : frame.width
                ]
            )
        else:
            img = frame.to_ndarray(format=self._image_format)

        if frame.rotation != 0:
            img = np.rot90(img, k=_rotation_k(frame), axes=(0, 1)).copy()
        if self._alpha:
            img = img[..., :3]
        return img

    def _aligned(self, frame):
        """Pad to a multiple of 32 and smear the border, video_types.py:381-397.

        Not cosmetic: on a 66x34 frame, skipping it differs by max 0.372, and 56 of the
        172 clips in the library are not a multiple of 32 wide.

        The graph is built from the FIRST frame's format, so a mid-stream pix_fmt change
        fails on `push()`. That is ComfyUI's bug, reproduced deliberately rather than
        diverged from - the whole decoder is asserted against it frame for frame.
        """
        import av

        if self._align is None:
            pad_w = ((frame.width + 31) // 32) * 32
            pad_h = ((frame.height + 31) // 32) * 32
            graph = av.filter.Graph()
            source = graph.add_buffer(
                width=frame.width, height=frame.height,
                format=frame.format.name, time_base=self._time_base,
            )
            pad = graph.add("pad", f"{pad_w}:{pad_h}:0:0")
            fill = graph.add(
                "fillborders",
                f"left=0:right={pad_w - frame.width}:top=0:"
                f"bottom={pad_h - frame.height}:mode=smear",
            )
            sink = graph.add("buffersink")
            source.link_to(pad)
            pad.link_to(fill)
            fill.link_to(sink)
            graph.configure()
            self._align = (graph, source, sink)
        self._align[1].push(frame)
        return self._align[2].pull()


class _FrameSink:
    """The output tensor, filled in place.

    Over-allocating is free in RSS - a torch CPU tensor's pages are not faulted in until
    they are written (measured on this host: an 8.05 GB `torch.empty` added 0.001 GB
    RSS, touching 0.74 GB of it added exactly 0.736 GB, and freeing returned it all) -
    which is what lets the capacity be an over-estimate instead of a growing buffer.
    Grow-and-copy would cost 2x the written prefix at the growth point, reintroducing
    the multiplier this whole change exists to remove.

    The frames are written already cropped, so the storage behind the returned tensor is
    crop-sized. Filling a full-frame buffer and returning a crop VIEW would satisfy every
    pixel assertion and every RSS budget while pinning full-frame storage for the
    lifetime of the MediaCache entry - which is precisely the defect being removed.
    """

    def __init__(self, capacity: int, height: int, width: int, scale: bool):
        import torch

        # float32 explicitly, never inherited from the first frame: a float64 sink would
        # satisfy `torch.equal` against the oracle while doubling the tensor and halving
        # what the byte ceiling's `* 4` is counting.
        self._buf = torch.empty((capacity, height, width, 3), dtype=torch.float32)
        self._scale = scale
        self.capacity = capacity
        self.filled = 0

    def write(self, img, count: int) -> None:
        """Broadcast one [h, w, 3] frame into the next `count` slots."""
        import torch

        tensor = torch.from_numpy(img)
        if self._scale:
            tensor = tensor.float().div_(255.0)
        self._buf[self.filled : self.filled + count] = tensor
        self.filled += count

    def take(self, n_out: int):
        return self._buf[:n_out]


class _PassResult:
    """What one demux pass learned. `needs_recount` means the caller must run another."""

    __slots__ = ("sink", "box", "audio", "n_src", "src_fps", "needs_recount", "reason")

    def __init__(self):
        self.sink = None
        self.box = None
        self.audio = None
        self.n_src = 0
        self.src_fps = 1.0
        self.needs_recount = False
        self.reason: str | None = None


def _trim_window(trim) -> tuple[float, float]:
    """`[start, end]` seconds -> `(start_time, duration)`, ComfyUI's own pair.

    A NEGATIVE start is deliberately NOT reproduced. `get_active_trim_window`
    (video_types.py:141-145) reads it as an offset from the end; `refs.validate_trim`
    rejects `start < 0` before it can reach `load_video`, so that branch is unreachable
    on the real path. Raising beats silently reinterpreting: a direct caller bypassing
    `validate_trim` would otherwise get end-relative behaviour from one decoder and
    absolute from the other.
    """
    if trim is None:
        return 0.0, 0.0
    start, end = float(trim[0]), float(trim[1])
    if start < 0:
        raise ValueError(
            f"trim start must be >= 0, got {start} - a negative start is an "
            "end-relative offset to ComfyUI and is not reproduced here"
        )
    return start, end - start


def _estimate_source_frames(container, stream, start_time: float, duration: float):
    """How many source frames the requested window probably holds, from metadata only.

    THE TWO DURATION FIELDS ARE IN DIFFERENT UNITS. `stream.duration` counts in
    `stream.time_base`; `container.duration` counts in `av.time_base` (microseconds).
    Multiplying either by `average_rate` without its own conversion is wrong by orders
    of magnitude and the error is SILENT - the fallbacks absorb it, at the cost of a
    recount on every clip, which reads as a mysterious performance regression rather
    than as a bug.

    Allowed to be wrong in either direction; `None` when metadata cannot answer.
    """
    import av

    rate = stream.average_rate or stream.guessed_rate
    if not rate:
        return None
    if not start_time and not duration and stream.frames:
        return int(stream.frames)
    if stream.duration is not None and stream.time_base:
        src_seconds = float(stream.duration * stream.time_base)
    elif container.duration:
        src_seconds = container.duration / av.time_base
    else:
        return None
    if duration:
        usable = max(0.0, min(src_seconds, start_time + duration) - start_time)
    else:
        usable = max(0.0, src_seconds - start_time)
    return math.ceil(usable * rate)


def _last_decodable_audio_stream(container, *, quiet: bool = False):
    """video_types.py:65-74: backwards for the first stream with a codec context.

    NOT `container.streams.audio[-1]`. Streams FFmpeg has no decoder for have no codec
    context and decoding their packets crashes the process - the iPhone APAC
    spatial-audio track this helper exists for. The shorthand passes a two-good-track
    test and then takes the process down on a real file.
    """
    streams = container.streams.audio
    stream = next((s for s in reversed(streams) if s.codec_context is not None), None)
    if stream is None and len(streams) and not quiet:
        logs.warn("video_audio_dropped", stage="select", error="no decodable audio stream")
    return stream


def _frame_start_seconds(frame, sample_rate: int) -> float:
    """When a RESAMPLED audio frame starts, in seconds. `_save_transcoded`'s form (:741-745).

    Two traps, and each one reintroduces the head-trim bug this replaces:

    `frame.time` is the tempting shorthand and is wrong - a frame carrying a pts with no
    time base yields NaN, `int(NaN)` raises rather than falling back, and every
    comparison against NaN is false, so a tail check written on it never fires and audio
    is retained to EOF.

    The fallback must be `Fraction(1, sample_rate)` (video_types.py:561), NOT the
    stream's time base. `AudioResampler(format='fltp')` REBASES pcm/mp2 pts into
    1/sample_rate while passing AAC through, so multiplying a rebased pts by mkv's
    1/1000 under-trims and by mpegts' 1/90000 over-trims. That is exactly the
    miscalculation being removed, on exactly the containers it targets.

    Its own function because neither state a fallback covers is reachable from a real
    file: every measured mkv arm comes back out of the resampler carrying
    `time_base=1/44100`, so an assertion placed on a container fixture could not fail.
    """
    from fractions import Fraction

    if frame.pts is None:
        return 0.0
    time_base = frame.time_base if frame.time_base else Fraction(1, sample_rate)
    return float(frame.pts * time_base)


class _AudioCollector:
    """Resample, head-trim and tail-trim one soundtrack's decoded frames.

    Shared by `_decode_pass` (a video's `video_audio_N`) and `_decode_audio_only` (an
    audio reference that points at a video file), so the same file trimmed the same
    way yields the same waveform through either socket - two copies of this loop
    would drift. `sample_rate` is the stream's advertised rate, 0 when unknown; the
    first decoded frame supplies it then.
    """

    def __init__(self, sample_rate: int, start_time: float, duration: float):
        self.sample_rate = sample_rate
        self.start_time = start_time
        self.duration = duration
        self.done = False
        self._resampler = None
        self._frames: list = []
        self._has_first = False

    def feed(self, raw) -> None:
        import av

        if self._resampler is None:
            # Deferred initialisation, not a look-ahead: the rate arrives WITH the first
            # frame, so no frame can precede it. This is why `probe_audio_params`
            # (video_types.py:76-89) is not used - its own docstring says "the caller
            # must seek back afterwards", and the seek it needs undone is what would
            # break the one-pass claim.
            rate = self.sample_rate or raw.sample_rate
            if not rate:
                return
            self.sample_rate = int(rate)
            self._resampler = av.audio.resampler.AudioResampler(format="fltp")
        start_time, duration, sample_rate = self.start_time, self.duration, self.sample_rate
        for frame in self._resampler.resample(raw):
            frame_start = _frame_start_seconds(frame, sample_rate)
            if duration and frame_start > start_time + duration:
                self.done = True
                return
            if not self._has_first:
                to_skip = max(0, int((start_time - frame_start) * sample_rate))
                if to_skip < frame.samples:
                    self._has_first = True
                    self._frames.append(frame.to_ndarray()[..., to_skip:])
            else:
                self._frames.append(frame.to_ndarray())

    def result(self) -> dict | None:
        """The AUDIO dict, or None when nothing usable was decoded."""
        import numpy as np
        import torch

        if not (self._frames and self.sample_rate):
            return None
        data = np.concatenate(self._frames, axis=1)
        if self.duration:
            limit = int(self.duration * self.sample_rate)
            if limit < data.shape[1]:
                # .copy(), NOT np.ascontiguousarray: a (1, N) mono buffer sliced to
                # (1, M) is still flagged C_CONTIGUOUS - a leading axis of extent 1
                # makes its stride irrelevant to the check - so ascontiguousarray
                # returns the same VIEW and the whole concatenation stays alive
                # behind it. (2, N) stereo does get copied, so the leak is silent on
                # exactly the shape most fixtures use.
                data = data[..., :limit].copy()
        return {
            "waveform": torch.from_numpy(data).unsqueeze(0),
            "sample_rate": int(self.sample_rate),
        }


def _decode_audio_only(path: str, trim) -> dict:
    """The soundtrack of a VIDEO file used as an audio reference; frames never decoded.

    Not core `nodes_audio.load`: it decodes `streams.audio[0]` unguarded, which is not
    the track `_decode_pass` picks for the same file's `video_audio_N` (the last
    decodable one) and crashes the process on a codec-less first track (iPhone APAC).
    """
    import os

    name = os.path.basename(path)
    start_time, duration = _trim_window(trim)
    with _open_container(path) as container:
        stream = _last_decodable_audio_stream(container)
        if stream is None:
            raise ValueError(f"reference audio {name!r} has no decodable audio track")
        collector = _AudioCollector(stream.codec_context.sample_rate or 0, start_time, duration)
        # Seek exactly as `_decode_pass` does - on the video stream - so the audio
        # decoder is fed the same packets from the same position. Seeking on the audio
        # stream instead lands elsewhere and changes the codec's pre-roll: measured,
        # an AAC head came back -0.09 where the video path gives 0.57.
        videos = container.streams.video
        seek_stream = videos[0] if videos else stream
        if start_time and seek_stream.time_base:
            container.seek(int(start_time / seek_stream.time_base), stream=seek_stream)
        for packet in container.demux(stream):
            for raw in packet.decode():
                collector.feed(raw)
                if collector.done:
                    break
            if collector.done:
                break
        audio = collector.result()
    if audio is None:
        raise ValueError(f"reference audio {name!r} yielded no decodable audio")
    return audio


def _cropped(img, box):
    if box is None:
        return img
    left, top, right, bottom = box
    return img[top:bottom, left:right]


def _decode_video(path, target_fps, crop, trim, fields):
    """The body of load_video: one streaming pass, plus an exact second one if needed."""
    import os

    start_time, duration = _trim_window(trim)
    result = _decode_pass(
        path, target_fps, crop, start_time, duration, n_src_hint=None, want_audio=True
    )
    if result.needs_recount:
        if result.reason is not None:
            logs.warn(
                "video_decode_recount",
                file=os.path.basename(path),
                reason=result.reason,
                src_frames=result.n_src,
            )
        audio = result.audio
        result = _decode_pass(
            path, target_fps, crop, start_time, duration,
            n_src_hint=result.n_src, want_audio=False,
        )
        result.audio = audio

    n_src, src_fps = result.n_src, result.src_fps
    fields["src_frames"] = n_src
    fields["fps"] = src_fps

    # On the RESULT, and never on the estimate: the estimate is explicitly allowed to be
    # wrong, so moving this check onto it would reject valid clips off bad metadata. It
    # looks like an obvious optimisation, which is why this says so.
    indices = resample_indices(n_src, src_fps, target_fps)
    if len(indices) < 5:
        clip_seconds = (n_src / src_fps) if src_fps else 0.0
        window = f" trimmed to {trim[0]:.2f}-{trim[1]:.2f}s" if trim is not None else ""
        raise ValueError(
            f"reference video {path!r}{window} has only {len(indices)} frame(s) at {target_fps}fps "
            f"(source: {n_src} frames, {clip_seconds:.2f}s) - MiniMax H3 needs at least 5"
        )

    out = result.sink.take(len(indices))
    fields["frames"] = len(indices)
    fields["audio"] = result.audio is not None
    return out, result.audio


def _decode_pass(path, target_fps, crop, start_time, duration, *, n_src_hint, want_audio):
    """One demux pass: select, rotate, crop and fill, reproducing video_types.py:308-448.

    `n_src_hint` is the true source-frame count when a previous pass counted it, else
    None (the capacity comes from metadata, and may be absent or wrong). A pass with a
    hint never asks for another.
    """
    import av

    result = _PassResult()
    with _open_container(path) as container:
        streams = container.streams.video
        if not streams:
            raise ValueError(f"no video stream found in {path!r}")
        video_stream = streams[0]
        # ONE Fraction -> float conversion. `round(i * Fraction(30000,1001) / 24)` rounds
        # exactly while the float spelling rounds a double, and they disagree at ties.
        src_fps = float(video_stream.average_rate) if video_stream.average_rate else 1.0
        result.src_fps = src_fps

        time_base = video_stream.time_base
        # int(), not round(): it moves a boundary frame, which changes n_src, which
        # changes the socket batch size. video_types.py:316-317.
        start_pts = int(start_time / time_base)
        end_pts = int((start_time + duration) / time_base)
        if start_pts != 0:
            container.seek(start_pts, stream=video_stream)

        capacity = None
        if n_src_hint is not None:
            capacity = _capacity_for(n_src_hint, src_fps, target_fps)
        else:
            estimate = _estimate_source_frames(container, video_stream, start_time, duration)
            if estimate is not None:
                # 25% headroom (floor 8) on top of the reachable count, because the
                # estimate is metadata and metadata is often a little short. Untouched
                # slack costs nothing in RSS, while a recount costs a whole second decode
                # - so the headroom is the cheaper side of that trade by a wide margin.
                # It stays modest because `vm.overcommit_memory=0` refuses a single
                # allocation exceeding RAM+swap outright.
                reachable = _capacity_for(estimate, src_fps, target_fps)
                capacity = reachable + max(8, reachable // 4)
        if capacity is None:
            # No usable metadata: count first, retaining nothing but the soundtrack,
            # then allocate exactly. Still 1x the final tensor at worst, unlike growing.
            result.needs_recount = True

        audio_stream = _last_decodable_audio_stream(container) if want_audio else None
        demuxed = [video_stream] if audio_stream is None else [video_stream, audio_stream]
        # Falsiness, not `is None`: MPEG-TS carries audio parameters in-band, so a stream
        # selected before any packet is decoded can present 0 (video_types.py:546 seeds
        # it that way and branches on `if not sample_rate` at :552). `Fraction(1, None)`
        # raises TypeError and `Fraction(1, 0)` raises ZeroDivisionError, so an `is None`
        # check turns a clean "no audio" into a crash on exactly the shape it exists for.
        sample_rate = 0
        if audio_stream is not None and audio_stream.codec_context is not None:
            sample_rate = audio_stream.codec_context.sample_rate or 0
        collector = _AudioCollector(sample_rate, start_time, duration)

        converter = _FrameConverter(time_base)
        sink = None
        box = None
        n_seen = 0
        next_out = 0
        last_frame = None
        video_done = False
        audio_done = audio_stream is None

        for packet in container.demux(*demuxed):
            if video_done and audio_done:
                break

            if packet.stream.type == "video":
                if video_done:
                    continue
                try:
                    for frame in packet.decode():
                        # Guarded `pts is None`, `_save_transcoded`'s form (:637):
                        # `get_components_internal` compares unguarded and would TypeError.
                        if frame.pts is not None and frame.pts < start_pts:
                            continue
                        if duration and frame.pts is not None and frame.pts >= end_pts:
                            video_done = True
                            break

                        count = 0
                        while _source_index(next_out + count, src_fps, target_fps) <= n_seen:
                            count += 1
                        if count and capacity is not None:
                            if next_out + count > capacity:
                                # The estimate was low. Release the buffer and keep
                                # decoding purely to learn the true n_src.
                                sink = None
                                capacity = None
                                result.needs_recount = True
                                result.reason = "capacity"
                            else:
                                img = converter.convert(frame)
                                if sink is None:
                                    sink, box = _build_sink(
                                        img, capacity, crop, converter.scale,
                                        path=path, src_fps=src_fps,
                                        estimated=n_src_hint is None,
                                    )
                                    if sink is None:
                                        capacity = None
                                        result.needs_recount = True
                                        result.reason = "budget"
                                if sink is not None:
                                    sink.write(_cropped(img, box), count)
                        # Retained unconditionally: the EOF clamp fills [next_out, n_out)
                        # from source frame n_src-1, which is not always itself selected
                        # (src_fps=12, target_fps=24, n_src=12 -> n_out=24, next_out=23).
                        last_frame = frame
                        next_out += count
                        n_seen += 1
                except av.error.InvalidDataError:
                    logs.debug("video_decode_error", file=path)

            elif packet.stream.type == "audio":
                if audio_done:
                    continue
                for raw in packet.decode():
                    collector.feed(raw)
                    if collector.done:
                        audio_done = True
                        break

        result.n_src = n_seen
        result.box = box
        result.sink = sink

        n_out = len(resample_indices(n_seen, src_fps, target_fps)) if n_seen else 0
        if sink is not None and last_frame is not None and next_out < n_out:
            sink.write(_cropped(converter.convert(last_frame), box), n_out - next_out)

        audio = collector.result()
        if audio is not None:
            result.audio = audio
        elif audio_stream is not None and not collector.sample_rate:
            logs.warn(
                "video_audio_dropped", stage="decode",
                error="stream never yielded a frame with a usable sample rate",
            )
    return result


def _build_sink(img, capacity, crop, scale, *, path, src_fps, estimated):
    """(sink, crop box), or (None, None) when the capacity busts the decode budget.

    An INFLATED estimate must not reject a valid clip, so a bust on an estimated
    capacity degrades to the count-first pass and the ceiling is enforced against the
    true `n_src` on the next one.
    """
    height, width = int(img.shape[0]), int(img.shape[1])
    box = _crop_box(crop, width, height) if crop is not None else None
    if box is not None:
        left, top, right, bottom = box
        height, width = bottom - top, right - left

    nbytes = capacity * height * width * 3 * 4
    if nbytes > _decode_budget():
        if estimated:
            return None, None
        raise DecodeBudgetExceeded(
            f"reference video {path!r} exceeds the decode budget: {src_fps:g}fps implies "
            f"{capacity} frames of {width}x{height}, {nbytes / 2**30:.2f} GiB, against a "
            f"limit of {_decode_budget() / 2**30:.2f} GiB "
            f"(raise {_DECODE_BUDGET_ENV} to override)"
        )
    return _FrameSink(capacity, height, width, scale), box


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


def _waveform_channels(waveform) -> int:
    """How many channels a [1,C,L] / [C,L] / [L] waveform has, WITHOUT building one.

    `_planar_waveform(...).shape[0]` answers the same question and costs a full clipped
    float32 copy of the whole soundtrack, which is discarded and then rebuilt by
    `_encode_audio` moments later. The shapes it collapses are the contract here: a
    batch dim is dropped, a bare [L] is one channel.
    """
    shape = getattr(waveform, "shape", None)
    if shape is None:
        import numpy as np

        shape = np.asarray(waveform).shape
    if len(shape) == 3:
        return int(shape[1])
    if len(shape) == 2:
        return int(shape[0])
    return 1


def _open_audio_stream(container, audio):
    """The AAC stream, or None if this soundtrack cannot get one.

    Returns None rather than raising: nothing has been muxed yet, so a video-only file
    is still the right answer, and the caller reports muxed=False - which is what puts
    the "NOT sent" note beside the tag instead of silently promising sound.
    """
    try:
        channels = _waveform_channels(audio["waveform"])
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
        return _probe_video(path)

    # audio
    import av

    with _open_container(path) as container:
        duration = float(container.duration / av.time_base) if container.duration else 0.0
    return {"kind": "audio", "width": None, "height": None, "fps": None, "duration": duration, "has_audio": True}


def _probe_video(path: str) -> dict:
    """One container open, raw PyAV, DISPLAY orientation.

    THE METADATA CONTRACT IS PART OF THIS, not a detail. `VideoFromFile` supplied fps and
    duration through a chain of fallbacks, so a straightforward one-open implementation
    returns None for clips whose stream metadata is thin - a regression that surfaces as
    a permanently pending tile rather than as an error, and only on the metadata-poor
    files nobody has in a fixture. So the rungs are explicit:

      fps       average_rate -> guessed_rate -> base_rate -> None
      duration  stream.duration * stream.time_base -> container.duration / av.time_base
                -> None                      (the two are in DIFFERENT units; see
                                              _estimate_source_frames)
      w/h       from the decoded keyframe, so a stale declared size still reports right

    None is a legitimate answer for each and the route renders it. What is NOT legitimate
    is a fabricated number - `frame_rate` degrades to Fraction(1) inside
    `get_components_internal` (video_types.py:437), and reporting 1 fps as if it were
    measured is worse than reporting nothing.

    Dimensions are DISPLAY dimensions: `frame.to_image()` does not apply the display
    matrix and `stream.width/height` stays raw, while the decoded tensor IS rotated and
    the browser's <video> IS rotated. Reporting the raw pair is what made a crop rect
    drawn on the tile disagree with the pixels `load_video` emitted.

    The error contract covers `av.open` too, not just the rotation decode: a corrupt
    container fails before any rotation fallback can run, `probe_route` (routes.py:69)
    has no `try`, and anything escaping becomes a 500 that leaves the tile pending
    forever. Every failure returns a complete, valid probe dict.
    """
    import os

    empty = {"kind": "video", "width": None, "height": None,
             "fps": None, "duration": None, "has_audio": False}
    try:
        import av

        with _open_container(path) as container:
            streams = container.streams.video
            if not streams:
                # An audio-only .mp4 is still a usable AUDIO reference; say it has sound.
                has_audio = _last_decodable_audio_stream(container, quiet=True) is not None
                return {**empty, "has_audio": has_audio}
            stream = streams[0]
            rate = stream.average_rate or stream.guessed_rate or stream.base_rate
            duration = None
            if stream.duration is not None and stream.time_base:
                duration = float(stream.duration * stream.time_base)
            elif container.duration:
                duration = container.duration / av.time_base
            k, width, height = _first_frame_rotation_k(container, stream)
            if width is None:
                width, height = stream.width or None, stream.height or None
            if k % 2 and width is not None and height is not None:
                width, height = height, width
            return {
                "kind": "video",
                "width": width,
                "height": height,
                "fps": float(rate) if rate else None,
                "duration": duration,
                # Decodable, not merely present: an APAC-only iPhone clip has an audio
                # stream the decoder skips, and the tile must not promise its sound.
                "has_audio": _last_decodable_audio_stream(container, quiet=True) is not None,
            }
    except Exception as e:
        logs.warn("probe_failed", file=os.path.basename(path),
                  error=f"{type(e).__name__}: {e}")
        return empty


def _first_frame_rotation_k(container, stream) -> tuple[int, int | None, int | None]:
    """(rotation quarter-turns, decoded width, decoded height); (0, None, None) on failure.

    A decode is the only route: PyAV 18.1.0 exposes no stream-level rotation getter and
    the display matrix is not in `stream.metadata`. This runs on the aiohttp loop, so it
    takes the thumb route's failure tolerance - a clip that will not decode still gets a
    dict, and a scalar would break the route's dictionary indexing.
    """
    try:
        frame = next(container.decode(stream))
    except Exception as e:
        logs.debug("probe_rotation_failed", error=f"{type(e).__name__}: {e}")
        return 0, None, None
    return _rotation_k(frame) % 4, int(frame.width), int(frame.height)


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

    The frame is rotated into DISPLAY orientation before the crop. `frame.to_image()`
    does not apply the display matrix, while `load_video` rotates and the browser's
    <video> rotates, so without this a crop rect drawn on a portrait-shot phone clip
    previewed one region and emitted another. `img.rotate(90*k, expand=True)` is
    byte-identical to `np.rot90(a, k, axes=(0, 1))` - verified for every k the decoder
    can produce, including the negative ones PyAV's signed rotation yields.
    """
    import io as _io

    from PIL import Image

    if _guess_kind(path) == "video":
        with _open_container(path) as container:
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
            k = _rotation_k(frame) % 4
            if k:
                img = img.rotate(90 * k, expand=True)
    else:
        img = Image.open(path).convert("RGB")

    if crop is not None:
        img = img.crop(_crop_box(crop, img.width, img.height))
    img.thumbnail((max_edge, max_edge))
    buf = _io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
