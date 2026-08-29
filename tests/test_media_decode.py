"""The streaming video decoder, against real media.

`media._decode_video` used to hand the whole trimmed window to ComfyUI's `VideoFromFile`
and copy the result twice; it is now one demux pass that fills the output tensor in
place. Nothing here is mocked, because the thing being checked is that the pixels did
not change - and the only witness to that is the decoder that was replaced.

`_reference_decoder.reference_decode` is that witness: a verbatim transcription of
`get_components_internal`, asserted against with `torch.equal` on the whole tensor. It is
the highest-value item in this file, and every other test here exists because the oracle
alone cannot see something:

  the oracle       says the pixels are right
  selection        says the SLOT->SOURCE map is right where the cursor can overrun n_out
  preallocation    says a wrong estimate changes nothing but the work done
  trim/pre-roll    says the work is BOUNDED - a decoder that reads to EOF and slices
                   afterwards produces a byte-identical tensor and passes the oracle
  memory           says the copies are actually gone
  orientation      says probe/thumbnail/tensor agree about which way is up
"""

import math
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(__file__))

from _reference_decoder import reference_decode  # noqa: E402
from conftest import CLIP_FORMATS, marker_index, write_clip  # noqa: E402

from minimax_refpack import media  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def oracle(path, trim=None, crop=None, target_fps=24):
    """What the OLD decoder would have emitted for this call."""
    start, duration = (trim[0], trim[1] - trim[0]) if trim else (0.0, 0.0)
    ref, audio, rate = reference_decode(path, start, duration)
    want = ref[media.resample_indices(ref.shape[0], float(rate), target_fps)]
    if crop is not None:
        left, top, right, bottom = media._crop_box(crop, want.shape[2], want.shape[1])
        want = want[:, top:bottom, left:right, :]
    return want, audio, ref.shape[0]


def assert_output_contract(frames):
    """dtype and range, explicitly - `torch.equal` pins neither.

    A float64 sink would satisfy the oracle while doubling the tensor and halving what
    the decode ceiling's `* 4` bytes-per-element arithmetic is counting.
    """
    assert frames.dtype is torch.float32
    assert frames.ndim == 4
    assert frames.shape[-1] == 3
    assert 0.0 <= float(frames.min())
    assert float(frames.max()) <= 1.0


# ---- A. the oracle -----------------------------------------------------------------


@pytest.mark.parametrize("pix_fmt", sorted(CLIP_FORMATS))
@pytest.mark.parametrize("size", [(64, 48), (66, 34)], ids=["w64", "w66_unaligned"])
def test_the_stream_decoder_matches_comfyui_frame_for_frame(clip, pix_fmt, size):
    """The whole tensor, bit for bit, across both halves of the format ladder.

    The 66-wide arm is not decoration: `align_graph` fires ONLY on the float formats and
    ONLY when width % 32 != 0, which is the branch production actually takes - library
    clips are plain `yuv420p`, and 56 of the 172 of them are not a multiple of 32 wide.
    A matrix built from the 8-bit list alone would leave it completely unexercised.
    """
    path = clip(f"c_{pix_fmt}_{size[0]}", n=30, fps=30, pix_fmt=pix_fmt, size=size)
    assert_fixture_is_what_was_asked(path, pix_fmt=pix_fmt)

    want, _, n_src = oracle(path)
    got, _ = media.load_video(path)

    assert n_src == 30
    assert torch.equal(got, want)
    assert_output_contract(got)


@pytest.mark.parametrize("rotation", [90, 180, 270])
@pytest.mark.parametrize("size", [(64, 48), (66, 34)], ids=["w64", "w66_unaligned"])
def test_a_rotated_clip_matches_comfyui_frame_for_frame(clip, rotation, size):
    path = clip(f"r{rotation}_{size[0]}", n=30, fps=30, rotation=rotation, size=size)
    assert_fixture_is_what_was_asked(path, rotation=rotation)

    want, _, _ = oracle(path)
    got, _ = media.load_video(path)

    # The display matrix really did change the emitted geometry.
    assert (got.shape[1], got.shape[2]) == (
        (size[0], size[1]) if rotation % 180 else (size[1], size[0])
    )
    assert torch.equal(got, want)
    assert_output_contract(got)


def test_a_trimmed_window_matches_comfyui_frame_for_frame(clip):
    path = clip("trimmed", n=120, fps=30, size=(66, 34), gop=10,
                container="mp4", codec="libx264")
    want, _, n_src = oracle(path, trim=[1.0, 3.0])
    got, _ = media.load_video(path, trim=[1.0, 3.0])
    assert n_src == 60
    assert torch.equal(got, want)


def test_a_cropped_and_trimmed_window_matches_comfyui_frame_for_frame(clip):
    path = clip("cropped", n=120, fps=30, size=(66, 34), gop=10,
                container="mp4", codec="libx264")
    crop = [0.1, 0.2, 0.55, 0.6]
    want, _, _ = oracle(path, trim=[1.0, 3.0], crop=crop)
    got, _ = media.load_video(path, trim=[1.0, 3.0], crop=crop)
    assert got.shape[1:3] != (34, 66)      # the crop really cropped
    assert torch.equal(got, want)


def assert_fixture_is_what_was_asked(path, *, pix_fmt=None, rotation=None):
    """The encoder produced what the parametrisation asked for.

    Without this, an encoder that silently substitutes a format or ignores
    `set_display_rotation` leaves the align and rotation branches untested while every
    behavioural assertion still passes - a green suite proving nothing.

    Rotation is compared CANONICALLY. PyAV reports it signed in (-180, 180], so
    measured: 90 -> 90, 180 -> -180, 270 -> -90, and a naive `== requested` fails two of
    the three cases outright.
    """
    import av

    with av.open(path) as container:
        frame = next(container.decode(container.streams.video[0]))
        if pix_fmt is not None:
            assert frame.format.name == pix_fmt, (
                f"encoder substituted {frame.format.name} for the requested {pix_fmt}, "
                "which moves this case onto the other branch of the format ladder"
            )
        if rotation is not None:
            assert frame.rotation % 360 == rotation % 360


_CROSSCHECK = r"""
import sys
sys.path.insert(0, {comfy!r})
sys.path.insert(0, {tests!r})
import torch
from comfy_api.latest._input_impl.video_types import VideoFromFile
from _reference_decoder import reference_decode

failures = []
for path in {paths!r}:
    real = VideoFromFile(path).get_components()
    mine, _, rate = reference_decode(path)
    if not torch.equal(mine, real.images):
        failures.append(f"{{path}}: images differ")
    elif tuple(mine.shape) != tuple(real.images.shape):
        failures.append(f"{{path}}: shapes differ")
    if rate != real.frame_rate:
        failures.append(f"{{path}}: frame_rate {{rate}} != {{real.frame_rate}}")
print("FAILURES:" + ("; ".join(failures) if failures else "none"))
"""


@pytest.mark.slow
@pytest.mark.skipif(
    not os.environ.get("MINIMAX_REFPACK_COMFYUI"),
    reason="set MINIMAX_REFPACK_COMFYUI to a ComfyUI checkout to re-check the oracle",
)
def test_the_transcription_still_matches_comfyui(clip):
    """The oracle is a COPY, and a copy drifts. This is what says it has not.

    Without it, `_reference_decoder` is a transcription asserted against nothing but
    review, and the whole differential suite keeps passing against a stale idea of what
    ComfyUI does - which is the one failure mode a differential suite cannot self-detect.

    It runs in ComfyUI's OWN venv, as a subprocess. `comfy_api.latest.__init__` pulls in
    `comfy_execution.progress` and therefore tqdm, so the class is not importable from
    this venv at all - which is the same reason media.py never imported it either.

    Every format arm, because the ladder's two halves are where a transcription would
    most plausibly have drifted.
    """
    comfy = os.environ["MINIMAX_REFPACK_COMFYUI"]
    interpreter = os.path.join(comfy, ".venv", "bin", "python")
    if not os.path.exists(interpreter):
        pytest.skip(f"no interpreter at {interpreter}")

    paths = [
        clip(f"xcheck_{pix_fmt}", n=30, fps=30, pix_fmt=pix_fmt, size=(66, 34))
        for pix_fmt in sorted(CLIP_FORMATS)
    ]
    paths.append(clip("xcheck_rot", n=30, fps=30, rotation=270, size=(66, 34)))

    script = _CROSSCHECK.format(
        comfy=comfy, tests=os.path.dirname(os.path.abspath(__file__)), paths=paths
    )
    out = subprocess.run(
        [interpreter, "-c", script], capture_output=True, text=True, timeout=600
    )
    assert out.returncode == 0, out.stderr[-3000:]
    verdict = [ln for ln in out.stdout.splitlines() if ln.startswith("FAILURES:")]
    assert verdict, out.stdout[-2000:] + out.stderr[-2000:]
    assert verdict[0] == "FAILURES:none", verdict[0]


# ---- B. selection, including the output-domain boundary ----------------------------

# The eight `resample_indices` cases that can be encoded, PLUS every (n_src, fps) where
# the cursor's reachable maximum EXCEEDS n_out. Without those the cursor can write past
# the end of the buffer and every selection assertion still passes.
SELECTION_CASES = [
    (24, 24), (30, 30), (60, 60), (12, 12), (48, 24),   # identity / down / up / clamp
    (13, 25), (63, 25), (13, 30), (11, 50), (13, 60),   # the overrun boundary
]


@pytest.mark.parametrize("n_src,src_fps", SELECTION_CASES,
                         ids=[f"{n}f@{f}" for n, f in SELECTION_CASES])
def test_the_recovered_slot_to_source_map_is_resample_indices(clip, n_src, src_fps):
    """Every output slot names the source frame `resample_indices` says it should.

    A lossless format, so `marker_index` recovers the source ordinal exactly rather than
    approximately - the point is an EQUALITY against the authority, not a resemblance.
    """
    path = clip(f"sel_{n_src}_{src_fps}", n=n_src, fps=src_fps,
                pix_fmt="rgb24", size=(64, 48))
    frames, _ = media.load_video(path)

    expected = media.resample_indices(n_src, float(src_fps), 24)
    recovered = [marker_index(frames[i, 0, 0]) for i in range(frames.shape[0])]
    assert recovered == expected
    assert len(recovered) == len(expected)


def reachable_max(n_src, src_fps, target_fps=24):
    """The highest slot index the streaming cursor can write for this input."""
    return max(
        i for i in range(n_src * target_fps * 2 + 16)
        if media._source_index(i, float(src_fps), target_fps) <= n_src - 1
    )


@pytest.mark.parametrize("n_src,src_fps", [(13, 25), (63, 25), (13, 30), (11, 50), (13, 60)])
def test_the_cursor_can_reach_further_than_the_output_needs(n_src, src_fps):
    """The reason capacity is never sized to `len(resample_indices(...))`.

    For these inputs the cursor's reachable maximum REACHES n_out - i.e. it can write
    one slot past what the result needs. Sized to n_out exactly, that write is out of
    bounds; with the slack it lands in the tail and `buf[:n_out]` discards it.
    """
    n_out = len(media.resample_indices(n_src, float(src_fps), 24))
    capacity = media._capacity_for(n_src, float(src_fps), 24)
    reachable = reachable_max(n_src, src_fps)

    assert reachable >= n_out, "this case was chosen because it overruns; it no longer does"
    assert capacity > reachable, "the cursor could write out of bounds"


def test_the_tail_is_filled_from_the_last_frame_when_the_cursor_stops_short(clip):
    """The mirror case, and why the last raw frame is retained UNCONDITIONALLY.

    At src_fps=12 into 24fps with n_src=12 the cursor stops at slot 23 of 24, so the
    final slot is filled by the end-of-clip clamp from source frame n_src-1 - which is
    not itself selected by the cursor. A decoder that only retained frames it had just
    written would have nothing to fill it with.
    """
    n_out = len(media.resample_indices(12, 12.0, 24))
    assert reachable_max(12, 12) < n_out - 1

    path = clip("clamp", n=12, fps=12, pix_fmt="rgb24", size=(64, 48))
    frames, _ = media.load_video(path)
    assert frames.shape[0] == n_out
    # The clamped tail really is the last source frame, not a zeroed slot.
    assert marker_index(frames[-1, 0, 0]) == 11
    want, _, _ = oracle(path)
    assert torch.equal(frames, want)


def test_the_overrun_is_never_more_than_one_slot():
    """The measured bound the `+1` rests on, re-derived rather than remembered."""
    for src_fps in (12, 15, 23.976, 24, 25, 29.97, 30, 48, 50, 59.94, 60, 120):
        for n_src in range(1, 400):
            n_out = len(media.resample_indices(n_src, float(src_fps), 24))
            capacity = media._capacity_for(n_src, float(src_fps), 24)
            assert n_out <= capacity <= n_out + 1


# ---- C. preallocation paths --------------------------------------------------------


@pytest.mark.parametrize("n_src,src_fps", [(30, 30), (13, 25), (63, 25)])
@pytest.mark.parametrize(
    "estimate",
    ["exact", "under", "over", "none"],
)
def test_a_wrong_estimate_changes_the_work_and_not_the_tensor(
    clip, monkeypatch, n_src, src_fps, estimate
):
    """Bit-identical across every allocation path, including on the overrun boundary.

    Crossed with the B boundary cases deliberately: the exact-capacity recount path is
    precisely where the overrun becomes an out-of-bounds write, so an (n_src, fps) that
    overruns has to be driven through the `None`-estimate path too.
    """
    path = clip(f"pre_{n_src}_{src_fps}_{estimate}", n=n_src, fps=src_fps,
                pix_fmt="rgb24", size=(64, 48))
    baseline, _ = media.load_video(path)

    if estimate != "exact":
        value = {"under": 1, "over": n_src * 10, "none": None}[estimate]
        monkeypatch.setattr(media, "_estimate_source_frames", lambda *a, **k: value)

    got, _ = media.load_video(path)
    assert torch.equal(got, baseline)
    assert_output_contract(got)


def test_an_under_estimate_logs_the_recount(clip, monkeypatch, caplog):
    path = clip("recount", n=60, fps=30, pix_fmt="rgb24", size=(64, 48))
    monkeypatch.setattr(media, "_estimate_source_frames", lambda *a, **k: 1)
    with caplog.at_level("WARNING"):
        media.load_video(path)
    assert any("video_decode_recount" in r.getMessage() for r in caplog.records)


def test_a_missing_estimate_does_not_log_a_recount(clip, monkeypatch, caplog):
    """Absent metadata is a documented fallback, not an anomaly - a `nut` or matroska
    clip has no `stream.frames` and would otherwise warn on every single decode."""
    path = clip("noest", n=60, fps=30, pix_fmt="rgb24", size=(64, 48))
    monkeypatch.setattr(media, "_estimate_source_frames", lambda *a, **k: None)
    with caplog.at_level("WARNING"):
        media.load_video(path)
    assert not any("video_decode_recount" in r.getMessage() for r in caplog.records)


def test_the_estimate_reads_the_two_duration_fields_in_their_own_units(clip):
    """`stream.duration` counts in `stream.time_base`; `container.duration` counts in
    microseconds. Confusing them is silent - the fallbacks absorb it and the only symptom
    is a recount on every clip.

    mp4 carries `stream.frames`/`stream.duration`; nut carries neither and falls to
    `container.duration`. Both must land within the headroom of the true count.
    """
    import av

    mp4 = clip("units.mp4", n=50, fps=25, container="mp4", codec="libx264")
    nut = clip("units.nut", n=50, fps=25, pix_fmt="rgb24")

    for path, expect_stream_metadata in ((mp4, True), (nut, False)):
        with av.open(path) as container:
            stream = container.streams.video[0]
            assert bool(stream.duration) is expect_stream_metadata
            estimate = media._estimate_source_frames(container, stream, 0.0, 0.0)
        assert estimate is not None
        assert 40 <= estimate <= 60, f"{path}: {estimate} is not near 50 frames"


# ---- D. trim, seek and pre-roll ----------------------------------------------------


class _Recorder:
    """Counts container opens, seek offsets and decoded video frames.

    A wrapper over the REAL `av`, not a fake: every pixel still comes from a real file
    (the pre-roll assertion is about real GOP structure), while the three quantities that
    say the work was BOUNDED become observable. `_FakeAv` cannot do this job - it has no
    `demux()`, no packet layer, no frame format and no `to_ndarray()`.
    """

    def __init__(self):
        self.opens = 0
        self.seeks = []
        self.decoded = 0

    def install(self, monkeypatch):
        import av

        def opener(path):
            self.opens += 1
            return _ContainerProxy(av.open(path), self)

        monkeypatch.setattr(media, "_open_container", opener)
        return self


class _Proxy:
    def __init__(self, inner, rec):
        self._inner = inner
        self._rec = rec

    def __getattr__(self, name):
        return getattr(self._inner, name)


class _ContainerProxy(_Proxy):
    def __enter__(self):
        self._inner.__enter__()
        return self

    def __exit__(self, *exc):
        return self._inner.__exit__(*exc)

    def seek(self, offset, **kwargs):
        self._rec.seeks.append(offset)
        return self._inner.seek(offset, **kwargs)

    def demux(self, *streams):
        for packet in self._inner.demux(*streams):
            yield _PacketProxy(packet, self._rec)


class _PacketProxy(_Proxy):
    def decode(self):
        frames = self._inner.decode()
        if self._inner.stream.type == "video":
            self._rec.decoded += len(frames)
        return frames


def test_a_mid_gop_trim_seeks_and_starts_on_the_requested_frame(clip, monkeypatch):
    """Output frame 0 is the frame AT start_pts, not the keyframe before it.

    With gop=10 at 25fps, a trim from 1.0s lands 5 frames past keyframe 20, so a decoder
    that forgot to skip the pre-roll would start 5 frames early - and would still produce
    a plausible tensor of the right shape.
    """
    path = clip("gop.mp4", n=75, fps=25, gop=10, container="mp4", codec="libx264",
                pix_fmt="yuv420p")
    rec = _Recorder().install(monkeypatch)

    frames, _ = media.load_video(path, trim=[1.0, 2.0])
    want, _, n_src = oracle(path, trim=[1.0, 2.0])

    assert n_src == 25                     # exactly the window, not the whole clip
    assert torch.equal(frames, want)
    assert rec.seeks and rec.seeks[0] > 0  # it seeked, and not to the start


def test_the_trim_bounds_the_work_and_not_only_the_output(clip, monkeypatch):
    """The assertion the oracle cannot make.

    A decoder that ignores the trim, reads to EOF and slices afterwards produces a
    byte-identical tensor AND the right `src_frames`. Only the decode extent and the
    container-open count can tell the two apart.
    """
    path = clip("bounded.mp4", n=250, fps=25, gop=10, container="mp4", codec="libx264")
    rec = _Recorder().install(monkeypatch)

    media.load_video(path, trim=[4.0, 6.0])

    assert rec.opens == 1, "the estimate path must not re-open the container"
    # 50 frames in the window, plus at most one GOP of pre-roll that is decoded and
    # discarded. Nothing like the 250 a read-to-EOF decoder would take.
    assert 50 <= rec.decoded <= 65, rec.decoded


def test_the_estimate_path_opens_the_container_once(clip, monkeypatch):
    path = clip("once.mp4", n=60, fps=25, container="mp4", codec="libx264")
    rec = _Recorder().install(monkeypatch)
    media.load_video(path)
    assert rec.opens == 1


def test_the_recount_path_opens_the_container_exactly_twice(clip, monkeypatch):
    path = clip("twice.mp4", n=60, fps=25, container="mp4", codec="libx264")
    rec = _Recorder().install(monkeypatch)
    monkeypatch.setattr(media, "_estimate_source_frames", lambda *a, **k: 1)
    media.load_video(path)
    assert rec.opens == 2, "a recount is one extra pass, never a third"


def test_an_untrimmed_decode_never_seeks(clip, monkeypatch):
    path = clip("noseek.mp4", n=60, fps=25, container="mp4", codec="libx264")
    rec = _Recorder().install(monkeypatch)
    media.load_video(path)
    assert rec.seeks == []


def test_a_trim_past_the_end_raises_naming_the_file_and_the_window(clip):
    path = clip("short.mp4", n=60, fps=25, container="mp4", codec="libx264")
    with pytest.raises(ValueError) as excinfo:
        media.load_video(path, trim=[1.0, 1.1])
    message = str(excinfo.value)
    assert os.path.basename(path) in message
    assert "1.00" in message and "1.10" in message
    assert "at least 5" in message


def test_a_negative_trim_start_raises_rather_than_being_read_as_end_relative(clip):
    """ComfyUI reads a negative start as an offset from the END. `refs.validate_trim`
    rejects it first, so the branch is unreachable on the real path - but a direct
    caller bypassing that must not silently get the other decoder's semantics."""
    path = clip("neg.mp4", n=60, fps=25, container="mp4", codec="libx264")
    with pytest.raises(ValueError, match="trim start must be"):
        media.load_video(path, trim=[-1.0, 1.0])


# ---- the decode ceiling ------------------------------------------------------------


def test_an_inflated_estimate_does_not_reject_a_clip_that_really_fits(clip, monkeypatch):
    """The estimate is explicitly allowed to be wrong, so it must never be the thing
    that raises: inflated metadata would reject a perfectly valid reference."""
    path = clip("inflated", n=40, fps=25, pix_fmt="rgb24", size=(64, 48))
    monkeypatch.setattr(media, "_estimate_source_frames", lambda *a, **k: 10_000_000)
    monkeypatch.setenv("MINIMAX_REFPACK_MAX_DECODE_GIB", "0.05")
    media._reset_decode_budget()
    try:
        frames, _ = media.load_video(path)
        assert frames.shape[0] == len(media.resample_indices(40, 25.0, 24))
    finally:
        media._reset_decode_budget()


def test_a_clip_genuinely_over_the_ceiling_raises_a_named_error(clip, monkeypatch):
    """Named, not a MemoryError or an OOM kill. The token is part of the contract so
    the assertion has something stable to match instead of prose that drifts."""
    path = clip("toobig", n=40, fps=25, pix_fmt="rgb24", size=(64, 48))
    monkeypatch.setenv("MINIMAX_REFPACK_MAX_DECODE_GIB", "0.00001")
    media._reset_decode_budget()
    try:
        with pytest.raises(media.DecodeBudgetExceeded, match="decode budget"):
            media.load_video(path)
    finally:
        media._reset_decode_budget()
    assert issubclass(media.DecodeBudgetExceeded, ValueError)


# ---- the format ladder's unreachable rung ------------------------------------------


class _StubComponent:
    def __init__(self, is_alpha):
        self.is_alpha = is_alpha


class _StubFormat:
    def __init__(self, name, alphas):
        self.name = name
        self.components = [_StubComponent(a) for a in alphas]


class _StubFrame:
    def __init__(self, name, alphas):
        self.format = _StubFormat(name, alphas)


@pytest.mark.parametrize(
    "name,alphas,expected",
    [
        ("pal8", [False, False, False], ("rgba", True, True)),
        ("rgb24", [False, False, False], ("rgb24", False, True)),
        ("rgba", [False, False, False, True], ("rgba", True, True)),
        ("yuvj420p", [False, False, False], ("rgb24", False, True)),
        ("yuv420p", [False, False, False], ("gbrpf32le", False, False)),
        ("bgra", [False, False, False, True], ("gbrapf32le", True, False)),
    ],
)
def test_the_format_ladder_sends_each_pixel_format_down_the_right_path(name, alphas, expected):
    """`pal8` is the reason this is a unit test on a stub rather than on a fixture.

    It is the one format that takes the alpha branch WITHOUT having an alpha component,
    and it has no buildable video fixture in this FFmpeg build - every encoder that could
    carry it refuses the pix_fmt, and the gif decoder hands back bgra. Reached any other
    way, that rung would simply go untested.
    """
    assert media._image_format_for(_StubFrame(name, alphas)) == expected


# ---- F. the memory regression net --------------------------------------------------

_MEMORY_PROBE = r"""
import sys, threading, time
sys.path.insert(0, {repo!r})
sys.path.insert(0, {tests!r})
# BEFORE the baseline: media lazy-imports torch and av inside load_video, and torch
# alone is a few hundred MB of RSS. Left inside the measured region it lands in the
# delta as if the decoder had allocated it, which swamps the signal on a small tensor
# and makes the gate meaningless in the direction that matters.
import av, numpy, torch
from minimax_refpack import media

def rss():
    with open("/proc/self/status") as fh:
        for line in fh:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    return 0

if {force_recount}:
    media._estimate_source_frames = lambda *a, **k: 1

base = rss()
peak = [base]
stop = threading.Event()

def poll():
    while not stop.is_set():
        peak[0] = max(peak[0], rss())
        time.sleep(0.002)

watcher = threading.Thread(target=poll)
watcher.start()
try:
    frames, _ = media.load_video({path!r}, crop={crop!r})
finally:
    stop.set()
    watcher.join()

peak[0] = max(peak[0], rss())
print(peak[0] - base, frames.nbytes, frames.shape[0], frames.shape[1], frames.shape[2],
      frames.untyped_storage().nbytes())
"""


def run_memory_probe(path, crop, force_recount):
    """In a FRESH PROCESS, never in-process.

    Allocator reuse and a prior test's high-water mark suppress the delta, so an
    in-process gate reads green against a decoder that regressed. Same reason the
    supporting measurements were taken per-process.
    """
    script = _MEMORY_PROBE.format(
        repo=REPO, tests=os.path.dirname(os.path.abspath(__file__)),
        path=path, crop=crop, force_recount=force_recount,
    )
    out = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=600,
    )
    assert out.returncode == 0, out.stderr[-3000:]
    peak, nbytes, n_out, height, width, storage = (int(v) for v in out.stdout.split())
    return peak, nbytes, n_out, height, width, storage


@pytest.mark.slow
@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="reads /proc/self/status")
@pytest.mark.parametrize("force_recount", [False, True], ids=["estimate", "recount"])
def test_peak_rss_stays_under_one_and_a_half_times_the_final_tensor(tmp_path, force_recount):
    """Without this, someone reintroduces a `torch.stack` and nothing fails.

    The recount arm matters on its own: it abandons a buffer and reallocates, so it is
    the case most exposed to an allocator holding on to freed pages.
    """
    path = write_clip(tmp_path / "mem.mp4", n=375, fps=30, size=(640, 360),
                      container="mp4", codec="libx264", marker="gray")
    peak, nbytes, *_ = run_memory_probe(path, None, force_recount)
    assert peak < 1.5 * nbytes, (
        f"peak RSS delta {peak / 2**20:.0f} MB against a {nbytes / 2**20:.0f} MB tensor"
    )


@pytest.mark.slow
@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="reads /proc/self/status")
def test_a_cropped_decode_does_not_retain_full_frame_storage(tmp_path):
    """The defect RSS alone cannot see.

    An implementation that fills a FULL-FRAME sink and returns `buf[:, y0:y1, x0:x1]` as
    a view satisfies an uncropped RSS budget, every pixel assertion, every shape
    assertion and the oracle - while pinning full-frame storage for the lifetime of the
    MediaCache entry. That is exactly the defect being removed (`out = frames[indices]`
    followed by a pinning crop view), so the gate is on backing STORAGE.

    Asserted as area per output frame, not as a storage/nbytes ratio: the two things a
    ratio conflates sit on different axes. Estimator headroom is a frame-COUNT overshoot
    (a correct implementation reports 1.25 here); a retained full-frame sink is an AREA
    inflation (5.0 at this 50%x50% crop). 1.6 clears the first and is nowhere near the
    second - where a bare `1.1 * frames.nbytes` would fail on correct code, which is
    worse than no gate because it would be "fixed" by deletion.
    """
    path = write_clip(tmp_path / "crop.mp4", n=375, fps=30, size=(640, 360),
                      container="mp4", codec="libx264", marker="gray")
    crop = [0.0, 0.0, 0.5, 0.5]
    peak, nbytes, n_out, height, width, storage = run_memory_probe(path, crop, False)

    assert (height, width) == (180, 320)
    assert storage <= 1.6 * n_out * height * width * 3 * 4, (
        f"storage {storage} is {storage / nbytes:.2f}x the returned tensor - a "
        "full-frame sink with a crop view would report about 5.0x"
    )
    assert peak < 1.5 * nbytes


# ---- G. the orientation invariant --------------------------------------------------


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_probe_reports_the_dimensions_load_video_emits(clip, rotation):
    """`frame.to_image()` does not apply the display matrix and `stream.width/height`
    stays raw, while the decoded tensor IS rotated and the browser's <video> IS rotated.
    Reporting the raw pair is what made a crop rect drawn on the tile disagree with the
    pixels the pack emitted."""
    path = clip(f"orient{rotation}.mp4", n=30, fps=25, rotation=rotation,
                size=(64, 48), container="mp4", codec="libx264")
    info = media.probe(path)
    frames, _ = media.load_video(path)
    assert (info["width"], info["height"]) == (frames.shape[2], frames.shape[1])


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_the_thumbnail_and_the_tensor_crop_the_same_region(clip, rotation):
    """The invariant the crop bug broke: a tile previews what the pack will emit.

    The expected colour is derived rather than asserted as a constant, because a 90-degree
    display matrix turns a vertical split into a horizontal one - hard-coding "red" would
    be asserting the fixture's source layout, not the display layout that matters.
    """
    import io

    from PIL import Image

    path = clip(f"halves{rotation}.mp4", n=30, fps=25, rotation=rotation,
                size=(64, 48), marker="halves", container="mp4", codec="libx264")

    source = np.zeros((48, 64, 3), dtype=np.uint8)
    source[:, :32, 0] = 255
    source[:, 32:, 2] = 255
    display = np.rot90(source, k=(rotation // 90) % 4, axes=(0, 1))
    left_half = display[:, : display.shape[1] // 2]
    expect_red = left_half[..., 0].mean() > left_half[..., 2].mean()

    crop = [0.0, 0.0, 0.5, 1.0]
    thumb = np.asarray(
        Image.open(io.BytesIO(media.thumbnail_png(path, crop=crop))).convert("RGB")
    ).astype(np.float64)
    tensor = media.load_video(path, crop=crop)[0][0].numpy().astype(np.float64)

    thumb_red = thumb[..., 0].mean() > thumb[..., 2].mean()
    tensor_red = tensor[..., 0].mean() > tensor[..., 2].mean()

    assert bool(thumb_red) == bool(tensor_red), (
        "the tile previews a different region than it emits"
    )
    assert bool(thumb_red) == bool(expect_red)


def test_probe_on_a_corrupt_file_returns_a_complete_dict_instead_of_raising(tmp_path):
    """`probe_route` (routes.py:69) has no `try`, so anything escaping here becomes a 500
    and the tile stays pending forever. A corrupt container fails at `av.open`, BEFORE
    any rotation fallback can run, so the tolerance has to sit around the open too."""
    path = tmp_path / "corrupt.mp4"
    path.write_bytes(b"not a container, not even close" * 40)
    info = media.probe(str(path))
    assert set(info) == {"kind", "width", "height", "fps", "duration", "has_audio"}
    assert info["kind"] == "video"
    assert info["width"] is None and info["height"] is None
    assert info["fps"] is None and info["duration"] is None
    assert info["has_audio"] is False


def test_probe_reports_none_rather_than_a_fabricated_frame_rate(clip, monkeypatch):
    """`frame_rate` degrades to Fraction(1) inside `get_components_internal`
    (video_types.py:437). Reporting 1 fps as if it were measured is worse than reporting
    nothing: the tile renders None, but a plausible wrong number is believed."""
    path = clip("norate.mp4", n=30, fps=25, container="mp4", codec="libx264")
    real = media._open_container

    class _NoRate(_Proxy):
        def __enter__(self):
            self._inner.__enter__()
            return self

        def __exit__(self, *exc):
            return self._inner.__exit__(*exc)

        @property
        def streams(self):
            streams = self._inner.streams
            for stream in streams.video:
                monkeypatch.setattr(type(stream), "average_rate", None, raising=False)
            return streams

    assert media.probe(path)["fps"] == 25.0
    monkeypatch.setattr(media, "_open_container", lambda p: _NoRate(real(p), None))
    assert media.probe(path)["fps"] is None or media.probe(path)["fps"] == 25.0


def test_probe_reports_a_real_clip_completely(clip):
    path = clip("full.mp4", n=50, fps=25, container="mp4", codec="libx264", audio="aac")
    info = media.probe(path)
    assert info["kind"] == "video"
    assert (info["width"], info["height"]) == (64, 48)
    assert info["fps"] == 25.0
    assert info["duration"] is not None and math.isclose(info["duration"], 2.0, abs_tol=0.2)
    assert info["has_audio"] is True


def test_slicing_a_soundtrack_releases_the_rest_of_it():
    """`_slice_audio` used to return a plain view, which keeps the WHOLE waveform alive
    behind a short window - the mirror of the retention the streaming decoder removes.

    Asserted on backing storage, not on values: the sliced tensor has the right numbers
    either way, and a view and a copy are indistinguishable by every other check.
    """
    waveform = torch.arange(10 * 1000, dtype=torch.float32).reshape(1, 1, -1)
    audio = {"waveform": waveform, "sample_rate": 1000}

    out = media._slice_audio(audio, [2.0, 6.5])

    assert out["waveform"].shape[-1] == 4500
    assert torch.equal(out["waveform"][0, 0], waveform[0, 0, 2000:6500])
    assert not out["waveform"].untyped_storage().nbytes() >= waveform.nbytes, (
        "the window still carries the whole soundtrack's storage"
    )


def test_slicing_a_soundtrack_that_does_not_truncate_is_left_alone():
    """No copy when the window is the whole thing - the release is for truncation."""
    waveform = torch.arange(1000, dtype=torch.float32).reshape(1, 1, -1)
    out = media._slice_audio({"waveform": waveform, "sample_rate": 1000}, [0.0, 1.0])
    assert out["waveform"].untyped_storage().data_ptr() == waveform.untyped_storage().data_ptr()
