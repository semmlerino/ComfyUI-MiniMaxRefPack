"""The VLM's copy streamed without the 24 fps stack (`vlm_reference_mp4`) against the
copy a node build sends (`encode_reference_mp4` of `load_video`).

Equality is on the mp4 BYTES, not on decoded pixels: a frozen off-pod prompt is only
the prompt a pod would write if the provider is handed the identical file, and a
pixel tolerance would let a different frame selection or conversion through.
"""

from __future__ import annotations

import os
import sys
from fractions import Fraction

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))

from conftest import CLIP_FORMATS  # noqa: E402

from minimax_refpack import media, prompt, refs, wired  # noqa: E402


def full(path, crop=None, trim=None, audio=True, **kwargs):
    frames, sound = media.load_video(path, 24, crop=crop, trim=trim)
    data, muxed = media.encode_reference_mp4(frames, 24, sound if audio else None, **kwargs)
    return data, muxed, audio and sound is not None


def streamed(path, crop=None, trim=None, audio=True, **kwargs):
    return media.vlm_reference_mp4(path, crop=crop, trim=trim, audio=audio, **kwargs)


def _refuse(name):
    def refuse(*args, **kwargs):
        raise AssertionError(f"called {name}")

    return refuse


def resized_clip(path, segments, fps=24):
    """One h264 stream whose frame size changes: `segments` is [((w, h), frames), ...],
    each encoded on its own and muxed back to back, as two clips joined with `-c copy`."""
    import av

    path = f"{path}.mkv"
    out = av.open(path, "w", format="matroska")
    stream = out.add_stream("libx264", rate=fps)
    stream.width, stream.height = segments[0][0]
    stream.pix_fmt = "yuv420p"
    start = 0
    for seg, ((w, h), n) in enumerate(segments):
        encoder = av.CodecContext.create("libx264", "w")
        encoder.width, encoder.height, encoder.pix_fmt = w, h, "yuv420p"
        encoder.time_base = Fraction(1, fps)
        encoder.options = {"bf": "0"}
        packets = []
        for i in range(n):
            a = np.full((h, w, 3), (seg * 90 + i * 3) % 256, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(a, format="rgb24")
            frame.pts = i
            packets.extend(encoder.encode(frame))
        packets.extend(encoder.encode(None))
        for packet in packets:
            packet.pts += start
            packet.dts += start
            packet.time_base = encoder.time_base
            packet.stream = stream
            out.mux(packet)
        start += n
    out.close()
    return path


def _counting(monkeypatch, owner, name):
    calls = []
    real = getattr(owner, name)

    def counted(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(owner, name, counted)
    return calls


@pytest.fixture
def budget(monkeypatch):
    """`budget(bytes)`: the decode budget, restored after the test."""

    def set_budget(nbytes):
        monkeypatch.setenv(media._DECODE_BUDGET_ENV, repr(nbytes / 2**30))
        media._reset_decode_budget()

    yield set_budget
    monkeypatch.delenv(media._DECODE_BUDGET_ENV, raising=False)
    media._reset_decode_budget()


@pytest.fixture
def same(monkeypatch):
    """`same(path, **kwargs)`: the streamed copy equals the full path's, and without the
    fallback, which reaches the full path itself and would make them equal trivially."""

    def check(path, **kwargs):
        want = full(path, **kwargs)
        with monkeypatch.context() as patched:
            patched.setattr(media, "load_video", _refuse("load_video"))
            got = streamed(path, **kwargs)
        assert got == want
        return got

    return check


# ---- byte identity -----------------------------------------------------------------


@pytest.mark.parametrize("pix_fmt", sorted(CLIP_FORMATS))
@pytest.mark.parametrize("size", [(64, 48), (66, 34)], ids=["w64", "w66_unaligned"])
def test_identical_across_the_format_ladder(clip, same, pix_fmt, size):
    path = clip(f"c_{pix_fmt}_{size[0]}", n=60, fps=30, pix_fmt=pix_fmt, size=size)
    same(path)


@pytest.mark.parametrize("fps", [8, 12, 24, 25, 30, 60])
def test_identical_at_every_source_rate(clip, same, fps):
    # Below 24 fps one source frame fills several slots; above it, slots skip frames.
    path = clip(f"r{fps}", n=3 * fps, fps=fps)
    same(path)


@pytest.mark.parametrize("n", [5, 8, 13, 14, 15, 16])
def test_identical_around_the_short_clip_rule(clip, same, n):
    # 13 slots and fewer go out whole at 24 fps; 14 is the first 8 fps resample.
    path = clip(f"n{n}", n=n, fps=24)
    same(path)


@pytest.mark.parametrize("rotation", [90, 270])
def test_identical_on_a_rotated_clip(clip, same, rotation):
    path = clip(f"rot{rotation}", n=60, fps=30, size=(66, 34), rotation=rotation)
    same(path)


def test_identical_when_downscaled(clip, same):
    path = clip("big", n=60, fps=30, size=(66, 34))
    same(path, long_edge=32)


def test_identical_on_a_cropped_and_trimmed_window(clip, same):
    path = clip("window", n=120, fps=30, size=(66, 34), gop=10, container="mp4", codec="libx264")
    crop, trim = [0.1, 0.2, 0.55, 0.6], [1.0, 3.0]
    same(path, crop=crop, trim=trim)


@pytest.mark.parametrize("audio", [True, False])
def test_identical_with_and_without_the_soundtrack(clip, same, audio):
    path = clip("sound", n=72, fps=24, audio="aac")
    got = same(path, audio=audio)
    assert got[1] is audio  # muxed exactly when asked for


def test_a_clip_without_sound_reports_it_unread(clip, same):
    path = clip("silent", n=48, fps=24)
    assert same(path)[2] is False


def test_identical_when_the_soundtrack_fails_after_the_header(clip, same, monkeypatch):
    def boom(container, stream, audio):
        raise RuntimeError("synthetic AAC encode failure")

    monkeypatch.setattr(media, "_encode_audio", boom)
    path = clip("retry", n=72, fps=24, audio="aac")
    # The retry re-encodes the kept frames, which the first attempt stamped with pts.
    assert same(path)[1] is False


# ---- what the stream never does ------------------------------------------------------


def test_the_stream_never_builds_the_stack(clip, monkeypatch):
    path = clip("nostack", n=60, fps=30)
    want = full(path)

    def no_stack(*args, **kwargs):
        raise AssertionError("built a 24 fps stack")

    monkeypatch.setattr(media, "_FrameSink", no_stack)
    assert streamed(path) == want


def test_the_stream_keeps_only_the_selected_slots(clip, monkeypatch):
    sinks = []
    real = media._VlmSink.build.__func__

    def spy(cls, *args):
        sink, box = real(cls, *args)
        sinks.append(sink)
        return sink, box

    monkeypatch.setattr(media._VlmSink, "build", classmethod(spy))
    path = clip("kept", n=300, fps=30)  # 10 s: 240 slots, 80 of them selected
    streamed(path)
    (sink,) = sinks
    selected = set(media.resample_indices(240, 24, 8))
    assert selected <= set(sink.kept)
    assert len(sink.kept) <= len(selected) + 14  # plus the short-clip prefix


def test_a_slot_the_stream_did_not_keep_falls_back_to_the_full_path(clip, monkeypatch):
    path = clip("fallback", n=60, fps=30)
    want = full(path)
    monkeypatch.setattr(media._VlmSink, "_keeps", lambda self, slot: slot % 2 == 1)
    assert streamed(path) == want


def test_the_stream_converts_only_frames_a_kept_slot_reads(clip, monkeypatch):
    path = clip("converted", n=300, fps=30)  # 240 slots, 80 of them selected
    calls = _counting(monkeypatch, media._FrameConverter, "convert")
    full(path)
    whole = len(calls)
    calls.clear()
    streamed(path)
    assert whole == 240
    assert len(calls) <= 80 + 14  # the selected slots, plus the short-clip prefix


def test_above_24_fps_the_copy_is_the_full_paths(clip):
    # A slot repeats in a 30 fps selection, and one kept frame cannot be encoded twice.
    path = clip("fast", n=60, fps=24)
    assert streamed(path, target_fps=30) == full(path, target_fps=30)


# ---- the full path's failures, unchanged ---------------------------------------------


def test_a_too_short_trim_fails_with_the_full_paths_message(clip):
    path = clip("short", n=120, fps=30, gop=10, container="mp4", codec="libx264")
    with pytest.raises(ValueError) as want:
        full(path, trim=[1.0, 1.1])
    with pytest.raises(ValueError) as got:
        streamed(path, trim=[1.0, 1.1])
    assert str(got.value) == str(want.value)


def test_a_clip_over_the_decode_budget_fails_as_on_a_pod(clip, budget):
    path = clip("budget", n=60, fps=30, size=(66, 34))
    budget(1024)
    with pytest.raises(media.DecodeBudgetExceeded) as want:
        full(path)
    with pytest.raises(media.DecodeBudgetExceeded) as got:
        streamed(path)
    assert str(got.value) == str(want.value)


def test_a_too_short_trim_over_the_budget_fails_on_the_budget(clip, budget):
    # The full path's exact pass raises the budget before the five-frame minimum.
    path = clip("short_budget", n=120, fps=30, gop=10, container="mp4", codec="libx264")
    budget(1024)
    with pytest.raises(media.DecodeBudgetExceeded) as want:
        full(path, trim=[1.0, 1.1])
    with pytest.raises(media.DecodeBudgetExceeded) as got:
        streamed(path, trim=[1.0, 1.1])
    assert str(got.value) == str(want.value)


def test_the_stream_keeps_nothing_once_the_stack_would_bust_the_budget(
    clip, budget, monkeypatch
):
    path = clip("busting", n=300, fps=30, size=(66, 34))
    budget(40 * 66 * 34 * 12)  # a stack of 40 slots, of the clip's 240
    with pytest.raises(media.DecodeBudgetExceeded) as want:
        full(path)
    made = _counting(monkeypatch, media, "_vlm_video_frame")
    converted = _counting(monkeypatch, media._FrameConverter, "convert")
    with pytest.raises(media.DecodeBudgetExceeded) as got:
        streamed(path)
    assert str(got.value) == str(want.value)
    # The prefix and the selected slots up to slot 40, and no conversion after them.
    assert len(made) <= 14 + 9
    assert len(converted) <= 14 + 9


@pytest.mark.parametrize(
    "segments",
    [
        [((64, 48), 36), ((48, 32), 36)],
        # Frame 31 fills slot 31, which the 8 fps copy never reads.
        [((64, 48), 31), ((48, 32), 1), ((64, 48), 30)],
    ],
    ids=["second_half", "one_skipped_frame"],
)
def test_a_size_change_mid_stream_fails_as_on_a_pod(tmp_path, segments):
    path = resized_clip(tmp_path / "resized", segments)
    with pytest.raises(RuntimeError):
        full(path)
    with pytest.raises(RuntimeError, match="changed size mid-stream"):
        streamed(path)


def test_a_size_change_the_crop_still_fits_is_identical(tmp_path, same):
    # The 48x32 half still holds the 32x24 box cut from the 64x48 first frame.
    path = resized_clip(tmp_path / "resized", [((64, 48), 36), ((48, 32), 36)])
    same(path, crop=[0, 0, 0.5, 0.5])


def test_the_soundtrack_is_decoded_even_when_it_is_not_sent(clip, monkeypatch):
    import av

    def corrupt(self, raw):
        raise av.error.InvalidDataError(1094995529, "synthetic corrupt AAC packet")

    path = clip("corrupt_sound", n=72, fps=24, audio="aac")
    monkeypatch.setattr(media._AudioCollector, "feed", corrupt)
    with pytest.raises(av.error.InvalidDataError):
        full(path, audio=False)
    with pytest.raises(av.error.InvalidDataError):
        streamed(path, audio=False)


# ---- the cache's routing ---------------------------------------------------------------


def test_a_vlm_only_cache_streams_once_and_never_decodes_the_stack(clip, monkeypatch):
    path = clip("cached", n=60, fps=30, audio="aac")
    want = full(path)
    monkeypatch.setattr(media, "load_video", _refuse("load_video"))
    calls = []
    real = media.vlm_reference_mp4

    def counted(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(media, "vlm_reference_mp4", counted)
    cache = media.MediaCache(vlm_only=True)
    assert cache.vlm_video(path) == want
    assert cache.vlm_video(path) == want
    assert len(calls) == 1


def test_a_vlm_only_cache_encodes_a_stack_it_already_holds(clip, monkeypatch):
    path = clip("held", n=60, fps=30)
    cache = media.MediaCache(vlm_only=True)
    cache.video(path, target_fps=24)
    monkeypatch.setattr(media, "vlm_reference_mp4", _refuse("vlm_reference_mp4"))
    assert cache.vlm_video(path) == full(path)


def test_a_build_cache_still_encodes_from_the_stack(clip, monkeypatch):
    path = clip("build", n=60, fps=30)
    monkeypatch.setattr(media, "vlm_reference_mp4", _refuse("vlm_reference_mp4"))
    assert media.MediaCache().vlm_video(path) == full(path)


def test_an_upstream_video_is_encoded_from_its_tensor_even_vlm_only(clip, monkeypatch):
    path = clip("upstream", n=60, fps=30)
    frames, _sound = media.load_video(path)
    cache = wired.WiredCache()
    cache.vlm_only = True
    cache.overrides["/nowhere/__upstream__/video_1.wired"] = (frames, None)
    monkeypatch.setattr(media, "vlm_reference_mp4", _refuse("vlm_reference_mp4"))
    data, muxed, read = cache.vlm_video("/nowhere/__upstream__/video_1.wired")
    assert (data, muxed, read) == (media.encode_reference_mp4(frames, 24)[0], False, False)


@pytest.mark.parametrize("sound", [True, False], ids=["soundtrack", "silent_clip"])
def test_the_prompt_payload_is_the_same_from_either_cache(clip, tmp_path, sound):
    clip("ref", n=72, fps=24, audio="aac" if sound else None)
    reference_set = refs.ReferenceSet([refs.Reference(kind="video", file="ref.mkv")])

    def content(cache):
        return prompt._build_content(reference_set, str(tmp_path), "idea", cache=cache)

    streamed_parts = content(media.MediaCache(vlm_only=True))
    assert streamed_parts == content(media.MediaCache())
    unread = "soundtrack could not be read" in streamed_parts[0]["text"]
    assert unread is not sound


# ---- the frame size without a decode ----------------------------------------------------


@pytest.mark.parametrize("rotation", [0, 90, 270])
@pytest.mark.parametrize("crop", [None, [0.1, 0.2, 0.55, 0.6]], ids=["whole", "cropped"])
@pytest.mark.parametrize("trim", [None, [0.5, 1.5]], ids=["untrimmed", "trimmed"])
def test_the_frame_size_is_the_decoded_one(clip, rotation, crop, trim):
    path = clip(f"size{rotation}", n=60, fps=30, size=(66, 34), rotation=rotation)
    frames, _sound = media.load_video(path, crop=crop, trim=trim)
    assert media.video_frame_size(path, crop, trim) == (frames.shape[2], frames.shape[1])


def test_the_frame_size_is_the_windows_own_after_a_size_change(tmp_path):
    path = resized_clip(tmp_path / "resized", [((64, 48), 36), ((48, 32), 36)])
    trim = [2.0, 2.9]  # inside the 48x32 half, which starts at 1.5 s
    frames, _sound = media.load_video(path, trim=trim)
    assert media.video_frame_size(path, None, trim) == (frames.shape[2], frames.shape[1])
    assert media.video_frame_size(path, None, trim) == (48, 32)


def test_the_frame_size_of_an_empty_window_is_an_error(clip):
    path = clip("empty", n=30, fps=30)
    with pytest.raises(ValueError, match="decoded no frame in 5.00-6.00s"):
        media.video_frame_size(path, None, [5.0, 6.0])
