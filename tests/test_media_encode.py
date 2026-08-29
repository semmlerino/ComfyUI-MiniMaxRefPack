"""Tests for minimax_refpack.media's MP4 encoder (`encode_reference_mp4`) and per-build
decode cache (`MediaCache`).

`av` is imported directly below, NOT through `pytest.importorskip`: the encoder's whole
job is turning already-decoded frames into bytes PyAV can decode back, so a suite that
skips itself when `av` is missing or broken would stay green through exactly the
regression it exists to catch.

There is no torch in this venv, so every frame stack here is a plain numpy array -
`_chunk_to_uint8`'s numpy branch is what all of these exercise; the torch fast path is
covered elsewhere. Frame stacks are gradient/random data rather than a solid color, so a
crop or downscale that silently produced garbage would still look wrong instead of
passing by accident.
"""

from __future__ import annotations

import io
import logging

import av
import numpy as np
import pytest

from minimax_refpack import logs, media


# ---- shared fixtures ------------------------------------------------------------


def _frames(n: int, height: int = 32, width: int = 32, seed: int = 0):
    """[n, H, W, 3] float32 in 0..1 - random per-pixel data, not a solid fill."""
    rng = np.random.default_rng(seed)
    return rng.random((n, height, width, 3), dtype=np.float32)


def _waveform(
    seconds: float, sample_rate: int = 44100, channels: int = 2
) -> dict[str, object]:
    """{"waveform": [1, C, L], "sample_rate"} - batch-first, the shape MediaCache.audio()
    and load_video's audio branch actually hand encode_reference_mp4."""
    n = int(round(seconds * sample_rate))
    t = np.linspace(0, seconds, n, endpoint=False, dtype=np.float32)
    tone = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    waveform = np.stack([tone] * channels, axis=0)  # [C, L]
    return {"waveform": waveform[None, ...], "sample_rate": sample_rate}


def _rate(stream: av.VideoStream) -> float:
    """`average_rate` is typed `Fraction | None` - the encoder always sets an explicit
    rate, so a None here would itself be a bug worth failing loudly on."""
    assert stream.average_rate is not None
    return float(stream.average_rate)


def _decode_container(
    data: bytes,
) -> tuple[
    list[av.VideoFrame], list[av.AudioFrame], av.VideoStream, av.AudioStream | None
]:
    """The real-decoder oracle every case below reads from: stream metadata alone
    (`streams.audio` non-empty) cannot tell a real soundtrack from a present-but-silent
    one, so every assertion here is made against actually-decoded frames.

    Returns (video_frames, audio_frames, video_stream, audio_stream_or_None).
    """
    with av.open(io.BytesIO(data), mode="r") as container:
        vstream = container.streams.video[0]
        astream = container.streams.audio[0] if container.streams.audio else None
        video_frames: list[av.VideoFrame] = []
        audio_frames: list[av.AudioFrame] = []
        frame_iter = (
            container.decode(vstream, astream)
            if astream is not None
            else container.decode(vstream)
        )
        for frame in frame_iter:
            if isinstance(frame, av.VideoFrame):
                video_frames.append(frame)
            elif isinstance(frame, av.AudioFrame):
                audio_frames.append(frame)
        return video_frames, audio_frames, vstream, astream


# ---- A. real-encoder round trip --------------------------------------------------


def test_real_encoder_round_trip_downscales_and_muxes_audio():
    # 1920x1080 so the 768px long-edge cap is actually exercised, not just declared.
    n_src, src_fps = 15, 24
    frames = _frames(n_src, height=1080, width=1920)
    expected_indices = media.resample_indices(n_src, src_fps, media.VLM_VIDEO_FPS)
    assert (
        len(expected_indices) >= media._MIN_VLM_FRAMES
    )  # sanity: normal path, no fallback
    video_seconds = len(expected_indices) / media.VLM_VIDEO_FPS
    audio = _waveform(video_seconds, channels=2)

    data, muxed = media.encode_reference_mp4(frames, src_fps=src_fps, audio=audio)

    video_frames, audio_frames, vstream, astream = _decode_container(data)
    assert muxed is True
    assert astream is not None
    assert len(audio_frames) > 0
    w, h = vstream.codec_context.width, vstream.codec_context.height
    assert max(w, h) <= media.VLM_VIDEO_LONG_EDGE
    assert w % 2 == 0 and w >= 2
    assert h % 2 == 0 and h >= 2
    assert _rate(vstream) == media.VLM_VIDEO_FPS
    assert len(video_frames) == len(expected_indices)


# ---- B. four real failure modes --------------------------------------------------


def test_five_frame_clip_stays_five_frames_not_two():
    """resample_indices(5, 24, 8) is exactly 2 - a clip that only just cleared
    load_video's >=5-frame minimum would otherwise reach the VLM as a 2-frame video.
    _MIN_VLM_FRAMES is the guard: below it, every source frame is kept and the clip is
    encoded at ITS OWN rate instead of being resampled down further."""
    n_src, src_fps = 5, 24
    assert len(media.resample_indices(n_src, src_fps, media.VLM_VIDEO_FPS)) == 2
    frames = _frames(n_src)

    data, muxed = media.encode_reference_mp4(frames, src_fps=src_fps)

    video_frames, _audio_frames, vstream, astream = _decode_container(data)
    assert len(video_frames) == n_src
    assert _rate(vstream) == src_fps  # source rate, not target_fps
    assert astream is None
    assert muxed is False


def test_one_pixel_input_yields_a_valid_encoder_not_an_einval():
    """The dimension rule is max(2, dim & ~1), not dim - dim % 2, precisely because
    _crop_box guarantees only that a box is never zero-width (a legal 1px crop). The
    subtraction form would build a 0-sized encoder here and avcodec_open2 would fail
    EINVAL; this asserts a real, decodable 2x2 stream comes out instead."""
    frames = _frames(5, height=1, width=1)

    data, muxed = media.encode_reference_mp4(frames, src_fps=24)

    video_frames, _audio_frames, vstream, _astream = _decode_container(data)
    assert vstream.codec_context.width == 2
    assert vstream.codec_context.height == 2
    assert len(video_frames) >= 1


def test_induced_audio_encode_failure_after_header_falls_back_to_video_only(
    monkeypatch,
):
    """Once the AAC stream is declared, a container cannot drop it if encoding then
    fails - the mp4 header is already written naming that stream, and adding one after
    the first mux() fails with 'Cannot rebase to zero time'. _AudioMuxFailed is the only
    honest recovery: throw the whole container away and rebuild video-only. Induced by
    monkeypatching _encode_audio itself, which only runs after the header is out."""

    def _boom(container, stream, audio):
        raise RuntimeError("synthetic AAC encode failure")

    monkeypatch.setattr(media, "_encode_audio", _boom)
    frames = _frames(8)
    audio = _waveform(1.0, channels=2)

    data, muxed = media.encode_reference_mp4(frames, src_fps=24, audio=audio)

    assert muxed is False
    video_frames, _audio_frames, _vstream, astream = _decode_container(data)
    assert astream is None
    assert len(video_frames) > 0


def test_unsupported_channel_count_never_opens_a_stream_before_the_header(monkeypatch):
    """6 channels is not in _LAYOUTS, so _open_audio_stream must return None BEFORE the
    first mux() runs - nothing has been muxed yet, so a video-only file is the honest
    answer without ever touching the _AudioMuxFailed rebuild path."""
    frames = _frames(8)
    audio = _waveform(1.0, channels=6)

    data, muxed = media.encode_reference_mp4(frames, src_fps=24, audio=audio)

    assert muxed is False
    _video_frames, _audio_frames, _vstream, astream = _decode_container(data)
    assert astream is None


def test_cache_methods_accept_none_crop_and_trim(monkeypatch):
    """`_cache_key` is `None if value is None else tuple(value)` precisely because a bare
    `tuple(None)` raises TypeError - and crop/trim=None is the COMMON path, not an edge
    case, so every MediaCache method must reach it without raising."""
    monkeypatch.setattr(media, "load_image", lambda path, crop=None, max_edge=0: "img")
    monkeypatch.setattr(
        media,
        "load_video",
        lambda path, target_fps=24, crop=None, trim=None: ("frames", None),
    )
    monkeypatch.setattr(
        media,
        "load_audio",
        lambda path, trim=None: {"waveform": "wf", "sample_rate": 1},
    )
    cache = media.MediaCache()

    assert cache.image("a.png", crop=None, max_edge=0) == "img"
    assert cache.video("a.mp4", crop=None, trim=None) == ("frames", None)
    assert cache.audio("a.wav", trim=None) == {"waveform": "wf", "sample_rate": 1}


# ---- C. muxed=True needs a real oracle -------------------------------------------


@pytest.mark.parametrize("channels", [1, 2])
def test_muxed_audio_is_really_decodable_not_just_present(channels):
    """An AAC stream can be present and carry nothing - which is exactly what a silently
    wrong [1,C,L] batch-first -> planar [C,L] conversion would produce, since PyAV's
    AudioFrame is not batch-first. Checking `streams.audio` alone would pass on an empty
    soundtrack; only decoding it back and counting real samples proves otherwise."""
    n_src, src_fps = 15, 24
    frames = _frames(n_src)
    expected_indices = media.resample_indices(n_src, src_fps, media.VLM_VIDEO_FPS)
    video_seconds = len(expected_indices) / media.VLM_VIDEO_FPS
    sample_rate = 44100
    audio = _waveform(video_seconds, sample_rate=sample_rate, channels=channels)

    data, muxed = media.encode_reference_mp4(frames, src_fps=src_fps, audio=audio)

    assert muxed is True
    video_frames, audio_frames, vstream, astream = _decode_container(data)
    assert astream is not None
    assert len(audio_frames) > 0
    total_samples = sum(f.samples for f in audio_frames)
    assert total_samples > 0
    assert audio_frames[0].to_ndarray().shape[0] == channels
    audio_duration = total_samples / sample_rate
    video_duration = len(video_frames) / _rate(vstream)
    # AAC frames are 1024 samples each, so the encoded tail rounds up to the next
    # 1024-sample boundary - allow one frame's worth of slack, not exact equality.
    assert abs(audio_duration - video_duration) < (1024 / sample_rate) + 0.05


# ---- D. cache-key isolation, misses and hits -------------------------------------


class _CountingLoader:
    """A stand-in for load_image/load_video/load_audio: records every call and returns a
    fresh object identity each time, so a hit vs. a miss is read off identity, not just a
    call count."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    def __call__(self, path: str, **kwargs: object) -> object:
        self.calls.append((path, kwargs))
        return object()


def _counting_loader() -> _CountingLoader:
    return _CountingLoader()


def test_cache_image_misses_on_differing_crop(monkeypatch):
    loader = _counting_loader()
    monkeypatch.setattr(media, "load_image", loader)
    cache = media.MediaCache()

    first = cache.image("a.png", crop=[0.0, 0.0, 0.5, 0.5])
    second = cache.image("a.png", crop=[0.0, 0.0, 1.0, 1.0])

    assert len(loader.calls) == 2
    assert first is not second


def test_cache_video_misses_on_differing_trim(monkeypatch):
    loader = _counting_loader()
    monkeypatch.setattr(media, "load_video", loader)
    cache = media.MediaCache()

    first = cache.video("clip.mp4", trim=[0.0, 1.0])
    second = cache.video("clip.mp4", trim=[1.0, 2.0])

    assert len(loader.calls) == 2
    assert first is not second


def test_cache_image_misses_on_differing_max_edge(monkeypatch):
    loader = _counting_loader()
    monkeypatch.setattr(media, "load_image", loader)
    cache = media.MediaCache()

    first = cache.image("a.png", max_edge=0)
    second = cache.image("a.png", max_edge=512)

    assert len(loader.calls) == 2
    assert first is not second


def test_cache_video_misses_on_differing_target_fps(monkeypatch):
    loader = _counting_loader()
    monkeypatch.setattr(media, "load_video", loader)
    cache = media.MediaCache()

    first = cache.video("clip.mp4", target_fps=24)
    second = cache.video("clip.mp4", target_fps=8)

    assert len(loader.calls) == 2
    assert first is not second


def test_cache_key_isolates_by_media_kind(monkeypatch):
    """Even for the identical path, the key's first element is the kind literal - an
    image, a video and an audio load of the "same" path must never collide."""
    img_loader = _counting_loader()
    vid_loader = _counting_loader()
    aud_loader = _counting_loader()
    monkeypatch.setattr(media, "load_image", img_loader)
    monkeypatch.setattr(media, "load_video", vid_loader)
    monkeypatch.setattr(media, "load_audio", aud_loader)
    cache = media.MediaCache()
    path = "same-name.bin"

    cache.image(path)
    cache.video(path)
    cache.audio(path)

    assert len(img_loader.calls) == 1
    assert len(vid_loader.calls) == 1
    assert len(aud_loader.calls) == 1


def test_cache_resolves_same_named_files_in_different_directories_apart(
    tmp_path, monkeypatch
):
    """Two references can legally share a basename in different directories - the key is
    built from the ABSOLUTE path, not the bare name, so they must never collide."""
    loader = _counting_loader()
    monkeypatch.setattr(media, "load_image", loader)
    cache = media.MediaCache()
    dir_a, dir_b = tmp_path / "a", tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()

    first = cache.image(str(dir_a / "same.png"))
    second = cache.image(str(dir_b / "same.png"))

    assert len(loader.calls) == 2
    assert first is not second


def test_cache_hit_loads_once_and_returns_the_same_object(monkeypatch):
    loader = _counting_loader()
    monkeypatch.setattr(media, "load_video", loader)
    cache = media.MediaCache()

    first = cache.video("clip.mp4", target_fps=24, crop=[0, 0, 1, 1], trim=[0, 1])
    second = cache.video("clip.mp4", target_fps=24, crop=[0, 0, 1, 1], trim=[0, 1])

    assert len(loader.calls) == 1
    assert first is second


def test_cache_emits_a_hit_or_miss_log_line(monkeypatch, caplog):
    monkeypatch.setattr(media, "load_image", lambda path, **kwargs: "img")
    cache = media.MediaCache()

    with caplog.at_level(logging.INFO, logger=logs.LOGGER_NAME):
        cache.image("dir/a.png")
        cache.image("dir/a.png")

    lines = [r.getMessage() for r in caplog.records if r.name == logs.LOGGER_NAME]
    cache_lines = [ln for ln in lines if "event=media_cache" in ln]
    assert len(cache_lines) == 2
    assert "kind=image" in cache_lines[0] and "hit=false" in cache_lines[0]
    assert "kind=image" in cache_lines[1] and "hit=true" in cache_lines[1]


# ---- E. the encoder this FFmpeg build actually has -------------------------------
# PyAV's own wheels bundle an FFmpeg with libx264 (av 18.1.0 ships ffmpeg 8.1.2 with it
# usable), but a PyAV linked against a distro or source FFmpeg built without it is real.
# It matters more since this change than before it: an untouched mp4 used to skip
# encoding altogether, so a bare add_stream failure would take away a path that worked.


def test_a_build_without_libx264_falls_back_instead_of_failing(monkeypatch, caplog):
    """mpeg4 is core FFmpeg, so the fallback list always ends somewhere that exists."""
    monkeypatch.setattr(media, "_VIDEO_ENCODERS", ("definitely-not-an-encoder", "mpeg4"))

    with caplog.at_level(logging.WARNING, logger=logs.LOGGER_NAME):
        data, muxed = media.encode_reference_mp4(_frames(15), src_fps=24)

    video_frames, _audio, vstream, _astream = _decode_container(data)
    assert len(video_frames) == len(media.resample_indices(15, 24, media.VLM_VIDEO_FPS))
    assert _rate(vstream) == media.VLM_VIDEO_FPS
    assert muxed is False

    line = "\n".join(r.getMessage() for r in caplog.records)
    assert "event=video_encoder_fallback" in line
    assert "encoder=mpeg4" in line
    # ...and it names what it could NOT use, so a pod log says why it degraded.
    assert "definitely-not-an-encoder" in line


def test_no_usable_encoder_at_all_raises_something_actionable(monkeypatch):
    """The video stream has no optional-degradation story the way the soundtrack does -
    there is no reference left without it - so this raises, and the message has to say
    what to do rather than surfacing PyAV's bare 'Unknown encoder'."""
    monkeypatch.setattr(media, "_VIDEO_ENCODERS", ("nope-one", "nope-two"))

    with pytest.raises(RuntimeError) as excinfo:
        media.encode_reference_mp4(_frames(15), src_fps=24)

    message = str(excinfo.value)
    assert "nope-one" in message and "nope-two" in message
    assert "libx264" in message

