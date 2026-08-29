"""Real media fixtures for the decoder suites.

`write_clip` builds a genuine container on disk. The decoder tests are differential -
they compare the streaming decoder against a verbatim transcription of ComfyUI's own
frame loop - so the pixels have to come from a real demux/decode, not from a fake.

WHAT THE FORMAT MATRIX IS FOR. video_types.py:363 splits every frame down one of two
paths: `yuvj420p`, `yuvj422p`, `yuvj444p`, `rgb24`, `rgba` and `pal8` decode to
rgb24/rgba and get divided by 255; EVERYTHING ELSE decodes to gbrpf32le/gbrapf32le
with no division, and only that second group ever reaches `align_graph` (:381). Library
clips are plain `yuv420p`, so the float+align path is the one production takes - a
matrix built only from the 8-bit list would leave it entirely unexercised.

WHAT THIS BUILD CANNOT PRODUCE (measured against PyAV 18.1.0 / FFmpeg 8.1.2, and the
reason each unbuildable arm is covered another way):

  pal8 video      every encoder that could carry it refuses the pix_fmt (rawvideo,
                  msrle, smc, msvideo1 -> OSError 95 / avcodec_open2 22) and the gif
                  decoder hands back `bgra`, not `pal8`. pal8's only distinct behaviour
                  is that it forces the alpha branch without an alpha COMPONENT, so it
                  is pinned by a unit test on `media._image_format_for` instead.
  mpegts + pcm    reopens with zero audio streams. Every mpegts codec that does mux
                  decodes to fltp and passes the resampler through unchanged, so the
                  over-trim direction has no container fixture either (see
                  test_media_decode.py's head-trim unit test).

`nut` is deliberately in the table twice over: it is lossless AND metadata-poor
(`stream.frames == 0`, `stream.duration is None`), which is what exercises the
estimator's fallback rungs and `probe()`'s.
"""

import numpy as np
import pytest

# (container, codec, extension, lossless) per pixel format. Measured, not guessed - an
# encoder that silently substitutes a format would move a case onto the other branch of
# the ladder and every behavioural assertion would still pass, so the tests assert
# `frame.format.name` against the request.
CLIP_FORMATS = {
    # --- float path (gbrpf32le / gbrapf32le), the only one align_graph fires on ---
    "yuv420p": ("matroska", "libx264", "mkv", False),   # what the library actually is
    "gbrp": ("nut", "rawvideo", "nut", True),
    "bgra": ("nut", "rawvideo", "nut", True),           # + alpha drop
    # --- 8-bit path (rgb24 / rgba, then /255) ---
    "rgb24": ("nut", "rawvideo", "nut", True),
    "yuvj420p": ("matroska", "libx264", "mkv", False),
    "rgba": ("nut", "rawvideo", "nut", True),           # + alpha drop
}

LOSSLESS_FORMATS = tuple(k for k, v in CLIP_FORMATS.items() if v[3])

# Audio arms, measured. `rebases` is whether AudioResampler(format='fltp') rewrites the
# frame pts into 1/sample_rate - which is what makes ComfyUI's `pts * stream.time_base`
# under-trim on the mkv arms and pass through cleanly on the mp4 one.
AUDIO_FORMATS = {
    "aac": ("mp4", "m4a", False),           # control: tb already 1/44100, passthrough
    "pcm_s16le": ("matroska", "mkv", True),  # tb 1/1000, decodes s16  -> rebased
    "mp2": ("matroska", "mkv", True),        # tb 1/1000, decodes s16p -> rebased
}


_EXTENSIONS = {"matroska": "mkv", "mp4": "mp4", "nut": "nut", "mov": "mov", "avi": "avi"}


def frame_marker(i: int) -> tuple[int, int, int]:
    """The RGB a `marker='index'` clip paints on source frame `i`.

    The index rides in R (low byte) and G (high byte) so a decoded frame names the
    source frame it came from - which is what lets a selection test recover the whole
    slot->source mapping and compare it against `resample_indices` exactly. Only exact
    on a lossless format; see LOSSLESS_FORMATS.
    """
    return (i & 0xFF, (i >> 8) & 0xFF, 0x40)


def marker_index(pixel) -> int:
    """Inverse of `frame_marker`, from any [.., 3] pixel or frame in 0..1 or 0..255."""
    arr = np.asarray(pixel, dtype=np.float64)
    while arr.ndim > 1:
        arr = arr.mean(axis=0)
    scale = 255.0 if arr.max() <= 1.0 else 1.0
    r, g = int(round(arr[0] * scale)), int(round(arr[1] * scale))
    return (g << 8) | r


def write_clip(
    path,
    *,
    n=24,
    fps=25,
    size=(64, 48),
    container=None,
    codec=None,
    pix_fmt="yuv420p",
    rotation=0,
    audio=None,
    gop=None,
    marker="index",
    audio_seconds=None,
    sample_rate=44100,
):
    """Write a real clip and return its path as a str.

    `pix_fmt` selects the container/codec from CLIP_FORMATS unless `container`/`codec`
    override it. `marker` is 'index' (per-frame recoverable id, see `frame_marker`),
    'halves' (display-left red, display-right blue - for the orientation invariant) or
    'gray' (a flat ramp).

    `audio` is a codec name from AUDIO_FORMATS, or None. Its waveform is a ramp where
    sample j carries j/N, so a decoded sample names its own index and a head-trim test
    can say which sample survived. Values stay in the -1..1 a real waveform occupies;
    the raw-index form would be unsatisfiable at any sample rate.

    `rotation` writes a display matrix. PyAV canonicalises it into (-180, 180], so a
    caller asserting on it must compare `frame.rotation % 360`, not `== rotation`:
    measured 90 -> 90, 180 -> -180, 270 -> -90.
    """
    import av

    fmt_container, fmt_codec, ext, _lossless = CLIP_FORMATS.get(
        pix_fmt, ("matroska", "libx264", "mkv", False)
    )
    container = container or fmt_container
    codec = codec or fmt_codec
    # The extension follows the CONTAINER that was resolved, not the pix_fmt's default:
    # `write_clip(p, container="mp4")` on the default yuv420p would otherwise name an
    # mp4 file `p.mkv`. Harmless to PyAV, which sniffs content, and a foot-gun to read.
    ext = _EXTENSIONS.get(container, ext)
    path = str(path)
    if not path.endswith(f".{ext}") and "." not in str(path).rsplit("/", 1)[-1]:
        path = f"{path}.{ext}"

    w, h = size
    out = av.open(path, "w", format=container)
    try:
        # `g`/`bf` as encoder OPTIONS, not `codec_context.gop_size` after the fact:
        # measured, the attribute is overridden by libx264's own rate control and a
        # gop=10 request came back with keyframes every 4 packets. B-frames are off so
        # decode order is presentation order and a mid-GOP trim test can name the frame
        # it expects.
        # `sc_threshold=0` is load-bearing: a marker clip is a full-frame colour change,
        # which x264 reads as a scene cut and answers with an IDR on almost every frame.
        # Measured, g=10 alone gave keyframes every 4 packets; with it, exactly 0/10/20/30.
        options = {"g": str(gop), "bf": "0", "sc_threshold": "0"} if gop else {}
        stream = out.add_stream(codec, rate=fps, options=options)
        stream.width, stream.height = w, h
        stream.pix_fmt = pix_fmt
        if rotation:
            stream.set_display_rotation(rotation)

        sound = None
        if audio is not None:
            sound = _add_audio(out, audio, sample_rate)

        for i in range(n):
            frame = av.VideoFrame.from_ndarray(_paint(i, w, h, marker), format="rgb24")
            for packet in stream.encode(frame):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)

        if sound is not None:
            seconds = audio_seconds if audio_seconds is not None else n / fps
            _encode_audio(out, sound, seconds, sample_rate)
    finally:
        out.close()
    return path


def _paint(i, w, h, marker):
    a = np.zeros((h, w, 3), dtype=np.uint8)
    if marker == "halves":
        # Painted in DISPLAY space by the caller's reckoning: the clip is written
        # unrotated, so a 90-degree display matrix turns these columns into rows and
        # the orientation test is what checks they come back as columns again.
        a[:, : w // 2, 0] = 255
        a[:, w // 2 :, 2] = 255
        return a
    if marker == "gray":
        a[...] = (i * 7) % 256
        return a
    r, g, b = frame_marker(i)
    a[..., 0], a[..., 1], a[..., 2] = r, g, b
    return a


def _add_audio(out, codec, sample_rate):
    return out.add_stream(codec, rate=sample_rate, layout="mono")


def _encode_audio(out, stream, seconds, sample_rate):
    """A mono ramp: sample j carries j/N, so a decoded value names its own index."""
    import av

    from fractions import Fraction

    total = max(1, int(round(seconds * sample_rate)))
    ramp = (np.arange(total, dtype=np.float32) / total)[None, :]
    resampler = av.AudioResampler(
        format=stream.format, layout=stream.layout, rate=stream.rate
    )
    time_base = Fraction(1, sample_rate)
    chunk = 4096

    def mux(frames):
        for resampled in frames:
            for packet in stream.encode(resampled):
                out.mux(packet)

    for start in range(0, total, chunk):
        frame = av.AudioFrame.from_ndarray(
            np.ascontiguousarray(ramp[:, start : start + chunk]),
            format="fltp",
            layout="mono",
        )
        frame.sample_rate = sample_rate
        frame.pts = start
        frame.time_base = time_base
        mux(resampler.resample(frame))
    mux(resampler.resample(None))
    for packet in stream.encode(None):
        out.mux(packet)


@pytest.fixture
def clip(tmp_path):
    """`clip(name, **write_clip_kwargs)` -> path under the test's tmp_path."""

    def _make(name="clip", **kwargs):
        return write_clip(tmp_path / name, **kwargs)

    return _make
