"""Tests for minimax_refpack.media - the pure resample math, load_video against REAL
clips, and the two seams a real file cannot stand in for.

There is no `VideoFromFile` left to fake: the decoder is ours (raw PyAV, one streaming
pass), so every load_video test below drives it over a genuine container built by
`tests/conftest.py`'s `write_clip`. That is strictly stronger than the fake it replaces -
`marker='index'` on a LOSSLESS pixel format paints each source frame with its own index,
so a selection assertion recovers the exact slot->source mapping out of the emitted
pixels instead of trusting a stub's bookkeeping.

Two seams remain, for reasons a fixture cannot remove. `_audio_load_fn` reaches
`comfy_extras.nodes_audio`, which is not importable outside a ComfyUI process (verified:
a bare `import comfy_extras` raises ModuleNotFoundError in this venv). `_open_container`
is wrapped in the thumbnail seek tests for the reason `_FakeAv`'s docstring gives.
"""

import pytest

from minimax_refpack import media
from tests.conftest import marker_index


# ---- resample_indices -------------------------------------------------------


def test_identity_when_fps_matches():
    assert media.resample_indices(24, 24, 24) == list(range(24))


def test_downsample_30_to_24():
    idx = media.resample_indices(30, 30, 24)
    assert len(idx) == 24
    assert idx == sorted(idx)
    assert idx[0] == 0
    assert max(idx) <= 29


def test_downsample_60_to_24():
    idx = media.resample_indices(60, 60, 24)
    assert len(idx) == 24
    assert idx == sorted(idx)
    assert idx[0] == 0
    assert max(idx) <= 59


def test_upsample_12_to_24_duplicates_frames_instead_of_inventing_them():
    idx = media.resample_indices(12, 12, 24)
    assert len(idx) == 24
    assert idx == sorted(idx)
    assert max(idx) <= 11
    assert len(set(idx)) < len(idx)  # upsampling must repeat source frames


def test_single_source_frame():
    assert media.resample_indices(1, 24, 24) == [0]
    assert media.resample_indices(1, 30, 24) == [0]


def test_zero_source_frames():
    assert media.resample_indices(0, 30, 24) == []


def test_indices_never_leave_source_bounds():
    for n_src, src_fps in [(1, 1), (5, 7), (100, 23), (2, 240)]:
        idx = media.resample_indices(n_src, src_fps, 24)
        assert all(0 <= i < n_src for i in idx)


def test_zero_or_negative_fps_is_rejected():
    with pytest.raises(ValueError):
        media.resample_indices(10, 0, 24)
    with pytest.raises(ValueError):
        media.resample_indices(10, -5, 24)


# ---- load_video's >=5-frame guard and passthrough ---------------------------
# Every one of these drives the real decoder over a real container. `write_clip` gives
# exact control over both `n` and `fps`, which is what lets each expected frame count
# below be a single unambiguous number rather than a range.


def test_load_video_raises_when_resampled_clip_is_too_short(clip):
    # 3 frames at 24fps -> 3 frames at the 24fps target, under the 5-frame floor
    path = clip("clip_too_short", n=3, fps=24)

    with pytest.raises(ValueError) as exc:
        media.load_video(path)

    assert "clip_too_short.mkv" in str(exc.value)


def test_load_video_does_not_pad_short_clips_up_to_five(clip):
    # 4 frames is still under 5 - must raise, never silently duplicate up to 5. The
    # count in the message is what proves nothing was padded on the way to the check.
    path = clip("four", n=4, fps=24)

    with pytest.raises(ValueError) as exc:
        media.load_video(path)

    assert "has only 4 frame(s)" in str(exc.value)


def test_load_video_passes_through_resampled_frames_and_audio(clip):
    # 30 frames at 30fps is 1.0s, which is 24 frames at the 24fps target - a count the
    # source does not have, so a decoder that skipped the resample would emit 30.
    path = clip("sound", n=30, fps=30, audio="pcm_s16le")

    frames, audio = media.load_video(path, target_fps=24)

    assert frames.shape == (24, 48, 64, 3)  # write_clip's default size is (w=64, h=48)
    assert float(frames.min()) >= 0.0 and float(frames.max()) <= 1.0
    assert audio is not None
    assert audio["sample_rate"] == 44100
    assert audio["waveform"].shape == (1, 1, 44100)  # the clip's own 1.0s of sound


def test_load_video_with_no_soundtrack_returns_none_audio(clip):
    path = clip("silent", n=30, fps=30)

    _frames, out_audio = media.load_video(path)

    assert out_audio is None


# ---- load_audio ---------------------------------------------------------------


def test_load_audio_wraps_waveform_with_a_batch_dim(monkeypatch):
    class FakeWaveform:
        def __init__(self):
            self.unsqueezed_dim = None

        def unsqueeze(self, dim):
            self.unsqueezed_dim = dim
            return self

    fake_wave = FakeWaveform()
    monkeypatch.setattr(media, "_audio_load_fn", lambda: (lambda path: (fake_wave, 44100)))

    result = media.load_audio("vo.wav")

    assert result == {"waveform": fake_wave, "sample_rate": 44100}
    assert fake_wave.unsqueezed_dim == 0


# ---- probe / thumbnail (image path only - real PIL, no ComfyUI needed) --------


def test_probe_an_image(tmp_path):
    from PIL import Image

    p = tmp_path / "a.png"
    Image.new("RGB", (40, 20), "red").save(p)

    info = media.probe(str(p))

    assert info == {"kind": "image", "width": 40, "height": 20, "fps": None, "duration": None, "has_audio": False}


def test_thumbnail_png_downscales_an_image(tmp_path):
    from PIL import Image

    p = tmp_path / "a.png"
    Image.new("RGB", (500, 100), "blue").save(p)

    png_bytes = media.thumbnail_png(str(p), max_edge=100)

    from io import BytesIO

    with Image.open(BytesIO(png_bytes)) as thumb:
        assert thumb.format == "PNG"
        assert max(thumb.size) <= 100


# ---- the one fraction->pixel rule (_crop_box) --------------------------------


def test_crop_box_rounds_half_up_and_stays_inside_the_frame():
    # 0.25*50 = 12.5 -> 13 (half-up; Python's round() would give 12 here)
    assert media._crop_box([0.25, 0.25, 0.5, 0.5], 100, 50) == (25, 13, 75, 38)
    assert media._crop_box([0.0, 0.0, 1.0, 1.0], 40, 20) == (0, 0, 40, 20)


def test_crop_box_never_collapses_to_zero_area():
    left, top, right, bottom = media._crop_box([0.999, 0.0, 0.001, 1.0], 6, 4)
    assert right - left >= 1 and bottom - top >= 1
    assert 0 <= left < right <= 6
    assert 0 <= top < bottom <= 4


# ---- load_video trim ----------------------------------------------------------


def test_load_video_passes_trim_window_to_decoder(clip):
    # The window is what the decoder reads, not the file. 7s at 24fps is 168 source
    # frames; [2.0, 6.5) is 108 of them, and the same clip loaded whole is the control
    # that says so - a decoder that ignored the trim would emit 168 for both.
    path = clip("window", n=168, fps=24, pix_fmt="rgb24")

    windowed, _ = media.load_video(path, trim=[2.0, 6.5])
    whole, _ = media.load_video(path)

    assert len(whole) == 168
    assert len(windowed) == 108


def test_load_video_trim_selects_the_source_window(clip):
    # 7s @ 24fps, trimmed to [2.0, 6.5): source frames 48..155 (start-inclusive,
    # end-exclusive), resampled 24->24 so all 108 survive, in order. `marker='index'` on
    # a lossless format means every emitted frame still names the source frame it was
    # filled from, so this asserts the whole slot->source mapping, not two endpoints.
    path = clip("window", n=168, fps=24, pix_fmt="rgb24")

    frames, _ = media.load_video(path, trim=[2.0, 6.5])

    assert [marker_index(f) for f in frames] == list(range(48, 156))


def test_load_video_trim_then_resamples_the_span(clip):
    # 60fps source, [1.0, 3.0) -> 120 source frames -> 48 output frames at 24fps: the
    # span's own 2.0s duration, preserved. Every one is drawn from inside the window.
    path = clip("fast", n=240, fps=60, pix_fmt="rgb24")

    frames, _ = media.load_video(path, trim=[1.0, 3.0])

    assert len(frames) == 48  # 2.0s at 24fps, not the window's 120 source frames
    sources = [marker_index(f) for f in frames]
    assert sources == sorted(sources)
    assert min(sources) >= 60 and max(sources) <= 179


def test_load_video_too_short_trim_raises_naming_file_and_window(clip):
    path = clip("short_trim", n=48, fps=24)

    with pytest.raises(ValueError) as exc:
        media.load_video(path, trim=[1.0, 1.1])

    msg = str(exc.value)
    assert "short_trim.mkv" in msg
    assert "1.00" in msg and "1.10" in msg


def test_load_video_trim_entirely_outside_the_clip_raises(clip):
    path = clip("onesec", n=24, fps=24)  # 1.0s

    with pytest.raises(ValueError) as exc:
        media.load_video(path, trim=[5.0, 9.0])

    assert "0 frame(s)" in str(exc.value)


def test_load_video_trim_slices_the_soundtrack_to_the_same_window(clip):
    # The soundtrack has to be cut to the SAME window as the frames or it drifts out of
    # sync with them. write_clip's audio is a ramp - sample j carries j/N - so a decoded
    # sample names its own position in the source and the head of the window is
    # checkable, not just its length.
    path = clip("audiotrim", n=120, fps=24, audio="pcm_s16le")  # 5.0s @ 44100

    _whole_frames, whole = media.load_video(path)
    _frames, out = media.load_video(path, trim=[1.0, 3.5])

    assert whole is not None and out is not None
    assert whole["waveform"].shape[-1] == 5 * 44100
    assert float(whole["waveform"][0, 0, 0]) == pytest.approx(0.0, abs=1e-3)

    assert out["sample_rate"] == 44100
    assert out["waveform"].shape[-1] == int(2.5 * 44100)
    # 1.0s into a 5.0s ramp is 0.2; pcm_s16le quantises at ~3e-5.
    assert float(out["waveform"][0, 0, 0]) == pytest.approx(0.2, abs=1e-3)


# ---- load_video crop ----------------------------------------------------------


def test_load_video_crop_crops_every_frame(clip):
    # 60x40 source; crop [0.5, 0.25, 0.5, 0.5] -> _crop_box (30, 10, 60, 30) = 30x20.
    path = clip("crop", n=24, fps=24, size=(60, 40))

    frames, _ = media.load_video(path, crop=[0.5, 0.25, 0.5, 0.5])

    assert frames.shape == (24, 20, 30, 3)


def test_load_video_crop_and_trim_compose(clip):
    path = clip("croptrim", n=168, fps=24, size=(60, 40))

    frames, _ = media.load_video(path, crop=[0.0, 0.0, 0.5, 0.5], trim=[2.0, 6.5])

    assert frames.shape == (108, 20, 30, 3)


# ---- load_audio trim -----------------------------------------------------------


def test_load_audio_trim_slices_the_waveform(monkeypatch):
    import numpy as np

    sr = 1000

    class FakeWaveform:
        """[C, L] that unsqueezes to a real numpy [1, C, L] so slicing is honest."""

        def __init__(self, arr):
            self.arr = arr

        def unsqueeze(self, dim):
            assert dim == 0
            return self.arr[None, ...]

    arr = np.arange(10 * sr, dtype=np.float32).reshape(1, -1)
    monkeypatch.setattr(media, "_audio_load_fn", lambda: (lambda path: (FakeWaveform(arr), sr)))

    out = media.load_audio("vo.wav", trim=[2.0, 6.5])

    assert out["sample_rate"] == sr
    assert out["waveform"].shape == (1, 1, int(6.5 * sr) - int(2.0 * sr))
    assert out["waveform"][0, 0, 0] == int(2.0 * sr)


# ---- thumbnail crop / video at_seconds ------------------------------------------


def test_thumbnail_png_applies_the_crop(tmp_path):
    from io import BytesIO

    from PIL import Image

    img = Image.new("RGB", (100, 100), "red")
    img.paste((0, 0, 255), (50, 0, 100, 100))
    p = tmp_path / "a.png"
    img.save(p)

    out = media.thumbnail_png(str(p), max_edge=200, crop=[0.5, 0.0, 0.5, 1.0])

    with Image.open(BytesIO(out)) as thumb:
        assert thumb.size == (50, 100)
        assert thumb.getpixel((25, 50)) == (0, 0, 255)


class _FakeAv:
    """Stub of the av surface thumbnail_png's video branch touches.

    NOT because av is missing - it is a hard dev dependency and is installed, and every
    load_video test above runs on real containers. What this buys is the SEEK, which a
    real decode cannot show: `thumbnail_png` is supposed to seek to the requested pts in
    the stream's own time base and then decode only the GOP between the landed keyframe
    and the target, and a decoder that ignored the seek entirely and read the clip from
    frame 0 would return the byte-identical image. The recorded `seeks`/`decoded` lists
    are the only instrument that can tell the two apart, so the three tests below assert
    on them and not just on the pixel that comes back.

    Deliberately NOT grown into a decoder fake for load_video: it has no `demux()`, no
    packet layer, no `to_ndarray()` and no frame `format`, all of which the streaming
    decoder uses, so it would only ever raise AttributeError there.
    """

    def __init__(self, frame_times, time_base):
        from fractions import Fraction

        from PIL import Image

        av_self = self
        self.seeks = []
        self.decoded = []
        self._seek_to = 0

        stream_time_base = Fraction(1, time_base)

        class FakeStream:
            time_base = stream_time_base

        stream = FakeStream()
        self.stream = stream

        class FakeFrame:
            # `rotation` because thumbnail_png rotates into DISPLAY orientation before
            # the crop; 0 is "no display matrix", which is what these fixtures are.
            rotation = 0

            def __init__(self, pts, color):
                self.pts = pts
                self._color = color

            def to_image(self):
                return Image.new("RGB", (20, 10), self._color)

        colors = ["red", "green", "blue", "yellow", "purple"]
        self.frames = [
            FakeFrame(int(t * time_base), colors[i % len(colors)])
            for i, t in enumerate(frame_times)
        ]

        class FakeStreams:
            video = [stream]

        class FakeContainer:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            @property
            def streams(self):
                return FakeStreams()

            def seek(self, offset, stream=None):
                av_self.seeks.append((offset, stream))
                # like a real backward seek: land on the last frame at/before offset
                av_self._seek_to = 0
                for i, f in enumerate(av_self.frames):
                    if f.pts <= offset:
                        av_self._seek_to = i

            def decode(self, stream):
                for f in av_self.frames[av_self._seek_to:]:
                    av_self.decoded.append(f.pts)
                    yield f

        self._container = FakeContainer()

    def open(self, path):
        return self._container


def _install_fake_av(monkeypatch, fake):
    """Patch `media._open_container` - the seam media.py's own docstring names for this.

    The `setitem(sys.modules, "av", fake)` this replaces does still reach thumbnail_png,
    measured: `_open_container` (media.py:123-135) does a function-LOCAL `import av`,
    which re-reads sys.modules on every call. But it works only for as long as that
    import stays function-local, and it substitutes av for the whole process rather than
    for one call site - a test that also touched a real container would get the stub.
    Patching the named accessor is what media.py asks for and depends on nothing.

    A fake that is never reached makes all three tests below vacuous, so each asserts on
    `fake.seeks`/`fake.decoded`, which stay empty unless the fake really was the
    container. Verified red: with this body stubbed out, all three fail on real av
    trying to open the nonexistent 'clip.mp4'.
    """
    monkeypatch.setattr(media, "_open_container", fake.open)


def test_thumbnail_png_seeks_video_to_at_seconds(monkeypatch):
    from io import BytesIO

    from PIL import Image

    fake = _FakeAv(frame_times=[0.0, 1.0, 2.0, 3.0], time_base=90000)
    _install_fake_av(monkeypatch, fake)

    out = media.thumbnail_png("clip.mp4", at_seconds=2.0)

    # sought in the stream's own time base, on the stream (CU video_types.py:316-320)
    assert fake.seeks == [(180000, fake.stream)]
    # decoded only from the landed keyframe up to the target, never the whole clip
    assert fake.decoded == [180000]
    with Image.open(BytesIO(out)) as thumb:
        assert thumb.getpixel((5, 5)) == (0, 0, 255)  # the 2.0s frame is blue


def test_thumbnail_png_at_seconds_past_the_end_keeps_the_last_frame(monkeypatch):
    from io import BytesIO

    from PIL import Image

    fake = _FakeAv(frame_times=[0.0, 1.0], time_base=90000)
    _install_fake_av(monkeypatch, fake)

    out = media.thumbnail_png("clip.mp4", at_seconds=99.0)

    # it did seek past the end and did decode; the target pts is simply never reached,
    # so the loop keeps the last frame it saw instead of failing the tile
    assert fake.seeks == [(99 * 90000, fake.stream)]
    assert fake.decoded == [90000]
    with Image.open(BytesIO(out)) as thumb:
        assert thumb.getpixel((5, 5)) == (0, 128, 0)  # the 1.0s frame (green)


def test_thumbnail_png_without_at_seconds_does_not_seek(monkeypatch):
    fake = _FakeAv(frame_times=[0.0, 1.0], time_base=90000)
    _install_fake_av(monkeypatch, fake)

    media.thumbnail_png("clip.mp4")

    assert fake.seeks == []
    assert fake.decoded == [0]


# ---- the reference-image size cap ---------------------------------------------
# Core sizes reference images off the SHORT edge (CU/comfy_extras/nodes_minimax_h3.py:301,
# REF_IMAGE_SHORT_EDGE = 2048 at :29), so a wide sheet reaches the VAE enormous at
# ref_image_size="max". Capping the LONG edge here is the guard.
#
# These run against REAL torch (a dev dependency - see pyproject's `dev` group). The
# assertions are about pixel dimensions and the 0..1 normalisation, both of which the
# real `torch.from_numpy` carries through from the numpy array unchanged.


def _png(tmp_path, name, size, color="red"):
    from PIL import Image

    p = tmp_path / name
    Image.new("RGB", size, color).save(p)
    return str(p)


def test_load_image_caps_the_long_edge(tmp_path):
    out = media.load_image(_png(tmp_path, "wide.png", (500, 250)), max_edge=200)

    assert out.shape[1:3] == (100, 200)  # [1, H, W, 3]


def test_load_image_caps_the_long_edge_of_a_tall_reference(tmp_path):
    out = media.load_image(_png(tmp_path, "tall.png", (250, 500)), max_edge=200)

    assert out.shape[1:3] == (200, 100)


def test_load_image_never_upscales_a_small_reference(tmp_path):
    out = media.load_image(_png(tmp_path, "small.png", (100, 50)), max_edge=2048)

    assert out.shape[1:3] == (50, 100)


def test_load_image_cap_of_zero_is_off(tmp_path):
    out = media.load_image(_png(tmp_path, "wide.png", (500, 250)), max_edge=0)

    assert out.shape[1:3] == (250, 500)


def test_load_image_caps_the_cropped_size_not_the_source(tmp_path):
    """Crop first, then cap. A crop that already brings the long edge under the cap
    leaves the pixels alone - the cap must never see the pre-crop dimensions."""
    path = _png(tmp_path, "wide.png", (800, 400))

    # crop to the left half: 400x400, already under a 500 cap
    untouched = media.load_image(path, crop=[0.0, 0.0, 0.5, 1.0], max_edge=500)
    assert untouched.shape[1:3] == (400, 400)

    # the same crop under a 200 cap does get resized
    capped = media.load_image(path, crop=[0.0, 0.0, 0.5, 1.0], max_edge=200)
    assert capped.shape[1:3] == (200, 200)


def test_load_image_cap_keeps_the_pixels_normalised(tmp_path):
    out = media.load_image(_png(tmp_path, "wide.png", (600, 300), (255, 0, 0)), max_edge=100)

    assert out.min() >= 0.0 and out.max() <= 1.0


# ---- structured logging -------------------------------------------------------


def _mmrp_lines(caplog):
    return [r.getMessage() for r in caplog.records if r.name == "MiniMaxRefPack"]


def test_load_image_logs_what_it_emitted(tmp_path, caplog):
    import logging

    p = _png(tmp_path, "wide.png", (500, 250))

    with caplog.at_level(logging.INFO, logger="MiniMaxRefPack"):
        media.load_image(p, crop=[0.0, 0.0, 0.5, 1.0], max_edge=200)

    line = next(ln for ln in _mmrp_lines(caplog) if "event=load_image" in ln)
    assert "src=500x250" in line
    assert "crop=[0,0,0.5,1]" in line
    assert "out=200x200" in line
    assert "ms=" in line


def test_load_video_logs_the_trim_and_what_survived_it(clip, caplog):
    import logging

    path = clip("window", n=168, fps=24)   # 7s @ 24fps; [2.0, 6.5) leaves 108 frames

    with caplog.at_level(logging.INFO, logger="MiniMaxRefPack"):
        media.load_video(path, trim=[2.0, 6.5])

    line = next(ln for ln in _mmrp_lines(caplog) if "event=load_video" in ln)
    assert "file=window.mkv" in line
    assert "trim=[2,6.5]" in line
    assert "frames=108" in line
    assert "ms=" in line


def test_a_failed_load_is_logged_as_a_failure(clip, caplog):
    import logging

    path = clip("short_trim", n=48, fps=24)

    with caplog.at_level(logging.INFO, logger="MiniMaxRefPack"):
        with pytest.raises(ValueError):
            media.load_video(path, trim=[1.0, 1.1])   # under 5 frames at 24fps

    line = next(ln for ln in _mmrp_lines(caplog) if "event=load_video" in ln)
    assert "ok=false" in line and "error=ValueError" in line


# ---- the VLM's copy of a clip ---------------------------------------------------
# This used to hold five tests for `video_clip_bytes`, which chose between inlining the
# original FILE and re-encoding a cropped/trimmed window. That split is gone: every clip
# is now encoded from the frames the sockets already decoded (media.encode_reference_mp4),
# so there is no container-mime lookup left to test and no untouched fast path to protect.
#
# Those five tests all monkeypatched `_transcode_window` and asserted only on the
# arguments it received, which meant the encoder itself had no regression net under it at
# all. tests/test_media_encode.py is the replacement and it runs the real encoder: it
# encodes, decodes the result back with PyAV, and asserts on structure - dimensions, frame
# rate, frame count, and a soundtrack that actually carries samples.


# ---- an audio reference that points at a video file ------------------------------


class _FakeAudioStream:
    def __init__(self, decodable):
        self.codec_context = object() if decodable else None
        self.time_base = None


class _FakeStreamsContainer:
    """Only `streams.audio`/`streams.video`: enough for track selection and the probe."""

    def __init__(self, audio_streams):
        self.streams = type("S", (), {"audio": audio_streams, "video": []})()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_load_audio_on_an_audio_file_never_opens_a_container(monkeypatch):
    def boom(path):
        raise AssertionError("an audio file must go through core's loader, not PyAV")

    class FakeWaveform:
        def unsqueeze(self, dim):
            return self

    monkeypatch.setattr(media, "_open_container", boom)
    monkeypatch.setattr(media, "_audio_load_fn", lambda: (lambda path: (FakeWaveform(), 8000)))

    assert media.load_audio("vo.wav")["sample_rate"] == 8000


def test_load_audio_on_a_video_without_a_decodable_track_names_the_file(monkeypatch):
    def core_loader():
        raise AssertionError("a video file must never reach core's audio loader")

    fake = _FakeStreamsContainer([_FakeAudioStream(False), _FakeAudioStream(False)])
    monkeypatch.setattr(media, "_open_container", lambda path: fake)
    monkeypatch.setattr(media, "_audio_load_fn", core_loader)

    with pytest.raises(ValueError, match=r"'clip\.mp4' has no decodable audio track"):
        media.load_audio("/in/clip.mp4")


def test_track_selection_skips_a_codec_less_first_track():
    first, second = _FakeAudioStream(False), _FakeAudioStream(True)
    chosen = media._last_decodable_audio_stream(_FakeStreamsContainer([first, second]))
    assert chosen is second


def test_probe_reports_a_clip_whose_only_audio_is_undecodable_as_silent(monkeypatch):
    class VideoStream:
        average_rate = 25
        duration = None
        time_base = None
        width, height = 64, 48

    fake = _FakeStreamsContainer([_FakeAudioStream(False)])
    fake.streams.video = [VideoStream()]
    fake.duration = None
    monkeypatch.setattr(media, "_open_container", lambda path: fake)
    monkeypatch.setattr(media, "_first_frame_rotation_k", lambda c, s: (0, 64, 48))

    info = media.probe("iphone.mov")

    assert info["kind"] == "video"
    assert info["has_audio"] is False
