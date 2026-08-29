"""Tests for minimax_refpack.media - the pure resample math and the ComfyUI-decoder
seams (VideoFromFile / nodes_audio.load). comfy_api/comfy_extras are not importable
outside a ComfyUI process (verified: bare `import comfy_api` raises ModuleNotFoundError
in this venv), so the seams are monkeypatched at their accessor functions rather than at
the real (unavailable) module path.
"""

import sys
import types

import pytest

from minimax_refpack import media


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


class _ListFrames:
    """Fake images tensor: .shape[0] + fancy list-indexing, no torch required."""

    def __init__(self, n, first=0):
        self.shape = (n,)
        self._items = list(range(first, first + n))

    def __getitem__(self, idx):
        return [self._items[i] for i in idx]


def _fake_video_from_file(images_n, frame_rate, audio=None):
    class FakeVideoFromFile:
        def __init__(self, path, *, start_time=0, duration=0):
            self.path = path
            self.start_time = start_time
            self.duration = duration

        def get_components(self):
            import math

            first = max(0, math.ceil(self.start_time * frame_rate - 1e-9))
            stop = images_n
            if self.duration:
                stop = min(
                    images_n,
                    math.ceil((self.start_time + self.duration) * frame_rate - 1e-9),
                )

            window_audio = (
                media._slice_audio(
                    audio, [self.start_time, self.start_time + self.duration]
                )
                if audio is not None and self.duration
                else audio
            )

            class FakeComponents:
                def __init__(self):
                    self.images = _ListFrames(max(0, stop - first), first=first)
                    self.frame_rate = frame_rate
                    self.audio = window_audio

            return FakeComponents()

    return FakeVideoFromFile


def test_load_video_raises_when_resampled_clip_is_too_short(monkeypatch):
    # 3 frames at 24fps -> 3 frames at target 24fps, under the 5-frame floor
    monkeypatch.setattr(media, "_video_from_file_cls", lambda: _fake_video_from_file(3, 24))
    with pytest.raises(ValueError) as exc:
        media.load_video("clip_too_short.mp4")
    assert "clip_too_short.mp4" in str(exc.value)


def test_load_video_does_not_pad_short_clips_up_to_five(monkeypatch):
    # 4 frames is still under 5 - must raise, never silently duplicate up to 5
    monkeypatch.setattr(media, "_video_from_file_cls", lambda: _fake_video_from_file(4, 24))
    with pytest.raises(ValueError):
        media.load_video("clip.mp4")


def test_load_video_passes_through_resampled_frames_and_audio(monkeypatch):
    audio = {"waveform": "wf", "sample_rate": 48000}
    monkeypatch.setattr(media, "_video_from_file_cls", lambda: _fake_video_from_file(30, 30, audio))
    frames, out_audio = media.load_video("clip.mp4", target_fps=24)
    assert len(frames) == 24
    assert out_audio is audio


def test_load_video_with_no_soundtrack_returns_none_audio(monkeypatch):
    monkeypatch.setattr(media, "_video_from_file_cls", lambda: _fake_video_from_file(30, 30, None))
    _frames, out_audio = media.load_video("clip.mp4")
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


def test_load_video_passes_trim_window_to_decoder(monkeypatch):
    calls = {}

    class Components:
        images = _ListFrames(108)
        frame_rate = 24
        audio = None

    class WindowedVideoFromFile:
        def __init__(self, path, *, start_time=0, duration=0):
            calls.update(path=path, start_time=start_time, duration=duration)

        def get_components(self):
            return Components()

    monkeypatch.setattr(media, "_video_from_file_cls", lambda: WindowedVideoFromFile)

    frames, _ = media.load_video("clip.mp4", trim=[2.0, 6.5])

    assert calls == {"path": "clip.mp4", "start_time": 2.0, "duration": 4.5}
    assert len(frames) == 108


def test_load_video_trim_selects_the_source_window(monkeypatch):
    # 10s @ 24fps, trimmed to [2.0, 6.5): source frames 48..155 (start-inclusive,
    # end-exclusive), resampled 24->24 so all 108 survive, in order.
    monkeypatch.setattr(media, "_video_from_file_cls", lambda: _fake_video_from_file(240, 24))
    frames, _ = media.load_video("clip.mp4", trim=[2.0, 6.5])
    assert len(frames) == 108
    assert frames[0] == 48
    assert frames[-1] == 155


def test_load_video_trim_then_resamples_the_span(monkeypatch):
    # 60fps source, [1.0, 3.0) -> 120 source frames -> 48 output frames at 24fps,
    # all drawn from inside the window.
    monkeypatch.setattr(media, "_video_from_file_cls", lambda: _fake_video_from_file(600, 60))
    frames, _ = media.load_video("clip.mp4", trim=[1.0, 3.0])
    assert len(frames) == 48
    assert min(frames) >= 60
    assert max(frames) <= 179


def test_load_video_too_short_trim_raises_naming_file_and_window(monkeypatch):
    monkeypatch.setattr(media, "_video_from_file_cls", lambda: _fake_video_from_file(240, 24))
    with pytest.raises(ValueError) as exc:
        media.load_video("clip.mp4", trim=[1.0, 1.1])
    msg = str(exc.value)
    assert "clip.mp4" in msg
    assert "1.00" in msg and "1.10" in msg


def test_load_video_trim_entirely_outside_the_clip_raises(monkeypatch):
    monkeypatch.setattr(media, "_video_from_file_cls", lambda: _fake_video_from_file(24, 24))
    with pytest.raises(ValueError):
        media.load_video("clip.mp4", trim=[5.0, 9.0])


def test_load_video_trim_slices_the_soundtrack_to_the_same_window(monkeypatch):
    import numpy as np

    sr = 1000
    audio = {
        "waveform": np.arange(10 * sr, dtype=np.float32).reshape(1, 1, -1),
        "sample_rate": sr,
    }
    monkeypatch.setattr(media, "_video_from_file_cls", lambda: _fake_video_from_file(240, 24, audio))
    _frames, out = media.load_video("clip.mp4", trim=[2.0, 6.5])
    assert out["sample_rate"] == sr
    assert out["waveform"].shape[-1] == int(6.5 * sr) - int(2.0 * sr)
    assert out["waveform"][0, 0, 0] == int(2.0 * sr)
    # the source dict is not mutated in place
    assert audio["waveform"].shape[-1] == 10 * sr


# ---- load_video crop ----------------------------------------------------------


def _fake_video_numpy(n, frame_rate, h, w, audio=None):
    import numpy as np

    class FakeVideoFromFile:
        def __init__(self, path, *, start_time=0, duration=0):
            self.path = path
            self.start_time = start_time
            self.duration = duration

        def get_components(self):
            import math

            first = max(0, math.ceil(self.start_time * frame_rate - 1e-9))
            stop = n
            if self.duration:
                stop = min(
                    n, math.ceil((self.start_time + self.duration) * frame_rate - 1e-9)
                )

            class FakeComponents:
                def __init__(self):
                    self.images = np.zeros(
                        (max(0, stop - first), h, w, 3), dtype=np.float32
                    )
                    self.frame_rate = frame_rate
                    self.audio = audio

            return FakeComponents()

    return FakeVideoFromFile


def test_load_video_crop_crops_every_frame(monkeypatch):
    monkeypatch.setattr(media, "_video_from_file_cls", lambda: _fake_video_numpy(24, 24, 40, 60))
    frames, _ = media.load_video("clip.mp4", crop=[0.5, 0.25, 0.5, 0.5])
    assert frames.shape == (24, 20, 30, 3)


def test_load_video_crop_and_trim_compose(monkeypatch):
    monkeypatch.setattr(media, "_video_from_file_cls", lambda: _fake_video_numpy(240, 24, 40, 60))
    frames, _ = media.load_video("clip.mp4", crop=[0.0, 0.0, 0.5, 0.5], trim=[2.0, 6.5])
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
    """Stub of the av surface thumbnail_png's video branch touches. av is not
    installed in this venv (same reason the ComfyUI decoders are stubbed), so the
    seek arithmetic is proven against a recorder instead."""

    def __init__(self, frame_times, time_base):
        from fractions import Fraction

        from PIL import Image

        av_self = self
        self.seeks = []
        self.decoded = []
        self._seek_to = 0

        class FakeStream:
            pass

        stream = FakeStream()
        stream.time_base = Fraction(1, time_base)
        self.stream = stream

        class FakeFrame:
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
    import sys

    monkeypatch.setitem(sys.modules, "av", fake)


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
# torch is not installed in this venv (the module lazy-imports it, same reason
# folder_paths and av get stubbed elsewhere in this file), so `fake_torch` stands in
# with an identity from_numpy - the assertions are all about pixel dimensions, which
# the numpy array carries unchanged.


@pytest.fixture
def fake_torch(monkeypatch):
    module = types.ModuleType("torch")
    module.from_numpy = lambda arr: arr
    monkeypatch.setitem(sys.modules, "torch", module)
    return module


def _png(tmp_path, name, size, color="red"):
    from PIL import Image

    p = tmp_path / name
    Image.new("RGB", size, color).save(p)
    return str(p)


def test_load_image_caps_the_long_edge(tmp_path, fake_torch):
    out = media.load_image(_png(tmp_path, "wide.png", (500, 250)), max_edge=200)

    assert out.shape[1:3] == (100, 200)  # [1, H, W, 3]


def test_load_image_caps_the_long_edge_of_a_tall_reference(tmp_path, fake_torch):
    out = media.load_image(_png(tmp_path, "tall.png", (250, 500)), max_edge=200)

    assert out.shape[1:3] == (200, 100)


def test_load_image_never_upscales_a_small_reference(tmp_path, fake_torch):
    out = media.load_image(_png(tmp_path, "small.png", (100, 50)), max_edge=2048)

    assert out.shape[1:3] == (50, 100)


def test_load_image_cap_of_zero_is_off(tmp_path, fake_torch):
    out = media.load_image(_png(tmp_path, "wide.png", (500, 250)), max_edge=0)

    assert out.shape[1:3] == (250, 500)


def test_load_image_caps_the_cropped_size_not_the_source(tmp_path, fake_torch):
    """Crop first, then cap. A crop that already brings the long edge under the cap
    leaves the pixels alone - the cap must never see the pre-crop dimensions."""
    path = _png(tmp_path, "wide.png", (800, 400))

    # crop to the left half: 400x400, already under a 500 cap
    untouched = media.load_image(path, crop=[0.0, 0.0, 0.5, 1.0], max_edge=500)
    assert untouched.shape[1:3] == (400, 400)

    # the same crop under a 200 cap does get resized
    capped = media.load_image(path, crop=[0.0, 0.0, 0.5, 1.0], max_edge=200)
    assert capped.shape[1:3] == (200, 200)


def test_load_image_cap_keeps_the_pixels_normalised(tmp_path, fake_torch):
    out = media.load_image(_png(tmp_path, "wide.png", (600, 300), (255, 0, 0)), max_edge=100)

    assert out.min() >= 0.0 and out.max() <= 1.0


# ---- structured logging -------------------------------------------------------


def _mmrp_lines(caplog):
    return [r.getMessage() for r in caplog.records if r.name == "MiniMaxRefPack"]


def test_load_image_logs_what_it_emitted(tmp_path, fake_torch, caplog):
    import logging

    p = _png(tmp_path, "wide.png", (500, 250))

    with caplog.at_level(logging.INFO, logger="MiniMaxRefPack"):
        media.load_image(p, crop=[0.0, 0.0, 0.5, 1.0], max_edge=200)

    line = next(ln for ln in _mmrp_lines(caplog) if "event=load_image" in ln)
    assert "src=500x250" in line
    assert "crop=[0,0,0.5,1]" in line
    assert "out=200x200" in line
    assert "ms=" in line


def test_load_video_logs_the_trim_and_what_survived_it(monkeypatch, caplog):
    import logging

    monkeypatch.setattr(media, "_video_from_file_cls", lambda: _fake_video_from_file(240, 24))

    with caplog.at_level(logging.INFO, logger="MiniMaxRefPack"):
        media.load_video("clip.mp4", trim=[2.0, 6.5])

    line = next(ln for ln in _mmrp_lines(caplog) if "event=load_video" in ln)
    assert "trim=[2,6.5]" in line
    assert "frames=108" in line
    assert "ms=" in line


def test_a_failed_load_is_logged_as_a_failure(monkeypatch, caplog):
    import logging

    monkeypatch.setattr(media, "_video_from_file_cls", lambda: _fake_video_from_file(240, 24))

    with caplog.at_level(logging.INFO, logger="MiniMaxRefPack"):
        with pytest.raises(ValueError):
            media.load_video("clip.mp4", trim=[1.0, 1.1])   # under 5 frames at 24fps

    line = next(ln for ln in _mmrp_lines(caplog) if "event=load_video" in ln)
    assert "ok=false" in line and "error=ValueError" in line


# ---- video bytes for the VLM ---------------------------------------------------
# The VLM takes whole videos (OpenRouter `video_url` parts, verified live against
# google/gemini-3-flash-preview: 10.12s clip -> 660 video tokens + 250 audio tokens).
# An untouched clip is therefore sent as the FILE, which decodes nothing at all; a
# cropped or trimmed one has to be re-encoded so the VLM sees what the socket emits.


def test_an_untouched_clip_is_sent_as_the_file_itself(tmp_path):
    p = tmp_path / "clip.mp4"
    p.write_bytes(b"\x00\x01not-really-an-mp4\x02")

    data, mime = media.video_clip_bytes(str(p))

    assert data == p.read_bytes()
    assert mime == "video/mp4"


def test_the_file_path_carries_the_soundtrack_so_it_needs_no_re_encode(tmp_path):
    """Documented as behaviour: nothing is decoded on this path."""
    p = tmp_path / "clip.mp4"
    p.write_bytes(b"x" * 32)

    assert media.video_clip_bytes(str(p))[1] == "video/mp4"
    assert media.video_clip_bytes(str(p), crop=None, trim=None)[0] == b"x" * 32


@pytest.mark.parametrize("suffix,mime", [
    (".mp4", "video/mp4"), (".mov", "video/mov"), (".webm", "video/webm"), (".mpeg", "video/mpeg"),
])
def test_containers_openrouter_accepts_pass_straight_through(tmp_path, suffix, mime):
    p = tmp_path / f"clip{suffix}"
    p.write_bytes(b"bytes")

    assert media.video_clip_bytes(str(p)) == (b"bytes", mime)


def test_a_container_openrouter_does_not_accept_is_re_encoded(tmp_path, monkeypatch):
    called = {}

    def fake_transcode(path, crop, trim):
        called["args"] = (path, crop, trim)
        return b"mp4"

    monkeypatch.setattr(media, "_transcode_window", fake_transcode)
    p = tmp_path / "clip.avi"
    p.write_bytes(b"avi bytes")

    data, mime = media.video_clip_bytes(str(p))

    assert (data, mime) == (b"mp4", "video/mp4")
    assert called["args"] == (str(p), None, None)


@pytest.mark.parametrize("crop,trim", [
    ([0.0, 0.0, 0.5, 1.0], None),
    (None, [2.0, 6.5]),
    ([0.0, 0.0, 0.5, 1.0], [2.0, 6.5]),
])
def test_an_edited_clip_is_re_encoded_so_the_vlm_sees_the_edit(tmp_path, monkeypatch, crop, trim):
    seen = {}

    def fake_transcode(path, c, t):
        seen["args"] = (c, t)
        return b"mp4"

    monkeypatch.setattr(media, "_transcode_window", fake_transcode)
    p = tmp_path / "clip.mp4"
    p.write_bytes(b"original")

    data, mime = media.video_clip_bytes(str(p), crop=crop, trim=trim)

    assert data == b"mp4" and mime == "video/mp4"
    assert seen["args"] == (crop, trim)
