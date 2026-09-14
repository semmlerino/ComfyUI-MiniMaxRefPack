"""The soundtrack half of the streaming decoder: head trim, decode extent, stream
selection, waveform shape and the decode budget.

This file exists because the audio path is where the rewrite deliberately CHANGES
output, and every one of those changes is invisible to the pixel oracle in
test_media_decode.py - a wrong soundtrack decodes into a perfectly valid tensor.

  head trim      ComfyUI computes it as `start_time - frame.pts * audio_stream.time_base`,
                 on a frame `AudioResampler(format='fltp')` may have REBASED into
                 1/sample_rate. On the two mkv arms that multiplies a rebased pts by
                 1/1000 and keeps ~44x too much head; `media._frame_start_seconds` uses
                 the frame's own base (`_save_transcoded`'s form) instead. The mp4/aac
                 arm is the control: the resampler passes it through, so it was already
                 right and the fix must not move it.
  decode extent  the tail check and the duration cap are two different guards and only
                 one of them is visible in the returned length. The cap truncates to the
                 window whether the decoder stopped at the tail or read to EOF, so the
                 length alone cannot see a tail that never fires (ComfyUI's tail compares
                 `frame.time`, which is NaN for a frame with a pts and no time base, and
                 every NaN comparison is false). The pre-cap buffer is asserted directly.
  selection      a track FFmpeg has no decoder for takes the PROCESS down when decoded,
                 so the last DECODABLE stream wins, not `streams.audio[-1]`. A fixture
                 with two good tracks cannot tell those two apart; this one carries a
                 real undecodable final track.
  channels       `_waveform_channels` replaces `_planar_waveform(...).shape[0]` to drop a
                 float32 copy of the whole soundtrack. Agreement tests alone would pass
                 against the copy it exists to remove, so the copy is poisoned instead.
  budget         `MINIMAX_REFPACK_MAX_DECODE_GIB` is read once and cached, which makes a
                 parametrised matrix false-green unless the cache is cleared between
                 cases - so the clearing seam is itself asserted.

Two head-trim states are unreachable from any container this FFmpeg build can write (a
fine time base surviving the resampler, and a frame with a pts but no base at all), so
they are pinned as unit tests on `_frame_start_seconds` with hand-built AudioFrames.
"""

import math
import os
import sys
from fractions import Fraction

import av
import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(__file__))

from conftest import AUDIO_FORMATS  # noqa: E402

from minimax_refpack import media  # noqa: E402

# The fixture is a 4s mono ramp where sample j carries j/N, so a decoded VALUE names the
# index it came from: 2.0s into a 4s ramp is 0.5, 3.0s is 0.75.
AUDIO_SECONDS = 4.0
SAMPLE_RATE = 44100
TOTAL_SAMPLES = int(AUDIO_SECONDS * SAMPLE_RATE)
TRIM = [2.0, 3.0]
WINDOW_SAMPLES = int((TRIM[1] - TRIM[0]) * SAMPLE_RATE)
HEAD_VALUE = TRIM[0] * SAMPLE_RATE / TOTAL_SAMPLES
TAIL_VALUE = TRIM[1] * SAMPLE_RATE / TOTAL_SAMPLES

# Per-arm amplitude tolerance, measured on this build. pcm_s16le and mp2 both decode
# through s16/s16p, so a ramp sample comes back within a couple of LSBs (1.2e-4 and
# 9.2e-5 observed). AAC is a perceptual transform codec and reproduces a near-DC ramp
# badly - 0.073 off at the head, 0.24 at the tail - so its amplitude can only carry a
# coarse assertion and the tight one is made against its own untrimmed decode instead.
HEAD_ATOL = {"aac": 0.10, "pcm_s16le": 5e-3, "mp2": 5e-3}
EXACT_ARMS = ("pcm_s16le", "mp2")


def audio_clip(clip, codec, **kwargs):
    """A 4s clip whose soundtrack is `codec`, in the container that arm needs."""
    container, ext, _rebases = AUDIO_FORMATS[codec]
    return clip(
        f"audio_{codec}.{ext}",
        n=100,
        fps=25,
        container=container,
        audio=codec,
        audio_seconds=AUDIO_SECONDS,
        **kwargs,
    )


def mono(audio):
    """The [1, 1, L] waveform of a mono soundtrack as a flat numpy array."""
    arr = audio["waveform"].numpy()
    assert arr.shape[:2] == (1, 1), f"expected a mono [1,1,L] waveform, got {arr.shape}"
    return arr[0, 0]


@pytest.fixture
def concat_spy(monkeypatch):
    """Capture the PRE-CAP soundtrack buffer `_decode_pass` concatenates.

    The returned dict is the only instrument that can see the tail check and the
    `.copy()`: the duration cap rewrites the length before the caller ever sees it, and
    a buffer still referenced by a returned view is indistinguishable from a released
    one at the API surface.
    """
    captured = []
    real = np.concatenate

    def spy(*args, **kwargs):
        out = real(*args, **kwargs)
        captured.append(out)
        return out

    monkeypatch.setattr(np, "concatenate", spy)
    return captured


def only_buffer(captured):
    assert len(captured) == 1, (
        f"expected exactly one np.concatenate in a decode, saw {len(captured)} - "
        "the spy is picking up someone else's call and every bound below is wrong"
    )
    return captured[0]


# ---- head trim -----------------------------------------------------------------


@pytest.mark.parametrize("codec", sorted(AUDIO_FORMATS))
def test_head_trim_starts_at_the_requested_second(clip, codec):
    """The first retained sample must be the one that plays at `trim[0]`.

    The ramp makes that checkable by VALUE: 2.0s into a 4s ramp is 0.5. Under the
    arithmetic this replaces, the two mkv arms multiply a pts already rebased into
    1/44100 by the stream's 1/1000, so `frame_start` runs 44x fast, the head skip
    collapses and retention starts around 0.045s - a first sample near 0.011.
    """
    path = audio_clip(clip, codec)
    _frames, audio = media.load_video(path, trim=TRIM)

    assert audio is not None
    assert audio["sample_rate"] == SAMPLE_RATE
    first = float(mono(audio)[0])
    assert first == pytest.approx(HEAD_VALUE, abs=HEAD_ATOL[codec]), (
        f"{codec}: retained from {first * AUDIO_SECONDS:.3f}s, wanted {TRIM[0]}s"
    )


@pytest.mark.parametrize("codec", sorted(AUDIO_FORMATS))
def test_head_trim_lands_on_the_untrimmed_decodes_own_samples(clip, codec):
    """The tight arm of the head trim, and the only one AAC can carry.

    Decoding the same file with no trim gives the same codec noise, so the trimmed
    window has to be a bit-exact slice of it starting at sample `2.0 * sample_rate`.
    That pins the head to within a millisecond on every arm - including the control,
    whose amplitude is too distorted to say anything tighter than 0.1.
    """
    path = audio_clip(clip, codec)
    _f, full_audio = media.load_video(path)
    _g, cut_audio = media.load_video(path, trim=TRIM)
    full, cut = mono(full_audio), mono(cut_audio)

    at = int(TRIM[0] * SAMPLE_RATE)
    probe = min(2000, cut.shape[0])
    shifts = range(-128, 129)
    errors = {
        k: float(np.abs(full[at + k : at + k + probe] - cut[:probe]).max())
        for k in shifts
        if 0 <= at + k and at + k + probe <= full.shape[0]
    }
    best = min(errors, key=lambda k: errors[k])
    assert errors[best] <= 1e-4, (
        f"{codec}: the trimmed window matches nothing in the untrimmed decode "
        f"(best offset {best}, max diff {errors[best]:.2e})"
    )
    assert abs(best) <= 64, (
        f"{codec}: window starts {best} samples "
        f"({best / SAMPLE_RATE * 1e3:.1f} ms) off {TRIM[0]}s"
    )


@pytest.mark.parametrize("codec", EXACT_ARMS)
def test_head_trim_keeps_the_whole_window_not_a_prefix_of_it(clip, codec):
    """Where the window ENDS, on the two arms whose amplitude survives the codec.

    A head trim that is right and a decode that stops early both start at 0.5; only the
    last sample says the far edge of the window is still there.
    """
    path = audio_clip(clip, codec)
    _frames, audio = media.load_video(path, trim=TRIM)
    last = float(mono(audio)[-1])
    assert last == pytest.approx(TAIL_VALUE, abs=HEAD_ATOL[codec]), (
        f"{codec}: window ends at {last * AUDIO_SECONDS:.3f}s, wanted {TRIM[1]}s"
    )


@pytest.mark.parametrize("codec", sorted(AUDIO_FORMATS))
def test_retained_sample_count_matches_the_window(clip, codec):
    """Length, not just position - a first-sample assertion cannot see a short read.

    Within one audio frame of `duration * sample_rate`: the cap trims the far edge to
    the sample, but a head trim that fired late or a tail that fired early lands here.
    """
    path = audio_clip(clip, codec)
    _frames, audio = media.load_video(path, trim=TRIM)
    count = mono(audio).shape[0]
    assert abs(count - WINDOW_SAMPLES) <= 2048, (
        f"{codec}: kept {count} samples for a {WINDOW_SAMPLES}-sample window"
    )


# ---- decode extent and the duration cap -----------------------------------------


@pytest.mark.parametrize("codec", sorted(AUDIO_FORMATS))
def test_decode_stops_at_the_tail_and_not_at_eof(clip, codec, concat_spy):
    """The assertion the returned length CANNOT make.

    The duration cap truncates to the window either way, so a tail check that never
    fires - ComfyUI's `frame.time > start + duration`, false forever when `frame.time`
    is NaN - is invisible downstream while the decoder reads to EOF. Here the window is
    1s and EOF is 2s away, so the pre-cap buffer would be about double.
    """
    path = audio_clip(clip, codec)
    media.load_video(path, trim=TRIM)
    buf = only_buffer(concat_spy)

    to_eof = TOTAL_SAMPLES - int(TRIM[0] * SAMPLE_RATE)
    assert buf.shape[1] <= WINDOW_SAMPLES + 8192, (
        f"{codec}: decoded {buf.shape[1]} samples for a {WINDOW_SAMPLES}-sample "
        f"window ({to_eof} would be a read to EOF)"
    )
    assert buf.shape[1] >= WINDOW_SAMPLES, (
        f"{codec}: decoded {buf.shape[1]} samples, short of the window itself"
    )


def test_duration_cap_truncates_the_overshoot_to_the_window(clip, concat_spy):
    """The cap is what makes the returned length exact, and it must actually engage."""
    path = audio_clip(clip, "pcm_s16le")
    _frames, audio = media.load_video(path, trim=TRIM)
    buf = only_buffer(concat_spy)

    assert buf.shape[1] > WINDOW_SAMPLES, (
        "the decode did not overshoot, so this file cannot see the cap at all"
    )
    assert mono(audio).shape[0] == WINDOW_SAMPLES


def test_mono_truncation_releases_the_decoded_buffer(clip, concat_spy):
    """`.copy()`, not `np.ascontiguousarray` - and MONO is the shape that shows it.

    A (1, N) buffer sliced to (1, M) is still flagged C_CONTIGUOUS, because a leading
    axis of extent 1 makes its stride irrelevant, so `ascontiguousarray` hands back the
    same view and the whole concatenation stays alive behind the returned waveform.
    (2, N) stereo does get copied, which is why the leak is silent on most fixtures.
    """
    path = audio_clip(clip, "pcm_s16le")
    _frames, audio = media.load_video(path, trim=TRIM)
    buf = only_buffer(concat_spy)

    assert buf.shape[0] == 1, "not a mono buffer - this test proves nothing on stereo"
    assert buf.shape[1] > WINDOW_SAMPLES, "no truncation happened, so nothing was copied"
    assert np.shares_memory(np.ascontiguousarray(buf[..., :WINDOW_SAMPLES]), buf), (
        "ascontiguousarray released a mono slice on this numpy, so the assertion below "
        "would pass against the broken form too"
    )
    assert np.shares_memory(audio["waveform"].numpy(), buf) is False


# ---- audio stream selection ------------------------------------------------------


def write_two_track_clip(path, *, second_rate=22050):
    """A real mkv with video + two audio tracks, each a constant of its own.

    Track 1 is mp2 at 44100 holding 0.25; track 2 is pcm_s16le at `second_rate` holding
    0.75. Distinct sample rates AND distinct amplitudes, so the returned dict names
    which track was decoded twice over.
    """
    path = str(path)
    out = av.open(path, "w", format="matroska")
    try:
        video = out.add_stream("libx264", rate=25)
        video.width, video.height = 64, 48
        video.pix_fmt = "yuv420p"
        first = out.add_stream("mp2", rate=44100, layout="mono")
        second = out.add_stream("pcm_s16le", rate=second_rate, layout="mono")
        for i in range(30):
            frame = av.VideoFrame.from_ndarray(
                np.full((48, 64, 3), i * 8, dtype=np.uint8), format="rgb24"
            )
            for packet in video.encode(frame):
                out.mux(packet)
        for packet in video.encode():
            out.mux(packet)
        _encode_constant(out, first, 0.25, 1.2, 44100)
        _encode_constant(out, second, 0.75, 1.2, second_rate)
    finally:
        out.close()
    return path


def _encode_constant(out, stream, value, seconds, rate):
    total = int(seconds * rate)
    data = np.full((1, total), value, dtype=np.float32)
    resampler = av.AudioResampler(
        format=stream.format, layout=stream.layout, rate=stream.rate
    )

    def mux(frames):
        for resampled in frames:
            for packet in stream.encode(resampled):
                out.mux(packet)

    for start in range(0, total, 4096):
        frame = av.AudioFrame.from_ndarray(
            np.ascontiguousarray(data[:, start : start + 4096]),
            format="fltp",
            layout="mono",
        )
        frame.sample_rate = rate
        frame.pts = start
        frame.time_base = Fraction(1, rate)
        mux(resampler.resample(frame))
    mux(resampler.resample(None))
    for packet in stream.encode(None):
        out.mux(packet)


def break_last_codec_id(src, dst):
    """Copy `src`, renaming the LAST track's mkv CodecID to one FFmpeg cannot map.

    `A_PCM/INT/LIT` -> `A_QQQ/INT/LIT` is the same length, so the EBML element keeps its
    size and everything around it stays valid; FFmpeg simply has no decoder for the
    result and PyAV hands back a stream with `codec_context is None` - the shape of the
    iPhone APAC spatial-audio track this selection rule exists for.
    """
    raw = open(src, "rb").read()
    at = raw.rfind(b"A_PCM/INT/LIT")
    assert at != -1, "no pcm CodecID in the fixture - nothing was broken"
    open(dst, "wb").write(raw[:at] + b"A_QQQ/INT/LIT" + raw[at + 13 :])
    return str(dst)


def test_the_last_audio_track_is_the_one_decoded(tmp_path):
    """Two decodable tracks: video_types.py takes the last, so this decoder must too."""
    path = write_two_track_clip(tmp_path / "two.mkv")
    _frames, audio = media.load_video(path)

    assert audio["sample_rate"] == 22050
    assert float(audio["waveform"].numpy().mean()) == pytest.approx(0.75, abs=1e-3)


def test_an_undecodable_final_track_falls_back_to_the_previous_one(tmp_path):
    """The case `container.streams.audio[-1]` gets wrong - and crashes the process on.

    Decoding a stream FFmpeg has no decoder for takes the interpreter down, so this
    cannot be written as "the shorthand returns the wrong track"; it is written as "the
    shorthand would have returned a track with no codec context at all", asserted on the
    stream objects, and then the whole decode is run to show it survives.
    """
    good = write_two_track_clip(tmp_path / "two.mkv")
    path = break_last_codec_id(good, tmp_path / "two_broken.mkv")

    with media._open_container(path) as container:
        streams = container.streams.audio
        assert len(streams) == 2
        assert streams[-1].codec_context is None, (
            "the last track is still decodable, so this fixture cannot tell "
            "_last_decodable_audio_stream apart from streams.audio[-1]"
        )
        chosen = media._last_decodable_audio_stream(container)
        assert chosen is streams[0]
        assert chosen is not streams[-1]

    _frames, audio = media.load_video(path)
    assert audio["sample_rate"] == 44100, "decoded the wrong track"
    assert float(audio["waveform"].numpy().mean()) == pytest.approx(0.25, abs=1e-3)


def test_a_clip_with_no_audio_decodes_to_none(clip):
    """No soundtrack is not an error, and must not be an empty waveform either."""
    path = clip("silent.mkv", n=30, fps=25)
    _frames, audio = media.load_video(path)
    assert audio is None

    with media._open_container(path) as container:
        assert container.streams.audio == ()
        assert media._last_decodable_audio_stream(container) is None


# ---- _frame_start_seconds, the states no container can produce -------------------


def audio_frame(pts, time_base):
    frame = av.AudioFrame(format="fltp", layout="mono", samples=1024)
    frame.sample_rate = SAMPLE_RATE
    if pts is not None:
        frame.pts = pts
    if time_base is not None:
        frame.time_base = time_base
    return frame


def test_frame_start_uses_the_frames_own_time_base():
    """The over-trim direction: a fine base (mpegts' 1/90000) must not be second-guessed.

    No container this build can write produces it - every mpegts codec that muxes
    decodes to fltp and the resampler passes the frame through carrying 1/44100 - so the
    only place it can be pinned is here.
    """
    pts = int(TRIM[0] * SAMPLE_RATE)
    frame = audio_frame(pts, Fraction(1, 90000))

    got = media._frame_start_seconds(frame, SAMPLE_RATE)
    assert got == pytest.approx(float(pts * Fraction(1, 90000)))
    assert got != pytest.approx(TRIM[0]), (
        "the frame's own base was ignored in favour of the sample rate"
    )


def test_frame_start_falls_back_to_one_over_sample_rate():
    """A pts with NO base at all: `Fraction(1, sample_rate)`, never a stream's base.

    Both wrong answers are asserted against, because "it did not raise" would pass for
    exactly the substitution that restores the bug: mkv's 1/1000 turns a 2.0s frame into
    88.2s, which skips no head and fires the tail on the first frame.
    """
    pts = int(TRIM[0] * SAMPLE_RATE)
    frame = audio_frame(pts, None)
    assert frame.time_base is None
    frame_time = frame.time
    assert frame_time is not None and math.isnan(frame_time), (
        "frame.time is no longer NaN without a base, so the tail check could be "
        "written on it after all - re-read _frame_start_seconds' docstring"
    )

    got = media._frame_start_seconds(frame, SAMPLE_RATE)
    assert got == pytest.approx(TRIM[0])
    assert got != pytest.approx(float(pts * Fraction(1, 1000))), "used mkv's 1/1000"
    assert got != pytest.approx(float(pts * Fraction(1, 90000))), "used mpegts' 1/90000"


def test_frame_start_of_a_frame_without_a_pts_is_zero():
    """`float(None * time_base)` is a TypeError; the head skip has to survive it."""
    assert media._frame_start_seconds(audio_frame(None, None), SAMPLE_RATE) == 0.0


# ---- _waveform_channels ----------------------------------------------------------


@pytest.mark.parametrize(
    "shape,channels",
    [((8,), 1), ((1, 8), 1), ((2, 8), 2), ((1, 1, 8), 1), ((1, 2, 8), 2)],
)
@pytest.mark.parametrize("kind", ["numpy", "torch"])
def test_waveform_channels_agrees_with_the_planar_copy(shape, channels, kind):
    """[L] / [C,L] / [1,C,L], both array types, against the function it replaced."""
    waveform = (
        np.zeros(shape, dtype=np.float32) if kind == "numpy" else torch.zeros(*shape)
    )
    assert media._waveform_channels(waveform) == channels
    assert media._waveform_channels(waveform) == media._planar_waveform(waveform).shape[0]


def test_waveform_channels_reads_a_plain_list():
    """No `.shape` attribute - the branch that keeps a hand-built dict working."""
    assert media._waveform_channels([0.0] * 8) == 1


def test_opening_an_audio_stream_builds_no_planar_copy(tmp_path, monkeypatch):
    """The assertion with teeth: agreement alone passes against the copy being removed.

    `_waveform_channels` implemented as `_planar_waveform(...).shape[0]` satisfies every
    test above while still clipping a float32 copy of the whole soundtrack into
    existence, one that `_open_audio_stream` discards and `_encode_audio` rebuilds
    moments later. Poisoning it is what makes that unimplementable.
    """

    def poisoned(_waveform):
        raise AssertionError("_planar_waveform must not run to count channels")

    monkeypatch.setattr(media, "_planar_waveform", poisoned)
    audio = {"waveform": torch.zeros(1, 1, SAMPLE_RATE), "sample_rate": SAMPLE_RATE}

    container = av.open(str(tmp_path / "out.mp4"), "w", format="mp4")
    try:
        stream = media._open_audio_stream(container, audio)
    finally:
        container.close()
    assert stream is not None, (
        "_open_audio_stream swallowed the poison as a setup failure and dropped the "
        "soundtrack - it is still building a planar copy to count channels"
    )


# ---- the decode budget environment variable --------------------------------------


@pytest.fixture
def budget_env(monkeypatch):
    """Clear the read-once cache around a case; `monkeypatch` restores the variable.

    Without the clear, seven parametrised values in one session all read the first
    one's answer and six of them assert nothing at all.
    """
    media._reset_decode_budget()
    yield monkeypatch
    monkeypatch.undo()
    media._reset_decode_budget()


@pytest.mark.parametrize(
    "raw",
    ["abc", "0", "-1", "nan", "inf", "-inf", "1e308", "", "   "],
)
def test_an_unusable_budget_leaves_the_default_standing(raw, budget_env):
    """Every one of these must be REJECTED, not clamped or accepted.

    `1e308` is the subtle one: finite and positive, so an input-side check alone lets it
    through, and `1e308 * 2**30` is `inf` - which silently disables every
    `bytes > limit` comparison one layer down. That is why the scaled value is checked
    too, and why this case is here rather than in a comment.
    """
    assert media._parse_decode_budget(raw) == media._MAX_DECODE_BYTES

    budget_env.setenv(media._DECODE_BUDGET_ENV, raw)
    media._reset_decode_budget()
    assert media._decode_budget() == media._MAX_DECODE_BYTES


def test_an_unset_budget_is_the_default(budget_env):
    budget_env.delenv(media._DECODE_BUDGET_ENV, raising=False)
    media._reset_decode_budget()
    assert media._decode_budget() == media._MAX_DECODE_BYTES
    assert media._parse_decode_budget(None) == media._MAX_DECODE_BYTES


def test_the_reset_seam_actually_re_reads_the_environment(budget_env):
    """Proves the matrix above is not vacuous.

    Two DIFFERENT valid values in one session have to yield two different limits. If
    they do not, the cache is never cleared and every rejection case above is asserting
    against whatever the first test in the session happened to set.
    """
    budget_env.setenv(media._DECODE_BUDGET_ENV, "1")
    media._reset_decode_budget()
    one = media._decode_budget()

    budget_env.setenv(media._DECODE_BUDGET_ENV, "2")
    assert media._decode_budget() == one, "the budget is not cached at all"

    media._reset_decode_budget()
    two = media._decode_budget()
    assert (one, two) == (2**30, 2 * 2**30)
    assert one != two


# ---- an audio reference to a video file: load_audio == load_video's soundtrack ------


@pytest.mark.parametrize("trim", [None, TRIM])
@pytest.mark.parametrize("codec", sorted(AUDIO_FORMATS))
def test_load_audio_on_a_video_matches_its_video_soundtrack(clip, codec, trim, monkeypatch):
    """The no-drift guard for `_AudioCollector`: the same file trimmed the same way must
    give the same waveform on `audio_N` as on `video_audio_N`. Named `.mp4`/`.mkv` so
    `_guess_kind` routes it down the video branch, never core's loader."""
    container, ext, _rebases = AUDIO_FORMATS[codec]
    ext = "mp4" if ext == "m4a" else ext
    path = clip(
        f"av_{codec}.{ext}", n=100, fps=25, container=container,
        audio=codec, audio_seconds=AUDIO_SECONDS,
    )

    def core_loader():
        raise AssertionError("a video file must never reach core's audio loader")

    monkeypatch.setattr(media, "_audio_load_fn", core_loader)
    decoded_video = []
    real_pass = media._decode_pass
    monkeypatch.setattr(
        media, "_decode_pass", lambda *a, **k: decoded_video.append(1) or real_pass(*a, **k)
    )

    audio = media.load_audio(path, trim=trim)
    assert decoded_video == [], "load_audio decoded video frames"
    _frames, expected = media.load_video(path, trim=trim)

    assert audio["sample_rate"] == expected["sample_rate"]
    assert audio["waveform"].shape == expected["waveform"].shape
    assert torch.equal(audio["waveform"], expected["waveform"])


def test_load_audio_on_a_video_takes_the_last_decodable_track(tmp_path):
    good = write_two_track_clip(tmp_path / "two.mkv")
    path = break_last_codec_id(good, tmp_path / "two_broken.mkv")

    audio = media.load_audio(path)

    assert audio["sample_rate"] == 44100
    assert float(audio["waveform"].numpy().mean()) == pytest.approx(0.25, abs=1e-3)


def test_load_audio_on_a_silent_video_raises_a_named_error(clip):
    path = clip("silent.mkv", n=30, fps=25, container="matroska", codec="libx264")
    with pytest.raises(ValueError, match="no decodable audio track"):
        media.load_audio(path)


def test_an_audio_only_mp4_probes_as_sound_and_decodes_trimmed(tmp_path):
    """No video stream: the probe must still say `has_audio`, and the trim seek falls
    back to the audio stream itself."""
    path = str(tmp_path / "voice.mp4")
    out = av.open(path, "w", format="mp4")
    try:
        stream = out.add_stream("aac", rate=SAMPLE_RATE, layout="mono")
        ramp = (np.arange(TOTAL_SAMPLES, dtype=np.float32) / TOTAL_SAMPLES)[None, :]
        frame = av.AudioFrame.from_ndarray(ramp, format="fltp", layout="mono")
        frame.sample_rate = SAMPLE_RATE
        for packet in stream.encode(frame):
            out.mux(packet)
        for packet in stream.encode(None):
            out.mux(packet)
    finally:
        out.close()

    assert media.probe(path)["has_audio"] is True
    audio = media.load_audio(path, trim=TRIM)
    assert audio["sample_rate"] == SAMPLE_RATE
    assert mono(audio).shape[0] == WINDOW_SAMPLES
    assert float(mono(audio)[WINDOW_SAMPLES // 2]) == pytest.approx(
        (HEAD_VALUE + TAIL_VALUE) / 2, abs=HEAD_ATOL["aac"]
    )
