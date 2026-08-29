"""One decode per reference, per build.

`build()` and the prompt writer both need the same pixels, and before this change they
each fetched their own. For ONE edited video reference with a soundtrack, on the
OpenRouter path, the baseline was:

  1. `media.load_video` for the node's sockets (seek, decode, resample, crop);
  2. an independent PyAV decode inside `media._transcode_window`, re-encoding the window
     at SOURCE resolution and SOURCE frame rate for the VLM;
  3. `media.load_video` a second time with identical arguments, purely to reach `audio` -
     its frames were thrown away into `_,`.

So: 2 `load_video` calls plus 1 separate decode. Images and trimmed standalone audio
were loaded twice each for the same reason.

The seam matters here. `tests/test_nodes.py`'s usual `_stub_prompt` replaces
`write_prompt` wholesale, so under it the prompt path never runs at all - the counter
would see one socket load and no transcode, which reads as "already deduplicated"
against unmodified code. These tests stub only the classifier/chat transport and the
encoder, so the REAL `_build_content` executes.

Both encoder names are patched with `raising=False` on purpose. `_transcode_window` is
gone after this change, and patching a name that no longer exists costs nothing; what it
buys is that this file reads identically before and after, and that the assertion below -
the survivor is handed FRAMES, never a path - fails loudly if a decode-from-the-file ever
comes back.
"""

import json
import sys
import types

import numpy as np
import pytest

from minimax_refpack import media, nodes, prompt


@pytest.fixture
def fake_folder_paths(tmp_path, monkeypatch):
    module = types.ModuleType("folder_paths")
    setattr(module, "get_input_directory", lambda: str(tmp_path))
    monkeypatch.setitem(sys.modules, "folder_paths", module)
    return tmp_path


class _FakeResponse:
    status_code = 200
    text = "ok"

    def json(self):
        return {"choices": [{"message": {"content": "a written prompt"}}]}


class Decodes:
    """How many times each loader actually ran, and what the VLM encoder was handed."""

    def __init__(self):
        self.image = 0
        self.video = 0
        self.audio = 0
        self.encodes: list[tuple[object, ...]] = []


@pytest.fixture
def counted(monkeypatch) -> Decodes:
    """Every real decode, counted at the module both nodes.py and prompt.py import.

    `nodes.media` and `prompt.media` are the same module object, so one patch covers
    both call sites - which is the whole point: the question is how many times a file is
    decoded for ONE build, not how many times each caller asks for it.
    """
    seen = Decodes()

    def load_image(path, crop=None, max_edge=0):
        seen.image += 1
        return np.random.rand(1, 8, 8, 3).astype(np.float32)

    def load_video(path, target_fps=24, crop=None, trim=None):
        seen.video += 1
        frames = np.random.rand(12, 8, 8, 3).astype(np.float32)
        audio = {
            "waveform": np.random.uniform(-1, 1, size=(1, 2, 480)).astype(np.float32),
            "sample_rate": 24000,
        }
        return frames, audio

    def load_audio(path, trim=None):
        seen.audio += 1
        return {
            "waveform": np.random.uniform(-1, 1, size=(1, 1, 480)).astype(np.float32),
            "sample_rate": 24000,
        }

    def fake_encode(*args, **kwargs):
        seen.encodes.append(args)
        return b"\x00\x00\x00 ftypmp42 fake", True

    def fake_transcode(*args, **kwargs):
        seen.encodes.append(args)
        return b"\x00\x00\x00 ftypmp42 fake"

    monkeypatch.setattr(media, "load_image", load_image)
    monkeypatch.setattr(media, "load_video", load_video)
    monkeypatch.setattr(media, "load_audio", load_audio)
    monkeypatch.setattr(media, "encode_reference_mp4", fake_encode, raising=False)
    monkeypatch.setattr(media, "_transcode_window", fake_transcode, raising=False)
    monkeypatch.setattr(prompt.requests, "post", lambda *a, **k: _FakeResponse())
    return seen


def _build(tmp_path, references, **kwargs):
    for ref in references:
        (tmp_path / ref["file"]).write_bytes(b"x")
    return nodes.MiniMaxH3ReferencePack().build(
        direction="a steer",
        openrouter_api_key="k",
        openrouter_model="m",
        references_json=json.dumps({"references": references}),
        job_type="standard",
        **kwargs,
    )


def test_an_edited_clip_with_sound_is_decoded_once_for_the_whole_build(
    fake_folder_paths, counted
):
    _build(
        fake_folder_paths,
        [
            {
                "kind": "video",
                "file": "clip.mp4",
                "use_soundtrack": True,
                "trim": [1.0, 3.0],
            },
        ],
    )

    assert counted.video == 1, (
        "the sockets and the VLM payload must share one decode; the baseline was 2 - the "
        "second reached only for `audio` and threw its frames away"
    )
    assert len(counted.encodes) == 1, (
        f"expected one VLM encode, got {len(counted.encodes)}"
    )
    assert not isinstance(counted.encodes[0][0], str), (
        "the VLM copy must be encoded from the frames already prepared for the sockets, "
        f"not decoded again from a path - got {counted.encodes[0][0]!r}"
    )


def test_an_image_reference_is_loaded_once(fake_folder_paths, counted):
    _build(fake_folder_paths, [{"kind": "image", "file": "a.jpg"}])

    assert counted.image == 1, (
        "the socket load and the VLM's copy must be the same load; they differed only "
        "because the prompt writer dropped max_reference_edge"
    )


def test_a_trimmed_standalone_audio_reference_is_decoded_once(
    fake_folder_paths, counted
):
    _build(
        fake_folder_paths,
        [
            {"kind": "audio", "file": "vo.wav", "trim": [1.0, 2.0]},
        ],
    )

    assert counted.audio == 1


def test_an_untouched_clip_is_also_decoded_exactly_once(fake_folder_paths, counted):
    """The old fast path decoded nothing at all here - it inlined the whole original
    file. That is why this case is asserted rather than assumed: it is the one reference
    shape that GAINS work, and it must gain exactly one decode, not two."""
    _build(
        fake_folder_paths,
        [
            {"kind": "video", "file": "plate.mp4", "use_soundtrack": True},
        ],
    )

    assert counted.video == 1
    assert len(counted.encodes) == 1
