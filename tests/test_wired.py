"""Upstream slot overrides must be shared by the renderer and the prompt writer."""
import json
import sys
import types

import pytest
import torch

from minimax_refpack import nodes, refs, wired


@pytest.fixture
def input_dir(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "folder_paths", types.SimpleNamespace(get_input_directory=lambda: str(tmp_path)))
    return str(tmp_path)


def test_image_override_preserves_other_slots_and_prompt(input_dir, monkeypatch):
    image = torch.zeros(1, 8, 8, 3)
    monkeypatch.setattr(nodes.media, "load_image", lambda *a, **k: "fallback")
    # Missing old file is allowed when replaced, but the other slot must still resolve.
    from pathlib import Path
    Path(input_dir, "second.png").touch()
    raw = json.dumps([{"kind": "image", "file": "missing.png", "roles": ["reference_generation"]},
                      {"kind": "image", "file": "second.png"}])
    out = dict(zip(refs.output_names(), nodes.MiniMaxH3ReferencePack().build(
        references_json=raw, image_1=image, direction="fixed", prompt_provider="none")))
    assert out["image_1"] is image
    assert out["image_2"] == "fallback"
    assert out["prompt"] == "fixed"


def test_auto_prompt_reads_same_upstream_image(input_dir, monkeypatch):
    image = torch.ones(1, 8, 8, 3)
    def write(**kwargs):
        reference = kwargs["references"].assign_tags()[0]
        assert reference.tag == "<Picture 1>"
        import os
        assert kwargs["cache"].image(os.path.join(input_dir, reference.file)) is image
        return "written"
    monkeypatch.setattr(nodes.prompt, "write_prompt", write)
    out = nodes.MiniMaxH3ReferencePack().build(image_1=image, direction="idea")
    assert out[refs.slot_index("prompt")] == "written"


def test_video_and_audio_share_cache(input_dir):
    frames = torch.zeros(22, 8, 8, 3)
    audio = {"waveform": torch.zeros(1, 1, 100), "sample_rate": 24000}
    reference_set, cache = wired.resolve(refs.ReferenceSet(), {"video_1": frames, "video_audio_1": audio}, input_dir)
    import os
    tagged = reference_set.assign_tags()[0]
    actual_frames, actual_audio = cache.video(os.path.join(input_dir, tagged.file))
    assert actual_frames is frames and actual_audio is audio
    assert tagged.audio_tag == "<Audio 1>"


def test_gap_and_batch_rejected(input_dir):
    with pytest.raises(ValueError, match="preceding"):
        wired.resolve(refs.ReferenceSet(), {"image_2": torch.zeros(1, 8, 8, 3)}, input_dir)
    with pytest.raises(ValueError, match="one picture"):
        wired.resolve(refs.ReferenceSet(), {"image_1": torch.zeros(2, 8, 8, 3)}, input_dir)


def test_upstream_bypasses_prefetch(input_dir, monkeypatch):
    monkeypatch.setattr(nodes.PREFETCHER, "consume", lambda key: pytest.fail("must not prefetch tensors"))
    nodes.MiniMaxH3ReferencePack().build(image_1=torch.zeros(1, 8, 8, 3), prompt_provider="none")


def test_audio_only_override_keeps_internal_video(input_dir, monkeypatch):
    frames = torch.zeros(22, 8, 8, 3)
    audio = {"waveform": torch.zeros(1, 1, 100), "sample_rate": 24000}
    monkeypatch.setattr(nodes.media, "load_video", lambda *a, **k: (frames, None))
    reference_set = refs.ReferenceSet([refs.Reference("video", "old.mp4", primary=True)])
    result, cache = wired.resolve(reference_set, {"video_audio_1": audio}, input_dir)
    import os
    reference = result.of_kind("video")[0]
    assert reference.primary
    actual_frames, actual_audio = cache.video(os.path.join(input_dir, reference.file))
    assert actual_frames is frames and actual_audio is audio


def test_standalone_audio_sent_from_cache(input_dir):
    from minimax_refpack import prompt
    audio = {"waveform": torch.zeros(1, 1, 240), "sample_rate": 24000}
    result, cache = wired.resolve(refs.ReferenceSet(), {"audio_1": audio}, input_dir)
    import os
    reference = result.of_kind("audio")[0]
    assert cache.audio(os.path.join(input_dir, reference.file)) is audio
    assert reference.file.endswith(".wired")
    assert prompt._vlm_wav_b64(audio)
