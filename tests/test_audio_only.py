""""Use audio only": the pure video -> audio reference conversion, as shipped.

videoToAudioOnly is imported from web/lib/references.js under node, so these tests run
the browser code itself. The chip, the modal button,
applyRefs and the plan-button error state need a real browser and are not covered.
"""

import json

import pytest
from _js import evaluate, requires_node


def run_js(expression: str):
    return evaluate(expression, "references")


def _convert(refs, index, cap=3):
    return run_js(
        "(() => { const refs = " + json.dumps(refs) + ";"
        " const before = JSON.stringify(refs);"
        f" const out = videoToAudioOnly(refs, {index}, {cap});"
        " return {out, untouched: JSON.stringify(refs) === before}; })()"
    )


PLATE = {
    "file": "plate.mp4",
    "use_soundtrack": True,
    "primary": True,
    "roles": ["video_editing", "audio_reuse", "reference_generation", "audio_reference"],
    "crop": [0, 0, 0.5, 0.5],
    "trim": [1.5, 4.0],
    "missing": False,
}


@requires_node
def test_moves_the_video_to_the_audio_row_keeping_trim_and_audio_roles():
    refs = {
        "images": [{"file": "a.png", "roles": []}],
        "videos": [PLATE, {"file": "other.mp4", "use_soundtrack": True, "roles": []}],
        "audios": [{"file": "vo.wav", "roles": []}],
        "taskPlan": None,
    }
    got = _convert(refs, 0)

    assert got["untouched"], "videoToAudioOnly mutated its input"
    assert got["out"]["error"] is None
    out = got["out"]["refs"]
    assert out["images"] == refs["images"]
    assert [v["file"] for v in out["videos"]] == ["other.mp4"]
    assert out["audios"] == [
        {"file": "vo.wav", "roles": []},
        {"file": "plate.mp4", "roles": ["audio_reuse", "audio_reference"],
         "missing": False, "trim": [1.5, 4.0]},
    ]
    assert "crop" not in out["audios"][1]
    assert "primary" not in out["audios"][1]


@requires_node
def test_an_untrimmed_muted_video_becomes_a_bare_audio_ref():
    video = {"file": "clip.mp4", "use_soundtrack": False, "roles": ["reference_generation"]}
    got = _convert({"images": [], "videos": [video], "audios": [], "taskPlan": None}, 0)

    assert got["out"]["refs"]["audios"] == [{"file": "clip.mp4", "roles": [], "missing": False}]


@requires_node
def test_refuses_when_the_audio_row_is_full():
    refs = {
        "images": [],
        "videos": [PLATE],
        "audios": [{"file": f"a{i}.wav", "roles": []} for i in range(3)],
        "taskPlan": None,
    }
    got = _convert(refs, 0, cap=3)

    assert got["out"]["error"]
    assert got["out"]["refs"] == refs


@pytest.mark.parametrize("index", [-1, 1, 0.5])
@requires_node
def test_refuses_a_bad_index(index):
    refs = {"images": [], "videos": [PLATE], "audios": [], "taskPlan": None}
    got = _convert(refs, index)

    assert got["out"]["error"]
    assert got["out"]["refs"] == refs
