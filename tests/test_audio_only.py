""""Use audio only": the pure video -> audio reference conversion, as shipped.

The block between web/refpack.js's MMRP-AUDIO-ONLY markers is extracted and executed
under node, so these tests run the browser code itself. The chip, the modal button,
applyRefs and the plan-button error state need a real browser and are not covered.
"""

import json
import pathlib
import shutil
import subprocess

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
REFPACK_JS = REPO_ROOT / "web" / "refpack.js"

NODE = shutil.which("node")
requires_node = pytest.mark.skipif(NODE is None, reason="node is not installed")


def _extract(marker: str) -> str:
    text = REFPACK_JS.read_text()
    start = text.find(f"// >>> {marker}")
    end = text.find(f"// <<< {marker}")
    assert start != -1 and end != -1, f"the {marker} markers are gone from web/refpack.js"
    return text[start:end]


def run_js(expression: str, *markers: str):
    script = "".join(_extract(m) for m in markers or ("MMRP-AUDIO-ONLY",))
    script += f"\nconsole.log(JSON.stringify({expression}));\n"
    proc = subprocess.run(
        [NODE, "--input-type=module", "-e", script],
        capture_output=True, text=True, check=False,
    )
    assert proc.returncode == 0, f"node failed: {proc.stderr}"
    return json.loads(proc.stdout.strip())


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


@requires_node
def test_the_block_runs_with_nothing_else_around_it():
    assert run_js("typeof videoToAudioOnly") == "function"
