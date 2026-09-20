"""Tile reorder: the pure move and midpoint-crossing index math, as shipped.

The block between web/refpack.js's MMRP-REORDER markers is extracted and executed
under node, so these tests run the browser code itself. The pointer listeners,
grab cursor and Escape-to-cancel path need a real canvas and are not covered.
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


def _extract(marker: str = "MMRP-REORDER") -> str:
    text = REFPACK_JS.read_text()
    start = text.find(f"// >>> {marker}")
    end = text.find(f"// <<< {marker}")
    assert start != -1 and end != -1, f"the {marker} markers are gone from web/refpack.js"
    return text[start:end]


def run_js(expression: str):
    assert NODE is not None
    script = _extract() + f"\nconsole.log(JSON.stringify({expression}));\n"
    proc = subprocess.run(
        [NODE, "--input-type=module", "-e", script],
        capture_output=True, text=True, check=False,
    )
    assert proc.returncode == 0, f"node failed: {proc.stderr}"
    return json.loads(proc.stdout.strip())


def _move(refs, kind, src, dst):
    return run_js(
        "(() => { const refs = " + json.dumps(refs) + ";"
        " const before = JSON.stringify(refs);"
        f" const out = moveRefInKind(refs, {json.dumps(kind)}, {src}, {dst});"
        " return {out, untouched: JSON.stringify(refs) === before}; })()"
    )


def _pictures(*names):
    return {
        "images": [{"file": n} for n in names],
        "videos": [],
        "audios": [],
    }


# The live canvas calls slotIndexAtX with CL.x0 / CL.tile / CL.gap. Tests use the
# same numbers so a geometry change that forgets the call site is a test failure,
# not a silent midpoint drift.
X0, TILE, GAP = 10, 131, 6


def test_the_markers_survive_in_the_shipped_file():
    assert "function moveRefInKind" in _extract()
    assert "function slotIndexAtX" in _extract()


def test_live_canvas_passes_the_shipped_tile_geometry():
    text = REFPACK_JS.read_text()
    assert "slotIndexAtX(pos.x, count, CL.x0, CL.tile, CL.gap)" in text
    assert "if (hit.type === \"tile\") beginReorder(node, e, hit);" in text
    assert "x0: 10," in text
    assert "tile: 131," in text
    assert "gap: 6," in text


@requires_node
def test_picture_one_dragged_onto_picture_two_swaps_them():
    got = _move(_pictures("one.png", "two.png"), "image", 0, 1)
    assert got["untouched"]
    assert got["out"]["error"] is None
    files = [r["file"] for r in got["out"]["refs"]["images"]]
    assert files == ["two.png", "one.png"]


@requires_node
def test_picture_two_dragged_onto_picture_one_swaps_them():
    got = _move(_pictures("one.png", "two.png"), "image", 1, 0)
    assert got["out"]["error"] is None
    files = [r["file"] for r in got["out"]["refs"]["images"]]
    assert files == ["two.png", "one.png"]


@requires_node
def test_dragging_the_first_of_three_to_the_last_shifts_the_middle():
    """Neighbour-swap as the pointer walks the row, not a pairwise exchange with the
    drop target: [a, b, c] dragged 0 -> 2 is [b, c, a], not [c, b, a]."""
    got = _move(_pictures("a.png", "b.png", "c.png"), "image", 0, 2)
    files = [r["file"] for r in got["out"]["refs"]["images"]]
    assert files == ["b.png", "c.png", "a.png"]


@requires_node
def test_crop_and_roles_travel_with_the_file():
    refs = {
        "images": [
            {"file": "a.png", "roles": ["reference_generation"], "crop": [0, 0, 0.5, 0.5]},
            {"file": "b.png", "roles": []},
        ],
        "videos": [],
        "audios": [],
    }
    got = _move(refs, "image", 0, 1)
    images = got["out"]["refs"]["images"]
    assert images[0]["file"] == "b.png"
    assert images[1] == {
        "file": "a.png", "roles": ["reference_generation"], "crop": [0, 0, 0.5, 0.5],
    }


@requires_node
def test_videos_and_audio_use_the_same_move():
    refs = {
        "images": [{"file": "keep.png"}],
        "videos": [{"file": "v1.mp4"}, {"file": "v2.mp4"}],
        "audios": [{"file": "a1.wav"}, {"file": "a2.wav"}, {"file": "a3.wav"}],
    }
    videos = _move(refs, "video", 0, 1)
    assert [r["file"] for r in videos["out"]["refs"]["videos"]] == ["v2.mp4", "v1.mp4"]
    assert [r["file"] for r in videos["out"]["refs"]["images"]] == ["keep.png"]
    audios = _move(refs, "audio", 2, 0)
    assert [r["file"] for r in audios["out"]["refs"]["audios"]] == [
        "a3.wav", "a1.wav", "a2.wav",
    ]


@requires_node
def test_same_index_is_a_noop_that_returns_the_input_object():
    got = run_js(
        "(() => { const refs = {images: [{file: 'a.png'}, {file: 'b.png'}],"
        " videos: [], audios: []};"
        " const out = moveRefInKind(refs, 'image', 1, 1);"
        " return {same: out.refs === refs, error: out.error}; })()"
    )
    assert got == {"same": True, "error": None}


@requires_node
def test_out_of_range_is_an_error_and_untouched():
    got = _move(_pictures("a.png", "b.png"), "image", 0, 2)
    assert got["untouched"]
    assert got["out"]["error"] == "No such reference."
    assert [r["file"] for r in got["out"]["refs"]["images"]] == ["a.png", "b.png"]


@requires_node
def test_slot_index_swaps_when_the_pointer_crosses_the_midpoint():
    # Two tiles: centres 75.5 and 212.5, midpoint 144. Math.round half-up.
    assert run_js(f"slotIndexAtX(10, 2, {X0}, {TILE}, {GAP})") == 0
    assert run_js(f"slotIndexAtX(143, 2, {X0}, {TILE}, {GAP})") == 0
    assert run_js(f"slotIndexAtX(144, 2, {X0}, {TILE}, {GAP})") == 1
    assert run_js(f"slotIndexAtX(300, 2, {X0}, {TILE}, {GAP})") == 1


@requires_node
def test_slot_index_clamps_and_ignores_a_single_tile():
    assert run_js(f"slotIndexAtX(-40, 3, {X0}, {TILE}, {GAP})") == 0
    assert run_js(f"slotIndexAtX(2000, 3, {X0}, {TILE}, {GAP})") == 2
    assert run_js(f"slotIndexAtX(2000, 1, {X0}, {TILE}, {GAP})") == 0
    assert run_js(f"slotIndexAtX(2000, 0, {X0}, {TILE}, {GAP})") == 0
