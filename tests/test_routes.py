"""Tests for the HTTP routes slice.

No running ComfyUI: routes.py guards its `server`/`folder_paths` imports (see its module
docstring), so importing it here is safe, and `routes.folder_paths` / `routes.media` are
swapped for fakes below. Handlers are called directly with a small request stub instead of
going through aiohttp's router.
"""

import asyncio
import json
import os

import pytest

from minimax_refpack import routes
from minimax_refpack.refs import Reference, ReferenceSet


class FakeRequest:
    """Stub matching the slice of aiohttp.web.Request the handlers actually use."""

    def __init__(self, query=None, match_info=None, body=None):
        self.query = query or {}
        self.match_info = match_info or {}
        self._body = body

    async def json(self):
        if self._body is None:
            raise ValueError("no body")
        return self._body


class FakeFolderPaths:
    """Stands in for the real `folder_paths` module (CU/folder_paths.py)."""

    def __init__(self, input_dir):
        self._input_dir = input_dir

    def get_input_directory(self):
        return self._input_dir

    def filter_files_content_types(self, files, content_types):
        # Mirrors CU/folder_paths.py:229's extension->mimetype->content-type approach
        # closely enough for tests: mimetypes.guess_type is what the real one uses too.
        import mimetypes

        wanted = set(content_types)
        out = []
        for f in files:
            mime, _ = mimetypes.guess_type(f, strict=False)
            if mime and mime.split("/")[0] in wanted:
                out.append(f)
        return out


class FakeMedia:
    """Stands in for minimax_refpack.media, which is a stub owned by another slice."""

    def __init__(self):
        self.thumb_calls = []

    def probe(self, path):
        return {"kind": "image", "width": 4, "height": 4, "fps": None, "duration": None, "has_audio": False}

    def thumbnail_png(self, path, max_edge=256, crop=None, at_seconds=None):
        self.thumb_calls.append({"path": path, "crop": crop, "at_seconds": at_seconds})
        return b"\x89PNG\r\nfake"


def run(coro):
    return asyncio.run(coro)


@pytest.fixture
def input_dir(tmp_path, monkeypatch):
    d = tmp_path / "input"
    d.mkdir()
    monkeypatch.setattr(routes, "folder_paths", FakeFolderPaths(str(d)))
    monkeypatch.setattr(routes, "media", FakeMedia())
    return str(d)


def body_json(resp):
    return json.loads(resp.text)


# ---- pack round trip ---------------------------------------------------------


def test_probe_and_thumb_happy_path(input_dir):
    open(os.path.join(input_dir, "ok.png"), "wb").close()

    resp = run(routes.probe_route(FakeRequest(query={"file": "ok.png"})))
    assert resp.status == 200
    assert body_json(resp)["kind"] == "image"

    resp = run(routes.thumb_route(FakeRequest(query={"file": "ok.png"})))
    assert resp.status == 200
    assert resp.content_type == "image/png"
    assert resp.body == b"\x89PNG\r\nfake"


def test_probe_and_thumb_404_when_absent(input_dir):
    assert run(routes.probe_route(FakeRequest(query={"file": "nope.png"}))).status == 404
    assert run(routes.thumb_route(FakeRequest(query={"file": "nope.png"}))).status == 404


def test_list_files_filters_by_kind(input_dir):
    open(os.path.join(input_dir, "a.png"), "wb").close()
    open(os.path.join(input_dir, "b.mp4"), "wb").close()
    open(os.path.join(input_dir, "c.wav"), "wb").close()

    resp = run(routes.list_files_route(FakeRequest(query={"kind": "image"})))
    assert body_json(resp) == {"files": ["a.png"]}

    resp = run(routes.list_files_route(FakeRequest(query={"kind": "video"})))
    assert body_json(resp) == {"files": ["b.mp4"]}


def test_list_files_sees_one_level_of_subdirectories(input_dir):
    """A pod stages media under `images/` and `videos/`; the picker must list those as
    `sub/name` (the form the node already resolves) without walking any deeper, and a
    directory whose name looks like a file is not a file."""
    os.makedirs(os.path.join(input_dir, "videos", "nested"))
    os.makedirs(os.path.join(input_dir, "images"))
    os.makedirs(os.path.join(input_dir, "dir.mp4"))
    for rel in ("root.mp4", "videos/plate.mp4", "videos/nested/deep.mp4", "images/still.png"):
        open(os.path.join(input_dir, rel), "wb").close()

    resp = run(routes.list_files_route(FakeRequest(query={"kind": "video"})))
    assert body_json(resp) == {"files": ["root.mp4", "videos/plate.mp4"]}

    resp = run(routes.list_files_route(FakeRequest(query={"kind": "image"})))
    assert body_json(resp) == {"files": ["images/still.png"]}


def test_subdirectory_names_round_trip_through_thumb_and_probe(input_dir):
    """What the listing hands out must be accepted back: `_safe_join` takes a
    subdirectory-qualified name, so the tile can be drawn and probed from it."""
    os.makedirs(os.path.join(input_dir, "images"))
    with open(os.path.join(input_dir, "images", "ok.png"), "wb") as fh:
        fh.write(b"\x89PNG")

    resp = run(routes.thumb_route(FakeRequest(query={"file": "images/ok.png"})))
    assert resp.status == 200
    assert routes.media.thumb_calls[-1]["path"].endswith(os.path.join("images", "ok.png"))
    assert run(routes.probe_route(FakeRequest(query={"file": "images/ok.png"}))).status == 200


def test_list_files_400_on_bad_kind(input_dir):
    resp = run(routes.list_files_route(FakeRequest(query={"kind": "mesh"})))
    assert resp.status == 400


# ---- thumb crop / t query params -------------------------------------------------


def test_thumb_passes_crop_and_time_through(input_dir):
    open(os.path.join(input_dir, "v.mp4"), "wb").close()

    resp = run(routes.thumb_route(FakeRequest(query={"file": "v.mp4", "crop": "0.1,0.2,0.5,0.5", "t": "2.5"})))

    assert resp.status == 200
    call = routes.media.thumb_calls[-1]
    assert call["crop"] == [0.1, 0.2, 0.5, 0.5]
    assert call["at_seconds"] == 2.5


def test_thumb_without_edit_params_is_unchanged(input_dir):
    open(os.path.join(input_dir, "ok.png"), "wb").close()

    resp = run(routes.thumb_route(FakeRequest(query={"file": "ok.png"})))

    assert resp.status == 200
    call = routes.media.thumb_calls[-1]
    assert call["crop"] is None
    assert call["at_seconds"] is None


@pytest.mark.parametrize("bad", [
    "",                  # explicit but empty
    "a,b,c,d",           # not numbers
    "0.1,0.2,0.5",       # wrong arity
    "0.6,0.5,0.6,0.6",   # past the right edge
    "-0.1,0,0.5,0.5",    # negative origin
    "0,0,0,1",           # zero area
])
def test_thumb_400_on_a_bad_crop(input_dir, bad):
    open(os.path.join(input_dir, "ok.png"), "wb").close()
    resp = run(routes.thumb_route(FakeRequest(query={"file": "ok.png", "crop": bad})))
    assert resp.status == 400


@pytest.mark.parametrize("bad", ["", "abc", "-1", "nan", "inf"])
def test_thumb_400_on_a_bad_time(input_dir, bad):
    open(os.path.join(input_dir, "ok.png"), "wb").close()
    resp = run(routes.thumb_route(FakeRequest(query={"file": "ok.png", "t": bad})))
    assert resp.status == 400


# ---- traversal table ----------------------------------------------------------

TRAVERSAL_ATTEMPTS = ["../../etc/passwd", "/etc/passwd", "..%2f..%2fetc", "subdir/../../x"]


@pytest.mark.parametrize("bad", TRAVERSAL_ATTEMPTS)
def test_probe_refuses_traversal(input_dir, bad):
    assert run(routes.probe_route(FakeRequest(query={"file": bad}))).status == 404


@pytest.mark.parametrize("bad", TRAVERSAL_ATTEMPTS)
def test_thumb_refuses_traversal(input_dir, bad):
    assert run(routes.thumb_route(FakeRequest(query={"file": bad}))).status == 404


def test_a_plain_valid_name_is_accepted_by_every_route(input_dir):
    open(os.path.join(input_dir, "ok.png"), "wb").close()

    assert run(routes.probe_route(FakeRequest(query={"file": "ok.png"}))).status == 200
    assert run(routes.thumb_route(FakeRequest(query={"file": "ok.png"}))).status == 200


def test_the_pack_store_is_gone():
    """Configs are files on the user's machine (2026-08-15); the server-side pack store
    was dead code and was deleted. This pins it — a route added back here would be a
    second, unused mechanism for saving a reference set."""
    paths = {path for _method, path, _handler in routes._ROUTES}
    assert not any("packs" in path for path in paths)
    assert not hasattr(routes, "save_pack_route")
    assert not hasattr(routes, "get_pack")


# ---- system prompt: read-only default ------------------------------------------


def test_no_write_route_exists_for_system_prompt():
    paths = {(method, path) for method, path, _handler in routes._ROUTES}
    assert ("GET", "/minimax_refpack/system_prompt") in paths
    assert ("POST", "/minimax_refpack/system_prompt") not in paths
    assert ("PUT", "/minimax_refpack/system_prompt") not in paths
    assert ("DELETE", "/minimax_refpack/system_prompt") not in paths


# ---- import safety -------------------------------------------------------------


def test_importing_routes_without_comfyui_does_not_raise():
    # If this test file imported cleanly, the module-level guard already proved this -
    # but assert the fallback state explicitly so a future edit can't silently import
    # a real `server`/`folder_paths` and hide the guard's absence.
    import importlib

    reloaded = importlib.reload(routes)
    assert reloaded.PromptServer is None or reloaded.PromptServer.instance is None


def test_list_files_for_audio_includes_video_files(input_dir):
    """An audio reference may use a video's soundtrack, so the audio picker lists both."""
    for name in ("a.png", "b.mp4", "c.wav"):
        open(os.path.join(input_dir, name), "wb").close()

    resp = run(routes.list_files_route(FakeRequest(query={"kind": "audio"})))
    assert sorted(body_json(resp)["files"]) == ["b.mp4", "c.wav"]
