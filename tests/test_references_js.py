"""The browser's tag rule agrees with refs.py's, reference set by reference set.

web/lib/references.js's assignTags draws the tags on the tiles; refs.py's
ReferenceSet.assign_tags writes them into the prompt MiniMax receives. A tile reading
<Audio 2> over a soundtrack the prompt calls <Audio 1> sends the prompt writer the wrong
binding with nothing failing anywhere, so both run on the same envelopes here.

JS reads each envelope through fromReferencesEnvelope, as the node does, so a video
without a use_soundtrack key takes the same default (on) on both sides.
"""

import json

import pytest
from _js import evaluate, requires_node

from minimax_refpack.refs import ReferenceSet


def _img(file):
    return {"kind": "image", "file": file}


def _vid(file, sound=None):
    ref = {"kind": "video", "file": file}
    if sound is not None:
        ref["use_soundtrack"] = sound
    return ref


def _aud(file):
    return {"kind": "audio", "file": file}


CASES = {
    "empty": [],
    "images only": [_img("a.png"), _img("b.jpg"), _img("c.webp")],
    "soundtrack before standalone audio": [_vid("clip.mp4", True), _aud("vo.wav")],
    "muted video leaves the audio counter alone": [_vid("clip.mp4", False), _aud("vo.wav")],
    "absent use_soundtrack defaults on": [_vid("clip.mp4"), _aud("vo.wav")],
    "everything at once": [
        _img("face.png"), _vid("a.mp4", True), _vid("b.mp4", False), _vid("c.mp4", True),
        _aud("one.wav"), _img("hat.png"), _aud("two.flac"),
    ],
}


def _python(references):
    tagged = ReferenceSet.from_obj({"references": references}).assign_tags()
    return [[t.kind, t.ref.file, t.tag, t.audio_tag] for t in tagged]


def _js(references):
    return evaluate(
        "(() => { const t = assignTags(fromReferencesEnvelope("
        + json.dumps({"references": references})
        + ")); return ["
        "...t.images.map(x => ['image', x.ref.file, x.tag, null]),"
        "...t.videos.map(x => ['video', x.ref.file, x.tag, x.audioTag]),"
        "...t.audios.map(x => ['audio', x.ref.file, x.tag, null])]; })()",
        "references",
    )


@requires_node
@pytest.mark.parametrize("references", CASES.values(), ids=CASES.keys())
def test_browser_and_backend_assign_the_same_tags(references):
    assert _js(references) == _python(references)


def test_the_cases_exercise_both_audio_sources():
    tags = {value for refs in CASES.values() for row in _python(refs) for value in row[2:]}
    assert {"<Audio 3>", "<Video 3>"} <= tags
