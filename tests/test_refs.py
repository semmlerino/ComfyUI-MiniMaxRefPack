"""Tests for the frozen reference contract.

The tag-order cases below are the whole point of this module: they encode what
ComfyUI's MiniMaxH3ReferenceToVideo actually does at nodes_minimax_h3.py:216-274.
If one of these changes, the node is lying to the user about <Picture N> / <Audio N>.
"""

import json
import os

import pytest

from minimax_refpack.refs import (
    MAX_AUDIOS,
    MAX_IMAGES,
    MAX_VIDEOS,
    Reference,
    ReferenceError,
    ReferenceSet,
    TaskPlan,
    empty_outputs,
    output_names,
    output_types,
    slot_index,
)


def img(name):
    return Reference(kind="image", file=name)


def vid(name, sound=False):
    return Reference(kind="video", file=name, use_soundtrack=sound)


def aud(name):
    return Reference(kind="audio", file=name)


# ---- tag numbering ---------------------------------------------------------


def test_images_are_numbered_in_order():
    s = ReferenceSet([img("a.jpg"), img("b.png"), img("c.webp")])
    assert [t.tag for t in s.assign_tags()] == ["<Picture 1>", "<Picture 2>", "<Picture 3>"]


def test_slot_equals_ordinal_because_the_node_compacts():
    s = ReferenceSet([img("a.jpg"), img("b.png")])
    assert [t.slot for t in s.assign_tags()] == [1, 2]


def test_video_soundtrack_takes_an_audio_number_before_standalone_audio():
    # nodes_minimax_h3.py:259 appends the soundtrack's audio item BEFORE the video at :263,
    # and standalone ref_audios come last at :273.
    s = ReferenceSet([vid("clip.mp4", sound=True), aud("vo.wav")])
    tagged = {t.file: t for t in s.assign_tags()}
    assert tagged["clip.mp4"].tag == "<Video 1>"
    assert tagged["clip.mp4"].audio_tag == "<Audio 1>"
    assert tagged["vo.wav"].tag == "<Audio 2>"


def test_standalone_audio_added_first_still_numbers_after_a_soundtrack():
    # insertion order in the UI must not change the numbering: kind order wins
    s = ReferenceSet([aud("vo.wav"), vid("clip.mp4", sound=True)])
    tagged = {t.file: t for t in s.assign_tags()}
    assert tagged["clip.mp4"].audio_tag == "<Audio 1>"
    assert tagged["vo.wav"].tag == "<Audio 2>"


def test_video_without_soundtrack_takes_no_audio_number():
    s = ReferenceSet([vid("a.mp4"), vid("b.mp4", sound=True), aud("vo.wav")])
    tagged = {t.file: t for t in s.assign_tags()}
    assert tagged["a.mp4"].audio_tag is None
    assert tagged["a.mp4"].tag == "<Video 1>"
    assert tagged["b.mp4"].tag == "<Video 2>"
    assert tagged["b.mp4"].audio_tag == "<Audio 1>"
    assert tagged["vo.wav"].tag == "<Audio 2>"


def test_videos_and_audio_are_numbered_independently():
    s = ReferenceSet([vid("a.mp4", sound=True), vid("b.mp4", sound=True)])
    tagged = {t.file: t for t in s.assign_tags()}
    assert (tagged["a.mp4"].tag, tagged["a.mp4"].audio_tag) == ("<Video 1>", "<Audio 1>")
    assert (tagged["b.mp4"].tag, tagged["b.mp4"].audio_tag) == ("<Video 2>", "<Audio 2>")


def test_full_house_numbering():
    s = ReferenceSet(
        [img("i1.jpg"), img("i2.jpg"), vid("v1.mp4", sound=True), vid("v2.mp4"), aud("a1.wav")]
    )
    got = {t.file: (t.tag, t.audio_tag) for t in s.assign_tags()}
    assert got == {
        "i1.jpg": ("<Picture 1>", None),
        "i2.jpg": ("<Picture 2>", None),
        "v1.mp4": ("<Video 1>", "<Audio 1>"),
        "v2.mp4": ("<Video 2>", None),
        "a1.wav": ("<Audio 2>", None),
    }


def test_tag_map_joins_a_videos_two_tags():
    s = ReferenceSet([vid("v1.mp4", sound=True)])
    assert s.tag_map() == {"v1.mp4": "<Video 1> <Audio 1>"}


def test_empty_set_tags_to_nothing():
    assert ReferenceSet().assign_tags() == []


# ---- caps ------------------------------------------------------------------


@pytest.mark.parametrize(
    "kind,cap,maker", [("image", MAX_IMAGES, img), ("video", MAX_VIDEOS, vid), ("audio", MAX_AUDIOS, aud)]
)
def test_cap_is_hard(kind, cap, maker):
    ok = ReferenceSet([maker(f"{i}.x") for i in range(cap)])
    ok.validate()
    over = ReferenceSet([maker(f"{i}.x") for i in range(cap + 1)])
    with pytest.raises(ReferenceError) as e:
        over.validate()
    assert str(cap) in str(e.value)


def test_caps_match_the_minimax_node():
    # nodes_minimax_h3.py:183,187,191,195
    assert (MAX_IMAGES, MAX_VIDEOS, MAX_AUDIOS) == (9, 3, 3)


# ---- json round trip -------------------------------------------------------


def test_json_round_trip():
    s = ReferenceSet([img("a.jpg"), vid("v.mp4", sound=True), aud("s.wav")])
    again = ReferenceSet.from_json(s.to_json())
    assert [r.to_dict() for r in again.references] == [r.to_dict() for r in s.references]


def test_legacy_json_round_trip_does_not_gain_task_metadata():
    raw = '{"references": [{"kind": "image", "file": "a.jpg"}]}'
    assert json.loads(ReferenceSet.from_json(raw).to_json()) == json.loads(raw)


def test_roles_and_task_plan_round_trip_without_moving_tags():
    s = ReferenceSet(
        [
            Reference(
                kind="image",
                file="face.png",
                roles=["reference_generation", "keyframe_completion"],
            ),
            Reference(
                kind="video",
                file="plate.mp4",
                use_soundtrack=True,
                primary=True,
                roles=["video_editing", "audio_reuse"],
            ),
        ],
        task_plan=TaskPlan(mode="explicit", specialization="character_replacement"),
    )

    again = ReferenceSet.from_json(s.to_json())

    assert again.task_plan == s.task_plan
    assert [r.roles for r in again.references] == [r.roles for r in s.references]
    assert [r.primary for r in again.references] == [False, True]
    assert [(t.tag, t.audio_tag) for t in again.assign_tags()] == [
        ("<Picture 1>", None),
        ("<Video 1>", "<Audio 1>"),
    ]


@pytest.mark.parametrize(
    "reference",
    [
        {"kind": "image", "file": "a.png", "roles": "reference_generation"},
        {"kind": "image", "file": "a.png", "roles": ["made_up"]},
        {"kind": "image", "file": "a.png", "roles": ["reference_generation", "reference_generation"]},
        {"kind": "audio", "file": "a.wav", "roles": ["video_editing"]},
    ],
)
def test_malformed_or_inapplicable_roles_are_rejected(reference):
    with pytest.raises(ReferenceError):
        ReferenceSet.from_obj({"references": [reference]})


@pytest.mark.parametrize(
    "task_plan",
    [
        "explicit",
        {"mode": "guess"},
        {"mode": "auto", "specialization": "character_replacement"},
        {"mode": "explicit", "specialization": "face_swap"},
    ],
)
def test_malformed_task_plan_is_rejected(task_plan):
    with pytest.raises(ReferenceError):
        ReferenceSet.from_obj({"references": [], "task_plan": task_plan})


@pytest.mark.parametrize("raw", ["", "   ", None])
def test_blank_widget_is_an_empty_set(raw):
    assert ReferenceSet.from_json(raw).is_empty()


def test_bare_list_is_accepted():
    s = ReferenceSet.from_json(json.dumps([{"kind": "image", "file": "a.jpg"}]))
    assert s.counts() == {"image": 1, "video": 0, "audio": 0}


def test_broken_json_is_a_reference_error():
    with pytest.raises(ReferenceError):
        ReferenceSet.from_json("{not json")


def test_unknown_kind_is_rejected():
    with pytest.raises(ReferenceError):
        ReferenceSet.from_json(json.dumps([{"kind": "mesh", "file": "a.obj"}]))


def test_missing_filename_is_rejected():
    with pytest.raises(ReferenceError):
        ReferenceSet.from_json(json.dumps([{"kind": "image", "file": "  "}]))


def test_use_soundtrack_only_sticks_to_videos():
    s = ReferenceSet.from_json(
        json.dumps([{"kind": "image", "file": "a.jpg", "use_soundtrack": True}])
    )
    assert s.references[0].use_soundtrack is False


def test_primary_is_optional_video_only_metadata():
    raw = {
        "references": [
            {"kind": "video", "file": "plate.mp4", "primary": True},
            {"kind": "video", "file": "guide.mp4"},
        ]
    }
    again = ReferenceSet.from_obj(raw)

    assert [reference.primary for reference in again.references] == [True, False]
    assert json.loads(again.to_json()) == {
        "references": [
            {
                "kind": "video",
                "file": "plate.mp4",
                "use_soundtrack": True,
                "primary": True,
            },
            {"kind": "video", "file": "guide.mp4", "use_soundtrack": True},
        ]
    }


@pytest.mark.parametrize(
    "reference",
    [
        {"kind": "video", "file": "plate.mp4", "primary": "yes"},
        {"kind": "image", "file": "face.png", "primary": True},
    ],
)
def test_malformed_or_inapplicable_primary_is_rejected(reference):
    with pytest.raises(ReferenceError, match="primary"):
        ReferenceSet.from_obj({"references": [reference]})


# ---- resolution ------------------------------------------------------------


def test_missing_files_names_what_is_gone(tmp_path):
    (tmp_path / "here.jpg").write_bytes(b"x")
    s = ReferenceSet([img("here.jpg"), img("gone.jpg"), aud("also_gone.wav")])
    assert s.missing_files(str(tmp_path)) == ["gone.jpg", "also_gone.wav"]


def test_a_pack_resolves_when_every_file_exists(tmp_path):
    (tmp_path / "here.jpg").write_bytes(b"x")
    assert ReferenceSet([img("here.jpg")]).resolve(str(tmp_path)) == []


def test_a_directory_is_not_a_file(tmp_path):
    os.makedirs(tmp_path / "adir.jpg")
    assert ReferenceSet([img("adir.jpg")]).missing_files(str(tmp_path)) == ["adir.jpg"]


# ---- output sockets --------------------------------------------------------


def test_outputs_in_declaration_order():
    names = output_names()
    assert len(names) == 22
    assert names[0] == "image_1"
    assert names[8] == "image_9"
    assert names[9:12] == ("video_1", "video_2", "video_3")
    assert names[12:15] == ("video_audio_1", "video_audio_2", "video_audio_3")
    assert names[15:18] == ("audio_1", "audio_2", "audio_3")
    assert names[18] == "prompt"
    assert names[19] == "debug"
    assert names[20:] == ("width", "height")


def test_the_19_media_and_prompt_slots_keep_their_indices():
    """ComfyUI stores a link by output slot INDEX. Every socket that existed before
    `debug` was appended must keep the index it had, or a saved workflow silently
    re-points its links (image_9 -> video_1, and so on)."""
    names = output_names()
    expected = (
        [f"image_{i}" for i in range(1, 10)]
        + [f"video_{i}" for i in range(1, 4)]
        + [f"video_audio_{i}" for i in range(1, 4)]
        + [f"audio_{i}" for i in range(1, 4)]
        + ["prompt"]
    )
    assert list(names[:19]) == expected
    assert slot_index("prompt") == 18


def test_output_types_line_up():
    types = output_types()
    assert len(types) == len(output_names())
    assert types[:12] == ("IMAGE",) * 12
    assert types[12:18] == ("AUDIO",) * 6
    assert types[18] == "STRING"
    assert types[19] == "STRING"
    assert types[20:] == ("INT", "INT")


def test_empty_outputs_is_none_but_the_strings_and_the_frame():
    out = empty_outputs()
    assert len(out) == 22
    assert out[:18] == [None] * 18
    assert out[18] == ""
    assert out[19] == ""
    assert out[20:] == [0, 0]


# ---- pack settings (model / reasoning effort) --------------------------------


def test_video_soundtrack_is_on_by_default():
    """A reference video's sound is part of the reference. Opt out, don't opt in."""
    assert Reference(kind="video", file="v.mp4").use_soundtrack is True
    # and a dict that simply omits the key inherits that default
    assert Reference.from_dict({"kind": "video", "file": "v.mp4"}).use_soundtrack is True
    # explicit False still wins
    assert Reference.from_dict(
        {"kind": "video", "file": "v.mp4", "use_soundtrack": False}
    ).use_soundtrack is False
    # non-videos never carry it
    assert Reference.from_dict({"kind": "image", "file": "i.png"}).use_soundtrack is False


def test_a_default_video_takes_an_audio_tag():
    s = ReferenceSet([Reference(kind="video", file="v.mp4"), aud("vo.wav")])
    tagged = {t.file: t for t in s.assign_tags()}
    assert tagged["v.mp4"].audio_tag == "<Audio 1>"
    assert tagged["vo.wav"].tag == "<Audio 2>"


# ---- crop / trim ------------------------------------------------------------


def test_crop_and_trim_round_trip():
    r = Reference.from_dict(
        {"kind": "video", "file": "v.mp4", "crop": [0.1, 0.2, 0.5, 0.5], "trim": [2.0, 6.5]}
    )
    assert r.crop == [0.1, 0.2, 0.5, 0.5]
    assert r.trim == [2.0, 6.5]
    d = r.to_dict()
    assert d["crop"] == [0.1, 0.2, 0.5, 0.5]
    assert d["trim"] == [2.0, 6.5]


def test_crop_and_trim_are_omitted_when_unset():
    """Pre-edit references_json values and v1 pack files must keep loading AND
    re-serialising unchanged, so an unedited reference emits no crop/trim keys."""
    for r in (img("a.jpg"), vid("v.mp4"), aud("s.wav")):
        d = r.to_dict()
        assert "crop" not in d
        assert "trim" not in d
    old = Reference.from_dict({"kind": "video", "file": "v.mp4"})
    assert old.crop is None and old.trim is None


@pytest.mark.parametrize("bad", [
    "half",                     # not a list
    [0.1, 0.2, 0.5],            # wrong arity
    [0.1, 0.2, 0.5, "h"],       # not numbers
    [-0.1, 0.0, 0.5, 0.5],      # negative origin
    [0.0, 0.0, 0.0, 0.5],       # zero area
    [0.0, 0.0, -0.5, 0.5],      # negative area
    [0.8, 0.0, 0.5, 0.5],       # x + w past the frame
    [0.0, 0.8, 0.5, 0.5],       # y + h past the frame
])
def test_bad_crops_are_rejected(bad):
    with pytest.raises(ReferenceError):
        Reference.from_dict({"kind": "image", "file": "a.jpg", "crop": bad})


@pytest.mark.parametrize("bad", [
    "later", [1.0], [1.0, 2.0, 3.0], ["a", "b"],
    [-1.0, 2.0],  # negative start
    [3.0, 3.0],   # end == start
    [4.0, 2.0],   # end < start
])
def test_bad_trims_are_rejected(bad):
    with pytest.raises(ReferenceError):
        Reference.from_dict({"kind": "audio", "file": "s.wav", "trim": bad})


def test_crop_only_sticks_to_images_and_videos():
    # mirrors use_soundtrack: an inapplicable field is dropped, not fatal
    box = [0, 0, 0.5, 0.5]
    assert Reference.from_dict({"kind": "audio", "file": "s.wav", "crop": box}).crop is None
    assert Reference.from_dict({"kind": "image", "file": "a.jpg", "crop": box}).crop == box
    assert Reference.from_dict({"kind": "video", "file": "v.mp4", "crop": box}).crop == box


def test_trim_only_sticks_to_videos_and_audio():
    assert Reference.from_dict({"kind": "image", "file": "a.jpg", "trim": [1.0, 2.0]}).trim is None
    assert Reference.from_dict({"kind": "video", "file": "v.mp4", "trim": [1.0, 2.0]}).trim == [1.0, 2.0]
    assert Reference.from_dict({"kind": "audio", "file": "s.wav", "trim": [1.0, 2.0]}).trim == [1.0, 2.0]


def test_crop_and_trim_survive_a_references_json_round_trip():
    """The pack store is gone, but this is the state the node actually restores from."""
    original = ReferenceSet([
        Reference(kind="video", file="v.mp4", crop=[0.0, 0.0, 0.5, 0.5], trim=[1.0, 3.0]),
    ])

    back = ReferenceSet.from_json(original.to_json()).references[0]

    assert back.crop == [0.0, 0.0, 0.5, 0.5]
    assert back.trim == [1.0, 3.0]
