import pytest

from minimax_refpack import task_plan
from minimax_refpack.refs import Reference, ReferenceError, ReferenceSet, TaskPlan


def explicit(*references, specialization="none"):
    return ReferenceSet(
        list(references),
        task_plan=TaskPlan(mode="explicit", specialization=specialization),
    )


def test_auto_and_legacy_sets_have_no_explicit_plan():
    assert task_plan.resolve(ReferenceSet()) is None
    assert task_plan.resolve(ReferenceSet(task_plan=TaskPlan(mode="auto"))) is None


def test_combined_prefix_is_derived_once_in_canonical_order():
    refs = explicit(
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
        specialization="character_replacement",
    )

    resolved = task_plan.resolve(refs)

    assert resolved.task_types == (
        "video editing",
        "keyframe completion",
        "reference generation",
        "audio reuse",
    )
    assert resolved.prefix == (
        "[video editing + keyframe completion + reference generation + audio reuse]"
    )
    assert resolved.overlay_ids == (
        "video_editing",
        "keyframe_completion",
        "reference_generation",
        "audio_reuse",
        "character_replacement",
    )


def test_rendered_plan_names_each_assets_actual_minimax_tag():
    refs = explicit(
        Reference(kind="image", file="face.png", roles=["reference_generation"]),
        Reference(
            kind="video",
            file="plate.mp4",
            use_soundtrack=True,
            primary=True,
            roles=["video_editing", "audio_reuse"],
        ),
    )

    rendered = task_plan.render_user_block(refs, task_plan.resolve(refs))

    assert "summary prefix: [video editing + reference generation + audio reuse]" in rendered
    assert "<Picture 1>: reference generation" in rendered
    assert "<Video 1>: video editing (primary video)" in rendered
    assert "<Audio 1>: audio reuse (soundtrack of <Video 1>)" in rendered


def test_only_one_video_can_be_the_edit_or_continuation_source():
    refs = explicit(
        Reference(kind="video", file="a.mp4", roles=["video_editing"]),
        Reference(kind="video", file="b.mp4", roles=["video_continuation"]),
    )
    with pytest.raises(ReferenceError, match="one primary video"):
        task_plan.resolve(refs)


def test_the_primary_video_must_be_video_one():
    refs = explicit(
        Reference(kind="video", file="guide.mp4", roles=["reference_generation"]),
        Reference(kind="video", file="plate.mp4", roles=["video_editing"]),
    )
    with pytest.raises(ReferenceError, match="<Video 1>"):
        task_plan.resolve(refs)


def test_primary_designation_must_match_the_edit_or_continuation_role():
    refs = explicit(
        Reference(
            kind="video",
            file="guide.mp4",
            primary=True,
            roles=["reference_generation"],
        ),
        Reference(kind="video", file="plate.mp4", roles=["video_editing"]),
    )
    with pytest.raises(ReferenceError, match="same video"):
        task_plan.resolve(refs)


def test_only_one_video_can_be_explicitly_designated_primary():
    refs = explicit(
        Reference(kind="video", file="a.mp4", primary=True, roles=["video_editing"]),
        Reference(kind="video", file="b.mp4", primary=True, roles=["reference_generation"]),
    )
    with pytest.raises(ReferenceError, match="one primary video"):
        task_plan.resolve(refs)


def test_replacement_specialization_requires_an_edit_source_and_image_guidance():
    no_edit = explicit(
        Reference(kind="image", file="face.png", roles=["reference_generation"]),
        specialization="character_replacement",
    )
    no_image = explicit(
        Reference(kind="video", file="plate.mp4", roles=["video_editing"]),
        specialization="object_replacement",
    )

    with pytest.raises(ReferenceError, match="video editing"):
        task_plan.resolve(no_edit)
    with pytest.raises(ReferenceError, match="image"):
        task_plan.resolve(no_image)


def test_a_video_soundtrack_role_requires_the_soundtrack_to_be_enabled():
    refs = explicit(
        Reference(
            kind="video",
            file="silent.mp4",
            use_soundtrack=False,
            roles=["reference_generation", "audio_reference"],
        )
    )
    with pytest.raises(ReferenceError, match="soundtrack"):
        task_plan.resolve(refs)


def test_explicit_mode_requires_at_least_one_role():
    with pytest.raises(ReferenceError, match="at least one reference role"):
        task_plan.resolve(explicit(Reference(kind="image", file="face.png")))
