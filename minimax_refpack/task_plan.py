"""Resolve per-reference roles into MiniMax's official summary task types.

This module is intentionally pure: it decides relationships and renders the small
authoritative task-plan block, while prompt.py owns provider calls and media payloads.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import refs


ROLE_TASK_TYPE: dict[str, str] = {
    "reference_generation": "reference generation",
    "keyframe_completion": "keyframe completion",
    "video_editing": "video editing",
    "video_continuation": "video continuation",
    "audio_reuse": "audio reuse",
    "audio_reference": "audio reference",
}

# Driver relationships come first, followed by concrete frames, general guidance and
# audio. This matches the guide's combined examples (`video editing + reference
# generation + audio reuse`, `video continuation + keyframe completion`) and makes the
# same set of roles serialize identically regardless of asset insertion order.
TASK_TYPE_ORDER: tuple[str, ...] = (
    "video editing",
    "video continuation",
    "keyframe completion",
    "reference generation",
    "audio reuse",
    "audio reference",
)

_AUDIO_ROLES = frozenset(("audio_reuse", "audio_reference"))
_PRIMARY_ROLES = frozenset(("video_editing", "video_continuation"))


@dataclass(frozen=True)
class ResolvedTaskPlan:
    """A validated explicit plan ready for prompt composition."""

    task_types: tuple[str, ...]
    specialization: str = "none"

    @property
    def prefix(self) -> str:
        return "[" + " + ".join(self.task_types) + "]"

    @property
    def overlay_ids(self) -> tuple[str, ...]:
        ids = tuple(task_type.replace(" ", "_") for task_type in self.task_types)
        if self.specialization != "none":
            ids += (self.specialization,)
        return ids


def resolve(reference_set: refs.ReferenceSet) -> ResolvedTaskPlan | None:
    """Return a validated explicit plan, or None for legacy/automatic routing."""
    plan = reference_set.task_plan
    if plan is None or plan.mode == "auto":
        return None

    reference_set.validate()
    role_count = sum(len(reference.roles) for reference in reference_set.references)
    if role_count == 0:
        raise refs.ReferenceError("explicit task-plan mode requires at least one reference role")

    videos = reference_set.of_kind("video")
    drivers: list[tuple[int, str]] = []
    primaries: list[int] = []
    for index, reference in enumerate(videos):
        if reference.primary:
            primaries.append(index)
        for role in reference.roles:
            if role in _PRIMARY_ROLES:
                drivers.append((index, role))
        if not reference.use_soundtrack and _AUDIO_ROLES.intersection(reference.roles):
            raise refs.ReferenceError(
                f"audio role assigned to {reference.file!r}, but its soundtrack is disabled"
            )

    if len(primaries) > 1:
        raise refs.ReferenceError("an explicit task plan may have only one primary video")
    if len(drivers) > 1:
        raise refs.ReferenceError(
            "an explicit task plan may have only one primary video for editing or continuation"
        )
    if primaries and not drivers:
        raise refs.ReferenceError(
            "the primary video requires a video editing or continuation role on the same video"
        )
    if primaries and drivers and primaries[0] != drivers[0][0]:
        raise refs.ReferenceError(
            "the primary designation and video editing/continuation role must be on the same video"
        )
    primary_index = primaries[0] if primaries else drivers[0][0] if drivers else None
    if primary_index is not None and primary_index != 0:
        raise refs.ReferenceError(
            "the primary edit/continuation source must be <Video 1>; put that asset in the first video slot"
        )

    all_roles = {
        role for reference in reference_set.references for role in reference.roles
    }
    if plan.specialization != "none":
        if "video_editing" not in all_roles:
            raise refs.ReferenceError(
                "replacement specialization requires a video editing source"
            )
        if not any(
            reference.kind == "image" and "reference_generation" in reference.roles
            for reference in reference_set.references
        ):
            raise refs.ReferenceError(
                "replacement specialization requires an image with reference generation guidance"
            )

    used_types = {ROLE_TASK_TYPE[role] for role in all_roles}
    task_types = tuple(task for task in TASK_TYPE_ORDER if task in used_types)
    return ResolvedTaskPlan(task_types=task_types, specialization=plan.specialization)


def render_user_block(
    reference_set: refs.ReferenceSet, resolved: ResolvedTaskPlan
) -> str:
    """Render the asset-specific, authoritative plan prepended to VLM user content."""
    lines = [
        "EXPLICIT TASK PLAN (authoritative):",
        f"summary prefix: {resolved.prefix}",
    ]
    if resolved.specialization != "none":
        lines.append("replacement specialization: " + resolved.specialization.replace("_", " "))

    for tagged in reference_set.assign_tags():
        reference = tagged.ref
        visual = [role for role in reference.roles if role not in _AUDIO_ROLES]
        audio = [role for role in reference.roles if role in _AUDIO_ROLES]

        if reference.kind != "audio":
            labels = [ROLE_TASK_TYPE[role] for role in visual]
            if reference.primary or any(role in _PRIMARY_ROLES for role in visual):
                labels = [
                    label + " (primary video)" if role in _PRIMARY_ROLES else label
                    for role, label in zip(visual, labels)
                ]
            lines.append(f"{tagged.tag}: " + (", ".join(labels) if labels else "unassigned"))

        if reference.kind == "video" and audio:
            # resolve() rejects this before rendering when the soundtrack is disabled.
            labels = ", ".join(ROLE_TASK_TYPE[role] for role in audio)
            lines.append(f"{tagged.audio_tag}: {labels} (soundtrack of {tagged.tag})")
        elif reference.kind == "audio":
            labels = [ROLE_TASK_TYPE[role] for role in audio]
            lines.append(f"{tagged.tag}: " + (", ".join(labels) if labels else "unassigned"))

    lines.append(
        "Use only these declared relationships to choose task types; an unassigned asset "
        "must not add a task type."
    )
    return "\n".join(lines)
