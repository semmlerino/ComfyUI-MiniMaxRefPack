# MiniMax Reference Planning

This context describes how media assets are selected and assigned relationships for a MiniMax full-reference video prompt.

## Language

**Input-directory asset**:
A media file already available in ComfyUI's input directory and eligible to be selected.
_Avoid_: Existing input, connected input

**Reference asset**:
An input-directory asset attached to the References Manager and assigned a MiniMax reference tag.
_Avoid_: Slot, upload

**Reference role**:
An official relationship that states how one reference asset contributes to the target video. A reference asset may have more than one role.
_Avoid_: Job type, asset type

**Primary video**:
The single reference video explicitly designated as the source that is directly edited or continued. Its designation is stored separately from its editing or continuation role, and both must refer to the same asset.
_Avoid_: Master toggle, selected video

**Task type**:
One of MiniMax's official relationships written in the `summary` prefix and derived from the reference roles.
_Avoid_: Mode, specialization

**Task plan**:
The choice to infer roles or declare them explicitly, together with any declared reference roles and replacement specialization.
_Avoid_: Prompt preset, job type

**Replacement specialization**:
Additional character- or object-replacement guidance applied to a video-editing task without creating a new task type.
_Avoid_: Replacement mode, replacement task type
