"""Exercise the crop/trim helpers and the crop/trim editor from the shipped browser modules."""

import json

import pytest

from _js import FAKE_DOM, evaluate, requires_node

pytestmark = requires_node


def _run(expression: str):
    return evaluate(expression, "edits")


@pytest.mark.parametrize(
    "start,end,duration,want",
    [
        (1.234, 3.456, 10, [1.23, 3.46]),
        (-1, 20, 10, [0, 10]),
        (5, 3, 10, [5, 5.05]),          # the in-point never passes the out-point
        (9.99, 10, 10, [9.95, 10]),     # nor sits within 0.05s of the clip's end
    ],
)
def test_clamp_trim_keeps_the_window_inside_the_clip(start, end, duration, want):
    assert _run(f"clampTrim({start}, {end}, {duration})") == pytest.approx(want)


def test_apply_edit_drops_what_covers_everything_or_the_kind_cannot_carry():
    got = _run("""(() => {
        const video = {file: 'v.mp4', crop: [0.5, 0, 0.5, 1], trim: [1, 2]};
        const videoResult = applyEdit(video, 'video', [0, 0, 1, 1], [0.001, 9.999], 10);
        const image = {file: 'a.png'};
        const imageResult = applyEdit(image, 'image', [0.25, 0.25, 0.5, 0.5], [1, 2], 10);
        const audio = {file: 's.wav'};
        applyEdit(audio, 'audio', [0.25, 0.25, 0.5, 0.5], [1.004, 2.5], 10);
        return {video, videoResult, image, imageResult, audio};
    })()""")

    assert got["video"] == {"file": "v.mp4"}
    assert got["videoResult"] == {"crop": None, "trim": None}
    assert got["image"] == {"file": "a.png", "crop": [0.25, 0.25, 0.5, 0.5]}
    assert got["imageResult"]["trim"] is None
    assert got["audio"] == {"file": "s.wav", "trim": [1, 2.5]}


def _editor(refs: dict, kind: str, body: str, *, probes: dict | None = None):
    """Open the real crop/trim editor on reference `kind`[0] and run `body`.

    `body` sees `modal`, `media`, `node`, `applied`, `dropped`, `converted` and
    `trimFields` ([in, out]), and must evaluate to one JSON value.
    """
    setup = FAKE_DOM + f"""
    const node = {{_mmrpRefs: {json.dumps(refs)},
        widgets: [{{name: 'width', value: 1280}}, {{name: 'height', value: 720}}]}};
    let applied = null;
    const dropped = [], converted = [];
    const {{ openEditModal }} = createEditUi({{
        fileUrl: (file) => `/view/${{file}}`,
        probeResults: new Map(Object.entries({json.dumps(probes or {})})),
        dropThumb: (file) => dropped.push(file),
        toggleSoundtrack: () => {{}},
        convertToAudioOnly: (target, index) => converted.push(index),
        applyRefs: (target, next) => {{ applied = next; target._mmrpRefs = next; }},
    }});
    openEditModal(node, {json.dumps(kind)}, 0);
    const modal = byClass(document.body, 'mmrp-edit-modal')[0];
    const media = all(modal).find(el => ['img', 'video', 'audio'].includes(el.tag));
    const trimFields = byClass(modal, 'mmrp-trim-num');
    const click = (text) => fire(byText(modal, text), 'click');
    """
    return evaluate(body, "edit_ui", setup=setup)


def _refs(**kinds):
    return {"images": [], "videos": [], "audios": [], "taskPlan": None, "control": None, **kinds}


def test_editor_trims_on_metadata_and_saves_a_cleared_crop_as_none():
    refs = _refs(videos=[{"file": "clip.mp4", "roles": [], "use_soundtrack": True,
                          "crop": [0.1, 0.1, 0.5, 0.5], "trim": [1, 3]}])
    got = _editor(refs, "video", """(() => {
        const before = trimFields.map(field => field.disabled);
        Object.assign(media, {duration: 10, videoWidth: 640, videoHeight: 360});
        fire(media, 'loadedmetadata');
        const loaded = trimFields.map(field => field.value);
        trimFields[0].value = '5';
        fire(trimFields[0], 'change');
        const clearCropWasEnabled = !byText(modal, 'Clear crop').disabled;
        click('Clear crop');
        click('Save');
        return {before, loaded, src: media.src, clearCropWasEnabled, video: applied.videos[0],
                dropped, open: !!modal.parentElement.parentElement};
    })()""")

    assert got["before"] == [True, True]
    assert got["loaded"] == ["1.00", "3.00"]
    assert got["src"] == "/view/clip.mp4"
    assert got["clearCropWasEnabled"] is True
    assert "crop" not in got["video"]
    assert got["video"]["trim"] == [5, 5.05]
    assert got["dropped"] == ["clip.mp4"]
    assert got["open"] is False


def test_editor_offers_audio_only_for_a_clip_with_sound():
    refs = _refs(videos=[{"file": "clip.mp4", "roles": [], "use_soundtrack": True}])
    unknown = _editor(refs, "video", "byText(modal, 'Audio only').disabled")
    got = _editor(refs, "video", """(() => {
        const disabled = byText(modal, 'Audio only').disabled;
        click('Audio only');
        return {disabled, converted, applied};
    })()""", probes={"clip.mp4": True})

    assert unknown is True  # the probe has not answered
    assert got == {"disabled": False, "converted": [0], "applied": None}


def test_image_editor_has_no_trim_or_preview():
    refs = _refs(images=[{"file": "a.png", "roles": []}])
    got = _editor(refs, "image", """(() => {
        Object.assign(media, {naturalWidth: 400, naturalHeight: 200});
        fire(media, 'load');
        click('1:1');
        click('Save');
        return {tag: media.tag, trim: trimFields.length, preview: !!byText(modal, 'Play original'),
                image: applied.images[0]};
    })()""")

    assert got["tag"] == "img"
    assert got["trim"] == 0
    assert got["preview"] is False
    assert got["image"]["crop"] == [0.25, 0, 0.5, 1]  # 1:1 on a 2:1 image
