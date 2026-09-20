"""MiniMaxH3ReferencePack: owns a reference set and fans it out to MiniMax H3's 18
reference sockets, plus a written prompt and a debug dump. Plain V1-style node (not
io.ComfyNode/v3): INPUT_TYPES/IS_CHANGED shape with an mtime+size cache-busting
signature.

Never a wrapper around the target node: comfy_extras/nodes_minimax_h3.py already skips
None per reference group (:219,236,270), so "always connected, mostly None" is what
upstream anticipates. MiniMax's node stays the sole encoder; we never touch its
VAE/tokenizer.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from typing import Any

from . import endpoint, logs, media, prefetch, prompt, refs

# Default long-edge cap for reference IMAGES. Named rather than inlined because it is
# the widget default AND the fallback when a workflow saved before the widget existed
# restores without it - those two must never drift apart.
DEFAULT_MAX_REFERENCE_EDGE = 2048

# MiniMaxH3ReferenceToVideo's width/height step, and the grid its reference canvas is
# rounded to (comfy_extras/nodes_minimax_h3.py CANVAS_MULTIPLE).
FRAME_MULTIPLE = 32


def match_frame(width: int, height: int, video_width: int, video_height: int) -> tuple[int, int]:
    """width x height's pixel area in the video's aspect ratio, on the 32 px grid.

    The core node never crops a reference video to the output canvas: it keeps the
    video's own ratio and leaves any mismatch to the model, which reframes or invents
    picture. So the area stays the user's (their megapixel choice) and the shape comes
    from the video. A 2.35:1 plate at a 1376x768 area comes out 1568x672.
    """
    area = width * height
    ratio = video_width / video_height
    return (
        max(FRAME_MULTIPLE, round(math.sqrt(area * ratio) / FRAME_MULTIPLE) * FRAME_MULTIPLE),
        max(FRAME_MULTIPLE, round(math.sqrt(area / ratio) / FRAME_MULTIPLE) * FRAME_MULTIPLE),
    )


def _switch_on(value) -> bool:
    """Only True, or the text "true", turns the switch on, for callers of build() itself.

    On a queued prompt this cannot help: execution.py bool()s a BOOLEAN input before the
    node runs, so a stray "False" arrives as True and "" as False. A graph saved before
    the widget existed restores the DOM widget's "" into this slot, and web/refpack.js
    (migrateMatchValue) resets anything but a real true before the graph is queued.
    """
    return value is True or (isinstance(value, str) and value.strip().lower() == "true")


def _provider_of(prompt_provider, use_openrouter=None) -> str:
    """The provider to use, accepting the shape a pre-0.3.2 workflow sends.

    `prompt_provider` replaced a BOOLEAN named `use_openrouter` in the same widget slot.
    web/refpack.js rewrites saved graphs on load, but that only helps a graph opened in a
    browser: an API client posting a stored prompt, or a graph loaded by a script, sends
    whatever it stored. So the server accepts both and never fails on the old shape.

    Belt and braces on purpose - the failure this guards is a workflow that will not
    queue at all, which is worse than the bug this release set out to fix.
    """
    if use_openrouter is not None and prompt_provider in (None, "", endpoint.DEFAULT_PROVIDER):
        # Only trust the legacy field when the new one carries nothing meaningful, so a
        # deliberate new value always wins over a stale boolean sitting beside it.
        return endpoint.normalize_provider(use_openrouter)
    return endpoint.normalize_provider(prompt_provider)


def _model_for(provider: str, openrouter_model: str, local_model_slug: str) -> str:
    """The model id for this provider, read from that provider's OWN field.

    There was a single generic `model_override` here once, applied to whichever provider
    was selected. It produced this, live against OpenRouter 2026-08-17:

        openrouter returned 400: google/gemma-4-e2b is not a valid model ID

    ...because configuring a local run and then switching back to `openrouter` left the
    local slug in the override, where it still won. The two fields no longer see each
    other, so there is no precedence rule left to get wrong: `openrouter` reads the
    dropdown, `local` reads the typed slug, neither can reach the other.

    Local with an empty slug is a hard error rather than a fallback to the dropdown: that
    dropdown lists OpenRouter's models, which a local server has never heard of, so
    falling back would just move the 400 to the other end.
    """
    if provider == "local":
        slug = (local_model_slug or "").strip()
        if not slug:
            raise ValueError(
                "prompt_provider is 'local' but local_model_slug is empty: type the model "
                "id your server reports, or click Local LLM to pick one. The "
                "openrouter_model dropdown is not used on this path."
            )
        return slug
    return openrouter_model or ""


def _credential_fingerprint(api_key) -> str:
    """A short hash of the provider key, so two jobs holding different credentials never
    share a cache identity - and the key itself never appears in one. Blank stays blank
    (the environment supplies the key on a pod, and that is one identity)."""
    if not api_key:
        return ""
    return hashlib.sha256(str(api_key).encode()).hexdigest()[:16]


def _files_signature(reference_set: refs.ReferenceSet, input_dir: str) -> str:
    """mtime+size of every referenced file, folded into one hash.

    Guards a real bug: a hidden state widget can hold byte-identical JSON across two
    different uploads (the JS re-uploads with overwrite=true under the same filename),
    so without this ComfyUI cache-hits IS_CHANGED and reruns emit the PREVIOUS
    reference's result.
    """
    h = hashlib.sha256()
    for r in reference_set.references:
        path = os.path.join(input_dir, r.file)
        try:
            st = os.stat(path)
            h.update(f"{r.file}:{st.st_mtime_ns}:{st.st_size}".encode("utf-8"))
        except OSError:
            h.update(f"{r.file}:missing".encode("utf-8"))
    return h.hexdigest()


@dataclass
class BuildResult:
    """What one build produced: the socket tuple, plus the two things the prefetcher
    keeps on their own when the decoded media is over its budget."""

    outputs: tuple[Any, ...]
    prompt_text: str
    debug_sink: list[str]
    media_bytes: int


def _media_bytes(outputs) -> int:
    """Bytes the decoded media in `outputs` occupies: IMAGE tensors and the waveforms
    inside AUDIO dicts. What a prepared pack holds in RAM until its job runs, which is
    what the prefetcher's budget is counted in."""
    total = 0
    for value in outputs:
        tensor = value.get("waveform") if isinstance(value, dict) else value
        # Duck-typed on purpose: this module never imports torch (the node tests stub
        # the loaders with plain strings), and a tensor is anything that can say how
        # many elements it has and how wide each one is.
        numel = getattr(tensor, "numel", None)
        element_size = getattr(tensor, "element_size", None)
        if callable(numel) and callable(element_size):
            count, width = numel(), element_size()
            if isinstance(count, int) and isinstance(width, int):
                total += count * width
    return total


class MiniMaxH3ReferencePack:
    """Owns a reference set (images/videos/audio) and fans it out to image_1..9,
    video_1..3, video_audio_1..3, audio_1..3 plus a VLM-written prompt. Wire the 19
    sockets into MiniMaxH3ReferenceToVideo by hand and save - empty slots emit None,
    which that node already skips per-group."""

    # The declaration order IS the on-canvas order, and litegraph restores saved values
    # POSITIONALLY, so this list is a wire format as much as a layout. Reordering it (as
    # happened for 0.3.3) scrambles every workflow saved before the change unless
    # web/refpack.js remaps them on load - see remapWidgetValues there, and keep the two
    # in step. Grouped by decision flow, agreed with Aviv 2026-08-17: the mode first,
    # then that mode's settings, then what to write, then the target video, then how the
    # references are prepared.
    #
    # `direction`, `references_json` and `system_prompt` lead because they are HIDDEN -
    # direction is bound to the DOM textarea, the other two live behind the modals - so
    # they cost no rows on the canvas and keep the visible block contiguous.
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "direction": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Steers the VLM's prompt writing (subject, mood, action).",
                }),
            },
            "optional": {
                "references_json": ("STRING", {
                    "multiline": True,
                    "dynamicPrompts": False,
                    "default": "",
                    "tooltip": "Reference list written by the modal. Do not hand-edit.",
                }),
                "system_prompt": ("STRING", {
                    "multiline": True,
                    "dynamicPrompts": False,
                    "default": "",
                    "tooltip": "VLM system prompt, editable per-workflow via the settings modal. "
                               "Blank falls back to the packaged default.",
                }),

                # --- the mode, first, because it decides everything under it ----------
                "prompt_provider": (list(endpoint.PROVIDERS), {
                    "default": endpoint.DEFAULT_PROVIDER,
                    "tooltip": "Who writes the prompt. openrouter = the hosted API (needs "
                               "a key). local = any OpenAI-compatible server, set api_base "
                               "below (Ollama, LM Studio, llama.cpp, vLLM); video is sent "
                               "as still frames and audio is not sent at all. none = no "
                               "call, your direction text passes through verbatim.",
                }),

                # --- the openrouter group; hidden by the UI on other providers --------
                "openrouter_api_key": ("STRING", {
                    "default": "",
                    "tooltip": "OpenRouter key. Blank falls back to OPENROUTER_API_KEY / LLM_KEY.",
                }),
                # Named for its provider, not generically, and read ONLY when
                # prompt_provider is `openrouter`. See _model_for().
                "openrouter_model": (prompt.available_models(), {
                    "default": prompt.DEFAULT_MODEL,
                    "tooltip": "Used only when prompt_provider is 'openrouter'. Lists "
                               "models that accept text, images, audio and video.",
                }),

                # OpenRouter-only, so it lives in the OpenRouter group and hides with it.
                # `sends_reasoning` is False for every other endpoint (endpoint.py), and
                # prompt.py gates the payload field on it: a plain OpenAI-compatible
                # server is more likely to reject an unknown top-level field outright
                # than to ignore it, which would turn a working local setup into a 400.
                "reasoning_effort": (list(prompt.REASONING_EFFORTS), {
                    "default": prompt.DEFAULT_REASONING_EFFORT,
                    "tooltip": "Used only when prompt_provider is 'openrouter'. How hard "
                               "the model thinks before writing. OpenRouter drops it for "
                               "models that don't reason; other endpoints never see it.",
                }),

                # --- the local group; hidden by the UI on other providers -------------
                "api_base": ("STRING", {
                    "default": "",
                    "tooltip": "Only used when prompt_provider is 'local'. The base URL of "
                               "an OpenAI-compatible server, ending in /v1. Ollama: "
                               "http://localhost:11434/v1 \u00b7 LM Studio: "
                               "http://localhost:1234/v1",
                }),
                "local_model_slug": ("STRING", {
                    "default": "",
                    "tooltip": "Used only when prompt_provider is 'local'. The model id "
                               "your own server reports, e.g. google/gemma-4-e2b or "
                               "qwen2.5vl:7b. The Local LLM button fills this in for you.",
                }),

                # --- what gets written (provider-neutral) -----------------------------
                "job_type": (list(prompt.MODES), {
                    "default": "auto",
                    "tooltip": "Legacy routing retained for saved workflows. New workflows "
                               "use the References Manager's Task plan: explicit asset roles "
                               "skip classification, while Infer roles uses auto.",
                }),

                # --- the target video --------------------------------------------------
                "width": ("INT", {
                    "default": 1280, "min": 0, "max": 8192,
                    "tooltip": "Target frame width, told to the VLM. 0 = unspecified.",
                }),
                "height": ("INT", {
                    "default": 720, "min": 0, "max": 8192,
                    "tooltip": "Target frame height, told to the VLM. 0 = unspecified.",
                }),
                "length_seconds": ("FLOAT", {
                    "default": 8.0, "min": 0.0, "max": 60.0, "step": 0.25,
                    "tooltip": "Target clip duration in seconds, told to the VLM. 0 = unspecified.",
                }),

                # --- how the references are prepared -----------------------------------
                "max_reference_edge": ("INT", {
                    "default": 2048, "min": 0, "max": 8192, "step": 64,
                    "tooltip": "Downscale a reference IMAGE whose long edge exceeds this "
                               "(0 = off). Never upscales. MiniMax sizes references off "
                               "their SHORT edge, so at ref_image_size=max a wide sheet "
                               "arrives huge and every sampling step pays for it. "
                               "Reference VIDEOS are not covered: they are decoded and "
                               "cached at source resolution, and the core node resizes "
                               "them at encode time.",
                }),

                # Appended, not grouped with width/height: widgets_values restores by
                # position, and the end is the one slot no saved graph already fills.
                "match_video_aspect": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "On: width x height sets only the pixel area, and the frame "
                               "takes the aspect ratio of the primary video (<Video 1> when "
                               "none is marked), on a 32 px grid. The width/height outputs "
                               "carry the frame; wire them into MiniMax's node so the render "
                               "and the prompt use the same one. No reference video: "
                               "width and height pass through.",
                }),

                # Display-only record of the idea typed before Auto Prompt rewrote
                # `direction`. Appended so every earlier slot stays where it was.
                # Freeze and PromptToWorkflow fill this once and then leave it.
                "original_prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "The idea typed before Auto Prompt rewrote direction. "
                               "Never sent to the VLM. A non-empty value is not overwritten.",
                }),
            },
        }

    RETURN_TYPES = refs.output_types()
    RETURN_NAMES = refs.output_names()
    FUNCTION = "build"
    CATEGORY = "MiniMax H3"

    @classmethod
    def IS_CHANGED(
        cls, direction="", openrouter_api_key="", openrouter_model="", references_json="",
        system_prompt="", width=0, height=0, length_seconds=0.0,
        prompt_provider=endpoint.DEFAULT_PROVIDER,
        reasoning_effort=prompt.DEFAULT_REASONING_EFFORT, job_type="auto",
        max_reference_edge=DEFAULT_MAX_REFERENCE_EDGE, api_base="", local_model_slug="",
        match_video_aspect=False, original_prompt="", use_openrouter=None, model=None,
        model_override=None, **kwargs
    ):
        openrouter_model = openrouter_model or (model or "")
        local_model_slug = local_model_slug or (model_override or "")
        import folder_paths

        reference_set = refs.ReferenceSet.from_json(references_json)
        sig = _files_signature(reference_set, folder_paths.get_input_directory())
        # direction/model/references_json/system_prompt decide when the prompt gets
        # rewritten - fold them in directly rather than relying on
        # file bytes alone, which can't see a text-only edit. width/height/length land
        # in the payload's TARGET FORMAT block, prompt_provider switches the whole call
        # and reasoning_effort changes the completion, so all five move the key too.
        # api_base and model_override change WHERE the call goes and WHAT answers it, so
        # they belong here as much as `model` does.
        #
        # A JSON list, not a "|" join. ComfyUI pairs this string with the literal inputs
        # when it keys its cache, but the prefetcher uses it ALONE as the identity of a
        # prepared pack, and under a plain join direction="x|||y" + system_prompt="z"
        # was the same key as direction="x" + system_prompt="y|||z". JSON escapes. The
        # provider key rides as a fingerprint, never in clear, for the same reason: two
        # jobs holding different credentials must not consume each other's prompt.
        return json.dumps([
            sig, direction, openrouter_model, references_json, system_prompt,
            str(width), str(height), str(length_seconds),
            _provider_of(prompt_provider, use_openrouter),
            str(reasoning_effort), str(job_type), str(max_reference_edge),
            str(api_base), str(local_model_slug), str(_switch_on(match_video_aspect)),
            _credential_fingerprint(openrouter_api_key),
        ], separators=(",", ":"))

    def _build(
        self, direction="", openrouter_api_key="", openrouter_model="", references_json="",
        system_prompt="", width=0, height=0, length_seconds=0.0,
        prompt_provider=endpoint.DEFAULT_PROVIDER,
        reasoning_effort=prompt.DEFAULT_REASONING_EFFORT, job_type="auto",
        max_reference_edge=DEFAULT_MAX_REFERENCE_EDGE, api_base="", local_model_slug="",
        match_video_aspect=False, use_openrouter=None, model=None, model_override=None,
        prewritten: prefetch.Prepared | None = None, prepared_how: str = "no",
    ) -> BuildResult:
        """The whole build, as it has always run. `build()` wraps it with the prefetch
        lookup, and the prefetcher calls it directly for a pending job.

        `prewritten` is a prompt the prefetcher already fetched for this exact key, so
        the provider is not asked a second time when only the media has to be redone;
        `prepared_how` is what the debug socket says about where the pack came from.
        """
        # Legacy kwarg names, for an API client replaying a prompt stored before 0.3.2.
        # The new field wins when both arrive, so a deliberate value is never overridden
        # by a stale one riding alongside it.
        openrouter_model = openrouter_model or (model or "")
        local_model_slug = local_model_slug or (model_override or "")
        import folder_paths

        started = time.perf_counter()
        reference_set = refs.ReferenceSet.from_json(references_json)
        reference_set.validate()  # raises refs.ReferenceError (a ValueError) over cap

        input_dir = folder_paths.get_input_directory()
        missing = reference_set.missing_files(input_dir)
        if missing:
            raise ValueError(
                "reference file(s) not found in the ComfyUI input directory: " + ", ".join(missing)
            )

        provider = _provider_of(prompt_provider, use_openrouter)
        model = _model_for(provider, openrouter_model, local_model_slug)

        counts = reference_set.counts()
        logs.log(
            "build", images=counts["image"], videos=counts["video"], audios=counts["audio"],
            cap=max_reference_edge or None, provider=provider,
            api_base=api_base or None, model=model, job_type=job_type,
        )

        outputs = refs.empty_outputs()
        # ONE cache for this build and no longer. The prompt writer needs the same pixels
        # the sockets do, and before this it fetched its own copy of every one of them.
        # It must not outlive the build: the browser re-uploads an edited reference under
        # the same filename with overwrite=true, which is exactly why IS_CHANGED above
        # hashes mtime+size rather than trusting the name.
        cache = media.MediaCache()
        aspect_source = None  # (tag, frames) of the video the frame follows
        for tagged in reference_set.assign_tags():
            path = os.path.join(input_dir, tagged.file)
            logs.log(
                "reference", kind=tagged.kind, slot=tagged.slot, tag=tagged.tag,
                file=tagged.file, crop=tagged.ref.crop, trim=tagged.ref.trim,
                soundtrack=tagged.audio_tag,
            )
            # crop/trim ride on the reference (refs.Reference); the loaders are the one
            # apply point, so the sockets and the VLM payload can never disagree. They
            # also ride inside references_json, so IS_CHANGED's key already moves on an
            # edit - confirmed by test_is_changed_key_moves_when_only_an_edit_changes.
            if tagged.kind == "image":
                outputs[refs.slot_index(f"image_{tagged.slot}")] = cache.image(
                    path, crop=tagged.ref.crop, max_edge=max_reference_edge
                )
            elif tagged.kind == "video":
                frames, audio = cache.video(path, crop=tagged.ref.crop, trim=tagged.ref.trim)
                outputs[refs.slot_index(f"video_{tagged.slot}")] = frames
                if aspect_source is None or (tagged.ref.primary and not aspect_source[2]):
                    aspect_source = (tagged.tag, frames, tagged.ref.primary)
                if tagged.ref.use_soundtrack and audio is not None:
                    outputs[refs.slot_index(f"video_audio_{tagged.slot}")] = audio
            else:  # audio
                outputs[refs.slot_index(f"audio_{tagged.slot}")] = cache.audio(
                    path, trim=tagged.ref.trim
                )

        frame_note = ""
        if _switch_on(match_video_aspect):
            if aspect_source is None:
                frame_note = "  (match_video_aspect: no reference video, kept as given)"
            else:
                if not (width and height):
                    raise ValueError(
                        "match_video_aspect needs width and height to set the pixel area; "
                        f"got {width} x {height}"
                    )
                tag, frames, _primary = aspect_source
                # The decoded IMAGE tensor, [frames, height, width, 3]: after crop, the
                # exact pixels MiniMax's node will size its reference canvas from.
                video_h, video_w = int(frames.shape[1]), int(frames.shape[2])
                given = (width, height)
                width, height = match_frame(width, height, video_w, video_h)
                frame_note = (
                    f"  (match_video_aspect: {tag} is {video_w} x {video_h}, "
                    f"area of {given[0]} x {given[1]})"
                )
            logs.log("frame", width=width, height=height, source=aspect_source and aspect_source[0])
        outputs[refs.slot_index("width")] = int(width or 0)
        outputs[refs.slot_index("height")] = int(height or 0)

        prompt_text = ""
        debug_sink: list[str] = []
        debug_header = [
            "=== MiniMax References Manager ===",
            f"prompt_provider: {provider}" + (f" ({api_base})" if provider == "local" else ""),
            f"model: {model}"
            + (" (local_model_slug)" if provider == "local" else " (openrouter_model)"),
            f"width: {width or '(unspecified)'}  height: {height or '(unspecified)'}  "
            f"length_seconds: {length_seconds or '(unspecified)'}",
            f"frame: {width} x {height}{frame_note}",
            f"reasoning_effort: {reasoning_effort}",
            f"max_reference_edge: {max_reference_edge or 'off'}",
            f"prefetch: {prepared_how}",
            (
                f"task_plan: {reference_set.task_plan.mode}"
                if reference_set.task_plan is not None
                else f"job_type: {job_type}"
            ),   # rewritten below once auto/explicit routing has resolved
            f"system_prompt: {'workflow override' if (system_prompt or '').strip() else 'packaged default'}",
            f"references: {len(reference_set.references)} "
            f"({', '.join(f'{t.tag} {t.file}' for t in reference_set.assign_tags()) or 'none'})",
        ]

        if provider == "none":
            # Opt-out: no call at all, the direction text passes through verbatim.
            # Deliberately NOT ridden on the empty-set skip below - the passthrough
            # holds whether or not any references are attached.
            prompt_text = direction
            debug_header.append("")
            logs.log("prompt_skipped", reason="prompt_provider=none", chars=len(direction))
            debug_header.append("Auto-prompting is OFF - no request was made. `direction` passes through verbatim:")
            debug_header.append(direction)
        elif reference_set.is_empty() and not (direction or "").strip():
            # The ONE remaining skip: no references AND nothing typed. A direction
            # alone is enough to write from (the branch below); an empty set with an
            # empty direction is not.
            debug_header.append("")
            logs.log("prompt_skipped", reason="no references and no direction")
            debug_header.append("No references and no direction - nothing to write from; no request was made.")
        else:
            if reference_set.is_empty():
                debug_header.append("note: no references attached - the prompt is written from the direction alone")
            try:
                if prewritten is not None:
                    # Fetched by the prefetcher for this exact key while the previous
                    # job rendered. The media was decoded again above because it was
                    # over the retention budget; the provider is not asked twice.
                    prompt_text = prewritten.prompt_text
                    debug_sink.extend(prewritten.debug_sink)
                else:
                    prompt_text = prompt.write_prompt(
                        references=reference_set,
                        input_dir=input_dir,
                        direction=direction,
                        api_key=openrouter_api_key,
                        model=model,
                        system_prompt=system_prompt,
                        width=width,
                        height=height,
                        length_seconds=length_seconds,
                        reasoning_effort=reasoning_effort,
                        job_type=job_type,
                        debug=debug_sink,
                        provider=provider,
                        api_base=api_base,
                        cache=cache,
                        max_reference_edge=max_reference_edge,
                    )
            except ValueError as e:
                # endpoint.resolve raises this for "local with no api_base". It is a user
                # error with a fixable cause, so it reads as one rather than as a crash.
                if "api_base" not in str(e):
                    raise
                raise ValueError(f"prompt generation failed: {e}") from e
            except prompt.PromptError as e:
                # Never let the key leak into a raised message, even if the writer's
                # own error text happened to echo it back.
                msg = str(e)
                if openrouter_api_key and openrouter_api_key in msg:
                    msg = msg.replace(openrouter_api_key, "***")
                raise ValueError(f"prompt generation failed: {msg}") from e

        # Hoist the resolved route into the header. With auto the header alone would only
        # say "auto"; with an explicit plan the derived prefix and overlays are the thing
        # worth seeing at a glance.
        if debug_sink:
            routing = next(
                (
                    ln
                    for ln in debug_sink[0].splitlines()
                    if ln.startswith(("job_type:", "task_plan:"))
                ),
                "",
            )
            if routing:
                debug_header = [
                    routing
                    if ln.startswith(("job_type:", "task_plan:"))
                    else ln
                    for ln in debug_header
                ]

        debug_text = "\n".join(debug_header)
        if debug_sink:
            where = "OpenRouter" if provider == "openrouter" else api_base
            debug_text += f"\n\n--- payload sent to {where} ---\n" + debug_sink[0]

        # Last line of defence. The key is only ever a header, never part of the payload
        # dict render_payload() sees, so this should be a no-op - but `debug` is a socket
        # a user will paste into a screenshot, and a silent leak there is unrecoverable.
        if openrouter_api_key and openrouter_api_key in debug_text:
            debug_text = debug_text.replace(openrouter_api_key, "***")

        outputs[refs.slot_index("prompt")] = prompt_text
        outputs[refs.slot_index("debug")] = debug_text
        media_bytes = _media_bytes(outputs)
        logs.log("build_done", prompt_chars=len(prompt_text), prefetch=prepared_how,
                 media_mb=media_bytes / 2**20,
                 ms=(time.perf_counter() - started) * 1000.0)
        return BuildResult(
            outputs=tuple(outputs), prompt_text=prompt_text, debug_sink=debug_sink,
            media_bytes=media_bytes,
        )

    def build(
        self, direction="", openrouter_api_key="", openrouter_model="", references_json="",
        system_prompt="", width=0, height=0, length_seconds=0.0,
        prompt_provider=endpoint.DEFAULT_PROVIDER,
        reasoning_effort=prompt.DEFAULT_REASONING_EFFORT, job_type="auto",
        max_reference_edge=DEFAULT_MAX_REFERENCE_EDGE, api_base="", local_model_slug="",
        match_video_aspect=False, original_prompt="", use_openrouter=None, model=None,
        model_override=None,
    ):
        """The node's entry point: the pack the prefetcher built for these exact inputs
        while the previous job rendered, else the build itself.

        The key is computed HERE, from build's own arguments, by the same IS_CHANGED that
        keys ComfyUI's cache - never taken from the prefetcher - so a key the thread
        computed from a queued prompt can only ever match, never mislead. Waiting on a
        prepare still under way costs the time build() would have spent itself.
        """
        del original_prompt  # display-only; never part of the VLM key or the build
        kwargs: dict[str, Any] = dict(
            direction=direction, openrouter_api_key=openrouter_api_key,
            openrouter_model=openrouter_model, references_json=references_json,
            system_prompt=system_prompt, width=width, height=height,
            length_seconds=length_seconds, prompt_provider=prompt_provider,
            reasoning_effort=reasoning_effort, job_type=job_type,
            max_reference_edge=max_reference_edge, api_base=api_base,
            local_model_slug=local_model_slug, match_video_aspect=match_video_aspect,
            use_openrouter=use_openrouter, model=model, model_override=model_override,
        )
        key = _key_for(**kwargs)
        with PREFETCHER.consume(key) as prepared:
            if prepared is not None and _key_for(**kwargs) != key:
                # The key hashes every reference's mtime+size, and consume() may have
                # waited on a prepare for minutes: a reference re-uploaded meanwhile
                # means the pack was built from the old file. Build from what is on
                # disk now instead - for a prompt-only hit too, whose prompt described
                # the old media.
                logs.log("prefetch_stale", digest=prefetch.key_digest(key))
                prepared = None
            if prepared is not None and prepared.outputs is not None:
                return prepared.outputs
            if prepared is None:
                result = self._build(**kwargs)
            else:
                result = self._build(
                    **kwargs, prewritten=prepared,
                    prepared_how="prompt only; media decoded now (over the retention budget)",
                )
        return result.outputs


def _key_for(**inputs) -> str:
    """The node's own cache key for a set of inputs - IS_CHANGED itself, so the
    prefetcher and build() cannot disagree about what "the same job" means."""
    return MiniMaxH3ReferencePack.IS_CHANGED(**inputs)


def _prefetch_build(**inputs) -> BuildResult:
    return MiniMaxH3ReferencePack()._build(
        **inputs, prepared_how="prepared while the previous job rendered"
    )


# Module-level on purpose: one set of prepared packs per process, shared by every
# instance of the node ComfyUI creates. Installed onto the queue by the package
# __init__ (prefetch.install), which is a no-op outside a running server.
PREFETCHER = prefetch.Prefetcher.from_env(key_fn=_key_for, build_fn=_prefetch_build)

NODE_CLASS_MAPPINGS = {"MiniMaxH3ReferencePack": MiniMaxH3ReferencePack}
# The class key stays MiniMaxH3ReferencePack forever — it is what saved workflows
# reference. Only the human-facing label changes.
NODE_DISPLAY_NAME_MAPPINGS = {"MiniMaxH3ReferencePack": "MiniMax References Manager"}
