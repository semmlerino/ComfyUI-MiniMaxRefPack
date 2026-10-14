// The working reference state: tags, references_json transport, the task plan, and
// the pure edits applied to it (audio-only conversion, reordering).

// ---------------------------------------------------------------------------
// The tag rule. Mirrors minimax_refpack/refs.py ReferenceSet.assign_tags() exactly:
//   1. images, slot order -> <Picture 1..n>
//   2. per video, slot order: soundtrack's <Audio j> is assigned BEFORE the video's own
//      <Video k>; standalone audio continues the same <Audio> counter afterwards
// `refs` is {images: [{file}], videos: [{file, use_soundtrack}], audios: [{file}]} — the
// working state keeps references grouped by kind, one array per kind, in socket order.
// ---------------------------------------------------------------------------
export function assignTags(refs) {
    const images = (refs && refs.images) || [];
    const videos = (refs && refs.videos) || [];
    const audios = (refs && refs.audios) || [];
    const tagged = { images: [], videos: [], audios: [] };

    images.forEach((ref, i) => tagged.images.push({ ref, tag: `<Picture ${i + 1}>` }));

    let audioN = 0;
    videos.forEach((ref, i) => {
        let audioTag = null;
        if (ref.use_soundtrack) {
            audioN += 1;
            audioTag = `<Audio ${audioN}>`;
        }
        tagged.videos.push({ ref, tag: `<Video ${i + 1}>`, audioTag });
    });

    audios.forEach((ref) => {
        audioN += 1;
        tagged.audios.push({ ref, tag: `<Audio ${audioN}>` });
    });

    return tagged;
}

// ---- references_json <-> working-state conversion + task planning ----------
// refs.py remains the backend authority; this copy gives the modal immediate
// validation instead of waiting for a queued prompt to fail.
export const TASK_ROLE_OPTIONS = {
    image: [
        ["reference_generation", "Generation guidance"],
        ["keyframe_completion", "Keyframe anchor"],
    ],
    video: [
        ["reference_generation", "Motion / camera / structure guidance"],
        ["video_editing", "Use as editing source"],
        ["video_continuation", "Use as continuation source"],
        ["audio_reuse", "Soundtrack: reuse signal"],
        ["audio_reference", "Soundtrack: reference qualities"],
    ],
    audio: [
        ["audio_reuse", "Reuse signal"],
        ["audio_reference", "Reference qualities"],
    ],
};
export const TASK_ROLE_TYPES = {
    reference_generation: "reference generation",
    keyframe_completion: "keyframe completion",
    video_editing: "video editing",
    video_continuation: "video continuation",
    audio_reuse: "audio reuse",
    audio_reference: "audio reference",
};
export const TASK_TYPE_ORDER = [
    "video editing", "video continuation", "keyframe completion",
    "reference generation", "audio reuse", "audio reference",
];
export const TASK_SPECIALIZATIONS = ["none", "character_replacement", "object_replacement"];

export function takeEdit(v) {
    return Array.isArray(v) ? v.slice() : null;
}

export function takeRoles(kind, roles) {
    const allowed = new Set((TASK_ROLE_OPTIONS[kind] || []).map(([id]) => id));
    const out = [];
    for (const role of Array.isArray(roles) ? roles : []) {
        if (allowed.has(role) && !out.includes(role)) out.push(role);
    }
    return out;
}

export function normalizeTaskPlan(raw) {
    if (!raw || typeof raw !== "object" || Array.isArray(raw)) return null;
    const mode = raw.mode === "explicit" ? "explicit" : raw.mode === "auto" ? "auto" : null;
    if (!mode) return null;
    let specialization = TASK_SPECIALIZATIONS.includes(raw.specialization)
        ? raw.specialization
        : "none";
    if (mode !== "explicit") specialization = "none";
    return { mode, specialization };
}

export function takeControl(raw) {
    return raw && typeof raw === "object" && !Array.isArray(raw)
        ? JSON.parse(JSON.stringify(raw)) : null;
}

export function fromReferencesList(list) {
    const refs = { images: [], videos: [], audios: [], taskPlan: null };
    for (const r of list || []) {
        if (!r || typeof r.file !== "string") continue;
        if (r.kind === "image")
            refs.images.push({
                file: r.file, roles: takeRoles("image", r.roles),
                missing: !!r.missing, crop: takeEdit(r.crop),
            });
        else if (r.kind === "video")
            refs.videos.push({
                file: r.file,
                // Backend and legacy workflow default: a video's soundtrack is on
                // unless the serialized reference explicitly disables it.
                use_soundtrack: r.use_soundtrack !== false,
                primary: r.primary === true,
                roles: takeRoles("video", r.roles),
                missing: !!r.missing,
                crop: takeEdit(r.crop),
                trim: takeEdit(r.trim),
            });
        else if (r.kind === "audio")
            refs.audios.push({
                file: r.file, roles: takeRoles("audio", r.roles),
                missing: !!r.missing, trim: takeEdit(r.trim),
            });
    }
    return refs;
}

export function fromReferencesEnvelope(data) {
    const refs = fromReferencesList((data && data.references) || []);
    refs.taskPlan = normalizeTaskPlan(data && data.task_plan);
    refs.control = takeControl(data && data.control);
    return refs;
}

export function toReferencesList(refs) {
    const out = [];
    const withMetadata = (d, r) => {
        if (Array.isArray(r.crop)) d.crop = r.crop.slice();
        if (Array.isArray(r.trim)) d.trim = r.trim.slice();
        if (Array.isArray(r.roles) && r.roles.length) d.roles = r.roles.slice();
        return d;
    };
    for (const r of refs.images)
        out.push(withMetadata({ kind: "image", file: r.file }, r));
    for (const r of refs.videos) {
        const video = withMetadata(
            { kind: "video", file: r.file, use_soundtrack: !!r.use_soundtrack }, r
        );
        if (r.primary) video.primary = true;
        out.push(video);
    }
    for (const r of refs.audios)
        out.push(withMetadata({ kind: "audio", file: r.file }, r));
    return out;
}

export function toReferencesEnvelope(refs) {
    const envelope = { references: toReferencesList(refs) };
    if (refs.taskPlan) {
        envelope.task_plan = { mode: refs.taskPlan.mode };
        if (refs.taskPlan.specialization && refs.taskPlan.specialization !== "none") {
            envelope.task_plan.specialization = refs.taskPlan.specialization;
        }
    }
    if (refs.control) envelope.control = takeControl(refs.control);
    return envelope;
}

export function newReference(kind, file) {
    const reference = { file, roles: [] };
    if (kind === "video") {
        reference.use_soundtrack = true;
        reference.primary = false;
    }
    return reference;
}

export function deriveTaskPlanState(refs, allowPrimaryReorder = false) {
    const plan = refs && refs.taskPlan;
    if (!plan) return { mode: "legacy", prefix: "Legacy job type", taskTypes: [], error: null };
    if (plan.mode === "auto") return { mode: "auto", prefix: "Auto detect", taskTypes: [], error: null };

    const all = [
        ...(refs.images || []).map((ref, index) => ({ kind: "image", index, ref })),
        ...(refs.videos || []).map((ref, index) => ({ kind: "video", index, ref })),
        ...(refs.audios || []).map((ref, index) => ({ kind: "audio", index, ref })),
    ];
    const roleCount = all.reduce((n, item) => n + (item.ref.roles || []).length, 0);
    if (!roleCount) {
        return { mode: "explicit", prefix: "No roles", taskTypes: [],
                 error: "Assign at least one reference role." };
    }

    const drivers = [];
    const primaries = [];
    for (const item of all.filter((item) => item.kind === "video")) {
        if (item.ref.primary) primaries.push(item.index);
        for (const role of item.ref.roles || []) {
            if (role === "video_editing" || role === "video_continuation") {
                drivers.push({ index: item.index, role });
            }
        }
        if (!item.ref.use_soundtrack && (item.ref.roles || []).some(
            (role) => role === "audio_reuse" || role === "audio_reference"
        )) {
            return { mode: "explicit", prefix: "Invalid plan", taskTypes: [],
                     error: `${item.ref.file}'s soundtrack is disabled.` };
        }
    }
    if (primaries.length > 1) {
        return { mode: "explicit", prefix: "Invalid plan", taskTypes: [],
                 error: "Choose only one primary video." };
    }
    if (drivers.length > 1) {
        return { mode: "explicit", prefix: "Invalid plan", taskTypes: [],
                 error: "Choose only one primary video for editing or continuation." };
    }
    if (primaries.length && !drivers.length) {
        return { mode: "explicit", prefix: "Invalid plan", taskTypes: [],
                 error: 'Select "Use as editing source" or "Use as continuation source" for the primary video.' };
    }
    if (primaries.length && drivers.length && primaries[0] !== drivers[0].index) {
        return { mode: "explicit", prefix: "Invalid plan", taskTypes: [],
                 error: "Put the primary designation and editing/continuation role on the same video." };
    }
    const primaryIndex = primaries.length ? primaries[0]
        : drivers.length ? drivers[0].index : null;
    if (primaryIndex !== null && primaryIndex !== 0 && !allowPrimaryReorder) {
        return { mode: "explicit", prefix: "Invalid plan", taskTypes: [],
                 error: "The primary edit/continuation source must be <Video 1>." };
    }

    const roles = new Set(all.flatMap((item) => item.ref.roles || []));
    const specialization = plan.specialization || "none";
    if (specialization !== "none" && !roles.has("video_editing")) {
        return { mode: "explicit", prefix: "Invalid plan", taskTypes: [],
                 error: 'A replacement specialization requires "Use as editing source".' };
    }
    if (specialization !== "none" && !(refs.images || []).some(
        (ref) => (ref.roles || []).includes("reference_generation")
    )) {
        return { mode: "explicit", prefix: "Invalid plan", taskTypes: [],
                 error: "A replacement specialization requires image generation guidance." };
    }

    const used = new Set([...roles].map((role) => TASK_ROLE_TYPES[role]).filter(Boolean));
    const taskTypes = TASK_TYPE_ORDER.filter((taskType) => used.has(taskType));
    return {
        mode: "explicit",
        prefix: `[${taskTypes.join(" + ")}]`,
        taskTypes,
        error: null,
    };
}

export function ensureReplacementPrimaryRole(refs) {
    const plan = refs && refs.taskPlan;
    if (!plan || plan.mode !== "explicit" || plan.specialization === "none") return false;
    const primary = (refs.videos || []).find((reference) => reference.primary === true);
    if (!primary || (primary.roles || []).some(
        (role) => role === "video_editing" || role === "video_continuation"
    )) return false;
    primary.roles = takeRoles("video", [...(primary.roles || []), "video_editing"]);
    return true;
}

export function referencesFingerprint(refs) {
    const envelope = toReferencesEnvelope(refs);
    // A preparation can finish while the role planner is open. Its independent
    // control receipt does not change the reference assets or their roles.
    delete envelope.control;
    return JSON.stringify(envelope);
}

export function normalizePrimaryVideo(refs) {
    const videos = refs && Array.isArray(refs.videos) ? refs.videos : [];
    const index = videos.findIndex((reference) => reference.primary === true);
    if (index > 0) videos.unshift(...videos.splice(index, 1));
    return refs;
}

export function emptyRefs() {
    return { images: [], videos: [], audios: [], taskPlan: null, control: null };
}

export function cloneRefs(refs) {
    const clone = (r) => ({
        ...r,
        roles: Array.isArray(r.roles) ? r.roles.slice() : [],
        crop: Array.isArray(r.crop) ? r.crop.slice() : r.crop,
        trim: Array.isArray(r.trim) ? r.trim.slice() : r.trim,
    });
    return {
        images: refs.images.map(clone),
        videos: refs.videos.map(clone),
        audios: refs.audios.map(clone),
        taskPlan: refs.taskPlan ? { ...refs.taskPlan } : null,
        control: takeControl(refs.control),
    };
}

// "Use audio only": turn a video reference into an audio reference on the same file,
// so MiniMax gets the clip's sound as <Audio n> and no <Video k> at all. The backend
// already accepts a video file under kind "audio" (media.load_audio decodes its last
// decodable track).
export const AUDIO_ONLY_ROLES = ["audio_reuse", "audio_reference"];

export function videoToAudioOnly(refs, index, audioCap) {
    // -> {refs, error}. Pure: returns a new top-level object, never mutates `refs`.
    // Kept: file, trim, missing and the audio roles. Dropped: crop, primary and the
    // video roles. A removed primary promotes nothing (removeRef's rule), and an
    // explicit plan may be left invalid - the caller reports that, it does not repair.
    const videos = Array.isArray(refs?.videos) ? refs.videos : [];
    const audios = Array.isArray(refs?.audios) ? refs.audios : [];
    if (!Number.isInteger(index) || index < 0 || index >= videos.length) {
        return { refs, error: "No such video reference." };
    }
    if (audios.length >= audioCap) {
        return { refs, error: `The audio row is full (${audioCap}/${audioCap}).` };
    }
    const video = videos[index];
    const audio = {
        file: video.file,
        roles: (Array.isArray(video.roles) ? video.roles : []).filter((role) =>
            AUDIO_ONLY_ROLES.includes(role)
        ),
        missing: !!video.missing,
    };
    if (Array.isArray(video.trim)) audio.trim = video.trim.slice();
    return {
        refs: {
            ...refs,
            videos: videos.filter((_, i) => i !== index),
            audios: [...audios, audio],
        },
        error: null,
    };
}

// Drag-to-swap inside one row. Tiles are canvas-painted, so this is array math,
// not HTML5 drag (that path is file drops).
//
// Semantics: MOVE, not pairwise-exchange with the drop target. Dragging the first
// of three onto the last yields [b, c, a], which is also what successive
// neighbour-swaps do as the pointer crosses each midpoint. Picture 1 onto
// Picture 2 is just the two-item case of the same rule.
export function moveRefInKind(refs, kind, fromIndex, toIndex) {
    // -> {refs, error}. Pure: new top-level object on a real move, never mutates
    // `refs`. Same-index is a documented no-op that returns the input object.
    const key = `${kind}s`;
    const arr = Array.isArray(refs?.[key]) ? refs[key] : [];
    if (!Number.isInteger(fromIndex) || !Number.isInteger(toIndex)
        || fromIndex < 0 || toIndex < 0
        || fromIndex >= arr.length || toIndex >= arr.length) {
        return { refs, error: "No such reference." };
    }
    if (fromIndex === toIndex) return { refs, error: null };
    const next = arr.slice();
    const [item] = next.splice(fromIndex, 1);
    next.splice(toIndex, 0, item);
    return { refs: { ...refs, [key]: next }, error: null };
}

export function slotIndexAtX(x, count, x0, tile, gap) {
    // Nearest tile centre. Crossing the midpoint between two tiles is the swap.
    // count < 2 has nowhere to go; a non-positive stride is a caller bug, not a
    // throw — return 0 rather than Infinity.
    if (!Number.isInteger(count) || count <= 1) return 0;
    const stride = tile + gap;
    if (!(stride > 0)) return 0;
    const i = Math.round((x - x0 - tile / 2) / stride);
    if (i < 0) return 0;
    if (i > count - 1) return count - 1;
    return i;
}
