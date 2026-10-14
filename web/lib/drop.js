// Drag-and-drop routing. Dropped files are sorted by what they ARE rather than by
// where the pointer landed: hitTest only covers DRAWN regions, so a drop on empty
// strip space has no section to report, and sorting by kind is also what makes a
// mixed drop fill all three sections in one gesture. DROP_KINDS keys the buckets, so
// it stays its own list rather than layout.js's KINDS.
export const DROP_KINDS = ["image", "video", "audio"];

// Consulted ONLY when the MIME type says nothing useful - browsers leave `type` empty
// for the less common containers (.mkv, .flac), and some hand back
// application/octet-stream for the same files. Deliberately not a fallback for every
// unrecognised type: `{name: "scan.png", type: "application/pdf"}` must be refused, not
// uploaded as an image because its name ends in .png. Not exhaustive and does not need
// to be - anything that misses lands in `rejected` under its own name, which is a
// visible refusal, not a silent one.
export const DROP_UNKNOWN_TYPES = ["", "application/octet-stream"];
export const DROP_EXTENSIONS = {
    image: ["png", "jpg", "jpeg", "gif", "webp", "bmp", "tif", "tiff", "avif", "heic", "heif"],
    video: ["mp4", "webm", "mov", "mkv", "avi", "m4v", "mpg", "mpeg", "wmv", "ogv"],
    audio: ["mp3", "wav", "flac", "ogg", "oga", "m4a", "aac", "opus", "wma", "aiff", "aif"],
};

export function dropKindOf(file) {
    // Duck-typed on {type, name}: `instanceof File` would pass in the browser and
    // reject the plain objects the tests feed it, which is the wrong way round.
    // The three prefixes are the same ones KIND_UPLOAD_META hands the file pickers.
    const type = String(file?.type ?? "").toLowerCase();
    for (const kind of DROP_KINDS) {
        if (type.startsWith(`${kind}/`)) return kind;
    }
    // A type the browser was confident about and that is not media is a refusal, not an
    // invitation to guess from the name.
    if (!DROP_UNKNOWN_TYPES.includes(type)) return null;
    const name = String(file?.name ?? "");
    const dot = name.lastIndexOf(".");
    if (dot < 0) return null;
    const ext = name.slice(dot + 1).toLowerCase();
    for (const kind of DROP_KINDS) {
        if (DROP_EXTENSIONS[kind].includes(ext)) return kind;
    }
    return null;
}

export function bucketDroppedFiles(files) {
    // -> {buckets: {image: [], video: [], audio: []}, rejected: [name, ...]}.
    // Pure: it decides nothing about caps or uploads, which stay addFiles' job.
    const buckets = {};
    for (const kind of DROP_KINDS) buckets[kind] = [];
    const rejected = [];
    for (const file of Array.from(files ?? [])) {
        const kind = dropKindOf(file);
        if (kind) buckets[kind].push(file);
        else rejected.push(String(file?.name ?? "").trim() || "(unnamed)");
    }
    return { buckets, rejected };
}

// A reference file that is a video container, by extension - for an AUDIO tile, which
// carries no probe kind of its own.
export function isVideoFile(file) {
    const name = String(file || "");
    const dot = name.lastIndexOf(".");
    return dot >= 0 && DROP_EXTENSIONS.video.includes(name.slice(dot + 1).toLowerCase());
}
