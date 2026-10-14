// ---------------------------------------------------------------------------
// Structured logging. Same line shape as the Python side (minimax_refpack/logs.py):
// `[MiniMaxRefPack] event=<name> key=value`, so one grep covers the browser console
// and the ComfyUI console, and a bug report from either reads the same way.
//
// info = state changes worth seeing by default (uploads, edits, previews, config).
// debug = per-tile chatter (thumb retries), hidden until the console's Verbose level.
// warn = something the user will notice going wrong.
// ---------------------------------------------------------------------------

export const LOG_PREFIX = "[MiniMaxRefPack]";

export function logValue(value) {
    if (typeof value === "boolean") return value ? "true" : "false";
    if (typeof value === "number") {
        // 3dp, trailing zeros dropped - matches logs.py so the two consoles agree
        return Number.isInteger(value) ? String(value) : String(Math.round(value * 1000) / 1000);
    }
    if (Array.isArray(value)) return `[${value.map(logValue).join(",")}]`;
    const text = String(value);
    if (!text) return '""';
    return /[ "]/.test(text) ? `"${text.replace(/"/g, "'")}"` : text;
}

export function logLine(event, fields) {
    let line = `${LOG_PREFIX} event=${event}`;
    for (const [key, value] of Object.entries(fields || {})) {
        if (value === undefined || value === null) continue;
        line += ` ${key}=${logValue(value)}`;
    }
    return line;
}

export const mlog = (event, fields) => console.info(logLine(event, fields));
export const mdebug = (event, fields) => console.debug(logLine(event, fields));
export const mwarn = (event, fields) => console.warn(logLine(event, fields));
