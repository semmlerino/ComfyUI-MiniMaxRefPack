"""Named Hearmeman-style rewriter system prompts.

`default` keeps packaged six-section (or a non-blank ⚙ `system_prompt`). Named
pins are full replacements and force `job_type=standard` so the classifier does
not run.

Files live in `rewrite_pins/<name>.txt`. The combo is append-only on the node:
Litegraph restores `widgets_values` by index, so this widget is last.
"""

from __future__ import annotations

from pathlib import Path

DEFAULT = "default"
PIN_DIR = Path(__file__).resolve().parent / "rewrite_pins"

# Canvas combo order. Keep MiniMax freeze_outbox_prompts.PINS names in step.
NAMES: tuple[str, ...] = (
    "hearmeman",
    "liberty",
    "talk",
    "wild",
    "horny",
    "act",
    "rub",
    "slow",
    "lost",
    "lens",
    "t2v_describe",
)

BLURBS: dict[str, str] = {
    "hearmeman": "Hearmeman three-field (favs)",
    "liberty": "micro-motion, no camera-brand filler",
    "talk": "vague: one in-language line, keep plate objects",
    "wild": "talk plus sloppier motion, nastier talk",
    "horny": "compulsive, blurted line",
    "act": "unhinged body; gasp not a speech",
    "rub": "kiss/lick/grind, not slam",
    "slow": "languid stuck contact; partner off-screen",
    "lost": "never looks at the lens; partner off-screen",
    "lens": "never looks away from the lens; partner off-screen",
    "t2v_describe": "text-only: transcribe references into prose, no tags",
}


def combo_values() -> list[str]:
    return [DEFAULT, *NAMES]


def normalize(value: str | None) -> str:
    key = (value or "").strip().lower()
    if key in ("", "graph"):
        return DEFAULT
    return key


def is_named(value: str | None) -> bool:
    return normalize(value) in NAMES


def load(name: str) -> str:
    key = normalize(name)
    if key not in NAMES:
        known = ", ".join(NAMES)
        raise ValueError(f"unknown rewrite pin {name!r}; choose one of: {known}")
    path = PIN_DIR / f"{key}.txt"
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise ValueError(f"rewrite pin {key} file is missing: {path}") from error
    if not text.strip():
        raise ValueError(f"rewrite pin {key} file is empty: {path}")
    return text


def apply(system_prompt: str, rewrite_pin: str) -> tuple[str, str | None]:
    """Return ``(system_prompt, job_type or None)``.

    A non-blank ``system_prompt`` widget always wins. A named pin loads its file
    and returns ``job_type='standard'``. ``default`` leaves the packaged path.
    """
    if (system_prompt or "").strip():
        return system_prompt, None
    if is_named(rewrite_pin):
        return load(rewrite_pin), "standard"
    return system_prompt or "", None
