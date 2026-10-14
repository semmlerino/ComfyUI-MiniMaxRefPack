"""What reaches the ComfyUI registry, and what must never be in it.

Background, 2026-08-17. 0.3.0 published and went Active. 0.3.3 and 0.3.4 published and
both sat at NodeVersionStatusPending. The only material difference in the uploaded zips
was `tests/`, and inside it `test_migration.py`, which builds a string and executes it
under node via `subprocess.run([node, "-e", script])`. That is a fine way to test shipped
JS and a textbook automated-scanner trigger, and it had no reason to be in a package a
user installs.

Two rules follow, and this file holds both:
  1. Development apparatus does not ship.
  2. The code that DOES ship contains no dynamic execution, so there is nothing for a
     scanner to find in the first place.
"""

import pathlib
import re

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
COMFYIGNORE = REPO_ROOT / ".comfyignore"
PACKAGE = REPO_ROOT / "minimax_refpack"

# `comfy node pack` ships git-tracked files minus .comfyignore patterns, so anything here
# that is git-tracked and unignored lands in the zip.
MUST_NOT_SHIP = ("tests/", "conftest.py")

# Patterns an automated malware scanner looks for. The shipped package has never needed
# any of them: it makes HTTP calls and decodes media, and nothing more.
DYNAMIC_EXECUTION = re.compile(
    r"\bsubprocess\b|\bos\.system\b|\bPopen\b|(?<![\w.])eval\s*\(|(?<![\w.])exec\s*\(",
)


def test_a_comfyignore_exists():
    assert COMFYIGNORE.exists(), (
        "without .comfyignore, `comfy node pack` ships every git-tracked file, which is "
        "how the test suite ended up inside a published package"
    )


@pytest.mark.parametrize("pattern", MUST_NOT_SHIP)
def test_development_apparatus_is_excluded_from_the_package(pattern):
    lines = [
        line.strip() for line in COMFYIGNORE.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    assert pattern in lines, f".comfyignore must exclude {pattern!r}; it lists {lines}"


@pytest.mark.parametrize(
    "path", sorted(PACKAGE.rglob("*.py")), ids=lambda p: p.name
)
def test_the_shipped_package_never_executes_anything_dynamically(path):
    """Not defensive style: it is what keeps the registry's scanner quiet.

    If a future change genuinely needs a subprocess, that is a deliberate decision worth
    making loudly rather than discovering when a release sits in Pending for a day.
    """
    for number, line in enumerate(path.read_text().splitlines(), 1):
        code = line.split("#", 1)[0]
        assert not DYNAMIC_EXECUTION.search(code), (
            f"{path.name}:{number} uses dynamic execution in SHIPPED code:\n  {line.strip()}"
        )


def test_the_web_assets_ship():
    """The inverse failure: over-broad ignores that drop something the node needs."""
    for needed in ("web/refpack.js", "web/refpack.css", "web/lib/", "web/lib/*.js", "__init__.py",
                   "requirements.txt"):
        lines = [
            line.strip() for line in COMFYIGNORE.read_text().splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        assert needed not in lines, f"{needed} is required at runtime and must ship"


# Dynamic code in shipped JS is as visible to the scanner as it is in Python, and the
# lib modules are loaded by ComfyUI as extensions in their own right, so they may not
# reach for ComfyUI's scripts either: refpack.js imports those and passes them in.
JS_FORBIDDEN = re.compile(r"(?<![\w.])eval\s*\(|\bnew\s+Function\s*\(|scripts/(app|api)\.js")


@pytest.mark.parametrize(
    "path", sorted((REPO_ROOT / "web" / "lib").glob("*.js")), ids=lambda p: p.name
)
def test_the_shipped_lib_modules_neither_execute_strings_nor_import_comfyui(path):
    for number, line in enumerate(path.read_text().splitlines(), 1):
        assert not JS_FORBIDDEN.search(line), f"web/lib/{path.name}:{number}: {line.strip()}"
