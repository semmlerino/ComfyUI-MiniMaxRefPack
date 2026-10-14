"""Run the shipped browser modules in web/lib under node.

Tests import the real files by URL, so they exercise the code ComfyUI serves, not a copy.
Imports are named, one per export, never `Object.assign(globalThis, module)`: a global
copy of every export would let a module that forgot one of its own imports resolve the
name anyway, pass here and throw a ReferenceError in the browser.

Code under test that needs browser globals (`document`, `window`, `setTimeout`) gets
fakes installed on `globalThis` by the test's own script; lexical `const document`
declarations in the test script are invisible to the imported modules.
"""

import functools
import json
import pathlib
import shutil
import subprocess
from typing import Any

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
WEB = REPO_ROOT / "web"
LIB = WEB / "lib"
MODULES = tuple(sorted(path.stem for path in LIB.glob("*.js")))

NODE = shutil.which("node")
requires_node = pytest.mark.skipif(NODE is None, reason="node is not installed")


def module_url(name: str) -> str:
    path = LIB / f"{name}.js"
    assert path.is_file(), f"no module web/lib/{name}.js"
    return path.as_uri()


def _node(script: str, timeout: float) -> subprocess.CompletedProcess[str]:
    assert NODE is not None
    return subprocess.run(
        [NODE, "--input-type=module", "-e", script],
        capture_output=True, text=True, check=False, timeout=timeout,
    )


@functools.cache
def exports(name: str) -> tuple[str, ...]:
    """The module's export names, as node reports them."""
    url = json.dumps(module_url(name))
    proc = _node(f"console.log(JSON.stringify(Object.keys(await import({url}))));", 10)
    assert proc.returncode == 0, f"web/lib/{name}.js does not import under node:\n{proc.stderr}"
    return tuple(json.loads(proc.stdout))


def imports(*modules: str) -> str:
    """One named import line per module, covering all of its exports."""
    return "".join(
        f"import {{ {', '.join(exports(name))} }} from {json.dumps(module_url(name))};\n"
        for name in modules
    )


def run(body: str, *modules: str, timeout: float = 10) -> Any:
    """Run `body` after importing `modules`; it must print exactly one JSON value."""
    proc = _node(imports(*modules) + body, timeout)
    assert proc.returncode == 0, f"node failed:\n{proc.stderr}"
    return json.loads(proc.stdout.strip())


def evaluate(expression: str, *modules: str, setup: str = "", timeout: float = 10) -> Any:
    """The JSON value of `expression` (awaited) after `setup`, with `modules` imported."""
    return run(f"{setup}\nconsole.log(JSON.stringify(await ({expression})));\n", *modules, timeout=timeout)
