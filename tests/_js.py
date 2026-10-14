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


# A fake DOM just deep enough to build and click the lib/*_ui.js dialogs under node:
# elements keep their children, classes, dataset, style and every listener; `fire`
# runs an element's listeners for one event. Install it with `setup=FAKE_DOM`.
# Lookups walk the tree (`all`, `byClass`, `byText`) rather than parsing selectors, so
# querySelectorAll finds nothing.
FAKE_DOM = """
class Element {
    constructor(tag) {
        this.tag = tag; this.children = []; this.style = {}; this.dataset = {}; this.attributes = {};
        this.handlers = {}; this.textContent = ''; this.value = ''; this.disabled = false;
        const classes = new Set(); this.classes = classes;
        this.classList = {add: name => classes.add(name), remove: name => classes.delete(name),
            contains: name => classes.has(name),
            toggle: (name, force) => (force ?? !classes.has(name)) ? classes.add(name) : classes.delete(name)};
    }
    set className(value) { this.classes.clear(); String(value).split(' ').filter(Boolean).forEach(c => this.classes.add(c)); }
    get className() { return [...this.classes].join(' '); }
    appendChild(child) { this.children.push(child); child.parentElement = this; return child; }
    remove() { const parent = this.parentElement; if (parent) parent.children.splice(parent.children.indexOf(this), 1); this.parentElement = null; }
    setAttribute(name, value) { this.attributes[name] = value; }
    getAttribute(name) { return this.attributes[name]; }
    addEventListener(name, callback) { (this.handlers[name] ||= []).push(callback); }
    removeEventListener(name, callback) { this.handlers[name] = (this.handlers[name] || []).filter(c => c !== callback); }
    replaceChildren(...children) { this.children = []; children.forEach(child => this.appendChild(child)); }
    querySelectorAll() { return []; }
    dispatchEvent(event) { fire(this, event.type, event); }
    focus() { document.activeElement = this; }
    getBoundingClientRect() { return {left: 0, top: 0, width: 100, height: 100}; }
    pause() { this.paused = true; }
    play() { this.paused = false; return Promise.resolve(); }
}
function fire(element, name, event = {}) {
    const full = {target: element, preventDefault() {}, stopPropagation() {}, ...event};
    const property = element['on' + name];
    if (typeof property === 'function') property(full);
    for (const callback of element.handlers[name] || []) callback(full);
}
const all = root => [root, ...root.children.flatMap(all)];
const byClass = (root, name) => all(root).filter(el => el.classList && el.classList.contains(name));
const byText = (root, text) => all(root).find(el => el.textContent === text);
globalThis.document = Object.assign(new Element('#document'), {
    body: new Element('body'), activeElement: null,
    createElement: tag => new Element(tag),
    createTextNode: text => Object.assign(new Element('#text'), {textContent: text}),
});
globalThis.window = new Element('window');
globalThis.requestAnimationFrame = () => 0;
// RefPack's log lines (lib/log.js) collect here: on stdout they would precede the JSON result.
const logged = [];
console.info = console.debug = console.warn = (line) => logged.push(line);
globalThis.cancelAnimationFrame = () => {};
"""
