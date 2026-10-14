"""The browser modules hold together: each loads alone, and refpack.js asks only for what exists.

ComfyUI imports every web/**/*.js as an extension, so each lib module must load with no
DOM and no ComfyUI around it - the same bare node these tests use. Importing a module also
links its own named imports, so a lib module asking another for a name it does not export
fails here.

refpack.js itself cannot be imported outside ComfyUI (it imports ../../scripts/app.js), so
its imports from ./lib/ are checked against the modules' real export lists instead. A
misspelled name there is a link error that stops the whole extension from loading.
"""

import re

import pytest
from _js import LIB, MODULES, WEB, exports, requires_node

LIB_IMPORT = re.compile(r'import\s*\{([^}]*)\}\s*from\s*"\./lib/([\w-]+)\.js"\s*;', re.DOTALL)


def _refpack_imports() -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for names, module in LIB_IMPORT.findall((WEB / "refpack.js").read_text()):
        found.setdefault(module, []).extend(n.strip() for n in names.split(",") if n.strip())
    return found


def test_every_lib_module_is_found():
    assert "references" in MODULES and "control_ui" in MODULES


@requires_node
@pytest.mark.parametrize("module", MODULES)
def test_each_lib_module_loads_bare_under_node(module):
    assert exports(module)


def test_refpack_imports_from_lib_with_one_statement_per_module():
    text = (WEB / "refpack.js").read_text()
    assert len(re.findall(r'from\s*"\./lib/', text)) == len(LIB_IMPORT.findall(text))
    assert _refpack_imports(), "refpack.js imports nothing from ./lib/"


@requires_node
def test_refpack_imports_only_names_the_lib_modules_export():
    for module, names in _refpack_imports().items():
        assert (LIB / f"{module}.js").is_file(), f"refpack.js imports missing ./lib/{module}.js"
        missing = sorted(set(names) - set(exports(module)))
        assert not missing, f"refpack.js imports {missing} from ./lib/{module}.js, which does not export them"
