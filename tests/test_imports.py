"""Every command module must import from the repository root."""

import importlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULES = sorted(
    ".".join(path.relative_to(ROOT).with_suffix("").parts)
    for path in (ROOT / "scripts").rglob("*.py")
    if path.name != "__init__.py"
)


@pytest.mark.parametrize("module", MODULES)
def test_script_module_imports(module: str) -> None:
    importlib.import_module(module)
