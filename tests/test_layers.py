"""Each layer imports only what sits below it (maps <- benchmarks <- station; serving beside them), and every in-repo import resolves."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = {
    "maps": {"benchmarks", "station", "serving"},
    "benchmarks": {"station", "serving"},
    "station": {"serving"},
}

# The vacuity floor: a scan that found fewer imports than this read nothing. The station is one module.
FLOOR = {"station": 20}


def _imports(layer: str) -> list[tuple[str, str]]:
    found = []
    for py in sorted(p for p in (ROOT / layer).rglob("*.py") if "node_modules" not in p.parts):
        for node in ast.walk(ast.parse(py.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                found += [(str(py.relative_to(ROOT)), a.name) for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                found.append((str(py.relative_to(ROOT)), node.module))
    return found


def _resolves(mod: str) -> bool:
    rel = ROOT.joinpath(*mod.split("."))
    return rel.with_suffix(".py").exists() or (rel / "__init__.py").exists()


@pytest.mark.parametrize("layer", sorted(FORBIDDEN))
def test_a_layer_imports_nothing_above_or_beside_it(layer):
    imports = _imports(layer)
    assert len(imports) > FLOOR.get(layer, 100)
    assert [(f, m) for f, m in imports if m.split(".")[0] in FORBIDDEN[layer]] == []


@pytest.mark.parametrize("layer", sorted(FORBIDDEN))
def test_every_in_repo_import_names_a_module_on_disk(layer):
    assert [(f, m) for f, m in _imports(layer) if m.split(".")[0] in FORBIDDEN and not _resolves(m)] == []
