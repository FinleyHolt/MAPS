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


# What a package's own modules (its tests apart) never import. Within the upper
# layers: the bench value and the run record sit under every bench, and the
# flight harness and the station reach a bench's engine through the bench
# value they are handed, never by import.
_ENGINE = ("maps.perception", "maps.llm", "maps.world_state", "maps.query", "maps.render", "maps.library",
           "maps.mission")
APART = {
    # Within the engine: perception, the transport and the world state stand
    # alone; the query reads none of them; the pictures read perception and
    # the query; the library reads the query; the mission composes them all.
    "maps/perception": tuple(p for p in _ENGINE if p != "maps.perception"),
    "maps/llm": tuple(p for p in _ENGINE if p != "maps.llm"),
    "maps/world_state": tuple(p for p in _ENGINE if p != "maps.world_state"),
    "maps/query": tuple(p for p in _ENGINE if p != "maps.query"),
    "maps/render": ("maps.llm", "maps.world_state", "maps.library", "maps.mission"),
    "maps/library": ("maps.perception", "maps.llm", "maps.world_state", "maps.render", "maps.mission"),
    "benchmarks/bench.py": ("benchmarks.citynav", "benchmarks.calcity", "benchmarks.flight", "benchmarks.run_record"),
    "benchmarks/run_record": ("benchmarks.citynav", "benchmarks.calcity", "benchmarks.flight"),
    "benchmarks/citynav": ("benchmarks.flight",),
    "benchmarks/calcity": ("benchmarks.flight",),
    "benchmarks/flight": ("benchmarks.citynav",),
    "station": ("benchmarks.citynav",),
}
# The vacuity floor of each: the fewest imports a scan that read the package finds.
APART_FLOOR = {"benchmarks/bench.py": 5, "benchmarks/run_record": 30, "benchmarks/citynav": 100,
               "benchmarks/calcity": 30, "benchmarks/flight": 100, "station": 20,
               "maps/perception": 60, "maps/llm": 30, "maps/world_state": 10, "maps/query": 40,
               "maps/render": 20, "maps/library": 40}


def _imports(layer: str, tests: bool = True) -> list[tuple[str, str]]:
    found = []
    root = ROOT / layer
    files = [root] if root.is_file() else root.rglob("*.py")
    for py in sorted(p for p in files if "node_modules" not in p.parts and (tests or "tests" not in p.parts)):
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


@pytest.mark.parametrize("package", sorted(APART))
def test_a_package_imports_none_of_the_packages_it_stands_apart_from(package):
    imports = _imports(package, tests=False)
    assert len(imports) > APART_FLOOR[package]
    assert [(f, m) for f, m in imports if any(m == a or m.startswith(a + ".") for a in APART[package])] == []
