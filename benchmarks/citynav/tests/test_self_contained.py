"""benchmarks.citynav is self-contained.

The folder serves ONLY the benchmark: it may import maps.* (the core
library below it), the stdlib, and third-party packages — never serving.*.
A new import that reaches sideways or upward fails here before it lands.
"""

from __future__ import annotations

import ast
from pathlib import Path

CITYNAV = Path(__file__).resolve().parents[1]

FORBIDDEN_ROOTS = ("serving",)


def _imports_of(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    mods: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods.extend(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            mods.append(node.module)
    return mods


def test_citynav_never_imports_sideways_or_upward():
    offenders: list[str] = []
    scanned = 0
    for py in sorted(CITYNAV.rglob("*.py")):
        if "__pycache__" in py.parts:
            continue
        scanned += 1
        for mod in _imports_of(py):
            root = mod.split(".", 1)[0]
            if root in FORBIDDEN_ROOTS:
                offenders.append(f"{py.relative_to(CITYNAV)}: import {mod}")
    assert offenders == [], (
        "benchmarks.citynav must stay self-contained (maps/stdlib/"
        "third-party only); found: " + "; ".join(offenders)
    )
    # VACUITY FLOOR: a walk that reads nothing reports no offenders.
    assert scanned >= 50, f"only {scanned} files walked under {CITYNAV}"


def test_citynav_internal_imports_resolve():
    # Every benchmarks.citynav.* import names a module that exists on disk —
    # a rename that leaves a stale internal import fails here, not at runtime.
    pkg_root = CITYNAV.parents[1]
    missing: list[str] = []
    scanned = 0
    for py in sorted(CITYNAV.rglob("*.py")):
        if "__pycache__" in py.parts:
            continue
        scanned += 1
        for mod in _imports_of(py):
            if not mod.startswith("benchmarks.citynav"):
                continue
            rel = Path(*mod.split("."))
            if not (
                (pkg_root / rel).with_suffix(".py").exists()
                or (pkg_root / rel / "__init__.py").exists()
            ):
                missing.append(f"{py.relative_to(CITYNAV)}: import {mod}")
    assert missing == [], "stale internal imports: " + "; ".join(missing)
    # VACUITY FLOOR: the walk must have happened.
    assert scanned >= 50, f"only {scanned} files walked under {CITYNAV}"
