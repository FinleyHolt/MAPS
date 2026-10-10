"""maps is the base layer and imports nothing above it.

maps.* may import maps, the stdlib, and third-party packages — never
benchmarks.* or serving.*. A new import that reaches upward or sideways
fails here before it lands.
"""

from __future__ import annotations

import ast
from pathlib import Path

MAPS = Path(__file__).resolve().parents[1]

FORBIDDEN_ROOTS = ("benchmarks", "serving")


def _imports_of(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    mods: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods.extend(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            mods.append(node.module)
    return mods


def test_maps_never_imports_up_or_sideways():
    offenders: list[str] = []
    scanned = 0
    for py in sorted(MAPS.rglob("*.py")):
        if "__pycache__" in py.parts:
            continue
        scanned += 1
        for mod in _imports_of(py):
            root = mod.split(".", 1)[0]
            if root in FORBIDDEN_ROOTS:
                offenders.append(f"{py.relative_to(MAPS)}: import {mod}")
    assert offenders == [], (
        "maps must stay the base layer (maps/stdlib/third-party only); "
        "found: " + "; ".join(offenders)
    )
    # VACUITY FLOOR: a walk that reads nothing reports no offenders. A moved
    # package root or a broken rglob would otherwise pass green.
    assert scanned >= 100, f"only {scanned} files walked under {MAPS}"


def test_maps_internal_imports_resolve():
    # Every maps.* import names a module that exists on disk — a rename
    # that leaves a stale internal import fails here, not at runtime.
    pkg_root = MAPS.parent
    missing: list[str] = []
    scanned = 0
    checked = 0
    for py in sorted(MAPS.rglob("*.py")):
        if "__pycache__" in py.parts:
            continue
        scanned += 1
        for mod in _imports_of(py):
            if not mod.startswith("maps"):
                continue
            checked += 1
            rel = Path(*mod.split("."))
            if not (
                (pkg_root / rel).with_suffix(".py").exists()
                or (pkg_root / rel / "__init__.py").exists()
            ):
                missing.append(f"{py.relative_to(MAPS)}: import {mod}")
    assert missing == [], "stale internal imports: " + "; ".join(missing)
    # VACUITY FLOOR: both the walk and the resolution must have happened.
    assert scanned >= 100, f"only {scanned} files walked under {MAPS}"
    assert checked >= 100, f"only {checked} maps.* imports resolved"
