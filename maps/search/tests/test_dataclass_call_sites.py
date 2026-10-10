"""Every construction of an imported dataclass names fields that exist.

Deleting a field is easy; finding the call site that still passes it is not.
A stale keyword on a construction no unit test reaches survives a full green
pytest and only surfaces as a runtime ``TypeError`` mid-episode.

The set of tracked classes is DERIVED, not listed. A hand-list rots in the
direction that hurts: of the four classes this file used to name, three had no
construction site outside their own module at all, while dozens of classes
went unwatched. What matters is not which class someone thought of, it is
whether a construction is far enough from the definition that nobody sees the
two together -- and an import is exactly that distance.

This is a source scan, not an import test: it reads every dataclass definition
under the three packages, then every call that constructs one through an
import, and checks the keywords against the real field set.
"""

from __future__ import annotations

import ast
import pathlib


REPO = pathlib.Path(__file__).resolve().parents[3]

PACKAGES = ("maps", "benchmarks")

_DATACLASS_DECORATORS = {"dataclass", "dataclasses.dataclass"}


def _decorator_names(node: ast.ClassDef) -> set[str]:
    out: set[str] = set()
    for dec in node.decorator_list:
        target = dec.func if isinstance(dec, ast.Call) else dec
        if isinstance(target, ast.Name):
            out.add(target.id)
        elif isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
            out.add(f"{target.value.id}.{target.attr}")
    return out


def _py_files(root: str):
    for path in sorted((REPO / root).rglob("*.py")):
        if "__pycache__" in str(path):
            continue
        yield path


def _parse(path: pathlib.Path) -> ast.Module | None:
    try:
        return ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        return None


def _definitions() -> dict[tuple[str, str], tuple[str, set[str]]]:
    """``(package, class_name) -> (defining_module, field_names)``.

    Keyed by package as well as name so ``Observation`` in
    ``maps.perception`` and ``Observation`` in ``maps.search`` -- or
    ``Pose`` in ``benchmarks`` and in ``robots`` -- stay separate contracts.
    A name-keyed map would union their fields and go quietly vacuous.
    """
    out: dict[tuple[str, str], tuple[str, set[str]]] = {}
    for pkg in PACKAGES:
        for path in _py_files(pkg):
            tree = _parse(path)
            if tree is None:
                continue
            rel = str(path.relative_to(REPO))
            for node in ast.walk(tree):
                if not isinstance(node, ast.ClassDef):
                    continue
                if not (_decorator_names(node) & _DATACLASS_DECORATORS):
                    continue
                fields = {
                    stmt.target.id
                    for stmt in node.body
                    if isinstance(stmt, ast.AnnAssign)
                    and isinstance(stmt.target, ast.Name)
                }
                key = (pkg, node.name)
                if key in out:
                    # Two dataclasses of one name in one package: union the
                    # fields rather than pick, so the guard stays sound (it
                    # can miss, never false-alarm).
                    prev_mod, prev_fields = out[key]
                    out[key] = (f"{prev_mod} + {rel}", prev_fields | fields)
                else:
                    out[key] = (rel, fields)
    return out


def _imported_dataclass_aliases(tree: ast.Module) -> dict[str, tuple[str, str, str]]:
    """``local_name -> (defining_package, real_class_name, imported_module)`` for this module.

    Resolves ``from maps.perception.geometry.sensor_spec import SensorSpec
    as _Spec``: the call site reads ``_Spec(...)``, and a scan keyed on the
    bare callee name never sees it. The module the name came from rides
    along so a plain class that shares a dataclass's name in the same
    package (the sim's ``Camera`` beside the discovery's) is not read as it.
    """
    out: dict[str, tuple[str, str, str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or not node.module:
            continue
        pkg = node.module.split(".")[0]
        if pkg not in PACKAGES:
            continue
        for alias in node.names:
            out[alias.asname or alias.name] = (pkg, alias.name, node.module)
    return out


def _defined_under(defining: str, module: str) -> bool:
    """Whether the import's ``module`` is a file that defines the class, or a
    package above one (a re-export through ``__init__``); ``defining`` is the
    definition map's path, or several joined by `` + ``."""
    root = module.replace(".", "/")
    return any(rel == root + ".py" or rel.startswith(root + "/") for rel in defining.split(" + "))


def _construction_sites(defs):
    """Yield ``(consumer_rel, lineno, def_key, kwargs)`` for every call that
    constructs a dataclass the module imported."""
    for pkg in PACKAGES:
        for path in _py_files(pkg):
            tree = _parse(path)
            if tree is None:
                continue
            rel = str(path.relative_to(REPO))
            aliases = _imported_dataclass_aliases(tree)
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                callee = node.func
                local = getattr(callee, "id", None) or getattr(callee, "attr", None)
                if not local:
                    continue
                resolved = aliases.get(local)
                if resolved is None:
                    continue
                def_pkg, cls_name, module = resolved
                key = (def_pkg, cls_name)
                if key not in defs or not _defined_under(defs[key][0], module):
                    continue
                kwargs = {k.arg for k in node.keywords if k.arg}
                if kwargs:
                    yield rel, node.lineno, key, kwargs


def _replace_sites(defs):
    """Yield ``(consumer_rel, lineno, def_key, kwargs)`` for every
    ``dataclasses.replace(<expr>, kw=...)`` whose target type can be read off
    the expression.

    ``replace`` is the blind spot a callee-name scan cannot see: the callee is
    ``replace``, not the class. Only the forms whose first argument names the
    class are resolvable; anything else is skipped.
    """
    for pkg in PACKAGES:
        for path in _py_files(pkg):
            tree = _parse(path)
            if tree is None:
                continue
            rel = str(path.relative_to(REPO))
            yield from _replace_sites_in_tree(tree, rel, defs)


def _replace_sites_in_tree(tree, rel: str, defs):
    """The per-file replace extraction — separated so the vacuity floor can
    prove the MECHANISM on a planted fixture even when the repo legitimately
    has zero class-resolvable replace sites (the corpus can go empty; the
    parser must not)."""
    aliases = _imported_dataclass_aliases(tree)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        is_replace = (
            isinstance(fn, ast.Attribute) and fn.attr == "replace"
        ) or (isinstance(fn, ast.Name) and fn.id == "replace")
        if not is_replace or not node.args:
            continue
        # First arg is `Cls(...)`, `Cls.factory(...)` or a bare `Cls`.
        target = node.args[0]
        while isinstance(target, ast.Call):
            target = target.func
        while isinstance(target, ast.Attribute):
            target = target.value
        local = getattr(target, "id", None)
        if not local:
            continue
        resolved = aliases.get(local)
        if resolved is None:
            continue
        def_pkg, cls_name, module = resolved
        key = (def_pkg, cls_name)
        if key not in defs or not _defined_under(defs[key][0], module):
            continue
        kwargs = {k.arg for k in node.keywords if k.arg}
        if kwargs:
            yield rel, node.lineno, key, kwargs


DEFS = _definitions()
CONSTRUCTIONS = sorted(_construction_sites(DEFS))
REPLACEMENTS = sorted(_replace_sites(DEFS))


def _offenders(sites) -> list[str]:
    out: list[str] = []
    for rel, lineno, key, kwargs in sites:
        module, fields = DEFS[key]
        unknown = sorted(kwargs - fields)
        if unknown:
            out.append(
                f"{rel}:{lineno} passes {unknown} to {key[1]} "
                f"(defined in {module}, fields: {sorted(fields)})"
            )
    return out


def test_the_scan_still_finds_constructions_to_check():
    """The derivation must not go vacuous.

    A derived guard fails open: delete every construction site, or break the
    alias resolution, and the two tests below pass while checking nothing.
    This pins that the scan is still looking at something.
    """
    assert len(DEFS) > 60, f"only {len(DEFS)} dataclasses found; scan is broken"
    assert len(CONSTRUCTIONS) >= 100, (
        f"only {len(CONSTRUCTIONS)} imported-dataclass constructions found; "
        "the alias resolution or the package walk has broken"
    )
    # The repo can LEGITIMATELY hold zero class-resolvable replace sites, so
    # the corpus count cannot be the floor. The floor is the MECHANISM: a planted
    # fixture whose replace site the parser must find — with its planted
    # bogus field flagged — or the replace scan has broken.
    (module, _fields), key = next(
        (DEFS[k], k) for k in sorted(DEFS)
    )
    dotted = module.removesuffix(".py").replace("/", ".")
    snippet = (
        f"import dataclasses\n"
        f"from {dotted} import {key[1]}\n"
        f"x = dataclasses.replace({key[1]}(), planted_bogus_field=1)\n"
    )
    found = list(_replace_sites_in_tree(
        ast.parse(snippet), "planted_fixture.py", DEFS))
    assert found and found[0][2] == key, (
        "the replace-site parser no longer finds a planted "
        f"dataclasses.replace({key[1]}(), ...) site; the scan is broken"
    )
    assert "planted_bogus_field" in found[0][3]


def test_every_imported_dataclass_construction_names_real_fields():
    offenders = _offenders(CONSTRUCTIONS)
    assert offenders == [], (
        "these construction sites pass fields the dataclass does not have:\n  "
        + "\n  ".join(offenders)
    )


def test_every_imported_dataclass_replace_names_real_fields():
    offenders = _offenders(REPLACEMENTS)
    assert offenders == [], (
        "these dataclasses.replace calls pass fields the dataclass does not "
        "have:\n  " + "\n  ".join(offenders)
    )
